"""The shipped K-member seed ensemble: one module that runs every member on the same input, averages their
prediction, mean field and residual quantiles (Vincentised), then applies the S12 gated decode and the band decode
ONCE on the average.  Used both for the AOT export and as the eager fallback inside the
submission, so the two paths are the same code.

Members must share the architecture; each keeps its own learned advective baseline.  `U`, `m_core` and `cond` are
input-derived (identical across members; `check_shared` verifies it) and are taken from member 0.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from .decode import band_from_quantiles
from .gated_decode import GATE_COEFFS, GATE_TABLES, gated_decode
from .scaffold import RPDEModel


def load_member(path: str, gamma: float, device="cuda") -> RPDEModel:
    ck = torch.load(path, map_location="cpu")
    kw = dict(ck.get("core_kwargs") or {})
    kw.pop("grad_checkpoint", None)
    m = RPDEModel(ck["core_name"], kw, dict(ck.get("model_cfg") or {}))
    m.load_state_dict(ck.get("ema_state") or ck["model_state"], strict=False)
    m.gamma = float(gamma)
    return m.to(device).eval()


def post(pred, lo, up):
    keep = torch.tensor([1.0, 1.0, 0.0], device=pred.device)
    pred = torch.nan_to_num(pred.float() * keep, nan=0.0, posinf=0.0, neginf=0.0)
    lo = torch.nan_to_num(lo.float() * keep, nan=0.0, posinf=0.0, neginf=0.0)
    up = torch.nan_to_num(up.float() * keep, nan=0.0, posinf=0.0, neginf=0.0)
    return pred, lo, torch.maximum(up, lo)


class EnsPredict(nn.Module):
    """x (B,20,32,64,3) float32 -> (pred, lower, upper), the ensemble's whole predict path."""

    def __init__(self, members: Sequence[RPDEModel], checks: bool = False, band_scale: float = 1.0,
                 energy_preserve: bool = False, gated: bool = True, gated_params: dict | None = None,
                 mirror_views: bool = False):
        super().__init__()
        # mirror test-time augmentation: every member also predicts the
        # y-mirrored input (H flipped, v negated -- the exact flow at -AoA) and the un-mirrored outputs join the member
        # list, so K members give 2K views for the average / Vincentised band / energy-preserving combine.  Only for
        # members TRAINED with data.mirror_aug (a plain model loses 2.7 lbw on the mirrored view).  Pure tensor ops.
        self.mirror_views = bool(mirror_views)
        # gated-decode parameters: the fitted tables and the coefficients, as written by rpde/tools/fit_gate.py --out
        self.gated_params = dict(gated_params or {})
        missing = [k for k in GATE_TABLES + GATE_COEFFS if k not in self.gated_params]
        if gated and missing:
            raise ValueError(f"gated decode needs {missing} in gated_params (rpde/tools/fit_gate.py --out)")
        self.ms = nn.ModuleList(members)
        self.checks = bool(checks)
        # per-cell energy-preserving combine of the members' fluctuations (energy_preserving_fluct); off = ENS2/ENS2b
        self.energy_preserve = bool(energy_preserve)
        self.gated = bool(gated)                     # False only for offline checks (the dumps have no gated decode)
        # half-width multiplier about the band centre; < 1 narrows the band (0.95 is the R-top held-out optimum
        # for 1-3 member ensembles, neutral in regime)
        self.band_scale = float(band_scale)

    def forward(self, x):
        outs = [m(x) for m in self.ms]
        if self.mirror_views:
            xm = torch.flip(x, dims=(2,)).clone()
            xm[..., 1] = -xm[..., 1]
            outs += [unmirror_output(m(xm)) for m in self.ms]
        o0 = outs[0]
        if self.checks:
            check_shared(outs[:len(self.ms)])      # the mirror views legitimately differ in U (freestream band of the flipped field) and cond
        k = float(len(outs))
        mean = sum(o["mean"].float() for o in outs) / k
        Q = sum(o["Q_nd"].float() for o in outs) / k
        if self.energy_preserve:
            pred = mean + energy_preserving_fluct(outs)
        else:
            pred = sum(o["pred"].float() for o in outs) / k
        if self.gated:
            pred = gated_decode(pred, mean, x, o0["U"], o0["m_core"], o0["cond"], self.gated_params)
        lo, up = band_from_quantiles(Q, mean, o0["U"].float(), o0["m_core"].float(), fast=True,
                                     levels=o0.get("levels"), checks=False)
        if self.band_scale != 1.0:
            ctr, half = 0.5 * (lo + up), 0.5 * (up - lo) * self.band_scale
            lo, up = ctr - half, ctr + half
        return post(pred, lo, up)


def unmirror_output(o: dict) -> dict:
    """Model outputs on mirror(x) -> the frame of x: (B,20,H,W,3) fields flip H and negate v; the residual quantiles
    of v map as q_tau(-r) = -q_{1-tau}(r): flip H, negate, reverse the knot order (the level set is symmetric)."""
    out = dict(o)
    for k in ("mean", "pred"):
        v = torch.flip(o[k], dims=(2,)).clone()
        v[..., 1] = -v[..., 1]
        out[k] = v
    Q = torch.flip(o["Q_nd"], dims=(2,)).clone()
    Q[..., 1, :] = -torch.flip(Q[..., 1, :], dims=(-1,))
    out["Q_nd"] = Q
    out["m_core"] = torch.flip(o["m_core"], dims=(-2,))
    return out


def energy_preserving_fluct(outs: List[dict], g_max: float = 16.0, eps: float = 1e-12) -> torch.Tensor:
    """The members' averaged fluctuation, rescaled per cell to the members' MEAN fluctuation energy.

    Averaging K fluctuations that disagree in phase shrinks var_t by (1 + rho)/2 (K=2); the lost energy sits where the
    members disagree, i.e. the chaotic shear-layer cells where the true TKE is largest.  Per cell (u,v summed):
    g = mean_k var_t(f_k) / var_t(mean_k f_k) >= 1 (Jensen), clipped to [1, g_max]; cells with no averaged energy
    keep g = 1.  The fluctuation keeps its zero temporal mean, so mvpe and the band are untouched.
    Paired R-top dump test: +0.038 lbw held-out, +0.027 in-regime at
    gamma 0.70 vs the shipped 0.75.
    """
    fk = [o["pred"].float() - o["mean"].float() for o in outs]
    f = sum(fk) / float(len(fk))
    ek = sum(t[..., :2].var(dim=1, unbiased=False).sum(-1) for t in fk) / float(len(fk))   # (B,H,W)
    ef = f[..., :2].var(dim=1, unbiased=False).sum(-1)
    g = torch.where(ef > eps, ek / ef.clamp(min=eps), torch.ones_like(ef)).clamp(1.0, g_max)
    return f * g.sqrt()[:, None, :, :, None]


def check_shared(outs: List[dict], tol: float = 1e-5):
    for key in ("U", "m_core", "cond"):
        ref = outs[0][key].float()
        for i, o in enumerate(outs[1:], 1):
            d = float((o[key].float() - ref).abs().max())
            if d > tol:
                raise ValueError(f"member {i} {key} differs from member 0 by {d:.3g}")
