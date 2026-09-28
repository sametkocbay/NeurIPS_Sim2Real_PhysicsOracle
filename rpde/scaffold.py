"""Scaffold: preprocessing, advective baseline, conditioning/FiLM, head, model.

Ships in the submission -> imports only torch, numpy, stdlib and rpde.constants.
Model scaffold: preprocessing, advective baseline, input volume, head and the full model.

Internal layout for the cores is channels-first over the 40-frame volume:
`(B, C, T=40, H=32, W=64)`; frames 0..19 are the input window, 20..39 the
advected baseline / future estimate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rpde.constants import (
    CHORD,
    DT,
    DX,
    FS_ROWS,
    H,
    QUANTILE_LEVELS,
    T_IN,
    T_OUT,
    T_VOL,
    W,
)

__all__ = [
    "Preproc",
    "shift_x",
    "advective_baseline",
    "advective_bank",
    "BaselineParams",
    "FiLM",
    "drop_path",
    "DropPath",
    "Head",
    "RPDEModel",
    "build_volume",
    "hindcast_residual",
]

_EPS = 1e-8
U_MIN = 1e-3


# ---------------------------------------------------------------------------
# 3.1  Preprocessing
# ---------------------------------------------------------------------------
class Preproc(nn.Module):
    """Raw physical window -> nondimensional fields, masks and conditioning.

    No parameters; kept as a Module so it moves with `.to(device)` buffers.

    `body_shadow=True`
    additionally splits `m_core` into `m_body` (one connected component of the
    core mask, 8-connectivity) and `m_shadow` (the rest), and takes the
    conditioning centroid/extent from `m_body` instead of the full mask.
    Default `False` -> every returned tensor is bit-identical to before.
    """

    def __init__(self, body_shadow: bool = False, body_mode: str = "centroid",
                 body_iters: int = 40) -> None:
        super().__init__()
        self.body_shadow = bool(body_shadow)
        self.body_mode = str(body_mode)
        if self.body_mode not in ("centroid", "topmost"):
            raise ValueError("body_mode must be 'centroid' or 'topmost', "
                             f"got {body_mode!r}")
        self.body_iters = int(body_iters)
        # x in [-1, 1] along W (columns, downstream), y in [-1, 1] along H with
        # +1 at row 0 (top of the FOV = largest physical y).
        xs = torch.linspace(-1.0, 1.0, W)
        ys = torch.linspace(1.0, -1.0, H)
        xy = torch.stack(
            [xs.view(1, W).expand(H, W), ys.view(H, 1).expand(H, W)], dim=0
        )  # (2, H, W)
        self.register_buffer("xy", xy.unsqueeze(0), persistent=False)  # (1,2,H,W)
        col = torch.arange(W, dtype=torch.float32) / max(W - 1, 1)
        row = torch.arange(H, dtype=torch.float32) / max(H - 1, 1)
        self.register_buffer("colnorm", col.view(1, 1, W), persistent=False)
        self.register_buffer("rownorm", row.view(1, H, 1), persistent=False)

    def _centroid_extent(self, m2: torch.Tensor):
        """m2: (B,H,W) 0/1 -> (cx, cy, ex, ey), normalised to [0,1], 0 if empty."""
        cnt = m2.sum(dim=(1, 2))                                # (B,)
        has = (cnt > 0).to(m2.dtype)
        den = cnt.clamp(min=1.0)
        cx = (m2 * self.colnorm).sum(dim=(1, 2)) / den
        cy = (m2 * self.rownorm).sum(dim=(1, 2)) / den
        big = torch.full_like(m2, 2.0)
        small = torch.full_like(m2, -1.0)
        colg = self.colnorm.expand_as(m2)
        rowg = self.rownorm.expand_as(m2)
        xmin = torch.where(m2 > 0, colg, big).amin(dim=(1, 2))
        xmax = torch.where(m2 > 0, colg, small).amax(dim=(1, 2))
        ymin = torch.where(m2 > 0, rowg, big).amin(dim=(1, 2))
        ymax = torch.where(m2 > 0, rowg, small).amax(dim=(1, 2))
        ex = ((xmax - xmin) * has).clamp(min=0.0)
        ey = ((ymax - ymin) * has).clamp(min=0.0)
        return cx * has, cy * has, ex, ey

    def body_split(self, m_core: torch.Tensor):
        """Split (B,1,H,W) core mask into (m_body, m_shadow).

        `m_body` is the 8-connected component of the mask grown from a single
        seed cell; `m_shadow = m_core - m_body`.  The seed is the mask cell
        closest to the mask centroid (`body_mode="centroid"`, the documented
        rule) or the topmost-then-leftmost mask cell (`"topmost"`).  The flood
        fill is `body_iters` iterations of (3x3 max-pool AND mask), which is
        exact once the iteration count exceeds the geodesic radius of the
        component (40 covers every mask in the real cache).
        """
        B, _, Hh, Ww = m_core.shape
        m2 = m_core[:, 0]                                       # (B,H,W)
        rows = torch.arange(Hh, device=m_core.device, dtype=m_core.dtype).view(1, Hh, 1)
        cols = torch.arange(Ww, device=m_core.device, dtype=m_core.dtype).view(1, 1, Ww)
        cnt = m2.sum(dim=(1, 2)).clamp(min=1.0).view(B, 1, 1)
        BIG = float(4 * (Hh * Hh + Ww * Ww) + 16)
        if self.body_mode == "topmost":
            key = rows * float(Ww) + cols
        else:
            cy = (m2 * rows).sum(dim=(1, 2)).view(B, 1, 1) / cnt
            cx = (m2 * cols).sum(dim=(1, 2)).view(B, 1, 1) / cnt
            key = (rows - cy) ** 2 + (cols - cx) ** 2
        key = key + BIG * (1.0 - m2)                            # only mask cells
        flat = key.reshape(B, -1).argmin(dim=1)                 # (B,)
        seed = torch.zeros(B, Hh * Ww, dtype=m_core.dtype, device=m_core.device)
        seed.scatter_(1, flat.view(B, 1), 1.0)
        cur = (seed.view(B, 1, Hh, Ww) * m_core)                # empty mask -> empty body
        for _ in range(self.body_iters):
            cur = F.max_pool2d(cur, 3, stride=1, padding=1) * m_core
        return cur, m_core - cur

    @staticmethod
    def freestream(u: torch.Tensor) -> torch.Tensor:
        """Median of the NONZERO u over FS_ROWS, all frames, all columns.

        u: (B, T, H, W) -> (B,), clamped to >= U_MIN.
        """
        band = u[:, :, FS_ROWS, :].reshape(u.shape[0], -1)
        band = torch.where(band != 0, band, torch.full_like(band, float("nan")))
        med = torch.nanmedian(band, dim=1).values
        med = torch.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
        return med.clamp(min=U_MIN)

    def forward(self, x: torch.Tensor, domain: Optional[torch.Tensor] = None,
                clock: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """`clock` (optional, (B,2)) overrides the window's derived clock:
        column 0 = the true convective step per output frame `dt*`, column 1 =
        the physical freestream used for the conditioning.  A non-positive entry
        means "derive it from the window", so `clock=None` (and real windows
        under the fix) reproduce the old path bit-for-bit.

        Simulation fields are stored NONDIMENSIONAL (freestream ~ 1.1), and this
        module is domain-blind, so without an override a sim window yields
        `dt* = DT*1.1/CHORD` and `a = 1.1*DT/DX ~ 6.5` cells/frame for EVERY Re,
        against a true motion of ~1.2 cells/frame -- see
        The override is supplied per window by the
        trainer from the regime's Re and the sim frame stride.
        """
        # x: (B, 20, 32, 64, 3) physical
        if x.dim() != 5 or x.shape[1] != T_IN or x.shape[2] != H or x.shape[3] != W:
            raise ValueError(f"Preproc expects (B,{T_IN},{H},{W},C), got {tuple(x.shape)}")
        B = x.shape[0]
        u = x[..., 0]
        v = x[..., 1]

        U = self.freestream(u)                                  # (B,)
        Ub = U.view(B, 1, 1, 1)
        u_nd = u / Ub
        v_nd = v / Ub

        zu = (u == 0).to(x.dtype)
        zv = (v == 0).to(x.dtype)
        m_core = (zu.min(dim=1).values).unsqueeze(1)            # (B,1,H,W) 1 where u==0 in ALL frames
        # context truncation (data.ctx_drop, training only): an input frame that is zero in EVERY cell is a hidden
        # frame, not a field.  The drop-out maps and the advective baseline's mean then run over the visible frames
        # only; with all frames visible (`fvalid` all ones) every expression below is the shipped one.
        fvalid = (zu.reshape(B, T_IN, -1).amin(dim=2) < 1.0).to(x.dtype)      # (B,20) 1 = frame has data
        # training only: in eval / AOT export there is no data-dependent branch and the graph is the shipped one
        hidden = bool(self.training) and bool((fvalid < 1.0).any())
        if hidden:
            nvis = fvalid.sum(dim=1).clamp(min=1.0).view(B, 1, 1, 1)
            fv = fvalid.view(B, T_IN, 1, 1)
            p_u = (zu * fv).sum(dim=1, keepdim=True) / nvis
            p_v = (zv * fv).sum(dim=1, keepdim=True) / nvis
        else:
            p_u = zu.mean(dim=1, keepdim=True)                  # (B,1,H,W)
            p_v = zv.mean(dim=1, keepdim=True)

        dtstar = DT * U / CHORD                                 # (B,)
        a = U * DT / DX                                         # (B,)
        u_ref = U
        if clock is not None:
            ck = torch.as_tensor(clock, dtype=x.dtype, device=x.device).reshape(B, 2)
            dt_ov, u_ov = ck[:, 0], ck[:, 1]
            # a is a pure function of dt*: U*DT/DX == (DT*U/CHORD)*(CHORD/DX)
            dtstar = torch.where(dt_ov > 0, dt_ov, dtstar)
            a = torch.where(dt_ov > 0, dt_ov * (CHORD / DX), a)
            u_ref = torch.where(u_ov > 0, u_ov, U)

        # core-mask centroid and extent, normalised to [0, 1] (from the BODY
        # component when the body/shadow split is on)
        m_body = m_shadow = None
        if self.body_shadow:
            m_body, m_shadow = self.body_split(m_core)
            cx, cy, ex, ey = self._centroid_extent(m_body[:, 0])
        else:
            cx, cy, ex, ey = self._centroid_extent(m_core[:, 0])

        if domain is None:
            dom = torch.zeros_like(U)
        else:
            dom = torch.as_tensor(domain, dtype=x.dtype, device=x.device).reshape(-1)
            if dom.numel() == 1:
                dom = dom.expand(B)
            dom = dom.to(x.dtype)

        if getattr(self, "cond_geom_off", False):
            # diagnosis 2026-09-26: withhold the mask-geometry key (angle identity) from FiLM
            cx, cy, ex, ey = (torch.zeros_like(U) for _ in range(4))
        cond = torch.stack(
            [
                (u_ref - 0.22) / 0.10,
                torch.log(dtstar.clamp(min=_EPS)),
                a / 2.0,
                cx,
                cy,
                ex,
                ey,
                dom,
            ],
            dim=1,
        )                                                       # (B, 8)

        out = dict(
            u_nd=u_nd, v_nd=v_nd, U=U, m_core=m_core, p_u=p_u, p_v=p_v,
            dtstar=dtstar, a=a, cond=cond, xy=self.xy,
            fvalid=fvalid if hidden else None,
        )
        if self.body_shadow:
            out["m_body"] = m_body
            out["m_shadow"] = m_shadow
        return out


# ---------------------------------------------------------------------------
# 3.2  Advective baseline
# ---------------------------------------------------------------------------
def shift_x(field: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Shift `field` downstream (towards larger column index) by `shift` cells.

    field: (N, C, H, W); shift: (N,) in cells, may carry gradient.
    Linear interpolation, zeros outside (inflow at column 0 gets 0).
    out[..., j] = field[..., j - shift]
    """
    N, C, Hh, Ww = field.shape
    dev, dt = field.device, field.dtype
    gx = torch.linspace(-1.0, 1.0, Ww, device=dev, dtype=dt).view(1, 1, Ww)
    gy = torch.linspace(-1.0, 1.0, Hh, device=dev, dtype=dt).view(1, Hh, 1)
    dsx = (2.0 * shift / max(Ww - 1, 1)).view(N, 1, 1)
    gxs = (gx - dsx).expand(N, Hh, Ww)
    gys = gy.expand(N, Hh, Ww)
    grid = torch.stack([gxs, gys], dim=-1)                      # (N,H,W,2)
    return F.grid_sample(field, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


class BaselineParams(nn.Module):
    """Learnable scalars of the advective baseline.

    `gamma` (convection fraction) is always active. The damping and the
    U-dependent mean blend are switchable, because the week-one measurement
    found no damping (tau -> infinity) and a constant blend beta = 0.75 best at
    every Re row:

    * `damping=False` (default): `D_k = 1` and `tau_star` is NOT used in the
      forward pass. The parameter still exists (init `tau_star0`) so the
      ablation can be switched on without changing the state_dict layout.
    * `beta_mode="const"` (default): `beta = sigmoid(beta_logit)`, a learnable
      scalar initialised to `beta0`; `U0` and `w` are unused.
    * `beta_mode="sigmoid"`: `beta = sigmoid((U - U0)/w)` as in spec 3.2;
      `beta_logit` is unused.

    Unused parameters simply receive no gradient (`p.grad is None`).
    """

    def __init__(self, gamma0: float = 0.6, tau_star0: float = 0.3,
                 U0: float = 0.129, w: float = 0.05, damping: bool = False,
                 beta_mode: str = "const", beta0: float = 0.75,
                 beta_fixed: bool = False, gamma_fixed: bool = False) -> None:
        super().__init__()
        if beta_mode not in ("const", "sigmoid"):
            raise ValueError(f"beta_mode must be 'const' or 'sigmoid', got {beta_mode!r}")
        self.damping = bool(damping)
        self.beta_mode = str(beta_mode)
        self.gamma = nn.Parameter(torch.tensor(float(gamma0)))
        self.tau_star = nn.Parameter(torch.tensor(float(tau_star0)))
        self.U0 = nn.Parameter(torch.tensor(float(U0)))
        self.w = nn.Parameter(torch.tensor(float(w)))
        b0 = min(max(float(beta0), 1e-6), 1.0 - 1e-6)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(b0 / (1.0 - b0))))
        # beta_fixed / gamma_fixed: the trained S14 family learns beta ~0.05, i.e. a
        # persistence anchor with no advected energy.  Freezing keeps the physics prior at its initial value.
        if bool(beta_fixed):
            self.beta_logit.requires_grad_(False)
        if bool(gamma_fixed):
            self.gamma.requires_grad_(False)

    def extra_repr(self) -> str:
        return f"damping={self.damping}, beta_mode={self.beta_mode!r}"


def _baseline_beta(params: Any, U: torch.Tensor, B: int,
                   dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The mean/fluctuation blend weight beta of the advective baseline, (B,)."""
    if str(getattr(params, "beta_mode", "const")) == "sigmoid":
        wpar = params.w.clamp(min=1e-3)
        return torch.sigmoid((U - params.U0) / wpar)            # (B,)
    bl = getattr(params, "beta_logit", None)
    b0 = torch.sigmoid(bl) if bl is not None else torch.tensor(
        0.75, dtype=dtype, device=device)
    return b0.reshape(1).expand(B)                              # (B,)


def advective_baseline(u_nd: torch.Tensor, v_nd: torch.Tensor, m_core: torch.Tensor,
                       a: torch.Tensor, U: torch.Tensor, params: Any,
                       t_out: int = T_OUT, fvalid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Advection of the last-frame fluctuation (spec 3.2, with the measured
    defaults of `BaselineParams`: no damping, constant mean blend beta = 0.75).

        M      = (1 - beta) * last + beta * mean_in
        F      = (last - mean_in) * (1 - m_core)
        base_k = M + beta * D_k * shift_x(F, gamma * a * k)      # then * (1 - m_core)
               = last + beta * (D_k * shift_x(F, s_k) - F)  on unmasked cells

    u_nd, v_nd : (B, T, H, W)      nondimensional input window (T = T_IN by
                 default; any T works, the mean/last are taken over what is given)
    m_core     : (B, 1, H, W)
    a          : (B,)  cells per frame ;  U : (B,)  m/s
    params     : object with .gamma (+ .tau_star/.damping, .beta_logit /
                 .U0/.w/.beta_mode); see `BaselineParams`
    t_out      : number of forecast frames (default `T_OUT` = 20; the hindcast
                 feature of `hindcast_residual` calls it with 10).  Every
                 expression below uses `t_out`, so `t_out=T_OUT` is arithmetically
                 identical to the original fixed-`T_OUT` code.
    returns    : base (B, 2, t_out, H, W)
    """
    B = u_nd.shape[0]
    vel = torch.stack([u_nd, v_nd], dim=1)                      # (B,2,20,H,W)
    if fvalid is not None:                                      # hidden input frames (context truncation): mean over
        fv = fvalid.to(vel.dtype).view(B, 1, -1, 1, 1)          # the visible frames only; the last frame is always visible
        mean_in = (vel * fv).sum(dim=2) / fv.sum(dim=2).clamp(min=1.0)
    else:
        mean_in = vel.mean(dim=2)                               # (B,2,H,W)
    last = vel[:, :, -1]                                        # (B,2,H,W)

    beta = _baseline_beta(params, U, B, u_nd.dtype, u_nd.device)
    M = (1.0 - beta.view(B, 1, 1, 1)) * last + beta.view(B, 1, 1, 1) * mean_in
    Fl = (last - mean_in) * (1.0 - m_core)                      # (B,2,H,W)

    tk = int(t_out)
    k = torch.arange(1, tk + 1, device=u_nd.device, dtype=u_nd.dtype)  # (tk,)
    s = params.gamma * a.view(B, 1) * k.view(1, tk)             # (B,tk) cells

    Fk = Fl.unsqueeze(1).expand(B, tk, 2, H, W).reshape(B * tk, 2, H, W)
    adv = shift_x(Fk, s.reshape(B * tk)).view(B, tk, 2, H, W)
    if bool(getattr(params, "damping", False)):
        tau_frames = (params.tau_star * CHORD / (U * DT)).clamp(min=1e-3)  # (B,)
        D = torch.exp(-k.view(1, tk) / tau_frames.view(B, 1))              # (B,tk)
        adv = adv * D.view(B, tk, 1, 1, 1)
    # beta multiplies the advected fluctuation as well, i.e.
    #   base_k = last + beta * (D_k * shift_x(F, s_k) - F)
    base = M.unsqueeze(1) + beta.view(B, 1, 1, 1, 1) * adv      # (B,tk,2,H,W)
    base = base.permute(0, 2, 1, 3, 4).contiguous()             # (B,2,20,H,W)
    return base * (1.0 - m_core).unsqueeze(2)


def advective_bank(u_nd: torch.Tensor, v_nd: torch.Tensor, m_core: torch.Tensor,
                   a: torch.Tensor, U: torch.Tensor, params: Any,
                   ratios, t_out: int = T_OUT) -> torch.Tensor:
    """Advective baselines at OTHER convection speeds, as differences (opt-in).

    `advective_baseline` moves the last-frame fluctuation at one speed:
    `gamma * a` cells per frame, the same number for every cell and every lead.
    A wake does not convect at one speed.  This helper evaluates the same
    baseline at `ratio * gamma` for each ratio in `ratios` and returns the
    DIFFERENCE from the primary baseline, because only the advected part of the
    baseline depends on the speed and the mean part `M` cancels:

        base_r - base = beta * (shift_x(F, r*gamma*a*k) - shift_x(F, gamma*a*k))

    Ratio 1.0 therefore returns exactly zero.  The caller decides what to do
    with the bank: feed it to the core as extra channels
    (`model.speed_bank_mode="channels"`) or mix it into the anchor with head
    coefficients (`"mix"`); see `RPDEModel`.

    ratios  : sequence of R positive floats, multipliers on `params.gamma`
    returns : (B, R, 2, t_out, H, W), zero on core-masked cells
    """
    B = u_nd.shape[0]
    R = len(ratios)
    vel = torch.stack([u_nd, v_nd], dim=1)                      # (B,2,20,H,W)
    mean_in = vel.mean(dim=2)
    last = vel[:, :, -1]
    beta = _baseline_beta(params, U, B, u_nd.dtype, u_nd.device)
    Fl = (last - mean_in) * (1.0 - m_core)                      # (B,2,H,W)

    tk = int(t_out)
    k = torch.arange(1, tk + 1, device=u_nd.device, dtype=u_nd.dtype)  # (tk,)
    r = torch.as_tensor(list(ratios), device=u_nd.device, dtype=u_nd.dtype)
    # (B, R+1, tk): row 0 is the primary speed, rows 1.. are the bank
    s = (params.gamma * a.view(B, 1, 1) * k.view(1, 1, tk)
         * torch.cat([torch.ones(1, device=r.device, dtype=r.dtype), r]).view(1, R + 1, 1))
    Fk = Fl.view(B, 1, 1, 2, H, W).expand(B, R + 1, tk, 2, H, W).reshape(-1, 2, H, W)
    adv = shift_x(Fk, s.reshape(-1)).view(B, R + 1, tk, 2, H, W)
    d = beta.view(B, 1, 1, 1, 1, 1) * (adv[:, 1:] - adv[:, :1])
    d = d.permute(0, 1, 3, 2, 4, 5)                             # (B,R,2,tk,H,W)
    return d * (1.0 - m_core).view(B, 1, 1, 1, H, W)


# ---------------------------------------------------------------------------
# 3.2b  Hindcast-residual feature (opt-in, `model.hindcast_feat`)
# ---------------------------------------------------------------------------
def hindcast_residual(u_nd: torch.Tensor, v_nd: torch.Tensor, m_core: torch.Tensor,
                      a: torch.Tensor, U: torch.Tensor, params: Any,
                      n_hist: int = T_IN // 2) -> torch.Tensor:
    """Per-cell predictability map read off the INPUT window alone.

    The advective baseline is run *inside* the input window: frames `0..n-1`
    play the role of the "last n frames" (their mean and their last frame give
    `M` and `F`) and are advected forward to hindcast frames `n..2n-1`, which
    are actually observed.  The per-cell, per-channel RMS of that hindcast
    residual over the `n` frames is the difficulty of this window at this cell,
    available at inference with zero labels.

    u_nd, v_nd : (B, T_IN, H, W) nondimensional input window
    m_core     : (B, 1, H, W);  a : (B,) cells/frame;  U : (B,) m/s
    params     : the model's own `BaselineParams` (used without gradient)
    returns    : (B, 2, H, W) = log1p(RMS residual), zero on core-masked cells

    The residual is averaged only over the frames where the observed value is
    non-zero, i.e. over the same "measured" set the scorer uses, so PIV
    drop-outs do not masquerade as unpredictability.
    """
    n = int(n_hist)
    with torch.no_grad():
        base = advective_baseline(u_nd[:, :n], v_nd[:, :n], m_core, a, U, params,
                                  t_out=n)                          # (B,2,n,H,W)
        act = torch.stack([u_nd[:, n:2 * n], v_nd[:, n:2 * n]], dim=1)
        r = act - base
        valid = (act != 0).to(act.dtype)
        cnt = valid.sum(dim=2).clamp(min=1.0)                       # (B,2,H,W)
        ms = ((r * r) * valid).sum(dim=2) / cnt
        h = torch.log1p(ms.clamp(min=0.0).sqrt())
        h = h * (1.0 - m_core)                                      # (B,1,H,W) bcast
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return h


# ---------------------------------------------------------------------------
# 3.3  Volume
# ---------------------------------------------------------------------------
def build_volume(u_nd: torch.Tensor, v_nd: torch.Tensor, base: torch.Tensor,
                 m_core: torch.Tensor, p_u: torch.Tensor, p_v: torch.Tensor,
                 xy: torch.Tensor, m_shadow: Optional[torch.Tensor] = None,
                 hind: Optional[torch.Tensor] = None,
                 bank: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(B, 8, 40, H, W) = [u_nd, v_nd, m_core, p_u, p_v, x_norm, y_norm, is_future].

    With `m_shadow` (the body/shadow split, opt-in) the mask channel carries the
    body component and the shadow is inserted right after it, giving
    (B, 9, 40, H, W) = [u_nd, v_nd, m_body, m_shadow, p_u, p_v, x, y, is_future].

    With `hind` (B, 2, H, W) (the hindcast-residual feature, opt-in) two more
    static channels are appended after `x, y`, giving in_ch 10 (or 11 with the
    body/shadow split).

    With `bank` (B, R, 2, 20, H, W) (the multi-speed convection bank, opt-in)
    2R channels are appended at the END, after `is_future`, so no existing
    channel index moves.  They are zero on the input frames and carry
    `base_r - base` on the future frames.
    """
    B = u_nd.shape[0]
    past = torch.stack([u_nd, v_nd], dim=1)                     # (B,2,20,H,W)
    vel = torch.cat([past, base], dim=2)                        # (B,2,40,H,W)
    ms = [m_core] if m_shadow is None else [m_core, m_shadow]
    extra = [] if hind is None else [hind]
    nstat = 5 + (0 if m_shadow is None else 1) + (0 if hind is None else 2)
    stat = torch.cat(ms + [p_u, p_v, xy.expand(B, 2, H, W)] + extra, dim=1)
    stat = stat.unsqueeze(2).expand(B, nstat, T_VOL, H, W)
    isf = torch.zeros(B, 1, T_VOL, H, W, dtype=u_nd.dtype, device=u_nd.device)
    isf[:, :, T_IN:] = 1.0
    parts = [vel, stat, isf]
    if bank is not None:
        R = bank.shape[1]
        bch = torch.zeros(B, 2 * R, T_VOL, H, W, dtype=u_nd.dtype, device=u_nd.device)
        bch[:, :, T_IN:] = bank.reshape(B, 2 * R, T_OUT, H, W)
        parts.append(bch)
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# 3.4  FiLM / adaLN-Zero helper + stochastic depth
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    """Per-sub-layer (scale, shift, gate) from the conditioning embedding.

    `Linear(cond_dim -> 3*d)`; gate slice zero-initialised so the residual
    branch is exactly zero at init. Scale is applied as `(1 + scale)`.
    """

    def __init__(self, cond_dim: int, d: int, n_sublayers: int = 1) -> None:
        super().__init__()
        self.d = d
        self.n = n_sublayers
        self.lin = nn.Linear(cond_dim, 3 * d * n_sublayers)
        nn.init.normal_(self.lin.weight, std=0.02)
        nn.init.zeros_(self.lin.bias)
        with torch.no_grad():
            wv = self.lin.weight.view(n_sublayers, 3, d, cond_dim)
            wv[:, 2].zero_()                                     # gate weights
            bv = self.lin.bias.view(n_sublayers, 3, d)
            bv[:, 2].zero_()                                     # gate bias

    def forward(self, cond: torch.Tensor):
        """cond: (B, cond_dim) -> scale, shift, gate each (B, n, d)."""
        h = self.lin(cond).view(-1, self.n, 3, self.d)
        return h[:, :, 0], h[:, :, 1], h[:, :, 2]


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """x: (..., d) channels-last; scale/shift: (B, d) broadcast over middle dims."""
    nd = x.dim() - scale.dim()
    view = (scale.shape[0],) + (1,) * nd + (scale.shape[-1],)
    return x * (1.0 + scale.view(view)) + shift.view(view)


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p <= 0.0 or not training:
        return x
    keep = 1.0 - p
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p:.3f}"


class CondEmbed(nn.Module):
    """Shared conditioning MLP feeding every FiLM layer of a core."""

    def __init__(self, cond_dim: int = 8, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.out_dim = hidden

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)


# ---------------------------------------------------------------------------
# 3.5  Head
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """Conv3d(core_channels -> 4 + 2n [+2], k=1) on frames 20..39.

    With the default five levels (n = 5) the layout is the shipped one:
    0:2 dmean (zero-init) | 2:4 fluct (zero-init) | 4:9 q-params u | 9:14 q-params v.
    Quantiles are built by cumulative softplus outward from the median
    (non-crossing).  Bias of the gap channels is set so q_0.9 - q_0.1 ~= `gap0`
    at init, whatever the level set.

    Opt-in:
      * `levels`   -- any strictly increasing level set containing 0.5
                      (`model.quantile_levels`); the conv gains 2*(n-5) channels.
      * `err_head` -- 2 extra channels at the END of the conv (so the first
                      4+2n stay where they were): the predicted log residual
                      magnitude, one per velocity channel.  `forward` then
                      returns a fourth element.
      * `mix_ch`   -- R more channels after those: the per-cell, per-lead
                      coefficients of the multi-speed convection bank
                      (`model.speed_bank_mode="mix"`).  Zero-init like the rest
                      of the conv, so the mixed anchor starts at exactly the
                      primary baseline.  `forward` returns them last.
    """

    def __init__(self, in_channels: int, gap0: float = 0.2,
                 quantile_hidden: int = 0, detach_trunk: bool = False,
                 levels=None, err_head: bool = False, mix_ch: int = 0) -> None:
        super().__init__()
        lv = tuple(float(v) for v in (QUANTILE_LEVELS if levels is None else levels))
        if len(lv) < 3 or any(b <= a for a, b in zip(lv, lv[1:])):
            raise ValueError(f"quantile levels must be strictly increasing: {lv}")
        if not (0.0 < lv[0] and lv[-1] < 1.0):
            raise ValueError(f"quantile levels must lie in (0,1): {lv}")
        self.levels = lv
        self.n_levels = len(lv)
        self.med_idx = next(i for i, v in enumerate(lv) if abs(v - 0.5) < 1e-12)
        self.err_head = bool(err_head)
        n = self.n_levels
        self.n_q = 2 * n
        self.err_off = 4 + self.n_q               # first err channel (if any)
        self.mix_ch = int(mix_ch)
        self.mix_off = self.err_off + (2 if self.err_head else 0)
        out_ch = self.mix_off + self.mix_ch
        self.conv = nn.Conv3d(in_channels, out_ch, kernel_size=1)
        # inference-only: accept (B,T,H,W,C) and apply the 1x1x1 conv as a
        # matmul, so the core can skip its (B,d,T,H,W) contiguous copy.
        self.channels_last_in = False
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # gap0 is the initial q_0.9 - q_0.1; spread it evenly over the gaps
        # between those two levels (n = 5 -> gap0/4, the shipped value).
        i_lo = min(range(n), key=lambda i: abs(lv[i] - 0.10))
        i_hi = min(range(n), key=lambda i: abs(lv[i] - 0.90))
        g = gap0 / max(1, i_hi - i_lo)
        b = math.log(math.expm1(g))          # softplus(b) == g
        with torch.no_grad():
            for base in (4, 4 + n):
                self.conv.bias[base] = 0.0           # median offset
                self.conv.bias[base + 1: base + n] = b
        # -- opt-in band-head options (both off by default) ------------------
        # `detach_trunk`: the 2n quantile channels are computed from detached
        #   features, so the pinball loss trains the head but never shapes the
        #   trunk (the cheap form of a cross-fitted
        #   head).  Identical in eval/no-grad; only the backward changes.
        # `quantile_hidden`: a small MLP (in_channels -> h -> 2n) for the
        #   quantile channels only, zero-init on the last layer so the initial
        #   quantile field is exactly the same constant as the linear head's.
        self.detach_trunk = bool(detach_trunk)
        self.quantile_hidden = int(quantile_hidden)
        if self.quantile_hidden > 0:
            self.qmlp = nn.Sequential(
                nn.Linear(in_channels, self.quantile_hidden), nn.GELU(),
                nn.Linear(self.quantile_hidden, self.n_q))
            nn.init.zeros_(self.qmlp[2].weight)
            with torch.no_grad():
                self.qmlp[2].bias.zero_()
                for base in (0, n):
                    self.qmlp[2].bias[base] = 0.0
                    self.qmlp[2].bias[base + 1: base + n] = b
        else:
            self.qmlp = None

    def forward(self, feat: torch.Tensor, q_add: Optional[torch.Tensor] = None):
        """feat: (B, C, 40, H, W) -> dmean, fluct (B,20,H,W,2), Q (B,20,H,W,2,n).

        `q_add` (B, 2n, 20, H, W), optional: added to the 2n quantile PARAMETERS only (before the median/softplus
        construction); dmean and fluct are untouched.  Used by the quantile-only full-resolution skip.

        With `err_head=True` a fourth element `err` (B,20,H,W,2) follows -- the
        predicted log residual magnitude.  With `mix_ch=R` a further element
        `mix` (B,20,H,W,R) follows -- the multi-speed bank coefficients.  The
        arity is conditional so every existing `dmean, fluct, Q = head(feat)`
        call site keeps working.

        With `channels_last_in` (set by `rpde.cores.axial.enable_fast_axial`)
        `feat` is (B, 40, H, W, C) instead and the same 1x1x1 conv weights are
        applied as an `F.linear`.
        """
        n, nq = self.n_levels, self.n_q
        fcl = None
        if getattr(self, "channels_last_in", False):
            w = self.conv.weight.reshape(self.conv.weight.shape[0], -1)
            fcl = feat[:, T_IN:]                                # (B,20,H,W,C)
            o = F.linear(fcl, w, self.conv.bias)                # (B,20,H,W,4+2n[+2])
        else:
            o = self.conv(feat[:, :, T_IN:])                    # (B,4+2n[+2],20,H,W)
            o = o.permute(0, 2, 3, 4, 1)                        # (B,20,H,W,4+2n[+2])
        dmean = o[..., 0:2]
        fluct = o[..., 2:4]
        # zero temporal mean over the 20 output frames: the fluctuation head adds
        # energy (tke) but can never shift the window-mean profile (mvpe, rel_l2 mean part)
        fluct = fluct - fluct.mean(dim=1, keepdim=True)
        if self.qmlp is None and not self.detach_trunk:         # default path
            qo = o[..., 4:4 + nq]
        else:
            if fcl is None:
                fcl = feat[:, :, T_IN:].permute(0, 2, 3, 4, 1)  # (B,20,H,W,C)
            fq = fcl.detach() if self.detach_trunk else fcl
            if self.qmlp is not None:
                qo = self.qmlp(fq)
            else:
                wq = self.conv.weight[4:4 + nq].reshape(nq, -1)
                qo = F.linear(fq, wq, self.conv.bias[4:4 + nq])
        if q_add is not None:
            qo = qo + q_add.permute(0, 2, 3, 4, 1).to(qo.dtype)
        qp = torch.stack([qo[..., 0:n], qo[..., n:2 * n]], dim=-2)  # (B,20,H,W,2,n)
        # median + cumulative softplus gaps, stacked outward: parameter j+1 is
        # the gap between levels j and j+1 (n = 5 reproduces g1..g4 exactly).
        gaps = [F.softplus(qp[..., j + 1]) for j in range(n - 1)]
        m = self.med_idx
        qs: list = [None] * n
        qs[m] = qp[..., 0]
        for j in range(m - 1, -1, -1):
            qs[j] = qs[j + 1] - gaps[j]
        for j in range(m + 1, n):
            qs[j] = qs[j - 1] + gaps[j - 1]
        Q = torch.stack(qs, dim=-1)                             # (B,20,H,W,2,n)
        if not self.err_head and not self.mix_ch:               # default path
            return dmean, fluct, Q
        outs = [dmean, fluct, Q]
        if self.err_head:
            outs.append(o[..., self.err_off:self.err_off + 2])
        if self.mix_ch:
            outs.append(o[..., self.mix_off:self.mix_off + self.mix_ch])
        return tuple(outs)


# ---------------------------------------------------------------------------
# 3.6  Model
# ---------------------------------------------------------------------------
DEFAULT_CFG: Dict[str, Any] = dict(
    shrink=False,          # (1-p) shrinkage measured at +8e-5 rel_l2: off by default
    gamma=0.75,
    baseline=dict(gamma0=0.6, tau_star0=0.3, damping=False, beta_mode="const",
                  beta0=0.75, U0=0.129, w=0.05, beta_fixed=False, gamma_fixed=False),
    # -- opt-in variants, all inert at their defaults -------------------------
    body_shadow=False,         # split m_core into body + shadow (in_ch 8 -> 9)
    body_mode="centroid",      # seed of the body flood fill: "centroid" | "topmost"
    body_iters=40,
    quantile_head_hidden=0,    # >0: small MLP for the 10 quantile channels only
    quantile_detach_trunk=False,  # also settable from loss.pinball_detach_trunk
    hindcast_feat=False,       # 2 static channels: log1p RMS residual of the
                               # advective baseline hindcasting the second half
                               # of the INPUT window (in_ch 8 -> 10)
    hindcast_n=T_IN // 2,      # frames used as history / hindcast horizon
    # -- wave-6 -------------------------------------
    quantile_levels=None,      # None = rpde.constants.QUANTILE_LEVELS (5 knots)
    err_head=False,            # 2 extra head channels: predicted log |residual|
    err_scale=False,           # decode hook: use exp(err_nd) as the band scale
    err_scale_k=1.0,           # multiplier on exp(err_nd) when err_scale is on
    # -- multi-speed convection bank ---------------------
    speed_bank=None,           # list of ratios on the learned gamma, e.g. [0.25, 0.5, 1.5]
    speed_bank_mode="channels",  # "channels" | "mix" | "both"
    # -- auxiliary sim-pressure target and multi-hypothesis fluctuation --
    aux_p=False,               # 1x1x1 head predicting the output frames' pressure / U^2 (out["p_nd"]); never an input
    fluct_hyp=1,               # K > 1: K fluctuation hypotheses, combined per cell energy-preserving (out["fluct_hyp"])
    fluct_fullres=False,       # with fluct_hyp > 1: ALL K guesses come from a small full-resolution conv head that
                               # also reads the advected baseline per pixel (3x3 across patch boundaries), not 1x1x1
)


class RPDEModel(nn.Module):
    """Preproc -> advective baseline -> volume -> core -> head -> physical fields."""

    def __init__(self, core_name: str = "stub",
                 core_kwargs: Optional[Dict[str, Any]] = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        from rpde.cores import build_core            # local import: avoids a cycle

        cfg = dict(DEFAULT_CFG) | dict(cfg or {})
        bl = dict(DEFAULT_CFG["baseline"]) | dict(cfg.get("baseline") or {})
        cfg["baseline"] = bl
        self.cfg = cfg
        self.core_name = core_name
        self.core_kwargs = dict(core_kwargs or {})
        self.shrink = bool(cfg.get("shrink", False))
        self.gamma = float(cfg.get("gamma", 0.75))

        self.body_shadow = bool(cfg.get("body_shadow", False))
        lv = cfg.get("quantile_levels", None)
        self.levels = tuple(float(v) for v in (QUANTILE_LEVELS if not lv else lv))
        self.err_head = bool(cfg.get("err_head", False))
        self.err_scale = bool(cfg.get("err_scale", False))
        self.err_scale_k = float(cfg.get("err_scale_k", 1.0) or 1.0)
        self.hindcast_feat = bool(cfg.get("hindcast_feat", False))
        self.hindcast_n = int(cfg.get("hindcast_n", T_IN // 2) or T_IN // 2)
        self.pre = Preproc(body_shadow=self.body_shadow,
                           body_mode=str(cfg.get("body_mode", "centroid")),
                           body_iters=int(cfg.get("body_iters", 40)))
        self.pre.cond_geom_off = bool(cfg.get("cond_geom_off", False))
        self.baseline = BaselineParams(**bl)
        # speed_bank=[r, ...] (opt-in, absent when empty): extra advective baselines at
        # ratio*gamma, as differences from the primary one (`advective_bank`). Ratio 1.0 is
        # dropped because its difference is identically zero. "channels" feeds the bank to
        # the core as 2R extra input channels (stem slice zero-init -> identical at step 0);
        # "mix" lets the head emit one coefficient per ratio, per cell and per lead, and adds
        # sum_r coeff_r * diff_r to the anchor (head zero-init -> identical at step 0).
        self.speed_ratios = tuple(float(r) for r in (cfg.get("speed_bank") or ())
                                  if abs(float(r) - 1.0) > 1e-9)
        mode = str(cfg.get("speed_bank_mode", "channels"))
        if mode not in ("channels", "mix", "both"):
            raise ValueError("speed_bank_mode must be 'channels', 'mix' or 'both', "
                             f"got {mode!r}")
        self.speed_bank_mode = mode
        nb = len(self.speed_ratios)
        self.speed_ch = nb if mode in ("channels", "both") else 0
        self.speed_mix = nb if mode in ("mix", "both") else 0
        in_ch = (9 if self.body_shadow else 8) + (2 if self.hindcast_feat else 0)
        in_ch += 2 * self.speed_ch
        self.core = build_core(core_name, in_ch=in_ch, cond_dim=8, **self.core_kwargs)
        if self.speed_ch:
            self._zero_bank_stem(in_ch)
        self.head = Head(self.core.out_channels,
                         quantile_hidden=int(cfg.get("quantile_head_hidden", 0) or 0),
                         detach_trunk=bool(cfg.get("quantile_detach_trunk", False)),
                         levels=self.levels, err_head=self.err_head,
                         mix_ch=self.speed_mix)
        # tke_head_hidden=h (opt-in, absent when 0): TKE-map head predicting the
        # future window's per-cell log energy; out["log_ehat_nd"].  The prediction path never reads it.
        self.p_head = nn.Conv3d(self.core.out_channels, 1, kernel_size=1) if cfg.get("aux_p") else None
        self.fluct_hyp = int(cfg.get("fluct_hyp", 1) or 1)
        self.fluct_fullres = bool(cfg.get("fluct_fullres", False)) and self.fluct_hyp > 1
        if self.fluct_fullres:
            C, K = self.core.out_channels, self.fluct_hyp
            self.hyp_head = nn.Sequential(
                nn.Conv3d(C + 2, 32, kernel_size=1), nn.GELU(),
                nn.Conv3d(32, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)), nn.GELU(),
                nn.Conv3d(32, 2 * K, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            self.hyp_head = (nn.Conv3d(self.core.out_channels, 2 * (self.fluct_hyp - 1), kernel_size=1)
                             if self.fluct_hyp > 1 else None)
        self.tke_head = None
        if int(cfg.get("tke_head_hidden", 0) or 0) > 0:
            from rpde.tke_map import TKEMapHead
            self.tke_head = TKEMapHead(self.core.out_channels, hidden=int(cfg["tke_head_hidden"]))
        # q_skip=C (opt-in, absent when 0): quantile-only full-resolution skip. The volume the core receives (after its
        # time-to-channel fold when the core uses one, so full 32x64 resolution before any patching) goes through
        # `q_skip_layers` per-frame 3x3 convs with C channels and a zero-initialised 1x1 projection to the 2n quantile
        # parameters, added in the head. Mean and fluctuation (the prediction) never see it.
        self.q_skip_ch = int(cfg.get("q_skip", 0) or 0)
        self.q_skip = None
        if self.q_skip_ch > 0:
            from rpde.cores.axial import HiresSkip
            self._q_t2c = bool(getattr(self.core, "t2c", False))
            self._q_tin = int(getattr(self.core, "t_in", T_IN))
            c_in = in_ch + (2 * self._q_tin if self._q_t2c else 0)
            self.q_skip = HiresSkip(c_in, self.q_skip_ch, self.head.n_q, int(cfg.get("q_skip_layers", 1) or 1))

    def _zero_bank_stem(self, in_ch: int) -> None:
        """Zero the stem weights that read the 2R bank channels.

        The bank channels are the last 2R of the first `in_ch` input channels,
        before any time-to-channel fold, so the slice is the same with and
        without `t2c`.  Zeroing them makes the model at step 0 exactly the model
        without the bank, while the gradient into that slice is non-zero from
        the first step (the channels themselves are not zero).
        """
        stem = getattr(self.core, "stem", None)
        w = getattr(stem, "weight", None)
        if w is None or w.dim() < 2:
            raise ValueError(f"core {self.core_name!r} has no stem conv: "
                             "speed_bank_mode 'channels' needs one")
        with torch.no_grad():
            w[:, in_ch - 2 * self.speed_ch:in_ch].zero_()

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, domain: Optional[torch.Tensor] = None,
                gamma: Optional[float] = None,
                clock: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        g = self.gamma if gamma is None else float(gamma)
        pp = self.pre(x, domain, clock)
        u_nd, v_nd = pp["u_nd"], pp["v_nd"]
        m_core, p_u, p_v, U = pp["m_core"], pp["p_u"], pp["p_v"], pp["U"]

        base = advective_baseline(u_nd, v_nd, m_core, pp["a"], U, self.baseline, fvalid=pp.get("fvalid"))
        bank = None
        if self.speed_ratios:
            bank = advective_bank(u_nd, v_nd, m_core, pp["a"], U, self.baseline,
                                  self.speed_ratios)               # (B,R,2,20,H,W)
        hind = None
        if self.hindcast_feat:
            hind = hindcast_residual(u_nd, v_nd, m_core, pp["a"], U, self.baseline,
                                     n_hist=self.hindcast_n)
        vol_bank = bank if self.speed_ch else None
        if self.body_shadow:
            vol = build_volume(u_nd, v_nd, base, pp["m_body"], p_u, p_v, pp["xy"],
                               m_shadow=pp["m_shadow"], hind=hind, bank=vol_bank)
        else:
            vol = build_volume(u_nd, v_nd, base, m_core, p_u, p_v, pp["xy"], hind=hind,
                               bank=vol_bank)
        feat = self.core(vol, pp["cond"])
        q_add = self._q_skip_forward(vol) if self.q_skip is not None else None
        hout = self.head(feat, q_add)
        dmean, fluct, Q = hout[0], hout[1], hout[2]
        err_nd = hout[3] if self.err_head else None
        mix = hout[-1] if self.speed_mix else None

        fcf = None
        if self.p_head is not None or self.hyp_head is not None:
            fcf = feat.permute(0, 4, 1, 2, 3) if getattr(self.head, "channels_last_in", False) else feat
            fcf = fcf[:, :, T_IN:]                                  # (B,C,20,H,W)
        fluct_hyp = None
        if self.hyp_head is not None:
            B_, _, T_, H_, W_ = fcf.shape
            if self.fluct_fullres:
                # the advected baseline per pixel (B,2,20,H,W) gives the head sub-patch structure
                hin = torch.cat([fcf, base.to(fcf.dtype)], dim=1)
                fluct_hyp = self.hyp_head(hin).view(B_, self.fluct_hyp, 2, T_, H_, W_).permute(0, 1, 3, 4, 5, 2)
                fluct_hyp = fluct_hyp - fluct_hyp.mean(dim=2, keepdim=True)
            else:
                ex = self.hyp_head(fcf).view(B_, self.fluct_hyp - 1, 2, T_, H_, W_).permute(0, 1, 3, 4, 5, 2)
                ex = ex - ex.mean(dim=2, keepdim=True)              # zero temporal mean, like the head's fluct
                fluct_hyp = torch.cat([fluct.unsqueeze(1), ex], dim=1)  # (B,K,20,H,W,2)
            fl_avg = fluct_hyp.mean(dim=1)
            ek = fluct_hyp.float().var(dim=2, unbiased=False).sum(-1).mean(dim=1)     # (B,H,W)
            ef = fl_avg.float().var(dim=1, unbiased=False).sum(-1)
            gcell = torch.where(ef > 1e-12, ek / ef.clamp(min=1e-12), torch.ones_like(ef)).clamp(1.0, 16.0)
            fluct = fl_avg * gcell.sqrt().to(fl_avg.dtype)[:, None, :, :, None]
        base_nd = base.permute(0, 2, 3, 4, 1).contiguous()      # (B,20,H,W,2)
        if mix is not None:
            bnd = bank.permute(0, 3, 4, 5, 2, 1)                # (B,20,H,W,2,R)
            base_nd = base_nd + (bnd * mix.unsqueeze(-2)).sum(dim=-1)
        mean_nd = base_nd + dmean
        Ub = U.view(-1, 1, 1, 1, 1)
        mk = (1.0 - m_core).permute(0, 2, 3, 1).unsqueeze(1)    # (B,1,H,W,1)

        mean2 = mean_nd * Ub * mk
        pred2 = (mean_nd + g * fluct) * Ub * mk
        if self.shrink:
            keep = torch.cat([1.0 - p_u, 1.0 - p_v], dim=1)     # (B,2,H,W)
            keep = keep.permute(0, 2, 3, 1).unsqueeze(1)        # (B,1,H,W,2)
            pred2 = pred2 * keep

        zeros = torch.zeros_like(mean2[..., :1])
        mean = torch.cat([mean2, zeros], dim=-1)                # (B,20,H,W,3)
        pred = torch.cat([pred2, torch.zeros_like(zeros)], dim=-1)

        out = dict(mean_nd=mean_nd, fluct_nd=fluct, Q_nd=Q, base_nd=base_nd,
                   mean=mean, pred=pred, U=U, m_core=m_core, p_u=p_u, p_v=p_v,
                   cond=pp["cond"], levels=self.levels)
        if err_nd is not None:
            out["err_nd"] = err_nd
        if fluct_hyp is not None:
            out["fluct_hyp"] = fluct_hyp
        if self.p_head is not None:
            out["p_nd"] = self.p_head(fcf)[:, 0]                  # (B,20,H,W)
        if self.tke_head is not None:
            from rpde.tke_map import energy_nd
            fh = feat.permute(0, 4, 1, 2, 3) if getattr(self.head, "channels_last_in", False) else feat
            e_model = energy_nd(((mean_nd + g * fluct) * mk).detach())
            e_in = energy_nd(torch.stack([u_nd, v_nd], -1).detach())
            out["log_ehat_nd"] = self.tke_head(fh, e_model, e_in)
        return out

    def _q_skip_forward(self, vol: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, 40, H, W) -> (B, 2n, 20, H, W) quantile-parameter correction at full resolution."""
        if self._q_t2c:
            B, _, T, H, W = vol.shape
            t = self._q_tin
            past = vol[:, :2, :t].reshape(B, 2 * t, 1, H, W).expand(-1, -1, T - t, -1, -1)
            v = torch.cat([vol[:, :, t:], past], dim=1)
        else:
            v = vol[:, :, T_IN:]
        return self.q_skip(v)

    # -- decode helpers ----------------------------------------------------
    def decode_err_scale(self, out: Dict[str, torch.Tensor]):
        """`exp(err_nd) * err_scale_k` when `decode.err_scale` is on, else None.

        Variant A's second arm: the auxiliary head's predicted residual
        magnitude replaces `(q75-q25)/1.349` as the scale fed to the reward
        argmax; the decode itself is unchanged.
        """
        if not self.err_scale or "err_nd" not in out:
            return None
        return torch.exp(out["err_nd"].float()) * self.err_scale_k

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def predict_with_bounds(self, x: torch.Tensor, gamma: Optional[float] = None,
                            spread: Optional[torch.Tensor] = None,
                            fast: bool = False, calib=None):
        """(pred, lower, upper) in physical units.

        `fast=True` runs the measured fast-inference configuration of
        `rpde.fastinf.FastInference` (TF32 + bf16 autocast + the fused axial
        path + the vectorised decode, all with the same weights); see
        It falls back to this eager path on CPU or
        if anything in the fast path raises.

        `calib` may be a `rpde.calibrate.CalibratedDecode`; it then replaces
        `band_from_quantiles` on either path (and `spread` is ignored, the
        calibrated decode carries its own scale transform).
        """
        from rpde.decode import band_from_quantiles

        if fast and not getattr(self, "_fastinf_disabled", False):
            try:
                from rpde.fastinf import FastInference
                fi = getattr(self, "_fastinf", None)
                if fi is None or fi.model is not self:
                    fi = FastInference(self)
                    self._fastinf = fi
                return fi.predict_with_bounds(x, gamma=gamma, spread=spread,
                                              calib=calib)
            except Exception as exc:                     # pragma: no cover
                # one warning, then this model stays on the eager path
                self._fastinf = None
                self._fastinf_disabled = True
                print(f"[fastinf] disabled, falling back to eager: {exc!r}")

        out = self(x, gamma=gamma)
        es = self.decode_err_scale(out)
        if calib is not None:
            lower, upper = calib.apply(
                out["Q_nd"].float(), out["mean"].float(), out["U"].float(),
                out["m_core"].float(), x=x, pred=out["pred"].float(),
                levels=self.levels, err_scale=es)
        else:
            lower, upper = band_from_quantiles(
                out["Q_nd"], out["mean"], out["U"], out["m_core"], spread=spread,
                levels=self.levels, err_scale=es,
            )
        return out["pred"], lower, upper
