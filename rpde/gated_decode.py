"""S9 decode: gated per-lead fluctuation decay + gated mean-shrink, applied to the prediction only (band untouched).

Both corrections act only on windows that lie beyond the training data in Re, judged from the input alone:
  Re_raw  = U_inf / U_PER_RE                                   (U_inf = the model's own freestream estimate)
  AoA_est = nearest centroid of the core-mask geometry (cx, cy, ex, ey) = cond[:, 3:7], standardised by `sd`
  excess  = (Re_raw - top[AoA_est]) / 1275                     (grid steps above the largest training Re at that AoA)
  decay   : excess in (decay_lo, decay_hi)  ->  pred' = M + c_k (pred - M),  c_k = 1 + c1 * k / 19
            every OTHER window gets the ungated decay c_out_k = 1 + c1_out * k / 19 (c1_out = 0 -> S9 behaviour)
  shrink  : excess > 0                       ->  pred'' = pred' - lam * (P - W) outside the core,
            P = temporal mean of the ORIGINAL prediction, W = temporal mean of the measured input frames,
            lam = lam for excess < decay_hi and lam_far beyond it (lam_far = lam -> S9 behaviour)
Validated offline on held-out prediction dumps.
Pure tensor ops, no data-dependent control flow -> exportable (AOTInductor). Submission-safe: torch only.
"""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from rpde.constants import U_PER_RE

__all__ = ["S9_PARAMS", "S12_PARAMS", "gated_decode"]

# Full-corpus tables (all 81 real recordings minus BAD_REAL): the column tops and the AoA-geometry centroids
# (mean cond[3:7] per AoA over the 3 dumped windows of every recording) + the per-feature std over those windows.
# Fitted offline on the training regimes (per-AoA mask centroids and the top training Re per AoA).
S9_PARAMS: Dict[str, object] = dict(
    aoa_keys=[0, 5, 10, 15, 20],
    tops=[26700.0, 21600.0, 26700.0, 26700.0, 25425.0],
    centroids=[[0.10816839, 0.74321777, 0.31361237, 0.51906127], [0.1081567, 0.72691208, 0.32690847, 0.53917056], [0.09397534, 0.70970106, 0.31216949, 0.55286729], [0.09518676, 0.71980542, 0.32413173, 0.5811128], [0.0908691, 0.68951124, 0.31613749, 0.63575274]],   # (5, 4)
    sd=[0.00957239, 0.02040497, 0.01088485, 0.03987769],   # (4,)
    c1=-0.6, decay_lo=0.0, decay_hi=1.75,
    lam=0.15,
    re_step=1275.0,
)

# S12 decode: the S9 tables plus two tweaks (fitted offline).
# A: a larger shrink beyond the decay gate (lam_far 0.25) -> R-corner +0.015.  B: a small ungated lead decay on the
# windows the gate leaves out (c1_out -0.2) -> in-regime +0.010/+0.011, R-mid-a +0.015, R-corner -0.007.
# Together: R-top unchanged (the gate never fires there), R-corner +0.008, in-regime +0.010, R-mid-a +0.015.
S12_PARAMS: Dict[str, object] = dict(S9_PARAMS, lam_far=0.25, c1_out=-0.2)


def gated_decode(pred: torch.Tensor, mean: torch.Tensor, x: torch.Tensor, U: torch.Tensor, m_core: torch.Tensor,
                 cond: torch.Tensor, params: Dict[str, object]) -> torch.Tensor:
    """pred, mean: (B,20,H,W,3) physical; x: (B,20,H,W,3) input; U: (B,); m_core: (B,1,H,W) or (B,H,W); cond: (B,8)."""
    dev, dt = pred.device, pred.dtype
    cents = torch.as_tensor(params["centroids"], device=dev, dtype=dt)             # (K,4)
    sd = torch.as_tensor(params["sd"], device=dev, dtype=dt)                       # (4,)
    tops = torch.as_tensor(params["tops"], device=dev, dtype=dt)                   # (K,)
    feat = cond[:, 3:7].to(dt)                                                     # (B,4)
    d2 = (((feat[:, None, :] - cents[None]) / sd) ** 2).sum(-1)                    # (B,K)
    top = tops[d2.argmin(1)]                                                       # (B,)
    excess = (U.to(dt) / U_PER_RE - top) / float(params["re_step"])
    g_dec = ((excess > float(params["decay_lo"])) & (excess < float(params["decay_hi"]))).to(dt)
    g_shr = (excess > 0).to(dt)

    T = pred.shape[1]
    k = torch.arange(T, device=dev, dtype=dt) / max(T - 1, 1)
    c = 1.0 + float(params["c1"]) * k                                              # (T,) inside the gate
    c_out = 1.0 + float(params.get("c1_out", 0.0)) * k                             # (T,) outside it
    gc = g_dec[:, None] * c[None, :] + (1.0 - g_dec)[:, None] * c_out[None, :]     # (B,T)
    f = pred - mean
    out = mean + gc[:, :, None, None, None] * f

    mk = m_core.reshape(m_core.shape[0], m_core.shape[-2], m_core.shape[-1]).to(dt)
    xv = x[..., :2].to(dt)
    val = (xv != 0).to(dt)
    W = (xv * val).sum(1) / val.sum(1).clamp(min=1.0)                             # (B,H,W,2)
    P = pred[..., :2].mean(1)                                                      # original prediction
    shift = (P - W) * (1.0 - mk)[..., None]
    lam = float(params["lam"])
    lam_far = float(params.get("lam_far", lam))
    lam_e = g_shr * torch.where(excess < float(params["decay_hi"]),
                                torch.full_like(excess, lam), torch.full_like(excess, lam_far))
    uv = out[..., :2] - lam_e[:, None, None, None, None] * shift[:, None]
    return torch.cat([uv, torch.zeros_like(out[..., 2:])], dim=-1)
