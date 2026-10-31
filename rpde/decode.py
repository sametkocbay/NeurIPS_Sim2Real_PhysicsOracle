"""Reward-maximising decode of the quantile head into an SPS band.

Ships in the submission -> torch + numpy + stdlib + rpde.constants only.
Band decode: quantile head -> (lower, upper) that maximises the expected SPS reward.

The scorer's per-element reward is `exp(-w/sigma) * 1{lower <= t <= upper}`
with `sigma = SIGMA_GLOBAL` in PHYSICAL units, so the width grid lives in
physical units and is converted to the per-sample nondimensional scale by
dividing by U.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from rpde.constants import QUANTILE_LEVELS, QUANTILE_LEVELS_9, SIGMA_GLOBAL

__all__ = ["band_from_quantiles", "default_width_grid", "cdf_from_quantiles",
           "argmax_width_vec", "argmax_width_coarse_fine", "resolve_levels", "levels_tensor", "iqr_scale",
           "quantile_at", "rescale_offsets", "median_index"]

_EPS = 1e-6
_MASK_BAND = 1e-3

# ---------------------------------------------------------------------------
# quantile levels
# ---------------------------------------------------------------------------
# The head's level set is a model config parameter (`model.quantile_levels`).
# Everything downstream takes `levels` as an argument; when it is not given the
# set is recovered from the number of knots, which covers both shipped sets and
# keeps every old call site (submission code included) working unchanged.
_KNOWN_LEVELS = {len(QUANTILE_LEVELS): QUANTILE_LEVELS,
                 len(QUANTILE_LEVELS_9): QUANTILE_LEVELS_9}


def resolve_levels(K: int, levels=None) -> Tuple[float, ...]:
    """The K quantile levels as a plain python tuple (no device sync)."""
    if levels is None:
        lv = _KNOWN_LEVELS.get(int(K))
        if lv is None:
            raise ValueError(
                f"{K} quantile knots: pass `levels=` explicitly "
                f"(known sets: {sorted(_KNOWN_LEVELS)})")
        return lv
    if torch.is_tensor(levels):
        levels = levels.detach().flatten().tolist()
    lv = tuple(float(v) for v in levels)
    if len(lv) != int(K):
        raise ValueError(f"{len(lv)} levels for {K} quantile knots")
    return lv


def levels_tensor(levels, device=None, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(levels):
        return levels.to(device=device, dtype=dtype)
    return torch.tensor(tuple(float(v) for v in levels), device=device, dtype=dtype)


def median_index(levels) -> int:
    """Index of the 0.5 level (the head builds the band outward from it)."""
    lv = tuple(float(v) for v in levels)
    for i, v in enumerate(lv):
        if abs(v - 0.5) < 1e-12:
            return i
    raise ValueError(f"quantile levels must contain 0.5, got {lv}")


def _interp_weights(levels: Tuple[float, ...], tau: float):
    """(i, j, w) with q_tau = (1-w)*Q[i] + w*Q[j]; (i, i, 0.0) on an exact knot."""
    for i, v in enumerate(levels):
        if abs(v - tau) < 1e-12:
            return i, i, 0.0
    if tau <= levels[0]:
        return 0, 0, 0.0
    if tau >= levels[-1]:
        return len(levels) - 1, len(levels) - 1, 0.0
    j = next(k for k, v in enumerate(levels) if v > tau)
    i = j - 1
    return i, j, (tau - levels[i]) / (levels[j] - levels[i])


def quantile_at(Q: torch.Tensor, levels, tau: float) -> torch.Tensor:
    """The tau-quantile of the knot set, interpolating when tau is not a knot."""
    lv = tuple(float(v) for v in levels)
    i, j, w = _interp_weights(lv, float(tau))
    if i == j:
        return Q[..., i]
    return Q[..., i] + (Q[..., j] - Q[..., i]) * w


def iqr_scale(Q: torch.Tensor, levels=None, eps: float = _EPS) -> torch.Tensor:
    """`s = (q_0.75 - q_0.25) / 1.349` -- the head's predicted scale.

    With the default five levels this is exactly `(Q[...,3]-Q[...,1])/1.349`
    (bit-identical); with any other set the 0.25/0.75 quantiles are taken from
    the knots, interpolating linearly when they are not knots themselves.
    """
    lv = resolve_levels(Q.shape[-1], levels)
    q25 = quantile_at(Q, lv, 0.25)
    q75 = quantile_at(Q, lv, 0.75)
    return ((q75 - q25) / 1.349).clamp(min=eps)


def rescale_offsets(Q: torch.Tensor, med: torch.Tensor,
                    fac: torch.Tensor) -> torch.Tensor:
    """Scale the quantile offsets about the median by `fac` (..., broadcast)."""
    return med.unsqueeze(-1) + (Q - med.unsqueeze(-1)) * fac.unsqueeze(-1)


def default_width_grid(device=None, dtype=torch.float32, n: int = 32,
                       lo: float = 0.02, hi: float = 2.0) -> torch.Tensor:
    """`n` candidate widths in physical units, from lo*sigma to hi*sigma.

    Geometric spacing: the reward `exp(-w/sigma)*cov(w)` is steep at small w and
    that is where nearly all of the SPS score lives, so log spacing resolves the
    optimum far better than a linear grid at the same cost.
    """
    return SIGMA_GLOBAL * torch.logspace(
        math.log10(lo), math.log10(hi), n, device=device, dtype=dtype)


def cdf_from_quantiles(z: torch.Tensor, Q: torch.Tensor,
                       levels: torch.Tensor) -> torch.Tensor:
    """tau(z) for the piecewise-linear quantile function through (levels, Q).

    z : (..., ) query, Q : (..., K) knots (non-decreasing), levels : (K,).
    Linear interpolation between knots, linear extrapolation beyond the outer
    knots with the outer slopes, clipped to [0, 1].

    Written as a sum of ramps:
        tau(z) = t0 + s0*(z-q0) + sum_{i=1..K-2} (s_i - s_{i-1}) * relu(z - q_i)
    which reproduces the interior segments exactly and keeps slope s0 below q0
    and s_{K-2} above q_{K-1}.
    """
    K = Q.shape[-1]
    gaps = (Q[..., 1:] - Q[..., :-1]).clamp(min=_EPS)           # (..., K-1)
    dlev = (levels[1:] - levels[:-1])                            # (K-1,)
    slopes = dlev / gaps                                         # (..., K-1)
    tau = levels[0] + slopes[..., 0] * (z - Q[..., 0])
    for i in range(1, K - 1):
        tau = tau + (slopes[..., i] - slopes[..., i - 1]) * (z - Q[..., i]).clamp(min=0.0)
    return tau.clamp(0.0, 1.0)


def _tau_lin(z: torch.Tensor, Q: torch.Tensor, slopes: torch.Tensor,
             levels: torch.Tensor) -> torch.Tensor:
    """`cdf_from_quantiles` with the slopes hoisted out (identical arithmetic)."""
    K = Q.shape[-1]
    tau = levels[0] + slopes[..., 0] * (z - Q[..., 0])
    for i in range(1, K - 1):
        tau = tau + (slopes[..., i] - slopes[..., i - 1]) * (z - Q[..., i]).clamp(min=0.0)
    return tau.clamp(0.0, 1.0)


def argmax_width_vec(Q: torch.Tensor, z0: torch.Tensor, Ub: torch.Tensor,
                     levels: torch.Tensor, width_grid: torch.Tensor,
                     sigma: float) -> torch.Tensor:
    """Vectorised SPS-reward argmax over the width grid -> width (nondimensional).

    Bit-exact replacement for the ``for j in range(width_grid.numel())`` loop of
    `band_from_quantiles` / `CalibratedDecode._argmax_width`:

    * the piecewise-linear slopes are computed once instead of once per width
      and per band edge (the loop recomputes them 2*n times);
    * all `n` widths are evaluated in one batch of kernels, turning ~13*2*n tiny
      launches into ~13*2 launches over an `n`-times larger tensor;
    * the loop keeps the FIRST width that attains the maximum reward
      (`take = r > best_r`) and the grid is strictly increasing, so the same
      choice is `argmin_w {w : r(w) = max_r}` -- taken here with a masked `amin`,
      which is deterministic and needs no tie-breaking convention from `argmax`.

    Q  : (..., K) quantile knots        z0 : (...,) band centre
    Ub : (B,1,1,1,1) freestream         width_grid : (n,) PHYSICAL widths
    """
    gaps = (Q[..., 1:] - Q[..., :-1]).clamp(min=_EPS)
    slopes = ((levels[1:] - levels[:-1]) / gaps).unsqueeze(0)
    Qe = Q.unsqueeze(0)
    n = int(width_grid.numel())
    w_ph = width_grid.reshape((n,) + (1,) * z0.dim())        # (n,1,...,1)
    w_nd = w_ph / Ub.unsqueeze(0)                            # (n,B,1,1,1,1)
    half = 0.5 * w_nd
    z = z0.unsqueeze(0)
    cov = (_tau_lin(z + half, Qe, slopes, levels)
           - _tau_lin(z - half, Qe, slopes, levels)).clamp(0.0, 1.0)
    r = torch.exp(-w_ph / sigma) * cov                       # (n,B,T,H,W,2)
    best_r = r.amax(dim=0, keepdim=True)
    big = torch.full((), float("inf"), device=r.device, dtype=r.dtype)
    best_w = torch.where(r >= best_r, w_nd.expand_as(r), big).amin(dim=0)
    # the loop starts from best_w = 0 and never updates if no width ever wins
    return torch.where(torch.isfinite(best_w), best_w, torch.zeros_like(best_w))


def argmax_width_coarse_fine(Q: torch.Tensor, z0: torch.Tensor, Ub: torch.Tensor, levels: torch.Tensor,
                             width_grid: torch.Tensor, sigma: float, n_fine: int = 8) -> torch.Tensor:
    """Two-stage width search: `n_coarse` widths, then `n_fine` between the winner's neighbours.

    The reward `exp(-w/sigma) * cov(w)` is unimodal in w (a decreasing exponential times a non-decreasing
    coverage), so the optimum lies between the coarse winner's two neighbours and a local refinement finds it.
    Cost is n_coarse + n_fine evaluations instead of 32, and the refinement is FINER than the 32-point grid
    there, which is why the result is not merely cheaper (measured +0.025 sps for a
    128-point grid, i.e. the 32-point one is slightly under-resolved).
    """
    gaps = (Q[..., 1:] - Q[..., :-1]).clamp(min=_EPS)
    slopes = ((levels[1:] - levels[:-1]) / gaps).unsqueeze(0)
    Qe = Q.unsqueeze(0)
    z = z0.unsqueeze(0)

    def reward(w_ph):                                   # w_ph: (n,1,...,1) physical widths
        w_nd = w_ph / Ub.unsqueeze(0)
        half = 0.5 * w_nd
        cov = (_tau_lin(z + half, Qe, slopes, levels)
               - _tau_lin(z - half, Qe, slopes, levels)).clamp(0.0, 1.0)
        return torch.exp(-w_ph / sigma) * cov, w_nd

    # the COARSE pass must be coarse: take every `stride`-th width of the grid, so the two passes cost
    # n_coarse + n_fine evaluations (8 + 8 = 16) against the full grid's 32.
    stride = max(1, int(width_grid.numel()) // 8)
    coarse = width_grid[::stride]
    n = int(coarse.numel())
    w_ph = coarse.reshape((n,) + (1,) * z0.dim())
    r, w_nd = reward(w_ph)
    idx = r.argmax(dim=0, keepdim=True)                                  # (1,B,T,H,W,2)
    lo_i = (idx - 1).clamp(min=0)
    hi_i = (idx + 1).clamp(max=n - 1)
    g = coarse.reshape((n,) + (1,) * z0.dim()).expand_as(r)
    w_lo = torch.gather(g, 0, lo_i)                                      # bracket, physical
    w_hi = torch.gather(g, 0, hi_i)
    t = torch.linspace(0.0, 1.0, n_fine, device=Q.device, dtype=Q.dtype).reshape((n_fine,) + (1,) * z0.dim())
    w_f = w_lo + (w_hi - w_lo) * t                                       # (n_fine,B,T,H,W,2) physical
    r_f, w_f_nd = reward(w_f)
    best = r_f.argmax(dim=0, keepdim=True)
    return torch.gather(w_f_nd, 0, best).squeeze(0)


def band_from_quantiles(Q_nd: torch.Tensor, mean: torch.Tensor, U: torch.Tensor,
                        m_core: torch.Tensor, width_grid: Optional[torch.Tensor] = None,
                        spread: Optional[torch.Tensor] = None,
                        sigma: float = SIGMA_GLOBAL,
                        fast: bool = False,
                        levels=None,
                        err_scale: Optional[torch.Tensor] = None,
                        checks: bool = True,
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantiles of the nondimensional residual -> physical (lower, upper).

    Q_nd    : (B, 20, H, W, 2, 5)  quantiles of (target_nd - mean_nd)
    mean    : (B, 20, H, W, 3)     physical mean field
    U       : (B,)                 freestream, physical
    m_core  : (B, 1, H, W)
    width_grid : optional (n,) candidate widths in PHYSICAL units
    spread  : optional (B, 20, H, W, 2) extra scale in the SAME (nondimensional)
              units as Q_nd; combined as s_eff^2 = s^2 + spread^2 by rescaling
              the quantile offsets about the median by s_eff / s.
    fast    : evaluate the width grid with `argmax_width_vec` (one batch of
              kernels instead of a python loop). Bit-exact, opt-in.
              "coarse_fine": two-stage search (see `argmax_width_coarse_fine`) -- cheaper AND finer, but NOT
              bit-exact against the 32-point grid.
    levels  : the head's quantile levels; inferred from the knot count when
              None (5 -> QUANTILE_LEVELS, 9 -> QUANTILE_LEVELS_9).
    err_scale : optional (B, 20, H, W, 2) scale, nondimensional, that REPLACES
              the head's `(q75-q25)/1.349` before the argmax (variant A's
              `decode.err_scale`); applied before `spread`.
    checks  : assert finite and ordered bounds (two host syncs). The bounds are finite
              and ordered by construction; False for graph export.

    Returns (lower, upper), both (B, 20, H, W, 3), physical, finite, ordered.
    """
    dev, dt = Q_nd.device, Q_nd.dtype
    lv = resolve_levels(Q_nd.shape[-1], levels)
    levels = levels_tensor(lv, device=dev, dtype=dt)
    med = Q_nd[..., median_index(lv)]                           # (B,20,H,W,2)

    if err_scale is not None:
        s = iqr_scale(Q_nd, lv)
        Q_nd = rescale_offsets(Q_nd, med, err_scale.to(dt).clamp(min=_EPS) / s)
    if spread is not None:
        s = iqr_scale(Q_nd, lv)
        s_eff = torch.sqrt(s * s + spread.to(dt) ** 2)
        Q_nd = rescale_offsets(Q_nd, med, s_eff / s)

    if width_grid is None:
        width_grid = default_width_grid(device=dev, dtype=dt)
    else:
        width_grid = torch.as_tensor(width_grid, device=dev, dtype=dt).reshape(-1)

    Ub = U.to(dt).view(-1, 1, 1, 1, 1)
    if fast == "coarse_fine":
        best_w = argmax_width_coarse_fine(Q_nd, med, Ub, levels, width_grid, sigma)
    elif fast:
        best_w = argmax_width_vec(Q_nd, med, Ub, levels, width_grid, sigma)
    else:
        best_r = torch.full_like(med, -1.0)
        best_w = torch.zeros_like(med)                          # nondimensional width
        for j in range(width_grid.numel()):
            w_ph = width_grid[j]
            w_nd = w_ph / Ub                                    # (B,1,1,1,1)
            half = 0.5 * w_nd
            cov = (cdf_from_quantiles(med + half, Q_nd, levels)
                   - cdf_from_quantiles(med - half, Q_nd, levels)).clamp(0.0, 1.0)
            r = torch.exp(-w_ph / sigma) * cov
            take = r > best_r
            best_r = torch.where(take, r, best_r)
            best_w = torch.where(take, half * 2.0 + torch.zeros_like(best_w), best_w)

    half = 0.5 * best_w
    lo2 = mean[..., :2] + (med - half) * Ub
    hi2 = mean[..., :2] + (med + half) * Ub

    # core-masked cells: harmless finite band (unscored)
    mk = m_core.to(dt).permute(0, 2, 3, 1).unsqueeze(1)         # (B,1,H,W,1)
    lo2 = torch.where(mk > 0, torch.full_like(lo2, -_MASK_BAND), lo2)
    hi2 = torch.where(mk > 0, torch.full_like(hi2, _MASK_BAND), hi2)

    zeros = torch.zeros_like(lo2[..., :1])
    lower = torch.cat([lo2, zeros], dim=-1)
    upper = torch.cat([hi2, torch.zeros_like(zeros)], dim=-1)

    lower = torch.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    upper = torch.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
    upper = torch.maximum(upper, lower)
    if not checks:
        return lower, upper
    if not (torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise AssertionError("decode produced non-finite bounds")
    if bool((lower > upper).any()):
        raise AssertionError("decode produced lower > upper")
    return lower, upper
