"""Training losses (spec 3.8).  Not shipped, but kept torch-only anyway.

    L = relL2(mean) + relL2(mean.detach() + gamma*fluct) + lambda_pin * L_pin + lambda_tke * L_tke
"""
from __future__ import annotations

from typing import Dict

import torch

from rpde.constants import CHORD, DT, QUANTILE_LEVELS, SIGMA_GLOBAL
from rpde.decode import _tau_lin, default_width_grid, levels_tensor, median_index

__all__ = ["compute_losses", "tke_field", "tke_field_masked", "horizon_frames",
           "pinball", "sps_reward_loss", "PROBE_COLS", "PROBE_ROWS", "TAU_STAR_DECORR"]

_EPS = 1e-8

# The scorer's mvpe probes at 32x64 (scoring.py::mvpe_rel_l2_per_sample with
# sub_s_real=2): probe_x = int(((i+1)*16 + 10)/2) for i = 0..3 and
# probe_y = 16 + 2*j for j = -4..4.
PROBE_COLS = (13, 21, 29, 37)
PROBE_ROWS = (8, 10, 12, 14, 16, 18, 20, 22, 24)


def tke_field(vel: torch.Tensor) -> torch.Tensor:
    """0.5*(var_t(u) + var_t(v)) over the time axis, population variance.

    Mirrors `scoring.py::kinetic_energy` exactly (mean of squared deviations
    from the temporal mean, no Bessel correction).
    vel: (B, T, H, W, 2) -> (B, H, W)
    """
    d = vel - vel.mean(dim=1, keepdim=True)
    ke = (d * d).mean(dim=1)                                    # (B,H,W,2)
    return 0.5 * (ke[..., 0] + ke[..., 1])


# 1/e decorrelation time of the fluctuating field in convective units,
# tau* = tau_frames * dt*, median over the real corpus (data analysis:
# 0.2845, running 0.231 at Re 3750 to 0.446 at Re 26700).
TAU_STAR_DECORR = 0.285


def horizon_frames(U: torch.Tensor, c: float = 2.0, T: int = 20,
                   tau_star: float = TAU_STAR_DECORR,
                   kmin: int = 4) -> torch.Tensor:
    """Per-sample TKE horizon K in frames.

        tau_frames = tau* * CHORD / (U_inf * DT)
        K          = clip(round(c * tau_frames), kmin, T)

    U: (B,) free-stream speed in m/s -> (B,) float K.
    """
    tau_frames = float(tau_star) * CHORD / (U * DT).clamp(min=1e-8)
    return torch.round(float(c) * tau_frames).clamp(float(kmin), float(T))


def tke_field_masked(vel: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`tke_field` restricted to the frames where ``w == 1``.

    vel: (B, T, H, W, 2); w: (B, T, 1, 1, 1) 0/1 -> (B, H, W).  With ``w``
    all-ones this is exactly ``tke_field`` (population variance over the
    selected frames, no Bessel correction), matching `scoring.py::kinetic_energy`.
    """
    n = w.sum(dim=1, keepdim=True).clamp(min=1.0)               # (B,1,1,1,1)
    mu = (vel * w).sum(dim=1, keepdim=True) / n
    d = (vel - mu) * w
    ke = (d * d).sum(dim=1) / n[:, 0]                           # (B,H,W,2)
    return 0.5 * (ke[..., 0] + ke[..., 1])


def pinball(err: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball loss of `err = y - q` at level tau (elementwise)."""
    return torch.maximum(tau * err, (tau - 1.0) * err)


# variant A: eps of the log residual magnitude, nondimensional
ERR_EPS = 1e-3


def sps_reward_loss(Q_nd: torch.Tensor, resid: torch.Tensor, valid: torch.Tensor,
                    n_valid: torch.Tensor, U: torch.Tensor, levels,
                    tau: float = 0.02, soft_temp: float = 0.005,
                    sigma: float = SIGMA_GLOBAL,
                    width_grid: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """Straight-through SPS band reward, as
    a training term:  L_sps = -mean_valid( reward of the band the decode ships ).

    Value = the scorer's exact per-element band reward `exp(-w/sigma) *
    1{|r - c| <= w/2}` for the decode's hard argmax width (`argmax_width_vec`
    arithmetic, uncalibrated); gradient = the smoothed version (softmax over the
    width grid at `soft_temp`, sigmoid coverage at `tau`), each factor written
    `hard + (soft - soft.detach())`.  `resid` must be built from a DETACHED
    mean, so the gradient reaches only the quantile knots (and the trunk
    through them) -- never the mean head.
    """
    dt = Q_nd.dtype
    lv = levels_tensor(levels, device=Q_nd.device, dtype=dt)
    med = Q_nd[..., median_index(levels)]                       # (B,T,H,W,2)
    d = (resid - med).abs()
    Ub = U.to(dt).view(-1, 1, 1, 1, 1)
    if width_grid is None:
        width_grid = default_width_grid(device=Q_nd.device, dtype=dt)
    gaps = (Q_nd[..., 1:] - Q_nd[..., :-1]).clamp(min=1e-6)
    slopes = ((lv[1:] - lv[:-1]) / gaps).unsqueeze(0)
    Qe = Q_nd.unsqueeze(0)
    n = int(width_grid.numel())
    w_ph = width_grid.reshape((n,) + (1,) * med.dim())          # (n,1,...,1)
    half = 0.5 * (w_ph / Ub.unsqueeze(0))                       # nondimensional half width
    z = med.unsqueeze(0)
    cov_pred = (_tau_lin(z + half, Qe, slopes, lv)
                - _tau_lin(z - half, Qe, slopes, lv)).clamp(0.0, 1.0)
    decay = torch.exp(-w_ph / float(sigma))
    R_pred = decay * cov_pred                                   # (n,B,T,H,W,2)
    p_soft = torch.softmax(R_pred / float(soft_temp), dim=0)
    # the decode keeps the FIRST (narrowest) width attaining the max; the grid is
    # increasing so argmax over dim 0 (first max) matches `argmax_width_vec`.
    hard_p = torch.zeros_like(p_soft).scatter_(
        0, R_pred.detach().argmax(dim=0, keepdim=True), 1.0)
    p = hard_p + (p_soft - p_soft.detach())
    dd = d.unsqueeze(0)
    cov_soft = torch.sigmoid((half - dd) / float(tau))
    cov_hard = (dd <= half).to(dt)
    cov_true = cov_hard + (cov_soft - cov_soft.detach())
    elem = (p * decay * cov_true).sum(dim=0)                    # (B,T,H,W,2)
    L_sps = -(elem * valid).sum() / n_valid
    with torch.no_grad():
        w_sel = (hard_p * (2.0 * half)).sum(dim=0)              # chosen width, nondim
        cov_sel = (hard_p * cov_hard).sum(dim=0)
        stats = dict(sps_reward=-L_sps.detach(),
                     sps_cov=(cov_sel * valid).sum() / n_valid,
                     sps_w_sigma=((w_sel * Ub) * valid).sum() / n_valid / float(sigma))
    return dict(L_sps=L_sps, **stats)


def compute_losses(out: Dict[str, torch.Tensor], y: torch.Tensor, gamma: float,
                   lambda_pin: float = 0.1, lambda_tke: float = 0.2,
                   lambda_tmean: float = 0.0,
                   lambda_probe: float = 0.0,
                   tke_horizon: str = "flat",
                   tke_horizon_c: float = 2.0,
                   lambda_tke_full: float | None = None,
                   lambda_err: float = 0.0,
                   lambda_sps: float = 0.0,
                   sps_tau: float = 0.02,
                   sps_temp: float = 0.005,
                   lambda_var: float = 0.0,
                   var_eps: float = 1e-4,
                   var_min_frames: int = 8,
                   var_weight: str = "uniform") -> Dict[str, torch.Tensor]:
    """out: RPDEModel.forward output; y: (B, 20, H, W, 3) physical target.

    Returns {"total", "loss", "L_mean", "L_pin", "L_tke", "n_valid"}; "loss" is
    an alias of "total".

    Two opt-in mean-field terms,
    both weighted 0 by default so the returned dict and the total are unchanged:

    * `lambda_tmean` * `L_tmean`: per-window relative L2 between the 20-frame
      TIME-MEAN of `mean_nd` and of `target_nd`, over valid cells only (the
      time mean of each is taken over that cell/channel's valid frames).
    * `lambda_probe` * `L_probe`: the mvpe surrogate -- at the scorer's four
      probe columns and nine probe rows, the per-column relative L2 between the
      time means (exactly `scoring.py::mvpe_rel_l2_per_sample`'s reduction, on
      `mean_nd` instead of `pred`), averaged over the columns and the batch.

    Both flow into the mean head (and through it into the trunk).

    `lambda_err` adds the auxiliary
    error-magnitude term: L1 between the head's `err_nd` and
    `log(|target_nd - mean_nd.detach()| + ERR_EPS)` on valid cells.  The
    gradient flows into the new channels AND the trunk (that is the point --
    the trunk must encode its own error magnitude); the target is detached, so
    the mean head is never pushed to change its own residual.

    `lambda_var` (default 0, benchmark_S7final lever 1) adds the per-cell
    log-variance match: mean over valid cells and channels of
    (log(var_20(pred_nd) + eps) - log(var_20(target_nd) + eps))^2, variance
    over the frames where the cell is valid (cells with fewer than
    `var_min_frames` valid frames are skipped).  Scale-free per cell, so every
    cell counts the same whatever its energy -- the spatial SHAPE of the
    injected fluctuation energy, which the global TKE norm does not see.
    Gradient into the fluctuation head only (mean detached, as in L_pred).

    `lambda_sps` adds the straight-through
    SPS band reward `sps_reward_loss` on `target_nd - mean_nd.detach()`; the
    gradient reaches the quantile knots and the trunk, never the mean head.
    `sps_tau` / `sps_temp` are the coverage-sigmoid and width-softmax
    temperatures (the trainer anneals `sps_tau`).

    The pinball term runs over the model's own quantile levels
    (`out["levels"]`, default `QUANTILE_LEVELS`), so a nine-level head is
    trained on its nine levels.
    """
    mean_nd = out["mean_nd"]                                    # (B,20,H,W,2)
    fluct_nd = out["fluct_nd"]
    Q_nd = out["Q_nd"]                                          # (B,20,H,W,2,n)
    levels = out.get("levels") or QUANTILE_LEVELS
    U = out["U"].to(mean_nd.dtype).view(-1, 1, 1, 1, 1)

    yv = y[..., :2].to(mean_nd.dtype)
    target_nd = yv / U
    valid = (yv != 0).to(mean_nd.dtype)
    n_valid = valid.sum().clamp(min=1.0)

    resid = target_nd - mean_nd
    Bn = mean_nd.shape[0]
    # masked MSE (reported) and per-window relative L2 (optimised): rel L2 is on the
    # same ~0.1 scale as the TKE and pinball terms, so the weights below mean what
    # they say; plain MSE (~0.01) let the TKE term dominate the trunk gradient 30:1.
    L_mse = ((resid * resid) * valid).sum() / n_valid
    tnorm = ((target_nd * target_nd) * valid).reshape(Bn, -1).sum(dim=1).clamp(min=_EPS).sqrt()
    L_mean = (((resid * resid) * valid).reshape(Bn, -1).sum(dim=1).sqrt() / tnorm).mean()

    L_pin = torch.zeros((), dtype=mean_nd.dtype, device=mean_nd.device)
    for i, tau in enumerate(levels):
        L_pin = L_pin + (pinball(resid - Q_nd[..., i], float(tau)) * valid).sum() / n_valid

    # the shipped prediction: mean (detached, it is trained by L_mean alone) plus the
    # fluctuation. The fluctuation head pays the rel-L2 price of mis-phased energy
    # (L_pred) and earns the TKE reward (L_tke); the optimiser finds the balance.
    pred_nd = mean_nd.detach() + float(gamma) * fluct_nd
    rp = target_nd - pred_nd
    L_pred = (((rp * rp) * valid).reshape(Bn, -1).sum(dim=1).sqrt() / tnorm).mean()
    ke_p = tke_field(pred_nd)
    ke_t = tke_field(target_nd)
    num = (ke_p - ke_t).reshape(Bn, -1).norm(dim=1)
    den = ke_t.reshape(Bn, -1).norm(dim=1).clamp(min=_EPS)
    L_tke = (num / den).mean()

    extra: Dict[str, torch.Tensor] = {}
    if str(tke_horizon) == "flat":
        total = L_mean + L_pred + float(lambda_pin) * L_pin + float(lambda_tke) * L_tke
    else:
        # Per-horizon TKE weighting: match the TKE
        # only over the prefix that is still predictable, K = clip(round(c *
        # tau_frames), 4, 20) frames, and keep the full-window term at a reduced
        # weight.  Don't pay the fluctuation head for keeping energy alive
        # beyond the decorrelation range.
        if str(tke_horizon) != "decorr":
            raise ValueError(f"tke_horizon must be 'flat' or 'decorr', got {tke_horizon!r}")
        Tw = mean_nd.shape[1]
        # K in float32 even under bf16 autocast: bf16 has 8 mantissa bits, which
        # would make the rounding of `c * tau_frames` to an integer frame count
        # jitter by +-1 near a .5 boundary.
        Kf = horizon_frames(out["U"].float().reshape(-1),
                            c=float(tke_horizon_c), T=int(Tw))
        tvec = torch.arange(Tw, device=mean_nd.device, dtype=torch.float32)
        wmask = (tvec.view(1, Tw, 1, 1, 1) < Kf.view(-1, 1, 1, 1, 1)).to(mean_nd.dtype)
        ke_ph = tke_field_masked(pred_nd, wmask)
        ke_th = tke_field_masked(target_nd, wmask)
        numh = (ke_ph - ke_th).reshape(Bn, -1).norm(dim=1)
        denh = ke_th.reshape(Bn, -1).norm(dim=1).clamp(min=_EPS)
        L_tke_h = (numh / denh).mean()
        lam_full = (float(lambda_tke) / 2.0 if lambda_tke_full is None
                    else float(lambda_tke_full))
        total = (L_mean + L_pred + float(lambda_pin) * L_pin
                 + float(lambda_tke) * L_tke_h + lam_full * L_tke)
        extra["L_tke_h"] = L_tke_h
        extra["L_K_mean"] = Kf.mean().detach()

    if float(lambda_tmean) != 0.0:
        # time mean over the valid frames of each (cell, channel)
        cnt = valid.sum(dim=1)                                  # (B,H,W,2)
        den_t = cnt.clamp(min=1.0)
        tm_p = (mean_nd * valid).sum(dim=1) / den_t
        tm_t = (target_nd * valid).sum(dim=1) / den_t
        cell = (cnt > 0).to(mean_nd.dtype)
        dtm = (tm_p - tm_t) * cell
        num = (dtm * dtm).reshape(Bn, -1).sum(dim=1).sqrt()
        den = ((tm_t * tm_t) * cell).reshape(Bn, -1).sum(dim=1).clamp(min=_EPS).sqrt()
        L_tmean = (num / den).mean()
        total = total + float(lambda_tmean) * L_tmean
        extra["L_tmean"] = L_tmean

    if float(lambda_probe) != 0.0:
        rows = torch.as_tensor(PROBE_ROWS, device=mean_nd.device)
        cols = torch.as_tensor(PROBE_COLS, device=mean_nd.device)
        # (B, 20, 9, 4, 2) -> time mean -> (B, 9, 4, 2); scoring.py takes the
        # plain mean over the 20 frames (zeros included), so this mirrors it.
        pp = mean_nd.index_select(2, rows).index_select(3, cols).mean(dim=1)
        tt = target_nd.index_select(2, rows).index_select(3, cols).mean(dim=1)
        d = (pp - tt).permute(0, 2, 1, 3).reshape(Bn, len(PROBE_COLS), -1)
        t2 = tt.permute(0, 2, 1, 3).reshape(Bn, len(PROBE_COLS), -1)
        num = d.norm(dim=2)                                     # (B, 4)
        den = t2.norm(dim=2).clamp(min=1e-8)
        L_probe = (num / den).mean(dim=1).mean()
        total = total + float(lambda_probe) * L_probe
        extra["L_probe"] = L_probe

    if float(lambda_err) != 0.0 and "err_nd" in out:
        tgt = torch.log((target_nd - mean_nd.detach()).abs() + ERR_EPS)
        L_err = ((out["err_nd"] - tgt).abs() * valid).sum() / n_valid
        total = total + float(lambda_err) * L_err
        extra["L_err"] = L_err

    if float(lambda_var) != 0.0:
        cnt = valid.sum(dim=1)                                  # (B,H,W,2) valid frames per cell/chan
        den_c = cnt.clamp(min=1.0)
        mu_p = (pred_nd * valid).sum(dim=1) / den_c
        mu_t = (target_nd * valid).sum(dim=1) / den_c
        var_p = (((pred_nd - mu_p.unsqueeze(1)) ** 2) * valid).sum(dim=1) / den_c
        var_t = (((target_nd - mu_t.unsqueeze(1)) ** 2) * valid).sum(dim=1) / den_c
        cell_ok = (cnt >= float(var_min_frames)).to(mean_nd.dtype)
        dlog = torch.log(var_p + float(var_eps)) - torch.log(var_t + float(var_eps))
        if str(var_weight) == "energy":
            # weight each cell by its share of the window's target fluctuation energy,
            # as the tke norm does (variance-predictability analysis: the whole oracle
            # prize is in shear + wake, none in the freestream)
            w = (var_t.detach() * cell_ok)
            w = w / w.reshape(Bn, -1).sum(dim=1).clamp(min=_EPS).view(Bn, 1, 1, 1)
            L_var = ((dlog * dlog) * w).reshape(Bn, -1).sum(dim=1).mean()
        elif str(var_weight) == "uniform":
            L_var = ((dlog * dlog) * cell_ok).sum() / cell_ok.sum().clamp(min=1.0)
        else:
            raise ValueError(f"var_weight must be 'uniform' or 'energy', got {var_weight!r}")
        total = total + float(lambda_var) * L_var
        extra["L_var"] = L_var
        with torch.no_grad():
            extra["var_logratio_mean"] = ((dlog * cell_ok).sum() / cell_ok.sum().clamp(min=1.0)).detach()

    if float(lambda_sps) != 0.0:
        r_sps = sps_reward_loss(Q_nd, (target_nd - mean_nd.detach()), valid, n_valid,
                                out["U"], levels, tau=float(sps_tau),
                                soft_temp=float(sps_temp))
        total = total + float(lambda_sps) * r_sps["L_sps"]
        extra["L_sps"] = r_sps["L_sps"]
        extra["sps_reward"] = r_sps["sps_reward"]
        extra["sps_cov"] = r_sps["sps_cov"]
        extra["sps_w_sigma"] = r_sps["sps_w_sigma"]

    return dict(total=total, loss=total, L_mean=L_mean, L_pred=L_pred, L_mse=L_mse.detach(),
                L_pin=L_pin, L_tke=L_tke, n_valid=n_valid.detach(), **extra)
