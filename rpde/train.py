"""Training entry point.

    python -m rpde.train --config configs/b_small.yaml --fold R-mid-a --seed 2 \
                         --name b_small_Rmida_s2 [--set optim.epochs=5 ...] [--wandb]

Writes ``runs/<name>/{config.json,best.pth,last.pth,log.txt,run.json}``.
Model selection is on the leaderboard-weighted score ``lbw`` of the validation
tail of the *training* trajectories (never the held-out row).

"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
# TF32 matmuls/convs on Ampere: ~2-3x faster fp32 Linear/conv with negligible accuracy cost
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import yaml

from . import data as D
from . import metrics as M
from .constants import QUANTILE_LEVELS, RUNS_DIR, T_OUT


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------
class Cfg(dict):
    """dict with attribute access, so the model side may use ``cfg.gamma`` or
    ``cfg["gamma"]`` interchangeably."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(x):
        if isinstance(x, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in x.items()})
        if isinstance(x, list):
            return [Cfg.wrap(v) for v in x]
        return x


DEFAULTS = dict(
    core=dict(name="stub", kwargs={}),
    model=dict(shrink=False, gamma=0.75,
               baseline=dict(gamma0=0.6, tau_star0=0.3, damping=False,
                             beta_mode="const", beta0=0.75, U0=0.129, w=0.05)),
    data=dict(),
    optim=dict(lr=3e-4, weight_decay=0.05, warmup_epochs=5, epochs=300,
               grad_clip=1.0, ema_decay=0.999, amp=True),
    loss=dict(lambda_pin=0.1, lambda_tke=0.2,
              # opt-in; 0 / False = the shipping path
              lambda_tmean=0.0, lambda_probe=0.0, pinball_detach_trunk=False,
              # opt-in; "flat" = the shipping path
              tke_horizon="flat", tke_horizon_c=2.0, lambda_tke_full=None,
              # opt-in; 0 = the shipping path
              lambda_err=0.0,
              # opt-in straight-through SPS reward term (wave 7); 0 = the shipping
              # path. Switched on at `sps_start_epoch`; `sps_tau` is annealed
              # geometrically tau0 -> tau1 from that epoch to the last one.
              lambda_sps=0.0, sps_start_epoch=20, sps_tau0=0.02, sps_tau1=0.001,
              sps_temp=0.005,
              # opt-in per-cell log-variance match of the injected fluctuation
              #; 0 = the shipping path
              lambda_var=0.0, var_eps=1e-4, var_weight="uniform",
              # opt-in: recompute lambda_tke and lambda_probe every epoch from the scorer's own
              # sensitivities at the latest validation error level; False = the shipping path
              score_aligned=False,
              # opt-in attached TKE-map head; 0 = shipping path.
              # The head trains from epoch 0 on detached features; its gradient into the trunk is 0 before
              # map_start_epoch and ramps linearly to 1 over map_ramp_epochs.
              lambda_map=0.0, map_start_epoch=40, map_ramp_epochs=10, map_lambda_log=0.01,
              # opt-in auxiliary sim-pressure target (needs data.aux_p and model.aux_p); 0 = shipping path
              lambda_p=0.0,
              # opt-in winner-takes-all over the fluctuation hypotheses (needs model.fluct_hyp > 1)
              lambda_wta=0.0, wta_all=0.05,
              # opt-in energy score over the fluctuation hypotheses (needs model.fluct_hyp > 1)
              lambda_es=0.0),
    # opt-in band decode; False = the shipping path
    decode=dict(err_scale=False, err_scale_k=1.0),
    eval=dict(every_epochs=1, patience=15, batch_size=None,
              save_every=0,    # >0: also keep the EMA weights every N epochs as ep<N>.pth (held-out curve vs epoch)
              save_from=0, save_to=0),   # >0: only keep ep<N>.pth for save_from <= N <= save_to (0 = no bound)
    wandb=dict(enabled=False, group=None, tags=[]),
    # opt-in ensemble distillation; `teacher_runs` empty
    # = the shipping path, bit-identical
    distill=dict(teacher_runs=None, teacher_ckpt="best", alpha=0.7,
                 mode="mean+Q", mix_pred=False, amp=True, fast=True),
)



# Scalars compute_losses reports that are NOT loss terms and so do not start with "L".  They are logged and
# stored exactly like the loss components; see the note in the filter below.
LOSS_DIAGNOSTICS = ("var_logratio_mean", "sps_reward", "sps_cov", "sps_w_sigma")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_sets(cfg: dict, sets: list[str]) -> dict:
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def load_config(path: str | None, sets: list[str] | None = None) -> dict:
    user = {}
    if path:
        with open(path) as fh:
            user = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULTS, user)
    return apply_sets(cfg, sets or [])


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha(root: str | Path = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root or Path(__file__).resolve().parent.parent),
             "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


class Logger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.fh.write(msg + "\n")

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# model-side adapters (the model modules are owned by the other agent; call
# them exactly as specified, but tolerate
# small signature differences)
# --------------------------------------------------------------------------
def build_model(cfg: dict):
    """The model, with `decode.*` folded into the model cfg.

    `decode.err_scale` / `decode.err_scale_k` are decode-time options of the
    auxiliary error head; they live under `decode`
    on the command line and as `model.err_scale` inside `RPDEModel.cfg`, so a
    run is reproducible from its `config.json` either way.
    """
    from .scaffold import RPDEModel
    core = cfg["core"]
    mcfg = dict(cfg["model"])
    dcfg = dict(cfg.get("decode") or {})
    for k in ("err_scale", "err_scale_k"):
        if k in dcfg and k not in cfg["model"]:
            mcfg[k] = dcfg[k]
    return RPDEModel(core["name"], dict(core.get("kwargs") or {}),
                     Cfg.wrap(mcfg))


def _kwmatch(fn, wanted: dict) -> dict:
    try:
        names = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return {}
    kw = {}
    for cands, val in wanted.items():
        for nm in cands:
            if nm in names:
                kw[nm] = val
                break
    return kw



# --------------------------------------------------------------------------
# score-aligned loss weights
# --------------------------------------------------------------------------
SCORE_W = dict(rel=0.468, tke=0.100, mvpe=0.097, sps=0.249)
ACC_C = dict(rel=0.5, tke=0.3, mvpe=0.2)          # sps accuracy-factor coefficients


def score_sensitivity(e_rel: float, e_tke: float, e_mvpe: float, band_q: float) -> dict:
    """d(final)/d(e_x) for the three accuracy errors, at the given error level.

    final = 0.468 s(e_rel) + 0.100 s(e_tke) + 0.097 s(e_mvpe) + 0.249 sps + ...,
    s(e) = 100/(1+e/2)  ->  ds/de = -50/(1+e/2)^2  (direct part), and
    sps = 100 * band_q * A,  A = sum_x c_x (1 - e_x/(0.5+e_x))  ->  dA/de_x = -c_x*0.5/(0.5+e_x)^2
    (the part through the sps accuracy factor).  Returned as positive magnitudes.
    """
    out = {}
    for k, e in (("rel", e_rel), ("tke", e_tke), ("mvpe", e_mvpe)):
        e = max(float(e), 1e-6)
        direct = SCORE_W[k] * 50.0 / (1.0 + 0.5 * e) ** 2
        via_sps = SCORE_W["sps"] * 100.0 * float(band_q) * ACC_C[k] * 0.5 / (0.5 + e) ** 2
        out[k] = direct + via_sps
    return out


def score_aligned_weights(e_rel, e_tke, e_mvpe, band_q):
    """(lambda_tke, lambda_probe) relative to the shipped-prediction rel term (weight 1)."""
    S = score_sensitivity(e_rel, e_tke, e_mvpe, band_q)
    return S["tke"] / S["rel"], S["mvpe"] / S["rel"]

def aux_pressure_loss(p_nd, p, U):
    """Per-window relative L2 of the centred sim pressure / U^2 over measured cells; real windows (p == 0) drop out."""
    t = p / (U.view(-1, 1, 1, 1) ** 2)
    v = (p != 0).float()
    n = v.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    tc = (t - (t * v).sum(dim=(2, 3), keepdim=True) / n) * v          # pressure is defined up to a constant per frame
    pc = (p_nd - (p_nd * v).sum(dim=(2, 3), keepdim=True) / n) * v
    B = p.shape[0]
    has = v.reshape(B, -1).sum(1) > 0
    if not bool(has.any()):
        return p_nd.sum() * 0.0
    num = (((pc - tc) ** 2).reshape(B, -1).sum(1)[has] + 1e-12).sqrt()   # real windows never reach the sqrt
    den = (tc ** 2).reshape(B, -1).sum(1)[has].clamp(min=1e-12).sqrt()
    return (num / den).mean()


def wta_fluct_loss(out, y, gamma, w_all=0.05):
    """min over hypotheses of the per-window rel L2 of mean.detach + gamma * f_k, plus w_all x their mean (no dead
    hypotheses).  The combined fluctuation is trained by the ordinary L_pred / L_tke."""
    fh = out["fluct_hyp"].float()                                       # (B,K,20,H,W,2)
    U = out["U"].float().view(-1, 1, 1, 1, 1)
    yt = y[..., :2].float() / U
    v = (y[..., :2] != 0).float()
    m = out["mean_nd"].float().detach()
    B, K = fh.shape[:2]
    tn = ((yt * yt) * v).reshape(B, -1).sum(1).clamp(min=1e-12).sqrt()
    r = torch.stack([(((yt - m - float(gamma) * fh[:, k]) ** 2) * v).reshape(B, -1).sum(1).sqrt() / tn
                     for k in range(K)], dim=1)                          # (B,K)
    return r.min(dim=1).values.mean() + float(w_all) * r.mean()


def es_fluct_loss(out, y, gamma):
    """Energy score of the K hypotheses as an ensemble forecast, per window, in rel-L2 units:
    mean_k ||mean.detach + gamma f_k - y|| - 1/(2K(K-1)) sum_{k != j} ||gamma (f_k - f_j)||, over measured cells.
    A proper scoring rule: minimised when the hypotheses' spread matches the real uncertainty."""
    fh = out["fluct_hyp"].float()                                       # (B,K,20,H,W,2)
    U = out["U"].float().view(-1, 1, 1, 1, 1)
    yt = y[..., :2].float() / U
    v = (y[..., :2] != 0).float()
    m = out["mean_nd"].float().detach()
    B, K = fh.shape[:2]
    tn = ((yt * yt) * v).reshape(B, -1).sum(1).clamp(min=1e-12).sqrt()

    def nrm(a):
        return (((a * a) * v).reshape(B, -1).sum(1) + 1e-12).sqrt() / tn
    skill = torch.stack([nrm(m + float(gamma) * fh[:, k] - yt) for k in range(K)], 1).mean(1)
    spread = sum(nrm(float(gamma) * (fh[:, k] - fh[:, j])) for k in range(K) for j in range(K) if j != k)
    return (skill - spread / (2.0 * K * (K - 1))).mean()


def call_losses(out, y, gamma, lam_pin, lam_tke, lam_tmean=0.0, lam_probe=0.0,
                tke_horizon="flat", tke_horizon_c=2.0, lam_tke_full=None,
                raw=False, lam_err=0.0, lam_sps=0.0, sps_tau=0.02, sps_temp=0.005,
                lam_var=0.0, var_eps=1e-4, var_weight="uniform"):
    from .losses import compute_losses
    wanted = {
        ("gamma",): gamma,
        ("lambda_pin", "lam_pin", "l_pin"): lam_pin,
        ("lambda_tke", "lam_tke", "l_tke"): lam_tke,
    }
    # only passed when actually enabled, so an older `compute_losses` without
    # these arguments still works on the default configuration
    if float(lam_tmean) != 0.0:
        wanted[("lambda_tmean", "lam_tmean")] = lam_tmean
    if float(lam_probe) != 0.0:
        wanted[("lambda_probe", "lam_probe")] = lam_probe
    if float(lam_err) != 0.0:
        wanted[("lambda_err", "lam_err")] = lam_err
    if float(lam_var) != 0.0:
        wanted[("lambda_var", "lam_var")] = float(lam_var)
        wanted[("var_eps",)] = float(var_eps)
        wanted[("var_weight",)] = str(var_weight)
    if float(lam_sps) != 0.0:
        wanted[("lambda_sps", "lam_sps")] = float(lam_sps)
        wanted[("sps_tau",)] = float(sps_tau)
        wanted[("sps_temp",)] = float(sps_temp)
    if str(tke_horizon or "flat") != "flat":
        wanted[("tke_horizon",)] = str(tke_horizon)
        wanted[("tke_horizon_c",)] = float(tke_horizon_c)
        if lam_tke_full is not None:
            wanted[("lambda_tke_full", "lam_tke_full")] = float(lam_tke_full)
    kw = _kwmatch(compute_losses, wanted)
    try:
        r = compute_losses(out, y, **kw)
    except TypeError:
        r = compute_losses(out, y, gamma, lam_pin, lam_tke)
    if torch.is_tensor(r):
        if raw:
            raise TypeError("call_losses(raw=True) needs the dict-returning compute_losses")
        return r, {"loss": float(r.detach())}
    if isinstance(r, (tuple, list)):
        if raw:
            raise TypeError("call_losses(raw=True) needs the dict-returning compute_losses")
        loss, comp = r[0], (r[1] if len(r) > 1 else {})
        return loss, {k: float(v) for k, v in dict(comp).items()}
    if isinstance(r, dict):
        for key in ("loss", "total", "L", "L_total"):
            if key in r:
                loss = r[key]
                break
        else:
            raise KeyError(f"compute_losses returned {sorted(r)} without a total")
        # The "L"-prefix rule silently dropped every DIAGNOSTIC the loss emits, because none of them is a
        # loss term: var_logratio_mean (the signed log-variance bias -- negative means the model
        # under-predicts fluctuation energy) and the three sps-surrogate diagnostics.  Consequence, found on
        # 2026-09-22: c_w44_fa_varu was the only run ever to compute var_logratio_mean and the number was
        # thrown away before the log line AND before run.json's train_curve, so the one experiment aimed at
        # the tke gap could not be diagnosed afterwards.  The sps surrogate has likewise been running since
        # epoch 40 of every recent run with no visibility into its reward, coverage or width.
        comp = {k: float(v) for k, v in r.items()
                if (torch.is_tensor(v) or isinstance(v, float))
                and (k.startswith("L") or k == "loss" or k in LOSS_DIAGNOSTICS)}
        comp["loss"] = float(loss.detach() if torch.is_tensor(loss) else loss)
        comp.pop("L_total", None)
        if raw:
            return loss, comp, r
        return loss, comp
    raise TypeError(f"unsupported compute_losses return type {type(r)}")


def call_bounds(out, spread=None, model=None):
    """``decode.band_from_quantiles`` on a forward-pass output dict."""
    from .decode import band_from_quantiles
    args = (out["Q_nd"], out["mean"], out["U"], out["m_core"])
    wanted = {}
    if spread is not None:
        wanted[("spread",)] = spread
    lv = out.get("levels") if isinstance(out, dict) else None
    if lv is not None:
        wanted[("levels",)] = lv
    es = model.decode_err_scale(out) if hasattr(model, "decode_err_scale") else None
    if es is not None:
        wanted[("err_scale",)] = es
    kw = _kwmatch(band_from_quantiles, wanted) if wanted else {}
    return band_from_quantiles(*args, **kw)


def call_predict_with_bounds(model, x, gamma=None):
    fn = model.predict_with_bounds
    kw = _kwmatch(fn, {("gamma",): gamma}) if gamma is not None else {}
    r = fn(x, **kw)
    if isinstance(r, dict):
        return r["pred"], r["lower"], r["upper"]
    return r[0], r[1], r[2]


# --------------------------------------------------------------------------
# optimiser
# --------------------------------------------------------------------------
NO_DECAY_HINTS = ("bias", "norm", "gate", "pos_emb", "pos_embed", "posemb",
                  "baseline", "logit_scale", "embedding")


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = name.lower()
        if p.ndim < 2 or any(h in low for h in NO_DECAY_HINTS):
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=weight_decay),
            dict(params=no_decay, weight_decay=0.0)]


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def make_ema(model, decay: float):
    def avg_fn(avg_p, cur_p, n):
        return decay * avg_p + (1.0 - decay) * cur_p
    return torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn, use_buffers=True)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_windows(model, loader, device, gamma, lam_pin, lam_tke,
                     amp: bool = False, check_shipping_path: bool = True,
                     collect_meta: bool = True, spread: float | None = None,
                     dcfg=None,
                     lam_tmean: float = 0.0, lam_probe: float = 0.0,
                     tke_horizon: str = "flat", tke_horizon_c: float = 2.0,
                     lam_tke_full=None, lam_err: float = 0.0) -> dict:
    model.eval()
    preds, lows, ups, ys = [], [], [], []
    meta = dict(re=[], aoa=[], domain=[], name=[], t0=[])
    comps: dict[str, float] = {}
    nb = 0
    ship_diff = float("nan")
    for bi, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        dom = batch["domain"].to(device).float()
        ck = D.clock_override(batch, dcfg)
        if ck is not None:
            ck = ck.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(x, domain=dom, gamma=gamma, clock=ck)
            loss, comp = call_losses(out, y, gamma, lam_pin, lam_tke,
                                     lam_tmean, lam_probe,
                                     tke_horizon, tke_horizon_c, lam_tke_full,
                                     lam_err=lam_err)
        out = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
        sp = None
        if spread is not None:
            sp = torch.full_like(out["Q_nd"][..., 0], float(spread))
        lower, upper = call_bounds(out, spread=sp, model=model)
        pred = out["pred"].float()
        for k, v in comp.items():
            comps[k] = comps.get(k, 0.0) + float(v)
        nb += 1
        if check_shipping_path and bi == 0 and hasattr(model, "predict_with_bounds"):
            try:
                p2, l2, u2 = call_predict_with_bounds(model, x, gamma)
                ship_diff = float(torch.max(torch.abs(
                    torch.as_tensor(p2).float().cpu() - pred.cpu())))
            except Exception:
                ship_diff = float("nan")
        preds.append(pred.cpu().numpy().astype(np.float32))
        ys.append(y.float().cpu().numpy().astype(np.float32))
        lows.append(torch.as_tensor(lower).float().cpu().numpy().astype(np.float32))
        ups.append(torch.as_tensor(upper).float().cpu().numpy().astype(np.float32))
        if collect_meta:
            for k in ("re", "aoa", "domain", "t0"):
                meta[k].extend([int(v) for v in batch[k]])
            meta["name"].extend(list(batch["name"]))
    if nb == 0:
        raise RuntimeError("empty evaluation loader")
    pred = np.concatenate(preds, 0)
    y = np.concatenate(ys, 0)
    lower = np.concatenate(lows, 0)
    upper = np.concatenate(ups, 0)
    res = M.score_windows(pred, y, lower, upper)
    res["loss_components"] = {k: v / nb for k, v in comps.items()}
    res["shipping_path_max_diff"] = ship_diff
    res["_arrays"] = dict(pred=pred, y=y, lower=lower, upper=upper)
    res["_meta"] = {k: np.asarray(v) if k != "name" else np.asarray(v, dtype=object)
                    for k, v in meta.items()}
    return res


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser("rpde.train")
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="full", choices=sorted(D.FOLDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None,
                    help="checkpoint (.pth) to continue from; restores model+EMA and, when "
                         "present, optimizer/scheduler/epoch/best. Old checkpoints without "
                         "optimizer state get a fresh AdamW with a short re-warmup.")
    ap.add_argument("--resume-from-ema", action="store_true",
                    help="with --resume-weights-only: initialise the MODEL weights from the "
                         "checkpoint's EMA state (the weights that were evaluated), so a "
                         "fine-tune starts from the scored point instead of the noisier raw weights")
    ap.add_argument("--resume-weights-only", action="store_true",
                    help="with --resume: load model+EMA weights but NOTHING else "
                         "(no optimizer, no epoch counter, no best) and run a fresh "
                         "`optim` schedule from epoch 0. This is the fine-tuning mode "
                         "used by the weight-space-direction endpoint experts "
                         "(endpoint expert).")
    ap.add_argument("--resume-epoch", type=int, default=None,
                    help="epochs already done, for checkpoints that carry no epoch counter")
    ap.add_argument("--rewarm-epochs", type=float, default=2.0,
                    help="linear LR re-warmup after a resume without optimizer state")
    return ap.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    cfg = load_config(args.config, args.sets)
    name = args.name or f"{Path(args.config).stem}_{args.fold}_s{args.seed}"
    run_dir = Path(args.runs_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "log.txt")
    t_start = time.time()

    seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dcfg = D.DataCfg.from_dict(cfg["data"])
    ocfg, lcfg, ecfg = cfg["optim"], cfg["loss"], cfg["eval"]
    gamma = float(cfg["model"]["gamma"])

    log(f"[run] {name}  fold={args.fold} seed={args.seed} device={device}")
    log(f"[cfg] {json.dumps(cfg, default=str)}")

    # ---- data ------------------------------------------------------------
    sp = D.split(args.fold)
    if dcfg.include_re:
        # endpoint-expert data filter: keep only these
        # Re rows, real AND sim.  The held-out row of the fold is untouched.
        n0 = (len(sp["train_real"]), len(sp["train_sim"]))
        sp["train_real"] = D.filter_re(sp["train_real"], dcfg.include_re)
        sp["train_sim"] = D.filter_re(sp["train_sim"], dcfg.include_re)
        log(f"[data] include_re={list(dcfg.include_re)} -> "
            f"real {n0[0]}->{len(sp['train_real'])}, sim {n0[1]}->{len(sp['train_sim'])}")
    corpus = D.Corpus()
    corpus, train_real = D.load_all(sp["train_real"], dcfg.limit_regimes, corpus)
    train_sim = []
    if dcfg.use_sim:
        corpus, train_sim = D.load_all(sp["train_sim"], dcfg.limit_regimes, corpus)
    log(f"[data] train_real={len(train_real)} train_sim={len(train_sim)} "
        f"heldout_real={len(sp['heldout_real'])} heldout_sim={len(sp['heldout_sim'])} "
        f"ram={corpus.nbytes()/1e9:.2f} GB")

    if bool(getattr(dcfg, "phase_aug", False)):
        n_ph = corpus.load_phases(train_real)
        log(f"[data] phase_aug: {n_ph} (regime, phase) arrays loaded for {len(train_real)} real regimes "
            f"({corpus.nbytes() / 1e9:.2f} GB in RAM)")
    aug = {k: getattr(dcfg, k) for k in ("mirror_aug", "rescale_aug", "real_noise_nu", "real_drop_p", "amp_jitter",
                                          "phase_aug", "ctx_drop", "mask_aug") if getattr(dcfg, k, None)}
    log(f"[data] train augmentation: {aug if aug else 'none'}")
    train_ds = D.TrainWindows(train_real, train_sim, corpus, dcfg, seed=args.seed)
    # coverage of the training regimes: start positions the sampler may draw vs every legal start (t0_max)
    for _dom in ("real", "sim"):
        _rs = [(r, h) for r, h in zip(train_ds.regs, train_ds._hi) if r.domain == _dom]
        _pool = train_real if _dom == "real" else train_sim
        _k = (lambda r: 1) if _dom == "real" else (lambda r: D.stride_for(r.re, dcfg))
        _all = sum(max(0, D.t0_max(r.T, _k(r)) + 1) for r in _pool)
        _got = sum(h + 1 for r, h in _rs)
        if _pool:
            log(f"[data] coverage {_dom}: regimes {len(_rs)}/{len(_pool)} starts {_got}/{_all} "
                f"({100.0 * _got / max(1, _all):.1f} %) train_all={bool(getattr(dcfg, 'train_all', False))}")
    val_ds = D.FixedWindows(list(train_real) + list(train_sim), corpus, dcfg,
                            mode="val", seed=args.seed, mask_pool=train_real)
    train_ld = D.make_loader(train_ds, dcfg)
    val_ld = D.make_loader(val_ds, dcfg, batch_size=ecfg.get("batch_size") or dcfg.batch_size)
    log(f"[data] windows/epoch={len(train_ds)} val_windows={len(val_ds)} "
        f"regimes={train_ds.n_real}real+{train_ds.n_sim}sim")
    if train_ds.regime_weights:
        log(f"[data] regime_weights={json.dumps(train_ds.regime_weights)} "
            f"-> per-regime sampling weights (x1000):")
        tab = train_ds.weight_table()
        for i in range(0, len(tab), 5):
            log("        " + "  ".join(f"{dom[0]}/{nm:>9s} {1000 * w:6.3f}"
                                       for dom, nm, w in tab[i:i + 5]))
    if train_ds.rescale_aug:
        log(f"[data] rescale_aug={json.dumps(train_ds.rescale_aug)} (REAL windows only)")

    # ---- model -----------------------------------------------------------
    model = build_model(cfg).to(device)
    # `loss.pinball_detach_trunk` is implemented in the head (the quantile
    # channels are read off detached features), so it is wired here; it changes
    # the backward only, never the forward.
    if bool(lcfg.get("pinball_detach_trunk", False)):
        model.head.detach_trunk = True
        log("[model] pinball_detach_trunk: quantile head reads detached trunk features")
    train_only = str(ocfg.get("train_only", "all") or "all")
    if train_only == "fluct_head":
        # tier-1 fine-tune (benchmark_S7final lever 1): freeze everything except the
        # head conv, and inside it zero the gradient of every output channel but
        # the two fluctuation channels (rows 2:4), so mean/quantile rows stay put.
        for pn, p in model.named_parameters():          # not `name`: that is the run name
            p.requires_grad_(pn.startswith("head.conv."))
        rows = torch.zeros(model.head.conv.out_channels, dtype=torch.bool)
        rows[2:4] = True
        def _mask_w(g, rows=rows):
            m = rows.to(g.device).view(-1, *([1] * (g.dim() - 1)))
            return g * m
        def _mask_b(g, rows=rows):
            return g * rows.to(g.device)
        model.head.conv.weight.register_hook(_mask_w)
        model.head.conv.bias.register_hook(_mask_b)
        log("[model] train_only=fluct_head: only head.conv rows 2:4 (fluct u,v) receive gradient")
    elif train_only != "all":
        raise ValueError(f"optim.train_only must be 'all' or 'fluct_head', got {train_only!r}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] core={cfg['core']['name']} params={n_params/1e6:.3f}M")
    ema = make_ema(model, float(ocfg["ema_decay"]))

    # ---- teacher ------------
    dscfg = dict(cfg.get("distill") or {})
    teacher = None
    if dscfg.get("teacher_runs"):
        from .distill import Teacher, combine_loss as distill_combine
        teacher = Teacher(list(dscfg["teacher_runs"]),
                          ckpt=str(dscfg.get("teacher_ckpt", "best")),
                          device=device,
                          amp=bool(dscfg.get("amp", True)) and device.type == "cuda",
                          fast=bool(dscfg.get("fast", True)))
        log(f"[distill] {len(teacher)} teachers alpha={float(dscfg.get('alpha', 0.7))} "
            f"mode={dscfg.get('mode', 'mean+Q')} mix_pred={bool(dscfg.get('mix_pred', False))} "
            f"fast={teacher.fast} "
            f"ckpt={dscfg.get('teacher_ckpt', 'best')}")
        for r in teacher.runs:
            log(f"[distill]   teacher {r}")

    opt = torch.optim.AdamW(param_groups(model, float(ocfg["weight_decay"])),
                            lr=float(ocfg["lr"]), betas=(0.9, 0.95))
    steps_per_epoch = max(1, math.ceil(len(train_ds) / dcfg.batch_size))
    total_steps = int(ocfg["epochs"]) * steps_per_epoch
    warmup_steps = int(ocfg["warmup_epochs"]) * steps_per_epoch
    amp = bool(ocfg["amp"]) and device.type == "cuda"

    # ---- resume ----------------------------------------------------------
    start_epoch, resume_best, resume_curve = 0, None, None
    ck_opt = ck_sched = None
    step_offset, rewarm_steps = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if getattr(args, "resume_from_ema", False) and ck.get("ema_state") is not None:
            model.load_state_dict(ck["ema_state"])
            log("[resume] model weights initialised from the checkpoint's EMA state")
        else:
            model.load_state_dict(ck["model_state"])
        ema.module.load_state_dict(ck.get("ema_state", ck["model_state"]))
        # AveragedModel copies the raw weights into the average on its first update when n_averaged == 0,
        # which silently discarded the restored EMA on every resume (found 2026-09-26: -0.2 val lbw dip at each
        # resume). Mark the restored average as already initialised.
        if hasattr(ema, "n_averaged"):
            ema.n_averaged.fill_(1)
        if args.resume_weights_only:
            log(f"[resume] {args.resume}: WEIGHTS ONLY (model + EMA); fresh optimizer "
                f"and a new {int(ocfg['epochs'])}-epoch schedule from epoch 0 "
                f"(lr {float(ocfg['lr']):g}, warmup {int(ocfg['warmup_epochs'])} ep)")
        done = None if args.resume_weights_only else ck.get("epoch")
        if args.resume_weights_only:
            done = 0
        if done is None:
            done = args.resume_epoch
        if done is None:
            rj = Path(args.resume).parent / "run.json"
            if rj.exists():
                with open(rj) as fh:
                    done = int(json.load(fh).get("epochs", 0))
        if done is None:
            raise SystemExit("--resume: checkpoint has no epoch counter; pass --resume-epoch N")
        if not args.resume_weights_only:
            start_epoch = int(done)
            step_offset = start_epoch * steps_per_epoch
            ck_opt, ck_sched = ck.get("opt_state"), ck.get("sched_state")
            resume_best = ck.get("best")
            if ck_opt is None:
                rewarm_steps = int(round(float(args.rewarm_epochs) * steps_per_epoch))
            rj = Path(args.resume).parent / "run.json"
            if rj.exists():
                with open(rj) as fh:
                    resume_curve = json.load(fh).get("train_curve")
            opt_note = ("yes" if ck_opt is not None
                        else f"NO (fresh AdamW, re-warmup {rewarm_steps} steps)")
            log(f"[resume] {args.resume}: epochs_done={start_epoch} "
                f"optimizer_state={opt_note} best={resume_best}")

    # optim.lr_hold_epoch (default off): the cosine is frozen at its value at that epoch
    # (diagnosis 2026-09-26: does the late held-out collapse need the LR decay?)
    _hold = ocfg.get("lr_hold_epoch")
    _hold_step = None if _hold in (None, "", "null") else int(float(_hold) * steps_per_epoch)

    def _lr(s: int) -> float:
        st = s + step_offset
        if _hold_step is not None:
            st = min(st, _hold_step)
        base = lr_lambda(st, warmup_steps, total_steps)
        if rewarm_steps > 0 and s < rewarm_steps:
            base *= (s + 1) / rewarm_steps
        return base
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    if ck_opt is not None:
        opt.load_state_dict(ck_opt)
        # the step counter lives in the lambda offset; the restored scheduler state only
        # matters for its own last_epoch, which we keep at 0 relative to the new process
        for g in opt.param_groups:
            g["lr"] = float(ocfg["lr"]) * _lr(0)

    # ---- run record ------------------------------------------------------
    meta = dict(argv=list(sys.argv if argv is None else ["rpde.train", *argv]),
                git_sha=git_sha(), fold=args.fold, seed=args.seed,
                n_params=int(n_params), torch_version=torch.__version__,
                start_time=time.strftime("%Y-%m-%dT%H:%M:%S"), end_time=None,
                device=str(device))
    with open(run_dir / "config.json", "w") as fh:
        json.dump(dict(config=cfg, fold=args.fold, seed=args.seed, name=name), fh, indent=1)
    record = dict(
        run_id=name, git_sha=meta["git_sha"],
        config_hash=str(abs(hash(json.dumps(cfg, sort_keys=True, default=str))) % (10 ** 12)),
        core=cfg["core"]["name"], recipe=("R0" if not dcfg.use_sim else "R2"),
        fold=args.fold, seed=args.seed, n_params=int(n_params),
        train_files=[r.name for r in train_real] + [f"sim/{r.name}" for r in train_sim],
        epochs=0, wall_h=0.0, metrics={}, diagnostics={}, train_curve=[],
        config=cfg,
        submission=dict(id=None, date=None,
                        hidden=dict(rel_l2=None, tke=None, mvpe=None, time=None,
                                    sps=None, final=None)),
    )

    def write_record():
        record["wall_h"] = (time.time() - t_start) / 3600.0
        with open(run_dir / "run.json", "w") as fh:
            json.dump(record, fh, indent=1, default=str)

    # ---- W&B -------------------------------------------------------------
    wb = None
    if args.wandb or cfg["wandb"].get("enabled"):
        try:
            import wandb
            wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "NIC_RealPDE"),
                            entity=os.environ.get("WANDB_ENTITY") or None,
                            group=cfg["wandb"].get("group") or args.fold,
                            tags=list(cfg["wandb"].get("tags") or []) + [args.fold,
                                                                         cfg["core"]["name"]],
                            name=name, config=cfg, dir=str(run_dir))
        except Exception as e:  # pragma: no cover
            log(f"[wandb] disabled ({e})")
            wb = None

    epoch_done = start_epoch  # number of completed epochs at the time of the save

    def save(tag: str, full: bool = True):
        d = dict(model_state=model.state_dict(),
                 ema_state=ema.module.state_dict(),
                 core_name=cfg["core"]["name"],
                 core_kwargs=dict(cfg["core"].get("kwargs") or {}),
                 model_cfg=dict(cfg["model"]), data_cfg=dict(cfg["data"]),
                 train_meta=meta, epoch=int(epoch_done),
                 best=dict(lbw=best_lbw, epoch=best_epoch, bad=bad))
        if full:  # optimizer + schedule position, so a resume is exact
            d["opt_state"] = opt.state_dict()
            d["sched_state"] = dict(step_offset=step_offset + sched.last_epoch)
        torch.save(d, run_dir / f"{tag}.pth")

    best_lbw, best_epoch, bad = -float("inf"), -1, 0
    if resume_best:
        best_lbw = float(resume_best.get("lbw", best_lbw))
        best_epoch = int(resume_best.get("epoch", best_epoch))
        bad = int(resume_best.get("bad", 0))
    if resume_curve:
        record["train_curve"] = list(resume_curve)
        record["epochs"] = start_epoch
        record["resumed_from"] = args.resume
    epochs = int(ocfg["epochs"])
    # score-aligned weights: error level from the latest validation (defaults = a fresh model)
    sa_level = dict(rel=0.10, tke=0.75, mvpe=0.10, band_q=0.50)
    score_aligned = bool(lcfg.get("score_aligned", False))
    for epoch in range(start_epoch, epochs):
        model.train()
        train_ds.set_epoch(epoch)
        if score_aligned:
            lam_tke_ep, lam_probe_ep = score_aligned_weights(
                sa_level["rel"], sa_level["tke"], sa_level["mvpe"], sa_level["band_q"])
        else:
            lam_tke_ep = float(lcfg["lambda_tke"])
            lam_probe_ep = float(lcfg.get("lambda_probe", 0.0) or 0.0)
        acc: dict[str, float] = {}
        nb = 0
        # SPS reward term: off before `sps_start_epoch`, tau annealed after it
        lam_sps = float(lcfg.get("lambda_sps", 0.0) or 0.0)
        sps_start = int(lcfg.get("sps_start_epoch", 0) or 0)
        sps_tau = float(lcfg.get("sps_tau0", 0.02) or 0.02)
        if lam_sps != 0.0 and epoch < sps_start:
            lam_sps = 0.0
        elif lam_sps != 0.0:
            t0, t1 = float(lcfg.get("sps_tau0", 0.02)), float(lcfg.get("sps_tau1", 0.001))
            frac = (epoch - sps_start) / max(1, (epochs - 1) - sps_start)
            sps_tau = t0 * (t1 / t0) ** min(1.0, max(0.0, frac))
        lam_map = float(lcfg.get("lambda_map", 0.0) or 0.0)
        map_alpha = 0.0
        if lam_map != 0.0 and getattr(model, "tke_head", None) is not None:
            m0, mr = int(lcfg.get("map_start_epoch", 40)), max(1, int(lcfg.get("map_ramp_epochs", 10)))
            map_alpha = min(1.0, max(0.0, (epoch - m0 + 1) / mr))
            model.tke_head.trunk_grad = map_alpha
        for batch in train_ld:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            dom = batch["domain"].to(device).float()
            tgt = (teacher.targets(x, domain=dom, gamma=gamma)
                   if teacher is not None else None)
            ck = D.clock_override(batch, dcfg)
            if ck is not None:
                ck = ck.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x, domain=dom, gamma=gamma, clock=ck)
                res = call_losses(out, y, gamma,
                                  float(lcfg["lambda_pin"]), lam_tke_ep,
                                  float(lcfg.get("lambda_tmean", 0.0) or 0.0),
                                  lam_probe_ep,
                                  str(lcfg.get("tke_horizon", "flat") or "flat"),
                                  float(lcfg.get("tke_horizon_c", 2.0) or 2.0),
                                  lcfg.get("lambda_tke_full"),
                                  raw=teacher is not None,
                                  lam_err=float(lcfg.get("lambda_err", 0.0) or 0.0),
                                  lam_sps=lam_sps, sps_tau=sps_tau,
                                  sps_temp=float(lcfg.get("sps_temp", 0.005) or 0.005),
                                  lam_var=float(lcfg.get("lambda_var", 0.0) or 0.0),
                                  var_eps=float(lcfg.get("var_eps", 1e-4) or 1e-4),
                                  var_weight=str(lcfg.get("var_weight", "uniform") or "uniform"))
                if teacher is None:
                    loss, comp = res
                else:
                    loss, comp = res[0], res[1]
                    comp["L_gt"] = comp["loss"]          # the ground-truth total
                    loss, dcomp = distill_combine(
                        out, y, tgt, loss, res[2],
                        alpha=float(dscfg.get("alpha", 0.7)),
                        lambda_pin=float(lcfg["lambda_pin"]),
                        mode=str(dscfg.get("mode", "mean+Q")),
                        mix_pred=bool(dscfg.get("mix_pred", False)))
                    comp.update(dcomp)
                if lam_map != 0.0 and "log_ehat_nd" in out:
                    from .tke_map import energy_nd, tke_map_loss
                    Ub = out["U"].float().view(-1, 1, 1, 1, 1)
                    e_true = energy_nd(y.float() / Ub)
                    valid = ((y[..., :2] != 0).all(-1).all(1) & (out["m_core"][:, 0] < 0.5)).float()
                    Lm = tke_map_loss(out["log_ehat_nd"].float(), e_true, valid,
                                      float(lcfg.get("map_lambda_log", 0.01)))
                    loss = loss + lam_map * Lm["loss"]
                    comp["L_map"] = float(Lm["loss"].detach())
                    comp["map_err"] = float(Lm["map_err"])
                lam_p = float(lcfg.get("lambda_p", 0.0) or 0.0)
                if lam_p != 0.0 and "p_nd" in out and "p" in batch:
                    Lp = aux_pressure_loss(out["p_nd"].float(), batch["p"].to(device).float(), out["U"].float())
                    loss = loss + lam_p * Lp
                    comp["L_p"] = float(Lp.detach())
                lam_wta = float(lcfg.get("lambda_wta", 0.0) or 0.0)
                if lam_wta != 0.0 and "fluct_hyp" in out:
                    Lw = wta_fluct_loss(out, y, gamma, float(lcfg.get("wta_all", 0.05)))
                    loss = loss + lam_wta * Lw
                    comp["L_wta"] = float(Lw.detach())
                lam_es = float(lcfg.get("lambda_es", 0.0) or 0.0)
                if lam_es != 0.0 and "fluct_hyp" in out:
                    Le = es_fluct_loss(out, y, gamma)
                    loss = loss + lam_es * Le
                    comp["L_es"] = float(Le.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if ocfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(ocfg["grad_clip"]))
            opt.step()
            sched.step()
            ema.update_parameters(model)
            for k, v in comp.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            nb += 1
        tr = {k: v / max(1, nb) for k, v in acc.items()}
        if lam_sps != 0.0:
            tr["sps_tau"] = sps_tau
        if lam_map != 0.0:
            tr["map_alpha"] = map_alpha
        if score_aligned:
            tr["w_tke"] = lam_tke_ep
            tr["w_probe"] = lam_probe_ep
        lr_now = opt.param_groups[0]["lr"]
        line = (f"[ep {epoch:3d}] lr={lr_now:.3e} " +
                " ".join(f"{k}={v:.5f}" for k, v in sorted(tr.items())))
        row = dict(epoch=epoch, lr=lr_now, **{f"train/{k}": v for k, v in tr.items()})

        if (epoch + 1) % int(ecfg["every_epochs"]) == 0 or epoch == epochs - 1:
            val = evaluate_windows(ema.module, val_ld, device, gamma,
                                   float(lcfg["lambda_pin"]), float(lcfg["lambda_tke"]),
                                   amp=amp, dcfg=dcfg,
                                   lam_tmean=float(lcfg.get("lambda_tmean", 0.0) or 0.0),
                                   lam_probe=float(lcfg.get("lambda_probe", 0.0) or 0.0),
                                   tke_horizon=str(lcfg.get("tke_horizon", "flat") or "flat"),
                                   tke_horizon_c=float(lcfg.get("tke_horizon_c", 2.0) or 2.0),
                                   lam_tke_full=lcfg.get("lambda_tke_full"),
                                   lam_err=float(lcfg.get("lambda_err", 0.0) or 0.0))
            vm = M.compact({k: v for k, v in val.items() if not k.startswith("_")})
            vm.pop("loss_components", None)
            per_dom = M.aggregate(val["_arrays"]["pred"], val["_arrays"]["y"],
                                  val["_meta"], by="domain",
                                  lower=val["_arrays"]["lower"], upper=val["_arrays"]["upper"])
            line += (f" | val lbw={val['lbw']:.3f} rel={val['rel_l2']:.4f} "
                     f"tke={val['tke']:.4f} mvpe={val['mvpe']:.4f} "
                     f"sps={val['sps_score']:.2f} cov={val['coverage']:.3f} "
                     f"Lval={val['loss_components'].get('loss', float('nan')):.5f}")
            row.update({f"val/{k}": v for k, v in vm.items()
                        if isinstance(v, (int, float))})
            for dname, dv in per_dom.items():
                row[f"val_{dname}/lbw"] = dv["lbw"]
                row[f"val_{dname}/rel_l2"] = dv["rel_l2"]
            row.update({f"val/{k}": v for k, v in val["loss_components"].items()})
            record["metrics"]["val"] = vm
            record["metrics"].update({f"val_{k}": M.compact(v) for k, v in per_dom.items()})
            if score_aligned:
                e_r, e_t, e_m = float(val["rel_l2"]), float(val["tke"]), float(val["mvpe"])
                a_fac = 0.5 * (1 - e_r / (0.5 + e_r)) + 0.3 * (1 - e_t / (0.5 + e_t)) + 0.2 * (1 - e_m / (0.5 + e_m))
                sa_level = dict(rel=e_r, tke=e_t, mvpe=e_m,
                                band_q=float(val["sps_score"]) / 100.0 / max(a_fac, 1e-6))
            if val["lbw"] > best_lbw:
                best_lbw, best_epoch, bad = val["lbw"], epoch, 0
                save("best")
                line += "  *"
            else:
                bad += 1
        record["epochs"] = epoch + 1
        epoch_done = epoch + 1
        record["train_curve"].append(row)
        log(line)
        if wb is not None:
            wb.log(row, step=epoch)
        save("last")
        _sf, _st = int(ecfg.get("save_from", 0) or 0), int(ecfg.get("save_to", 0) or 0)
        if (int(ecfg.get("save_every", 0) or 0) > 0 and (epoch + 1) % int(ecfg["save_every"]) == 0
                and (epoch + 1) >= _sf and (_st <= 0 or (epoch + 1) <= _st)):
            save(f"ep{epoch + 1}", full=False)
        write_record()
        if bad >= int(ecfg["patience"]):
            log(f"[stop] no val improvement for {bad} validations (best epoch {best_epoch})")
            break
        # RPDE_TIME_BUDGET_MIN (default off): stop cleanly before a wall-clock limit;
        # the next chunk resumes exactly from last.pth (optimizer + schedule position).
        _bud = os.environ.get("RPDE_TIME_BUDGET_MIN")
        if _bud:
            _el = time.time() - t_start
            _per = _el / max(1, epoch + 1 - start_epoch)
            if _el + 1.3 * _per > 60.0 * float(_bud):
                log(f"[stop] time budget {_bud} min: {_el/60:.1f} min used after epoch {epoch + 1}")
                break

    meta["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    record["best_epoch"] = best_epoch
    record["best_val_lbw"] = best_lbw if math.isfinite(best_lbw) else None
    save("last")
    write_record()
    log(f"[done] best val lbw={best_lbw:.4f} @ epoch {best_epoch} "
        f"wall={(time.time()-t_start)/60:.1f} min -> {run_dir}")
    if wb is not None:
        wb.finish()
    log.close()
    return str(run_dir)


if __name__ == "__main__":
    main()
