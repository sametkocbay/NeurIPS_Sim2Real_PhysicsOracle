"""Data layer: cache index, regime table, folds, windows, sim alignment.

Local-only module (never shipped): may import yaml/scipy/h5py.  It reads the
32x64 float32 cache produced by ``rpde/tools/make_cache.py``:

    <CACHE_ROOT>/{real,sim}/<Re>_<AoA>.npz
        f       (T, 32, 64, C)  float32   C = 2 real [u, v], 3 sim [u, v, p]
        t       (T,)            float32
        mask    (32, 64)        bool      True = measured (u never exactly 0)
        re_nom, aoa, re_attr    scalars

Real data is dimensional (m/s), sim is nondimensional (U_inf ~ 1).  Nothing is
normalised here: `U_inf`, masks and drop-out maps are the model's job
(scaffold), so training and inference see the same code path.


"""
from __future__ import annotations

import json
import os
import re as _re
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import numpy.lib.format as _npfmt
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from .constants import (
    BAD_REAL,
    CACHE_ROOT,
    DTSTAR_SIM,
    FS_ROWS,
    H,
    HIDDEN_ROWS,
    MISSING_CORNERS,
    T_IN,
    T_OUT,
    T_VOL,
    U_PER_RE,
    W,
)
from .constants import sim_stride as auto_sim_stride

EVAL_STRIDE = 40          # >= T_OUT: no target frame inside another window's input
VAL_TAIL = 120            # frames (start indices) reserved at the end of every
                          # training trajectory for validation
DOMAIN_ID = {"real": 0, "sim": 1}
NOISE_NU = (0.015, 0.022)  # additive noise as a fraction of U_inf (REPORT Q2d)


# --------------------------------------------------------------------------
# regime table
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Regime:
    domain: str          # "real" | "sim"
    re: int
    aoa: int
    name: str            # "<Re>_<AoA>"
    T: int
    path: str

    @property
    def domain_id(self) -> int:
        return DOMAIN_ID[self.domain]

    @property
    def key(self) -> tuple:
        return (self.domain, self.re, self.aoa)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.domain}/{self.name}(T={self.T})"


def _npz_len(path: Path) -> int:
    """Number of frames in ``f`` without decompressing the array."""
    with zipfile.ZipFile(path) as z:
        with z.open("f.npy") as fh:
            ver = _npfmt.read_magic(fh)
            shape, _, _ = _npfmt._read_array_header(fh, ver)
    return int(shape[0])


def _index_path(root: Path, domain: str) -> Path:
    return root / f"_index_{domain}.json"


def _scan_domain(domain: str, root: Path) -> list[dict]:
    """(name, T) for every file of a domain; cached on disk keyed by mtime/size."""
    d = root / domain
    if not d.is_dir():
        raise FileNotFoundError(f"cache directory missing: {d}")
    files = sorted(d.glob("*.npz"))
    stamp = {p.name: [int(p.stat().st_mtime), int(p.stat().st_size)] for p in files}
    cache_file = _index_path(root, domain)
    cached = None
    if cache_file.is_file():
        try:
            with open(cache_file) as fh:
                blob = json.load(fh)
            if blob.get("stamp") == stamp:
                cached = blob["entries"]
        except Exception:
            cached = None
    if cached is None:
        cached = [{"name": p.stem, "T": _npz_len(p)} for p in files]
        try:
            with open(cache_file, "w") as fh:
                json.dump({"stamp": stamp, "entries": cached}, fh)
        except OSError:
            pass
    return cached


def parse_name(name: str) -> tuple[int, int]:
    m = _re.match(r"(\d+)_(\d+)$", Path(name).stem)
    if m is None:
        raise ValueError(f"unparseable case name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def regimes(domain: str, cache_root: str | os.PathLike | None = None) -> list[Regime]:
    """Every available regime of a domain, sorted by (Re, AoA).  BAD_REAL dropped."""
    root = Path(cache_root or CACHE_ROOT)
    out: list[Regime] = []
    for e in _scan_domain(domain, root):
        name = e["name"]
        if domain == "real" and name in BAD_REAL:
            continue
        re_nom, aoa = parse_name(name)
        out.append(Regime(domain, re_nom, aoa, name, int(e["T"]),
                          str(root / domain / f"{name}.npz")))
    return sorted(out, key=lambda r: (r.re, r.aoa))


def re_band(re_nom: int) -> str:
    """Re band used by the aggregation tables."""
    if re_nom <= 8850:
        return "low"
    if re_nom <= 16500:
        return "mid"
    return "high"


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------
def _never(re_nom: int, aoa: int) -> bool:
    return False


FOLDS: dict[str, dict[str, Callable[[int, int], bool]]] = {
    "full": dict(real_holdout=_never, sim_holdout=_never),
    "R-mid-a": dict(real_holdout=lambda re, aoa: re == 16500, sim_holdout=_never),
    "R-mid-b": dict(real_holdout=lambda re, aoa: re == 13950, sim_holdout=_never),
    "R-top": dict(real_holdout=lambda re, aoa: re == 26700, sim_holdout=_never),
    "R-corner": dict(real_holdout=lambda re, aoa: re >= 22875 and aoa in (5, 20),
                     sim_holdout=_never),
    "S-hidden": dict(real_holdout=_never,
                     sim_holdout=lambda re, aoa: re in HIDDEN_ROWS),
    # Novel-regime folds (final phase: private test at unseen AoA / Re, no sim there).
    # "x" = the held-out regime is removed from real AND sim.
    "A-10": dict(real_holdout=lambda re, aoa: aoa == 10, sim_holdout=_never),
    "A-10x": dict(real_holdout=lambda re, aoa: aoa == 10, sim_holdout=lambda re, aoa: aoa == 10),
    "A-20": dict(real_holdout=lambda re, aoa: aoa == 20, sim_holdout=_never),   # angle EXTRAPOLATION (diagnosis 09-26)
    "R-top-x": dict(real_holdout=lambda re, aoa: re == 26700, sim_holdout=lambda re, aoa: re >= 26700),
}


def split(fold: str, cache_root: str | os.PathLike | None = None) -> dict[str, list[Regime]]:
    """Whole-Re-row split.  Sim at a held-out *real* Re stays in ``train_sim``
    (that is exactly the hidden-set situation)."""
    if fold not in FOLDS:
        raise KeyError(f"unknown fold {fold!r}; known: {sorted(FOLDS)}")
    f = FOLDS[fold]
    real = regimes("real", cache_root)
    sim = regimes("sim", cache_root)
    ho_r, ho_s = f["real_holdout"], f["sim_holdout"]
    return dict(
        train_real=[r for r in real if not ho_r(r.re, r.aoa)],
        heldout_real=[r for r in real if ho_r(r.re, r.aoa)],
        train_sim=[r for r in sim if not ho_s(r.re, r.aoa)],
        heldout_sim=[r for r in sim if ho_s(r.re, r.aoa)],
    )


def limit(regs: Sequence[Regime], n: int | None) -> list[Regime]:
    """Evenly spaced subset of a regime list (keeps the Re range, incl. the top row)."""
    regs = list(regs)
    if n is None or n <= 0 or n >= len(regs):
        return regs
    idx = np.unique(np.round(np.linspace(0, len(regs) - 1, n)).astype(int))
    return [regs[i] for i in idx]


def _limit_for(limit_regimes, domain: str) -> int | None:
    if limit_regimes is None:
        return None
    if isinstance(limit_regimes, dict):
        v = limit_regimes.get(domain)
        return None if v is None else int(v)
    return int(limit_regimes)


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def window_span(k: int = 1) -> int:
    """Number of raw frames a 40-frame window at stride ``k`` covers."""
    return (T_VOL - 1) * k + 1


def t0_max(T: int, k: int = 1) -> int:
    """Largest legal window start (may be negative for a too-short trajectory)."""
    return T - window_span(k)


def t0_max_f(T: int, k: float = 1.0) -> int:
    """``t0_max`` for a possibly fractional stride (re-scaling augmentation)."""
    return int(np.floor(T - 1 - float(k) * (T_VOL - 1)))


def train_hi_f(T: int, k: float = 1.0) -> int:
    """``train_hi`` for a possibly fractional stride."""
    return t0_max_f(T, k) - VAL_TAIL


def eval_starts(T: int, k: int = 1) -> list[int]:
    """Fixed grid from 0 at stride 40 over the whole trajectory."""
    tm = t0_max(T, k)
    if tm < 0:
        return []
    return list(range(0, tm + 1, EVAL_STRIDE))


def val_starts(T: int, k: int = 1) -> list[int]:
    """Validation tail: the last three stride-40 starts of a training trajectory.

    For real data (k = 1) this is exactly ``T-120, T-80, T-40`` -- the last 120
    frames.  For strided sim windows the same rule is applied to the *start*
    index (the last 120 legal starts), see the implementation note in the spec.
    """
    tm = t0_max(T, k)
    if tm < 0:
        return []
    return sorted({s for s in (tm - 80, tm - 40, tm) if s >= 0})


def train_hi(T: int, k: int = 1) -> int:
    """Largest training start; training windows never start in the validation tail."""
    return t0_max(T, k) - VAL_TAIL


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class DataCfg:
    real_frac: float = 0.75
    sim_boost: float = 2.0
    sim_stride: object = "auto"     # "auto" | int | {re: int}
    sim_clock: bool = False         # opt-in: feed the TRUE sim dt*/advection and the
                                    # regime's physical U to Preproc
    sim_mask: bool = True
    sim_dropout: bool = True
    sim_noise: bool = True
    use_sim: bool = True
    batch_size: int = 8
    windows_per_epoch: int = 2000
    limit_regimes: object = None    # int | {"real": n, "sim": m} | None
    val_domains: tuple = ("real",)          # model selection on the target domain only
    noise_nu: tuple = NOISE_NU
    num_workers: int = 0
    # -- wave-4 opt-ins, all inert at their defaults --
    mirror_aug: float = 0.0         # p of mirroring a TRAIN window in y: flow at +AoA is exactly the mirror
                                    # of flow at -AoA with v -> -v, so this is a physically exact augmentation.
                                    # Applied to x, y and the mask together; conditioning is recomputed from
                                    # the mirrored mask, so nothing downstream needs to know.
    rescale_aug: object = None      # {"p":0.5,"k_choices":[2,3],"mode":"stride"}
                                    # or {"p":0.5,"k_range":[1.5,3.0],"mode":"interp"}
    regime_weights: object = None   # {"re_ge":21600,"re_factor":2.0,
                                    #  "aoa_eq":20,"aoa_factor":2.0}
    include_re: object = None       # [3750, 5025, 6300]: keep only these Re rows
    aux_p: bool = False             # opt-in: every window carries "p" (20,H,W), the sim pressure of the OUTPUT
                                    # frames (zero for real), for the auxiliary pressure target (loss.lambda_p)
    # -- input-side augmentation of TRAIN real windows: the members lose 0.5 lbw on
    # unseen regimes between their val-selected epoch and epoch 86 while in-regime val drops 0.19, i.e. they memorise
    # seen trajectories.  Both act on the 20 INPUT frames only (the target is untouched) and only on measured cells.
    real_noise_nu: object = None    # (lo, hi): additive Gaussian noise, sigma = nu * U_inf, nu ~ U(lo, hi) per window
    real_drop_p: float = 0.0        # extra per-cell drop-out probability on the input frames (PIV-like holes)
    amp_jitter: float = 0.0         # TRAIN real windows: fluctuation about each cell's 40-frame mean scaled by 1 + U(-a, a)
                                    # (input and target together; measured cells only) -- a turbulence-intensity jitter
    ctx_drop: object = None         # {"p":0.5,"k":[5,10]}: with prob p zero the FIRST k input frames of a TRAIN window (real and
                                    # sim); a memorised trajectory needs a long history, a short context forces local dynamics.
                                    # Preproc detects all-zero frames and the baseline / drop-out maps use the visible frames.
    mask_aug: object = None         # {"p_dilate":0.3,"holes":[1,3],"hole_hw":[[2,4],[4,8]],"hole_frames":[1,5]}: TRAIN real windows:
                                    # with p_dilate grow the static core mask by one cell (all 40 frames; the conditioning geometry
                                    # jitters with it), and cut `holes` random rectangles out of a few consecutive INPUT frames.
    phase_aug: bool = False         # TRAIN real windows drawn from a random 2x2 sub-pixel phase of the raw 64x128 PIV:
                                    # the other three [i::2, j::2] grids are independently measured 32x64 fields of the
                                    # same flows; val/eval stay on (0,0)
    train_tail: bool = False        # diagnosis 2026-09-26: TRAIN real windows may also start inside the validation tail
                                    # (the last 120 frames, 14 % of the real data). Validation then scores seen frames;
                                    # select the epoch by a counterpart run, not by val.
    train_all: bool = False         # final wave 2026-09-27: 100 % of every TRAINING regime, real AND sim -- windows may start
                                    # anywhere (t0_max, validation tails included) and no regime is dropped for being too short
                                    # for a window after the tail.  Val then scores seen frames: pick the epoch by held-out folds.

    @classmethod
    def from_dict(cls, d: dict | None) -> "DataCfg":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"unknown data config keys: {sorted(unknown)}")
        if "val_domains" in d and d["val_domains"] is not None:
            d["val_domains"] = tuple(d["val_domains"])
        if "noise_nu" in d and d["noise_nu"] is not None:
            d["noise_nu"] = tuple(float(x) for x in d["noise_nu"])
        if "real_noise_nu" in d and d["real_noise_nu"] is not None:
            d["real_noise_nu"] = tuple(float(x) for x in d["real_noise_nu"])
        return cls(**d)


def stride_for(re_nom: int, cfg: DataCfg | None = None) -> int:
    """Sim frame stride k(Re) -- ``constants.sim_stride`` unless overridden."""
    s = "auto" if cfg is None else cfg.sim_stride
    if s is None or s == "auto":
        return auto_sim_stride(re_nom)
    if isinstance(s, dict):
        for key in (re_nom, str(re_nom)):
            if key in s:
                return int(s[key])
        return auto_sim_stride(re_nom)
    return int(s)


def filter_re(regs: Sequence[Regime], include_re) -> list[Regime]:
    """Keep only the regimes whose nominal Re is in ``include_re`` (``None`` = all).

    Used by the weight-space-direction endpoint fine-tunes: the expert sees ONLY the
    endpoint rows, real and sim alike.
    """
    if not include_re:
        return list(regs)
    keep = {int(r) for r in include_re}
    return [r for r in regs if r.re in keep]


def parse_rescale_aug(spec) -> dict | None:
    """Validate ``data.rescale_aug``; ``None`` when the augmentation is off."""
    if not spec:
        return None
    d = dict(spec)
    unknown = set(d) - {"p", "k_choices", "k_range", "mode"}
    if unknown:
        raise KeyError(f"unknown rescale_aug keys: {sorted(unknown)}")
    p = float(d.get("p", 0.0))
    if p <= 0.0:
        return None
    mode = str(d.get("mode", "stride"))
    if mode not in ("stride", "interp"):
        raise ValueError(f"rescale_aug.mode must be 'stride' or 'interp', got {mode!r}")
    out = dict(p=p, mode=mode)
    if mode == "stride":
        out["k_choices"] = [int(k) for k in d.get("k_choices", [2, 3])]
        if any(k < 1 for k in out["k_choices"]):
            raise ValueError("rescale_aug.k_choices must be >= 1")
    else:
        lo, hi = d.get("k_range", (1.5, 3.0))
        out["k_range"] = (float(lo), float(hi))
        # k < 1 (interp mode) presents the window as a LOWER Re: denser frames, velocities scaled down; the fractional
        # window bound train_hi_f handles any k > 0
        if not (out["k_range"][0] > 0.0 and out["k_range"][1] >= out["k_range"][0]):
            raise ValueError("rescale_aug.k_range must satisfy 0 < lo <= hi")
    return out


def parse_regime_weights(spec) -> dict | None:
    if not spec:
        return None
    d = dict(spec)
    unknown = set(d) - {"re_ge", "re_factor", "aoa_eq", "aoa_factor"}
    if unknown:
        raise KeyError(f"unknown regime_weights keys: {sorted(unknown)}")
    return d


def regime_factor(reg: Regime, rw: dict | None) -> float:
    """Multiplicative sampling-weight factor of one regime (1.0 when off).

    ``re_factor`` applies when ``Re >= re_ge``, ``aoa_factor`` when
    ``AoA == aoa_eq``; the two multiply.  Applies to real and sim alike.
    """
    if not rw:
        return 1.0
    m = 1.0
    if rw.get("re_ge") is not None and reg.re >= int(rw["re_ge"]):
        m *= float(rw.get("re_factor", 1.0))
    if rw.get("aoa_eq") is not None and reg.aoa == int(rw["aoa_eq"]):
        m *= float(rw.get("aoa_factor", 1.0))
    return m


# --------------------------------------------------------------------------
# in-memory corpus
# --------------------------------------------------------------------------
def to3(a: np.ndarray) -> np.ndarray:
    """(..., 2) -> (..., 3) with p = 0, as the scorer sees real data."""
    if a.shape[-1] == 3:
        return np.ascontiguousarray(a, dtype=np.float32)
    out = np.zeros(a.shape[:-1] + (3,), dtype=np.float32)
    out[..., :2] = a
    return out


class Corpus:
    """Holds the decompressed cache in RAM (~3.4 GB for everything).

    Also caches, per real file, the two quantities the sim forward model needs:
    the core mask (cells whose ``u`` is zero in every frame) and the per-cell
    drop-out probability map (fraction of frames with ``u == 0``).
    """

    def __init__(self, cache_root: str | os.PathLike | None = None):
        self.root = Path(cache_root or CACHE_ROOT)
        self._f: dict[tuple, np.ndarray] = {}
        self._valid: dict[tuple, np.ndarray] = {}
        self._pmap: dict[tuple, np.ndarray] = {}

    # -- loading -----------------------------------------------------------
    def load(self, reg: Regime) -> None:
        if reg.key in self._f:
            return
        with np.load(reg.path) as z:
            f = np.ascontiguousarray(z["f"], dtype=np.float32)
            valid = np.asarray(z["mask"], dtype=bool)
        self._f[reg.key] = f
        self._valid[reg.key] = valid
        if reg.domain == "real":
            self._pmap[reg.key] = (f[..., 0] == 0.0).mean(0).astype(np.float32)

    def load_all(self, regs: Sequence[Regime]) -> "Corpus":
        for r in regs:
            self.load(r)
        return self

    PHASES = ((0, 1), (1, 0), (1, 1))

    def load_phases(self, regs: Sequence[Regime]) -> int:
        """Load the three other sub-pixel phases of every REAL regime from ``<root>/real_p<ij>/<name>.npz``.
        Returns the number of (regime, phase) arrays loaded; regimes whose phase files are missing are skipped
        (they then always draw phase (0,0))."""
        n = 0
        for r in regs:
            if r.domain != "real":
                continue
            for (i, j) in self.PHASES:
                key = (r.key, i, j)
                if key in self._f:
                    n += 1
                    continue
                path = self.root / f"real_p{i}{j}" / f"{r.name}.npz"
                if not path.is_file():
                    continue
                with np.load(path) as z:
                    self._f[key] = np.ascontiguousarray(z["f"], dtype=np.float32)
                n += 1
        return n

    def has_phases(self, reg: Regime) -> bool:
        return all((reg.key, i, j) in self._f for (i, j) in self.PHASES)

    @property
    def n_loaded(self) -> int:
        return len(self._f)

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self._f.values()))

    # -- access ------------------------------------------------------------
    def f(self, reg: Regime, phase: tuple | None = None) -> np.ndarray:
        self.load(reg)
        if phase and phase != (0, 0):
            return self._f[(reg.key, int(phase[0]), int(phase[1]))]
        return self._f[reg.key]

    def valid(self, reg: Regime) -> np.ndarray:
        self.load(reg)
        return self._valid[reg.key]

    def pmap(self, reg: Regime) -> np.ndarray:
        self.load(reg)
        return self._pmap[reg.key]


def load_all(split_part: Sequence[Regime], limit_regimes=None,
             corpus: Corpus | None = None) -> tuple[Corpus, list[Regime]]:
    """Read the (optionally limited) regimes of one split part into RAM once."""
    regs = list(split_part)
    if regs:
        regs = limit(regs, _limit_for(limit_regimes, regs[0].domain))
    corpus = corpus or Corpus()
    corpus.load_all(regs)
    return corpus, regs


def match_real(re_nom: int, aoa: int, pool: Sequence[Regime]) -> Regime | None:
    """Real file whose mask / drop-out map stands in for a sim regime:
    exact (Re, AoA), else the nearest Re at the same AoA, else nearest overall."""
    if not pool:
        return None
    exact = [r for r in pool if r.re == re_nom and r.aoa == aoa]
    if exact:
        return exact[0]
    same_aoa = [r for r in pool if r.aoa == aoa]
    if same_aoa:
        return min(same_aoa, key=lambda r: abs(r.re - re_nom))
    return min(pool, key=lambda r: (abs(r.aoa - aoa), abs(r.re - re_nom)))


# --------------------------------------------------------------------------
# window construction
# --------------------------------------------------------------------------
def _u_inf(w: np.ndarray) -> float:
    """Median of the nonzero streamwise velocity over the freestream band."""
    u = w[:, FS_ROWS, :, 0]
    u = u[u != 0.0]
    return float(np.median(u)) if u.size else 1.0


def rescale_real(f: np.ndarray, t0: int, k: float, mode: str) -> np.ndarray:
    """Re-window a REAL trajectory at frame stride ``k`` and scale by ``k``.

    The window covers raw frames ``t0 + k*[0..39]`` and every velocity is
    multiplied by ``k``, so U_inf, ``dt* = DT*U/CHORD`` and the per-frame
    advection ``a = U*DT/DX`` all scale by ``k``: the sample presents as
    ``Re' = k*Re`` filmed at the original frame rate.

    ``mode="stride"``: integer ``k``, raw frames are taken as they are, so the
    per-frame drop-out pattern and the static mask are exactly a real window's.
    ``mode="interp"``: fractional ``k``, linear interpolation in time, with the
    zero pattern (mask + drop-out) copied from the NEAREST raw frame so the
    drop-out statistics stay those of real data instead of being smeared.
    """
    T = f.shape[0]
    if mode == "stride":
        ki = int(round(float(k)))
        idx = t0 + ki * np.arange(T_VOL)
        w = f[idx][..., :2].astype(np.float32, copy=True)
        return w * np.float32(ki)
    pos = float(t0) + float(k) * np.arange(T_VOL, dtype=np.float64)
    i0 = np.floor(pos).astype(np.int64).clip(0, T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    fr = (pos - i0).astype(np.float32).reshape(T_VOL, 1, 1, 1)
    a = f[i0][..., :2].astype(np.float32)
    b = f[i1][..., :2].astype(np.float32)
    w = (1.0 - fr) * a + fr * b
    near = np.rint(pos).astype(np.int64).clip(0, T - 1)
    w[f[near][..., :2] == 0.0] = 0.0
    return np.ascontiguousarray(w * np.float32(k))


def make_window(reg: Regime, t0: int, corpus: Corpus, cfg: DataCfg,
                rng: np.random.Generator, mask_pool: Sequence[Regime] = (),
                mask_src: dict | None = None, rescale=None,
                augment: bool = False, phase: tuple | None = None) -> dict:
    """One training/eval sample.  Sim windows go through the forward model of
    Sim-to-real input transforms: stride, noise, mask, drop-out.

    ``rescale=(k, mode)`` (real only, opt-in) re-windows at stride ``k`` and
    multiplies the velocities by ``k`` -- see ``rescale_real``.

    ``augment=True`` (TrainWindows only) enables the opt-in input-side augmentation of real windows
    (``cfg.real_noise_nu`` / ``cfg.real_drop_p``); with both at their defaults nothing is drawn from ``rng``."""
    f = corpus.f(reg, phase) if reg.domain == "real" else corpus.f(reg)
    if reg.domain == "real":
        k = 1
        if rescale is not None:
            k = rescale[0]
            w = rescale_real(f, t0, rescale[0], rescale[1])
        else:
            w = f[t0:t0 + T_VOL, :, :, :2]
        if augment and float(getattr(cfg, "amp_jitter", 0.0) or 0.0) > 0.0:
            w = np.array(w, dtype=np.float32, copy=True)          # the cache array itself is never modified
            meas = w != 0.0
            cnt = meas.sum(0, keepdims=True).clip(min=1)
            mean40 = (w * meas).sum(0, keepdims=True) / cnt           # per-cell mean over its measured frames
            eps = float(rng.uniform(-cfg.amp_jitter, cfg.amp_jitter))
            w = np.where(meas, mean40 + (1.0 + eps) * (w - mean40), 0.0).astype(np.float32)
        if augment and cfg.mask_aug:
            ma = dict(cfg.mask_aug)
            w = np.array(w, dtype=np.float32, copy=True)
            if float(rng.random()) < float(ma.get("p_dilate", 0.0)):
                core = (w[..., 0] == 0.0).all(0)                       # static core mask of this window (H,W)
                dil = core.copy()
                dil[1:] |= core[:-1]; dil[:-1] |= core[1:]; dil[:, 1:] |= core[:, :-1]; dil[:, :-1] |= core[:, 1:]
                w[:, dil & ~core] = 0.0                                 # one-cell dilation, input and target
            lo_n, hi_n = ma.get("holes", [1, 3]); (h_lo, h_hi), (w_lo, w_hi) = ma.get("hole_hw", [[2, 4], [4, 8]])
            f_lo, f_hi = ma.get("hole_frames", [1, 5])
            for _ in range(int(rng.integers(int(lo_n), int(hi_n) + 1))):
                hh, ww = int(rng.integers(h_lo, h_hi + 1)), int(rng.integers(w_lo, w_hi + 1))
                r0, c0 = int(rng.integers(0, H - hh + 1)), int(rng.integers(0, W - ww + 1))
                nf = int(rng.integers(f_lo, f_hi + 1)); t_a = int(rng.integers(0, T_IN - nf + 1))
                w[t_a:t_a + nf, r0:r0 + hh, c0:c0 + ww] = 0.0           # input frames only
        if augment and (cfg.real_noise_nu or float(cfg.real_drop_p or 0.0) > 0.0):
            w = np.array(w, dtype=np.float32, copy=True)          # the cache array itself is never modified
            xin = w[:T_IN]
            measured = xin != 0.0
            if cfg.real_noise_nu:
                nu = float(rng.uniform(*cfg.real_noise_nu))
                xin[measured] += rng.normal(0.0, nu * _u_inf(xin), size=int(measured.sum())).astype(np.float32)
            if float(cfg.real_drop_p or 0.0) > 0.0:
                drop = rng.random(xin.shape[:3]) < float(cfg.real_drop_p)
                xin[drop] = 0.0
    else:
        k = stride_for(reg.re, cfg)
        idx = t0 + k * np.arange(T_VOL)
        w = f[idx][..., :2].astype(np.float32, copy=True)
        pw = f[idx][..., 2].astype(np.float32, copy=True) if cfg.aux_p else None
        # 4. noise first, so that the zeroing below leaves exact zeros
        if cfg.sim_noise:
            nu = float(rng.uniform(*cfg.noise_nu))
            w = w + rng.normal(0.0, nu * _u_inf(w), size=w.shape).astype(np.float32)
        src = None
        if mask_src is not None:
            src = mask_src.get((reg.re, reg.aoa))
        if src is None and mask_pool:
            src = match_real(reg.re, reg.aoa, mask_pool)
        if src is not None:
            # 2. body + shadow mask of the matched real regime
            if cfg.sim_mask:
                w = w * corpus.valid(src)[None, :, :, None]
                if pw is not None:
                    pw = pw * corpus.valid(src)[None]
            # 3. simulated per-frame drop-out with the real probability map
            if cfg.sim_dropout:
                drop = rng.random((T_VOL, H, W)) < corpus.pmap(src)[None]
                w[drop] = 0.0
                if pw is not None:
                    pw[drop] = 0.0
    if augment and cfg.ctx_drop and float(rng.random()) < float(dict(cfg.ctx_drop).get("p", 0.0)):
        k_lo, k_hi = dict(cfg.ctx_drop).get("k", [5, 10])
        k = int(rng.integers(int(k_lo), int(k_hi) + 1))
        w = np.array(w, dtype=np.float32, copy=True)
        w[:k] = 0.0                                              # hide the first k input frames (the target is untouched)
    w = np.ascontiguousarray(w, dtype=np.float32)
    x = to3(w[:T_IN])
    y = to3(w[T_IN:T_VOL])
    out = dict(
        x=torch.from_numpy(x), y=torch.from_numpy(y),
        domain=reg.domain_id, re=reg.re, aoa=reg.aoa, name=reg.name,
        t0=int(t0), stride=int(round(float(k))), k_used=float(k),
    )
    if cfg.aux_p:
        pp = pw[T_IN:T_VOL] if reg.domain != "real" else np.zeros((T_VOL - T_IN,) + w.shape[1:3], np.float32)
        out["p"] = torch.from_numpy(np.ascontiguousarray(pp, dtype=np.float32))
    return out


class _Base(Dataset):
    def __init__(self, corpus: Corpus, cfg: DataCfg, seed: int = 0,
                 mask_pool: Sequence[Regime] = ()):
        self.corpus = corpus
        self.cfg = cfg
        self.seed = int(seed)
        self.mask_pool = list(mask_pool)
        self._mask_src: dict[tuple[int, int], Regime | None] = {}

    def _src(self, reg: Regime):
        key = (reg.re, reg.aoa)
        if key not in self._mask_src:
            self._mask_src[key] = match_real(reg.re, reg.aoa, self.mask_pool)
        return self._mask_src[key]

    def _make(self, reg: Regime, t0: int, rng, rescale=None, mirror: bool = False,
              augment: bool = False, phase: tuple | None = None) -> dict:
        if reg.domain != "real":
            self._src(reg)          # populate the (re, aoa) -> real-file cache
        w = make_window(reg, t0, self.corpus, self.cfg, rng,
                        mask_pool=self.mask_pool, mask_src=self._mask_src,
                        rescale=rescale, augment=augment, phase=phase)
        return _mirror_window(w) if mirror else w


def _mirror_window(w: dict) -> dict:
    """Reflect a window across the y axis: flip the H axis and negate v.

    Exact symmetry of the Navier-Stokes solution (flow at +AoA is the mirror of flow at -AoA), so the mirrored
    sample is as physical as the original.  `x` and `y` are (T, H, W, C) TORCH tensors with C = (u, v, p); the
    body mask travels inside them, so flipping x and y is the whole transform.  torch has no negative strides,
    hence torch.flip rather than [::-1].
    """
    out = dict(w)
    for k in ("x", "y"):
        v = out.get(k)
        if v is None:
            continue
        v = torch.flip(v, dims=(1,)) if torch.is_tensor(v) else np.ascontiguousarray(v[:, ::-1])
        v = v.clone() if torch.is_tensor(v) else v
        v[..., 1] = -v[..., 1]                           # v is antisymmetric under a y flip
        out[k] = v
    if out.get("p") is not None:                         # pressure is a scalar: flip only
        out["p"] = torch.flip(out["p"], dims=(1,))
    return out


class TrainWindows(_Base):
    """``windows_per_epoch`` random windows, balanced by regime.

    Regime weights: real regimes uniform summing to ``real_frac``; sim regimes
    weighted by ``sim_boost`` at the hidden rows and the missing corners,
    summing to ``1 - real_frac``.  The draw for index ``i`` of epoch ``e`` is a
    pure function of ``(seed, e, i)``, so the data order is reproducible with
    ``num_workers=0`` and ``shuffle=False``.
    """

    def __init__(self, train_real: Sequence[Regime], train_sim: Sequence[Regime],
                 corpus: Corpus, cfg: DataCfg, seed: int = 0,
                 mask_pool: Sequence[Regime] | None = None):
        super().__init__(corpus, cfg, seed,
                         mask_pool=train_real if mask_pool is None else mask_pool)
        tall = bool(getattr(cfg, "train_all", False))
        hi_fn = t0_max if tall else train_hi
        real = [r for r in train_real if hi_fn(r.T, 1) >= 0]
        sim = []
        if cfg.use_sim:
            sim = [r for r in train_sim if hi_fn(r.T, stride_for(r.re, cfg)) >= 0]
        self.regs = list(real) + list(sim)
        if not self.regs:
            raise ValueError("no training regime has a usable window range")
        wr = np.ones(len(real), dtype=np.float64)
        ws = np.array([cfg.sim_boost if (r.re in HIDDEN_ROWS
                                         or (r.re, r.aoa) in MISSING_CORNERS) else 1.0
                       for r in sim], dtype=np.float64)
        # opt-in regime oversampling: multiplies the
        # weight of a regime inside its own domain group, so `real_frac` and the
        # real/sim split are untouched.  Skipped entirely when off, so the
        # default weight vector is bit-identical.
        self.regime_weights = parse_regime_weights(getattr(cfg, "regime_weights", None))
        if self.regime_weights:
            if len(real):
                wr = wr * np.array([regime_factor(r, self.regime_weights) for r in real],
                                   dtype=np.float64)
            if len(sim):
                ws = ws * np.array([regime_factor(r, self.regime_weights) for r in sim],
                                   dtype=np.float64)
        frac = cfg.real_frac if (len(real) and len(sim)) else (1.0 if len(real) else 0.0)
        w = np.concatenate([
            wr / wr.sum() * frac if len(real) else np.zeros(0),
            ws / ws.sum() * (1.0 - frac) if len(sim) else np.zeros(0),
        ])
        self.weights = w / w.sum()
        self.n_real, self.n_sim = len(real), len(sim)
        self.epoch = 0
        self._hi = [(t0_max(r.T, 1) if (r.domain == "real" and bool(getattr(cfg, "train_tail", False)))
                     else hi_fn(r.T, 1 if r.domain == "real" else stride_for(r.re, cfg)))
                    for r in self.regs]
        self.train_all = tall
        self.rescale_aug = parse_rescale_aug(getattr(cfg, "rescale_aug", None))
        # per-regime, per-k largest legal start for the rescaling augmentation
        self._hi_k: dict[tuple[int, float], int] = {}

    def weight_table(self) -> list[tuple[str, str, float]]:
        """``(domain, name, weight)`` of every training regime, in draw order."""
        return [(r.domain, r.name, float(w)) for r, w in zip(self.regs, self.weights)]

    def _draw_k(self, rng) -> float:
        a = self.rescale_aug
        if a["mode"] == "stride":
            return float(a["k_choices"][int(rng.integers(0, len(a["k_choices"])))])
        lo, hi = a["k_range"]
        return float(rng.uniform(lo, hi))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.cfg.windows_per_epoch)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        j = int(rng.choice(len(self.regs), p=self.weights))
        reg = self.regs[j]
        hi, rescale = self._hi[j], None
        # opt-in Re-rescaling augmentation, REAL windows only (sim windows
        # already carry their own stride alignment).  Nothing is drawn from
        # `rng` when the augmentation is off -> the default stream is unchanged.
        if self.rescale_aug is not None and reg.domain == "real":
            if float(rng.random()) < self.rescale_aug["p"]:
                k = self._draw_k(rng)
                key = (j, k)
                if key not in self._hi_k:
                    self._hi_k[key] = t0_max_f(reg.T, k) if self.train_all else train_hi_f(reg.T, k)
                if self._hi_k[key] >= 0:
                    hi, rescale = self._hi_k[key], (k, self.rescale_aug["mode"])
        t0 = int(rng.integers(0, hi + 1))
        # mirror AFTER the window is drawn, and only when enabled, so the default rng stream is untouched
        mirror = bool(self.cfg.mirror_aug) and float(rng.random()) < float(self.cfg.mirror_aug)
        # sub-pixel phase LAST and only when enabled, so every other draw of the stream is unchanged; real regimes
        # without phase files (or sim) stay on (0,0)
        phase = None
        if bool(self.cfg.phase_aug) and reg.domain == "real" and rescale is None and self.corpus.has_phases(reg):
            k = int(rng.integers(0, 4))
            phase = ((0, 0), (0, 1), (1, 0), (1, 1))[k]
        return self._make(reg, t0, rng, rescale=rescale, mirror=mirror, augment=True, phase=phase)


class FixedWindows(_Base):
    """Deterministic window list: ``mode='val'`` (training tails) or
    ``mode='eval'`` (stride-40 grid over the whole trajectory)."""

    def __init__(self, regs: Sequence[Regime], corpus: Corpus, cfg: DataCfg,
                 mode: str = "eval", seed: int = 0,
                 mask_pool: Sequence[Regime] = ()):
        super().__init__(corpus, cfg, seed, mask_pool=mask_pool)
        if mode not in ("val", "eval"):
            raise ValueError(f"mode must be 'val' or 'eval', got {mode!r}")
        self.mode = mode
        starts = val_starts if mode == "val" else eval_starts
        self.items: list[tuple[Regime, int]] = []
        for r in regs:
            if mode == "val" and cfg.val_domains and r.domain not in cfg.val_domains:
                continue
            k = 1 if r.domain == "real" else stride_for(r.re, cfg)
            for t0 in starts(r.T, k):
                self.items.append((r, t0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        reg, t0 = self.items[i]
        # deterministic per (seed, file, t0): sim augmentation is reproducible
        rng = np.random.default_rng([self.seed, zlib.crc32(reg.name.encode()),
                                     reg.domain_id, int(t0)])
        return self._make(reg, t0, rng)


def collate(batch: list[dict]) -> dict:
    """``default_collate``; ``name`` comes back as a list of str."""
    return default_collate(batch)


def make_loader(dataset: Dataset, cfg: DataCfg, batch_size: int | None = None,
                shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size or cfg.batch_size,
                      shuffle=shuffle, num_workers=cfg.num_workers,
                      collate_fn=collate, drop_last=False)


def clock_override(batch, cfg: DataCfg | None = None):
    """(B,2) [dt*_true, U_physical] for the batch, or None when the fix is off.

    Real windows get zeros (= "derive from the window", the shipping path).
    Simulation windows get `dt* = DTSTAR_SIM * stride` -- the sim's own
    convective step per stored frame times the stride actually used -- and the
    regime's physical freestream `U_PER_RE * Re` for the conditioning.
    """
    if cfg is not None and not bool(getattr(cfg, "sim_clock", False)):
        return None
    dom = torch.as_tensor(batch["domain"]).reshape(-1).float()
    re = torch.as_tensor(batch["re"]).reshape(-1).float()
    k = torch.as_tensor(batch["stride"]).reshape(-1).float()
    is_sim = (dom > 0).float()
    dt_ov = is_sim * (DTSTAR_SIM * k.clamp(min=1.0))
    u_ov = is_sim * (U_PER_RE * re)
    return torch.stack([dt_ov, u_ov], dim=1)
