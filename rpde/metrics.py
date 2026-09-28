"""Official metrics, the leaderboard combination, and subset aggregation.

Every number that can appear in a report comes from the organizers'
``scoring.py`` (imported from ``KIT_DIR``); nothing here reimplements a metric.

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .constants import KIT_DIR, LBW, TIME_SCORE_REF
from .constants import lbw as lbw_score

_kit = str(Path(KIT_DIR))
if _kit not in sys.path:
    sys.path.insert(0, _kit)

from scoring import (  # noqa: E402
    SIGMA_GLOBAL,
    aggregate_sps,
    measured_channels,
    mvpe_rel_l2_per_sample,
    rel_l2_per_sample,
    score_error,
    score_sps,
    score_time,
    tke_rel_l2_per_sample,
)

__all__ = [
    "SIGMA_GLOBAL", "aggregate_sps", "mvpe_rel_l2_per_sample", "rel_l2_per_sample",
    "score_error", "score_sps", "score_time", "tke_rel_l2_per_sample",
    "lbw_score", "score_windows", "aggregate", "subset_masks", "re_band",
]

# re_band lives in data.py but is needed here without importing the data layer
def re_band(re_nom: int) -> str:
    if re_nom <= 8850:
        return "low"
    if re_nom <= 16500:
        return "mid"
    return "high"


def _as5(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 5 or a.shape[-1] != 3:
        raise ValueError(f"expected (N, 20, 32, 64, 3), got {a.shape}")
    return a


def score_windows(pred, y, lower=None, upper=None, time_s: float = TIME_SCORE_REF) -> dict:
    """Official subscores for a set of windows.

    ``pred``/``y``: (N, 20, 32, 64, 3) float32 with ``p = 0``.  Pooling follows
    ``scoring.py``: the per-window error is averaged first, then ``score_error``
    is applied to that mean.  Returns the per-window arrays as well.
    """
    pred, y = _as5(pred), _as5(y)
    if pred.shape != y.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {y.shape}")
    c = measured_channels(y)
    rel = rel_l2_per_sample(pred, y, c)
    tke = tke_rel_l2_per_sample(pred, y, c)
    mvpe = mvpe_rel_l2_per_sample(pred, y)
    out = dict(
        n=int(pred.shape[0]), c=int(c),
        rel_l2_per_window=np.asarray(rel, dtype=np.float64),
        tke_per_window=np.asarray(tke, dtype=np.float64),
        mvpe_per_window=np.asarray(mvpe, dtype=np.float64),
        rel_l2=float(np.mean(rel)), tke=float(np.mean(tke)), mvpe=float(np.mean(mvpe)),
    )
    out["rel_l2_score"] = score_error(out["rel_l2"])
    out["tke_score"] = score_error(out["tke"])
    out["mvpe_score"] = score_error(out["mvpe"])

    if lower is not None and upper is not None:
        lower, upper = _as5(lower), _as5(upper)
        sps, coverage = aggregate_sps(pred, y, c, lower=lower, upper=upper)
        scored = y[..., :c] != 0.0
        width = (upper - lower)[..., :c]
        out["mean_width"] = float(width[scored].mean()) if scored.any() else float("nan")
        out["mean_width_sigma"] = out["mean_width"] / SIGMA_GLOBAL
        out["bounds"] = True
    else:
        sps, coverage = aggregate_sps(pred, y, c)     # scorer's default +-5% band
        out["mean_width"] = float("nan")
        out["mean_width_sigma"] = float("nan")
        out["bounds"] = False
    out["sps"] = float(sps)
    out["sps_score"] = score_sps(sps)
    out["coverage"] = float(coverage)
    out["time_score"] = float(time_s)
    out["lbw"] = lbw_score(out["rel_l2_score"], out["tke_score"], out["mvpe_score"],
                           out["sps_score"], time_s)
    return out


# --------------------------------------------------------------------------
# subsets
# --------------------------------------------------------------------------
def subset_masks(meta: dict, by: str) -> dict[str, np.ndarray]:
    """Boolean row masks over the window table.

    ``meta`` holds parallel arrays/lists: ``re``, ``aoa``, ``domain`` (0/1),
    ``name``, ``heldout`` (0/1).  ``by`` in
    {"all", "regime", "re", "aoa", "band", "heldout", "domain"}.
    """
    n = len(meta["re"])
    re_ = np.asarray(meta["re"])
    aoa = np.asarray(meta["aoa"])
    dom = np.asarray(meta.get("domain", np.zeros(n, dtype=int)))
    hel = np.asarray(meta.get("heldout", np.zeros(n, dtype=int)))
    dname = np.array(["real", "sim"])[dom]
    if by == "all":
        return {"all": np.ones(n, dtype=bool)}
    if by == "regime":
        keys = np.array([f"{d}/{r}_{a}" for d, r, a in zip(dname, re_, aoa)])
    elif by == "re":
        keys = np.array([f"Re{r}" for r in re_])
    elif by == "aoa":
        keys = np.array([f"AoA{a}" for a in aoa])
    elif by == "band":
        keys = np.array([re_band(int(r)) for r in re_])
    elif by == "heldout":
        keys = np.array(["heldout" if h else "train" for h in hel])
    elif by == "domain":
        keys = dname
    else:
        raise ValueError(f"unknown grouping {by!r}")
    return {str(k): (keys == k) for k in sorted(set(keys.tolist()))}


def aggregate(pred, y, meta: dict, by: str = "regime", lower=None, upper=None,
              time_s: float = TIME_SCORE_REF, min_n: int = 1) -> dict[str, dict]:
    """``score_windows`` per subset.  Keys as produced by ``subset_masks``."""
    out: dict[str, dict] = {}
    for key, m in subset_masks(meta, by).items():
        if int(m.sum()) < min_n:
            continue
        lo = None if lower is None else lower[m]
        up = None if upper is None else upper[m]
        out[key] = score_windows(pred[m], y[m], lo, up, time_s=time_s)
    return out


def compact(d: dict) -> dict:
    """Scalar-only view of a ``score_windows`` dict (for JSON / printing)."""
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}


def table(rows: dict[str, dict], cols=("n", "rel_l2", "tke", "mvpe", "coverage",
                                       "rel_l2_score", "tke_score", "mvpe_score",
                                       "sps_score", "lbw")) -> str:
    """Fixed-width text table of ``{subset: score_windows(...)}``."""
    w0 = max([len("subset")] + [len(k) for k in rows]) if rows else 6
    wc = {c: max(12, len(c) + 2) for c in cols}
    head = "subset".ljust(w0) + "".join(f"{c:>{wc[c]}}" for c in cols)
    lines = [head, "-" * len(head)]
    for k, v in rows.items():
        cells = []
        for c in cols:
            x = v.get(c, float("nan"))
            cells.append(f"{int(x):>{wc[c]}d}" if c == "n"
                         else f"{float(x):>{wc[c]}.4f}")
        lines.append(k.ljust(w0) + "".join(cells))
    return "\n".join(lines)
