#!/usr/bin/env python
"""Figures of the README.

    python tools/make_figures.py progress                  # docs/figures/progress{,_dark}.svg  (from submissions.csv)
    python tools/make_figures.py qualitative [--regime 26700_10]   # docs/figures/qualitative{,_dark}.png
                                                           # (GPU + data/cache32 + weights/; runs the eager ensemble)
Needs matplotlib (not a training dependency).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"

THEMES = {
    "light": dict(fg="#1f2328", muted="#59636e", grid="#d1d9e0", bg="none", line="#0969da", dot="#0969da",
                  bad="#cf222e", final="#1a7f37", shade=["#dafbe1", "#fff8c5", "#ddf4ff", "#fbefff", "#ffebe9", "#eaeef2"]),
    "dark": dict(fg="#e6edf3", muted="#9198a1", grid="#3d444d", bg="none", line="#4493f8", dot="#4493f8",
                 bad="#f85149", final="#3fb950", shade=["#12261e", "#2e2a14", "#0f2438", "#271a33", "#2f1a1a", "#1c2128"]),
}

# phases of the development (inclusive date ranges) and their label
PHASES = [
    ("2026-09-09", "2026-09-10", "physics\nbaseline"),
    ("2026-09-11", "2026-09-13", "learned\nmodel"),
    ("2026-09-14", "2026-09-16", "band\nloss"),
    ("2026-09-17", "2026-09-20", "compilation\n& decode"),
    ("2026-09-21", "2026-09-22", "architecture"),
    ("2026-09-23", "2026-09-28", "ensembles &\nsim-to-real"),
]
D = lambda d, h=0: dt.datetime(2026, 9, int(d), int(h), int(round((d % 1) * 0)))  # noqa: E731
# label placement per panel: name -> (text, x as (day, hour), y); dips drawn in the "bad" colour
LEFT = {"S1": ("S1  77.63\nphysics baseline", (10, 2), 78.35),
        "S2": ("S2  76.99\nuntrained core (slower)", (10, 14), 76.72),
        "S3": ("S3  79.91\naxial transformer", (10, 4), 80.75),
        "S4": ("S4  80.53\n2x2 patches", (12, 20), 79.85),
        "S7final": ("S7final  81.29\npinball + SPS reward", (14, 4), 80.25)}
RIGHT = {"S9": ("S9  81.28\ncompiled graph,\ngated decode", (17, 8), 81.40),
         "S11s2": ("S11s2  81.61\nMLP ratio 8", (16, 6), 81.80),
         "S13": ("S13  81.82\n20 frames in one token", (18, 6), 81.99),
         "S14": ("S14  81.84\n1x2 patches, 3.4 M params", (20, 20), 81.56),
         "ENS2": ("ENS2  81.93\ntwo-seed ensemble", (21, 0), 82.08),
         "ENS2c": ("fp16 outputs", (23, 0), 81.72),
         "ENS4g": ("mixed-patch\nensemble", (25, 8), 81.74),
         "FINAL": ("FINAL  82.01\nmirror augmentation, 100 % data,\nenergy-preserving combine", (23, 18), 81.40)}
BAD = {"S2", "ENS2c", "ENS4g"}


def _panel(ax, t, s, rows, c, theme, xlim, ylim, labels, phase_y):
    best = np.maximum.accumulate(s)
    for (a, b, lab), col in zip(PHASES, c["shade"]):
        x0 = dt.datetime.fromisoformat(a) - dt.timedelta(hours=12)
        x1 = dt.datetime.fromisoformat(b) + dt.timedelta(hours=12)
        ax.axvspan(x0, x1, color=col, alpha=0.9 if theme == "light" else 0.8, lw=0, zorder=0)
        xm = max(x0, xlim[0]) + (min(x1, xlim[1]) - max(x0, xlim[0])) / 2
        if min(x1, xlim[1]) - max(x0, xlim[0]) > dt.timedelta(hours=20):
            ax.text(xm, phase_y, lab, ha="center", va="top", fontsize=8.5, color=c["muted"], zorder=1)
    tb = t + [t[-1] + dt.timedelta(hours=10)]
    ax.step(tb, list(best) + [best[-1]], where="post", color=c["line"], lw=2.0, zorder=2)
    ax.scatter(t, s, s=24, color=c["dot"], zorder=3, edgecolor="white" if theme == "light" else "#0d1117", lw=0.8)
    ax.scatter([t[-1]], [s[-1]], s=64, color=c["final"], zorder=4, edgecolor="none")
    for r, ti, si in zip(rows, t, s):
        if r["name"] in labels:
            txt, (d, h), y = labels[r["name"]]
            col = c["bad"] if r["name"] in BAD else (c["final"] if r["name"] == "FINAL" else c["fg"])
            ax.annotate(txt, xy=(ti, si), xytext=(dt.datetime(2026, 9, d, h), y), fontsize=8.4, color=col,
                        ha="left", va="center", zorder=5,
                        arrowprops=dict(arrowstyle="-", color=col if r["name"] in BAD else c["muted"], lw=0.7,
                                        shrinkA=2, shrinkB=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=c["grid"], lw=0.6, zorder=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def progress():
    rows = list(csv.DictReader(open(FIG / "submissions.csv")))
    t = [dt.datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M") for r in rows]
    s = np.array([float(r["final_score"]) for r in rows])
    for theme, c in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": c["fg"],
                             "axes.labelcolor": c["fg"], "xtick.color": c["muted"], "ytick.color": c["muted"],
                             "axes.edgecolor": c["grid"], "svg.fonttype": "none"})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.14))
        fig.patch.set_alpha(0.0)
        _panel(a1, t, s, rows, c, theme, (dt.datetime(2026, 9, 9, 8), dt.datetime(2026, 9, 17, 0)), (76.45, 81.75),
               LEFT, 81.72)
        _panel(a2, t, s, rows, c, theme, (dt.datetime(2026, 9, 16, 0), dt.datetime(2026, 9, 28, 12)), (81.15, 82.16),
               RIGHT, 82.155)
        a1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        a2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        a1.set_ylabel("hidden-set final score")
        a1.set_title("development phase, 9-27 Sep 2026", fontsize=10, color=c["fg"], loc="left")
        a2.set_title("zoom: from 16 Sep", fontsize=10, color=c["fg"], loc="left")
        a1.plot([], [], color=c["line"], lw=2, label="best so far")
        a1.scatter([], [], s=24, color=c["dot"], label="submission")
        a1.legend(loc="lower right", frameon=False, fontsize=8.5)
        fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.1)
        out = FIG / ("progress.svg" if theme == "light" else "progress_dark.svg")
        fig.savefig(out, transparent=True)
        plt.close(fig)
        print("wrote", out)


def qualitative(regime: str, lead_idx=(0, 9, 19)):
    import torch
    sys.path.insert(0, str(ROOT))
    from rpde.ensemble_predict import EnsPredict, load_member
    z = np.load(ROOT / "data" / "cache32" / "real" / f"{regime}.npz") if not os.environ.get("RPDE_CACHE") else \
        np.load(Path(os.environ["RPDE_CACHE"]) / "real" / f"{regime}.npz")
    f = z["f"][..., :2]
    s0 = f.shape[0] - 40                                # the last window of the trajectory
    x = np.zeros((1, 20, 32, 64, 3), np.float32)
    x[0, ..., :2] = f[s0:s0 + 20]
    y = f[s0 + 20:s0 + 40]
    mask = ~z["mask"]
    members = [load_member(str(ROOT / "weights" / w), 0.70, "cuda") for w in ("fm_all_s3_ep59.pth", "fm_all_s4_ep59.pth")]
    ens = EnsPredict(members, band_scale=0.95, energy_preserve=True).eval()
    with torch.no_grad():
        p, lo, up = (a[0].cpu().numpy() for a in ens(torch.from_numpy(x).cuda()))
    speed = lambda a: np.hypot(a[..., 0], a[..., 1])  # noqa: E731
    for theme, c in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": c["fg"],
                             "axes.labelcolor": c["fg"], "xtick.color": c["muted"], "ytick.color": c["muted"]})
        fig, axs = plt.subplots(4, len(lead_idx), figsize=(10.5, 6.6), constrained_layout=True)
        fig.patch.set_alpha(0.0)
        vmax = float(np.percentile(speed(y), 99.5))
        emax = float(np.percentile(np.abs(speed(p[..., :2]) - speed(y)), 99.5))
        wmax = float(np.percentile(np.abs(up - lo)[..., 0][:, ~mask], 99.5))
        rows = [("measured |u|", lambda k: speed(y[k]), "viridis", 0, vmax),
                ("predicted |u|", lambda k: speed(p[k, ..., :2]), "viridis", 0, vmax),
                ("abs. error", lambda k: np.abs(speed(p[k, ..., :2]) - speed(y[k])), "magma", 0, emax),
                ("band width (u)", lambda k: (up - lo)[k, ..., 0], "cividis", 0, wmax)]
        for i, (lab, fn, cmap, lo_v, hi_v) in enumerate(rows):
            for j, k in enumerate(lead_idx):
                a = np.ma.array(fn(k), mask=mask)
                im = axs[i, j].imshow(a, cmap=cmap, vmin=lo_v, vmax=hi_v, origin="upper", interpolation="nearest")
                axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                for sp in axs[i, j].spines.values():
                    sp.set_color(c["grid"])
                if i == 0:
                    axs[i, j].set_title(f"lead {k + 1} / 20", fontsize=10, color=c["fg"])
                if j == 0:
                    axs[i, j].set_ylabel(lab, fontsize=10)
            cb = fig.colorbar(im, ax=axs[i, :], shrink=0.9, pad=0.01, aspect=12)
            cb.ax.tick_params(labelsize=8, colors=c["muted"]); cb.outline.set_edgecolor(c["grid"])
            cb.set_label("m/s", fontsize=8, color=c["muted"])
        re_, aoa = regime.split("_")
        fig.suptitle(f"Re {int(re_):,}, angle of attack {aoa} deg: last window of the trajectory (training data)",
                     fontsize=10.5, color=c["fg"])
        out = FIG / ("qualitative.png" if theme == "light" else "qualitative_dark.png")
        fig.savefig(out, dpi=150, transparent=True)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["progress", "qualitative"])
    ap.add_argument("--regime", default="26700_10")
    a = ap.parse_args()
    progress() if a.what == "progress" else qualitative(a.regime)
