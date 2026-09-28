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

THEMES = {  # warm neutrals with a clay accent
    "light": dict(fg="#28261f", muted="#6f6b5e", faint="#a8a494", grid="#e4e1d6", accent="#c96442",
                  dot="#c96442", other="#8f8b7c", bad="#b54a4a", final="#c96442", bg="#ffffff"),
    "dark": dict(fg="#ece9df", muted="#aaa597", faint="#6f6b5e", grid="#3a3830", accent="#e0805f",
                 dot="#e0805f", other="#9a968a", bad="#e07a72", final="#e0805f", bg="#141310"),
}

# annotation per submission: (text, offset in points (dx, dy), horizontal alignment); None = marker only
LABELS = {
    "S1": ("physics baseline", (6, 22), "left"),
    "S2": ("· untrained core", (12, 3), "left"),
    "S3": ("axial transformer", (-10, 26), "right"),
    "S4": ("2x2 patches", (8, -24), "left"),
    "S6lr": ("learning rate", (-6, 28), "right"),
    "S7final": ("pinball loss +\nband reward", (-10, 30), "right"),
    "S9": ("compiled graph +\ngated decode", (12, -12), "left"),
    "S11s2": ("MLP ratio 8", (-10, 26), "right"),
    "S12": ("decode retune", (6, -30), "left"),
    "S13": ("one temporal\ntoken", (-10, 30), "right"),
    "S14": ("1x2 patches,\n3.4 M params", (8, -30), "left"),
    "ENS2": ("2-seed\nensemble", (-10, 30), "right"),
    "ENS2c": ("fp16 outputs", (10, -96), "left"),
    "ENS4g": ("mixed-patch\nensemble", (6, -46), "left"),
    "FINAL": ("mirror aug., all data,\nenergy-preserving combine", (-14, 22), "right"),
}
BREAK, LO, HI = 81.0, 76.55, 82.3      # y axis: [LO, BREAK] linear on 1 unit, [BREAK, HI] linear on UP units
UP = 1.75


def _fwd(y):
    y = np.asarray(y, dtype=float)
    return np.where(y < BREAK, (y - LO) / (BREAK - LO), 1.0 + (y - BREAK) / (HI - BREAK) * UP)


def _inv(v):
    v = np.asarray(v, dtype=float)
    return np.where(v < 1.0, LO + v * (BREAK - LO), BREAK + (v - 1.0) / UP * (HI - BREAK))


def progress():
    rows = list(csv.DictReader(open(FIG / "submissions.csv")))
    names = [r["name"] for r in rows]
    s = np.array([float(r["final_score"]) for r in rows])
    days = [dt.datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").strftime("%d %b") for r in rows]
    best = np.maximum.accumulate(s)
    improved = np.r_[True, s[1:] > best[:-1]]
    x = np.arange(len(rows))
    for theme, c in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": c["fg"],
                             "axes.labelcolor": c["fg"], "xtick.color": c["muted"], "ytick.color": c["muted"],
                             "axes.edgecolor": c["grid"], "svg.fonttype": "none"})
        fig, ax = plt.subplots(figsize=(12, 5.6))
        fig.patch.set_facecolor(c["bg"])
        ax.set_facecolor(c["bg"])
        ax.set_yscale("function", functions=(_fwd, _inv))
        # horizontal grid and the scale break
        for yt in (77, 78, 79, 80, 81.2, 81.4, 81.6, 81.8, 82.0):
            ax.axhline(yt, color=c["grid"], lw=0.6, zorder=1)
        ax.axhline(BREAK, color=c["faint"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(len(rows) - 0.45, BREAK, "scale changes", ha="right", va="bottom", fontsize=7.5, color=c["faint"])
        # best-so-far staircase
        xs = np.r_[x, len(rows) - 0.5]
        ax.step(xs, np.r_[best, best[-1]], where="post", color=c["accent"], lw=1.8, alpha=0.9, zorder=2)
        # markers: improving submissions filled, the others hollow
        ax.scatter(x[improved], s[improved], s=38, color=c["dot"], zorder=4, edgecolor=c["bg"], lw=1.0)
        ax.scatter(x[~improved], s[~improved], s=34, facecolor=c["bg"], edgecolor=c["other"], lw=1.3, zorder=4)
        ax.scatter([x[-1]], [s[-1]], s=150, facecolor="none", edgecolor=c["final"], lw=1.4, zorder=4)
        # annotations: score + change, leader line ending at the marker
        for xi, n, si, imp in zip(x, names, s, improved):
            if n not in LABELS:
                continue
            txt, (dx, dy), ha = LABELS[n]
            col = c["fg"] if imp else c["bad"]
            head = f"{n}  {si:.2f}"
            full = f"{head}  {txt}" if txt.startswith("·") else f"{head}\n{txt}"
            ax.annotate(full, xy=(xi, si), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        va="bottom" if dy > 0 else "top", fontsize=8.2, color=col, linespacing=1.2, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=c["faint"], lw=0.7, shrinkA=1, shrinkB=4.5,
                                        connectionstyle="arc3,rad=0"))
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(LO, HI)
        ax.set_yticks([77, 78, 79, 80, 81, 81.2, 81.4, 81.6, 81.8, 82.0])
        ax.set_yticklabels(["77", "78", "79", "80", "81", "81.2", "81.4", "81.6", "81.8", "82.0"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{d}" for n, d in zip(names, days)], fontsize=7.6)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel("hidden-set final score")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        fig.tight_layout()
        out = FIG / ("progress.svg" if theme == "light" else "progress_dark.svg")
        fig.savefig(out, facecolor=c["bg"])
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
