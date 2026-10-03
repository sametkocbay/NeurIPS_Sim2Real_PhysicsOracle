#!/usr/bin/env python
"""Generate the improvement figure from the recorded development submissions.

    python tools/make_improvement_figure.py --output-dir final/figures --font-scale 1.15 --theme light
    python tools/make_improvement_figure.py --output-dir /tmp/improvement-review --preview /tmp/improvement.png

Needs matplotlib (not a training dependency); no model, GPU or flow-field data required.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tools" / "figure_data"
FIGURES = ROOT / "final" / "figures"

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
    "S3": ("axial transformer", (-10, 42), "right"),
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


def improvement(output_dir=None, font_scale=1.0, theme="both", preview=None):
    rows = list(csv.DictReader(open(DATA / "submissions.csv")))
    names = [r["name"] for r in rows]
    s = np.array([float(r["final_score"]) for r in rows])
    days = [dt.datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M").strftime("%d %b") for r in rows]
    best = np.maximum.accumulate(s)
    improved = np.r_[True, s[1:] > best[:-1]]
    x = np.arange(len(rows))
    for theme_name, c in THEMES.items():
        if theme != "both" and theme_name != theme:
            continue
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5 * font_scale, "text.color": c["fg"],
                             "axes.labelcolor": c["fg"], "xtick.color": c["muted"], "ytick.color": c["muted"],
                             "axes.edgecolor": c["grid"], "svg.fonttype": "path", "svg.hashsalt": "physicsoracle-progress"})
        fig, ax = plt.subplots(figsize=(12, 5.6))
        fig.patch.set_facecolor(c["bg"])
        ax.set_facecolor(c["bg"])
        ax.set_yscale("function", functions=(_fwd, _inv))
        # horizontal grid and the scale break
        for yt in (77, 78, 79, 80, 81.2, 81.4, 81.6, 81.8, 82.0):
            ax.axhline(yt, color=c["grid"], lw=0.6, zorder=1)
        ax.axhline(BREAK, color=c["faint"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(len(rows) - 0.45, BREAK, "scale changes", ha="right", va="bottom", fontsize=7.5 * font_scale, color=c["faint"])
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
                        va="bottom" if dy > 0 else "top", fontsize=8.2 * font_scale, color=col, linespacing=1.2, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=c["faint"], lw=0.7, shrinkA=1, shrinkB=4.5,
                                        connectionstyle="arc3,rad=0"))
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(LO, HI)
        ax.set_yticks([77, 78, 79, 80, 81, 81.2, 81.4, 81.6, 81.8, 82.0])
        ax.set_yticklabels(["77", "78", "79", "80", "81", "81.2", "81.4", "81.6", "81.8", "82.0"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{d}" for n, d in zip(names, days)], fontsize=7.6 * font_scale)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", length=0)
        ax.set_ylabel("hidden-set final score")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        annotations = [item for item in ax.texts if isinstance(item, Annotation)]
        boxes = [Text.get_window_extent(item, renderer) for item in annotations]
        for i, bounds in enumerate(boxes):
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise RuntimeError(f"Annotation is clipped: {annotations[i].get_text()!r}")
            for j in range(i):
                other = boxes[j]
                if (min(bounds.x1, other.x1) > max(bounds.x0, other.x0) + 1 and
                        min(bounds.y1, other.y1) > max(bounds.y0, other.y0) + 1):
                    raise RuntimeError(f"Annotations overlap: {annotations[i].get_text()!r} / {annotations[j].get_text()!r}")
        destination = Path(output_dir) if output_dir else FIGURES
        destination.mkdir(parents=True, exist_ok=True)
        out = destination / ("progress.svg" if theme_name == "light" else "progress_dark.svg")
        fig.savefig(out, facecolor=c["bg"], metadata={"Date": None, "Title": "PhysicsOracle development progress"})
        out.write_text("\n".join(line.rstrip() for line in out.read_text().splitlines()) + "\n")
        if preview:
            preview_path = Path(preview)
            if theme == "both":
                preview_path = preview_path.with_name(f"{preview_path.stem}_{theme_name}{preview_path.suffix}")
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview_path, dpi=160, facecolor=c["bg"])
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, help="Directory for improvement SVGs (default: final/figures).")
    ap.add_argument("--font-scale", type=float, default=1.0, help="Figure font-size multiplier.")
    ap.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    ap.add_argument("--preview", type=Path, help="PNG preview of the improvement figure.")
    a = ap.parse_args()
    if a.font_scale <= 0:
        ap.error("--font-scale must be positive")
    improvement(a.output_dir, a.font_scale, a.theme, a.preview)
