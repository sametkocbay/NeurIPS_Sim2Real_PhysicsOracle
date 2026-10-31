#!/usr/bin/env python
"""Real PIV windows used to trace, autotune and check the AOT build (rpde/tools/aot_build_ens.py).

The first five real regimes (sorted by Re, AoA), the first three stride-40 windows of each: 15 windows of shape
(20, 32, 64, 3) float32 with p = 0, exactly what the evaluator feeds predict(). Real-scale inputs matter: random
N(0,1) inputs overflow the fp16 fast path and make the build's eager-vs-AOT check fail spuriously.

    python rpde/tools/make_sample_inputs.py [--cache data/cache32] [--out build/sample_inputs.npy]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

T_IN, T_OUT, STRIDE = 20, 20, 40
BAD_REAL = {"7575_0"}          # organiser announcement: duplicate of 6300_0


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(root / "data" / "cache32"))
    ap.add_argument("--out", default=str(root / "build" / "sample_inputs.npy"))
    a = ap.parse_args()
    key = lambda n: tuple(int(v) for v in re.match(r"(\d+)_(\d+)", n).groups())  # noqa: E731
    names = sorted((p.stem for p in (Path(a.cache) / "real").glob("*.npz") if p.stem not in BAD_REAL), key=key)[:5]
    xs = []
    for n in names:
        f = np.load(Path(a.cache) / "real" / f"{n}.npz")["f"][..., :2]
        starts = np.arange(0, f.shape[0] - (T_IN + T_OUT) + 1, STRIDE)[:3]
        x = np.zeros((len(starts), T_IN, 32, 64, 3), dtype=np.float32)
        x[..., :2] = np.stack([f[s:s + T_IN] for s in starts])
        xs.append(x)
    x = np.concatenate(xs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, x)
    print(f"{x.shape} from {names} -> {a.out}")


if __name__ == "__main__":
    main()
