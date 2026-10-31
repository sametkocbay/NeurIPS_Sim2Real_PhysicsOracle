#!/usr/bin/env python
"""Build the 32x64 training cache from the raw competition data.

Input  (the public training data, see README.md):
    <raw>/train_real/<Re>_<AoA>.h5   PIV, keys u, v, t, x, y, re, aoa           (64x128 per frame)
    <raw>/train_sim/<Re>_<AoA>.h5    CFD, keys u, v, p, t, x, y, re, aoa
Output (what rpde.data reads):
    <cache>/real/<Re>_<AoA>.npz, <cache>/sim/<Re>_<AoA>.npz with
      f      (T, 32, 64, C) float32, C = 2 (u, v) for real, 3 (u, v, p) for sim  -- every 2nd row/column
      t      (T,) float32
      mask   (32, 64) bool, True = measured (u not identically zero over the trajectory)
      re_nom, aoa (from the file name), re_attr (the file's `re` attribute)

    python rpde/tools/make_cache.py --raw <raw> [--cache data/cache32] [--workers 16]

The subsampling [::2, ::2] is exactly the evaluation grid (the hidden inputs are 20 x 32 x 64).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from pathlib import Path

import h5py
import numpy as np

SUB = 2


def parse_name(path) -> tuple[int, int]:
    m = re.match(r"(\d+)_(\d+)", Path(path).stem)
    return int(m.group(1)), int(m.group(2))


def process(job):
    raw, cache, domain, name = job
    with h5py.File(Path(raw) / f"train_{domain}" / f"{name}.h5", "r") as h:
        u = np.asarray(h["u"])
        v = np.asarray(h["v"])
        pr = np.asarray(h["p"]) if "p" in h else None
        t = np.asarray(h["t"], dtype=np.float64)
        re_attr = int(h["re"][()])
    re_nom, aoa = parse_name(name)
    us = np.ascontiguousarray(u[:, ::SUB, ::SUB]).astype(np.float32)
    vs = np.ascontiguousarray(v[:, ::SUB, ::SUB]).astype(np.float32)
    if pr is not None:
        f = np.stack([us, vs, np.ascontiguousarray(pr[:, ::SUB, ::SUB]).astype(np.float32)], axis=-1)
    else:
        f = np.stack([us, vs], axis=-1)
    mask = ~np.all(f[..., 0] == 0.0, axis=0)          # u only: v crosses zero physically
    np.savez(Path(cache) / domain / f"{name}.npz", f=f, t=t.astype(np.float32),
             mask=mask, re_nom=re_nom, aoa=aoa, re_attr=re_attr)
    return f"{domain}/{name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True, help="directory holding train_real/ and train_sim/")
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[2] / "data" / "cache32"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = []
    for domain in ("real", "sim"):
        (Path(a.cache) / domain).mkdir(parents=True, exist_ok=True)
        names = sorted((p.stem for p in (Path(a.raw) / f"train_{domain}").glob("*.h5")), key=parse_name)
        jobs += [(a.raw, a.cache, domain, n) for n in names]   # real 7575_0 is kept here, dropped by rpde.data
    with mp.Pool(a.workers) as pool:
        done = pool.map(process, jobs, chunksize=1)
    print(f"wrote {len(done)} files to {a.cache}")


if __name__ == "__main__":
    main()
