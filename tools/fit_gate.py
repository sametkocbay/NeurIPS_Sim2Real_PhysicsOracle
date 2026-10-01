#!/usr/bin/env python
"""Fit the data tables of the gated decode (rpde/gated_decode.py) from the training cache.

    python tools/fit_gate.py [--out gate.json] [--check]

--out writes everything the gated decode needs: the fitted tables plus the decode coefficients of
configs/final.yaml (submission.gate).

The gate decides whether an input lies above the training Reynolds range for its angle of attack. It needs, from
train_real (7575_0 excluded):
  tops       the largest Reynolds number recorded at each angle of attack
  centroids  the mean mask-geometry features cond[3:7] (mask centroid x, y and extent x, y) of each angle's windows
  sd         the standard deviation of those features over all windows
The windows are the validation tails of every real recording (the last three stride-40 windows, rpde.data.val_starts)
and, for the recordings of the highest Reynolds row, the stride-40 grid over the whole recording (rpde.data.eval_starts):
during development that row was the held-out row of the fold whose predictions the tables were first computed on.
The decode coefficients (c1, lam, lam_far, c1_out, decay_lo, decay_hi, re_step) are fixed hyperparameters and are not
fitted here. --check compares the fitted tables with REFERENCE, the tables of Codabench submission 948641.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rpde import data as D  # noqa: E402
from rpde.constants import AOA_GRID  # noqa: E402
from rpde.gated_decode import GATE_COEFFS, GATE_TABLES  # noqa: E402
from rpde.scaffold import Preproc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the tables of Codabench submission 948641 (printed to 8 digits), for --check
REFERENCE = dict(
    aoa_keys=[0, 5, 10, 15, 20],
    tops=[26700.0, 21600.0, 26700.0, 26700.0, 25425.0],
    centroids=[[0.10816839, 0.74321777, 0.31361237, 0.51906127], [0.1081567, 0.72691208, 0.32690847, 0.53917056],
               [0.09397534, 0.70970106, 0.31216949, 0.55286729], [0.09518676, 0.71980542, 0.32413173, 0.5811128],
               [0.0908691, 0.68951124, 0.31613749, 0.63575274]],
    sd=[0.00957239, 0.02040497, 0.01088485, 0.03987769],
)


CONFIG = os.path.join(ROOT, "configs", "final.yaml")


def gate_params(tables: dict, config: str = CONFIG) -> dict:
    """The gated decode's full parameter set: the fitted tables and the coefficients of the config."""
    coeffs = yaml.safe_load(open(config))["submission"]["gate"]
    return {**{k: tables[k] for k in GATE_TABLES}, **{k: coeffs[k] for k in GATE_COEFFS}}


def fit(config: str = CONFIG) -> dict:
    cfg = yaml.safe_load(open(config))
    dcfg = D.DataCfg.from_dict(cfg["data"])
    real = D.split("full")["train_real"]
    corpus, real = D.load_all(real, None, D.Corpus())
    top_re = max(r.re for r in real)
    sets = [D.FixedWindows([r for r in real if r.re != top_re], corpus, dcfg, mode="val", seed=0, mask_pool=real),
            D.FixedWindows([r for r in real if r.re == top_re], corpus, dcfg, mode="eval", seed=0, mask_pool=real)]
    pre = Preproc().eval()
    feats, aoas = [], []
    with torch.no_grad():
        for ds in sets:
            for i in range(len(ds)):
                reg, _ = ds.items[i]
                x = torch.as_tensor(ds[i]["x"])[None].float()
                feats.append(pre(x)["cond"][0, 3:7].numpy())
                aoas.append(int(reg.aoa))
    feats, aoas = np.stack(feats), np.asarray(aoas)
    keys = sorted(set(AOA_GRID))
    return dict(
        aoa_keys=keys,
        tops=[float(max(r.re for r in real if int(r.aoa) == a)) for a in keys],
        centroids=[feats[aoas == a].mean(0).tolist() for a in keys],
        sd=(feats.std(0) + 1e-9).tolist(),
        n_windows=int(len(feats)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    g = fit()
    print(json.dumps(g))
    if a.out:
        json.dump(gate_params(g), open(a.out, "w"))
    if a.check:
        for k in GATE_TABLES:
            ref, got = np.asarray(REFERENCE[k], dtype=float), np.asarray(g[k], dtype=float)
            print(f"{k:10s} max |fitted - reference| = {np.abs(ref - got).max():.3e}")


if __name__ == "__main__":
    main()
