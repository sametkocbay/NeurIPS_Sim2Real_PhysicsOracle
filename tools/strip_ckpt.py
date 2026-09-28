#!/usr/bin/env python
"""Strip a training checkpoint down to what a submission needs.

    python tools/strip_ckpt.py runs/fm_all_s3/ep59.pth weights/fm_all_s3_ep59.pth

Keeps the EMA weights (the ones every submission loads) plus the architecture
description read back by `rpde.ensemble_predict.load_member`:
`core_name`, `core_kwargs`, `model_cfg` (+ `epoch`, `best`, `train_meta` for
provenance).  Drops `model_state` (the EMA copy is what ships), `opt_state`,
`sched_state` and `data_cfg`: ~57 MB -> ~7 MB per member.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

KEEP = ("core_name", "core_kwargs", "model_cfg", "epoch", "best", "train_meta")


def strip(src, dst, keep_model_state: bool = False) -> dict:
    ck = torch.load(str(src), map_location="cpu")
    state = ck.get("ema_state") or ck["model_state"]
    out = {"ema_state": {k: v.detach().cpu() if torch.is_tensor(v) else v
                         for k, v in state.items()}}
    for k in KEEP:
        if k in ck:
            out[k] = ck[k]
    if keep_model_state and "model_state" in ck:
        out["model_state"] = ck["model_state"]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(dst))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser("strip_ckpt")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-model-state", action="store_true")
    a = ap.parse_args(argv)
    strip(a.src, a.dst, a.keep_model_state)
    s0 = Path(a.src).stat().st_size / 1e6
    s1 = Path(a.dst).stat().st_size / 1e6
    print(f"{a.src} ({s0:.2f} MB) -> {a.dst} ({s1:.2f} MB)")


if __name__ == "__main__":
    main()
