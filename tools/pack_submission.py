#!/usr/bin/env python
"""Assemble the Codabench submission zip from the AOT build and the member checkpoints, then test it in a clean room.

    python tools/pack_submission.py --aot build/aot \\
        --members weights/fm_all_s3_ep59.pth weights/fm_all_s4_ep59.pth --out dist/submission.zip

Zip layout (everything at the archive root):
    submission.py           the lean runtime (submission/submission.py)
    ens_cfg.json            gamma / band_scale / energy_preserve / gated / mirror_views the aot/ graphs were BUILT with
                            (read from the build reports; only the eager fallback reads it)
    model_0.pth, model_1.pth  members in the SAME order as the AOT build (stripped: EMA weights + architecture)
    aot/predict_b{B}.so, aot/kernels/*.cubin
    rpde/...                the modules the eager fallback imports
The member order must equal the order given to tools/aot_build_ens.py.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.strip_ckpt import strip  # noqa: E402

CAP_BYTES = 256 * 1024 * 1024
RUNTIME_RPDE = ["__init__.py", "constants.py", "scaffold.py", "decode.py", "gated_decode.py", "ensemble_predict.py",
                "fastinf.py", "cores/__init__.py", "cores/axial.py"]

CLEAN_ROOM = r'''
import sys, importlib.abc, numpy as np
BLOCK = {"scipy", "h5py", "einops", "matplotlib", "pandas", "sklearn", "cv2", "yaml"}
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ModuleNotFoundError(f"blocked in clean room: {name}")
        return None
sys.meta_path.insert(0, _Block())
sys.path.insert(0, sys.argv[1])
import submission
for call in range(2):
    x = (np.random.rand(3, 20, 32, 64, 3).astype(np.float32) * 0.3)
    x[..., 2] = 0.0
    x[:, :, 22:, :9, :] = 0.0
    out = submission.predict(x, {})
    p, lo, hi = (np.asarray(out[k]) for k in ("prediction", "lower", "upper"))
    assert p.shape == (3, 20, 32, 64, 3) and lo.shape == p.shape and hi.shape == p.shape
    assert p.dtype == np.float32
    assert np.isfinite(p).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
    assert (p[..., 2] == 0).all(), "p channel must be zero"
    assert (lo <= hi).all(), "lower > upper somewhere"
print("clean-room OK: 2 calls, shapes", p.shape, "dtype", p.dtype, "compiled batch sizes", sorted(submission._G))
'''


def ens_cfg_from_reports(aot_dir: Path) -> dict:
    reps = [json.load(open(p)) for p in sorted(glob.glob(str(aot_dir / "build_report_*.json")))]
    assert reps, f"no build_report_*.json in {aot_dir}"
    cfg = {(r["gamma"], r.get("band_scale", 1.0), bool(r.get("energy_preserve", False)),
            json.dumps(r.get("gated", {}), sort_keys=True), bool(r.get("mirror_views", False))) for r in reps}
    assert len(cfg) == 1, f"inconsistent AOT build reports: {cfg}"
    g, s, e, gd, mv = cfg.pop()
    return {"gamma": g, "band_scale": s, "energy_preserve": e, "gated": json.loads(gd), "mirror_views": mv}


def assemble(aot: Path, members, stage: Path, ens_cfg: dict | None = None) -> dict:
    """Write the package layout (submission.py, ens_cfg.json, model_<i>.pth, rpde/, aot/) into `stage`.

    With `ens_cfg` given, a package without compiled graphs is allowed (the runtime then runs the eager fallback,
    configured from that ens_cfg); otherwise the build reports define the decode settings."""
    aot, stage = Path(aot), Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "submission" / "submission.py", stage / "submission.py")
    has_aot = bool(sorted(aot.glob("build_report_*.json")))
    cfg = ens_cfg_from_reports(aot) if has_aot or ens_cfg is None else dict(ens_cfg)
    if has_aot and ens_cfg is not None:
        assert cfg == ens_cfg, f"compiled graphs were built with {cfg}, expected {ens_cfg}"
    (stage / "ens_cfg.json").write_text(json.dumps(cfg))
    print("ens_cfg.json:", cfg)
    for i, ck in enumerate(members):
        strip(ck, stage / f"model_{i}.pth")
        print(f"model_{i}.pth <- {ck}")
    for f in RUNTIME_RPDE:
        (stage / "rpde" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "rpde" / f, stage / "rpde" / f)
    (stage / "aot" / "kernels").mkdir(parents=True, exist_ok=True)
    sos = sorted(aot.glob("predict_b*.so"))
    assert sos or ens_cfg is not None, f"no predict_b*.so in {aot}"
    if not sos:
        print(f"WARNING: no compiled graphs in {aot}; the package runs the eager fallback")
    for so in sos:
        shutil.copy(so, stage / "aot" / so.name)
    for cb in (aot / "kernels").glob("*.cubin"):
        shutil.copy(cb, stage / "aot" / "kernels" / cb.name)
    return cfg


def clean_room(pkg: Path) -> str:
    """Import `pkg`/submission.py in a fresh interpreter (no repository on the path) and call predict() twice."""
    r = subprocess.run([sys.executable, "-c", CLEAN_ROOM, str(pkg)], env=dict(os.environ, PYTHONPATH=""),
                       cwd=str(pkg), capture_output=True, text=True, timeout=900)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("clean-room test FAILED")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aot", required=True, help="directory written by tools/aot_build_ens.py")
    ap.add_argument("--members", nargs="+", required=True, help="checkpoints, in AOT build order")
    ap.add_argument("--out", default=str(ROOT / "dist" / "submission.zip"))
    a = ap.parse_args()
    aot, out = Path(a.aot), Path(a.out)
    stage = Path(tempfile.mkdtemp(prefix="rpde_pack_"))
    try:
        assemble(aot, a.members, stage)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(stage.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    z.write(f, str(f.relative_to(stage)))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    sos = sorted(aot.glob("predict_b*.so"))
    extracted = sum(i.file_size for i in zipfile.ZipFile(out).infolist())
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    print(f"zip: {out}  ({out.stat().st_size / 1e6:.1f} MB, {extracted / 1e6:.1f} MB extracted)  sha256[:12]={sha}  "
          f"compiled batch sizes {[so.name for so in sos]}")
    assert extracted < CAP_BYTES, "over the 256 MB extracted cap"

    tmp = Path(tempfile.mkdtemp(prefix="rpde_cleanroom_"))
    try:
        with zipfile.ZipFile(out) as z:
            z.extractall(tmp)
        clean_room(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
