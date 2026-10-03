#!/usr/bin/env python
"""Stage the final-phase training package (Docker build context and zip) from the repository.

    python tools/make_final_package.py [--out dist/final]

Writes <out>/package/ (Dockerfile, solution.py, README.md, METHOD.md, the code train() runs and the logs of the
training runs behind the Codabench submission) and
<out>/physicsoracle_track1.zip with the same files. Build the image from <out>/package/ so that the image and the zip
hold the same code.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "Dockerfile": "final/Dockerfile",
    "README.md": "final/README.md",
    "METHOD.md": "final/METHOD.md",
    "figures/architecture.svg": "final/figures/architecture.svg",
    "figures/progress.svg": "final/figures/progress.svg",
    "solution.py": "solution.py",
    "configs/final.yaml": "configs/final.yaml",
    "submission/submission.py": "submission/submission.py",
    "third_party/realpde_kit/scoring.py": "third_party/realpde_kit/scoring.py",
    "third_party/realpde_kit/README.md": "third_party/realpde_kit/README.md",
    **{f"tools/{f}": f"tools/{f}" for f in ("__init__.py", "make_cache.py", "make_sample_inputs.py",
                                             "aot_build_ens.py", "pack_submission.py", "strip_ckpt.py", "fit_gate.py",
                                             "make_final_package.py")},
    # logs of the training runs behind Codabench submission 948641 (exported from Weights & Biases)
    **{f"training_logs_948641/{r}/{f}": f"logs/wandb/{r}/{f}" for r in ("fm_all_s3", "fm_all_s4")
       for f in ("output_log.txt", "history.txt", "config.yaml")},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "dist" / "final"))
    a = ap.parse_args()
    out = Path(a.out)
    pkg = out / "package"
    shutil.rmtree(pkg, ignore_errors=True)
    files = dict(FILES)
    files.update({str(p.relative_to(ROOT)): str(p.relative_to(ROOT)) for p in sorted((ROOT / "rpde").rglob("*.py"))})
    for dst, src in files.items():
        if not (ROOT / src).is_file():
            print(f"WARNING: missing {src}")
            continue
        (pkg / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / src, pkg / dst)
    z = out / "physicsoracle_track1.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                zf.write(f, str(Path("physicsoracle_track1") / f.relative_to(pkg)))
    n = sum(1 for f in pkg.rglob("*") if f.is_file())
    print(f"{pkg}: {n} files;  {z}: {z.stat().st_size / 1e6:.2f} MB  sha256[:12]="
          f"{hashlib.sha256(z.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
