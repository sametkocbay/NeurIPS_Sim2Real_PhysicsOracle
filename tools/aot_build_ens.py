#!/usr/bin/env python
"""AOT-compile the K-member ensemble's WHOLE predict path (rpde.ensemble_predict.EnsPredict: members + averaging +
energy-preserving combine + S12 gated decode + band decode + post) into one shared object per batch size (math SDPA,
max-autotune, each member's fp16 fast path applied), then reload it through the compiler-free runner and compare
against the eager EnsPredict: max |dpred|, |dband|, timing.  Needs a CUDA GPU, the CUDA toolkit (CUDA_HOME) and a host
C++ compiler; run the batch groups one after another (parallel builds into one directory race on temp files).

    python tools/aot_build_ens.py build/aot weights/fm_all_s3_ep59.pth,weights/fm_all_s4_ep59.pth \
        gamma=0.70 band_scale=0.95 energy_preserve=1 gated=build/gate.json batches=5,10 \
        [inputs=build/sample_inputs.npy] [autotune=1]
Writes <out_dir>/predict_b{B}.so, <out_dir>/kernels/*.cubin and <out_dir>/build_report_<batches>.json.
"""
import glob, json, os, re, shutil, sys, time
from contextlib import ExitStack

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = os.path.abspath(sys.argv[1])
ckpts = sys.argv[2].split(",")
opts = dict(a.split("=", 1) for a in sys.argv[3:])
gamma = float(opts.get("gamma", "0.70"))
band_scale = float(opts.get("band_scale", "1.0"))
energy_preserve = opts.get("energy_preserve", "0") == "1"
gated = opts.get("gated", "{}")                           # JSON, or a JSON file (tools/fit_gate.py --out): gate parameters
gated = json.load(open(gated)) if os.path.isfile(gated) else json.loads(gated)
mirror_views = opts.get("mirror_views", "0") == "1"
batches = [int(b) for b in opts.get("batches", "5").split(",")]
autotune = opts.get("autotune", "1") == "1"
os.makedirs(f"{out_dir}/kernels", exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{out_dir}/_inductor_cache_{'_'.join(map(str, batches))}")

from rpde.aot_runner import AOTModel  # noqa: E402
from rpde.ensemble_predict import EnsPredict, load_member  # noqa: E402
from rpde.fastinf import RECOMMENDED, FastInference  # noqa: E402

dev = torch.device("cuda")
members = [load_member(c, gamma, dev) for c in ckpts]
fis = [FastInference(m, **{**RECOMMENDED, "cuda_graph": False}) for m in members]
# real-scale example windows (tools/make_sample_inputs.py), at least max(batches) of them
xall = torch.from_numpy(np.load(opts.get("inputs", "build/sample_inputs.npy"))).to(dev)
print(f"[ens-aot] K={len(members)} gamma={gamma} band_scale={band_scale} energy_preserve={energy_preserve} gated={gated} mirror_views={mirror_views} batches={batches} autotune={autotune} inputs={tuple(xall.shape)}",
      flush=True)


def fast_ctx():
    st = ExitStack()
    st.enter_context(torch.no_grad())
    st.enter_context(torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False))
    for fi in fis:
        st.enter_context(fi._apply())
    return st


def bench(fn, B, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts)) / B


rep_path = f"{out_dir}/build_report_{'_'.join(map(str, batches))}.json"
report = dict(ckpts=ckpts, gamma=gamma, band_scale=band_scale, energy_preserve=energy_preserve, gated=gated, mirror_views=mirror_views, autotune=autotune, gpu=torch.cuda.get_device_name(), runs={})
for B in batches:
    x = xall[:B].contiguous()
    so = f"{out_dir}/predict_b{B}.so"
    t0 = time.perf_counter()
    with fast_ctx():
        pm = EnsPredict(members, checks=True, band_scale=band_scale, energy_preserve=energy_preserve, gated_params=gated, mirror_views=mirror_views).eval()
        ref = pm(x)                                  # shared-input check + fills cached buffers
        pm.checks = False
        path = torch._export.aot_compile(pm, (x,), options={"aot_inductor.output_path": so, "max_autotune": autotune})
        ref = pm(x)
    build_s = time.perf_counter() - t0
    blob = open(path, "rb").read()
    names = set(m.decode() for m in re.findall(rb"[\w/.\-]+\.cubin", blob))
    for n in names:
        if os.path.isfile(n):
            shutil.copy(n, f"{out_dir}/kernels")
    for n in glob.glob(f"{out_dir}/*.cubin"):          # Inductor also writes cubins the .so does not name by path
        shutil.copy(n, f"{out_dir}/kernels")
    runner = AOTModel(so, f"{out_dir}/kernels", [torch.float32] * 3, [5] * 3)
    got = runner(x)
    dp = float((got[0] - ref[0]).abs().max())
    db = float(torch.maximum((got[1] - ref[1]).abs().max(), (got[2] - ref[2]).abs().max()))
    ok = bool(torch.isfinite(got[0]).all() and (got[1] <= got[2]).all() and (got[0][..., 2] == 0).all())
    report["runs"][B] = dict(build_s=round(build_s, 1), n_cubin=len(names), max_dpred=dp, max_dband=db, sane=ok,
                             aot_ms_per_sample=bench(lambda: runner(x), B))
    print(f"[ens-aot] B={B}: {report['runs'][B]}", flush=True)
    json.dump(report, open(rep_path, "w"), indent=1)
print(f"[ens-aot] done -> {rep_path}", flush=True)
