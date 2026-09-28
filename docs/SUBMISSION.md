# Building the submission

The zip that ran on the evaluation server contains the two members, the whole predict path compiled ahead of time
(AOTInductor) for the batch sizes the server uses, and a small runtime. Build it from the epoch-59 snapshots of your
own training, or directly from the included `weights/`.

## Requirements

- the environment of the evaluation container: Python 3.10, `torch==2.2.2` (CUDA 12.1), `numpy==1.26.4`;
- an NVIDIA GPU of the same architecture as the evaluation server (compute capability 8.0, e.g. A100; the server used
  an A800) -- the compiled kernels are architecture specific;
- the CUDA 12.1 toolkit (`CUDA_HOME` set, `nvcc` on the path) and a host C++ compiler (gcc 11 works): AOTInductor
  compiles C++/CUDA at build time. The runtime needs neither.

## Steps

```bash
# 1. real example windows for tracing, autotuning and the correctness check (needs data/cache32, see TRAINING.md)
python tools/make_sample_inputs.py                        # -> build/sample_inputs.npy  (15 windows)

# 2. compile the ensemble for batch sizes 1, 2, 3, 4, 5 and 10 (about 5 min per group).
#    Run the groups one after another: parallel builds into one directory collide on temporary files.
M=weights/fm_all_s3_ep59.pth,weights/fm_all_s4_ep59.pth    # or runs/fm_all_s3/ep59.pth,runs/fm_all_s4/ep59.pth
for b in 5,10 2,4 3,1; do
  python tools/aot_build_ens.py build/aot $M gamma=0.70 band_scale=0.95 energy_preserve=1 batches=$b
done

# 3. package + clean-room test
python tools/pack_submission.py --aot build/aot --members weights/fm_all_s3_ep59.pth weights/fm_all_s4_ep59.pth \
    --out dist/submission.zip
```

Each build prints, per batch size, the maximum difference between the compiled graph and the eager PyTorch model
(`max_dpred`, about 5e-5 to 1e-4 m/s: the fp16 fast path) and a sanity flag; `build/aot/build_report_*.json` keeps
it. `pack_submission.py` checks that all build reports agree on the decode settings, writes them to `ens_cfg.json`,
strips the members, zips everything with `submission.py` at the archive root, checks the 256 MB limit and calls
`predict()` twice in a clean subprocess (no repository on the path, scipy/h5py/yaml/... blocked, as on the server).
The member order passed to `pack_submission.py` must equal the order given to `aot_build_ens.py`.

The final submission was ~108 MB (124 MB extracted): 2 x 13.8 MB of weights, 6 compiled graphs of ~14.8 MB and
their CUDA kernels.

## Contents of the zip

```
submission.py            runtime: predict(input_array, metadata) -> {"prediction", "lower", "upper"}
ens_cfg.json             {"gamma": 0.7, "band_scale": 0.95, "energy_preserve": true, "gated": {}, "mirror_views": false}
model_0.pth, model_1.pth members (EMA weights + architecture description)
aot/predict_b{1,2,3,4,5,10}.so, aot/kernels/*.cubin
rpde/                    the modules the eager fallback needs
```

## The runtime (`submission/submission.py`)

The server times every `predict()` call, so the runtime does nothing that the prediction does not need:

- **at import** (not timed): load the compiled graphs through their C API with `ctypes` (no compiler at run time; if
  the package directory is not executable the graphs are loaded from a copy in `/dev/shm` or the temp directory), run
  a warm-up of every batch size, and allocate pinned host buffers for the outputs;
- **per call**: copy the input through a pinned buffer to the GPU, run the graph for that batch size (other sizes are
  split into the compiled sizes), copy the three outputs into fresh pinned buffers and return them as float32 arrays;
  no logging, no extra copies, no dtype conversions;
- **fallback**: if the compiled graphs cannot be loaded or fail, the same ensemble runs eagerly in PyTorch
  (`rpde.ensemble_predict.EnsPredict`, members on their fp16 fast path), configured from `ens_cfg.json`.

Measured in the evaluation image: 1.09 ms per sample inside `predict()` for the server's calls (batch 5), and the
compiled and eager paths agree to < 1e-4 m/s on arbitrary batch sizes, dtypes and memory layouts.
