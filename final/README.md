# PhysicsOracle: Track 1 training package

Physics-scaffolded axial transformer, two-member ensemble, compiled ahead of time. The method is described in
`METHOD.md`.

## Codabench
- Username: physicsoracle
- Submission id this package reproduces: 948641
- final_score of that submission: 82.010 (rel_l2 94.892, tke 79.708, mvpe 94.216, time 91.862, sps 44.377)

## How to reproduce
- Build command: the default, `docker build -t <name> .` (linux/amd64).
- Train command: the default,
  `python -c "import solution; solution.train('/data', '/output', 'cuda')"`, with `/data` read-only.
- Evaluation: the default. `solution.load_predictor('/output', 'cuda')` returns an object whose `predict(x)` takes a
  float32 array `(N, 20, 32, 64, 3)` for any N and returns a dict with `prediction`, `lower` and `upper`, each of the
  same shape and float32.

`train()` runs four stages:

| stage | what | time on one RTX 4090 |
|---|---|---|
| 1. cache | `tools/make_cache.py`: every second row and column of `train_real/*.h5` and `train_sim/*.h5` (the evaluation grid), into `/output/work/cache32` (3.4 GB). Then `tools/fit_gate.py` fits the data tables of the gated decode on `train_real` (`/output/logs/gate.json`) | 12 s |
| 2. members | `python -m rpde.train --config configs/final.yaml --fold full --seed {3,4}`, one after the other; each stops after epoch 59 of a 100-epoch cosine schedule and keeps that epoch's EMA weights | 93 min per member |
| 3. compile | `tools/aot_build_ens.py`: the two members, ensemble combine and band decode, compiled with AOTInductor for batch sizes 1, 2, 3, 4, 5 and 10 (three groups). The graphs are compiled for the GPU that runs `train()` | about 15 min |
| 4. package | `tools/pack_submission.py`: writes the Codabench package layout into `/output` (`submission.py`, `ens_cfg.json`, `model_0.pth`, `model_1.pth`, `aot/`, `rpde/`) and calls `predict()` in a clean interpreter | 10 s |

`load_predictor` imports `/output/submission.py`, the same runtime that ran on Codabench. At import it loads the
compiled graphs and warms them up; per call it runs the graph for that batch size (other sizes are split into
compiled sizes). If the graphs cannot be loaded it runs the same ensemble eagerly in PyTorch, with the same
outputs to within 1e-4 m/s.

`/output/logs/` keeps the training logs of both members (`fm_all_s3/log.txt`, `fm_all_s4/log.txt`), the compile
reports, `stages.log` (all stage output) and `summary.json` (stage times, peak GPU memory). `/output/work/` (cache and
full checkpoints) is deleted at the end; set `RPDE_KEEP_WORK=1` to keep it.

## Hardware your submitted model was trained on
- GPU model and count: one NVIDIA GeForce RTX 4090 per member, on two rented 4 x RTX 4090 servers (seed 3: 24 GB
  cards; seed 4: 48 GB cards), started 30 min apart.
- CPU: 64 cores / 128 threads per server (8 data-loader workers per member).
- Memory: 504 GiB per server; one member needs < 16 GB (the cache is held in memory, 3.6 GB).
- Training time on this hardware: 82 min per member to epoch 59 (the original runs continued to epoch 100: 157 and
  143 min), including a validation pass after every epoch. `train()` trains the members one after the other and stops
  at epoch 59; on one RTX 4090 the whole of `train()` takes about 3.4 h (2 x 93 min of training, about 15 min of
  compilation, 12 s for the cache and the gate tables).
- Peak GPU memory on this hardware: 3.17 GiB allocated by PyTorch per member (3.31 GiB reserved), measured in the
  reproduction run below.

Reproduction run of this package, on one RTX 4090 (24 GB) with `--network none` and 16 CPUs, `docker run` as in the
organisers' instructions: the seed-3 member reproduces the original run bit for bit. All 59 epoch lines of its log
(losses and validation scores) equal those of the original run, and its epoch-59 EMA weights equal the submitted
weights exactly (all 121 tensors). The seed-4 member was stopped after 44 epochs, all 44 identical to the original
log. The logs are in `training_logs_reproduction/`. On other GPU models the weights differ at the level of numerical
noise (cuDNN autotuning, TF32 and bf16).

## Environment
- Base image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel` (Python 3.10, torch 2.2.2, CUDA 12.1, gcc 11.4). The
  devel variant is needed because the compile stage calls nvcc and a host C++ compiler. CUDA 12.1 is below the 12.4
  limit.
- Added packages and versions (pinned the same way in the Dockerfile): `h5py==3.11.0`, `pyyaml==6.0.1` (already in the
  base image), `numpy==1.26.4` (already in the base image). `CUDA_HOME=/usr/local/cuda`.

## Data, weights and code
- Training data used: only the released training data in `/data`: all 81 files of `train_real/` and all 100 files of
  `train_sim/` (`u`, `v`; the simulated pressure is not used). `mean_std_real.pt` is not used (the model normalises by
  a freestream velocity estimated from each input window). `7575_0` is not mounted and is also excluded in code.
- Augmentation: simulated windows are strided to the real convective time step and given a real PIV mask, drop-out
  holes and measurement noise; half of all windows are mirrored in the cross-stream direction (flip y, negate v: the
  exact flow at the negative angle of attack). Details in `METHOD.md`, section 3.
- Official baseline checkpoints used, if any: none. Training starts from random initialisation.
- Statistics of the training data used at inference: the gated decode (`rpde/gated_decode.py`) needs the highest
  training Reynolds number per angle of attack and the mean mask-geometry features per angle. `train()` computes them
  from `train_real` (`tools/fit_gate.py`) and writes them to `/output/logs/gate.json` and `/output/ens_cfg.json`;
  they equal the tables of submission 948641 to float32 precision. Nothing from the test data is used anywhere.
- Third-party code and its license: `third_party/realpde_kit/scoring.py`, unmodified from the organisers' Track 1
  starting kit (v9), used for the validation score logged during training. Libraries: PyTorch (BSD-3-Clause), NumPy
  (BSD-3-Clause), h5py (BSD-3-Clause), PyYAML (MIT). All other code is ours.

## Randomness
- Seed(s): 3 and 4, one per member, set in `solution.py`. `rpde.train` seeds Python, NumPy and torch, and every training
  window is drawn from `numpy.random.default_rng([seed, epoch, index])`, so the data order is fixed. The weights are
  not bit-reproducible across GPU models (cuDNN autotuning, TF32 and bf16 autocast).
- How runs or checkpoints are selected, if at all: no selection. Both members are always used, and the epoch (59)
  is fixed in the config. We chose epoch 59, the fluctuation scale 0.70 and the band scale 0.95 during development, by
  training the same recipe with whole Reynolds-number rows of `train_real` held out (folds `R-top`, `R-mid-a`,
  `R-corner` in `rpde/data.py`) and scoring the held-out rows. These are constants of the method; `train()` does not
  repeat that study.

## Contact
- Name and email, reachable during the Decision Phase: TODO

## Notes
- Logs of the training runs behind submission 948641 are in `training_logs_948641/` (exported from Weights & Biases,
  run ids `jcse9a3x` and `tv71r279`). For each member there is `output_log.txt` (the console log, one line per epoch,
  epochs 0-99; it starts at epoch 0 because the logger attached after start-up), `history.txt` (every logged metric
  per epoch, tab separated; epoch 58 there is the shipped snapshot `ep59.pth`) and `config.yaml` (the resolved
  configuration). Those runs used an earlier version of this code with the same recipe; they
  continued to epoch 100 and saved snapshots of epochs 35-75, of which epoch 59 was shipped.
- If the compiled graphs cannot be built on your machine, `train()` still finishes and the predictor runs the eager
  fallback (same outputs, about 4x slower per call).
