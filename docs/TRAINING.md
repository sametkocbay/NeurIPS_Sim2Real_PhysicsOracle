# Training

The submission is an ensemble of **two members** that share one recipe (`configs/final.yaml`) and differ only in the
random seed (**3** and **4**). Each member is trained for the full 100-epoch schedule and the **EMA weights of epoch
59** are shipped.

## 1. Environment

Python 3.10, `torch==2.2.2` (CUDA 12.1 build), `numpy==1.26.4`, `pyyaml`, and `h5py` for the cache builder
(`pyproject.toml`; `uv sync` installs it). Optional: `wandb` for live curves (`--wandb`).

Hardware per member: one GPU with >= 8 GB (an RTX 4090 or A100 runs at about 70 s per epoch including the per-epoch
validation, about 2 h in total), 8 CPU cores for the data loader, >= 16 GB RAM (the whole corpus is held in memory,
~3.6 GB). The two members are independent and can train in parallel on two GPUs.

## 2. Data

The public training data is mirrored on Hugging Face
([`AI4Science-WestlakeU/RealPDE-Competition-Data`](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data)),
about 16 GB of h5 files:

```bash
mkdir -p /path/to/raw && cd /path/to/raw
curl -fL https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data/resolve/main/train_real.tar.gz | tar xz
curl -fL https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data/resolve/main/train_sim.tar.gz  | tar xz
```

`train_real/` holds 82 PIV trajectories `<Re>_<AoA>.h5` (`u, v` at 64 x 128, 282-868 frames), `train_sim/` 100 CFD
trajectories (`u, v, p`, 1000 frames). Build the 32 x 64 cache the training reads (the evaluation grid is every second
row and column):

```bash
python tools/make_cache.py --raw /path/to/raw            # -> data/cache32/{real,sim}/<Re>_<AoA>.npz  (~3.4 GB)
```

`7575_0` (a duplicate of `6300_0`, announced by the organisers) is written to the cache but never used. Point the
training at another cache location with `RPDE_CACHE=<dir>`.

## 3. Train

```bash
python -m rpde.train --config configs/final.yaml --fold full --seed 3 --name fm_all_s3
python -m rpde.train --config configs/final.yaml --fold full --seed 4 --name fm_all_s4
```

Output in `runs/<name>/` (override with `--runs-dir` or `RPDE_RUNS`):

| file | content |
|---|---|
| `ep59.pth` | **the shipped weights** (EMA snapshot after epoch 59) |
| `last.pth` | epoch 100, with optimizer state (resume with `--resume`) |
| `best.pth` | best validation epoch (not used, see below) |
| `log.txt`, `run.json`, `config.json` | per-epoch losses and validation scores, the resolved config |

Strip a snapshot to the submission format (EMA weights + architecture only, 13.8 MB):
`python tools/strip_ckpt.py runs/fm_all_s3/ep59.pth weights/fm_all_s3_ep59.pth`. The files in `weights/` were made
this way from the original runs.

## 4. The recipe

| | |
|---|---|
| Model | axial transformer, width 128, 6 blocks, 8 heads, MLP ratio 8, RMSNorm, shared FiLM, 1 x 2 spatial patches, all 20 input frames in one temporal token; 3.44 M parameters |
| Batch | 8 windows; 6000 windows per epoch; **75 % real / 25 % simulated** windows |
| Windows | 20 input + 20 target frames, drawn uniformly per regime; **`train_all`**: from the whole trajectory (no frames held out) |
| Sim-to-real | simulated windows are strided to the real convective time step and get the real PIV mask, drop-out holes and measurement noise |
| Augmentation | **mirror** (p = 0.5): flip the window in y and negate v, i.e. the exact flow at -AoA |
| Optimiser | AdamW, lr 1e-3, weight decay 0.05, 5 warm-up epochs, cosine decay over 100 epochs, grad clip 1.0, AMP; EMA of the weights with decay 0.9995 |
| Loss | relative L2 of the mean + relative L2 of the prediction (mean detached, so it trains the fluctuation) + 4.0 x pinball loss of the 5 quantiles + 0.2 x TKE-map loss + (from epoch 40) 1.0 x a differentiable SPS band reward; measured cells only |
| Seeds | **3 and 4** |
| Shipped epoch | **59** (EMA), fixed; no early stopping |

The data draws are a pure function of `(seed, epoch, index)`, so a rerun reproduces the same batches. The weights are
not bit-reproducible across GPUs (cuDNN autotuning and TF32 are on), so a retrained member differs from `weights/` at
the level of a seed-to-seed difference.

## 5. Why epoch 59, and why no early stopping

With `train_all` every frame is used for training, so the built-in validation (the last 120 frames of each trajectory)
scores frames the model has seen and cannot pick the epoch. The epoch was chosen on **held-out Reynolds-number rows**
instead: the same recipe was trained with whole regimes removed (`--fold R-top`: Re 26,700; `--fold R-mid-a`:
Re 16,500; `--fold R-corner`: Re >= 22,875 at AoA 5 and 20; two seeds each), and the 2-member ensemble was scored on
the held-out rows for every epoch 35-75.

| epoch | 53 | 55 | 57 | 58 | **59** | 60 | 62 | 63 | 64 | 66 | 68 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean gap to each fold's peak (score points) | -0.024 | -0.014 | -0.015 | -0.012 | **-0.008** | -0.007 | -0.008 | -0.006 | -0.014 | -0.035 | -0.057 | -0.166 |

Epochs 58-63 are within ~0.015 of every fold's peak; after epoch 64 the score falls quickly (the model starts to
memorise the seen trajectories), and the final epoch 100 is clearly worse on unseen regimes even though the in-regime
validation recovers late in training. Epoch 59 has the best worst case (-0.009) and leaves 5 epochs of margin before
the drop. The held-out folds are available in `rpde/data.py` (`FOLDS`) to reproduce this study, e.g.
`python -m rpde.train --config configs/final.yaml --fold R-top --seed 3 --name fold_rtop_s3 --set eval.save_from=35 --set eval.save_to=75`.

The decode settings were calibrated the same way on the held-out folds: band scale 0.95 is optimal on every fold;
the fluctuation scale 0.70 is within ~0.01 of the best compromise (the folds disagree: R-top prefers ~0.62, R-corner
~0.85), and per-seed calibration gives the same result as calibrating the ensemble.
