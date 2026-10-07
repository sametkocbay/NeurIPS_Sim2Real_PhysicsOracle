# PhysicsOracle: Track 1 training package

Two-member physics-scaffolded axial transformer, trained from scratch and compiled ahead of time. See [METHOD.md](METHOD.md) for the model, training recipe and uncertainty bounds.

## Codabench

- Username: physicsoracle
- Submission id this package reproduces: 948641
- final_score of that submission: 82.01030

## How to reproduce

- Build command (same as default):
```bash
docker build -t physicsoracle .
```
- Train command (same as default):
```bash
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /path/to/train_data:/data:ro \
  -v /path/to/output:/output \
  physicsoracle \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```


## Hardware we trained the submitted model on

- GPU model and count: 4 x NVIDIA GeForce RTX 4090 (used one per member)
- CPU: AMD EPYC 7V12, 64 cores
- Memory: 504 GiB
- Training time on this hardware: 156.6 min (we trained to 100 epochs but submitted the epoch-59 checkpoint)
- Peak GPU memory on this hardware: 3.31 GiB (one member)


## Environment

- Base image: `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel`, with Python 3.10, PyTorch 2.2.2, Triton 2.2.0, CUDA 12.1 and cuDNN 8. The development image supplies the CUDA and C++ tools needed for ahead-of-time compilation; `CUDA_HOME=/usr/local/cuda`.
- Added packages and versions (pinned the same way in the Dockerfile): `h5py==3.11.0`, `pyyaml==6.0.1`, `numpy==1.26.4`.


## Data, weights and code

- **Training data used:** only the released `train_real` and `train_sim` data: 81 real and 100 simulated trajectories, downsampled to 32 × 64. Windows are sampled with 75% real and 25% simulated probability. Simulated regimes without matching real measurements receive twice the sampling weight of other simulated regimes. The final runs reserve no independent validation split; 243 monitoring windows overlap with data available for training.
- **Augmentation:** simulation frames are subsampled to approximate the real convective time step. Simulated input and target frames receive real measurement masks, random missing vectors and noise. Every window has a 50% probability of vertical reflection, with the cross-stream velocity negated.
- **Official baseline checkpoints used:** none; both members start from random initialization. 
- **Third-party code and its license:** `third_party/realpde_kit/scoring.py`, copied unmodified from the organizers' Track 1 starting kit v9, supplies the monitoring subscores.

Note: The gated decode uses reference tables fitted from `train_real` during `train()`, stored in `/output/logs/gate.json` and `/output/ens_cfg.json`. No test data is used for training or fitting these tables. All weights used at inference are produced by `train()`.

## Randomness

- **Seed(s):** 3 and 4. Python, NumPy and PyTorch are seeded; window sampling and augmentation are determined by the seed, epoch and sample index. Exact weight reproduction across GPU models is not guaranteed.
- **How runs or checkpoints are selected:** both members' EMA weights after 59 epochs are always used. The seeds, stopping epoch and decode settings are fixed in `configs/final.yaml`; `train()` does not repeat development sweeps or choose among runs. The trainer also saves a checkpoint selected by the monitoring score, but it is not used for submission.


## Contact
- Team email: physicsoracle.awesome@gmail.com

**Team Members**
- Samet Kocbay (samet.kocbay@tum.de)
- Oscar Breiner (oscar.breiner@tum.de)
- Maximilian Leutschafft (maximilian.leutschafft@tum.de)


## Notes

- The assembled package includes the original W&B exports in `training_logs_948641/` as the submission logs. In the repository, these are under `logs/wandb/`. Logged epoch 58 corresponds to the checkpoint after 59 completed epochs.
- Keep `ens_cfg.json` alongside the model checkpoints in the output directory. The eager Python/PyTorch fallback requires it to load the ensemble settings and gated-decode reference tables.
- `train()` writes its logs to `/output/logs/`: `solution.log`, `stages.log`, the member training logs, `gate.json` and `summary.json` (stage times, peak GPU memory and the compiled batch sizes, expected 1, 2, 3, 4, 5 and 10).
- Adding `-e RPDE_SMOKE=1` to the train command runs the same pipeline with one short epoch and one compiled group, to check the setup in minutes. Its output is not the submitted model.
