# PhysicsOracle: Track 1 method

## 1. Summary
A parameter-free physics scaffold estimates the freestream velocity, reads the airfoil geometry from the PIV mask and
builds a convected-turbulence baseline forecast. A 3.4 M-parameter axial transformer predicts the correction to that
baseline, a zero-mean fluctuation field and residual quantiles. Two seeds are combined by an energy-preserving ensemble
decode and compiled ahead of time into one graph. What matters most is the physics normalisation and baseline, which
make one model valid from Re 3,750 to 27,975. Then come joint training on real PIV and on simulation made to look like
PIV, and the band decode that maximises the expected SPS reward.

## 2. Model
- **Architecture and number of parameters:** 3,440,787 parameters per member, two members.
  - **Scaffold** (`rpde/scaffold.py`):
    - Freestream `U` is the median of the measured `u` in the top 8 rows, and all velocities are divided by `U`.
    - The core mask (cells with `u` exactly zero over the window) gives the airfoil and its shadow.
    - Drop-out maps mark the cells with missing PIV vectors.
    - An 8-dimensional conditioning vector holds `U`, `log dt*`, the advection per frame, the mask centroid (x, y) and
      extent (x, y), and a real/sim flag.
    - The advective baseline is the input time mean plus `beta` times the last fluctuation, convected downstream by
      `gamma * a * k` cells at lead `k`. `beta` and `gamma` are two learned scalars.
  - **Core** (`rpde/cores/axial.py`):
    - The 20 past frames are folded into channels, and the input volume `(8, 40, 32, 64)` is cut into 1 x 2 spatial
      patches: 1024 tokens of width 128.
    - 6 blocks, each with RMSNorm, 8-head attention along x, attention along y, and an MLP (ratio 8, dropout 0.1).
    - Every sub-layer is modulated by FiLM from one shared conditioning MLP.
    - A transposed convolution maps the tokens back to `(128, 20, 32, 64)`.
  - **Head:** a 1 x 1 convolution per future frame and cell gives:
    - a correction to the baseline mean (2 channels);
    - a fluctuation field with zero temporal mean (2 channels);
    - 5 non-crossing residual quantiles per channel, at levels 0.1, 0.25, 0.5, 0.75 and 0.9.
- **Inputs and outputs, including preprocessing and normalization:**
  - The input is the raw window `(N, 20, 32, 64, 3)` in physical units (`p` is ignored).
  - Normalisation is per window, by its own freestream estimate `U`. No dataset statistics are used and
    `mean_std_real.pt` is not read.
  - A member outputs `mean = (baseline + dmean) * U`, `fluct * U` and the quantiles of `target - mean`, with zeros
    inside the mask.
  - The final output is `prediction`, `lower` and `upper` in physical units, with `p = 0`.
- **Starting point:** random initialisation (`torch.manual_seed(seed)`); no official checkpoint is used.
- **Where the numbers are:**
  - `configs/final.yaml`: the training recipe of a member (sections `core` to `eval`) and every decision of the final
    ensemble (section `submission`: seeds, epoch, decode settings, gate coefficients, compiled batch sizes).
  - `rpde/constants.py`: facts about the data and the task (grid, time step, chord, Reynolds and angle grids,
    quantile levels, the scorer's constants).
  - `rpde/data.py`: the parameters of the PIV-likeness augmentation of simulated windows (noise level, drop-out holes).
  - The gated decode's data tables are fitted in `train()` (section 6).

## 3. Data
- **Training data used and how the two are combined:**
  - All 81 `train_real` trajectories and all 100 `train_sim` trajectories, at 32 x 64 (every second row and column,
    the evaluation grid).
  - Each window (20 input + 20 target frames) is drawn by first picking a regime: real regimes uniformly, with 75 % of
    the probability, and simulated regimes with 25 %. Then a start frame is picked uniformly, and every frame of every
    trajectory can be drawn (`train_all`).
  - Some simulated regimes have no real recording: the two Reynolds rows absent from `train_real` (15,225 and 27,975)
    and 8 (Re, angle) combinations missing from it. Their simulated trajectories get twice the sampling weight inside
    the simulated share, so that the gaps of the real data are covered (`HIDDEN_ROWS`, `MISSING_CORNERS` in
    `rpde/constants.py`, read off the file lists of `train_real` and `train_sim`).
  - Simulated trajectories are strided to the real convective time step of their Reynolds number.
  - 6000 windows per epoch.
- **Validation split carved out of the training data, if any:** none in `train()`.
  - The validation score logged per epoch (243 windows from the last 120 frames of each trajectory) uses frames that
    are also trained on. It is only logged and steers nothing.
  - During development, held-out folds of whole Reynolds rows (`R-top`, `R-mid-a`, `R-corner`; `rpde/data.py`) fixed
    the epoch and the decode constants (section 4).
- **Augmentation:**
  - Simulated windows get the PIV mask of a real recording, PIV-like drop-out holes and measurement noise.
  - Every window is mirrored with p = 0.5: flip in y and negate v, which is the exact flow at the negative angle of
    attack.
  - The draws are a function of `(seed, epoch, index)`.
- **Data left out, and why:** `train_real/7575_0` (a duplicate of `6300_0`, per the organisers) and the simulated
  pressure (the target has `p = 0`).

## 4. Training
One row per stage, in the order `train()` runs them. Both members run the same stage; the second member runs it again
with another seed.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1 (seed 3), then 1 (seed 4) | train_real + train_sim, 75/25 | Relative L2 of the mean, plus relative L2 of the prediction (mean detached), plus 4.0 x pinball loss of the 5 quantiles, plus 0.2 x TKE-map loss, plus (from epoch 40) 1.0 x a differentiable SPS band reward (its tau annealed from 0.02 to 0.001). Measured cells only | AdamW (betas 0.9, 0.95), lr 1e-3, weight decay 0.05 (not on biases, norms, FiLM gates, position embeddings or the baseline scalars), 5 warm-up epochs, then cosine decay over 100 epochs; gradient clipping 1.0; bf16 autocast; EMA of the weights, decay 0.9995 | 59 epochs of 750 steps (44,250 steps). The schedule is laid out for 100 epochs and training stops after epoch 59 | 8 |

- **How runs or checkpoints are selected, if at all:** no selection inside `train()`. Each member ships its EMA weights
  after epoch 59, a fixed number in `configs/final.yaml`.
  - That epoch was chosen during development on the held-out folds. The 2-member ensemble was scored on the held-out
    rows at every epoch from 35 to 75.
  - Epochs 58-63 are within 0.015 score points of each fold's best, and later epochs lose quickly (epoch 100: -0.17).
  - Epoch 59 has the best worst case.
- **Seeds, and the run-to-run spread you have seen:** seeds 3 and 4.
  - At epoch 59 their logged validation scores are 85.88 and 85.91 (rel_l2 0.0757 and 0.0755, sps 55.44 and 55.52).
  - The two-member ensemble scored +0.09 above a single member on the hidden development set.

## 5. Uncertainty bounds
- **Averaging:** the members' quantiles of `target - mean` are averaged level by level.
- **Band:** the band is centred on the ensemble mean. Its half-width per cell is chosen from 32 candidates to maximise
  the expected SPS reward `exp(-width / sigma) * P(target inside)`, under the piecewise-linear distribution given by
  the 5 quantiles.
- **Scaling:** the half-width is then multiplied by 0.95, a constant chosen on the held-out folds.
- **Safety:** `lower <= prediction <= upper` holds by construction.

## 6. Inference
- **Track 1: what `predict` does per call, numeric precision, any compilation or caching:**
  - **What runs:** `predict` copies the input through a pinned host buffer to the GPU and runs one compiled graph for
    that batch size. The graph contains both members, the energy-preserving combine, the gated decode and the band
    decode.
  - **Outputs:** the three outputs are copied back into pinned host buffers and returned as float32 arrays.
  - **Precision:** the transformer core runs with fp16 weights, the scaffold and the decode in fp32; attention uses
    the math kernel. The compiled graph agrees with the eager fp16 path to within 1e-4 m/s on the prediction.
  - **Compilation:** AOTInductor with max-autotune, for batch sizes 1, 2, 3, 4, 5 and 10, built at the end of
    `train()` on the training GPU. Other batch sizes are split into compiled sizes.
  - **Load time:** `load_predictor` loads the graphs through their C API (no compiler at run time), runs a warm-up of
    every batch size and allocates the output buffers. Nothing is cached between calls except buffers.
  - **Ensemble decode, per cell:**
    1. Average the members' means and quantiles.
    2. Rescale the averaged fluctuation to the members' mean temporal energy (gain clipped to [1, 16]). This undoes the
       energy that averaging two out-of-phase fluctuations removes.
    3. `prediction = mean + 0.70 * fluctuation`.
    4. Gated decode: for inputs whose estimated Reynolds number lies above the largest training Re for their estimated
       angle, damp the fluctuation with lead time and shrink the mean slightly. The angle is estimated as the nearest
       per-angle centroid of the mask-geometry features. `train()` fits the tables (largest Re per angle, centroids
       and feature spread) on `train_real` (`tools/fit_gate.py`). The decay and shrink coefficients are hyperparameters
       chosen on the held-out folds (`rpde/gated_decode.py`).

## 7. Speed
What we did for the Time score (91.862 on Codabench):
- Ahead-of-time compilation of the whole predict path into one graph per batch size, with no Python in the timed path.
- An fp16 core and a model of only 3.4 M parameters (moving from 10.4 M to 3.4 M parameters also raised the score).
- All 20 input frames folded into one temporal token, so the core has 1024 tokens.
- Warm-up and buffer allocation at load time, and pinned host buffers.

We measured 1.1 ms per window at batch 5 on an A100, and 1.6 ms on an RTX 4090.

## 8. What mattered
Changes in the hidden development score between our submissions:

| Change | Effect |
|---|---|
| Physics baseline alone, then the axial transformer on top of it, with all training data | +2.28 |
| Training recipe: patching and time folding, learning rate, pinball loss and the differentiable band reward | +1.38 |
| Inference: ahead-of-time compilation, gated decode above the training Re range, wider MLP | +0.43 |
| Architecture: all 20 frames in one temporal token, 1 x 2 patches, 3.4 M instead of 10.4 M parameters | +0.12 |
| Two-seed ensemble in one compiled graph | +0.09 |
| Mirror augmentation, training on all frames, energy-preserving combine, fluctuation scale 0.70, band scale 0.95, fixed epoch 59 | +0.08 |
| **Total**, from 77.63 (physics baseline) to 82.01 | **+4.38** |

On the held-out folds, mirror augmentation alone adds +0.44 per model at an angle of attack beyond the training range.

## 9. Compute
- **Hardware of the run behind the submission:** one NVIDIA RTX 4090 per member, on rented 4 x RTX 4090 servers (64
  cores, 504 GiB).
- **Training time:** 82 min per member to epoch 59, including a validation pass per epoch. The original runs continued
  to epoch 100.
- **Peak GPU memory:** 3.17 GiB allocated by PyTorch per member (3.31 GiB reserved), measured in a reproduction run
  of this package on an RTX 4090.
- **Full `train()` on one RTX 4090:** about 3.4 h: 2 x 93 min of training (with a validation pass after every epoch)
  and about 15 min of compilation. On an RTX 4090 the reproduction run gave the submitted seed-3 weights bit for bit.

## 10. References
- **Ideas the method builds on:**
  - Axial attention: Ho et al., *Axial Attention in Multidimensional Transformers*, 2019.
  - FiLM: Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
  - Frozen turbulence: Taylor, *The spectrum of turbulence*, Proc. R. Soc. A, 1938.
  - Pinball (quantile) loss: Koenker and Bassett, *Regression Quantiles*, Econometrica 1978.
- **Software:**
  - PyTorch 2.2.2, including AOTInductor and Triton.
  - NumPy, h5py, PyYAML.
  - The organisers' `scoring.py` (Track 1 starting kit v9), for the validation score.
