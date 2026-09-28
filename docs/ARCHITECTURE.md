# Architecture

![architecture](architecture.svg)

One member maps a window of 20 past PIV frames to the 20 next frames plus residual quantiles. Two members are then
combined and decoded into `prediction, lower, upper`. Code references are to `rpde/`.

## 1. Physics scaffold (no learning except two baseline scalars)

**Preproc** (`scaffold.py`, `Preproc`). From the raw window `x (B, 20, 32, 64, 3)`:
- *core mask*: cells that are exactly zero in `u` over all 20 frames (airfoil and its PIV shadow);
- *freestream* `U`: median of the measured `u` in the top 8 rows (the undisturbed flow). All velocities are divided by
  `U`, so the network sees nondimensional fields and one model covers Re 3,750-27,975;
- *drop-out maps* `p_u, p_v`: per-cell fraction of missing PIV vectors (holes);
- *advection per frame* `a = U dt / dx` and the convective time step `dt* = U dt / chord`;
- the **conditioning vector** `cond (B, 8)`: normalised `U`, `log dt*`, `a`, the mask centroid (x, y) and extent
  (x, y) (which encode the angle of attack) and a real/sim flag.

**Advective baseline** (`advective_baseline`). A frozen-turbulence forecast of the 20 future frames: the time mean of
the input plus `beta` times the last fluctuation, convected downstream by `gamma * a * k` cells at lead `k`. `gamma` and
`beta` are learned scalars (trained values: gamma 0.81, beta 0.05, i.e. mostly the convected time mean). The network
predicts the **correction** to this baseline.

**Input volume** (`build_volume`): `(B, 8, 40, 32, 64)`: `u, v` (observed for the 20 past frames, the baseline for
the 20 future frames), core mask, drop-out maps, `x, y` coordinates and an is-future flag.

## 2. Axial transformer core (`cores/axial.py`, 3.44 M parameters)

- **Tokenise.** The past frames are folded into channels (`t2c`), then a `Conv3d` with kernel = stride = (20, 1, 2)
  turns the volume into `(B, 128, 1, 32, 32)`: 1024 space tokens of width 128, one token per 1 x 2 cells, with all 20
  future frames carried in the channels. Learned position embedding.
- **6 axial blocks.** Each: RMSNorm -> multi-head attention along x -> attention along y -> MLP (ratio 8), pre-norm
  residuals, 8 heads. Every sub-layer is modulated by **FiLM** (scale, shift, zero-initialised gate) from a conditioning
  MLP `8 -> 64` shared by all blocks, so the flow regime (Re, angle) steers every layer.
- **Un-patch.** `ConvTranspose3d` back to `(B, 128, 20, 32, 64)`: one feature vector per future frame and cell.

## 3. Head (`scaffold.py`, `Head`) and member output

A 1 x 1 convolution per future frame and cell gives
- `dmean` (2): correction to the baseline; `mean = (baseline + dmean) * U`, zero inside the mask;
- `fluct` (2): a fluctuation field with zero time mean (the turbulent part, which the mean alone cannot carry);
- 5 residual quantiles per channel (levels 0.1, 0.25, 0.5, 0.75, 0.9), non-crossing by construction (cumulative
  softplus increments), in nondimensional units.

`prediction = mean + gamma * fluct`. The quantiles describe the distribution of `target - mean` and feed the band.

## 4. Ensemble decode (`ensemble_predict.py`, `EnsPredict`)

The two members (seeds 3 and 4, epoch 59) run on the same input, then:
1. `mean` and the quantiles are averaged over the members (the quantiles level by level);
2. **energy-preserving combine**: averaging two fluctuations that disagree in phase removes energy exactly where the
   true turbulence is strongest. The averaged fluctuation is rescaled per cell to the members' mean temporal energy
   (gain `mean_k var_t(f_k) / var_t(mean_k f_k)`, clipped to [1, 16]);
3. `prediction = mean + 0.70 * fluctuation`;
4. **gated decode** (`gated_decode.py`): for windows whose estimated Re lies above the highest training Re for their
   estimated angle, the fluctuation is damped with lead time and the mean shrunk slightly (fitted offline);
5. **band** (`decode.py`, `band_from_quantiles`): centred on the mean; the half-width per cell is the value that
   maximises the expected SPS reward `exp(-width / sigma) * P(target inside)` under the quantile distribution, chosen
   from 32 candidates, then scaled by 0.95;
6. `p = 0`, NaN guard, `lower <= upper`.

The whole of section 4 together with both members is compiled ahead of time into one graph per batch size (see
[SUBMISSION.md](SUBMISSION.md)).

## 5. Training signal (see [TRAINING.md](TRAINING.md))

Relative L2 on the mean, relative L2 on the prediction (trains the fluctuation), pinball loss on the quantiles, a TKE
map loss and, from epoch 40, a differentiable version of the SPS band reward. 75 % of the windows are real PIV, 25 %
simulated windows made to look like PIV (strided to the real time step, real masks, drop-out holes, noise). Half of
the windows are mirrored in y (the exact flow at -AoA), which teaches the angle dependence as a symmetry.
