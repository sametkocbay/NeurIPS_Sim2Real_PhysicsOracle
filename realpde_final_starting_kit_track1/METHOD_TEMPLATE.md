# <Team name>: Track <1 or 2> method

## 1. Summary
In two or three sentences: what the method is, and which part of it matters most for the score.

## 2. Model
- Architecture and number of parameters:
- Inputs and outputs, including preprocessing and normalization:
- Starting point (random initialization, or which official baseline checkpoint):

## 3. Data
- Training data used (`train_sim`, `train_real`) and how the two are combined:
- Validation split carved out of the training data, if any:
- Augmentation:
- Data left out, and why:

## 4. Training
One row per stage, in the order `train()` runs them.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |

- How runs or checkpoints are selected, if at all:
- Seeds, and the run-to-run spread you have seen:

## 5. Uncertainty bounds
How `lower` and `upper` are computed, or "not returned":

## 6. Inference
- Track 1: what `predict` does per call, numeric precision, any compilation or caching:
- Track 2: the adaptation rule in `ttt_step`: which parameters are updated, learning rate, steps per window, and what `reset_ttt_state` restores:

## 7. Speed
What you did for the Time score:

## 8. What mattered
The changes that moved your development score most, with rough effect sizes:

## 9. Compute
Hardware, training time, and peak GPU memory of the run behind your submission:

## 10. References
Papers, third-party code and libraries the method builds on:
