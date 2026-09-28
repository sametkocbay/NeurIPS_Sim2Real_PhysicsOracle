# Organiser scoring reference

`scoring.py` is copied unmodified from the official starting kit of the NeurIPS 2026 RealPDE Competition,
Track 1 (Simulation-to-Real), `realpde_t1_starting_kit_v9`, provided by the organisers
(realpde-competition@googlegroups.com). It computes the five leaderboard subscores (rel_l2, tke, mvpe, time, sps).

`rpde.metrics` imports it for the validation score during training. Override the location with `RPDE_KIT=<dir>`.
