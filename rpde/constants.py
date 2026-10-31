"""Geometry, scorer and bookkeeping constants. Submission-safe (stdlib only).

Every number here is either an organizer constant or a measurement from the
week-one analysis. Do not tune them here; tune in configs.
"""
from __future__ import annotations

import os

# --- evaluation geometry --------------------------------------------------
T_IN = 20
T_OUT = 20
T_VOL = T_IN + T_OUT          # 40-frame volume the cores operate on
H = 32
W = 64
C_IO = 3                      # [u, v, p]; p == 0 for real data

# --- physical constants (measured, REPORT.md §1.2 / §3.2) -----------------
DX = 3.4216e-3                # m, grid spacing at 32x64 (2 x 1.7108e-3; 64x128 has 1.711 mm)
DT = 0.02                     # s per real frame; sim frame in sim time units
CHORD = 0.0692                # m, NACA4418 chord (from U_inf-Re regression)
NU = 1.0e-6                   # m^2/s, water
U_PER_RE = 1.445e-5           # m/s per unit Re:  U_inf = U_PER_RE * Re
DTSTAR_SIM = 0.019            # sim convective step per stored frame (flat in Re)

# --- scorer constants (starting kit scoring.py) ---------------------------
SIGMA_GLOBAL = 0.0563870259
T_NUMERICAL_SEC = 0.72896

# --- freestream band for U_inf estimation (row 0 = top of FOV, largest y) --
FS_ROWS = slice(0, 8)

# --- regime bookkeeping ---------------------------------------------------
RE_GRID = [3750 + 1275 * k for k in range(20)]
AOA_GRID = [0, 5, 10, 15, 20]
HIDDEN_ROWS = (15225, 27975)
MISSING_CORNERS = [
    (3750, 15), (17775, 0), (22875, 5), (22875, 20),
    (24150, 5), (25425, 5), (26700, 5), (26700, 20),
]
BAD_REAL = {"7575_0"}         # organizer notice 2026-08-12: duplicate of 6300_0

# --- leaderboard combination (recovered; treated as given) ----------------
LBW = dict(rel_l2=0.468, tke=0.100, mvpe=0.097, time=0.097, sps=0.249, bias=0.53)
TIME_SCORE_REF = 90.5         # time_score at ~8 ms/sample, used when time is held fixed

# --- uncertainty head ------------------------------------------------------
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
# Opt-in nine-level head.
# The decode's piecewise-linear quantile function saturates at the outer knots:
# with the five default levels the coverage of a band can never exceed 0.8 plus
# the two extrapolated tails, i.e. it stops growing at |z| ~ 1.69 s; the 0.05 /
# 0.95 knots push that to ~2 s.
QUANTILE_LEVELS_9 = (0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95)

# --- paths (overridable) ---------------------------------------------------
# Resolved relative to the checkout; every path can be overridden with an environment variable.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.environ.get("RPDE_CACHE") or os.path.join(_REPO, "data", "cache32")
KIT_DIR = os.environ.get("RPDE_KIT") or os.path.join(_REPO, "third_party", "realpde_kit")
RUNS_DIR = os.environ.get("RPDE_RUNS") or os.path.join(_REPO, "runs")


def sim_stride(re_nom: int, lo: int = 1, hi: int = 8) -> int:
    """Frame stride that aligns a sim window's convective horizon with real
    data at the same Re:  k = round(dt*_real(Re) / dt*_sim)."""
    dtstar_real = DT * U_PER_RE * re_nom / CHORD
    return int(min(hi, max(lo, round(dtstar_real / DTSTAR_SIM))))


def lbw(rel_l2_s: float, tke_s: float, mvpe_s: float, sps_s: float,
        time_s: float = TIME_SCORE_REF) -> float:
    """Leaderboard-weighted final score from the five 0-100 subscores."""
    return (LBW["rel_l2"] * rel_l2_s + LBW["tke"] * tke_s + LBW["mvpe"] * mvpe_s
            + LBW["time"] * time_s + LBW["sps"] * sps_s + LBW["bias"])
