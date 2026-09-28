"""Core registry.  `build_core(name, in_ch, cond_dim, **kw) -> Core`.

A Core is an `nn.Module` with an `out_channels` attribute and
`forward(vol: (B, in_ch, 40, 32, 64), cond: (B, cond_dim)) -> (B, out_channels, 40, 32, 64)`.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

__all__ = ["build_core", "CORES"]

CORES = ("axial",)


def build_core(name: str, in_ch: int = 8, cond_dim: int = 8, **kw: Any) -> nn.Module:
    name = str(name).lower()
    if name == "axial":
        from rpde.cores.axial import AxialCore
        return AxialCore(in_ch=in_ch, cond_dim=cond_dim, **kw)
    raise ValueError(f"unknown core {name!r}; expected one of {CORES}")
