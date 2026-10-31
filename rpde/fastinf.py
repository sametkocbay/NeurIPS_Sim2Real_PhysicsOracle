"""Opt-in fast inference for `RPDEModel` -- same architecture, same weights.

Nothing here changes the network: it only chooses the precision, the SDPA
backend, the fused (but weight-identical) elementwise path of the axial core,
the vectorised decode and -- optionally -- a CUDA-graph replay of the forward
pass.  Every option degrades safely: an unsupported backend, a failed graph
capture or a missing CUDA device falls back to the eager fp32 path.

Submission-safe: torch + stdlib only.

Measured on an A100-40GB, batch 3, 24 real windows --
"""
from __future__ import annotations

import contextlib
import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["FastInference", "CudaGraphRunner", "RECOMMENDED", "set_tf32"]

_DTYPES = {
    None: None, "fp32": None, "none": None, "off": None,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "half": torch.float16,
}

# The configuration measured best on an A100-40GB
# (31.9 ms/sample vs 95.4 fp32 eager, 3.0x; max |dpred| 3.5e-2 sigma; official
# subscores unchanged to 3 decimals).
RECOMMENDED: Dict[str, Any] = dict(
    tf32=True,
    autocast=None,             # not needed once the core weights are fp16
    half_weights="fp16",       # fp16 copy of the core; fp32 master untouched
    sdp="mem_efficient",       # head dim 16: cutlass beats flash and math here
    fast_axial=True,
    fast_decode=True,
    cudnn_benchmark=True,
    inference_mode=True,
    cuda_graph=False,          # measured +1.8% only; kept as an option
)


def set_tf32(on: bool = True) -> None:
    """TF32 for matmul and cuDNN (a global torch setting, not per-model)."""
    torch.backends.cuda.matmul.allow_tf32 = bool(on)
    torch.backends.cudnn.allow_tf32 = bool(on)
    try:
        torch.set_float32_matmul_precision("high" if on else "highest")
    except Exception:                                       # pragma: no cover
        pass


def _sdp_ctx(mode: Optional[str]):
    """A context-manager factory restricting the SDPA backend, or None."""
    if mode in (None, "", "default", "auto"):
        return None
    fn = getattr(torch.backends.cuda, "sdp_kernel", None)
    if fn is None:                                          # pragma: no cover
        return None
    kw = dict(enable_flash=False, enable_math=False, enable_mem_efficient=False)
    for part in str(mode).split("+"):
        part = part.strip()
        if part == "flash":
            kw["enable_flash"] = True
        elif part in ("mem", "mem_efficient", "efficient"):
            kw["enable_mem_efficient"] = True
        elif part == "math":
            kw["enable_math"] = True
        else:
            raise ValueError(f"unknown sdp backend {mode!r}")
    return lambda: fn(**kw)


class HalfCore(nn.Module):
    """A half/bfloat16 COPY of the core with fp32 I/O.

    The copy is deep, so the model keeps its fp32 master weights and the
    default eager path is unaffected; `FastInference` swaps this module in for
    the duration of a fast forward and swaps the fp32 core back afterwards.
    """

    def __init__(self, core: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.core = copy.deepcopy(core).to(dtype)
        self.dtype = dtype
        self.out_channels = int(core.out_channels)

    def forward(self, vol: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.core(vol.to(self.dtype), cond.to(self.dtype))
        return out.float()


# ---------------------------------------------------------------------------
# CUDA graphs
# ---------------------------------------------------------------------------
class CudaGraphRunner:
    """Capture `fn(*args)` once and replay it for identically shaped inputs.

    The captured tensors are static: inputs are copied into the capture buffers
    and outputs are cloned out (a graph replay overwrites them).  Capture needs
    a side-stream warm-up; anything that raises during capture is the caller's
    signal to fall back to eager.
    """

    def __init__(self, fn, args, warmup: int = 3, clone: bool = True) -> None:
        self.fn = fn
        self.clone = bool(clone)
        self.static_in = tuple(a.detach().clone() for a in args)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn(*self.static_in)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = fn(*self.static_in)

    def matches(self, args) -> bool:
        return (len(args) == len(self.static_in)
                and all(tuple(a.shape) == tuple(s.shape) and a.dtype == s.dtype
                        for a, s in zip(args, self.static_in)))

    def __call__(self, *args):
        for s, a in zip(self.static_in, args):
            s.copy_(a)
        self.graph.replay()
        out = self.static_out
        if not self.clone:
            return out
        if isinstance(out, dict):
            return {k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in out.items()}
        if isinstance(out, (list, tuple)):
            return type(out)(v.clone() if torch.is_tensor(v) else v for v in out)
        return out.clone()


class GraphedCore(nn.Module):
    """Drop-in replacement for `model.core` that replays a captured graph.

    Captures lazily on the first call (so the shapes come from real data) and
    permanently reverts to eager if capture raises.
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core
        self.out_channels = int(core.out_channels)
        self.runner: Optional[CudaGraphRunner] = None
        self.error: Optional[str] = None

    def forward(self, vol: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.runner is None and self.error is None:
            try:
                self.runner = CudaGraphRunner(self.core, (vol, cond))
            except Exception as exc:
                self.error = repr(exc)
        if self.runner is not None and self.runner.matches((vol, cond)):
            return self.runner(vol, cond)
        return self.core(vol, cond)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
class FastInference:
    """Runs `RPDEModel` with a chosen inference configuration.

    Parameters (all optional, defaults = `RECOMMENDED`):
      tf32             TF32 matmul/cuDNN + float32_matmul_precision("high")
      autocast         None/"fp32" | "bf16" | "fp16"  (autocast for the core)
      half_weights     None | "bf16" | "fp16"  -- cast the core's *weights*,
                       keeping preproc/baseline/head/decode in fp32
      sdp              "default" | "flash" | "mem_efficient" | "math" | "a+b"
      fast_axial       fused elementwise path of the axial core
      fast_decode      vectorised (bit-exact) width grid in the decode
      width_n          decode width-grid size (32 = shipped default)
      cudnn_benchmark  torch.backends.cudnn.benchmark
      inference_mode   torch.inference_mode() instead of torch.no_grad()
      cuda_graph       capture+replay ("model" or "core" scope)
    """

    def __init__(self, model: nn.Module, *,
                 tf32: bool = True,
                 autocast: Optional[str] = None,
                 half_weights: Optional[str] = "fp16",
                 sdp: Optional[str] = "mem_efficient",
                 fast_axial: bool = True,
                 fast_decode: bool = True,
                 width_n: int = 32,
                 cudnn_benchmark: bool = True,
                 inference_mode: bool = True,
                 cuda_graph: bool = False,
                 graph_scope: str = "model",
                 safety_check: bool = True) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.cuda = self.device.type == "cuda"
        self.autocast_dtype = _DTYPES[autocast] if self.cuda else None
        # half-precision kernels (LayerNorm in particular) are CUDA-only in
        # torch 2.2, so on CPU the fp32 path is the only correct one
        self.half_weights = _DTYPES[half_weights] if self.cuda else None
        self.sdp = sdp
        self.fast_decode = bool(fast_decode)
        self.width_n = int(width_n)
        self.cuda_graph = bool(cuda_graph) and self.cuda
        self.graph_scope = str(graph_scope)
        self.safety_check = bool(safety_check)
        self._core_override: Optional[nn.Module] = None
        self.inference_mode = bool(inference_mode) and not self.cuda_graph
        self.errors: list = []
        self._graphs: Dict[Any, CudaGraphRunner] = {}
        self._failed: set = set()
        self._grid: Optional[torch.Tensor] = None

        model.eval()
        if self.cuda:
            set_tf32(bool(tf32))
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        self._sdp_factory = _sdp_ctx(sdp) if self.cuda else None

        # The fused axial path is toggled per call (see `_apply`) so the eager
        # path stays bit-identical for callers that use both.  The module lists
        # are cached: walking `.modules()` twice per forward is pure overhead.
        self.fast_axial = bool(fast_axial)
        self._toggle: list = []
        if self.fast_axial:
            from rpde.cores.axial import AxialBlock, AxialCore
            from rpde.scaffold import Head
            mods = list(model.modules())
            has_axial = any(isinstance(m, AxialCore) for m in mods)
            self._toggle = [m for m in mods
                            if isinstance(m, (AxialBlock, AxialCore))
                            or (has_axial and isinstance(m, Head))]
            self.fast_axial = bool(self._toggle)
            self.axial_report = self._set_fast(True)
            self._set_fast(False)          # armed, but off until a fast forward
        else:
            self.axial_report = None

        # The core is never replaced on the model: the override is swapped in
        # for the duration of a fast forward, so `predict_with_bounds(fast=False)`
        # keeps using the untouched fp32 module.
        if self.half_weights is not None:
            # the copy is taken with the fused flags ON, so it keeps them
            self._set_fast(True)
            self._core_override = HalfCore(model.core, self.half_weights)
            self._set_fast(False)
        if self.cuda_graph and self.graph_scope == "core":
            self._core_override = GraphedCore(self._core_override or model.core)

    def _set_fast(self, on: bool) -> dict:
        """Flip the fused axial path on the model's own modules."""
        from rpde.cores.axial import AxialBlock, AxialCore
        rep = dict(blocks=0, cores=0, heads=0)
        for m in self._toggle:
            if isinstance(m, AxialBlock):
                m.fast = on
                rep["blocks"] += 1
            elif isinstance(m, AxialCore):
                m.fast = on
                m.channels_last_out = on
                rep["cores"] += 1
            else:                                    # Head
                m.channels_last_in = on
                rep["heads"] += 1
        return rep

    def core_module(self) -> nn.Module:
        """The module a fast forward actually runs (override or the model's)."""
        return self._core_override if self._core_override is not None else self.model.core

    @contextlib.contextmanager
    def _apply(self):
        """Fused path on + fp16 core swapped in, both undone on exit.

        Leaves the model exactly as it was found, so a later
        `predict_with_bounds(x)` (no `fast`) is still bit-identical to the
        untouched eager reference.
        """
        orig = self.model.core
        if self.fast_axial:
            self._set_fast(True)                 # must precede the swap: the
        if self._core_override is not None:      # override keeps its own flags
            self.model.core = self._core_override
        try:
            yield
        finally:
            self.model.core = orig
            if self.fast_axial:
                self._set_fast(False)

    # -- contexts ---------------------------------------------------------
    def _ctx(self):
        st = contextlib.ExitStack()
        st.enter_context(torch.inference_mode() if self.inference_mode
                         else torch.no_grad())
        if self.autocast_dtype is not None and self.cuda:
            st.enter_context(torch.autocast("cuda", dtype=self.autocast_dtype))
        if self._sdp_factory is not None:
            st.enter_context(self._sdp_factory())
        return st

    def width_grid(self, device, dtype=torch.float32) -> Optional[torch.Tensor]:
        if self.width_n == 32:
            return None                       # the shipped default grid
        from rpde.decode import default_width_grid
        if self._grid is None or self._grid.device != device:
            self._grid = default_width_grid(device=device, dtype=dtype,
                                            n=self.width_n)
        return self._grid

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, gamma: Optional[float] = None) -> Dict[str, torch.Tensor]:
        with self._ctx(), self._apply():
            if self.cuda_graph and self.graph_scope == "model":
                key = (tuple(x.shape), x.dtype, gamma)
                runner = self._graphs.get(key)
                if runner is None and key not in self._failed:
                    try:
                        runner = CudaGraphRunner(
                            lambda t: self.model(t, gamma=gamma), (x,))
                        self._graphs[key] = runner
                    except Exception as exc:
                        self._failed.add(key)
                        self.errors.append(f"graph(model): {exc!r}")
                if runner is not None:
                    return runner(x)
            return self.model(x, gamma=gamma)

    def predict_with_bounds(self, x: torch.Tensor, gamma: Optional[float] = None,
                            spread: Optional[torch.Tensor] = None,
                            calib=None):
        """(pred, lower, upper), physical units, decode always in fp32.

        `calib` may be a `rpde.calibrate.CalibratedDecode`; it then replaces
        `band_from_quantiles` and receives the same fp32 inputs.
        """
        from rpde.decode import band_from_quantiles

        out = self.forward(x, gamma=gamma)
        if self.safety_check and not bool(torch.isfinite(out["pred"]).all()):
            # reduced precision produced a non-finite field: redo in fp32 eager
            print("[fastinf] non-finite output, recomputing in fp32 eager")
            with torch.no_grad():
                out = self.model(x, gamma=gamma)
        with torch.no_grad():
            Q = out["Q_nd"].float()
            mean = out["mean"].float()
            U = out["U"].float()
            mk = out["m_core"].float()
            lv = out.get("levels") if isinstance(out, dict) else None
            es = None
            if hasattr(self.model, "decode_err_scale"):
                es = self.model.decode_err_scale(out)
            if calib is not None:
                lower, upper = calib.apply(
                    Q, mean, U, mk, x=x, pred=out["pred"].float(),
                    fast=self.fast_decode, levels=lv, err_scale=es)
            else:
                lower, upper = band_from_quantiles(
                    Q, mean, U, mk, width_grid=self.width_grid(Q.device, Q.dtype),
                    spread=spread, fast=self.fast_decode, levels=lv, err_scale=es)
        return out["pred"].float(), lower, upper
