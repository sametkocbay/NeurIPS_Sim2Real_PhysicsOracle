"""Run an AOTInductor-compiled model (.so) from Python with NO compiler at runtime.

torch 2.2 only ships a C++ launcher for AOTInductor artefacts (building it needs a compiler).
This module calls the artefact's C API directly through ctypes:
  AOTInductorModelContainerCreate / Run  (exported by the .so)
  aoti_torch_create_tensor_from_blob / get_data_ptr / get_sizes / get_strides / delete_tensor_object
                                         (the C shim exported by libtorch, already loaded by `import torch`)
Inputs are zero-copy views of torch tensors; outputs are wrapped zero-copy through
`__cuda_array_interface__` and cloned (or copied into caller-provided buffers).

Submission-safe: torch + stdlib only.
"""
from __future__ import annotations

import ctypes
import glob
import os
from typing import List, Optional, Sequence

import torch

__all__ = ["AOTModel"]

_c_p = ctypes.c_void_p
_i64 = ctypes.c_int64


def _shim() -> ctypes.CDLL:
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    for name in ("libtorch_cpu.so", "libtorch.so"):
        cand = glob.glob(os.path.join(lib_dir, name))
        if cand:
            lib = ctypes.CDLL(cand[0], mode=ctypes.RTLD_GLOBAL)
            if hasattr(lib, "aoti_torch_create_tensor_from_blob"):
                return lib
    raise RuntimeError("aoti_torch shim not found in libtorch")


_DT = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16",
       torch.int64: "int64", torch.int32: "int32", torch.bool: "bool", torch.float64: "float64"}
_TYPESTR = {torch.float32: "<f4", torch.float16: "<f2", torch.float64: "<f8", torch.int64: "<i8",
            torch.int32: "<i4", torch.bool: "|b1"}


class _CAI:
    """Minimal __cuda_array_interface__ holder for a raw device pointer."""

    def __init__(self, ptr: int, shape, strides_elems, dtype: torch.dtype) -> None:
        itemsize = torch.empty((), dtype=dtype).element_size()
        self.__cuda_array_interface__ = dict(
            shape=tuple(int(s) for s in shape), typestr=_TYPESTR[dtype], data=(int(ptr), False), version=2,
            strides=tuple(int(s) * itemsize for s in strides_elems))


class AOTModel:
    """Load `so_path` (+ its kernel directory) and run it on CUDA tensors.

    `out_dtypes` / `out_ndims` describe the outputs (torch 2.2's shim exposes neither the dtype nor the rank).
    """

    def __init__(self, so_path: str, cubin_dir: Optional[str], out_dtypes: Sequence[torch.dtype],
                 out_ndims: Sequence[int], device_index: int = 0) -> None:
        self.shim = _shim()
        self.lib = ctypes.CDLL(os.path.abspath(so_path), mode=ctypes.RTLD_LOCAL)
        s, L = self.shim, self.lib
        s.aoti_torch_create_tensor_from_blob.argtypes = [_c_p, _i64, ctypes.POINTER(_i64), ctypes.POINTER(_i64), _i64,
                                                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(_c_p)]
        s.aoti_torch_create_tensor_from_blob.restype = ctypes.c_int32
        s.aoti_torch_get_data_ptr.argtypes = [_c_p, ctypes.POINTER(_c_p)]
        s.aoti_torch_get_sizes.argtypes = [_c_p, ctypes.POINTER(ctypes.POINTER(_i64))]
        s.aoti_torch_get_strides.argtypes = [_c_p, ctypes.POINTER(ctypes.POINTER(_i64))]
        s.aoti_torch_delete_tensor_object.argtypes = [_c_p]
        for fn in (s.aoti_torch_get_data_ptr, s.aoti_torch_get_sizes, s.aoti_torch_get_strides,
                   s.aoti_torch_delete_tensor_object):
            fn.restype = ctypes.c_int32
        self.dev_cuda = s.aoti_torch_device_type_cuda()
        self.dtype_code = {t: getattr(s, f"aoti_torch_dtype_{n}")() for t, n in _DT.items()}
        L.AOTInductorModelContainerCreate.argtypes = [ctypes.POINTER(_c_p), ctypes.c_size_t, ctypes.c_bool, ctypes.c_char_p]
        L.AOTInductorModelContainerRun.argtypes = [_c_p, ctypes.POINTER(_c_p), ctypes.c_size_t, ctypes.POINTER(_c_p),
                                                   ctypes.c_size_t, _c_p, _c_p]
        L.AOTInductorModelContainerGetNumInputs.argtypes = [_c_p, ctypes.POINTER(ctypes.c_size_t)]
        L.AOTInductorModelContainerGetNumOutputs.argtypes = [_c_p, ctypes.POINTER(ctypes.c_size_t)]
        for fn in (L.AOTInductorModelContainerCreate, L.AOTInductorModelContainerRun,
                   L.AOTInductorModelContainerGetNumInputs, L.AOTInductorModelContainerGetNumOutputs):
            fn.restype = ctypes.c_int32
        self.device_index = int(device_index)
        self.handle = _c_p()
        with torch.cuda.device(self.device_index):
            rc = L.AOTInductorModelContainerCreate(ctypes.byref(self.handle), 1, False,
                                                   None if cubin_dir is None else os.path.abspath(cubin_dir).encode())
        if rc != 0:
            raise RuntimeError(f"AOTInductorModelContainerCreate failed ({rc})")
        n_in, n_out = ctypes.c_size_t(), ctypes.c_size_t()
        L.AOTInductorModelContainerGetNumInputs(self.handle, ctypes.byref(n_in))
        L.AOTInductorModelContainerGetNumOutputs(self.handle, ctypes.byref(n_out))
        self.n_in, self.n_out = int(n_in.value), int(n_out.value)
        self.out_dtypes = list(out_dtypes)
        self.out_ndims = [int(n) for n in out_ndims]
        self._pinned = {}
        if len(self.out_dtypes) != self.n_out:
            raise ValueError(f"model has {self.n_out} outputs, got {len(self.out_dtypes)} dtypes")

    def _handle(self, t: torch.Tensor) -> _c_p:
        nd = t.dim()
        sizes = (_i64 * nd)(*t.shape)
        strides = (_i64 * nd)(*t.stride())
        h = _c_p()
        rc = self.shim.aoti_torch_create_tensor_from_blob(
            _c_p(t.data_ptr()), nd, sizes, strides, 0, self.dtype_code[t.dtype], self.dev_cuda,
            self.device_index, ctypes.byref(h))
        if rc != 0:
            raise RuntimeError(f"create_tensor_from_blob failed ({rc})")
        return h

    def __call__(self, *inputs: torch.Tensor, host: bool = False) -> List[torch.Tensor]:
        """Outputs as owned CUDA tensors, or (host=True) as CPU tensors copied straight from the
        artefact's buffers into cached pinned memory (no intermediate device clone)."""
        if len(inputs) != self.n_in:
            raise ValueError(f"expected {self.n_in} inputs")
        in_arr = (_c_p * self.n_in)(*[self._handle(t) for t in inputs])     # stolen by Run
        out_arr = (_c_p * self.n_out)()
        stream = _c_p(torch.cuda.current_stream().cuda_stream)
        rc = self.lib.AOTInductorModelContainerRun(self.handle, in_arr, self.n_in, out_arr, self.n_out, stream, None)
        if rc != 0:
            raise RuntimeError(f"AOTInductorModelContainerRun failed ({rc})")
        outs, pending = [], []
        for i in range(self.n_out):
            h = _c_p(out_arr[i])
            ptr, nd = _c_p(), self.out_ndims[i]
            sz, st = ctypes.POINTER(_i64)(), ctypes.POINTER(_i64)()
            self.shim.aoti_torch_get_data_ptr(h, ctypes.byref(ptr))
            self.shim.aoti_torch_get_sizes(h, ctypes.byref(sz))
            self.shim.aoti_torch_get_strides(h, ctypes.byref(st))
            shape = [sz[k] for k in range(nd)]
            strides = [st[k] for k in range(nd)]
            view = torch.as_tensor(_CAI(ptr.value or 0, shape, strides, self.out_dtypes[i]),
                                   device=f"cuda:{self.device_index}")
            if host:
                key = (i, tuple(shape), self.out_dtypes[i])
                buf = self._pinned.get(key)
                if buf is None:
                    buf = torch.empty(shape, dtype=self.out_dtypes[i], pin_memory=True)
                    self._pinned[key] = buf
                buf.copy_(view, non_blocking=True)
                outs.append(buf)
                pending.append(h)
            else:
                outs.append(view.clone())                  # own the memory before the handle is released
                self.shim.aoti_torch_delete_tensor_object(h)
        if host:
            torch.cuda.current_stream().synchronize()      # copies done; the handles' memory may go now
            for h in pending:
                self.shim.aoti_torch_delete_tensor_object(h)
        return outs
