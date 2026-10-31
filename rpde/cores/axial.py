"""Core A: axial spatiotemporal transformer.

Full-resolution transformer over the 40x32x64 volume with attention factorised
along time (T=40), streamwise x (W=64) and cross-stream y (H=32). Pre-LN,
residual, adaLN-Zero (FiLM) gating with zero-init gates, so every block is the
identity at initialisation and the model equals the advective baseline.

Submission-safe: torch + rpde only (no einops).
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint as _ckpt
import torch.nn as nn
import torch.nn.functional as F

from rpde.constants import H as H_DEF, T_VOL, W as W_DEF
from rpde.scaffold import CondEmbed, FiLM, drop_path, modulate

__all__ = ["AxialCore", "enable_fast_axial", "fuse_qkv"]


class MHSA(nn.Module):
    """Multi-head self-attention over the last-but-one axis of (N, L, d)."""

    def __init__(self, d: int, heads: int = 8, attn_drop: float = 0.0) -> None:
        super().__init__()
        if d % heads:
            raise ValueError(f"d={d} not divisible by heads={heads}")
        self.h = heads
        self.dh = d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.attn_drop = float(attn_drop)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor,
                attn_bias: torch.Tensor = None) -> torch.Tensor:
        N, L, d = x.shape
        qkv = self.qkv(x).view(N, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # (N, h, L, dh)
        if attn_bias is None:                                  # default path
            o = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        else:                                                  # rel_time_bias
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias.to(q.dtype),
                dropout_p=self.attn_drop if self.training else 0.0)
        o = o.transpose(1, 2).reshape(N, L, d)
        return self.proj(o)


def _fold(x: torch.Tensor, axis: str):
    """(B,T,H,W,d) -> (N, L, d) with `axis` as the sequence, plus an unfold fn."""
    B, T, Hh, Ww, d = x.shape
    if axis == "T":
        y = x.permute(0, 2, 3, 1, 4).reshape(B * Hh * Ww, T, d)

        def un(z):
            return z.view(B, Hh, Ww, T, d).permute(0, 3, 1, 2, 4)
    elif axis == "W":
        y = x.reshape(B * T * Hh, Ww, d)

        def un(z):
            return z.view(B, T, Hh, Ww, d)
    elif axis == "S":                     # merged spatial axis (full attention)
        y = x.reshape(B * T, Hh * Ww, d)

        def un(z):
            return z.view(B, T, Hh, Ww, d)
    elif axis == "H":
        y = x.permute(0, 1, 3, 2, 4).reshape(B * T * Ww, Hh, d)

        def un(z):
            return z.view(B, T, Ww, Hh, d).permute(0, 1, 3, 2, 4)
    else:
        raise ValueError(axis)
    return y, un


class SwiGLU(nn.Module):
    """Gated MLP: W3( silu(x W1) * (x W2) ), W1/W2 fused into one Linear."""

    def __init__(self, d: int, h: int, drop: float = 0.0) -> None:
        super().__init__()
        self.w12 = nn.Linear(d, 2 * h)
        self.w3 = nn.Linear(h, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(self.drop(F.silu(a) * b)))


class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction: x * rsqrt(mean(x^2) + eps) * weight.

    The profile puts "block glue" (norms + FiLM + residual gates) at 25 % of the compiled model, the largest
    single line. RMSNorm drops the mean pass and the bias, so it is roughly a third cheaper than LayerNorm and
    is the standard drop-in in modern transformers. `norm="layer"` keeps the shipped layer bit-identical.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = float(eps)
        self.bias = None                      # modulate() adds FiLM's shift, so no bias is needed here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _norm(kind: str, d: int) -> nn.Module:
    if kind == "layer":
        return nn.LayerNorm(d)
    if kind == "rms":
        return RMSNorm(d)
    raise ValueError(f"norm must be 'layer' or 'rms', got {kind!r}")


class AxialBlock(nn.Module):
    """attn_t -> attn_x -> attn_y -> MLP, each pre-LN, residual, FiLM-gated.

    Two opt-in variants (both off by default, the default path is untouched):

    * ``spatial_attn="full"``: ``attn_x``/``attn_y`` are replaced by ONE
      attention ``attn_s`` over all ``Hp*Wp`` positions of a frame, so the block
      is ``attn_t -> attn_s -> MLP`` (3 sub-layers instead of 4).
    * ``rel_time_bias=True``: a learned relative-position bias table
      (``heads x (2*T_core-1)``, zero-init) is added to the *temporal* attention
      logits through SDPA's additive ``attn_mask``.
    """

    N_SUB = 4

    def __init__(self, d: int, cond_dim: int, heads: int = 8, mlp_ratio: int = 4,
                 drop_path: float = 0.0, mlp_drop: float = 0.1,
                 attn_drop: float = 0.0, spatial_attn: str = "axial",
                 rel_time_bias: bool = False, t_core: int = T_VOL, mlp_type: str = "gelu",
                 norm: str = "layer", film_shared: bool = False,
                 rel_pos_bias: bool = False, n_x: int = W_DEF, n_y: int = H_DEF) -> None:
        super().__init__()
        self.dp = float(drop_path)
        self.spatial_attn = str(spatial_attn).lower()
        if self.spatial_attn not in ("axial", "full"):
            raise ValueError(f"spatial_attn must be 'axial' or 'full', got {spatial_attn!r}")
        self.full_spatial = self.spatial_attn == "full"
        # one time token (patch_t == T_core, "full collapse"): attention over a length-1 axis is the identity
        # apart from its projections, so the sub-layer is dropped entirely and the frames mix in the channels.
        self.no_time = int(t_core) == 1
        self.n_sub = (3 if self.full_spatial else self.N_SUB) - (1 if self.no_time else 0)
        self.attn_t = None if self.no_time else MHSA(d, heads, attn_drop)
        if self.full_spatial:
            self.attn_s = MHSA(d, heads, attn_drop)
        else:
            self.attn_x = MHSA(d, heads, attn_drop)
            self.attn_y = MHSA(d, heads, attn_drop)
        if mlp_type == "swiglu":
            # gated MLP at the SAME parameter count as the GELU MLP: 3*d*h = 2*d*(ratio*d) -> h = 2/3 * ratio * d (x32)
            h = max(32, int(round(2.0 / 3.0 * mlp_ratio * d / 32.0)) * 32)
            self.mlp = SwiGLU(d, h, mlp_drop)
        elif mlp_type == "gelu":
            self.mlp = nn.Sequential(
                nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(mlp_drop),
                nn.Linear(mlp_ratio * d, d), nn.Dropout(mlp_drop),
            )
        else:
            raise ValueError(f"mlp_type must be 'gelu' or 'swiglu', got {mlp_type!r}")
        self.norms = nn.ModuleList([_norm(norm, d) for _ in range(self.n_sub)])
        # film_shared: ONE (scale, shift, gate) triple per block instead of one per sub-layer -- three quarters
        # less FiLM work, and the conditioning signal is the same for every sub-layer of a block anyway.
        self.film_shared = bool(film_shared)
        self.film = FiLM(cond_dim, d, n_sublayers=1 if self.film_shared else self.n_sub)
        # relative time bias (variant 2): zero-init, so at initialisation the
        # temporal attention is bit-identical to the default path
        self.rel_time_bias = bool(rel_time_bias)
        if self.rel_time_bias:
            self.t_core = int(t_core)
            self.rel_t = nn.Parameter(torch.zeros(heads, 2 * self.t_core - 1))
            idx = torch.arange(self.t_core)
            self.register_buffer(
                "rel_idx", (idx.view(1, -1) - idx.view(-1, 1) + self.t_core - 1).long(),
                persistent=False)
        # relative SPATIAL bias: the x/y analogue of rel_time_bias.  The absolute pos_x/pos_y embeddings make
        # the core position-dependent, so the same wake structure at two x positions is processed by different
        # parameters.  A relative bias restores shift equivariance along the attended axis.  Zero-init, so the
        # path is bit-identical to the default at initialisation.
        self.rel_pos_bias = bool(rel_pos_bias)
        if self.rel_pos_bias:
            for ax, n in (("x", int(n_x)), ("y", int(n_y))):
                setattr(self, f"rel_{ax}", nn.Parameter(torch.zeros(heads, 2 * n - 1)))
                idx = torch.arange(n)
                self.register_buffer(f"rel_idx_{ax}", (idx.view(1, -1) - idx.view(-1, 1) + n - 1).long(),
                                     persistent=False)
        # inference-only fused elementwise chain; see `enable_fast_axial`
        self.fast = False

    def _pos_bias(self, ax: str) -> torch.Tensor:
        b = getattr(self, f"rel_{ax}")[:, getattr(self, f"rel_idx_{ax}")]
        n = b.shape[-1]
        pad = (-n) % 8
        if pad:
            b = F.pad(b, (0, pad))[..., :n]
        return b.unsqueeze(0)

    def _time_bias(self) -> torch.Tensor:
        """(1, heads, T, T) additive bias, last dim contiguous and row-aligned.

        The memory-efficient SDPA kernel wants the bias' last dimension
        contiguous with an aligned row stride, so the table is padded to a
        multiple of 8 columns and narrowed back (a view, still differentiable).
        """
        b = self.rel_t[:, self.rel_idx]                        # (h, T, T)
        T = b.shape[-1]
        pad = (-T) % 8
        if pad:
            b = F.pad(b, (0, pad))[..., :T]
        return b.unsqueeze(0)

    # -- fused elementwise helpers (inference only) -----------------------
    @staticmethod
    def _bview(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(B, d) -> (B, 1, ..., 1, d) broadcastable against `x`."""
        return t.view((t.shape[0],) + (1,) * (x.dim() - 2) + (t.shape[-1],))

    def _mod(self, norm, x, scale, shift):
        """`modulate(norm(x), scale, shift)` in two passes instead of four.

        The LayerNorm affine (per-channel) is folded into the FiLM scale/shift
        (per-sample, per-channel), so the (B,T,H,W,d) activation is normalised
        once and then written once by a single fused multiply-add, instead of
        LN-affine + multiply + add.  Identical up to fp reassociation.
        """
        s1 = 1.0 + scale
        w = self._bview(norm.weight * s1, x)
        # RMSNorm has no bias (FiLM supplies the shift), so the fold is just the FiLM shift there
        b = self._bview(shift if norm.bias is None else norm.bias * s1 + shift, x)
        if norm.bias is None:                                   # RMSNorm: no mean pass, no normalized_shape
            xh = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm.eps).to(x.dtype)
        else:
            xh = F.layer_norm(x, norm.normalized_shape, None, None, norm.eps)
        return torch.addcmul(b, xh, w)

    def _attn_sub(self, x, attn, norm, axis, scale, shift, gate, bias=None):
        if self.fast and not self.training:
            h = self._mod(norm, x, scale, shift)
            y, un = _fold(h, axis)
            y = un(attn(y) if bias is None else attn(y, bias))
            return torch.addcmul(x, y, self._bview(gate, x))
        h = modulate(norm(x), scale, shift)
        y, un = _fold(h, axis)
        y = attn(y) if bias is None else attn(y, bias)
        y = un(y)
        g = gate.view(gate.shape[0], 1, 1, 1, gate.shape[-1])
        return x + drop_path(g * y, self.dp, self.training)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s, b, g = self.film(cond)                              # (B, n_sub, d) each
        if self.film_shared:                                   # broadcast the single triple to every sub-layer
            s, b, g = (v.expand(-1, self.n_sub, -1) for v in (s, b, g))
        tb = self._time_bias() if self.rel_time_bias else None
        k = 0
        if not self.no_time:
            x = self._attn_sub(x, self.attn_t, self.norms[0], "T", s[:, 0], b[:, 0], g[:, 0], tb)
            k = 1
        if self.full_spatial:
            x = self._attn_sub(x, self.attn_s, self.norms[k], "S", s[:, k], b[:, k], g[:, k])
            j = k + 1
        else:
            xb = self._pos_bias("x") if self.rel_pos_bias else None
            yb = self._pos_bias("y") if self.rel_pos_bias else None
            x = self._attn_sub(x, self.attn_x, self.norms[k], "W", s[:, k], b[:, k], g[:, k], xb)
            x = self._attn_sub(x, self.attn_y, self.norms[k + 1], "H", s[:, k + 1], b[:, k + 1], g[:, k + 1], yb)
            j = k + 2
        if self.fast and not self.training:
            h = self.mlp(self._mod(self.norms[j], x, s[:, j], b[:, j]))
            return torch.addcmul(x, h, self._bview(g[:, j], x))
        h = self.mlp(modulate(self.norms[j](x), s[:, j], b[:, j]))
        gg = g[:, j].view(g.shape[0], 1, 1, 1, g.shape[-1])
        return x + drop_path(gg * h, self.dp, self.training)


class RefineBlock(nn.Module):
    """Full-resolution residual refinement on (B, d, T, H, W): 1x1 down to `ch`, two per-frame 3x3 convs, 1x1 up.
    The up-projection is zero-initialised, so the block is the identity at initialisation."""

    def __init__(self, d: int, ch: int = 32) -> None:
        super().__init__()
        self.down = nn.Conv3d(d, ch, kernel_size=1)
        self.c1 = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.c2 = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.up = nn.Conv3d(ch, d, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.down(x))
        h = F.gelu(self.c1(h))
        h = F.gelu(self.c2(h))
        return x + self.up(h)


class HiresSkip(nn.Module):
    """Full-resolution bypass: (B, C_in, T, H, W) -> (B, d, T, H, W); per-frame 3x3 convs, zero-init 1x1 output."""

    def __init__(self, c_in: int, ch: int, d: int, layers: int = 1) -> None:
        super().__init__()
        convs, c = [], c_in
        for _ in range(max(1, layers)):
            convs.append(nn.Conv3d(c, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            c = ch
        self.convs = nn.ModuleList(convs)
        self.out = nn.Conv3d(ch, d, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        h = v
        for c in self.convs:
            h = F.gelu(c(h))
        return self.out(h)


class AxialCore(nn.Module):
    out_channels: int

    def __init__(self, in_ch: int = 8, cond_dim: int = 8, d: int = 128, heads: int = 8,
                 depth: int = 8, mlp_ratio: int = 4, drop_path: float = 0.1,
                 mlp_drop: float = 0.1, attn_drop: float = 0.0,
                 cond_hidden: int = 64, T: int = T_VOL, H: int = H_DEF, W: int = W_DEF,
                 grad_checkpoint: bool = False, patch_hw: int = 1, patch_t: int = 1, t2c: bool = False,
                 t_in: int = 20, spatial_attn: str = "axial",
                 rel_time_bias: bool = False, query_frames: bool = False,
                 patch_h: int = None, patch_w: int = None,
                 refine: int = 0, refine_ch: int = 32,
                 hires_skip: int = 0, hires_skip_layers: int = 1,
                 mlp_type: str = "gelu", norm: str = "layer", film_shared: bool = False,
                 rel_pos_bias: bool = False,
                 overlap_stem: bool = False, overlap_unpatch: bool = False,
                 **_ignored) -> None:
        super().__init__()
        self.out_channels = int(d)
        self.grad_checkpoint = bool(grad_checkpoint)
        # --- token-reduction options (both default off; identical to the original when off)
        # patch_hw: 2x2 (or pxp) spatial patch embedding in the stem + a linear un-patch
        #           (ConvTranspose) after the blocks -> tokens / p^2, full-res features out.
        # t2c:      time-to-channel: the past frames' (u,v) become 2*t_in extra channels of
        #           every future frame and the core runs on the T - t_in future frames only;
        #           the output is zero-padded on the past frames so `Head` slicing is unchanged.
        # patch_h / patch_w: anisotropic patches (e.g. 1 x 2 = full cross-stream resolution); default = patch_hw
        self.patch_hw = int(patch_hw)
        self.patch_h = int(patch_h) if patch_h else self.patch_hw
        self.patch_w = int(patch_w) if patch_w else self.patch_hw
        self.t2c = bool(t2c)
        self.t_in = int(t_in)
        self.T_core = (T - self.t_in) if self.t2c else T
        # patch_t: fold `pt` consecutive FUTURE frames into one token (the time analogue of patch_hw) and unfold
        # them again after the blocks.  Tokens drop by pt, so every sub-layer gets cheaper, not just the time one.
        # pt == T_core is the full collapse: one token per spatial patch, all frames carried in the channels.
        self.patch_t = int(patch_t)
        if self.T_core % self.patch_t:
            raise ValueError(f"patch_t={self.patch_t} must divide T_core={self.T_core}")
        self.T_tok = self.T_core // self.patch_t
        self.Hp, self.Wp = H // self.patch_h, W // self.patch_w
        in_ch_eff = in_ch + (2 * self.t_in if self.t2c else 0)
        pt = self.patch_t
        if self.patch_h == 1 and self.patch_w == 1:
            self.stem = nn.Conv3d(in_ch_eff, d, kernel_size=(pt, 3, 3), stride=(pt, 1, 1), padding=(0, 1, 1))
            self.unpatch = (None if pt == 1 else
                            nn.ConvTranspose3d(d, d, kernel_size=(pt, 1, 1), stride=(pt, 1, 1)))
        else:
            ph, pw = self.patch_h, self.patch_w
            # overlap_stem: (p+1)x(p+1) window at stride p (neighbouring patches share one row/column of cells);
            # overlap_unpatch: (2p)x(2p) transposed conv at stride p (each output cell blends 2x2 neighbouring tokens).
            # Same token grid, so the trunk cost is unchanged. Both default off (original layers, bit-identical).
            if overlap_stem:
                self.stem = nn.Conv3d(in_ch_eff, d, kernel_size=(pt, ph + 1, pw + 1), stride=(pt, ph, pw),
                                      padding=(0, ph // 2, pw // 2))
            else:
                self.stem = nn.Conv3d(in_ch_eff, d, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
            if overlap_unpatch:
                self.unpatch = nn.ConvTranspose3d(d, d, kernel_size=(pt, 2 * ph, 2 * pw), stride=(pt, ph, pw),
                                                  padding=(0, ph // 2, pw // 2))
            else:
                self.unpatch = nn.ConvTranspose3d(d, d, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        # refine: full-resolution residual conv blocks after the un-patch (detail below the patch size).
        # Zero-initialised output projection -> identity at init; absent (no parameters) when refine == 0.
        self.refine = nn.ModuleList([RefineBlock(d, int(refine_ch)) for _ in range(int(refine))]) if refine else None
        # hires_skip=C: U-net-style full-resolution bypass around the patch bottleneck. The stem INPUT volume (full H x W)
        # goes through `hires_skip_layers` per-frame 3x3 convs with C channels and a zero-initialised 1x1 projection to d,
        # added to the un-patched features. Identity at init; absent (no parameters) when 0.
        self.hires_skip = HiresSkip(in_ch_eff, int(hires_skip), d, int(hires_skip_layers)) if hires_skip else None
        self.pos_t = nn.Parameter(torch.zeros(self.T_tok, d))
        self.pos_y = nn.Parameter(torch.zeros(self.Hp, d))
        self.pos_x = nn.Parameter(torch.zeros(self.Wp, d))
        for p in (self.pos_t, self.pos_y, self.pos_x):
            nn.init.normal_(p, std=0.02)
        # --- learned query frames (variant 3): a per-future-step embedding added
        # to the stem output.  Zero-initialised, so at init the core is exactly the
        # default one; the baseline stays an input (the stem input is unchanged).
        self.query_frames = bool(query_frames)
        if self.query_frames and self.patch_t != 1:
            raise ValueError("query_frames and patch_t > 1 are not supported together")
        self.T_fut = self.T_core if self.t2c else (T - self.t_in)
        if self.query_frames:
            self.query_emb = nn.Parameter(torch.zeros(self.T_fut, d))
        self.cond = CondEmbed(cond_dim, cond_hidden)
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            AxialBlock(d, self.cond.out_dim, heads=heads, mlp_ratio=mlp_ratio,
                       drop_path=rates[i], mlp_drop=mlp_drop, attn_drop=attn_drop,
                       spatial_attn=spatial_attn, rel_time_bias=rel_time_bias,
                       t_core=self.T_tok, mlp_type=mlp_type, norm=norm, film_shared=film_shared,
                       rel_pos_bias=rel_pos_bias, n_x=self.Wp, n_y=self.Hp)
            for i in range(depth)
        ])
        self.norm_out = _norm(norm, d)
        self.fast = False              # see `enable_fast_axial`
        self.channels_last_out = False # emit (B,T,H,W,d); `Head` must agree
        self._pos = None               # cached pos_t+pos_y+pos_x (fast path)

    def _pos_sum(self, x: torch.Tensor) -> torch.Tensor:
        """The three broadcast position embeddings, summed once and cached."""
        pos = self._pos
        if (pos is None or pos.dtype != x.dtype or pos.device != x.device
                or pos.shape[-1] != x.shape[-1]):
            d = x.shape[-1]
            pos = (self.pos_t.view(1, -1, 1, 1, d)
                   + self.pos_y.view(1, 1, -1, 1, d)
                   + self.pos_x.view(1, 1, 1, -1, d)).to(dtype=x.dtype)
            self._pos = pos
        return pos

    def forward(self, vol: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.t2c:
            B, _, T, H, W = vol.shape
            past = vol[:, :2, :self.t_in]                                  # (B,2,t_in,H,W)
            past_ch = past.reshape(B, 2 * self.t_in, 1, H, W).expand(-1, -1, T - self.t_in, -1, -1)
            vol = torch.cat([vol[:, :, self.t_in:], past_ch], dim=1)      # (B,in+2*t_in,T-t_in,H,W)
        vol_full = vol
        x = self.stem(vol)                                     # (B,d,Tc,Hp,Wp)
        x = x.permute(0, 2, 3, 4, 1)                           # (B,Tc,Hp,Wp,d)
        if self.fast and not self.training:
            x = x + self._pos_sum(x)
        else:
            x = (x + self.pos_t.view(1, -1, 1, 1, x.shape[-1])
                   + self.pos_y.view(1, 1, -1, 1, x.shape[-1])
                   + self.pos_x.view(1, 1, 1, -1, x.shape[-1]))
        if self.query_frames:
            q = self.query_emb.to(x.dtype)
            if self.T_fut != self.T_core:            # non-t2c: future frames only
                q = F.pad(q, (0, 0, self.T_core - self.T_fut, 0))
            x = x + q.view(1, -1, 1, 1, x.shape[-1])
        c = self.cond(cond)
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if use_ckpt:
                x = _ckpt.checkpoint(blk, x, c, use_reentrant=False)
            else:
                x = blk(x, c)
        x = self.norm_out(x)
        if (self.unpatch is None and not self.t2c and self.refine is None
                and self.hires_skip is None and self.patch_t == 1):       # original path
            if self.channels_last_out:
                return x                                       # (B,T,H,W,d)
            return x.permute(0, 4, 1, 2, 3).contiguous()       # (B,d,T,H,W)
        out = x.permute(0, 4, 1, 2, 3)                         # (B,d,Tc,Hp,Wp)
        if self.unpatch is not None:
            out = self.unpatch(out)                            # (B,d,Tc,H,W)
        if self.hires_skip is not None:
            out = out + self.hires_skip(vol_full).to(out.dtype)
        if self.refine is not None:
            for rb in self.refine:
                out = rb(out)
        if self.t2c:                                           # zero-pad the past frames
            B, d, Tc, H, W = out.shape
            out = torch.cat([out.new_zeros(B, d, self.t_in, H, W), out], dim=2)
        if self.channels_last_out:
            return out.permute(0, 2, 3, 4, 1).contiguous()     # (B,T,H,W,d)
        return out.contiguous()                                # (B,d,T,H,W)


# ---------------------------------------------------------------------------
# Inference-only, weight-preserving transforms
# ---------------------------------------------------------------------------
def enable_fast_axial(model: nn.Module, on: bool = True,
                      channels_last_head: bool = True) -> dict:
    """Switch every AxialBlock/AxialCore in `model` to the fused inference path.

    Same weights, same architecture; only the order in which the elementwise
    work is issued changes (fewer passes over the 42 MB activations):

    * LayerNorm affine folded into the FiLM modulate -> one `addcmul`;
    * residual `x + gate * y` -> one `addcmul`;
    * the three position embeddings summed once and cached;
    * `channels_last_head`: the core returns (B,T,H,W,d) and `Head` applies its
      1x1x1 conv as an `F.linear`, dropping the (B,d,T,H,W) `.contiguous()` copy.

    Differences vs the default path are floating-point reassociation only.
    The fused path is skipped automatically while `training` (drop-path).
    Returns a small report; call with `on=False` to restore the default path.
    """
    from rpde.scaffold import Head                    # local: avoids a cycle

    rep = dict(blocks=0, cores=0, heads=0)
    for m in model.modules():
        if isinstance(m, AxialBlock):
            m.fast = bool(on)
            rep["blocks"] += 1
        elif isinstance(m, AxialCore):
            m.fast = bool(on)
            m._pos = None
            m.channels_last_out = bool(on and channels_last_head)
            rep["cores"] += 1
    if rep["cores"]:
        for m in model.modules():
            if isinstance(m, Head):
                m.channels_last_in = bool(on and channels_last_head)
                rep["heads"] += 1
    return rep


def fuse_qkv(model: nn.Module) -> dict:
    """Merge separate q/k/v projections of every MHSA into one Linear (exact).

    `MHSA.__init__` already ships a single fused `qkv = Linear(d, 3d)`, and
    `FiLM` already ships a single `Linear(cond_dim, 3*d*n_sublayers)` covering
    scale/shift/gate of all four sub-layers of a block -- so on the current
    architecture there is nothing left to fuse and this reports
    `already_fused` only.  The merge branch below stays for the case a variant
    reintroduces separate `q`/`k`/`v` Linears: weights are concatenated along
    the output dimension, which is bit-exact for the three separate matmuls
    only up to the reduction order of the fused GEMM (fp noise).
    """
    rep = dict(fused=0, already_fused=0, film_already_fused=0)
    for m in model.modules():
        if isinstance(m, MHSA):
            if hasattr(m, "qkv") and isinstance(m.qkv, nn.Linear):
                rep["already_fused"] += 1
                continue
            q, k, v = (getattr(m, n, None) for n in ("q", "k", "v"))
            if not all(isinstance(t, nn.Linear) for t in (q, k, v)):
                continue
            d_in = q.in_features
            has_bias = q.bias is not None
            lin = nn.Linear(d_in, 3 * q.out_features, bias=has_bias,
                            device=q.weight.device, dtype=q.weight.dtype)
            with torch.no_grad():
                lin.weight.copy_(torch.cat([q.weight, k.weight, v.weight], 0))
                if has_bias:
                    lin.bias.copy_(torch.cat([q.bias, k.bias, v.bias], 0))
            m.qkv = lin
            for n in ("q", "k", "v"):
                delattr(m, n)
            rep["fused"] += 1
        elif isinstance(m, FiLM):
            rep["film_already_fused"] += 1
    return rep
