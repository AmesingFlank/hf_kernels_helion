"""Helion gated-activation kernels: ``act(x[..., :d]) * x[..., d:]``.

Matches the API of ``kernels-community/activation`` (silu_and_mul, gelu_and_mul,
gelu_tanh_and_mul), which write the result into a preallocated ``out`` whose
last dim is half of the input's.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.experimental
import helion.language as hl

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


@helion.experimental.aot_kernel()
def _silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """out = silu(x[..., :d]) * x[..., d:], flattened to [num_rows, d]."""
    n_rows, d = out.size()
    for tile_r, tile_d in hl.tile([n_rows, d]):
        a = x[tile_r, tile_d].to(torch.float32)
        b = x[tile_r, tile_d.index + d]
        silu = a * torch.sigmoid(a)
        out[tile_r, tile_d] = (silu.to(b.dtype) * b).to(out.dtype)


@helion.experimental.aot_kernel()
def _gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """out = gelu(x[..., :d]) * x[..., d:] (exact erf gelu)."""
    n_rows, d = out.size()
    for tile_r, tile_d in hl.tile([n_rows, d]):
        a = x[tile_r, tile_d].to(torch.float32)
        b = x[tile_r, tile_d.index + d]
        gelu = a * 0.5 * (1.0 + torch.erf(a * 0.7071067811865476))
        out[tile_r, tile_d] = (gelu.to(b.dtype) * b).to(out.dtype)


@helion.experimental.aot_kernel()
def _gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    """out = gelu_tanh(x[..., :d]) * x[..., d:] (tanh approximation)."""
    n_rows, d = out.size()
    for tile_r, tile_d in hl.tile([n_rows, d]):
        a = x[tile_r, tile_d].to(torch.float32)
        b = x[tile_r, tile_d.index + d]
        inner = _SQRT_2_OVER_PI * (a + 0.044715 * a * a * a)
        gelu = a * 0.5 * (1.0 + torch.tanh(inner))
        out[tile_r, tile_d] = (gelu.to(b.dtype) * b).to(out.dtype)
