"""Helion depthwise causal 1D convolution, matching causal_conv1d_fn.

out[b, d, l] = sum_w x[b, d, l - (width-1) + w] * weight[d, w] + bias[d]
with optional SiLU activation. Layout x: (batch, dim, seqlen), weight: (dim, width).
This is the core forward used by Mamba / SSM blocks.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _causal_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, apply_silu: hl.constexpr
) -> torch.Tensor:
    B, D, L = x.shape
    width = hl.specialize(weight.shape[1])
    out = torch.empty_like(x)
    for tile_b, tile_d, tile_l in hl.tile([B, D, L]):
        acc = hl.zeros([tile_b, tile_d, tile_l], dtype=torch.float32)
        for w in range(width):
            src = tile_l.index - (width - 1) + w
            xv = torch.where(
                (src >= 0)[None, None, :],
                x[tile_b, tile_d, src.clamp(0, L - 1)].to(torch.float32),
                0.0,
            )
            wv = weight[tile_d, w].to(torch.float32)
            acc = acc + xv * wv[None, :, None]
        acc = acc + bias[tile_d].to(torch.float32)[None, :, None]
        if apply_silu:
            acc = acc * torch.sigmoid(acc)
        out[tile_b, tile_d, tile_l] = acc.to(out.dtype)
    return out
