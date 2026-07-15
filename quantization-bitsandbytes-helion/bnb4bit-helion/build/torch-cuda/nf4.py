"""Helion 4-bit (NF4) weight-only GEMM, matching bitsandbytes' gemm_4bit_forward.

bitsandbytes stores a 4-bit-quantized weight matrix as packed uint8 (two 4-bit
codes per byte) plus a per-block ``absmax`` scale (one fp value per ``blocksize``
weight elements). Dequantization is:

    w = NF4_CODE[code] * absmax[block_of(code)]

then ``out = input @ w`` (weight is [N, K] row-major of an [N,K] weight, packed
along K). We fold the LUT + blockwise scale into the K-loop of a tiled GEMM.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# Standard NF4 codebook (QLoRA / bitsandbytes), 16 entries.
NF4_CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


@helion.kernel(static_shapes=False)
def _nf4_gemm(
    inp: torch.Tensor,          # [M, K] bf16 activations
    w_packed: torch.Tensor,     # [N, K//2] uint8, two NF4 codes per byte (along K)
    absmax: torch.Tensor,       # [N, K//blocksize] fp32 per-block scale
    code: torch.Tensor,         # [16] fp32 NF4 codebook
    blocksize: int,
) -> torch.Tensor:
    M, K = inp.shape
    N, Kp = w_packed.shape
    out = torch.empty([M, N], dtype=torch.float32, device=inp.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_kp in hl.tile(Kp):
            b = w_packed[tile_n, tile_kp].to(torch.int32)   # [n, kp]
            lo = b & 0xF
            hi = (b >> 4) & 0xF
            # LUT gather: code[lo], code[hi]
            w_lo = code[lo]
            w_hi = code[hi]
            # block index for scaling: full K position // blocksize.
            # packed col kp -> unpacked cols 2*kp (lo) and 2*kp+1 (hi).
            k_lo = tile_kp.index * 2
            k_hi = tile_kp.index * 2 + 1
            amax_lo = absmax[tile_n, k_lo // blocksize]
            amax_hi = absmax[tile_n, k_hi // blocksize]
            w_lo = w_lo * amax_lo
            w_hi = w_hi * amax_hi
            # interleave lo/hi back to full K order along the contraction dim
            w_full = torch.stack([w_lo, w_hi], dim=2).reshape(
                tile_n.block_size, tile_kp.block_size * 2
            )  # [n, 2*kp]
            a = inp[tile_m, (tile_kp.begin * 2):(tile_kp.begin * 2 + tile_kp.block_size * 2)].to(torch.float32)
            acc = hl.dot(a, w_full.transpose(0, 1), acc=acc)  # [m,2kp]x[2kp,n]
        out[tile_m, tile_n] = acc
    return out
