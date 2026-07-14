"""Helion multi-scale deformable attention (single level), matching the core of
deformable-detr's ms_deform_attn.

For each query/head, sample the value map at P continuous locations via bilinear
interpolation and combine with attention weights:
    out[b,q,h,:] = Σ_p attn_weight[b,q,h,p] · bilinear(value[b,:,h,:], loc[b,q,h,p])
value is a single feature level of grid (Hgrid, Wgrid), flattened to
Nkv=Hgrid*Wgrid. Tiles over queries; grids over (batch, head) so the bilinear
gather index stays 1-D (a Helion requirement for data-dependent gathers).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _deform_attn(
    value: torch.Tensor, sampling_loc: torch.Tensor, attn_weight: torch.Tensor,
    Hgrid: int, Wgrid: int, Dh: hl.constexpr, P: hl.constexpr, H: hl.constexpr,
) -> torch.Tensor:
    B, Nkv, _, _ = value.shape
    _, Nq, _, _, _ = sampling_loc.shape
    out = torch.empty([B, Nq, H, Dh], dtype=torch.float32, device=value.device)
    for tb in hl.grid(B):
        for th in hl.grid(H):
            for tq in hl.tile(Nq):
                acc = hl.zeros([tq, Dh], dtype=torch.float32)
                for p in range(P):
                    x = sampling_loc[tb, tq, th, p, 0].to(torch.float32) * Wgrid - 0.5
                    y = sampling_loc[tb, tq, th, p, 1].to(torch.float32) * Hgrid - 0.5
                    x0 = torch.floor(x); y0 = torch.floor(y)
                    wx = x - x0; wy = y - y0
                    x0i = x0.to(torch.int32); y0i = y0.to(torch.int32)
                    aw = attn_weight[tb, tq, th, p].to(torch.float32)
                    yy = y0i.clamp(0, Hgrid - 1); xx = x0i.clamp(0, Wgrid - 1)
                    inb = ((y0i >= 0) & (y0i < Hgrid) & (x0i >= 0) & (x0i < Wgrid)).to(torch.float32)
                    acc = acc + (aw * (1 - wy) * (1 - wx) * inb)[:, None] * value[tb, yy * Wgrid + xx, th, :]
                    yy = y0i.clamp(0, Hgrid - 1); xx = (x0i + 1).clamp(0, Wgrid - 1)
                    inb = ((y0i >= 0) & (y0i < Hgrid) & (x0i + 1 >= 0) & (x0i + 1 < Wgrid)).to(torch.float32)
                    acc = acc + (aw * (1 - wy) * wx * inb)[:, None] * value[tb, yy * Wgrid + xx, th, :]
                    yy = (y0i + 1).clamp(0, Hgrid - 1); xx = x0i.clamp(0, Wgrid - 1)
                    inb = ((y0i + 1 >= 0) & (y0i + 1 < Hgrid) & (x0i >= 0) & (x0i < Wgrid)).to(torch.float32)
                    acc = acc + (aw * wy * (1 - wx) * inb)[:, None] * value[tb, yy * Wgrid + xx, th, :]
                    yy = (y0i + 1).clamp(0, Hgrid - 1); xx = (x0i + 1).clamp(0, Wgrid - 1)
                    inb = ((y0i + 1 >= 0) & (y0i + 1 < Hgrid) & (x0i + 1 >= 0) & (x0i + 1 < Wgrid)).to(torch.float32)
                    acc = acc + (aw * wy * wx * inb)[:, None] * value[tb, yy * Wgrid + xx, th, :]
                out[tb, tq, th, :] = acc
    return out
