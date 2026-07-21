"""Helion RMS norm forward (from the Helion rms_norm example)."""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.aot_kernel()
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        xt = x[tile_m, :].to(torch.float32)
        inv = torch.rsqrt(torch.mean(xt * xt, dim=-1) + eps)
        out[tile_m, :] = (xt * inv[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    return out
