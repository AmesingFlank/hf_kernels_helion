"""Helion MoE expert MLP (gpt-oss / triton-kernels style): grouped GEMM + SwiGLU.

Tokens are routed and packed per expert (group_offsets). For each expert:
  h = SiLU(A @ Wg[e]) * (A @ Wu[e]) ; out = h @ Wd[e]
Here we implement the gate/up fused expert projection with SwiGLU, the MoE hot
path. A:[T,H] packed by expert; Wg,Wu:[E,H,I]; group_offsets:[E+1] -> out:[T,I].
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _moe_swiglu(A: torch.Tensor, Wg: torch.Tensor, Wu: torch.Tensor,
                group_offsets: torch.Tensor) -> torch.Tensor:
    T, H = A.shape
    E, _, I = Wg.shape
    out = torch.empty([T, I], dtype=A.dtype, device=A.device)
    for e in hl.grid(E):
        start = group_offsets[e]
        end = group_offsets[e + 1]
        M_g = end - start
        if M_g != 0:
            for tm, ti in hl.tile([M_g, I]):
                gate = hl.zeros([tm, ti], dtype=torch.float32)
                up = hl.zeros([tm, ti], dtype=torch.float32)
                for tk in hl.tile(H):
                    a = A[start + tm.index, tk]
                    gate = torch.addmm(gate, a, Wg[e, tk, ti])
                    up = torch.addmm(up, a, Wu[e, tk, ti])
                h = (gate * torch.sigmoid(gate)) * up
                out[start + tm.index, ti] = h.to(out.dtype)
    return out
