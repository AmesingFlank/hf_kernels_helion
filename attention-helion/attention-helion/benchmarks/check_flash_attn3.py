"""Isolated flash-attn3 probe (runs in its own process; the CUDA failure it
triggers poisons the context, so it must not share a process with real work)."""
from __future__ import annotations
import torch
import kernels

fa3 = kernels.get_kernel("kernels-community/flash-attn3", version=1)
print("loaded:", hasattr(fa3, "flash_attn_func"))
B, S, H, D = 2, 128, 8, 64
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
try:
    out = fa3.flash_attn_func(q, q, q, causal=False)
    torch.cuda.synchronize()
    print("RAN OK, shape:", tuple((out[0] if isinstance(out, tuple) else out).shape))
except Exception as e:
    print("FAILED:", type(e).__name__)
    print("MSG:", str(e).splitlines()[0][:120])
