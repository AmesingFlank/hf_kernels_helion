---
library_name: kernels
license: apache-2.0
tags:
- kernels
- helion
- attention
---

# attention (Helion)

A scaled-dot-product / flash-attention kernel written in
[Helion](https://github.com/pytorch/helion) and packaged for the
[`kernels` library](https://github.com/huggingface/kernels). It is a
Triton-backend kernel that ships **pre-tuned configs** (via
`helion.aot_kernel`), so the first call selects a known-good configuration and
compiles it — no autotuning search at run time.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
import torch
from kernels import get_kernel

# trust_remote_code=True is required: this is a third-party (non
# kernels-community) kernel that runs Python at load time.
kernel = get_kernel("HelionDSL/attention", version=1, trust_remote_code=True)

# flash-attn layout: (batch, seq, heads, head_dim), fp16/bf16
q = torch.randn(2, 512, 8, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = kernel.flash_attn_func(q, k, v, causal=False)
```

`get_kernel` requires a pinned version or revision — use `version=1` (the stable
kernel-API major, backed by the repo's `v1` branch) or `revision="main"` /
`revision="<commit>"` for an explicit Hub revision.

## Available functions

- `flash_attn_func(q, k, v, softmax_scale=None, causal=False)` — flash-attn
  compatible entry point. Inputs and output are `(batch, seq, heads, head_dim)`.
- `attention(q, k, v, causal=False, out=None)` — SDPA-layout entry point. Inputs
  and output are `(batch, heads, seq, head_dim)`.

Both accept `torch.float16` / `torch.bfloat16` and support `causal=True`.

## Available layers

- `Attention` — an `nn.Module` wrapper (`layers.Attention(causal=False)`) whose
  `forward(q, k, v)` calls the kernel.

## Notes

- **Backend:** Helion → Triton. Correctness is verified against
  `torch.nn.functional.scaled_dot_product_attention`.
- **Pre-tuned:** ships `_helion_aot_attention_cuda_sm100.py` and
  `_helion_aot_attention_cuda_sm90.py`; the right one is selected for your GPU
  (with compute-capability fallback). On a shape/GPU without a shipped config,
  the kernel falls back to normal Helion autotuning.
