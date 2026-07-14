# sage-attention-helion

A [Helion](https://github.com/pytorch/helion) implementation of **SageAttention2**
— INT8-quantized flash attention — packaged as a
[`kernels`](https://github.com/huggingface/kernels) kernel.

This is a faithful port of the algorithm in
[`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention)'s
`qk_int8_pv_fp16` path (the `_qattn_sm80` CUDA kernel that `sageattn()`
dispatches to on Ampere/Blackwell GPUs). SageAttention does every step in CUDA;
here the numerics are written in Helion, and PyTorch is reused only for the
smooth-K mean (exactly as the reference computes `km` in Python).

## Algorithm

For inputs Q, K, V of shape `(batch, heads, seq, 128)`:

1. **smooth-K** — subtract the per-`(batch, head)` sequence mean of K,
   `k ← k − mean_seq(k)`. Mathematically a no-op inside softmax (it shifts every
   logit in a row by the same constant), but it dramatically shrinks the INT8
   quantization error of K. This is the core idea of SageAttention.
2. **INT8 quantization** of Q and K at *per-token* granularity
   (`scale = rowmax(|x|) / 127`).
3. **INT8 QK<sup>T</sup>** via IMMA (int8 inputs, int32 accumulate), then
   dequantize by `q_scale · k_scale` and fold in the softmax scale.
4. **online (flash) softmax** in base-2, streaming over K/V tiles.
5. **P @ V** with V in fp16 and FP32 accumulation.

Head dim is fixed at **128** (the SageAttention2 head dim).

## Usage

```python
from kernels import get_kernel
import torch

sage = get_kernel("kernels-community/sage-attention-helion")

q = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.bfloat16)

# HND layout (B, H, S, D); also supports tensor_layout="NHD" and is_causal=True
out = sage.sageattn(q, k, v, tensor_layout="HND", is_causal=False)

# flash-attn-style wrapper: (B, S, H, D) layout
out = sage.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
```

## Accuracy & performance

Benchmarked on an **NVIDIA B200** against `thu-ml/SageAttention` 2.2.0's
INT8-QK/FP16-PV CUDA kernel, **built from source for sm_100** (the prebuilt
`flashrt/sageattention2-blackwell` kernel targets only consumer Blackwell
sm_120/sm_121 and cannot run on the datacenter B200). Both are INT8-quantized
attention; numerically both track full-precision SDPA to ~1% relative error:

| vs SDPA (relative error) | value |
|---|---|
| this Helion kernel | ~0.0097 |
| SageAttention2 CUDA reference | ~0.0115 |
| this kernel vs the CUDA reference | ~0.013 (INT8 noise floor) |

See `../../benchmark_results_triton.md` for the full head-to-head speed table.

## Upstream

Algorithm from [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention)
(Apache-2.0). Built with
[kernel-builder](https://github.com/huggingface/kernel-builder).
