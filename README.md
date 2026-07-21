# hf_kernels_helion

[Helion](https://github.com/pytorch/helion) ports of the most-downloaded
[`kernels-community`](https://huggingface.co/kernels-community) kernels, each
packaged as a Hugging Face [`kernels`](https://github.com/huggingface/kernels)
kernel via [kernel-builder](https://github.com/huggingface/kernel-builder) and
benchmarked head-to-head against the real reference kernel on an **NVIDIA
B200**. Helion lets you write a kernel once in a high-level tile language and
have it compiled and autotuned per input shape; these ports cover
elementwise/activation, normalization, rotary embeddings, state-space models
(Mamba, RWKV), MoE/grouped GEMM, quantized GEMM, deformable and paged attention,
and INT8-quantized (SageAttention2) and full-precision flash attention.

These are benchmarked on Helion's default **Triton** backend, and ship
**pre-tuned configs** (`@helion.aot_kernel`) so downloaders skip autotuning
entirely — the first call is a sub-second compile of the shipped config instead
of minutes of search. Across the 38 kernel×shape pairs, shipping the
pre-tuned configs cuts total first-use autotuning **4065 s →
38 s (~106× faster time-to-first-run)** while retaining
performance to **geomean 1.02× of per-shape-optimal**. See
[`benchmark_results_triton_aot.md`](benchmark_results_triton_aot.md) for the
per-shape pre-tuned-vs-autotuned comparison and
[`aot_kernel_instructions.md`](aot_kernel_instructions.md) for how to use
pre-tuned kernels and add tunings for new hardware.

The table below reports the pre-tuned kernels. `Helion speed vs reference` =
reference latency / Helion latency (>1 → Helion is faster). Every row is
numerically verified against the reference; per-shape verification and
first-call compile times are in
[`benchmark_results_triton_aot.md`](benchmark_results_triton_aot.md).


## Aggregated benchmark results — Triton backend (pre-tuned / AOT)

| Task | Reference | Helion speed vs reference |
|---|---|---|
| activation — medium 8x2048x4096 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 4.25× |
| activation — large 8x4096x8192 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 3.65× |
| rotary — large 16x2048x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 3.25× |
| rwkv — large 16x1024x2048 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 2.47× |
| tinygrad-rms — large 65536x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 2.46× |
| layer-norm — large 16384x8192 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 2.31× |
| rwkv — medium 8x1024x1024 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 2.17× |
| tinygrad-rms — medium 16384x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 1.64× |
| sage-attention — small 2x8x512x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 1.50× |
| activation — small 8x1024x2048 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 1.35× |
| rotary — medium 8x512x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 1.31× |
| finegrained-fp8 — small 512x512x2048 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 1.11× |
| finegrained-fp8 — medium 2048x2048x4096 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.83× |
| rwkv — small 4x256x512 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 0.81× |
| sage-attention — large 8x16x2048x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 0.77× |
| finegrained-fp8 — large 4096x4096x8192 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.65× |
| mamba-ssm — small 2x256x128 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.60× |
| sage-attention — medium 4x16x1024x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 0.58× |
| tinygrad-rms — small 4096x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 0.51× |
| attention — small 2x512x8x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.46× |
| causal-conv1d — small 8x768x512 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.43× |
| layer-norm — small 256x768 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.43× |
| rotary — small 2x128x8x32 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 0.38× |
| layer-norm — medium 2048x2048 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.35× |
| attention — medium 4x1024x16x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.33× |
| mamba-ssm — large 8x2048x1024 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.33× |
| megablocks — small G8x1996x1024x1024 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.28× |
| mamba-ssm — medium 4x1024x512 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.26× |
| attention — large 8x2048x16x128 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.25× |
| deformable-detr — large 8x4000x8x64 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.21× |
| megablocks — large G32x16580x4096x4096 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.20× |
| megablocks — medium G16x8189x2048x2048 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.16× |
| causal-conv1d — medium 16x2048x2048 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.15× |
| paged-attention — medium 32x16x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.15× |
| causal-conv1d — large 32x4096x4096 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.14× |
| paged-attention — small 16x8x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.13× |
| deformable-detr — medium 4x2000x8x32 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.11× |
| deformable-detr — small 2x900x8x32 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.11× |

