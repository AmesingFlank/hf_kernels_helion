# hf_kernels_helion

[Helion](https://github.com/pytorch/helion) ports of the most-downloaded
[`kernels-community`](https://huggingface.co/kernels-community) kernels, each
packaged as a Hugging Face [`kernels`](https://github.com/huggingface/kernels)
kernel via [kernel-builder](https://github.com/huggingface/kernel-builder) and
benchmarked head-to-head against the real reference kernel on an **NVIDIA
B200**. Helion lets you write a kernel once in a high-level tile language and
have it compiled (through its Triton backend) and autotuned per input shape;
these ports cover elementwise/activation, normalization, rotary embeddings,
state-space models (Mamba, RWKV), MoE/grouped GEMM, quantized GEMM, deformable
and paged attention, and INT8-quantized (SageAttention2) and full-precision
flash attention. The table below aggregates every verified head-to-head; per-kernel
tables (with reference-vs-Helion latency for each shape) are in
[`benchmark_results_triton.md`](benchmark_results_triton.md).

`Helion speed vs reference` = reference latency / Helion latency (>1 → Helion is
faster). Every row is numerically verified against the reference. `Autotune
time` is the wall-clock time Helion's LLM-guided autotuner spent searching
configs for that shape, in its own fresh process.


## Aggregated benchmark results

| Task | Reference | Helion speed vs reference | Autotune time |
|---|---|---|---|
| activation — medium 8x2048x4096 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 4.61× | 114 s |
| activation — large 8x4096x8192 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 3.67× | 103 s |
| rotary — large 16x2048x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 3.33× | 83 s |
| layer-norm — large 16384x8192 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 2.69× | 82 s |
| tinygrad-rms — large 65536x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 2.46× | 78 s |
| rwkv — medium 8x1024x1024 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 2.09× | 75 s |
| tinygrad-rms — medium 16384x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 1.83× | 111 s |
| activation — small 8x1024x2048 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 1.68× | 142 s |
| sage-attention — small 2x8x512x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 1.60× | 131 s |
| finegrained-fp8 — small 512x512x2048 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 1.40× | 82 s |
| rotary — medium 8x512x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 1.32× | 98 s |
| rwkv — large 16x1024x2048 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 1.23× | 77 s |
| sage-attention — large 8x16x2048x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 0.84× | 135 s |
| rwkv — small 4x256x512 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 0.83× | 88 s |
| finegrained-fp8 — medium 2048x2048x4096 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.81× | 99 s |
| finegrained-fp8 — large 4096x4096x8192 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.75× | 89 s |
| tinygrad-rms — small 4096x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 0.60× | 119 s |
| sage-attention — medium 4x16x1024x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 0.58× | 141 s |
| mamba-ssm — small 2x256x128 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.48× | 101 s |
| attention — small 2x512x8x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.43× | 120 s |
| layer-norm — medium 2048x2048 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.42× | 134 s |
| layer-norm — small 256x768 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.40× | 137 s |
| causal-conv1d — small 8x768x512 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.39× | 101 s |
| rotary — small 2x128x8x32 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 0.32× | 155 s |
| attention — medium 4x1024x16x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.32× | 81 s |
| attention — large 8x2048x16x128 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.31× | 88 s |
| megablocks — small G8x2195x1024x1024 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.27× | 71 s |
| deformable-detr — large 8x4000x8x64 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.24× | 113 s |
| mamba-ssm — large 8x2048x1024 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.23× | 88 s |
| megablocks — large G32x17221x4096x4096 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.21× | 152 s |
| mamba-ssm — medium 4x1024x512 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.15× | 92 s |
| megablocks — medium G16x8289x2048x2048 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.14× | 86 s |
| causal-conv1d — medium 16x2048x2048 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.14× | 80 s |
| causal-conv1d — large 32x4096x4096 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.14× | 98 s |
| deformable-detr — medium 4x2000x8x32 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.12× | 110 s |
| paged-attention — small 16x8x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.12× | 116 s |
| paged-attention — medium 32x16x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.12× | 194 s |
| deformable-detr — small 2x900x8x32 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.12× | 101 s |

