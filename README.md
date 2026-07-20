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

Helion has two codegen backends, and the same kernels are benchmarked through
both: the default **Triton** backend and the newer **CuteDSL** (`cute`) backend
(CUTLASS CuTe DSL, `HELION_BACKEND=cute`). The aggregated tables below give the
per-shape results for each; full per-kernel tables are in
[`benchmark_results_triton.md`](benchmark_results_triton.md) and
[`benchmark_results_cute.md`](benchmark_results_cute.md).

`Helion speed vs reference` = reference latency / Helion latency (>1 → Helion is
faster). `verified ✗` marks a shape where the Helion output did not match the
reference (kept for transparency). `Autotune time` is the wall-clock time
Helion's LLM-guided autotuner spent searching configs for that shape, in its own
fresh process.

## Pre-tuned (AOT) kernels

The Triton kernels ship **pre-tuned configs** (`@helion.experimental.aot_kernel`)
so downloaders skip autotuning entirely — first call is a sub-second compile
instead of minutes of search. Across the 38 kernel×shape pairs, shipping the
pre-tuned configs cuts total autotune time **4065 s → 39 s (~105× faster
time-to-first-run)** while retaining performance to **geomean 1.02× of
per-shape-optimal**. See
[`benchmark_results_triton_aot.md`](benchmark_results_triton_aot.md) for the
per-shape comparison and [`aot_kernel_instructions.md`](aot_kernel_instructions.md)
for how to use pre-tuned kernels and add tunings for new hardware.


## Aggregated benchmark results — Triton backend

| Task | Reference | Helion speed vs reference | Verified | Autotune time |
|---|---|---|---|---|
| activation — medium 8x2048x4096 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 4.61× | ✓ | 114 s |
| activation — large 8x4096x8192 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 3.67× | ✓ | 103 s |
| rotary — large 16x2048x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 3.33× | ✓ | 83 s |
| layer-norm — large 16384x8192 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 2.69× | ✓ | 82 s |
| tinygrad-rms — large 65536x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 2.46× | ✓ | 78 s |
| rwkv — medium 8x1024x1024 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 2.09× | ✓ | 75 s |
| tinygrad-rms — medium 16384x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 1.83× | ✓ | 111 s |
| activation — small 8x1024x2048 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 1.68× | ✓ | 142 s |
| sage-attention — small 2x8x512x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 1.60× | ✓ | 131 s |
| finegrained-fp8 — small 512x512x2048 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 1.40× | ✓ | 82 s |
| rotary — medium 8x512x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 1.32× | ✓ | 98 s |
| rwkv — large 16x1024x2048 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 1.23× | ✓ | 77 s |
| sage-attention — large 8x16x2048x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 0.84× | ✓ | 135 s |
| rwkv — small 4x256x512 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 0.83× | ✓ | 88 s |
| finegrained-fp8 — medium 2048x2048x4096 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.81× | ✓ | 99 s |
| finegrained-fp8 — large 4096x4096x8192 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.75× | ✓ | 89 s |
| tinygrad-rms — small 4096x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 0.60× | ✓ | 119 s |
| sage-attention — medium 4x16x1024x128 | [SageAttention](https://github.com/thu-ml/SageAttention) | 0.58× | ✓ | 141 s |
| mamba-ssm — small 2x256x128 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.48× | ✓ | 101 s |
| attention — small 2x512x8x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.43× | ✓ | 120 s |
| layer-norm — medium 2048x2048 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.42× | ✓ | 134 s |
| layer-norm — small 256x768 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.40× | ✓ | 137 s |
| causal-conv1d — small 8x768x512 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.39× | ✓ | 101 s |
| rotary — small 2x128x8x32 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 0.32× | ✓ | 155 s |
| attention — medium 4x1024x16x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.32× | ✓ | 81 s |
| attention — large 8x2048x16x128 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.31× | ✓ | 88 s |
| megablocks — small G8x2195x1024x1024 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.27× | ✓ | 71 s |
| deformable-detr — large 8x4000x8x64 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.24× | ✓ | 113 s |
| mamba-ssm — large 8x2048x1024 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.23× | ✓ | 88 s |
| megablocks — large G32x17221x4096x4096 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.21× | ✓ | 152 s |
| mamba-ssm — medium 4x1024x512 | [mamba-ssm](https://huggingface.co/kernels/kernels-community/mamba-ssm) | 0.15× | ✓ | 92 s |
| causal-conv1d — medium 16x2048x2048 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.14× | ✓ | 80 s |
| causal-conv1d — large 32x4096x4096 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.14× | ✓ | 98 s |
| megablocks — medium G16x8289x2048x2048 | [megablocks](https://huggingface.co/kernels/kernels-community/megablocks) | 0.14× | ✓ | 86 s |
| paged-attention — small 16x8x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.12× | ✓ | 116 s |
| paged-attention — medium 32x16x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.12× | ✓ | 194 s |
| deformable-detr — small 2x900x8x32 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.12× | ✓ | 101 s |
| deformable-detr — medium 4x2000x8x32 | [deformable-detr](https://huggingface.co/kernels/kernels-community/deformable-detr) | 0.12× | ✓ | 110 s |

## Aggregated benchmark results — CuteDSL backend

| Task | Reference | Helion speed vs reference | Verified | Autotune time |
|---|---|---|---|---|
| rotary — large 16x2048x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 2.64× | ✓ | 280 s |
| activation — medium 8x2048x4096 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 2.46× | ✓ | 253 s |
| tinygrad-rms — large 65536x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 2.44× | ✓ | 267 s |
| activation — large 8x4096x8192 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 2.06× | ✓ | 243 s |
| layer-norm — large 16384x8192 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 1.92× | ✓ | 269 s |
| rotary — medium 8x512x32x64 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 0.94× | ✓ | 347 s |
| tinygrad-rms — medium 16384x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 0.92× | ✓ | 303 s |
| activation — small 8x1024x2048 | [activation](https://huggingface.co/kernels/kernels-community/activation) | 0.92× | ✓ | 336 s |
| rwkv — large 16x1024x2048 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 0.84× | ✓ | 250 s |
| rwkv — medium 8x1024x1024 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 0.82× | ✓ | 259 s |
| rwkv — small 4x256x512 | [rwkv](https://huggingface.co/kernels/kernels-community/rwkv) | 0.69× | ✓ | 241 s |
| tinygrad-rms — small 4096x1024 | [tinygrad-rms](https://huggingface.co/kernels/kernels-community/tinygrad-rms) | 0.31× | ✓ | 360 s |
| attention — small 2x512x8x64 | [flash-attn4](https://huggingface.co/kernels/kernels-community/flash-attn4) | 0.28× | ✓ | 402 s |
| layer-norm — small 256x768 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.26× | ✓ | 369 s |
| rotary — small 2x128x8x32 | [rotary](https://huggingface.co/kernels/kernels-community/rotary) | 0.23× | ✓ | 341 s |
| causal-conv1d — small 8x768x512 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.23× | ✓ | 329 s |
| causal-conv1d — medium 16x2048x2048 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.22× | ✓ | 260 s |
| layer-norm — medium 2048x2048 | [layer-norm](https://huggingface.co/kernels/kernels-community/layer-norm) | 0.21× | ✓ | 339 s |
| causal-conv1d — large 32x4096x4096 | [causal-conv1d](https://huggingface.co/kernels/kernels-community/causal-conv1d) | 0.20× | ✓ | 246 s |
| finegrained-fp8 — small 512x512x2048 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.05× | ✗ | 235 s |
| paged-attention — medium 32x16x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.04× | ✗ | 239 s |
| paged-attention — small 16x8x64 | [paged-attention](https://huggingface.co/kernels/kernels-community/paged-attention) | 0.02× | ✗ | 201 s |
| finegrained-fp8 — medium 2048x2048x4096 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.01× | ✗ | 281 s |
| finegrained-fp8 — large 4096x4096x8192 | [finegrained-fp8](https://huggingface.co/kernels/kernels-community/finegrained-fp8) | 0.01× | ✗ | 511 s |

## CuteDSL backend — failure cases

Every (kernel, input size) that produces a **verified** result on the Triton backend but **not** on the CuteDSL backend, with the reason. Covers three modes: the kernel doesn't compile on `cute`, it compiles but is numerically wrong, or its autotune doesn't converge in budget. Full per-kernel context is in [`benchmark_results_cute.md`](benchmark_results_cute.md).

| Task | Input size | Why the Helion CuteDSL kernel didn't work |
|---|---|---|
| paged-attention | small 16x8x64 | Compiles and runs on cute, but the output fails `torch.allclose` vs the reference (numerically incorrect). |
| paged-attention | medium 32x16x64 | Compiles and runs on cute, but the output fails `torch.allclose` vs the reference (numerically incorrect). |
| mamba-ssm | small 2x256x128 | Does not compile — the sequential selective-scan recurrence raises `TypeError: Expected a TensorSSA or Numeric(Float), but got ArithValue` in CuTe codegen. |
| mamba-ssm | medium 4x1024x512 | Does not compile — the sequential selective-scan recurrence raises `TypeError: Expected a TensorSSA or Numeric(Float), but got ArithValue` in CuTe codegen. |
| mamba-ssm | large 8x2048x1024 | Does not compile — the sequential selective-scan recurrence raises `TypeError: Expected a TensorSSA or Numeric(Float), but got ArithValue` in CuTe codegen. |
| megablocks | small G8x2195x1024x1024 | Autotuning hangs — the jagged grouped-GEMM wedges in CuTe compilation (CPU-bound, no config ever benchmarked) and hits the wall-clock timeout. |
| megablocks | medium G16x8289x2048x2048 | Autotuning hangs — the jagged grouped-GEMM wedges in CuTe compilation (CPU-bound, no config ever benchmarked) and hits the wall-clock timeout. |
| megablocks | large G32x17221x4096x4096 | Autotuning hangs — the jagged grouped-GEMM wedges in CuTe compilation (CPU-bound, no config ever benchmarked) and hits the wall-clock timeout. |
| deformable-detr | small 2x900x8x32 | Does not compile — the bilinear-sampling gather raises `BackendUnsupported: unresolved CuTe layout mismatch`. |
| deformable-detr | medium 4x2000x8x32 | Does not compile — the bilinear-sampling gather raises `BackendUnsupported: unresolved CuTe layout mismatch`. |
| deformable-detr | large 8x4000x8x64 | Does not compile — the bilinear-sampling gather raises `BackendUnsupported: unresolved CuTe layout mismatch`. |
| finegrained-fp8 | small 512x512x2048 | Compiles and runs on cute, but the output fails `torch.allclose` vs the reference (numerically incorrect). |
| finegrained-fp8 | medium 2048x2048x4096 | Compiles and runs on cute, but the output fails `torch.allclose` vs the reference (numerically incorrect). |
| finegrained-fp8 | large 4096x4096x8192 | Compiles and runs on cute, but the output fails `torch.allclose` vs the reference (numerically incorrect). |
| attention | medium 4x1024x16x64 | LLM autotuner did not converge within the 700 s per-shape budget on cute (this kernel's smaller shape(s) did). |
| attention | large 8x2048x16x128 | LLM autotuner did not converge within the 700 s per-shape budget on cute (this kernel's smaller shape(s) did). |
| sage-attention | small 2x8x512x128 | Does not compile — the INT8-quant step `torch.round` raises `InductorLoweringError` (the `cute` backend has no lowering for `aten.round.default`). |
| sage-attention | medium 4x16x1024x128 | Does not compile — the INT8-quant step `torch.round` raises `InductorLoweringError` (the `cute` backend has no lowering for `aten.round.default`). |
| sage-attention | large 8x16x2048x128 | Does not compile — the INT8-quant step `torch.round` raises `InductorLoweringError` (the `cute` backend has no lowering for `aten.round.default`). |

