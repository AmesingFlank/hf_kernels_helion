# Helion kernels (triton backend) — benchmark vs `kernels-community` references

Helion kernels here are compiled through Helion's default **Triton
backend**. Most also ship pre-tuned (AOT) configs — see
`benchmark_results_triton_aot.md` for the pre-tuned-vs-autotuned comparison.

Helion ports of `kernels-community` kernels, benchmarked on **NVIDIA B200**
against the **real HuggingFace kernel** (each reference's build variant pulled
locally into `~/hf_kernels_refs` and loaded via `get_local_kernel`, so
benchmarking makes zero Hub calls). Comparison is Helion-vs-reference only.

**Autotuning:** LLM-guided (`HELION_AUTOTUNER=LLMGuidedSearch`, Bedrock
`claude-haiku-4.5`, `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0`). The `autotune (s)`
column is the wall-clock time the LLM autotuner spent searching configs for that
input size.

`speedup` = ref_ms / helion_ms  (>1 → Helion faster). Every row is numerically
verified against the reference (`torch.allclose`); a ✗ in the `verified` column
means the Helion output did NOT match the reference on that shape (kept for
transparency). Each input size is autotuned in its **own fresh process with all
caches cleared**, so every `autotune (s)` is a real measurement.

This covers the full `kernels-community` download list from **activation** down
to **deep-gemm** (18 kernels; flash-attn variants excluded per the task), plus
two dedicated attention comparisons that run on Blackwell — a plain SDPA Helion
kernel vs **flash-attn4** (CuTeDSL), and a Helion **SageAttention2**
(INT8-quantized) kernel vs **thu-ml/SageAttention** built from source for sm_100.
Kernels with no head-to-head row either have no loadable reference build for
this system (torch 2.13/cu130, sm_100 Blackwell) or expose private data formats
the Helion op can't be called against on identical inputs — noted per kernel.
That makes **13 head-to-head tables** below.

## activation — `silu_and_mul`
vs **kernels-community/activation (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 8x1024x2048 | 0.0257 | 0.0430 | 1.68× | 142 | ✓ |
| medium 8x2048x4096 | 0.0310 | 0.1430 | 4.61× | 114 | ✓ |
| large 8x4096x8192 | 0.1224 | 0.4492 | 3.67× | 103 | ✓ |

## quantization-bitsandbytes — `gemm_4bit (NF4)`
vs **kernels-community/quantization-bitsandbytes**

_Reference publishes **CPU-only** builds (no CUDA variant) — cannot run on the B200._

## causal-conv1d — `causal_conv1d_fn (+SiLU)`
vs **kernels-community/causal-conv1d (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 8x768x512 | 0.0352 | 0.0136 | 0.39× | 101 | ✓ |
| medium 16x2048x2048 | 0.5620 | 0.0801 | 0.14× | 80 | ✓ |
| large 32x4096x4096 | 4.1575 | 0.5806 | 0.14× | 98 | ✓ |

## rotary — `apply_rotary`
vs **kernels-community/rotary (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 2x128x8x32 | 0.0530 | 0.0169 | 0.32× | 155 | ✓ |
| medium 8x512x32x64 | 0.0562 | 0.0740 | 1.32× | 98 | ✓ |
| large 16x2048x32x64 | 0.1635 | 0.5449 | 3.33× | 83 | ✓ |

## paged-attention — `paged_attention_v1`
vs **kernels-community/paged-attention (vLLM C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 16x8x64 | 0.0617 | 0.0074 | 0.12× | 116 | ✓ |
| medium 32x16x64 | 0.0618 | 0.0074 | 0.12× | 194 | ✓ |

## mamba-ssm — `selective_scan_fn`
vs **kernels-community/mamba-ssm (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 2x256x128 | 0.0545 | 0.0260 | 0.48× | 101 | ✓ |
| medium 4x1024x512 | 0.3345 | 0.0500 | 0.15× | 92 | ✓ |
| large 8x2048x1024 | 1.4605 | 0.3400 | 0.23× | 88 | ✓ |

## megablocks — `gg_ops.gmm (grouped GEMM)`
vs **kernels-community/megablocks (AITER grouped GEMM)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small G8x2195x1024x1024 | 0.2435 | 0.0652 | 0.27× | 71 | ✓ |
| medium G16x8289x2048x2048 | 0.8242 | 0.1190 | 0.14× | 86 | ✓ |
| large G32x17221x4096x4096 | 2.7935 | 0.5887 | 0.21× | 152 | ✓ |

## deformable-detr — `ms_deform_attn (single-level)`
vs **kernels-community/deformable-detr (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 2x900x8x32 | 0.1433 | 0.0168 | 0.12× | 101 | ✓ |
| medium 4x2000x8x32 | 0.5678 | 0.0699 | 0.12× | 110 | ✓ |
| large 8x4000x8x64 | 2.2937 | 0.5534 | 0.24× | 113 | ✓ |

## tinygrad-rms — `tinygrad_rms_norm`
vs **kernels-community/tinygrad-rms (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 4096x1024 | 0.0298 | 0.0179 | 0.60× | 119 | ✓ |
| medium 16384x1024 | 0.0302 | 0.0552 | 1.83× | 111 | ✓ |
| large 65536x1024 | 0.0809 | 0.1992 | 2.46× | 78 | ✓ |

## rwkv — `wkv (forward)`
vs **kernels-community/rwkv (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 4x256x512 | 0.0650 | 0.0541 | 0.83× | 88 | ✓ |
| medium 8x1024x1024 | 0.2519 | 0.5270 | 2.09× | 75 | ✓ |
| large 16x1024x2048 | 0.4994 | 0.6150 | 1.23× | 77 | ✓ |

## layer-norm — `dropout_add_ln_fwd (RMSNorm path)`
vs **kernels-community/layer-norm (flash C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 256x768 | 0.0272 | 0.0109 | 0.40× | 137 | ✓ |
| medium 2048x2048 | 0.0262 | 0.0111 | 0.42× | 134 | ✓ |
| large 16384x8192 | 0.0804 | 0.2164 | 2.69× | 82 | ✓ |

## mra — `mm_to_sparse (block-sparse)`
vs **kernels-community/mra (C++/CUDA)**

_Reference `mm_to_sparse` uses a private block-index encoding I couldn't match on same inputs (Helion kernel verified vs PyTorch separately)._

## punica-sgmv — `add_lora_sgmv_cutlass`
vs **kernels-community/punica-sgmv (C++/CUDA)**

_Reference `add_lora_sgmv_cutlass` takes **raw pointer-array** weights (private CUDA ABI); can't construct matching inputs._

## yoso — `lsh_cumulation`
vs **kernels-community/yoso (C++/CUDA)**

_Reference `lsh_cumulation` implements the full LSH hashtable-collision algorithm (8 private args); different from the simple bucket-sum Helion kernel._

## finegrained-fp8 — `w8a8_block_fp8_matmul`
vs **kernels-community/finegrained-fp8 (Triton)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 512x512x2048 | 0.0511 | 0.0715 | 1.40× | 82 | ✓ |
| medium 2048x2048x4096 | 0.1758 | 0.1420 | 0.81× | 99 | ✓ |
| large 4096x4096x8192 | 1.2803 | 0.9590 | 0.75× | 89 | ✓ |

## gpt-oss-triton-kernels — `matmul_ogs (MoE)`
vs **kernels-community/gpt-oss-triton-kernels (Triton)**

_Reference `matmul_ogs` takes private routing dataclasses (RoutingData / gather-scatter indices), not plain tensors._

## quantization-eetq — `w8_a16_gemm (int8)`
vs **kernels-community/quantization-eetq**

_Reference builds stop at **torch 2.11 / cu128**; no torch-2.13 + cu130 variant for this box._

## deep-gemm — `fp8 blockwise GEMM`
vs **kernels-community/deep-gemm**

_Reference builds stop at **torch 2.11**; `get_kernel` 404s for v1. No loadable variant here._

## attention — `flash_attn_func`
vs **kernels-community/flash-attn4 (Blackwell CuTeDSL)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 2x512x8x64 | 0.0819 | 0.0356 | 0.43× | 120 | ✓ |
| medium 4x1024x16x64 | 0.1082 | 0.0343 | 0.32× | 81 | ✓ |
| large 8x2048x16x128 | 0.8581 | 0.2692 | 0.31× | 88 | ✓ |

## sage-attention — `sageattn (INT8 quant attn)`
vs **thu-ml/SageAttention 2.2.0 (INT8-QK/FP16-PV CUDA, built from source for sm_100)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 2x8x512x128 | 0.0930 | 0.1488 | 1.60× | 131 | ✓ |
| medium 4x16x1024x128 | 0.3018 | 0.1757 | 0.58× | 141 | ✓ |
| large 8x16x2048x128 | 1.0292 | 0.8604 | 0.84× | 135 | ✓ |

