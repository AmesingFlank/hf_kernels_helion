# Helion kernels (cute backend) — benchmark vs `kernels-community` references

Helion kernels here are compiled through Helion's **CuteDSL (`cute`)
backend** (`HELION_BACKEND=cute`) — CUTLASS CuTe DSL codegen targeting the
B200's Blackwell tensor cores — rather than the default Triton backend. See
`benchmark_results_triton.md` for the same kernels on the Triton backend.
**Not every Helion kernel compiles on the CuteDSL backend yet** (the backend is
newer); kernels that don't are noted in place.

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
this system (torch 2.13/cu130, sm_100 Blackwell), expose private data formats
the Helion op can't be called against on identical inputs, or (on the CuteDSL
backend) don't yet compile — noted per kernel.
That makes **9 head-to-head tables** below.

## activation — `silu_and_mul`
vs **kernels-community/activation (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 8x1024x2048 | 0.0470 | 0.0430 | 0.92× | 336 | ✓ |
| medium 8x2048x4096 | 0.0583 | 0.1432 | 2.46× | 253 | ✓ |
| large 8x4096x8192 | 0.2179 | 0.4491 | 2.06× | 243 | ✓ |

## quantization-bitsandbytes — `gemm_4bit (NF4)`
vs **kernels-community/quantization-bitsandbytes**

_Reference publishes **CPU-only** builds (no CUDA variant) — cannot run on the B200._

## causal-conv1d — `causal_conv1d_fn (+SiLU)`
vs **kernels-community/causal-conv1d (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 8x768x512 | 0.0579 | 0.0136 | 0.23× | 329 | ✓ |
| medium 16x2048x2048 | 0.3629 | 0.0802 | 0.22× | 260 | ✓ |
| large 32x4096x4096 | 2.9626 | 0.5804 | 0.20× | 246 | ✓ |

## rotary — `apply_rotary`
vs **kernels-community/rotary (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 2x128x8x32 | 0.0779 | 0.0183 | 0.23× | 341 | ✓ |
| medium 8x512x32x64 | 0.0784 | 0.0741 | 0.94× | 347 | ✓ |
| large 16x2048x32x64 | 0.2104 | 0.5563 | 2.64× | 280 | ✓ |

## paged-attention — `paged_attention_v1`
vs **kernels-community/paged-attention (vLLM C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 16x8x64 | 0.4381 | 0.0089 | 0.02× | 201 | ✗ |
| medium 32x16x64 | 0.2193 | 0.0089 | 0.04× | 239 | ✗ |

## mamba-ssm — `selective_scan_fn`
vs **kernels-community/mamba-ssm (C++/CUDA)**

_Helion kernel **does not compile on the CuteDSL backend** — the sequential selective-scan recurrence hits `TypeError: Expected a TensorSSA or Numeric(Float), but got ArithValue` in CuTe codegen, so the autotuner can't build a baseline. Works on the Triton backend (see `benchmark_results_triton.md`)._

## megablocks — `gg_ops.gmm (grouped GEMM)`
vs **kernels-community/megablocks (AITER grouped GEMM)**

_Autotuning **hangs on the CuteDSL backend** — the jagged grouped-GEMM kernel wedges in CuTe compilation (CPU-bound, no configs ever benchmarked) and hits the wall-clock timeout with no result. Works on the Triton backend (see `benchmark_results_triton.md`)._

## deformable-detr — `ms_deform_attn (single-level)`
vs **kernels-community/deformable-detr (C++/CUDA)**

_Helion kernel **does not compile on the CuteDSL backend** — the bilinear-sampling gather raises `BackendUnsupported: unresolved CuTe layout mismatch`. Works on the Triton backend (see `benchmark_results_triton.md`)._

## tinygrad-rms — `tinygrad_rms_norm`
vs **kernels-community/tinygrad-rms (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 4096x1024 | 0.0581 | 0.0178 | 0.31× | 360 | ✓ |
| medium 16384x1024 | 0.0602 | 0.0556 | 0.92× | 303 | ✓ |
| large 65536x1024 | 0.0818 | 0.1999 | 2.44× | 267 | ✓ |

## rwkv — `wkv (forward)`
vs **kernels-community/rwkv (C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 4x256x512 | 0.0787 | 0.0543 | 0.69× | 241 | ✓ |
| medium 8x1024x1024 | 0.6832 | 0.5573 | 0.82× | 259 | ✓ |
| large 16x1024x2048 | 0.7331 | 0.6151 | 0.84× | 250 | ✓ |

## layer-norm — `dropout_add_ln_fwd (RMSNorm path)`
vs **kernels-community/layer-norm (flash C++/CUDA)**

| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |
|---|---|---|---|---|---|
| small 256x768 | 0.0480 | 0.0127 | 0.26× | 369 | ✓ |
| medium 2048x2048 | 0.0516 | 0.0106 | 0.21× | 339 | ✓ |
| large 16384x8192 | 0.1158 | 0.2220 | 1.92× | 269 | ✓ |

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
| small 512x512x2048 | 1.3544 | 0.0716 | 0.05× | 235 | ✗ |
| medium 2048x2048x4096 | 24.3607 | 0.1422 | 0.01× | 281 | ✗ |
| large 4096x4096x8192 | 193.4872 | 0.9588 | 0.01× | 511 | ✗ |

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
| small 2x512x8x64 | 0.1217 | 0.0339 | 0.28× | 402 | ✓ |

_The `medium` and `large` shapes are omitted: on the CuteDSL backend the LLM autotuner did not converge within the 700 s per-shape wall-clock budget for those shapes (the `small` shape did). Autotuning is markedly slower on `cute` than on Triton for this kernel._

## sage-attention — `sageattn (INT8 quant attn)`
vs **thu-ml/SageAttention 2.2.0 (INT8-QK/FP16-PV CUDA, built from source for sm_100)**

_Helion kernel **does not compile on the CuteDSL backend** — the INT8 quantization step `torch.round` raises `InductorLoweringError: Error in codegen for aten.round.default` (the `cute` backend has no lowering for `round`). Works on the Triton backend (see `benchmark_results_triton.md`); the plain-SDPA attention kernel above, which has no rounding, does run on `cute`._

