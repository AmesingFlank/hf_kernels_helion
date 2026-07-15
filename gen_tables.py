"""Generate benchmark_results_<backend>.md from results/<backend>/<kernel>.json
— one table per kernel. Uses a FIXED header (does not read the existing doc), so
re-running is idempotent.

Backend-parameterized:
  GEN_BACKEND=triton (default) -> reads results/triton/, writes benchmark_results_triton.md
  GEN_BACKEND=cute             -> reads results/cute/,   writes benchmark_results_cute.md
"""
import json, os

HH = "/home/dev/hf_kernels_helion"
_BACKEND = os.environ.get("GEN_BACKEND", "triton")
_IS_CUTE = _BACKEND == "cute"
OUT = f"{HH}/benchmark_results_{_BACKEND}.md"
_RESULTS_DIR = f"{HH}/results/{_BACKEND}"


def load(name):
    p = f"{_RESULTS_DIR}/{name}.json"
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            return d if d else None
        except Exception:
            return None
    return None


# (json key, repo, op, ref description)
ORDER = [
    ("activation", "activation", "silu_and_mul", "kernels-community/activation (C++/CUDA)"),
    ("bitsandbytes", "quantization-bitsandbytes", "gemm_4bit (NF4)", "kernels-community/quantization-bitsandbytes"),
    ("causal_conv1d", "causal-conv1d", "causal_conv1d_fn (+SiLU)", "kernels-community/causal-conv1d (C++/CUDA)"),
    ("rotary", "rotary", "apply_rotary", "kernels-community/rotary (C++/CUDA)"),
    ("paged", "paged-attention", "paged_attention_v1", "kernels-community/paged-attention (vLLM C++/CUDA)"),
    ("mamba", "mamba-ssm", "selective_scan_fn", "kernels-community/mamba-ssm (C++/CUDA)"),
    ("megablocks", "megablocks", "gg_ops.gmm (grouped GEMM)", "kernels-community/megablocks (AITER grouped GEMM)"),
    ("deformable", "deformable-detr", "ms_deform_attn (single-level)", "kernels-community/deformable-detr (C++/CUDA)"),
    ("tinygrad_rms", "tinygrad-rms", "tinygrad_rms_norm", "kernels-community/tinygrad-rms (C++/CUDA)"),
    ("rwkv", "rwkv", "wkv (forward)", "kernels-community/rwkv (C++/CUDA)"),
    ("layer_norm", "layer-norm", "dropout_add_ln_fwd (RMSNorm path)", "kernels-community/layer-norm (flash C++/CUDA)"),
    ("mra", "mra", "mm_to_sparse (block-sparse)", "kernels-community/mra (C++/CUDA)"),
    ("punica_sgmv", "punica-sgmv", "add_lora_sgmv_cutlass", "kernels-community/punica-sgmv (C++/CUDA)"),
    ("yoso", "yoso", "lsh_cumulation", "kernels-community/yoso (C++/CUDA)"),
    ("fp8", "finegrained-fp8", "w8a8_block_fp8_matmul", "kernels-community/finegrained-fp8 (Triton)"),
    ("gpt_oss", "gpt-oss-triton-kernels", "matmul_ogs (MoE)", "kernels-community/gpt-oss-triton-kernels (Triton)"),
    ("eetq", "quantization-eetq", "w8_a16_gemm (int8)", "kernels-community/quantization-eetq"),
    ("deep_gemm", "deep-gemm", "fp8 blockwise GEMM", "kernels-community/deep-gemm"),
    ("attention", "attention", "flash_attn_func", "kernels-community/flash-attn4 (Blackwell CuTeDSL)"),
    ("sage", "sage-attention", "sageattn (INT8 quant attn)", "thu-ml/SageAttention 2.2.0 (INT8-QK/FP16-PV CUDA, built from source for sm_100)"),
]

# How many kernels actually produced a head-to-head table on this backend.
_n_tables = sum(1 for k, _, _, _ in ORDER if load(k))
_FOOTER_NOTE = f"That makes **{_n_tables} head-to-head tables** below."

_BACKEND_BLURB = (
    """Helion kernels here are compiled through Helion's **CuteDSL (`cute`)
backend** (`HELION_BACKEND=cute`) — CUTLASS CuTe DSL codegen targeting the
B200's Blackwell tensor cores — rather than the default Triton backend. See
`benchmark_results_triton.md` for the same kernels on the Triton backend.
**Not every Helion kernel compiles on the CuteDSL backend yet** (the backend is
newer); kernels that don't are noted in place."""
    if _IS_CUTE else
    """Helion kernels here are compiled through Helion's default **Triton
backend**. See `benchmark_results_cute.md` for the same kernels on the CuteDSL
(`cute`) backend."""
)

HEADER = f"""# Helion kernels ({_BACKEND} backend) — benchmark vs `kernels-community` references

{_BACKEND_BLURB}

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
{_FOOTER_NOTE}
"""

# reason a kernel has no head-to-head row (reference-side, backend-independent)
NOTE = {
    "bitsandbytes": "Reference publishes **CPU-only** builds (no CUDA variant) — cannot run on the B200.",
    "eetq": "Reference builds stop at **torch 2.11 / cu128**; no torch-2.13 + cu130 variant for this box.",
    "deep_gemm": "Reference builds stop at **torch 2.11**; `get_kernel` 404s for v1. No loadable variant here.",
    "mra": "Reference `mm_to_sparse` uses a private block-index encoding I couldn't match on same inputs (Helion kernel verified vs PyTorch separately).",
    "punica_sgmv": "Reference `add_lora_sgmv_cutlass` takes **raw pointer-array** weights (private CUDA ABI); can't construct matching inputs.",
    "yoso": "Reference `lsh_cumulation` implements the full LSH hashtable-collision algorithm (8 private args); different from the simple bucket-sum Helion kernel.",
    "gpt_oss": "Reference `matmul_ogs` takes private routing dataclasses (RoutingData / gather-scatter indices), not plain tensors.",
}

# Kernels that have a working triton head-to-head but do NOT compile on the
# CuteDSL backend. Used only when GEN_BACKEND=cute.
CUTE_FAIL_NOTE = {
    "mamba": "Helion kernel **does not compile on the CuteDSL backend** — the sequential selective-scan recurrence hits `TypeError: Expected a TensorSSA or Numeric(Float), but got ArithValue` in CuTe codegen, so the autotuner can't build a baseline. Works on the Triton backend (see `benchmark_results_triton.md`).",
    "megablocks": "Autotuning **hangs on the CuteDSL backend** — the jagged grouped-GEMM kernel wedges in CuTe compilation (CPU-bound, no configs ever benchmarked) and hits the wall-clock timeout with no result. Works on the Triton backend (see `benchmark_results_triton.md`).",
    "deformable": "Helion kernel **does not compile on the CuteDSL backend** — the bilinear-sampling gather raises `BackendUnsupported: unresolved CuTe layout mismatch`. Works on the Triton backend (see `benchmark_results_triton.md`).",
}

lines = [HEADER]
for key, repo, opname, refname in ORDER:
    rows = load(key)
    lines.append(f"## {repo} — `{opname}`")
    lines.append(f"vs **{refname}**")
    lines.append("")
    if not rows:
        note = (CUTE_FAIL_NOTE.get(key) if _IS_CUTE else None) or NOTE.get(
            key, "reference kernel not available for head-to-head on this system.")
        lines.append("_" + note + "_")
        lines.append("")
        continue
    lines.append("| input size | Helion (ms) | ref (ms) | speedup | autotune (s) | verified |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['size']} | {r['helion_ms']:.4f} | {r['ref_ms']:.4f} | "
                     f"{r['speedup']:.2f}× | {r['autotune_s']:.0f} | {'✓' if r['ok'] else '✗'} |")
    lines.append("")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"regenerated {OUT} - {_n_tables} kernels with data (backend={_BACKEND})")
