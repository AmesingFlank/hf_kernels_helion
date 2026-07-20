"""Generate README.md for the hf_kernels_helion repo: intro + one aggregated
table per backend (Task | Reference | Helion speed vs reference | Autotune
time), sorted by decreasing Helion speedup. Data comes from results/<backend>/.

Renders the triton table always, and the cute table if results/cute/ exists.
"""
import json, os

HH = "/home/dev/hf_kernels_helion"
OUT = f"{HH}/README.md"

# (json key, task label, reference display name, reference URL)
KERNELS = [
    ("activation",    "activation",    "activation",     "https://huggingface.co/kernels/kernels-community/activation"),
    ("causal_conv1d", "causal-conv1d", "causal-conv1d",  "https://huggingface.co/kernels/kernels-community/causal-conv1d"),
    ("rotary",        "rotary",        "rotary",         "https://huggingface.co/kernels/kernels-community/rotary"),
    ("paged",         "paged-attention","paged-attention","https://huggingface.co/kernels/kernels-community/paged-attention"),
    ("mamba",         "mamba-ssm",     "mamba-ssm",      "https://huggingface.co/kernels/kernels-community/mamba-ssm"),
    ("megablocks",    "megablocks",    "megablocks",     "https://huggingface.co/kernels/kernels-community/megablocks"),
    ("deformable",    "deformable-detr","deformable-detr","https://huggingface.co/kernels/kernels-community/deformable-detr"),
    ("tinygrad_rms",  "tinygrad-rms",  "tinygrad-rms",   "https://huggingface.co/kernels/kernels-community/tinygrad-rms"),
    ("rwkv",          "rwkv",          "rwkv",           "https://huggingface.co/kernels/kernels-community/rwkv"),
    ("layer_norm",    "layer-norm",    "layer-norm",     "https://huggingface.co/kernels/kernels-community/layer-norm"),
    ("fp8",           "finegrained-fp8","finegrained-fp8","https://huggingface.co/kernels/kernels-community/finegrained-fp8"),
    ("attention",     "attention",     "flash-attn4",    "https://huggingface.co/kernels/kernels-community/flash-attn4"),
    ("sage",          "sage-attention","SageAttention",  "https://github.com/thu-ml/SageAttention"),
]

INTRO = """# hf_kernels_helion

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
"""


def load(backend, name):
    p = f"{HH}/results/{backend}/{name}.json"
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
        return d or None
    except Exception:
        return None


def build_rows(backend):
    rows = []  # (speedup, task, ref_md, speedup_str, verified, autotune_str)
    for key, task, refname, url in KERNELS:
        data = load(backend, key)
        if not data:
            continue
        ref_md = f"[{refname}]({url})"
        for r in data:
            rows.append((
                r["speedup"],
                f"{task} — {r['size']}",
                ref_md,
                f"{r['speedup']:.2f}×",
                "✓" if r["ok"] else "✗",
                f"{r['autotune_s']:.0f} s",
            ))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def render_table(title, rows):
    out = [f"## {title}", ""]
    if not rows:
        out += ["_No results for this backend yet._", ""]
        return out
    out += ["| Task | Reference | Helion speed vs reference | Verified | Autotune time |",
            "|---|---|---|---|---|"]
    for _, task, ref_md, sp, ok, at in rows:
        out.append(f"| {task} | {ref_md} | {sp} | {ok} | {at} |")
    out.append("")
    return out


# Why a Helion kernel fails to produce a verified result on the CuteDSL backend.
CUTE_COMPILE_FAIL = {
    "mamba": "Does not compile — the sequential selective-scan recurrence raises `TypeError: Expected a TensorSSA or Numeric(Float), but got ArithValue` in CuTe codegen.",
    "megablocks": "Autotuning hangs — the jagged grouped-GEMM wedges in CuTe compilation (CPU-bound, no config ever benchmarked) and hits the wall-clock timeout.",
    "deformable": "Does not compile — the bilinear-sampling gather raises `BackendUnsupported: unresolved CuTe layout mismatch`.",
    "sage": "Does not compile — the INT8-quant step `torch.round` raises `InductorLoweringError` (the `cute` backend has no lowering for `aten.round.default`).",
}
CUTE_NUMERIC_FAIL = "Compiles and runs on cute, but the output fails `torch.allclose` vs the reference (numerically incorrect)."
CUTE_TIMEOUT_FAIL = "LLM autotuner did not converge within the 700 s per-shape budget on cute (this kernel's smaller shape(s) did)."


def build_cute_failures():
    """Enumerate every (task, size) that has a Triton result but no *verified*
    cute result — i.e. compile failures, numerically-wrong (ok=False) shapes,
    and autotune timeouts. Shapes come from the Triton run (every kernel ran
    there); reasons are the actual errors observed on cute."""
    rows = []  # (task, size, reason)
    for key, task, refname, url in KERNELS:
        tri = load("triton", key) or []
        cute = load("cute", key) or []
        cute_by_size = {r["size"]: r for r in cute}
        for tr in tri:
            size = tr["size"]
            cr = cute_by_size.get(size)
            if cr is not None:
                if not cr["ok"]:
                    rows.append((task, size, CUTE_NUMERIC_FAIL))
            elif key in CUTE_COMPILE_FAIL:
                rows.append((task, size, CUTE_COMPILE_FAIL[key]))
            elif key == "attention":
                rows.append((task, size, CUTE_TIMEOUT_FAIL))
    return rows


def render_failure_table(rows):
    out = ["## CuteDSL backend — failure cases", "",
           "Every (kernel, input size) that produces a **verified** result on the "
           "Triton backend but **not** on the CuteDSL backend, with the reason. "
           "Covers three modes: the kernel doesn't compile on `cute`, it compiles "
           "but is numerically wrong, or its autotune doesn't converge in budget. "
           "Full per-kernel context is in "
           "[`benchmark_results_cute.md`](benchmark_results_cute.md).", ""]
    if not rows:
        out += ["_No CuteDSL failures._", ""]
        return out
    out += ["| Task | Input size | Why the Helion CuteDSL kernel didn't work |",
            "|---|---|---|"]
    for task, size, reason in rows:
        out.append(f"| {task} | {size} | {reason} |")
    out.append("")
    return out


lines = [INTRO, ""]
triton_rows = build_rows("triton")
cute_rows = build_rows("cute")
lines += render_table("Aggregated benchmark results — Triton backend", triton_rows)
cute_failures = []
if cute_rows or os.path.isdir(f"{HH}/results/cute"):
    lines += render_table("Aggregated benchmark results — CuteDSL backend", cute_rows)
    cute_failures = build_cute_failures()
    lines += render_failure_table(cute_failures)

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}: {len(triton_rows)} triton rows, {len(cute_rows)} cute rows, "
      f"{len(cute_failures)} cute failures")
