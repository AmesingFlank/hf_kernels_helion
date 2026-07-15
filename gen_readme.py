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


lines = [INTRO, ""]
triton_rows = build_rows("triton")
cute_rows = build_rows("cute")
lines += render_table("Aggregated benchmark results — Triton backend", triton_rows)
if cute_rows or os.path.isdir(f"{HH}/results/cute"):
    lines += render_table("Aggregated benchmark results — CuteDSL backend", cute_rows)
    lines += [
        "> On the CuteDSL backend, some kernels do not yet compile "
        "(`mamba-ssm`, `megablocks`, `deformable-detr`, `sage-attention`) and so "
        "have no rows, and `paged-attention` / `finegrained-fp8` compile but do "
        "not match the reference numerically (verified ✗). "
        "See [`benchmark_results_cute.md`](benchmark_results_cute.md) for details.",
        "",
    ]

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}: {len(triton_rows)} triton rows, {len(cute_rows)} cute rows")
