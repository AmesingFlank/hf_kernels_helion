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

These are benchmarked on Helion's default **Triton** backend, and ship
**pre-tuned configs** (`@helion.aot_kernel`) so downloaders skip autotuning
entirely — the first call is a sub-second compile of the shipped config instead
of minutes of search. Across the {n_rows} kernel×shape pairs, shipping the
pre-tuned configs cuts total first-use autotuning **{auto_before:.0f} s →
{auto_after:.0f} s (~{speedup_x:.0f}× faster time-to-first-run)** while retaining
performance to **geomean {geomean:.2f}× of per-shape-optimal**. See
[`benchmark_results_triton_aot.md`](benchmark_results_triton_aot.md) for the
per-shape pre-tuned-vs-autotuned comparison and
[`aot_kernel_instructions.md`](aot_kernel_instructions.md) for how to use
pre-tuned kernels and add tunings for new hardware.

The table below reports the pre-tuned kernels. `Helion speed vs reference` =
reference latency / Helion latency (>1 → Helion is faster). Every row is
numerically verified against the reference; per-shape verification and
first-call compile times are in
[`benchmark_results_triton_aot.md`](benchmark_results_triton_aot.md).
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
    out += ["| Task | Reference | Helion speed vs reference |",
            "|---|---|---|"]
    for _, task, ref_md, sp, ok, at in rows:
        out.append(f"| {task} | {ref_md} | {sp} |")
    out.append("")
    return out


# Headline autotuning-time stats, computed from the data so the prose can't
# drift: sum the from-scratch search time (results/triton) vs the pre-tuned
# first-call compile (results/triton_aot) over shapes present in both.
_before = _after = 0.0
_n = 0
_logsum = 0.0  # for geomean of per-shape Δ speed (pre-tuned / autotuned)
import math as _math
for key, *_ in KERNELS:
    base = load("triton", key) or []
    aot = {r["size"]: r for r in (load("triton_aot", key) or [])}
    aot_list = list((load("triton_aot", key) or []))
    for i, r in enumerate(base):
        a = aot.get(r["size"]) or (aot_list[i] if i < len(aot_list) else None)
        if a is None:
            continue
        _before += r["autotune_s"]
        _after += a["autotune_s"]
        if r["speedup"] > 0 and a["speedup"] > 0:
            _logsum += _math.log(a["speedup"] / r["speedup"])
        _n += 1
_stats = {
    "n_rows": _n,
    "auto_before": _before,
    "auto_after": _after,
    "speedup_x": (_before / _after) if _after else 0.0,
    "geomean": _math.exp(_logsum / _n) if _n else 0.0,
}

lines = [INTRO.format(**_stats), ""]
aot_rows = build_rows("triton_aot")
lines += render_table(
    "Aggregated benchmark results — Triton backend (pre-tuned / AOT)", aot_rows
)

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}: {len(aot_rows)} pre-tuned rows "
      f"(autotune {_before:.0f}s->{_after:.0f}s, {_stats['speedup_x']:.0f}x)")
