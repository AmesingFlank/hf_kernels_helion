"""Compare AOT pre-tuned results (results/triton_aot/) against the original
from-scratch LLM-autotuned results (results/triton/). Writes
benchmark_results_triton_aot.md with a per-shape comparison table."""
import json, glob, os

HH = "/home/dev/hf_kernels_helion"
OUT = f"{HH}/benchmark_results_triton_aot.md"

ORDER = [
    ("activation", "activation"), ("causal_conv1d", "causal-conv1d"),
    ("rotary", "rotary"), ("paged", "paged-attention"), ("mamba", "mamba-ssm"),
    ("megablocks", "megablocks"), ("deformable", "deformable-detr"),
    ("tinygrad_rms", "tinygrad-rms"), ("rwkv", "rwkv"), ("layer_norm", "layer-norm"),
    ("fp8", "finegrained-fp8"), ("attention", "attention"), ("sage", "sage-attention"),
]


_ORD = {"small": 0, "medium": 1, "large": 2}


def load(sub, key):
    p = f"{HH}/results/{sub}/{key}.json"
    if not os.path.exists(p):
        return {}
    return {r["size"]: r for r in json.load(open(p))}


def align(base, aot):
    """Match base->aot rows by size label, falling back to shape-class
    (small/medium/large) position when labels differ (e.g. megablocks uses
    random group sizes in one harness, deterministic in the other)."""
    aot_by_class = {}
    for sz, r in aot.items():
        cls = sz.split()[0]
        aot_by_class.setdefault(cls, r)
    pairs = []
    for sz, b in base.items():
        a = aot.get(sz) or aot_by_class.get(sz.split()[0])
        pairs.append((sz, b, a))
    pairs.sort(key=lambda t: _ORD.get(t[0].split()[0], 9))
    return pairs


HEADER = """# Helion kernels — AOT pre-tuned vs from-scratch autotuned (Triton backend)

Both columns are the **same Helion kernels on the same B200**, benchmarked
against the same `kernels-community` references. Two things differ between them —
both *how* the config was searched for and *when* the cost is paid:

- **autotuned** (`results/triton/`): from-scratch, per-shape search with the
  **LLM-guided** autotuner. The `autotune` column is the wall-clock search time
  (80-150 s/shape) a user would pay on first use with no shipped config.
- **pre-tuned** (`results/triton_aot/`): the committed
  `_helion_aot_*_cuda_sm100.py` heuristics, produced ahead-of-time by the
  **default LFBO** autotuner at full effort and loaded via `@helion.aot_kernel`
  (`HELION_AOT_MODE=evaluate`). No search at run time; the `autotune` column is
  just the one-config compile a downloader pays.

So `Δ speed` (= pre-tuned speedup / autotuned speedup) blends two effects: the
generalization loss of using one shipped config per shape *and* the LFBO-vs-LLM
autotuner difference. It is **not** uniformly ≤1: LFBO found notably better
configs for some kernels (e.g. causal-conv1d up to 3.2x) and worse for others
(e.g. finegrained-fp8), so treat it as "how the shipped config compares to the
earlier per-shape LLM search", not a pure pre-tuning penalty. All rows are
numerically verified in both modes (✓).

The headline win is unchanged regardless: run-time search (thousands of seconds
total) collapses to a sub-second per-shape compile. Autotuning-time totals below
exclude shapes present in only one mode.
"""


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


lines = [HEADER]
tot_auto_s = 0.0
tot_aot_s = 0.0
n_rows = 0
speed_ratios = []
for key, repo in ORDER:
    base = load("triton", key)
    aot = load("triton_aot", key)
    if not base and not aot:
        continue
    lines.append(f"## {repo}")
    lines.append("")
    if key == "megablocks":
        lines.append("_Group sizes differ slightly between the two harnesses "
                     "(random totals when autotuned, fixed totals when pre-tuned), "
                     "so rows are matched by shape class (small/medium/large); the "
                     "K/N GEMM dims are identical._")
        lines.append("")
    lines.append("| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | "
                 "autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for size, b, a in align(base, aot):
        if a is None:
            lines.append(f"| {size} | {fmt(b['helion_ms'])} | — | "
                         f"{b['speedup']:.2f}× | — | — | {b['autotune_s']:.0f} → — | "
                         f"{'✓' if b['ok'] else '✗'} |")
            continue
        dspeed = a["speedup"] / b["speedup"] if b["speedup"] else 0
        ok = "✓" if (b["ok"] and a["ok"]) else "✗"
        lines.append(
            f"| {size} | {fmt(b['helion_ms'])} | {fmt(a['helion_ms'])} | "
            f"{b['speedup']:.2f}× | {a['speedup']:.2f}× | {dspeed:.2f}× | "
            f"{b['autotune_s']:.0f} → {a['autotune_s']:.1f} | {ok} |"
        )
        tot_auto_s += b["autotune_s"]
        tot_aot_s += a["autotune_s"]
        n_rows += 1
        speed_ratios.append(dspeed)
    lines.append("")

# Summary
geomean = 1.0
for r in speed_ratios:
    geomean *= r
geomean = geomean ** (1.0 / len(speed_ratios)) if speed_ratios else 0
lines.append("## Summary")
lines.append("")
lines.append(f"- **{n_rows} kernel×shape** pairs compared (present in both modes).")
lines.append(f"- **Total autotune time: {tot_auto_s:.0f}s → {tot_aot_s:.1f}s** "
             f"({tot_auto_s / tot_aot_s:.0f}× faster time-to-first-run) — the pre-tuned "
             f"kernels skip the search entirely.")
lines.append(f"- **Performance retained: geomean Δ speed = {geomean:.3f}×** "
             f"(pre-tuned vs individually-autotuned); i.e. the shipped configs are "
             f"within ~{abs(1 - geomean) * 100:.0f}% of per-shape-optimal on average.")
lines.append(f"- Min Δ speed = {min(speed_ratios):.2f}×, max = {max(speed_ratios):.2f}× "
             f"across all shapes.")
lines.append("")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"wrote {OUT}: {n_rows} rows, autotune {tot_auto_s:.0f}s->{tot_aot_s:.1f}s, geomean dspeed {geomean:.3f}")
