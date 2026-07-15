"""One-time recovery: parse benchmark_results_triton.md back into
results/triton/<kernel>.json, so the triton source data lives in the repo (the
original /tmp JSONs were lost on session teardown). Idempotent.
"""
import json, re, os

HH = "/home/dev/hf_kernels_helion"
DOC = f"{HH}/benchmark_results_triton.md"
OUTDIR = f"{HH}/results/triton"
os.makedirs(OUTDIR, exist_ok=True)

# map "## <repo> — `<op>`" repo name -> json key + the kernel field used in rows
REPO_TO_KEY = {
    "activation": ("activation", "activation.silu_and_mul"),
    "causal-conv1d": ("causal_conv1d", "causal_conv1d"),
    "rotary": ("rotary", "rotary.apply_rotary"),
    "paged-attention": ("paged", "paged_attention_v1"),
    "mamba-ssm": ("mamba", "mamba.selective_scan"),
    "megablocks": ("megablocks", "megablocks.gmm"),
    "deformable-detr": ("deformable", "deformable.ms_deform_attn"),
    "tinygrad-rms": ("tinygrad_rms", "tinygrad_rms_norm"),
    "rwkv": ("rwkv", "rwkv.wkv"),
    "layer-norm": ("layer_norm", "layer_norm.rms_fwd"),
    "finegrained-fp8": ("fp8", "finegrained_fp8.w8a8_block"),
    "attention": ("attention", "attention.flash_attn_func"),
    "sage-attention": ("sage", "sage_attention.sageattn"),
}

text = open(DOC).read()
# split into sections at "## "
sections = re.split(r"\n## ", text)
row_re = re.compile(
    r"^\|\s*(.+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)×\s*\|\s*([\d.]+)\s*\|\s*([✓✗])\s*\|$"
)

written = 0
for sec in sections:
    header = sec.splitlines()[0] if sec else ""
    m = re.match(r"^(.+?) — ", header)
    if not m:
        continue
    repo = m.group(1).strip().lstrip("# ").strip()
    if repo not in REPO_TO_KEY:
        continue
    key, kernel_field = REPO_TO_KEY[repo]
    rows = []
    for line in sec.splitlines():
        rm = row_re.match(line.strip())
        if not rm:
            continue
        size, hms, rms, sp, at, ok = rm.groups()
        rows.append({
            "kernel": kernel_field,
            "size": size,
            "helion_ms": float(hms),
            "ref_ms": float(rms),
            "speedup": float(sp),
            "autotune_s": float(at),
            "ok": ok == "✓",
        })
    if rows:
        json.dump(rows, open(f"{OUTDIR}/{key}.json", "w"), indent=2)
        written += 1
        print(f"  {key}: {len(rows)} rows")

print(f"reconstructed {written} triton result files into {OUTDIR}")
