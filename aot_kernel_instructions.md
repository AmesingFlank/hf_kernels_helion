# AOT (ahead-of-time) pre-tuned kernels

These Helion kernels ship with **pre-tuned configurations** so that when you
download and run them, they execute a known-good config immediately instead of
spending minutes autotuning on your machine. This document explains how that
works, how to use it, and how to add pre-tunings for new hardware.

## TL;DR

- **Users:** do nothing. Just call the kernel. If a pre-tuned config exists for
  your GPU it is used automatically (first call ≈ one compile, no autotuning).
  If not, the kernel falls back to normal Helion autotuning.
- **Maintainers, new GPU:** run the tuning sweep on that GPU and commit the
  generated `_helion_aot_<file>_<device>_<compute>.py` files (see
  [Adding pre-tunings for new hardware](#adding-pre-tunings-for-new-hardware)).

## Why AOT

A Helion kernel is generic tile code; the *config* (block sizes, num_warps,
num_stages, indexing strategy, …) that makes it fast is hardware- and
shape-specific and is normally discovered by **autotuning** — a search that can
take 1–5 minutes *per input shape* the first time a shape is seen. That cost is
fine for a library author tuning once, but unacceptable for a user who just
wants to run the kernel. AOT moves the search to *build time* (ours) and ships
the answer as a tiny lookup table (a "heuristic") next to the kernel.

## How it works

Each kernel is declared with the AOT decorator instead of the plain one:

```python
import helion
import helion.experimental

@helion.experimental.aot_kernel()          # instead of @helion.kernel()
def _silu_and_mul(out, x):
    ...
```

`@aot_kernel()` configures the kernel to:
- use `AOTAutotuneCache` (heuristic-based config selection),
- run with `static_shapes=False` (one heuristic serves many shapes),
- install a **shape-based specialization key** that, at runtime, is answered by
  a generated heuristic file if one is present.

At runtime Helion looks (via `find_heuristic_file`) for a file named

```
_helion_aot_<kernel-source-filename>_<device_kind>_<compute>.py
```

sitting **next to the kernel's source module**. For example the activation
kernel lives in `act.py`, so on a B200 (device `cuda`, compute `sm100`) the
file is `_helion_aot_act_cuda_sm100.py`. These files are committed in each
kernel's `torch-ext/<pkg>/` directory (source of truth) and copied into the
loadable `build/torch-cuda/` directory by `rebuild_noarch.py`.

A generated heuristic file is pure Python with no Helion dependency — it is just
a decision function plus a config table:

```python
def key__silu_and_mul(*args) -> int:
    # inspects arg shapes/dtypes, returns an index into the config list
    return 0

def autotune__silu_and_mul(*args) -> dict:
    _C = [ {'block_sizes': [1, 1024], 'num_warps': 4, 'num_stages': 1, ...} ]
    return _C[key__silu_and_mul(*args)]
```

**Compute-capability fallback.** Discovery tries the exact compute first, then
older compatible ones (`sm100 → sm90 → sm89 → …`). So an `sm90` heuristic will
be used on an `sm100` GPU if no `sm100` file exists — usually still a good
config, just not re-verified on the newer part.

**No heuristic present?** The kernel silently falls back to normal autotuning.
Nothing breaks; you just pay the one-time search cost.

## Using pre-tuned kernels (users)

Nothing to configure. `HELION_AOT_MODE` defaults to `evaluate`, which is the
"use the shipped heuristic" mode. Just:

```python
from kernels import get_kernel
act = get_kernel("kernels-community/activation-helion")
act.silu_and_mul(out, x)   # uses the pre-tuned config for your GPU + shape
```

To confirm a pre-tuned config is being used, time the first call: with a
heuristic it is ~sub-second (a single compile); without one it is tens of
seconds to minutes (a full autotune).

You can point Helion at an alternate directory of heuristic files with
`HELION_HEURISTIC_DIR=/path` (useful for A/B testing tunings).

## Adding pre-tunings for new hardware

The pre-tuned files are per-GPU. If you have a GPU for which no
`_helion_aot_*_<device>_<compute>.py` exists, generate them once:

All tooling lives under `scripts/`; run the commands below from the repo root.

1. **Environment.** Ensure `helion`, `torch`, and (for our sweep) Bedrock creds
   for LLM-guided autotuning are available. Our tuning env is captured in
   `scripts/aot_env.sh`.

2. **Run the sweep.** From the repo root:

   ```bash
   bash scripts/run_aot_all.sh              # all kernels
   # or one kernel via the runner directly:
   source scripts/aot_env.sh
   python -m helion.experimental.aot_runner \
       --phase all --goal max_slowdown --threshold 1.15 --max-configs 8 \
       -k activation -- python scripts/aot_tune.py activation
   ```

   The runner executes four phases (see [The workflow](#the-workflow-under-the-hood)),
   and the **build** phase writes `_helion_aot_<file>_<device>_<compute>.py`
   next to each kernel's `build/torch-cuda` source.

3. **Sync to source of truth.**

   ```bash
   python scripts/sync_aot_heuristics.py    # copies build/ heuristics -> torch-ext/
   ```

4. **Commit** the new `_helion_aot_*_<device>_<compute>.py` files. They live
   alongside the existing ones — different GPUs coexist (e.g.
   `_helion_aot_act_cuda_sm100.py` and `_helion_aot_act_cuda_sm90.py`).

### Adjusting which shapes are tuned

`scripts/aot_tune.py` defines, per kernel, the exact input shapes exercised
during tuning (they mirror the benchmark shapes in `scripts/rebench_llm.py`). To
tune for different shapes, edit the `run_<kernel>()` functions there. The heuristic
generalizes across shapes via a decision tree, but it is most reliable on (and
near) the shapes it was trained on. Prefer covering the shapes your workload
actually uses.

### Knobs

- `--max-configs N` — max distinct configs the heuristic may select between
  (more configs = better per-shape fit, larger table). We use 8.
- `--single-config` — force one config for all shapes (smallest table; good when
  shapes are similar or batch-only-varying).
- `--goal max_slowdown --threshold 1.15` — accept a heuristic whose worst-case
  shape is within 15% of that shape's individually-tuned optimum.
- `--backend decision_tree|nearest_neighbor` — decision tree (default) is
  compact; nearest-neighbor stores training shapes and matches at runtime.
- `batched=[[...]|None,...]` on the decorator — mark batch dims so different
  batch sizes share one config (excludes those dims from the key).

## The workflow under the hood

`python -m helion.experimental.aot_runner --phase all -- <cmd>` runs, driven by
the `HELION_AOT_MODE` env var it sets on the subprocess:

1. **collect** (`HELION_AOT_MODE=collect`) — autotune each distinct shape the
   benchmark exercises; save `tuned_configs_<hwid>.json`.
2. **measure** (`=measure`) — re-run, benchmarking every discovered config on
   every shape; save `measurements_<hwid>.csv`.
3. **build** — train a decision tree over the measurements and emit the
   `_helion_aot_*.py` heuristic next to each kernel source.
4. **evaluate** (`=evaluate`, also the default at deploy time) — validate the
   heuristic hits the perf goal, and confirm the deployed kernel loads it.

Intermediate artifacts land in `.helion_aot/<run-id>/` (gitignored). The only
files that must be committed for runtime use are the
`_helion_aot_<file>_<device>_<compute>.py` heuristics beside each kernel.

## Files in this repo

| File | Role |
|---|---|
| `<kernel>/…/torch-ext/<pkg>/*.py` | kernel source, decorated with `@aot_kernel` |
| `<kernel>/…/torch-ext/<pkg>/_helion_aot_*_<dev>_<compute>.py` | committed pre-tuned heuristic (source of truth) |
| `<kernel>/…/build/torch-cuda/…` | loadable copy (via `scripts/rebuild_noarch.py`); what `get_kernel` runs |
| `scripts/aot_tune.py` | per-kernel tuning-shape driver |
| `scripts/run_aot_all.sh` | sweep all kernels through the AOT runner |
| `scripts/sync_aot_heuristics.py` | copy generated heuristics `build/` → `torch-ext/` |
| `scripts/aot_env.sh` | autotuner env (LLM-guided, Bedrock) used during collect |

## Caveats

- Pre-tuned configs are **hardware-specific**. A config tuned on an A100 may be
  merely OK on an H100 and vice-versa; the compute-fallback picks the closest
  available, but re-tuning on the target GPU is best.
- Heuristics are **shape-generalized**, not shape-exact beyond the training set.
  On a wildly different shape the selected config may be suboptimal (still
  correct). Extend `aot_tune.py` and re-run to cover new shape regimes.
- These are **Triton-backend** tunings. The CuteDSL backend
  (`HELION_BACKEND=cute`) would need its own sweep and its own heuristic files.
