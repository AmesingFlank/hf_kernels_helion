#!/usr/bin/env bash
# Run all Helion kernels through the CuteDSL backend with LLM-guided autotuning,
# one fresh process per (kernel, shape) so each autotune time is a real
# measurement. Mirrors the triton methodology; writes results/cute/<kernel>.json
# (in-repo, survives /tmp cleanup).
#
# Per-kernel timeout: most kernels autotune in ~300s; megablocks WEDGES in CuTe
# compilation (CPU-bound, never benchmarks a config), so it gets a short timeout
# to confirm-and-move-on instead of burning 15 min/shape.
set -u
cd /home/dev/hf_kernels_helion
export HELION_BACKEND=cute
LOG=/home/dev/hf_kernels_helion/cute_run.log   # in-repo (gitignored) so it survives /tmp cleanup
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

KERNELS=(activation causal_conv1d rotary paged mamba megablocks deformable tinygrad_rms rwkv layer_norm fp8 attention)
SHAPES=(small medium large)

timeout_for() {  # per-kernel autotune timeout (seconds)
  case "$1" in
    megablocks) echo 240 ;;   # known to wedge; just confirm
    *)          echo 700 ;;
  esac
}

for k in "${KERNELS[@]}"; do
  TMO=$(timeout_for "$k")
  for s in "${SHAPES[@]}"; do
    echo "########## cute $k $s (timeout ${TMO}s) ##########"
    REBENCH_SHAPE="$s" timeout "$TMO" ~/.venv/bin/python scripts/rebench_llm.py "$k" 2>&1 \
      | grep -E "helion=|WROTE|=== |Error|Traceback|BackendUnsupported|CompilationError|NotImplemented|ok=False|InvalidConfig" | tail -6
    echo "---- exit ${PIPESTATUS[0]} ----"
  done
done
echo "ALL_VENV_CUTE_DONE"
