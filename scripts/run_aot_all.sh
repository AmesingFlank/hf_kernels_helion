#!/usr/bin/env bash
# Run the AOT collect/measure/build/evaluate workflow for every benchmarkable
# triton kernel, using LLM-guided autotuning during collect. Each kernel's
# heuristic is written next to its build/torch-cuda source, then synced to
# torch-ext. Logs in-repo (durable across /tmp wipes).
set -u
cd /home/dev/hf_kernels_helion
source scripts/aot_env.sh

# activation already done in the prototype; include it anyway for a clean sweep
KERNELS=(activation rotary layer_norm tinygrad_rms rwkv causal_conv1d fp8 mamba paged megablocks deformable attention sage)

LOG=/home/dev/hf_kernels_helion/aot_all.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

for k in "${KERNELS[@]}"; do
  echo "############## AOT workflow: $k ##############"
  # per-kernel timeout guard: megablocks/mamba can be slow; cap generously
  timeout 1800 ~/.venv/bin/python -m helion.experimental.aot_runner \
    --phase all --goal max_slowdown --threshold 1.15 --max-configs 8 \
    -k "$k" -- ~/.venv/bin/python scripts/aot_tune.py "$k" \
    2>&1 | grep -E "PHASE|Collected|Recorded|Saved combined|Selected|max_slowdown|PASS|FAIL|Traceback|Error|\[aot_tune\]" | tail -20
  echo "---- AOT $k exit ${PIPESTATUS[0]} ----"
  # sync any newly generated heuristic to torch-ext source of truth
  ~/.venv/bin/python scripts/sync_aot_heuristics.py 2>&1 | tail -2
done
echo "############## AOT_ALL_DONE ##############"
