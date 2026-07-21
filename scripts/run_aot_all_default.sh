#!/usr/bin/env bash
# Re-pretune all benchmarkable triton HF kernels with Helion's DEFAULT autotuner
# (LFBOTreeSearch, full effort) — NOT the LLM autotuner. Heuristics land next to
# each kernel's build/torch-cuda source, then get synced to torch-ext.
set -u
cd /home/dev/hf_kernels_helion
source scripts/aot_env_default.sh

KERNELS=(activation rotary layer_norm tinygrad_rms rwkv causal_conv1d fp8 mamba paged megablocks deformable attention sage)

LOG=/home/dev/hf_kernels_helion/aot_all_default.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

for k in "${KERNELS[@]}"; do
  echo "############## AOT (default LFBO, full) workflow: $k ##############"
  timeout 3600 ~/.venv/bin/python -m helion.autotuner.aot_runner \
    --phase all --goal max_slowdown --threshold 1.15 --max-configs 8 \
    -k "$k" -- ~/.venv/bin/python scripts/aot_tune.py "$k" \
    2>&1 | grep -E "PHASE|Collected|Recorded|Saved combined|Selected|max_slowdown|PASS|FAIL|Traceback|Error|\[aot_tune\]" | tail -20
  echo "---- AOT $k exit ${PIPESTATUS[0]} ----"
  ~/.venv/bin/python scripts/sync_aot_heuristics.py 2>&1 | tail -2
done
echo "############## AOT_ALL_DEFAULT_DONE ##############"
