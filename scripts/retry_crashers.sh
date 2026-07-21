#!/usr/bin/env bash
# Retry the two kernels that hit the non-deterministic misaligned-address crash
# under LFBO (rwkv, causal_conv1d). Retry each up to 3 times until it produces a
# heuristic (exit 0). The crash is non-deterministic — a re-run usually clears it.
set -u
cd /home/dev/hf_kernels_helion
source scripts/aot_env_default.sh
LOG=/home/dev/hf_kernels_helion/aot_retry.log; : > "$LOG"
exec > >(tee -a "$LOG") 2>&1
for k in rwkv causal_conv1d; do
  for attempt in 1 2 3; do
    echo "###### retry $k attempt $attempt ######"
    timeout 900 ~/.venv/bin/python -m helion.autotuner.aot_runner \
      --phase all --goal max_slowdown --threshold 1.15 --max-configs 8 \
      -k "$k" -- ~/.venv/bin/python scripts/aot_tune.py "$k" \
      2>&1 | grep -E "Selected|Saved combined|misaligned|\[aot_tune\]" | tail -8
    rc=${PIPESTATUS[0]}
    echo "---- $k attempt $attempt exit $rc ----"
    ~/.venv/bin/python scripts/sync_aot_heuristics.py >/dev/null 2>&1
    [ "$rc" = "0" ] && { echo "$k SUCCESS on attempt $attempt"; break; }
  done
done
echo "###### RETRY_DONE ######"
