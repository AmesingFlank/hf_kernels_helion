#!/usr/bin/env bash
# Run the AOT collect/measure/build/evaluate workflow for one kernel (arg1),
# using LLM-guided autotuning during collect. Writes heuristic next to the
# kernel's build/torch-cuda source. Logs in-repo (durable).
set -u
cd /home/dev/hf_kernels_helion
source aot_env.sh
K="$1"
LOG="/home/dev/hf_kernels_helion/aot_${K}.log"; : > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "###### AOT workflow: $K ######"
~/.venv/bin/python -m helion.experimental.aot_runner \
  --phase all --goal max_slowdown --threshold 1.15 --max-configs 8 \
  -k "$K" -- ~/.venv/bin/python aot_tune.py "$K"
echo "###### AOT $K exit $? ######"
