#!/usr/bin/env bash
set -u
cd /home/dev/hf_kernels_helion
export HELION_BACKEND=cute
LOG=/home/dev/hf_kernels_helion/cute_sage.log; : > "$LOG"
exec > >(tee -a "$LOG") 2>&1
for s in small medium large; do
  echo "########## cute sage $s ##########"
  REBENCH_SHAPE="$s" timeout 700 ~/venv_torch_213/bin/python rebench_llm.py sage 2>&1 \
    | grep -E "helion=|WROTE|=== |Error|Traceback|BackendUnsupported|ok=False|InvalidConfig" | tail -6
  echo "---- exit ${PIPESTATUS[0]} ----"
done
echo "SAGE_CUTE_DONE"
