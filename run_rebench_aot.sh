#!/usr/bin/env bash
# Re-benchmark all triton kernels in AOT (pre-tuned) evaluate mode, per-shape
# fresh process. Fast now — no autotune search. Writes results/triton_aot/.
set -u
cd /home/dev/hf_kernels_helion
KERNELS=(activation rotary layer_norm causal_conv1d fp8 mamba paged megablocks tinygrad_rms rwkv deformable attention sage)
SHAPES=(small medium large)
LOG=/home/dev/hf_kernels_helion/rebench_aot.log; : > "$LOG"
exec > >(tee -a "$LOG") 2>&1
for k in "${KERNELS[@]}"; do
  for s in "${SHAPES[@]}"; do
    # sage benchmark needs the sageattention ref -> venv_torch_213; others use .venv
    PY=~/.venv/bin/python
    [ "$k" = "sage" ] && PY=~/venv_torch_213/bin/python
    REBENCH_AOT=1 REBENCH_SHAPE="$s" timeout 300 "$PY" rebench_llm.py "$k" 2>&1 \
      | grep -E "helion=|WROTE|Error|Traceback" | tail -3
  done
done
echo "REBENCH_AOT_DONE"
