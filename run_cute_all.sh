#!/usr/bin/env bash
# Run all Helion kernels through the CuteDSL backend with LLM-guided autotuning,
# one fresh process per (kernel, shape) so each autotune time is a real
# measurement. Mirrors the triton methodology; writes /tmp/rebench_cute_<k>.json.
set -u
cd /home/dev/hf_kernels_helion
export HELION_BACKEND=cute

# 12 kernels that run under ~/.venv (torch 2.14). sage needs venv_torch_213 and
# is run separately by the caller.
KERNELS=(activation causal_conv1d rotary paged mamba megablocks deformable tinygrad_rms rwkv layer_norm fp8 attention)
SHAPES=(small medium large)

for k in "${KERNELS[@]}"; do
  for s in "${SHAPES[@]}"; do
    echo "########## cute $k $s ##########"
    REBENCH_SHAPE="$s" timeout 900 ~/.venv/bin/python rebench_llm.py "$k" 2>&1 \
      | grep -E "helion=|WROTE|=== |Error|Traceback|rc=|CompilationError|NotImplemented|ok=False" | tail -6
    echo "---- exit ${PIPESTATUS[0]} ----"
  done
done
echo "ALL_VENV_CUTE_DONE"
