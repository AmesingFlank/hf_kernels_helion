#!/usr/bin/env bash
#
# Pre-autotune ALL Helion kernels in this repo using the LLM-guided autotuner
# (Bedrock / Claude Haiku), at FULL effort, 7 shapes per kernel.
#
#   bash scripts/pretune_all_bedrock_llm.sh              # all kernels
#   bash scripts/pretune_all_bedrock_llm.sh rwkv fp8     # only these kernels
#   EFFORT=quick bash scripts/pretune_all_bedrock_llm.sh # faster, lower quality
#
# It exports the LLM autotuner env vars and calls scripts/aot_tune.py (which
# itself reads HELION_AUTOTUNER from the environment, so the autotuner choice
# lives here, not in the Python). Heuristics are written next to each kernel as
# _helion_aot_<file>_<device>_<compute>.py (this machine's GPU) and synced into
# torch-ext/. A full log is written to pretune_llm.log at the repo root.
#
# Prereqs: activate the venv first (has torch/helion/kernels + boto3), and have
# AWS credentials available (ambient IAM role or env) for Bedrock.
set -euo pipefail

# --- repo root (this script lives in scripts/) --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# --- LLM-guided autotuner (Bedrock) -------------------------------------------
export HELION_AUTOTUNER=LLMGuidedSearch
export HELION_LLM_PROVIDER=bedrock
export HELION_LLM_MODEL="${HELION_LLM_MODEL:-us.anthropic.claude-haiku-4-5-20251001-v1:0}"
export AWS_REGION="${AWS_REGION:-us-east-2}"
# Operational settings for AOT-tuning get_local_kernel modules (aot_tune.py also
# sets these as defaults, but be explicit so this script is self-contained).
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0
export HELION_AUTOTUNE_IGNORE_ERRORS=1

EFFORT="${EFFORT:-full}"          # full (default) | quick
PY="${PY:-python}"                # honor an explicit interpreter if set

# --- preflight ----------------------------------------------------------------
echo "=== preflight ==="
command -v "$PY" >/dev/null || { echo "ERROR: '$PY' not found — activate your venv first."; exit 1; }
"$PY" - <<'PYCHECK' || { echo "ERROR: environment not ready (see above)."; exit 1; }
import sys
for mod in ("torch", "helion", "kernels", "boto3"):
    __import__(mod)
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"  ok: torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
PYCHECK
# Bedrock needs AWS creds; warn early rather than failing mid-sweep.
"$PY" - <<'AWSCHECK' || echo "  WARN: could not confirm AWS credentials; Bedrock calls may fail."
import boto3
c = boto3.client("sts")
print("  ok: AWS identity", c.get_caller_identity()["Arn"])
AWSCHECK

echo "=== launching LLM (Bedrock) pre-tuning: effort=$EFFORT, autotuner=$HELION_AUTOTUNER ==="
LOG="$REPO_ROOT/pretune_llm.log"
# tee everything to a durable in-repo log; aot_tune.py prints per-kernel progress.
"$PY" scripts/aot_tune.py --effort "$EFFORT" "$@" 2>&1 | tee "$LOG"

echo
echo "=== done. Commit the generated heuristics: ==="
echo "    git add '*/_helion_aot_*.py' && git commit -m 'AOT: LLM-tuned configs for <gpu>' && git push"
