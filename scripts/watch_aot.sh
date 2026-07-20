#!/usr/bin/env bash
cd /home/dev/hf_kernels_helion
while pgrep -f "run_aot_all.sh" >/dev/null 2>&1; do
  sleep 180
  ~/.venv/bin/python scripts/sync_aot_heuristics.py >/dev/null 2>&1
  if [ -n "$(git status --porcelain '*/torch-ext/*/_helion_aot_*.py' '*/build/torch-cuda/_helion_aot_*.py' 2>/dev/null)" ]; then
    git add '*/_helion_aot_*.py' 2>/dev/null
    git -c user.email=dev@local -c user.name=dev commit -qm "AOT heuristics checkpoint ($(ls */*/torch-ext/*/_helion_aot_*.py 2>/dev/null | wc -l) kernels)" 2>/dev/null \
      && git push origin main 2>/dev/null && echo "pushed checkpoint"
  fi
done
echo "=== AOT SWEEP ENDED ==="
~/.venv/bin/python scripts/sync_aot_heuristics.py 2>&1 | tail -3
git add -A '*/_helion_aot_*.py' 2>/dev/null
git -c user.email=dev@local -c user.name=dev commit -qm "AOT heuristics: final sweep" 2>/dev/null && git push origin main 2>/dev/null
echo "heuristics generated:"; ls */*/torch-ext/*/_helion_aot_*.py 2>/dev/null | wc -l
