#!/usr/bin/env bash
cd /home/dev/hf_kernels_helion
while pgrep -f "run_aot_all_default.sh" >/dev/null 2>&1; do
  sleep 180
  ~/.venv/bin/python scripts/sync_aot_heuristics.py >/dev/null 2>&1
  if [ -n "$(git status --porcelain '*/_helion_aot_*.py' 2>/dev/null)" ]; then
    git add '*/_helion_aot_*.py' 2>/dev/null
    git -c user.email=dev@local -c user.name=dev commit -qm "AOT (default LFBO) heuristics checkpoint" 2>/dev/null \
      && git push origin main 2>/dev/null && echo "pushed checkpoint"
  fi
done
echo "=== DEFAULT AOT SWEEP ENDED ==="
~/.venv/bin/python scripts/sync_aot_heuristics.py 2>&1 | tail -2
