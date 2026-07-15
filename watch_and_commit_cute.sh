#!/usr/bin/env bash
# While the cute run is alive, periodically commit any new results/cute/*.json
# so progress survives session teardown. On completion, do a final commit.
cd /home/dev/hf_kernels_helion
while pgrep -f "run_cute_all.sh" >/dev/null 2>&1; do
  sleep 180
  if ! git diff --quiet results/cute 2>/dev/null || [ -n "$(git status --porcelain results/cute)" ]; then
    git add results/cute 2>/dev/null
    git -c user.email=dev@local -c user.name=dev commit -qm "cute results: checkpoint $(ls results/cute | wc -l) kernels" 2>/dev/null \
      && git push origin main 2>/dev/null && echo "checkpoint pushed: $(ls results/cute)"
  fi
done
echo "=== CUTE RUN ENDED ==="
git add results/cute 2>/dev/null
git -c user.email=dev@local -c user.name=dev commit -qm "cute results: final $(ls results/cute | wc -l) kernels" 2>/dev/null \
  && git push origin main 2>/dev/null && echo "final commit pushed"
echo "=== summary ==="
grep -E "helion=|ALL_VENV" cute_run.log | tail -40
for f in results/cute/*.json; do
  [ -f "$f" ] && echo "$(basename $f): $(~/.venv/bin/python -c "import json;print(len(json.load(open('$f'))))" 2>/dev/null) rows"
done
