#!/usr/bin/env bash
# Build the large distillation set (diag_sibling_values.py build) in parallel shards:
# every saved-game collection except the held-out test ones, split into groups by
# logs/shared_diag/big_groups.json (see the shared-network diagnosis). Pinned to
# cores 40-63, niced; one process per group.
set -euo pipefail
cd "$(dirname "$0")/.."
groups=logs/shared_diag/big_groups.json
n=$(python3 -c "import json; print(len(json.load(open('$groups'))['groups']))")
for k in $(seq 0 $((n - 1))); do
  cols=$(python3 -c "import json; print(' '.join('--collection ' + c for c in json.load(open('$groups'))['groups'][$k]))")
  OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True taskset -c 40-63 nice -n 10 \
    uv run python scripts/diag_sibling_values.py build $cols --games 100000 --per-game "${PER_GAME:-5}" \
    --max-children "${MAX_CHILDREN:-32}" --no-played --seed $((${SEED_BASE:-100} + k)) --out "logs/shared_diag/${OUT:-big}/shard$k" \
    > "logs/shared_diag/${OUT:-big}/shard$k.log" 2>&1 &
done
wait
