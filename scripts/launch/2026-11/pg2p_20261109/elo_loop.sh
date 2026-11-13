#!/bin/bash
# For each new hundred U (after evals.sh has measured it): U vs U-200, U-300 and U-600 (transitivity.sh), then the Elo
# fit (elo.py) appended to elo.txt. Ends with the PG run. Started 2026-11-09 13:50.
set -uo pipefail
X=/home/revys/longrun/experiments/pg2p_20261109
while :; do
  u=$(grep -o '^policy-only u[0-9]*00_vs_u' $X/results.txt | grep -o '[0-9]*' | sort -n | tail -1)
  if [ -n "$u" ] && [ ! -f $X/elo_done_$u ] && [ "$u" -ge 1000 ]; then
    bs=""; for k in 200 300 600; do [ $((u - k)) -ge 0 ] && bs="$bs $((u - k))"; done
    $X/transitivity.sh $u $bs
    { date '+%F %T'; python3 $X/elo.py; } >> $X/elo.txt
    touch $X/elo_done_$u
  else
    systemctl --user is-active -q rl18xx-pg2p || break
    sleep 300
  fi
done
