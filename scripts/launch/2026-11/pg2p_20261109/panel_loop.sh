#!/bin/bash
# Fixed-panel view (2026-11-09 19:30): each hundred >= 1100 vs u700, u900 and u1100 (policy-only, same settings as
# evals.sh), so progress is read against opponents that don't move. Added because far-back scores levelled off
# (u1300 vs u700 .902 < u1200 vs u700 .935) while each hundred kept beating the previous one. Ends with the PG run.
set -uo pipefail
X=/home/revys/longrun/experiments/pg2p_20261109
PANEL="700 900 1100"
while :; do
  did=0
  for f in $(ls $X/elo_done_* 2>/dev/null | sort -t_ -k3 -n); do
    u=${f##*_}; [ "$u" -ge 1100 ] || continue; [ -f $X/panel_done_$u ] && continue
    bs=""; for b in $PANEL; do [ "$b" -lt "$u" ] && bs="$bs $b"; done
    $X/transitivity.sh $u $bs
    python3 $X/panel.py > $X/panel.txt
    touch $X/panel_done_$u; did=1
  done
  [ $did = 1 ] && continue
  systemctl --user is-active -q rl18xx-pg2p || systemctl --user is-active -q rl18xx-pg2p-elo || break
  sleep 300
done
