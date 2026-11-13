#!/bin/bash
# Puts a gen6 checkpoint into the running advisor / bot services in place (the user, 2026-11-12: "just do u2400").
# Same unit names, ports, working directories, environment and memory limit as the gen5 swap (gen5_20261107/swap_bots.sh);
# only arguments naming gen5 u2800 change: its learner -> gen6's learner, its critic -> gen6's critic. --value (value_v2/26),
# --auction-policy (SL ckpt 10) and lr1 promoted/4 (--play-value / --search-value where set) stay.
#   swap_bots.sh <update>     old command lines -> swap_bots.log (rollback: re-run them with systemd-run as below)
set -uo pipefail
G=/home/revys/longrun/experiments/gen6_20261110
R6=/home/revys/longrun/work/lr1/model_checkpoints_pg/gen6_20261110
UNITS="rl18xx-bot-backend rl18xx-advisor rl18xx-bot-backend-next"
LOG=$G/swap_bots.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
U6=${1:?usage: swap_bots.sh <gen6 update>}
NEWL=$R6/learner/$U6.pth; NEWC=$R6/critic/$U6.pth
[ -f $NEWL ] && [ -f $NEWC ] || { log "no learner+critic pair for gen6 u$U6"; exit 1; }
log "swapping the advisor/bot services from gen5 u2800 to gen6 u$U6"

fails=0
for u in $UNITS; do
  pid=$(systemctl --user show -p MainPID --value $u)
  if [ -z "$pid" ] || [ "$pid" = 0 ]; then log "$u: not running; skipped"; continue; fi
  wd=$(systemctl --user show -p WorkingDirectory --value $u)
  mapfile -d '' old < /proc/$pid/cmdline
  new=()
  for a in "${old[@]}"; do
    case "$a" in
      */gen5_20261107/learner/2800.pth) new+=("$NEWL") ;;
      */gen5_20261107/critic/2800.pth) new+=("$NEWC") ;;
      *) new+=("$a") ;;
    esac
  done
  port=$(printf '%s\n' "${old[@]}" | grep -A1 -x -- --port | tail -1)
  log "$u (port $port, $wd) old: ${old[*]}"
  systemctl --user stop $u; systemctl --user reset-failed $u 2>/dev/null
  systemd-run --user --unit $u -p MemoryMax=12G -p WorkingDirectory=$wd \
    -E CUDA_VISIBLE_DEVICES= -E OMP_NUM_THREADS=4 -E PYTHONDONTWRITEBYTECODE=1 "${new[@]}" >> $LOG 2>&1
  ok=""
  for _ in $(seq 60); do
    sleep 5
    h=$(curl -s -m 5 -H 'Origin: chrome-extension://abcdefghijklmnop' http://127.0.0.1:$port/api/health)
    if [[ $h == *'"ok":true'* ]]; then ok=$h; break; fi
    systemctl --user is-active -q $u || break
  done
  if [ -n "$ok" ]; then log "$u up: $ok"
  else log "$u: NOT healthy after the swap ($(systemctl --user is-active $u))"; fails=$((fails + 1)); fi
done
log "swap done ($fails failed)"
exit $fails
