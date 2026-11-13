#!/bin/bash
# Puts gen5's final checkpoint into the running advisor / bot services (the user, 2026-11-10 23:50: "when gen5 is
# done, be sure to update the running bot/advisor to use the new model"). In place: same unit names, ports, working
# directories, environment and memory limit; only the arguments naming gen4 change -- gen4's learner
# (rab_place_20261014/learner/1800.pth) -> gen5's learner, gen4's critic -> gen5's critic. Everything else stays:
# --value (value_v2/26, the win estimates), --auction-policy (SL ckpt 10, the human-like auction advice), and
# --play-value / --search-value where they name lr1's promoted/4.
#   swap_bots.sh <G5>      swap now to gen5 update G5
#   swap_bots.sh --wait    wait for gen5's plateau stop (STOP file + unit rl18xx-gen5 gone), then swap to its last
#                          snapshot; gen5 ending without a STOP file swaps nothing. Run as unit rl18xx-bot-swap.
# The old command lines go to swap_bots.log (rollback: re-run them with systemd-run as below).
set -uo pipefail
G=/home/revys/longrun/experiments/gen5_20261107
R5=/home/revys/longrun/work/lr1/model_checkpoints_pg/gen5_20261107
UNITS="rl18xx-bot-backend rl18xx-advisor rl18xx-bot-backend-next"
LOG=$G/swap_bots.log
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
state() { systemctl --user show -p ActiveState --value rl18xx-gen5; }

if [ "${1:-}" = --wait ]; then
  log "waiting for gen5's plateau stop"
  while :; do
    s=$(state)
    if [ "$s" = inactive ] || [ "$s" = failed ]; then
      sleep 30; s=$(state)
      if [ "$s" = inactive ] || [ "$s" = failed ]; then
        [ -f $R5/STOP ] && break
        log "gen5 ended ($s) without a STOP file; bots NOT swapped"; exit 1
      fi
    fi
    sleep 60
  done
  G5=$(ls $R5/learner/ | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | tail -1)
else
  G5=${1:?usage: swap_bots.sh <G5> | --wait}
fi
NEWL=$R5/learner/$G5.pth; NEWC=$R5/critic/$G5.pth
[ -f $NEWL ] && [ -f $NEWC ] || { log "no learner+critic pair for gen5 u$G5"; exit 1; }
log "swapping the advisor/bot services to gen5 u$G5"

fails=0
for u in $UNITS; do
  pid=$(systemctl --user show -p MainPID --value $u)
  if [ -z "$pid" ] || [ "$pid" = 0 ]; then log "$u: not running; skipped"; continue; fi
  wd=$(systemctl --user show -p WorkingDirectory --value $u)
  mapfile -d '' old < /proc/$pid/cmdline
  new=()
  for a in "${old[@]}"; do
    case "$a" in
      */rab_place_20261014/learner/1800.pth) new+=("$NEWL") ;;
      */rab_place_20261014/critic/1800.pth) new+=("$NEWC") ;;
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
