#!/bin/bash
# Puts a title-generic 1830 checkpoint (a policy-gradient-generic run's learner + critic) into the running advisor /
# bot services in place (docs/rl_model.md "Serving generic checkpoints"), the pattern of gen6's swap_bots.sh /
# swap_head.sh (longrun/experiments/gen6_20261110): same unit names, ports, environment and memory limit. The code
# moves to DEPLOY -- a frozen deployment at or after the commit that added generic serving (`bash
# scripts/longrun/deploy.sh create <tag> <dest>`) -- and the four seat-playing checkpoints become generic:
#   --play-policy / --play-auction-policy -> generic:<learner>, --play-value / --play-auction-value -> generic:<critic>.
# Everything else stays (--play-value-head score, the advice networks --policy / --auction-policy / --value /
# --search-value, which stay legacy, --bots, ports, ...). --policy-only [T] makes the seats play the learner's own move
# without searching (--play-mode policy --play-temperature T, default 0: the most likely move and price).
#
# Dry run by default: prints each unit's old and new command line. --apply swaps; old command lines -> $LOG
# (rollback: re-run them with systemd-run as below, or swap_head.sh's way).
#   swap_bots_generic.sh <deploy dir> <learner.pth> <critic.pth> [--policy-only [T]] [--apply]
# SWAP_TEST_CMDLINE="<a command line>" prints its rewrite and exits (no systemd; tests the rewrite offline).
set -uo pipefail
usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 3 ] || usage
DEPLOY=$(realpath -m "$1"); LEARNER=$(realpath -m "$2"); CRITIC=$(realpath -m "$3"); shift 3
MODE=search; TEMPERATURE=0; APPLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --policy-only) MODE=policy
      if [[ ${2:-} =~ ^[0-9.]+$ ]]; then TEMPERATURE=$2; shift; fi ;;
    --apply) APPLY=1 ;;
    *) usage ;;
  esac
  shift
done
UNITS=${UNITS:-"rl18xx-bot-backend rl18xx-advisor rl18xx-bot-backend-next"}

# old command line (python main.py advisor ...) -> new: the deployment's python, the generic play checkpoints.
rewrite() {
  local -a old=("$@") out=("$DEPLOY/.venv/bin/python")
  local skip=0 a
  for a in "${old[@]:1}"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case "$a" in
      --play-policy|--play-value|--play-auction-policy|--play-auction-value|--play-mode|--play-temperature) skip=1 ;;
      --play-policy=*|--play-value=*|--play-auction-policy=*|--play-auction-value=*|--play-mode=*|--play-temperature=*) ;;
      *) out+=("$a") ;;
    esac
  done
  out+=(--play-policy "generic:$LEARNER" --play-value "generic:$CRITIC")
  out+=(--play-auction-policy "generic:$LEARNER" --play-auction-value "generic:$CRITIC")
  [ "$MODE" = policy ] && out+=(--play-mode policy --play-temperature "$TEMPERATURE")
  printf '%s\0' "${out[@]}"
}

if [ -n "${SWAP_TEST_CMDLINE:-}" ]; then
  read -r -a test_old <<< "$SWAP_TEST_CMDLINE"
  rewrite "${test_old[@]}" | tr '\0' ' '; echo
  exit 0
fi

LOG=${LOG:-$HOME/longrun/experiments/swap_bots_generic.log}
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
[ -f "$LEARNER" ] && [ -f "$CRITIC" ] || { echo "no learner/critic at $LEARNER / $CRITIC" >&2; exit 1; }
grep -q evaluates_games "$DEPLOY/rl18xx/agent/alphazero/rust_mcts_player.py" 2>/dev/null \
  || { echo "$DEPLOY has no generic serving (rust_mcts_player.py without evaluates_games)" >&2; exit 1; }
bash "$DEPLOY/scripts/longrun/deploy.sh" verify "$DEPLOY" > /dev/null 2>&1 || { echo "$DEPLOY does not verify" >&2; exit 1; }
[ $APPLY = 1 ] && log "swapping the advisor/bot services to generic $LEARNER + $CRITIC ($MODE) on $DEPLOY"

fails=0
for u in $UNITS; do
  pid=$(systemctl --user show -p MainPID --value "$u")
  if [ -z "$pid" ] || [ "$pid" = 0 ]; then echo "$u: not running; skipped"; continue; fi
  mapfile -d '' old < "/proc/$pid/cmdline"
  mapfile -d '' new < <(rewrite "${old[@]}")
  port=$(printf '%s\n' "${old[@]}" | grep -A1 -x -- --port | tail -1)
  if [ $APPLY = 0 ]; then
    printf '%s (port %s)\n  old: %s\n  new: %s\n' "$u" "$port" "${old[*]}" "${new[*]}"
    continue
  fi
  log "$u (port $port) old: ${old[*]}"
  systemctl --user stop "$u"; systemctl --user reset-failed "$u" 2>/dev/null
  systemd-run --user --unit "$u" -p MemoryMax=12G -p WorkingDirectory="$DEPLOY" \
    -E CUDA_VISIBLE_DEVICES= -E OMP_NUM_THREADS=4 -E PYTHONDONTWRITEBYTECODE=1 "${new[@]}" >> "$LOG" 2>&1
  ok=""
  for _ in $(seq 90); do
    sleep 5
    h=$(curl -s -m 5 -H 'Origin: chrome-extension://abcdefghijklmnop' "http://127.0.0.1:$port/api/health")
    if [[ $h == *'"ok":true'* && $h == *'generic:'* ]]; then ok=$h; break; fi
    systemctl --user is-active -q "$u" || break
  done
  if [ -n "$ok" ]; then log "$u up: $ok"; else log "$u: NOT healthy ($(systemctl --user is-active "$u"))"; fails=$((fails + 1)); fi
done
[ $APPLY = 1 ] && log "swap done ($fails failed)"
exit $fails
