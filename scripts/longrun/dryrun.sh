#!/bin/bash
# The long-run harness end to end on a tiny CPU search-learning run (docs/longrun_runbook.md, "Dry run").
#
#   scripts/longrun/dryrun.sh <deployment> <scratch dir>     [CPUS=24-31,56-63] [HUMAN_GAMES=<dir>]
#
# Installs three instances under the user's systemd (lr-dryrun, lr-dryrun-nan, lr-dryrun-oom), checks each
# claim below, prints PASS / FAIL lines, and uninstalls them again (their files stay in the scratch dir):
#   1  the first start is fresh; the run reaches snapshots
#   2  kill -9 of main.py: every process of the run goes, systemd restarts it, it resumes from the snapshot
#   3  kill -9 of a spawned child (an inference server): the same
#   4  no history line for STALL_MINUTES: the watchdog alerts, and clears when updates resume
#   5  deploy.sh update is refused while the run uses the deployment
#   6  systemctl stop leaves no process behind
#   7  a modified deployment: the start is refused (76), not restarted, alerted
#   8  a non-finite loss: exit 75, not restarted, alerted
#   9  over MemoryMax: OOM-killed inside its cgroup, restarted, then the start limit stops it; the user's
#      session and other services are untouched; the watchdog reports the crash loop
#  10  the watchdog's disk / GPU thresholds and ALERT_COMMAND
# No GPU (CUDA_VISIBLE_DEVICES is empty); everything runs pinned to $CPUS at nice 19.
set -uo pipefail

DEPLOY=$(realpath "${1:?deployment}")
BASE=$(realpath -m "${2:?scratch dir}")
CPUS=${CPUS:-24-31,56-63}
HERE=$DEPLOY/scripts/longrun
REPO=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_repo"])' "$DEPLOY/DEPLOYMENT.json")
HUMAN_GAMES=${HUMAN_GAMES:-$REPO/human_games}
WORK=$BASE/work
RUN=model_checkpoints_search/dryrun
INSTANCES=(lr-dryrun lr-dryrun-nan lr-dryrun-oom)
failures=0

log() { echo "$(date +%T) $*"; }
pass() { log "PASS  $*"; }
fail() { log "FAIL  $*"; failures=$((failures + 1)); }
check() { local what=$1; shift; if "$@"; then pass "$what"; else fail "$what"; fi; }
wait_for() {  # wait_for <seconds> <command...>: until the command succeeds
    local deadline=$(( $(date +%s) + $1 )); shift
    until "$@"; do [ "$(date +%s)" -ge "$deadline" ] && return 1; sleep 1; done
}
prop() { systemctl --user show "rl18xx-longrun@$1.service" -p "$2" --value; }
main_py() { pgrep -P "$(prop "$1" MainPID)" -f "main.py search-learning" | head -1; }
cgroup_procs() { cat "/sys/fs/cgroup$(prop "$1" ControlGroup)/cgroup.procs" 2>/dev/null; }
ident() { for p in "$@"; do echo "$p:$(awk '{print $22}' "/proc/$p/stat" 2>/dev/null)"; done; }  # pid:start time
alive() { local gone=0; for id in "$@"; do [ "$(ident "${id%%:*}")" = "$id" ] && gone=1; done; [ $gone -eq 1 ]; }
snapshot() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot_update"])' "$WORK/$RUN/state.json" 2>/dev/null || echo -1; }
last_update() { tail -1 "$WORK/$RUN/history.jsonl" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read() or "{}").get("update", -1))'; }
starts() { cat "$WORK/longrun/$1/starts.jsonl" 2>/dev/null | wc -l; }
alert_logged() { grep -q -- "$2" "$WORK/longrun/$1/alerts.log" 2>/dev/null; }

TRAIN="--workers 2 --games-per-worker 2 --task-seconds 3 --n-full 8 --n-fast 0 --p-full 0.5 --parallel-readouts 4 \
--max-decisions 40 --inference-batch-size 64 --window-rows 400 --rows-per-update 16 --min-window-rows 16 \
--steps-per-update 1 --minibatch 8 --microbatch 4 --snapshot-every 5 --anchor-every 0 --max-updates 100000 \
--critic-positions-per-game 0 --server-replicas 2 --dedupe-maps --reuse-tree total --midgame-fraction 0.3 \
--auction-fraction 0.15"
START="--policy model_checkpoints_pg/tiny_start/learner/0.pth --value model_checkpoints_pg/tiny_start/critic/0.pth"

install_instance() {  # install_instance <name> <run dir> <train args> [install.sh options...]
    local name=$1 run_dir=$2 train=$3; shift 3
    bash "$HERE/install.sh" "$name" --deploy "$DEPLOY" --workdir "$WORK" --run-dir "$run_dir" \
        --start-args "$START" --train-args "$train" --cpus "$CPUS" --nice 19 --restart-sec 10 \
        --memory-high 12G --memory-max 16G --set CUDA_VISIBLE_DEVICES= --set WATCH_INTERVAL_SEC=5 \
        --set STALL_MINUTES=1 --set GPU_CHECK=0 --set "ALERT_COMMAND=echo \$ALERT_LEVEL \$ALERT_KEY >> $WORK/alert_command.out" \
        "$@" > /dev/null
}

cleanup() {
    for name in "${INSTANCES[@]}"; do
        systemctl --user stop "rl18xx-longrun@$name.service" "rl18xx-longrun-watchdog@$name.service" 2>/dev/null
        bash "$HERE/install.sh" --uninstall "$name" > /dev/null 2>&1
    done
}
trap cleanup EXIT

log "dry run: deployment $DEPLOY ($(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["commit"][:10], d["ref"])' "$DEPLOY/DEPLOYMENT.json")), scratch $BASE"
mkdir -p "$WORK/logs"
ln -sfn "$HUMAN_GAMES" "$WORK/human_games"
[ -f "$WORK/model_checkpoints_pg/tiny_start/learner/0.pth" ] \
    || CUDA_VISIBLE_DEVICES= taskset -c "$CPUS" nice -n 19 "$DEPLOY/.venv/bin/python" -s "$HERE/make_tiny_start.py" \
        "$WORK/model_checkpoints_pg/tiny_start" > /dev/null

# ----------------------------------------------------------------------------- 1: fresh start
install_instance lr-dryrun "$RUN" "$TRAIN"
systemctl --user start rl18xx-longrun@lr-dryrun.service
wait_for 180 bash -c "[ \$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"snapshot_update\"])' $WORK/$RUN/state.json 2>/dev/null || echo 0) -ge 5 ]"
check "1 first start is fresh" grep -q '"mode": "fresh"' "$WORK/longrun/lr-dryrun/starts.jsonl"
check "1 reached snapshot $(snapshot) (update $(last_update))" [ "$(snapshot)" -ge 5 ]
check "1 watchdog started with the run" systemctl --user is-active --quiet rl18xx-longrun-watchdog@lr-dryrun.service
log "     processes in the unit: $(cgroup_procs lr-dryrun | wc -l) ($(systemd-cgls --user-unit rl18xx-longrun@lr-dryrun.service --no-pager 2>/dev/null | grep -c spawn_main) spawned)"

# ----------------------------------------------------------------------------- 2, 3: kills
kill_and_resume() {  # kill_and_resume <label> <pid to kill>
    local label=$1 victim=$2 before restarts snap ids
    mapfile -t ids < <(ident $(cgroup_procs lr-dryrun))
    before=$(starts lr-dryrun); restarts=$(prop lr-dryrun NRestarts)
    kill -9 "$victim"
    sleep 0.5
    snap=$(snapshot)  # (read after the kill: it can't move any more)
    log "     $label: killed $victim (kill -9) at update $(last_update), snapshot $snap; ${#ids[@]} processes in the unit"
    # Every process of the old invocation goes before the new one starts (RestartSec=10).
    wait_for 30 bash -c "! systemctl --user is-active --quiet rl18xx-longrun@lr-dryrun.service" || true
    if wait_for 30 eval '! alive "${ids[@]}"'; then pass "$label: all ${#ids[@]} processes of the killed run exited"
    else fail "$label: survivors $(for i in "${ids[@]}"; do alive "$i" && echo -n "${i%%:*} "; done)"; fi
    wait_for 120 bash -c "[ \$(cat $WORK/longrun/lr-dryrun/starts.jsonl | wc -l) -gt $before ]"
    check "$label: systemd restarted it (NRestarts $restarts -> $(prop lr-dryrun NRestarts))" [ "$(prop lr-dryrun NRestarts)" -gt "$restarts" ]
    local rec; rec=$(tail -1 "$WORK/longrun/lr-dryrun/starts.jsonl")
    check "$label: resumed from snapshot $snap ($rec)" python3 -c "import json,sys; r=json.loads(sys.argv[1]); sys.exit(not (r['mode']=='resume' and r['snapshot_update']==$snap))" "$rec"
    local mark; mark=$(wc -l < "$WORK/$RUN/history.jsonl")
    wait_for 120 bash -c "[ \$(wc -l < $WORK/$RUN/history.jsonl) -gt $mark ]"
    local next; next=$(sed -n "$((mark + 1))p" "$WORK/$RUN/history.jsonl" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["update"])')
    check "$label: history continues at update $next = snapshot + 1" [ "$next" -eq $((snap + 1)) ]
}
sleep 3
kill_and_resume "2 kill main.py" "$(main_py lr-dryrun)"
wait_for 120 bash -c "[ \$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"snapshot_update\"])' $WORK/$RUN/state.json) -gt $(snapshot) ]"
child=$(pgrep -P "$(main_py lr-dryrun)" -f spawn_main | sort -n | head -1)  # the first spawned: an inference server
kill_and_resume "3 kill a spawned child" "$child"
check "2-3 the watchdog alerted on the crashes" alert_logged lr-dryrun "WARN \[lr-dryrun\] crash"

# ----------------------------------------------------------------------------- 4: stall
main=$(main_py lr-dryrun)
log "     4: SIGSTOP main.py ($main) for 100 s (STALL_MINUTES=1)"
kill -STOP "$main"
wait_for 110 alert_logged lr-dryrun "WARN \[lr-dryrun\] stall"
check "4 stall alert" alert_logged lr-dryrun "WARN \[lr-dryrun\] stall"
kill -CONT "$main"
wait_for 90 alert_logged lr-dryrun "stall: cleared"
check "4 stall cleared once updates resumed" alert_logged lr-dryrun "stall: cleared"

# ----------------------------------------------------------------------------- 5, 6: in use, stop
out=$(bash "$HERE/deploy.sh" update "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$DEPLOY/DEPLOYMENT.json")" "$DEPLOY" --allow-untagged 2>&1)
check "5 deploy.sh update refused while running ($(echo "$out" | head -1 | cut -c1-90))" bash -c "[[ \"$out\" == *'refusing to update'* ]]"
mapfile -t ids < <(ident $(cgroup_procs lr-dryrun))
systemctl --user stop rl18xx-longrun@lr-dryrun.service
check "6 stopped: unit inactive" [ "$(prop lr-dryrun ActiveState)" = inactive ]
check "6 stopped: none of its ${#ids[@]} processes left" eval '! alive "${ids[@]}"'
left=$(bash "$HERE/deploy.sh" in-use "$DEPLOY"; pgrep -f "$WORK" | grep -v $$)
check "6 stopped: nothing uses the deployment or the work dir${left:+ ($left)}" [ -z "$left" ]
check "6 the stop was recorded" grep -q '"event": "stop"' "$WORK/longrun/lr-dryrun/exits.jsonl"

# ----------------------------------------------------------------------------- 7: modified deployment
victim=$DEPLOY/rl18xx/agent/alphazero/search_learning.py
chmod u+w "$victim" && echo "# an edit" >> "$victim"
systemctl --user start rl18xx-longrun@lr-dryrun.service
wait_for 60 bash -c "[ \"\$(systemctl --user show rl18xx-longrun@lr-dryrun.service -p ActiveState --value)\" = failed ]"
check "7 modified deployment: start refused with 76, not restarted" \
    [ "$(prop lr-dryrun ExecMainStatus)" = 76 -a "$(prop lr-dryrun NRestarts)" = 0 ]
wait_for 30 alert_logged lr-dryrun "refused"
check "7 the watchdog alerted" alert_logged lr-dryrun "CRIT \[lr-dryrun\] refused"
git -C "$DEPLOY" show "HEAD:${victim#$DEPLOY/}" > "$victim" && chmod a-w "$victim"
check "7 restored: deploy.sh verify passes again" bash "$HERE/deploy.sh" verify "$DEPLOY"
systemctl --user reset-failed rl18xx-longrun@lr-dryrun.service

# ----------------------------------------------------------------------------- 8: non-finite loss
install_instance lr-dryrun-nan model_checkpoints_search/dryrun_nan "$TRAIN --lr 1e30 --critic-lr 1e30 --steps-per-update 2"
systemctl --user start rl18xx-longrun@lr-dryrun-nan.service
wait_for 180 bash -c "[ \"\$(systemctl --user show rl18xx-longrun@lr-dryrun-nan.service -p ActiveState --value)\" = failed ]"
check "8 non-finite: exit 75, not restarted" [ "$(prop lr-dryrun-nan ExecMainStatus)" = 75 -a "$(prop lr-dryrun-nan NRestarts)" = 0 ]
wait_for 30 alert_logged lr-dryrun-nan "nonfinite"
check "8 the watchdog alerted: $(grep -m1 nonfinite "$WORK/longrun/lr-dryrun-nan/alerts.log" | cut -c1-150)" alert_logged lr-dryrun-nan "CRIT \[lr-dryrun-nan\] nonfinite"

# ----------------------------------------------------------------------------- 9: OOM, crash loop
rc_before=$(systemctl --user show claude-rc.service -p ActiveEnterTimestampMonotonic --value 2>/dev/null)
install_instance lr-dryrun-oom model_checkpoints_search/dryrun_oom "$TRAIN" --memory-high infinity --memory-max 1500M --restart-sec 3
systemctl --user start rl18xx-longrun@lr-dryrun-oom.service
wait_for 300 bash -c "[ \"\$(systemctl --user show rl18xx-longrun@lr-dryrun-oom.service -p Result --value)\" = start-limit-hit ]"
check "9 OOM: restarted until the start limit ($(prop lr-dryrun-oom NRestarts) restarts, result $(prop lr-dryrun-oom Result))" \
    [ "$(prop lr-dryrun-oom Result)" = start-limit-hit ]
check "9 OOM kills were confined to the run's cgroup" \
    bash -c "journalctl --user -u rl18xx-longrun@lr-dryrun-oom.service --no-pager | grep -q 'oom-kill\|OOM'"
check "9 the user's session and claude-rc are untouched" \
    [ "$(systemctl --user show claude-rc.service -p ActiveEnterTimestampMonotonic --value 2>/dev/null)" = "$rc_before" ]
wait_for 30 alert_logged lr-dryrun-oom "crash_loop"
check "9 the watchdog reported the crash loop" alert_logged lr-dryrun-oom "CRIT \[lr-dryrun-oom\] crash_loop"
check "9 the watchdog reported the failed unit" alert_logged lr-dryrun-oom "CRIT \[lr-dryrun-oom\] failed"

# ----------------------------------------------------------------------------- 10: thresholds, hooks
out=$(env -i HOME="$HOME" PATH="$PATH" WORKDIR="$WORK" RUN_DIR="$RUN" STATE_DIR="$BASE/thresholds" DISK_MIN_GB=1000000 \
    GPU_TEMP_MAX_C=1 ALERT_COMMAND="echo \$ALERT_KEY >> $BASE/thresholds.cmd" \
    python3 "$HERE/watchdog.py" --instance lr-dryrun --once 2>&1)
check "10 disk threshold alerts" grep -q "disk:" <<< "$out"
if command -v nvidia-smi > /dev/null; then check "10 GPU temperature threshold alerts" grep -q "gpu0_temp" <<< "$out"; fi
check "10 ALERT_COMMAND ran" grep -q "disk:" "$BASE/thresholds.cmd"
check "10 ALERT_COMMAND ran for the instances too" [ -s "$WORK/alert_command.out" ]
echo "$out" | sed 's/^/     /'

log "alerts raised:"
cat "$WORK"/longrun/*/alerts.log | sed 's/^/     /'
[ $failures -eq 0 ] && log "dry run: all checks passed" || log "dry run: $failures check(s) FAILED"
exit $failures
