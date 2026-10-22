#!/bin/bash
# The long-run service's ExecStart / ExecStop (rl18xx-longrun@.service; docs/longrun_runbook.md).
#
#   launch.sh start    verify the deployment, then run main.py: a fresh run the first time, --resume from the
#                      run's last snapshot after that; exit with main.py's status, or
#                        75  the training stopped on a non-finite loss / gradient (no automatic restart)
#                        76  the deployment changed since it was built, or the configuration is incomplete
#                        77  the snapshot to resume from is missing or unreadable
#   launch.sh stop     (ExecStop) wait for a snapshot write in progress, then let systemd signal the run
#
# Settings come from the instance's environment file (~/.config/rl18xx-longrun/<instance>.env, written by
# install.sh): DEPLOY, WORKDIR, RUN_DIR, START_ARGS (first start only: --policy / --value), TRAIN_ARGS (every
# start: every setting the run should keep), and optionally LOG, STATE_DIR, COMMAND, PYTHON.
set -uo pipefail

INSTANCE=${LONGRUN_INSTANCE:-manual}
: "${DEPLOY:?DEPLOY (the frozen deployment) is not set}" "${WORKDIR:?WORKDIR is not set}" "${RUN_DIR:?RUN_DIR is not set}"
COMMAND=${COMMAND:-search-learning}
PYTHON=${PYTHON:-$DEPLOY/.venv/bin/python}
cd "$WORKDIR" || exit 76
LOG=${LOG:-logs/longrun_$INSTANCE.log}
STATE_DIR=${STATE_DIR:-longrun/$INSTANCE}
mkdir -p "$(dirname "$LOG")" "$STATE_DIR"

say() {
    local line
    line="$(date -Iseconds) longrun[$INSTANCE]: $*"
    echo "$line"
    echo "$line" >> "$LOG"
}

# One JSON line per event in $STATE_DIR/<file> (the watchdog reads them).
event() {  # event <file> key=value...
    python3 - "$STATE_DIR/$1" "${@:2}" <<'PY'
import json, sys
from datetime import datetime
record = {"time": datetime.now().astimezone().isoformat(timespec="seconds")}
for item in sys.argv[2:]:
    key, _, value = item.partition("=")
    record[key] = int(value) if value.lstrip("-").isdigit() else value
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(record) + "\n")
PY
}

snapshot_mtime_gap() {  # >0 while a snapshot write is in progress (optimizer.pth written, state.json not yet)
    [ -f "$RUN_DIR/optimizer.pth" ] && [ -f "$RUN_DIR/state.json" ] || { echo 0; return; }
    echo $(( $(stat -c %Y "$RUN_DIR/optimizer.pth") - $(stat -c %Y "$RUN_DIR/state.json") ))
}

case ${1:-} in
stop)
    # optimizer.pth and state.json are not written atomically (the checkpoints are): don't let the SIGTERM
    # that follows land in the middle of a snapshot.
    for _ in $(seq 120); do
        [ "$(snapshot_mtime_gap)" -gt 0 ] && [ $(( $(date +%s) - $(stat -c %Y "$RUN_DIR/optimizer.pth") )) -lt 120 ] || break
        sleep 1
    done
    event exits.jsonl event=stop pid="${MAINPID:-}"
    say "stop requested (systemd); signalling the run"
    exit 0
    ;;
start) ;;
*) echo "usage: launch.sh start|stop" >&2; exit 2 ;;
esac

# ----------------------------------------------------------------------------- start

if ! problem=$(bash "$DEPLOY/scripts/longrun/deploy.sh" verify "$DEPLOY" 2>&1); then
    say "refusing to start: the deployment $DEPLOY changed since it was built: $problem"
    event exits.jsonl event=refused rc=76 reason=deployment_modified
    exit 76
fi
commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$DEPLOY/DEPLOYMENT.json")

# A shared lock for the run's lifetime (main.py inherits it): deploy.sh update refuses while it is held.
exec 9< "$DEPLOY/.longrun.lock"
flock -s -n 9 || { say "cannot lock $DEPLOY/.longrun.lock"; exit 76; }

read -r -a train_args <<< "${TRAIN_ARGS:-}"
read -r -a start_args <<< "${START_ARGS:-}"

if [ -f "$RUN_DIR/STOP" ]; then
    say "removing a stale $RUN_DIR/STOP (a STOP file ends the run at its next update)"
    rm -f "$RUN_DIR/STOP"
fi

if [ -f "$RUN_DIR/state.json" ]; then
    if ! snapshot=$("$PYTHON" -s "$DEPLOY/scripts/longrun/snapshot_check.py" "$RUN_DIR" 2>&1); then
        say "refusing to resume: $snapshot (docs/longrun_runbook.md, 'Recovering a snapshot')"
        event exits.jsonl event=refused rc=77 reason=snapshot_invalid detail="$snapshot"
        exit 77
    fi
    mode=resume
    args=(--resume "$RUN_DIR" "${train_args[@]}")
    say "resuming $RUN_DIR from its snapshot at update $snapshot (deployment $commit)"
else
    if [ -e "$RUN_DIR" ]; then
        # A run that never reached its first snapshot: nothing to resume, so start over beside it. One that
        # has had snapshots but lost state.json is not that: stop and let a human look.
        lines=$(cat "$RUN_DIR/history.jsonl" 2>/dev/null | wc -l)
        every=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("snapshot_every", 10))' \
            "$RUN_DIR/config.json" 2>/dev/null || echo 10)
        if [ "$lines" -ge "$every" ]; then
            say "refusing to start: $RUN_DIR has $lines updates of history but no state.json"
            event exits.jsonl event=refused rc=77 reason=state_missing
            exit 77
        fi
        aside="$RUN_DIR.nosnapshot-$(date +%Y%m%dT%H%M%S)"
        mv "$RUN_DIR" "$aside"
        say "$RUN_DIR never reached a snapshot ($lines updates): moved to $aside, starting fresh"
    fi
    [ ${#start_args[@]} -gt 0 ] || { say "START_ARGS (--policy / --value) is empty"; exit 76; }
    mode=fresh
    snapshot=0
    args=(--out-dir "$(dirname "$RUN_DIR")" --run-name "$(basename "$RUN_DIR")" "${start_args[@]}" "${train_args[@]}")
    say "starting $RUN_DIR fresh (deployment $commit)"
fi

# Only the deployment's code: no user site-packages, no inherited PYTHONPATH.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
say "$PYTHON -s $DEPLOY/main.py $COMMAND ${args[*]}"
offset=$(stat -c %s "$LOG")
"$PYTHON" -s "$DEPLOY/main.py" "$COMMAND" "${args[@]}" >> "$LOG" 2>&1 &
child=$!
event starts.jsonl mode=$mode snapshot_update="$snapshot" pid=$child commit="$commit" \
    invocation="${INVOCATION_ID:-}"
wait $child
rc=$?

if [ $rc -eq 0 ]; then
    say "main.py exited cleanly (max updates reached or a STOP file)"
    event exits.jsonl event=exit rc=0 reason=clean
    exit 0
fi
# The training's own non-finite stops (search_learning._update / _update_value_only).
cause=$(tail -c +$((offset + 1)) "$LOG" | tail -n 300 \
    | grep -E "^(RuntimeError|[A-Za-z.]*Error): .*(non-finite|not finite|diverged)" | tail -1)
if [ -n "$cause" ]; then
    say "main.py stopped on a non-finite value: $cause -- not restarting"
    event exits.jsonl event=exit rc=75 reason=nonfinite detail="$cause"
    exit 75
fi
if [ $rc -gt 128 ]; then
    reason="signal $((rc - 128))"
else
    reason=$(tail -c +$((offset + 1)) "$LOG" | grep -E "^[A-Za-z_.]+(Error|Exception|Exit)\b" | tail -1 | cut -c1-200)
fi
say "main.py exited with status $rc (${reason:-no error line})"
event exits.jsonl event=exit rc=$rc reason=crash detail="$reason"
exit $rc
