#!/bin/bash
# The long run's daily jobs (rl18xx-longrun-daily@.service, started by its timer; docs/longrun_runbook.md).
#
#   daily.sh        retention -> style statistics -> integrity check -> status report, from the workdir
#
# Settings come from the daily instance's environment file (install.sh --kind daily): DEPLOY, WORKDIR, STATE_DIR
# (default longrun/<instance>: every job's output), RETENTION_CONFIG, SAVE_DATA, and the status report's inputs
# (GEN_RUN_DIR, VALUE_RUN_DIR, LADDER_PANEL, WATCH_DIRS, REPORT_LOG, USAGE_LOG, USAGE_PATHS, WORKTREES_DIR, ...;
# scripts/longrun/status_report.py). A job whose setting is empty is skipped; extra arguments per job:
# RETENTION_ARGS, STYLE_ARGS, INTEGRITY_ARGS, REPORT_ARGS. Each job's output goes to $STATE_DIR/daily.log; their
# exit statuses to $STATE_DIR/daily_steps.json, which the status report (last) reads. Each job gets at most
# STEP_TIMEOUT seconds (default 2700; exit 124 when it is stopped), so one stuck job can't keep the status report
# from running inside the unit's 3-h limit. Exits non-zero if a job failed, so systemd marks the run failed (an
# integrity mismatch exits 1).
set -uo pipefail

INSTANCE=${LONGRUN_INSTANCE:-daily}
: "${DEPLOY:?DEPLOY (the frozen deployment) is not set}" "${WORKDIR:?WORKDIR is not set}"
cd "$WORKDIR" || exit 1
PYTHON=${PYTHON:-$DEPLOY/.venv/bin/python}
export STATE_DIR=${STATE_DIR:-longrun/$INSTANCE}
mkdir -p "$STATE_DIR"
LOG=$STATE_DIR/daily.log
# Only the deployment's code.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

declare -A rc
order=()
step() {  # step <name> <command...>
    local name=$1; shift
    order+=("$name")
    echo "$(date -Iseconds) daily[$INSTANCE] $name: $*" >> "$LOG"
    timeout --kill-after=60 "${STEP_TIMEOUT:-2700}" "$@" >> "$LOG" 2>&1
    rc[$name]=$?
    [ "${rc[$name]}" -eq 124 ] && echo "$(date -Iseconds) daily[$INSTANCE] $name: stopped after ${STEP_TIMEOUT:-2700} s" >> "$LOG"
    echo "$(date -Iseconds) daily[$INSTANCE] $name: exit ${rc[$name]}" >> "$LOG"
}
L=$DEPLOY/scripts/longrun

[ -n "${RETENTION_CONFIG:-}" ] && step retention "$PYTHON" -s "$L/retention.py" --config "$RETENTION_CONFIG" \
    --log "$STATE_DIR/retention.jsonl" --quiet ${RETENTION_ARGS:-}
if [ -n "${SAVE_DATA:-}" ]; then
    step style_stats "$PYTHON" -s "$L/style_stats.py" --data "$SAVE_DATA" --out "$STATE_DIR/style_stats.jsonl" \
        --baseline first ${STYLE_ARGS:-}
    step integrity "$PYTHON" -s "$L/integrity_check.py" --data "$SAVE_DATA" --log "$STATE_DIR/integrity.jsonl" \
        --checksums "$STATE_DIR/shard_checksums.txt" ${INTEGRITY_ARGS:-}
fi

{
    printf '{"time": "%s", "steps": {' "$(date -Iseconds)"
    sep=""
    for name in "${order[@]}"; do printf '%s"%s": %d' "$sep" "$name" "${rc[$name]}"; sep=", "; done
    printf '}}\n'
} > "$STATE_DIR/daily_steps.json"

step status_report "$PYTHON" -s "$L/status_report.py" ${REPORT_ARGS:-}

failed=0
for name in "${order[@]}"; do [ "${rc[$name]}" -eq 0 ] || failed=1; done
exit $failed
