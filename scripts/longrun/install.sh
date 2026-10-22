#!/bin/bash
# Install a long-run instance under the user's systemd (docs/longrun_runbook.md). Writes, and nothing else:
#   ~/.config/rl18xx-longrun/<instance>.env                         the instance's settings
#   ~/.config/systemd/user/rl18xx-longrun@.service                  the unit templates, from the deployment
#   ~/.config/systemd/user/rl18xx-longrun-watchdog@.service
#   ~/.config/systemd/user/rl18xx-longrun@<instance>.service.d/10-limits.conf   memory, CPUs, restart delay
# then `systemctl --user daemon-reload`. It starts and enables nothing.
#
#   install.sh <instance> --deploy DIR --workdir DIR --run-dir PATH --start-args "..." --train-args "..."
#              [--memory-high 160G] [--memory-max 180G] [--cpus "0-63"] [--nice 0] [--restart-sec 60]
#              [--set KEY=VALUE]...       (watchdog thresholds, CUDA_VISIBLE_DEVICES, LOG, STATE_DIR, ...)
#   install.sh --uninstall <instance>   removes the instance's .env and drop-in (and the templates if it was
#                                       the last instance); never the run's data
#
# --run-dir is relative to --workdir (the run's working directory: human_games/, the checkpoint directories
# and runs/ resolve there). --start-args go only to the first start (--policy, --value); --train-args to
# every start, so they must name every setting the run keeps (a resume before 5634a72 resets an omitted
# resumable setting to its default).
set -euo pipefail

die() { echo "install.sh: $*" >&2; exit 1; }
CONF=${XDG_CONFIG_HOME:-$HOME/.config}
ENV_DIR=$CONF/rl18xx-longrun
UNIT_DIR=$CONF/systemd/user

# One systemd EnvironmentFile value, single-quoted: no escapes, no variable expansion ($ALERT_MESSAGE in
# ALERT_COMMAND stays for the shell that runs it). check_value runs before anything is written.
check_value() { case $1 in *[\'\\]*|*$'\n'*) die "values can't contain ', \\ or newlines: $1" ;; esac; }
quote() { printf "'%s'" "$1"; }

if [ "${1:-}" = "--uninstall" ]; then
    instance=${2:?instance}
    systemctl --user is-active --quiet "rl18xx-longrun@$instance.service" && die "rl18xx-longrun@$instance is running"
    rm -f "$ENV_DIR/$instance.env"
    rm -rf "$UNIT_DIR/rl18xx-longrun@$instance.service.d"
    if ! ls "$ENV_DIR"/*.env > /dev/null 2>&1; then
        rm -f "$UNIT_DIR/rl18xx-longrun@.service" "$UNIT_DIR/rl18xx-longrun-watchdog@.service"
    fi
    systemctl --user daemon-reload
    systemctl --user reset-failed "rl18xx-longrun@$instance.service" "rl18xx-longrun-watchdog@$instance.service" 2>/dev/null || true
    echo "uninstalled $instance (the run's files are untouched)"
    exit 0
fi

instance=${1:?usage: install.sh <instance> --deploy DIR ...}; shift
[[ "$instance" =~ ^[A-Za-z0-9_.-]+$ ]] || die "instance names are letters, digits, _ . -"
deploy="" workdir="" run_dir="" start_args="" train_args=""
memory_high=160G memory_max=180G cpus="" nice=0 restart_sec=60
extra=()
while [ $# -gt 0 ]; do
    case $1 in
        --deploy) deploy=$(realpath "$2"); shift 2 ;;
        --workdir) workdir=$(realpath "$2"); shift 2 ;;
        --run-dir) run_dir=$2; shift 2 ;;
        --start-args) start_args=$2; shift 2 ;;
        --train-args) train_args=$2; shift 2 ;;
        --memory-high) memory_high=$2; shift 2 ;;
        --memory-max) memory_max=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --nice) nice=$2; shift 2 ;;
        --restart-sec) restart_sec=$2; shift 2 ;;
        --set) [[ "$2" == *=* ]] || die "--set KEY=VALUE"; extra+=("$2"); shift 2 ;;
        *) die "unknown option $1" ;;
    esac
done
[ -n "$deploy" ] && [ -n "$workdir" ] && [ -n "$run_dir" ] || die "--deploy, --workdir and --run-dir are required"
[ -f "$deploy/DEPLOYMENT.json" ] || die "$deploy is not a deployment (scripts/longrun/deploy.sh create)"
[ -x "$deploy/.venv/bin/python" ] || die "$deploy has no .venv"
[ -n "$start_args" ] || [ -f "$workdir/$run_dir/state.json" ] || [ -f "$run_dir/state.json" ] \
    || die "--start-args (--policy / --value) are needed for a run that has no snapshot yet"
src=$deploy/scripts/longrun
for value in "$deploy" "$workdir" "$run_dir" "$start_args" "$train_args" "${extra[@]}"; do check_value "$value"; done

mkdir -p "$ENV_DIR" "$UNIT_DIR/rl18xx-longrun@$instance.service.d"
[ -f "$ENV_DIR/$instance.env" ] && cp "$ENV_DIR/$instance.env" "$ENV_DIR/$instance.env.bak"
{
    echo "# rl18xx long run $instance (scripts/longrun/install.sh, $(date -Iseconds)). Read by launch.sh and watchdog.py."
    echo "DEPLOY=$(quote "$deploy")"
    echo "LAUNCHER=$(quote "$src/launch.sh")"
    echo "WATCHDOG=$(quote "$src/watchdog.py")"
    echo "WORKDIR=$(quote "$workdir")"
    echo "RUN_DIR=$(quote "$run_dir")"
    echo "START_ARGS=$(quote "$start_args")"
    echo "TRAIN_ARGS=$(quote "$train_args")"
    for kv in "${extra[@]}"; do echo "${kv%%=*}=$(quote "${kv#*=}")"; done
} > "$ENV_DIR/$instance.env"

install -m 0644 "$src/rl18xx-longrun@.service" "$src/rl18xx-longrun-watchdog@.service" "$UNIT_DIR/"
{
    echo "# install.sh, $(date -Iseconds)"
    echo "[Service]"
    echo "MemoryHigh=$memory_high"
    echo "MemoryMax=$memory_max"
    echo "RestartSec=$restart_sec"
    echo "Nice=$nice"
    [ -n "$cpus" ] && echo "CPUAffinity=${cpus//,/ }"
    true
} > "$UNIT_DIR/rl18xx-longrun@$instance.service.d/10-limits.conf"
systemctl --user daemon-reload

cat <<EOF
installed rl18xx-longrun@$instance
  settings  $ENV_DIR/$instance.env
  limits    $UNIT_DIR/rl18xx-longrun@$instance.service.d/10-limits.conf
start it:   systemctl --user start rl18xx-longrun@$instance        (also starts its watchdog)
status:     python3 $src/watchdog.py --instance $instance status
at boot:    systemctl --user enable rl18xx-longrun@$instance       (only if you want it; see the runbook)
EOF
