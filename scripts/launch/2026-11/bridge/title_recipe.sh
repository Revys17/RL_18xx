#!/bin/bash
# Shared steps of the new titles' GPU recipes (1867, 1822 = 1822+ rules mix, 1822PNW, 1822MX, 1822CA): sourced by
# title_<title>.sh, which sets the per-title variables below and calls `title_main "$@"`. The plan, the per-title
# settings and the GPU estimates are in docs/rl_model.md "GPU run queue".
#
# Steps (any subset, in this order; none = pretrain pg):
#   pretrain  supervised two-stage pretraining (default-size model) on the title's human dataset, as the 1830 GPU check
#             (gpu_check_1830.sh: 10 epochs, batch 256, lr 1e-3 cosine, patience 4, value refit), then
#             scripts/generic_sl_report.py -> <sl run>/family_report.txt (top-1 per descriptor family vs chance)
#   pg        policy-gradient-generic from the SL checkpoint with gen6's recipe adapted (docs/training_runs_2026_11.md;
#             departures named in the title scripts); resumes from state.json when the run exists. Runs in the
#             foreground: start it with nohup / systemd-run for a long run.
#   evals     one round of policy-only evals (scripts/eval_generic.py, T 0.5, every game from the opening) of a learner
#             snapshot (EVAL_UPDATE, default the newest multiple of EVAL_EVERY): the fixed panel (4p 1v3 alone vs
#             SL@0.5, SL@1.0 and the frozen snapshots PANEL_UPDATES), each player count vs SL@0.5, and 4p 2v2 vs the
#             snapshot CHAIN_BACK updates earlier; then panel_status.py (panel means and the plateau rule)
#   evalloop  `evals` for every new multiple of EVAL_EVERY until the run stops (STOP file or max updates)
#
# Environment (defaults in brackets): CPUS [24-31,56-63], WORKERS [32], DEVICE [cuda], TAG [20261114],
# HG [human_games] (datasets and start files), SL_OUT / SL / OUT_DIR / PG_RUN / EVAL_OUT (paths), MAX_UPDATES,
# RESUME_MAX_UPDATES (a resume's new --max-updates), EVAL_GAMES [400], EVAL_WORKERS [16], EVAL_UPDATE,
# MIX_FLOOR [10]: every player count of the PG mix gets at least this percent of the games (gen6 style; the counts
# above it keep their proportions; 0 = the title's PLAYER_MIX as given; PLAYER_MIX itself can be overridden too).
# DRY=1: the CPU dry run -- tiny model / steps / games, CPU servers, every output under DRY_ROOT (nothing in the
# repository); the real settings are untouched by it.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)  # the repository root
cd "$ROOT" || exit 1

# --- per-title settings (set by title_<title>.sh before sourcing) ------------------------------------------------
: "${TITLE:?}" "${DATA_NAME:?}" "${PLAYER_MIX:?}" "${CAPS:?}" "${MAX_DECISIONS:?}"
: "${MICROBATCH:?}" "${INFER_BATCH:?}" "${GAMES_PER_TASK:?}" "${TITLE_MAX_UPDATES:?}"
START_SUFFIX=${START_SUFFIX:-}  # 1822: _v2 (every rules combination)
RULES_MIX=${RULES_MIX:-}        # 1822: "default" = the human games' combinations (titles.TitleSpec.rules_mix)

TAG=${TAG:-20261114}
DRY=${DRY:-}
CPUS=${CPUS:-24-31,56-63}
HG=${HG:-human_games}
DATA=${DATA:-$HG/$DATA_NAME}
NT='{n}'  # the player-count placeholder of start-file templates (a bare {n} would close ${...:-...} early)
START_T=${START_T:-$HG/start_positions_${TITLE}_${NT}p${START_SUFFIX}.jsonl}
MID_T=${MID_T:-$HG/midgame_positions_${TITLE}_${NT}p${START_SUFFIX}.jsonl}
SNAPSHOT_EVERY=10
EVAL_EVERY=${EVAL_EVERY:-200}
PANEL_PLAYERS=4                        # the panel's table: the learner alone against three copies (fair 0.25)
PANEL_UPDATES=${PANEL_UPDATES:-200 500}  # frozen early snapshots in the panel (from u500 on the panel is complete)
CHAIN_BACK=${CHAIN_BACK:-200}

if [ -n "$DRY" ]; then
  # The CPU dry run: same commands, tiny everything. No GPU, no repository writes.
  export CUDA_VISIBLE_DEVICES=
  DRY_ROOT=${DRY_ROOT:?DRY=1 needs DRY_ROOT (a scratch directory)}
  DEVICE=cpu
  SL_OUT=${SL_OUT:-$DRY_ROOT/$TITLE/sl}
  OUT_DIR=${OUT_DIR:-$DRY_ROOT/$TITLE/pg}
  EVAL_OUT=${EVAL_OUT:-$DRY_ROOT/$TITLE/eval}
  WORKERS=${WORKERS:-3}
  EVAL_GAMES=${EVAL_GAMES:-1}  # rounded up to one game per seat arrangement
  EVAL_WORKERS=${EVAL_WORKERS:-3}
  EVAL_GAMES_PER_TASK=${EVAL_GAMES_PER_TASK:-2}
  EVAL_EVERY=1
  PANEL_UPDATES="1 2"
  CHAIN_BACK=1
  MAX_UPDATES=${MAX_UPDATES:-2}
  PRETRAIN_ARGS=(--model-preset small --device cpu --workers 2 --threads 2 --batch-size 32 --max-steps 60
    --eval-every 30 --train-max-games 30 --val-max-games 2 --buffer-positions 2048 --refit-max-steps 20
    --refit-eval-steps 10 --log-every 10 --final-eval --report-max-games 2)
  PG_SCALE=(--workers "$WORKERS" --games-per-task 2 --positions-per-seat 8 --opponent-slots 1
    --slot-refresh-every 1 --league-snapshot-every 1 --league-max-snapshots 2 --anchor-every 1
    --rows-per-update 200 --minibatch 128 --microbatch 64 --snapshot-every 1 --save-game-every 2
    --server-cpu --server-threads 1 --threads 2 --inference-batch-size 64 --server-replicas 1)
  SNAPSHOT_EVERY=1
  EVAL_SERVER=(--server)  # CPU servers here (CUDA hidden)
  PY=(taskset -c "$CPUS" nice -n 19 env OMP_NUM_THREADS=1 "$ROOT/.venv/bin/python")
else
  DEVICE=${DEVICE:-cuda}
  SL_OUT=${SL_OUT:-model_checkpoints_generic/bridge_${TITLE}_sl_$TAG}
  OUT_DIR=${OUT_DIR:-model_checkpoints_generic_pg}
  WORKERS=${WORKERS:-32}
  EVAL_GAMES=${EVAL_GAMES:-400}
  EVAL_WORKERS=${EVAL_WORKERS:-16}
  MAX_UPDATES=${MAX_UPDATES:-$TITLE_MAX_UPDATES}
  # The 1830 GPU check's settings (= the legacy SL ckpt 10 recipe) + the final evaluation on the validation split.
  PRETRAIN_ARGS=(--model-preset default --device "$DEVICE" --workers 6 --threads 4 --epochs 10 --batch-size 256
    --lr 1e-3 --final-eval)
  # gen6's scale (32 workers, 32 rows per learner seat, 3 opponent slots per count, own snapshots every 50 (4 kept),
  # anchor every 50, 65,536 rows per update, minibatch 2,048, snapshot every 10, every 20th game saved), the
  # learner server in 2 replicas; microbatch and server batch per title (activation memory: docs/rl_model.md
  # "Model fit across titles").
  PG_SCALE=(--workers "$WORKERS" --games-per-task "$GAMES_PER_TASK" --positions-per-seat 32 --opponent-slots 3
    --slot-refresh-every 2 --league-snapshot-every 50 --league-max-snapshots 4 --anchor-every 50
    --rows-per-update 65536 --minibatch 2048 --microbatch "$MICROBATCH" --snapshot-every 10 --save-game-every 20
    --inference-batch-size "$INFER_BATCH" --server-replicas 2 --threads 8)
  EVAL_SERVER=(--server)  # each side on a generic inference server (the GPU)
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ulimit -n "$(ulimit -Hn)"
  PY=(taskset -c "$CPUS" nice -n 19 env OMP_NUM_THREADS=2 "$ROOT/.venv/bin/python")
fi
# ECHO=1: print each python command instead of running it (with SL_OUT / OUT_DIR / EVAL_OUT pointing somewhere
# disposable: the steps still create their directories and the league file).
[ -n "${ECHO:-}" ] && PY=(echo "${PY[@]}")
SL=${SL:-$SL_OUT/final.pth}
PG_RUN=${PG_RUN:-gpg${TITLE}_$TAG}
POP=${POP:-$OUT_DIR/populations/league_generic_${TITLE}_v1.json}
EVAL_OUT=${EVAL_OUT:-logs/eval/$PG_RUN}
PANEL_DIR=${PANEL_DIR:-$OUT_DIR/panels/$PG_RUN}
RUN_DIR=$OUT_DIR/$PG_RUN
COUNTS=$(echo "$PLAYER_MIX" | tr ',' '\n' | cut -d: -f1 | tr '\n' ' ')

# PLAYER_MIX ("3:24,4:61,...", any weights) with every count raised to at least $2 percent of the games and the others
# scaled down in proportion (repeated until none falls below the floor); one decimal. A floor of 0 returns the mix.
floor_mix() {
  echo "$1" | awk -v floor="$2" -F, '{
    n = NF; total = 0
    for (i = 1; i <= n; i++) { split($i, kv, ":"); key[i] = kv[1]; w[i] = kv[2] + 0; total += w[i]; pinned[i] = 0 }
    if (floor <= 0) { print $0; exit }
    if (floor * n > 100) { print "MIX_FLOOR " floor " x " n " counts > 100%" > "/dev/stderr"; exit 1 }
    for (i = 1; i <= n; i++) w[i] = 100 * w[i] / total
    changed = 1
    while (changed) {
      changed = 0; rest = 100; free = 0
      for (i = 1; i <= n; i++) if (pinned[i]) rest -= floor; else free += w[i]
      for (i = 1; i <= n; i++) {
        if (pinned[i]) { v[i] = floor; continue }
        v[i] = w[i] * rest / free
        if (v[i] < floor) { pinned[i] = 1; changed = 1 }
      }
    }
    out = ""
    for (i = 1; i <= n; i++) { x = sprintf("%.1f", v[i]); sub(/\.0$/, "", x); out = out (i > 1 ? "," : "") key[i] ":" x }
    print out
  }'
}
MIX_FLOOR=${MIX_FLOOR:-10}
PG_MIX=$(floor_mix "$PLAYER_MIX" "$MIX_FLOOR") || exit 1

# The decision cap of an n-player game (CAPS "3:1200,4:1200,...", else MAX_DECISIONS).
cap_for() {
  local kv
  for kv in ${CAPS//,/ }; do
    [ "${kv%%:*}" = "$1" ] && { echo "${kv#*:}"; return; }
  done
  echo "$MAX_DECISIONS"
}

step_pretrain() {
  mkdir -p "$SL_OUT"
  if [ -e "$SL_OUT/final.pth" ]; then
    echo "pretrain: $SL_OUT/final.pth exists, kept"
  else
    [ -e "$DATA/meta.json" ] || { echo "pretrain: no dataset at $DATA"; return 1; }
    "${PY[@]}" main.py pretrain-generic --data "$DATA" --out "$SL_OUT" "${PRETRAIN_ARGS[@]}" \
      2>&1 | tee -a "$SL_OUT/train.log"
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "pretrain failed"; return 1; }
  fi
  "${PY[@]}" scripts/generic_sl_report.py "$SL_OUT" --data "$DATA" --out "$SL_OUT/family_report.txt"
}

step_pg() {
  local rules=()
  [ -n "$RULES_MIX" ] && rules=(--rules-mix "$RULES_MIX")
  if [ -f "$RUN_DIR/state.json" ]; then
    echo "pg: resuming $RUN_DIR"
    "${PY[@]}" main.py policy-gradient-generic --resume "$RUN_DIR" \
      ${RESUME_MAX_UPDATES:+--max-updates "$RESUME_MAX_UPDATES"}
    return
  fi
  [ -f "$SL" ] || { echo "pg: no SL checkpoint at $SL (run the pretrain step first)"; return 1; }
  mkdir -p "$(dirname "$POP")"
  [ -f "$POP" ] || printf '{"note": "%s %s: the generic SL checkpoint (own snapshots join per run)", %s}\n' \
    "$TITLE" "$PG_RUN" "\"members\": [{\"path\": \"$SL\", \"note\": \"generic SL (pretrain step)\"}]" > "$POP"
  "${PY[@]}" main.py policy-gradient-generic --run-name "$PG_RUN" --out-dir "$OUT_DIR" --title "$TITLE" \
    --policy "$SL" --player-mix "$PG_MIX" "${rules[@]}" \
    --max-decisions "$MAX_DECISIONS" --max-decisions-by-players "$CAPS" \
    "${PG_SCALE[@]}" --population "$POP" --pfsp-power 2.0 --reward placement \
    --ppo-epochs 1 --clip 0.2 --lr 2e-5 --critic-lr 3e-5 --kl-coef 0.02 --kl-target 0.08 --kl-coef-min 0.005 \
    --entropy-coef 0.0 --learner-temperature 0.5 --opponent-temperature 0.5 --opponent-price-eps 0.0 \
    --max-updates "$MAX_UPDATES" --pool-refresh-every 5 \
    --auction-fraction 0.5 --midgame-fraction 0.3 --random-start-fraction 0.2 --auction-stall-decisions 0 \
    --start-positions "$START_T" --midgame-positions "$MID_T"
}

# Newest learner snapshot that is a multiple of $1 (snapshots at multiples of SNAPSHOT_EVERY are kept for good).
newest_snapshot() {
  local every=$1 u best=
  for f in "$RUN_DIR"/learner/*.pth; do
    [ -e "$f" ] || continue
    u=$(basename "$f" .pth)
    [ $((u % every)) = 0 ] && [ $((u % SNAPSHOT_EVERY)) = 0 ] && { [ -z "$best" ] || [ "$u" -gt "$best" ]; } && best=$u
  done
  echo "$best"
}

# One eval_generic.py invocation: name players a_seats match-args...
run_eval() {
  local name=$1 n=$2 a=$3
  shift 3
  if [ -e "$EVAL_OUT/$name/summary.json" ]; then
    echo "evals: $name done, kept"
    return
  fi
  "${PY[@]}" scripts/eval_generic.py --title "$TITLE" "$@" --num-players "$n" --a-seats "$a" \
    --games "$EVAL_GAMES" --workers "$EVAL_WORKERS" --games-per-task "${EVAL_GAMES_PER_TASK:-10}" "${EVAL_SERVER[@]}" \
    --auction-start --auction-stall-decisions 0 --max-decisions "$(cap_for "$n")" --save-games "${EVAL_SAVE:-6}" \
    --out "$EVAL_OUT/$name" > "$EVAL_OUT/$name.log" 2>&1
  local status=$?
  grep -h '^\[' "$EVAL_OUT/$name.log" | sed "s#^#u$U $name #" | tee -a "$EVAL_OUT/results.txt"
  [ $status = 0 ] || echo "evals: $name failed ($status), see $EVAL_OUT/$name.log" | tee -a "$EVAL_OUT/results.txt"
}

step_evals() {
  U=${EVAL_UPDATE:-$(newest_snapshot "$EVAL_EVERY")}
  [ -n "$U" ] || { echo "evals: no snapshot at a multiple of $EVAL_EVERY in $RUN_DIR/learner yet"; return 1; }
  local learner="$RUN_DIR/learner/$U.pth" p panel=() n
  [ -f "$learner" ] || { echo "evals: no $learner"; return 1; }
  mkdir -p "$EVAL_OUT" "$PANEL_DIR"
  # Freeze the panel's snapshots (copies: the panel never moves, whatever happens to the run directory).
  for p in $PANEL_UPDATES; do
    [ -f "$RUN_DIR/learner/$p.pth" ] && [ ! -f "$PANEL_DIR/$p.pth" ] && cp "$RUN_DIR/learner/$p.pth" "$PANEL_DIR/$p.pth"
  done
  local L="$learner@0.5"
  # (a) the fixed outside panel, 4 players, the learner alone (fair 0.25): SL at T 0.5 and 1.0, frozen snapshots
  panel=(--match "$L" "$SL@0.5" --match "$L" "$SL@1.0")
  for p in $PANEL_UPDATES; do
    [ -f "$PANEL_DIR/$p.pth" ] && [ "$p" -lt "$U" ] && panel+=(--match "$L" "$PANEL_DIR/$p.pth@0.5")
  done
  run_eval "u${U}_panel_${PANEL_PLAYERS}p_1v3" "$PANEL_PLAYERS" 1 "${panel[@]}"
  # (b) every player count vs the SL start at T 0.5: alone at the other counts, half the table at 4 players
  for n in $COUNTS; do
    if [ "$n" = 4 ]; then
      run_eval "u${U}_4p_2v2_sl" 4 2 --match "$L" "$SL@0.5"
    else
      run_eval "u${U}_${n}p_1v$((n - 1))_sl" "$n" 1 --match "$L" "$SL@0.5"
    fi
  done
  # (c) within-family progress: 4p 2v2 vs the snapshot CHAIN_BACK updates earlier
  local back=$((U - CHAIN_BACK))
  if [ "$back" -gt 0 ] && [ -f "$RUN_DIR/learner/$back.pth" ]; then
    run_eval "u${U}_4p_2v2_u$back" 4 2 --match "$L" "$RUN_DIR/learner/$back.pth@0.5"
  fi
  "${PY[@]}" "$ROOT/scripts/launch/2026-11/bridge/panel_status.py" "$EVAL_OUT" --panel-updates $PANEL_UPDATES \
    | tee "$EVAL_OUT/panel_status.txt"
}

step_evalloop() {
  local done_u=
  while :; do
    local u
    u=$(newest_snapshot "$EVAL_EVERY")
    if [ -n "$u" ] && [ "$u" != "$done_u" ] && [ "$u" -gt 0 ]; then
      EVAL_UPDATE=$u step_evals
      done_u=$u
    fi
    if [ -e "$RUN_DIR/STOP" ] || { [ -n "$u" ] && [ "$u" -ge "$MAX_UPDATES" ]; }; then
      echo "evalloop: run stopped"
      return
    fi
    sleep "${EVALLOOP_SLEEP:-600}"
  done
}

title_main() {
  local steps=("$@") s
  [ ${#steps[@]} -eq 0 ] && steps=(pretrain pg)
  echo "$TITLE recipe: steps ${steps[*]}${DRY:+ (DRY RUN under $DRY_ROOT)}; SL $SL; PG run $RUN_DIR; counts $COUNTS (PG mix $PG_MIX)"
  for s in "${steps[@]}"; do
    case "$s" in
      pretrain | pg | evals | evalloop) "step_$s" || { echo "$TITLE: step $s failed"; return 1; } ;;
      *) echo "unknown step $s (pretrain pg evals evalloop)"; return 2 ;;
    esac
  done
}
