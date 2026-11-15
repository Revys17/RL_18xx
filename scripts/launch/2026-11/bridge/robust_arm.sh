#!/bin/bash
# The robustness arms on generic 1830 (docs/robustness_test_plan.md Step 1; mechanisms in docs/rl_model.md
# "Robustness mechanisms"): gen6's recipe as pg_generic_1830.sh runs it, from the 1830 GPU check's generic SL
# checkpoint, plus each arm's change; the same seed, a 300-update first pass, pg_generic_1830.sh's evals. NOT LAUNCHED.
#
#   bash scripts/launch/2026-11/bridge/robust_arm.sh A0                 # = `pg`: new run, or resume from state.json
#   nohup bash .../robust_arm.sh A3 pg > logs/robust_A3.log 2>&1 &      # foreground runs: nohup / systemd-run
#   nohup bash .../robust_arm.sh A3 exploiters > logs/robust_A3_exp.log 2>&1 &   # A3 / A4+A3: the refresh loop
#   bash .../robust_arm.sh A1 evals                                     # one eval round (newest u % EVAL_EVERY)
#   nohup bash .../robust_arm.sh A1 evalloop > logs/robust_A1_evals.log 2>&1 &   # rounds until the run stops
#   RESUME_MAX_UPDATES=600 bash .../robust_arm.sh A2 pg                 # the second pass of a promising arm
#   DRY=1 DRY_ROOT=<scratch> HG=<main checkout>/human_games bash .../robust_arm.sh A4 pg evals   # CPU dry run
#   ECHO=1 OUT_DIR=<scratch> bash .../robust_arm.sh A2 pg               # print the command
#
# Arms (each = A0 + its change):
#   A0  baseline: pg_generic_1830.sh's recipe (mix 2:15,3:20,4:35,5:20,6:10, placement, per-count PFSP, KL to an
#       anchor moved every 50, learner / opponent T 0.5, half the games from the real-rules auction, 30% mid-game,
#       20% random endings) with its own league file (the SL + own snapshots), 300 updates, --seed 1830.
#   A1  human-imitation seats: the league file holds the SL at T 1.0 and at T 0.5 (two members, each pinned at
#       12.5% of the draws at every player count: 25% of opponent seats), own snapshots as A0. REF_KL=<coef> adds
#       the piKL-style pull, --ref-kl-coef <coef> toward the SL at REF_T (default the learner's 0.5: KL 0 at the
#       start; 1.0 = the SL's human spread), and names the run ..._refkl<coef>.
#   A2  self-prioritized starts: --start-priority (|reward - V| of the learner's rows, rank temperature 1,
#       staleness 0.3, uniform share 0.3, ema 0.5) over the human SR1 starts and mid-game cuts; the start mix itself
#       (auction / mid-game / random / human) is A0's.
#   A3  main exploiters: the main = A0 + --population-reload-every 2; step `exploiters` (exploiter_loop.py) trains
#       an exploiter from the main's snapshot every EXP_EVERY (100) updates for EXP_UPDATES (100) against a frozen
#       copy of it in every other seat (placement reward, T EXP_T 0.5, entropy EXP_ENTROPY 0.01, no KL, lr EXP_LR
#       4e-5), then adds it to the main's league file pinned at EXP_MIN_SHARE (0.1; the newest EXP_PINNED = 2 keep
#       their pins). Exploiter runs: $OUT_DIR/<run>_mx<u>/ (every 20th game saved: mine them with M2).
#   A4  A1 + A2 (WITH_A3=1: + A3, run name ..._a3).
#
# Departures from pg_generic_1830.sh (all arms): max updates 300 (the plan's first pass; RESUME_MAX_UPDATES extends),
# --seed 1830 (every arm draws the same starts and seats per worker task until the policies differ; runs stay
# asynchronous, not reproducible), a league file per run, evals every EVAL_EVERY 100 updates with two extra matches
# (4p 1v3 alone vs the SL at T 1.0 -- M3's human-imitation opponent -- and, with exploiters, vs each loop exploiter).
#
# Environment: CPUS [24-31,56-63], TAG [20261115], SL, OUT_DIR [model_checkpoints_generic_pg], HG [human_games],
# SEED [1830], MAX_UPDATES [300], RESUME_MAX_UPDATES, EVAL_EVERY [100], EVAL_GAMES [400], EVAL_WORKERS [16],
# SL10 / GEN6 (the legacy outside panel), EXP_* (above), REF_KL / REF_T (A1, A4), WITH_A3 (A4).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)  # the repository root
cd "$ROOT" || exit 1

ARM=${1:-}
shift || true
case "$ARM" in A0 | A1 | A2 | A3 | A4) ;; *) echo "usage: $0 <A0|A1|A2|A3|A4> [pg] [exploiters] [evals] [evalloop]"; exit 2 ;; esac
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(pg)

TAG=${TAG:-20261115}
DRY=${DRY:-}
CPUS=${CPUS:-24-31,56-63}
HG=${HG:-human_games}
SEED=${SEED:-1830}
REF_KL=${REF_KL:-}
REF_T=${REF_T:-}
WITH_A3=${WITH_A3:-}
EXP_EVERY=${EXP_EVERY:-100}
EXP_UPDATES=${EXP_UPDATES:-100}
EXP_MIN_SHARE=${EXP_MIN_SHARE:-0.1}
EXP_PINNED=${EXP_PINNED:-2}
EXP_T=${EXP_T:-0.5}
EXP_ENTROPY=${EXP_ENTROPY:-0.01}
EXP_LR=${EXP_LR:-4e-5}
NT='{n}'  # the player-count placeholder of start-file templates

# The arm's mechanisms.
WANT_A1= WANT_A2= WANT_A3=
case "$ARM" in
  A1) WANT_A1=1 ;;
  A2) WANT_A2=1 ;;
  A3) WANT_A3=1 ;;
  A4) WANT_A1=1; WANT_A2=1; [ -n "$WITH_A3" ] && WANT_A3=1 ;;
esac
SUFFIX=
[ "$ARM" = A4 ] && [ -n "$WANT_A3" ] && SUFFIX=_a3
[ -n "$WANT_A1" ] && [ -n "$REF_KL" ] && SUFFIX="${SUFFIX}_refkl$REF_KL"

if [ -n "$DRY" ]; then
  # The CPU dry run: the same commands, tiny everything, CPU servers, every output under DRY_ROOT.
  export CUDA_VISIBLE_DEVICES=
  DRY_ROOT=${DRY_ROOT:?DRY=1 needs DRY_ROOT (a scratch directory)}
  OUT_DIR=${OUT_DIR:-$DRY_ROOT/pg}
  SL=${SL:-$DRY_ROOT/sl/final.pth}  # made by a tiny pretrain-generic below when missing
  MAX_UPDATES=${MAX_UPDATES:-3}
  EVAL_EVERY=${EVAL_EVERY:-1}
  EVAL_GAMES=${EVAL_GAMES:-1}  # rounded up to one game per seat arrangement
  EVAL_WORKERS=${EVAL_WORKERS:-3}
  EXP_EVERY=1 EXP_UPDATES=1 EXP_POLL=5
  PG_SCALE=(--workers 3 --games-per-task 2 --positions-per-seat 8 --opponent-slots 1 --slot-refresh-every 1
    --league-snapshot-every 1 --league-max-snapshots 2 --anchor-every 1 --rows-per-update 200 --minibatch 128
    --microbatch 64 --snapshot-every 1 --save-game-every 2 --server-cpu --server-threads 1 --threads 2
    --inference-batch-size 64 --server-replicas 1)
  EXP_SNAPSHOT_EVERY=1
  EVAL_SERVER=()  # generic players in the pool workers (CPU)
  # tiny legacy stand-ins for the outside panel (made below when missing; the real ones live under /home/revys/longrun;
  # the legacy loader wants a numbered file name)
  SL10=${SL10:-$DRY_ROOT/standin/sl10/10.pth}
  GEN6=${GEN6:-$DRY_ROOT/standin/gen6/learner/2400.pth}
  PY=(taskset -c "$CPUS" nice -n 19 env OMP_NUM_THREADS=1 "$ROOT/.venv/bin/python")
else
  OUT_DIR=${OUT_DIR:-model_checkpoints_generic_pg}
  SL=${SL:-model_checkpoints_generic/bridge_1830_check_20261113/final.pth}  # gpu_check_1830.sh's $OUT/final.pth
  MAX_UPDATES=${MAX_UPDATES:-300}
  EVAL_EVERY=${EVAL_EVERY:-100}
  EVAL_GAMES=${EVAL_GAMES:-400}
  EVAL_WORKERS=${EVAL_WORKERS:-16}
  EXP_POLL=${EXP_POLL:-300}
  # pg_generic_1830.sh's scale (= gen6's: 32 workers x 96 games, 32 rows per learner seat, 3 opponent slots per count,
  # slot refresh 2, own snapshots every 50 (4 kept), anchor every 50, 65,536 rows, minibatch 2,048 / microbatch 1,024,
  # snapshot every 10, every 20th game saved) + the learner server in 2 replicas
  PG_SCALE=(--workers 32 --games-per-task 96 --positions-per-seat 32 --opponent-slots 3 --slot-refresh-every 2
    --league-snapshot-every 50 --league-max-snapshots 4 --anchor-every 50 --rows-per-update 65536 --minibatch 2048
    --microbatch 1024 --snapshot-every 10 --save-game-every 20 --inference-batch-size 1024 --server-replicas 2
    --threads 8)
  EXP_SNAPSHOT_EVERY=10
  EVAL_SERVER=(--generic-server)  # generic players on generic inference servers (the GPU)
  SL10=${SL10:-/home/revys/longrun/work/lr1/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth}
  GEN6=${GEN6:-/home/revys/longrun/work/lr1/model_checkpoints_pg/gen6_20261110/learner/2400.pth}
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ulimit -n "$(ulimit -Hn)"
  PY=(taskset -c "$CPUS" nice -n 19 env OMP_NUM_THREADS=2 "$ROOT/.venv/bin/python")
fi
[ -n "${ECHO:-}" ] && PY=(echo "${PY[@]}")
RUN=${RUN:-robust_${ARM}_${TAG}${SUFFIX}}
RUN_DIR=$OUT_DIR/$RUN
POP=${POP:-$OUT_DIR/populations/$RUN.json}
EVAL_OUT=${EVAL_OUT:-${DRY:+$DRY_ROOT/eval/$RUN}}
EVAL_OUT=${EVAL_OUT:-logs/eval/$RUN}

# The game settings of gen6's recipe, shared by the main run and its exploiters.
GAME_ARGS=(--title 1830 --player-mix 2:15,3:20,4:35,5:20,6:10
  --max-decisions 1200 --max-decisions-by-players 2:1000,3:1100,5:1300,6:1500 --reward placement
  --random-start-fraction 0.2 --midgame-fraction 0.3 --auction-fraction 0.5 --no-auction-unlock
  --auction-stall-decisions 150 --start-positions "$HG/start_positions_1830_${NT}p_v2.jsonl"
  --midgame-positions "$HG/midgame_positions_1830_${NT}p_v2.jsonl" --opponent-price-eps 0.0 --ppo-epochs 1
  --clip 0.2 --critic-lr 3e-5 --seed "$SEED")

write_population() {
  [ -f "$POP" ] && return
  mkdir -p "$(dirname "$POP")"
  if [ -n "$WANT_A1" ]; then
    printf '{"note": "%s: the generic SL at T 1.0 and 0.5, each pinned at 12.5%% at every count (A1); own snapshots join", "members": [{"path": "%s", "temperature": 1.0, "min_share": 0.125}, {"path": "%s", "temperature": 0.5, "min_share": 0.125}]}\n' \
      "$RUN" "$SL" "$SL" > "$POP"
  else
    printf '{"note": "%s: the generic SL (own snapshots join)", "members": [{"path": "%s"}]}\n' "$RUN" "$SL" > "$POP"
  fi
}

dry_sl() {  # DRY: a tiny generic SL checkpoint from 60 steps of pretrain-generic on the real dataset
  [ -z "$DRY" ] || [ -f "$SL" ] && return 0
  mkdir -p "$(dirname "$SL")"
  "${PY[@]}" main.py pretrain-generic --data "$HG/generic_1830_clean_all_v1" --out "$(dirname "$SL")" \
    --model-preset small --device cpu --workers 2 --threads 2 --batch-size 32 --max-steps 60 --eval-every 30 \
    --train-max-games 30 --val-max-games 2 --buffer-positions 2048 --refit-max-steps 20 --refit-eval-steps 10 \
    --log-every 10 > "$(dirname "$SL")/train.log" 2>&1 || { echo "dry: pretrain failed ($(dirname "$SL")/train.log)"; return 1; }
}

dry_standins() {  # DRY: tiny legacy (AlphaZeroTransformer) checkpoints standing in for SL ckpt 10 and gen6 u2400
  [ -z "$DRY" ] || { [ -f "$SL10" ] && [ -f "$GEN6" ]; } && return 0
  "${PY[@]}" - "$SL10" "$GEN6" <<'PYEOF'
import sys
import torch
from rl18xx.agent.alphazero.config import ModelTransformerConfig
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.policy_gradient import _save
for seed, path in enumerate(sys.argv[1:]):
    torch.manual_seed(seed)
    config = ModelTransformerConfig(
        device=torch.device("cpu"), d_entity=32, econ_transformer_layers=1, econ_transformer_heads=2,
        econ_transformer_ff_dim=64, d_map=32, hex_transformer_layers=1, hex_transformer_heads=2,
        hex_transformer_ff_dim=64, cross_attn_heads=2, d_trunk=64, num_res_blocks=1, value_head_layers=1,
    )
    _save(AlphaZeroTransformerModel(config), __import__("pathlib").Path(path))
PYEOF
}

step_pg() {
  if [ -f "$RUN_DIR/state.json" ]; then
    echo "pg: resuming $RUN_DIR"
    "${PY[@]}" main.py policy-gradient-generic --resume "$RUN_DIR" \
      ${RESUME_MAX_UPDATES:+--max-updates "$RESUME_MAX_UPDATES"}
    return
  fi
  dry_sl || return 1
  [ -f "$SL" ] || [ -n "${ECHO:-}" ] || { echo "pg: no SL checkpoint at $SL (gpu_check_1830.sh first)"; return 1; }
  write_population
  local arm=()
  if [ -n "$WANT_A1" ] && [ -n "$REF_KL" ]; then
    arm+=(--ref-kl-coef "$REF_KL" --ref-checkpoint "$SL")
    [ -n "$REF_T" ] && arm+=(--ref-temperature "$REF_T")
  fi
  [ -n "$WANT_A2" ] && arm+=(--start-priority --start-priority-score abs_adv --start-priority-uniform 0.3
    --start-priority-staleness 0.3 --start-priority-temperature 1.0 --start-priority-ema 0.5)
  [ -n "$WANT_A3" ] && arm+=(--population-reload-every 2)
  "${PY[@]}" main.py policy-gradient-generic --run-name "$RUN" --out-dir "$OUT_DIR" --policy "$SL" "${GAME_ARGS[@]}" \
    "${PG_SCALE[@]}" --population "$POP" --pfsp-power 2.0 --lr 2e-5 --kl-coef 0.02 --kl-target 0.08 \
    --kl-coef-min 0.005 --entropy-coef 0.0 --learner-temperature 0.5 --opponent-temperature 0.5 \
    --max-updates "$MAX_UPDATES" --pool-refresh-every 5 "${arm[@]}"
}

step_exploiters() {
  [ -n "$WANT_A3" ] || { echo "exploiters: arm $ARM has no exploiters (A3, or A4 with WITH_A3=1)"; return 1; }
  # The exploiter recipe: the main's games (GAME_ARGS) and scale, the frozen main in every other seat (the loop adds
  # --exploiter-of / --policy / --value / --max-updates), placement reward, no KL, an entropy bonus.
  "${PY[@]}" scripts/launch/2026-11/bridge/exploiter_loop.py --main-run "$RUN_DIR" --population "$POP" \
    --every "$EXP_EVERY" --updates "$EXP_UPDATES" --min-share "$EXP_MIN_SHARE" --pinned-latest "$EXP_PINNED" \
    --exploiter-snapshot-every "$EXP_SNAPSHOT_EVERY" --poll "$EXP_POLL" --python "${PY[*]}" \
    ${MAIN_MAX_UPDATES:+--main-max-updates "$MAIN_MAX_UPDATES"} -- \
    "${GAME_ARGS[@]}" "${PG_SCALE[@]}" --anchor-every 0 --kl-coef 0 --lr "$EXP_LR" --entropy-coef "$EXP_ENTROPY" \
    --learner-temperature "$EXP_T" --opponent-temperature 0.5 --opponent-slots 1
}

# Newest learner snapshot at a multiple of EVAL_EVERY.
newest_snapshot() {
  local u best=
  for f in "$RUN_DIR"/learner/*.pth; do
    [ -e "$f" ] || continue
    u=$(basename "$f" .pth)
    [ $((u % EVAL_EVERY)) = 0 ] && [ "$u" -gt 0 ] && { [ -z "$best" ] || [ "$u" -gt "$best" ]; } && best=$u
  done
  echo "$best"
}

run_eval() {  # name players a_seats B-player
  local name=$1 n=$2 a=$3 b=$4
  if [ -e "$EVAL_OUT/$name/summary.json" ]; then
    echo "evals: $name done, kept"
    return
  fi
  OMP_NUM_THREADS=1 "${PY[@]}" scripts/eval_policy_only.py --match "$L" "$b" --num-players "$n" --a-seats "$a" \
    --games "$EVAL_GAMES" --auction-start --workers "$EVAL_WORKERS" "${EVAL_SERVER[@]}" --save-games 6 \
    --seed 0 --out "$EVAL_OUT/$name" > "$EVAL_OUT/$name.log" 2>&1
  local status=$?
  tail -1 "$EVAL_OUT/$name.log" | sed "s#^#$name #" | tee -a "$EVAL_OUT/results.txt"
  [ $status = 0 ] || echo "evals: $name failed ($status), see $EVAL_OUT/$name.log" | tee -a "$EVAL_OUT/results.txt"
}

step_evals() {
  # pg_generic_1830.sh's round (policy-only, T 0.5, real-rules auction starts, 400 games): vs the generic SL (4p 2v2,
  # 1v3, 2p), the legacy outside panel (SL ckpt 10 4p 2v2 / 1v3, gen6 u2400 at 2-6 players); plus 4p 1v3 alone vs the
  # SL at T 1.0 and vs each loop exploiter of this run (M3-style).
  U=${EVAL_UPDATE:-$(newest_snapshot)}
  [ -n "$U" ] || { echo "evals: no snapshot at a multiple of $EVAL_EVERY in $RUN_DIR/learner yet"; return 1; }
  dry_standins || return 1
  L="generic:$RUN_DIR/learner/$U.pth@0.5"
  mkdir -p "$EVAL_OUT"
  local spec e
  for spec in "4 2 generic:$SL@0.5 sl" "4 1 generic:$SL@0.5 sl" "2 1 generic:$SL@0.5 sl" "4 1 generic:$SL@1.0 sl_t1" \
              "4 2 $SL10@0.5 ckpt10" "4 1 $SL10@0.5 ckpt10" "4 1 $GEN6@0.5 gen6" "3 1 $GEN6@0.5 gen6" \
              "5 2 $GEN6@0.5 gen6" "6 3 $GEN6@0.5 gen6" "2 1 $GEN6@0.5 gen6"; do
    set -- $spec
    run_eval "u${U}_${1}p_${2}v$(( $1 - $2 ))_$4" "$1" "$2" "$3"
  done
  if [ -n "$WANT_A3" ] && [ -f "$POP" ]; then
    for e in $("${ROOT}/.venv/bin/python" -c "
import json, sys
data = json.load(open(sys.argv[1]))
for m in (data['members'] if isinstance(data, dict) else data):
    if isinstance(m, dict) and m.get('exploiter_loop'):
        print(m['path'])" "$POP"); do
      run_eval "u${U}_4p_1v3_$(basename "$(dirname "$(dirname "$e")")")" 4 1 "generic:$e@0.5"
    done
  fi
}

step_evalloop() {
  local done_u= u
  while :; do
    u=$(newest_snapshot)
    if [ -n "$u" ] && [ "$u" != "$done_u" ]; then
      EVAL_UPDATE=$u step_evals
      done_u=$u
    fi
    if [ -e "$RUN_DIR/STOP" ] || { [ -n "$u" ] && [ "$u" -ge "${RESUME_MAX_UPDATES:-$MAX_UPDATES}" ]; }; then
      echo "evalloop: run stopped"
      return
    fi
    sleep "${EVALLOOP_SLEEP:-600}"
  done
}

echo "robust arm $ARM: steps ${STEPS[*]}${DRY:+ (DRY RUN under $DRY_ROOT)}; SL $SL; run $RUN_DIR; league $POP"
for s in "${STEPS[@]}"; do
  case "$s" in
    pg | exploiters | evals | evalloop) "step_$s" || { echo "$ARM: step $s failed"; exit 1; } ;;
    *) echo "unknown step $s (pg exploiters evals evalloop)"; exit 2 ;;
  esac
done
