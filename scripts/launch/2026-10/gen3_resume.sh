#!/bin/bash
# Round 3 after the 2026-10-13 reboot: gx2_bankrupt / gx2_rust were resumed from their
# snapshots by hand (180 / 170 -> 300). When one finishes, the placement exploiter starts
# (from checkpoint 10, 2 seats vs gen2/1500, 300 updates); each finished exploiter gets a clean
# 2v2 vs gen2/1500 and 60 self-play games for its style profile. Then gen3 = the main run resumed
# from 1500 to 2000 against population v5 (v4 + the three + gen2/1500), the new hunter and rust
# exploiter pinned at 10% (the old ones at 5%), 15% of games starting at the auction. Checks every
# 100 (clean @0.5): 2v2 vs gen2/1500, pg4/750 and the three new exploiters; 1v3 vs 3x gen2/1500.
# At 2000: MCTS @64 vs gen2/1500 and the new hunter's bankruptcy count (vs gen3 and vs gen2).
cd "$(dirname "$0")/../../.."  # the repository root
LOG=logs/gen3.log
RUN=model_checkpoints_pg/gen1_20261009
L=$RUN/learner
GEN2=$L/1500.pth
PG4=model_checkpoints_pg/pg4_20261007/learner/750.pth
CK10=model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
POP=model_checkpoints_pg/populations/league_v5.json
say() { echo "$(date -Iseconds) $*" >> $LOG; }
PG=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python main.py policy-gradient)
EV=(env OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_policy_only.py)
GOAL=(--policy $CK10 --value $RUN/critic/1500.pth --opponent $GEN2 --sl-opponent-fraction 1.0 --learner-seats 2
      --anchor-every 50 --learner-temperature 0.5 --opponent-temperature 0.5 --minibatch 2048 --microbatch 256
      --lr 2e-5 --kl-target 0.08 --midgame-fraction 0.3 --save-game-every 20)
running() { ps -eo args | grep -qE "[p]olicy-gradient .*$1"; }

evals() {  # evals NAME: clean 2v2 vs gen2/1500 and a style-profile sample of a finished exploiter
  local name=$1 final=model_checkpoints_pg/${1}_20261013/learner/300.pth
  if [ ! -f $final ]; then say "$name: no $final"; return; fi
  say "$name: done ($(grep -E 'update [0-9]+:' logs/policy_gradient_$name.log | tail -1 | sed 's/.*\(update [0-9]*:.*\)/\1/' | cut -c1-120))"
  "${EV[@]}" --match $final@0.5 $GEN2@0.5 --games 1200 --workers 8 --out logs/eval/${name}_2v2 \
    2> logs/eval_${name}_2v2.err | grep " vs " | sed "s#^#$(date -Iseconds) $name 2v2 #" >> $LOG
  "${EV[@]}" --match $final@0.5 $final@0.5 --games 60 --save-games 60 --workers 8 \
    --out logs/eval/member_profiles/$name > /dev/null 2>&1
  say "$name: evals done"
}

say "gen3_resume.sh: waiting for the resumed gx2_bankrupt / gx2_rust"
while running gx2_bankrupt && running gx2_rust; do sleep 60; done
say "placement exploiter (300 updates vs gen2/1500): started"
"${PG[@]}" "${GOAL[@]}" --reward placement --max-updates 300 --workers 28 --microbatch 1024 --run-name gx2_placement_20261013 \
  > logs/policy_gradient_gx2_placement.log 2>&1 &
PLACE_PID=$!
for name in gx2_bankrupt gx2_rust; do
  while running $name; do sleep 60; done
  evals $name
done
wait $PLACE_PID
evals gx2_placement

.venv/bin/python - <<'PY' >> $LOG 2>&1
import json
from pathlib import Path
pops = Path("model_checkpoints_pg/populations")
v = json.loads((pops / "league_v4.json").read_text())
for m in v["members"]:  # round-2 hostile exploiters keep a smaller pin
    if m.get("min_share"):
        m["min_share"] = 0.05
new = [("gx2_bankrupt", "bankruptcy hunter", 0.1), ("gx2_rust", "rust", 0.1), ("gx2_placement", "placement", None)]
for name, note, pin in new:
    path = Path(f"model_checkpoints_pg/{name}_20261013/learner/300.pth")
    if not path.is_file():
        print(f"missing {path}")
        continue
    member = {"path": str(path), "note": f"goal exploiter round 3: {note} (vs gen2/1500)"}
    if pin:
        member["min_share"] = pin
    v["members"].append(member)
v["members"].append({"path": "model_checkpoints_pg/gen1_20261009/learner/1500.pth", "note": "gen2/1500 (main agent before gen3)"})
(pops / "league_v5.json").write_text(json.dumps(v, indent=2) + "\n")
print(f"population v5: {len(v['members'])} members")
PY

"${PG[@]}" --resume $RUN --max-updates 2000 --population $POP --midgame-fraction 0.3 --auction-fraction 0.15 --microbatch 1024 \
  --save-game-every 20 > logs/policy_gradient_gen3.log 2>&1 &
PID=$!
say "gen3 started (pid $PID): resumed from 1500 against $POP, 15% auction starts"

H2=model_checkpoints_pg/gx2_bankrupt_20261013/learner/300.pth
R2=model_checkpoints_pg/gx2_rust_20261013/learner/300.pth
P2=model_checkpoints_pg/gx2_placement_20261013/learner/300.pth
check() {  # check NAME PREFIX cmd...: log the " vs " lines; retry twice on failure
  local name=$1 prefix=$2
  shift 2
  for attempt in 1 2 3; do
    if "$@" > logs/eval_$name.out 2> logs/eval_$name.err; then
      grep " vs " logs/eval_$name.out | sed "s#^#$(date -Iseconds) $prefix#" >> $LOG
      return 0
    fi
    say "$name: failed (attempt $attempt): $(grep -E 'Error|died' logs/eval_$name.err | tail -1 | cut -c1-200)"
    sleep 120
  done
}
for M in $(seq 1600 100 2000); do
  until [ -f $L/$M.pth ]; do
    if ! kill -0 $PID 2>/dev/null; then say "gen3 ended before $M ($(tail -1 logs/policy_gradient_gen3.log | cut -c1-160))"; exit 0; fi
    sleep 30
  done
  sleep 5
  check gen3_2v2_$M "2v2 " "${EV[@]}" --match $L/$M.pth@0.5 $GEN2@0.5 --match $L/$M.pth@0.5 $PG4@0.5 \
    --match $L/$M.pth@0.5 $H2@0.5 --match $L/$M.pth@0.5 $R2@0.5 --match $L/$M.pth@0.5 $P2@0.5 \
    --games 1200 --workers 16 --out logs/eval/gen3_2v2_$M
  check gen3_1v3_$M "1v3 " "${EV[@]}" --match $L/$M.pth@0.5 $GEN2@0.5 --a-seats 1 --games 1200 --workers 16 \
    --out logs/eval/gen3_1v3_$M
done
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/eval_head_to_head.py \
  --match $L/2000.pth@64 $GEN2@64 --games 240 --start-positions human_games/start_positions_1830_4p.jsonl --workers 24 \
  --out logs/eval/h2h_gen3_2000_vs_gen2 > logs/eval_h2h_gen3_2000.log 2>&1
python3 -c "import json; [print('h2h', s['A'][-35:], 'vs gen2/1500', round(s['A_score'],3), '+-', round(s['A_score_se'],3)) for s in json.load(open('logs/eval/h2h_gen3_2000_vs_gen2/summary.json'))]" 2>&1 | sed "s#^#$(date -Iseconds) #" >> $LOG
"${EV[@]}" --match $H2@0.5 $L/2000.pth@0.5 --match $H2@0.5 $GEN2@0.5 --games 400 --save-games 400 --workers 16 \
  --out logs/eval/bankrupt_check_gen3 > /dev/null 2>&1
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/count_bankruptcies.py logs/eval/bankrupt_check_gen3 2>&1 | grep -v Warn | sed "s#^#$(date -Iseconds) bankruptcies #" >> $LOG
wait $PID
say "gen3 exited $?"
