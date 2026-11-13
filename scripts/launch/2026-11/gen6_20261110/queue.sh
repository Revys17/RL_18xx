#!/bin/bash
# Launches gen6 when gen5 stops on its plateau rule (the user, 2026-11-10 22:55: "Yeah if gen5 stalls just go ahead
# with gen6"). Waits for gen5's STOP file (touched by gen5's evals2.sh after two flat hundreds) and for unit
# rl18xx-gen5 to be gone, then: G5 = gen5's last snapshot (learner + critic), the league file league_v8_gen5.json
# (league_v7_gen4 + gen5/G5 + pg2p u2000), unit rl18xx-gen6 (run.sh, MPS env), unit rl18xx-gen6-evals (evals.sh).
# If gen5 ends WITHOUT a STOP file (its 4000 cap, or a failure that won't restart) nothing launches.
# G5_PIN=<u> starts gen6 from that snapshot instead (2026-11-11: the user chose u2800). Touch HOLD in this directory to keep it from launching. Run as unit rl18xx-gen6-queue.
set -uo pipefail
X=/home/revys/longrun/experiments/gen6_20261110
W=/home/revys/longrun/work/lr1
R5=$W/model_checkpoints_pg/gen5_20261107
D=/home/revys/longrun/deploy/gen6-5d0ac8e
POP=$W/model_checkpoints_pg/populations/league_v8_gen5.json
PG2P=model_checkpoints_pg/pg2p_20261109/learner/2000.pth
MPS=(-E CUDA_MPS_PIPE_DIRECTORY=/home/revys/longrun/ops/mps/pipe -E CUDA_MPS_LOG_DIRECTORY=/home/revys/longrun/ops/mps/log)
log() { echo "$(date '+%F %T') $*" >> $X/results.txt; }
state() { systemctl --user show -p ActiveState --value rl18xx-gen5; }

log "queue: waiting for gen5's plateau stop (STOP file + rl18xx-gen5 gone), then gen6"
while :; do
  [ -f $X/HOLD ] && { log "queue: HOLD file; not launching gen6"; exit 0; }
  s=$(state)
  if [ "$s" = inactive ] || [ "$s" = failed ]; then
    sleep 30; s=$(state)   # Restart=on-failure: a failed run comes back after 120 s, so look twice
    if [ "$s" = inactive ] || [ "$s" = failed ]; then
      [ -f $R5/STOP ] && break
      log "queue: gen5 ended ($s) without a STOP file (not a plateau); gen6 NOT launched"; exit 1
    fi
  fi
  sleep 60
done

G5=${G5_PIN:-$(ls $R5/learner/ | sed 's/\.pth$//' | grep -E '^[0-9]+$' | sort -n | tail -1)}  # G5_PIN: a chosen snapshot
[ -n "$G5" ] && [ -f $R5/critic/$G5.pth ] || { log "queue: no learner+critic pair for gen5's last snapshot ($G5)"; exit 1; }
bash $D/scripts/longrun/deploy.sh verify $D > /dev/null 2>&1 \
  || { log "queue: deployment $D does not verify; gen6 NOT launched"; exit 1; }
if [ ! -f $POP ]; then
  python3 - "$POP" "$G5" "$PG2P" <<'EOF' || { log "queue: could not write the league file"; exit 1; }
import json, sys
pop, g5, pg2p = sys.argv[1], sys.argv[2], sys.argv[3]
base = json.load(open(pop.replace("league_v8_gen5.json", "league_v7_gen4.json")))
members = base["members"] + [
    {"path": f"model_checkpoints_pg/gen5_20261107/learner/{g5}.pth", "note": f"gen5/{g5} (gen6's start)"},
    {"path": pg2p, "note": "pg2p/2000 (the 2-player PG run's final; strongest 2-player policy)"},
]
json.dump({"note": f"gen6 (2026-11-10): league_v7_gen4 (31 members) + gen5/{g5} + pg2p/2000", "members": members},
          open(pop, "w"), indent=1)
EOF
fi
log "queue: gen5 stopped; launching gen6 from gen5 u$G5 (deployment $D, league $(basename $POP))"
systemd-run --user --unit rl18xx-gen6 -p MemoryHigh=110G -p MemoryMax=130G -p Restart=on-failure -p RestartSec=120 \
  "${MPS[@]}" -E D=$D -E G5=$G5 /usr/bin/bash $X/run.sh
sleep 90
systemctl --user is-active -q rl18xx-gen6 || { log "queue: rl18xx-gen6 not active 90 s after launch"; exit 1; }
systemd-run --user --unit rl18xx-gen6-evals "${MPS[@]}" -E D=$D -E G5=$G5 /usr/bin/bash $X/evals.sh
log "queue: gen6 + evals started"
