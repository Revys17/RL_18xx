#!/bin/bash
# Is the run's progress transitive? transitivity.sh U B1 B2 ...: u U against older selves u B at the same temperature 0.5
# (B 0 = checkpoint 10). Same settings as evals.sh; shares its GPU lock. First run 2026-11-09 08:10: 500 0 200.
set -uo pipefail
source /home/revys/longrun/experiments/pg2p_20261109/common.sh
cd $W
eval "$(sed -n '/^po() {/,/^}/p' $X/evals.sh)"
u=$1; shift
for b in "$@"; do po u${u}_vs_u${b} "$(ckpt $u)@0.5" "$(ckpt $b)@0.5"; done
log "transitivity check $u done"
