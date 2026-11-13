# 2-player policy gradient + league from checkpoint 10 (DESIGN.md). Sourced by every script.
X=/home/revys/longrun/experiments/pg2p_20261109
D=/home/revys/longrun/deploy/az0-2p-213ec33   # all 2026-11-08 audit fixes (samplers, league crediting, bankrupt last)
PY=$D/.venv/bin/python
W=/home/revys/longrun/work/lr1
NP=2
RUN=${RUN:-pg2p_20261109}
CK10=$W/model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth
GEN4=$W/model_checkpoints_pg/rab_place_20261014/learner/1800.pth
CRITIC=model_checkpoints_value/l2p_v0/promoted/0.pth   # the 2p loop's V0: a value of checkpoint 10's own 2p games
UPDATES=${UPDATES:-2000}
MAXDEC=1500
STALL=150
ckpt() { if [ "$1" = 0 ]; then echo "$CK10"; else echo "$W/model_checkpoints_pg/$RUN/learner/$1.pth"; fi; }
log() { echo "$(date '+%F %T') $*" >> $X/results.txt; }
