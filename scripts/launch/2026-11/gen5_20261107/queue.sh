#!/bin/bash
# Waits for the 2-player loop to end (unit rl18xx-l2p), then starts gen5 as unit rl18xx-gen5 (restarts resume).
until ! systemctl --user is-active -q rl18xx-l2p; do sleep 300; done
[ -f /home/revys/longrun/experiments/gen5_20261107/HOLD ] && { echo "HOLD file: not starting gen5"; exit 0; }
echo "$(date '+%F %T') gen5: start" >> /home/revys/longrun/experiments/gen5_20261107/results.txt
systemd-run --user --unit rl18xx-gen5 -p MemoryHigh=110G -p MemoryMax=130G -p Restart=on-failure -p RestartSec=120 \
  bash /home/revys/longrun/experiments/gen5_20261107/run.sh
