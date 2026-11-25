# Auction / critic probes (2026-11-25)

Ad-hoc probes behind `docs/robustness_measurement.md` "Why search stops the bots contesting". CPU, run from the repo
root with `.venv/bin/python` (`CUDA_VISIBLE_DEVICES=`). Checkpoint paths (gen6 u2400 in lr1's work dir, A4 u1528) are
written into the scripts.

- `match_auctions.py <collection>... | <human corpus dir>`: per private, the share with 2+ bidders, mean price over
  face, and the buyer's win rate uncontested vs contested, by side (A / B of an `eval_head_to_head` collection).
- `root_moves.py gen6|u1528 CA:165 ...`: the live search at the position after the scripted bids (`SEARCH=` env for
  other settings): each root move's prior, visits, Q, price grandchildren, and the critic's 1-ply value per move.
- `value_line.py gen6|u1528 <bid|pass> ...`: the critic's value of every seat along a scripted auction line.
- `pass_swing.py <games> <per game>`: the critics' value change on a pass that changes only whose turn it is.
- `mover_calibration.py <games> <per game>`: actual placement points minus the critic's prediction, mover vs others.
