# Launch scripts, October 2026

How the later runs in `docs/training_runs_2026_10.md` were actually started and
checked, kept as a record (they wait on each other's log lines and name exact
checkpoints, so they are not meant to be rerun as they are). Each `cd`s to the
repository root; results went to `logs/<name>.log` and `logs/eval/`.

| Script | What it ran |
|---|---|
| `gen3_resume.sh` | round-3 goal exploiters (bankruptcy hunter, rust, placement) against gen2/1500, population v5, then gen3 = the gen1 run resumed 1500 -> 2000 with 15% auction starts, with checks every 100 updates |
| `reward_ab.sh` | three branches from gen3/2000 differing only in the reward (win / placement / blend), 300 updates each, checks at 150 and 300, round robin |
| `after_ab.sh` | picks the round-robin winner and continues it as gen4 (300 -> 800) |
| `gen4b.sh`, `gen4c.sh` | gen4 800 -> 1300 against population v7, then 1040 -> 1800 at 96 games per worker task |
| `value_v3.sh` | value_v3: value_v2 fine-tuned on 400k policy games of gen4/800, scored on human and agent games |
| `qtarget_pilot.sh` | the Q-target pilot: `--target cq` from az4 u100 with and without KataGo surprise weighting (the flattening test, docs/search_learning_design.md "Q-based policy targets"), then each arm's per-step entropies and near-greedy check |
| `battery.sh` | the validation battery before the October pause: the generational ladder (2v2 and 1v3), gen4 against every population member, search tests (void: pre-3217c28 tree), the bankruptcy hunter, style samples |

Reusable tools that grew out of these live in `scripts/` (`count_bankruptcies.py`,
`search_decision_diag.py`, `search_override_rollouts.py`).
