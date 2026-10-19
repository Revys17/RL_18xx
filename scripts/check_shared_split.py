"""Check an AlphaGo-style split pair of runs (docs/search_learning_design.md,
"AlphaGo-style split"): every checkpoint of the policy-gradient run (``--shared
--value-follow``) and of the value run (``search-learning --freeze-policy
--policy-follow``) is compared part by part with the other run's checkpoints, and
the histories' follow columns are printed.

A PG checkpoint's value side must equal some value-run checkpoint's (the one it
followed) and its policy path must differ from the start only by PPO; a value-run
checkpoint's policy path must equal some PG checkpoint's, its value side being its
own training.

    uv run python scripts/check_shared_split.py --pg model_checkpoints_ag/ag1_pg --value model_checkpoints_ag/ag1_val
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parts(path):
    from rl18xx.agent.alphazero.shared_split import checkpoint_state, is_value_key

    state = checkpoint_state(path)
    value = {k: v for k, v in state.items() if is_value_key(k) and not k.startswith(("aux_seat_head", "aux_global_head"))}
    policy = {k: v for k, v in state.items() if not is_value_key(k)}
    return policy, value


def same(a: dict, b: dict) -> bool:
    import torch

    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def checkpoints(run: Path) -> list:
    return sorted(((int(p.stem), p) for p in (run / "learner").glob("*.pth") if p.stem.isdigit()))


def history(run: Path, keys) -> list:
    path = run / "history.jsonl"
    if not path.exists():
        return []
    return [{k: r.get(k) for k in keys} for r in map(json.loads, path.read_text().splitlines()) if r]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pg", type=Path, required=True)
    parser.add_argument("--value", type=Path, required=True)
    args = parser.parse_args()
    pg = {n: parts(p) for n, p in checkpoints(args.pg)}
    val = {n: parts(p) for n, p in checkpoints(args.value)}
    print(f"PG checkpoints {sorted(pg)}, value-run checkpoints {sorted(val)}")
    ok = True
    for n, (policy, value) in sorted(pg.items()):
        src = [m for m, (_, v) in val.items() if same(value, v)]
        print(f"PG {n}: value side = value run {src or 'none (its start)'}")
    for n, (policy, value) in sorted(val.items()):
        src = [m for m, (p, _) in pg.items() if same(policy, p)]
        print(f"value run {n}: policy path = PG {src or 'none'}")
        ok &= bool(src) or n == min(val)
    if len(pg) > 1:
        first, last = pg[min(pg)][0], pg[max(pg)][0]
        print(f"PG policy path changed between {min(pg)} and {max(pg)}: {not same(first, last)}")
    if len(val) > 1:
        first, last = val[min(val)][1], val[max(val)][1]
        print(f"value side changed between value-run {min(val)} and {max(val)}: {not same(first, last)}")
    for r in history(args.pg, ("update", "value_follow_update", "value_follow_loaded", "critic_row_brier", "kl_anchor", "pg_loss")):
        print("PG history", r)
    for r in history(args.value, ("update", "policy_follow_update", "policy_follow_loaded", "critic_ce", "critic_outcome_ce", "rows_trained", "games")):
        print("value history", r)
    print("OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
