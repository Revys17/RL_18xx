"""A ``pretrain-generic`` run against chance: the per-family top-1 report the new titles' supervised runs are judged by
(``docs/rl_model.md`` "GPU run queue"; the CPU smokes' tables in "1867: data, starts, caps and the CPU smoke" and "1822
family: ... the CPU smoke" are this report's).

Prints the training windows (first / last), every joint-stage validation, the value refit, top-1 by player count and
-- from the last joint validation -- top-1 per descriptor family next to chance (mean 1 / legal count over the same
validation games: the run's ``val_max_games`` first ones, all by default). A family is ``BELOW`` when its top-1 is under
chance with at least ``--min-n`` positions; ``final_eval`` (``--final-eval``) is printed too when the run has it.

    uv run python scripts/generic_sl_report.py <run dir> [--data <dataset dir>] [--out <file>]

``--data`` defaults to the run's own dataset (``config.json``'s first source).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def family_chance(data: str, max_games) -> dict:
    """``{family: (n, chance top-1)}`` over the first ``max_games`` validation games of the dataset."""
    from rl18xx.agent.alphazero import generic_data as gd
    from rl18xx.agent.alphazero import model_generic as mg

    ds = gd.GenericDataset(data, check_layouts=False)
    families = mg.vocab()["families"]
    chance = defaultdict(list)
    ids = ds.game_ids("validation")
    for gid in ids[: max_games or len(ids)]:
        record = ds.record(gid)
        sizes = np.diff(np.asarray(record["legal_offsets"]))
        for fam, size in zip(np.asarray(record["family"]).tolist(), sizes.tolist()):
            chance[families[int(fam)]].append(1.0 / max(int(size), 1))
    return {fam: (len(v), float(np.mean(v))) for fam, v in chance.items()}


def report(run: Path, data: str = None, min_n: int = 20) -> str:
    config = json.loads((run / "config.json").read_text())
    pretrain = config.get("pretrain", {})
    data = data or config["sources"][0]["path"]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
    train = [h for h in history if h.get("kind") == "train"]
    evals = [h for h in history if h.get("kind") == "eval" and h.get("stage", "joint") == "joint"]
    lines = [f"run {run}", f"dataset {data}"]
    if train:
        first, last = train[0], train[-1]
        lines.append(
            f"train windows: first total {first['total']:.3f} policy_ce {first['policy_ce']:.3f} top1 "
            f"{first['top1']:.3f}; last (step {last.get('step')}) total {last['total']:.3f} policy_ce {last['policy_ce']:.3f} top1 "
            f"{last['top1']:.3f}; elapsed {last.get('elapsed')}"
        )
    for e in evals:
        lines.append(
            f"  validation step {e['step']} (epoch {e.get('epoch')}): positions {e['positions']} top1 {e['top1']:.3f} "
            f"top5 {e['top5']:.3f} (chance {e['chance_top1']:.3f}) policy_ce {e['policy_ce']:.3f} price_nll "
            f"{e.get('price_nll', float('nan')):.3f} win CE {e['value_ce']:.3f} (equal odds {e['value_baseline']:.3f}) "
            f"winner {e['winner_acc']:.3f}"
        )
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("joint"):
        lines.append(f"joint stage: {json.dumps(summary['joint'])}")
    if summary.get("refit"):
        r = summary["refit"]
        lines.append(
            f"value refit: {r.get('steps')} steps, validation win CE {r.get('val_value_ce', float('nan')):.3f}, "
            f"winner {r.get('val_winner_acc', float('nan')):.3f}"
        )
    if not evals:
        lines.append("no validation yet")
        return "\n".join(lines)
    last = evals[-1]
    lines.append(
        "top-1 by player count (last validation): "
        + ", ".join(f"{k}p {v['top1']:.3f} (n {v['n']})" for k, v in sorted(last.get("by_players", {}).items()))
    )
    chance = family_chance(data, pretrain.get("val_max_games"))
    lines.append(f"top-1 per family (last validation, step {last['step']}) vs chance (mean 1 / legal count):")
    lines.append(f"    {'family':22} {'n':>7} {'top1':>6} {'chance':>6}")
    below = []
    for fam, v in sorted(last.get("top1_by_family", {}).items(), key=lambda kv: -kv[1]["n"]):
        c = chance.get(fam, (0, float("nan")))[1]
        if c >= 0.999:
            flag = "forced"
        elif v["top1"] > c:
            flag = "above"
        elif v["n"] >= min_n:
            flag = "BELOW"
            below.append(fam)
        else:
            flag = "(small)"
        lines.append(f"    {fam:22} {v['n']:7d} {v['top1']:6.3f} {c:6.3f} {flag}")
    lines.append(f"families below chance with >= {min_n} positions: {', '.join(below) if below else 'none'}")
    for label, entry in (summary.get("final_eval") or {}).items():
        if isinstance(entry, dict):
            lines.append(
                f"final_eval {label} ({entry.get('split')}): top1 {entry.get('top1', float('nan')):.3f} policy_ce "
                f"{entry.get('policy_ce', float('nan')):.3f} win CE {entry.get('value_ce', float('nan')):.3f} "
                f"(equal odds {entry.get('value_baseline', float('nan')):.3f})"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", default=None, help="Dataset directory (default: the run's first source)")
    parser.add_argument("--min-n", type=int, default=20, help="Positions a family needs to be flagged BELOW chance")
    parser.add_argument("--out", type=Path, default=None, help="Also write the report here")
    args = parser.parse_args()
    text = report(args.run, args.data, args.min_n)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
