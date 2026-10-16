"""How a set of 4-player games was played: openings, train timing, payouts, length.

Each game is replayed in the Rust engine to place its moves in their stock /
operating rounds. Per collection:

- **opening**: the corporations parred before the first operating round (the
  B&O's par comes with its private), most common sets first
- **B&O in OR set 1**: the B&O laid a tile or bought a train before the second
  stock round (it floated in SR1; a new corporation has no train to run yet)
- **first 3 / 4 / D**: the stock round after which each train type was first bought
- **payout**: share of dividend decisions that paid out
- **stock rounds** per finished game, and the share of games ending in a bankruptcy

    uv run python scripts/style_profile.py logs/eval/member_profiles/gen4_final logs/eval/member_profiles/ck10
    uv run python scripts/style_profile.py human_games/1830_clean_all --limit 400   # human 4-player games

A collection is an eval output directory (``games/*.json``: ``make_game_record``
files or bare action lists) or a directory of 18xx.games exports; only
4-player games under the base rules (or ``optional_6_train``) that replay are
counted.
"""

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRAIN_TYPES = ("3", "4", "D")


def _actions_of(path: Path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data, {}
    players = data.get("players") or []
    if players and len(players) != 4:
        return None, None
    rules = (data.get("settings") or {}).get("optional_rules") or []
    if any(r != "optional_6_train" for r in rules):
        return None, None
    return data["actions"], rules


def profile_game(path: Path):
    """Style facts of one game, or None if it isn't a replayable 4-player game."""
    from rl18xx.agent.alphazero.start_positions import new_game

    actions, rules = _actions_of(path)
    if actions is None:
        return None
    game = new_game(4)
    if rules:
        return None  # new_game builds the base rules; skip the few 6-train games rather than misread them
    stock_rounds, previous_round = 0, None
    opening, first_train, bo_first_set = set(), {}, False
    dividends = payouts = 0
    try:
        for action in actions:
            round_name = game.round.__class__.__name__
            if round_name == "Stock" and previous_round != "Stock":
                stock_rounds += 1
            previous_round = round_name
            kind = action.get("type")
            if kind == "par" and stock_rounds <= 1 and round_name != "Operating":
                opening.add(action.get("corporation"))
            if kind in ("lay_tile", "buy_train") and action.get("entity") == "B&O" and stock_rounds == 1:
                bo_first_set = True
            if kind == "buy_train":
                train = str(action.get("train", "")).split("-")[0]
                if train in TRAIN_TYPES and train not in first_train:
                    first_train[train] = stock_rounds
            elif kind == "dividend":
                dividends += 1
                payouts += action.get("kind") == "payout"
            game._game.process_action(action)
    except Exception:
        return None
    if not game.finished:
        return None
    return {
        "opening": "+".join(sorted(c for c in opening if c and c != "B&O")) or "-",
        "bo_first_set": bo_first_set,
        "first_train": first_train,
        "payout": payouts / dividends if dividends else None,
        "stock_rounds": stock_rounds,
        "bankrupt": any(p.bankrupt for p in game._game.players),
    }


def summarize(label: str, records: list) -> dict:
    n = len(records)
    openings = collections.Counter(r["opening"] for r in records)
    out = {
        "label": label,
        "games": n,
        "openings": [(k, round(v / n, 3)) for k, v in openings.most_common(4)],
        "bo_first_set": round(sum(r["bo_first_set"] for r in records) / n, 3),
        "payout": round(statistics.mean(r["payout"] for r in records if r["payout"] is not None), 3),
        "stock_rounds": round(statistics.mean(r["stock_rounds"] for r in records), 1),
        "bankrupt": round(sum(r["bankrupt"] for r in records) / n, 3),
    }
    for t in TRAIN_TYPES:
        srs = [r["first_train"][t] for r in records if t in r["first_train"]]
        out[f"first_{t}"] = round(statistics.median(srs), 1) if srs else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collections", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="At most this many games per collection")
    parser.add_argument("--json", type=Path, help="Also write the summaries here")
    args = parser.parse_args()
    summaries = []
    for collection in args.collections:
        files = sorted((collection / "games").glob("*.json")) if (collection / "games").is_dir() else sorted(collection.glob("*.json"))
        records = []
        for path in files:
            if args.limit and len(records) >= args.limit:
                break
            record = profile_game(path)
            if record is not None:
                records.append(record)
        if not records:
            print(f"{collection}: no replayable 4-player games")
            continue
        s = summarize(collection.name, records)
        summaries.append(s)
        openings = ", ".join(f"{k} {v:.0%}" for k, v in s["openings"])
        print(
            f"{s['label']:16s} {s['games']:4d} games | {openings} | B&O in OR set 1 {s['bo_first_set']:.0%} | first 3/4/D "
            f"after SR {s['first_3']}/{s['first_4']}/{s['first_D']} | payout {s['payout']:.0%} | {s['stock_rounds']} SRs | "
            f"bankrupt {s['bankrupt']:.0%}"
        )
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
