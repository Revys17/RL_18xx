"""Build the human self-play start positions: each usable recorded game's
actions up to its first Stock Round, one game per line.

Self-play can start at Stock Round 1 instead of the private auction
(``SelfPlayHyperparams.start_positions_path``; see
``rl18xx/agent/alphazero/start_positions.py``). This takes the cleaned human
games (``pretraining.load_games_from_json``), keeps those with the requested
player count and no optional rules -- finished or not (~13% are unfinished;
only their auction is used) -- replays each through the Rust
engine until the auction (and the B&O par) is over, and writes

    {"id": "<game id>", "num_players": 4, "actions": [{"type": "bid", ...}, ...]}

per line. Actions keep only what ``process_action`` reads (no user ids,
timestamps or chat). Every prefix is replayed again strictly, in both engines,
and must reach the same first-Stock-Round position.

The cleaned exports carry only the optional rules the engines implement, so
the raw 18xx.games exports (``--raw-dir``) are consulted for the rest (e.g.
``multiple_brown_from_ipo``).

Usage:
    uv run python scripts/build_start_positions.py [--clean-dir human_games/1830_clean_all] \
        [--raw-dir human_games/1830 --raw-dir human_games/scrape_2026_10/1830] \
        [--num-players 4] [--out human_games/start_positions_1830_4p.jsonl]
"""

import argparse
import collections
import json
import logging
import os
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rl18xx.agent.alphazero.pretraining import load_games_from_json  # noqa: E402
from rl18xx.agent.alphazero.start_positions import (  # noqa: E402
    apply_actions,
    human_start_prefix,
    in_auction,
    new_game,
    position_summary,
    python_game,
)


def raw_optional_rules(game_id: str, raw_dirs: list) -> list | None:
    """The optional rules of a game's raw 18xx.games export, or None if there is none."""
    for raw_dir in raw_dirs:
        path = Path(raw_dir) / f"{game_id}.json"
        if path.exists():
            with open(path) as f:
                return list((json.load(f).get("settings") or {}).get("optional_rules") or [])
    return None


def check_prefix(prefix: list, num_players: int) -> str | None:
    """None if ``prefix`` replays strictly to a first Stock Round identically in both engines."""
    try:
        rust = apply_actions(new_game(num_players), prefix)
    except Exception:
        return "strict_replay_error"
    if in_auction(rust):
        return "strict_replay_in_auction"
    try:
        python = python_game(num_players, prefix)
    except Exception:
        return "python_engine_error"
    if position_summary(rust) != position_summary(python):
        return "engine_parity"
    return None


def _sort_key(game_id: str):
    return (0, int(game_id), "") if game_id.isdigit() else (1, 0, game_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", default="human_games/1830_clean_all")
    parser.add_argument(
        "--raw-dir",
        action="append",
        default=None,
        help="raw 18xx.games exports, for optional rules (default: human_games/1830, human_games/scrape_2026_10/1830)",
    )
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--out", default=None, help="default: human_games/start_positions_1830_<N>p.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    raw_dirs = args.raw_dir or ["human_games/1830", "human_games/scrape_2026_10/1830"]
    out = Path(args.out or f"human_games/start_positions_1830_{args.num_players}p.jsonl")

    games = load_games_from_json(args.clean_dir)
    excluded = collections.Counter()
    records = []
    for game in games:
        gid = str(game["id"])
        if game.get("status") == "error" or "title" not in game:
            excluded["cleaning_dropped"] += 1
            continue
        if len(game.get("players") or []) != args.num_players:
            excluded["player_count"] += 1
            continue
        if (game.get("settings") or {}).get("optional_rules"):
            excluded["optional_rules"] += 1
            continue
        rules = raw_optional_rules(gid, raw_dirs)
        if rules is None:
            excluded["no_raw_export"] += 1
            continue
        if rules:
            excluded["optional_rules"] += 1
            continue
        prefix, reason = human_start_prefix(game)
        reason = reason or check_prefix(prefix, args.num_players)
        if reason:
            excluded[reason] += 1
            continue
        records.append({"id": gid, "num_players": args.num_players, "actions": prefix})
    records.sort(key=lambda r: _sort_key(r["id"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.chmod(tmp, 0o644)  # mkstemp creates it 0600
    os.replace(tmp, out)

    print(f"{len(games)} cleaned games in {args.clean_dir}")
    for reason, count in excluded.most_common():
        print(f"  excluded {reason}: {count}")
    print(f"{len(records)} usable {args.num_players}-player starts -> {out}")
    if not records:
        return
    lengths = [len(r["actions"]) for r in records]
    print(
        f"prefix actions: min {min(lengths)}, median {statistics.median(lengths)}, max {max(lengths)}; "
        f"passes dropped as unprocessable are not included"
    )
    pars, cash = collections.Counter(), collections.defaultdict(list)
    for record in records:
        summary = position_summary(apply_actions(new_game(args.num_players), record["actions"]))
        pars[summary["bo_par"]] += 1
        for player, amount in summary["cash"].items():
            cash[player].append(amount)
    print("B&O par:", dict(sorted(pars.items())))
    print("median SR1 cash by seat:", {p: statistics.median(c) for p, c in sorted(cash.items())})


if __name__ == "__main__":
    main()
