"""Build the mid-game start positions: recorded human games up to the beginning
of one of their later Stock Rounds, for self-play games that start mid-game.

Games that start at Stock Round 1 rarely reach the positions that drive most of
1830's strategy (trains about to rust, a company left without trains, a dump or
a bankruptcy in the air), so a share of self-play games can start from where
humans had taken a game instead (``start_positions.sample_start_position``,
``midgame_fraction``). This keeps the same games as
``scripts/build_start_positions.py`` -- the requested player count and no
optional rules -- replays each through the Rust engine with the pretraining
importer's leniency (``start_positions.human_midgame_cuts``), and writes

    {"id": "<game id>", "num_players": 4, "actions": [...], "cuts": {"2": 151, "3": 233, ...}}

per line: the applied actions up to the last cut, and the prefix length at
which each of Stock Rounds ``--first-round``..``--last-round`` begins. Every
prefix is replayed again strictly and must stop at that Stock Round with a
player to act. Actions keep what the engine logged (no user ids, timestamps or
chat).

Usage:
    uv run python scripts/build_midgame_positions.py [--clean-dir human_games/1830_clean_all] \\
        [--num-players 4] [--first-round 2] [--last-round 6] [--out human_games/midgame_positions_1830_4p.jsonl]
"""

import argparse
import collections
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_start_positions import _sort_key, raw_optional_rules  # noqa: E402

from rl18xx.agent.alphazero.pretraining import load_games_from_json  # noqa: E402
from rl18xx.agent.alphazero.start_positions import apply_actions, human_midgame_cuts, new_game  # noqa: E402


def check_cuts(actions: list, cuts: dict, num_players: int) -> str | None:
    """None if every cut replays strictly to the start of a Stock Round, unfinished."""
    for stock_round, length in sorted(cuts.items()):
        try:
            game = apply_actions(new_game(num_players), actions[:length])
        except Exception:
            return "strict_replay_error"
        if game.finished or game._game.round.round_type != "Stock":
            return "cut_not_at_stock_round"
    return None


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
    parser.add_argument("--first-round", type=int, default=2)
    parser.add_argument("--last-round", type=int, default=6)
    parser.add_argument("--out", default=None, help="default: human_games/midgame_positions_1830_<N>p.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    raw_dirs = args.raw_dir or ["human_games/1830", "human_games/scrape_2026_10/1830"]
    out = Path(args.out or f"human_games/midgame_positions_1830_{args.num_players}p.jsonl")

    games = load_games_from_json(args.clean_dir)
    excluded = collections.Counter()
    rounds = collections.Counter()
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
        actions, cuts, reason = human_midgame_cuts(game, args.first_round, args.last_round)
        reason = reason or check_cuts(actions, cuts, args.num_players)
        if reason:
            excluded[reason] += 1
            continue
        rounds.update(cuts.keys())
        records.append(
            {"id": gid, "num_players": args.num_players, "actions": actions, "cuts": {str(k): v for k, v in cuts.items()}}
        )
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
    print(f"  starts by Stock Round: {dict(sorted(rounds.items()))}")
    print(f"{len(records)} usable {args.num_players}-player games -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
