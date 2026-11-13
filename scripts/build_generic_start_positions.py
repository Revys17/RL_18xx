"""Build start files for the title-generic pipeline (RL bridge milestone C2; design §5.6): human
post-auction positions (first Stock Round) and mid-game cuts (the start of Stock Rounds 2-6), per player
count, for any title with a human-game replay (``generic_play.replay_human_game``).

Generalizes ``scripts/build_start_positions.py`` / ``build_midgame_positions.py``: each usable game is
replayed by the engine, its own action log (``raw_actions``) is cut where Stock Rounds begin
(``generic_play.stock_round_cuts``), and every prefix is replayed again strictly and must stop at an
unfinished Stock Round. Games with optional rules are left out (starts are played under the base rules).
Lines carry ``title`` and, for titles whose engine takes a seed (1867, the 1822 family), ``seed``::

    start_positions_<title>_<N>p<suffix>.jsonl    {"id", "title", "num_players", ["seed",] "actions"}
    midgame_positions_<title>_<N>p<suffix>.jsonl  {"id", "title", "num_players", ["seed",] "actions", "cuts"}

Sources: 1830 reads cleaned exports (``main.py clean``; replayed as the generic importer replays them,
``--raw-dir`` consulted for optional rules as the legacy builders do); 1867 and the 1822 family read raw
18xx.games exports (replayed as the corpus audits do, ``tests/title_replay_harness``: run from the repository
root). A title whose game opens with a Stock Round (the 1822 family) has empty SR1 starts (a fresh game):
only its mid-game file is useful.

Random openings are not built here: generators play them at run time (``generic_play.random_opening_actions``).
Random starts and auction starts need no file.

    uv run python scripts/build_generic_start_positions.py --title 1830 --source human_games/1830_clean_20261017 \\
        --out-dir human_games --suffix _generic --num-players 4
    uv run python scripts/build_generic_start_positions.py --title 1867 \\
        --source human_games/1867 --source human_games/scrape_2026_10/1867 --out-dir human_games --jobs 8
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def raw_optional_rules(game_id: str, raw_dirs: list):
    """The optional rules of a game's raw 18xx.games export, or None if there is none."""
    for raw_dir in raw_dirs:
        path = Path(raw_dir) / f"{game_id}.json"
        if path.exists():
            with open(path) as f:
                return list((json.load(f).get("settings") or {}).get("optional_rules") or [])
    return None


def game_files(sources: list) -> list:
    seen = {}
    for src in sources:
        p = Path(src)
        for f in sorted(p.glob("*.json")) if p.is_dir() else [p]:
            seen.setdefault(f.stem, f)
    return [seen[k] for k in sorted(seen, key=lambda s: (len(s), s))]


_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args
    logging.getLogger().setLevel(logging.ERROR)


def process(path: str) -> tuple:
    """``(game id, n, start record or None, midgame record or None, excluded reason or None)``."""
    from rl18xx.agent.alphazero.generic_play import replay_human_game, start_records

    args = _ARGS
    gid = Path(path).stem
    try:
        game = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return gid, 0, None, None, "unreadable"
    game.setdefault("id", gid)
    if args.title == "1830":
        game["id"] = gid  # pretraining.load_games_from_json's convention
        if game.get("status") == "error" or "title" not in game:
            return gid, 0, None, None, "cleaning_dropped"
    else:
        if str(game.get("title")) != args.title:
            return gid, 0, None, None, "other_title"
        if game.get("status") not in ("finished", "archived", "active", None):
            return gid, 0, None, None, f"status_{game.get('status')}"
    n = len(game.get("players") or [])
    if args.num_players and n not in args.num_players:
        return gid, n, None, None, "player_count"
    if (game.get("settings") or {}).get("optional_rules"):
        return gid, n, None, None, "optional_rules"
    if args.title == "1830" and args.raw_dir:
        rules = raw_optional_rules(gid, args.raw_dir)
        if rules is None:
            return gid, n, None, None, "no_raw_export"
        if rules:
            return gid, n, None, None, "optional_rules"
    replayed, reason = replay_human_game(args.title, game)
    if replayed is None:
        return gid, n, None, None, (reason or "replay_failed").split(":")[0]
    if replayed.optional_rules:
        return gid, n, None, None, "optional_rules"
    start, midgame, reason = start_records(args.title, replayed, args.first_round, args.last_round)
    return gid, n, start, midgame, reason


def _write(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--title", default="1830")
    parser.add_argument("--source", action="append", default=None, help="Game JSON directories or files")
    parser.add_argument(
        "--raw-dir", action="append", default=None,
        help="1830: raw 18xx.games exports, for optional rules "
        "(default human_games/1830, human_games/scrape_2026_10/1830)",
    )
    parser.add_argument("--num-players", type=int, action="append", default=None, help="(default: the title's range)")
    parser.add_argument("--first-round", type=int, default=2)
    parser.add_argument("--last-round", type=int, default=6)
    parser.add_argument("--out-dir", default="human_games")
    parser.add_argument("--suffix", default="", help="File name suffix, e.g. _generic or _v3")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N files (by id)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(args.title)
    if not args.num_players:
        args.num_players = list(spec.player_counts)
    if args.source is None:
        args.source = ["human_games/1830_clean_20261017"] if args.title == "1830" else list(spec.human_games_dirs)
    if args.title == "1830" and args.raw_dir is None:
        args.raw_dir = ["human_games/1830", "human_games/scrape_2026_10/1830"]
    files = game_files(args.source)
    if args.limit:
        files = files[: args.limit]

    excluded = collections.Counter()
    starts, midgames = collections.defaultdict(list), collections.defaultdict(list)
    rounds = collections.Counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init, initargs=(args,)) as pool:
        for gid, n, start, midgame, reason in pool.imap_unordered(process, [str(f) for f in files], chunksize=8):
            if start is None and midgame is None:
                excluded[reason or "unknown"] += 1
                continue
            if reason:
                excluded[f"sr1_{reason}"] += 1
            if start is not None:
                starts[n].append(start)
            if midgame is not None:
                midgames[n].append(midgame)
                rounds.update(midgame["cuts"].keys())

    def key(r):
        return (0, int(r["id"]), "") if str(r["id"]).isdigit() else (1, 0, str(r["id"]))

    out_dir = Path(args.out_dir)
    print(f"{len(files)} {args.title} game files from {args.source}")
    for reason, count in excluded.most_common():
        print(f"  excluded {reason}: {count}")
    print(f"  mid-game starts by Stock Round: {dict(sorted(rounds.items()))}")
    for n in sorted(args.num_players):
        s_path = out_dir / f"start_positions_{args.title}_{n}p{args.suffix}.jsonl"
        m_path = out_dir / f"midgame_positions_{args.title}_{n}p{args.suffix}.jsonl"
        _write(s_path, sorted(starts[n], key=key))
        _write(m_path, sorted(midgames[n], key=key))
        print(f"{n}p: {len(starts[n])} SR1 starts -> {s_path}; {len(midgames[n])} mid-game games -> {m_path}")


if __name__ == "__main__":
    main()
