"""Build start files for the title-generic pipeline (RL bridge milestone C2; design §5.6): human
post-auction positions (first Stock Round) and mid-game cuts (the start of Stock Rounds 2-6), per player
count, for any title with a human-game replay (``generic_play.replay_human_game``).

Generalizes ``scripts/build_start_positions.py`` / ``build_midgame_positions.py``: each usable game is
replayed by the engine, its own action log (``raw_actions``) is cut where Stock Rounds begin
(``generic_play.stock_round_cuts``), and every prefix is replayed again strictly and must stop at an
unfinished Stock Round. Games with optional rules are left out (starts are played under the base rules), except
for a title that plays a rules mix (``titles.TitleSpec.rules_mix``: 1822's 1822+ and tax-haven combinations), whose
games under any combination of its ``optional_rules`` are kept, each line with its canonical ``optional_rules``
(``[]`` for base rules: a run draws a game's combination first, then a start of that combination). Lines carry
``title`` and, for titles whose engine takes a seed (1867, the 1822 family), ``seed``::

    start_positions_<title>_<N>p<suffix>.jsonl    {"id", "title", "num_players", ["seed",] ["optional_rules",]
                                                   "actions"}
    midgame_positions_<title>_<N>p<suffix>.jsonl  {"id", "title", "num_players", ["seed",] ["optional_rules",]
                                                   "actions", "cuts"}

Sources: 1830 reads cleaned exports (``main.py clean``; replayed as the generic importer replays them,
``--raw-dir`` consulted for optional rules as the legacy builders do); 1867 and the 1822 family read raw
18xx.games exports (replayed as the corpus audits do, ``tests/title_replay_harness``: run from the repository
root). A title whose opening auction is its first Stock Round (the 1822 family's bid boxes) gets the position
where that Stock Round ends instead (OR1, or the Choices round before it; ``generic_play.opening_end_cut``;
lines carry ``"start": "sr1_end"``): what an SR1 start is to 1830's and 1867's auctions.

Random openings are not built here: generators play them at run time (``generic_play.random_opening_actions``).
Random starts and auction starts need no file.

    uv run python scripts/build_generic_start_positions.py --title 1830 --source human_games/1830_clean_20261017 \\
        --out-dir human_games --suffix _generic --num-players 4
    uv run python scripts/build_generic_start_positions.py --title 1867 \\
        --source human_games/1867 --source human_games/scrape_2026_10/1867 --out-dir human_games --jobs 8
    uv run python scripts/build_generic_start_positions.py --title 1822 --suffix _v2 \\
        --source human_games/1822 --source human_games/scrape_2026_10/1822 --out-dir human_games --jobs 8
    # 1830's mid-game cuts extended to Stock Round 10 (the v2 files' SR2-6 plus SR7-10; robustness arm A2's library
    # reaches the loot-and-dumps of SR 6-9): human_games/midgame_positions_1830_<N>p_v2_sr10.jsonl, N = 2..6
    uv run python scripts/build_generic_start_positions.py --title 1830 --last-round 10 --suffix _v2_sr10 \\
        --only midgame --strip-meta --jobs 8
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
#: Action fields the engine ignores (``--strip-meta`` drops them, as the legacy 1830 builders did).
META_KEYS = ("created_at", "user")


def _init(args):
    global _ARGS
    _ARGS = args
    logging.getLogger().setLevel(logging.ERROR)


def process(path: str) -> tuple:
    """``(game id, n, start record or None, midgame record or None, excluded reason or None)``."""
    from rl18xx.agent.alphazero.generic_play import replay_human_game, start_records
    from rl18xx.agent.alphazero.titles import title_spec

    args = _ARGS
    spec = title_spec(args.title)
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
    rules = (game.get("settings") or {}).get("optional_rules")
    if rules and not (spec.rules_mix and spec.rules_in_scope(rules)):
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
    if replayed.optional_rules and not (spec.rules_mix and spec.rules_in_scope(replayed.optional_rules)):
        return gid, n, None, None, "optional_rules"
    start, midgame, reason = start_records(args.title, replayed, args.first_round, args.last_round)
    if args.strip_meta:  # the legacy files' actions: no timestamps or user ids (half the size)
        for record in (start, midgame):
            if record is not None:
                record["actions"] = [{k: v for k, v in a.items() if k not in META_KEYS} for a in record["actions"]]
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
    parser.add_argument(
        "--only", choices=("start", "midgame"), default=None,
        help="Write only the SR1 start files or only the mid-game files (default both)",
    )
    parser.add_argument(
        "--strip-meta", action="store_true",
        help="Drop the actions' created_at / user fields (the legacy 1830 files' form; the engine ignores them)",
    )
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
    by_rules = collections.defaultdict(collections.Counter)  # rules name -> {(kind, n): lines}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init, initargs=(args,)) as pool:
        for gid, n, start, midgame, reason in pool.imap_unordered(process, [str(f) for f in files], chunksize=8):
            if start is None and midgame is None:
                excluded[reason or "unknown"] += 1
                continue
            if reason:
                excluded[f"sr1_{reason}"] += 1
            name = spec.rules_name((start or midgame).get("optional_rules"))
            if start is not None:
                starts[n].append(start)
                by_rules[name][("start", n)] += 1
            if midgame is not None:
                midgames[n].append(midgame)
                rounds.update(midgame["cuts"].keys())
                by_rules[name][("midgame", n)] += 1

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
        if args.only != "midgame":
            _write(s_path, sorted(starts[n], key=key))
        if args.only != "start":
            _write(m_path, sorted(midgames[n], key=key))
        kind = "SR1-end" if starts[n] and starts[n][0].get("start") == "sr1_end" else "SR1"
        cuts = collections.Counter(int(k) for m in midgames[n] for k in m["cuts"])
        s_out = f" -> {s_path}" if args.only != "midgame" else " (not written)"
        m_out = f" -> {m_path}" if args.only != "start" else " (not written)"
        print(f"{n}p: {len(starts[n])} {kind} starts{s_out}; {len(midgames[n])} mid-game games, "
              f"{sum(cuts.values())} cuts {dict(sorted(cuts.items()))}{m_out}")
    if spec.rules_mix or len(by_rules) > 1:
        counts = sorted(args.num_players)
        print("starts / mid-game games by rules combination and player count:")
        print(f"  {'rules':16} " + " ".join(f"{str(n) + 'p':>9}" for n in counts))
        for name, c in sorted(by_rules.items(), key=lambda kv: -sum(kv[1].values())):
            cells = " ".join(f"{c[('start', n)]:4}/{c[('midgame', n)]:<4}" for n in counts)
            print(f"  {name:16} {cells}")


if __name__ == "__main__":
    main()
