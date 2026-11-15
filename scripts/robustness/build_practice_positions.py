"""Practice positions for the robustness arms (docs/robustness_test_plan.md; arm A5p in
``scripts/launch/2026-11/bridge/robust_arm.sh``): positions at the start of the Stock Round one and two Stock Rounds
before every looted-broke dump (weakness 2 of docs/bot_weaknesses_2026_11.md) of the cleaned 1830 human corpus and of
Caleb's games against the bots (``data/caleb_games``).

A looted-broke dump is M2's ``looted_broke_dump`` (``behaviour_stats.broke_dumps``): a president sells out of a
floated, trainless company short of the next train's price, having first sold its train away for <= $10 or had it buy
one for >= 1.5x list within 150 actions. The games are replayed in the Python engine to find them
(``behaviour_events``: the same detection as M2's counts); the positions are cut from the Rust engine's own log of the
game (as the mid-game start files: ``generic_play.replay_human_game`` for the corpus, the converted actions for
Caleb's games), each prefix checked to replay strictly to the start of that Stock Round (Rust ``turn`` = the Stock
Round's number). A dump in Stock Round k is preceded by Stock Rounds k-1 and k-2; one in Operating Round k.x by k and
k-1 (``--srs-before 1 2``); a Stock Round cut shared by several dumps is written once.

Files (the mid-game start format, so the generators read them as they read mid-game files; one per player count)::

    practice_positions_1830_<N>p_v1.jsonl
    {"id": "<game id>" | "caleb-<site id>", "title": "1830", "num_players": N, "source": "human" | "caleb",
     "actions": [...up to the last cut...], "cuts": {"<k>": <prefix length>, ...},
     "dumps": [{"sr": 7, "round": "Stock", "corp": "NYC", "dumper": 1, "victim": 3, "victim_bankrupt": true,
                "phase": "5", "at": <action index or site id>}, ...],
     "cut_dumps": {"<k>": [[<dump index>, <Stock Rounds before>], ...]}}

A run draws them as ``practice:<id>:sr<k>`` starts (``policy-gradient-generic --practice-positions <template>
--practice-fraction f``; through A2's priorities with ``--start-priority``).

    uv run python -P scripts/robustness/build_practice_positions.py            # human_games/practice_positions_1830_*
    uv run python -P scripts/robustness/build_practice_positions.py --out-dir <dir> --workers 6 \\
        --source <hg>/1830_clean_20261017 --raw-dir <hg>/1830 --raw-dir <hg>/scrape_2026_10/1830  # hg: human_games
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

TITLE = "1830"
SUFFIX = "_v1"
DEFAULT_SOURCES = ["human_games/1830_clean_20261017"]
DEFAULT_RAW_DIRS = ["human_games/1830", "human_games/scrape_2026_10/1830"]
DEFAULT_CALEB = str(HERE / "data" / "caleb_games")


def practice_path(out_dir, num_players: int, suffix: str = SUFFIX) -> Path:
    return Path(out_dir) / f"practice_positions_{TITLE}_{int(num_players)}p{suffix}.jsonl"


def cut_rounds(dump: dict, srs_before) -> list:
    """The Stock Rounds whose start precedes ``dump`` by ``srs_before`` Stock Rounds (>= 1): a dump in Stock Round k
    is preceded by k-1, k-2, ...; one in an Operating Round after Stock Round k by k, k-1, ..."""
    k = int(dump["sr"])
    last = k - 1 if dump.get("round") == "Stock" else k
    return [last - (b - 1) for b in srs_before if last - (b - 1) >= 1]


def _clean(action: dict) -> dict:
    return {k: v for k, v in action.items() if k not in ("created_at", "user")}


def engine_log(path: str, data: dict, source: str, raw_dirs, allow_optional_rules: bool = False) -> tuple:
    """``(game id, players, engine actions, None)`` or ``(id, players, None, reason)``: the Rust engine's own log of
    a corpus game (the generic importer's replay, as the start files) or a converted site game's actions. Games under
    optional rules are left out (runs play 1830's base rules) unless ``allow_optional_rules``: then their cuts are
    kept where the prefix replays strictly under the base rules (checked by the caller)."""
    from rl18xx.agent.alphazero.generic_play import replay_human_game

    if source == "caleb":
        gid = str(data.get("id") or f"caleb-{Path(path).name.split('.')[0]}")
        rules = (data.get("settings") or {}).get("optional_rules") or []
        if rules and not allow_optional_rules:
            return gid, len(data["players"]), None, "optional_rules"
        return gid, len(data["players"]), [_clean(a) for a in data["actions"]], None
    gid = Path(path).name.split(".")[0]
    data = dict(data, id=gid)
    n = len(data.get("players") or [])
    if data.get("status") == "error" or "title" not in data:
        return gid, n, None, "cleaning_dropped"
    if (data.get("settings") or {}).get("optional_rules") and not allow_optional_rules:
        return gid, n, None, "optional_rules"
    if raw_dirs:
        from build_generic_start_positions import raw_optional_rules

        rules = raw_optional_rules(gid, raw_dirs)
        if rules is None:
            return gid, n, None, "no_raw_export"
        if rules and not allow_optional_rules:
            return gid, n, None, "optional_rules"
    replayed, reason = replay_human_game(TITLE, data)
    if replayed is None:
        return gid, n, None, (reason or "replay_failed").split(":")[0]
    if replayed.optional_rules and not allow_optional_rules:
        return gid, n, None, "optional_rules"
    return gid, replayed.num_players, list(replayed.actions), None


def _task(args) -> dict:
    """One game: its looted-broke dumps and the practice cuts before them (a pool task)."""
    path, source, raw_dirs, srs_before, include_unlooted, allow_optional_rules = args
    from behaviour_stats import analyse_item, broke_dumps
    from robust_games import load_item, read_json

    from rl18xx.agent.alphazero.generic_play import apply_actions, new_title_game, raw_game, stock_round_cuts

    t0 = time.time()
    out = {"path": path, "source": source}
    try:
        data = read_json(path)
        item = load_item(path, default_group="human" if source == "human" else None)
    except Exception as e:
        return {**out, "reason": f"load: {type(e).__name__}"}
    n = item.num_players
    out["num_players"] = n
    if not 2 <= n <= 6:
        return {**out, "reason": "player_count"}
    try:
        rec = analyse_item(item, keep_events=True)
    except Exception as e:
        return {**out, "reason": f"python_replay: {type(e).__name__}"}
    dumps = [d for d in broke_dumps(rec["events"]) if d["looted"] or include_unlooted]
    out["dumps_found"] = len(dumps)
    out["python_error"] = rec.get("error")
    if not dumps:
        return {**out, "reason": "no_dump", "secs": round(time.time() - t0, 2)}
    gid, n_rust, actions, reason = engine_log(path, data, source, raw_dirs, allow_optional_rules)
    out["id"] = gid
    if actions is None:
        return {**out, "reason": reason}
    if n_rust != n:
        return {**out, "reason": "player_count_mismatch"}
    wanted = {}
    for i, d in enumerate(dumps):
        for b, k in zip(srs_before, cut_rounds(d, srs_before)):
            wanted.setdefault(k, []).append([i, b])
    try:
        cuts = stock_round_cuts(TITLE, n, actions, last_round=max(wanted))
    except Exception as e:
        return {**out, "reason": f"rust_replay: {type(e).__name__}"}
    kept, dropped = {}, collections.Counter()
    for k in sorted(wanted):
        if k not in cuts:
            dropped["no_cut"] += 1
            continue
        try:
            game = raw_game(apply_actions(new_title_game(TITLE, n), actions[: cuts[k]]))
        except Exception:
            dropped["strict_replay_error"] += 1
            continue
        if game.finished or game.round.round_type != "Stock" or int(game.turn) != k:
            dropped["cut_not_at_stock_round"] += 1
            continue
        kept[k] = cuts[k]
    out["dropped_cuts"] = dict(dropped)
    if not kept:
        return {**out, "reason": "no_cut_replays"}
    record = {
        "id": gid,
        "title": TITLE,
        "num_players": n,
        "source": source,
        "actions": actions[: max(kept.values())],
        "cuts": {str(k): int(v) for k, v in sorted(kept.items())},
        "dumps": [
            {"sr": int(d["sr"]), "round": d.get("round"), "corp": d["corp"], "dumper": d["from"], "victim": d["to"],
             "looted": bool(d["looted"]), "victim_bankrupt": bool(d["victim_bankrupt"]), "phase": d.get("phase"),
             "at": d.get("site_id")}
            for d in dumps
        ],
        "cut_dumps": {str(k): wanted[k] for k in sorted(kept)},
    }
    if source == "caleb":
        record["names"] = (data.get("rl18xx") or {}).get("names")
    rules = (data.get("settings") or {}).get("optional_rules") or []
    if rules:
        record["played_under"] = list(rules)  # (--allow-optional-rules) the prefixes replay under the base rules
    return {**out, "record": record, "secs": round(time.time() - t0, 2)}


def _sources(args) -> list:
    from robust_games import iter_sources

    tasks = []
    for path, _, _ in iter_sources(args.source):
        tasks.append((path, "human"))
    if args.caleb:
        for path, _, _ in iter_sources([args.caleb]):
            tasks.append((path, "caleb"))
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def _write(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _key(record: dict):
    gid = str(record["id"])
    return (0, int(gid), "") if gid.isdigit() else (1, 0, gid)


def build(args) -> dict:
    tasks = [(path, source, args.raw_dir, tuple(args.srs_before), args.include_unlooted, args.allow_optional_rules)
             for path, source in _sources(args)]
    print(f"{len(tasks)} game files ({sum(t[1] == 'caleb' for t in tasks)} of Caleb's)", flush=True)
    t0 = time.time()
    reasons = collections.Counter()
    by_count = collections.defaultdict(list)
    stats = collections.defaultdict(collections.Counter)
    dropped = collections.Counter()
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(_task, tasks, chunksize=8)
    else:
        pool = None
        results = map(_task, tasks)
    for i, res in enumerate(results):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(tasks)} files ({time.time() - t0:.0f}s)", flush=True)
        n = res.get("num_players")
        src = res["source"]
        if res.get("dumps_found"):
            stats[(src, n)]["games_with_dump"] += 1
            stats[(src, n)]["dumps"] += res["dumps_found"]
        dropped.update(res.get("dropped_cuts") or {})
        if "record" not in res:
            reasons[(src, res.get("reason"))] += 1
            continue
        record = res["record"]
        by_count[n].append(record)
        stats[(src, n)]["games"] += 1
        stats[(src, n)]["positions"] += len(record["cuts"])
        stats[(src, n)]["kept_dumps"] += len(record["dumps"])
        for k in record["cuts"]:
            stats[(src, n)][f"sr{k}"] += 1
    if pool:
        pool.shutdown()
    print(f"done in {time.time() - t0:.0f}s")
    for (src, reason), c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {src}: {reason}: {c}")
    if dropped:
        print(f"  cuts dropped: {dict(dropped)}")
    summary = {}
    for n in range(2, 7):
        records = sorted(by_count.get(n, []), key=_key)
        path = practice_path(args.out_dir, n, args.suffix)
        if not args.dry_run:
            _write(path, records)
        human, caleb = stats[("human", n)], stats[("caleb", n)]
        srs = collections.Counter()
        for c in (human, caleb):
            srs.update({int(k[2:]): v for k, v in c.items() if k.startswith("sr")})
        summary[n] = {
            "file": str(path), "games": len(records), "positions": sum(len(r["cuts"]) for r in records),
            "human": {k: human[k] for k in ("games_with_dump", "dumps", "games", "kept_dumps", "positions")},
            "caleb": {k: caleb[k] for k in ("games_with_dump", "dumps", "games", "kept_dumps", "positions")},
            "positions_by_sr": dict(sorted(srs.items())),
        }
        print(f"{n}p: {summary[n]['positions']} positions from {len(records)} games "
              f"(human: {human['positions']} positions / {human['games']} games / {human['kept_dumps']} dumps of "
              f"{human['dumps']} found in {human['games_with_dump']} games; Caleb: {caleb['positions']} positions / "
              f"{caleb['games']} games / {caleb['kept_dumps']} dumps); by Stock Round {summary[n]['positions_by_sr']}"
              + ("" if args.dry_run else f" -> {path}"), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=None,
                        help=f"cleaned 1830 exports (directories or files; default {DEFAULT_SOURCES})")
    parser.add_argument("--raw-dir", action="append", default=None,
                        help=f"raw 18xx.games exports, for optional rules (default {DEFAULT_RAW_DIRS})")
    parser.add_argument("--caleb", default=DEFAULT_CALEB, help="Caleb's converted site games ('' to leave them out)")
    parser.add_argument("--srs-before", type=int, nargs="+", default=[1, 2],
                        help="cut this many Stock Rounds before each dump (default 1 2)")
    parser.add_argument("--include-unlooted", action="store_true", help="every broke dump, looted or not")
    parser.add_argument(
        "--allow-optional-rules", action="store_true",
        help="keep games played under optional rules where the prefix replays strictly under the base rules "
        "(default: leave them out, as the start files do)",
    )
    parser.add_argument("--out-dir", default="human_games")
    parser.add_argument("--suffix", default=SUFFIX)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--limit", type=int, default=None, help="only the first N files (tests)")
    parser.add_argument("--dry-run", action="store_true", help="count, write nothing")
    args = parser.parse_args(argv)
    args.source = args.source or DEFAULT_SOURCES
    args.raw_dir = args.raw_dir if args.raw_dir is not None else DEFAULT_RAW_DIRS
    if any(b < 1 for b in args.srs_before):
        raise SystemExit("--srs-before: 1 or more")
    return build(args)


if __name__ == "__main__":
    main()
