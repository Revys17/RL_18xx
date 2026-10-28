#!/usr/bin/env python3
"""The nightly saved-data integrity check (docs/long_run_recipe.md M6, and "4. Data": the shard list's checksums).

    integrity_check.py --data <saved games dir> [--games 50] [--hours 24] [--log longrun/integrity.jsonl]
                       [--checksums longrun/shard_checksums.txt] [--verify-old 20] [--allow-missing]

1. ``--games`` random games of the shards written in the last ``--hours`` (shard mtime) are rebuilt into rows twice
   (``search_data.rebuild_rows(check=True)``: the action log replays, and every row's position has the mover and the
   legal moves the generator saw); the two rebuilds' encodings and targets must agree bit for bit (the encoder is
   deterministic, so a difference is a bug that would corrupt the value's data).
2. ``--checksums``: the sha256 of every shard not yet listed is appended (``<sha256>  <shard name>  <bytes>``), and
   ``--verify-old`` listed shards, drawn at random, are hashed again: a changed or vanished shard is a mismatch (a
   shard moved off the disk by hand: move its line too, or pass ``--allow-missing``).

One JSON line per run goes to ``--log``. Any mismatch: exit 1 and a CRIT ``integrity`` alert through the watchdog's
alert path (STATE_DIR's alerts.log, notify-send, ALERT_COMMAND; the daily job's instance). Run with the deployment's
venv. The recipe's action on a mismatch: stop promoting values from the affected shards; investigate before the
value trainer's next cycle.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))


def same_value(a, b) -> bool:
    """Bit-for-bit equality of two row fields (arrays by dtype, shape and bytes)."""
    import numpy as np

    if hasattr(a, "numpy") and not isinstance(a, np.ndarray):
        a = a.numpy()
    if hasattr(b, "numpy") and not isinstance(b, np.ndarray):
        b = b.numpy()
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def compare_rows(first, second) -> list:
    """The fields (``row k: name``) where two rebuilds of a game differ."""
    import dataclasses

    if len(first) != len(second):
        return [f"{len(first)} rows vs {len(second)}"]
    diffs = []
    for k, (a, b) in enumerate(zip(first, second)):
        for f in dataclasses.fields(a):
            if not same_value(getattr(a, f.name), getattr(b, f.name)):
                diffs.append(f"row {k}: {f.name}")
    return diffs


def check_game(game) -> tuple:
    """(rows, problem or None): the game rebuilt twice with ``check=True``, compared."""
    from rl18xx.agent.alphazero.search_data import rebuild_rows

    try:
        first = rebuild_rows(game, check=True)
        second = rebuild_rows(game, check=True)
    except Exception as e:  # noqa: BLE001 -- a replay that fails is the finding
        return 0, f"rebuild failed: {e!r}"
    if not first:
        return 0, None
    if any(r.game_state is None or r.node_features is None for r in first):
        return len(first), "a rebuilt row has no encoding"
    diffs = compare_rows(first, second)
    return len(first), (f"two rebuilds differ: {', '.join(diffs[:5])}" if diffs else None)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums(path) -> dict:
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                out[parts[1]] = parts[0]
    except OSError:
        pass
    return out


def checksums(data_dir, path, verify_old, rng, allow_missing=False) -> dict:
    """Append the new shards' sha256 to ``path`` and re-hash ``verify_old`` listed ones."""
    from rl18xx.agent.alphazero.search_data import shard_paths

    listed = read_checksums(path)
    shards = {p.name: p for p in shard_paths(data_dir)}
    old = [name for name in listed]
    verify = rng.sample(old, min(verify_old, len(old)))
    problems, missing = [], []
    for name in verify:
        if name not in shards:
            missing.append(name)
            continue
        if sha256(shards[name]) != listed[name]:
            problems.append(f"shard {name} changed since it was listed (sha256)")
    if missing and not allow_missing:
        problems += [f"shard {name} is listed but gone" for name in missing]
    new = [name for name in sorted(shards) if name not in listed]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for name in new:
            f.write(f"{sha256(shards[name])}  {name}  {shards[name].stat().st_size}\n")
    return {"listed": len(new), "verified": len(verify) - len(missing), "missing": missing, "problems": problems}


def run(args) -> dict:
    from rl18xx.agent.alphazero.search_data import read_shard, shard_paths

    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    until = time.time()
    shards = [p for p in shard_paths(args.data) if until - args.hours * 3600 <= p.stat().st_mtime < until]
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data": str(args.data),
        "shards_in_window": len(shards),
        "games": 0,
        "rows": 0,
        "mismatches": [],
    }
    games, total = [], 0  # a uniform sample (reservoir) of detached games: a day's shards don't fit in memory
    for path in shards:
        try:
            shard = read_shard(path)
        except Exception as e:  # noqa: BLE001 -- an unreadable complete shard is a mismatch
            record["mismatches"].append({"shard": path.name, "error": f"unreadable: {e!r}"})
            continue
        for game in shard:
            total += 1
            if len(games) < args.games:
                games.append((path.name, game.detached()))
            else:
                j = rng.randrange(total)
                if j < args.games:
                    games[j] = (path.name, game.detached())
    record["games_in_window"] = total
    for name, game in games:
        rows, problem = check_game(game)
        record["games"] += 1
        record["rows"] += rows
        if problem:
            record["mismatches"].append({"shard": name, "uid": game.uid, "error": problem})
    if args.checksums:
        summary = checksums(args.data, args.checksums, args.verify_old, rng, args.allow_missing)
        record["checksums_listed"] = summary["listed"]
        record["checksums_verified"] = summary["verified"]
        record["checksums_missing"] = summary["missing"]
        record["mismatches"] += [{"error": p} for p in summary["problems"]]
    record["ok"] = not record["mismatches"]
    return record


def alert(level, key, message):
    import watchdog

    state_dir = Path(os.environ.get("STATE_DIR") or "longrun/daily")
    if not state_dir.is_absolute():
        state_dir = Path(os.environ.get("WORKDIR", ".")) / state_dir
    watchdog.send_alert(state_dir, os.environ.get("LONGRUN_INSTANCE", "daily"), level, key, message,
                        os.environ.get("ALERT_COMMAND", "").strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="the generator's --save-data directory")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--hours", type=float, default=24.0, help="games of shards written in the last N hours")
    parser.add_argument("--log", type=Path, default=Path("longrun/integrity.jsonl"))
    parser.add_argument("--checksums", type=Path, default=None, help="the shard checksum list to extend and verify")
    parser.add_argument("--verify-old", type=int, default=20, help="listed shards re-hashed per run")
    parser.add_argument("--allow-missing", action="store_true", help="a listed shard that is gone is no mismatch")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-alert", action="store_true")
    args = parser.parse_args()
    try:
        record = run(args)
    except Exception as e:  # noqa: BLE001 -- record that the check itself failed
        record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "data": str(args.data),
                  "error": repr(e), "mismatches": [], "ok": False}
        traceback.print_exc()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"integrity: {record.get('games', 0)} games, {record.get('rows', 0)} rows rebuilt twice, "
          f"{record.get('checksums_listed', 0)} shards listed, {record.get('checksums_verified', 0)} re-hashed; "
          f"{len(record['mismatches'])} mismatch(es)" + (f"; error {record['error']}" if record.get("error") else ""))
    for m in record["mismatches"][:20]:
        print(f"  MISMATCH {m}")
    if record["mismatches"]:
        if not args.no_alert:
            alert("CRIT", "integrity", f"{len(record['mismatches'])} saved-data mismatch(es): "
                                       f"{record['mismatches'][0]} -- stop promoting values from these shards "
                                       f"(docs/long_run_recipe.md M6)")
        return 1
    if record.get("error"):
        if not args.no_alert:
            alert("WARN", "integrity", f"the integrity check failed to run: {record['error']}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
