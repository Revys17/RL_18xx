#!/usr/bin/env python3
"""Style statistics of the long run's saved games, and the drift guard (docs/long_run_recipe.md P6, "3. Opponents").

    style_stats.py --data <saved games dir> [--out longrun/style_stats.jsonl] [--hours 24 | --since T --until T]
                   [--max-games 5000] [--baseline first|<YYYY-MM-DD>|<file>]

Reads the ``--save-data`` shards written in the window (by shard mtime: a shard is renamed into place when its last
game is in it; default the last 24 h), samples up to ``--max-games`` games (0 = all; ~16 ms a game) and replays each
in the Rust engine (the action log alone doesn't say who presides over a corporation or which operating round it
is). Per game:

- pars after the start (the start's own auction and B&O par were not the system's choices): seat, corporation, price;
- dividend decisions (a dividend after a run with revenue: payout or withhold), by the president's seat;
- bankruptcies (the bankrupt corporation's president's seat);
- the first purchase of each train type (2, 3, 4, 5, 6, D): the operating round it happened in (the count of
  operating rounds so far, 1 = the first OR) and its action index in the log;
- decisions, actions, termination (``max_length`` -- or ``max_decisions`` -- is the length cap).

Train timing and game length are over games played from the first stock round (a mid-game start's log begins with a
human game's first rounds); the length-cap share is over all.

Two scopes. ``main``: the main system's seats (``seat_members``; a game without it is all main) for the per-seat
statistics (pars, dividends, bankruptcies per 4 main seats), and the games where all four seats are the main system
for the per-game ones (train timing, length, the length cap). ``all``: every seat and game.

Appends one JSON line per run to ``--out``. ``--baseline`` compares with an earlier line (``first``: the file's first
line; a date: that day's last line; or a JSON / JSONL file's last record) and sets the drift guard's flags: any
corporation's share of the main system's pars above 0.5 (at least ``--min-pars`` pars), or its withhold rate doubled
against the baseline (at least ``--min-dividends`` decisions on both sides). A flag means: raise the league share to
50% for the next week. Run with the deployment's venv (engine_rs).
"""

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRAIN_TYPES = ("2", "3", "4", "5", "6", "D")
LENGTH_CAP = ("max_length", "max_decisions")
MAIN = "main"


# ----------------------------------------------------------------------------- one game


def _seat_of_player(entity):
    try:
        return int(entity) - 1
    except (TypeError, ValueError):
        return None


def _president(rs, corporation):
    """The seat of the corporation's president now (None: no player owns it)."""
    corp = rs.corporation_by_id(str(corporation))
    owner = getattr(corp, "owner_id_str", "") if corp is not None else ""
    return int(owner.split(":")[1]) - 1 if owner.startswith("player:") else None


def _par_price(action):
    price = action.get("share_price", action.get("price"))
    try:
        return int(str(price).split(",")[0])
    except (TypeError, ValueError):
        return None


def _revenue(action):
    total = sum(float(r.get("revenue") or 0) for r in action.get("routes") or [])
    return total + float(action.get("extra_revenue") or 0)


def game_summary(game) -> dict:
    """What the statistics need of one saved game (``search_data.SavedGame``), from a replay in the Rust engine."""
    from engine_rs import BaseGame

    meta = game.meta
    n = int(meta["num_players"])
    members = list(meta.get("seat_members") or [MAIN] * n)
    rs = BaseGame({i + 1: f"Player {i + 1}" for i in range(n)})
    rs.set_auction_unlock(bool(meta.get("auction_unlock")))
    start = int(meta.get("start_actions") or 0)
    pars, dividends, bankrupt, first = [], [], [], {}
    operating_rounds, revenue = set(), {}
    actions = game.actions
    for i, action in enumerate(actions):
        kind = action.get("type")
        rnd = rs.round
        if rnd.round_type == "Operating":
            operating_rounds.add((rs.turn, rnd.round_num))
        if kind == "buy_train":
            train = str(action.get("train", "")).split("-")[0]
            if train and train not in first:
                first[train] = [len(operating_rounds), i, _president(rs, action.get("entity"))]
        elif i >= start:
            if kind == "par":
                pars.append([_seat_of_player(action.get("entity")), action.get("corporation"), _par_price(action)])
            elif kind == "run_routes":
                revenue[action.get("entity")] = _revenue(action)
            elif kind == "dividend":
                if revenue.pop(action.get("entity"), 0.0) > 0:  # a choice: payout was possible
                    dividends.append([_president(rs, action.get("entity")), action.get("kind")])
            elif kind == "bankrupt":
                seat = (
                    _seat_of_player(action.get("entity"))
                    if action.get("entity_type") == "player"
                    else _president(rs, action.get("entity"))
                )
                bankrupt.append(seat)
        rs.process_action(action)
    return {
        "uid": game.uid,
        "start_kind": meta.get("start_kind"),
        "num_players": n,
        "members": members,
        "termination": meta.get("termination"),
        "decisions": int(meta.get("decisions") or 0),
        "actions": len(actions),
        "value_version": meta.get("value_version"),
        "pars": pars,
        "dividends": dividends,
        "bankrupt": bankrupt,
        "first_train": first,
        "operating_rounds": len(operating_rounds),
    }


# ----------------------------------------------------------------------------- aggregation


def _dist(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
    }


def _shares(counter):
    total = sum(counter.values())
    return {str(k): v / total for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))} if total else {}


def aggregate(summaries, scope: str) -> dict:
    """The statistics of one scope ("main" or "all"; module docstring)."""
    def counted(g, seat):
        if scope != MAIN:
            return True
        return seat is not None and 0 <= seat < len(g["members"]) and g["members"][seat] == MAIN

    games = [g for g in summaries if all(m == MAIN for m in g["members"])] if scope == MAIN else list(summaries)
    whole = [g for g in games if g.get("start_kind") != "midgame"]  # games played from the first stock round
    seats = sum(sum(1 for s in range(g["num_players"]) if counted(g, s)) for g in summaries)
    pars = [(corp, price) for g in summaries for seat, corp, price in g["pars"] if counted(g, seat)]
    divs = [kind for g in summaries for seat, kind in g["dividends"] if counted(g, seat)]
    broke = sum(1 for g in summaries for seat in g["bankrupt"] if counted(g, seat))
    par_share = _shares(Counter(corp for corp, _ in pars))
    top = max(par_share.items(), key=lambda kv: kv[1]) if par_share else (None, None)
    four_seats = seats / 4 if seats else 0
    out = {
        "seats": seats,
        "games": len(games),
        "pars": len(pars),
        "par_share": par_share,
        "par_share_max": {"corporation": top[0], "share": top[1]},
        "par_price": _shares(Counter(price for _, price in pars if price is not None)),
        "par_price_mean": statistics.fmean([p for _, p in pars if p is not None]) if pars else None,
        "dividend_decisions": len(divs),
        "withhold_rate": sum(1 for k in divs if k == "withhold") / len(divs) if divs else None,
        "bankruptcies": broke,
        "bankruptcies_per_game": broke / four_seats if four_seats else None,
        "train_first": {},
        "whole_games": len(whole),
        "decisions": _dist([g["decisions"] for g in whole]),
        "actions": _dist([g["actions"] for g in whole]),
        "operating_rounds": _dist([g["operating_rounds"] for g in whole]),
        "length_cap_share": sum(1 for g in games if g["termination"] in LENGTH_CAP) / len(games) if games else None,
        "terminations": dict(Counter(str(g["termination"]) for g in games)),
    }
    for train in TRAIN_TYPES:
        bought = [g["first_train"][train] for g in whole if train in g["first_train"]]
        out["train_first"][train] = {
            "bought_share": len(bought) / len(whole) if whole else None,
            "or": _dist([b[0] for b in bought]),
            "action": _dist([b[1] for b in bought]),
        }
    return out


# ----------------------------------------------------------------------------- baseline and drift guard


def read_jsonl(path):
    records = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return records


def load_baseline(spec, out_path):
    """The record a ``--baseline`` names (module docstring), or None."""
    if not spec:
        return None
    if spec == "first":
        records = read_jsonl(out_path)
        return records[0] if records else None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec):
        same_day = [r for r in read_jsonl(out_path) if str(r.get("window", {}).get("until", "")).startswith(spec)]
        return same_day[-1] if same_day else None
    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"--baseline {spec}: no such file (or 'first', or a YYYY-MM-DD in {out_path})")
    text = path.read_text().strip()
    try:
        return json.loads(text)
    except ValueError:
        records = read_jsonl(path)
        return records[-1] if records else None


def _scalar(stats, path):
    value = stats
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, (int, float)) else None


COMPARED = {
    "withhold_rate": ("withhold_rate",),
    "par_share_max": ("par_share_max", "share"),
    "par_price_mean": ("par_price_mean",),
    "bankruptcies_per_game": ("bankruptcies_per_game",),
    "decisions_mean": ("decisions", "mean"),
    "length_cap_share": ("length_cap_share",),
    **{f"first_{t}_or_mean": ("train_first", t, "or", "mean") for t in ("3", "D")},
}


def drift(main, base_main=None, min_pars=20, min_dividends=50, par_max=0.5, withhold_factor=2.0, floor=0.01):
    """The drift guard's flags (module docstring): {"par_share": {...}, "withhold": {...}}; ``flag`` None = n/a."""
    top = main.get("par_share_max") or {}
    par = {"corporation": top.get("corporation"), "share": top.get("share"), "pars": main.get("pars", 0),
           "threshold": par_max, "flag": None}
    if main.get("pars", 0) >= min_pars and top.get("share") is not None:
        par["flag"] = top["share"] > par_max
    withhold = {"rate": main.get("withhold_rate"), "baseline": None, "factor": withhold_factor, "flag": None}
    if base_main is not None:
        withhold["baseline"] = base_main.get("withhold_rate")
        if (
            main.get("dividend_decisions", 0) >= min_dividends
            and base_main.get("dividend_decisions", 0) >= min_dividends
            and withhold["rate"] is not None
            and withhold["baseline"] is not None
        ):
            withhold["flag"] = withhold["rate"] >= withhold_factor * max(withhold["baseline"], floor)
    return {"par_share": par, "withhold": withhold}


def compare(record, baseline) -> dict:
    """Deltas of the headline statistics against a baseline record (main scope)."""
    out = {}
    for name, path in COMPARED.items():
        now, then = _scalar(record["main"], path), _scalar(baseline.get("main", {}), path)
        out[name] = {"now": now, "baseline": then, "change": None if now is None or then is None else now - then}
    return out


# ----------------------------------------------------------------------------- shards


def window_shards(data_dir, since, until):
    from rl18xx.agent.alphazero.search_data import shard_paths

    out = []
    for path in shard_paths(data_dir):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since <= mtime < until:
            out.append(path)
    return out


def collect(data_dir, since, until, max_games=5000, seed=0):
    """(summaries, shards read, games in the window, unreadable shards). The shards are streamed into a uniform
    sample (reservoir) of detached games: a day's games don't fit in memory (~90 MB a shard of 256 games, ~400
    shards a day), and a game still tied to its shard keeps all of it."""
    from rl18xx.agent.alphazero.search_data import read_shard

    shards = window_shards(data_dir, since, until)
    rng = random.Random(seed)
    games, bad, total = [], [], 0
    for path in shards:
        try:
            shard = read_shard(path)
        except Exception as e:  # noqa: BLE001 -- one bad shard must not hide the rest (integrity_check flags it)
            bad.append(f"{path.name}: {e!r}")
            continue
        for game in shard:
            total += 1
            if not max_games or len(games) < max_games:
                games.append(game.detached())
            else:
                j = rng.randrange(total)
                if j < max_games:
                    games[j] = game.detached()
    summaries, failed = [], 0
    for game in games:
        try:
            summaries.append(game_summary(game))
        except Exception as e:  # noqa: BLE001
            failed += 1
            bad.append(f"game {game.uid}: {e!r}")
    return summaries, len(shards), total, bad


def run(args) -> dict:
    until = datetime.fromisoformat(args.until).timestamp() if args.until else time.time()
    since = datetime.fromisoformat(args.since).timestamp() if args.since else until - args.hours * 3600
    summaries, shards, total, bad = collect(args.data, since, until, args.max_games, args.seed)
    versions = [s["value_version"] for s in summaries if s["value_version"] is not None]
    record = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window": {
            "since": datetime.fromtimestamp(since).astimezone().isoformat(timespec="seconds"),
            "until": datetime.fromtimestamp(until).astimezone().isoformat(timespec="seconds"),
        },
        "data": str(args.data),
        "shards": shards,
        "games_in_window": total,
        "games": len(summaries),
        "league_games": sum(1 for s in summaries if any(m != MAIN for m in s["members"])),
        "value_versions": [min(versions), max(versions)] if versions else None,
        "errors": bad[:20],
        "main": aggregate(summaries, MAIN),
        "all": aggregate(summaries, "all"),
    }
    baseline = load_baseline(args.baseline, args.out)  # (read before this run's line is appended)
    record["baseline"] = {"time": baseline.get("time"), "window": baseline.get("window")} if baseline else None
    record["compare"] = compare(record, baseline) if baseline else None
    record["drift"] = drift(record["main"], baseline.get("main") if baseline else None, args.min_pars,
                            args.min_dividends)
    record["flags"] = [k for k, v in record["drift"].items() if v["flag"]]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    return record


def describe(record) -> str:
    m = record["main"]
    top = m["par_share_max"]
    lines = [
        f"{record['games']} of {record['games_in_window']} games from {record['shards']} shards "
        f"({record['window']['since']} .. {record['window']['until']}); {record['league_games']} with league seats",
        f"main: {m['pars']} pars, top {top['corporation']} {top['share'] or 0:.2f}, mean par "
        f"{m['par_price_mean'] or 0:.0f}; withhold {m['withhold_rate'] if m['withhold_rate'] is not None else 'n/a'} "
        f"of {m['dividend_decisions']}; bankruptcies/game {m['bankruptcies_per_game']}; "
        f"decisions {m['decisions'].get('mean')}; length cap {m['length_cap_share']}",
        "first trains (OR, main games): "
        + ", ".join(
            f"{t} {v['or'].get('mean', float('nan')):.1f}" for t, v in m["train_first"].items() if v["or"]["n"]
        ),
        f"drift flags: {', '.join(record['flags']) or 'none'}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="the generator's --save-data directory")
    parser.add_argument("--out", type=Path, default=Path("longrun/style_stats.jsonl"))
    parser.add_argument("--hours", type=float, default=24.0, help="the window: the last N hours (default 24)")
    parser.add_argument("--since", default=None, help="window start (ISO time; instead of --hours)")
    parser.add_argument("--until", default=None, help="window end (ISO time; default now)")
    parser.add_argument("--max-games", type=int, default=5000, help="games replayed, sampled (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", default=None, help="first | YYYY-MM-DD | a JSON / JSONL file")
    parser.add_argument("--min-pars", type=int, default=20, help="pars needed before the par-share flag applies")
    parser.add_argument("--min-dividends", type=int, default=50, help="decisions needed for the withhold flag")
    args = parser.parse_args()
    record = run(args)
    print(describe(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
