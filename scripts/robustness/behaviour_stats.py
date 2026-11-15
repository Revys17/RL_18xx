"""M2 of docs/robustness_test_plan.md: how often the five tactics behind the bots' losses to Caleb happen in a set
of games, compared with the human corpus and gen6's self-play (docs/robustness_measurement.md).

Per game (replayed in the Python reference engine, ``behaviour_events``), counted from the first move the recorded
players chose themselves (a policy-gradient game's human start prefix is skipped):

1. auction: each private's price, over face, contested (2+ bidders) or not, and whether its buyer won;
2. loot-and-dump: ``broke_dump`` (a president sells out of a floated, trainless company that can't afford the next
   train), ``broke_dump_p4`` (phase 4+), ``looted_broke_dump`` (the dumper first sold its train away for <= $10 or
   bought one for >= 1.5x list within 150 actions), ``broke_dump_victim_bankrupt``;
3. sell-outs: ``pres_sells_lost`` (a stock-round sale that hands the presidency over), ``handover_cash300`` (... of a
   floated company holding >= $300), ``takeover_bought`` (a presidency taken by buying);
4. doomed buys: ``buy_into_trainless_broke`` (a share of a floated, trainless, broke company someone else runs),
   ``bought_then_dumped_same_sr``, ``took_trainless_broke`` (its presidency taken by buying);
5. suitcasing: ``trainless_turn`` (a company ends an operating turn without a train), ``suitcase2`` / ``suitcase3``
   (a company trainless for 2+ / 3+ ORs in a row under one president);

plus ``bankrupt``, ``contrib`` / ``contrib$`` (forced train contributions), ``dump``, ``pres_sells``,
``shuffle_$1`` / ``overpay_own_train`` (train sales between one president's companies).

Each count is kept for the game ("all") and for the group of the player it is attributed to (the victim of a dump,
the seller of a sell-out, the buyer of a doomed buy, the president of a suitcase); groups come from the saved game
(``robust_games``: A/B sides of an eval, learner/league of a policy-gradient game, bot/human of a site game). Rates
are per game for "all" and per seat-game for a group.

    uv run python scripts/robustness/behaviour_stats.py run logs/eval/<collection> --out logs/robustness/<name>
    uv run python scripts/robustness/behaviour_stats.py compare logs/robustness/<name>/summary.json
    # the cached baselines (data/behaviour_baselines.json): human corpus and gen6 self-play
    uv run python scripts/robustness/behaviour_stats.py run human_games/1830_clean_20261017 --num-players 4 \
        --finished-only --label human --save-baseline human
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from behaviour_events import FACE, PHASE_ORDER, TRAIN_PRICE  # noqa: E402

BASELINES = HERE / "data" / "behaviour_baselines.json"
PRIVATES = ("CA", "MH", "DH", "CS")
LOOT_WINDOW = 150  # actions before a dump in which a loot counts (the analysis' window)
# The rows of the comparison table, by weakness (docs/bot_weaknesses_2026_11.md's numbering).
REPORT = [
    ("2 loot-and-dump", ["broke_dump", "broke_dump_p4", "looted_broke_dump", "broke_dump_victim_bankrupt"]),
    ("3 sell-outs", ["pres_sells_lost", "handover_cash300", "takeover_bought", "dump"]),
    ("4 doomed buys", ["buy_into_trainless_broke", "bought_then_dumped_same_sr", "took_trainless_broke"]),
    ("5 suitcasing", ["trainless_turn", "suitcase2", "suitcase3"]),
    ("money", ["bankrupt", "contrib", "contrib$", "shuffle_$1", "overpay_own_train"]),
]


def cheapest_train(name: Optional[str]) -> int:
    return TRAIN_PRICE.get(name or "", 10**6)


def _broke(corp: dict, depot_next: Optional[str]) -> bool:
    return bool(corp["floated"]) and not corp["trains"] and corp["cash"] < cheapest_train(depot_next)


def is_looted(ev: dict, dump: dict) -> bool:
    """Whether the dumper of ``dump`` (a ``pres_changes`` event) looted the company first: within the
    ``LOOT_WINDOW`` actions before, it sold one of the company's trains to another of its companies for <= $10, or
    had it buy one from another of its companies for >= 1.5x list."""
    for x in ev["train_sales"]:
        if x["site_id"] >= dump["site_id"] or x["site_id"] < dump["site_id"] - LOOT_WINDOW:
            continue
        if x["buyer_pres"] != x["seller_pres"] or x["buyer_pres"] != dump["from"]:
            continue
        lst = TRAIN_PRICE.get(x["train"], 0)
        if (x["seller"] == dump["corp"] and x["price"] <= 10) or (
                x["buyer"] == dump["corp"] and x["price"] >= 1.5 * lst):
            return True
    return False


def victim_went_bankrupt(ev: dict, dump: dict) -> bool:
    return any(b["player"] == dump["to"] and b["site_id"] >= dump["site_id"] for b in ev["bankruptcies"])


def broke_dumps(ev: dict, start_k: int = 0) -> list:
    """The game's ``broke_dump`` events (from action ``start_k`` on): each ``pres_changes`` event where a president
    sold out of a floated, trainless company short of the next train, with ``looted`` (:func:`is_looted`) and
    ``victim_bankrupt``. The practice-position builder reads these (``build_practice_positions.py``)."""
    out = []
    for p in ev["pres_changes"]:
        if start_k and p.get("site_id", 0) < start_k:
            continue
        if p["cause"] != "sold" or not _broke(p["corp_after"], p.get("depot_next")):
            continue
        out.append({**p, "looted": is_looted(ev, p), "victim_bankrupt": victim_went_bankrupt(ev, p)})
    return out


# --------------------------------------------------------------------------------------------- per game
def game_metrics(ev: dict, start_k: int = 0, group_of=lambda pid: None) -> Counter:
    """Counts of a game's events from action ``start_k`` on: ``"<metric>"`` for the game and
    ``"<metric>|<group>"`` for the group of the player each count is attributed to."""
    m = Counter()

    def add(name, pid=None, n=1):
        m[name] += n
        g = group_of(pid) if pid is not None else None
        if g is not None:
            m[f"{name}|{g}"] += n

    def keep(x):
        return x.get("site_id", 0) >= start_k if start_k else True

    sid_key = "site_id"
    for p in ev["pres_changes"]:
        if not keep(p):
            continue
        ca = p["corp_after"]
        if p["cause"] == "sold":
            add("dump", p["to"])
            if _broke(ca, p.get("depot_next")):
                add("broke_dump", p["to"])
                if PHASE_ORDER.get(p["phase"], 0) >= 4:
                    add("broke_dump_p4", p["to"])
                if is_looted(ev, p):
                    add("looted_broke_dump", p["to"])
                if victim_went_bankrupt(ev, p):
                    add("broke_dump_victim_bankrupt", p["to"])
            if ca["floated"] and ca["cash"] >= 300:
                add("handover_cash300", p["from"])
        elif p["cause"] == "bought":
            add("takeover_bought", p["to"])
            if _broke(ca, p.get("depot_next")):
                add("took_trainless_broke", p["to"])
    for s in ev["own_sales"]:
        if not keep(s) or s.get("round") != "Stock" or not s["was_pres"]:
            continue
        add("pres_sells", s["player"])
        if s["pres_after"] != s["player"]:
            add("pres_sells_lost", s["player"])
    for b in ev["buys"]:
        if not keep(b) or b.get("round") != "Stock":
            continue
        if b["pres"] is not None and b["pres"] != b["player"] and _broke(b["corp_before"], b.get("depot_next")):
            add("buy_into_trainless_broke", b["player"])
    for p in ev["pres_changes"]:
        if not keep(p) or p["cause"] != "sold" or p.get("round") != "Stock":
            continue
        if any(b["player"] == p["to"] and b["corp"] == p["corp"] and b.get("sr") == p.get("sr")
               and b[sid_key] < p[sid_key] for b in ev["buys"]):
            add("bought_then_dumped_same_sr", p["to"])
    runs = defaultdict(int)
    best = defaultdict(int)
    for x in ev["corp_turns"]:
        if not keep(x):
            continue
        if not x["trains_end"]:
            add("trainless_turn", x["pres"])
            runs[x["corp"]] += 1
            best[(x["corp"], x["pres"])] = max(best[(x["corp"], x["pres"])], runs[x["corp"]])
        else:
            runs[x["corp"]] = 0
            if not x["ran"] and x["trains_start"]:
                add("has_train_no_run", x["pres"])
    for (_, pres), n in best.items():
        if pres is None:
            continue
        if n >= 2:
            add("suitcase2", pres)
        if n >= 3:
            add("suitcase3", pres)
    for b in ev["bankruptcies"]:
        if keep(b):
            add("bankrupt", b["player"])
    for c in ev["contributions"]:
        if keep(c):
            add("contrib", c["player"])
            add("contrib$", c["player"], c["amount"])
    for x in ev["train_sales"]:
        if not keep(x) or x["buyer_pres"] is None or x["buyer_pres"] != x["seller_pres"]:
            continue
        if x["price"] <= 10:
            add("shuffle_$1", x["buyer_pres"])
        elif x["price"] >= 1.5 * TRAIN_PRICE.get(x["train"], 0):
            add("overpay_own_train", x["buyer_pres"])
    if any(keep(b) for b in ev["bankruptcies"]):
        m["game_with_bankruptcy"] += 1
    return m


def auction_rows(ev: dict, group_of=lambda pid: None) -> list:
    """One row per private sold in the game's auction: price, over face, contested, the buyer's group and place."""
    end = ev.get("game_end") or {}
    values = {int(k): v["value"] for k, v in (end.get("players") or {}).items()}
    ordered = sorted(values.values(), reverse=True)
    rows = []
    for co, a in ev["auction"].items():
        if co is None or co not in FACE:
            continue
        v = values.get(int(a["buyer"]))
        rows.append({
            "co": co, "price": a["price"], "over": a["price"] - FACE[co], "contested": bool(a["contested"]),
            "group": group_of(a["buyer"]),
            "won": (1.0 if v is not None and ordered and v >= ordered[0] else 0.0) if end.get("finished") else None,
            "place": (ordered.index(v) if v is not None else None) if end.get("finished") else None,
        })
    return rows


def analyse_item(item, keep_events: bool = False) -> dict:
    """Replay one game and count its behaviours (a pool task)."""
    from behaviour_events import extract, replay_rows, start_k

    t0 = time.time()
    tries = [item.auction_unlock] if item.auction_unlock is not None else [False, True]
    rows = err = game = None
    for unlock in tries:
        rows, err, game = replay_rows(item.actions, item.num_players, item.optional_rules, bool(unlock),
                                      site_ids=item.site_ids)
        if err is None:
            break
    groups = {int(k): v for k, v in (item.groups or {}).items()}

    def group_of(pid):
        return groups.get(int(pid)) if pid is not None else None

    ev = extract(rows, lambda pid: group_of(pid) == "bot")
    sk = start_k(rows, item.start)
    rec = {
        "file": item.path, "num_players": item.num_players, "error": err, "finished": bool(game.finished),
        "start": item.start, "start_k": sk, "groups": {str(k): v for k, v in groups.items()},
        "n_actions": len(item.actions), "secs": round(time.time() - t0, 2),
    }
    if sk is not None:
        start_id = sk
        if item.site_ids is not None and sk < len(item.site_ids):
            start_id = item.site_ids[sk] or 0
        rec["counts"] = dict(game_metrics(ev, start_id, group_of))
        rec["auction"] = auction_rows(ev, group_of) if sk == 0 else []
    if keep_events:
        rec["events"] = ev
    return rec


def _task(args) -> Optional[dict]:
    path, index_record, settings, filters, keep_events = args
    from robust_games import load_item

    try:
        item = load_item(path, index_record, settings, default_group=filters.get("default_group"))
    except Exception as e:
        return {"file": path, "error": f"load: {type(e).__name__}: {e}"}
    if filters.get("num_players") and item.num_players != filters["num_players"]:
        return None
    if filters.get("finished_only") and item.finished_hint is False:
        return None
    try:
        rec = analyse_item(item, keep_events)
    except Exception as e:
        return {"file": path, "error": f"replay: {type(e).__name__}: {e}"}
    if filters.get("finished_only") and not rec["finished"]:
        return None
    return rec


# ------------------------------------------------------------------------------------------- aggregation
def summarize(records: Iterable[dict], label: str = "") -> dict:
    """Rates over the games' records: per game ("all") and per seat-game of each group; the auction table."""
    games = 0
    seat_games = Counter()
    totals = Counter()
    auction = []
    errors = 0
    skipped_start = 0
    for r in records:
        if r is None:
            continue
        if r.get("error"):
            errors += 1
            continue
        if "counts" not in r:
            skipped_start += 1
            continue
        games += 1
        for g in (r.get("groups") or {}).values():
            seat_games[g] += 1
        totals.update(r["counts"])
        auction.extend(r.get("auction") or [])
    rates = {}
    for key, n in totals.items():
        name, _, group = key.partition("|")
        if group:
            rates.setdefault(name, {})[group] = n / max(1, seat_games[group])
        else:
            rates.setdefault(name, {})["all"] = n / max(1, games)
    return {
        "label": label, "games": games, "errors": errors, "skipped_start": skipped_start,
        "seat_games": dict(seat_games), "rates": rates, "totals": dict(totals),
        "auction": auction_table(auction),
    }


def auction_table(rows: list) -> dict:
    out = {}
    for co in PRIVATES:
        r = [x for x in rows if x["co"] == co]
        if not r:
            continue
        prices = [x["price"] for x in r]
        contested = [x for x in r if x["contested"]]
        uncontested = [x for x in r if not x["contested"]]
        won = [x["won"] for x in r if x.get("won") is not None]
        row = {
            "n": len(r), "mean": statistics.mean(prices), "median": statistics.median(prices),
            "contested": len(contested) / len(r),
            "mean_contested": statistics.mean(x["price"] for x in contested) if contested else None,
            "mean_uncontested": statistics.mean(x["price"] for x in uncontested) if uncontested else None,
            "over_face": statistics.mean(x["over"] for x in r),
            "buyer_wins": statistics.mean(won) if won else None,
        }
        by_group = defaultdict(list)
        for x in r:
            if x.get("group"):
                by_group[x["group"]].append(x)
        if by_group:
            row["by_group"] = {
                g: {"n": len(xs), "mean": statistics.mean(x["price"] for x in xs),
                    "contested": sum(x["contested"] for x in xs) / len(xs),
                    "buyer_wins": (statistics.mean(x["won"] for x in xs if x.get("won") is not None)
                                   if any(x.get("won") is not None for x in xs) else None)}
                for g, xs in sorted(by_group.items())
            }
        out[co] = row
    return out


def _fmt(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "-"
    return f"{x:.{digits}f}"


def format_report(summaries: list, baselines: Optional[dict] = None) -> str:
    """A text table: each summary's per-game rates (and its groups' per seat-game rates) next to the baselines'."""
    cols = list(summaries)
    base = [dict(v, label=k) for k, v in (baselines or {}).items() if isinstance(v, dict) and "rates" in v]
    lines = []
    head = ["metric"] + [f"{s['label'] or 'games'} (n={s['games']})" for s in cols] + [
        f"{b['label']} (n={b['games']})" for b in base]
    lines.append(" | ".join(head))
    for title, names in REPORT:
        lines.append(f"-- {title}")
        for name in names:
            cells = [name]
            for s in cols + base:
                r = s["rates"].get(name, {})
                cell = _fmt(r.get("all", 0.0))
                groups = {g: v for g, v in r.items() if g != "all"}
                if groups and s in cols:
                    cell += " [" + ", ".join(f"{g} {_fmt(v)}" for g, v in sorted(groups.items())) + "]"
                cells.append(cell)
            lines.append(" | ".join(cells))
    lines.append("-- 1 auction (price; contested share; buyer wins)")
    for co in PRIVATES:
        cells = [co]
        for s in cols + base:
            a = s.get("auction", {}).get(co)
            if not a:
                cells.append("-")
                continue
            cell = f"${a['mean']:.0f} (contested {a['contested']:.0%}; wins {_fmt(a['buyer_wins'])})"
            if s in cols and a.get("by_group"):
                cell += " [" + ", ".join(f"{g} ${v['mean']:.0f} c{v['contested']:.0%}"
                                         for g, v in a["by_group"].items()) + "]"
            cells.append(cell)
        lines.append(" | ".join(cells))
    for s in cols + base:
        rate = s["rates"].get("game_with_bankruptcy", {}).get("all", 0.0)
        lines.append(f"{s['label'] or 'games'}: games ending with a bankruptcy {rate:.3f}; seat-games by group "
                     f"{s.get('seat_games')}; replay errors {s.get('errors', 0)}")
    return "\n".join(lines)


def load_baselines(path=BASELINES) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def save_baseline(name: str, summary: dict, source: str, path=BASELINES) -> None:
    data = load_baselines(path)
    keep = {k: summary[k] for k in ("label", "games", "errors", "skipped_start", "seat_games", "rates", "auction")}
    keep["source"] = source
    keep["computed"] = time.strftime("%Y-%m-%d")
    data[name] = keep
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------------------------- CLI
def _index_players(rec: dict) -> Optional[int]:
    """A collection index record's player count (``num_players``, or its ``sides`` / ``seats`` list), else None."""
    if rec.get("num_players"):
        return int(rec["num_players"])
    for key in ("sides", "seats"):
        if isinstance(rec.get(key), list) and rec[key]:
            return len(rec[key])
    return None


def run(args) -> dict:
    from robust_games import iter_sources

    sources = list(iter_sources(args.inputs, min_update=args.min_update, max_update=args.max_update))
    if args.num_players:  # a collection index that names the seats: other counts never sampled or read
        sources = [s for s in sources if _index_players(s[1]) in (None, args.num_players)]
    if args.sample and len(sources) > args.sample:
        sources = random.Random(args.seed).sample(sources, args.sample)
    if args.limit:
        sources = sources[: args.limit]
    filters = {"num_players": args.num_players, "finished_only": args.finished_only,
               "default_group": args.default_group}
    tasks = [(path, rec, settings, filters, args.keep_events) for path, rec, settings in sources]
    print(f"{len(tasks)} game files", flush=True)
    records = []
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    sink = open(out / "games.jsonl", "w") if out else None
    t0 = time.time()
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(_task, tasks, chunksize=8)
    else:
        pool = None
        results = map(_task, tasks)
    for i, rec in enumerate(results):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(tasks)} files, {len(records)} games kept ({time.time() - t0:.0f}s)", flush=True)
        if rec is None:
            continue
        records.append({k: v for k, v in rec.items() if k != "events"})
        if sink:
            sink.write(json.dumps(rec, default=str) + "\n")
    if pool:
        pool.shutdown()
    if sink:
        sink.close()
    summary = summarize(records, args.label or (Path(args.inputs[0]).name if args.inputs else ""))
    summary["seconds"] = round(time.time() - t0, 1)
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.save_baseline:
        save_baseline(args.save_baseline, summary, " ".join(args.inputs) + (
            f" (num_players={args.num_players}, finished_only={args.finished_only}, min_update={args.min_update})"),
            args.baselines)
    print(format_report([summary], load_baselines(args.baselines)))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="count the behaviours in a set of games")
    p.add_argument("inputs", nargs="+", help="collections, directories of game files, or game files")
    p.add_argument("--label", default="")
    p.add_argument("--num-players", type=int, default=None, help="keep games of this player count")
    p.add_argument("--finished-only", action="store_true", help="keep finished games only")
    p.add_argument("--min-update", type=int, default=None, help="policy-gradient collections: from this update on")
    p.add_argument("--max-update", type=int, default=None, help="policy-gradient collections: up to this update")
    p.add_argument("--sample", type=int, default=None, help="a random sample of this many files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    p.add_argument("--default-group", default=None,
                   help="group label for games that name none (e.g. human for the human corpus)")
    p.add_argument("--keep-events", action="store_true", help="write each game's events into games.jsonl")
    p.add_argument("--out", default=None, help="directory for games.jsonl (per game) and summary.json")
    p.add_argument("--baselines", default=str(BASELINES))
    p.add_argument("--save-baseline", default=None, help="store the summary under this name in --baselines")
    c = sub.add_parser("compare", help="print summaries next to the baselines")
    c.add_argument("summaries", nargs="+")
    c.add_argument("--baselines", default=str(BASELINES))
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        summaries = [json.loads(Path(s).read_text()) for s in args.summaries]
        print(format_report(summaries, load_baselines(args.baselines)))


if __name__ == "__main__":
    main()
