"""Native router vs Python AutoRouter, over many operating-round states.

At every run_routes of a game, compares the best total revenue of the Rust
router (``BaseGame.calculate_routes`` → ``router::calculate_corp_routes``)
with the Python AutoRouter's (``rl18xx/game/engine/autorouter.py``), and times
the Rust search. Two sources of states:

  * ``random`` — seeded random native games (2–6 players by seed, both auction
    rules); the logged run_routes total is what the native decode paid, and the
    log is replayed in the Python engine for the AutoRouter.
  * ``human`` — cleaned human games (``human_games/1830_clean*/``), replayed in
    both engines; also reports how the human's own routes compare.

The AutoRouter keeps only each train's top ``route_limit`` (10,000) routes and
gives up its combination search after 10 s, so the oracle here is its own
candidates searched exhaustively (``exact``; branch-and-bound over the same
bitfields). Where Rust beats even that, rerun the state with a higher
``route_limit`` (``AutoRouter.compute(corp, route_limit=...)``).

Usage::

    uv run python tests/router_parity_sweep.py random 100 400 [--workers 16] [--json out.jsonl]
    uv run python tests/router_parity_sweep.py human human_games/1830_clean_2026_10/*.json [--workers 16]
    ... --no-python   # Rust timings only
"""

import argparse
import copy
import json
import logging
import multiprocessing as mp
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

MAX_MOVES = 3000


def exact_combo_revenue(router, sorted_routes):
    """The best total over the AutoRouter's own per-train candidates and
    hexside bitfields — what ``js_evaluate_combos`` finds by brute force,
    without its timeout."""
    lists = []
    for routes in sorted_routes:
        items = []
        for r in routes:
            bits = 0
            for i, word in enumerate(r.bitfield):
                bits |= word << (32 * i)
            items.append((r.revenue(), bits))
        items.sort(key=lambda x: -x[0])
        lists.append(items)
    rest = [0] * (len(lists) + 1)
    for i in range(len(lists) - 1, -1, -1):
        rest[i] = rest[i + 1] + (lists[i][0][0] if lists[i] else 0)
    best = [0]

    def search(i, used, revenue):
        best[0] = max(best[0], revenue)
        if i == len(lists) or revenue + rest[i] <= best[0]:
            return
        for r_revenue, bits in lists[i]:
            if revenue + r_revenue + rest[i + 1] <= best[0]:
                break
            if not bits & used:
                search(i + 1, used | bits, revenue + r_revenue)
        search(i + 1, used, revenue)

    search(0, 0, 0)
    return best[0]


def python_optimum(py_game, corp_sym):
    """{auto, exact, timeout, t_py} for ``corp_sym`` in ``py_game``."""
    from rl18xx.game.engine.autorouter import AutoRouter

    flashes = []
    router = AutoRouter(py_game, flash=flashes.append)
    captured = {}
    evaluate = router.js_evaluate_combos

    def capture(sorted_routes, route_timeout):
        captured["routes"] = sorted_routes
        return evaluate(sorted_routes, route_timeout)

    router.js_evaluate_combos = capture
    t0 = time.perf_counter()
    best = router.compute(py_game.corporation_by_id(corp_sym))
    t_py = time.perf_counter() - t0
    return {
        "auto": sum(r.revenue() for r in best if r is not None),
        "exact": exact_combo_revenue(router, captured.get("routes", [])),
        "timeout": bool(flashes),
        "t_py": t_py,
    }


def rust_search(game, corp_sym, reps):
    """(revenue, best-of-``reps`` seconds) of the Rust search on a clone with
    a warm graph cache."""
    clone = game.pickle_clone()
    clone.calculate_routes(corp_sym)
    best_t = None
    for _ in range(reps):
        t0 = time.perf_counter()
        _, revenue = clone.calculate_routes(corp_sym)
        dt = time.perf_counter() - t0
        best_t = dt if best_t is None else min(best_t, dt)
    return revenue, best_t


def run_random(seed, do_python, reps):
    import engine_rs
    from rl18xx.game.engine.actions import BaseAction
    from rl18xx.game.gamemap import GameMap

    num_players = [4, 3, 5, 2, 6][seed % 5]
    unlock = seed % 2 == 1
    players = {i: f"Player {i}" for i in range(1, num_players + 1)}
    rng = random.Random(seed)
    game = engine_rs.BaseGame(players)
    game.set_auction_unlock(unlock)
    times = []
    for _ in range(MAX_MOVES):
        if game.finished:
            break
        if "run_routes" in game.legal_action_types():
            times.append(rust_search(game, game.current_entity_id.split(":", 1)[1], reps)[1])
        idx = int(rng.choice(game.factored_legal_indices()))
        price_range = game.price_range_for_index(idx)
        price = None
        if price_range is not None and price_range[0] != price_range[1]:
            price = rng.randint(price_range[0], price_range[1])
        game.apply_action_index(idx, price)
    rows = []
    py_game = None
    if do_python:
        py_game = GameMap().game_by_title("1830")(players)
        py_game.auction_unlock = unlock
    for i, action in enumerate(game.raw_actions):
        if action["type"] == "run_routes" and do_python:
            row = {"i": i, "corp": action["entity"], "rust": sum(r["revenue"] for r in action["routes"])}
            row.update(python_optimum(py_game, action["entity"]))
            rows.append(row)
        if do_python:
            py_game.process_action(BaseAction.action_from_dict(copy.deepcopy(action), py_game))
    return {"source": f"seed {seed}", "rows": rows, "rust_times": times}


def run_human(path, do_python, reps):
    import engine_rs
    from rl18xx.agent.alphazero.pretraining import engine_optional_rules
    from rl18xx.game.engine.actions import BaseAction
    from rl18xx.game.gamemap import GameMap

    record = json.loads(Path(path).read_text())
    if "players" not in record:  # a dropped-game record
        return {"source": path, "rows": [], "rust_times": []}
    players = {int(p["id"]): f"Player {p['id']}" for p in record["players"]}
    rules = engine_optional_rules(record)
    game = engine_rs.BaseGame(players, optional_rules=rules)
    py_game = GameMap().game_by_title("1830")(players, optional_rules=rules) if do_python else None
    rows, times = [], []
    for i, action in enumerate(record["actions"]):
        if action["type"] == "run_routes":
            revenue, t = rust_search(game, action["entity"], reps)
            times.append(t)
            row = {
                "i": i,
                "corp": action["entity"],
                "rust": revenue,
                "human": sum(r.get("revenue", 0) for r in action["routes"]),
            }
            if do_python:
                row.update(python_optimum(py_game, action["entity"]))
            rows.append(row)
        game.process_action(copy.deepcopy(action))
        if do_python:
            py_game.process_action(BaseAction.action_from_dict(copy.deepcopy(action), py_game))
    return {"source": path, "rows": rows, "rust_times": times}


def _job(args):
    mode, item, do_python, reps = args
    try:
        return run_random(int(item), do_python, reps) if mode == "random" else run_human(item, do_python, reps)
    except Exception as exc:  # noqa: BLE001 — report and keep sweeping
        return {"source": str(item), "rows": [], "rust_times": [], "error": f"{type(exc).__name__}: {exc}"}


def summarize(results):
    counts = Counter()
    worst = []
    times = []
    for res in results:
        times += res["rust_times"]
        if "error" in res:
            counts["errors"] += 1
        for row in res["rows"]:
            counts["states"] += 1
            if "exact" not in row:
                continue
            key = (
                "rust == exact"
                if row["rust"] == row["exact"]
                else ("rust > exact" if row["rust"] > row["exact"] else "rust < exact")
            )
            counts[key] += 1
            counts["autorouter timed out"] += row["timeout"]
            counts["auto != exact"] += row["auto"] != row["exact"]
            if "human" in row:
                counts["human < exact"] += row["human"] < row["exact"]
                counts["human > exact"] += row["human"] > row["exact"]
            if row["rust"] != row["exact"]:
                worst.append((res["source"], row))
    print(dict(counts))
    for source, row in worst[:20]:
        print("  differs:", source, row)
    if times:
        times.sort()
        q = lambda p: times[min(len(times) - 1, int(p * len(times)))] * 1e6  # noqa: E731
        print(
            f"rust route search: n={len(times)} mean={statistics.mean(times) * 1e6:.0f}us p50={q(.5):.0f}us "
            f"p90={q(.9):.0f}us p99={q(.99):.0f}us max={times[-1] * 1e6:.0f}us"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["random", "human"])
    parser.add_argument("items", nargs="+", help="random: SEED_LO SEED_HI; human: game JSON files")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", help="write one result per line here")
    parser.add_argument("--no-python", action="store_true", help="Rust timings only")
    parser.add_argument("--reps", type=int, default=3, help="time each Rust search as the best of this many runs")
    args = parser.parse_args()
    items = list(range(int(args.items[0]), int(args.items[1]))) if args.mode == "random" else args.items
    jobs = [(args.mode, item, not args.no_python, args.reps) for item in items]
    out = open(args.json, "w") if args.json else None
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap_unordered(_job, jobs):
            results.append(res)
            if out:
                out.write(json.dumps(res) + "\n")
                out.flush()
            differ = sum(1 for r in res["rows"] if "exact" in r and r["rust"] != r["exact"])
            print(f"{res['source']}: {len(res['rows'])} run_routes, {differ} differ {res.get('error', '')}", flush=True)
    summarize(results)


if __name__ == "__main__":
    main()
