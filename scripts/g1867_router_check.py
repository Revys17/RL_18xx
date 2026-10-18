"""1867 router check: at every recorded human ``run_routes`` the engine's
own optimum (``g1867_auto_routes_json``, called BEFORE the recorded action
is applied) must be at least the recorded revenue, and equal in the large
majority. Reports equal / above / BELOW counts (below = the router misses a
legal run a human found — a router bug or a rule the router does not know)
with example game ids and file action ids, plus router time per call
(mean / p99 / max). No player data.

Recorded revenue = the sum of the routes' ``revenue``; a route recorded
without one is priced by the engine's replay pricer (``route_revenue_py``).
Every engine run is also re-priced through ``route_revenue_py`` from its own
``connections`` (the replay path a record without revenue takes): a
different price means the emitted connections do not describe the run.

``--ruby`` sends every "above" case (the engine beat the human) to Ruby HEAD
(``scripts/g1867_router_ruby.rb`` in docker, the rig of
``scripts/ruby_diff.py``): Ruby validates and prices the engine's run
(revenue omitted); a Ruby error or a different total is a router bug (an
illegal or mispriced route). Every "below" case sends the HUMAN's run
(revenue omitted) instead: when Ruby HEAD prices it at or under the
engine's optimum the record's revenue is stale (an older server's
pricing), otherwise it is a real router shortfall. Ruby's per-route prices
must match the engine's too (`ruby_route_split_differs`). Old records Ruby
HEAD cannot replay up to the decision (pre-#9655 merger order, ...) count
as `ruby_cannot_replay` (unverifiable). Rig: the Ruby clone in
``logs/ruby/18xx`` and the gem dir ``logs/ruby/gems`` (``require_all``), as
in docs/g1867_remaining_work.md.

Corpora (defaults): the 2026-10 scrape and the first 300 files of
``human_games/1867`` (sorted, as ``tests/g1867_corpus_audit.py --limit
300``); 2-player games, games with optional rules (grid_market) and
unfinished games are skipped; a game whose replay stops on a rejected
action keeps the run_routes before it.

    CUDA_VISIBLE_DEVICES="" RUBY_DOCKER_CPUSET=40-63 uv run python \\
        scripts/g1867_router_check.py --jobs 6 --ruby --json /tmp/r.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import multiprocessing as mp
import resource
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_rs  # noqa: E402

from tests.title_replay_harness import game_seed, ruby_filtered_actions  # noqa: E402

TITLE = "1867"
DEFAULT_CORPORA = ["human_games/scrape_2026_10/1867/*.json", "human_games/1867/*.json:300"]


def _limit_memory(gb: float) -> None:
    if gb > 0:
        b = int(gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (b, b))


def skip_reason(game: dict) -> str | None:
    if game.get("title") != TITLE:
        return "title"
    if len(game.get("players") or []) == 2:
        return "2p"
    if (game.get("settings") or {}).get("optional_rules"):
        return "optional_rules"
    if game.get("status") not in ("finished", "archived"):
        return "unfinished"
    return None


def check_game(path: str) -> dict:
    game = json.load(open(path))
    gid = Path(path).stem
    out = {
        "id": gid,
        "path": path,
        "counts": Counter(),
        "below": [],
        "above": [],
        "times": [],
        "inexact": 0,
        "conn_mismatch": [],
        "error": None,
        "checks": [],
        "steps": [],
        "slow": [],
    }
    reason = skip_reason(game)
    if reason:
        out["error"] = f"skipped {reason}"
        return out
    seat = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    g = engine_rs.BaseGame.new_titled(TITLE, {v: f"Player {v}" for v in seat.values()}, None, game_seed(game))
    raw = json.loads(json.dumps(game["actions"]))
    for a in raw:
        a["_fid"] = a.get("id")
        for au in a.get("auto_actions") or []:
            au["_fid"] = a.get("id")
    for a in ruby_filtered_actions(raw):
        fid = a.pop("_fid", None)
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        if a["type"] == "run_routes" and not g.finished:
            corp = str(a["entity"])
            recorded = 0
            for r in a.get("routes") or []:
                if r.get("revenue") is not None:
                    recorded += int(r["revenue"])
                else:
                    out["counts"]["priced_record"] += 1
                    recorded += g.route_revenue_py(corp, r["train"], r["connections"])
            t0 = time.perf_counter()
            res = json.loads(g.g1867_auto_routes_json(corp))
            out["times"].append(time.perf_counter() - t0)
            out["steps"].append(res.get("steps", [0, 0]))
            best = res["revenue"]
            mix = "+".join(sorted(r["train"].rsplit("-", 1)[0] for r in a.get("routes") or [])) or "-"
            out["slow"].append((out["times"][-1], fid, out["steps"][-1], mix))
            out["slow"] = sorted(out["slow"], reverse=True)[:3]
            if not res.get("exact", True):
                out["inexact"] += 1
            for r in res["routes"]:
                p = g.route_revenue_py(corp, r["train"], r["connections"])
                if p != r["revenue"]:
                    out["conn_mismatch"].append((fid, r["train"], r["revenue"], p))
            if best == recorded:
                out["counts"]["equal"] += 1
            elif best > recorded:
                out["counts"]["above"] += 1
                out["above"].append((fid, recorded, best, mix))
                rs = [{k: v for k, v in r.items() if k != "revenue"} for r in res["routes"]]
                out["checks"].append(
                    {
                        "fid": fid,
                        "corp": corp,
                        "routes": rs,
                        "engine": best,
                        "engine_routes": [r["revenue"] for r in res["routes"]],
                        "kind": "above",
                    }
                )
            else:
                out["counts"]["below"] += 1
                out["below"].append((fid, recorded, best, mix))
                rs = [{k: v for k, v in r.items() if k not in ("revenue", "revenue_str")} for r in a["routes"]]
                out["checks"].append({"fid": fid, "corp": corp, "routes": rs, "engine": best, "kind": "below"})
        try:
            g.process_action(a)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = f"replay rejected file id {fid}: {exc}"[:200]
            return out
    return out


def check_game_safe(args: tuple[str, float, int]) -> dict:
    path, mem_gb, express_budget = args
    _limit_memory(mem_gb)
    if express_budget:
        engine_rs.BaseGame.g1867_router_set_express_budget(express_budget)
    try:
        return check_game(path)
    except BaseException as exc:  # noqa: BLE001
        return {
            "id": Path(path).stem,
            "path": path,
            "counts": Counter(),
            "below": [],
            "above": [],
            "times": [],
            "inexact": 0,
            "conn_mismatch": [],
            "error": f"crash: {exc}"[:200],
            "checks": [],
            "steps": [],
            "slow": [],
        }


def ruby_validate(path: str, checks: list[dict], work: Path) -> list[dict]:
    """Run scripts/g1867_router_ruby.rb for one game's checks (docker)."""
    import subprocess

    from scripts.ruby_diff import CPUSET

    work.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    cin = work / f"{stem}.checks.json"
    cout = work / f"{stem}.out.json"
    cin.write_text(json.dumps(checks))
    if cout.exists():
        cout.unlink()
    repo = Path("logs/ruby/18xx").resolve()
    gems = Path("logs/ruby/gems").resolve()
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--cpuset-cpus",
            CPUSET,
            "-e",
            "ENGINE_LOG_LEVEL=3",
            "-v",
            f"{gems}:/usr/local/bundle",
            "-v",
            f"{repo}:/18xx:ro",
            "-v",
            f"{Path(path).resolve().parent}:/games:ro",
            "-v",
            f"{Path(__file__).resolve().parent}:/probe:ro",
            "-v",
            f"{work.resolve()}:/out",
            "ruby:3.2-slim",
            "ruby",
            "/probe/g1867_router_ruby.rb",
            f"/games/{Path(path).name}",
            f"/out/{cin.name}",
            f"/out/{cout.name}",
        ],
        check=False,
        capture_output=True,
    )
    if not cout.exists():
        return [{"fid": c["fid"], "error": "ruby produced no output"} for c in checks]
    return json.loads(cout.read_text())


def corpus_paths(specs: list[str]) -> list[str]:
    paths: list[str] = []
    for spec in specs:
        pattern, _, limit = spec.partition(":")
        ps = sorted(globmod.glob(pattern))
        paths += ps[: int(limit)] if limit else ps
    return paths


def summarize(name: str, results: list[dict]) -> dict:
    totals: Counter = Counter()
    below, above, conn = [], [], []
    times: list[float] = []
    steps: list[list[int]] = []
    slow: list = []
    errors: Counter = Counter()
    inexact = 0
    for r in results:
        totals.update(r["counts"])
        below += [(r["id"], *x) for x in r["below"]]
        above += [(r["id"], *x) for x in r["above"]]
        conn += [(r["id"], *x) for x in r["conn_mismatch"]]
        times += r["times"]
        steps += r.get("steps", [])
        slow += [(t, r["id"], fid, st, mix) for t, fid, st, mix in r.get("slow", [])]
        inexact += r["inexact"]
        if r["error"]:
            errors[r["error"].split(" file id")[0].split(":")[0]] += 1
    times.sort()
    n = len(times)
    timing = {
        "calls": n,
        "mean_ms": 1000 * sum(times) / max(n, 1),
        "p99_ms": 1000 * times[min(n - 1, int(0.99 * n))] if n else 0.0,
        "max_ms": 1000 * times[-1] if n else 0.0,
        "max_steps_normal": max((x[0] for x in steps), default=0),
        "max_steps_express": max((x[1] for x in steps), default=0),
        "calls_over_100ms": sum(1 for t in times if t > 0.1),
        "calls_over_1s": sum(1 for t in times if t > 1.0),
        "slowest": [(round(1000 * t), gid, fid, st, mix) for t, gid, fid, st, mix in sorted(slow, reverse=True)[:10]],
    }
    print(f"== {name}: {len(results)} games, {n} run_routes; errors/skips {dict(errors)}")
    print(
        f"   equal {totals['equal']}  above {totals['above']}  BELOW {totals['below']}"
        f"  (records priced by the engine: {totals['priced_record']}; inexact calls: {inexact})"
    )
    print(
        f"   router time: mean {timing['mean_ms']:.2f} ms, p99 {timing['p99_ms']:.1f} ms,"
        f" max {timing['max_ms']:.0f} ms; most walk steps: normal {timing['max_steps_normal']},"
        f" 5+5E {timing['max_steps_express']}"
    )
    print(
        f"   calls over 100 ms: {timing['calls_over_100ms']}, over 1 s: {timing['calls_over_1s']};"
        " slowest (ms, game, file id, [normal, 5+5E steps], trains):",
        timing["slowest"][:6],
    )
    print("   below examples (game, file id, recorded, engine, trains):", below[:12])
    print("   above examples:", above[:8])
    if conn:
        print(f"   CONNECTIONS re-priced differently: {len(conn)}", conn[:8])
    return {
        "totals": dict(totals),
        "below": below,
        "above": above,
        "conn_mismatch": conn,
        "errors": dict(errors),
        "timing": timing,
        "inexact": inexact,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="glob[:limit] (sorted, first `limit` files); repeatable. Default: the scrape + 300 old games",
    )
    ap.add_argument("--games", nargs="*", default=None, help="explicit game files (overrides --corpus)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--mem-gb", type=float, default=3.0, help="address-space cap per worker (0: none)")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--ruby", action="store_true", help="Ruby-validate every 'above' and 'below' case")
    ap.add_argument("--work", default="logs/ruby/router1867")
    ap.add_argument(
        "--express-budget",
        type=int,
        default=0,
        help="experiments: the 5+5E walk's step budget (0: the engine default)",
    )
    args = ap.parse_args()
    groups: list[tuple[str, list[str]]]
    if args.games:
        groups = [("games", args.games)]
    else:
        groups = [(spec, corpus_paths([spec])) for spec in (args.corpus or DEFAULT_CORPORA)]
    report: dict = {}
    all_results: list[dict] = []
    for name, paths in groups:
        work = [(p, args.mem_gb, args.express_budget) for p in paths]
        if args.jobs > 1:
            with mp.get_context("fork").Pool(args.jobs, maxtasksperchild=20) as pool:
                results = pool.map(check_game_safe, work, chunksize=1)
        else:
            results = [check_game_safe(w) for w in work]
        report[name] = summarize(name, results)
        all_results += results
    mism, inexact = engine_rs.BaseGame.g1867_router_counters()
    if args.jobs <= 1:
        print(f"router counters (this process): price mismatches {mism}, inexact calls {inexact}")

    if args.ruby:
        from concurrent.futures import ThreadPoolExecutor

        todo = [(r["path"], r["checks"]) for r in all_results if r["checks"]]
        with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 4))) as ex:
            outs = list(ex.map(lambda t: (t[0], t[1], ruby_validate(t[0], t[1], Path(args.work))), todo))
        ruby: Counter = Counter()
        ruby_below: Counter = Counter()
        ruby_bad = []
        for path, checks, res in outs:
            by_fid = {c["fid"]: c for c in checks}
            for r in res:
                c = by_fid.get(r["fid"])
                gid = Path(path).stem
                err = r.get("error")
                # Ruby HEAD cannot replay the game up to this action (old
                # records: pre-#9655 merger order etc.) -- unverifiable
                stopped = bool(err) and err.startswith("RuntimeError: Ruby stopped")
                if c and c["kind"] == "below":
                    # Ruby HEAD's price of the HUMAN run vs the engine's optimum
                    if stopped:
                        ruby_below["ruby_cannot_replay"] += 1
                    elif err:
                        ruby_below["human_run_rejected"] += 1
                        ruby_bad.append((gid, r["fid"], f"below, human run rejected: {err}"))
                    elif r["total"] <= c["engine"]:
                        ruby_below["record_stale"] += 1
                        ruby_bad.append((gid, r["fid"], f"below, stale: ruby {r['total']} engine {c['engine']}"))
                    else:
                        ruby_below["router_short"] += 1
                        ruby_bad.append((gid, r["fid"], f"below: ruby {r['total']} engine {c['engine']}"))
                elif stopped:
                    ruby["ruby_cannot_replay"] += 1
                elif err:
                    ruby["ruby_error"] += 1
                    ruby_bad.append((gid, r["fid"], err))
                elif c and r["total"] == c["engine"]:
                    ruby["ruby_agrees"] += 1
                    if r.get("revenues") != c.get("engine_routes"):
                        ruby["ruby_route_split_differs"] += 1
                        split = f"per-route ruby {r.get('revenues')} engine {c.get('engine_routes')}"
                        ruby_bad.append((gid, r["fid"], split))
                else:
                    ruby["ruby_differs"] += 1
                    ruby_bad.append((gid, r["fid"], f"ruby {r['total']} engine {c and c['engine']}"))
        print(f"Ruby on 'above' runs (the engine's run, priced by Ruby HEAD): {dict(ruby)}")
        print(f"Ruby on 'below' runs (the human run, priced by Ruby HEAD): {dict(ruby_below)}")
        for b in ruby_bad[:25]:
            print("  ", b)
        report["ruby"] = {"above": dict(ruby), "below": dict(ruby_below), "bad": ruby_bad}
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
