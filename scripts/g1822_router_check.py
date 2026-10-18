"""1822 router check: at every recorded human ``run_routes`` the engine's
own optimum (``g1822_router_best``) must be at least the recorded revenue,
and equal in the large majority. Reports equal / above / BELOW counts
(below = the router misses a legal run a human found — a router bug or a
rule the router does not know) with example game ids and file action ids,
plus router time per call. No player data.

``--ruby`` sends every "above" case (the engine beat the human) to Ruby HEAD
(docker rig of scripts/ruby_diff.py): Ruby validates and prices the
engine's run (revenue omitted); a Ruby error or a different total is a
router bug (an illegal or mispriced route). Every "below" case sends the
HUMAN's run (revenue omitted) instead: when Ruby HEAD prices it at or
under the engine's optimum the record's revenue is stale (an older
server's pricing), otherwise it is a real router shortfall.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_router_check.py \\
        --glob 'human_games/scrape_2026_10/1822/*.json' --jobs 8 --json /tmp/r.json
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_rs  # noqa: E402

from tests.title_replay_harness import SEAT_ID_BASE, game_seed, ruby_filtered_actions  # noqa: E402


TITLE = "1822"


def check_game(path: str) -> dict:
    game = json.load(open(path))
    gid = Path(path).stem
    out = {
        "id": gid,
        "counts": Counter(),
        "below": [],
        "above": [],
        "time": 0.0,
        "calls": 0,
        "slow": [],
        "error": None,
        "checks": [],
    }
    if game.get("title") != TITLE or game.get("status") not in ("finished", "archived"):
        out["error"] = "skipped"
        return out
    base = SEAT_ID_BASE.get(TITLE, 1000)
    seat = {p["id"]: base + i + 1 for i, p in enumerate(game["players"])}
    rules = list((game.get("settings") or {}).get("optional_rules") or [])
    try:
        g = engine_rs.BaseGame.new_titled(
            TITLE, {v: f"Player {v - base}" for v in seat.values()}, rules or None, game_seed(game)
        )
    except ValueError as exc:
        out["error"] = f"construction: {exc}"
        return out
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
            recorded = sum(int(r.get("revenue") or 0) for r in a.get("routes") or [])
            t0 = time.perf_counter()
            best, _routes = g.g1822_router_best(str(a["entity"]))
            dt = time.perf_counter() - t0
            out["time"] += dt
            out["calls"] += 1
            if dt > 0.25:
                out["slow"].append((fid, round(dt, 3)))
            if best == recorded:
                out["counts"]["equal"] += 1
            elif best > recorded:
                out["counts"]["above"] += 1
                if len(out["above"]) < 3:
                    out["above"].append((fid, recorded, best))
                rs = json.loads(_routes)
                for r in rs:
                    r.pop("revenue", None)
                out["checks"].append(
                    {"fid": fid, "corp": str(a["entity"]), "routes": rs, "engine": best, "kind": "above"}
                )
            else:
                out["counts"]["below"] += 1
                out["below"].append((fid, recorded, best))
                rs = [{k: v for k, v in r.items() if k not in ("revenue", "revenue_str")} for r in a["routes"]]
                out["checks"].append(
                    {"fid": fid, "corp": str(a["entity"]), "routes": rs, "engine": best, "kind": "below"}
                )
        try:
            g.process_action(a)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = f"replay rejected file id {fid}: {exc}"[:200]
            return out
    return out


def ruby_validate(path: str, checks: list[dict], work: Path) -> list[dict]:
    """Run scripts/g1822_router_ruby.rb for one game's checks (docker)."""
    import subprocess

    from scripts.ruby_diff import CPUSET

    work.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    cin = work / f"{stem}.checks.json"
    cout = work / f"{stem}.out.json"
    cin.write_text(json.dumps(checks))
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
            "/probe/g1822_router_ruby.rb",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default="1822", help="an 1822-family title (1822, 1822PNW, ...)")
    ap.add_argument("--glob", default=None, help="default: human_games/scrape_2026_10/<title>/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--ruby", action="store_true", help="Ruby-validate every 'above' and 'below' case")
    ap.add_argument("--work", default="logs/ruby/router")
    args = ap.parse_args()
    global TITLE
    TITLE = args.title
    paths = sorted(globmod.glob(args.glob or f"human_games/scrape_2026_10/{TITLE}/*.json"))[: args.limit]
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            results = pool.map(check_game, paths, chunksize=2)
    else:
        results = [check_game(p) for p in paths]
    totals: Counter = Counter()
    below, above = [], []
    t, n = 0.0, 0
    errors = Counter()
    slow = []
    for r in results:
        totals.update(r["counts"])
        below += [(r["id"], *x) for x in r["below"]]
        above += [(r["id"], *x) for x in r["above"]]
        t += r["time"]
        n += r["calls"]
        slow += [(r["id"], *x) for x in r["slow"]]
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1
    print(f"{len(paths)} games, {n} run_routes; errors {dict(errors)}")
    print(f"equal {totals['equal']}  above {totals['above']}  BELOW {totals['below']}")
    print(f"router time: {t:.1f}s total, {1000 * t / max(n, 1):.1f} ms/call")
    slow.sort(key=lambda x: -x[2])
    print(f"calls over 0.25 s: {len(slow)}; slowest (game, file id, s): {slow[:8]}")
    print("below examples (game, file id, recorded, engine):", below[:12])
    print("above examples:", above[:8])
    ruby = Counter()
    ruby_bad = []
    if args.ruby:
        from concurrent.futures import ThreadPoolExecutor

        todo = [(p, r["checks"]) for p, r in zip(paths, results) if r["checks"]]
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            outs = list(ex.map(lambda t: (t[0], t[1], ruby_validate(t[0], t[1], Path(args.work))), todo))
        ruby_below = Counter()
        for path, checks, res in outs:
            by_fid = {c["fid"]: c for c in checks}
            for r in res:
                c = by_fid.get(r["fid"])
                if c and c["kind"] == "below":
                    # Ruby HEAD's price of the HUMAN run vs the engine's optimum
                    if "error" in r:
                        ruby_below["human_run_rejected"] += 1
                    elif r["total"] <= c["engine"]:
                        ruby_below["record_stale"] += 1
                    else:
                        ruby_below["router_short"] += 1
                        ruby_bad.append((Path(path).stem, r["fid"], f"below: ruby {r['total']} engine {c['engine']}"))
                elif "error" in r:
                    ruby["ruby_error"] += 1
                    ruby_bad.append((Path(path).stem, r["fid"], r["error"]))
                elif c and r["total"] == c["engine"]:
                    ruby["ruby_agrees"] += 1
                else:
                    ruby["ruby_differs"] += 1
                    ruby_bad.append((Path(path).stem, r["fid"], f"ruby {r['total']} engine {c and c['engine']}"))
        print(f"Ruby on 'above' runs: {dict(ruby)}")
        print(f"Ruby on 'below' runs (the human run, priced by Ruby HEAD): {dict(ruby_below)}")
        ruby["below"] = dict(ruby_below)
        for b in ruby_bad[:15]:
            print("  ", b)
    if args.json_out:
        json.dump(
            {
                "totals": dict(totals),
                "below": below,
                "above": above,
                "errors": dict(errors),
                "ms_per_call": 1000 * t / max(n, 1),
                "slow": slow,
                "ruby": dict(ruby),
                "ruby_bad": ruby_bad,
            },
            open(args.json_out, "w"),
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
