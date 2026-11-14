"""Which tile instance the 1822 family's generator offers, against the human records and Ruby HEAD.

Ruby's UI offers, for each tile name, the FIRST instance of that name in the game's supply (`@game.tiles`:
`Tracker#potential_tiles` keeps `uniq(&:name)`), and a lay sends that instance's id. The supply's order is the
initial instances still unlaid (index order), then the instances appended since (a tile upgraded off the map,
an unlimited tile's extra instance). The generator offers the same instance (`BaseGame.g1822_offered_tile_instance`,
since 2026-11-14; before, the lowest-numbered instance not on the map). In 1822PNW an instance upgraded off the
map keeps its builder cubes and brings back their discount when laid again, so the instance changes the cost.

For every game of a title's corpus (or given game files: `--glob`, e.g. `scripts/g1822_fuzz.py` walks), at every
human decision whose generated moves (the rules' set) include tile lays this script counts the positions where
the offered instance of some tile name differs from the old rule's (`changed_positions`, `changed_lays`), and at
every human `lay_tile` it compares the instance the human sent (= Ruby's UI's offer) with the generator's
(`new == human`) and the old rule's. ``--ruby`` also asks Ruby HEAD (the docker rig of scripts/ruby_diff.py,
`scripts/g1822_tile_instance_ruby.rb`) for its first instance at every human lay of the games where the new and
old rule differ somewhere (``--ruby-all``: every game with a lay).

    CUDA_VISIBLE_DEVICES= taskset -c 24-31,56-63 nice -19 uv run python -P scripts/g1822_tile_instance_check.py \\
        --title 1822PNW --jobs 8 [--ruby]
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import multiprocessing as mp
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS = "human_games/scrape_2026_10/{title}/*.json"


def old_instance(on_map: set, name: str) -> str:
    """The rule before 2026-11-14: the lowest-numbered instance not on the map."""
    for n in range(10_000):
        if f"{name}-{n}" not in on_map:
            return f"{name}-{n}"
    return f"{name}-0"


def check_game(args) -> dict:
    path, title = args
    from scripts.g1822_generator_profile import human_actions, new_game

    game = json.load(open(path))
    out = {"id": Path(path).stem, "path": path, "positions": 0, "lay_positions": 0, "changed_positions": 0,
           "changed_lays": 0, "lays": [], "error": None}  # fmt: skip
    if game.get("title") != title:
        out["error"] = "skipped"
        return out
    try:
        g, seat = new_game(title, game)
    except ValueError as exc:
        out["error"] = f"construction: {exc}"
        return out
    for fid, auto, a in human_actions(game, seat):
        if not auto and not g.finished:
            out["positions"] += 1
            moves = json.loads(g.g1822_legal_moves_json("rules"))
            lays = [m["wire"] for m in moves if m["wire"]["type"] == "lay_tile" and "-" in str(m["wire"].get("tile"))]
            names = {str(w["tile"]).rsplit("-", 1)[0] for w in lays}
            names.discard("BC")
            if names:
                out["lay_positions"] += 1
                on_map = {h.tile.id for h in g.hexes}
                changed = {n for n in names if g.g1822_offered_tile_instance(n) != old_instance(on_map, n)}
                if changed:
                    out["changed_positions"] += 1
                    out["changed_lays"] += sum(1 for w in lays if str(w["tile"]).rsplit("-", 1)[0] in changed)
            if a["type"] == "lay_tile" and "-" in str(a.get("tile")):
                name = str(a["tile"]).rsplit("-", 1)[0]
                if name != "BC":
                    on_map = {h.tile.id for h in g.hexes}
                    out["lays"].append(
                        {"fid": fid, "name": name, "human": a["tile"], "new": g.g1822_offered_tile_instance(name),
                         "old": old_instance(on_map, name)}  # fmt: skip
                    )
        try:
            g.process_action(a)
        except BaseException as exc:  # noqa: BLE001 -- a record the replay rejects: stop there
            out["error"] = f"replay rejected file id {fid}: {str(exc)[:120]}"
            break
    return out


def ruby_firsts(path: str, lays: list, work: Path) -> dict:
    """{fid: Ruby HEAD's first instance of the lay's tile name} (docker)."""
    from scripts.ruby_diff import CPUSET, ensure_gems

    work.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    cin = work / f"{stem}.instances.json"
    cout = work / f"{stem}.instances.out.json"
    cin.write_text(json.dumps([{"fid": x["fid"], "name": x["name"]} for x in lays]))
    repo = Path("logs/ruby/18xx").resolve()
    gems = Path("logs/ruby/gems").resolve()
    ensure_gems(gems)
    subprocess.run(
        ["docker", "run", "--rm", "--cpuset-cpus", CPUSET, "-e", "ENGINE_LOG_LEVEL=3",
         "-v", f"{gems}:/usr/local/bundle", "-v", f"{repo}:/18xx:ro",
         "-v", f"{Path(path).resolve().parent}:/games:ro",
         "-v", f"{Path(__file__).resolve().parent}:/probe:ro", "-v", f"{work.resolve()}:/out",
         "ruby:3.2-slim", "ruby", "/probe/g1822_tile_instance_ruby.rb",
         f"/games/{Path(path).name}", f"/out/{cin.name}", f"/out/{cout.name}"],
        check=False, capture_output=True,
    )  # fmt: skip
    if not cout.exists():
        return {}
    return {r["fid"]: r.get("first", "error: " + str(r.get("error"))) for r in json.loads(cout.read_text())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--title", default="1822PNW")
    ap.add_argument("--glob", action="append", default=None, help="default: the title's 2026-10 scrape")
    ap.add_argument("--corpus-root", default=".", help="where human_games/ lives")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ruby", action="store_true", help="ask Ruby HEAD at every human lay of the affected games")
    ap.add_argument("--ruby-all", action="store_true", help="--ruby on every game with a human lay")
    ap.add_argument("--work", default="logs/ruby/instances")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import engine_rs  # noqa: F401

    globs = args.glob or [str(Path(args.corpus_root) / CORPUS.format(title=args.title))]
    paths = sorted(p for g in globs for p in globmod.glob(g))
    with mp.get_context("fork").Pool(args.jobs) as pool:
        results = list(pool.imap_unordered(check_game, [(p, args.title) for p in paths], chunksize=1))
    games = [r for r in results if r["error"] != "skipped"]
    lays = [x for r in games for x in r["lays"]]
    print(
        f"{args.title}: {len(games)} games, {sum(r['positions'] for r in games)} human decisions, "
        f"{sum(r['lay_positions'] for r in games)} with tile lays; the offered instance changes at "
        f"{sum(r['changed_positions'] for r in games)} of them ({sum(r['changed_lays'] for r in games)} lay moves)"
    )
    c = Counter((x["new"] == x["human"], x["old"] == x["human"]) for x in lays)
    print(
        f"human lays: {len(lays)}; new == human {sum(v for (n, _), v in c.items() if n)}, "
        f"old == human {sum(v for (_, o), v in c.items() if o)}; (new, old) == human: {dict(c)}"
    )
    for x in [x for r in games for x in r["lays"] if x["new"] != x["human"]][:10]:
        print("  new != human:", x)
    affected = [r for r in games if any(x["new"] != x["old"] for x in r["lays"]) or r["changed_positions"]]
    print(f"games where the rules differ somewhere: {len(affected)}")
    if args.ruby or args.ruby_all:
        todo = [r for r in (games if args.ruby_all else affected) if r["lays"]]
        work = Path(args.work)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(1, args.jobs // 2)) as ex:
            firsts = list(ex.map(lambda r: ruby_firsts(r["path"], r["lays"], work / args.title), todo))
        rc = Counter()
        bad = []
        for r, f in zip(todo, firsts):
            for x in r["lays"]:
                ruby = f.get(x["fid"])
                if ruby is None:
                    rc["ruby: no answer (past a record Ruby rejects)"] += 1
                    continue
                rc[f"new == ruby: {x['new'] == ruby}"] += 1
                rc[f"old == ruby: {x['old'] == ruby}"] += 1
                rc[f"human == ruby: {x['human'] == ruby}"] += 1
                if x["new"] != ruby and len(bad) < 10:
                    bad.append((r["id"], x, ruby))
        print(f"Ruby HEAD at the human lays of {len(todo)} games: {dict(rc)}")
        for b in bad:
            print("  new != ruby:", b)
    if args.json:
        Path(args.json).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
