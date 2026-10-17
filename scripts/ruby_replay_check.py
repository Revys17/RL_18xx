"""Replay games OUR engine generated in the real Ruby engine (tobymao/18xx).

Corpus parity (tests/parity_runner.py, tests/cleaning_diff.py) only sees moves
humans can make on 18xx.games, so a rule our engines both get wrong in the
direction of allowing too much never shows up there. This replays games the
engine itself produced -- self-play / eval games saved in 18xx.games export
format, and random walks generated here -- through Ruby, and reports per game
the first action Ruby rejects or the first action after which Ruby's state
differs from the Rust engine's (the Rust and Python engines are kept at parity
by the corpus gates) -- states after every action: round, acting entity,
phase, bank, player cash by seat, every corporation's price / cash / trains /
share holders, private owners, and the map (laid tiles, tokens by city).
Results are grouped by cause.

Statuses: ok; diverges (first differing state); ruby_rejects (Ruby refuses an
action ours took); both_reject (a game recorded under older rules of ours:
both engines now refuse the same action); rust_rejects; result_differs.

Known, accounted-for differences (see ruby_replay_dump.rb):
  * implicit passes -- our 1830 deliberately forbids buying trains from
    corporations other players run and privates from other players, so where
    Ruby offers only that (plus pass) our engine moves on and Ruby waits for a
    pass. Each is inserted, counted and grouped (summary "implicit_passes");
    any other `why` is a finding. A recorded purchase of that kind is a bug in
    our restriction (summary "cross_player_purchases").
  * G19's two cities are numbered the other way round in our engines
    (SWAPPED_PREPRINTED_CITIES); city references are translated for Ruby.

The docker rig is scripts/ruby_diff.py's (CPUSET, ensure_gems), but not its
Ruby driver: ruby_dump.rb replays RECORDED games through Ruby's own filtered
actions, while games our engine generated need Ruby to insert those implicit
passes, keyed by our engine's acting entity / step / round, before each action
(ruby_replay_dump.rb), and a finer per-action state (share holders, privates,
the map).

Setup (once; read-only use of the clone, never contact 18xx.games):
    git clone --depth 1 https://github.com/tobymao/18xx.git logs/ruby/18xx
    docker image inspect ruby:3.2-slim   # `docker pull ruby:3.2-slim` if missing
    (the require_all gem is installed into --gems on first use)

Usage:
    # games saved by eval_policy_only / policy-gradient (skips auction_unlock games)
    CUDA_VISIBLE_DEVICES="" uv run python scripts/ruby_replay_check.py \\
        --games-glob 'logs/eval/member_profiles/*/games/*.json' --out logs/ruby_replay/selfplay.json
    # (or --games-list FILE with one path per line)
    # N random walks from our engine (half with multiple_brown_from_ipo)
    ... scripts/ruby_replay_check.py --random 300 --out logs/ruby_replay/random.json

The Ruby containers run with --network none. Game files are inputs only:
nothing here writes player data (states are keyed by seat).
"""

from __future__ import annotations

import argparse
import collections
import glob as globmod
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.ruby_diff import CPUSET, ensure_gems  # noqa: E402

RULE = "multiple_brown_from_ipo"


# ----- inputs ------------------------------------------------------------------


def random_walk_game(seed: int, num_players: int, rules: list, max_moves: int) -> dict:
    """A uniform random walk over the native legal indices (prices uniform in
    the slot's range), exported in 18xx.games format."""
    import engine_rs

    rng = random.Random(seed)
    names = {i: f"Player {i}" for i in range(1, num_players + 1)}
    game = engine_rs.BaseGame(names, optional_rules=rules)
    for _ in range(max_moves):
        if game.finished:
            break
        legal = [int(i) for i in game.factored_legal_indices()]
        if not legal:
            break
        idx = rng.choice(legal)
        pr = game.price_range_for_index(idx)
        price = rng.randint(pr[0], pr[1]) if pr is not None and pr[0] != pr[1] else None
        game.apply_action_index(idx, price)
    actions = []
    for i, action in enumerate(game.raw_actions, start=1):
        action = dict(action)
        action["id"] = i
        actions.append(action)
    return {
        "id": seed,
        "title": "1830",
        "players": [{"id": i, "name": n} for i, n in names.items()],
        "settings": {"optional_rules": rules, "seed": 0},
        "actions": actions,
        "rl18xx": {"kind": "random_walk", "seed": seed, "finished": bool(game.finished)},
    }


def write_random_games(n: int, out_dir: Path, max_moves: int, start_seed: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(n):
        seed = start_seed + k
        players = (4, 4, 4, 2, 3, 5, 6)[k % 7]
        rules = [RULE] if k % 2 else []
        path = out_dir / f"rw_{seed}_{players}p{'_mbfi' if rules else ''}.json"
        if not path.exists():
            path.write_text(json.dumps(random_walk_game(seed, players, rules, max_moves)))
        paths.append(path)
    return paths


# ----- Ruby ----------------------------------------------------------------------

# Preprinted hexes whose two cities our engines number the other way round from
# Ruby (same physical cities; Ruby G19 is `path=a:3,b:_0;path=a:0,b:_1` with
# NYNH's home in city 0, ours `path=a:3,b:_1;path=a:0,b:_0` with it in city 1).
# Until the hex is upgraded, a city reference in our action log is translated
# before Ruby sees it, and our map snapshot lists its cities in Ruby's order.
SWAPPED_PREPRINTED_CITIES = {"G19"}


def _swap_index(ref: str, hex_id: str, sep_count: int) -> str:
    parts = ref.split("-")
    if parts[0] != hex_id or len(parts) != sep_count + 1 or parts[-1] not in ("0", "1"):
        return ref
    parts[-1] = "1" if parts[-1] == "0" else "0"
    return "-".join(parts)


# Our step names -> Ruby's step classes (ruby_replay_dump.rb compares them).
RUBY_STEP = {
    "LayTile": "Track",
    "RunRoutes": "Route",
    "PlaceToken": "Token",
    "BuySellPar": "BuySellParShares",
}


def ruby_input(data: dict) -> dict:
    """Our action log in Ruby's city numbering (see SWAPPED_PREPRINTED_CITIES),
    each action tagged with who acts in our engine before it and in which step
    (`_rs_ent`, `_rs_step` as Ruby's class name), so the Ruby driver can tell
    where Ruby still waits for a pass our engine took implicitly."""
    import engine_rs

    data = json.loads(json.dumps(data))
    rules = (data.get("settings") or {}).get("optional_rules") or []
    game = engine_rs.BaseGame({p["id"]: p["name"] for p in data["players"]}, optional_rules=rules)
    for action in data["actions"]:
        step = game.active_step_type()
        action["_rs_step"] = RUBY_STEP.get(step, step)
        action["_rs_ent"] = _entity_label(game.acting_entity_id() or game.current_entity_id)
        rtype = game.round.round_type
        action["_rs_round"] = f"{rtype}:{game.turn}:{game.round.round_num if rtype == 'Operating' else 0}"
        try:
            game.process_action({k: v for k, v in action.items() if not k.startswith("_")})
        except BaseException:  # noqa: BLE001 -- the comparison reports it
            break
    # Our game ended after its last action: Ruby may still wait for passes.
    data["_rs_finished"] = bool(game.finished)
    preprinted = set(SWAPPED_PREPRINTED_CITIES)
    for action in data["actions"]:
        for hx in list(preprinted):
            if action.get("type") == "place_token" and str(action.get("city", "")).startswith(f"{hx}-0-"):
                action["city"] = _swap_index(action["city"], hx, 2)  # preprinted tile id "G19-0"
            if action.get("type") == "run_routes":
                for route in action.get("routes") or []:
                    route["nodes"] = [_swap_index(n, hx, 1) for n in route.get("nodes") or []]
            if action.get("type") == "lay_tile" and action.get("hex") == hx:
                preprinted.discard(hx)
    return data


def ruby_dump(game_path: Path, out: Path, ruby_repo: Path, gems: Path, cpuset: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpuset-cpus",
            cpuset,
            "-e",
            "ENGINE_LOG_LEVEL=3",
            "-v",
            f"{gems}:/usr/local/bundle",
            "-v",
            f"{ruby_repo.resolve()}:/18xx:ro",
            "-v",
            f"{game_path.resolve().parent}:/games:ro",
            "-v",
            f"{Path(__file__).resolve().parent}:/probe:ro",
            "-v",
            f"{out.parent.resolve()}:/out",
            "ruby:3.2-slim",
            "ruby",
            "/probe/ruby_replay_dump.rb",
            f"/games/{game_path.name}",
            f"/out/{out.name}",
        ],
        check=False,
        capture_output=True,
    )


# ----- Rust ----------------------------------------------------------------------


def _owner_label(owner: str) -> str:
    if owner.startswith("player:"):
        return "p" + owner.split(":", 1)[1]
    if owner == "market":
        return "pool"
    if owner.startswith("ipo:") or owner == "":
        return "ipo"
    return owner.split(":", 1)[-1]


def _entity_label(entity_id: str):
    if not entity_id:
        return None
    kind, _, name = entity_id.partition(":")
    return f"p{name}" if kind == "player" else name


def rust_map(game) -> list:
    """Laid tiles and placed tokens per city, as ruby_replay_dump.rb's map_state."""
    out = []
    for h in game.hexes:
        t = h.tile
        tokens = [sorted(tk.corporation_id for tk in c.tokens if tk is not None) for c in t.cities]
        pre = t.name in ("", h.id) or t.name.startswith("preprinted")
        if pre and h.id in SWAPPED_PREPRINTED_CITIES:
            tokens = tokens[::-1]
        if pre and not any(tokens):
            continue
        name = re.sub(r"-\d+$", "", t.name)  # Rust names laid tiles by instance id ("8-0")
        out.append([h.id, "pre" if pre else name, 0 if pre else t.rotation, tokens])
    return sorted(out, key=lambda x: x[0])


def rust_snap(game, action_id: int) -> dict:
    corps = {}
    for c in game.corporations:
        holders = collections.Counter()
        for s in c.shares:
            holders[_owner_label(s.owner)] += s.percent
        corps[c.sym] = {
            "p": c.share_price.price if c.share_price else None,
            "cash": c.cash,
            "fl": bool(c.floated),
            "tr": sorted(t.name for t in c.trains),
            "sh": {k: v for k, v in holders.items() if v},
        }
    return {
        "id": action_id,
        "round": game.round.round_type,
        "ent": _entity_label(game.acting_entity_id() or game.current_entity_id),
        "phase": game.phase.name,
        "corps": corps,
        "pcash": {str(p.id): p.cash for p in game.players},
        "privates": {co.sym: _entity_label(co.owner) for co in game.companies if not co.closed},
        "finished": bool(game.finished),
        "bank": game.bank.cash,
        "map": rust_map(game),
    }


def cross_player_purchase(game, action: dict):
    """A purchase our 1830 rules forbid on purpose (Ruby allows it): a train
    from a corporation another player presides, or a private from another
    player. Our games must never contain one. Returns a description or None."""
    if action.get("entity_type") != "corporation" or action["type"] not in ("buy_train", "buy_company"):
        return None
    buyer = game.corporation_by_id(action["entity"])
    if buyer is None:
        return None
    if action["type"] == "buy_train":
        for corp in game.corporations:
            if corp.sym != buyer.sym and any(t.id == action.get("train") for t in corp.trains):
                if corp.owner_id_str != buyer.owner_id_str:
                    return f"{buyer.sym} ({buyer.owner_id_str}) buys {action['train']} from {corp.sym} ({corp.owner_id_str})"
        return None
    company = game.company_by_id(action.get("company", ""))
    if company is not None and company.owner.startswith("player:") and company.owner != buyer.owner_id_str:
        return f"{buyer.sym} ({buyer.owner_id_str}) buys {company.sym} from {company.owner}"
    return None


def rust_states(data: dict):
    """(states by action id, rejection or None, forbidden cross-player
    purchases). Seats are the player ids."""
    import engine_rs

    seats = [p["id"] for p in data["players"]]
    assert seats == list(range(1, len(seats) + 1)), "expects players numbered by seat"
    rules = (data.get("settings") or {}).get("optional_rules") or []
    game = engine_rs.BaseGame({p["id"]: p["name"] for p in data["players"]}, optional_rules=rules)
    states = {}
    forbidden = []
    for action in data["actions"]:
        bad = cross_player_purchase(game, action)
        if bad:
            forbidden.append({"id": action["id"], "detail": bad})
        try:
            game.process_action({k: v for k, v in action.items() if k != "auto_actions"})
        except BaseException as exc:  # noqa: BLE001 -- incl. pyo3 panics
            return states, {"id": action["id"], "exception": str(exc)[:300]}, forbidden
        states[action["id"]] = rust_snap(game, action["id"])
    if game.finished:
        states["result"] = {str(k): round(v) for k, v in game.result().items()}
    return states, None, forbidden


# ----- comparison ---------------------------------------------------------------


def diff_states(rs: dict, rb: dict) -> list[str]:
    out = []
    for key in ("round", "phase", "bank", "finished"):
        if rs[key] != rb[key]:
            out.append(f"{key} rs={rs[key]} rb={rb[key]}")
    if not rs["finished"] and rs["ent"] != rb["ent"]:
        out.append(f"ent rs={rs['ent']} rb={rb['ent']}")
    for seat, cash in rs["pcash"].items():
        if rb["pcash"].get(seat) != cash:
            out.append(f"p{seat} cash rs={cash} rb={rb['pcash'].get(seat)}")
    for sym, a in rs["corps"].items():
        b = rb["corps"].get(sym)
        if b is None:
            continue
        for key in ("p", "cash", "fl", "tr", "sh"):
            if a[key] != b[key]:
                out.append(f"{sym} {key} rs={a[key]} rb={b[key]}")
    if rs["privates"] != rb["privates"]:
        out.append(f"privates rs={rs['privates']} rb={rb['privates']}")
    return out


def check_game(path: Path, rb_file: Path) -> dict:
    data = json.loads(path.read_text())
    actions = {a["id"]: a for a in data["actions"]}
    rs_states, rs_exc, forbidden = rust_states(data)
    rb_rows = [json.loads(line) for line in open(rb_file)] if rb_file.exists() else []
    if not rb_rows:
        return {"file": path.name, "status": "ruby_crashed"}
    rb_exc = next((r for r in rb_rows if "exception" in r), None)
    # Implicit passes Ruby needed before this point (see ruby_replay_dump.rb).
    inserted = []
    base = {"file": path.name, "cross_player_purchases": forbidden}
    rb_map = []
    for row in rb_rows:
        if "exception" in row or row.get("final"):
            continue
        inserted += [dict(x, id=row["id"]) for x in row.get("inserted", [])]
        rb_map = row.get("map", rb_map)
        rs = rs_states.get(row["id"])
        if rs is None:
            break
        d = diff_states(rs, row)
        if not d and rs["map"] != rb_map:
            ours = {h[0]: h[1:] for h in rs["map"]}
            theirs = {h[0]: h[1:] for h in rb_map}
            d = [
                f"map {hx} rs={ours.get(hx)} rb={theirs.get(hx)}"
                for hx in sorted(set(ours) | set(theirs))
                if ours.get(hx) != theirs.get(hx)
            ]
        if d:
            return {
                **base,
                "status": "diverges",
                "id": row["id"],
                "action": actions[row["id"]],
                "prev": actions.get(row["id"] - 1),
                "detail": d[:12],
                "ruby_step": row.get("step"),
                "inserted": inserted,
            }
    if rb_exc is not None and rs_exc is not None and rs_exc["id"] == rb_exc["id"]:
        # A game recorded under older rules of ours: both engines now refuse
        # the same action.
        return {
            **base,
            "status": "both_reject",
            "id": rb_exc["id"],
            "action": actions[rb_exc["id"]],
            "prev": actions.get(rb_exc["id"] - 1),
            "detail": [rb_exc["exception"], rs_exc["exception"]],
            "inserted": inserted,
        }
    if rb_exc is not None:
        return {
            **base,
            "status": "ruby_rejects",
            "id": rb_exc["id"],
            "action": actions[rb_exc["id"]],
            "prev": actions.get(rb_exc["id"] - 1),
            "detail": [rb_exc["exception"]],
            "rust_rejects": rs_exc,
            "inserted": inserted,
        }
    if rs_exc is not None:
        return {
            **base,
            "status": "rust_rejects",
            "id": rs_exc["id"],
            "action": actions[rs_exc["id"]],
            "detail": [rs_exc["exception"]],
            "inserted": inserted,
        }
    rb_final = next((r for r in rb_rows if r.get("final")), {})
    if "result" in rs_states and rb_final.get("finished"):
        rb_result = {k: round(v) for k, v in (rb_final.get("result") or {}).items()}
        if rb_result != rs_states["result"]:
            return {
                **base,
                "status": "result_differs",
                "detail": [f"rs={rs_states['result']} rb={rb_result}"],
                "inserted": inserted,
            }
    return {**base, "status": "ok", "actions": len(actions), "inserted": inserted}


def _one_game(job) -> dict:
    """Ruby dump (cached) + comparison for one game, in a worker process."""
    p, work, ruby_repo, gems, cpuset, refresh = job
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{p.parent.parent.name}__{p.stem}")
    out = work / "dumps" / f"{tag}.rb.jsonl"
    try:
        if refresh or not out.exists() or out.stat().st_size == 0:
            inp = work / "inputs" / f"{tag}.json"
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_text(json.dumps(ruby_input(json.loads(p.read_text()))))
            ruby_dump(inp, out, ruby_repo, gems, cpuset)
        res = check_game(p, out)
    except Exception as exc:  # noqa: BLE001
        res = {"file": p.name, "status": "check_error", "detail": [f"{type(exc).__name__}: {exc}"[:300]]}
    res["path"] = str(p)
    return res


def cause_key(res: dict) -> str:
    """A coarse grouping key: status, action type, and the message/fields with
    numbers and names blurred."""
    action = res.get("action") or {}
    if res["status"] in ("ruby_rejects", "both_reject"):
        msg = re.sub(r"\b\d+\b", "N", res["detail"][0])
        msg = re.sub(r"\b[A-Z][A-Z&]+\b", "X", msg)
        return f"{res['status']} {action.get('type')}: {msg[:120]}"
    if res["status"] == "diverges":
        fields = sorted(
            {
                re.sub(r"^\S+ ", "", d.split(" rs=")[0]) if " " in d.split(" rs=")[0] else d.split(" rs=")[0]
                for d in res["detail"]
            }
        )
        return f"diverges after {action.get('type')}: {','.join(fields)}"
    return res["status"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="*", help="game JSON files")
    ap.add_argument("--games-glob", action="append", default=[], help="glob(s) of game JSON files")
    ap.add_argument("--games-list", help="a file listing game JSON paths (one per line)")
    ap.add_argument("--random", type=int, default=0, help="also generate N random-walk games")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--random-max-moves", type=int, default=3000)
    ap.add_argument("--work", default="logs/ruby_replay", help="cache dir (random games, Ruby dumps)")
    ap.add_argument("--ruby-repo", default="logs/ruby/18xx")
    ap.add_argument("--gems", default="logs/ruby/gems")
    ap.add_argument("--cpuset", default=CPUSET, help="docker --cpuset-cpus")
    ap.add_argument("--jobs", type=int, default=16, help="parallel Ruby containers")
    ap.add_argument("--refresh", action="store_true", help="re-run cached Ruby dumps")
    ap.add_argument("--out", help="write per-game results + grouped causes as JSON")
    args = ap.parse_args()

    work = Path(args.work)
    paths = [Path(p) for p in args.games]
    for pattern in args.games_glob:
        paths += [Path(p) for p in sorted(globmod.glob(pattern))]
    if args.games_list:
        paths += [Path(line.strip()) for line in Path(args.games_list).read_text().splitlines() if line.strip()]
    skipped = 0
    kept = []
    for p in paths:
        meta = json.loads(p.read_text()).get("rl18xx") or {}
        if meta.get("auction_unlock"):
            skipped += 1  # the variant isn't in Ruby
            continue
        kept.append(p)
    if args.random:
        kept += write_random_games(args.random, work / "random", args.random_max_moves, args.random_seed)
    if not kept:
        ap.error("no games")
    ruby_repo, gems = Path(args.ruby_repo), Path(args.gems).resolve()
    if not (ruby_repo / "lib" / "engine.rb").exists():
        ap.error(f"no tobymao/18xx clone at {ruby_repo} (see the module docstring)")
    ensure_gems(gems)

    jobs = [(p, work, ruby_repo, gems, args.cpuset, args.refresh) for p in kept]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(_one_game, jobs, chunksize=4))

    causes = collections.defaultdict(list)
    for r in results:
        causes[cause_key(r)].append(r["path"])
    passes = collections.Counter()
    pass_games = collections.Counter()
    for r in results:
        whys = [x["why"] for x in r.get("inserted", [])]
        passes.update(whys)
        pass_games.update(set(whys))
    summary = {
        "games": len(results),
        "skipped_auction_unlock": skipped,
        "status": collections.Counter(r["status"] for r in results).most_common(),
        "causes": sorted(((k, len(v)) for k, v in causes.items()), key=lambda kv: -kv[1]),
        # Where Ruby waits for an explicit pass our engine takes implicitly
        # (e.g. trains or privates only other players' -- our deliberate 1830
        # restriction): expected, counted, not bugs.
        "implicit_passes": {why: {"passes": n, "games": pass_games[why]} for why, n in passes.most_common()},
        # Purchases our rules forbid on purpose: any is a bug in our restriction.
        "cross_player_purchases": sum(len(r.get("cross_player_purchases") or []) for r in results),
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "causes": causes, "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
