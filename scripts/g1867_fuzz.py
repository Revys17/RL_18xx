"""Random-walk fuzzing of the 1867 legal-move generator through the WIRE
path, recording each game as an 18xx.games-style JSON for a Ruby replay.

Every decision: the engine enumerates its legal moves as 18xx.games action
dicts (``g1867_legal_moves_json``), the driver picks one (non-pass moves
favoured, a random price inside a ranged move's range) and sends the dict
back through ``process_action`` — the same path a replay uses. A walk
fails on an exception, a rejected generated move, or an unfinished state
with no moves. Each walk is written to ``--out``/<name>.json (players
"Player N", ids 1..N) so ``scripts/ruby_diff.py --title 1867 <file>`` can
replay it in Ruby HEAD: a Ruby rejection or a state divergence there is a
generator soundness bug (an illegal move our process accepts) or an
engine divergence.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_fuzz.py \\
        --players 3 4 5 6 --seeds 1-10 --moves 4000 --out logs/fuzz/1867
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_rs  # noqa: E402

PRICE_FIELD = {"bid": "price", "buy_train": "price", "buy_company": "price"}


def pick(moves: list[dict], rng: random.Random) -> dict:
    non_pass = [m for m in moves if m["wire"]["type"] != "pass"]
    m = rng.choice(non_pass) if non_pass and rng.random() < 0.75 else rng.choice(moves)
    wire = dict(m["wire"])
    if m["range"]:
        lo, hi, step = m["range"]
        k = (hi - lo) // max(step, 1)
        r = rng.random()
        price = lo if r < 0.4 else lo + k * step if r < 0.5 else lo + rng.randint(0, min(k, 40)) * step
        wire[PRICE_FIELD[wire["type"]]] = price
    return wire


def walk(players: int, seed: int, max_moves: int, policy: str | None = None) -> dict:
    ids = {i + 1: f"Player {i + 1}" for i in range(players)}
    g = engine_rs.BaseGame.new_titled("1867", ids, None, seed)
    rng = random.Random(seed * 1_000_003 + players)
    actions: list[dict] = []
    types: Counter = Counter()
    gen_time: Counter = Counter()
    gen_max: Counter = Counter()
    error = None
    for n in range(max_moves):
        if g.finished:
            break
        t0 = time.perf_counter()
        try:
            moves = json.loads(g.g1867_legal_moves_json(policy))
        except BaseException as exc:  # noqa: BLE001 — incl. pyo3 panics
            error = f"move {n}: generator raised {exc!r}"
            break
        dt = time.perf_counter() - t0
        if moves:
            key = moves[0]["step"]
            gen_time[key] += dt
            gen_max[key] = max(gen_max[key], dt)
        if not moves:
            error = f"move {n}: no legal moves (offers {g.g1867_step_offers()})"
            break
        wire = pick(moves, rng)
        wire["id"] = len(actions) + 1
        types[wire["type"]] += 1
        try:
            g.process_action(dict(wire))
        except BaseException as exc:  # noqa: BLE001
            error = f"move {n}: generated move rejected: {json.dumps(wire)[:300]}: {exc}"
            break
        actions.append(wire)
    result = {str(pid): v for pid, v in g.result().items()}
    # Routes go to Ruby WITHOUT revenue: Ruby then validates and prices
    # every route itself (a preset revenue skips Route#revenue's checks),
    # and our replay of the file prices them with the same rules.
    recorded = json.loads(json.dumps(actions))
    for a in recorded:
        for r in a.get("routes") or []:
            r.pop("revenue", None)
    return {
        "game": {
            "id": f"fuzz{players}p{seed}",
            "title": "1867",
            "status": "finished" if g.finished else "active",
            "settings": {"seed": seed, "optional_rules": []},
            "players": [{"id": pid, "name": name} for pid, name in ids.items()],
            "result": result,
            "actions": recorded,
        },
        "error": error,
        "finished": bool(g.finished),
        "moves": len(actions),
        "types": types,
        "gen_time": gen_time,
        "gen_max": gen_max,
    }


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--seeds", default="1-4")
    ap.add_argument("--moves", type=int, default=4000)
    ap.add_argument("--out", default=None, help="directory for the recorded games")
    ap.add_argument(
        "--policy",
        choices=["title", "rules", "restricted"],
        default="title",
        help="cross-player purchase policy: the title's (default), the rules' (all Ruby allows), or none",
    )
    ap.add_argument("--timing", action="store_true", help="print generator time by step kind per walk")
    args = ap.parse_args()
    policy = None if args.policy == "title" else args.policy
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    failures = 0
    total_types: Counter = Counter()
    for players in args.players:
        for seed in parse_seeds(args.seeds):
            t0 = time.perf_counter()
            rep = walk(players, seed, args.moves, policy)
            dt = time.perf_counter() - t0
            name = f"fuzz_{players}p_s{seed}{'_' + args.policy if args.policy != 'title' else ''}"
            if out:
                (out / f"{name}.json").write_text(json.dumps(rep["game"]))
            status = "ERROR " + rep["error"] if rep["error"] else "finished" if rep["finished"] else "cut"
            print(f"{name}: {rep['moves']} moves in {dt:.1f}s, {status}", flush=True)
            failures += bool(rep["error"])
            total_types.update(rep["types"])
            if args.timing:
                slow = sorted(rep["gen_time"].items(), key=lambda kv: -kv[1])[:4]
                print("    generator time by step: " + ", ".join(
                    f"{k} {v:.1f}s (max {rep['gen_max'][k]:.2f}s)" for k, v in slow))
    print(f"{failures} failing walks; action types {dict(total_types)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
