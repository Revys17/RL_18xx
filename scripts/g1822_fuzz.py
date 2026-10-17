"""Random-walk fuzzing of the 1822 legal-move generator through the WIRE
path, recording each game as an 18xx.games-style JSON for a Ruby replay.

Every decision: the engine enumerates its legal moves as 18xx.games action
dicts (``g1822_legal_moves_json``), the driver picks one (non-pass moves
favoured, a random price inside a ranged move's range) and sends the dict
back through ``process_action`` — the same path a replay uses. A walk
fails on an exception, a rejected generated move, or an unfinished state
with no moves. Each walk is written to ``--out``/<name>.json (players
"Player N", ids 1001+, seed and optional rules in settings) so
``scripts/ruby_diff.py --title 1822 <file>`` can replay it in Ruby HEAD:
a Ruby rejection or a state divergence there is a generator soundness bug
(an illegal move our process accepts) or an engine divergence.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_fuzz.py \\
        --players 3 4 5 6 --seeds 1-10 --moves 3000 --out logs/fuzz/1822
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_rs  # noqa: E402

PRICE_FIELD = {"bid": "price", "buy_train": "price", "payoff_player_debt_partial": "amount"}


def pick(moves: list[dict], rng: random.Random) -> dict:
    non_pass = [m for m in moves if m["wire"]["type"] != "pass"]
    m = rng.choice(non_pass) if non_pass and rng.random() < 0.75 else rng.choice(moves)
    wire = dict(m["wire"])
    if m["range"]:
        lo, hi, step = m["range"]
        k = (hi - lo) // max(step, 1)
        r = rng.random()
        price = lo if r < 0.25 else lo + k * step if r < 0.5 else lo + rng.randint(0, k) * step
        wire[PRICE_FIELD[wire["type"]]] = price
    return wire


def walk(players: int, seed: int, rules: list[str], max_moves: int, policy: str | None = None) -> dict:
    ids = {1000 + i + 1: f"Player {i + 1}" for i in range(players)}
    g = engine_rs.BaseGame.new_titled("1822", ids, rules or None, seed)
    rng = random.Random(seed * 1_000_003 + players)
    actions: list[dict] = []
    error = None
    dead_end = False
    for n in range(max_moves):
        if g.finished:
            break
        try:
            moves = json.loads(g.g1822_legal_moves_json(policy))
        except BaseException as exc:  # noqa: BLE001 — incl. pyo3 panics
            error = f"move {n}: generator raised {exc!r}"
            break
        if not moves:
            if g.g1822_ruby_dead_end():
                dead_end = True  # Ruby cannot continue either
            else:
                error = f"move {n}: no legal moves ({g.g1822_snapshot()[:300]})"
            break
        wire = pick(moves, rng)
        wire["id"] = len(actions) + 1
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
            "title": "1822",
            "status": "finished" if g.finished else "active",
            "settings": {"seed": seed, "optional_rules": rules},
            "players": [{"id": pid, "name": name} for pid, name in ids.items()],
            "result": result,
            "actions": recorded,
        },
        "error": error,
        "dead_end": dead_end,
        "finished": bool(g.finished),
        "moves": len(actions),
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
    ap.add_argument("--moves", type=int, default=3000)
    ap.add_argument("--rules", choices=["base", "plus", "both"], default="both")
    ap.add_argument("--out", default=None, help="directory for the recorded games")
    ap.add_argument(
        "--policy",
        choices=["title", "rules", "restricted"],
        default="title",
        help="cross-player purchase policy: the title's (default), the rules' (all Ruby allows), or none",
    )
    args = ap.parse_args()
    policy = None if args.policy == "title" else args.policy
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for players in args.players:
        for seed in parse_seeds(args.seeds):
            variants = {"base": [[]], "plus": [["plus_expansion"]], "both": [[], ["plus_expansion"]]}[args.rules]
            for rules in variants:
                rep = walk(players, seed, rules, args.moves, policy)
                name = f"fuzz_{players}p_s{seed}{'_plus' if rules else ''}"
                if out:
                    (out / f"{name}.json").write_text(json.dumps(rep["game"]))
                status = (
                    "ERROR " + rep["error"]
                    if rep["error"]
                    else "finished" if rep["finished"] else "ruby dead end" if rep["dead_end"] else "cut"
                )
                print(f"{name}: {rep['moves']} moves, {status}")
                failures += bool(rep["error"])
    print(f"{failures} failing walks")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
