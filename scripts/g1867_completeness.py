"""Corpus completeness for the 1867 legal-move generator (RL-readiness
contract, item 4).

Replays recorded games through the Rust engine; before every HUMAN
decision (server auto-actions are skipped) it asks the generator for the
legal moves and checks that the recorded action matches EXACTLY ONE of
them. A move matches on its type, entity and the parameters that change
the outcome (company / corporation / bundle holder and size / hex, tile
and rotation / city / train type and seller / choice / kind), a ranged
parameter (bid, company price, price for another corporation's train) by
range membership. ``run_routes`` is not matched here (routes are the router
check's business). Matching uses the rules' move set (policy "rules"); an
exact match the title's cross-player policy excludes (a train / private
bought from another player, a merger round add of another player's minor)
is reported as "policy". Reports exact /
missing / ambiguous / policy by action family with example game ids and
file action ids (no player data).

Corpus: finished 3-6 player games without optional rules (the generator's
scope); games the replay itself rejects are counted up to the rejection.

``--index`` checks the RL layout ``1867-v1`` instead (docs/rl_descriptors.md
§8): every human decision (``run_routes`` included, matched by type) against
the state's indexed moves (``rl_legal_moves_json("rules")``) must match
exactly one index; for that index ``rl_decode_index`` must give the human
action back and ``rl_apply_index`` must leave the state the human action
leaves (``--no-apply-check`` skips the latter). Rows by layout block; it
also prints the human price census of the price-bearing blocks
(``--census-rows`` keeps every priced decision).

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_completeness.py \\
        --glob 'human_games/scrape_2026_10/1867/*.json' --jobs 8 --json /tmp/c.json [--index]
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_rs  # noqa: E402

from tests.title_replay_harness import game_seed, ruby_filtered_actions  # noqa: E402

# The matching lives in the library (the 1867 importer of the generic supervised data uses the same):
# rl18xx/agent/alphazero/generic_import_1867.py.
from rl18xx.agent.alphazero.generic_import_1867 import POLICY_TYPES, matching, replay_only_reason  # noqa: E402
from rl18xx.agent.alphazero.generic_import_1867 import action_key as _key  # noqa: E402

SKIP = {"run_routes"}


def matches(g, human: dict, moves: list[dict]) -> int:
    return len(matching(g, human, moves))


def audit_game(path: str) -> dict:
    game = json.load(open(path))
    gid = Path(path).stem
    out = {"id": gid, "counts": Counter(), "examples": defaultdict(list), "error": None}
    if game.get("title") != "1867" or game.get("status") not in ("finished", "archived"):
        out["error"] = "skipped"
        return out
    if (game.get("settings") or {}).get("optional_rules") or not 3 <= len(game["players"]) <= 6:
        out["error"] = "out of scope"
        return out
    seat = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    g = engine_rs.BaseGame.new_titled("1867", {v: f"Player {v}" for v in seat.values()}, None, game_seed(game))
    raw = json.loads(json.dumps(game["actions"]))
    for a in raw:
        a["_fid"] = a.get("id")
        for au in a.get("auto_actions") or []:
            au["_fid"] = a.get("id")
    for a in ruby_filtered_actions(raw, mark_auto=True):
        fid = a.pop("_fid", None)
        auto = a.pop("_auto", False)
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        if not auto and a["type"] not in SKIP and not g.finished:
            try:
                moves = json.loads(g.g1867_legal_moves_json("rules"))
                n = matches(g, a, moves)
                if n == 1 and a["type"] in POLICY_TYPES and not matches(g, a, json.loads(g.g1867_legal_moves_json())):
                    n = -1  # legal, but the title's cross-player policy excludes it
            except BaseException as exc:  # noqa: BLE001
                out["error"] = f"generator raised at file id {fid}: {exc!r}"[:300]
                return out
            kind = "exact" if n == 1 else "missing" if n == 0 else "policy" if n == -1 else "ambiguous"
            if kind == "missing":
                kind = replay_only_reason(g, a) or kind
            out["counts"][(a["type"], kind)] += 1
            if kind != "exact" and len(out["examples"][(a["type"], kind)]) < 3:
                out["examples"][(a["type"], kind)].append(fid)
        try:
            g.process_action(a)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = f"replay rejected file id {fid}: {exc}"[:300]
            return out
    return out


# ---------------------------------------------------------------------------
# Index level (--index): the RL layout 1867-v1 (docs/rl_descriptors.md)
# ---------------------------------------------------------------------------

# 1867 train face prices (title data, g1867.rs trains()).
TRAIN_FACE = {"2": 100, "3": 225, "4": 350, "5": 550, "6": 650, "7": 800, "8": 1000, "2+2": 600, "5+5E": 1500}


def _snap(g) -> tuple:
    """The rules state, for "the index applies as the human action did".
    Interchangeable objects count by kind: a tile by its name (the
    instance "8-1" or "8-2" is the same tile), a corporation's certificates
    as a multiset of (owner, percent, president)."""
    corps = tuple(
        (
            c.sym,
            c.cash,
            c.share_price.price if c.share_price else None,
            c.floated,
            tuple(sorted(t.name for t in c.trains)),
            tuple(sorted((str(s.owner), s.percent, s.president) for s in c.shares)),
        )
        for c in g.corporations
    )
    tiles = tuple(
        (
            h.id,
            h.tile.name.rsplit("-", 1)[0] if h.tile.name.rsplit("-", 1)[-1].isdigit() else h.tile.name,
            h.tile.rotation,
            tuple(tuple(t.corporation_id if t else None for t in c.tokens) for c in h.tile.cities),
        )
        for h in g.hexes
    )
    return (
        g.debug_state(),
        tuple(p.cash for p in g.players),
        corps,
        tuple((co.sym, str(co.owner), co.closed) for co in g.companies),
        tiles,
        tuple(sorted(t.name for t in g.depot.trains)),
        tuple(sorted(t.name for t in g.depot.discarded)),
        g.bank.cash,
        bool(g.finished),
    )


def _census_row(g, a: dict, m: dict) -> dict:
    """One human price decision of an index with a price range."""
    lo, hi, step = m["range"]
    row = {"block": m["block"], "price": a.get("price"), "lo": lo, "hi": hi, "step": step}
    w = m["wire"]
    if m["block"] in ("Bid.company", "BuyCompany"):
        co = next((c for c in g.companies if c.sym == w.get("company")), None)
        row["face"] = co.value if co else None
    elif m["block"] == "BuyTrain.corp":
        name, _owner = g.g1867_train_info(a.get("train", ""))
        row["train"] = name
        row["face"] = TRAIN_FACE.get(name)
    elif m["block"] == "Bid.corporation":
        row["opening"] = lo == 100
    return row


def audit_game_index(path: str, apply_check: bool = True) -> dict:
    """Every human decision against the state's indexed legal moves
    (``rl_legal_moves_json("rules")``): exactly one matching move = exactly
    one index (the layout is injective per state, which the API checks).
    For an exact match, ``rl_decode_index`` must give the human action back
    (key and price) and, with ``apply_check``, ``rl_apply_index`` must leave
    the state the human action leaves. ``run_routes`` matches the RunRoutes
    index by type (the route itself is the router check's business)."""
    game = json.load(open(path))
    gid = Path(path).stem
    out = {"id": gid, "counts": Counter(), "examples": defaultdict(list), "error": None, "census": [], "pars": []}
    if game.get("title") != "1867" or game.get("status") not in ("finished", "archived"):
        out["error"] = "skipped"
        return out
    if (game.get("settings") or {}).get("optional_rules") or not 3 <= len(game["players"]) <= 6:
        out["error"] = "out of scope"
        return out
    out["players"] = len(game["players"])
    seat = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    g = engine_rs.BaseGame.new_titled("1867", {v: f"Player {v}" for v in seat.values()}, None, game_seed(game))
    raw = json.loads(json.dumps(game["actions"]))
    for a in raw:
        a["_fid"] = a.get("id")
        for au in a.get("auto_actions") or []:
            au["_fid"] = a.get("id")
    for a in ruby_filtered_actions(raw, mark_auto=True):
        fid = a.pop("_fid", None)
        auto = a.pop("_auto", False)
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        if not auto and not g.finished:
            try:
                moves = json.loads(g.rl_legal_moves_json("rules"))
                if a["type"] in SKIP:
                    hits = [
                        m
                        for m in moves
                        if m["wire"]["type"] == a["type"] and str(m["wire"].get("entity")) == str(a.get("entity"))
                    ]
                else:
                    hits = matching(g, a, moves)
                if len(hits) == 1:
                    m = hits[0]
                    idx, block = m["index"], m["block"]
                    kind = "exact"
                    price = a.get("price") if m["range"] else None
                    if a["type"] in POLICY_TYPES and idx not in set(g.rl_legal_indices()):
                        kind = "policy"  # legal, but the title's cross-player policy excludes it
                    if a["type"] not in SKIP:
                        w = json.loads(g.rl_decode_index(idx, price, "rules"))
                        if _key(g, w) != _key(g, a) or (price is not None and w.get("price") != price):
                            kind = "decode_mismatch"
                        elif apply_check:
                            by_human = g.pickle_clone()
                            by_human.process_action(dict(a))
                            by_index = g.pickle_clone()
                            by_index.rl_apply_index(idx, price, "rules")
                            if _snap(by_human) != _snap(by_index):
                                kind = "apply_mismatch"
                    if m["range"]:
                        out["census"].append(_census_row(g, a, m))
                    if block == "Par":
                        out["pars"].append(_key(g, a)[1])
                elif not hits:
                    block = f"type:{a['type']}"
                    kind = replay_only_reason(g, a) or "missing"
                else:
                    block = "+".join(sorted({h["block"] for h in hits}))
                    kind = "ambiguous"
            except BaseException as exc:  # noqa: BLE001
                out["error"] = f"index path raised at file id {fid}: {exc!r}"[:300]
                return out
            out["counts"][(block, kind)] += 1
            if kind != "exact" and len(out["examples"][(block, kind)]) < 3:
                out["examples"][(block, kind)].append(fid)
        try:
            g.process_action(a)
        except BaseException as exc:  # noqa: BLE001
            out["error"] = f"replay rejected file id {fid}: {exc}"[:300]
            return out
    return out


def _quantiles(xs: list, qs=(0.5, 0.75, 0.9, 0.95, 0.99)) -> list:
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else []


def summarize_census(rows: list[dict], pars: list) -> dict:
    """Where human prices fall in their legal ranges, per priced block, and
    how 1830's price cells (price_pmf "Bid" / "BuyTrain" / "BuyCompany",
    the step from the move) resolve them."""
    from rl18xx.agent.alphazero.price_pmf import price_cells

    cell_type = {
        "Bid.company": "Bid",
        "Bid.corporation": "Bid",
        "BuyTrain.corp": "BuyTrain",
        "BuyCompany": "BuyCompany",
    }
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["block"]].append(r)
        if r["block"] == "Bid.corporation":
            groups["Bid.corporation:" + ("opening" if r.get("opening") else "raise")].append(r)
    for block, rs in sorted(groups.items()):
        n = len(rs)
        steps = [(r["price"] - r["lo"]) // max(r["step"], 1) for r in rs]
        widths = [(r["hi"] - r["lo"]) // max(r["step"], 1) for r in rs]
        s = {
            "n": n,
            "at_lo": sum(r["price"] == r["lo"] for r in rs) / n,
            "at_hi": sum(r["price"] == r["hi"] for r in rs) / n,
            "degenerate_range": sum(r["lo"] == r["hi"] for r in rs) / n,
            "steps_above_lo_q50_75_90_95_99": _quantiles(steps),
            "range_steps_q50_90": _quantiles(widths, (0.5, 0.9)),
            "price_q50_90_99": _quantiles([r["price"] for r in rs], (0.5, 0.9, 0.99)),
        }
        if any("face" in r for r in rs):
            faced = [r for r in rs if r.get("face")]
            s["at_face"] = sum(r["price"] == r["face"] for r in faced) / max(len(faced), 1)
            s["face_in_range"] = sum(r["lo"] <= r["face"] <= r["hi"] for r in faced) / max(len(faced), 1)
            s["price_over_face_q10_50_90"] = _quantiles([r["price"] / r["face"] for r in faced], (0.1, 0.5, 0.9))
        if block == "BuyTrain.corp":
            s["at_hi_minus_1"] = sum(r["price"] == r["hi"] - 1 for r in rs) / n
            s["by_train"] = dict(Counter(r.get("train") for r in rs).most_common())
        if block.startswith("Bid"):
            s["beyond_20_step_ladder"] = sum(st >= 20 for st in steps) / n
        ct = cell_type.get(block.split(":")[0])
        if ct:
            on_atom, bin_sizes = 0, []
            for r in rs:
                cells = price_cells(ct, r["lo"], r["hi"], bid_step=r["step"])
                c = cells.cell_of(r["price"])
                if c < cells.n_atoms:
                    on_atom += 1
                else:
                    bin_sizes.append(int(cells.counts[c]))
            s["cells_1830"] = {
                "type": ct,
                "on_atom": on_atom / n,
                "bin_size_q50_90": _quantiles(bin_sizes, (0.5, 0.9)),
            }
        out[block] = s
    out["Par"] = {"n": len(pars), "by_price": dict(sorted(Counter(pars).items()))}
    return out


def main_index(args, paths: list[str]) -> int:
    from functools import partial

    fn = partial(audit_game_index, apply_check=not args.no_apply_check)
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            results = pool.map(fn, paths, chunksize=2)
    else:
        results = [fn(p) for p in paths]
    totals: Counter = Counter()
    examples: dict = defaultdict(list)
    errors = Counter()
    census, pars = [], []
    by_players = Counter()
    for r in results:
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1
        if r.get("players"):
            by_players[r["players"]] += 1
        for k, v in r["counts"].items():
            totals[k] += v
        for k, ids in r["examples"].items():
            if len(examples[k]) < 6:
                examples[k].extend(f"{r['id']}@{i}" for i in ids[: 6 - len(examples[k])])
        census.extend(r["census"])
        pars.extend(r["pars"])
    kinds = ["exact", "policy", "missing", "ambiguous", "decode_mismatch", "apply_mismatch"]
    print(f"{len(paths)} games ({dict(sorted(by_players.items()))} in scope by players); errors: {dict(errors)}")
    print(f"{'block':28} " + " ".join(f"{k[:9]:>9}" for k in kinds) + "  replay-only")
    blocks = sorted({b for b, _ in totals})
    for b in blocks:
        ro = sum(v for (bb, k), v in totals.items() if bb == b and k not in kinds)
        ex = [e for (bb, k), e in examples.items() if bb == b and k != "exact"]
        print(
            f"{b:28} " + " ".join(f"{totals[(b, k)]:9}" for k in kinds) + f"  {ro:6}  {[x for e in ex for x in e][:4]}"
        )
    n = sum(totals.values())
    one = sum(v for (b, k), v in totals.items() if k in ("exact", "policy"))
    ro = sum(v for (b, k), v in totals.items() if k not in kinds)
    print(f"exactly one index (rules' moves): {one}/{n} = {one / max(n, 1):.4%}; replay-only classes {ro}")
    for k in kinds[1:]:
        print(f"  {k}: {sum(v for (b, kk), v in totals.items() if kk == k)}")
    summary = summarize_census(census, pars)
    print("price census:")
    print(json.dumps(summary, indent=1))
    if args.json_out:
        json.dump(
            {
                "games": len(paths),
                "by_players": dict(by_players),
                "errors": dict(errors),
                "totals": {f"{b}|{k}": v for (b, k), v in totals.items()},
                "examples": {f"{b}|{k}": v for (b, k), v in examples.items()},
                "census": summary,
            },
            open(args.json_out, "w"),
            indent=1,
        )
    if args.census_rows:
        with open(args.census_rows, "w") as f:
            for r in census:
                f.write(json.dumps(r) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="human_games/scrape_2026_10/1867/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument(
        "--index",
        action="store_true",
        help="index level: every human decision against the RL layout 1867-v1 (exactly one index, decode and "
        "apply checks), plus the human price census of the price-bearing indices",
    )
    ap.add_argument("--no-apply-check", action="store_true", help="--index: skip the apply-by-index state check")
    ap.add_argument("--census-rows", default=None, help="--index: write every priced human decision (JSONL)")
    args = ap.parse_args()
    paths = sorted(globmod.glob(args.glob))[: args.limit]
    if args.index:
        return main_index(args, paths)
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            results = pool.map(audit_game, paths, chunksize=2)
    else:
        results = [audit_game(p) for p in paths]
    totals: Counter = Counter()
    examples: dict = defaultdict(list)
    errors = Counter()
    for r in results:
        if r["error"]:
            errors[r["error"].split(":")[0]] += 1
        for k, v in r["counts"].items():
            totals[k] += v
        for k, ids in r["examples"].items():
            if len(examples[k]) < 6:
                examples[k].extend(f"{r['id']}@{i}" for i in ids[: 6 - len(examples[k])])
    families = sorted({t for t, _ in totals})
    print(f"{len(paths)} games; errors: {dict(errors)}")
    ro = Counter()
    for (t, k), v in totals.items():
        if k not in ("exact", "missing", "ambiguous", "policy"):
            ro[k] += v
    print(f"replay-only classes among the misses: {dict(ro)}")
    print(f"{'family':16} {'exact':>8} {'missing':>8} {'ambig':>6} {'policy':>6}")
    for f in families:
        e, m, a = totals[(f, "exact")], totals[(f, "missing")], totals[(f, "ambiguous")]
        p = totals[(f, "policy")]
        print(
            f"{f:16} {e:8} {m:8} {a:6} {p:6}  {examples.get((f, 'missing'), [])[:4]} "
            f"{examples.get((f, 'ambiguous'), [])[:2]}"
        )
    all_ro = sum(ro.values())
    all_e = sum(v for (t, k), v in totals.items() if k in ("exact", "policy"))
    all_p = sum(v for (t, k), v in totals.items() if k == "policy")
    all_n = sum(totals.values())
    print(f"overall exact (rules) {all_e}/{all_n} = {all_e / max(all_n, 1):.4%}")
    print(f"excluded by the cross-player policy: {all_p}/{all_n} = {all_p / max(all_n, 1):.4%}")
    print(f"replay-only (documented classes): {all_ro}/{all_n} = {all_ro / max(all_n, 1):.4%}")
    if args.json_out:
        json.dump(
            {
                "games": len(paths),
                "errors": dict(errors),
                "totals": {f"{t}|{k}": v for (t, k), v in totals.items()},
                "examples": {f"{t}|{k}": v for (t, k), v in examples.items()},
            },
            open(args.json_out, "w"),
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
