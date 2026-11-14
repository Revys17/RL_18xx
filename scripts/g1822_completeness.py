"""Corpus completeness for the 1822 legal-move generator (RL-readiness
contract, item 4).

Replays recorded games through the Rust engine; before every HUMAN
decision (server auto-actions are skipped) it asks the generator for the
legal moves and checks that the recorded action matches EXACTLY ONE of
them. A move matches on its type, entity and the parameters that change
the outcome (company / corporation / bundle holder and size / hex, tile
and rotation / city / train type and seller / choice / kind), a ranged
parameter (bid, price for another corporation's train, partial payoff)
by range membership. ``run_routes`` is not matched here (routes are the
router check's business). Matching uses the rules' move set (policy
"rules"); an exact match the title's cross-player policy excludes (a
train / private / minor bought from another player) is reported as
"policy". Reports exact / missing / ambiguous / policy by action family
with example game ids (no player data).

``--index`` checks the title's RL layout (``<title>-v1``,
docs/rl_descriptors.md §9) instead, on the games of the player counts the
pipeline trains (3-6; PNW and MX 3-5): every human decision (``run_routes``
included, matched by type) against the state's indexed moves
(``rl_legal_moves_json("rules")``) must match exactly one index; for that
index ``rl_decode_index`` must give the human action back (key and price)
and ``rl_apply_index`` must leave the state the human action leaves
(``--no-apply-check`` skips that). Rows by layout block; it also prints the
human price census of the price-bearing indices (``--census-rows`` keeps
every priced decision) and the generator's time per decision
(``rl_legal_indices()`` under the title's policy: p50 / p90 / p99 / mean).

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_completeness.py \\
        --glob 'human_games/scrape_2026_10/1822/*.json' --jobs 8 --json /tmp/c.json [--index]
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_rs  # noqa: E402

from rl18xx.agent.alphazero import generic_import_1822 as gi22  # noqa: E402
from tests.title_replay_harness import SEAT_ID_BASE, game_seed, ruby_filtered_actions  # noqa: E402

TITLE = "1822"
SKIP = {"run_routes"}
# The player counts the RL pipeline trains each title at (titles.py): --index checks these games only.
PLAYERS = {"1822": (3, 6), "1822PNW": (3, 5), "1822MX": (3, 5), "1822CA": (3, 6)}
# Action types the cross-player policy (GameTitle::generator_cross_player) can exclude.
POLICY_TYPES = gi22.POLICY_TYPES


# The matching lives in the importer of the supervised data (generic_import_1822), so its labels are these.
_in_range = gi22._in_range
_key = gi22.action_key  # the outcome-relevant parameters of a wire action (state BEFORE it)
_price = gi22.action_price
matches = gi22.matches  # how many generated moves the human action matches


def audit_game(path: str) -> dict:
    game = json.load(open(path))
    gid = Path(path).stem
    out = {"id": gid, "counts": Counter(), "examples": defaultdict(list), "error": None}
    if game.get("title") != TITLE or game.get("status") not in ("finished", "archived"):
        out["error"] = "skipped"
        return out
    base = SEAT_ID_BASE.get(TITLE, 0)
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
    for a in ruby_filtered_actions(raw, mark_auto=True):
        fid = a.pop("_fid", None)
        auto = a.pop("_auto", False)
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        if not auto and a["type"] not in SKIP and not g.finished:
            try:
                moves = json.loads(g.g1822_legal_moves_json("rules"))
                n = matches(g, a, moves)
                if n == 1 and a["type"] in POLICY_TYPES and not matches(g, a, json.loads(g.g1822_legal_moves_json())):
                    n = -1  # legal, but the title's cross-player policy excludes it
            except BaseException as exc:  # noqa: BLE001
                out["error"] = f"generator raised at file id {fid}: {exc!r}"[:300]
                return out
            kind = "exact" if n == 1 else "missing" if n == 0 else "policy" if n == -1 else "ambiguous"
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
# Index level (--index): the title's RL layout (docs/rl_descriptors.md §9)
# ---------------------------------------------------------------------------


matching = gi22.matching  # the indexed legal moves the human action matches (as matches counts them)


def _snapshot(g) -> str:
    """``g1822_snapshot`` with every corporation's trains as a multiset: discarding one of two
    trains of a type (the generator offers the first, a human may pick the other) leaves the same
    trains in another order."""
    snap = json.loads(g.g1822_snapshot())
    for c in (snap.get("corps") or {}).values():
        if isinstance(c, dict) and isinstance(c.get("tr"), list):
            c["tr"] = sorted(c["tr"])
    return json.dumps(snap, sort_keys=True)


def _snap(g) -> tuple:
    """The rules state, for "the index applies as the human action did". Interchangeable objects
    count by kind: a tile by its name, a corporation's certificates as a multiset of (owner,
    percent, president), its trains as a multiset of names."""
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
        _snapshot(g),
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
    row = {"block": m["block"], "price": _price(a), "lo": lo, "hi": hi, "step": step}
    w = m["wire"]
    if m["block"] == "Bid.company":
        co = next((c for c in g.companies if c.sym == w.get("company")), None)
        row["kind"] = (w.get("company") or "?")[:1]
        row["face"] = co.value if co else None
    elif m["block"] == "BuyTrain.corp":
        name, _owner = g.g1822_train_info(a.get("train", ""))
        row["train"] = name
    return row


def audit_game_index(path: str, apply_check: bool = True) -> dict:
    """Every human decision against the state's indexed legal moves: exactly one matching move =
    exactly one index (the layout is injective per state, which the API checks). For an exact
    match, ``rl_decode_index`` must give the human action back (key and price) and, with
    ``apply_check``, ``rl_apply_index`` must leave the state the human action leaves.
    ``run_routes`` matches the RunRoutes index by type (the route itself is the router check's
    business)."""
    game = json.load(open(path))
    gid = Path(path).stem
    out = {"id": gid, "counts": Counter(), "examples": defaultdict(list), "error": None, "census": [], "times": []}
    if game.get("title") != TITLE or game.get("status") not in ("finished", "archived"):
        out["error"] = "skipped"
        return out
    lo_p, hi_p = PLAYERS.get(TITLE, (3, 6))
    if not lo_p <= len(game["players"]) <= hi_p:
        out["error"] = "out of scope (players)"
        return out
    out["players"] = len(game["players"])
    base = SEAT_ID_BASE.get(TITLE, 0)
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
    for a in ruby_filtered_actions(raw, mark_auto=True):
        fid = a.pop("_fid", None)
        auto = a.pop("_auto", False)
        if a.get("entity_type") == "player":
            a["entity"] = a["user"] = seat.get(a["entity"], a["entity"])
        if not auto and not g.finished:
            try:
                t0 = time.perf_counter()
                titled = set(g.rl_legal_indices())
                out["times"].append(time.perf_counter() - t0)
                rows = json.loads(g.rl_legal_moves_json("rules"))
                if a["type"] in SKIP:
                    hits = [
                        m
                        for m in rows
                        if m["wire"]["type"] == a["type"] and str(m["wire"].get("entity")) == str(a.get("entity"))
                    ]
                else:
                    hits = matching(g, a, rows)
                if len(hits) == 1:
                    m = hits[0]
                    idx, block = m["index"], m["block"]
                    kind = "exact"
                    price = _price(a) if m["range"] else None
                    if idx not in titled:
                        kind = "policy"  # legal, but the title's cross-player policy excludes it
                    if a["type"] not in SKIP:
                        w = json.loads(g.rl_decode_index(idx, price, "rules"))
                        if _key(g, w) != _key(g, a) or (price is not None and _price(w) != price):
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
                    block = block.split(":")[0] + (":<company>" if ":" in block else "")
                elif not hits:
                    block = f"type:{a['type']}"
                    kind = "missing"
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


def summarize_census(rows: list[dict]) -> dict:
    """Where human prices fall in their legal ranges, per priced block (bids also per bid-box kind:
    M minors, C concessions, P privates), and how ``price_pmf``'s cell layouts, taken in step units
    (docs/rl_descriptors.md §6), resolve them: ``bits`` = E[log2 |cell of the human price|] (0 =
    every human price is a cell of its own), ``on_atom`` = the share on an atom."""
    from rl18xx.agent.alphazero.price_pmf import price_cells

    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["block"]].append(r)
        if r["block"] == "Bid.company":
            groups[f"Bid.company:{r.get('kind')}"].append(r)
    for block, rs in sorted(groups.items()):
        n = len(rs)
        steps = [(r["price"] - r["lo"]) // max(r["step"], 1) for r in rs]
        widths = [(r["hi"] - r["lo"]) // max(r["step"], 1) for r in rs]
        s = {
            "n": n,
            "at_lo": round(sum(r["price"] == r["lo"] for r in rs) / n, 4),
            "at_hi": round(sum(r["price"] == r["hi"] for r in rs) / n, 4),
            "at_hi_minus_1_step": round(sum(r["price"] == r["hi"] - r["step"] for r in rs) / n, 4),
            "degenerate_range": round(sum(r["lo"] == r["hi"] for r in rs) / n, 4),
            "steps_above_lo_q50_75_90_95_99": _quantiles(steps),
            "range_steps_q50_90": _quantiles(widths, (0.5, 0.9)),
            "price_q50_90_99": _quantiles([r["price"] for r in rs], (0.5, 0.9, 0.99)),
            "beyond_20_steps": round(sum(st >= 20 for st in steps) / n, 4),
        }
        faced = [r for r in rs if r.get("face")]
        if faced:
            s["at_face"] = round(sum(r["price"] == r["face"] for r in faced) / len(faced), 4)
            s["price_over_face_q10_50_90"] = _quantiles([r["price"] / r["face"] for r in faced], (0.1, 0.5, 0.9))
        if block == "BuyTrain.corp":
            s["by_train"] = dict(Counter(r.get("train") for r in rs).most_common())
        cells = {}
        for ct in ("Bid", "BuyTrain", "BuyCompany"):
            on_atom, bits = 0, 0.0
            for r in rs:
                k = (r["price"] - r["lo"]) // max(r["step"], 1)
                pc = price_cells(ct, 0, (r["hi"] - r["lo"]) // max(r["step"], 1), 1)
                c = pc.cell_of(k)
                on_atom += c < pc.n_atoms
                bits += math.log2(max(int(pc.counts[c]), 1))
            cells[ct] = {"on_atom": round(on_atom / n, 4), "bits": round(bits / n, 3)}
        s["cells_in_step_units"] = cells
        out[block] = s
    return out


def main_index(args, paths: list[str]) -> int:
    from functools import partial

    fn = partial(audit_game_index, apply_check=not args.no_apply_check)
    # Build the layout once before forking (the first index call of a process builds it: ~0.5 s for
    # 1822's 0.75M descriptors), so the per-decision times are the generator's.
    info = engine_rs.rl_layout_info(TITLE)
    print(f"{TITLE}: layout {info['version']} ({info['total']} indices, fingerprint {info['fingerprint']})")
    del info
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            results = pool.map(fn, paths, chunksize=1)
    else:
        results = [fn(p) for p in paths]
    totals: Counter = Counter()
    examples: dict = defaultdict(list)
    errors = Counter()
    error_examples = defaultdict(list)
    census, times = [], []
    by_players = Counter()
    for r in results:
        if r["error"]:
            key = r["error"].split(":")[0]
            errors[key] += 1
            if len(error_examples[key]) < 4:
                error_examples[key].append(f"{r['id']}: {r['error'][:160]}")
        if r.get("players"):
            by_players[r["players"]] += 1
        for k, v in r["counts"].items():
            totals[k] += v
        for k, ids in r["examples"].items():
            if len(examples[k]) < 6:
                examples[k].extend(f"{r['id']}@{i}" for i in ids[: 6 - len(examples[k])])
        census.extend(r["census"])
        times.extend(r["times"])
    kinds = ["exact", "policy", "missing", "ambiguous", "decode_mismatch", "apply_mismatch"]
    in_scope = dict(sorted(by_players.items()))
    print(f"{TITLE}: {len(paths)} games ({in_scope} in scope by players); errors: {dict(errors)}")
    for key, ex in error_examples.items():
        if key not in ("skipped", "out of scope (players)"):
            print(f"  {key}: {ex}")
    print(f"{'block':28} " + " ".join(f"{k[:9]:>9}" for k in kinds))
    for b in sorted({b for b, _ in totals}):
        ex = [e for (bb, k), e in examples.items() if bb == b and k != "exact"]
        print(f"{b:28} " + " ".join(f"{totals[(b, k)]:9}" for k in kinds) + f"  {[x for e in ex for x in e][:4]}")
    n = sum(totals.values())
    one = sum(v for (b, k), v in totals.items() if k in ("exact", "policy"))
    print(f"exactly one index (rules' moves): {one}/{n} = {one / max(n, 1):.4%}")
    for k in kinds[1:]:
        print(f"  {k}: {sum(v for (b, kk), v in totals.items() if kk == k)}")
    ms = [t * 1000 for t in times]
    timing = {
        "decisions": len(ms),
        "ms_p50_p90_p99": [round(x, 2) for x in _quantiles(ms, (0.5, 0.9, 0.99))],
        "ms_max": round(max(ms), 1) if ms else None,
        "ms_mean": round(sum(ms) / max(len(ms), 1), 2),
    }
    print(f"rl_legal_indices() per decision (title's policy): {timing}")
    summary = summarize_census(census)
    print("price census:")
    print(json.dumps(summary, indent=1))
    if args.json_out:
        json.dump(
            {
                "title": TITLE,
                "games": len(paths),
                "by_players": dict(by_players),
                "errors": dict(errors),
                "error_examples": dict(error_examples),
                "totals": {f"{b}|{k}": v for (b, k), v in totals.items()},
                "examples": {f"{b}|{k}": v for (b, k), v in examples.items()},
                "timing": timing,
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
    ap.add_argument("--title", default="1822", help="an 1822-family title (1822, 1822PNW, ...)")
    ap.add_argument("--glob", default=None, help="default: human_games/scrape_2026_10/<title>/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument(
        "--index",
        action="store_true",
        help="index level: every human decision against the title's RL layout (exactly one index, decode and "
        "apply checks), the human price census of the price-bearing indices and the generator's time per decision",
    )
    ap.add_argument("--no-apply-check", action="store_true", help="--index: skip the apply-by-index state check")
    ap.add_argument("--census-rows", default=None, help="--index: write every priced human decision (JSONL)")
    args = ap.parse_args()
    global TITLE
    TITLE = args.title
    paths = sorted(globmod.glob(args.glob or f"human_games/scrape_2026_10/{TITLE}/*.json"))[: args.limit]
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
    print(f"{'family':28} {'exact':>8} {'missing':>8} {'ambig':>6} {'policy':>6}")
    for f in families:
        e, m, a = totals[(f, "exact")], totals[(f, "missing")], totals[(f, "ambiguous")]
        p = totals[(f, "policy")]
        print(
            f"{f:28} {e:8} {m:8} {a:6} {p:6}  {examples.get((f, 'missing'), [])[:3]} "
            f"{examples.get((f, 'ambiguous'), [])[:2]}"
        )
    all_e = sum(v for (t, k), v in totals.items() if k in ("exact", "policy"))
    all_p = sum(v for (t, k), v in totals.items() if k == "policy")
    all_n = sum(totals.values())
    print(f"overall exact (rules) {all_e}/{all_n} = {all_e / max(all_n, 1):.4%}")
    print(f"excluded by the cross-player policy: {all_p}/{all_n} = {all_p / max(all_n, 1):.4%}")
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
