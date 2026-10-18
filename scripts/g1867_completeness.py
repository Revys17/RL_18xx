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
bought from another player) is reported as "policy". Reports exact /
missing / ambiguous / policy by action family with example game ids and
file action ids (no player data).

Corpus: finished 3-6 player games without optional rules (the generator's
scope); games the replay itself rejects are counted up to the rejection.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1867_completeness.py \\
        --glob 'human_games/scrape_2026_10/1867/*.json' --jobs 8 --json /tmp/c.json
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

SKIP = {"run_routes"}
# Action types the cross-player policy (GameTitle::generator_cross_player) can exclude.
POLICY_TYPES = {"buy_train", "buy_company"}


def _in_range(value, rng) -> bool:
    if rng is None or value is None:
        return False
    lo, hi, step = rng
    return lo <= value <= hi and (value - lo) % max(step, 1) == 0


def _shares_key(g, a: dict):
    ids = list(a.get("shares") or [])
    owners = sorted(g.g1867_share_owner(s) for s in ids)
    corp = ids[0].rsplit("_", 1)[0] if ids else a.get("corporation")
    pres = any(s.endswith("_0") for s in ids)
    return (corp, tuple(owners), int(a.get("percent") or 0), pres)


def _key(g, a: dict):
    """The outcome-relevant parameters of a wire action (state BEFORE it)."""
    t = a["type"]
    if t == "bid":
        return (a.get("company"), a.get("corporation"))
    if t in ("buy_shares", "sell_shares"):
        return _shares_key(g, a)
    if t == "par":
        sp = a.get("share_price")
        if isinstance(sp, str):
            sp = int(sp.split(",")[0])
        return (a.get("corporation"), sp)
    if t == "lay_tile":
        return (a.get("hex"), str(a.get("tile")).rsplit("-", 1)[0], int(a.get("rotation") or 0))
    if t == "place_token":
        city = a.get("city")
        return tuple(g.g1867_city_ref(city)) if city else (a.get("hex"),)
    if t == "remove_token":
        return tuple(g.g1867_city_ref(a.get("city"))) + (int(a.get("slot") or 0),)
    if t == "dividend":
        return (a.get("kind"),)
    if t == "choose":
        return (str(a.get("choice")),)
    if t == "buy_train":
        name, owner = g.g1867_train_info(a.get("train", ""))
        exch = a.get("exchange")
        return (name, owner, g.g1867_train_info(exch)[0] if exch else None)
    if t == "discard_train":
        return (g.g1867_train_info(a.get("train", ""))[0],)
    if t == "merge":
        return (a.get("corporation"),)
    if t == "buy_company":
        return (a.get("company"),)
    return ()


def _price(a: dict):
    return a.get("price") if a["type"] in ("bid", "buy_train", "buy_company") else None


def matches(g, human: dict, moves: list[dict]) -> int:
    t = human["type"]
    ent = str(human.get("entity"))
    hk = _key(g, human)
    hp = _price(human)
    n = 0
    for m in moves:
        w = m["wire"]
        if w["type"] != t or str(w.get("entity")) != ent:
            continue
        if _key(g, w) != hk:
            continue
        if m["range"] is not None:
            if not _in_range(hp, m["range"]):
                continue
        elif hp is not None and w.get("price") != hp:
            continue
        n += 1
    return n


def replay_only_reason(g, human: dict) -> str | None:
    """A missed human action in a documented replay-only class (see the
    generator status notes in docs/g1867_remaining_work.md), else None."""
    offers = g.g1867_step_offers()
    if not offers:
        return None
    kind, ent, _types = offers[0]
    t, who = human["type"], str(human.get("entity"))
    if kind == "MajorTrainless" and t in ("choose", "pass") and who != ent:
        return "trainless_out_of_order"  # Ruby accepts any queued major; we decide in queue order
    if kind == "PostMergerShares" and t in ("buy_shares", "pass") and who != ent:
        return "dealing_out_of_turn"  # Ruby's actions ignore the dealing order
    if kind == "ReduceTokens" and t in ("buy_shares", "pass") and human.get("entity_type") == "player":
        return "pre_9655_merger_order"  # old records deal shares before removing tokens
    return None


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="human_games/scrape_2026_10/1867/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    paths = sorted(globmod.glob(args.glob))[: args.limit]
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
