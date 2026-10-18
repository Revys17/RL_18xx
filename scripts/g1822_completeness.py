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

    CUDA_VISIBLE_DEVICES="" uv run python scripts/g1822_completeness.py \\
        --glob 'human_games/scrape_2026_10/1822/*.json' --jobs 8 --json /tmp/c.json
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

from tests.title_replay_harness import SEAT_ID_BASE, game_seed, ruby_filtered_actions  # noqa: E402

TITLE = "1822"
SKIP = {"run_routes"}
# Action types the cross-player policy (GameTitle::generator_cross_player) can exclude.
POLICY_TYPES = {"buy_train", "acquire_company", "merge"}


def _in_range(value, rng) -> bool:
    if rng is None:
        return False
    lo, hi, step = rng
    return lo <= value <= hi and (value - lo) % max(step, 1) == 0


def _shares_key(g, a: dict):
    ids = list(a.get("shares") or [])
    owners = sorted(g.g1822_share_owner(s) for s in ids)
    corp = ids[0].rsplit("_", 1)[0] if ids else a.get("corporation")
    pres = any(s.endswith("_0") for s in ids)
    return (corp, tuple(owners), int(a.get("percent") or 0), pres)


def _key(g, a: dict):
    """The outcome-relevant parameters of a wire action (state BEFORE it)."""
    t = a["type"]
    if t == "bid":
        return (a.get("company"),)
    if t in ("buy_shares", "sell_shares"):
        return _shares_key(g, a)
    if t == "par":
        sp = a.get("share_price")
        if isinstance(sp, str):
            sp = int(sp.split(",")[0])
        return (a.get("corporation"), sp)
    if t == "lay_tile":
        tile = str(a.get("tile")).split("-")[0]
        return (
            a.get("hex"),
            tile,
            # (a builder cube's rotation changes nothing)
            0 if tile == "BC" else int(a.get("rotation") or 0),
            tuple(sorted(a.get("combo_entities") or [])),
        )
    if t in ("place_token", "remove_token"):
        # (1822MX's NdeM token auction names the token by its city)
        city = a.get("city")
        return tuple(g.g1822_city_ref(city)) if city else (a.get("hex"),)
    if t == "hex_token":
        return (a.get("hex"),)
    if t == "dividend":
        return (a.get("kind"),)
    if t in ("choose", "choose_ability"):
        return (str(a.get("choice")),)
    if t == "buy_train":
        name, owner = g.g1822_train_info(a.get("train", ""))
        return (name, owner, a.get("variant") if owner == "depot" else None, bool(a.get("exchange")))
    if t == "discard_train":
        return (g.g1822_train_info(a.get("train", ""))[0],)
    if t == "merge":
        return (a.get("corporation"),)
    if t == "acquire_company":
        return (a.get("company"),)
    if t == "assign":
        return (a.get("target"), a.get("target_type"))
    return ()


def _price(a: dict):
    t = a["type"]
    if t == "bid":
        return a.get("price")
    if t == "payoff_player_debt_partial":
        return a.get("amount")
    if t == "buy_train":
        return a.get("price")
    return None


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
        elif t == "buy_train" and hp is not None and w.get("price") != hp:
            continue
        elif t == "payoff_player_debt_partial":
            continue
        n += 1
    return n


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--title", default="1822", help="an 1822-family title (1822, 1822PNW, ...)")
    ap.add_argument("--glob", default=None, help="default: human_games/scrape_2026_10/<title>/*.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    global TITLE
    TITLE = args.title
    paths = sorted(globmod.glob(args.glob or f"human_games/scrape_2026_10/{TITLE}/*.json"))[: args.limit]
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
