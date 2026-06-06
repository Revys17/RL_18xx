#!/usr/bin/env python
"""Dual-engine cleaning diagnostic — pinpoint the FIRST Rust↔Python divergence
during a human-game IMPORT.

Why this exists
---------------
The production human-import parity metric (``tests/parity_runner.py`` human mode)
runs the cleaning pipeline (`pretraining._get_game_object_for_game_with_reason`)
**separately** on each engine and compares the final outcome. That tells you a
game diverges, but not *where* or *why*. The simple ways to localize it all fail:

  * a raw ``filter_actions`` replay breaks at blocking steps (Route/etc.) before
    it ever reaches the divergence — the real cleaning resolves those with
    pass/run_routes insertion;
  * ``game.raw_actions`` does not round-trip to a fresh game;
  * the cleaning does per-engine action substitution + pass-insertion, so you
    cannot just diff two recorded action streams.

So we instrument the cleaning *loop itself*. We run BOTH engines through the
**same** cleaning decisions in lockstep:

  * the PYTHON engine is the oracle and makes every cleaning decision
    (entity-owner resolution, pass insertion, run_routes insertion, skip,
    action-helper substitution, drop branches);
  * every action that actually gets APPLIED (the main action plus any inserted
    pass / run_routes) is applied to BOTH engines;
  * after each applied action we call ``compare_state(rust_raw, python)`` AND
    check whether Rust raised where Python did not.

The first action where Rust raises or ``compare_state`` reports a difference is
the divergence. Because Python drives the decisions, both engines receive an
identical action stream — so any divergence is a pure Rust ``process_action``
bug, isolated from cleaning-decision differences. (Up to the first divergence
the two engines' states are identical, so it does not matter which engine the
decisions are read from — they would be the same either way.)

This loop is a faithful mirror of
``pretraining._get_game_object_for_game_with_reason``. The decision *logic*
(filter_actions / should_add_pass / should_skip_action /
check_action_in_action_helper) is imported from ``pretraining`` so it stays in
sync; only the loop skeleton is duplicated, with the dual-apply + compare bolted
on at the three ``process_action`` call sites.

Usage
-----
    uv run python tests/cleaning_diff.py human_games/1830/26825.json
    uv run python tests/cleaning_diff.py human_games/1830/26825.json human_games/1830/26861.json
    uv run python tests/cleaning_diff.py --json /tmp/d.json human_games/1830/*.json

The result for each game is one of:
  * {"status": "parity"}                     both engines agreed throughout
  * {"status": "dropped", "reason": ...}     Python dropped the game (no divergence found)
  * {"status": "rust_error", ...}            Rust raised where Python applied an action
  * {"status": "state_divergence", ...}      compare_state flagged a difference after an applied action
  * {"status": "python_error", ...}          Python itself failed to apply (not a Rust bug)

Each divergence record carries: the filtered-action ``index``, the applied
``action`` dict, the ``label`` (main/inserted-pass/inserted-run_routes), the
divergent ``fields`` (compare_state output) or ``error`` text, and the
game/round context.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402
from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    filter_actions,
    should_add_pass,
    should_skip_action,
    check_action_in_action_helper,
    RouteStep,
    BuySellParShares,
)
from tests.validate_rust_engine import compare_state  # noqa: E402


def _ctx(py):
    """Round/step/phase/entity context, matching parity_runner's _ctx."""
    try:
        rnd = type(py.round).__name__
        stp = type(py.active_step()).__name__ if py.active_step() else None
        phase = getattr(getattr(py, "phase", None), "name", "?")
        ent = py.current_entity
        ent_s = getattr(ent, "name", getattr(ent, "id", str(ent)))
        return {"round": rnd, "op_step": stp, "phase": phase, "entity": str(ent_s)}
    except Exception:
        return {"round": "?", "op_step": "?", "phase": "?", "entity": "?"}


class _Divergence(Exception):
    """Raised internally to unwind the loop at the first divergence."""

    def __init__(self, record):
        self.record = record


def diagnose_game(game: dict):
    """Run both engines through the same cleaning decisions in lockstep.

    Returns a result dict (see module docstring). The Python engine is the
    oracle and makes all decisions; both engines receive the identical applied
    action stream; the first divergent applied action is reported.
    """
    optional_rules = bool(game["settings"].get("optional_rules"))
    num_players = len(game["players"])
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}

    # Python oracle (drives decisions) + Rust mirror (compared only).
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    py_state = game_class(players)
    ru_state = RustGameAdapter(RustGame(players))

    player_mapping = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    filtered_actions = filter_actions(game["actions"])

    # Box so the closure can rebind py_state (process_action may return a clone).
    state = {"py": py_state}

    def apply_both(idx, action, label):
        """Apply ``action`` to BOTH engines, then compare. Records the first
        divergence by raising _Divergence. ``idx`` is the filtered-action index;
        ``label`` distinguishes main vs synthesized pass/run_routes."""
        ctx = _ctx(state["py"])
        # 1) Python (oracle) — must accept; if it raises, that is not a Rust bug.
        try:
            state["py"] = state["py"].process_action(action)
        except Exception as exc:
            raise _Divergence({
                "status": "python_error", "index": idx, "label": label,
                "action": action, "error": f"{type(exc).__name__}: {exc}", **ctx,
            })
        # 2) Rust — a raise here (incl. pyo3 PanicException, a BaseException) is
        #    a divergence: Python applied this exact action and Rust could not.
        try:
            ru_state.process_action(action)
        except BaseException as exc:
            raise _Divergence({
                "status": "rust_error", "index": idx, "label": label,
                "action": action, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                **ctx,
            })
        # 3) State must match exactly after the applied action.
        try:
            errs = compare_state(ru_state._game, state["py"])
        except BaseException as exc:
            errs = [f"compare_state raised: {type(exc).__name__}: {exc}"]
        if errs:
            raise _Divergence({
                "status": "state_divergence", "index": idx, "label": label,
                "action": action, "fields": list(errs)[:20], **ctx,
            })

    try:
        for i, action in enumerate(filtered_actions):
            gs = state["py"]  # decisions read the oracle's current state

            if action["entity_type"] == "player":
                action["entity"] = player_mapping[action["entity"]]
                action["user"] = action["entity"]
            else:
                entity_owner = gs.get(action["entity_type"], action["entity"]).player()
                if action.get("user", None):
                    if action["user"] not in player_mapping:
                        pass
                    elif entity_owner.id != player_mapping[action["user"]]:
                        pass
                    action["user"] = entity_owner.id

            # --- drop branches (mirror the production loop; oracle-driven) ---
            if action["type"] == "buy_company":
                company_purchaser = gs.get(action["entity_type"], action["entity"]).player()
                company_owner = gs.company_by_id(action["company"]).player()
                if company_purchaser.id != company_owner.id:
                    return {"status": "dropped", "reason": "cross_player_company_purchase",
                            "scanned": i}

            if action["type"] == "buy_train":
                train_purchaser = gs.get(action["entity_type"], action["entity"])
                train_owner = gs.train_by_id(action["train"]).owner
                if train_owner.is_corporation():
                    if train_purchaser.player() != train_owner.player():
                        return {"status": "dropped", "reason": "cross_player_train_purchase",
                                "scanned": i}
                if not action.get("exchange"):
                    bought = gs.train_by_id(action["train"])
                    if bought is not None and not bought.owner.is_corporation():
                        depot_trains = gs.depot.depot_trains()
                        buyable_names = {t.name for t in depot_trains}
                        if bought.name not in buyable_names and bought in gs.depot.upcoming:
                            return {"status": "dropped", "reason": "ruby_depot_phase_skip",
                                    "scanned": i}

            if action["entity_type"] == "company" and action["entity"] == "MH":
                mh_owner = gs.company_by_id("MH").player()
                current_player = gs.current_entity.player()
                if mh_owner != current_player:
                    return {"status": "dropped", "reason": "mh_out_of_turn", "scanned": i}

            if (
                action["entity_type"] == "company"
                and action["entity"] in ("CS", "DH")
                and action["type"] == "lay_tile"
                and not gs.round.operating
            ):
                return {"status": "dropped", "reason": "company_tile_lay_outside_or",
                        "scanned": i}

            # --- pass insertion (APPLIED to both) ---
            if should_add_pass(action, gs):
                pass_action = {
                    "type": "pass",
                    "entity": gs.current_entity.id,
                    "entity_type": gs.current_entity.__class__.__name__.lower(),
                    "user": gs.current_entity.player().id,
                }
                apply_both(i, pass_action, "inserted-pass")
                gs = state["py"]

            # --- run_routes insertion (APPLIED to both) ---
            active_step = gs.round.active_step()
            if (
                isinstance(active_step, RouteStep)
                and action["type"] in ("buy_train", "dividend", "discard_train")
                and action.get("entity_type") == "corporation"
                and gs.current_entity.id == action.get("entity")
            ):
                run_routes_action = {
                    "type": "run_routes",
                    "entity": gs.current_entity.id,
                    "entity_type": "corporation",
                    "user": gs.current_entity.player().id,
                    "routes": [],
                }
                apply_both(i, run_routes_action, "inserted-run_routes")
                gs = state["py"]

            if should_skip_action(filtered_actions, action, gs, i):
                continue

            if isinstance(gs.round.active_step(), BuySellParShares) and action["type"] == "buy_shares":
                if action["entity_type"] != "company":
                    shares = [gs.share_by_id(share) for share in action["shares"]]
                    if not gs.round.active_step().can_buy_shares(gs.current_entity, shares):
                        return {"status": "dropped", "reason": "illegal_share_buy", "scanned": i}

            if action["type"] not in ["pass", "bankrupt"]:
                replacement_action = check_action_in_action_helper(action, gs)
                if replacement_action is not None:
                    action = replacement_action

            if (
                action["entity_type"] == "corporation"
                and action["type"] in ("lay_tile", "place_token", "run_routes", "dividend",
                                        "buy_train", "buy_company", "pass")
                and action["entity"] != gs.current_entity.id
            ):
                return {"status": "dropped", "reason": "entity_mismatch", "scanned": i}

            # --- the main action (APPLIED to both) ---
            apply_both(i, action, "main")

        return {"status": "parity", "scanned": len(filtered_actions)}
    except _Divergence as d:
        return d.record


def trace_clean(game: dict, use_rust: bool):
    """Run the REAL production cleaning on a SINGLE engine, recording every
    APPLIED action (the actual dict handed to ``process_action``, including
    synthesized pass / run_routes and action-helper substitutions).

    This mirrors ``_get_game_object_for_game_with_reason`` exactly for the
    chosen engine — decisions AND application both run on that engine — so the
    recorded stream is what production cleaning actually does. Catching the
    exact failure point (which may be a DECISION step such as
    ``check_action_in_action_helper`` building a SellShares, not
    ``process_action``) is the whole point: that is where ``rust_import_error``
    cases like 26861 surface.

    Returns {"applied": [ {index,label,action} ... ],
             "outcome": {"status": "parity"|"dropped"|"error", ...}}.
    """
    num_players = len(game["players"])
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    if use_rust:
        gs = RustGameAdapter(RustGame(players))
    else:
        gs = GameMap().game_by_title("1830")(players)

    player_mapping = {p["id"]: i + 1 for i, p in enumerate(game["players"])}
    filtered_actions = filter_actions(game["actions"])
    applied = []

    def record_and_apply(idx, action, label):
        ctx = _ctx(gs)
        applied.append({"index": idx, "label": label, "action": action, **ctx})
        return gs.process_action(action)

    i = 0
    cur = {"action": None}  # action currently being processed (for error reporting)
    try:
        for i, action in enumerate(filtered_actions):
            cur["action"] = action
            if action["entity_type"] == "player":
                action["entity"] = player_mapping[action["entity"]]
                action["user"] = action["entity"]
            else:
                entity_owner = gs.get(action["entity_type"], action["entity"]).player()
                if action.get("user", None) and action["user"] in player_mapping:
                    action["user"] = entity_owner.id
                elif action.get("user", None):
                    action["user"] = entity_owner.id

            if action["type"] == "buy_company":
                if gs.get(action["entity_type"], action["entity"]).player().id != \
                        gs.company_by_id(action["company"]).player().id:
                    return {"applied": applied,
                            "outcome": {"status": "dropped", "reason": "cross_player_company_purchase", "scanned": i}}
            if action["type"] == "buy_train":
                tp = gs.get(action["entity_type"], action["entity"])
                to = gs.train_by_id(action["train"]).owner
                if to.is_corporation() and tp.player() != to.player():
                    return {"applied": applied,
                            "outcome": {"status": "dropped", "reason": "cross_player_train_purchase", "scanned": i}}
                if not action.get("exchange"):
                    bought = gs.train_by_id(action["train"])
                    if bought is not None and not bought.owner.is_corporation():
                        buyable = {t.name for t in gs.depot.depot_trains()}
                        if bought.name not in buyable and bought in gs.depot.upcoming:
                            return {"applied": applied,
                                    "outcome": {"status": "dropped", "reason": "ruby_depot_phase_skip", "scanned": i}}
            if action["entity_type"] == "company" and action["entity"] == "MH":
                if gs.company_by_id("MH").player() != gs.current_entity.player():
                    return {"applied": applied,
                            "outcome": {"status": "dropped", "reason": "mh_out_of_turn", "scanned": i}}
            if (action["entity_type"] == "company" and action["entity"] in ("CS", "DH")
                    and action["type"] == "lay_tile" and not gs.round.operating):
                return {"applied": applied,
                        "outcome": {"status": "dropped", "reason": "company_tile_lay_outside_or", "scanned": i}}

            if should_add_pass(action, gs):
                pass_action = {"type": "pass", "entity": gs.current_entity.id,
                               "entity_type": gs.current_entity.__class__.__name__.lower(),
                               "user": gs.current_entity.player().id}
                gs = record_and_apply(i, pass_action, "inserted-pass")

            active_step = gs.round.active_step()
            if (isinstance(active_step, RouteStep)
                    and action["type"] in ("buy_train", "dividend", "discard_train")
                    and action.get("entity_type") == "corporation"
                    and gs.current_entity.id == action.get("entity")):
                rr = {"type": "run_routes", "entity": gs.current_entity.id,
                      "entity_type": "corporation", "user": gs.current_entity.player().id, "routes": []}
                gs = record_and_apply(i, rr, "inserted-run_routes")

            if should_skip_action(filtered_actions, action, gs, i):
                continue

            if isinstance(gs.round.active_step(), BuySellParShares) and action["type"] == "buy_shares":
                if action["entity_type"] != "company":
                    shares = [gs.share_by_id(s) for s in action["shares"]]
                    if not gs.round.active_step().can_buy_shares(gs.current_entity, shares):
                        return {"applied": applied,
                                "outcome": {"status": "dropped", "reason": "illegal_share_buy", "scanned": i}}

            if action["type"] not in ["pass", "bankrupt"]:
                repl = check_action_in_action_helper(action, gs)
                if repl is not None:
                    action = repl

            if (action["entity_type"] == "corporation"
                    and action["type"] in ("lay_tile", "place_token", "run_routes", "dividend",
                                            "buy_train", "buy_company", "pass")
                    and action["entity"] != gs.current_entity.id):
                return {"applied": applied,
                        "outcome": {"status": "dropped", "reason": "entity_mismatch", "scanned": i}}

            gs = record_and_apply(i, action, "main")

        return {"applied": applied, "outcome": {"status": "parity", "scanned": len(filtered_actions)}}
    except BaseException as exc:
        return {"applied": applied,
                "outcome": {"status": "error", "index": i, "cur_action": cur["action"],
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            **_ctx(gs)}}


def _canon(action):
    """Canonical comparison key for an applied action dict (drop volatile id)."""
    return json.dumps({k: v for k, v in action.items() if k != "id"}, sort_keys=True, default=str)


def diagnose_decisions(game: dict):
    """Trace the real per-engine cleaning and find where the two engines' applied
    action streams (i.e. their cleaning DECISIONS) first diverge, or where Rust
    errors during a decision/application that Python survives. Catches the
    decision-level class (e.g. 26861's rust_import_error)."""
    tp = trace_clean(game, use_rust=False)
    tr = trace_clean(game, use_rust=True)
    ap, ar = tp["applied"], tr["applied"]

    n = min(len(ap), len(ar))
    for k in range(n):
        if _canon(ap[k]["action"]) != _canon(ar[k]["action"]):
            return {"status": "decision_divergence", "index": ar[k]["index"],
                    "label": ar[k]["label"], "applied_step": k,
                    "py_action": ap[k]["action"], "rust_action": ar[k]["action"],
                    "round": ar[k].get("round"), "op_step": ar[k].get("op_step"),
                    "phase": ar[k].get("phase"), "entity": ar[k].get("entity")}

    op, orr = tp["outcome"], tr["outcome"]
    # Same applied prefix; outcomes differ.
    if orr["status"] == "error" and op["status"] != "error":
        return {"status": "rust_error", "index": orr.get("index"), "label": "decision/apply",
                "action": orr.get("cur_action"),
                "error": orr["error"], "py_outcome": op["status"],
                "round": orr.get("round"), "op_step": orr.get("op_step"),
                "phase": orr.get("phase"), "entity": orr.get("entity")}
    if op["status"] == "error" and orr["status"] != "error":
        return {"status": "python_error", "index": op.get("index"),
                "error": op["error"], "rust_outcome": orr["status"]}
    if op.get("status") == "dropped" and orr.get("status") == "dropped" and op.get("reason") != orr.get("reason"):
        return {"status": "reason_mismatch", "py_reason": op.get("reason"), "rust_reason": orr.get("reason")}
    if (op.get("status") == "dropped") != (orr.get("status") == "dropped"):
        return {"status": "reason_mismatch",
                "py_reason": op.get("reason") if op.get("status") == "dropped" else None,
                "rust_reason": orr.get("reason") if orr.get("status") == "dropped" else None}
    if len(ap) != len(ar):
        return {"status": "stream_length_mismatch", "py_applied": len(ap), "rust_applied": len(ar),
                "py_outcome": op, "rust_outcome": orr}
    return {"status": "parity", "scanned": len(ap)}


def run(path: str):
    """Full diagnosis: lockstep (process_action class) first; if that is clean,
    fall back to per-engine decision tracing (decision/adapter class)."""
    game = json.load(open(path))
    res = diagnose_game(game)
    if res["status"] in ("parity", "dropped"):
        # process_action is fine — check whether the real per-engine cleaning
        # decisions diverge (the rust_import_error / reason_mismatch class).
        dres = diagnose_decisions(game)
        if dres["status"] != "parity":
            dres["found_by"] = "decision-trace"
            res = dres
        else:
            res.setdefault("found_by", "lockstep")
    else:
        res["found_by"] = "lockstep"
    res["game_id"] = Path(path).stem
    return res


def _fmt(res):
    gid = res.get("game_id", "?")
    status = res["status"]
    via = f" [{res['found_by']}]" if res.get("found_by") else ""
    if status == "parity":
        return f"[{gid}] PARITY{via} — both engines agreed"
    if status == "dropped":
        return f"[{gid}] DROPPED by Python{via}: {res['reason']} (after {res.get('scanned')} actions)"
    if status == "reason_mismatch":
        return f"[{gid}] REASON_MISMATCH{via} — py={res.get('py_reason')!r} rust={res.get('rust_reason')!r}"
    if status == "stream_length_mismatch":
        return (f"[{gid}] STREAM_LENGTH_MISMATCH{via} — py_applied={res['py_applied']} "
                f"rust_applied={res['rust_applied']}")
    if status == "diagnostic_crash":
        return f"[{gid}] DIAGNOSTIC_CRASH: {res.get('error')}\n{res.get('trace','')}"
    head = (f"[{gid}] {status.upper()}{via} @ filtered-action #{res.get('index')} "
            f"({res.get('label')}) round={res.get('round')} step={res.get('op_step')} "
            f"phase={res.get('phase')} entity={res.get('entity')}")
    lines = [head]
    if "action" in res:
        lines.append(f"    action:      {json.dumps(res['action'], default=str)[:400]}")
    if "py_action" in res:
        lines.append(f"    py_action:   {json.dumps(res['py_action'], default=str)[:400]}")
        lines.append(f"    rust_action: {json.dumps(res['rust_action'], default=str)[:400]}")
    if "error" in res:
        lines.append(f"    error: {res['error']}")
    if "py_outcome" in res:
        lines.append(f"    (python outcome at that point: {res['py_outcome']})")
    for f in res.get("fields", []):
        lines.append(f"    field: {f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("games", nargs="+", help="human game JSON file paths")
    ap.add_argument("--json", help="write all results as a JSON array to this file")
    args = ap.parse_args()

    results = []
    for path in args.games:
        try:
            res = run(path)
        except BaseException as exc:
            import traceback
            res = {"game_id": Path(path).stem, "status": "diagnostic_crash",
                   "error": f"{type(exc).__name__}: {exc}",
                   "trace": traceback.format_exc()[-1200:]}
        results.append(res)
        print(_fmt(res))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.json}")

    # Exit non-zero if any game showed a real Rust divergence.
    bad = [r for r in results if r["status"] in ("rust_error", "state_divergence")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
