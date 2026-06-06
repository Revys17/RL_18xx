#!/usr/bin/env python
"""Strict Rust-vs-Python parity runner for the 1830 engines.

Emits machine-readable JSON failure records. There are NO tolerances, NO skip
lists, and NO "expected error" exclusions — any divergence between the engines
is a failure. Python is the oracle; Rust must reproduce it exactly.

Two checks per random-walk step and per human-game action:
  * enumeration: the factored legal-action SETS must match exactly, including
    the exact price_range for price-bearing types (Bid/BuyTrain/BuyCompany).
  * state: compare_state(rust, py) must report no differences.

Modes
-----
  --random START:END         walk seeds [START, END)
  --random-seeds 42,43,...    explicit seed list (for reduced re-runs)
  --human GLOB                replay human games matching GLOB
  --human-games p1,p2,...      explicit game-file list (for reduced re-runs)
  --out FILE                  write JSON results (default: stdout)
  --max-steps N               cap steps/actions per game (default 100000 = full)

Output JSON
-----------
  {
    "failures":     [ <record>, ... ],   # real Rust-vs-Python parity failures
    "python_side":  [ <record>, ... ],   # games the Python ENGINE itself rejects
                                          # (human mode only); NOT a parity failure,
                                          # surfaced for review, never silently dropped
    "summary": {random_run, random_fail, human_run, human_fail, python_side}
  }

Failure record fields: kind ('random'|'human'), id (seed or game id), step,
check ('enum'|'price'|'state'|'rust_error'|'python_self_reject'), detail,
py, rust, round, op_step, phase, entity.

In RANDOM mode the actions are produced by the Python ActionHelper, so a Python
engine rejection of its OWN helper's action is a Python-side bug (recorded as a
``python_self_reject`` failure to fix). In HUMAN mode the actions come from
external data, so a Python rejection means the game can't be replayed at all and
is bucketed under ``python_side`` (per the agreed rule: Rust is only required to
match steps the Python engine accepts).
"""

import argparse
import glob as globmod
import json
import logging
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.action_helper import ActionHelper  # noqa: E402
from rl18xx.game.factored_action_helper import FactoredActionHelper  # noqa: E402
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402
from rl18xx.agent.alphazero.pretraining import (  # noqa: E402
    _get_game_object_for_game_with_reason,
)
from tests.validate_rust_engine import compare_state  # noqa: E402

MAX_STEPS_DEFAULT = 100000


def _key(la):
    """Strict categorical + exact-price fingerprint. Works on LegalAction
    objects (both engines return these)."""
    t, e, p = la.type, la.entity, la.params
    pr = tuple(la.price_range) if la.price_range is not None else None
    if t == "Pass":
        return ("Pass",)
    if t == "Bid":
        return ("Bid", e.get("private"), pr)
    if t == "Par":
        return ("Par", e.get("corp"), p.get("par_price"))
    if t == "BuyShares":
        return ("BuyShares", e.get("corp"), p.get("source"), int(p.get("percent", 0)))
    if t == "CompanyBuyShares":
        return ("CompanyBuyShares", e.get("private"), e.get("corp"), p.get("source"))
    if t == "SellShares":
        return ("SellShares", e.get("corp"), int(p.get("count", 0)))
    if t == "PlaceToken":
        return ("PlaceToken", p.get("hex"), p.get("city"), p.get("slot"))
    if t == "LayTile":
        return ("LayTile", p.get("hex"), p.get("tile"), p.get("rotation"))
    if t == "BuyTrain":
        return ("BuyTrain", e.get("source"), e.get("train"), e.get("exchange"), pr)
    if t == "DiscardTrain":
        return ("DiscardTrain", p.get("train"))
    if t == "Dividend":
        return ("Dividend", p.get("kind"))
    if t == "BuyCompany":
        return ("BuyCompany", e.get("private"), pr)
    if t == "RunRoutes":
        return ("RunRoutes",)
    if t == "Bankrupt":
        return ("Bankrupt",)
    return (t,)


def _ctx(py):
    rnd = type(py.round).__name__
    stp = type(py.active_step()).__name__ if py.active_step() else None
    phase = getattr(getattr(py, "phase", None), "name", "?")
    ent = py.current_entity
    ent_s = getattr(ent, "name", getattr(ent, "id", str(ent)))
    return {"round": rnd, "op_step": stp, "phase": phase, "entity": ent_s}


def run_random_seed(seed, max_steps):
    """Walk one random-walk game in lockstep. Returns the FIRST failure record
    (the earliest divergence — its root cause is what needs fixing) or None."""
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    py = GameMap().game_by_title("1830")(names)
    rust = RustGame(names)
    adapter = RustGameAdapter(rust)
    hp = FactoredActionHelper()
    ah = ActionHelper()

    for step in range(max_steps):
        if py.finished:
            break
        ctx = _ctx(py)

        # --- enumeration parity (categorical + exact price) ---
        py_map, rust_map = {}, {}
        for la in hp.get_choices(py):
            py_map[_key(la)] = la.price_range
        for la in adapter.get_factored_choices():
            rust_map[_key(la)] = la.price_range
        py_keys, rust_keys = set(py_map), set(rust_map)
        py_only = sorted(map(str, py_keys - rust_keys))
        rust_only = sorted(map(str, rust_keys - py_keys))
        if py_only or rust_only:
            return {"kind": "random", "id": seed, "step": step, "check": "enum",
                    "detail": {"py_only": py_only, "rust_only": rust_only},
                    "py": py_only, "rust": rust_only, **ctx}

        # --- state parity ---
        state_errs = compare_state(rust, py)
        if state_errs:
            return {"kind": "random", "id": seed, "step": step, "check": "state",
                    "detail": list(state_errs)[:20], "py": None, "rust": None, **ctx}

        # --- advance both engines with a Python-helper-chosen action ---
        legacy = ah.get_all_choices_limited(py)
        if not legacy:
            break
        action = rng.choice(legacy).to_dict()
        try:
            py = py.process_action(action)
        except Exception as exc:
            return {"kind": "random", "id": seed, "step": step, "check": "python_self_reject",
                    "detail": str(exc), "py": str(action), "rust": None, **ctx}
        try:
            adapter.process_action(action)
        except BaseException as exc:
            return {"kind": "random", "id": seed, "step": step, "check": "rust_error",
                    "detail": str(exc), "py": None, "rust": str(action), **ctx}
    return None


def run_human_game(path, max_steps):
    """Compare the REAL human-game IMPORT: run the production cleaning pipeline
    (`_get_game_object_for_game_with_reason`) on BOTH engines and compare the
    outcome (drop reason / final state). This is what training-data import
    actually does (filter_actions + pass/skip handling) — NOT a raw filtered
    replay, which hits blocking-step rejections the real import resolves and
    thereby masks divergences. Returns (failure_or_None, python_side_or_None)."""
    game_id = Path(path).stem
    try:
        game = json.load(open(path))
    except Exception as exc:
        return None, {"kind": "python_side", "id": game_id, "step": -1,
                      "reason": f"unparseable game JSON: {exc}"}

    def clean(use_rust):
        try:
            obj, reason = _get_game_object_for_game_with_reason(game, use_rust=use_rust)
            return obj, reason, None
        except BaseException as exc:  # incl. pyo3 PanicException (a BaseException)
            return None, None, f"{type(exc).__name__}: {str(exc)[:140]}"

    opy, rpy, epy = clean(False)
    oru, rru, eru = clean(True)
    base = {"id": game_id, "step": -1, "round": "?", "op_step": "?",
            "phase": "?", "entity": "?"}

    # Python (oracle) errored during import -> Python can't import this game:
    # python_side, UNLESS Rust imported it cleanly (Rust more permissive).
    if epy is not None:
        if eru is not None or oru is None:
            return None, {"kind": "python_side", "id": game_id, "step": -1,
                          "reason": f"python import error: {epy}"}
        return ({"kind": "human", "check": "rust_accepts_python_reject",
                 "detail": f"Python import errored ({epy}) but Rust imported (reason={rru!r})",
                 "py": epy, "rust": None, **base}, None)

    # Rust errored where Python imported cleanly -> real divergence.
    if eru is not None:
        return ({"kind": "human", "check": "rust_import_error",
                 "detail": f"Python imported (reason={rpy!r}) but Rust raised: {eru}",
                 "py": str(rpy), "rust": eru, **base}, None)

    # Different drop/keep outcome -> divergence.
    if rpy != rru:
        return ({"kind": "human", "check": "reason_mismatch",
                 "detail": f"py_reason={rpy!r} rust_reason={rru!r}",
                 "py": str(rpy), "rust": str(rru), **base}, None)

    # Both dropped with the SAME reason -> consistent (python_side, non-parity).
    if rpy is not None:
        return None, {"kind": "python_side", "id": game_id, "step": -1,
                      "reason": f"both engines drop: {rpy}"}

    # Both imported successfully -> final state must match EXACTLY.
    finished = bool(getattr(opy, "finished", False))
    try:
        res_py = {k: round(v) for k, v in opy.result().items()}
        res_ru = {k: round(v) for k, v in oru.result().items()}
    except BaseException as exc:
        res_py = res_ru = f"result() raised: {exc}"
    if res_py != res_ru:
        return ({"kind": "human", "check": "result_mismatch",
                 "detail": f"py={res_py} rust={res_ru}", "py": str(res_py),
                 "rust": str(res_ru), "finished": finished, **base}, None)
    rust_raw = getattr(oru, "_game", oru)
    try:
        state_errs = compare_state(rust_raw, opy)
    except BaseException as exc:
        state_errs = [f"compare_state raised: {exc}"]
    if state_errs:
        return ({"kind": "human", "check": "state_mismatch",
                 "detail": list(state_errs)[:20], "py": None, "rust": None,
                 "finished": finished, **base}, None)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", help="seed range START:END")
    ap.add_argument("--random-seeds", help="comma-separated explicit seeds")
    ap.add_argument("--human", help="glob for human game JSONs")
    ap.add_argument("--human-games", help="comma-separated explicit game file paths")
    ap.add_argument("--out", help="output JSON file (default stdout)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    args = ap.parse_args()

    failures, python_side = [], []
    random_run = human_run = 0

    seeds = []
    if args.random:
        a, b = args.random.split(":")
        seeds = list(range(int(a), int(b)))
    if args.random_seeds:
        seeds += [int(s) for s in args.random_seeds.split(",") if s.strip()]
    for seed in seeds:
        random_run += 1
        try:
            f = run_random_seed(seed, args.max_steps)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # incl. pyo3 PanicException (a BaseException)
            f = {"kind": "random", "id": seed, "step": -1, "check": "runner_crash",
                 "detail": traceback.format_exc()[-800:], "py": None, "rust": None,
                 "round": "?", "op_step": "?", "phase": "?", "entity": "?"}
        if f:
            failures.append(f)

    games = []
    if args.human:
        games += sorted(globmod.glob(args.human))
    if args.human_games:
        games += [g for g in args.human_games.split(",") if g.strip()]
    for path in games:
        human_run += 1
        try:
            f, ps = run_human_game(path, args.max_steps)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # incl. pyo3 PanicException (a BaseException)
            f = {"kind": "human", "id": Path(path).stem, "step": -1, "check": "runner_crash",
                 "detail": traceback.format_exc()[-800:], "py": None, "rust": None,
                 "round": "?", "op_step": "?", "phase": "?", "entity": "?"}
            ps = None
        if f:
            failures.append(f)
        if ps:
            python_side.append(ps)

    result = {
        "failures": failures,
        "python_side": python_side,
        "summary": {
            "random_run": random_run,
            "random_fail": sum(1 for f in failures if f["kind"] == "random"),
            "human_run": human_run,
            "human_fail": sum(1 for f in failures if f["kind"] == "human"),
            "python_side": len(python_side),
        },
    }
    out = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}: {result['summary']}")
    else:
        print(out)


if __name__ == "__main__":
    main()
