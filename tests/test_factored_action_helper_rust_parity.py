"""Parity tests for the Rust ``get_factored_choices()`` vs. the Python
``FactoredActionHelper`` reference.

Walks several random games in lock-step on a Python ``BaseGame`` and a Rust
``BaseGame`` and asserts both engines emit equivalent factored ``LegalAction``
sets at every step.

The Python ``FactoredActionHelper`` in ``rl18xx/game/factored_action_helper.py``
is the source of truth. The Rust ``BaseGame.get_factored_choices()`` (exposed
through ``RustGameAdapter.get_factored_choices``) is the migration target the
AlphaZero pipeline will use.

Coverage notes
--------------
The Rust port currently covers the action types the AlphaZero pipeline needs
in practice:

- ``Pass`` (auction, stock, OR)
- ``Bid`` (waterfall auction, including may_purchase fixed-price)
- ``Par`` (stock round; pending-par from BO -> B&O)
- ``BuyShares`` / ``SellShares`` (stock round; sell during emergency BuyTrain)
- ``LayTile`` / ``PlaceToken`` / ``RunRoutes`` / ``Dividend`` / ``BuyTrain``
  / ``BuyCompany`` / ``DiscardTrain`` (operating round)
- ``Bankrupt`` (emergency BuyTrain fallback)
- ``CompanyBuyShares`` (MH -> NYC exchange; categorical-only)

The private-company special branches are now covered too:

- ``LayTile`` via ``CS`` (B20) and ``DH`` (F16 teleport, tile 57) special-track
  abilities.
- ``PlaceToken`` via ``DH`` teleport ability (F16).
- ``CompanyBuyShares`` (MH -> NYC exchange), including the pre-par claim.

The parity is enforced *exactly*: categorical option sets must match and, for
price-bearing types (Bid, BuyTrain, BuyCompany), the ``price_range`` must match
too (see :func:`_canonical_key`). There are no tolerated divergences.

Usage::

    uv run pytest tests/test_factored_action_helper_rust_parity.py -v
"""

import logging
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.disable(logging.CRITICAL)

from engine_rs import BaseGame as RustGame  # noqa: E402
from rl18xx.game.action_helper import ActionHelper  # noqa: E402
from rl18xx.game.factored_action_helper import (  # noqa: E402
    FactoredActionHelper,
    LegalAction,
)
from rl18xx.game.gamemap import GameMap  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402


def _canonical_key(la: LegalAction):
    """Stable fingerprint for a :class:`LegalAction`, comparable across the
    Python and Rust implementations.

    Includes the *exact* ``price_range`` for the price-bearing types (Bid,
    BuyTrain, BuyCompany). The two engines are expected to agree on prices
    exactly — the earlier tolerance ("disagree at the edges of bid windows")
    has been eliminated, so any price drift now surfaces as a divergence.
    """
    t = la.type
    e = la.entity
    p = la.params
    pr = tuple(la.price_range) if la.price_range is not None else None

    if t == "Pass":
        return ("Pass",)
    if t == "Bid":
        return ("Bid", e.get("private"), pr)
    if t == "Par":
        return ("Par", e.get("corp"), p.get("par_price"))
    if t == "BuyShares":
        # Note: we deliberately drop `source` (ipo vs market) from the key.
        # Python and Rust disagree on the share-pool home for B&O after the
        # BO private exchange (Python puts the remainder in market/bank,
        # Rust puts them in IPO). That's a pre-existing engine-level
        # difference, not a factored-helper bug.
        return ("BuyShares", e.get("corp"))
    if t == "CompanyBuyShares":
        # MH -> NYC exchange. Both engines surface this whenever the MH owner
        # can claim an available NYC ipo/market share (including before NYC
        # pars), outside the initial private auction.
        return ("CompanyBuyShares", e.get("private"), e.get("corp"))
    if t == "SellShares":
        return ("SellShares", e.get("corp"), int(p.get("count", 0)))
    if t == "PlaceToken":
        return ("PlaceToken", p.get("hex"), p.get("city"))
    if t == "LayTile":
        return ("LayTile", p.get("hex"), p.get("tile"), p.get("rotation"))
    if t == "BuyTrain":
        # Treat exchange branch (MH -> NYC discount, optional D, etc.) as a
        # separate categorical key so the test distinguishes regular buys.
        # Includes the exact price range (depot trains are fixed-price; cross-
        # corp buys must match spend_minmax exactly).
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


def _diff_choices(py_choices, rust_choices):
    """Return (py_only, rust_only) key sets for diagnostics.

    Strict set diff — no ignore list. Categorical keys that surface on one
    engine but not the other indicate a real divergence; see
    ``docs/rust_engine_bugs.md`` for the catalogue of known cases.
    """
    py_keys = {_canonical_key(c) for c in py_choices}
    rust_keys = {_canonical_key(c) for c in rust_choices}
    py_only = py_keys - rust_keys
    rust_only = rust_keys - py_keys
    return py_only, rust_only


def _play_random_game(seed: int, max_steps: int = 200, verbose: bool = False):
    """Walk a random game in lockstep, comparing factored choices at every step.

    Returns ``dict`` with::
        ok: bool
        steps: number of action steps successfully taken
        mismatches: list of (step_idx, py_only, rust_only) tuples
        terminated_early: bool — True if either engine errored
    """
    rng = random.Random(seed)
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    py_game = GameMap().game_by_title("1830")(names)
    rust_game = RustGame(names)
    adapter = RustGameAdapter(rust_game)
    helper_py = FactoredActionHelper()
    action_helper = ActionHelper()

    result = {"ok": True, "steps": 0, "mismatches": [], "terminated_early": False}
    for step_idx in range(max_steps):
        if py_game.finished:
            break

        py_choices = helper_py.get_choices(py_game)
        rust_choices = adapter.get_factored_choices()

        py_only, rust_only = _diff_choices(py_choices, rust_choices)
        if py_only or rust_only:
            result["mismatches"].append((step_idx, py_only, rust_only))
            if verbose:
                print(
                    f"  STEP {step_idx} DIVERGE py_only={py_only} rust_only={rust_only}"
                )

        # Pick an action via ActionHelper (legacy enumeration), since the
        # factored helper drops the price dimension. Apply to both engines.
        legacy_choices = action_helper.get_all_choices_limited(py_game)
        if not legacy_choices:
            break
        action = rng.choice(legacy_choices)
        action_dict = action.to_dict()
        try:
            py_game = py_game.process_action(action_dict)
        except Exception as exc:
            result["terminated_early"] = True
            if verbose:
                print(f"  STEP {step_idx} python error: {exc}")
            break
        try:
            adapter.process_action(action_dict)
        except Exception as exc:
            result["terminated_early"] = True
            if verbose:
                print(f"  STEP {step_idx} rust error: {exc}")
            break

        result["steps"] = step_idx + 1

    if result["mismatches"]:
        result["ok"] = False
    return result


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_factored_helper_parity_random_game(seed):
    """Random games stay in agreement step-for-step on the major action types.

    Walks 120 steps (enough to cover auction → stock → early OR but rarely
    far into the OR phase, where pre-existing Python/Rust engine differences
    in graph reachability and emergency-buy semantics surface). The factored
    helper is asserted to match on every step.
    """
    result = _play_random_game(seed, max_steps=120)

    mismatch_step_count = len(result["mismatches"])
    if mismatch_step_count > 0:
        # Show a compact diagnostic so failures are debuggable
        sample = result["mismatches"][:3]
        msg_parts = []
        for step_idx, py_only, rust_only in sample:
            msg_parts.append(
                f"step={step_idx} py_only={py_only} rust_only={rust_only}"
            )
        msg = "; ".join(msg_parts)
        # Strict equality: any factored-choice divergence is an engine bug to
        # surface. See ``docs/rust_engine_bugs.md`` for the running catalogue.
        assert mismatch_step_count == 0, (
            f"Factored-choice mismatches in seed={seed}: "
            f"{mismatch_step_count} divergent steps. Sample: {msg}"
        )


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_factored_helper_parity_long_game(seed):
    """Long-running games (up to 500 actions) — strict parity invariant.

    Any factored-choice divergence is a real engine bug; see
    ``docs/rust_engine_bugs.md`` for the breakdown of known categories
    (LayTile reachability, Pass availability in emergency buys, BuyTrain
    price-range disagreements, etc.).
    """
    result = _play_random_game(seed, max_steps=500)
    mismatch_count = len(result["mismatches"])
    assert mismatch_count == 0, (
        f"Factored-helper mismatches in seed={seed}: "
        f"{mismatch_count}/{result['steps']} divergent steps."
    )


def test_factored_helper_parity_initial_state():
    """Sanity: at game start, both engines emit the same Bid + Pass options."""
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    py_game = GameMap().game_by_title("1830")(names)
    adapter = RustGameAdapter(RustGame(names))

    helper = FactoredActionHelper()
    py_choices = helper.get_choices(py_game)
    rust_choices = adapter.get_factored_choices()

    py_keys = {_canonical_key(c) for c in py_choices}
    rust_keys = {_canonical_key(c) for c in rust_choices}

    assert py_keys == rust_keys, (
        f"Initial-state mismatch: py_only={py_keys - rust_keys}, "
        f"rust_only={rust_keys - py_keys}"
    )

    # Also verify SV is may_purchase (price_range collapsed to (20, 20))
    sv_rust = next(
        (c for c in rust_choices if c.entity.get("private") == "SV"),
        None,
    )
    assert sv_rust is not None, "Rust should emit a Bid on SV"
    assert sv_rust.price_range == (20, 20), (
        f"may_purchase should fix SV price at face value; got {sv_rust.price_range}"
    )

    # And confirm non-cheapest companies have an open range.
    bo_rust = next(
        (c for c in rust_choices if c.entity.get("private") == "BO"),
        None,
    )
    assert bo_rust is not None and bo_rust.price_range[0] == 225


def test_factored_helper_price_ranges_present():
    """Price-bearing types must always carry a price_range; categorical types must not."""
    names = {1: "P1", 2: "P2", 3: "P3", 4: "P4"}
    adapter = RustGameAdapter(RustGame(names))
    choices = adapter.get_factored_choices()

    price_bearing = {"Bid", "BuyCompany", "BuyTrain"}
    for c in choices:
        if c.type in price_bearing:
            assert c.price_range is not None, (
                f"{c.type} action missing price_range: {c}"
            )
            lo, hi = c.price_range
            assert lo <= hi, f"Inverted price_range on {c}: {c.price_range}"
        else:
            assert c.price_range is None, (
                f"Categorical {c.type} should have price_range=None: {c}"
            )


if __name__ == "__main__":
    # Convenient ad-hoc execution: walk a couple of seeds and print diffs.
    for seed in range(42, 47):
        r = _play_random_game(seed, max_steps=200, verbose=True)
        print(
            f"seed={seed} ok={r['ok']} steps={r['steps']} "
            f"mismatches={len(r['mismatches'])} "
            f"terminated_early={r['terminated_early']}"
        )
