"""Tests for ``rl18xx.game.factored_action_helper``.

The factored helper is validated in two ways:

1. A structural check on the freshly-initialised 1830 auction state: we know
   exactly which entries should be present and what their price ranges should
   look like.
2. A parity sweep across ~50 random moves: at each game state the legacy
   :class:`ActionHelper` enumeration must be a (modulo-price) subset of the
   :class:`FactoredActionHelper` output. The price dimension collapsing is
   intentional — the factored helper returns one entry per
   (action_type, entity) for price-bearing types.
"""

import random

import pytest

from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.factored_action_helper import (
    FactoredActionHelper,
    LegalAction,
    _categorical_key,
    _factored_key,
    categorical_parity_test,
)
from rl18xx.game.gamemap import GameMap


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_game_1830():
    """A freshly-initialised 4-player 1830 game (waterfall auction)."""
    game_map = GameMap()
    game_cls = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    return game_cls(players)


# ---------------------------------------------------------------------------
# auction-phase structural test
# ---------------------------------------------------------------------------


def test_auction_phase_bid_actions(fresh_game_1830):
    """At game start every private company should appear exactly once as a
    Bid LegalAction with the engine-provided price range, and there should
    be a single Pass entry."""
    helper = FactoredActionHelper()
    choices = helper.get_choices(fresh_game_1830)

    bids = [la for la in choices if la.type == "Bid"]
    passes = [la for la in choices if la.type == "Pass"]

    private_syms = {company.sym for company in fresh_game_1830.companies}
    assert {bid.entity["private"] for bid in bids} == private_syms
    assert len(bids) == 6, f"expected one bid per private, got {len(bids)}"
    assert len(passes) == 1

    # Sanity: every Bid has a price_range tuple with non-negative bounds and
    # min <= max.
    for bid in bids:
        assert bid.price_range is not None
        lo, hi = bid.price_range
        assert isinstance(lo, int) and isinstance(hi, int)
        assert 0 <= lo <= hi
        assert bid.entity["private"] in private_syms

    # SV is the buy-it-now company — its min and max should match its $20 value.
    sv_bid = next(b for b in bids if b.entity["private"] == "SV")
    assert sv_bid.price_range == (20, 20)

    # Player 1 has $600 starting cash; max bid for non-SV privates is that
    # cash minus committed bids = 600 (no bids placed yet).
    cs_bid = next(b for b in bids if b.entity["private"] == "CS")
    assert cs_bid.price_range[0] == 45  # min_bid = value + 5 increment for CS
    assert cs_bid.price_range[1] == 600

    # Pass entity should be the active player.
    assert passes[0].entity == {"player": "Player 1"}


def test_legal_action_dataclass_defaults():
    la = LegalAction(type="Pass")
    assert la.entity == {}
    assert la.params == {}
    assert la.price_range is None


def test_categorical_parity_test_helper_on_fresh_game(fresh_game_1830):
    """The module-level parity helper should pass on a fresh game."""
    assert categorical_parity_test(fresh_game_1830) is True


# ---------------------------------------------------------------------------
# parity sweep across ~50 random moves
# ---------------------------------------------------------------------------


def _factored_keyset(game) -> set:
    return {
        k
        for k in (_factored_key(la) for la in FactoredActionHelper().get_choices(game))
        if k is not None
    }


def _legacy_keyset(game) -> set:
    return {
        k
        for k in (_categorical_key(a) for a in ActionHelper().get_all_choices_limited(game))
        if k is not None
    }


def test_parity_random_walk(fresh_game_1830):
    """Walk a random game for ~50 steps. At every state the legacy helper's
    categorical fingerprints must be a subset of the factored helper's."""
    rng = random.Random(20260520)
    game = fresh_game_1830
    helper = ActionHelper()

    steps = 0
    max_steps = 50  # spec target
    mismatches: list[tuple[int, str, set]] = []

    while steps < max_steps and not game.finished:
        factored_keys = _factored_keyset(game)
        legacy_keys = _legacy_keyset(game)

        missing = legacy_keys - factored_keys
        if missing:
            step_label = game.active_step().__class__.__name__
            mismatches.append((steps, step_label, missing))

        choices = helper.get_all_choices_limited(game)
        if not choices:
            break
        action = rng.choice(choices)
        try:
            game.process_action(action)
        except Exception:
            # If the engine rejects a "legal" action, that's an ActionHelper bug
            # (not a FactoredActionHelper bug) — log and stop the walk.
            break
        steps += 1

    assert not mismatches, (
        "factored helper missed categorical options that ActionHelper emitted: "
        + "\n".join(
            f"  step {i} ({label}): missing {sorted(str(m) for m in missing)}"
            for i, label, missing in mismatches[:5]
        )
    )
    assert steps > 0, "random walk made no progress"


def test_parity_random_walk_seed_b(fresh_game_1830):
    """Second random walk with a different seed — guards against seed-specific
    coincidences."""
    rng = random.Random(424242)
    game = fresh_game_1830
    helper = ActionHelper()

    mismatches: list[tuple[int, str, set]] = []
    for step_idx in range(50):
        if game.finished:
            break
        factored_keys = _factored_keyset(game)
        legacy_keys = _legacy_keyset(game)

        missing = legacy_keys - factored_keys
        if missing:
            mismatches.append((step_idx, game.active_step().__class__.__name__, missing))

        choices = helper.get_all_choices_limited(game)
        if not choices:
            break
        try:
            game.process_action(rng.choice(choices))
        except Exception:
            break

    assert not mismatches, (
        "factored helper missed categorical options on second seed: "
        + "\n".join(
            f"  step {i} ({label}): missing {sorted(str(m) for m in missing)}"
            for i, label, missing in mismatches[:5]
        )
    )


def test_parity_long_walk_covers_operating_round(fresh_game_1830):
    """Longer walk (up to 400 steps) to exercise operating-round step types
    (tile lay, place token, buy train, dividend). At every state the legacy
    helper's categorical fingerprints must be a subset of the factored
    helper's."""
    rng = random.Random(0xC0FFEE)
    game = fresh_game_1830
    helper = ActionHelper()

    mismatches: list[tuple[int, str, set]] = []
    step_classes: set[str] = set()
    factored_types: set[str] = set()

    for step_idx in range(400):
        if game.finished:
            break
        step_classes.add(game.active_step().__class__.__name__)
        factored = FactoredActionHelper().get_choices(game)
        for la in factored:
            factored_types.add(la.type)

        factored_keys = {k for k in (_factored_key(la) for la in factored) if k is not None}
        legacy_keys = _legacy_keyset(game)
        missing = legacy_keys - factored_keys
        if missing:
            mismatches.append((step_idx, game.active_step().__class__.__name__, missing))

        choices = helper.get_all_choices_limited(game)
        if not choices:
            break
        try:
            game.process_action(rng.choice(choices))
        except Exception:
            break

    assert not mismatches, (
        "factored helper missed categorical options on long walk: "
        + "\n".join(
            f"  step {i} ({label}): missing {sorted(str(m) for m in missing)}"
            for i, label, missing in mismatches[:5]
        )
    )
    # Sanity: we should have reached operating-round territory at least.
    assert len(step_classes) >= 3, f"long walk only saw {step_classes}"
