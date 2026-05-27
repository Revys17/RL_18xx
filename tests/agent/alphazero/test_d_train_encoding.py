"""Tests for the two new D-train depot BuyTrain slots.

The action mapper's legacy ``["depot"]`` slot collapses any depot train to
"buy the cheapest", which silently locked the agent out of choosing a D
over a 6 (or paying the trade-in discounted price) while a 6 was still in
the depot. Two new slots, appended at the END of ``self.actions`` so all
prior indices stay stable, fix that:

  - ``["depot", "D", "full"]``    — buy D at full price (no exchange).
  - ``["depot", "D", "trade-in"]`` — buy D with $300 discount, auto-pick
                                     the lowest-tier 4/5/6 donor.

These tests exercise encode/decode/mask for both slots on a synthetic
phase-6 game state.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.game.engine.actions import BuyTrain
from rl18xx.game.gamemap import GameMap


# ----- helpers ---------------------------------------------------------------


def _make_game(num_players: int = 4):
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    return game_class(players)


def _advance_phase_to_6(game) -> None:
    """Bump the phase index to 6 (the D-train phase). Faster than playing
    through the full game just to surface a D-train in the depot."""
    # Phases: 2, 3, 4, 5, 6, D — so phase 6 is index 4.
    while game.phase.name != "6":
        game.phase.next_phase()


def _force_current_entity(game, entity):
    """Make ``state.current_entity`` return ``entity`` for the duration of the
    test, without polluting the BaseGame class itself.

    We attach a per-instance override using a thin subclass so other tests
    running in the same process don't see a leaked property.
    """
    # Use ``__class__`` swap to a subclass that overrides the property at the
    # class level (the only way to override a property on a Python instance).
    base = type(game)
    holder = type(
        f"_TestPatched_{base.__name__}",
        (base,),
        {"current_entity": property(lambda self: entity)},
    )
    game.__class__ = holder


def _make_phase6_state_with_d():
    """Return ``(game, corp, d_train)`` in a phase-6 state where ``corp``
    has at least one 4, 5, and 6 train and the depot has a D-train ready
    to buy."""
    game = _make_game()
    _advance_phase_to_6(game)

    # Pick any corporation and give it some cash + a 4/5/6 train each so
    # the trade-in path has a donor.
    corp = game.corporations[0]
    corp.cash = 2000

    # Pull a 4, 5, and 6 train out of the bank's stock and assign them to
    # the corporation. Use ``train_by_id`` to find specific train instances.
    for tname in ("4", "5", "6"):
        # Find a train in the global ``game.trains`` list with this name
        # and reassign it to the corp.
        candidate = next(t for t in game.trains if t.name == tname)
        # Detach from current owner (depot) and attach to corp.
        if candidate.owner is not None and hasattr(candidate.owner, "trains"):
            try:
                candidate.owner.trains.remove(candidate)
            except ValueError:
                pass
        if candidate in game.depot.upcoming:
            game.depot.upcoming.remove(candidate)
        candidate.owner = corp
        corp.trains.append(candidate)
    game.depot.depot_trains_cache = None

    # Locate the first D-train in the depot upcoming. Sanity-check.
    d_train = next(t for t in game.depot.upcoming if t.name == "D")
    assert d_train.owner.name == "The Depot"
    assert d_train.name == "D"
    return game, corp, d_train


# ----- tests -----------------------------------------------------------------


def test_action_space_size_includes_new_slots():
    """Action space grew from 26535 to 26537 with the two new D slots at
    the very end of the action list."""
    mapper = ActionMapper()
    assert mapper.action_encoding_size == 26537
    assert mapper.actions[-2] == (BuyTrain, ["depot", "D", "full"])
    assert mapper.actions[-1] == (BuyTrain, ["depot", "D", "trade-in"])
    assert mapper.action_offsets["BuyTrainDFull"] == 26535
    assert mapper.action_offsets["BuyTrainDTradeIn"] == 26536


def test_encode_d_full():
    """A BuyTrain action for a depot D-train at full price (no exchange)
    must map to the D-full slot (26535)."""
    game, corp, d_train = _make_phase6_state_with_d()
    mapper = ActionMapper()
    action = BuyTrain(corp, d_train, d_train.price)
    assert mapper.get_index_for_action(action, game) == 26535


def test_encode_d_trade_in():
    """A BuyTrain action for a depot D-train with an exchange donor (e.g.
    a 4-train) and the discounted $800 price must map to the D-trade-in
    slot (26536)."""
    game, corp, d_train = _make_phase6_state_with_d()
    mapper = ActionMapper()
    donor = next(t for t in corp.trains if t.name == "4")
    discounted_price = d_train.price - 300  # 1830: $300 off for 4/5/6 trade-in
    action = BuyTrain(corp, d_train, discounted_price, exchange=donor)
    assert mapper.get_index_for_action(action, game) == 26536


def test_decode_d_full():
    """Decoding the D-full slot must yield a BuyTrain at face price with
    no exchange and a D-train target."""
    game, corp, d_train = _make_phase6_state_with_d()
    # The mapper decodes ``state.current_entity``-as-actor; force it to
    # the corp by faking the active step's current entity.
    # We rely on the inner ``map_index_to_action`` only reading
    # ``state.depot.depot_trains(entity)`` and ``entity.trains`` so we
    # just monkey-patch ``current_entity``.
    _force_current_entity(game, corp)

    mapper = ActionMapper()
    action = mapper.map_index_to_action(26535, game)
    assert isinstance(action, BuyTrain)
    assert action.train.name == "D"
    assert action.exchange is None
    assert action.price == d_train.price  # 1100


def test_decode_d_trade_in():
    """Decoding the D-trade-in slot must yield a BuyTrain with a 4-train
    donor (the lowest tier owned), and the canonical discounted price."""
    game, corp, d_train = _make_phase6_state_with_d()
    _force_current_entity(game, corp)

    mapper = ActionMapper()
    action = mapper.map_index_to_action(26536, game)
    assert isinstance(action, BuyTrain)
    assert action.train.name == "D"
    assert action.exchange is not None
    assert action.exchange.name == "4"
    # 1830: 1100 - 300 = 800.
    assert action.price == 800


def test_decode_d_trade_in_picks_5_when_no_4():
    """If the corp owns 5/6 but no 4, the donor should be the 5."""
    game, corp, d_train = _make_phase6_state_with_d()
    # Drop the 4-train.
    fours = [t for t in corp.trains if t.name == "4"]
    for t in fours:
        corp.trains.remove(t)
    _force_current_entity(game, corp)

    mapper = ActionMapper()
    action = mapper.map_index_to_action(26536, game)
    assert action.exchange.name == "5"


def test_decode_d_trade_in_raises_when_no_donor():
    """If the corp owns no 4/5/6, decoding the trade-in slot must raise."""
    game, corp, d_train = _make_phase6_state_with_d()
    # Drop all donor candidates.
    corp.trains[:] = [t for t in corp.trains if t.name not in ("4", "5", "6")]
    _force_current_entity(game, corp)

    mapper = ActionMapper()
    with pytest.raises(ValueError, match="No 4/5/6 owned"):
        mapper.map_index_to_action(26536, game)


def test_legal_mask_lights_up_new_slots():
    """In a phase-6 state where the corp can buy a D (with or without
    trade-in), both new slots must be marked legal. With no 4/5/6 owned,
    only the D-full slot lights up."""
    game, corp, d_train = _make_phase6_state_with_d()
    # The legal-action enumeration needs to hit a BuyTrain step. The
    # easiest hook is the factored LegalAction shape, which the mapper
    # routes via ``index_for_factored``. Build LegalActions by hand —
    # this exercises the same code path the engine would, without
    # requiring us to drive the full operating-round flow.
    from rl18xx.game.factored_action_helper import LegalAction

    mapper = ActionMapper()
    la_full = LegalAction(
        type="BuyTrain",
        entity={"corp": corp.name, "source": "depot", "train": "D"},
        price_range=(d_train.price, d_train.price),
    )
    la_trade = LegalAction(
        type="BuyTrain",
        entity={
            "corp": corp.name,
            "source": "depot",
            "train": "D",
            "exchange": "4",
        },
        price_range=(800, 800),
    )

    full_idx = mapper.index_for_factored(la_full, game)
    trade_idx = mapper.index_for_factored(la_trade, game)
    assert full_idx == 26535
    assert trade_idx == 26536

    mask = mapper.convert_indices_to_mask([full_idx, trade_idx])
    assert mask.shape == (26537,)
    assert mask[26535] == 1.0
    assert mask[26536] == 1.0


def test_legal_mask_d_full_only_when_no_donor():
    """Even with no 4/5/6 trains owned the D-full slot must remain legal
    via ``index_for_factored``; the trade-in LegalAction simply won't
    be enumerated upstream when there's no donor."""
    game, corp, d_train = _make_phase6_state_with_d()
    corp.trains[:] = [t for t in corp.trains if t.name not in ("4", "5", "6")]

    from rl18xx.game.factored_action_helper import LegalAction

    mapper = ActionMapper()
    la_full = LegalAction(
        type="BuyTrain",
        entity={"corp": corp.name, "source": "depot", "train": "D"},
        price_range=(d_train.price, d_train.price),
    )
    assert mapper.index_for_factored(la_full, game) == 26535


def test_legacy_pi_padding_in_dataset(tmp_path):
    """A SelfPlayDataset row written with a legacy 26535-wide pi must be
    zero-padded to the current 26537 action size on read."""
    import io
    import lmdb
    import lz4.frame
    import torch
    from rl18xx.agent.alphazero.dataset import SelfPlayDataset

    legacy_size = 26535
    game_state = torch.zeros(1, 16, dtype=torch.float32)
    node_data = torch.zeros(2, 4, dtype=torch.float32)
    edge_index = torch.zeros(2, 1, dtype=torch.long)
    edge_attr = torch.zeros(1, dtype=torch.long)
    state = (game_state, node_data, edge_index, edge_attr)

    legal_actions = torch.tensor([0, 1, 2], dtype=torch.long)
    pi = torch.zeros(legacy_size, dtype=torch.float32)
    pi[0] = 1.0
    value = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    buffer = io.BytesIO()
    torch.save((state, legal_actions, pi, value), buffer)
    compressed = lz4.frame.compress(buffer.getvalue())

    lmdb_dir = tmp_path / "legacy_db"
    env = lmdb.open(str(lmdb_dir), map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"00000000", compressed)
    finally:
        env.close()

    ds = SelfPlayDataset(lmdb_dir)
    _, _, _, pi_out, _, _ = ds[0]
    assert pi_out.shape == (26537,)
    # The padded tail should be zero (no preference for the new D slots).
    assert pi_out[26535].item() == 0.0
    assert pi_out[26536].item() == 0.0
    # The original head slot should still be 1.0.
    assert pi_out[0].item() == 1.0


def test_existing_indices_unchanged():
    """Inserting the new slots at the END must leave every other
    action_offset stable so trained checkpoints stay aligned."""
    mapper = ActionMapper()
    # Spot-check a few prominent offsets.
    assert mapper.action_offsets["Pass"] == 0
    assert mapper.action_offsets["Bid"] == 1
    # Two new offsets sit at the tail and equal the previous total.
    assert mapper.action_offsets["BuyTrainDFull"] == 26535
    assert mapper.action_offsets["BuyTrainDTradeIn"] == 26536
