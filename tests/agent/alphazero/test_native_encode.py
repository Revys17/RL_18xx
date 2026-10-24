"""The native leaf encoding (``BaseGame.encode_canonical_py`` / ``mcts._native_encode``
and the lockstep batches) against ``mcts._rust_encode``, the adapter path the
search used per leaf: bit-for-bit on positions of every round type and 2-6
players, and the stacked batch layout of ``engine_rs.encode_games``.

``scripts/check_native_encode.py`` runs the same comparison over human games.
"""

import random

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.mcts import _native_encode, _rust_encode  # noqa: E402
from rl18xx.agent.alphazero.policy_selfplay import _sample_price  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402


def random_positions(num_players: int, seed: int, steps: int = 700, every: int = 3) -> list:
    """Positions along one game of uniformly random legal moves (random legal prices)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    game = engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(num_players)})
    game.set_auction_unlock(True)
    adapter = RustGameAdapter(game)
    positions = []
    for step in range(steps):
        if game.finished:
            break
        legal = [int(i) for i in game.factored_legal_indices()]
        if not legal:
            break
        if step % every == 0:
            positions.append(game.pickle_clone())
        index = rng.choice(legal)
        game.apply_action_index(index, _sample_price(adapter, index, None, np_rng, 0.0))
    return positions


def assert_same_encoding(game) -> None:
    expected = _rust_encode(RustGameAdapter(game))
    got = _native_encode(game)
    assert len(got) == len(expected) == 8
    for i in (0, 1):
        assert got[i].dtype == expected[i].dtype == torch.float32
        assert got[i].shape == expected[i].shape
        # Bit-for-bit: compare the raw float32 bits (NaN-safe, -0.0 != 0.0).
        assert np.array_equal(got[i].numpy().view(np.uint32), expected[i].numpy().view(np.uint32))
    assert got[2] is expected[2] and got[3] is expected[3]  # the shared static edge tensors
    assert tuple(got[4:]) == tuple(expected[4:])


@pytest.mark.parametrize("num_players,seed", [(4, 0), (4, 1), (2, 2), (3, 3), (5, 4), (6, 5)])
def test_native_encoding_is_bit_identical(num_players, seed):
    positions = random_positions(num_players, seed)
    rounds, rotations = set(), set()
    for game in positions:
        assert_same_encoding(game)
        enc = _native_encode(game)
        rounds.add(enc[4])
        rotations.add(enc[6])
    # The walk covers the auction, stock and operating rounds and every seat.
    assert rounds == {0, 1, 2}
    assert rotations == set(range(num_players))


def test_encode_games_stacks_the_same_rows():
    positions = random_positions(4, 7, steps=300, every=10)
    batch = engine_rs.encode_games(positions)
    assert batch["game_states"].shape[0] == len(positions)
    assert batch["num_players"] == 4
    for row, game in enumerate(positions):
        expected = _rust_encode(RustGameAdapter(game))
        assert np.array_equal(batch["game_states"][row], expected[0].numpy()[0])
        assert np.array_equal(batch["node_features"][row], expected[1].numpy())
        assert int(batch["round_types"][row]) == expected[4]
        assert int(batch["rotations"][row]) == expected[6]


def test_encode_games_rejects_mixed_player_counts():
    games = [random_positions(4, 8, steps=3)[0], random_positions(3, 9, steps=3)[0]]
    with pytest.raises(ValueError):
        engine_rs.encode_games(games)


def test_standing_bids_encode_the_same_in_every_game_instance():
    # A player's cash committed to standing auction bids is a float sum; the engine keeps the
    # bids in a HashMap whose order differs between game instances, which made one position
    # (here 45 + 75 + 120 of 600) encode a bit apart in different games. Summed in the
    # auction's company order, every replay of the position encodes it as the adapter does.
    passes = [{"type": "pass", "entity": p, "entity_type": "player"} for p in (2, 3, 4)]
    actions = []
    for company, price in (("CS", 45), ("DH", 75), ("MH", 120)):
        actions += [{"type": "bid", "entity": 1, "entity_type": "player", "company": company, "price": price}]
        actions += passes
    sums = {np.float32(np.float32(np.float32(a) / 600 + np.float32(b) / 600) + np.float32(c) / 600)
            for a, b, c in ((45, 75, 120), (45, 120, 75), (120, 75, 45))}
    assert len(sums) > 1  # the order matters for these bids
    encodings = set()
    for _ in range(20):
        game = engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(4)})
        for action in actions[:-3]:  # player 1's three bids standing, player 2 to act
            game.process_action(dict(action))
        assert_same_encoding(game)
        encodings.add(game.encode_canonical_py()[0].tobytes())
    assert len(encodings) == 1
