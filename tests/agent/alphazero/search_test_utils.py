"""Shared helpers for the Rust search tests: deterministic stand-in networks and
positions from random play."""

import random
import zlib

import numpy as np
import torch

import engine_rs
from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE
from rl18xx.agent.alphazero.policy_selfplay import _sample_price
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
from rl18xx.agent.alphazero.self_play import _slice_price_components
from rl18xx.rust_adapter import RustGameAdapter


def state_key(encoded) -> int:
    gs = np.ascontiguousarray(encoded[0].numpy(), dtype=np.float32)
    nf = np.ascontiguousarray(encoded[1].numpy(), dtype=np.float32)
    return zlib.crc32(nf.tobytes(), zlib.crc32(gs.tobytes()))


class HashNet:
    """Priors, values and price logits drawn from a generator seeded by the
    encoded state: a deterministic function of the position, as a network is."""

    def __init__(self):
        self.last_price_components = None
        self.calls = 0
        self.states = 0

    def outputs(self, encoded):
        rng = np.random.default_rng(state_key(encoded))
        probs = rng.random(POLICY_SIZE).astype(np.float32) ** 4
        probs /= probs.sum()
        num_players = int(encoded[7])
        value = np.zeros(VALUE_SIZE, dtype=np.float32)
        value[:num_players] = rng.dirichlet(np.ones(num_players)).astype(np.float32)
        price = rng.normal(size=(price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS)).astype(np.float32)
        return probs, value, price

    def run_many_encoded(self, encoded_states, legal_indices=None):
        self.calls += 1
        self.states += len(encoded_states)
        outs = [self.outputs(s) for s in encoded_states]
        self.last_price_components = {
            "price_logits": torch.from_numpy(np.stack([o[2] for o in outs])),
            "slot_index": dict(price_pmf.SLOT_INDEX),
            "num_slots": price_pmf.NUM_SLOTS,
        }
        return (
            torch.from_numpy(np.stack([o[0] for o in outs])),
            None,
            torch.from_numpy(np.stack([o[1] for o in outs])),
        )

    def run_encoded(self, encoded):
        probs, _, values = self.run_many_encoded([encoded])
        return probs[0], None, values[0]


class RiggedNet:
    """Uniform priors and equal values, except for positions given a value with
    :meth:`rig`: what a search should then find is known."""

    def __init__(self):
        self.last_price_components = None
        self.values: dict = {}

    def rig(self, game, seat: int, value: float) -> None:
        """Position ``game`` (an ``engine_rs.BaseGame``) is worth ``value`` to
        seat ``seat`` (absolute, players sorted by id) and the rest of the win
        share to the others equally."""
        from rl18xx.agent.alphazero.mcts import _rust_encode

        encoded = _rust_encode(RustGameAdapter(game))
        rotation, num_players = int(encoded[6]), int(encoded[7])
        canonical = np.zeros(VALUE_SIZE, dtype=np.float32)
        canonical[:num_players] = (1.0 - value) / (num_players - 1)
        canonical[(seat - rotation) % num_players] = value
        self.values[state_key(encoded)] = canonical

    def run_many_encoded(self, encoded_states, legal_indices=None):
        n = len(encoded_states)
        probs = np.full((n, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32)
        values = np.zeros((n, VALUE_SIZE), dtype=np.float32)
        for i, s in enumerate(encoded_states):
            num_players = int(s[7])
            values[i, :num_players] = 1.0 / num_players
            rigged = self.values.get(state_key(s))
            if rigged is not None:
                values[i] = rigged
        self.last_price_components = None
        return torch.from_numpy(probs), None, torch.from_numpy(values)

    def run_encoded(self, encoded):
        probs, _, values = self.run_many_encoded([encoded])
        return probs[0], None, values[0]


def has_open_price(game, index: int) -> bool:
    price_range = game.price_range_for_index(int(index))
    return price_range is not None and price_range[0] != price_range[1]


def random_positions(num_players: int, seed: int, steps: int = 400, every: int = 7) -> list:
    """``engine_rs.BaseGame`` positions along one game of random legal moves."""
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
        if step % every == 0 and len(legal) > 1:
            positions.append(game.pickle_clone())
        index = rng.choice(legal)
        game.apply_action_index(index, _sample_price(adapter, index, None, np_rng, 0.0))
    return positions


def ending_positions(num_players: int, seed: int, last: int = 12) -> list:
    """The positions of the last ``last`` moves of a random game that finishes,
    plus the finished position (searches there reach terminal leaves)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    game = engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(num_players)})
    game.set_auction_unlock(True)
    adapter = RustGameAdapter(game)
    history = []
    for _ in range(20000):
        if game.finished:
            return history[-last:] + [game.pickle_clone()]
        legal = [int(i) for i in game.factored_legal_indices()]
        if len(legal) > 1:
            history.append(game.pickle_clone())
        index = rng.choice(legal)
        game.apply_action_index(index, _sample_price(adapter, index, None, np_rng, 0.0))
    raise RuntimeError(f"random game {seed} did not finish")


def make_player(network, game, seed: int = 0, **overrides) -> RustMCTSPlayer:
    """A seeded Rust-MCTS player at ``game`` (an ``engine_rs.BaseGame``)."""
    settings = dict(
        network=network,
        dirichlet_noise_weight=0.0,
        enable_resign=False,
        noresign_holdout_rate=0.0,
        player_count_distribution={len(game.players): 1.0},
    )
    settings.update(overrides)
    player = RustMCTSPlayer(SelfPlayConfig(**settings))
    player.initialize_game(RustGameAdapter(game.pickle_clone()))
    player._rust_player.seed(seed)
    return player


def expand_root(player) -> None:
    """The eval's ``_expand_root``: one evaluation of a fresh root."""
    leaf = player.root.select_leaf()
    leaf.ensure_encoded()
    probs, _, value = player._backend.run_encoded(leaf.encoded_game_state)
    price = _slice_price_components(getattr(player._backend, "last_price_components", None), 0)
    leaf.incorporate_results(probs, value, leaf, price_components=price)


def tree_summary(player) -> dict:
    tree = player._rust_player
    return {
        "n": float(tree.n_at_root()),
        "child_n": np.asarray(tree.child_n_at_root(), dtype=np.float32),
        "q": np.asarray(tree.root_q_vector(), dtype=np.float32),
        "prices": {k: dict(v) for k, v in tree.price_grandchildren_at_root().items()},
        "arena": int(tree.arena_size()),
    }
