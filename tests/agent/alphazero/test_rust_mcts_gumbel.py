"""Gumbel root selection and completed-Q policy targets in the Rust tree
(``engine-rs/src/gumbel.rs``; ``SelfPlayConfig.gumbel_k``).

Rigged values check what the root does with them: the target shifts toward
the better move after 4 simulations, sequential halving spends exactly the
budget (one batch or many, deduplicated or not), with k = 1 the root plays the
argmax of noise + logits, and the target is a distribution over the legal
moves. Searched in lockstep or alone, a Gumbel tree ends identical
(test_lockstep_search.py).
"""

import math

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.config import SelfPlayConfig  # noqa: E402
from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator  # noqa: E402
from rl18xx.agent.alphazero.mcts import POLICY_SIZE  # noqa: E402

from .search_test_utils import (  # noqa: E402
    HashNet,
    RiggedNet,
    expand_root,
    has_open_price,
    make_player,
    random_positions,
)

GUMBEL = dict(mcts_mean_q=True, c_puct_init=0.15, c_puct_by_round={}, dedupe_leaves=True)


def considered_visits_sequence(m: int, n: int) -> list:
    """mctx ``seq_halving.get_sequence_of_considered_visits``, verbatim."""
    if m <= 1:
        return list(range(n))
    log2max = int(math.ceil(math.log2(m)))
    sequence, visits, num_considered = [], [0] * m, m
    while len(sequence) < n:
        extra = max(1, int(n / (log2max * num_considered)))
        for _ in range(extra):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return sequence[:n]


def halving_allocation(m: int, n: int) -> list:
    """The visit counts sequential halving leaves on the m kept moves (sorted)."""
    visits = [0] * m
    for level in considered_visits_sequence(m, n):
        visits[visits.index(level)] += 1
    return sorted(visits, reverse=True)


def _positions(min_legal: int, open_prices: bool = None):
    games = random_positions(4, 0, steps=400, every=5) + random_positions(4, 1, steps=400, every=5)
    out = []
    for g in games:
        legal = [int(i) for i in g.factored_legal_indices()]
        if len(legal) < min_legal:
            continue
        priced = any(has_open_price(g, i) for i in legal)
        if open_prices is None or priced == open_prices:
            out.append(g)
    return out


def test_target_moves_toward_the_better_move_at_n4():
    game = _positions(min_legal=3, open_prices=False)[0]
    net = RiggedNet()
    player = make_player(
        net, game, gumbel_k=2, gumbel_noise_scale=0.0, parallel_readouts=1, num_readouts=4, min_readouts=4, **GUMBEL
    )
    expand_root(player)
    tree = player._rust_player
    legal = list(tree.legal_action_indices_at_root())
    mover = tree.root_active_player_index()
    # Uniform priors and no noise: the two kept moves are the first two legal
    # ones. The second is worth 0.6 to the mover, the first 0.2.
    worse, better = legal[0], legal[1]
    for move, value in ((worse, 0.2), (better, 0.6)):
        net.rig(tree.get_game_for_idx(tree.maybe_add_child(tree.root_idx, move)), mover, value)
    assert player.suggest_move() == better
    stats = tree.gumbel_root_stats()
    assert stats["num_considered"] == 2
    assert sum(stats["search_visits"]) == 4
    target = np.asarray(tree.gumbel_policy_target())
    prior = 1.0 / len(legal)
    assert target[better] > prior > target[worse]
    assert target[better] == target[legal].max()
    # Recorded as the move's policy target.
    player.play_move(better)
    np.testing.assert_array_equal(player.searches_pi[-1], target)


@pytest.mark.parametrize("dedupe", [True, False])
@pytest.mark.parametrize("budget,k,par", [(37, 8, 8), (16, 16, 4), (50, 4, 1), (9, 3, 32)])
def test_sequential_halving_spends_exactly_the_budget(dedupe, budget, k, par):
    net = HashNet()
    games = _positions(min_legal=k)
    checked = 0
    for seed, game in enumerate(games[:6]):
        settings = dict(GUMBEL, dedupe_leaves=dedupe)
        player = make_player(
            net, game, seed=seed, gumbel_k=k, parallel_readouts=par, num_readouts=budget, min_readouts=budget,
            **settings,
        )
        expand_root(player)
        tree = player._rust_player
        n_before = tree.n_at_root()
        player.suggest_move()
        assert tree.n_at_root() - n_before == budget
        stats = tree.gumbel_root_stats()
        visits = np.asarray(stats["search_visits"])
        considered = np.asarray(stats["considered"])
        m = min(k, len(visits))
        assert considered.sum() == m
        assert visits.sum() == budget
        assert visits[~considered].sum() == 0
        assert sorted(visits[considered].tolist(), reverse=True) == halving_allocation(m, budget)
        assert tree.gumbel_remaining() == 0
        checked += 1
    assert checked >= 3


def test_lockstep_gumbel_spends_exactly_the_budget():
    net = HashNet()
    games = _positions(min_legal=6)[:8]
    players = [
        make_player(net, g, seed=s, gumbel_k=6, parallel_readouts=8, num_readouts=30, min_readouts=30, **GUMBEL)
        for s, g in enumerate(games)
    ]
    searches = [p.start_search() for p in players]
    LockstepSearcher({None: ModelEvaluator(net)}).run(searches)
    for p in players:
        # 1 for the root's expansion, then exactly the budget.
        assert p._rust_player.n_at_root() == 31
        assert sum(p._rust_player.gumbel_root_stats()["search_visits"]) == 30


@pytest.mark.parametrize("noise_scale", [1.0, 0.0])
def test_k1_plays_the_argmax_of_noise_plus_logits(noise_scale):
    net = HashNet()
    for seed, game in enumerate(_positions(min_legal=4)[:6]):
        player = make_player(
            net,
            game,
            seed=seed,
            gumbel_k=1,
            gumbel_noise_scale=noise_scale,
            parallel_readouts=4,
            num_readouts=10,
            min_readouts=10,
            **GUMBEL,
        )
        expand_root(player)
        move = player.suggest_move()
        stats = player._rust_player.gumbel_root_stats()
        noise, logits = np.asarray(stats["noise"]), np.asarray(stats["logits"])
        best = int(np.argmax(noise + logits))
        assert move == stats["legal"][best]
        assert stats["search_visits"][best] == 10
        if noise_scale == 0.0:
            assert not noise.any()
            assert best == int(np.argmax(logits))  # the prior's argmax


def test_target_is_a_distribution_over_the_legal_moves():
    net = HashNet()
    games = random_positions(4, 2, steps=300, every=9)
    priced = 0
    for seed, game in enumerate(games):
        player = make_player(
            net, game, seed=seed, gumbel_k=4, parallel_readouts=8, num_readouts=16, min_readouts=16, **GUMBEL
        )
        expand_root(player)
        player.suggest_move()
        tree = player._rust_player
        target = np.asarray(tree.gumbel_policy_target(), dtype=np.float64)
        legal = np.asarray(tree.legal_action_indices_at_root())
        assert target.shape == (POLICY_SIZE,)
        assert (target >= 0).all() and np.isfinite(target).all()
        assert abs(target.sum() - 1.0) < 1e-5
        off = np.ones(POLICY_SIZE, dtype=bool)
        off[legal] = False
        assert not target[off].any()
        # A price-bearing root move aggregates its price grandchildren.
        child_n = np.asarray(tree.child_n_at_root())
        for slot, by_price in tree.price_grandchildren_at_root().items():
            assert sum(by_price.values()) == child_n[slot]
            priced += 1
    assert priced > 0  # the auction's bids were searched with their prices


def test_gumbel_is_off_by_default():
    config = SelfPlayConfig(network=None)
    assert config.gumbel_k == 0
    game = random_positions(4, 3, steps=20, every=10)[0]
    player = make_player(HashNet(), game, parallel_readouts=8, num_readouts=16, min_readouts=16)
    assert not player.gumbel_enabled
    expand_root(player)
    player.suggest_move()
    assert not player._rust_player.gumbel_search_active


def test_eval_player_spec_options():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "eval_head_to_head.py"
    spec = importlib.util.spec_from_file_location("eval_head_to_head", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.resolve_checkpoint = lambda part: Path(part)
    _, overrides = module.parse_player("x.pth@32:gumbel=8,gscale=0,cvisit=25,cscale=0.5", 64)
    assert overrides["gumbel_k"] == 8
    assert overrides["gumbel_noise_scale"] == 0.0
    assert overrides["gumbel_c_visit"] == 25.0
    assert overrides["gumbel_c_scale"] == 0.5
    assert overrides["num_readouts"] == 32
