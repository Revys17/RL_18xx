"""Lockstep search (``rl18xx/agent/alphazero/lockstep.py``,
``engine-rs/src/mcts_lockstep.rs``) against ``RustMCTSPlayer.suggest_move``.

With the same seed, network and position, a tree searched in lockstep with
others -- its leaves collected and encoded natively, sent in one request with
other trees' leaves, results scattered back in Rust -- ends identical to the
same tree searched alone through ``tree_search`` (visit counts, values, price
grandchildren), under the default search and the value-driven options.
"""

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, PolicyRequest  # noqa: E402
from rl18xx.agent.alphazero.mcts import _rust_encode  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from .search_test_utils import (  # noqa: E402
    HashNet,
    ending_positions,
    expand_root,
    make_player,
    random_positions,
    tree_summary,
)

DEFAULT = dict(parallel_readouts=8, num_readouts=40, min_readouts=40)
VALUE_DRIVEN = dict(
    DEFAULT, mcts_mean_q=True, c_puct_init=0.15, c_puct_by_round={}, dedupe_leaves=True, prior_temperature=0.5
)
GUMBEL = dict(VALUE_DRIVEN, gumbel_k=8)


def _positions():
    # The auction (price grandchildren under bids), stock and operating rounds,
    # 3 and 4 players.
    return random_positions(4, 0, steps=260, every=13) + random_positions(3, 1, steps=200, every=25)


def _alone(net, game, seed, config):
    player = make_player(net, game, seed=seed, **config)
    expand_root(player)
    player.suggest_move()
    return player


def _assert_same_tree(a, b):
    sa, sb = tree_summary(a), tree_summary(b)
    assert sa["n"] == sb["n"]
    assert np.array_equal(sa["child_n"], sb["child_n"])
    assert np.array_equal(sa["q"], sb["q"])
    assert sa["prices"] == sb["prices"]
    assert sa["arena"] == sb["arena"]


@pytest.mark.parametrize("config", [DEFAULT, VALUE_DRIVEN, GUMBEL], ids=["default", "value_driven", "gumbel"])
def test_lockstep_trees_match_trees_searched_alone(config):
    net = HashNet()
    positions = _positions()
    assert len(positions) >= 20
    alone = [_alone(net, game, k, config) for k, game in enumerate(positions)]
    # All trees in one searcher; two network keys (the same network) and two
    # player counts, so a round sends several requests.
    players = [make_player(net, game, seed=k, **config) for k, game in enumerate(positions)]
    searches = [p.start_search(key=k % 2) for k, p in enumerate(players)]
    searcher = LockstepSearcher({0: ModelEvaluator(net), 1: ModelEvaluator(net)})
    searcher.run(searches)
    assert all(s.done for s in searches)
    for a, b in zip(alone, players):
        _assert_same_tree(a, b)
        if config.get("gumbel_k"):
            assert a.pick_move() == b.pick_move()
            assert np.array_equal(a._rust_player.gumbel_policy_target(), b._rust_player.gumbel_policy_target())
        else:
            assert a._rust_player.pick_best_action() == b._rust_player.pick_best_action()
    # Leaves of many trees per request: fewer requests than trees x rounds.
    assert searcher.stats.requests < searcher.stats.rounds * len(players)


def test_lockstep_play_continues_on_the_reused_subtree():
    """Two moves in a row by the same tree (the eval keeps a seat's tree across
    its own consecutive moves): the second search starts from the subtree."""
    net = HashNet()
    game = random_positions(4, 2, steps=120, every=40)[-1]
    a = make_player(net, game, seed=3, **VALUE_DRIVEN)
    b = make_player(net, game, seed=3, **VALUE_DRIVEN)
    expand_root(a)
    searcher = LockstepSearcher({None: ModelEvaluator(net)})
    for _ in range(2):
        a.suggest_move()
        searcher.run([b.start_search()])
        _assert_same_tree(a, b)
        move = a._rust_player.pick_best_action()
        assert b._rust_player.pick_best_action() == move
        a.play_move(move)
        b.play_move(move)
        if a._rust_player.num_legal_at_root() <= 1:
            break


def test_policy_requests_get_the_networks_outputs():
    net = HashNet()
    games = random_positions(4, 4, steps=100, every=20)
    requests = [PolicyRequest(g, [int(i) for i in g.factored_legal_indices()]) for g in games]
    player = make_player(net, games[0], seed=0, **DEFAULT)
    searcher = LockstepSearcher({None: ModelEvaluator(net)})
    searcher.round([player.start_search()], requests)
    for game, r in zip(games, requests):
        probs, value, price = net.outputs(_rust_encode(RustGameAdapter(game)))
        np.testing.assert_array_equal(r.priors, probs[r.legal])
        np.testing.assert_array_equal(r.value, value)
        np.testing.assert_array_equal(r.price_row, price)
    # A round of requests alone (no leaves for that network's trees).
    alone = [PolicyRequest(g, [int(i) for i in g.factored_legal_indices()]) for g in games]
    finished = make_player(net, games[0], seed=0, **DEFAULT)
    searcher.round([finished.start_search()], [])
    searcher.round([], alone)
    assert all(r.done for r in alone)


@pytest.mark.parametrize("use_score_values", [True, False])
def test_native_terminal_value_matches_python(use_score_values):
    from rl18xx.agent.alphazero.mcts import terminal_backup_value

    for seed in (0, 1, 4):
        game = ending_positions(4, seed)[-1]
        assert game.finished
        player = make_player(HashNet(), game, use_score_values=use_score_values, **DEFAULT)
        expected = player._compute_terminal_value(player._rust_player.root_idx)
        if use_score_values:
            expected = terminal_backup_value(expected, len(game.players))
        got = np.asarray(engine_rs.mcts_terminal_value(game, use_score_values), dtype=np.float32)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("config", [DEFAULT, VALUE_DRIVEN, GUMBEL], ids=["default", "value_driven", "gumbel"])
def test_lockstep_matches_alone_near_the_end(config):
    """Positions a few moves from the end: searches reach finished games, whose
    values both paths back up without the network."""
    net = HashNet()
    positions = [g for g in ending_positions(4, 1) + ending_positions(4, 4) if not g.finished]
    alone = [_alone(net, game, k, config) for k, game in enumerate(positions)]
    players = [make_player(net, game, seed=k, **config) for k, game in enumerate(positions)]
    searcher = LockstepSearcher({None: ModelEvaluator(net)})
    searcher.run([p.start_search() for p in players])
    assert searcher.stats.terminal_backups > 0
    for a, b in zip(alone, players):
        _assert_same_tree(a, b)


def test_tree_keeps_the_whole_action_log():
    """Tree nodes log only the actions since the root (cloning full logs was most
    of an expansion's cost late in a game); the games the tree hands out carry
    the whole log, which replays to the same position."""
    from rl18xx.agent.alphazero.mcts import _native_encode

    net = HashNet()
    game = random_positions(4, 6, steps=200, every=20)[-1]
    start_log = list(game.raw_actions)
    player = make_player(net, game, seed=1, **VALUE_DRIVEN)
    searcher = LockstepSearcher({None: ModelEvaluator(net)})
    played = []
    for _ in range(6):
        search = player.start_search()
        if search is not None:
            searcher.run([search])
            move = player._rust_player.pick_best_action()
        else:
            move = int(player._rust_player.legal_action_indices_at_root()[0])
        player.play_move(move)
        played.append(player.played_actions[-1])
        played.extend(player.forced_action_dicts[-1])
    state = player.get_game_state()
    log = list(state._game.raw_actions)  # the adapter numbers them (``id``)
    assert start_log and len(log) >= len(start_log) + 6
    assert log == start_log + played
    assert player._rust_player.root_move_number() == len(log)
    replay = engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(4)})
    replay.set_auction_unlock(True)
    for action in log:
        replay.process_action(dict(action))
    expected, got = _native_encode(replay), _native_encode(state._game)
    assert np.array_equal(expected[0].numpy(), got[0].numpy())
    assert np.array_equal(expected[1].numpy(), got[1].numpy())
