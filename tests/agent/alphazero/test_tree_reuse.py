"""Tree reuse in lockstep search and search learning's generator
(``RustMCTSPlayer.advance_root_to`` / ``root_log_matches``, ``TreeSearch(budget=)``,
``SearchLearningConfig.reuse_tree``).

- Advancing keeps the chosen child's statistics exactly: the new root's
  per-move visits, values and priors are the child's, its visit count the
  root's count for that move.
- A search continued on a kept tree is the search done without interruption:
  N1 readouts and then a "total" search to N1 + N2 end with the tree of one
  N1 + N2 search, bit for bit; "additional" adds the budget on top.
- ``root_log_matches`` tells a kept tree at the game's position from one that
  left it, and the generator only searches a kept tree at the game's position.
"""

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, TreeSearch  # noqa: E402
from rl18xx.agent.alphazero.policy_selfplay import has_price_choice  # noqa: E402

from .search_test_utils import HashNet, make_player, random_positions, tree_summary  # noqa: E402

VALUE_DRIVEN = dict(
    parallel_readouts=8, num_readouts=40, min_readouts=40, mcts_mean_q=True, c_puct_init=0.15, c_puct_by_round={},
    dedupe_leaves=True, prior_temperature=0.5,
)


def _positions():
    return random_positions(4, 0, steps=260, every=13) + random_positions(3, 1, steps=200, every=25)


def _search(player, net, readouts, budget="additional"):
    searcher = LockstepSearcher({None: ModelEvaluator(net)})
    search = TreeSearch(player, readouts, budget=budget)
    searcher.run([search])
    return search


def _same(a, b):
    sa, sb = tree_summary(a), tree_summary(b)
    assert sa["n"] == sb["n"] and sa["arena"] == sb["arena"] and sa["prices"] == sb["prices"]
    assert np.array_equal(sa["child_n"], sb["child_n"]) and np.array_equal(sa["q"], sb["q"])


def _stats_equal(x, y):
    assert list(x["legal"]) == list(y["legal"]) and x["expanded"] == y["expanded"]
    for key in ("child_n", "child_w", "child_prior"):
        assert np.array_equal(np.asarray(x[key]), np.asarray(y[key])), key
    assert x["price_child_n"] == y["price_child_n"]


def test_advancing_keeps_the_chosen_childs_statistics():
    net = HashNet()
    checked = 0
    for k, game in enumerate(_positions()):
        a = make_player(net, game, seed=k, **VALUE_DRIVEN)
        b = make_player(net, game, seed=k, **VALUE_DRIVEN)
        _search(a, net, 40)
        _search(b, net, 40)
        ta, tb = a._rust_player, b._rust_player
        move = int(ta.pick_best_action())
        price = ta.most_visited_price_for_slot(move)
        visits = (
            ta.price_grandchildren_at_root()[move][price] if price is not None else ta.child_n_at_root()[move]
        )
        child = tb.maybe_add_child(tb.root_idx, move, price)  # the existing child (or grandchild)
        expected = tb.node_stats(child)
        expected_game = tb.get_game_for_idx(child)
        ta.advance_root_to(move, price)
        assert ta.n_at_root() == visits
        _stats_equal(ta.node_stats(ta.root_idx), expected)
        root = ta.root_game_object()
        assert list(root.raw_actions) == list(expected_game.raw_actions)
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize("gumbel", [False, True])
def test_a_continued_search_is_the_uninterrupted_search(gumbel):
    """N1 then a "total" search to N1 + N2 = one N1 + N2 search (PUCT); with a
    Gumbel root, a second search keeps the first one's visits and spends its
    own budget of exactly N2."""
    net = HashNet()
    config = dict(VALUE_DRIVEN, gumbel_k=8) if gumbel else VALUE_DRIVEN
    for k, game in enumerate(_positions()[:12]):
        one = make_player(net, game, seed=k, **config)
        split = make_player(net, game, seed=k, **config)
        _search(one, net, 64)
        first = _search(split, net, 24)
        if gumbel:
            n1 = split._rust_player.n_at_root()
            second = _search(split, net, 40, budget="total")
            assert second.inherited == n1
            if not split._rust_player.is_terminal(split._rust_player.root_idx):
                assert split._rust_player.n_at_root() == n1 + 40
            continue
        assert first.inherited == 0.0
        second = _search(split, net, 64, budget="total")
        assert second.inherited > 0
        _same(one, split)


def test_additional_budget_searches_on_top_of_the_inherited_visits():
    net = HashNet()
    game = _positions()[5]
    player = make_player(net, game, seed=3, **VALUE_DRIVEN)
    _search(player, net, 40)
    n = player._rust_player.n_at_root()
    total = _search(player, net, 16, budget="total")  # already has 1 + 16: no evaluations
    assert total.done and total.inherited == n and player._rust_player.n_at_root() == n
    _search(player, net, 16, budget="additional")
    assert player._rust_player.n_at_root() >= n + 16


def _play_forced(game):
    """Apply single-legal-move positions without an open price (the generator's ``_advance``)."""
    while not game.finished:
        legal = [int(i) for i in game.factored_legal_indices()]
        if len(legal) != 1 or has_price_choice(game, legal[0]):
            return
        lo = game.price_range_for_index(legal[0])
        game.apply_action_index(legal[0], None if lo is None else int(lo[0]))


def test_root_log_matches_tells_the_games_position_from_another():
    net = HashNet()
    matched = mismatched = 0
    for k, game in enumerate(_positions()[:15]):
        player = make_player(net, game, seed=k, **VALUE_DRIVEN)
        _search(player, net, 24)
        tree = player._rust_player
        legal = [int(i) for i in tree.legal_action_indices_at_root()]
        move = int(tree.pick_best_action())
        price = tree.most_visited_price_for_slot(move)
        if price is None:
            pr = game.price_range_for_index(move)
            price = None if pr is None else int(pr[0])
        start = int(tree.root_move_number())
        assert tree.root_log_matches(game, start)
        other = game.pickle_clone()
        played = game.pickle_clone()
        played.apply_action_index(move, price)
        _play_forced(played)
        tree.advance_root_to(move, price)
        if tree.root_log_matches(played, start):
            matched += 1
            assert np.array_equal(
                engine_rs.encode_games([tree.root_game_object()])["game_states"],
                engine_rs.encode_games([played])["game_states"],
            )
        alternatives = [m for m in legal if m != move]
        if alternatives:
            m = alternatives[0]
            pr = other.price_range_for_index(m)
            other.apply_action_index(m, None if pr is None else int(pr[0]))
            _play_forced(other)
            assert not tree.root_log_matches(other, start)
            mismatched += 1
    assert matched >= 10 and mismatched >= 10


@pytest.fixture
def start_file(tmp_path):
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def _settings(start_file, reuse, **overrides):
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", learner_temperature=0.5, prior_temperature=0.5,
        start_positions=start_file, random_start_fraction=1.0, n_full=16, n_fast=8, p_full=0.25,
        games_per_worker=4, max_decisions=40, parallel_readouts=4, reuse_tree=reuse,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return sl.worker_settings(cfg, version=3)


@pytest.mark.parametrize("reuse", ["total", "additional"])
def test_the_generator_searches_kept_trees_only_at_the_games_position(start_file, reuse, monkeypatch):
    gen = sl.Generator(_settings(start_file, reuse), {"learner": ModelEvaluator(HashNet())})
    original = sl.Generator._search
    checked = {"reused": 0}

    def checked_search(self, g, rs, readouts, key):
        search = original(self, g, rs, readouts, key)
        if search.inherited > 0:
            checked["reused"] += 1
            root = engine_rs.encode_games([search.tree.root_game_object()])
            here = engine_rs.encode_games([rs])
            assert np.array_equal(root["game_states"], here["game_states"])
            assert np.array_equal(root["node_features"], here["node_features"])
            assert search.tree.root_move_number() == len(rs.raw_actions)
        return search

    monkeypatch.setattr(sl.Generator, "_search", checked_search)
    out = gen.run(10.0)
    stats = out["stats"]
    assert checked["reused"] > 0 and stats.get("reused_searches", 0) == checked["reused"]
    assert stats["inherited_visits"] > 0 and stats["final_visits"] > stats["inherited_visits"]
    assert out["games"] and all(g["searched"] == len(g["rows"]) for g in out["games"])
    for g in out["games"]:
        for r in g["rows"]:
            assert float(r.target.sum()) == pytest.approx(1.0, abs=1e-5)


def test_total_reuse_spends_fewer_evaluations_per_search(start_file):
    rates = {}
    for reuse in ("off", "total"):
        gen = sl.Generator(_settings(start_file, reuse), {"learner": ModelEvaluator(HashNet())})
        stats = gen.run(8.0)["stats"]
        searches = stats.get("fast_search_decisions", 0) + stats.get("searched_decisions", 0)
        rates[reuse] = stats["leaves"] / max(searches, 1)
    assert rates["total"] < rates["off"]


def test_a_version_change_drops_the_kept_trees(start_file):
    settings = _settings(start_file, "total")
    gen = sl.Generator(settings, {"learner": ModelEvaluator(HashNet())})
    gen.run(4.0)
    newer = dict(settings, version=4)
    stats = gen.run(4.0, newer)["stats"]
    assert stats.get("reuse_dropped_version", 0) > 0
