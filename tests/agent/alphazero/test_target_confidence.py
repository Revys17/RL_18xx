"""Confidence-filtered policy targets (``search_targets.departure_confidence`` /
``confident_target``, ``SearchLearningConfig.target_confidence``) and the per-move value
variance they read from the Rust tree (``root_target_stats()["child_q_var"]``)."""

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero import search_targets as st  # noqa: E402

from .search_test_utils import HashNet, expand_root, make_player, random_positions  # noqa: E402
from .test_search_learning import _generate, _settings, start_file  # noqa: E402,F401

PUCT = dict(mcts_mean_q=True, c_puct_init=0.05, c_puct_by_round={}, dedupe_leaves=True, prior_temperature=0.5)


def _games(count=6, min_legal=3):
    games = random_positions(4, 4, steps=400, every=7) + random_positions(4, 5, steps=400, every=7)
    return [g for g in games if len(g.factored_legal_indices()) >= min_legal][:count]


# ----------------------------------------------------------------------------- the tree's variance


def test_child_q_var_is_the_variance_of_the_values_backed_up_through_each_move():
    game = _games(count=1, min_legal=4)[0]
    player = make_player(HashNet(), game, parallel_readouts=1, num_readouts=8, min_readouts=8,
                         softpick_move_cutoff=0, **PUCT)
    expand_root(player)
    tree = player._rust_player
    mover = tree.root_active_player_index()
    legal = list(tree.legal_action_indices_at_root())
    rng = np.random.default_rng(0)
    backed = {}
    for k, move in enumerate(legal[:3]):
        child = tree.maybe_add_child(tree.root_idx, move)
        values = []
        for _ in range(k + 1 if k < 2 else 7):  # 1, 2 and 7 backups
            v = np.zeros(6, dtype=np.float32)
            v[:4] = rng.dirichlet(np.ones(4)).astype(np.float32)
            tree.backup_value(child, v.tolist())
            values.append(float(v[mover]))
        backed[k] = np.array(values, dtype=np.float64)
    stats = st.root_stats(tree)
    for k, values in backed.items():
        assert stats["child_n"][k] == len(values)
        assert stats["child_q"][k] == pytest.approx(values.mean(), rel=1e-5)
        assert stats["child_q_var"][k] == pytest.approx(values.var(), abs=1e-6)  # population variance
    assert np.isnan(stats["child_q_var"][3:]).all()  # unvisited


def test_child_q_var_follows_searches_and_kept_subtrees():
    """After a real search every visited move has a finite, non-negative variance (0 with
    one visit); after advancing, the new root's variances are its own moves'."""
    net = HashNet()
    for seed, game in enumerate(_games(count=4)):
        player = make_player(net, game, seed=seed, parallel_readouts=8, num_readouts=64, min_readouts=64,
                             softpick_move_cutoff=0, **PUCT)
        expand_root(player)
        player.suggest_move()
        tree = player._rust_player
        stats = st.root_stats(tree)
        visited = stats["child_n"] > 0
        assert (stats["child_q_var"][visited] >= 0).all() and np.isnan(stats["child_q_var"][~visited]).all()
        assert (stats["child_q_var"][stats["child_n"] == 1] < 1e-7).all()
        assert st.pooled_q_var(stats) >= 0
        move = int(tree.pick_best_action())
        tree.advance_root_to(move, tree.most_visited_price_for_slot(move))
        if tree.is_expanded(tree.root_idx) and not tree.is_terminal(tree.root_idx):
            after = st.root_stats(tree)
            seen = after["child_n"] > 0
            assert np.isfinite(after["child_q_var"][seen]).all()


# ----------------------------------------------------------------------------- confidence


def _stats(n, q, var, prior):
    n, q = np.asarray(n, dtype=np.float64), np.asarray(q, dtype=np.float64)
    return {
        "child_n": n, "child_q": np.where(n > 0, q, np.nan), "completed_q": q,
        "child_q_var": np.where(n > 0, np.asarray(var, dtype=np.float64), np.nan),
        "child_prior": np.asarray(prior, dtype=np.float64),
    }


def test_departure_confidence_grows_with_the_gap_and_the_visits():
    prior = [0.7, 0.2, 0.1]
    target = np.array([0.2, 0.7, 0.1])  # departs: move 1 over the policy's move 0
    agree = np.array([0.8, 0.1, 0.1])
    s = _stats([40, 40, 0], [0.25, 0.27, 0.25], [0.002, 0.002, 0], prior)
    assert st.departure_confidence(s, agree) == float("inf")
    z = st.departure_confidence(s, target, prior_weight=0.0)
    assert z == pytest.approx(0.02 / np.sqrt(0.002 / 40 * 2))
    more = st.departure_confidence(_stats([160, 160, 0], [0.25, 0.27, 0.25], [0.002, 0.002, 0], prior), target, 0.0)
    assert more == pytest.approx(2 * z)
    wider = st.departure_confidence(_stats([40, 40, 0], [0.25, 0.29, 0.25], [0.002, 0.002, 0], prior), target, 0.0)
    assert wider == pytest.approx(2 * z)
    worse = st.departure_confidence(_stats([40, 40, 0], [0.27, 0.25, 0.25], [0.002, 0.002, 0], prior), target, 0.0)
    assert worse == pytest.approx(-z)
    # An unvisited move: no evidence.
    assert st.departure_confidence(_stats([40, 0, 2], [0.25, 0.3, 0.25], [0.002, 0, 0.001], prior), target) == 0.0


def test_one_or_two_visits_borrow_the_roots_pooled_variance():
    prior = [0.7, 0.2, 0.1]
    target = np.array([0.2, 0.7, 0.1])
    s = _stats([60, 1, 3], [0.25, 0.30, 0.20], [0.003, 0.0, 0.001], prior)
    pooled = st.pooled_q_var(s)
    n, q, v = s["child_n"], s["completed_q"], np.nan_to_num(s["child_q_var"])
    mean = (n * q).sum() / n.sum()
    assert pooled == pytest.approx((n * (v + q * q)).sum() / n.sum() - mean**2)
    raw = st.departure_confidence(s, target, prior_weight=0.0)  # a single visit's variance is 0
    shrunk = st.departure_confidence(s, target, prior_weight=2.0)
    var_t = (1 * 0.0 + 2 * pooled) / 3
    var_p = (60 * 0.003 + 2 * pooled) / 62
    assert shrunk == pytest.approx(0.05 / np.sqrt(var_t / 1 + var_p / 60))
    assert shrunk < raw


def test_confident_target_keeps_strong_departures_and_shrinks_the_rest():
    prior = np.array([0.7, 0.2, 0.1])
    target = np.array([0.2, 0.7, 0.1])
    s = _stats([40, 40, 0], [0.25, 0.27, 0.25], [0.002, 0.002, 0], prior)
    z = st.departure_confidence(s, target)
    kept, z_kept = st.confident_target(s, target, threshold=z - 0.1)
    assert z_kept == pytest.approx(z) and np.array_equal(kept, target)
    reverted, _ = st.confident_target(s, target, threshold=z + 0.1)
    np.testing.assert_allclose(reverted, prior)
    soft, _ = st.confident_target(s, target, threshold=2 * z, mode="soft")
    np.testing.assert_allclose(soft, prior + 0.5 * (target - prior))
    assert soft.sum() == pytest.approx(1.0)
    agree = np.array([0.9, 0.05, 0.05])
    same, z_agree = st.confident_target(s, agree, threshold=100.0)
    assert z_agree == float("inf") and np.array_equal(same, agree)
    with pytest.raises(ValueError):
        st.confident_target(s, target, 1.0, mode="nope")
    # A distrusted departure shrinks to the policy whatever its confidence; agreement never shrinks.
    distrusted, _ = st.confident_target(s, target, threshold=-1e9, distrust=True)
    np.testing.assert_allclose(distrusted, prior)
    same, _ = st.confident_target(s, agree, threshold=1e9, distrust=True)
    assert same is agree


def test_the_share_signal_is_the_departures_share_of_the_two_moves_visits():
    prior = np.array([0.7, 0.2, 0.1])
    target = np.array([0.2, 0.7, 0.1])
    s = _stats([30, 10, 5], [0.25, 0.27, 0.2], [0.002, 0.002, 0.001], prior)
    assert st.departure_share(s, target) == pytest.approx(10 / 40)
    assert st.departure_share(s, np.array([0.8, 0.1, 0.1])) == float("inf")
    assert st.departure_signal(s, target, "share") == pytest.approx(0.25)
    kept, _ = st.confident_target(s, target, 0.2, signal="share")
    assert kept is target
    shrunk, _ = st.confident_target(s, target, 0.5, signal="share")
    np.testing.assert_allclose(shrunk, prior)
    with pytest.raises(ValueError):
        st.departure_signal(s, target, "nope")


def test_a_real_roots_departures_are_scored():
    net = HashNet()
    scored = 0
    for seed, game in enumerate(_games(count=8)):
        player = make_player(net, game, seed=seed, parallel_readouts=8, num_readouts=48, min_readouts=48,
                             softpick_move_cutoff=0, **PUCT)
        expand_root(player)
        player.suggest_move()
        stats = st.root_stats(player._rust_player)
        for c_scale in (1.0, 3.0, 30.0):
            target = st.completed_q_target(stats, 50.0, c_scale)
            z = st.departure_confidence(stats, target)
            out, _ = st.confident_target(stats, target, 1.0)
            assert out.sum() == pytest.approx(1.0) and (out >= 0).all()
            departs = int(np.argmax(target)) != int(np.argmax(stats["child_prior"]))
            assert np.isfinite(z) == departs
            scored += departs
    assert scored > 0


# ----------------------------------------------------------------------------- search learning


def test_generator_rows_keep_or_shrink_departures(start_file):
    base = dict(target="cq", n_full=8, critic_positions_per_game=0, gumbel_c_scale=3.0)
    settings = _settings(start_file, **base, target_confidence=1e9)  # every departure shrinks to the policy
    gen, out = _generate(settings)
    rows = [r for g in out["games"] for r in g["rows"]]
    stats = out["stats"]
    assert rows and stats["confidence_rows"] >= len(rows)  # games still in progress hold rows too
    assert stats["confidence_departures"] == stats["confidence_shrunk"]
    for r in rows:
        assert float(r.target.sum()) == pytest.approx(1.0, abs=1e-5)
    # With the threshold at -inf nothing shrinks; without the option nothing is counted.
    settings = _settings(start_file, **base, target_confidence=-1e9)
    _, out = _generate(settings)
    assert out["stats"].get("confidence_shrunk", 0) == 0 and out["stats"]["confidence_rows"] > 0
    _, out = _generate(_settings(start_file, **base))
    assert "confidence_rows" not in out["stats"]


def test_shrunk_rows_hold_the_policys_distribution(start_file, monkeypatch):
    """A shrunk row's target is the tree's priors (the policy at T) over its legal moves."""
    seen = []
    original = st.confident_target

    def spy(stats, target, threshold, mode="prior", prior_weight=2.0, signal="z", distrust=False):
        out, z = original(stats, target, threshold, mode, prior_weight, signal, distrust)
        seen.append((stats["child_prior"] / stats["child_prior"].sum(), target, out, z))
        return out, z

    monkeypatch.setattr(sl, "confident_target", spy)
    settings = _settings(start_file, target="cq", n_full=8, critic_positions_per_game=0, gumbel_c_scale=3.0,
                         target_confidence=1e9)
    _, out = _generate(settings)
    rows = [r for g in out["games"] for r in g["rows"]]
    assert rows and seen  # only departures reach confident_target
    kept_targets = {}
    for prior, target, kept, z in seen:
        assert np.isfinite(z) and int(np.argmax(target)) != int(np.argmax(prior))
        np.testing.assert_allclose(kept, prior, atol=1e-12)  # the threshold is never met
        kept_targets[kept.astype(np.float32).tobytes()] = True
    # Rows of finished games: a departure's row holds the policy; any other row agrees with it.
    shrunk = [r for r in rows if r.target.astype(np.float32).tobytes() in kept_targets]
    assert shrunk and len(shrunk) <= len(seen)


def test_decision_steps_classify_by_round_and_move_types():
    """A stock round's decisions are "stock"; elsewhere the first of tile lay (track), token,
    routes, dividend, train purchase among the legal moves -- the step types of
    ``scripts/diag_sibling_values.py`` (which now calls ``decision_step``)."""
    seen = set()
    for game in random_positions(4, 11, steps=600, every=3):
        legal = [int(x) for x in game.factored_legal_indices()]
        if len(legal) < 2:
            continue
        step = sl.decision_step(game, legal)
        types = {game.decode_index_to_map(ix, None).get("type") for ix in legal}
        assert step in sl.STEP_TYPES
        if game.round.round_type == "Stock":
            assert step == "stock"
        elif "lay_tile" in types:
            assert step == "track"
        elif step == "train":
            assert "buy_train" in types and not {"place_token", "run_routes", "dividend"} & types
        seen.add(step)
    assert {"stock", "track"} <= seen


def test_generator_distrusts_departures_in_the_named_steps(start_file, monkeypatch):
    calls = []
    original = sl.decision_step

    def spy(rs, legal):
        step = original(rs, legal)
        calls.append(step)
        return step

    monkeypatch.setattr(sl, "decision_step", spy)
    # Every step distrusted: every departure shrinks; no threshold set.
    settings = _settings(start_file, target="cq", n_full=8, critic_positions_per_game=0, gumbel_c_scale=3.0,
                         target_distrust_steps=tuple(sl.STEP_TYPES))
    _, out = _generate(settings)
    stats = out["stats"]
    assert stats["confidence_departures"] > 0 and len(calls) == stats["confidence_departures"]
    assert stats["confidence_shrunk"] == stats["confidence_distrusted"] == stats["confidence_departures"]
    # No step distrusted and no threshold: the option is off, nothing is read.
    calls.clear()
    _, out = _generate(_settings(start_file, target="cq", n_full=8, critic_positions_per_game=0))
    assert not calls and "confidence_rows" not in out["stats"]


def test_config_validation_and_worker_settings():
    cfg = sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="", target="cq", target_confidence=2.0,
                                  target_confidence_mode="soft")
    settings = sl.worker_settings(cfg, 0)
    assert settings["target_confidence"] == 2.0 and settings["target_confidence_mode"] == "soft"
    assert sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="").target_confidence is None
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="", target_confidence_mode="x"))
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="",
                                             target_confidence_signal="nope"))
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="",
                                             target_distrust_steps=("tracks",)))
    cfg = sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="", target_distrust_steps=("track",),
                                  target_confidence_signal="share")
    settings = sl.worker_settings(cfg, 0)
    assert settings["target_distrust_steps"] == ("track",) and settings["target_confidence_signal"] == "share"
