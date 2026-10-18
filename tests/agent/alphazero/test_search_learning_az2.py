"""az2's search-learning options (docs/search_learning_design.md, "az2"): one shared
network, n-step value targets from the search's root values, the blended search
value on the shared path, auxiliary outcome heads, KataGo's policy surprise
weighting, Gumbel targets at small budgets, and the value-head refit."""

import copy
import json
import math

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero import value_refit as vr  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE, _rust_encode  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from .search_test_utils import HashNet, ending_positions, random_positions  # noqa: E402
from .test_search_learning import _row, _settings, _tiny_model  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture
def start_file(tmp_path):
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


class AbsoluteValueNet:
    """Uniform priors; every position is worth ``value`` per absolute seat (the
    canonical reply is that vector rotated to the mover's frame), so a search's
    root value is known exactly."""

    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.last_price_components = None

    def run_many_encoded(self, encoded_states, legal_indices=None):
        n = len(encoded_states)
        probs = np.full((n, POLICY_SIZE), 1.0 / POLICY_SIZE, dtype=np.float32)
        values = np.zeros((n, VALUE_SIZE), dtype=np.float32)
        for i, s in enumerate(encoded_states):
            rotation, players = int(s[6]), int(s[7])
            values[i, :players] = np.roll(self.value[:players], -rotation)
        return torch.from_numpy(probs), None, torch.from_numpy(values)


def _generator(settings, net):
    return sl.Generator(settings, {"learner": ModelEvaluator(net)})


def _random_value_heads(model, seed):
    """Non-zero value-head outputs (the heads' output layers start at zero)."""
    g = torch.Generator().manual_seed(seed)
    for head in (model.win_loss_head, model.score_head):
        with torch.no_grad():
            head[-1].weight.copy_(torch.randn(head[-1].weight.shape, generator=g) * 0.5)
            head[-1].bias.zero_()
    return model


# ----------------------------------------------------------------------------- n-step value targets


def test_nstep_target_mixes_the_outcome_with_the_root_value_n_searches_later():
    outcome = np.array([0.0, 1.0, 0.0, 0.0])
    values = [
        np.full(4, 0.25),
        np.array([0.1, 0.6, 0.2, 0.1]),
        np.array([0.2, 0.2, 0.2, 0.4]),
        np.array([0, 1, 0.5, 0.5]),
    ]
    t = sl.nstep_win_target(outcome, values, 0, 2, 0.5)
    np.testing.assert_allclose(t, 0.5 * outcome + 0.5 * values[2], atol=1e-6)
    t = sl.nstep_win_target(outcome, values, 1, 2, 0.5)  # values[3] normalized: 0 / 0.5 / 0.25 / 0.25
    np.testing.assert_allclose(t, [0.0, 0.75, 0.125, 0.125], atol=1e-6)
    np.testing.assert_allclose(sl.nstep_win_target(outcome, values, 1, 2, 1.0), [0, 0.5, 0.25, 0.25], atol=1e-6)
    for index, n, w in ((2, 2, 0.5), (3, 1, 0.5), (0, 0, 0.5), (0, 2, 0.0), (-1, 1, 0.5)):
        np.testing.assert_allclose(sl.nstep_win_target(outcome, values, index, n, w), outcome)  # the outcome alone
    assert float(sl.nstep_win_target(outcome, values, 0, 1, 0.3).sum()) == pytest.approx(1.0)


def test_lambda_returns_over_a_games_searches():
    z = np.array([1.0, 0.0, 0.0, 0.0])
    v = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.2, 0.2, 0.2]), np.array([0.0, 0.0, 2.0, 2.0])]
    lam = 0.6
    g2 = z
    g1 = (1 - lam) * v[2] / 4.0 + lam * g2
    g0 = (1 - lam) * v[1] + lam * g1
    returns = sl.lambda_returns(z, v, lam)
    for got, want in zip(returns, (g0, g1, g2)):
        np.testing.assert_allclose(got, want, atol=1e-9)
    np.testing.assert_allclose(sl.lambda_returns(z, v, 1.0)[0], z)  # lambda 1: the outcome
    np.testing.assert_allclose(sl.lambda_returns(z, v, 0.0)[0], v[1])  # lambda 0: the next search's value
    np.testing.assert_allclose(sl.lambda_win_target(z, returns, 0, 0.5), 0.5 * z + 0.5 * g0, atol=1e-6)
    np.testing.assert_allclose(sl.lambda_win_target(z, returns, -1, 0.5), z)


def test_generated_rows_take_lambda_targets(start_file):
    v_abs = np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float32)
    settings = _settings(start_file, p_full=1.0, n_full=6, games_per_worker=2, max_decisions=20)
    settings.update(value_lambda=0.5, value_bootstrap_weight=1.0)
    games = [g for g in _generator(settings, AbsoluteValueNet(v_abs)).run(6.0)["games"] if g["rows"]]
    assert games
    for g in games:
        rows = g["rows"]
        z = sl.share_of_winners(rows[0].fractions)
        expected = sl.lambda_returns(z, [r.root_value for r in rows], 0.5)
        for r, want in zip(rows, expected):
            np.testing.assert_allclose(r.win_target, want, atol=1e-5)
        # Every root value is v_abs: G_i = v_abs + 0.5 ** (searches after i) * (z - v_abs).
        k = len(rows) - 1 - rows[0].search_index
        np.testing.assert_allclose(rows[0].win_target, v_abs + 0.5**k * (z - v_abs), atol=1e-5)


def test_rows_carry_the_searchs_root_value_in_absolute_seat_order_and_nstep_targets(start_file):
    v_abs = np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float32)
    settings = _settings(start_file, p_full=1.0, n_full=6, games_per_worker=2, max_decisions=20)
    settings.update(value_nstep=3, value_bootstrap_weight=0.4)
    gen = _generator(settings, AbsoluteValueNet(v_abs))
    out = gen.run(6.0)
    games = [g for g in out["games"] if g["rows"]]
    assert games
    for g in games:
        rows = g["rows"]
        assert [r.search_index for r in rows] == list(range(len(rows)))  # every decision searched (no fast searches)
        z = sl.share_of_winners(rows[0].fractions)
        for r in rows:
            np.testing.assert_allclose(r.root_value, v_abs, atol=1e-5)  # absolute order, every leaf's value
            expected = 0.6 * z + 0.4 * v_abs if r.search_index + 3 < len(rows) else z
            np.testing.assert_allclose(r.win_target, expected, atol=1e-5)
        ce_sum, ce_rows = g["root_value_ce"]
        assert ce_rows == len(rows)
        assert ce_sum / ce_rows == pytest.approx(float(-(z * np.log(v_abs)).sum()), rel=1e-4)


def test_fast_searches_count_as_searched_decisions_for_the_bootstrap(start_file):
    settings = _settings(start_file, p_full=0.3, n_full=4, parallel_readouts=2, games_per_worker=3, max_decisions=40)
    settings["n_fast"] = 2
    gen = _generator(settings, HashNet())
    gen.run(6.0)
    seen_gap = False
    for g in gen.games.values():
        indices = [r.search_index for r in g.rows]
        assert indices == sorted(set(indices)) and len(g.search_values) >= len(g.rows)
        for r in g.rows:
            assert r.root_value is g.search_values[r.search_index]
            assert float(r.root_value.sum()) == pytest.approx(1.0, abs=1e-4)
        seen_gap |= len(g.search_values) > len(g.rows)
    assert seen_gap


def test_critic_positions_per_game_zero_flags_every_row(start_file):
    gen = _generator(_settings(start_file, critic_positions_per_game=0, p_full=1.0), HashNet())
    games = [g for g in gen.run(5.0)["games"] if g["rows"]]
    assert games and all(r.critic for g in games for r in g["rows"])


def test_value_losses_agree_with_critic_losses_on_outcome_targets():
    from rl18xx.agent.alphazero.policy_gradient import critic_losses

    rng = np.random.default_rng(0)
    fractions = rng.dirichlet(np.ones(4), size=6).astype(np.float32)
    fractions[0] = [0.3, 0.3, 0.2, 0.2]  # a tie
    win_logits = torch.randn(6, VALUE_SIZE)
    win_logits[:, 4:] = -1e4  # padded seats, as the forward masks them
    score = torch.randn(6, VALUE_SIZE)
    score[:, 4:] = 0.0
    score_target = torch.rand(6, 4)
    win = torch.from_numpy(np.stack([sl.share_of_winners(f) for f in fractions]))
    v = sl.value_losses(win_logits, score, win, score_target)
    c = critic_losses(win_logits, score, torch.from_numpy(fractions), score_target)
    for key in ("ce", "mse", "winner_hit"):
        assert float(v[key]) == pytest.approx(float(c[key]), rel=1e-5)
    assert float(v["outcome_ce"]) == pytest.approx(float(c["ce"]), rel=1e-5)
    # With another target the outcome CE stays against the outcome.
    other = torch.full((6, 4), 0.25)
    v2 = sl.value_losses(win_logits, score, other, score_target, outcome_win=win)
    assert float(v2["outcome_ce"]) == pytest.approx(float(c["ce"]), rel=1e-5)
    assert float(v2["ce"]) != pytest.approx(float(c["ce"]), rel=1e-3)


# ----------------------------------------------------------------------------- surprise weighting


def test_surprise_weights_formula_cap_and_mean_one():
    s = torch.tensor([0.0, 0.1, 0.1, 2.2])
    w, capped = sl.surprise_weights(s, 0.5, cap=100.0)
    raw = 0.5 + 0.5 * s / s.mean()  # mean exactly 1 already
    assert torch.allclose(w, raw) and float(capped) == 0
    w, capped = sl.surprise_weights(s, 0.5, cap=1.5)
    clipped = raw.clamp(max=1.5)
    assert torch.allclose(w, clipped * 4 / clipped.sum()) and float(capped) == 1
    assert float(w.mean()) == pytest.approx(1.0)
    w, _ = sl.surprise_weights(torch.zeros(3), 0.5, 8.0)
    assert torch.equal(w, torch.ones(3))


def test_surprise_weighting_reweights_each_rows_target_ce():
    legal = [5, 6]
    rows = [_row(legal, [0.5, 0.5]), _row(legal, [1.0, 0.0])]
    batch = sl.search_batch(rows, CPU)
    logits = torch.zeros(2, POLICY_SIZE, requires_grad=True)  # uniform: row 0 at its target, row 1 off by ln 2
    out = sl.search_losses(logits, None, logits.detach(), None, batch, 1.0, 0.0, surprise_weight=0.5, surprise_cap=8.0)
    kl1 = math.log(2.0)
    assert float(out["search_kl"]) == pytest.approx(kl1, rel=1e-5)
    w = np.array([0.5, 1.5])  # (1 - a) + a * s / mean(s), s = (0, ln 2)
    ce = np.array([math.log(2.0), math.log(2.0)])
    assert float(out["total"].detach()) == pytest.approx(float((w * ce).sum()), rel=1e-5)
    assert float(out["surprise_disagree_rows"]) == 0  # ties: argmax agrees on both
    out["total"].backward()
    g = logits.grad[:, legal].numpy()
    np.testing.assert_allclose(g[0], 0.0, atol=1e-6)  # at its target: no gradient whatever the weight
    np.testing.assert_allclose(g[1], 1.5 * np.array([0.5 - 1.0, 0.5]), atol=1e-6)  # w * (p - target)
    plain = sl.search_losses(logits.detach(), None, logits.detach(), None, batch, 1.0, 0.0)
    assert float(plain["total"]) == pytest.approx(float(ce.sum()), rel=1e-5) and "surprise_weight_sq" not in plain


# ----------------------------------------------------------------------------- the shared network


def _window(rows):
    window = sl.ReplayWindow(len(rows))
    for r in rows:
        window.add(r)
    return window


def _rows(start_file, n=24, **overrides):
    settings = _settings(start_file, critic_positions_per_game=0, **overrides)
    settings.update(value_nstep=2, value_bootstrap_weight=0.5)
    gen = _generator(settings, HashNet())
    rows = [r for g in gen.run(5.0)["games"] for r in g["rows"]]
    assert len(rows) >= n
    return rows[:n]


def _cfg(**overrides):
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p",
        value_checkpoint="",
        learner_temperature=0.5,
        steps_per_update=1,
        minibatch=24,
        microbatch=8,
        kl_coef=0.0,
        shared=True,
        lr=1e-3,
        critic_lr=1e-3,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_learner_param_groups_put_the_value_heads_at_critic_lr():
    model = _tiny_model(0)
    model.add_aux_heads(sl.AUX_SEAT_TARGETS, sl.AUX_GLOBAL_TARGETS)
    groups = sl.learner_param_groups(model, _cfg(lr=1e-5, critic_lr=3e-5))
    assert [g["lr"] for g in groups] == [1e-5, 3e-5]
    heads = {
        id(p)
        for h in (model.win_loss_head, model.score_head, model.aux_seat_head, model.aux_global_head)
        for p in h.parameters()
    }
    assert {id(p) for p in groups[1]["params"]} == heads
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == len(list(model.parameters()))
    assert len(sl.learner_param_groups(model, _cfg(shared=False))) == 1


def test_shared_update_trains_policy_and_value_from_one_forward(start_file):
    rows = _rows(start_file)
    learner = _random_value_heads(_tiny_model(0), 5)  # zero heads would score every target ln 4
    anchor = copy.deepcopy(learner)
    cfg = _cfg()
    opt = torch.optim.AdamW(sl.learner_param_groups(learner, cfg), weight_decay=0.0)
    calls = {"n": 0}
    forward = learner.forward

    def counting(*args, **kwargs):
        calls["n"] += 1
        return forward(*args, **kwargs)

    learner.forward = counting
    rng = np.random.default_rng(0)
    first = sl._update(learner, None, anchor, opt, None, _window(rows), cfg, 0.0, CPU, rng)
    assert calls["n"] == 3  # one forward per microbatch: the policy and value losses share it
    assert first["critic_rows_trained"] == len(rows) and not any(r.critic for r in rows)
    assert math.isfinite(first["critic_ce"]) and math.isfinite(first["critic_outcome_ce"])
    assert first["critic_ce"] != pytest.approx(first["critic_outcome_ce"])  # n-step targets differ from the outcome
    for r in rows:
        r.critic = True
    for _ in range(15):
        last = sl._update(learner, None, anchor, opt, None, _window(rows), cfg, 0.0, CPU, rng)
        for r in rows:
            r.critic = True
    assert last["ce"] < first["ce"] - 0.05 and last["critic_ce"] < first["critic_ce"] - 0.05


def _one_shared_step(rows, stop_grad: bool, value_weight: float):
    learner = _random_value_heads(_tiny_model(0), 6)  # zero output layers would pass no gradient back
    anchor = copy.deepcopy(learner)
    for r in rows:
        r.critic = True
    cfg = _cfg(value_stop_grad=stop_grad, value_loss_weight=value_weight, score_loss_weight=value_weight)
    opt = torch.optim.SGD(sl.learner_param_groups(learner, cfg), lr=0.1)
    sl._update(learner, None, anchor, opt, None, _window(rows), cfg, 0.0, CPU, np.random.default_rng(0))
    return learner


def test_the_value_loss_reaches_the_trunk_unless_stop_grad(start_file):
    rows = _rows(start_file)
    start = _random_value_heads(_tiny_model(0), 6)
    policy_only = _one_shared_step(rows, stop_grad=False, value_weight=0.0)
    joint = _one_shared_step(rows, stop_grad=False, value_weight=1.0)
    stopped = _one_shared_step(rows, stop_grad=True, value_weight=1.0)

    def trunk_step(m):
        return torch.cat(
            [
                (p.detach() - q.detach()).flatten()
                for (n, p), q in zip(m.named_parameters(), start.parameters())
                if n.startswith("res_blocks")
            ]
        )

    def heads(m):
        return torch.cat([p.detach().flatten() for p in m.win_loss_head.parameters()])

    def cosine(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm()))

    # The value heads' group is clipped on its own, so with stop-grad the trunk takes the policy's step exactly.
    assert torch.allclose(trunk_step(stopped), trunk_step(policy_only), atol=1e-8)
    assert cosine(trunk_step(joint), trunk_step(policy_only)) < 1 - 1e-4  # the value gradient moved it too
    assert (heads(stopped) - heads(policy_only)).abs().max() > 1e-6  # the heads learned either way


# ----------------------------------------------------------------------------- the blended value on one network


def test_a_shared_value_model_blends_its_own_heads_from_one_forward(tmp_path):
    from rl18xx.agent.alphazero.composite_model import SharedValueModel, load_policy_value
    from rl18xx.agent.alphazero.policy_gradient import _save

    model = _random_value_heads(_tiny_model(0), 1).eval()
    path = tmp_path / "run" / "learner" / "3.pth"
    _save(model, path)
    states = [_rust_encode(RustGameAdapter(g)) for g in random_positions(4, 2, steps=200, every=23)[:5]]
    plain = load_policy_value(str(path))
    assert not isinstance(plain, SharedValueModel) and not isinstance(
        load_policy_value(f"{path}#win"), SharedValueModel
    )
    blended = load_policy_value(f"{path}#blend=0.25")
    scored = load_policy_value(f"{path}#score")
    assert isinstance(blended, SharedValueModel) and isinstance(scored, SharedValueModel)
    with torch.no_grad():
        p0, _, v0 = plain.run_many_encoded(states)
        _, win, score, _ = plain._forward_encoded_batch(states)
        calls = {"n": 0}
        forward = blended.model.forward

        def counting(*args, **kwargs):
            calls["n"] += 1
            return forward(*args, **kwargs)

        blended.model.forward = counting
        p1, _, v1 = blended.run_many_encoded(states)
        _, _, v2 = scored.run_many_encoded(states)
    assert calls["n"] == 1
    assert torch.allclose(p1, p0, atol=1e-6) and torch.allclose(v0, torch.softmax(win, dim=-1), atol=1e-6)
    assert torch.allclose(v1, 0.75 * torch.softmax(win, dim=-1) + 0.25 * score, atol=1e-6)
    assert torch.allclose(v2, score, atol=1e-6)
    assert blended.last_price_components is not None
    # The inference server's packed fast path gives the same replies.
    parts = [
        (
            np.concatenate([s[0].numpy().reshape(1, -1) for s in states]),
            np.stack([s[1].numpy() for s in states]),
            np.array([[int(s[4]), 0, int(s[6]), int(s[7])] for s in states]),
            None,
        )
    ]
    assert blended.supports_packed
    with torch.no_grad():
        legal = [np.arange(10, dtype=np.int64)] * len(states)
        pp, _, vp = blended.run_packed(blended.packed_inputs(parts), legal_indices=legal)
        p_ref, _, v_ref = blended.run_many_encoded(states, legal_indices=legal)
    assert torch.allclose(vp, v1, atol=1e-5) and torch.allclose(vp, v_ref, atol=1e-5)
    assert torch.allclose(pp[:, :10], p_ref[:, :10], atol=1e-5)
    with pytest.raises(ValueError):
        load_policy_value(f"{path}#blend=1.5")


def test_server_spec_for_shared_and_composite_runs():
    cfg = _cfg(value_head="blend=0.3")
    assert sl.server_spec(cfg, "L.pth") == "L.pth#blend=0.3"
    cfg = _cfg(shared=False, value_head="win")
    assert sl.server_spec(cfg, "L.pth", "C.pth") == "L.pth+C.pth"
    cfg = _cfg(shared=False, value_head="score")
    assert sl.server_spec(cfg, "L.pth", "C.pth") == "L.pth+C.pth#score"


# ----------------------------------------------------------------------------- auxiliary targets


def test_aux_heads_are_training_only_and_survive_a_checkpoint(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _load, _save

    model = _random_value_heads(_tiny_model(0), 2).eval()
    states = [_rust_encode(RustGameAdapter(g)) for g in random_positions(4, 3, steps=150, every=31)[:3]]
    with torch.no_grad():
        before = model._forward_encoded_batch(states)
    model.add_aux_heads(sl.AUX_SEAT_TARGETS, sl.AUX_GLOBAL_TARGETS)
    with torch.no_grad():
        model.aux_seat_head[-1].weight.normal_()
        after = model._forward_encoded_batch(states)
    for a, b in zip(before[:3], after[:3]):
        assert torch.equal(a, b)  # the forward never runs them
    _save(model, tmp_path / "learner" / "1.pth")
    loaded = _load(tmp_path / "learner" / "1.pth")
    assert loaded.config.aux_seat_targets == 2 and loaded.config.aux_global_targets == 8
    assert torch.equal(
        loaded.aux_seat_head[-1].weight, model.aux_seat_head[-1].weight.to(loaded.aux_seat_head[-1].weight.device)
    )
    store, hook = loaded.capture_value_input()
    with torch.no_grad():
        loaded._forward_encoded_batch(states, value_only=True)
    hook.remove()
    seat, glob = loaded.aux_outputs(store["seat_input"])
    assert seat.shape == (3, VALUE_SIZE, 2) and glob.shape == (3, 8)
    plain = _tiny_model(0)
    assert plain.aux_seat_head is None and plain.aux_global_head is None


def test_aux_outcomes_at_a_games_end():
    from rl18xx.agent.alphazero.policy_gradient import outcome
    from rl18xx.agent.alphazero.self_play import _compute_net_worth

    game = RustGameAdapter(ending_positions(4, 5, last=1)[-1])
    assert game.finished
    _, fractions = outcome(game)
    seat, floated = sl.aux_outcomes(game, fractions)
    worth = _compute_net_worth(game)
    total = sum(max(v, 0) for v in worth.values())
    cash = {p.id: p.cash for p in game._game.players}
    np.testing.assert_allclose(seat[:, 0], fractions)
    np.testing.assert_allclose(seat[:, 1], [cash[pid] / total for pid in sorted(worth)], rtol=1e-6)
    assert floated.shape == (8,) and set(np.unique(floated)) <= {0.0, 1.0}
    assert floated.tolist() == [float(c.floated) for c in game._game.corporations]


def test_rows_get_aux_targets_and_the_batch_rotates_them(start_file):
    gen = _generator(_settings(start_file, critic_positions_per_game=0, p_full=1.0), HashNet())
    rows = [r for g in gen.run(5.0)["games"] for r in g["rows"]]
    assert rows
    rows = [r for r in rows if r.rotation != 0][:4] or rows[:4]
    batch = sl.search_batch(rows, CPU)
    for b, r in enumerate(rows):
        assert r.aux_seat.shape == (4, 2) and r.aux_global.shape == (8,)
        np.testing.assert_allclose(batch["aux_seat"][b].numpy(), np.roll(r.aux_seat, -r.rotation, axis=0))
        np.testing.assert_allclose(batch["win_target"][b].numpy(), np.roll(r.win_target, -r.rotation))


# ----------------------------------------------------------------------------- Gumbel at small budgets


@pytest.mark.parametrize("n_full", [8, 16])
def test_gumbel_targets_at_small_budgets_with_fast_searches(start_file, n_full):
    settings = _settings(
        start_file,
        target="gumbel",
        n_full=n_full,
        p_full=0.25,
        parallel_readouts=4,
        games_per_worker=3,
        max_decisions=40,
    )
    settings["n_fast"] = 4
    gen = _generator(settings, HashNet())
    out = gen.run(6.0)
    stats = out["stats"]
    assert stats.get("fast_search_decisions", 0) > stats.get("searched_decisions", 0) > 0
    assert stats.get("fast_decisions", 0) == stats.get("single_move_decisions", 0)  # no policy moves with a choice
    rows = [r for g in out["games"] for r in g["rows"]] + [r for g in gen.games.values() for r in g.rows]
    assert rows
    for r in rows:
        assert float(r.target.sum()) == pytest.approx(1.0, abs=1e-5) and (r.target >= 0).all()
        assert r.root_value is not None and float(r.root_value.sum()) == pytest.approx(1.0, abs=1e-4)


# ----------------------------------------------------------------------------- the value-head refit


def test_value_refit_fits_heads_to_the_critic_and_keeps_the_policy(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _load, _save

    learner, critic = _tiny_model(0).eval(), _random_value_heads(_tiny_model(1), 3).eval()
    games = random_positions(4, 4, steps=500, every=3)
    rng = np.random.default_rng(0)
    positions = []
    for i, g in enumerate(games):
        state, nodes, _, rotation, _ = g.encode_canonical_py()
        win = np.eye(4, dtype=np.float32)[i % 4]
        positions.append(
            vr.Position(state, nodes, int(rotation), win, win * 1.0, min(2, i * 3 // len(games)), f"x/{i // 8}")
        )
    data = vr.extract(positions, learner, {"critic": critic}, CPU, batch_size=32)
    assert data.seat_input.shape == (len(positions), 4, learner.config.d_entity + learner.config.d_trunk)
    np.testing.assert_allclose(data.win[0], np.roll(positions[0].win, -positions[0].rotation))
    cfg = vr.RefitConfig(outcome_weight=0.0, lr=3e-3, max_epochs=40, patience=40, batch_size=16)
    fit = vr.fit_heads(learner, data, data, "critic", cfg, CPU)
    assert fit["history"][-1]["ce"] < fit["history"][0]["ce"] - 0.05
    pred = vr.head_predictions(fit["win_head"], fit["score_head"], data, CPU)
    kl_before = (
        (
            data.predictions["critic"]["win"]
            * np.log(data.predictions["critic"]["win"] / data.predictions["learner"]["win"])
        )
        .sum(1)
        .mean()
    )
    kl_after = (
        (data.predictions["critic"]["win"] * np.log(data.predictions["critic"]["win"] / pred["win"])).sum(1).mean()
    )
    assert kl_after < 0.5 * kl_before
    source = tmp_path / "src" / "learner" / "9.pth"
    _save(learner, source)
    (tmp_path / "src" / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    out = vr.write_refit_checkpoint(
        source, fit["win_head"], fit["score_head"], tmp_path / "out" / "learner" / "9.pth", {"x": 1}
    )
    assert json.loads((tmp_path / "out" / "config.json").read_text())["learner_temperature"] == 0.5
    refit = _load(out).eval()
    gs = torch.from_numpy(np.stack([p.game_state for p in positions[:6]])).float()
    nodes = torch.from_numpy(np.stack([p.node_features for p in positions[:6]])).float()
    with torch.no_grad():
        a = learner.to(CPU)(gs, nodes)
        b = refit.to(CPU)(gs, nodes)
    assert torch.equal(a[0], b[0])  # the policy is untouched
    # (the fit read fp16 features)
    np.testing.assert_allclose(torch.softmax(b[1], -1)[:, :4].numpy(), pred["win"][:6], atol=5e-4)
    report = vr.stage_report(data, pred)
    assert report["n"] == len(positions) and 0 <= report["winner_hit"] <= 1


def test_value_finetune_keeps_the_policy_when_the_trunk_is_frozen_and_reports_drift():
    learner, critic = _tiny_model(0).eval(), _random_value_heads(_tiny_model(1), 7).eval()
    games = random_positions(4, 6, steps=300, every=3)
    positions = []
    for i, g in enumerate(games):
        state, nodes, _, rotation, _ = g.encode_canonical_py()
        win = np.eye(4, dtype=np.float32)[i % 4]
        legal = np.asarray(g.factored_legal_indices(), dtype=np.int32)
        positions.append(vr.Position(state, nodes, int(rotation), win, win, i * 3 // len(games), f"y/{i // 6}", legal))
    data = vr.extract(positions, learner, {"critic": critic}, CPU, batch_size=32)
    frozen = vr.FinetuneConfig(trunk_lr=0.0, head_lr=3e-3, batch_size=16, epochs=6, eval_every=5, outcome_weight=1.0)
    out = vr.finetune(learner, positions, data, positions, data, "critic", frozen, CPU)
    drift = vr.policy_drift(learner, out["model"], positions, 0.5, CPU)
    assert drift["policy_kl"] < 1e-9 and drift["top1_agreement"] == 1.0  # only the value heads moved
    assert out["best_step"] > 0 and out["history"][-1]["outcome_ce"] != out["history"][0]["outcome_ce"]
    moving = vr.FinetuneConfig(trunk_lr=1e-3, head_lr=3e-3, batch_size=16, epochs=2, eval_every=4, policy_kl_weight=0.0)
    out = vr.finetune(learner, positions, data, positions, data, "critic", moving, CPU)
    assert {"policy_kl", "top1_agreement", "outcome_ce"} <= set(out["history"][-1])
    assert vr.policy_drift(learner, out["model"], positions, 0.5, CPU)["policy_kl"] >= 0.0
    pred = vr.model_predictions(out["model"], positions[:5], CPU)
    assert pred["win"].shape == (5, 4) and np.allclose(pred["win"].sum(1), 1.0, atol=1e-5)


def test_stage_report_scores_ce_and_winner_hit_by_stage():
    data = vr.Extracted(
        seat_input=None,
        win=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0.5]], dtype=np.float32),
        placement=np.zeros((3, 4), dtype=np.float32),
        stage=np.array([0, 1, 2]),
        games=["a", "a", "b"],
    )
    pred = {
        "win": np.array([[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]),
        "score": np.zeros((3, 4)),
    }
    r = vr.stage_report(data, pred)
    assert r["win_ce_early"] == pytest.approx(-math.log(0.7))
    assert r["win_ce_mid"] == pytest.approx(-math.log(0.1))
    assert r["win_ce_late"] == pytest.approx(-math.log(0.4))
    assert (r["winner_hit_early"], r["winner_hit_mid"], r["winner_hit_late"]) == (1.0, 0.0, 1.0)


# ----------------------------------------------------------------------------- end to end


def _save_shared_start(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _save

    source = tmp_path / "source"
    (source / "learner").mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    _save(_random_value_heads(_tiny_model(0), 4), source / "learner" / "100.pth")
    return source / "learner" / "100.pth"


def test_a_tiny_cpu_az2_run_shared_gumbel_nstep_surprise_aux_speedups_resumes(tmp_path, start_file, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    policy = _save_shared_start(tmp_path)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint=str(policy),
        value_checkpoint="",
        out_dir=str(tmp_path / "runs"),
        run_name="az2",
        shared=True,
        value_head="blend=0.3",
        target="gumbel",
        n_full=8,
        n_fast=4,
        p_full=0.5,
        parallel_readouts=4,
        workers=2,
        games_per_worker=2,
        task_seconds=2.0,
        max_decisions=16,
        start_positions=start_file,
        random_start_fraction=1.0,
        inference_batch_size=64,
        window_rows=400,
        rows_per_update=10,
        min_window_rows=10,
        steps_per_update=2,
        minibatch=8,
        microbatch=4,
        max_updates=2,
        snapshot_every=1,
        anchor_every=1,
        kl_target=0.05,
        critic_positions_per_game=0,
        value_nstep=3,
        value_bootstrap_weight=0.5,
        surprise_weight=0.5,
        aux_loss_weight=0.1,
        lr=1e-4,
        critic_lr=3e-4,
        reuse_tree="total",
        dedupe_maps=True,
        server_replicas=2,  # the pilot's speed options
    )
    run_dir = sl.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    for h in history:
        for key in ("ce", "critic_ce", "critic_outcome_ce", "surprise_ess", "aux_mse", "aux_bce", "search_value_ce"):
            assert math.isfinite(h[key]), key
        assert h["critic_rows_trained"] > 0 and 0 < h["surprise_ess"] <= 1.0 + 1e-6
        assert "reused_search_share" in h and h["map_rows_per_state"] <= 1.0
    assert not (run_dir / "critic").exists()
    for update in (0, 1, 2):
        assert (run_dir / "learner" / f"{update}.pth").exists()
    from rl18xx.agent.alphazero.policy_gradient import _load

    trained = _load(run_dir / "learner" / "2.pth")
    assert trained.aux_seat_head is not None  # the aux heads are saved with the learner
    config = json.loads((run_dir / "config.json").read_text())
    assert config["shared"] and config["value_head"] == "blend=0.3"

    again = copy.copy(cfg)
    again.max_updates = 3
    assert sl.run(again, resume=str(run_dir)) == run_dir
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2, 3]
    assert json.loads((run_dir / "state.json").read_text())["snapshot_update"] == 3
