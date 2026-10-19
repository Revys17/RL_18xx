"""Q-based policy targets on PUCT roots (rl18xx/agent/alphazero/search_targets.py),
min-max normalized Q in PUCT selection (``RustMCTSPlayer.set_q_normalize``), and
search learning's ``cq`` / ``rpo`` targets and fast-search rows."""

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero import search_targets as st  # noqa: E402
from rl18xx.agent.alphazero.config import SelfPlayConfig  # noqa: E402
from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator  # noqa: E402
from rl18xx.agent.alphazero.mcts import POLICY_SIZE  # noqa: E402
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer  # noqa: E402

from .search_test_utils import HashNet, RiggedNet, expand_root, make_player, random_positions  # noqa: E402
from .test_search_learning import _generate, _row, _settings, start_file  # noqa: E402,F401

PUCT = dict(mcts_mean_q=True, c_puct_init=0.05, c_puct_by_round={}, dedupe_leaves=True, prior_temperature=0.5)


def _games(min_legal=3, count=8, seed=4):
    games = random_positions(4, seed, steps=400, every=7) + random_positions(4, seed + 1, steps=400, every=7)
    return [g for g in games if len(g.factored_legal_indices()) >= min_legal][:count]


def _searched(net, game, seed=0, readouts=48, **overrides):
    player = make_player(
        net, game, seed=seed, parallel_readouts=8, num_readouts=readouts, min_readouts=readouts,
        softpick_move_cutoff=0, **{**PUCT, **overrides},
    )
    expand_root(player)
    player.suggest_move()
    return player


# ----------------------------------------------------------------------------- root statistics


def test_root_target_stats_match_the_trees_own_statistics():
    net = HashNet()
    priced = 0
    for seed, game in enumerate(_games(count=10)):
        tree = _searched(net, game, seed)._rust_player
        stats = st.root_stats(tree)
        node = tree.node_stats(tree.root_idx)
        mover = tree.root_active_player_index()
        n = np.asarray(node["child_n"], dtype=np.float64)
        w = np.asarray(node["child_w"], dtype=np.float64)[:, mover]
        assert stats["legal"].tolist() == list(node["legal"]) == list(tree.legal_action_indices_at_root())
        assert stats["mover"] == mover
        np.testing.assert_array_equal(stats["child_n"], n)
        visited = n > 0
        np.testing.assert_allclose(stats["child_q"][visited], w[visited] / n[visited], rtol=1e-6)
        assert np.isnan(stats["child_q"][~visited]).all()
        np.testing.assert_allclose(stats["completed_q"][visited], stats["child_q"][visited], rtol=1e-6)
        np.testing.assert_allclose(stats["completed_q"][~visited], stats["v_mix"], rtol=1e-6)
        assert stats["root_n"] == tree.n_at_root() == 1 + n.sum()
        assert stats["c_puct"] == pytest.approx(2 * np.log((1 + stats["root_n"] + 19652) / 19652) + 2 * 0.05)
        # A price-bearing slot's N aggregates its price grandchildren.
        for slot, by_price in tree.price_grandchildren_at_root().items():
            i = int(np.flatnonzero(stats["legal"] == slot)[0])
            assert stats["open_price"][i] and sum(by_price.values()) == stats["child_n"][i]
            priced += 1
    assert priced > 0


def test_cq_target_is_the_gumbel_roots_target():
    net = HashNet()
    for seed, game in enumerate(_games(count=6)):
        player = _searched(net, game, seed, readouts=32, gumbel_k=8, c_puct_init=0.15)
        tree = player._rust_player
        stats = st.root_stats(tree)
        ours = st.completed_q_target(stats, c_visit=50.0, c_scale=1.0)
        theirs = np.asarray(tree.gumbel_policy_target(), dtype=np.float64)[stats["legal"]]
        np.testing.assert_allclose(ours, theirs, atol=2e-6)


def _rigged_root(gap: float, readouts: int, **overrides):
    """A searched root whose second legal move is worth ``0.25 + gap`` to the mover
    and every other child 0.25 (uniform priors)."""
    game = _games(min_legal=4, count=1, seed=8)[0]
    net = RiggedNet()
    player = make_player(
        net, game, parallel_readouts=1, num_readouts=readouts, min_readouts=readouts, softpick_move_cutoff=0,
        **{**PUCT, "dedupe_leaves": False, **overrides},
    )
    expand_root(player)
    tree = player._rust_player
    legal = list(tree.legal_action_indices_at_root())
    mover = tree.root_active_player_index()
    for move in legal:
        value = 0.25 + gap if move == legal[1] else 0.25
        net.rig(tree.get_game_for_idx(tree.maybe_add_child(tree.root_idx, move)), mover, value)
    player.suggest_move()
    return tree, legal


def test_q_targets_move_mass_to_the_better_move():
    tree, legal = _rigged_root(gap=0.05, readouts=24)
    stats = st.root_stats(tree)
    prior = stats["child_prior"] / stats["child_prior"].sum()
    for target in (st.completed_q_target(stats), st.rpo_target(stats), st.rpo_target(stats, normalize=True)):
        assert abs(target.sum() - 1.0) < 1e-9 and (target >= 0).all()
        assert int(np.argmax(target)) == 1
        assert target[1] > prior[1]
        # Equal priors: moves with equal (completed) q get equal mass, a lower q less.
        q = stats["completed_q"]
        for i in range(len(q)):
            for j in range(len(q)):
                if abs(q[i] - q[j]) < 1e-7:
                    assert target[i] == pytest.approx(target[j], rel=1e-6)
                elif q[i] < q[j]:
                    assert target[i] < target[j]


def test_rpo_solves_the_regularized_problem():
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = int(rng.integers(2, 30))
        prior = rng.dirichlet(np.full(k, 0.3)) + 1e-9
        prior /= prior.sum()
        q = rng.uniform(0.15, 0.35, size=k)
        lam = float(rng.choice([1e-4, 1e-3, 1e-2, 0.1, 1.0]))
        pi = st.solve_rpo(prior, q, lam)
        assert abs(pi.sum() - 1.0) < 1e-9 and (pi >= 0).all()

        def objective(y):
            return q @ y - lam * float((prior * (np.log(prior) - np.log(np.clip(y, 1e-300, None)))).sum())

        best = objective(pi)
        for _ in range(20):  # no nearby distribution does better
            y = np.clip(pi * np.exp(rng.normal(scale=0.05, size=k)), 1e-300, None)
            assert objective(y / y.sum()) <= best + 1e-9
    prior = np.array([0.7, 0.2, 0.1])
    q = np.array([0.2, 0.3, 0.25])
    np.testing.assert_allclose(st.solve_rpo(prior, q, 1e4), prior, atol=1e-4)  # strong regularization: the prior
    assert st.solve_rpo(prior, q, 1e-7)[1] > 0.999  # none: the best q


def test_rpo_lambda_and_unvisited_completion():
    tree, legal = _rigged_root(gap=0.05, readouts=6)
    stats = st.root_stats(tree)
    n = stats["child_n"].sum()
    assert st.rpo_lambda(stats) == pytest.approx(stats["c_puct"] * np.sqrt(n) / (len(legal) + n))
    assert st.rpo_lambda(stats, c=2.0) == pytest.approx(2.0 * np.sqrt(n) / (len(legal) + n))
    unvisited = stats["child_n"] == 0
    assert unvisited.any()
    assert np.allclose(st.rpo_q(stats, "vmix")[unvisited], stats["v_mix"])
    assert np.allclose(st.rpo_q(stats, "parent")[unvisited], stats["root_value"])
    assert np.allclose(st.rpo_q(stats, "min")[unvisited], np.nanmin(stats["child_q"]))
    normalized = st.rpo_q(stats, "vmix", normalize=True)
    assert normalized.min() >= -1e-9 and normalized.max() <= 1 + 1e-9


# ----------------------------------------------------------------------------- q normalization


def _child_n(player):
    return np.asarray(player._rust_player.child_n_at_root())


def test_q_normalize_off_is_the_default_tree():
    net = HashNet()
    for seed, game in enumerate(_games(count=6)):
        default = _searched(net, game, seed)
        explicit = make_player(
            net, game, seed=seed, parallel_readouts=8, num_readouts=48, min_readouts=48, softpick_move_cutoff=0,
            mcts_q_normalize=False, **PUCT,
        )
        explicit._rust_player.set_q_normalize(False)
        expand_root(explicit)
        explicit.suggest_move()
        assert not default._rust_player.q_normalize_enabled
        np.testing.assert_array_equal(_child_n(default), _child_n(explicit))
        np.testing.assert_array_equal(
            np.asarray(default._rust_player.pruned_child_n_at_root()),
            np.asarray(explicit._rust_player.pruned_child_n_at_root()),
        )


def test_q_normalize_lets_a_small_value_gap_compete_with_exploration():
    # A gap of 0.01 win share at MuZero's c_puct (1.25 = 2 * 0.625): raw Q leaves the
    # visits to the uniform prior; normalized, the better move takes most of them.
    raw, legal = _rigged_root(gap=0.01, readouts=64, c_puct_init=0.625)
    normalized, _ = _rigged_root(gap=0.01, readouts=64, c_puct_init=0.625, mcts_q_normalize=True)
    assert normalized.q_normalize_enabled
    share = lambda tree: np.asarray(tree.child_n_at_root())[legal[1]] / (tree.n_at_root() - 1)  # noqa: E731
    assert share(normalized) > 0.5 > 2 * share(raw)
    # Pruning reads the selection's (normalized) PUCT terms.
    pruned = np.asarray(normalized.pruned_child_n_at_root())[legal]
    assert pruned[1] == np.asarray(normalized.child_n_at_root())[legal].max()


def test_q_normalize_needs_mean_q():
    game = _games(count=1)[0]
    with pytest.raises(ValueError):
        make_player(HashNet(), game, mcts_q_normalize=True)
    player = RustMCTSPlayer(SelfPlayConfig(network=HashNet(), mcts_mean_q=True, mcts_q_normalize=True))
    from rl18xx.rust_adapter import RustGameAdapter

    player.initialize_game(RustGameAdapter(game.pickle_clone()))
    assert player._rust_player.q_normalize_enabled


def test_eval_player_option_qnorm():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "eval_head_to_head.py"
    spec = importlib.util.spec_from_file_location("eval_head_to_head", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = str(Path(__file__).resolve())  # any existing file stands in for a .pth
    ckpt = checkpoint[: -len(".py")] + ".pth"
    Path(ckpt).write_bytes(b"")
    try:
        _, overrides = module.parse_player(f"{ckpt}@200/0.625:mean,qnorm,dedupe", 64)
        assert overrides["mcts_q_normalize"] and overrides["mcts_mean_q"] and overrides["c_puct_init"] == 0.625
        with pytest.raises(SystemExit):
            module.parse_player(f"{ckpt}@200:qnorm", 64)
    finally:
        Path(ckpt).unlink()


# ----------------------------------------------------------------------------- search learning


@pytest.mark.parametrize("target", ["cq", "rpo"])
def test_generator_rows_carry_q_targets(start_file, target):
    settings = _settings(start_file, target=target, n_full=8, critic_positions_per_game=0, q_normalize=True,
                         c_puct_init=0.5)
    gen, out = _generate(settings)
    assert gen.config.mcts_q_normalize and gen.config.gumbel_k == 0
    rows = [r for g in out["games"] for r in g["rows"]]
    assert rows
    for r in rows:
        assert not r.fast and r.weight == 1.0
        assert r.target.shape == r.legal.shape and (r.target >= 0).all()
        assert float(r.target.sum()) == pytest.approx(1.0, abs=1e-5)


def test_fast_rows_are_marked_weighted_and_train_the_value(start_file):
    settings = _settings(
        start_file, target="cq", p_full=0.25, n_full=4, parallel_readouts=2, games_per_worker=4, max_decisions=60,
        critic_positions_per_game=0,
    )
    settings.update(n_fast=2, fast_row_weight=0.3)
    _, out = _generate(settings, seconds=8.0)
    rows = [r for g in out["games"] for r in g["rows"]]
    fast = [r for r in rows if r.fast]
    full = [r for r in rows if not r.fast]
    assert fast and full and len(fast) > len(full)
    assert all(r.weight == pytest.approx(0.3) for r in fast) and all(r.weight == 1.0 for r in full)
    assert all(r.critic for r in rows)  # fast_row_value (default): every row trains the value
    assert sum(g["searched"] for g in out["games"]) == len(full)
    assert out["stats"]["fast_rows"] >= len(fast)
    settings["fast_row_value"] = False
    _, out = _generate(settings, seconds=6.0)
    rows = [r for g in out["games"] for r in g["rows"]]
    assert any(r.fast for r in rows)
    assert all(r.critic == (not r.fast) for r in rows)


def test_fast_rows_weigh_into_the_loss_and_report_apart():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(6):
        legal = np.sort(rng.choice(200, size=5, replace=False))
        target = rng.dirichlet(np.ones(5))
        row = _row(legal, target)
        if i >= 3:
            row.fast, row.weight = True, 0.25
        rows.append(row)
    batch = sl.search_batch(rows, torch.device("cpu"))
    assert batch["fast"].tolist() == [False] * 3 + [True] * 3
    logits = torch.randn(len(rows), POLICY_SIZE)
    out = sl.search_losses(logits, None, logits.clone(), None, batch, 0.5, kl_coef=0.0)
    alone = [sl.search_losses(logits[i : i + 1], None, logits[i : i + 1], None,
                              sl.search_batch([rows[i]], torch.device("cpu")), 0.5, 0.0) for i in range(6)]
    ce = [float(a["ce_fast"] if r.fast else a["ce"]) for a, r in zip(alone, rows)]
    assert float(out["total"]) == pytest.approx(sum(ce[:3]) + 0.25 * sum(ce[3:]), rel=1e-5)
    assert float(out["ce"]) == pytest.approx(sum(ce[:3]), rel=1e-5)
    assert float(out["ce_fast"]) == pytest.approx(sum(ce[3:]), rel=1e-5)
    assert float(out["rows_fast"]) == 3
    # Without fast rows, the batch and the losses are as before.
    plain = sl.search_batch(rows[:3], torch.device("cpu"))
    assert plain["fast"] is None and plain["row_weight"] is None
    out = sl.search_losses(logits[:3], None, logits[:3], None, plain, 0.5, 0.0)
    assert "ce_fast" not in out and float(out["total"]) == pytest.approx(sum(ce[:3]), rel=1e-6)


def test_update_reports_fast_rows_apart():
    rows = []
    rng = np.random.default_rng(2)
    for i in range(8):
        legal = np.sort(rng.choice(300, size=4, replace=False))
        row = _row(legal, rng.dirichlet(np.ones(4)), critic=True)
        if i % 2:
            row.fast, row.weight = True, 0.5
        rows.append(row)
    window = sl.ReplayWindow(16)
    for r in rows:
        window.add(r)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(POLICY_SIZE))
            self.v = torch.nn.Parameter(torch.zeros(6))
            self.last_price_components = {"price_logits": None}

        def forward(self, game_state, nodes, lay_tile_rows=None, value_only=False):
            b = game_state.shape[0]
            return self.w.expand(b, -1), self.v.expand(b, -1), self.v.expand(b, -1), None

    learner, anchor, critic = Net(), Net(), Net()
    cfg = sl.SearchLearningConfig(policy_checkpoint="", value_checkpoint="", learner_temperature=0.5,
                                  steps_per_update=2, minibatch=8, microbatch=4)
    opt = torch.optim.SGD(learner.parameters(), lr=0.1)
    copt = torch.optim.SGD(critic.parameters(), lr=0.1)
    stats = sl._update(learner, critic, anchor, opt, copt, window, cfg, 0.0, torch.device("cpu"),
                       np.random.default_rng(0))
    assert stats["fast_rows_trained"] == 8 and stats["rows_trained"] == 16
    for key in ("ce", "search_kl", "target_entropy", "top1_agreement"):
        assert np.isfinite(stats[key]) and np.isfinite(stats[f"{key}_fast"])

    def entropy(r):
        return float(-(r.target * np.log(r.target)).sum())

    # Every row is drawn in each step (minibatch = window): the means are over each kind's rows.
    assert stats["target_entropy"] == pytest.approx(np.mean([entropy(r) for r in rows if not r.fast]), rel=1e-5)
    assert stats["target_entropy_fast"] == pytest.approx(np.mean([entropy(r) for r in rows if r.fast]), rel=1e-5)
