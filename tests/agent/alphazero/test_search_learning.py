"""Search learning (rl18xx/agent/alphazero/search_learning.py): the target
cross-entropy at the policy's temperature, the generator's rows and its
playout-cap sampling, the update, and a tiny CPU run end to end (inference
servers, checkpoints, resume)."""

import copy
import json
import math

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import price_pmf  # noqa: E402
from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero.lockstep import ModelEvaluator  # noqa: E402
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, _rust_encode  # noqa: E402

from .search_test_utils import HashNet, expand_root, make_player, random_positions  # noqa: E402

STATE_WIDTH, HEXES, FEATURES = 394, 93, 50  # the 4-player encoding


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending (no human data needed)."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def _settings(start_file, **overrides):
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", learner_temperature=0.5, prior_temperature=0.5,
        start_positions=start_file, random_start_fraction=1.0, n_full=6, p_full=0.5, games_per_worker=3,
        max_decisions=24, parallel_readouts=4,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    settings = sl.worker_settings(cfg, version=7)
    return settings


def _row(legal, target, price=None, rotation=0, critic=False):
    return sl.SearchRow(
        game_state=np.zeros(STATE_WIDTH, dtype=np.float32),
        node_features=np.zeros((HEXES, FEATURES), dtype=np.float32),
        rotation=rotation,
        legal=np.asarray(legal, dtype=np.int32),
        target=np.asarray(target, dtype=np.float32),
        price=price,
        version=0,
        fractions=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32),
        score_target=np.array([1.0, 0.5, 0.2, 0.0], dtype=np.float32),
        critic=critic,
    )


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.max() - np.log(np.exp(x - x.max()).sum())


# ----------------------------------------------------------------------------- the loss


def test_target_cross_entropy_is_taken_at_the_policys_temperature():
    legal = [3, 10, 200, 5000]
    target = np.array([0.1, 0.6, 0.2, 0.1])
    rows = [_row(legal, target), _row([7, 8], [1.0, 0.0])]
    batch = sl.search_batch(rows, torch.device("cpu"))
    logits = torch.zeros(2, POLICY_SIZE)
    logits[0, legal] = torch.tensor([1.0, -0.5, 0.3, 2.0])
    logits[1, [7, 8]] = torch.tensor([0.2, 0.1])
    logits[:, 0] = 50.0  # not legal: no effect
    for temperature in (1.0, 0.5):
        out = sl.search_losses(logits, None, logits.clone(), None, batch, temperature, kl_coef=0.1)
        expected0 = -(target * _log_softmax(logits[0, legal].numpy() / temperature)).sum()
        expected1 = -_log_softmax(logits[1, [7, 8]].numpy() / temperature)[0]
        assert float(out["ce"]) == pytest.approx(expected0 + expected1, rel=1e-5)
        entropy = -(target * np.log(target)).sum()
        assert float(out["search_kl"]) == pytest.approx(expected0 + expected1 - entropy, rel=1e-5)
        assert float(out["kl_anchor"]) == pytest.approx(0.0, abs=1e-6)  # anchor = learner
        assert float(out["total"]) == pytest.approx(float(out["ce"]), rel=1e-5)


def test_the_loss_is_least_when_the_tempered_policy_is_the_target_and_pulls_toward_it():
    legal = [11, 12, 13]
    target = np.array([0.7, 0.2, 0.1])
    batch = sl.search_batch([_row(legal, target)], torch.device("cpu"))
    temperature = 0.5
    logits = torch.zeros(1, POLICY_SIZE)
    logits[0, legal] = torch.tensor(np.log(target) * temperature, dtype=torch.float32)  # softmax(l / T) = target
    out = sl.search_losses(logits, None, logits.clone(), None, batch, temperature, kl_coef=0.0)
    assert float(out["search_kl"]) == pytest.approx(0.0, abs=1e-5)
    assert float(out["top1_agreement"]) == 1.0
    # From a flat policy, gradient descent raises the target's best move.
    flat = torch.zeros(1, POLICY_SIZE, requires_grad=True)
    sl.search_losses(flat, None, flat.detach().clone(), None, batch, temperature, kl_coef=0.0)["total"].backward()
    grad = flat.grad[0, legal].numpy()
    assert grad[0] < 0 < grad[2]  # (p - target) / T
    assert np.allclose(grad, (1 / 3 - target) / temperature, atol=1e-5)
    assert float(flat.grad[0, 0]) == 0.0


def test_kl_to_the_anchor_and_the_price_target():
    slot = price_pmf.SLOT_INDEX[("Bid", ("CA",))]
    cells = price_pmf.price_cells("Bid", 165, 300)
    cell_target = np.zeros(price_pmf.NUM_CELLS, dtype=np.float32)
    cell_target[[0, 2]] = [0.75, 0.25]
    row = _row([1, 2], [0.5, 0.5], price=(slot, 165, 300, cell_target))
    batch = sl.search_batch([row], torch.device("cpu"))
    assert batch["price_rows"].tolist() == [0] and batch["price_masks"][0].numpy().tolist() == cells.nonempty.tolist()
    logits = torch.zeros(1, POLICY_SIZE)
    anchor = logits.clone()
    anchor[0, 1] = 1.0
    price = torch.zeros(1, price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS)
    out = sl.search_losses(logits, price, anchor, price.clone(), batch, 1.0, kl_coef=0.5, price_weight=2.0)
    live = cells.nonempty.sum()
    assert float(out["price_ce"]) == pytest.approx(math.log(live), rel=1e-5)  # uniform over the live cells
    p_anchor = np.exp(_log_softmax([1.0, 0.0]))
    kl = 0.5 * np.log(0.5 / p_anchor[0]) + 0.5 * np.log(0.5 / p_anchor[1])
    assert float(out["kl_anchor"]) == pytest.approx(kl, rel=1e-5)
    assert float(out["total"]) == pytest.approx(float(out["ce"]) + 2.0 * math.log(live) + 0.5 * kl, rel=1e-5)


def test_price_cell_target_pools_the_grandchildren_by_cell():
    slot = price_pmf.SLOT_INDEX[("BuyTrain", ("NYC", "3"))]
    cells = price_pmf.price_cells("BuyTrain", 1, 400)
    targets = [(slot, 1.0, 0.5, 1.0, 400.0), (slot, 400.0, 0.25, 1.0, 400.0), (slot, 399.0, 0.25, 1.0, 400.0)]
    got_slot, lo, hi, dist = sl.price_cell_target(targets)
    assert (got_slot, lo, hi) == (slot, 1, 400)
    assert dist.sum() == pytest.approx(1.0)
    assert dist[cells.cell_of(1)] == pytest.approx(0.5) and dist[cells.cell_of(400)] == pytest.approx(0.25)
    assert sl.price_cell_target([]) is None


def test_batches_rotate_outcomes_into_the_movers_frame():
    rows = [_row([1, 2], [0.5, 0.5], rotation=2, critic=True), _row([3], [1.0], rotation=0, critic=False)]
    batch = sl.search_batch(rows, torch.device("cpu"))
    assert batch["critic_rows"].tolist() == [0]
    assert batch["fractions"][0].tolist() == pytest.approx([0.2, 0.1, 0.4, 0.3])  # seat 2 first
    assert batch["score_target"][0].tolist() == pytest.approx([0.2, 0.0, 1.0, 0.5])


def test_policy_target_pruning_drops_the_visits_exploration_explains():
    net = HashNet()
    games = [g for g in random_positions(4, 3, steps=300, every=11) if len(g.factored_legal_indices()) >= 4]
    pruned_any = 0
    for seed, game in enumerate(games[:10]):
        player = make_player(
            net, game, seed=seed, mcts_mean_q=True, c_puct_init=0.15, c_puct_by_round={}, dedupe_leaves=True,
            parallel_readouts=8, num_readouts=64, min_readouts=64,
        )
        expand_root(player)
        player.suggest_move()
        tree = player._rust_player
        legal = np.asarray(tree.legal_action_indices_at_root())
        raw = np.asarray(tree.child_n_at_root())[legal]
        pruned = np.asarray(tree.pruned_child_n_at_root())[legal]
        best = int(np.argmax(raw))
        assert (pruned <= raw).all() and pruned[best] == raw[best] > 0
        assert not (pruned == 1).any()
        assert not np.asarray(tree.pruned_child_n_at_root())[np.setdiff1d(np.arange(POLICY_SIZE), legal)].any()
        pruned_any += int((pruned < raw).any())
    assert pruned_any > 0


# ----------------------------------------------------------------------------- generation


def _generate(settings, seconds=6.0, evaluators=None):
    gen = sl.Generator(settings, evaluators or {"learner": ModelEvaluator(HashNet())})
    out = gen.run(seconds)
    return gen, out


@pytest.mark.parametrize("target", ["visits", "gumbel"])
def test_rows_hold_the_searched_positions_targets_and_outcomes(start_file, target):
    settings = _settings(start_file, target=target, critic_positions_per_game=3)
    _, out = _generate(settings)
    games = out["games"]
    assert games and sum(len(g["rows"]) for g in games) > 0
    for g in games:
        assert g["searched"] == len(g["rows"])
        assert sum(r.critic for r in g["rows"]) == min(3, len(g["rows"]))
        for r in g["rows"]:
            assert r.version == 7
            assert r.game_state.shape == (STATE_WIDTH,) and r.node_features.shape == (HEXES, FEATURES)
            assert r.game_state.dtype == np.float32 and 0 <= r.rotation < 4
            assert len(r.legal) > 1 and np.all(np.diff(r.legal) > 0) and r.legal.dtype == np.int32
            assert r.target.shape == r.legal.shape and (r.target >= 0).all()
            assert float(r.target.sum()) == pytest.approx(1.0, abs=1e-5)
            assert float(r.fractions.sum()) == pytest.approx(1.0, abs=1e-5)
            assert float(r.score_target.sum()) == pytest.approx(1.7)  # placement points 1, 0.5, 0.2, 0 (ties share)
            if r.price is not None:
                slot, lo, hi, cells = r.price
                live = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot], lo, hi).nonempty
                assert float(cells.sum()) == pytest.approx(1.0) and not cells[~live].any()


def test_a_searched_rows_state_is_the_positions_encoding(start_file):
    settings = _settings(start_file, p_full=1.0)
    gen = sl.Generator(settings, {"learner": ModelEvaluator(HashNet())})
    g = gen._new_game()
    assert gen._step_game(0, g)
    state, nodes, round_type, rotation, num_players = g.pending
    expected = _rust_encode(g.game)
    assert np.array_equal(state, expected[0].numpy()[0]) and np.array_equal(nodes, expected[1].numpy())
    assert (round_type, rotation, num_players) == (expected[4], expected[6], expected[7])


def test_playout_cap_randomization_searches_p_full_of_the_decisions(start_file):
    counts = {}
    for p_full in (0.0, 0.25, 1.0):
        settings = _settings(
            start_file, p_full=p_full, n_full=2, parallel_readouts=2, games_per_worker=4, max_decisions=60
        )
        gen, out = _generate(settings, seconds=8.0)
        stats = out["stats"]
        choice_decisions = stats.get("searched_decisions", 0) + stats.get("fast_decisions", 0) - stats.get(
            "single_move_decisions", 0
        )
        counts[p_full] = (stats.get("searched_decisions", 0), choice_decisions, out)
    assert counts[0.0][0] == 0 and not any(g["rows"] for g in counts[0.0][2]["games"])
    assert counts[1.0][0] == counts[1.0][1] > 0  # every decision with a choice of move
    searched, decisions, _ = counts[0.25]
    assert decisions > 200
    share = searched / decisions
    assert abs(share - 0.25) < 4 * math.sqrt(0.25 * 0.75 / decisions)


def test_opponent_seats_play_their_policy_and_give_no_rows(start_file):
    settings = _settings(start_file, p_full=1.0)
    settings["opponent_servers"] = ["opp"]
    settings["learner_seats"] = 2
    net = HashNet()
    _, out = _generate(settings, evaluators={"learner": ModelEvaluator(net), "opp": ModelEvaluator(net)})
    assert out["games"]
    for g in out["games"]:
        learner = {seat for seat, server in enumerate(g["seat_servers"]) if server == "learner"}
        assert len(learner) == 2
        assert {r.rotation for r in g["rows"]} <= learner


def test_learner_temperature_comes_from_the_policys_run(tmp_path):
    (tmp_path / "learner").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    assert sl.run_temperature(tmp_path / "learner" / "100.pth") == 0.5
    assert sl.run_temperature(tmp_path / "other" / "100.pth") == 1.0


# ----------------------------------------------------------------------------- training


def _tiny_model(seed: int):
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    torch.manual_seed(seed)
    config = ModelTransformerConfig(
        device=torch.device("cpu"), d_entity=32, econ_transformer_layers=1, econ_transformer_heads=2,
        econ_transformer_ff_dim=64, d_map=32, hex_transformer_layers=1, hex_transformer_heads=2,
        hex_transformer_ff_dim=64, cross_attn_heads=2, d_trunk=64, num_res_blocks=1, value_head_layers=1,
    )
    return AlphaZeroTransformerModel(config)


def _fixed_rows(start_file, n=24):
    _, out = _generate(_settings(start_file, critic_positions_per_game=0), seconds=5.0)
    rows = [r for g in out["games"] for r in g["rows"]]
    assert len(rows) >= n
    return rows[:n]


def test_updates_fit_the_search_targets_on_a_fixed_batch(start_file):
    rows = _fixed_rows(start_file)
    learner, critic = _tiny_model(0), _tiny_model(1)
    anchor = copy.deepcopy(learner)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="v", learner_temperature=0.5, steps_per_update=1,
        minibatch=len(rows), microbatch=8, kl_coef=0.0,
    )
    window = sl.ReplayWindow(len(rows))
    for r in rows:
        window.add(r)
    opts = [torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0) for m in (learner, critic)]
    rng = np.random.default_rng(0)
    first = sl._update(learner, critic, anchor, *opts, window, cfg, 0.0, torch.device("cpu"), rng)
    assert first["critic_rows_trained"] == len(rows) and first["rows_trained"] == len(rows)
    assert all(not r.critic for r in rows)  # the critic trains on a position once
    for _ in range(15):
        last = sl._update(learner, critic, anchor, *opts, window, cfg, 0.0, torch.device("cpu"), rng)
    assert last["critic_rows_trained"] == 0
    assert last["ce"] < first["ce"] - 0.05
    assert all(math.isfinite(v) for v in last.values() if isinstance(v, float) and not math.isnan(v))


def test_microbatches_add_up_to_the_minibatch_step(start_file):
    rows = _fixed_rows(start_file, n=12)
    base = [_tiny_model(0), _tiny_model(1)]
    anchor = _tiny_model(2)
    moved = []
    for micro in (12, 4):
        learner, critic = (copy.deepcopy(m) for m in base)
        for r in rows:
            r.critic = True
        window = sl.ReplayWindow(len(rows))
        for r in rows:
            window.add(r)
        cfg = sl.SearchLearningConfig(
            policy_checkpoint="p", value_checkpoint="v", learner_temperature=0.5, steps_per_update=1,
            minibatch=12, microbatch=micro,
        )
        opts = [torch.optim.SGD(m.parameters(), lr=0.1) for m in (learner, critic)]
        sl._update(learner, critic, anchor, *opts, window, cfg, 0.02, torch.device("cpu"), np.random.default_rng(0))
        moved.append([p.detach().clone() for p in learner.parameters()])
    diffs = [float((a - b).abs().max()) for a, b in zip(*moved)]
    steps = [float((a - b.detach()).abs().max()) for a, b in zip(moved[0], base[0].parameters())]
    assert max(diffs) < 0.02 * max(steps)


def test_replay_window_keeps_the_latest_rows():
    window = sl.ReplayWindow(3)
    for i in range(5):
        window.add(i)
    assert sorted(window.rows) == [2, 3, 4]
    assert sorted(window.sample(10, np.random.default_rng(0))) == [2, 3, 4]


# ----------------------------------------------------------------------------- end to end


def _save_checkpoints(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _save

    source = tmp_path / "source"
    (source / "learner").mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    _save(_tiny_model(0), source / "learner" / "100.pth")
    _save(_tiny_model(1), source / "critic" / "100.pth")
    return source / "learner" / "100.pth", source / "critic" / "100.pth"


def test_a_tiny_cpu_run_trains_saves_reloads_and_resumes(tmp_path, start_file, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    policy, value = _save_checkpoints(tmp_path)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint=str(policy), value_checkpoint=str(value), out_dir=str(tmp_path / "runs"), run_name="tiny",
        p_full=0.5, n_full=4, parallel_readouts=2, workers=2, games_per_worker=2, task_seconds=2.0,
        max_decisions=16, start_positions=start_file, random_start_fraction=1.0, inference_batch_size=64,
        window_rows=400, rows_per_update=10, min_window_rows=10, steps_per_update=2, minibatch=8, microbatch=4,
        max_updates=2, snapshot_every=1, anchor_every=0, kl_target=0.05, critic_positions_per_game=4,
    )
    run_dir = sl.run(cfg)
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2]
    config = json.loads((run_dir / "config.json").read_text())
    assert config["learner_temperature"] == 0.5 and config["prior_temperature"] == 0.5  # from the policy's run
    for h in history:
        assert h["rows_trained"] == 16 and h["new_rows"] >= 10 and h["searched_rows_per_hour"] > 0
        assert math.isfinite(h["ce"]) and math.isfinite(h["kl_anchor"]) and h["reuse_update"] > 0
    for update in (0, 1, 2):
        assert (run_dir / "learner" / f"{update}.pth").exists() and (run_dir / "critic" / f"{update}.pth").exists()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["snapshot_update"] == 2

    again = copy.copy(cfg)
    again.max_updates = 3
    resumed = sl.run(again, resume=str(run_dir))
    assert resumed == run_dir
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["update"] for h in history] == [1, 2, 3]
    assert history[2]["kl_coef"] in (state["kl_coef"], state["kl_coef"] * 1.5, max(state["kl_coef"] / 1.5, 0.005))
    assert json.loads((run_dir / "state.json").read_text())["snapshot_update"] == 3
    assert (run_dir / "learner" / "3.pth").exists()


def test_sharpen_target_one_hot_and_power():
    from rl18xx.agent.alphazero.search_learning import sharpen_target

    legal = np.array([7, 3, 9])
    target = np.array([0.5, 0.3, 0.2])
    one_hot = sharpen_target(target, legal, move=3, temperature=0.0)
    assert one_hot.tolist() == [0.0, 1.0, 0.0]
    np.testing.assert_allclose(sharpen_target(target, legal, 7, 1.0), target)
    sharp = sharpen_target(target, legal, 7, 0.5)
    np.testing.assert_allclose(sharp, target**2 / (target**2).sum())
    assert sharp[0] > target[0]
    # An empty target falls back to the move played.
    assert sharpen_target(np.zeros(3), legal, 9, 1.0).tolist() == [0.0, 0.0, 1.0]


def test_fast_searches_play_the_other_decisions_without_rows(start_file):
    # n_fast > 0: every learner decision with a choice is searched -- p_full of them
    # fully (rows), the rest by a fast search (played, no row); none by the policy.
    settings = _settings(
        start_file, p_full=0.25, n_full=2, parallel_readouts=2, games_per_worker=4, max_decisions=60
    )
    settings["n_fast"] = 2
    _, out = _generate(settings, seconds=8.0)
    stats = out["stats"]
    fast_search = stats.get("fast_search_decisions", 0)
    searched = stats.get("searched_decisions", 0)
    policy_with_choice = stats.get("fast_decisions", 0) - stats.get("single_move_decisions", 0)
    assert fast_search > searched > 0
    assert policy_with_choice == 0
    rows = sum(len(g["rows"]) for g in out["games"])
    assert rows <= searched


def test_visit_move_explores_only_the_first_decisions():
    # AlphaZero-style exploration: full_move_temperature applies only while the game has
    # made fewer than explore_decisions decisions (0: always); afterwards the most visited.
    from collections import Counter
    from types import SimpleNamespace

    from rl18xx.agent.alphazero.search_learning import Generator

    class Tree:
        def pick_best_action(self):
            return 7

    legal, visits = np.array([3, 7]), np.array([10.0, 10.0])

    def gen(t, k):
        return SimpleNamespace(
            settings={"full_move_temperature": t, "explore_decisions": k},
            rng=np.random.default_rng(0),
            counts=Counter(),
        )

    g = gen(0.0, 0)
    assert all(Generator._visit_move(g, Tree(), legal, visits, d) == 7 for d in range(20))
    g = gen(1.0, 10)
    early = {Generator._visit_move(g, Tree(), legal, visits, 3) for _ in range(50)}
    assert early == {3, 7}  # sampled
    assert g.counts["explored_moves"] == 50
    assert all(Generator._visit_move(g, Tree(), legal, visits, d) == 7 for d in (10, 11, 500))
    assert g.counts["explored_moves"] == 50  # past the cutoff: the most visited, not counted
    g = gen(1.0, 0)
    assert {Generator._visit_move(g, Tree(), legal, visits, 500) for _ in range(50)} == {3, 7}  # 0: always


def test_generation_explores_the_early_decisions(start_file):
    settings = _settings(
        start_file, target="visits", p_full=0.25, n_full=2, parallel_readouts=2, games_per_worker=4, max_decisions=60
    )
    settings["n_fast"] = 2
    settings["full_move_temperature"] = 1.0
    settings["explore_decisions"] = 5
    _, out = _generate(settings, seconds=8.0)
    stats = out["stats"]
    games = stats.get("games_in_progress", 0) + len(out["games"])
    assert 0 < stats.get("explored_moves", 0) <= 5 * max(games, 1)
