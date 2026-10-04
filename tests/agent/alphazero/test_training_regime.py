"""Self-play training regime: sampled windows, the value generalization check, and stopping."""
import math

import pytest
import torch

from rl18xx.agent.alphazero.config import ModelTransformerConfig, SelfPlayConfig, TrainingConfig
from rl18xx.agent.alphazero.dataset import SelfPlayDataset
from rl18xx.agent.alphazero.mcts import VALUE_SIZE
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.dataset import collate_examples
from rl18xx.agent.alphazero.train import compute_losses, move_batch_to_device, train, value_cross_entropy

from tests.agent.alphazero.test_train_divergence import _UniformNet


@pytest.fixture
def selfplay_lmdb(tmp_path, monkeypatch):
    """A small real self-play LMDB (one short 4-player game)."""
    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()
    config = SelfPlayConfig(
        network=_UniformNet(), num_readouts=4, parallel_readouts=2, min_readouts=2, use_score_values=True,
        player_count_distribution={4: 1.0}, enable_resign=False, auction_stall_moves=40, game_id="regime",
    )
    self_play.SelfPlay(config).run_game()
    return tmp_path / "training_examples" / "selfplay" / "dummy"


def test_training_samples_a_subset_of_the_window(selfplay_lmdb, tmp_path):
    total = len(SelfPlayDataset(selfplay_lmdb))
    assert total > 20
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
    config = TrainingConfig(
        train_dir=selfplay_lmdb, num_epochs=1, batch_size=8, max_training_window=0, train_samples_per_iteration=16,
    )
    _, metrics = train(config, model, model_checkpoint_dir=str(tmp_path / "checkpoints"))
    assert metrics.training_examples == 16


def test_value_stop_grad_keeps_value_gradients_out_of_the_trunk(selfplay_lmdb, tmp_path):
    """With ``value_stop_grad`` the value loss trains only the value heads, so
    the trunk can't learn features that identify games."""
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu")))
    config = TrainingConfig(
        train_dir=selfplay_lmdb, num_epochs=1, batch_size=8, max_training_window=0, train_samples_per_iteration=8,
        value_stop_grad=True,
    )
    train(config, model, model_checkpoint_dir=str(tmp_path / "checkpoints"))
    assert model.value_stop_grad

    dataset = SelfPlayDataset(selfplay_lmdb)
    game_state, data, legal, pi, value, price_targets = move_batch_to_device(
        collate_examples([dataset[i] for i in range(4)]), torch.device("cpu")
    )
    # A short fixture game can end level (a stalled auction), and a uniform
    # target against an untrained head gives no gradient at all.
    value = torch.zeros_like(value)
    value[:, 0] = 1.0
    model.zero_grad()
    compute_losses(model, game_state, data, legal, pi, value, config, price_targets)["value_loss"].backward()
    grads = {name: param.grad for name, param in model.named_parameters()}
    value_head = [g for name, g in grads.items() if "win_loss_head" in name]
    elsewhere = [g for name, g in grads.items() if "win_loss_head" not in name and "score_head" not in name]
    assert any(g is not None and g.abs().sum() > 0 for g in value_head)
    assert all(g is None or g.abs().sum() == 0 for g in elsewhere)


def test_value_cross_entropy_of_a_uniform_head_is_log_players(selfplay_lmdb):
    """The generalization check reports the training loop's value loss: a
    head that can't tell 4 players apart scores ln 4."""

    class UniformValueModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, game_state_data, batch_data):
            logits = torch.full((len(game_state_data), VALUE_SIZE), -1e4)
            logits[:, :4] = 0.0
            return None, logits, None, None

    dataset = SelfPlayDataset(selfplay_lmdb)
    assert value_cross_entropy(UniformValueModel(), dataset, list(range(10))) == pytest.approx(math.log(4), abs=1e-4)


def test_stop_signal_exits_after_cleaning_up(monkeypatch):
    """SIGTERM/SIGINT used to clean up the children (inference server,
    self-play pool) and then return, so the loop carried on without them."""
    from rl18xx.agent.alphazero import loop

    with pytest.raises(SystemExit) as exc:
        loop.cleanup_and_exit(signum=15)
    assert exc.value.code == 143
    loop.cleanup_and_exit()  # the atexit path just cleans up


def test_an_earlier_runs_status_files_are_not_this_iterations(tmp_path, monkeypatch):
    """Iteration indices restart at 0 every run; files an earlier run left for
    this index were read back as this iteration's games (resign calibration,
    phase counts, unlock rate)."""
    import os

    from rl18xx.agent.alphazero import loop

    monkeypatch.setattr(loop, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path)
    for name in ("L3_G0.json", "L3_G150.json", "L4_G0.json", "L31_G0.json"):
        (tmp_path / name).write_text("{}")
    os.utime(tmp_path / "L3_G150.json", (1000.0, 1000.0))  # an earlier run's game
    started = (tmp_path / "L3_G0.json").stat().st_mtime
    assert [p.name for p in loop._iteration_status_files(3, since=started)] == ["L3_G0.json"]
    assert len(list(tmp_path.iterdir())) == 4  # nothing deleted


def test_selfplay_overrides_keep_only_selfplay_config_fields():
    from rl18xx.agent.alphazero import loop

    cfg = {"selfplay_overrides": {"mcts_mean_q": True, "parallel_readouts": 8, "not_a_field": 1}}
    assert loop._selfplay_overrides(cfg) == {"mcts_mean_q": True, "parallel_readouts": 8}
    assert loop._selfplay_overrides(None) == {}


def test_eval_player_spec_searches_like_self_play():
    from rl18xx.agent.alphazero import loop

    pointer = {"session": "S", "checkpoint_num": 12}
    assert loop._eval_player_spec(pointer, {}, 64) == "S/12@64"
    overrides = {"c_puct_init": 0.5, "mcts_mean_q": True, "fpu_reduction": 0.1, "parallel_readouts": 8}
    assert loop._eval_player_spec(pointer, overrides, 64) == "S/12@64/0.5:mean=0.1,par=8"


def test_periodic_evaluator_runs_every_n_iterations_and_logs_the_score(tmp_path, monkeypatch):
    """The eval runs in a background process (stubbed here) every eval_every
    iterations; its score lands in TensorBoard and eval_history.jsonl."""
    import json
    import textwrap

    from rl18xx.agent.alphazero import loop

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "loop_config.json"
    config_path.write_text(json.dumps({"eval_every": 2, "eval_games": 6}))
    monkeypatch.setattr(loop, "LOOP_CONFIG_PATH", config_path)
    monkeypatch.setattr(loop, "EVAL_HISTORY_PATH", tmp_path / "eval_history.jsonl")
    fake_eval = tmp_path / "fake_eval.py"
    fake_eval.write_text(textwrap.dedent('''
        import json, sys
        args = sys.argv[1:]
        out = args[args.index("--out") + 1]
        json.dump([{"A_score": 0.6, "A_score_se": 0.05, "games": 6, "argv": args}], open(out + "/summary.json", "w"))
    '''))
    monkeypatch.setattr(loop, "EVAL_SCRIPT", fake_eval)
    monkeypatch.setattr(loop, "get_current_best", lambda _dir: {"session": "S", "checkpoint_num": 9})
    logged = []

    class Metrics:
        def add_scalar(self, name, value, step):
            logged.append((name, value, step))

    evaluator = loop.PeriodicEvaluator({"session": "S", "checkpoint_num": 7}, "T", Metrics())
    evaluator.maybe_start(0, 64)  # iteration 1: not due
    assert evaluator.proc is None
    evaluator.maybe_start(1, 64)  # iteration 2: due
    assert evaluator.proc is not None
    evaluator.poll(wait=True)
    assert ("Eval/Score_vs_Start", 0.6, 1) in logged
    record = json.loads((tmp_path / "eval_history.jsonl").read_text())
    assert record["loop"] == 2 and record["score"] == 0.6
    argv = json.loads((tmp_path / "logs/eval/loop_T/iter_2/summary.json").read_text())[0]["argv"]
    assert argv[argv.index("--match") + 1 : argv.index("--match") + 3] == ["S/9@64", "S/7@64"]
