"""The AlphaGo-style split (``shared_split``): one shared value-encoder network whose
policy path trains by policy gradient (``policy-gradient --shared``) and whose value
side trains on search games (``search-learning --shared --freeze-policy``), each run
following the other's latest checkpoint."""

import math

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import policy_gradient as pg  # noqa: E402
from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402
from rl18xx.agent.alphazero import shared_split as ss  # noqa: E402
from rl18xx.agent.alphazero.mcts import _rust_encode  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from .search_test_utils import HashNet, random_positions  # noqa: E402
from .test_search_learning_az2 import _generator, _random_value_heads, start_file  # noqa: E402,F401
from .test_search_learning import _settings  # noqa: E402

CPU = torch.device("cpu")


def _tiny_shared(seed: int):
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    torch.manual_seed(seed)
    config = ModelTransformerConfig(
        device=torch.device("cpu"), d_entity=32, econ_transformer_layers=1, econ_transformer_heads=2,
        econ_transformer_ff_dim=64, d_map=32, hex_transformer_layers=1, hex_transformer_heads=2,
        hex_transformer_ff_dim=64, cross_attn_heads=2, d_trunk=64, num_res_blocks=1, value_head_layers=1,
        value_entity_encoder=True,
    )
    return _random_value_heads(AlphaZeroTransformerModel(config), seed + 100).eval()


def _states(n=4):
    return [_rust_encode(RustGameAdapter(g)) for g in random_positions(4, 3, steps=150, every=31)[:n]]


def _outputs(model, states):
    with torch.no_grad():
        policy, win, score, _ = model._forward_encoded_batch(states)
    return policy, win, score


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _changed(before, params) -> bool:
    return any(not torch.equal(a, b.detach()) for a, b in zip(before, params))


def test_the_value_side_and_policy_path_partition_the_network():
    model = _tiny_shared(0)
    value, policy = ss.value_parameters(model), ss.policy_parameters(model)
    assert value and policy
    assert len(value) + len(policy) == len(list(model.parameters()))
    assert {id(p) for p in value}.isdisjoint({id(p) for p in policy})
    assert {id(p) for p in model.value_econ_transformer.parameters()} <= {id(p) for p in value}
    assert not any(ss.is_value_key(k) for k in model.state_dict() if k.startswith(("econ_transformer.", "price_head.")))


def test_load_part_moves_one_side_bit_for_bit():
    a, b = _tiny_shared(0), _tiny_shared(1)
    states = _states()
    policy_a, win_a, _ = _outputs(a, states)
    ss.load_part(a, b.state_dict(), "value")
    policy_after, win_after, _ = _outputs(a, states)
    assert torch.equal(policy_after, policy_a)  # the policy path is untouched
    assert not torch.allclose(win_after, win_a)
    ss.load_part(a, b.state_dict(), "policy")
    policy_b, win_b, score_b = _outputs(b, states)
    p, w, s = _outputs(a, states)
    assert torch.equal(p, policy_b) and torch.equal(w, win_b) and torch.equal(s, score_b)


def test_load_part_refuses_a_network_without_a_value_encoder():
    from .test_search_learning import _tiny_model

    plain = _tiny_model(0)
    with pytest.raises((KeyError, ValueError)):
        ss.load_part(plain, _tiny_shared(1).state_dict(), "value")


def test_follower_loads_only_newer_checkpoints(tmp_path):
    run = tmp_path / "run"
    source, model = _tiny_shared(1), _tiny_shared(0)
    follower = ss.Follower(run, "value")
    assert follower.poll(model) is None  # no run yet
    pg._save(source, run / "learner" / "1.pth")
    assert follower.poll(model) == 1
    assert follower.poll(model) is None
    for key, value in source.state_dict().items():
        if ss.is_value_key(key):
            assert torch.equal(model.state_dict()[key], value)
    pg._save(_tiny_shared(2), run / "learner" / "3.pth")
    pg._save(_tiny_shared(3), run / "learner" / "2.pth")  # older than what it has
    assert follower.poll(model) == 3 and follower.poll(model) is None
    assert ss.latest_checkpoint(run)[0] == 3 and follower.loads == 2


def _pg_rows(n=12):
    rows = []
    for g in random_positions(4, 5, steps=300, every=19)[:n]:
        legal = [int(x) for x in g.factored_legal_indices()]
        enc = _rust_encode(RustGameAdapter(g))
        rows.append((enc, np.asarray(legal, dtype=np.int32), legal[0], float(np.log(1.0 / len(legal))), None, 0.7,
                     np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32), None))
    return rows


def _pg_cfg(**kw):
    cfg = pg.PGConfig(policy_checkpoint="p", value_checkpoint="", shared=True, minibatch=6, microbatch=3,
                      lr=1e-3, critic_lr=1e-3, learner_temperature=0.5, kl_coef=0.0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_pg_shared_update_trains_the_policy_path_and_leaves_the_value_side():
    learner = _tiny_shared(0)
    anchor = _tiny_shared(0)
    cfg = _pg_cfg()
    opt = torch.optim.AdamW(pg.shared_param_groups(learner, cfg), weight_decay=0.0)
    assert [g["name"] for g in opt.param_groups] == ["policy_path", "value_side"]
    rows, baseline = pg.subtract_baselines(learner, _pg_rows(), cfg, CPU)
    assert math.isfinite(baseline["critic_row_brier"]) and all(r[5] != 0.7 for r in rows)
    value_before, policy_before = _snapshot(ss.value_parameters(learner)), _snapshot(ss.policy_parameters(learner))
    stats = pg._update(learner, None, anchor, opt, None, rows, cfg, 0.0, CPU, np.random.default_rng(0))
    assert not _changed(value_before, ss.value_parameters(learner))
    assert _changed(policy_before, ss.policy_parameters(learner))
    assert stats["optimizer_steps"] == 2 and learner.value_stop_grad is False


def test_pg_value_pg_weight_trains_the_value_side_on_detached_trunk_features():
    learner = _tiny_shared(0)
    anchor = _tiny_shared(0)
    cfg = _pg_cfg(value_pg_weight=1.0)
    groups = pg.shared_param_groups(learner, cfg)
    groups[0]["lr"] = 0.0  # the policy loss moves nothing: whatever moves came from the value loss
    opt = torch.optim.SGD(groups, lr=0.1)
    value_before, policy_before = _snapshot(ss.value_parameters(learner)), _snapshot(ss.policy_parameters(learner))
    stats = pg._update(learner, None, anchor, opt, None, _pg_rows(), cfg, 0.0, CPU, np.random.default_rng(0))
    assert _changed(value_before, ss.value_parameters(learner))
    assert not _changed(policy_before, ss.policy_parameters(learner))
    assert math.isfinite(stats["critic_ce"])


def _search_rows(start_file, n=24):
    settings = _settings(start_file, critic_positions_per_game=0)
    settings.update(value_lambda=0.97, value_bootstrap_weight=0.5)
    rows = [r for g in _generator(settings, HashNet()).run(5.0)["games"] for r in g["rows"]]
    assert len(rows) >= n
    return rows[:n]


def test_value_only_search_learning_trains_the_value_side_once_per_row(start_file):  # noqa: F811
    rows = _search_rows(start_file)
    learner = _tiny_shared(0)
    cfg = sl.SearchLearningConfig(
        policy_checkpoint="p", value_checkpoint="", learner_temperature=0.5, shared=True, freeze_policy=True,
        steps_per_update=4, minibatch=8, microbatch=4, critic_lr=1e-3, aux_loss_weight=0.0,
    )
    sl._validate(cfg)
    groups = sl.learner_param_groups(learner, cfg)
    assert len(groups) == 1 and {id(p) for p in groups[0]["params"]} == {id(p) for p in ss.value_parameters(learner)}
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    window = sl.ReplayWindow(len(rows))
    for r in rows:
        window.add(r)
    policy_before, value_before = _snapshot(ss.policy_parameters(learner)), _snapshot(ss.value_parameters(learner))
    stats = sl._update_value_only(learner, opt, window, cfg, CPU, np.random.default_rng(0))
    assert stats["critic_rows_trained"] == len(rows) == stats["rows_trained"]
    assert not any(r.critic for r in rows)
    assert math.isfinite(stats["critic_ce"]) and math.isnan(stats["ce"])
    assert not _changed(policy_before, ss.policy_parameters(learner))
    assert _changed(value_before, ss.value_parameters(learner))
    again = sl._update_value_only(learner, opt, window, cfg, CPU, np.random.default_rng(0))
    assert again["rows_trained"] == 0  # each row trains the value once


def test_split_options_validate():
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(policy_checkpoint="p", value_checkpoint="v", freeze_policy=True))
    with pytest.raises(SystemExit):
        sl._validate(sl.SearchLearningConfig(policy_checkpoint="p", value_checkpoint="", shared=True,
                                             policy_follow="x"))
    sl._validate(sl.SearchLearningConfig(policy_checkpoint="p", value_checkpoint="", shared=True,
                                         freeze_policy=True, policy_follow="x"))
