"""Policy-gradient refinement (policy_gradient): games, advantages and losses."""
import math

import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero import policy_gradient as pg
from rl18xx.agent.alphazero.mcts import POLICY_SIZE

CRITIC_VALUE = 0.3


class _Client:
    """Stands in for an inference server: uniform priors, no price head, the
    critic's value CRITIC_VALUE for the mover (canonical slot 0)."""

    last_price_components = None

    def __init__(self):
        self.rotations = []

    def send(self, states, legal_indices=None):
        return states

    def receive(self, states):
        return self.run_many_encoded(states)

    def run_many_encoded(self, states, legal_indices=None):
        self.rotations.extend(int(s[6]) for s in states)
        probs = [torch.full((POLICY_SIZE,), 1.0 / POLICY_SIZE) for _ in states]
        values = [torch.tensor([CRITIC_VALUE, 0.2, 0.2, 0.3, 0.0, 0.0]) for _ in states]
        return probs, None, values


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def _settings(start_file, **overrides):
    settings = {
        "positions_per_seat": 5,
        "learner_seats": 2,
        "sl_opponent_fraction": 0.5,
        "pool_ready": False,
        "pool_label": None,
        "opponent_temperature": 1.0,
        "opponent_price_eps": 0.05,
        "price_eps": 0.05,
        "max_decisions": 40,
        "start_positions": start_file,
        "random_start_fraction": 1.0,
        "gae_lambda": 1.0,
    }
    settings.update(overrides)
    return settings


def test_seat_advantages_are_monte_carlo_at_lambda_one_and_td_at_zero():
    values = [0.2, 0.4, 0.1]
    assert pg.seat_advantages(values, 1.0, 1.0) == pytest.approx([0.8, 0.6, 0.9])
    assert pg.seat_advantages(values, 1.0, 0.0) == pytest.approx([0.2, -0.3, 0.9])


def test_learner_seats_play_against_one_opponent_and_only_their_decisions_are_kept(monkeypatch, start_file):
    clients = {"learner": _Client(), "sl": _Client(), "pool": _Client()}
    monkeypatch.setattr(pg, "_CLIENTS", clients)
    result = pg.play_pg_games(3, _settings(start_file))
    assert len(result["games"]) == 3
    assert not clients["pool"].rotations  # no snapshots yet: every opponent is the supervised policy
    for game in result["games"]:
        assert game["opponent"] == "sl" and len(game["learner_seats"]) == 2
        assert game["termination"] == "max_length" and game["decisions"] == 40
        assert sum(game["win_share"]) == pytest.approx(1.0)
        assert len(game["rows"]) == min(10, game["learner_decisions"])
        for encoded, legal, choice, old_logp, price_info, advantage, fractions in game["rows"]:
            assert int(encoded[6]) in game["learner_seats"]  # the mover is a learner seat
            assert len(legal) > 1 and choice in legal
            if price_info is None:
                assert old_logp == pytest.approx(-math.log(len(legal)))  # uniform prior
            assert advantage == pytest.approx(game["win_share"][int(encoded[6])] - CRITIC_VALUE)
            assert fractions.shape == (4,) and float(fractions.sum()) == pytest.approx(1.0, abs=1e-5)
    assert clients["learner"].rotations and clients["sl"].rotations


def test_games_use_the_pool_once_it_has_snapshots(monkeypatch, start_file):
    clients = {"learner": _Client(), "sl": _Client(), "pool": _Client()}
    monkeypatch.setattr(pg, "_CLIENTS", clients)
    result = pg.play_pg_games(4, _settings(start_file, pool_ready=True, sl_opponent_fraction=0.0, max_decisions=8))
    assert {g["opponent"] for g in result["games"]} == {"pool"}
    assert clients["pool"].rotations and not clients["sl"].rotations


def _batch(logits_rows, legal_rows, choices, old_logp, advantage, price=None):
    """A pg_losses batch over a tiny action space (POLICY_SIZE is only used by batch_tensors)."""
    n = len(choices)
    legal = torch.zeros(n, len(logits_rows[0]), dtype=torch.bool)
    for b, ix in enumerate(legal_rows):
        legal[b, ix] = True
    batch = {
        "legal": legal,
        "choice": torch.tensor(choices),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32),
        "advantage": torch.tensor(advantage, dtype=torch.float32),
        "price_rows": torch.zeros(0, dtype=torch.long),
        "price_slots": torch.zeros(0, dtype=torch.long),
        "price_cells": torch.zeros(0, dtype=torch.long),
        "price_masks": torch.zeros(0, 4, dtype=torch.bool),
    }
    if price:
        batch.update(price)
    return batch


def test_policy_gradient_raises_good_moves_and_lowers_bad_ones():
    logits = torch.zeros(2, 4, requires_grad=True)
    batch = _batch([[0.0] * 4] * 2, [[0, 1, 2], [0, 1, 2]], [0, 1], [-math.log(3)] * 2, [0.5, -0.5])
    out = pg.pg_losses(logits, None, logits.detach(), None, batch, clip=0.2, kl_coef=0.1, entropy_coef=0.0)
    assert float(out["kl_sl"]) == pytest.approx(0.0, abs=1e-7)  # learner == supervised
    assert float(out["approx_kl"]) == pytest.approx(0.0, abs=1e-6)  # on-policy
    out["total"].backward()
    assert logits.grad[0, 0] < 0 and logits.grad[1, 1] > 0  # descent raises row 0's move, lowers row 1's
    assert logits.grad[0, 3] == 0  # illegal moves get nothing


def test_clipped_ratio_stops_the_push_and_kl_pulls_back_to_the_supervised_policy():
    # The learner already plays move 0 far more often than when it was sampled.
    logits = torch.tensor([[3.0, 0.0, 0.0]], requires_grad=True)
    batch = _batch([[0.0] * 3], [[0, 1, 2]], [0], [-math.log(3)], [1.0])
    out = pg.pg_losses(logits, None, torch.zeros(1, 3), None, batch, clip=0.2, kl_coef=0.0, entropy_coef=0.0)
    assert float(out["clip_frac"]) == 1.0
    out["total"].backward()
    assert torch.all(logits.grad == 0)

    logits.grad = None
    out = pg.pg_losses(logits, None, torch.zeros(1, 3), None, batch, clip=0.2, kl_coef=1.0, entropy_coef=0.0)
    assert float(out["kl_sl"]) > 0
    out["total"].backward()
    assert logits.grad[0, 0] > 0  # descent lowers the move the supervised policy doesn't favor


def test_price_cells_count_in_the_moves_probability():
    logits = torch.zeros(1, 3, requires_grad=True)
    price_logits = torch.zeros(1, 2, 4, requires_grad=True)  # (B, slots, cells)
    price = {
        "price_rows": torch.tensor([0]),
        "price_slots": torch.tensor([1]),
        "price_cells": torch.tensor([2]),
        "price_masks": torch.tensor([[True, True, True, False]]),
    }
    # Sampled with the probabilities the learner still has: index 1/3, cell 1/3.
    batch = _batch([[0.0] * 3], [[0, 1, 2]], [1], [2 * -math.log(3)], [1.0], price=price)
    out = pg.pg_losses(logits, price_logits, logits.detach(), price_logits.detach(), batch, 0.2, 0.0, 0.0)
    assert float(out["approx_kl"]) == pytest.approx(0.0, abs=1e-6)
    out["total"].backward()
    assert price_logits.grad[0, 1, 2] < 0 and price_logits.grad[0, 1, 3] == 0 and torch.all(price_logits.grad[0, 0] == 0)


def test_batch_rotates_value_targets_into_the_movers_frame(monkeypatch, start_file):
    monkeypatch.setattr(pg, "_CLIENTS", {"learner": _Client(), "sl": _Client(), "pool": _Client()})
    rows = [r for g in pg.play_pg_games(2, _settings(start_file))["games"] for r in g["rows"]]
    batch = pg.batch_tensors(rows, torch.device("cpu"))
    for b, row in enumerate(rows):
        rotation = int(row[0][6])
        assert batch["fractions"][b, 0] == pytest.approx(float(row[6][rotation]))
        assert batch["legal"][b].sum() == len(row[1]) and batch["legal"][b, row[2]]
    assert batch["game_state"].shape == (len(rows), rows[0][0][0].numel())


def test_update_moves_the_learner_and_critic_but_not_the_supervised_policy(monkeypatch, start_file):
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    monkeypatch.setattr(pg, "_CLIENTS", {"learner": _Client(), "sl": _Client(), "pool": _Client()})
    rows = [r for g in pg.play_pg_games(1, _settings(start_file, max_decisions=12))["games"] for r in g["rows"]][:6]
    torch.manual_seed(0)
    models = [AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))) for _ in range(3)]
    learner, critic, sl = models
    sl.load_state_dict(learner.state_dict())
    before = {name: [p.detach().clone() for p in m.parameters()] for name, m in zip("lcs", models)}
    cfg = pg.PGConfig(policy_checkpoint="", value_checkpoint="", minibatch=4, lr=1e-3, critic_lr=1e-3)
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0) for m in (learner, critic)]
    stats = pg._update(learner, critic, sl, *opts, rows, cfg, 0.02, torch.device("cpu"), np.random.default_rng(0))
    assert all(math.isfinite(v) for v in stats.values())
    assert stats["rows"] == len(rows)

    def changed(name, model):
        return any(not torch.equal(a, b) for a, b in zip(before[name], model.parameters()))

    assert changed("l", learner) and changed("c", critic) and not changed("s", sl)


def test_value_only_forward_matches_the_full_forward():
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
    from rl18xx.agent.alphazero.start_positions import new_game

    torch.manual_seed(0)
    model = AlphaZeroTransformerModel(ModelTransformerConfig(device=torch.device("cpu"))).eval()
    encoded = [_rust_encode(new_game(4))] * 2
    with torch.no_grad():
        _, _, values = model.run_many_encoded(encoded)
        assert torch.allclose(model.run_values_encoded(encoded), values)
        policy, *_ = model._forward_encoded_batch(encoded, value_only=True)
    assert policy is None and model.last_price_components is None
