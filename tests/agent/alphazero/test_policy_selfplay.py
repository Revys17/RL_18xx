"""Search-free self-play for value-network data (policy_selfplay)."""
import numpy as np
import pytest
import torch

from rl18xx.agent.alphazero import inference_server, policy_selfplay
from rl18xx.agent.alphazero.mcts import POLICY_SIZE


class _UniformClient:
    """Stands in for the inference server: uniform priors, no price head."""

    last_price_components = None

    def __init__(self):
        self.batches = []

    def run_many_encoded(self, states, legal_indices=None):
        self.batches.append(len(states))
        probs = [torch.full((POLICY_SIZE,), 1.0 / POLICY_SIZE) for _ in states]
        return probs, None, [torch.zeros(6) for _ in states]


@pytest.fixture
def start_file(tmp_path):
    """An empty start file: every game takes a random auction ending."""
    path = tmp_path / "starts.jsonl"
    path.write_text("")
    return str(path)


def test_policy_games_keep_a_few_positions_per_game_valued_by_the_outcome(monkeypatch, start_file):
    client = _UniformClient()
    monkeypatch.setattr(inference_server, "get_worker_client", lambda: client)
    settings = {
        "positions_per_game": 3,
        "temperature": 1.0,
        "max_decisions": 40,
        "start_positions": start_file,
        "random_start_fraction": 1.0,
        "price_eps": 0.05,
    }
    result = policy_selfplay.play_policy_games(4, settings)
    games = result["games"]
    assert len(games) == 4
    assert max(client.batches) == 4  # the games' decisions go to the network together
    for game in games:
        assert game["start"] == "random"
        assert game["termination"] == "max_length" and game["decisions"] == 40
        assert len(game["rows"]) == 3
        for state, legal, pi, value, price_targets in game["rows"]:
            assert len(state) == 8 and int(state[7]) == 4  # full encoding, rotation at 6
            assert len(legal) > 1  # decisions only, never forced moves
            assert abs(float(pi.sum()) - 1.0) < 1e-5 and float(pi[legal].sum()) == pytest.approx(1.0, abs=1e-5)
            assert value.shape == (4,) and float(value.sum()) == pytest.approx(1.0, abs=1e-5)
            assert price_targets == []
        values = {tuple(row[3].tolist()) for row in game["rows"]}
        assert len(values) == 1  # every kept position of a game gets that game's outcome
    assert result["decisions"] == 4 * 40


def test_sampled_prices_lie_in_the_legal_range():
    from rl18xx.agent.alphazero.start_positions import new_game

    game = new_game(4)
    rng = np.random.default_rng(0)
    for index in game._game.factored_legal_indices():
        price = policy_selfplay._sample_price(game, int(index), None, rng, 0.05)
        price_range = game._game.price_range_for_index(int(index))
        if price_range is None:
            assert price is None
        else:
            assert price_range[0] <= price <= price_range[1]
