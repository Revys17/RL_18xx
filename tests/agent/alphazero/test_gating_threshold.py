"""Gating promotes a candidate against its fair share of a 4-player gate."""
import pytest

from rl18xx.agent.alphazero import loop


def test_gate_threshold_keeps_its_two_player_meaning():
    assert loop.gate_win_rate_needed(0.55, num_players=2) == pytest.approx(0.55)
    assert loop.gate_win_rate_needed(0.55, num_players=4) == pytest.approx(0.275)


@pytest.mark.parametrize("win_rate, promoted", [(0.3, True), (0.2, False)])
def test_gating_compares_against_the_scaled_threshold(monkeypatch, win_rate, promoted):
    """A candidate winning 30% of 4-player gate games (fair share 25%) clears
    the default 0.55 threshold; one winning 20% doesn't. The raw comparison
    against 0.55 rejected both."""
    promotions = []

    class FakeModel:
        def architecture_name(self):
            return "AlphaZeroTransformer"

    class FakeMetrics:
        def add_scalar(self, *args, **kwargs):
            pass

    monkeypatch.setattr(loop, "session_name_for", lambda model: "session")
    monkeypatch.setattr(loop, "get_latest_model", lambda checkpoint_dir: object())
    monkeypatch.setattr(loop, "evaluate_candidate", lambda **kwargs: win_rate)
    monkeypatch.setattr(loop, "set_current_best", lambda *args: promotions.append(args))
    monkeypatch.setattr(loop, "append_model_history", lambda record: None)
    monkeypatch.setattr(loop, "update_loop_status", lambda status: None)

    result = loop._run_gating_iteration(
        loop=1, model=FakeModel(), candidate_checkpoint_num=7, no_gate=False, gate_games=10,
        gate_threshold=0.55, num_readouts=50, status={}, metrics=FakeMetrics(),
    )
    assert result.promoted is promoted
    assert bool(promotions) is promoted
