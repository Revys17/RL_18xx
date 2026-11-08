"""``policy_selfplay.tempered``: every sampler's softmax(log p / T) over the legal moves -- what
training's log_softmax(logits / T) over the legal moves reads -- even when the legal moves hold
almost none of the full softmax's mass (every sampler clipped at 1e-12 before 2026-11-08)."""

import numpy as np
import pytest

from rl18xx.agent.alphazero.policy_selfplay import tempered


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


@pytest.mark.parametrize("temperature", [0.5, 1.0, 0.7, 2.0])
def test_tempered_is_the_legal_softmax_at_the_temperature(temperature):
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, 26537)
    logits[:20] -= 40.0  # 20 legal moves with ~1e-17 of the full softmax each, below the old 1e-12 floor
    legal = np.arange(20)
    full = _softmax(logits)
    assert full[legal].max() < 1e-12
    p = tempered(full[legal].astype(np.float32), temperature)  # the server's float32 probabilities
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p, _softmax(logits[legal] / temperature), rtol=1e-4)
    assert p.max() / p.min() > 10  # not flattened


def test_tempered_falls_back_to_uniform_without_legal_mass():
    assert np.allclose(tempered(np.zeros(4), 0.5), 0.25)
    assert np.allclose(tempered(np.array([np.nan, 1.0]), 0.5), 0.5)


def test_a_move_without_probability_stays_unplayed():
    p = tempered(np.array([0.0, 1e-30, 1e-31]), 0.5)
    assert p[0] == 0.0 and p[1] == pytest.approx(100 / 101)
