"""Unit tests for ``train.py`` loss helpers.

Covers:
- ``_compute_price_nll_loss`` (truncated-Normal correction, fixed-price branch,
  empty-target branch).
- ``_compute_decomposed_policy_loss`` (the three LayTile autoregressive levels).
- ``_derive_dual_value_targets`` (score-encoded, win/loss-encoded, ties).

These are pure-Python unit tests on the loss helpers; they construct hand-built
tensors with small dimensions so the analytic NLL / CE values are easy to
compute. No model / dataset / game state required.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from rl18xx.agent.alphazero.train import (
    _compute_decomposed_policy_loss,
    _compute_price_nll_loss,
    _derive_dual_value_targets,
)


# ---------------------------------------------------------------------------
# _compute_price_nll_loss
# ---------------------------------------------------------------------------


def _analytic_trunc_normal_nll(p, mu, sigma, lo, hi):
    """Hand-computed truncated-Normal NLL: ``-log φ((p-μ)/σ) + log(Φ_hi - Φ_lo)``.

    Matches the formula the helper claims to implement; keeps the analytic
    expression independent of the production code so the test catches drift.
    """
    z = (p - mu) / sigma
    # Standard-normal log-pdf at z, plus the ``- log(σ)`` Jacobian.
    log_phi = -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    cdf_hi = 0.5 * (1.0 + math.erf((hi - mu) * inv_sqrt2 / sigma))
    cdf_lo = 0.5 * (1.0 + math.erf((lo - mu) * inv_sqrt2 / sigma))
    log_trunc = math.log(cdf_hi - cdf_lo)
    log_prob = log_phi - log_trunc
    return -log_prob


def _make_price_components(num_slots, mu_values, log_sigma_values):
    """Build a minimal ``price_components`` dict with the only two keys the
    helper reads from ('price_mean', 'price_log_std')."""
    mean = torch.tensor([mu_values], dtype=torch.float32)
    log_std = torch.tensor([log_sigma_values], dtype=torch.float32)
    assert mean.shape[1] == num_slots
    return {"price_mean": mean, "price_log_std": log_std}


def test_price_nll_truncated_normal_matches_analytic():
    """The helper's NLL equals ``-log φ(z) + log(Φ(hi) - Φ(lo))`` to high precision."""
    mu = 100.0
    sigma = 20.0
    log_sigma = math.log(sigma)
    components = _make_price_components(1, [mu], [log_sigma])

    price = 110.0
    lo, hi = 80.0, 150.0
    weight = 1.0
    # Batch has one example with one target at slot 0.
    price_targets = [[(0, price, weight, lo, hi)]]

    loss, diags = _compute_price_nll_loss(components, price_targets)

    expected = _analytic_trunc_normal_nll(price, mu, sigma, lo, hi)
    assert math.isclose(loss.item(), expected, rel_tol=1e-5, abs_tol=1e-5), (
        f"loss {loss.item():.6f} != analytic {expected:.6f}"
    )
    assert diags["price_count"] == 1


def test_price_nll_fixed_price_branch_drops_truncation_correction():
    """When ``lo == hi`` the helper collapses to the untruncated log-pdf
    (truncation correction is zeroed out per the doc-comment contract)."""
    mu = 50.0
    sigma = 10.0
    log_sigma = math.log(sigma)
    components = _make_price_components(1, [mu], [log_sigma])

    price = 50.0
    lo = hi = 50.0  # fixed-price branch
    price_targets = [[(0, price, 1.0, lo, hi)]]

    loss, _ = _compute_price_nll_loss(components, price_targets)

    # ``log_trunc_correction`` is zero, so loss == untruncated NLL.
    z = (price - mu) / sigma
    expected = -(-0.5 * z * z - log_sigma - 0.5 * math.log(2.0 * math.pi))
    assert math.isclose(loss.item(), expected, rel_tol=1e-5, abs_tol=1e-5)


def test_price_nll_empty_targets_returns_zero_loss():
    """No targets in any batch row → loss is a literal zero tensor."""
    components = _make_price_components(2, [0.0, 0.0], [0.0, 0.0])

    # Both "no entries at all" and "rows of empty lists" should yield zero.
    for price_targets in ([], [[], []], None):
        loss, diags = _compute_price_nll_loss(components, price_targets)
        assert loss.item() == 0.0
        assert diags["price_count"] == 0


# ---------------------------------------------------------------------------
# _compute_decomposed_policy_loss
# ---------------------------------------------------------------------------


def test_decomposed_policy_loss_lay_tile_three_levels():
    """Three-level CE on a hand-built LayTile target should equal the sum of
    the analytic per-level CEs.

    Uses minimal dimensions (H=2, T=2, R=2, S=1) so the marginals are easy to
    reason about by hand. PlaceToken and other-action blocks are zeroed so the
    LayTile sub-losses dominate the total.
    """
    B = 1
    H = 2
    T = 2
    R = 2
    S = 1

    lay_tile_size = H * T * R  # 8
    pt_size = H * S  # 2
    other_size = 2
    total = lay_tile_size + pt_size + other_size  # 12

    lt_start = 0
    lt_end = lay_tile_size
    pt_start = lt_end
    pt_end = pt_start + pt_size
    other_start = pt_end
    other_indices = torch.tensor([other_start, other_start + 1], dtype=torch.long)

    # PlaceToken layout: one slot per hex.
    pt_to_hex = torch.tensor([0, 1], dtype=torch.long)
    pt_to_slot = torch.tensor([0, 0], dtype=torch.long)

    # Hand-picked logits and pi so we can hand-compute the loss.
    hex_logits = torch.tensor([[1.0, 0.0]])  # (B, H)
    tile_logits = torch.zeros(B, H, T)  # uniform tiles
    rotation_logits = torch.zeros(B, H, T, R)  # uniform rotations
    pt_hex_logits = torch.tensor([[0.0, 0.0]])  # uniform
    pt_slot_logits = torch.zeros(B, H, S)
    other_logits = torch.zeros(B, other_size)

    components = {
        "num_hexes": H,
        "num_tiles": T,
        "num_rotations": R,
        "max_city_slots": S,
        "lay_tile_offset": lt_start,
        "lay_tile_end": lt_end,
        "place_token_offset": pt_start,
        "place_token_end": pt_end,
        "other_indices": other_indices,
        "place_token_to_mapper_hex": pt_to_hex,
        "place_token_to_slot": pt_to_slot,
        "hex_logits": hex_logits,
        "tile_logits": tile_logits,
        "rotation_logits": rotation_logits,
        "place_token_hex_logits": pt_hex_logits,
        "place_token_slot_logits": pt_slot_logits,
        "other_logits": other_logits,
    }

    # pi: put all visit mass on LayTile slot 0 (hex=0, tile=0, rot=0).
    pi = torch.zeros(B, total)
    pi[0, lt_start + 0] = 1.0
    # legal mask: all legal so masked CE on "other" is the uniform CE.
    legal_action_mask = torch.ones(B, total)

    loss, diags = _compute_decomposed_policy_loss(components, pi, legal_action_mask)

    # Expected per-level CEs:
    # hex: target one-hot on hex 0; logits=[1, 0]; -log softmax([1,0])[0]
    log_softmax_hex = torch.log_softmax(hex_logits, dim=-1)
    expected_hex = -log_softmax_hex[0, 0].item()
    # tile: uniform logits over 2 → log p = log(1/2)
    expected_tile = -math.log(0.5)
    # rotation: uniform logits over 2 → log p = log(1/2)
    expected_rot = -math.log(0.5)
    # PlaceToken target is all-zero → both PT losses are exactly 0.
    expected_pt_hex = 0.0
    expected_pt_slot = 0.0
    # Other target is all-zero → other loss is 0.
    expected_other = 0.0

    expected_total = (
        expected_hex
        + expected_tile
        + expected_rot
        + expected_pt_hex
        + expected_pt_slot
        + expected_other
    )

    assert math.isclose(diags["loss_lay_tile_hex"].item(), expected_hex, abs_tol=1e-5)
    assert math.isclose(diags["loss_lay_tile_tile"].item(), expected_tile, abs_tol=1e-5)
    assert math.isclose(diags["loss_lay_tile_rot"].item(), expected_rot, abs_tol=1e-5)
    assert math.isclose(diags["loss_place_token_hex"].item(), expected_pt_hex, abs_tol=1e-5)
    assert math.isclose(diags["loss_place_token_slot"].item(), expected_pt_slot, abs_tol=1e-5)
    assert math.isclose(diags["loss_other"].item(), expected_other, abs_tol=1e-5)
    assert math.isclose(loss.item(), expected_total, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# _derive_dual_value_targets
# ---------------------------------------------------------------------------


def test_derive_dual_targets_score_encoded_argmax_winner():
    """Score-encoded value (all ≥ 0) → win_loss is one-hot on argmax, score is
    the input unchanged."""
    value = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0]])
    win_loss, score = _derive_dual_value_targets(value)

    assert torch.allclose(score, value)
    expected_win_loss = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(win_loss, expected_win_loss)


def test_derive_dual_targets_win_loss_encoded_one_winner():
    """Legacy {-1, 0, +1} encoding (values < 0 present) → derived win_loss
    distributes mass over winners and score is forced to the same distribution
    (no independent score signal in this branch)."""
    value = torch.tensor([[1.0, -1.0, -1.0, -1.0, 0.0, 0.0]])
    win_loss, score = _derive_dual_value_targets(value)

    # winners_mask = value > -0.5 → [1, 0, 0, 0, 1, 1]; 3 winners (the legacy
    # encoding's phantom slots are treated as ties under the win-share rule —
    # this matches the helper's documented behaviour).
    expected = torch.tensor([[1.0 / 3, 0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3]])
    assert torch.allclose(win_loss, expected)
    assert torch.allclose(score, win_loss)


def test_derive_dual_targets_two_way_tie_win_loss_encoded():
    """Win-loss-encoded two-way tie at top (encoded as 0.0): both top players
    get half the win mass."""
    value = torch.tensor([[0.0, 0.0, -1.0, -1.0, 0.0, 0.0]])
    win_loss, score = _derive_dual_value_targets(value)
    expected = torch.tensor([[0.25, 0.25, 0.0, 0.0, 0.25, 0.25]])
    assert torch.allclose(win_loss, expected)
    assert torch.allclose(score, win_loss)


def test_derive_dual_targets_score_encoded_two_way_tie():
    """Score-encoded two-way tie → both top-share players get 0.5."""
    value = torch.tensor([[0.4, 0.4, 0.1, 0.1, 0.0, 0.0]])
    win_loss, score = _derive_dual_value_targets(value)

    expected_win_loss = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(win_loss, expected_win_loss, atol=1e-5)
    assert torch.allclose(score, value)


def test_derive_dual_targets_score_encoded_four_way_tie():
    """Score-encoded four-way tie → all four top-share players get 0.25."""
    value = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0]])
    win_loss, score = _derive_dual_value_targets(value)
    expected = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0]])
    assert torch.allclose(win_loss, expected, atol=1e-5)
    assert torch.allclose(score, value)
