"""Unit tests for ``train.py`` loss helpers.

Covers:
- ``_compute_price_nll_loss`` (exact price-pmf NLL, visit weights, skipped and
  empty targets).
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

from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.price_pmf import NUM_CELLS
from rl18xx.agent.alphazero.train import (
    _compute_decomposed_policy_loss,
    _compute_price_nll_loss,
    _derive_dual_value_targets,
)


# ---------------------------------------------------------------------------
# _compute_price_nll_loss
# ---------------------------------------------------------------------------


def _components(slot_logits: dict, num_slots: int = 60):
    """``price_components`` for a one-example batch: zero logits except the
    given ``{slot: NUM_CELLS logits}`` rows."""
    logits = torch.zeros(1, num_slots, NUM_CELLS)
    for slot, row in slot_logits.items():
        logits[0, slot] = torch.as_tensor(row, dtype=torch.float32)
    return {"price_logits": logits}


def test_price_nll_is_the_exact_log_probability_of_the_price():
    """NLL = -(log_softmax over non-empty cells)[cell(price)] + log |cell|,
    for an atom (the min bid) and for a price inside a bin."""
    row = torch.linspace(-1.0, 1.0, NUM_CELLS)
    bid_slot = price_pmf.SLOT_INDEX[("Bid", ("BO",))]
    train_slot = price_pmf.SLOT_INDEX[("BuyTrain", ("PRR", "4"))]
    for slot, action_type, price, lo, hi in (
        (bid_slot, "Bid", 225, 225, 600),
        (bid_slot, "Bid", 437, 225, 600),  # off the $5 ladder: residual cell
        (train_slot, "BuyTrain", 237, 1, 441),
    ):
        loss, diags = _compute_price_nll_loss(_components({slot: row}), [[(slot, price, 1.0, lo, hi)]])
        cells = price_pmf.PriceCells(action_type, lo, hi)
        expected = -cells.log_prob(row.numpy(), price)
        assert math.isclose(loss.item(), expected, rel_tol=1e-5, abs_tol=1e-5), (action_type, price)
        assert diags["price_count"] == 1


def test_price_nll_weights_visit_fractions():
    slot = price_pmf.SLOT_INDEX[("BuyTrain", ("NYC", "3"))]
    row = torch.randn(NUM_CELLS)
    targets = [[(slot, 1, 0.75, 1, 300), (slot, 300, 0.25, 1, 300)]]
    loss, _ = _compute_price_nll_loss(_components({slot: row}), targets)
    cells = price_pmf.PriceCells("BuyTrain", 1, 300)
    expected = -(0.75 * cells.log_prob(row.numpy(), 1) + 0.25 * cells.log_prob(row.numpy(), 300))
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)


def test_price_nll_skips_fixed_and_out_of_range_targets():
    """Fixed-price (lo == hi) and illegal prices carry no distribution: no loss."""
    for targets in ([[(0, 50, 1.0, 50, 50)]], [[(0, 700, 1.0, 225, 600)]], [[(99, 300, 1.0, 225, 600)]]):
        loss, diags = _compute_price_nll_loss(_components({}), targets)
        assert loss.item() == 0.0
        assert diags["price_count"] == 0


def test_price_nll_empty_targets_returns_zero_loss():
    """No targets in any batch row → loss is a literal zero tensor."""
    for price_targets in ([], [[], []], None):
        loss, diags = _compute_price_nll_loss(_components({}), price_targets)
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
