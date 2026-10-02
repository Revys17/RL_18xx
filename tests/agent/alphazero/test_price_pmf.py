"""The exact price pmf: cells partition every legal range, the distribution
normalizes over every legal integer price, and the Rust mirror agrees."""

import numpy as np
import pytest

from rl18xx.agent.alphazero.price_pmf import NUM_CELLS, PriceCells, proposal_order

try:
    from engine_rs import price_pmf_cell_of, price_pmf_counts, price_pmf_member
except ImportError:  # pragma: no cover - engine built without the price pmf
    price_pmf_counts = None

RANGES = [
    ("Bid", 225, 600),  # opening B&O bid
    ("Bid", 25, 40),  # ladder runs past the cash
    ("Bid", 47, 512),  # min bid off the $5 grid
    ("Bid", 20, 21),
    ("BuyTrain", 1, 441),
    ("BuyTrain", 1, 2),  # max - 1 coincides with the min
    ("BuyTrain", 1, 3),
    ("BuyTrain", 1, 30),  # fewer interior prices than bins
    ("BuyTrain", 123, 1337),  # emergency-buy minimum
    ("BuyCompany", 80, 320),
    ("BuyCompany", 10, 13),
]


def _random_ranges(n=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = ("Bid", "BuyTrain", "BuyCompany")[rng.integers(3)]
        lo = int(rng.integers(1, 400))
        out.append((t, lo, lo + int(rng.integers(1, 1200))))
    return out


@pytest.mark.parametrize("action_type,lo,hi", RANGES + _random_ranges(40))
def test_cells_partition_the_legal_range(action_type, lo, hi):
    cells = PriceCells(action_type, lo, hi)
    assert cells.counts.sum() == hi - lo + 1
    seen = set()
    for c in range(NUM_CELLS):
        for u in range(cells.counts[c]):
            p = cells.member(c, u)
            assert cells.cell_of(p) == c
            seen.add(p)
    assert seen == set(range(lo, hi + 1))


@pytest.mark.parametrize("action_type,lo,hi", RANGES)
def test_every_legal_price_has_probability_and_they_sum_to_one(action_type, lo, hi):
    logits = np.random.default_rng(1).normal(size=NUM_CELLS)
    cells = PriceCells(action_type, lo, hi)
    probs = np.exp([cells.log_prob(logits, p) for p in range(lo, hi + 1)])
    assert np.all(probs > 0)
    assert probs.sum() == pytest.approx(1.0)


def test_atoms_hold_the_prices_humans_pile_onto():
    bid = PriceCells("Bid", 225, 600)
    assert [bid.cell_of(p) for p in (225, 230, 320)] == [0, 1, 19]
    assert bid.counts[:20].tolist() == [1] * 20
    train = PriceCells("BuyTrain", 1, 441)
    assert (train.cell_of(1), train.cell_of(441), train.cell_of(440)) == (0, 1, 2)
    company = PriceCells("BuyCompany", 80, 320)
    assert (company.cell_of(80), company.cell_of(320)) == (0, 1)


def test_non_finite_logits_fall_back_to_uniform_over_nonempty_cells():
    cells = PriceCells("BuyTrain", 1, 441)
    probs = cells.cell_probs(np.full(NUM_CELLS, np.nan))
    assert probs[cells.nonempty] == pytest.approx(1.0 / cells.nonempty.sum())
    assert probs[~cells.nonempty].sum() == 0


def test_proposal_order_starts_at_the_mode_and_covers_every_nonempty_cell():
    cells = PriceCells("BuyTrain", 1, 441)
    logits = np.zeros(NUM_CELLS)
    logits[1] = 5.0
    probs = cells.cell_probs(logits)
    order = proposal_order(probs, 0.05, np.random.default_rng(0))
    assert order[0] == 1
    assert sorted(order) == np.flatnonzero(cells.nonempty).tolist()


@pytest.mark.skipif(price_pmf_counts is None, reason="engine_rs built without the price pmf")
@pytest.mark.parametrize("action_type,lo,hi", RANGES + _random_ranges(300))
def test_rust_cells_match_python(action_type, lo, hi):
    cells = PriceCells(action_type, lo, hi)
    assert price_pmf_counts(action_type, lo, hi, 5) == cells.counts.tolist()
    for p in range(lo, hi + 1):
        assert price_pmf_cell_of(action_type, p, lo, hi, 5) == cells.cell_of(p)
    for c in np.flatnonzero(cells.nonempty):
        for u in {0, int(cells.counts[c]) // 2, int(cells.counts[c]) - 1}:
            assert price_pmf_member(action_type, int(c), u, lo, hi, 5) == cells.member(int(c), u)
