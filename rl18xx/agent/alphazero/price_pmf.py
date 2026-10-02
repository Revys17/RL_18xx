"""Exact price distribution for price-bearing actions.

The price head assigns a probability to **every** legal integer price of a
Bid / cross-corp BuyTrain / BuyCompany, so any human or searched price is a
valid training target verbatim and search can play any legal price. The legal
range ``[lo, hi]`` is partitioned into *cells*:

  * **atoms** — single prices humans pile onto (the min-bid + step ladder for
    bids; $1 / max / max-1 for cross-corp trains; min / max for companies);
  * **bins** — the remaining prices, split into range-relative bins whose mass
    is spread uniformly over the integers they hold.

The head emits one logit per cell (``NUM_CELLS`` per slot), so

    P(price) = softmax(logits over non-empty cells)[cell(price)] / |cell(price)|

The cells are the density's resolution, not an action list. See
docs/arbitrary_price_actions.md. Mirrored exactly (integer arithmetic only) by
``engine-rs/src/price_pmf.rs``; ``tests/agent/alphazero/test_price_pmf.py``
checks parity.
"""

from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from typing import Optional

import numpy as np

NUM_CELLS = 24
BID_LADDER = 20  # min_bid + step * j, j < BID_LADDER: covers every human bid in the corpus
BID_STEP_1830 = 5

# action_type -> (number of atom cells, number of bin cells)
LAYOUT = {
    "Bid": (BID_LADDER, 1),
    "BuyTrain": (3, 20),
    "BuyCompany": (2, 8),
}

# ContinuousPriceHead-era slot layout, kept: (action_type, entity_key) per slot.
PRICE_HEAD_COMPANIES = ("SV", "CS", "DH", "MH", "CA", "BO")
PRICE_HEAD_CORPORATIONS = ("PRR", "NYC", "CPR", "B&O", "C&O", "ERIE", "NYNH", "B&M")
PRICE_HEAD_TRAIN_TYPES = ("2", "3", "4", "5", "6", "D")
SLOTS: list[tuple[str, tuple]] = (
    [("Bid", (c,)) for c in PRICE_HEAD_COMPANIES]
    + [("BuyTrain", (corp, t)) for corp in PRICE_HEAD_CORPORATIONS for t in PRICE_HEAD_TRAIN_TYPES]
    + [("BuyCompany", (c,)) for c in PRICE_HEAD_COMPANIES]
)
NUM_SLOTS = len(SLOTS)
SLOT_INDEX = {key: i for i, key in enumerate(SLOTS)}
SLOT_TYPES = [t for t, _ in SLOTS]


def structural_cell_mask(action_type: str) -> np.ndarray:
    """Cells an action type can ever use (``n_atoms + n_bins`` of ``NUM_CELLS``)."""
    n_atoms, n_bins = LAYOUT[action_type]
    mask = np.zeros(NUM_CELLS, dtype=bool)
    mask[: n_atoms + n_bins] = True
    return mask


def _atom_spec(action_type: str, lo: int, hi: int, bid_step: int) -> list[int]:
    if action_type == "Bid":
        return [lo + bid_step * j for j in range(BID_LADDER)]
    if action_type == "BuyTrain":
        return [lo, hi, hi - 1]
    if action_type == "BuyCompany":
        return [lo, hi]
    raise ValueError(f"not a price-bearing action type: {action_type!r}")


class PriceCells:
    """The cell partition of one legal range ``[lo, hi]`` (``lo <= hi``)."""

    def __init__(self, action_type: str, lo: int, hi: int, bid_step: int = BID_STEP_1830):
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise ValueError(f"empty price range [{lo}, {hi}]")
        self.action_type = action_type
        self.lo, self.hi = lo, hi
        self.n_atoms, self.n_bins = LAYOUT[action_type]
        # Atom j is live iff in range and not a repeat of an earlier atom.
        self.atom_cell: dict[int, int] = {}
        for j, price in enumerate(_atom_spec(action_type, lo, hi, bid_step)):
            if lo <= price <= hi and price not in self.atom_cell:
                self.atom_cell[price] = j
        self._atoms_sorted = sorted(self.atom_cell)
        self.counts = np.zeros(NUM_CELLS, dtype=np.int64)
        for j in self.atom_cell.values():
            self.counts[j] = 1
        for b in range(self.n_bins):
            start, end = self.bin_bounds(b)
            n_atoms_inside = bisect_right(self._atoms_sorted, end) - bisect_right(self._atoms_sorted, start - 1)
            self.counts[self.n_atoms + b] = max(0, end - start + 1 - n_atoms_inside)

    @property
    def nonempty(self) -> np.ndarray:
        return self.counts > 0

    def bin_bounds(self, b: int) -> tuple[int, int]:
        """Integer prices ``[start, end]`` whose bin is ``b`` (atoms included)."""
        width, n = self.hi - self.lo + 1, self.n_bins
        return self.lo + (b * width + n - 1) // n, self.lo + ((b + 1) * width + n - 1) // n - 1

    def cell_of(self, price: int) -> int:
        price = int(price)
        if not self.lo <= price <= self.hi:
            raise ValueError(f"price {price} outside [{self.lo}, {self.hi}]")
        atom = self.atom_cell.get(price)
        if atom is not None:
            return atom
        return self.n_atoms + ((price - self.lo) * self.n_bins) // (self.hi - self.lo + 1)

    def member(self, cell: int, u: int) -> int:
        """The ``u``-th (0-based, ascending) price in ``cell``; ``0 <= u < counts[cell]``."""
        if not 0 <= u < self.counts[cell]:
            raise ValueError(f"member {u} of cell {cell} with {self.counts[cell]} prices")
        if cell < self.n_atoms:
            return next(p for p, j in self.atom_cell.items() if j == cell)
        start, end = self.bin_bounds(cell - self.n_atoms)
        price = start + u
        for atom in self._atoms_sorted:  # skip the atoms that sit inside the bin
            if start <= atom <= price:
                price += 1
        return price

    def sample_price(self, cell: int, rng: np.random.Generator) -> int:
        return self.member(cell, int(rng.integers(0, int(self.counts[cell]))))

    def cell_probs(self, logits: Optional[np.ndarray]) -> np.ndarray:
        """Softmax of the head's logits over this range's non-empty cells.

        Missing or non-finite logits fall back to uniform over the non-empty
        cells, so search still has usable proposals."""
        mask = self.nonempty
        if logits is None:
            logits = np.zeros(NUM_CELLS)
        logits = np.asarray(logits, dtype=np.float64)
        if not np.all(np.isfinite(logits[mask])):
            logits = np.zeros(NUM_CELLS)
        z = np.where(mask, logits, -np.inf)
        z = z - z[mask].max()
        p = np.where(mask, np.exp(z), 0.0)
        return p / p.sum()

    def log_prob(self, logits: Optional[np.ndarray], price: int) -> float:
        cell = self.cell_of(price)
        return float(np.log(self.cell_probs(logits)[cell]) - np.log(self.counts[cell]))


@lru_cache(maxsize=1 << 16)
def price_cells(action_type: str, lo: int, hi: int, bid_step: int = BID_STEP_1830) -> PriceCells:
    """Cached :class:`PriceCells` (treat the result as read-only)."""
    return PriceCells(action_type, lo, hi, bid_step)


def proposal_order(probs: np.ndarray, explore_eps: float, rng: np.random.Generator) -> list[int]:
    """Order in which progressive widening materializes a slot's cells.

    The head's most likely cell first (at one or two expansions the mode beats
    a random draw), then the rest by Gumbel-top-k — sampling without
    replacement — from ``(1 - eps) * probs + eps * uniform``, so every
    non-empty cell is eventually reached even where the head is confidently
    wrong. Empty cells (``probs == 0``) never appear."""
    live = np.flatnonzero(probs > 0)
    first = int(live[np.argmax(probs[live])])
    mixed = (1.0 - explore_eps) * probs[live] + explore_eps / len(live)
    keys = np.log(mixed) + rng.gumbel(size=len(live))
    rest = [int(live[i]) for i in np.argsort(-keys, kind="stable") if live[i] != first]
    return [first] + rest
