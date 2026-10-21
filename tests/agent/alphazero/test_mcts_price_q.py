"""Price-grandchild selection follows the search's Q rule (2026-10-21).

An open-price move (a bid, a train between corporations, a private sold to a
corporation) picks among its price grandchildren by PUCT once progressive
widening is capped. Until 2026-10-21 that level read minigo's W / (1 + N)
whatever ``mcts_mean_q`` said, so on 0-1 win shares the most-visited price --
the price head's mode, proposed first -- kept winning. Both engines now score
prices like categorical children: under mean Q the mean value, an in-flight
virtual loss a zero-value visit, an unvisited price at first-play urgency from
the move's own value. Rigged trees, built the same way in both engines.
"""

import math

import numpy as np
import pytest

from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import SelfPlayConfig
from rl18xx.agent.alphazero.mcts import BID_STEP, MCTSNode, POLICY_SIZE, VALUE_SIZE
from rl18xx.rust_adapter import RustGameAdapter

engine_rs = pytest.importorskip("engine_rs")

# Widening capped at one grandchild for the visit counts used here, so the
# slot's selection is PUCT among the grandchildren the test made.
PW = dict(pw_c=0.5, pw_alpha=0.25, min_price_children=1)


def _fresh_game() -> RustGameAdapter:
    return RustGameAdapter(engine_rs.BaseGame({i + 1: f"Player {i + 1}" for i in range(4)}))


class RiggedTrees:
    """A Python and a Rust tree on the same fresh 4-player game, roots expanded
    with uniform priors and a price head confident in one cell of the first
    open-price Bid slot. Grandchildren of that slot are added, backed up and
    given virtual losses in both trees alike."""

    def __init__(self, mean_q: bool, fpu_reduction: float = 0.0):
        cfg = SelfPlayConfig(
            network=None,
            dirichlet_noise_weight=0.0,
            backup_discount=1.0,
            c_puct_init=0.05,
            c_puct_by_round={},
            mcts_mean_q=mean_q,
            fpu_reduction=fpu_reduction,
            **PW,
        )
        self.py = MCTSNode(_fresh_game(), config=cfg)
        self.rs = engine_rs.RustMCTSPlayer(_fresh_game(), cfg.pw_c, cfg.pw_alpha, cfg.min_price_children)
        self.rs.set_search_config(cfg.c_puct_init, float(cfg.c_puct_base), {}, 1000)
        if mean_q:
            self.rs.set_q_config(True, fpu_reduction)

        self.slot = next(
            i
            for i in self.py.legal_action_indices
            if self.py.action_types_by_idx.get(i) == "Bid"
            and self.py.price_ranges_by_idx.get(i) is not None
            and self.py.price_ranges_by_idx[i][0] != self.py.price_ranges_by_idx[i][1]
        )
        self.price_range = self.py.price_ranges_by_idx[self.slot]
        self.cells = price_pmf.price_cells("Bid", *self.price_range, BID_STEP)
        self.mode = int(np.flatnonzero(self.cells.nonempty)[0])
        components = {
            "price_logits": np.zeros((price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS), dtype=np.float32),
            "slot_index": dict(price_pmf.SLOT_INDEX),
            "num_slots": price_pmf.NUM_SLOTS,
        }
        company = ActionMapper().actions[self.slot][1][0]
        components["price_logits"][price_pmf.SLOT_INDEX[("Bid", (company,))], self.mode] = 5.0

        priors = np.zeros(POLICY_SIZE, dtype=np.float32)
        priors[self.py.legal_action_indices] = 1.0 / len(self.py.legal_action_indices)
        value = np.full(VALUE_SIZE, 0.25, dtype=np.float32)
        self.py.incorporate_results(priors, value, up_to=self.py, price_components=components)
        assert self.rs.select_leaf() == 0
        self.rs.incorporate_results(0, priors, value, components)
        self.mover = self.py.active_player_index
        assert self.mover == self.rs.root_active_player_index()
        self.nodes = {}  # price -> (Python grandchild, Rust arena index)

    def price_in(self, *, mode: bool) -> int:
        """A legal price in (or outside) the head's modal cell, in a cell no
        grandchild holds yet."""
        taken = {self.cells.cell_of(p) for p in self.nodes}
        lo, hi = self.price_range
        return next(
            p
            for p in range(lo, hi + 1)
            if (self.cells.cell_of(p) == self.mode) == mode and self.cells.cell_of(p) not in taken
        )

    def add(self, price: int) -> int:
        self.nodes[price] = (
            self.py.maybe_add_child(self.slot, price=price),
            self.rs.maybe_add_child(0, self.slot, price),
        )
        return price

    def backup(self, price: int, x: float, times: int = 1):
        """Back up ``times`` values worth ``x`` to the root's mover."""
        value = np.full(VALUE_SIZE, 0.0, dtype=np.float32)
        value[:4] = (1.0 - x) / 3
        value[self.mover] = x
        node, idx = self.nodes[price]
        for _ in range(times):
            node.backup_value(value, up_to=self.py)
            self.rs.backup_value(idx, [float(v) for v in value])

    def add_virtual_loss(self, price: int):
        node, idx = self.nodes[price]
        node.add_virtual_loss(up_to=self.py)
        self.rs.add_virtual_loss(idx)

    def revert_virtual_loss(self, price: int):
        node, idx = self.nodes[price]
        node.revert_virtual_loss(up_to=self.py)
        self.rs.revert_virtual_loss(idx)

    def terms(self) -> tuple[dict, dict]:
        """``{price: (q, u)}`` per engine."""
        py = {p: (q, u) for p, _, q, u in self.py._price_child_terms(self.slot, "Bid", self.price_range)}
        rs = {int(p): qu for p, qu in self.rs.price_child_scores(0, self.slot).items()}
        return py, rs

    def select(self) -> tuple[int, int]:
        """The price each engine's capped PW descent takes."""
        py = self.py._select_or_expand_price_child(self.slot, self.price_range).sampled_price
        idx = self.rs._select_or_expand_price_child(0, self.slot)
        rs = next(p for p, (_, i) in self.nodes.items() if i == idx)
        return py, rs


def _modal_vs_better(mean_q: bool) -> tuple[RiggedTrees, int, int]:
    """The head's modal price with 10 visits worth 0.25, another with 1 visit
    worth 0.40 (minigo reads them 0.227 and 0.20)."""
    trees = RiggedTrees(mean_q)
    modal = trees.add(trees.price_in(mode=True))
    better = trees.add(trees.price_in(mode=False))
    trees.backup(modal, 0.25, times=10)
    trees.backup(better, 0.40)
    return trees, modal, better


def _assert_terms_agree(trees: RiggedTrees):
    py, rs = trees.terms()
    assert py.keys() == rs.keys()
    for price in py:
        assert py[price] == pytest.approx(rs[price], rel=1e-5, abs=1e-6), price


def test_mean_q_selects_the_better_less_visited_price():
    trees, modal, better = _modal_vs_better(mean_q=True)
    assert trees.select() == (better, better)
    py, _ = trees.terms()
    assert py[better][0] == pytest.approx(0.40)
    assert py[modal][0] == pytest.approx(0.25)
    _assert_terms_agree(trees)


def test_minigo_keeps_the_most_visited_price():
    # The default Q is unchanged: W / (1 + N), so the modal price's ten visits
    # outweigh the better price's one.
    trees, modal, better = _modal_vs_better(mean_q=False)
    assert trees.select() == (modal, modal)
    py, _ = trees.terms()
    assert py[modal][0] == pytest.approx(2.5 / 11)
    assert py[better][0] == pytest.approx(0.40 / 2)
    _assert_terms_agree(trees)


def test_virtual_loss_is_a_zero_value_visit_under_mean_q():
    trees, modal, better = _modal_vs_better(mean_q=True)
    trees.add_virtual_loss(better)
    py, _ = trees.terms()
    assert py[better][0] == pytest.approx(0.40 / 2)  # one 0.40 visit, one in flight at 0
    assert trees.py.price_child_VL[trees.slot][better] == 1
    assert trees.py.child_VL_compressed[trees.py.legal_action_indices.index(trees.slot)] == 1
    _assert_terms_agree(trees)
    # A price with only a descent in flight reads 0: not -1, nor first-play.
    fresh = trees.add(trees.price_in(mode=False))
    trees.add_virtual_loss(fresh)
    py, _ = trees.terms()
    assert py[fresh][0] == 0.0
    _assert_terms_agree(trees)
    trees.revert_virtual_loss(fresh)
    trees.revert_virtual_loss(better)
    py, _ = trees.terms()
    assert py[better][0] == pytest.approx(0.40)
    assert trees.py.price_child_VL[trees.slot] == {modal: 0, better: 0, fresh: 0}
    _assert_terms_agree(trees)


def test_an_unvisited_price_starts_at_the_moves_value():
    fpu_reduction = 0.1
    trees = RiggedTrees(mean_q=True, fpu_reduction=fpu_reduction)
    modal = trees.add(trees.price_in(mode=True))
    other = trees.add(trees.price_in(mode=False))
    trees.backup(modal, 0.25, times=10)
    trees.backup(other, 0.40)
    fresh = trees.add(trees.price_in(mode=False))
    probs = trees.py._slot_cell_probs(trees.slot, "Bid", trees.price_range)[1]
    mass = {p: probs[trees.cells.cell_of(p)] for p in (modal, other, fresh)}
    visited = (mass[modal] + mass[other]) / sum(mass.values())
    slot_value = (10 * 0.25 + 0.40) / 11
    py, _ = trees.terms()
    assert py[fresh][0] == pytest.approx(slot_value - fpu_reduction * math.sqrt(visited), rel=1e-5)
    _assert_terms_agree(trees)


def test_a_price_grandchilds_mean_value_is_its_own():
    trees, modal, better = _modal_vs_better(mean_q=True)
    node = trees.nodes[better][0]
    assert node.mean_value(trees.mover) == pytest.approx(0.40)
    trees.add_virtual_loss(better)
    assert node.mean_value(trees.mover) == pytest.approx(0.40)
