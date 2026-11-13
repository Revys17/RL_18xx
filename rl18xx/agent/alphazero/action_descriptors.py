"""A title's action descriptors as numpy arrays (``docs/rl_descriptors.md``).

Every policy index of a title's RL layout has a descriptor: one row of small ints saying what the
index means in title-independent terms (its action family, the entities it refers to, hex / city /
tile / rotation, a number, a categorical choice, how its price is expressed, flags). A generic
policy head scores a state's legal indices from their rows (``descriptors.gather(legal)``) instead
of keeping one output per index. The tables are built in Rust (``engine_rs.rl_layout_info``) and
cached here per title::

    d = action_descriptors("1830")
    rows = d.gather(game.factored_legal_indices())      # int32 [n_legal, len(d.fields)]
    fam = rows[:, d.col("family")]                        # ids into d.families
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

NONE = -1


@dataclass(frozen=True)
class ActionDescriptors:
    title: str
    #: Layout version ("1830-v1"); a change of any index's meaning needs a new one.
    version: str
    #: 16 hex digits over blocks, vocabularies and every descriptor (pinned per title in Rust).
    fingerprint: str
    total: int
    #: int32 ``[total, len(fields)]``, read-only.
    table: np.ndarray
    fields: Tuple[str, ...]
    #: Global vocabularies (append-only; ids are positions).
    families: Tuple[str, ...]
    map_families: Tuple[int, ...]
    ent_kinds: Tuple[str, ...]
    price_kinds: Tuple[str, ...]
    price_cell_types: Tuple[object, ...]
    choices: Tuple[str, ...]
    flags: Dict[str, int]
    #: The title's vocabularies the entity / hex / tile fields index: corporation, company, train,
    #: hex, tile.
    vocab: Dict[str, Tuple[str, ...]]
    #: The layout's blocks: name, family, family_id, offset, size, price_kind, flags, axes.
    blocks: Tuple[dict, ...]
    #: How legal indices are computed: "factored" (1830: ``factored_legal_indices``) or "generator"
    #: (1867: ``rl_legal_indices`` over the title's legal-move generator).
    kind: str = "factored"
    #: How a legal priced index's ``(lo, hi, step)`` (``BaseGame.rl_price_range``) reads: "step" --
    #: legal prices are ``lo + step * k`` and the price cells are taken in step units (generator
    #: layouts, 1867); "price" -- every integer in ``[lo, hi]`` is legal, cells in price space, ``step``
    #: only the "Bid" ladder (1830, as the legacy head). See :func:`range_price_cells`.
    price_units: str = "price"

    @property
    def price_grid(self) -> bool:
        """Whether legal prices lie on the ``lo + step * k`` grid (``price_units == "step"``)."""
        return self.price_units == "step"

    def col(self, field: str) -> int:
        return self.fields.index(field)

    def column(self, field: str) -> np.ndarray:
        return self.table[:, self.col(field)]

    def gather(self, indices: Sequence[int]) -> np.ndarray:
        """The rows of ``indices`` (e.g. a state's legal indices), int32 ``[len, len(fields)]``."""
        return self.table[np.asarray(indices, dtype=np.int64)]

    def describe(self, index: int) -> dict:
        """One index's descriptor with every id resolved to its name (debugging, tests)."""
        row = dict(zip(self.fields, (int(v) for v in self.table[index])))
        out = {"index": int(index), "family": self.families[row["family"]]}
        kind_vocab = {"corporation": "corporation", "company": "company", "train": "train"}
        for slot in ("ent0", "ent1"):
            if row[f"{slot}_kind"] != NONE:
                kind = self.ent_kinds[row[f"{slot}_kind"]]
                out[slot] = (kind, self.vocab[kind_vocab[kind]][row[slot]] if kind in kind_vocab else row[slot])
        if row["via"] != NONE:
            out["via"] = self.vocab["company"][row["via"]]
        if row["hex"] != NONE:
            out["hex"] = self.vocab["hex"][row["hex"]]
        if row["city"] != NONE:
            out["city"] = row["city"]
        if row["tile"] != NONE:
            out["tile"] = self.vocab["tile"][row["tile"]]
        if row["rot"] != NONE:
            out["rot"] = row["rot"]
        if row["param"] != NONE:
            out["param"] = row["param"]
        if row["choice"] != NONE:
            out["choice"] = self.choices[row["choice"]]
        if row["price_kind"] != NONE:
            out["price_kind"] = self.price_kinds[row["price_kind"]]
        out["flags"] = [name for name, bit in self.flags.items() if row["flags"] & bit]
        return out

    def block_of(self, index: int) -> dict:
        for b in self.blocks:
            if b["offset"] <= index < b["offset"] + b["size"]:
                return b
        raise IndexError(f"{self.title}: index {index} outside 0-{self.total - 1}")


@lru_cache(maxsize=None)
def action_descriptors(title: str) -> ActionDescriptors:
    """The descriptor table of ``title``'s RL layout. RuntimeError "not bridge-ready" for a title
    without one (the 1822 family until its layout lands). 1867's layout (``1867-v1``, kind
    "generator") indexes its legal-move generator's moves: ``game.rl_legal_indices()``."""
    import engine_rs

    info = engine_rs.rl_layout_info(title)
    table = np.ascontiguousarray(info["descriptors"], dtype=np.int32)
    table.setflags(write=False)
    vocab = {k: tuple(v) for k, v in info["vocab"].items()}
    return ActionDescriptors(
        title=info["title"],
        version=info["version"],
        fingerprint=info["fingerprint"],
        total=int(info["total"]),
        table=table,
        fields=tuple(info["fields"]),
        families=tuple(info["families"]),
        map_families=tuple(info["map_families"]),
        ent_kinds=tuple(info["ent_kinds"]),
        price_kinds=tuple(info["price_kinds"]),
        price_cell_types=tuple(info["price_cell_types"]),
        choices=tuple(info["choices"]),
        flags=dict(info["flags"]),
        vocab=vocab,
        blocks=tuple(dict(b) for b in info["blocks"]),
        kind=str(info["kind"]),
        price_units=str(info["price_units"]),
    )


class RangePriceCells:
    """The price head's cells of one legal range, addressed in prices (``docs/rl_descriptors.md`` §6).

    ``grid`` (generator layouts): legal prices are ``lo + step * k``; the cells are
    ``price_pmf.price_cells(cell_type, 0, K, 1)`` over ``k`` (``K = (hi - lo) // step``), so a cell
    holds legal prices only. Otherwise (1830) every integer in ``[lo, hi]`` is legal and the cells are
    ``price_pmf.price_cells(cell_type, lo, hi, step)`` -- the legacy head's, with ``step`` the "Bid"
    ladder. ``cell_of`` / ``member`` take and give prices either way; ``counts`` / ``nonempty`` /
    ``n_atoms`` are the underlying cells'."""

    def __init__(self, cell_type: str, lo: int, hi: int, step: int, grid: bool):
        from rl18xx.agent.alphazero.price_pmf import price_cells

        self.lo, self.hi, self.step, self.grid = int(lo), int(hi), max(int(step), 1), bool(grid)
        if self.grid:
            self.cells = price_cells(cell_type, 0, (self.hi - self.lo) // self.step, 1)
        else:
            self.cells = price_cells(cell_type, self.lo, self.hi, self.step)
        self.n_atoms = self.cells.n_atoms
        self.counts = self.cells.counts

    @property
    def nonempty(self) -> np.ndarray:
        return self.cells.nonempty

    def cell_of(self, price: int) -> int:
        """The cell of a legal price (ValueError off the range or, on a grid, off the grid)."""
        price = int(price)
        if not self.lo <= price <= self.hi:
            raise ValueError(f"price {price} outside [{self.lo}, {self.hi}]")
        if not self.grid:
            return self.cells.cell_of(price)
        k, r = divmod(price - self.lo, self.step)
        if r:
            raise ValueError(f"price {price} is off the ${self.step} grid from {self.lo}")
        return self.cells.cell_of(k)

    def member(self, cell: int, u: int) -> int:
        """The ``u``-th (ascending) legal price in ``cell``."""
        x = self.cells.member(cell, u)
        return self.lo + self.step * x if self.grid else x

    def sample_price(self, cell: int, rng: np.random.Generator) -> int:
        return self.member(cell, int(rng.integers(max(int(self.counts[cell]), 1))))


def range_price_cells(cell_type: str, lo: int, hi: int, step: int, grid: bool) -> RangePriceCells:
    """:class:`RangePriceCells` of a legal range (``grid``: the layout's :attr:`ActionDescriptors.price_grid`)."""
    return RangePriceCells(cell_type, lo, hi, step, grid)


def descriptor_vocab() -> dict:
    """The global (title-independent) vocabularies alone."""
    import engine_rs

    return dict(engine_rs.rl_descriptor_vocab())


__all__: List[str] = [
    "ActionDescriptors",
    "RangePriceCells",
    "action_descriptors",
    "descriptor_vocab",
    "range_price_cells",
    "NONE",
]
