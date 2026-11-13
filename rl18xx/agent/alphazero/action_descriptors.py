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
    without one (1867 and the 1822 family until their layouts land)."""
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
    )


def descriptor_vocab() -> dict:
    """The global (title-independent) vocabularies alone."""
    import engine_rs

    return dict(engine_rs.rl_descriptor_vocab())


__all__: List[str] = ["ActionDescriptors", "action_descriptors", "descriptor_vocab", "NONE"]
