"""Title-generic policy / value network (RL bridge milestone B2; ``docs/rl_model.md``).

One architecture for every registered title. It reads B1's typed tokens (``generic_tokens.encode_batch``,
``docs/rl_encoder.md``) and, per position, the descriptor rows of the position's **legal** actions (A1,
``docs/rl_descriptors.md``), and returns

* policy logits over the legal actions only: a flat list of legal rows over the whole batch with each
  row's position (``legal_pos``), normalized by a segment softmax per position, so positions with
  different legal counts pack without padding to a layout. A row's score comes from its descriptor --
  family, entity pointers resolved to the encoder's entity tokens, hex / tile / rotation, param,
  choice, via, flags -- read against the fused state. Map families keep the factorization
  P(hex) * P(tile | hex) * P(rot | hex, tile) (tile lays) / P(hex) * P(city | hex) (tokens) over the
  legal values, under a state-dependent score for the move's group;
* a price head for price-bearing legal rows: logits over ``price_pmf``'s ``NUM_CELLS`` cells,
  conditioned on the row's descriptor, the state and the row's legal range, the cell structure taken
  from the descriptor's ``price_kind`` (``price_cell_types``) and the range from the state -- the
  1830 price head's semantics (``P(price) = softmax(nonempty cells)[cell] / |cell|``), never a menu;
* per-seat value heads over ``MAX_SEATS`` (6) seats, padding masked: win logits (softmax over the
  present seats) and a score head (placement points). ``value_only=True`` skips the policy and price
  heads (a critic is this architecture used value-only).

Inputs are built on the CPU by :func:`build_inputs` (numpy; :func:`to_device` moves them); losses for
supervised pretraining and policy gradient are :func:`policy_loss`, :func:`price_loss`,
:func:`value_losses`; checkpoints (:func:`save_checkpoint` / :func:`load_checkpoint`) carry the titles,
their layout fingerprints, ``ENCODER_VERSION`` and the config, and refuse a mismatched encoder.

The frozen 1830 model (``model_transformer.py``) is untouched: it stays the production 1830 network.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.action_descriptors import NONE, action_descriptors, descriptor_vocab, range_price_cells
from rl18xx.agent.alphazero.generic_tokens import encode_batch, raw_game, schema, spec_info

LOGGER = logging.getLogger(__name__)

ARCHITECTURE = "GenericTokenModel"
#: Version of this module's checkpoint payload (``generic_meta``), not of the encoder.
CHECKPOINT_FORMAT = 1
NUM_CELLS = price_pmf.NUM_CELLS
#: Large negative logit for masked entries: finite under bf16 / fp16, ``exp`` underflows to 0 in fp32.
MASKED = -1e4

# Descriptor columns (``docs/rl_descriptors.md`` §2); checked against the engine's field list.
DESCRIPTOR_FIELDS = (
    "family", "ent0_kind", "ent0", "ent1_kind", "ent1", "via", "hex", "city", "tile", "rot", "param", "choice",
    "price_kind", "flags",
)  # fmt: skip
(
    C_FAMILY, C_E0K, C_E0, C_E1K, C_E1, C_VIA, C_HEX, C_CITY, C_TILE, C_ROT, C_PARAM, C_CHOICE, C_PRICE_KIND, C_FLAGS,
) = range(len(DESCRIPTOR_FIELDS))  # fmt: skip
#: Columns that say what a map move's group is (everything but where on the map).
GROUP_COLUMNS = (C_FAMILY, C_E0K, C_E0, C_E1K, C_E1, C_VIA, C_PARAM, C_CHOICE, C_PRICE_KIND, C_FLAGS)
NUM_FLAG_BITS = 8

# Entity-kind ids of the descriptor (``ent_kinds``): player 0, corporation 1, company 2, train 3.
EK_PLAYER, EK_CORPORATION, EK_COMPANY, EK_TRAIN = range(4)
_ENT_KIND_NAMES = ("player", "corporation", "company", "train")

#: How a family's ``param`` is scaled to a feature: by the title's highest market price (Par's $) or a
#: constant (percents / 100, a ``Choose`` option's position / 10). Other families have no param.
_PARAM_SCALE_BY_FAMILY = {
    "Par": "max_market_price",
    "BuyShares": 100.0,
    "SellShares": 100.0,
    "RedeemShares": 100.0,
    "IssueShares": 100.0,
    "Choose": 10.0,
}


# ---------------------------------------------------------------------------------------------------
# Vocabularies and static per-title tables
# ---------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def vocab() -> dict:
    """The global (append-only) vocabularies the model embeds, from the encoder schema and the
    descriptor vocabulary. A checkpoint records them; :func:`load_checkpoint` refuses ids that moved."""
    s = schema()
    d = descriptor_vocab()
    if tuple(d["fields"]) != DESCRIPTOR_FIELDS:
        raise RuntimeError(f"descriptor fields changed: {d['fields']} (model_generic expects {DESCRIPTOR_FIELDS})")
    return {
        "titles": tuple(s["titles"]),
        "round_kinds": tuple(s["round_kinds"]),
        "step_kinds": tuple(s["step_kinds"]),
        "labels": tuple(s["labels"]),
        "entity_types": tuple(s["entity_types"]),
        "relation_kinds": tuple(s["relation_kinds"]),
        "families": tuple(d["families"]),
        "map_families": tuple(int(x) for x in d["map_families"]),
        "ent_kinds": tuple(d["ent_kinds"]),
        "choices": tuple(d["choices"]),
        "price_kinds": tuple(d["price_kinds"]),
        "price_cell_types": tuple(d["price_cell_types"]),
        "fields": tuple(d["fields"]),
    }


def _family_id(name: str) -> int:
    return vocab()["families"].index(name)


@lru_cache(maxsize=None)
def title_tables() -> dict:
    """Static per-title arrays (numpy), indexed by the encoder's title id (``meta[:, 0]``; row 0 =
    unknown, all padding): hex geometry, the static entity types, descriptor entity bases, the tile
    table and its exits per rotation, normalizers. Built from ``rl_spec_info`` for every title."""
    s = schema()
    titles = s["titles"]
    infos = [None] + [spec_info(t) for t in titles]
    n = len(infos)
    h_max = max(i["num_hexes"] for i in infos[1:])
    e_max = max(i["num_entities"] for i in infos[1:])
    t_max = max(i["num_tiles"] for i in infos[1:])
    tf_names = {name: k for k, name in enumerate(s["tile_features"])}
    hex_dist = np.full((n, h_max, h_max), 255, dtype=np.uint8)
    hex_dir = np.full((n, h_max, h_max), 6, dtype=np.uint8)
    hex_nbr = np.full((n, h_max, 6), -1, dtype=np.int64)
    ent_type = np.zeros((n, e_max), dtype=np.int64)
    desc_base = np.zeros((n, 4), dtype=np.int64)
    tile_feat = np.zeros((n, t_max, s["f_tile"]), dtype=np.float32)
    tile_exits = np.zeros((n, t_max, 6, 6), dtype=np.float32)  # [title, tile, rot, edge]
    num_tiles = np.zeros(n, dtype=np.int64)
    num_hexes = np.zeros(n, dtype=np.int64)
    num_entities = np.zeros(n, dtype=np.int64)
    bank = np.ones(n, dtype=np.float32)
    max_market = np.ones(n, dtype=np.float32)
    for t, info in enumerate(infos):
        if info is None:
            continue
        nh, ne, nt = info["num_hexes"], info["num_entities"], info["num_tiles"]
        num_hexes[t], num_entities[t], num_tiles[t] = nh, ne, nt
        axial = np.asarray(info["hex_axial"], dtype=np.int64)
        dq = axial[:, None, 0] - axial[None, :, 0]
        dr = axial[:, None, 1] - axial[None, :, 1]
        hex_dist[t, :nh, :nh] = np.minimum((np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2, 254)
        nbr = np.asarray(info["hex_neighbors"], dtype=np.int64)
        hex_nbr[t, :nh] = nbr
        for h in range(nh):
            for k in range(6):
                if nbr[h, k] >= 0:
                    hex_dir[t, h, nbr[h, k]] = k
        ent_type[t, :ne] = np.asarray(info["entity_types"], dtype=np.int64)
        base = info["descriptor_entity_base"]
        desc_base[t] = [base[k] for k in _ENT_KIND_NAMES]
        tf = np.asarray(info["tile_feat"], dtype=np.float32)
        tile_feat[t, :nt] = tf
        tile_exits[t, :nt] = _tile_exits(tf, tf_names)
        bank[t] = float(info["normalizers"]["bank_cash"]) or 1.0
        max_market[t] = float(info["normalizers"]["max_market_price"]) or 1.0
    return {
        "hex_dist": hex_dist,
        "hex_dir": hex_dir,
        "hex_nbr": hex_nbr,
        "ent_type": ent_type,
        "desc_base": desc_base,
        "tile_feat": tile_feat,
        "tile_exits": tile_exits,
        "num_tiles": num_tiles,
        "num_hexes": num_hexes,
        "num_entities": num_entities,
        "bank": bank,
        "max_market": max_market,
        "h_max": h_max,
        "e_max": e_max,
        "t_max": t_max,
    }


def _tile_exits(tile_feat: np.ndarray, names: dict) -> np.ndarray:
    """(T, rot, edge) 0/1: the edges a catalog tile's track reaches when laid at each rotation.
    Rotation r moves edge e to (e + r) mod 6 (checked against laid tiles in the tests)."""
    exits0 = np.zeros((len(tile_feat), 6), dtype=np.float32)
    for i in range(6):
        for j in range(i + 1, 6):
            for prefix in ("track", "via"):
                hit = tile_feat[:, names[f"{prefix}_{i}_{j}"]] > 0
                exits0[hit, i] = exits0[hit, j] = 1.0
        for kind in ("city", "town", "offboard"):
            exits0[tile_feat[:, names[f"edge{i}_{kind}"]] > 0, i] = 1.0
    return np.stack([np.roll(exits0, r, axis=1) for r in range(6)], axis=1)


@lru_cache(maxsize=None)
def _hex_feature_columns() -> dict:
    names = {name: k for k, name in enumerate(schema()["hex_features"])}
    edge_track = np.zeros((len(names), 6), dtype=np.float32)  # hex feature -> "edge k has track"
    for i in range(6):
        for j in range(i + 1, 6):
            for prefix in ("track", "via"):
                edge_track[names[f"{prefix}_{i}_{j}"], [i, j]] = 1.0
        for kind in ("city", "town", "offboard"):
            edge_track[names[f"edge{i}_{kind}"], i] = 1.0
    return {
        "edge_track": edge_track,
        "edge_stop": [[names[f"edge{k}_{kind}"] for kind in ("city", "town", "offboard")] for k in range(6)],
        "wall": [names[f"wall_{k}"] for k in range(6)],
        "border": [names[f"border_cost_{k}"] for k in range(6)],
    }


# ---------------------------------------------------------------------------------------------------
# Legal actions and batch inputs (CPU, numpy)
# ---------------------------------------------------------------------------------------------------


@dataclass
class LegalActions:
    """One position's legal actions: descriptor rows, and for price-bearing rows the legal range."""

    #: int32 (n, 14): descriptor rows (``docs/rl_descriptors.md`` §2), one per legal action.
    rows: np.ndarray
    #: int64 (n, 3): (lo, hi, step) of each row with a ``price_kind``; -1 elsewhere.
    price_range: np.ndarray
    #: The rows' layout indices (None for rows built without a layout, e.g. a test stub).
    indices: Optional[np.ndarray] = None
    #: Whether legal prices lie on the ``lo + step * k`` grid and the price cells are taken in step
    #: units (a generator layout, 1867: ``ActionDescriptors.price_grid``); False = every integer of the
    #: range, cells in price space with ``step`` the "Bid" ladder (1830, the legacy head's cells).
    price_grid: bool = False
    #: A generator layout's moves as 18xx.games action dicts, aligned with ``rows`` (``rl_legal_moves_json``'s
    #: ``wire``; None for 1830): ``generic_play.TitlePlay.apply`` processes the chosen one directly instead of
    #: decoding its index again (``rl_apply_index`` re-enumerates the moves: half of 1867's per-decision CPU).
    wires: Optional[list] = None

    def __len__(self) -> int:
        return len(self.rows)

    def price_cells(self, k: int):
        """The price head's cells of legal row ``k``'s range, on this list's grid (:func:`price_cells_for`):
        ``cell_of`` / ``sample_price`` / ``member`` give legal prices only (1867: on the move's step grid).
        Every caller that samples or scores a price of a legal row goes through here."""
        lo, hi, step = (int(x) for x in self.price_range[k])
        return price_cells_for(int(self.rows[k, C_PRICE_KIND]), lo, hi, step, grid=bool(self.price_grid))


def legal_actions(game, indices: Optional[Sequence[int]] = None, policy: Optional[str] = None) -> LegalActions:
    """The legal actions of a position of a title with an RL layout: its legal indices (or ``indices``
    when the caller already has them), their descriptor rows, and each price-bearing row's
    ``(lo, hi, step)`` (``BaseGame.rl_price_range``, ``docs/rl_descriptors.md`` §6). This is the one
    place a legal range's step is decided (the supervised data and the policy players call it).

    * 1830 (layout kind "factored"): ``factored_legal_indices``; ranges as ``price_range_for_index``
      with ``step`` the bid ladder ($5) for a bid, 1 otherwise; ``price_grid`` False.
    * A generator layout (1867): ``rl_legal_moves_json(policy)`` (None = the title's cross-player
      policy); each move's own range and step; ``price_grid`` True."""
    g = raw_game(game)
    d = action_descriptors(g.title)
    if d.kind == "generator":
        moves = json.loads(g.rl_legal_moves_json(policy))
        ranges = {m["index"]: m["range"] for m in moves}
        wire_of = {m["index"]: m["wire"] for m in moves}
        idx = np.asarray(sorted(ranges) if indices is None else indices, dtype=np.int64)
        rows = d.gather(idx)
        price_range = np.full((len(idx), 3), -1, dtype=np.int64)
        for i in range(len(idx)):
            if int(idx[i]) not in ranges:
                raise ValueError(f"{g.title}: index {idx[i]} is not legal at this position")
            if rows[i, C_PRICE_KIND] >= 0:
                r = ranges[int(idx[i])]
                if r is None:
                    raise ValueError(
                        f"{g.title}: legal index {idx[i]} has price_kind {rows[i, C_PRICE_KIND]} but no range"
                    )
                price_range[i] = r
        wires = [wire_of[int(i)] for i in idx]
        return LegalActions(rows=rows, price_range=price_range, indices=idx, price_grid=True, wires=wires)
    idx = np.asarray(g.factored_legal_indices() if indices is None else indices, dtype=np.int64)
    rows = d.gather(idx)
    price_range = np.full((len(idx), 3), -1, dtype=np.int64)
    for i in np.flatnonzero(rows[:, C_PRICE_KIND] >= 0):
        r = g.rl_price_range(int(idx[i]))
        if r is None:
            raise ValueError(f"{g.title}: legal index {idx[i]} has price_kind {rows[i, C_PRICE_KIND]} but no range")
        price_range[i] = r
    return LegalActions(rows=rows, price_range=price_range, indices=idx, price_grid=d.price_grid)


def _unique_rows(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``np.unique(keys, axis=0)``'s first-occurrence indices and inverse (1-D)."""
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return first.astype(np.int64), np.asarray(inverse, dtype=np.int64).reshape(-1)


def build_inputs(
    games: Sequence,
    legal: Optional[Sequence[LegalActions]] = None,
    dedupe_maps: bool = False,
    config: Optional["GenericModelConfig"] = None,
) -> dict:
    """The model's inputs for ``games`` (``engine_rs.BaseGame`` objects or adapters, any titles and
    player counts) as numpy arrays: the encoder's token arrays (``encode_batch``) plus, with
    ``legal`` (one :class:`LegalActions` per position), the flat legal list:

    * ``legal_rows`` (N, 14), ``legal_pos`` (N,) the position of each row, ``legal_offsets`` (B + 1,);
    * map-family factorization keys -- ``map_rows`` (Nm,) rows of map families; ``map_k1`` /
      ``map_k2`` (Nm,) their (group, hex) and (group, hex, tile | city) ids; ``k1_rep`` / ``k1_group``
      and ``k2_rep`` / ``k2_k1`` (a representative row and the parent id of every key). Keys are
      content-based, so nothing depends on the order of the rows;
    * prices -- ``price_rows`` (P,) rows with a ``price_kind``, ``price_range`` (P, 3) (lo, hi, step),
      ``price_cell_mask`` (P, NUM_CELLS) the non-empty cells of each range (``price_pmf``).

    Without ``legal`` the inputs serve ``value_only`` forwards. With ``config``, ids are checked
    against the model's vocabulary sizes (a checkpoint older than an appended vocabulary entry)."""
    arrays = encode_batch(list(games), dedupe_maps)
    if legal is None:
        return arrays
    if len(legal) != len(games):
        raise ValueError(f"{len(legal)} legal lists for {len(games)} positions")
    return add_legal_inputs(arrays, legal, config=config)


def add_legal_inputs(
    arrays: dict, legal: Sequence[LegalActions], config: Optional["GenericModelConfig"] = None
) -> dict:
    """Add the flat legal list of ``legal`` (one :class:`LegalActions` per position of ``arrays``, in
    order) to token arrays -- ``encode_batch``'s, or a supervised batch's collated encodings
    (``generic_data.collate``) -- as :func:`build_inputs` describes. Returns ``arrays`` (updated)."""
    if len(legal) != len(arrays["meta"]):
        raise ValueError(f"{len(legal)} legal lists for {len(arrays['meta'])} positions")
    v = vocab()
    counts = np.array([len(la) for la in legal], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    rows = (
        np.concatenate([np.asarray(la.rows, dtype=np.int64).reshape(-1, len(DESCRIPTOR_FIELDS)) for la in legal])
        if len(legal)
        else np.zeros((0, len(DESCRIPTOR_FIELDS)), dtype=np.int64)
    )
    pos = np.repeat(np.arange(len(legal), dtype=np.int64), counts)
    price_all = (
        np.concatenate([np.asarray(la.price_range, dtype=np.int64).reshape(-1, 3) for la in legal])
        if len(legal)
        else np.zeros((0, 3), dtype=np.int64)
    )
    grid_all = np.repeat(np.array([bool(la.price_grid) for la in legal], dtype=bool), counts)
    arrays["legal_rows"] = rows
    arrays["legal_pos"] = pos
    arrays["legal_offsets"] = offsets

    # -- map families: group, (group, hex), (group, hex, tile | city) keys ---------------------------
    fam = rows[:, C_FAMILY]
    lay, place, remove = _family_id("LayTile"), _family_id("PlaceToken"), _family_id("RemoveToken")
    map_rows = np.flatnonzero(np.isin(fam, v["map_families"])).astype(np.int64)
    if len(map_rows):
        r = rows[map_rows]
        _, group = _unique_rows(np.column_stack([pos[map_rows], r[:, list(GROUP_COLUMNS)]]))
        k1_first, k1 = _unique_rows(np.column_stack([group, r[:, C_HEX]]))
        rf = r[:, C_FAMILY]
        x2 = np.where(rf == lay, r[:, C_TILE], np.where((rf == place) | (rf == remove), r[:, C_CITY], NONE))
        k2_first, k2 = _unique_rows(np.column_stack([k1, x2]))
        arrays.update(
            map_rows=map_rows,
            map_k1=k1,
            map_k2=k2,
            k1_rep=map_rows[k1_first],
            k1_group=group[k1_first],
            k2_rep=map_rows[k2_first],
            k2_k1=k1[k2_first],
        )
    else:
        empty = np.zeros(0, dtype=np.int64)
        arrays.update(
            map_rows=empty, map_k1=empty, map_k2=empty, k1_rep=empty, k1_group=empty, k2_rep=empty, k2_k1=empty
        )

    # -- prices: the legal range's non-empty cells ---------------------------------------------------
    price_rows = np.flatnonzero(rows[:, C_PRICE_KIND] >= 0).astype(np.int64)
    cell_mask = np.zeros((len(price_rows), NUM_CELLS), dtype=bool)
    for j, i in enumerate(price_rows):
        cells = price_cells_for(int(rows[i, C_PRICE_KIND]), *price_all[i], grid=bool(grid_all[i]))
        cell_mask[j] = cells.nonempty
    arrays["price_rows"] = price_rows
    arrays["price_range"] = price_all[price_rows]
    arrays["price_grid"] = grid_all[price_rows]
    arrays["price_cell_mask"] = cell_mask
    if config is not None:
        check_inputs(config, arrays)
    return arrays


def price_cells_for(price_kind: int, lo: int, hi: int, step: int, grid: bool = False):
    """The cell partition of a legal range for a descriptor ``price_kind`` (its ``price_cell_types``
    entry: "Bid" ladder at ``step``, "BuyTrain", "BuyCompany"; ``amount`` has no cells yet).
    ``grid`` False (1830): ``price_pmf.price_cells(cell_type, lo, hi, step)``, the legacy cells.
    ``grid`` True (a generator layout): the cells in step units, so they hold legal prices only
    (``action_descriptors.RangePriceCells``; ``cell_of`` / ``member`` in prices either way)."""
    v = vocab()
    cell_type = v["price_cell_types"][price_kind]
    if cell_type is None:
        raise NotImplementedError(f"price kind {v['price_kinds'][price_kind]!r} has no cell spec yet")
    if lo < 0 or hi < lo:
        raise ValueError(f"bad legal range [{lo}, {hi}] for a {cell_type} price")
    if grid:
        return range_price_cells(cell_type, int(lo), int(hi), int(step), grid=True)
    return price_pmf.price_cells(cell_type, int(lo), int(hi), int(step) if step > 0 else price_pmf.BID_STEP_1830)


def price_targets(inputs: dict, prices: np.ndarray) -> Dict[str, np.ndarray]:
    """Price-head targets for ``inputs["price_rows"]``: ``prices`` (P,) the observed price of each
    priced row (-1 = none). Returns ``price_cell`` (P,), ``price_log_count`` (P,) (log |cell|) and
    ``price_valid`` (P,) -- a target counts only inside a range with more than one price."""
    prices = np.asarray(prices, dtype=np.int64)
    rows = inputs["legal_rows"]
    pr = inputs["price_rows"]
    cell = np.zeros(len(pr), dtype=np.int64)
    log_count = np.zeros(len(pr), dtype=np.float32)
    valid = np.zeros(len(pr), dtype=bool)
    grids = inputs.get("price_grid")
    for j, i in enumerate(pr):
        price = int(prices[j])
        lo, hi, step = (int(x) for x in inputs["price_range"][j])
        grid = bool(grids[j]) if grids is not None else False
        if price < 0 or hi <= lo or not lo <= price <= hi or (grid and (price - lo) % max(step, 1)):
            continue
        cells = price_cells_for(int(rows[i, C_PRICE_KIND]), lo, hi, step, grid=grid)
        c = cells.cell_of(price)
        cell[j], log_count[j], valid[j] = c, math.log(int(cells.counts[c])), True
    return {"price_cell": cell, "price_log_count": log_count, "price_valid": valid}


def check_inputs(config: "GenericModelConfig", arrays: dict) -> None:
    """Raise if an id in ``arrays`` is outside the model's vocabularies (e.g. a family appended
    after the checkpoint was trained) or an entity / tile index outside its tables."""
    c = config.resolved()
    checks = [
        ("title", arrays["meta"][:, 0], c.num_titles),
        ("round kind", arrays["meta"][:, 1], c.num_round_kinds),
        ("step kind", arrays["meta"][:, 2], c.num_step_kinds),
        ("entity type", arrays["ent_type"], c.num_entity_types),
        ("relation kind", arrays["rel"][..., 2], c.num_relation_kinds),
        ("label", arrays["hex_ids"][..., 0], c.num_labels),
    ]
    if "legal_rows" in arrays:
        rows = arrays["legal_rows"]
        checks += [
            ("family", rows[:, C_FAMILY], c.num_families),
            ("choice", rows[:, C_CHOICE] + 1, c.num_choices + 1),
            ("price kind", rows[:, C_PRICE_KIND] + 1, c.num_price_kinds + 1),
        ]
    for name, ids, size in checks:
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() >= size):
            raise ValueError(f"{name} id {int(ids.max())} outside the model's vocabulary of {size}")
    if arrays["ent_type"].shape[1] > c.max_entities:
        raise ValueError(f"{arrays['ent_type'].shape[1]} entities > max_entities {c.max_entities}")


def to_device(arrays: dict, device: torch.device | str = "cpu") -> dict:
    """The input arrays as tensors on ``device`` (ints as int64, masks as bool, floats as float32)."""
    out = {}
    for key, value in arrays.items():
        if value is None:
            out[key] = None
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if tensor.dtype in (torch.int32, torch.int16, torch.uint8):
            tensor = tensor.long()
        out[key] = tensor.to(device)
    return out


# ---------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------


@dataclass
class GenericModelConfig:
    """Sizes of :class:`GenericPolicyValueModel`. The default (~8M parameters) mirrors the 1830 v2
    model's budget; the vocabulary sizes are filled from the engine's schema (``resolved``) and saved
    with every checkpoint."""

    d_model: int = 256
    num_heads: int = 8
    ff_mult: int = 2
    #: Map encoder layers (hex self-attention; runs once per distinct map with ``dedupe_maps``).
    hex_layers: int = 4
    #: Entity self-attention layers before the fusion.
    entity_layers: int = 1
    #: Fusion rounds, each: entities <- hexes, hexes <- entities, entity self-attention.
    fusion_layers: int = 2
    d_trunk: int = 256
    trunk_blocks: int = 2
    #: Conditioning embedding (title + round kind + step kind + seat count) for FiLM.
    d_cond: int = 64
    #: Width of the action / price scorers.
    d_head: int = 192
    #: Width of the per-edge neighbor features of the rotation factor.
    d_edge: int = 32
    value_head_layers: int = 3
    max_hex_distance: int = 12
    #: Rows of the per-(title, entity) and per-(title, tile) id embeddings, and their rank (an id
    #: embedding is a small correction on top of the features: ``d_id`` then projected to ``d_model``).
    max_entities: int = 160
    max_tiles: int = 128
    d_id: int = 64
    #: The price head reads the legal range (lo, hi, step in units of the starting cash).
    price_range_features: bool = True
    #: The rotation factor reads the neighbors across the edges a lay's track reaches.
    rot_edge_features: bool = True
    # Vocabulary sizes (None: from the engine's schema).
    num_titles: Optional[int] = None
    num_round_kinds: Optional[int] = None
    num_step_kinds: Optional[int] = None
    num_labels: Optional[int] = None
    num_entity_types: Optional[int] = None
    num_relation_kinds: Optional[int] = None
    num_families: Optional[int] = None
    num_choices: Optional[int] = None
    num_price_kinds: Optional[int] = None
    f_hex: Optional[int] = None
    f_ent: Optional[int] = None
    f_tile: Optional[int] = None
    s_max: Optional[int] = None
    p_ent: Optional[int] = None
    p_hex: Optional[int] = None
    max_seats: Optional[int] = None

    def resolved(self) -> "GenericModelConfig":
        """A copy with every vocabulary size filled from the current schema where None."""
        s, v = schema(), vocab()
        current = dict(
            num_titles=len(v["titles"]) + 1,
            num_round_kinds=len(v["round_kinds"]),
            num_step_kinds=len(v["step_kinds"]),
            num_labels=len(v["labels"]) + 2,
            num_entity_types=len(v["entity_types"]),
            num_relation_kinds=len(v["relation_kinds"]),
            num_families=len(v["families"]),
            num_choices=len(v["choices"]),
            num_price_kinds=len(v["price_kinds"]),
            f_hex=s["f_hex"],
            f_ent=s["f_ent"],
            f_tile=s["f_tile"],
            s_max=s["s_max"],
            p_ent=s["p_ent"],
            p_hex=s["p_hex"],
            max_seats=s["max_seats"],
        )
        values = asdict(self)
        for key, value in current.items():
            if values[key] is None:
                values[key] = value
        return GenericModelConfig(**values)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "GenericModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown GenericModelConfig fields: {sorted(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------------


def segment_logsumexp(x: Tensor, segment: Tensor, num_segments: int) -> Tensor:
    """logsumexp of ``x`` over each segment id (empty segments: -inf). fp32."""
    x = x.float()
    peak = torch.full((num_segments,), float("-inf"), device=x.device, dtype=x.dtype)
    peak = peak.scatter_reduce(0, segment, x.detach(), reduce="amax", include_self=True)
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    total = torch.zeros(num_segments, device=x.device, dtype=x.dtype).index_add(
        0, segment, torch.exp(x - peak[segment])
    )
    return peak + torch.log(total)


def segment_log_softmax(x: Tensor, segment: Tensor, num_segments: int) -> Tensor:
    """``log_softmax`` of ``x`` within each segment (``segment`` (N,) ids in [0, num_segments))."""
    return x.float() - segment_logsumexp(x, segment, num_segments)[segment]


class AttentionBlock(nn.Module):
    """Pre-norm attention + feed-forward. Queries ``x`` attend to ``context`` with an additive
    per-head bias ``(B, heads, Lq, Lk)`` (or None) and a key mask ``(B, Lk)``."""

    def __init__(self, d: int, heads: int, ff_mult: int):
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} not divisible by {heads} heads")
        self.heads = heads
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, x: Tensor, context: Tensor, bias: Optional[Tensor], key_mask: Tensor) -> Tensor:
        b, lq, d = x.shape
        h = self.heads
        q = self.q(self.norm_q(x)).view(b, lq, h, d // h).transpose(1, 2)
        k, v = self.kv(self.norm_kv(context)).chunk(2, dim=-1)
        k = k.reshape(b, -1, h, d // h).transpose(1, 2)
        v = v.reshape(b, -1, h, d // h).transpose(1, 2)
        mask = torch.zeros(b, 1, 1, key_mask.shape[1], device=x.device, dtype=q.dtype)
        mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if bias is not None:
            mask = mask + bias.to(q.dtype)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(attn.transpose(1, 2).reshape(b, lq, d))
        return x + self.ff(self.norm_ff(x))


class MaskedPool(nn.Module):
    """A learned query attending over a token sequence (padding masked) -> (B, d)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm = nn.LayerNorm(d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.LayerNorm(d)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        h = self.heads
        k, v = self.kv(self.norm(x)).chunk(2, dim=-1)
        k = k.reshape(b, n, h, d // h).transpose(1, 2)
        v = v.reshape(b, n, h, d // h).transpose(1, 2)
        q = self.query.to(k.dtype).expand(b, -1, -1).reshape(b, 1, h, d // h).transpose(1, 2)
        attn_mask = torch.zeros(b, 1, 1, n, device=x.device, dtype=q.dtype).masked_fill(
            ~mask[:, None, None, :], float("-inf")
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(out.transpose(1, 2).reshape(b, d))


class FiLMResBlock(nn.Module):
    """1830's trunk block: residual MLP with FiLM inside (identity at init)."""

    def __init__(self, d: int, d_cond: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.fc2 = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.film = nn.Linear(d_cond, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = F.gelu(self.ln1(self.fc1(x)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        out = (1 + gamma) * out + beta
        return F.gelu(self.ln2(self.fc2(out)) + x)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.GELU(), nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out)
    )


def _gather_tokens(tokens: Tensor, batch: Tensor, index: Tensor, none: Optional[Tensor] = None) -> Tensor:
    """``tokens[batch, index]`` for any shape of ``index`` (``batch`` broadcastable to it); -1 entries
    give ``none`` (a learned "none" vector) or zeros."""
    valid = index >= 0
    out = tokens[batch.expand_as(index), index.clamp(min=0)]
    if none is None:
        return out * valid[..., None].to(out.dtype)
    return torch.where(valid[..., None], out, none.to(out.dtype))


# ---------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------


class GenericPolicyValueModel(nn.Module):
    """Typed-token policy / value network for any registered title (``docs/rl_model.md``).

    1. **Map encoder** (per distinct map): hex features + label + laid-tile features / id, each slot's
       token and reservation resolved to the corporation's static embedding (title, index, type), then
       ``hex_layers`` of hex self-attention biased by clipped distance, direction, track connectivity
       across the shared edge and "same corporation's tokens".
    2. **Entities** (per position): a per-type projection of the features + type, (title, index) id
       and seat embeddings, one hop of pointers (owner, link, phases, home / destination hex), the
       acting entity marked; ``entity_layers`` of self-attention with the relation bias.
    3. **Fusion** (per position): the map's hex tokens plus the acting entity's tokens / home on each
       hex, then ``fusion_layers`` rounds of entities <- hexes and hexes <- entities (biased where a
       corporation holds, reserves or calls a hex home) and entity self-attention.
    4. **Trunk**: global token, acting entity token, pooled entities and hexes -> ``d_trunk``, FiLM
       residual blocks on embedding(title) + embedding(round kind) + embedding(step kind) +
       embedding(seats).
    5. **Heads** (fp32): the descriptor policy scorer with the map factorization, the price head, the
       per-seat win / score heads.
    """

    def __init__(self, config: Optional[GenericModelConfig] = None):
        super().__init__()
        c = (config or GenericModelConfig()).resolved()
        self.config = c
        d, heads = c.d_model, c.num_heads
        v = vocab()
        t = title_tables()
        if t["e_max"] > c.max_entities:
            raise ValueError(f"a title has {t['e_max']} entities > max_entities {c.max_entities}")
        if t["t_max"] > c.max_tiles:
            raise ValueError(f"a title has {t['t_max']} tiles > max_tiles {c.max_tiles}")
        if len(v["titles"]) + 1 > c.num_titles:
            LOGGER.warning("the engine knows %d titles, this model %d", len(v["titles"]), c.num_titles - 1)
        g_cols = {name: k for k, name in enumerate(schema()["entity_features"]["global"])}
        self.starting_cash_col = g_cols["starting_cash"]
        self.value_stop_grad = False

        # -- static tables (rebuilt from the engine; not saved: the checkpoint's fingerprints guard them) --
        nt = c.num_titles

        def table(name, array, dtype):
            data = torch.as_tensor(np.asarray(array), dtype=dtype)
            if data.shape[0] < nt:  # a model trained before a title was registered: pad
                data = torch.cat([data, data.new_zeros((nt - data.shape[0],) + tuple(data.shape[1:]))])
            self.register_buffer(name, data[:nt], persistent=False)

        table("t_hex_dist", t["hex_dist"].astype(np.int64).clip(max=c.max_hex_distance), torch.long)
        table("t_hex_dir", t["hex_dir"], torch.long)
        table("t_hex_nbr", t["hex_nbr"], torch.long)
        table("t_ent_type", t["ent_type"], torch.long)
        table("t_desc_base", t["desc_base"], torch.long)
        table("t_tile_feat", t["tile_feat"], torch.float32)
        table("t_tile_exits", t["tile_exits"], torch.float32)
        table("t_num_tiles", t["num_tiles"], torch.long)
        table("t_bank", t["bank"], torch.float32)
        param_scale = np.zeros((nt, c.num_families), dtype=np.float32)
        for fam_name, scale in _PARAM_SCALE_BY_FAMILY.items():
            if fam_name in v["families"][: c.num_families]:
                f = v["families"].index(fam_name)
                param_scale[: len(t["max_market"]), f] = t["max_market"] if isinstance(scale, str) else scale
        self.register_buffer("t_param_scale", torch.as_tensor(param_scale).clamp(min=1.0), persistent=False)
        is_map = torch.zeros(c.num_families, dtype=torch.bool)
        for f in v["map_families"]:
            if f < c.num_families:
                is_map[f] = True
        self.register_buffer("is_map_family", is_map, persistent=False)
        self.lay_family = _family_id("LayTile")
        self.token_families = (_family_id("PlaceToken"), _family_id("RemoveToken"))
        hexcols = _hex_feature_columns()
        self.register_buffer("hex_edge_track", torch.as_tensor(hexcols["edge_track"]), persistent=False)
        self.hex_edge_stop = hexcols["edge_stop"]
        self.hex_wall = hexcols["wall"]
        self.hex_border = hexcols["border"]

        # -- embeddings shared by the map and the entities ------------------------------------------------
        self.title_emb = nn.Embedding(c.num_titles, d)
        self.type_emb = nn.Embedding(c.num_entity_types, d)
        self.ent_id_emb = nn.Sequential(
            nn.Embedding(c.num_titles * c.max_entities, c.d_id), nn.Linear(c.d_id, d, bias=False)
        )
        nn.init.normal_(self.ent_id_emb[0].weight, std=0.02)  # per (title, entity index)
        self.tile_feat_proj = nn.Linear(c.f_tile, d)
        self.tile_id_emb = nn.Sequential(
            nn.Embedding(c.num_titles * c.max_tiles, c.d_id), nn.Linear(c.d_id, d, bias=False)
        )
        nn.init.zeros_(self.tile_id_emb[0].weight)  # per (title, tile)

        # -- 1. map encoder ------------------------------------------------------------------------------
        self.hex_in = nn.Linear(c.f_hex, d)
        self.label_emb = nn.Embedding(c.num_labels, d)
        self.laid_tile_proj = nn.Linear(d, d)
        self.slot_token_proj = nn.Linear(d, d)
        self.slot_res_proj = nn.Linear(d, d)
        self.slot_special = nn.Parameter(torch.randn(2, d) * 0.02)  # neutral token, empty slot
        self.map_in_norm = nn.LayerNorm(d)
        self.dist_bias = nn.Embedding(c.max_hex_distance + 1, heads)
        self.dir_bias = nn.Embedding(7, heads)
        self.track_bias = nn.Parameter(torch.zeros(heads))
        self.same_corp_bias = nn.Parameter(torch.zeros(heads))
        for emb in (self.dist_bias, self.dir_bias):
            nn.init.zeros_(emb.weight)
        self.map_layers = nn.ModuleList(AttentionBlock(d, heads, c.ff_mult) for _ in range(c.hex_layers))
        self.map_out_norm = nn.LayerNorm(d)

        # -- 2. entities ---------------------------------------------------------------------------------
        self.ent_in = nn.ModuleList(nn.Linear(c.f_ent, d) for _ in range(c.num_entity_types - 1))  # type t at t - 1
        self.seat_emb = nn.Embedding(c.max_seats + 1, d)  # 0 = not a seat
        self.ptr_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(c.p_ent))
        self.hex_ptr_proj = nn.ModuleList(nn.Linear(d, d) for _ in range(c.p_hex))
        self.acting_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.ent_in_norm = nn.LayerNorm(d)
        self.rel_bias = nn.Embedding(2 * c.num_relation_kinds, 2 * heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.entity_layers = nn.ModuleList(AttentionBlock(d, heads, c.ff_mult) for _ in range(c.entity_layers))

        # -- 3. fusion -----------------------------------------------------------------------------------
        self.cond_title = nn.Embedding(c.num_titles, c.d_cond)
        self.cond_round = nn.Embedding(c.num_round_kinds, c.d_cond)
        self.cond_step = nn.Embedding(c.num_step_kinds, c.d_cond)
        self.cond_seats = nn.Embedding(c.max_seats + 1, c.d_cond)
        self.cond_tokens = nn.Linear(c.d_cond, d)
        self.acting_hex_proj = nn.Linear(4, d)  # acting entity: token here, reserved here, home, destination
        self.fusion_bias_eh = nn.Parameter(torch.zeros(4, heads))  # holds, reserves, home, destination
        self.fusion_bias_he = nn.Parameter(torch.zeros(4, heads))
        self.fusion_ent_from_hex = nn.ModuleList(AttentionBlock(d, heads, c.ff_mult) for _ in range(c.fusion_layers))
        self.fusion_hex_from_ent = nn.ModuleList(AttentionBlock(d, heads, c.ff_mult) for _ in range(c.fusion_layers))
        self.fusion_ent_self = nn.ModuleList(AttentionBlock(d, heads, c.ff_mult) for _ in range(c.fusion_layers))
        self.ent_out_norm = nn.LayerNorm(d)
        self.hex_out_norm = nn.LayerNorm(d)

        # -- 4. trunk ------------------------------------------------------------------------------------
        self.ent_pool = MaskedPool(d, heads)
        self.hex_pool = MaskedPool(d, heads)
        self.none_acting = nn.Parameter(torch.zeros(d))
        self.trunk_in = nn.Linear(4 * d, c.d_trunk)
        self.trunk_norm = nn.LayerNorm(c.d_trunk)
        self.trunk_blocks = nn.ModuleList(FiLMResBlock(c.d_trunk, c.d_cond) for _ in range(c.trunk_blocks))

        # -- 5a. policy: descriptor scorer ------------------------------------------------------------------
        dh = c.d_head
        self.ctx_proj = nn.Linear(c.d_trunk, dh)
        self.family_emb = nn.Embedding(c.num_families, dh)
        self.choice_emb = nn.Embedding(c.num_choices + 1, dh)  # 0 = none
        self.price_kind_emb = nn.Embedding(c.num_price_kinds + 1, dh)
        self.ent0_proj = nn.Linear(d, dh)
        self.ent1_proj = nn.Linear(d, dh)
        self.via_proj = nn.Linear(d, dh)
        self.hexref_proj = nn.Linear(d, dh)
        self.none_ref = nn.Parameter(torch.randn(4, d) * 0.02)  # ent0, ent1, via, hex: "none"
        self.param_proj = nn.Linear(3, dh)
        self.flags_proj = nn.Linear(NUM_FLAG_BITS, dh)
        self.base_mlp = _mlp(dh, dh, 1)
        # map factors: P(hex | group), P(tile | city | hex, group), P(rot | hex, tile, group)
        self.f1_hex = nn.Linear(d, dh)
        self.f1_mlp = _mlp(dh, dh, 1)
        self.f2_hex = nn.Linear(d, dh)
        self.f2_tile = nn.Linear(d, dh)
        self.f2_city = nn.Linear(d, dh)
        self.city_emb = nn.Embedding(c.s_max + 1, d)  # 0 = none
        self.f2_mlp = _mlp(dh, dh, 1)
        self.f3_hex = nn.Linear(d, dh)
        self.f3_tile = nn.Linear(d, dh)
        self.rot_emb = nn.Embedding(7, dh)  # 0 = none
        self.edge_tok = nn.Linear(d, c.d_edge)
        self.edge_raw = nn.Linear(7, c.d_edge)
        self.edge_dir = nn.Parameter(torch.randn(6, c.d_edge) * 0.02)
        self.f3_edge = nn.Linear(c.d_edge, dh)
        self.f3_mlp = _mlp(dh, dh, 1)

        # -- 5b. price head ------------------------------------------------------------------------------------
        self.price_row = nn.Linear(dh, dh)
        self.price_range_proj = nn.Linear(7, dh)
        self.price_mlp = _mlp(dh, dh, NUM_CELLS)

        # -- 5c. value heads (per seat, one MLP for every seat) -----------------------------------------------
        self.win_head = self._value_mlp()
        self.score_head = self._value_mlp()

        for head in (self.win_head, self.score_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.price_mlp[-1].weight)
        nn.init.zeros_(self.price_mlp[-1].bias)

    def _value_mlp(self) -> nn.Sequential:
        c = self.config
        layers: List[nn.Module] = []
        d_in, hidden = c.d_model + c.d_trunk, c.d_trunk // 2
        for _ in range(c.value_head_layers - 1):
            layers += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d_in = hidden
        layers.append(nn.Linear(d_in, 1))
        return nn.Sequential(*layers)

    def architecture_name(self) -> str:
        return ARCHITECTURE

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- pieces ------------------------------------------------------------------------------------------

    def _static_entities(self, titles: Tensor, num: int) -> Tensor:
        """(len(titles), num, d) static entity embeddings: (title, index) id + the title's static type."""
        index = torch.arange(num, device=titles.device)
        ids = titles[:, None] * self.config.max_entities + index[None, :]
        return self.ent_id_emb(ids) + self.type_emb(self.t_ent_type[:, :num][titles])

    def _tile_embedding(self, titles: Tensor, tiles: Tensor) -> Tensor:
        """Embedding of tile ``tiles`` (title manifest index, -1 = none -> 0) of ``titles``."""
        valid = (tiles >= 0) & (tiles < self.t_num_tiles[titles])
        t = tiles.clamp(min=0)
        emb = self.tile_feat_proj(self.t_tile_feat[titles, t]) + self.tile_id_emb(titles * self.config.max_tiles + t)
        return emb * valid[..., None].to(emb.dtype)

    def _map_encoder(self, inputs: dict, map_title: Tensor, num_entities: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per distinct map: hex tokens (M, H, d), token and reservation counts (M, H, E) and the raw
        per-edge features of the rotation factor (M, H, 6, 7)."""
        hex_feat, hex_ptr, hex_res = inputs["hex_feat"], inputs["hex_ptr"], inputs["hex_res"]
        hex_mask, hex_ids = inputs["hex_mask"], inputs["hex_ids"]
        m, h, _ = hex_feat.shape
        e = num_entities
        x = self.hex_in(hex_feat) + self.label_emb(hex_ids[..., 0]) + self.title_emb(map_title)[:, None]
        x = x + self.laid_tile_proj(self._tile_embedding(map_title[:, None].expand(m, h), hex_ids[..., 1] - 1))
        # slots: corporations' tokens and reservations through their static embeddings
        tok_count = hex_feat.new_zeros(m, h * e)
        flat_hex = torch.arange(h, device=hex_feat.device)[None, :, None] * e
        tok_count.index_put_(
            (torch.arange(m, device=x.device)[:, None, None].expand_as(hex_ptr), (flat_hex + hex_ptr.clamp(min=0))),
            (hex_ptr >= 0).to(tok_count.dtype),
            accumulate=True,
        )
        tok_count = tok_count.view(m, h, e)
        res_count = hex_feat.new_zeros(m, h * e)
        res_count.index_put_(
            (torch.arange(m, device=x.device)[:, None, None].expand_as(hex_res), (flat_hex + hex_res.clamp(min=0))),
            (hex_res >= 0).to(res_count.dtype),
            accumulate=True,
        )
        res_count = res_count.view(m, h, e)
        static = self._static_entities(map_title, e)  # (M, E, d)
        neutral = (hex_ptr == -2).sum(-1, keepdim=True).to(x.dtype)
        empty = ((hex_ptr == -1) & (inputs["hex_slot_city"] >= 0)).sum(-1, keepdim=True).to(x.dtype)
        x = x + self.slot_token_proj(torch.bmm(tok_count.to(static.dtype), static))
        x = x + self.slot_res_proj(torch.bmm(res_count.to(static.dtype), static))
        x = x + neutral * self.slot_special[0] + empty * self.slot_special[1]
        x = self.map_in_norm(x)

        # attention bias: distance, direction, track across the shared edge, same corporation's tokens
        dist = self.t_hex_dist[:, :h, :h][map_title]
        direction = self.t_hex_dir[:, :h, :h][map_title]
        bias = (self.dist_bias(dist) + self.dir_bias(direction)).permute(0, 3, 1, 2)  # (M, heads, H, H)
        edge_track = (hex_feat @ self.hex_edge_track > 0).to(x.dtype)  # (M, H, 6)
        nbr = self.t_hex_nbr[map_title][:, :h]  # (M, H, 6)
        valid_nbr = nbr >= 0
        bm = torch.arange(m, device=x.device)[:, None, None]
        opposite = edge_track[bm, nbr.clamp(min=0), (torch.arange(6, device=x.device) + 3) % 6]  # (M, H, 6)
        conn = edge_track * opposite * valid_nbr.to(x.dtype)
        track = x.new_zeros(m, h, h)
        track.index_put_(
            (bm.expand_as(nbr), torch.arange(h, device=x.device)[None, :, None].expand_as(nbr), nbr.clamp(min=0)),
            conn,
            accumulate=True,
        )
        has = (tok_count > 0).to(x.dtype)
        same = (torch.bmm(has, has.transpose(1, 2)) > 0).to(x.dtype)
        bias = bias + track[:, None] * self.track_bias[None, :, None, None]
        bias = bias + same[:, None] * self.same_corp_bias[None, :, None, None]
        for layer in self.map_layers:
            x = layer(x, x, bias, hex_mask)
        x = self.map_out_norm(x) * hex_mask[..., None].to(x.dtype)
        # per-edge raw features for the rotation factor: neighbor exists / has track on the facing edge /
        # a stop on it, this hex's wall and border cost on the edge
        facing = (torch.arange(6, device=x.device) + 3) % 6
        stops = torch.tensor(self.hex_edge_stop, device=x.device)[facing]  # (6, 3)
        nbr_stop = hex_feat[bm[..., None], nbr.clamp(min=0)[..., None], stops[None, None]]  # (M, H, 6, 3)
        raw = torch.cat(
            [
                valid_nbr.to(x.dtype)[..., None],
                opposite[..., None],
                nbr_stop * valid_nbr[..., None].to(x.dtype),
                hex_feat[..., self.hex_wall][..., None],
                hex_feat[..., self.hex_border][..., None],
            ],
            dim=-1,
        )  # (M, H, 6, 7)
        return x, tok_count, res_count, raw

    def _entity_tokens(self, inputs: dict, hex_in: Tensor) -> Tensor:
        ent_type, ent_feat, ent_mask, meta = inputs["ent_type"], inputs["ent_feat"], inputs["ent_mask"], inputs["meta"]
        b, e, _ = ent_feat.shape
        flat_feat, flat_type = ent_feat.reshape(b * e, -1), ent_type.reshape(-1)
        rows, values = [], []
        for t in range(1, self.config.num_entity_types):
            sel = (flat_type == t).nonzero(as_tuple=True)[0]
            if sel.numel():
                rows.append(sel)
                values.append(self.ent_in[t - 1](flat_feat[sel]))
        if values:
            projected = torch.cat(values)
            x = projected.new_zeros(b * e, projected.shape[-1]).index_copy(0, torch.cat(rows), projected).view(b, e, -1)
        else:
            x = ent_feat.new_zeros(b, e, self.config.d_model)
        titles = meta[:, 0]
        index = torch.arange(e, device=x.device)
        x = x + self.type_emb(ent_type) + self.ent_id_emb(titles[:, None] * self.config.max_entities + index[None, :])
        seat = torch.where((index >= 1) & (index <= self.config.max_seats), index, torch.zeros_like(index))
        x = x + self.seat_emb(seat)[None]
        base = x
        bi = torch.arange(b, device=x.device)[:, None]
        for col, proj in enumerate(self.ptr_proj):
            x = x + proj(_gather_tokens(base, bi, inputs["ent_ptr"][..., col]))
        for col, proj in enumerate(self.hex_ptr_proj):
            x = x + proj(_gather_tokens(hex_in, bi, inputs["ent_hex"][..., col]))
        acting = meta[:, 3]
        x = x + (index[None, :] == acting[:, None]).to(x.dtype)[..., None] * self.acting_emb
        return self.ent_in_norm(x) * ent_mask[..., None].to(x.dtype)

    def _relation_bias(self, inputs: dict, num_entities: int) -> Tensor:
        rel, rel_val, rel_mask = inputs["rel"], inputs["rel_val"], inputs["rel_mask"]
        b, r = rel_val.shape
        heads = self.config.num_heads
        bias = rel_val.new_zeros(b, heads, num_entities * num_entities)
        if r == 0:
            return bias.view(b, heads, num_entities, num_entities)
        src, dst, kind = rel[..., 0].clamp(min=0), rel[..., 1].clamp(min=0), rel[..., 2]
        n_rel = self.config.num_relation_kinds
        for direction, (i, j) in enumerate(((src, dst), (dst, src))):
            params = self.rel_bias(kind + direction * n_rel)  # (b, r, 2 * heads)
            const, scale = params[..., :heads], params[..., heads:]
            value = (const + scale * rel_val[..., None]) * rel_mask[..., None].to(rel_val.dtype)
            index = (i * num_entities + j)[:, None, :].expand(-1, heads, -1)
            bias = bias.scatter_add(2, index, value.transpose(1, 2).to(bias.dtype))
        return bias.view(b, heads, num_entities, num_entities)

    def _conditioning(self, meta: Tensor) -> Tensor:
        seats = meta[:, 4].clamp(0, self.config.max_seats)
        return (
            self.cond_title(meta[:, 0])
            + self.cond_round(meta[:, 1])
            + self.cond_step(meta[:, 2])
            + self.cond_seats(seats)
        )

    # -- forward -------------------------------------------------------------------------------------------

    def forward(self, inputs: dict, value_only: bool = False) -> dict:
        """``inputs``: :func:`build_inputs` arrays as tensors (:func:`to_device`). Returns

        * ``logits`` (N,): each legal row's score (map rows: group score + log of the map factors);
          ``logp`` (N,): log-probabilities, normalized over each position's legal rows;
        * ``price_logp`` (P, NUM_CELLS): log-probabilities over the cells of each priced row's range
          (empty cells at about -1e4);
        * ``win_logits`` (B, MAX_SEATS) (absent seats -1e4), ``score`` (B, MAX_SEATS) (absent 0),
          ``present`` (B, MAX_SEATS);
        * ``trunk`` (B, d_trunk), ``entities`` (B, E, d), ``hexes`` (B, H, d).

        With ``value_only`` (or inputs without legal rows) the policy and price outputs are None."""
        meta = inputs["meta"]
        b, e = inputs["ent_type"].shape
        map_index = inputs.get("map_index")
        m = inputs["hex_feat"].shape[0]
        if map_index is None:
            map_index = torch.arange(b, device=meta.device)
            map_title = meta[:, 0]
        else:
            map_title = meta.new_zeros(m).scatter(0, map_index, meta[:, 0])

        # 1. map (per distinct map), then each position's copy
        hex_map, tok_count, res_count, edge_raw = self._map_encoder(inputs, map_title, e)
        hexes = hex_map[map_index]
        hex_mask = inputs["hex_mask"][map_index]
        h = hexes.shape[1]
        holds = tok_count[map_index].transpose(1, 2)  # (B, E, H)
        reserves = res_count[map_index].transpose(1, 2)
        bi = torch.arange(b, device=meta.device)[:, None]
        hex_ids = torch.arange(h, device=meta.device)
        ent_hex = inputs["ent_hex"]
        home = (ent_hex[..., 0, None] == hex_ids).to(hexes.dtype)  # (B, E, H)
        dest = (ent_hex[..., 1, None] == hex_ids).to(hexes.dtype)

        # 2. entities
        hex_in = hexes  # one hop of the home / destination pointers reads the map's tokens
        ent = self._entity_tokens(inputs, hex_in)
        ent_mask = inputs["ent_mask"].clone()
        ent_mask[:, 0] = True  # the global token always exists
        cond = self._conditioning(meta)
        ent = ent + self.cond_tokens(cond)[:, None]
        rel_bias = self._relation_bias(inputs, e)
        for layer in self.entity_layers:
            ent = layer(ent, ent, rel_bias, ent_mask)

        # 3. fusion: the acting entity's marks on the map, then entities <-> hexes
        acting = meta[:, 3]
        has_acting = (acting >= 0)[:, None, None].to(hexes.dtype)
        a = acting.clamp(min=0)
        acting_marks = torch.stack(
            [holds[bi[:, 0], a], reserves[bi[:, 0], a], home[bi[:, 0], a], dest[bi[:, 0], a]], dim=-1
        )  # (B, H, 4)
        hexes = (
            hexes
            + self.acting_hex_proj((acting_marks > 0).to(hexes.dtype) * has_acting)
            + self.cond_tokens(cond)[:, None]
        )
        marks = torch.stack([holds, reserves, home, dest], dim=-1)  # (B, E, H, 4)
        bias_eh = torch.einsum("behk,kn->bneh", marks, self.fusion_bias_eh.to(marks.dtype))
        bias_he = torch.einsum("behk,kn->bnhe", marks, self.fusion_bias_he.to(marks.dtype))
        safe_hex_mask = hex_mask.clone()
        safe_hex_mask[:, 0] = True
        for ent_from_hex, hex_from_ent, ent_self in zip(
            self.fusion_ent_from_hex, self.fusion_hex_from_ent, self.fusion_ent_self
        ):
            ent_next = ent_from_hex(ent, hexes, bias_eh, safe_hex_mask)
            hexes = hex_from_ent(hexes, ent, bias_he, ent_mask)
            ent = ent_self(ent_next, ent_next, rel_bias, ent_mask)
        ent = self.ent_out_norm(ent) * inputs["ent_mask"][..., None].to(ent.dtype)
        hexes = self.hex_out_norm(hexes) * hex_mask[..., None].to(hexes.dtype)

        # 4. trunk
        acting_tok = torch.where(
            (acting >= 0)[:, None], ent[torch.arange(b, device=meta.device), a], self.none_acting.to(ent.dtype)
        )
        pooled = torch.cat(
            [ent[:, 0], acting_tok, self.ent_pool(ent, ent_mask), self.hex_pool(hexes, safe_hex_mask)], dim=-1
        )
        x = self.trunk_norm(self.trunk_in(pooled))
        for block in self.trunk_blocks:
            x = block(x, cond)

        out = {"trunk": x, "entities": ent, "hexes": hexes, "logits": None, "logp": None, "price_logp": None}

        # 5. heads in fp32 (as 1830's: per-position offsets dwarf bf16's spacing)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf, entf, hexf = x.float(), ent.float(), hexes.float()
            if not value_only and "legal_rows" in inputs:
                out.update(self._policy_and_price(inputs, xf, entf, hexf, edge_raw.float(), map_index))
            out.update(self._values(inputs, xf, entf))
        return out

    def _values(self, inputs: dict, trunk: Tensor, ent: Tensor) -> dict:
        s = self.config.max_seats
        seats = ent[:, 1 : 1 + s]
        present = inputs["ent_mask"][:, 1 : 1 + s]
        if seats.shape[1] < s:  # a batch whose titles have fewer entity slots than seats can't happen
            raise ValueError("entity tokens shorter than the seat block")
        t = trunk.detach() if self.value_stop_grad else trunk
        if self.value_stop_grad:
            seats = seats.detach()
        seat_in = torch.cat([seats, t[:, None].expand(-1, s, -1)], dim=-1)
        win = self.win_head(seat_in).squeeze(-1).masked_fill(~present, MASKED)
        score = self.score_head(seat_in).squeeze(-1).masked_fill(~present, 0.0)
        return {"win_logits": win, "score": score, "present": present}

    def _resolve_entity(self, kind: Tensor, index: Tensor, pos: Tensor, meta: Tensor) -> Tensor:
        """Entity token index of descriptor entity (kind, index) at positions ``pos``; -1 = none."""
        title = meta[pos, 0]
        base = self.t_desc_base[title, kind.clamp(min=0)]
        n = meta[pos, 4].clamp(min=1)
        seat = 1 + torch.remainder(index - meta[pos, 5], n)
        tok = torch.where(kind == EK_PLAYER, seat, base + index)
        return torch.where((kind >= 0) & (index >= 0), tok, torch.full_like(tok, -1))

    def _policy_and_price(self, inputs, trunk, ent, hexes, edge_raw, map_index) -> dict:
        c = self.config
        rows, pos, meta = inputs["legal_rows"], inputs["legal_pos"], inputs["meta"]
        b = trunk.shape[0]
        title = meta[pos, 0]
        fam = rows[:, C_FAMILY]
        is_map = self.is_map_family[fam]
        e0 = self._resolve_entity(rows[:, C_E0K], rows[:, C_E0], pos, meta)
        e1 = self._resolve_entity(rows[:, C_E1K], rows[:, C_E1], pos, meta)
        via = torch.where(
            rows[:, C_VIA] >= 0, self.t_desc_base[title, EK_COMPANY] + rows[:, C_VIA], torch.full_like(fam, -1)
        )
        hexref = torch.where(is_map, torch.full_like(fam, -1), rows[:, C_HEX])
        param = rows[:, C_PARAM]
        scale = self.t_param_scale[title, fam]
        has_param = param >= 0
        p = torch.where(has_param, param.float() / scale, torch.zeros_like(scale)).clamp(-16, 16)
        param_feat = torch.stack([p, p * p, has_param.float()], dim=-1)
        flags = rows[:, C_FLAGS]
        bits = torch.stack([((flags >> k) & 1).float() for k in range(NUM_FLAG_BITS)], dim=-1)
        ctx = self.ctx_proj(trunk)
        base_h = (
            ctx[pos]
            + self.family_emb(fam)
            + self.ent0_proj(_gather_tokens(ent, pos, e0, self.none_ref[0]))
            + self.ent1_proj(_gather_tokens(ent, pos, e1, self.none_ref[1]))
            + self.via_proj(_gather_tokens(ent, pos, via, self.none_ref[2]))
            + self.hexref_proj(_gather_tokens(hexes, pos, hexref, self.none_ref[3]))
            + self.choice_emb(rows[:, C_CHOICE] + 1)
            + self.price_kind_emb(rows[:, C_PRICE_KIND] + 1)
            + self.param_proj(param_feat)
            + self.flags_proj(bits)
        )
        logits = self.base_mlp(base_h).squeeze(-1)  # (N,): a map row's group score
        components = {"group_logits": logits}

        mr = inputs["map_rows"]
        if mr.numel():
            k1_rep, k1_group, k2_rep, k2_k1 = inputs["k1_rep"], inputs["k1_group"], inputs["k2_rep"], inputs["k2_k1"]
            num_groups = int(k1_group.max()) + 1 if k1_group.numel() else 0
            num_k1, num_k2 = k1_rep.shape[0], k2_rep.shape[0]

            def hex_tok(r):
                return _gather_tokens(hexes, pos[r], rows[r, C_HEX])

            def tile_emb(r):
                is_lay = (fam[r] == self.lay_family)[:, None].float()
                return (
                    self._tile_embedding(
                        title[r], torch.where(fam[r] == self.lay_family, rows[r, C_TILE], torch.full_like(fam[r], -1))
                    )
                    * is_lay
                )

            # level 1: P(hex | group)
            u1 = self.f1_mlp(base_h[k1_rep] + self.f1_hex(hex_tok(k1_rep))).squeeze(-1)
            lp1 = segment_log_softmax(u1, k1_group, num_groups)
            # level 2: P(tile | hex, group) for tile lays, P(city | hex, group) for tokens
            u2_in = base_h[k2_rep] + self.f2_hex(hex_tok(k2_rep)) + self.f2_tile(tile_emb(k2_rep))
            is_tok = (fam[k2_rep] == self.token_families[0]) | (fam[k2_rep] == self.token_families[1])
            tok_keys = is_tok.nonzero(as_tuple=True)[0]
            if tok_keys.numel():  # only token keys get the city term (a batch's other rows never see it)
                city = self.f2_city(self._city_vectors(inputs, ent, k2_rep[tok_keys], map_index))
                u2_in = u2_in.index_add(0, tok_keys, city)
            u2 = self.f2_mlp(u2_in).squeeze(-1)
            lp2 = segment_log_softmax(u2, k2_k1, num_k1)
            # level 3: P(rot | hex, tile, group) (other map families: one row per key, log 1 = 0)
            rot = rows[mr, C_ROT]
            u3_in = base_h[mr] + self.f3_hex(hex_tok(mr)) + self.f3_tile(tile_emb(mr)) + self.rot_emb(rot + 1)
            if c.rot_edge_features:
                u3_in = u3_in + self.f3_edge(self._edge_term(inputs, hexes, edge_raw, mr, map_index))
            u3 = self.f3_mlp(u3_in).squeeze(-1)
            lp3 = segment_log_softmax(u3, inputs["map_k2"], num_k2)
            map_logits = logits[mr].float() + lp1[inputs["map_k1"]] + lp2[inputs["map_k2"]] + lp3
            logits = logits.float().index_put((mr,), map_logits)
            components.update(log_p_hex=lp1, log_p_second=lp2, log_p_rot=lp3, u1=u1, u2=u2, u3=u3)

        logp = segment_log_softmax(logits, pos, b)
        out = {"logits": logits, "logp": logp, "policy_components": components}

        pr = inputs["price_rows"]
        if pr.numel():
            h = self.price_row(base_h[pr])
            if c.price_range_features:
                h = h + self.price_range_proj(self._range_features(inputs, pos[pr]))
            price_logits = self.price_mlp(h).float()
            price_logits = price_logits.masked_fill(~inputs["price_cell_mask"], MASKED)
            out["price_logp"] = torch.log_softmax(price_logits, dim=-1)
        else:
            out["price_logp"] = trunk.new_zeros(0, NUM_CELLS)
        return out

    def _range_features(self, inputs: dict, pos: Tensor) -> Tensor:
        """A priced row's legal range in units of the starting cash per player (the encoder's money unit)."""
        meta = inputs["meta"]
        sc = (inputs["ent_feat"][pos, 0, self.starting_cash_col] * self.t_bank[meta[pos, 0]]).clamp(min=1.0)
        lo, hi, step = (inputs["price_range"][:, k].float() for k in range(3))
        width = (hi - lo).clamp(min=0)
        feats = torch.stack(
            [
                lo / sc,
                hi / sc,
                width / sc,
                step.clamp(min=0) / sc,
                torch.log1p(width) / math.log(1000.0),
                (width == 0).float(),
                torch.log1p(lo.clamp(min=0)) / 7.0,
            ],
            dim=-1,
        )
        return feats.clamp(-16, 16)

    def _city_vectors(self, inputs: dict, ent: Tensor, rep: Tensor, map_index: Tensor) -> Tensor:
        """Per token row: the mean over its city's slots of the slot's occupant (entity token, or the
        neutral / empty embedding) and reservation, + the city's index embedding."""
        rows, pos = inputs["legal_rows"], inputs["legal_pos"]
        b = pos[rep]
        mi = map_index[b]
        hx = rows[rep, C_HEX].clamp(min=0)
        city = rows[rep, C_CITY]
        ptr = inputs["hex_ptr"][mi, hx]  # (K, S)
        res = inputs["hex_res"][mi, hx]
        slot_city = inputs["hex_slot_city"][mi, hx]
        in_city = (slot_city == city[:, None]) & (city[:, None] >= 0)
        occupant = _gather_tokens(ent, b[:, None], ptr)
        occupant = (
            occupant
            + (ptr == -2)[..., None].float() * self.slot_special[0]
            + (ptr == -1)[..., None].float() * self.slot_special[1]
        )
        occupant = occupant + _gather_tokens(ent, b[:, None], res)
        w = in_city.float()
        mean = (occupant * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)
        return mean + self.city_emb((city + 1).clamp(0, self.config.s_max))

    def _edge_term(self, inputs: dict, hexes: Tensor, edge_raw: Tensor, mr: Tensor, map_index: Tensor) -> Tensor:
        """Per map row: the sum over the edges a tile lay's track reaches (at its rotation) of the
        neighbor across that edge (projected token + raw edge features + direction); 0 for non-lays."""
        rows, pos, meta = inputs["legal_rows"], inputs["legal_pos"], inputs["meta"]
        b = pos[mr]
        title = meta[b, 0]
        hx = rows[mr, C_HEX].clamp(min=0)
        is_lay = rows[mr, C_FAMILY] == self.lay_family
        tile = torch.where(is_lay, rows[mr, C_TILE], torch.zeros_like(hx)).clamp(min=0)
        rot = rows[mr, C_ROT].clamp(0, 5)
        exits = self.t_tile_exits[title, tile, rot] * is_lay[:, None].float()  # (Nm, 6)
        nbr = self.t_hex_nbr[title, hx]  # (Nm, 6)
        nbr_tok = self.edge_tok(_gather_tokens(hexes, b[:, None], nbr))  # (Nm, 6, d_edge)
        raw = self.edge_raw(edge_raw[map_index[b], hx])  # (Nm, 6, d_edge)
        edge = nbr_tok + raw + self.edge_dir
        return torch.einsum("nk,nkd->nd", exits, edge)

    def run(self, inputs: dict) -> dict:
        """Inference (fp32, no grad): ``probs`` (N,) per position's legal rows, ``price_probs``
        (P, NUM_CELLS), ``win`` (B, MAX_SEATS) win shares, ``score`` (B, MAX_SEATS)."""
        with torch.no_grad():
            out = self.forward(inputs)
        return {
            "probs": None if out["logp"] is None else out["logp"].exp(),
            "price_probs": None if out["price_logp"] is None else out["price_logp"].exp(),
            "win": torch.softmax(out["win_logits"].float(), dim=-1),
            "score": out["score"],
        }


# ---------------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------------


def policy_loss(
    out: dict,
    inputs: dict,
    target_index: Optional[Tensor] = None,
    target_probs: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Cross-entropy of the policy against each position's target, over its legal rows only.

    ``target_index`` (B,): the target's index in the position's legal list (0 .. n_b - 1; -1 = no
    target), or ``target_probs`` (N,): a distribution over each position's legal rows (sums to 1 over
    a position with a target, 0 elsewhere). ``weight`` (B,) optional per-position weights. Mean over the
    positions with a target (weighted)."""
    logp = out["logp"]
    pos = inputs["legal_pos"]
    b = inputs["legal_offsets"].shape[0] - 1
    if target_index is not None:
        has = target_index >= 0
        flat = (inputs["legal_offsets"][:-1] + target_index.clamp(min=0)).clamp(max=max(logp.shape[0] - 1, 0))
        nll = -logp[flat] * has.float()
    elif target_probs is not None:
        per_row = -(target_probs.float() * logp)
        nll = logp.new_zeros(b).index_add(0, pos, per_row)
        has = logp.new_zeros(b).index_add(0, pos, target_probs.float()) > 0
    else:
        raise ValueError("policy_loss needs target_index or target_probs")
    w = has.float() if weight is None else has.float() * weight.float()
    return (nll * w).sum() / w.sum().clamp(min=1e-8)


def chosen_log_prob(out: dict, inputs: dict, index: Tensor) -> Tensor:
    """(B,) log-probability of each position's chosen legal row (``index`` into its legal list), e.g.
    the policy-gradient ratio's numerator; a price-bearing move adds its price's
    ``price_logp[cell] - log |cell|``."""
    return out["logp"][inputs["legal_offsets"][:-1] + index]


def policy_entropy(out: dict, inputs: dict) -> Tensor:
    """(B,) entropy of each position's policy over its legal rows (0 for a position without any)."""
    logp = out["logp"]
    b = inputs["legal_offsets"].shape[0] - 1
    return logp.new_zeros(b).index_add(0, inputs["legal_pos"], -(logp.exp() * logp))


def price_loss(
    out: dict, price_cell: Tensor, price_log_count: Tensor, price_valid: Tensor, weight: Optional[Tensor] = None
) -> Tensor:
    """Exact NLL of observed prices (``price_targets``): ``-(log P(cell) - log |cell|)``, weighted mean
    over the valid targets (0 without any) -- 1830's ``_compute_price_nll_loss``, keyed by row."""
    logp = out["price_logp"]
    if logp is None or logp.shape[0] == 0:
        return out["trunk"].new_zeros(())
    nll = -(logp.gather(1, price_cell[:, None]).squeeze(1) - price_log_count.float())
    w = price_valid.float() if weight is None else price_valid.float() * weight.float()
    return (nll * w).sum() / w.sum().clamp(min=1e-8)


def value_losses(out: dict, win_target: Tensor, score_target: Tensor) -> Dict[str, Tensor]:
    """Win CE over the present seats (``win_target`` (B, seats): a distribution over the present
    seats, e.g. the share of winners) and score MSE over the present seats (placement points).
    Targets may be narrower than ``MAX_SEATS`` (padded with 0)."""
    win_logits, score, present = out["win_logits"], out["score"], out["present"]
    pad = win_logits.shape[1] - win_target.shape[1]
    if pad:
        win_target = F.pad(win_target, (0, pad))
        score_target = F.pad(score_target, (0, pad))
    log_v = torch.log_softmax(win_logits.float(), dim=1)
    ce = -(win_target.float() * log_v * present.float()).sum(dim=1).mean()
    sq = ((score.float() - score_target.float()) ** 2) * present.float()
    mse = sq.sum() / present.float().sum().clamp(min=1.0)
    hit = win_target.gather(1, log_v.argmax(dim=1, keepdim=True)).mean()
    return {"ce": ce, "mse": mse, "winner_hit": hit.detach()}


# ---------------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------------

STATE_DICT_KEY = "state_dict"
CONFIG_KEY = "config"
ARCHITECTURE_KEY = "architecture"
META_KEY = "generic_meta"


def checkpoint_meta(titles: Sequence[str]) -> dict:
    """What a checkpoint must agree with to load: the encoder version, the titles and each one's
    layout (version + fingerprint; None for a title without a layout yet) and the vocabularies."""
    layouts = {}
    for title in titles:
        try:
            d = action_descriptors(title)
            layouts[title] = {"version": d.version, "fingerprint": d.fingerprint, "total": d.total}
        except RuntimeError:  # not bridge-ready: no layout yet
            layouts[title] = None
    v = vocab()
    return {
        "format": CHECKPOINT_FORMAT,
        "encoder_version": int(schema()["version"]),
        "titles": list(titles),
        "layouts": layouts,
        "vocab": {k: list(val) for k, val in v.items()},
        "max_seats": int(schema()["max_seats"]),
        "num_cells": NUM_CELLS,
    }


def save_checkpoint(model: GenericPolicyValueModel, path, titles: Sequence[str], extra: Optional[dict] = None) -> None:
    """Write ``model`` with its config and :func:`checkpoint_meta` (atomic rename)."""
    meta = checkpoint_meta(titles)
    if extra:
        meta["extra"] = dict(extra)
    payload = {
        ARCHITECTURE_KEY: ARCHITECTURE,
        STATE_DICT_KEY: model.state_dict(),
        CONFIG_KEY: model.config.to_json(),
        META_KEY: meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class CheckpointMismatch(ValueError):
    """A checkpoint that does not fit the current encoder / layouts / vocabularies."""


def check_checkpoint_meta(meta: dict, check_layouts: bool = True) -> None:
    """Raise :class:`CheckpointMismatch` unless ``meta`` fits the running engine: same encoder version,
    every saved vocabulary a prefix of the current one (append-only), same layout fingerprint for every
    title that had a layout."""
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointMismatch(f"checkpoint format {meta.get('format')} != {CHECKPOINT_FORMAT}")
    current = int(schema()["version"])
    if int(meta.get("encoder_version", -1)) != current:
        raise CheckpointMismatch(
            f"checkpoint encoder version {meta.get('encoder_version')} != running encoder {current}"
        )
    if int(meta.get("max_seats", -1)) != int(schema()["max_seats"]):
        raise CheckpointMismatch(f"checkpoint MAX_SEATS {meta.get('max_seats')} != {schema()['max_seats']}")
    if int(meta.get("num_cells", -1)) != NUM_CELLS:
        raise CheckpointMismatch(f"checkpoint price cells {meta.get('num_cells')} != {NUM_CELLS}")
    v = vocab()
    for key, saved in meta.get("vocab", {}).items():
        now = [list(x) if isinstance(x, tuple) else x for x in v.get(key, ())]
        if key == "map_families":
            if list(saved) != list(now)[: len(saved)]:
                raise CheckpointMismatch(f"map families changed: {saved} vs {now}")
            continue
        if list(saved) != list(now)[: len(saved)]:
            raise CheckpointMismatch(f"vocabulary {key!r} is not an append-only extension of the checkpoint's")
    if check_layouts:
        for title, saved in meta.get("layouts", {}).items():
            if saved is None:
                continue
            try:
                d = action_descriptors(title)
            except RuntimeError as err:
                raise CheckpointMismatch(f"{title}: checkpoint has layout {saved['version']}, the engine none ({err})")
            if (d.version, d.fingerprint) != (saved["version"], saved["fingerprint"]):
                raise CheckpointMismatch(
                    f"{title}: checkpoint layout {saved['version']} {saved['fingerprint']} "
                    f"!= engine {d.version} {d.fingerprint}"
                )


def load_checkpoint(
    path, device: torch.device | str = "cpu", check_layouts: bool = True
) -> Tuple[GenericPolicyValueModel, dict]:
    """``(model, generic_meta)`` from :func:`save_checkpoint`'s file; refuses (CheckpointMismatch) a
    checkpoint of another architecture, encoder version, layout fingerprint or non-append vocabulary."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get(ARCHITECTURE_KEY) != ARCHITECTURE:
        found = payload.get(ARCHITECTURE_KEY) if isinstance(payload, dict) else type(payload).__name__
        raise CheckpointMismatch(f"{path}: not a {ARCHITECTURE} checkpoint ({found})")
    meta = payload[META_KEY]
    check_checkpoint_meta(meta, check_layouts=check_layouts)
    model = GenericPolicyValueModel(GenericModelConfig.from_json(payload[CONFIG_KEY]))
    model.load_state_dict(payload[STATE_DICT_KEY], strict=True)
    return model.to(device), meta


__all__ = [
    "ARCHITECTURE",
    "CheckpointMismatch",
    "GenericModelConfig",
    "GenericPolicyValueModel",
    "LegalActions",
    "add_legal_inputs",
    "build_inputs",
    "check_checkpoint_meta",
    "check_inputs",
    "chosen_log_prob",
    "checkpoint_meta",
    "legal_actions",
    "load_checkpoint",
    "policy_loss",
    "policy_entropy",
    "price_cells_for",
    "price_loss",
    "price_targets",
    "save_checkpoint",
    "segment_log_softmax",
    "title_tables",
    "to_device",
    "value_losses",
    "vocab",
]
