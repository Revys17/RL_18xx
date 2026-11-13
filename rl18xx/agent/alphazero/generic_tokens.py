"""Title-generic typed-token inputs (RL bridge milestone B1).

``encode_batch`` wraps the native batched encoder ``engine_rs.rl_encode_batch``
(any titles and player counts in one call; schema in ``docs/rl_encoder.md`` and
``engine_rs.rl_encoder_schema()``), ``TitleGeometry`` holds each title's static
hex geometry, and ``TokenTrunkStub`` is a deliberately small PyTorch module
that consumes the arrays -- hex and entity token embeddings, pointer
resolution, a relation attention bias, a hex-distance bias, entity -> hex
cross-attention, FiLM on title / round / step and the per-seat value logits --
to show the arrays are usable. It has no policy head: the descriptor-conditioned
policy and price heads and the real model are milestone B2
(``docs/rl_bridge_work_packages.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import engine_rs

# The arrays ``encode_batch`` returns, grouped by their leading dimension.
HEX_KEYS = ("hex_feat", "hex_ptr", "hex_res", "hex_slot_city", "hex_ids", "hex_mask")
ENTITY_KEYS = ("ent_type", "ent_feat", "ent_ptr", "ent_hex", "ent_mask")
RELATION_KEYS = ("rel", "rel_val", "rel_mask")


@lru_cache(maxsize=None)
def schema() -> dict:
    """``engine_rs.rl_encoder_schema()``: widths, feature names, vocabularies."""
    return engine_rs.rl_encoder_schema()


@lru_cache(maxsize=None)
def spec_info(title: str) -> dict:
    """``engine_rs.rl_spec_info(title)``: hex order and geometry, entity names, tile table, normalizers."""
    return engine_rs.rl_spec_info(title)


def raw_game(game):
    """The ``engine_rs.BaseGame`` behind a game: itself, or a ``RustGameAdapter``'s
    (``titles.new_game("1830", n)`` returns an adapter)."""
    return getattr(game, "_game", game)


def encode_batch(games: list, dedupe_maps: bool = False) -> dict:
    """Encode positions (``engine_rs.BaseGame`` objects or adapters around them,
    any titles / player counts) into stacked numpy arrays. With ``dedupe_maps``
    the hex arrays hold each distinct hex block once and ``map_index`` (B,)
    names each position's. Entity, hex and tile orders are the descriptor
    vocabularies' (``docs/rl_descriptors.md`` §4; ``spec_info(title)
    ["descriptor_entity_base"]`` maps a descriptor's entity to its token)."""
    return engine_rs.rl_encode_batch([raw_game(g) for g in games], dedupe_maps)


def to_torch(arrays: dict, device: torch.device | str = "cpu") -> dict:
    """The arrays as tensors (int arrays as int64, masks as bool)."""
    out = {}
    for key, value in arrays.items():
        if value is None:
            out[key] = None
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if tensor.dtype == torch.int32:
            tensor = tensor.long()
        out[key] = tensor.to(device)
    return out


class TitleGeometry:
    """Per-title hex distance matrices (from ``rl_spec_info``'s axial
    coordinates), stacked and padded for a batch's titles."""

    def __init__(self, max_distance: int = 8):
        self.max_distance = max_distance
        self._by_title: dict[int, torch.Tensor] = {}
        self._titles = {i + 1: t for i, t in enumerate(schema()["titles"])}

    def distances(self, title_id: int) -> torch.Tensor:
        if title_id not in self._by_title:
            axial = torch.from_numpy(np.asarray(spec_info(self._titles[title_id])["hex_axial"], dtype=np.int64))
            dq = axial[:, None, 0] - axial[None, :, 0]
            dr = axial[:, None, 1] - axial[None, :, 1]
            dist = (dq.abs() + dr.abs() + (dq + dr).abs()) // 2
            self._by_title[title_id] = dist.clamp(max=self.max_distance)
        return self._by_title[title_id]

    def batch(self, title_ids: torch.Tensor, num_hexes: int) -> torch.Tensor:
        """(B, H, H) clipped hex distances; padding hexes at ``max_distance``."""
        out = torch.full((len(title_ids), num_hexes, num_hexes), self.max_distance, dtype=torch.long)
        for b, t in enumerate(title_ids.tolist()):
            d = self.distances(int(t))
            out[b, : d.shape[0], : d.shape[1]] = d
        return out


@dataclass
class TokenStubConfig:
    d_model: int = 64
    num_heads: int = 4
    hex_layers: int = 1
    entity_layers: int = 1
    max_distance: int = 8
    dropout: float = 0.0


class BiasedLayer(nn.Module):
    """Pre-norm transformer layer whose attention takes an additive
    (B, heads, Lq, Lk) bias (relations, hex distances, token pointers)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.dropout = dropout

    def forward(self, x, context, bias, key_mask):
        b, lq, d = x.shape
        h = self.heads
        q = self.q(self.norm_q(x)).view(b, lq, h, d // h).transpose(1, 2)
        k, v = self.kv(self.norm_kv(context)).chunk(2, dim=-1)
        k = k.view(b, -1, h, d // h).transpose(1, 2)
        v = v.view(b, -1, h, d // h).transpose(1, 2)
        mask = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.out(attn.transpose(1, 2).reshape(b, lq, d))
        return x + self.ff(self.norm_ff(x))


class TokenTrunkStub(nn.Module):
    """A minimal consumer of ``encode_batch`` arrays (CPU-testable, no policy head).

    * entity tokens: a per-type input projection of ``ent_feat`` + a type
      embedding + the owner pointer's embedding;
    * hex tokens: a projection of ``hex_feat`` + label embedding, then each slot's
      token pointer and reservation resolved to the corporation's entity
      embedding (pointers are resolved in the model, as the design says);
    * entity self-attention with a relation bias (a learned per-head scalar per
      relation kind and direction, scaled by the relation's value);
    * hex self-attention with a per-head bias per clipped hex distance;
    * entity -> hex cross-attention, biased where the corporation has a token;
    * FiLM on embedding(title) + embedding(round kind) + embedding(step kind);
    * per-seat value logits (``MAX_SEATS``; absent seats -1e4).
    """

    def __init__(self, config: TokenStubConfig | None = None):
        super().__init__()
        c = config or TokenStubConfig()
        s = schema()
        d = c.d_model
        self.config = c
        self.f_ent = s["f_ent"]
        self.max_seats = s["max_seats"]
        num_types = len(s["entity_types"])
        self.num_rel = len(s["relation_kinds"])
        self.ent_in = nn.Parameter(torch.randn(num_types, self.f_ent, d) * self.f_ent**-0.5)
        self.ent_type_emb = nn.Embedding(num_types, d)
        self.owner_proj = nn.Linear(d, d)
        self.hex_in = nn.Linear(s["f_hex"], d)
        self.label_emb = nn.Embedding(len(s["labels"]) + 2, d)
        self.slot_emb = nn.Embedding(s["s_max"], d)
        self.token_proj = nn.Linear(d, d)
        self.res_proj = nn.Linear(d, d)
        self.title_emb = nn.Embedding(len(s["titles"]) + 1, d)
        self.round_emb = nn.Embedding(len(s["round_kinds"]), d)
        self.step_emb = nn.Embedding(len(s["step_kinds"]), d)
        # relation bias: (kind, direction) -> per-head (constant, value scale)
        self.rel_bias = nn.Embedding(2 * self.num_rel, 2 * c.num_heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.dist_bias = nn.Embedding(c.max_distance + 1, c.num_heads)
        nn.init.zeros_(self.dist_bias.weight)
        self.token_bias = nn.Parameter(torch.zeros(c.num_heads))
        self.entity_layers = nn.ModuleList(BiasedLayer(d, c.num_heads, c.dropout) for _ in range(c.entity_layers))
        self.hex_layers = nn.ModuleList(BiasedLayer(d, c.num_heads, c.dropout) for _ in range(c.hex_layers))
        self.cross = BiasedLayer(d, c.num_heads, c.dropout)
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.trunk_norm = nn.LayerNorm(d)
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.geometry = TitleGeometry(c.max_distance)

    @staticmethod
    def _gather_entities(ent: torch.Tensor, ptr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Entity embeddings at pointer positions (any trailing shape), and the valid mask."""
        b = ent.shape[0]
        valid = ptr >= 0
        flat = ptr.clamp(min=0).reshape(b, -1)
        out = torch.gather(ent, 1, flat[..., None].expand(-1, -1, ent.shape[-1]))
        return out.reshape(*ptr.shape, ent.shape[-1]) * valid[..., None], valid

    def _relation_bias(self, rel, rel_val, rel_mask, num_entities) -> torch.Tensor:
        b, r = rel_val.shape
        heads = self.config.num_heads
        bias = torch.zeros(b, heads, num_entities * num_entities, dtype=rel_val.dtype, device=rel_val.device)
        if r == 0:
            return bias.view(b, heads, num_entities, num_entities)
        src, dst, kind = rel[..., 0].clamp(min=0), rel[..., 1].clamp(min=0), rel[..., 2]
        for direction, (i, j) in enumerate(((src, dst), (dst, src))):
            params = self.rel_bias(kind + direction * self.num_rel)  # (b, r, 2*heads)
            const, scale = params[..., :heads], params[..., heads:]
            value = (const + scale * rel_val[..., None]) * rel_mask[..., None]
            index = (i * num_entities + j)[:, None, :].expand(-1, heads, -1)
            bias.scatter_add_(2, index, value.transpose(1, 2))
        return bias.view(b, heads, num_entities, num_entities)

    def forward(self, inputs: dict) -> dict:
        ent_type, ent_feat, ent_mask = inputs["ent_type"], inputs["ent_feat"], inputs["ent_mask"]
        b, e, _ = ent_feat.shape
        map_index = inputs.get("map_index")

        def per_position(x):
            return x if map_index is None else x[map_index]

        hex_feat = per_position(inputs["hex_feat"])
        hex_mask = per_position(inputs["hex_mask"])
        hex_ptr, hex_res = per_position(inputs["hex_ptr"]), per_position(inputs["hex_res"])
        hex_ids = per_position(inputs["hex_ids"])
        h = hex_feat.shape[1]
        meta = inputs["meta"]

        # -- entity tokens ---------------------------------------------------------
        weights = self.ent_in[ent_type]  # (b, e, f, d)
        ent = torch.einsum("bef,befd->bed", ent_feat, weights) + self.ent_type_emb(ent_type)
        owner, _ = self._gather_entities(ent, inputs["ent_ptr"][..., 0])
        ent = ent + self.owner_proj(owner)

        # -- hex tokens (pointers resolved to the corporations' embeddings) ------------
        hx = self.hex_in(hex_feat) + self.label_emb(hex_ids[..., 0])
        slots = self.slot_emb.weight[None, None]  # (1, 1, s, d)
        tok, tok_valid = self._gather_entities(ent, hex_ptr)
        res, _ = self._gather_entities(ent, hex_res)
        hx = hx + (self.token_proj(tok + slots) * tok_valid[..., None]).sum(2) + self.res_proj(res).sum(2)

        # -- entity self-attention with the relation bias -------------------------------
        rel_bias = self._relation_bias(inputs["rel"], inputs["rel_val"], inputs["rel_mask"], e)
        safe_ent_mask = ent_mask.clone()
        safe_ent_mask[:, 0] = True  # the global token always exists
        for layer in self.entity_layers:
            ent = layer(ent, ent, rel_bias, safe_ent_mask)

        # -- hex self-attention with the distance bias ---------------------------------
        dist = self.geometry.batch(meta[:, 0].cpu(), h).to(hex_feat.device)
        dist_bias = self.dist_bias(dist).permute(0, 3, 1, 2)
        safe_hex_mask = hex_mask.clone()
        safe_hex_mask[:, 0] = True
        for layer in self.hex_layers:
            hx = layer(hx, hx, dist_bias, safe_hex_mask)

        # -- entities read the map: bias where the corporation holds a token ------------
        holds = torch.zeros(b, e, h, dtype=hx.dtype, device=hx.device)
        rows = hex_ptr.clamp(min=0)  # (b, h, s)
        hex_index = torch.arange(h, device=hx.device)[None, :, None].expand_as(rows)
        flat = (rows * h + hex_index).reshape(b, -1)
        holds.view(b, -1).scatter_add_(1, flat, (hex_ptr >= 0).to(hx.dtype).reshape(b, -1))
        cross_bias = holds[:, None] * self.token_bias[None, :, None, None]
        ent = self.cross(ent, hx, cross_bias, safe_hex_mask)

        # -- trunk, FiLM on title / round / step -------------------------------------------
        cond = self.title_emb(meta[:, 0]) + self.round_emb(meta[:, 1]) + self.step_emb(meta[:, 2])
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        hex_mean = (hx * hex_mask[..., None]).sum(1) / hex_mask.sum(1, keepdim=True).clamp(min=1)
        trunk = self.trunk_norm(ent[:, 0] + hex_mean) * (1 + gamma) + beta

        # -- per-seat value -----------------------------------------------------------------
        seats = ent[:, 1 : 1 + self.max_seats]
        seat_in = torch.cat([seats, trunk[:, None].expand_as(seats)], dim=-1)
        value_logits = self.value_head(seat_in).squeeze(-1)
        present = ent_mask[:, 1 : 1 + self.max_seats]
        value_logits = value_logits.masked_fill(~present, -1e4)
        return {"trunk": trunk, "entities": ent, "hexes": hx, "value_logits": value_logits}
