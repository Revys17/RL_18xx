"""Dense auxiliary value targets for 1830 (KataGo's ownership / score targets, for this game).

A game gives the value one outcome per ~700 decisions. KataGo's network also predicts every
point's final owner and the final score -- hundreds of low-noise labels per position -- and
lets those gradients shape the shared trunk. These are 1830's analogues, all read from a
finished game (its replay and its action log):

per player (canonical seat order: the mover first)
    ``p_final``  (4, 4)  net-worth share, cash share, bankrupt, presidencies / 8
    ``pc_final`` (4, 8)  percent of each corporation held / 100
    ``p_short``  (4,)    net-worth share ``SHORT_HORIZON`` actions later (or at the end)
per corporation (title order)
    ``c_final``  (8, 4)  floated, share price / 350, trains held / 4 (capped), cash / 1000 (capped)
    ``c_pres``   (8,)    the final president's canonical seat, 4 = none
    ``c_next``   (8,)    the corporation's next run's revenue / 400 (capped), ``c_next_mask`` 1 if
                         it runs again
per hex (the encoder's order: ``encoder.HEX_COORDS_ORDERED``)
    ``h_color``  (93,)   final tile colour: 0 empty, 1 yellow, 2 green, 3 brown, 4 gray, 5 other
    ``h_token``  (93, 8) a token of each corporation in the hex at the end
    ``h_route``  (93, 8) the hex is on each corporation's last run

:class:`DenseHeads` predicts them from the value network's own activations (player and
corporation tokens of the entity encoder, the map encoder's per-hex embeddings, the trunk),
captured by forward hooks, so the network class is unchanged and a checkpoint trained with
them loads as a plain one. Training-only: nothing at inference reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

NUM_PLAYERS = 4
NUM_CORPS = 8
SHORT_HORIZON = 100  # engine actions
COLOR_CLASSES = {"white": 0, "yellow": 1, "green": 2, "brown": 3, "gray": 4}
NUM_COLORS = 6
PER_GAME_KEYS = ("p_final", "pc_final", "c_final", "c_pres", "h_color", "h_token", "h_route")
PER_POSITION_KEYS = ("p_short", "c_next", "c_next_mask")


def _hex_order() -> list:
    from rl18xx.agent.alphazero.encoder import HEX_COORDS_ORDERED

    return list(HEX_COORDS_ORDERED)


def _corp_order() -> list:
    from rl18xx.agent.alphazero.encoder import CORPORATION_IDS_ORDERED

    return list(CORPORATION_IDS_ORDERED)


def net_worth_shares(rs) -> np.ndarray:
    """Per seat (player-id order) its net worth's share of the players' clipped total."""
    from rl18xx.agent.alphazero.self_play import _compute_net_worth
    from rl18xx.rust_adapter import RustGameAdapter

    worth = _compute_net_worth(RustGameAdapter(rs))
    w = np.array([max(float(worth[pid]), 0.0) for pid in sorted(worth)], dtype=np.float64)
    return (w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))).astype(np.float32)


def final_targets(rs, actions: list) -> dict:
    """The per-game targets of a finished game ``rs`` (an ``engine_rs.BaseGame``) and its action
    log, in absolute seat order (player-id order); :func:`canonical` rotates them per position."""
    corps = _corp_order()
    corp_index = {sym: i for i, sym in enumerate(corps)}
    hexes = _hex_order()
    hex_index = {h: i for i, h in enumerate(hexes)}
    players = sorted(rs.players, key=lambda p: int(p.id))
    seat_of = {f"player:{int(p.id)}": s for s, p in enumerate(players)}
    by_sym = {c.sym: c for c in rs.corporations}

    from rl18xx.agent.alphazero.self_play import _compute_net_worth
    from rl18xx.rust_adapter import RustGameAdapter

    worth = _compute_net_worth(RustGameAdapter(rs))
    worth_total = sum(max(float(v), 0.0) for v in worth.values()) or 1.0
    shares = np.array([max(float(worth[pid]), 0.0) / worth_total for pid in sorted(worth)], dtype=np.float32)
    p_final = np.zeros((NUM_PLAYERS, 4), dtype=np.float32)
    pc_final = np.zeros((NUM_PLAYERS, NUM_CORPS), dtype=np.float32)
    for s, p in enumerate(players[:NUM_PLAYERS]):
        p_final[s, 0] = shares[s]
        p_final[s, 1] = float(p.cash) / worth_total
        p_final[s, 2] = float(bool(p.bankrupt))
        for sym, i in corp_index.items():
            pc_final[s, i] = float(p.percent_of(by_sym[sym])) / 100.0
    c_final = np.zeros((NUM_CORPS, 4), dtype=np.float32)
    c_pres = np.full(NUM_CORPS, NUM_PLAYERS, dtype=np.int64)
    for sym, i in corp_index.items():
        c = by_sym[sym]
        floated = bool(c.floated)
        c_final[i, 0] = float(floated)
        price = c.share_price.price if c.share_price is not None else 0
        c_final[i, 1] = float(price) / 350.0 if floated else 0.0
        c_final[i, 2] = min(len(c.trains), 4) / 4.0
        c_final[i, 3] = min(max(float(c.cash), 0.0), 1000.0) / 1000.0
        owner = c.owner_id_str
        if owner in seat_of:
            c_pres[i] = seat_of[owner]
            p_final[seat_of[owner], 3] += 1.0 / NUM_CORPS

    h_color = np.full(len(hexes), NUM_COLORS - 1, dtype=np.int64)
    h_token = np.zeros((len(hexes), NUM_CORPS), dtype=np.float32)
    for h in rs.hexes:
        k = hex_index.get(h.id)
        if k is None:
            continue
        h_color[k] = COLOR_CLASSES.get(str(h.tile.color), NUM_COLORS - 1)
        for city in h.tile.cities:
            for token in city.tokens:
                if token is not None and token.corporation_id in corp_index:
                    h_token[k, corp_index[token.corporation_id]] = 1.0
    h_route = np.zeros((len(hexes), NUM_CORPS), dtype=np.float32)
    last_run = {}
    for a in actions:
        if a.get("type") == "run_routes" and a.get("entity") in corp_index:
            last_run[a["entity"]] = a.get("routes") or []
    for sym, routes in last_run.items():
        for route in routes:
            for hx in route.get("hexes", []):
                if hx in hex_index:
                    h_route[hex_index[hx], corp_index[sym]] = 1.0
    return {
        "p_final": p_final,
        "pc_final": pc_final,
        "c_final": c_final,
        "c_pres": c_pres,
        "h_color": h_color,
        "h_token": h_token,
        "h_route": h_route,
    }


def run_revenues(actions: list) -> dict:
    """Corporation -> ``[(action index, revenue)]`` of its runs, in game order."""
    corps = set(_corp_order())
    out = {sym: [] for sym in corps}
    for i, a in enumerate(actions):
        if a.get("type") == "run_routes" and a.get("entity") in corps:
            revenue = sum(int(r.get("revenue", 0)) for r in a.get("routes") or []) + int(a.get("extra_revenue") or 0)
            out[a["entity"]].append((i, revenue))
    return out


def next_run_targets(revenues: dict, i: int) -> tuple:
    """``(c_next, c_next_mask)`` at action index ``i``: each corporation's next run's revenue
    after ``i`` (/ 400, capped at 1) and whether there is one."""
    corps = _corp_order()
    nxt = np.zeros(NUM_CORPS, dtype=np.float32)
    mask = np.zeros(NUM_CORPS, dtype=np.float32)
    for k, sym in enumerate(corps):
        for idx, revenue in revenues.get(sym, []):
            if idx >= i:
                nxt[k] = min(revenue, 400) / 400.0
                mask[k] = 1.0
                break
    return nxt, mask


def canonical(targets: dict, rotation: int) -> dict:
    """Per-game targets rotated into a position's canonical frame (``rotation`` = the mover's
    seat): seat axes rolled so the mover is first, presidents' seats renumbered."""
    r = int(rotation)
    out = dict(targets)
    for key in ("p_final", "pc_final", "p_short"):
        if key in targets:
            out[key] = np.roll(targets[key], -r, axis=0)
    if "c_pres" in targets:
        pres = targets["c_pres"].copy()
        seated = pres < NUM_PLAYERS
        pres[seated] = (pres[seated] - r) % NUM_PLAYERS
        out["c_pres"] = pres
    return out


# ---------------------------------------------------------------------------
# Heads and loss (training only)
# ---------------------------------------------------------------------------


@dataclass
class DenseWeights:
    """Per-group loss weights (each group's loss is a mean, ~0.01-1)."""

    player: float = 1.0
    holdings: float = 1.0
    short: float = 1.0
    corp: float = 1.0
    president: float = 0.5
    next_run: float = 1.0
    hex_color: float = 0.5
    hex_token: float = 1.0
    hex_route: float = 1.0


def build_dense_heads(model):
    """:class:`DenseHeads` sized for ``model`` (a ``ModelTransformer`` value network)."""
    import torch.nn as nn

    c = model.config

    def mlp(d_in, d_out):
        return nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, d_out))

    class DenseHeads(nn.Module):
        """Player / corporation / hex heads on the network's tokens, map embeddings and trunk."""

        def __init__(self):
            super().__init__()
            self.player = mlp(c.d_entity + c.d_trunk, 4 + NUM_CORPS + 1)
            self.corp = mlp(c.d_entity + c.d_trunk, 4 + NUM_PLAYERS + 1 + 1)
            self.hex = mlp(c.d_map + c.d_trunk, NUM_COLORS + 2 * NUM_CORPS)

    return DenseHeads()


class ActivationTap:
    """Forward hooks capturing a ``ModelTransformer``'s entity embeddings, per-hex map
    embeddings and trunk output on each forward (gradients flow through them)."""

    def __init__(self, model):
        self.model = model
        self.entity = self.hexes = self.trunk = None
        self.handles = [
            model.econ_transformer.register_forward_hook(self._entity),
            model.hex_transformer.register_forward_hook(self._hexes),
            model.res_blocks[-1].register_forward_hook(self._trunk),
        ]

    def _entity(self, module, inputs, output):
        self.entity = output[1]

    def _hexes(self, module, inputs, output):
        self.hexes = output[0]

    def _trunk(self, module, inputs, output):
        self.trunk = output

    def remove(self):
        for h in self.handles:
            h.remove()


def dense_loss(heads, tap: ActivationTap, batch: dict, weights: Optional[DenseWeights] = None) -> tuple:
    """``(total, per-group losses)`` of the dense targets in ``batch`` (canonical-frame tensors,
    keys as in this module) from the activations ``tap`` captured on the batch's forward."""
    import torch
    import torch.nn.functional as F

    w = weights or DenseWeights()
    econ = tap.model.econ_transformer
    trunk = tap.trunk.float()
    ent = tap.entity.float()
    players = ent[:, econ.player_token_start : econ.player_token_start + NUM_PLAYERS]
    corps = ent[:, econ.player_token_end : econ.player_token_end + NUM_CORPS]
    hexes = tap.hexes.float()

    def with_trunk(tokens):
        return torch.cat([tokens, trunk.unsqueeze(1).expand(-1, tokens.shape[1], -1)], dim=-1)

    p_out = heads.player(with_trunk(players))  # (B, 4, 13)
    c_out = heads.corp(with_trunk(corps))  # (B, 8, 10)
    h_out = heads.hex(with_trunk(hexes))  # (B, 93, 22)

    pf = batch["p_final"]
    losses = {}
    losses["player"] = F.mse_loss(p_out[..., [0, 1, 3]], pf[..., [0, 1, 3]]) + F.binary_cross_entropy_with_logits(
        p_out[..., 2], pf[..., 2]
    )
    losses["holdings"] = F.mse_loss(p_out[..., 4 : 4 + NUM_CORPS], batch["pc_final"])
    losses["short"] = F.mse_loss(p_out[..., 4 + NUM_CORPS], batch["p_short"])
    cf = batch["c_final"]
    losses["corp"] = F.binary_cross_entropy_with_logits(c_out[..., 0], cf[..., 0]) + F.mse_loss(
        c_out[..., 1:4], cf[..., 1:4]
    )
    losses["president"] = F.cross_entropy(c_out[..., 4 : 4 + NUM_PLAYERS + 1].reshape(-1, NUM_PLAYERS + 1),
                                          batch["c_pres"].reshape(-1))
    mask = batch["c_next_mask"]
    losses["next_run"] = ((c_out[..., -1] - batch["c_next"]) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
    losses["hex_color"] = F.cross_entropy(h_out[..., :NUM_COLORS].reshape(-1, NUM_COLORS), batch["h_color"].reshape(-1))
    losses["hex_token"] = F.binary_cross_entropy_with_logits(h_out[..., NUM_COLORS : NUM_COLORS + NUM_CORPS],
                                                             batch["h_token"])
    losses["hex_route"] = F.binary_cross_entropy_with_logits(h_out[..., NUM_COLORS + NUM_CORPS :], batch["h_route"])
    total = sum(getattr(w, k) * v for k, v in losses.items())
    return total, {k: float(v.detach()) for k, v in losses.items()}
