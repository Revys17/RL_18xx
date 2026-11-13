"""Policy-only play for title-generic checkpoints (RL bridge milestone C1; ``docs/rl_model.md``).

A :class:`GenericPolicy` scores a batch of positions with sparse legal lists in one forward
(``model_generic.legal_actions`` + ``build_inputs``: only the legal moves are scored, no 26,537-wide
dense policy), samples a move at a temperature exactly as the legacy policy-only players do
(``policy_selfplay.tempered`` over the legal moves) and a price from the price head's cells of the move's
legal range (``P(price) = softmax(cells)[cell] / |cell|``; ``price_eps`` mixes in uniform cells as
``policy_gradient.choose_price`` does). Inference runs in the calling process (fp32, no grad, CPU by
default): ``scripts/eval_policy_only.py`` loads one per pool worker for a ``generic:<checkpoint>``
player, so generic checkpoints play legacy ones under the legacy eval's rules and starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rl18xx.agent.alphazero import model_generic as mg
from rl18xx.agent.alphazero.generic_tokens import raw_game

#: Player-spec prefix for a generic checkpoint in ``eval_policy_only.py`` (``generic:<path>[@T]``).
GENERIC_PREFIX = "generic:"


@dataclass
class PolicyOutput:
    """One position's policy: probabilities over its legal list (in the list's order), the price
    head's log-probabilities per priced row of the list, and win shares per seat (encoder frame)."""

    legal: mg.LegalActions
    probs: np.ndarray
    price_logp: Dict[int, np.ndarray]
    win: np.ndarray


def _tempered(p: np.ndarray, temperature: float) -> np.ndarray:
    from rl18xx.agent.alphazero.policy_selfplay import tempered

    return tempered(p, temperature)


class GenericPolicy:
    """A generic checkpoint as a batched policy-only player."""

    def __init__(self, path: str, device: str = "cpu"):
        self.path = str(path)
        self.device = torch.device(device)
        self.model, self.meta = mg.load_checkpoint(self.path, self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, games: Sequence, legal_indices: Optional[Sequence[Sequence[int]]] = None) -> List[PolicyOutput]:
        """Score positions (engine games or adapters) given their legal layout indices (computed when
        None); one forward for the batch."""
        if not games:
            return []
        if legal_indices is None:
            legal = [mg.legal_actions(g) for g in games]
        else:
            legal = [mg.legal_actions(g, idx) for g, idx in zip(games, legal_indices)]
        arrays = mg.build_inputs(list(games), legal, config=self.model.config)
        out = self.model(mg.to_device(arrays, self.device))
        logp = out["logp"].float().cpu().numpy()
        price_logp = out["price_logp"].float().cpu().numpy() if out["price_logp"] is not None else None
        win = torch.softmax(out["win_logits"].float(), dim=-1).cpu().numpy()
        offsets = arrays["legal_offsets"]
        per_price: List[Dict[int, np.ndarray]] = [dict() for _ in games]
        for j, row in enumerate(arrays["price_rows"]):
            b = int(arrays["legal_pos"][row])
            per_price[b][int(row - offsets[b])] = price_logp[j]
        return [
            PolicyOutput(
                legal=legal[b],
                probs=np.exp(logp[offsets[b] : offsets[b + 1]].astype(np.float64)),
                price_logp=per_price[b],
                win=win[b],
            )
            for b in range(len(games))
        ]

    def choose(
        self,
        game,
        output: PolicyOutput,
        temperature: float,
        rng: np.random.Generator,
        price_eps: float = 0.0,
    ) -> tuple:
        """``(layout index, price or None)`` sampled from ``output`` at ``temperature``."""
        return choose(game, output, temperature, rng, price_eps)


def choose(game, output: PolicyOutput, temperature: float, rng: np.random.Generator, price_eps: float = 0.0) -> tuple:
    """``(layout index, price or None)`` sampled from ``output`` at ``temperature`` (:meth:`GenericPolicy.choose`)."""
    p = _tempered(output.probs, temperature)
    k = int(rng.choice(len(p), p=p))
    index = int(output.legal.indices[k])
    return index, sample_price(game, output, k, rng, price_eps)


# ---------------------------------------------------------------------------------------------------
# Served by a generic inference server (``generic_inference``; eval_policy_only.py --generic-server)
# ---------------------------------------------------------------------------------------------------

_SERVED_CLIENTS: dict = {}


def served_client(name: str, client):
    """A ``GenericInferenceClient`` on the queues of a pool worker's server client (``policy_gradient.worker_init``
    connects every server with a legacy ``InferenceClient``; the generic one shares its worker slot)."""
    if name not in _SERVED_CLIENTS:
        from rl18xx.agent.alphazero.generic_inference import GenericInferenceClient

        _SERVED_CLIENTS[name] = GenericInferenceClient(
            request_q=client.request_q, reply_q=client.reply_q, worker_id=client.worker_id
        )
    return _SERVED_CLIENTS[name]


def served_request(games: Sequence, legal_indices: Sequence[Sequence[int]]) -> tuple:
    """``(arrays, legal)``: the inputs of a server request for positions with these legal indices."""
    legal = [mg.legal_actions(g, idx) for g, idx in zip(games, legal_indices)]
    return mg.build_inputs(list(games), legal), legal


def served_outputs(arrays: dict, legal: Sequence[mg.LegalActions], reply) -> List[PolicyOutput]:
    """:class:`PolicyOutput` per position from a server's ``GenericReply`` (asked ``want_price_logp``)."""
    offsets = arrays["legal_offsets"]
    per_price: List[Dict[int, np.ndarray]] = [dict() for _ in legal]
    for j, row in enumerate(arrays["price_rows"]):
        b = int(arrays["legal_pos"][row])
        per_price[b][int(row - offsets[b])] = reply.price_logp[j]
    return [
        PolicyOutput(
            legal=legal[b],
            probs=np.exp(reply.logp[offsets[b] : offsets[b + 1]].astype(np.float64)),
            price_logp=per_price[b],
            win=reply.win[b],
        )
        for b in range(len(legal))
    ]


def sample_price(game, output: PolicyOutput, k: int, rng: np.random.Generator, eps: float = 0.0) -> Optional[int]:
    """The price of legal row ``k``: None for a move without a price, the engine price for a fixed one
    (a depot train, ``lo == hi``), else a cell from the price head over the row's legal range (mixed with
    ``eps`` uniform over the range's non-empty cells) and a price inside it."""
    g = raw_game(game)
    index = int(output.legal.indices[k])
    price_range = g.price_range_for_index(index)
    if price_range is None:
        return None
    lo, hi = (int(x) for x in price_range)
    kind = int(output.legal.rows[k, mg.C_PRICE_KIND])
    if kind < 0 or lo == hi:
        return lo
    lo, hi, step = (int(x) for x in output.legal.price_range[k])
    cells = mg.price_cells_for(kind, lo, hi, step)
    live = cells.nonempty
    logits = output.price_logp.get(k)
    probs = np.where(live, np.exp(logits.astype(np.float64)), 0.0) if logits is not None else live.astype(np.float64)
    probs = probs / probs.sum() if probs.sum() > 0 else live / live.sum()
    mixed = np.where(live, (1.0 - eps) * probs + eps / live.sum(), 0.0)
    mixed = mixed / mixed.sum()
    cell = int(rng.choice(len(mixed), p=mixed))
    return int(cells.sample_price(cell, rng))


_CACHE: Dict[tuple, GenericPolicy] = {}


def cached_policy(path: str, device: str = "cpu") -> GenericPolicy:
    """One :class:`GenericPolicy` per (checkpoint, device) per process."""
    key = (str(path), str(device))
    if key not in _CACHE:
        _CACHE[key] = GenericPolicy(path, device)
    return _CACHE[key]


def is_generic_spec(spec: str) -> bool:
    return str(spec).startswith(GENERIC_PREFIX)


__all__ = [
    "GENERIC_PREFIX",
    "GenericPolicy",
    "PolicyOutput",
    "cached_policy",
    "choose",
    "is_generic_spec",
    "sample_price",
    "served_client",
    "served_outputs",
    "served_request",
]
