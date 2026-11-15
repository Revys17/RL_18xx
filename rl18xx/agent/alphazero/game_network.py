"""Policy, prices and values for 1830 positions given as games: generic checkpoints next to legacy ones.

The legacy search and the legacy players hand their networks the frozen 1830 encoding (``mcts._rust_encode``);
a title-generic checkpoint (``model_generic``) reads typed tokens (``rl_encode_batch``) and scores the legal
moves from their descriptors. :class:`GameNetwork` takes the positions themselves (``RustGameAdapter`` objects)
and their legal 1830 layout indices, and answers in the legacy search's terms, whatever each part is:

* a policy per round (the auction policy for private-auction positions, the policy for the rest), each a
  :class:`GenericNetwork` or a legacy model (``AlphaZeroTransformerModel``);
* a value, generic or legacy, read from its win head, its score head (placement points) or a blend
  (``composite_model.parse_value_head``);
* per position (:meth:`GameNetwork.evaluate`): the policy's probabilities over the legal moves (temperature 1),
  its price head as the legacy price head's rows (``price_pmf.SLOT_INDEX``: one row of ``NUM_CELLS`` cell logits
  per (action type, entity) slot) and the value per player in absolute seat order (players sorted by id, the
  Rust tree's order).

For 1830 a generic model's legal indices *are* the frozen layout's (``action_descriptors("1830")``), and the
price cells of a priced legal row are the legacy slot's cells over the same range (``price_pmf.price_cells``;
``tests/agent/advisor/test_generic_play.py`` checks both), so moves and prices carry over one to one. A generic
value is in the encoder's frame (seat 0 = the player to move, ``meta[5]`` the rotation over the seating order):
:func:`absolute_values` maps it to the players' ids.

The Rust tree (``RustMCTSPlayer.tree_search``) uses a network with ``evaluates_games`` set through
:meth:`GameNetwork.run_many_games`: it hands over the leaf games instead of legacy encodings and gets dense
priors (each policy's training temperature applied, as ``advisor._RoundPolicy`` does), the legacy price
components and absolute values back. Legacy networks never take that branch. Only the Python tree loop does:
the lockstep searcher (``lockstep.py``) encodes leaves natively and has no game-network path.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from rl18xx.agent.alphazero import price_pmf
from rl18xx.agent.alphazero.generic_policy import GENERIC_PREFIX

_AUCTION = "Auction"


def is_generic_spec(spec) -> bool:
    """Whether a checkpoint spec names a generic checkpoint (``generic:<path>``)."""
    return str(spec).startswith(GENERIC_PREFIX)


def spec_path(spec) -> str:
    """The checkpoint path of a spec (``generic:<path>`` or a legacy path)."""
    spec = str(spec)
    return spec[len(GENERIC_PREFIX) :] if is_generic_spec(spec) else spec


def _value_size() -> int:
    from rl18xx.agent.alphazero.mcts import VALUE_SIZE

    return VALUE_SIZE


def absolute_values(seat_values: np.ndarray, rotation: int, num_players: int, player_ids: Sequence) -> np.ndarray:
    """A generic per-seat vector (seat k = player ``(k + rotation) mod n`` of the seating order ``player_ids``)
    as ``VALUE_SIZE`` values in absolute order (players sorted by id; padding 0)."""
    out = np.zeros(_value_size(), dtype=np.float32)
    order = {pid: i for i, pid in enumerate(sorted(player_ids))}
    for k in range(num_players):
        out[order[player_ids[(k + rotation) % num_players]]] = seat_values[k]
    return out


@dataclass
class GenericOutput:
    """One position's generic forward: probabilities over its legal list (in the list's order), each priced
    legal row's cell log-probabilities, and the value heads per player in absolute order."""

    legal: np.ndarray  # int64 (n,) layout indices
    probs: Optional[np.ndarray]  # float64 (n,), None for a value-only forward
    price_logp: dict  # legal index -> float32 (NUM_CELLS,)
    win: np.ndarray  # float32 (VALUE_SIZE,)
    score: np.ndarray  # float32 (VALUE_SIZE,)


class GenericNetwork:
    """A generic checkpoint (``model_generic.save_checkpoint``) loaded for serving: fp32, eval mode, no grad."""

    def __init__(self, path, device="cpu", dedupe_maps: bool = True):
        from rl18xx.agent.alphazero import model_generic as mg

        self.path = str(path)
        self.device = torch.device(device)
        # The map encoder runs once per distinct map of a batch (a search's leaves mostly share theirs).
        self.dedupe_maps = bool(dedupe_maps)
        self.model, self.meta = mg.load_checkpoint(self.path, self.device)
        self.model.eval()
        if "1830" not in (self.meta.get("titles") or []):
            raise mg.CheckpointMismatch(f"{self.path}: not trained on 1830 (titles {self.meta.get('titles')})")

    def __repr__(self) -> str:
        return f"GenericNetwork({self.path!r})"

    @torch.no_grad()
    def forward(self, games: Sequence, legal_indices: Optional[Sequence[Sequence[int]]] = None) -> List[GenericOutput]:
        """One forward for ``games`` (adapters); with ``legal_indices`` the policy and price heads too, else
        value-only."""
        from rl18xx.agent.alphazero import model_generic as mg

        games = list(games)
        legal = None if legal_indices is None else [mg.legal_actions(g, idx) for g, idx in zip(games, legal_indices)]
        arrays = mg.build_inputs(games, legal, dedupe_maps=self.dedupe_maps, config=self.model.config)
        out = self.model(mg.to_device(arrays, self.device), value_only=legal is None)
        win = torch.softmax(out["win_logits"].float(), dim=-1).cpu().numpy()
        score = out["score"].float().cpu().numpy()
        meta = arrays["meta"]
        results = []
        for b, game in enumerate(games):
            ids = [p.id for p in game.players]
            rotation, n = int(meta[b, 5]), int(meta[b, 4])
            results.append(
                GenericOutput(
                    legal=np.asarray(legal[b].indices if legal else [], dtype=np.int64),
                    probs=None,
                    price_logp={},
                    win=absolute_values(win[b], rotation, n, ids),
                    score=absolute_values(score[b], rotation, n, ids),
                )
            )
        if legal is None:
            return results
        logp = out["logp"].float().cpu().numpy().astype(np.float64)
        offsets = arrays["legal_offsets"]
        for b, result in enumerate(results):
            p = np.exp(logp[offsets[b] : offsets[b + 1]])
            result.probs = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / max(1, len(p)))
        if out["price_logp"] is not None:
            price_logp = out["price_logp"].float().cpu().numpy()
            for j, row in enumerate(arrays["price_rows"]):
                b = int(arrays["legal_pos"][row])
                index = int(legal[b].indices[int(row - offsets[b])])
                results[b].price_logp[index] = price_logp[j]
        return results


@dataclass
class Evaluation:
    """One position as :class:`GameNetwork` sees it."""

    legal: np.ndarray  # int64 (n,) legal layout indices
    probs: np.ndarray  # float64 (n,) the policy's probabilities over them, temperature 1
    price_logits: np.ndarray  # float32 (NUM_SLOTS, NUM_CELLS): the legacy price head's rows (zeros: no output)
    value: np.ndarray  # float32 (VALUE_SIZE,) the value head's output per player, absolute order
    temperature: float  # the policy's training temperature (the search's priors are softmax(logits / it))
    role: str  # "auction_policy" / "policy": which policy scored the position


def _rust_encode(game):
    from rl18xx.agent.alphazero.mcts import _rust_encode as encode

    return encode(game)


class GameNetwork:
    """A policy per round (generic or legacy) and a value (generic or legacy) for positions given as games
    (module docstring). ``lock``: taken around the forwards (the advisor's inference lock)."""

    evaluates_games = True

    def __init__(
        self,
        policy,
        value,
        auction_policy=None,
        policy_temperature: float = 1.0,
        auction_policy_temperature: float = 1.0,
        value_head: str = "win",
        lock=None,
    ):
        from rl18xx.agent.alphazero.composite_model import parse_value_head

        self.policy = policy
        self.auction_policy = auction_policy if auction_policy is not None else policy
        self.value = value
        self.policy_temperature = float(policy_temperature)
        self.auction_policy_temperature = float(auction_policy_temperature)
        self.kind, self.blend = parse_value_head(value_head)
        self.value_head = value_head
        self._lock = lock if lock is not None else contextlib.nullcontext()
        self.last_price_components = None

    def eval(self):
        return self

    def get_name(self) -> str:
        return "game-network"

    # ------------------------------------------------------------------ parts
    def _role(self, game) -> str:
        return "auction_policy" if game._game.round.round_type == _AUCTION else "policy"

    def _head(self, win: np.ndarray, score: np.ndarray) -> np.ndarray:
        if self.kind == "win":
            return win
        if self.kind == "score":
            return score
        return ((1.0 - self.blend) * win + self.blend * score).astype(np.float32)

    def _legacy_values(self, model, encoded: list) -> np.ndarray:
        from rl18xx.agent.alphazero.mcts import unrotate_value

        if self.kind == "win":
            values = model.run_values_encoded(encoded)
        elif self.kind == "score":
            values = model.run_scores_encoded(encoded)
        else:
            win, score = model.run_values_and_scores_encoded(encoded)
            values = (1.0 - self.blend) * win + self.blend * score
        values = values.float().cpu().numpy()
        return np.stack([unrotate_value(v, int(e[6]), int(e[7])) for v, e in zip(values, encoded)]).astype(np.float32)

    @staticmethod
    def _slot_rows(game, price_logp: dict) -> np.ndarray:
        """A generic position's price outputs as the legacy head's slot rows (``price_head_slot_for_index``)."""
        rows = np.zeros((price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS), dtype=np.float32)
        for index, logp in price_logp.items():
            slot = game._game.price_head_slot_for_index(int(index))
            if slot is not None:
                rows[int(slot[0])] = logp
        return rows

    # -------------------------------------------------------------- evaluate
    def evaluate(self, games: Sequence, legal_indices: Optional[Sequence[Sequence[int]]] = None) -> List[Evaluation]:
        """Each position's :class:`Evaluation` (one forward per network; a generic policy that is also the value
        gives both from one forward)."""
        games = list(games)
        if legal_indices is None:
            legal_indices = [[int(i) for i in g._game.factored_legal_indices()] for g in games]
        legal_indices = [np.asarray(list(idx), dtype=np.int64) for idx in legal_indices]
        n = len(games)
        probs, prices, values, roles = [None] * n, [None] * n, [None] * n, [None] * n
        encoded: dict = {}

        def enc(i):
            if i not in encoded:
                encoded[i] = _rust_encode(games[i])
            return encoded[i]

        groups: dict = {}
        for i, game in enumerate(games):
            roles[i] = self._role(game)
            model = self.auction_policy if roles[i] == "auction_policy" else self.policy
            groups.setdefault(id(model), (model, []))[1].append(i)
        with self._lock, torch.no_grad():
            for model, rows in groups.values():
                if isinstance(model, GenericNetwork):
                    outs = model.forward([games[i] for i in rows], [legal_indices[i] for i in rows])
                    for i, out in zip(rows, outs):
                        probs[i] = out.probs
                        prices[i] = self._slot_rows(games[i], out.price_logp)
                        if model is self.value:
                            values[i] = self._head(out.win, out.score)
                    continue
                p, _, _ = model.run_many_encoded([enc(i) for i in rows])
                p = p.float().cpu().numpy()
                components = getattr(model, "last_price_components", None)
                logits = components["price_logits"].float().cpu().numpy() if components is not None else None
                for j, i in enumerate(rows):
                    q = p[j][legal_indices[i]].astype(np.float64)
                    probs[i] = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / max(1, len(q)))
                    prices[i] = (
                        logits[j].astype(np.float32)
                        if logits is not None
                        else np.zeros((price_pmf.NUM_SLOTS, price_pmf.NUM_CELLS), dtype=np.float32)
                    )
            rest = [i for i in range(n) if values[i] is None]
            if rest:
                if isinstance(self.value, GenericNetwork):
                    for i, out in zip(rest, self.value.forward([games[i] for i in rest])):
                        values[i] = self._head(out.win, out.score)
                else:
                    for i, v in zip(rest, self._legacy_values(self.value, [enc(i) for i in rest])):
                        values[i] = v
        return [
            Evaluation(
                legal=legal_indices[i],
                probs=probs[i],
                price_logits=prices[i],
                value=np.asarray(values[i], dtype=np.float32),
                temperature=(
                    self.auction_policy_temperature if roles[i] == "auction_policy" else self.policy_temperature
                ),
                role=roles[i],
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------ the search
    def run_many_games(self, games: Sequence, legal_indices: Sequence[Sequence[int]]):
        """The Rust tree's leaf evaluation (``RustMCTSPlayer.tree_search``): ``(priors (B, POLICY_SIZE) float32,
        None, values (B, VALUE_SIZE) float32 in absolute order)``, the priors each policy's probabilities at its
        training temperature (the tree renormalizes over the legal moves) and ``last_price_components`` the
        legacy price head's rows."""
        from rl18xx.agent.alphazero.mcts import POLICY_SIZE
        from rl18xx.agent.alphazero.policy_selfplay import tempered

        evaluations = self.evaluate(games, legal_indices)
        priors = np.zeros((len(evaluations), POLICY_SIZE), dtype=np.float32)
        for b, e in enumerate(evaluations):
            if len(e.legal):
                priors[b, e.legal] = tempered(e.probs, e.temperature)
        values = np.stack([e.value for e in evaluations]).astype(np.float32)
        self.last_price_components = {
            "price_logits": torch.from_numpy(np.stack([e.price_logits for e in evaluations])),
            "slot_index": dict(price_pmf.SLOT_INDEX),
            "num_slots": price_pmf.NUM_SLOTS,
        }
        return priors, None, values


__all__ = [
    "Evaluation",
    "GameNetwork",
    "GenericNetwork",
    "GenericOutput",
    "absolute_values",
    "is_generic_spec",
    "spec_path",
]
