"""The model's view of a followed game: win estimates, recommended moves, the
moves actually played, and an optional search.

Three checkpoints, loaded once (:class:`AdvisorModels`):

- the **policy** recommends moves (default: the strongest policy-gradient
  snapshot measured so far);
- the **auction policy** does so during the private auction: the
  policy-gradient runs start every game after the waterfall auction, so their
  snapshots never trained on auction decisions (default: the supervised policy
  they started from);
- the **value** network estimates each player's chance of winning (default:
  value_v2, the best outcome predictor on held-out human games).

Probabilities are the policy's softmax at temperature 1 over the legal moves
(``scripts/eval_policy_only.py``'s inference path). A move with an open price
-- a bid, a private bought from its owner, a train bought from another
corporation -- gets its price from the price head (the most likely price cell,
no uniform mixing) and the next likeliest cells as alternatives. Win estimates
are the value net's per-seat win probabilities (``VALUE_SIZE`` = 6 seats, the
first ``num_players`` real) in seat order, normalised to sum to one.

"Think harder" (:meth:`Advisor.start_search`) runs the Rust MCTS
(``RustMCTSPlayer``, as ``scripts/eval_head_to_head.py`` does) from the
current position with the policy (auction policy in auction positions) for
priors and the value net at the leaves, one search at a time on a background
thread.

In a hotseat game the model can also play seats itself (:meth:`Advisor.start_move`,
``autoplay``): every decision is searched and the most visited move played, as the
head-to-head evals play (no sampling), with the *play* systems -- by default the
advice search's networks, or others (``main.py advisor --play-policy / --play-value``,
and ``--play-auction-policy / --play-auction-value`` for searches from the private
auction: most runs start after it, and searching it with their value nets bid $315 on
the B&O, while the supervised policy alone kept raising one private to $410).

The play systems may be title-generic checkpoints (``generic:<path>``, ``model_generic``):
their searches evaluate leaves through ``game_network.GameNetwork`` (the leaf games, not
legacy encodings), and ``--play-mode policy`` plays the play policy's own move without a
search (the most likely move and price cell, or sampled at ``--play-temperature``) --
generic or legacy. The advice, the win estimates and "Think harder" stay legacy.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from rl18xx.agent.advisor.autoplay import AutoplayError, bot_position, match_train_ids, plan_move, to_site_action
from rl18xx.agent.advisor.describe import DescribeContext, describe_action, describe_actor, map_positions

LOGGER = logging.getLogger(__name__)

TOP_MOVES = 5
PRICE_OPTIONS = 3
RECENT_MOVES = 5
DEFAULT_READOUTS = 200
MAX_READOUTS = 5000
# Search settings, from 240-game head-to-heads with gen4 and its critic (2026-10-16/17):
# value-driven PUCT (mean Q, unvisited moves at the parent's value, no first-play reduction),
# each distinct leaf backed up once, 8 leaves per batch, and priors at the policy's training
# temperature. With c_puct_init 0.15 @200 it beat gen4's own policy 0.933 (2v2) and won 0.787
# alone against three copies (1v3, even = 0.25); 0.05 then beat 0.15 (0.600) and 0.02 (0.562).
SEARCH_OVERRIDES = dict(
    mcts_mean_q=True, fpu_reduction=0.0, c_puct_init=0.05, c_puct_by_round={}, dedupe_leaves=True, parallel_readouts=8
)
# Autoplay keeps a game's search tree after the model's move (the chosen move's subtree) and goes on
# with it at the model's next move, as the training generator does: the tree follows the moves played
# in between (another of its seats, a human) down the children its search explored, and the root is
# topped up to the budget, inherited visits included (the head-to-head evals' ``budget=total``), so
# the model plays as those evals measured it, with fewer evaluations. A move the search never explored
# (the tree has nothing below it) starts a fresh tree. Trees kept, LRU:
KEPT_TREES = 8
# A played move the policy gave less than this is "surprising"; at least
# EXPECTED (or its top choice) "expected"; in between "plausible".
SURPRISING = 0.05
EXPECTED = 0.25
_AUCTION_ROUND_TYPE = 2  # encoder.ROUND_TYPE_MAP["Auction"]


def checkpoint_label(path) -> str:
    """A short name for a checkpoint: its last three path parts, without ``.pth`` (``generic:`` kept)."""
    from rl18xx.agent.alphazero.game_network import GENERIC_PREFIX, is_generic_spec, spec_path

    parts = Path(spec_path(path)).with_suffix("").parts
    return (GENERIC_PREFIX if is_generic_spec(path) else "") + "/".join(parts[-3:])


def load_model(path, device):
    """One checkpoint's model, in eval mode, built on ``device``."""
    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint

    path = Path(path)
    if not path.is_file():
        # Checkpoints are gitignored, so they exist only in the checkout that trained them.
        raise FileNotFoundError(
            f"No checkpoint at {path.resolve()}. Run from the checkout that holds the checkpoints "
            f"or pass --policy / --auction-policy / --value as absolute paths."
        )
    model = _load_model_from_session_checkpoint(path.parent, path, device=device)
    model.eval()
    return model


def load_network(spec, device):
    """A legacy checkpoint's model (:func:`load_model`), or for ``generic:<path>`` a title-generic checkpoint
    (``game_network.GenericNetwork``)."""
    from rl18xx.agent.alphazero.game_network import GenericNetwork, is_generic_spec, spec_path

    if not is_generic_spec(spec):
        return load_model(spec, device)
    path = Path(spec_path(spec))
    if not path.is_file():
        raise FileNotFoundError(f"No generic checkpoint at {path.resolve()}.")
    return GenericNetwork(path, device)


PLAY_MODES = ("search", "policy")


@dataclass
class AdvisorModels:
    policy: object
    auction_policy: object
    value: object
    labels: dict = field(default_factory=dict)  # "policy" / "auction_policy" / "value" -> checkpoint label
    # The temperature each policy was trained at (a policy-gradient learner *is*
    # softmax(logits / learner_temperature)); the search's priors use it.
    policy_temperature: float = 1.0
    auction_policy_temperature: float = 1.0
    # The search's leaf value (None: ``value``, which also gives the win estimates).
    search_value: object = None
    # The systems that play seats (autoplay), each a policy and its search's leaf value
    # (None: the advice ones above): one from the first stock round on, one for the
    # private auction -- which most runs never played (they start after it).
    play_policy: object = None
    play_value: object = None
    play_auction_policy: object = None
    play_auction_value: object = None
    play_policy_temperature: float = 1.0
    play_auction_policy_temperature: float = 1.0
    # Which output of the play value the play searches read (``composite_model.parse_value_head``):
    # "win" (the win head), "score" (a policy-gradient critic's placement-points head: a losing seat
    # still prefers 3rd to bankruptcy) or "blend=<w>". The advice search always reads "win".
    play_value_head: str = "win"
    # How the play systems choose a move: "search" (every decision searched, the most visited move) or
    # "policy" (the play policy's own move, no search: its most likely move and price cell at
    # ``play_temperature`` 0, else a move sampled from softmax(log p / T) and a price from its price head).
    play_mode: str = "search"
    play_temperature: float = 0.0

    def value_head_for(self, play: Optional[str] = None) -> str:
        """The value head a search reads: the play systems' ``play_value_head``, the advice search's win head."""
        return self.play_value_head if play in ("main", "auction") else "win"

    @classmethod
    def load(
        cls,
        policy,
        auction_policy,
        value,
        device,
        search_value=None,
        play_policy=None,
        play_value=None,
        play_auction_policy=None,
        play_auction_value=None,
        play_value_head: str = "win",
        play_mode: str = "search",
        play_temperature: float = 0.0,
        play_prior_temperature: Optional[float] = None,
    ) -> "AdvisorModels":
        """Load the checkpoints (a path given twice is loaded once). The play systems
        default to the advice search's: ``play_policy`` to ``policy``, ``play_value`` to
        the search's value, ``play_auction_policy`` to ``auction_policy`` and
        ``play_auction_value`` to ``play_value``. Only the play systems take generic
        checkpoints (``generic:<path>``). ``play_prior_temperature``: the temperature T the play
        policies are read at, softmax(logits / T) -- the play searches' priors and ``play_mode``
        "policy"'s probabilities (None: each checkpoint's training temperature, ``training_temperature``)."""
        from rl18xx.agent.alphazero.composite_model import parse_value_head
        from rl18xx.agent.alphazero.game_network import is_generic_spec, spec_path

        for name, spec in (("--policy", policy), ("--auction-policy", auction_policy), ("--value", value)):
            if is_generic_spec(spec):
                raise ValueError(f"{name}: generic checkpoints serve the play systems only (--play-*)")
        if search_value and is_generic_spec(search_value):
            raise ValueError("--search-value: generic checkpoints serve the play systems only (--play-*)")
        if play_mode not in PLAY_MODES:
            raise ValueError(f"play mode {play_mode!r}: one of {', '.join(PLAY_MODES)}")
        if float(play_temperature) < 0:
            raise ValueError("the play temperature must be >= 0")
        if play_prior_temperature is not None and not float(play_prior_temperature) > 0:
            raise ValueError("the play prior temperature must be > 0")
        parse_value_head(play_value_head)
        loaded: dict = {}

        def get(spec):
            key = ("generic:" if is_generic_spec(spec) else "") + str(Path(spec_path(spec)).resolve())
            if key not in loaded:
                LOGGER.info("Loading %s on %s", spec, device)
                loaded[key] = load_network(spec, device)
            return loaded[key]

        def temperature(spec) -> float:
            if play_prior_temperature is not None:
                return float(play_prior_temperature)
            return training_temperature(spec_path(spec))

        play_policy = play_policy or policy
        play_value = play_value or search_value or value
        play_auction_policy = play_auction_policy or auction_policy
        play_auction_value = play_auction_value or play_value
        return cls(
            policy=get(policy),
            auction_policy=get(auction_policy),
            value=get(value),
            labels={
                "policy": checkpoint_label(policy),
                "auction_policy": checkpoint_label(auction_policy),
                "value": checkpoint_label(value),
                "search_value": checkpoint_label(search_value or value),
                "play_policy": checkpoint_label(play_policy),
                "play_value": checkpoint_label(play_value),
                "play_auction_policy": checkpoint_label(play_auction_policy),
                "play_auction_value": checkpoint_label(play_auction_value),
                "play_value_head": play_value_head,
                "play_mode": play_mode if play_mode == "search" else f"policy@{float(play_temperature):g}",
                "play_prior_temperature": (
                    "training" if play_prior_temperature is None else f"{float(play_prior_temperature):g}"
                ),
            },
            search_value=get(search_value) if search_value else None,
            policy_temperature=training_temperature(policy),
            auction_policy_temperature=training_temperature(auction_policy),
            play_policy=get(play_policy),
            play_value=get(play_value),
            play_auction_policy=get(play_auction_policy),
            play_auction_value=get(play_auction_value),
            play_policy_temperature=temperature(play_policy),
            play_auction_policy_temperature=temperature(play_auction_policy),
            play_value_head=play_value_head,
            play_mode=play_mode,
            play_temperature=float(play_temperature),
        )

    def search_network(self, play: Optional[str] = None):
        """``(policy for priors, leaf value)`` of the advice search, or of a play system:
        ``play="main"`` (from the first stock round on) or ``"auction"`` (a search from a
        private-auction position: its whole tree, the stock round after included)."""
        if play == "auction" and self.play_auction_policy is not None:
            t = self.play_auction_policy_temperature
            policy = _RoundPolicy(self.play_auction_policy, self.play_auction_policy, t, t)
            return policy, self.play_auction_value or self.play_value or self.value
        if play == "main" and self.play_policy is not None:
            policy = _RoundPolicy(
                self.play_policy,
                self.play_auction_policy or self.play_policy,
                self.play_policy_temperature,
                self.play_auction_policy_temperature,
            )
            return policy, self.play_value or self.search_value or self.value
        policy = _RoundPolicy(
            self.policy, self.auction_policy, self.policy_temperature, self.auction_policy_temperature
        )
        return policy, self.search_value or self.value

    def _play_parts(self, play: str) -> tuple:
        """``(policy, auction policy, their temperatures, value)`` of a play system, as ``search_network``."""
        if play == "auction":
            t = self.play_auction_policy_temperature
            policy = self.play_auction_policy or self.play_policy or self.auction_policy
            return policy, policy, t, t, self.play_auction_value or self.play_value or self.value
        return (
            self.play_policy or self.policy,
            self.play_auction_policy or self.play_policy or self.auction_policy,
            self.play_policy_temperature,
            self.play_auction_policy_temperature,
            self.play_value or self.search_value or self.value,
        )

    def is_game_system(self, play: Optional[str]) -> bool:
        """Whether a play system (``"main"`` / ``"auction"``) has a generic part, so its searches evaluate
        leaves through ``game_network.GameNetwork``; the advice search (None) never does."""
        from rl18xx.agent.alphazero.game_network import GenericNetwork

        if play not in ("main", "auction"):
            return False
        policy, auction_policy, _, _, value = self._play_parts(play)
        return any(isinstance(m, GenericNetwork) for m in (policy, auction_policy, value))

    def game_network(self, play: str, lock=None):
        """A play system as a ``game_network.GameNetwork`` (generic or legacy parts): the policy (the auction
        policy for auction positions), the value read by ``play_value_head``."""
        from rl18xx.agent.alphazero.game_network import GameNetwork

        if play not in ("main", "auction"):
            raise ValueError(f"play system {play!r}: main or auction")
        policy, auction_policy, t, ta, value = self._play_parts(play)
        return GameNetwork(
            policy,
            value,
            auction_policy=auction_policy,
            policy_temperature=t,
            auction_policy_temperature=ta,
            value_head=self.value_head_for(play),
            lock=lock,
        )


def training_temperature(path) -> float:
    """The ``learner_temperature`` of the policy-gradient run a ``learner/<n>.pth``
    checkpoint comes from (its ``config.json``), else 1 (a supervised checkpoint)."""
    path = Path(path)
    config = path.parent.parent / "config.json"
    if path.parent.name != "learner" or not config.is_file():
        return 1.0
    try:
        return float(json.loads(config.read_text()).get("learner_temperature", 1.0))
    except (ValueError, OSError):
        return 1.0


class _RoundPolicy:
    """The policy for each position of a batch: the auction policy for auction
    positions, the policy for the rest (what the search reads its priors from)."""

    def __init__(self, policy, auction_policy, policy_temperature: float = 1.0, auction_policy_temperature: float = 1.0):
        self.policy = policy
        self.auction_policy = auction_policy
        # Priors are each policy's softmax(logits / its training temperature): p ** (1 / t)
        # here, and the search renormalizes over the legal moves.
        self.temperatures = {id(policy): policy_temperature, id(auction_policy): auction_policy_temperature}
        self.last_price_components = None

    @property
    def device(self):
        return self.policy.device

    def eval(self):
        return self

    def encoder_type(self):
        return self.policy.encoder_type()

    def get_name(self):
        return "round-policy"

    def run_many_encoded(self, encoded_game_states, *args, **kwargs):
        groups: dict = {}
        for i, state in enumerate(encoded_game_states):
            model = self.auction_policy if int(state[4]) == _AUCTION_ROUND_TYPE else self.policy
            groups.setdefault(id(model), (model, []))[1].append(i)
        n = len(encoded_game_states)
        probs, log_probs, values, prices = [None] * n, [None] * n, [None] * n, [None] * n
        components = None
        for model, rows in groups.values():
            p, lp, v = model.run_many_encoded([encoded_game_states[i] for i in rows])
            temperature = self.temperatures.get(id(model), 1.0)
            if temperature != 1.0:
                p = torch.pow(torch.clamp(p, min=0.0), 1.0 / temperature)
            pc = getattr(model, "last_price_components", None)
            for j, i in enumerate(rows):
                probs[i], values[i] = p[j], v[j]
                log_probs[i] = lp[j] if lp is not None else None
                if pc is not None:
                    prices[i] = pc["price_logits"][j]
                    components = pc
        if components is not None and all(row is not None for row in prices):
            self.last_price_components = {**components, "price_logits": torch.stack(prices)}
        else:
            self.last_price_components = None
        return torch.stack(probs), None, torch.stack(values)

    def run_encoded(self, encoded_game_state):
        probs, _, values = self.run_many_encoded([encoded_game_state])
        return probs[0], None, values[0]


class _Locked:
    """A network whose forward passes take the advisor's inference lock (the
    models keep per-call state, e.g. ``last_price_components``)."""

    def __init__(self, network, lock):
        self._network = network
        self._lock = lock
        self.last_price_components = None

    def __getattr__(self, name):
        return getattr(self._network, name)

    def run_many_encoded(self, *args, **kwargs):
        with self._lock:
            out = self._network.run_many_encoded(*args, **kwargs)
            self.last_price_components = getattr(self._network, "last_price_components", None)
        return out

    def run_encoded(self, encoded_game_state):
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0] if log_probs is not None else None, values[0]


@dataclass
class SearchJob:
    id: str
    game_id: object
    position: Optional[str]
    readouts: int
    status: str = "running"  # running / done / error
    visits: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    kind: str = "think"  # think ("Think harder") / move (the model playing a seat)

    def to_dict(self) -> dict:
        return {
            "job": self.id,
            "kind": self.kind,
            "game_id": self.game_id,
            "position": self.position,
            "readouts": self.readouts,
            "status": self.status,
            "progress": min(1.0, self.visits / max(1, self.readouts)),
            "visits": int(self.visits),
            "seconds": round((self.finished or time.time()) - self.started, 1),
            "result": self.result,
            "error": self.error,
        }


class SearchBusy(RuntimeError):
    def __init__(self, job: SearchJob):
        super().__init__("A search is already running")
        self.job = job


# ------------------------------------------------------------------ helpers
def has_price_choice(rs, index: int) -> bool:
    from rl18xx.agent.alphazero.policy_selfplay import has_price_choice as _has

    return _has(rs, index)


def is_decision(rs, legal: list) -> bool:
    """More than one legal move, or one whose price is still open."""
    return len(legal) > 1 or (len(legal) == 1 and has_price_choice(rs, legal[0]))


def price_options(rs, index: int, price_row: Optional[np.ndarray], top: int = PRICE_OPTIONS) -> Optional[dict]:
    """The price of a legal move: None for a move without one; ``fixed`` for a
    set price; else the price head's likeliest cells (``options``: ``price``,
    the cell's ``low``-``high`` prices and its ``probability``), the first
    being the recommendation (``price``)."""
    from rl18xx.agent.alphazero import price_pmf

    price_range = rs.price_range_for_index(int(index))
    if price_range is None:
        return None
    lo, hi = int(price_range[0]), int(price_range[1])
    slot = rs.price_head_slot_for_index(int(index))
    if lo == hi or slot is None:
        return {"fixed": True, "price": lo, "range": [lo, hi], "options": []}
    slot_index, lo, hi = int(slot[0]), int(slot[1]), int(slot[2])
    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot_index], lo, hi)
    probs = cells.cell_probs(price_row[slot_index] if price_row is not None else None)
    options = []
    for cell in np.argsort(-probs, kind="stable"):
        cell = int(cell)
        if probs[cell] <= 0 or len(options) >= top:
            break
        count = int(cells.counts[cell])
        low, high = cells.member(cell, 0), cells.member(cell, count - 1)
        options.append(
            {
                "price": int(cells.member(cell, count // 2)),
                "low": int(low),
                "high": int(high),
                "probability": float(probs[cell]),
            }
        )
    return {"fixed": False, "price": options[0]["price"], "range": [lo, hi], "options": options}


def price_probability(rs, index: int, price: int, price_row: Optional[np.ndarray]) -> Optional[float]:
    """The price head's probability for the cell of a played ``price``."""
    from rl18xx.agent.alphazero import price_pmf

    slot = rs.price_head_slot_for_index(int(index))
    if slot is None or price is None:
        return None
    slot_index, lo, hi = int(slot[0]), int(slot[1]), int(slot[2])
    if lo == hi or not lo <= int(price) <= hi:
        return None
    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot_index], lo, hi)
    probs = cells.cell_probs(price_row[slot_index] if price_row is not None else None)
    return float(probs[cells.cell_of(int(price))])


def _trade_in_index() -> int:
    from rl18xx.agent.alphazero.action_mapper import ActionMapper

    return ActionMapper().action_offsets["BuyTrainDTradeIn"]


def describe_index(game, index: int, price, ctx: DescribeContext, fixed_price: bool = False) -> dict:
    """``{type, actor, description, hex/tile/rotation, map}`` of a legal index,
    played at ``price`` (None for moves without one)."""
    action = dict(game._game.decode_index_to_map(int(index), None if price is None else int(price)))
    if fixed_price:
        action["fixed_price"] = True
    if int(index) == _trade_in_index():
        action["trade_in"] = True
    out = {
        "type": action.get("type"),
        "actor": describe_actor(action, ctx),
        "description": describe_action(action, ctx),
    }
    if action.get("type") in ("lay_tile", "place_token") and action.get("hex"):
        hex_id = action["hex"]
        out["hex"] = hex_id
        out["map"] = map_positions()["hexes"].get(hex_id)
        if action.get("type") == "lay_tile":
            out["tile"] = str(action.get("tile", "")).split("-")[0]
            out["rotation"] = int(action.get("rotation", 0))
    return out


def _normalised(values) -> list:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = values.sum()
    return (values / total if total > 0 else np.full(len(values), 1.0 / max(1, len(values)))).tolist()


# ------------------------------------------------------------------ advisor
class Advisor:
    """Turns a :class:`~rl18xx.agent.advisor.live_game.LiveGame` into advice (thread-safe)."""

    def __init__(self, models: AdvisorModels, top_moves: int = TOP_MOVES, recent_moves: int = RECENT_MOVES):
        self.models = models
        self.top_moves = top_moves
        self.recent_moves = recent_moves
        self._lock = threading.RLock()  # one forward pass at a time
        self._cache: OrderedDict = OrderedDict()  # (game id, position) -> advice
        self._jobs: OrderedDict = OrderedDict()
        self._jobs_lock = threading.Lock()
        self._running: Optional[SearchJob] = None
        self._moves: OrderedDict = OrderedDict()  # (game id, position, site's last action, seats, readouts) -> job
        self._trees: OrderedDict = OrderedDict()  # game id -> (root's position key, system, RustMCTSPlayer)
        self._trees_lock = threading.Lock()
        self._rng = np.random.default_rng()  # --play-mode policy at a temperature

    # -------------------------------------------------------------- inference
    def _policy_role(self, game) -> str:
        return "auction_policy" if game._game.round.round_type == "Auction" else "policy"

    def _run_policy(self, role: str, encoded: list) -> tuple:
        """``(probabilities (B, POLICY_SIZE), price logits (B, slots, cells) or None)``."""
        model = getattr(self.models, role)
        with self._lock, torch.no_grad():
            probs, _, _ = model.run_many_encoded(encoded)
            components = getattr(model, "last_price_components", None)
            prices = components["price_logits"].float().cpu().numpy() if components is not None else None
        return probs.float().cpu().numpy(), prices

    def _run_values(self, encoded: list) -> np.ndarray:
        from rl18xx.agent.alphazero.mcts import unrotate_value

        with self._lock, torch.no_grad():
            values = self.models.value.run_values_encoded(encoded).float().cpu().numpy()
        return np.stack([unrotate_value(v, int(e[6]), int(e[7]))[: int(e[7])] for v, e in zip(values, encoded)])

    def _analyse(self, games: list) -> list:
        """``(role, legal indices, move probabilities, price logits row)`` per position
        (batched by policy)."""
        from rl18xx.agent.alphazero.mcts import _rust_encode

        out: list = [None] * len(games)
        groups: dict = {}
        for i, game in enumerate(games):
            groups.setdefault(self._policy_role(game), []).append(i)
        for role, rows in groups.items():
            encoded = [_rust_encode(games[i]) for i in rows]
            probs, prices = self._run_policy(role, encoded)
            for j, i in enumerate(rows):
                legal = [int(x) for x in games[i]._game.factored_legal_indices()]
                p = probs[j][legal].astype(np.float64)
                p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / max(1, len(legal)))
                out[i] = (role, legal, p, prices[j] if prices is not None else None)
        return out

    # ------------------------------------------------------------------ advice
    def advise(self, live, server_acting=None) -> dict:
        """Advice for the game's current position (see ``app`` for the shape)."""
        key = (live.game_id, live.position_key, live.error is None)
        with self._lock:
            cached = self._cache.get(key)
        if cached is None:
            cached = self._advise(live)
            with self._lock:
                self._cache[key] = cached
                self._cache.move_to_end(key)
                while len(self._cache) > 32:
                    self._cache.popitem(last=False)
        advice = dict(cached)
        warnings = list(advice.get("warnings", []))
        mismatch = live.acting_mismatch(server_acting)
        if mismatch:
            warnings.append(mismatch)
        advice["warnings"] = warnings
        return advice

    def _advise(self, live) -> dict:
        game = live.rust
        rs = game._game
        warnings = []
        if live.untested:
            warnings.append(
                f"Untested: the model was trained on 4-player games; this one has {live.num_players} players."
            )
        if live.error:
            warnings.append(
                f"The engine could not follow action {live.error['action']} ({live.error['type']}: "
                f"{live.error['message']}), so no moves are suggested; the win estimate is for the position "
                "before it."
            )
        advice = {
            "supported": True,
            "reason": None,
            "game_id": live.game_id,
            "position": live.position_key,
            "players": [{"seat": i + 1, "id": p["id"], "name": p["name"]} for i, p in enumerate(live.players)],
            "untested_player_count": live.untested,
            "engine_error": live.error,
            "finished": bool(game.finished),
            "acting": live.acting(),
            "checkpoints": dict(self.models.labels),
            "warnings": warnings,
            "moves": [],
            "num_legal": 0,
            "recent": [],
        }
        if game.finished:
            result = game.result()
            worth = [float(result[pid]) for pid in sorted(result)]
            best = max(worth)
            share = _normalised([1.0 if w >= best - 1e-6 else 0.0 for w in worth])
            advice["win"] = {
                "final": True,
                "players": [
                    {"seat": i + 1, "name": live.names[i + 1], "probability": share[i], "net_worth": worth[i]}
                    for i in range(live.num_players)
                ],
            }
            return advice

        from rl18xx.agent.alphazero.mcts import _rust_encode

        values = self._run_values([_rust_encode(game)])[0]
        advice["win"] = {
            "final": False,
            "model": self.models.labels.get("value"),
            "raw_sum": float(np.sum(values)),
            "players": [
                {"seat": i + 1, "name": live.names[i + 1], "probability": p} for i, p in enumerate(_normalised(values))
            ],
        }
        ctx = DescribeContext.from_game(game, live.names)
        advice["round"] = {"type": ctx.round_type, "step": ctx.step}
        if not live.error:
            role, legal, probs, price_row = self._analyse([game])[0]
            advice["policy"] = {"role": role, "checkpoint": self.models.labels.get(role), "temperature": 1.0}
            advice["num_legal"] = len(legal)
            advice["forced"] = not is_decision(rs, legal)
            order = np.argsort(-probs, kind="stable")[: self.top_moves]
            for k in order:
                index = legal[int(k)]
                price = price_options(rs, index, price_row)
                entry = {"index": index, "probability": float(probs[int(k)]), "price": price}
                fixed = bool(price and price["fixed"])
                entry.update(describe_index(game, index, price["price"] if price else None, ctx, fixed))
                advice["moves"].append(entry)
        advice["recent"] = self._recent(live)
        return advice

    def _recent(self, live) -> list:
        """The model's probability for each of the last few decisions actually played."""
        from rl18xx.agent.alphazero.action_mapper import ActionMapper
        from rl18xx.agent.alphazero.pretraining import AmbiguousActionMatch, _action_dict_to_factored_index

        chosen = []
        for record in live.recent_records():
            if len(chosen) >= self.recent_moves:
                break
            if not record.applied or record.before.finished:
                continue
            if "decision" not in record.analysis:
                state = record.before
                legal = [int(x) for x in state._game.factored_legal_indices()]
                record.analysis["decision"] = is_decision(state._game, legal)
                if record.analysis["decision"]:
                    try:
                        index = _action_dict_to_factored_index(
                            record.action, state, state.get_factored_choices(), ActionMapper()
                        )
                    except AmbiguousActionMatch:
                        index = None
                    record.analysis["index"] = index
            if record.analysis["decision"]:
                chosen.append(record)
        pending = [r for r in chosen if "probability" not in r.analysis]
        for record, (role, legal, probs, price_row) in zip(pending, self._analyse([r.before for r in pending])):
            self._fill_recent(record, live, role, legal, probs, price_row)
        return [r.analysis["entry"] for r in chosen]

    def _fill_recent(self, record, live, role, legal, probs, price_row) -> None:
        state = record.before
        rs = state._game
        ctx = DescribeContext.from_game(state, live.names)
        index = record.analysis.get("index")
        action = record.action
        played_price = action.get("price")
        entry = {"action": record.index + 1, "type": action.get("type"), "num_legal": len(legal), "policy": role}
        if index is None or index not in legal:
            entry.update({"probability": None, "rank": None, "label": "unknown"})
            entry["actor"] = describe_actor({"entity": action.get("entity")}, ctx)
            entry["description"] = f"{action.get('type')} (not in the model's move list)"
        else:
            p = float(probs[legal.index(index)])
            rank = 1 + int(np.sum(probs > p + 1e-12))
            label = "expected" if rank == 1 or p >= EXPECTED else ("plausible" if p >= SURPRISING else "surprising")
            entry.update({"index": index, "probability": p, "rank": rank, "label": label})
            price_range = rs.price_range_for_index(index)
            price = None
            if price_range is not None:
                price = int(played_price) if played_price is not None else int(price_range[0])
                entry["price"] = price
                entry["price_probability"] = price_probability(rs, index, price, price_row)
            try:
                fixed = price_range is not None and int(price_range[0]) == int(price_range[1])
                entry.update(describe_index(state, index, price, ctx, fixed))
            except Exception:  # a price the decoder rejects: name the move without it
                entry.update(describe_index(state, index, None if price_range is None else int(price_range[0]), ctx))
        record.analysis["probability"] = entry.get("probability")
        record.analysis["entry"] = entry

    # ------------------------------------------------------------------ search
    def start_search(self, live, readouts: int = DEFAULT_READOUTS) -> SearchJob:
        """Start an MCTS search from the game's current position on a background
        thread; raises :class:`SearchBusy` while another one runs."""
        readouts = max(1, min(int(readouts), MAX_READOUTS))
        if live.finished or live.error:
            raise ValueError("There is no position to search: the game is over or the engine could not follow it")
        with self._jobs_lock:
            if self._running is not None and self._running.status == "running":
                raise SearchBusy(self._running)
            job = SearchJob(uuid.uuid4().hex[:12], live.game_id, live.position_key, readouts)
            self._running = job
            self._jobs[job.id] = job
            while len(self._jobs) > 20:
                self._jobs.popitem(last=False)
        game = live.rust.pickle_clone()
        names = dict(live.names)
        threading.Thread(target=self._search, args=(job, game, names), name="advisor-search", daemon=True).start()
        return job

    def job(self, job_id: str) -> Optional[SearchJob]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def running_search(self) -> Optional[SearchJob]:
        """The search running now (a "Think harder" or a seat's move), if any."""
        with self._jobs_lock:
            job = self._running
        return job if job is not None and job.status == "running" else None

    def _search(self, job: SearchJob, game, names: dict) -> None:
        try:
            job.result = self._run_search(job, game, names)
            job.status = "done"
        except Exception as e:  # reported to the page
            LOGGER.warning("search %s failed: %s", job.id, type(e).__name__)
            job.error = f"{type(e).__name__}: {e}"
            job.status = "error"
        finally:
            job.finished = time.time()

    def _grow_tree(self, job: SearchJob, game, play: Optional[str] = None, search: bool = True):
        """A ``RustMCTSPlayer`` at ``game`` with its root evaluated and, with ``search``,
        ``job.readouts`` readouts run (progress in ``job.visits``); the advice search's
        networks, or a play system's (``play``: see ``AdvisorModels.search_network``). The
        player picks the most visited move (no sampling), as the head-to-head evals play."""
        from rl18xx.agent.alphazero.composite_model import PolicyValueComposite
        from rl18xx.agent.alphazero.config import SelfPlayConfig
        from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer

        if self.models.is_game_system(play):
            # A generic part: the leaves go to the networks as games (game_network.GameNetwork).
            network = self.models.game_network(play, lock=self._lock)
        else:
            network = _Locked(
                PolicyValueComposite(
                    *self.models.search_network(play=play), value_head=self.models.value_head_for(play)
                ),
                self._lock,
            )
        num_players = len(game.players)
        config = SelfPlayConfig(
            network=network,
            num_readouts=job.readouts,
            min_readouts=job.readouts,
            dirichlet_noise_weight=0.0,
            enable_resign=False,
            noresign_holdout_rate=0.0,
            player_count_distribution={num_players: 1.0},
            auction_unlock=False,
            softpick_move_cutoff=0,
            **SEARCH_OVERRIDES,
        )
        player = RustMCTSPlayer(config)
        player.initialize_game(game)
        tree = player._rust_player
        # Evaluate the root first, as SelfPlay.play (and eval_head_to_head) do.
        if getattr(network, "evaluates_games", False) is True:
            player.tree_search(parallel_readouts=1)  # the root alone (as suggest_move's root evaluation)
        else:
            self._evaluate_root(player, network)
        while search and tree.n_at_root() < job.readouts:
            before = tree.n_at_root()
            player.tree_search()
            job.visits = tree.n_at_root()
            if job.visits <= before:  # nothing left to expand (e.g. every line ends the game)
                break
        return player

    @staticmethod
    def _evaluate_root(player, network) -> None:
        """The legacy root evaluation (a legacy network: the root's legacy encoding)."""
        from rl18xx.agent.alphazero.self_play import _slice_price_components

        with torch.no_grad():
            leaf = player.root.select_leaf()
            leaf.ensure_encoded()
            probs, _, value = network.run_encoded(leaf.encoded_game_state)
            components = _slice_price_components(network.last_price_components, 0)
            leaf.incorporate_results(probs, value, leaf, price_components=components)

    def _kept_tree(self, job: SearchJob, system: str, game):
        """The tree kept after the model's last move in ``job``'s game, moved along the moves played
        since then to ``game``'s position (``KEPT_TREES``), or None (dropped) where it can't follow."""
        from rl18xx.agent.advisor.live_game import action_key, position_key

        with self._trees_lock:
            kept = self._trees.pop(job.game_id, None)
        if kept is None or kept[1] != system:
            return None
        player = kept[2]
        tree = player._rust_player
        if kept[0] != job.position:
            actions = list(game.raw_actions)
            log = [action_key(a) for a in actions]
            followed = 0
            while True:
                root_log = [action_key(a) for a in tree.root_game_object().raw_actions]
                if len(root_log) >= len(log) or root_log != log[: len(root_log)]:
                    break
                move = self._explored_move(tree, actions[len(root_log)])
                if move is None:
                    LOGGER.info(
                        "game %s: the kept tree never explored %s; a fresh tree",
                        job.game_id,
                        action_key(actions[len(root_log)]),
                    )
                    return None
                try:
                    tree.advance_root_to(*move)
                except Exception:
                    return None
                followed += 1
            if position_key(tree.root_game_object().raw_actions) != job.position:
                return None
            LOGGER.info("game %s: the kept tree followed %d move(s) played since", job.game_id, followed)
        LOGGER.info("game %s: going on with the kept tree (%d visits)", job.game_id, tree.n_at_root())
        return player

    @staticmethod
    def _explored_move(tree, action: dict):
        """``(index, price)`` of the move the tree's search explored at its root that plays ``action``
        (an engine action dict), or None."""
        from rl18xx.agent.advisor.live_game import action_key
        from rl18xx.rust_adapter import RustGameAdapter

        wanted = action_key(action)
        visits = np.asarray(tree.child_n_at_root(), dtype=np.float64)
        root = RustGameAdapter(tree.root_game_object())
        for index in tree.legal_action_indices_at_root():
            index = int(index)
            if visits[index] <= 0:
                continue
            probe = root.pickle_clone()
            price_range = probe._game.price_range_for_index(index)
            price = int(action["price"]) if price_range is not None and action.get("price") is not None else None
            before = len(probe.raw_actions)
            try:
                probe._game.apply_action_index(index, price)
            except Exception:  # e.g. a price outside this move's range: not this move
                continue
            logged = list(probe.raw_actions)
            if len(logged) > before and action_key(logged[before]) == wanted:
                return index, price
        return None

    def _top_up(self, job: SearchJob, player) -> None:
        """Search a kept tree until its root has ``job.readouts`` visits, inherited ones included."""
        tree = player._rust_player
        job.visits = tree.n_at_root()
        while tree.n_at_root() < job.readouts:
            before = tree.n_at_root()
            player.tree_search()
            job.visits = tree.n_at_root()
            if job.visits <= before:
                break

    def _keep_tree(self, game_id, system: str, player, raw_actions: list) -> None:
        """Keep ``player``'s tree, advanced past the model's move, for its next move in the game."""
        from rl18xx.agent.advisor.live_game import position_key

        with self._trees_lock:
            self._trees[game_id] = (position_key(raw_actions), system, player)
            self._trees.move_to_end(game_id)
            while len(self._trees) > KEPT_TREES:
                self._trees.popitem(last=False)

    def _run_search(self, job: SearchJob, game, names: dict) -> dict:
        from rl18xx.rust_adapter import RustGameAdapter

        num_players = len(game.players)
        player = self._grow_tree(job, game)
        tree = player._rust_player
        legal = [int(i) for i in tree.legal_action_indices_at_root()]
        visits = np.asarray(tree.child_n_at_root(), dtype=np.float64)[legal]
        total = float(visits.sum()) or 1.0
        prices = tree.price_grandchildren_at_root()
        root = RustGameAdapter(tree.root_game_object())
        ctx = DescribeContext.from_game(root, names)
        moves = []
        for k in np.argsort(-visits, kind="stable")[: TOP_MOVES * 2]:
            index = legal[int(k)]
            if visits[int(k)] <= 0:
                break
            price_range = root._game.price_range_for_index(index)
            price = None
            if price_range is not None:
                by_price = prices.get(index) or {}
                price = int(max(by_price, key=lambda p: (by_price[p], p))) if by_price else int(price_range[0])
            entry = {"index": index, "visits": int(visits[int(k)]), "share": float(visits[int(k)] / total)}
            entry["price"] = price
            fixed = price_range is not None and int(price_range[0]) == int(price_range[1])
            entry.update(describe_index(root, index, price, ctx, fixed))
            moves.append(entry)
        q = [float(x) for x in tree.root_q_vector()][:num_players]
        return {
            "visits": int(tree.n_at_root()),
            "moves": moves,
            "win": [{"seat": i + 1, "name": names[i + 1], "probability": p} for i, p in enumerate(_normalised(q))],
        }

    # -------------------------------------------------------------- autoplay
    def start_move(self, live, site, seats, readouts: int = DEFAULT_READOUTS) -> SearchJob:
        """The model's move for one of ``seats`` in ``live``'s current position, given
        the site's view of it (``autoplay.SiteView``): a job whose result's ``action``
        is the move in the site's format. An implicit pass is done at once; a decision
        is searched with ``readouts`` on a background thread (one search at a time:
        :class:`SearchBusy` while another runs). Asking again for the same position
        returns the same job. Raises :class:`~rl18xx.agent.advisor.autoplay.AutoplayError`
        when the model has no move to make."""
        readouts = max(1, min(int(readouts), MAX_READOUTS))
        seats = sorted({int(s) for s in seats})
        plan = plan_move(live, site, seats)
        key = (
            live.game_id,
            live.position_key,
            site.last_id,
            site.entity_type,
            str(site.entity_id),
            tuple(seats),
            readouts,
        )
        with self._jobs_lock:
            existing = self._moves.get(key)
            if existing is not None and existing.status != "error":
                return existing
            if plan.kind == "search" and self._running is not None and self._running.status == "running":
                raise SearchBusy(self._running)
            job = SearchJob(uuid.uuid4().hex[:12], live.game_id, live.position_key, readouts, kind="move")
            self._jobs[job.id] = job
            self._moves[key] = job
            while len(self._jobs) > 20:
                self._jobs.popitem(last=False)
            while len(self._moves) > 20:
                self._moves.popitem(last=False)
            if plan.kind == "search":
                self._running = job
        base = {
            "seat": plan.seat,
            "player": live.names.get(plan.seat),
            "last_id": site.last_id,
            "position": live.position_key,
        }
        if plan.kind == "pass":
            acting = {"kind": site.entity_type, "id": site.entity_id}
            job.result = {
                **base,
                "kind": "implicit_pass",
                "action": plan.action,
                "actor": (
                    f"{site.entity_id} ({live.names.get(plan.seat)})"
                    if site.entity_type != "player"
                    else live.names.get(plan.seat)
                ),
                "description": "Pass",
                "reason": plan.reason,
                "acting": acting,
            }
            job.status = "done"
            job.finished = time.time()
            return job
        game = bot_position(live)  # the bots don't shuffle trains; the followed game keeps the real rules
        names = dict(live.names)
        threading.Thread(
            target=self._move, args=(job, game, names, live, base, site), name="advisor-move", daemon=True
        ).start()
        return job

    def _move(self, job: SearchJob, game, names: dict, live, base: dict, site=None) -> None:
        try:
            job.result = {**base, **self._run_move(job, game, names, live, site)}
            job.status = "done"
        except Exception as e:  # reported to the page
            LOGGER.warning("move %s failed: %s: %s", job.id, type(e).__name__, e)
            job.error = f"{type(e).__name__}: {e}"
            job.status = "error"
        finally:
            job.finished = time.time()

    def _run_move(self, job: SearchJob, game, names: dict, live, site=None) -> dict:
        from rl18xx.rust_adapter import RustGameAdapter

        rs = game._game
        legal = [int(i) for i in rs.factored_legal_indices()]
        system = "auction" if rs.round.round_type == "Auction" else "main"
        if self.models.play_mode == "policy":
            return self._run_policy_move(game, legal, system, names, live, site)
        searched = is_decision(rs, legal) and len(legal) > 1
        player = self._kept_tree(job, system, game) if searched else None
        inherited = 0
        if player is not None:
            inherited = int(player._rust_player.n_at_root())
            self._top_up(job, player)
        else:
            player = self._grow_tree(job, game, play=system, search=searched)
        tree = player._rust_player
        root = RustGameAdapter(tree.root_game_object())
        ctx = DescribeContext.from_game(root, names)
        index = player.pick_move() if searched else legal[0]
        visits = np.asarray(tree.child_n_at_root(), dtype=np.float64)
        total = float(visits[legal].sum()) or 1.0
        root_visits = int(tree.n_at_root())
        num_players = len(game.players)
        q = [float(x) for x in tree.root_q_vector()][:num_players]
        price_range = root._game.price_range_for_index(index)
        raw_before = len(root.raw_actions)
        player.play_move(index)  # the root's most visited price for an open-price move
        logged = list(tree.root_game_object().raw_actions)
        if len(logged) <= raw_before:
            raise RuntimeError(f"the engine logged no action for move {index}")
        if searched:
            self._keep_tree(job.game_id, system, player, logged)
        action = dict(logged[raw_before])
        price = action.get("price")
        fixed = price_range is not None and int(price_range[0]) == int(price_range[1])
        described = describe_index(root, index, None if price is None else int(price), ctx, fixed)
        out = {
            "kind": "search" if searched else ("priced" if price_range is not None and not fixed else "forced"),
            "system": system,
            # The site's ids for the trains it names (its engine may number two trains the other way).
            "action": match_train_ids(to_site_action(action, live, logged[:raw_before]), root, site),
            "index": index,
            "visits": root_visits if searched else 0,
            "inherited": inherited,  # of those, from the tree kept after the model's last move
            "share": float(visits[index] / total) if searched else 1.0,
            "num_legal": len(legal),
            "actor": described["actor"],
            "description": described["description"],
            "win": [{"seat": i + 1, "name": names[i + 1], "probability": p} for i, p in enumerate(_normalised(q))],
        }
        for key in ("hex", "tile", "rotation", "map"):
            if key in described:
                out[key] = described[key]
        return out

    def _run_policy_move(self, game, legal: list, system: str, names: dict, live, site=None) -> dict:
        """``--play-mode policy``: the play system's policy plays its own move, no search -- at
        ``play_temperature`` 0 its most likely move and, for an open price, the middle price of its price
        head's most likely cell (the advice's recommendation, :func:`price_options`); above 0 a move sampled
        from softmax(log p / T) (``policy_selfplay.tempered``, as the policy-only evals play) and a price
        sampled from the head's cells. ``game``: the bot's copy of the position (``bot_position``)."""
        from rl18xx.agent.alphazero import price_pmf
        from rl18xx.agent.alphazero.policy_selfplay import tempered

        rs = game._game
        network = self.models.game_network(system, lock=self._lock)
        evaluation = network.evaluate([game], [legal])[0]
        temperature = float(self.models.play_temperature)
        if len(legal) == 1:
            k = 0
        elif temperature <= 0:
            k = int(np.argmax(evaluation.probs))
        else:
            k = int(self._rng.choice(len(legal), p=tempered(evaluation.probs, temperature)))
        index = int(legal[k])
        price_range = rs.price_range_for_index(index)
        price = None
        open_price = has_price_choice(rs, index)
        if price_range is not None:
            price = int(price_range[0])
            if open_price and temperature <= 0:
                price = int(price_options(rs, index, evaluation.price_logits)["price"])
            elif open_price:
                slot_index, lo, hi = (int(x) for x in rs.price_head_slot_for_index(index))
                cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot_index], lo, hi)
                probs = cells.cell_probs(evaluation.price_logits[slot_index])
                price = int(cells.sample_price(int(self._rng.choice(len(probs), p=probs)), self._rng))
        fixed = price_range is not None and int(price_range[0]) == int(price_range[1])
        ctx = DescribeContext.from_game(game, names)
        described = describe_index(game, index, price, ctx, fixed)
        log = list(rs.raw_actions)
        after = game.pickle_clone()  # ``game`` stays the position before the move (match_train_ids reads it)
        after._game.apply_action_index(index, price)
        logged = list(after._game.raw_actions)
        if len(logged) <= len(log):
            raise RuntimeError(f"the engine logged no action for move {index}")
        action = dict(logged[len(log)])
        num_players = len(game.players)
        decision = len(legal) > 1
        out = {
            "kind": "policy" if decision else ("priced" if open_price else "forced"),
            "system": system,
            "action": match_train_ids(to_site_action(action, live, log), game, site),
            "index": index,
            "visits": 0,
            "inherited": 0,
            "share": float(evaluation.probs[k]),  # the policy's probability of the move (temperature 1)
            "temperature": temperature,
            "num_legal": len(legal),
            "actor": described["actor"],
            "description": described["description"],
            "win": [
                {"seat": i + 1, "name": names[i + 1], "probability": p}
                for i, p in enumerate(_normalised(evaluation.value[:num_players]))
            ],
        }
        for key in ("hex", "tile", "rotation", "map"):
            if key in described:
                out[key] = described[key]
        return out
