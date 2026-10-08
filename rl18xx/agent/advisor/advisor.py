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
"""

from __future__ import annotations

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

from rl18xx.agent.advisor.describe import DescribeContext, describe_action, describe_actor, map_positions

LOGGER = logging.getLogger(__name__)

TOP_MOVES = 5
PRICE_OPTIONS = 3
RECENT_MOVES = 5
DEFAULT_READOUTS = 200
MAX_READOUTS = 5000
# A played move the policy gave less than this is "surprising"; at least
# EXPECTED (or its top choice) "expected"; in between "plausible".
SURPRISING = 0.05
EXPECTED = 0.25
_AUCTION_ROUND_TYPE = 2  # encoder.ROUND_TYPE_MAP["Auction"]


def checkpoint_label(path) -> str:
    """A short name for a checkpoint: its last three path parts, without ``.pth``."""
    parts = Path(path).with_suffix("").parts
    return "/".join(parts[-3:])


def load_model(path, device):
    """One checkpoint's model, in eval mode, built on ``device``."""
    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    model = _load_model_from_session_checkpoint(path.parent, path, device=device)
    model.eval()
    return model


@dataclass
class AdvisorModels:
    policy: object
    auction_policy: object
    value: object
    labels: dict = field(default_factory=dict)  # "policy" / "auction_policy" / "value" -> checkpoint label

    @classmethod
    def load(cls, policy, auction_policy, value, device) -> "AdvisorModels":
        """Load the three checkpoints (a path given twice is loaded once)."""
        loaded: dict = {}

        def get(path):
            key = str(Path(path).resolve())
            if key not in loaded:
                LOGGER.info("Loading %s on %s", path, device)
                loaded[key] = load_model(path, device)
            return loaded[key]

        return cls(
            policy=get(policy),
            auction_policy=get(auction_policy),
            value=get(value),
            labels={
                "policy": checkpoint_label(policy),
                "auction_policy": checkpoint_label(auction_policy),
                "value": checkpoint_label(value),
            },
        )


class _RoundPolicy:
    """The policy for each position of a batch: the auction policy for auction
    positions, the policy for the rest (what the search reads its priors from)."""

    def __init__(self, policy, auction_policy):
        self.policy = policy
        self.auction_policy = auction_policy
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

    def to_dict(self) -> dict:
        return {
            "job": self.id,
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

    def _run_search(self, job: SearchJob, game, names: dict) -> dict:
        from rl18xx.agent.alphazero.composite_model import PolicyValueComposite
        from rl18xx.agent.alphazero.config import SelfPlayConfig
        from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
        from rl18xx.agent.alphazero.self_play import _slice_price_components
        from rl18xx.rust_adapter import RustGameAdapter

        network = _Locked(
            PolicyValueComposite(_RoundPolicy(self.models.policy, self.models.auction_policy), self.models.value),
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
        )
        player = RustMCTSPlayer(config)
        player.initialize_game(game)
        tree = player._rust_player
        # Evaluate the root first, as SelfPlay.play (and eval_head_to_head) do.
        with torch.no_grad():
            leaf = player.root.select_leaf()
            leaf.ensure_encoded()
            probs, _, value = network.run_encoded(leaf.encoded_game_state)
            components = _slice_price_components(network.last_price_components, 0)
            leaf.incorporate_results(probs, value, leaf, price_components=components)
        while tree.n_at_root() < job.readouts:
            before = tree.n_at_root()
            player.tree_search()
            job.visits = tree.n_at_root()
            if job.visits <= before:  # nothing left to expand (e.g. every line ends the game)
                break

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
