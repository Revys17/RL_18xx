"""Self-play start positions: games that begin at the first Stock Round.

Self-play's opening waterfall auction is pathological — temperature-1 players
lock it (every player's cash committed to bids, nobody able to afford the next
private) or pass forever, which is why self-play needed the ``auction_unlock``
rule variant and the ``auction_stall_moves`` backstop. With
``SelfPlayHyperparams.start_positions_path`` set, a self-play game skips the
auction and starts at Stock Round 1 from either

* a **human start** — the real post-auction position of a recorded game: its
  action prefix up to the first Stock Round, written one game per line to a
  JSONL file by ``scripts/build_start_positions.py``; or
* a **random start** (``random_start_fraction`` of the games, and every game
  whose player count has no human starts) — a legal auction realizing a random
  allocation of the privates (:func:`random_start_actions`).

A start is a list of action dicts applied to a fresh game, so both engines and
the game's action log stay valid, and ``RustMCTSPlayer.extract_data`` rebuilds
the positions by replaying it on an empty game. The auction is real 1830 in
both cases: start games are built with ``auction_unlock`` off.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from rl18xx.rust_adapter import RustGameAdapter

LOGGER = logging.getLogger(__name__)

HUMAN_START_PREFIX = "human:"
RANDOM_START = "random"

# What ``process_action`` reads from an auction action (bid / pass / par).
# Everything else in a recorded action — ``user``, ``created_at``, ``id`` — is
# dropped from start positions.
ACTION_KEYS = ("type", "entity", "entity_type", "company", "price", "corporation", "share_price")
START_ACTION_TYPES = ("bid", "pass", "par")


@dataclass(frozen=True)
class StartPosition:
    """Actions that take a fresh game to its first Stock Round."""

    label: str  # "human:<game id>" or "random"
    actions: tuple


def new_game(num_players: int, auction_unlock: bool = False) -> RustGameAdapter:
    """An empty Rust game with players "Player 1".."Player N" (self-play's naming)."""
    from engine_rs import BaseGame as RustGame

    game = RustGame({i + 1: f"Player {i + 1}" for i in range(num_players)})
    game.set_auction_unlock(auction_unlock)
    return RustGameAdapter(game)


def in_auction(game: RustGameAdapter) -> bool:
    return game._game.round.round_type == "Auction"


def apply_actions(game, actions: Iterable[dict]):
    """Apply action dicts to ``game`` (copies, so the engine can't alias them)."""
    for action in actions:
        game.process_action(dict(action))
    return game


def clean_action(action: dict) -> dict:
    """Only the keys ``process_action`` needs for an auction action."""
    return {key: action[key] for key in ACTION_KEYS if key in action}


def position_summary(game) -> dict:
    """Comparable summary of a post-auction position. Works on a ``RustGameAdapter``
    and on a Python ``BaseGame`` alike (for parity checks)."""
    bo = game.corporation_by_id("B&O")
    return {
        "round": game.round.__class__.__name__,
        "cash": {p.id: p.cash for p in game.players},
        "owners": {c.sym: getattr(c.owner, "id", None) for c in game.companies},
        "bo_par": bo.share_price.price if bo.share_price else None,
        "bo_president": getattr(bo.owner, "id", None),
        "priority": game.priority_deal_player().id,
        "current": game.current_entity.id,
        "bank": game.bank.cash,
    }


def python_game(num_players: int, actions: Iterable[dict]):
    """The Python reference engine after ``actions`` (``auction_unlock`` off)."""
    from rl18xx.game.engine.actions import BaseAction
    from rl18xx.game.gamemap import GameMap

    game = GameMap().game_by_title("1830")({i + 1: f"Player {i + 1}" for i in range(num_players)})
    for action in actions:
        game.process_action(BaseAction.action_from_dict(dict(action), game))
    return game


# ----------------------------------------------------------------- human starts
def human_start_prefix(game: dict) -> tuple[Optional[list[dict]], Optional[str]]:
    """A cleaned human game's actions up to its first Stock Round.

    ``game`` is a cleaned export as ``pretraining.load_games_from_json`` loads
    it (players "Player 1".."Player N" with ids 1..N). It is replayed through
    the Rust engine with ``auction_unlock`` off and the pretraining importer's
    leniency: a pass the engine rejects is dropped
    (``pretraining._process_pass_leniently``); any other action must apply.
    Returns ``(prefix, None)`` — the applied actions, stripped to
    :data:`ACTION_KEYS` — or ``(None, reason)``.
    """
    from rl18xx.agent.alphazero.pretraining import _process_pass_leniently

    players = game.get("players") or []
    ids = [p.get("id") if isinstance(p, dict) else None for p in players]
    if not 2 <= len(players) <= 6 or ids != list(range(1, len(players) + 1)):
        return None, "player_ids"
    state = new_game(len(players))
    prefix = []
    for raw in game.get("actions") or []:
        if not in_auction(state):
            break
        action = clean_action(raw)
        if action.get("type") not in START_ACTION_TYPES:
            return None, f"action_type_{action.get('type')}"
        if action["type"] == "pass":
            if _process_pass_leniently(state, dict(action), use_rust=True):
                prefix.append(action)
            continue
        try:
            state.process_action(dict(action))
        except Exception as e:
            LOGGER.debug("Game %s: engine rejected %s: %s", game.get("id"), action, e)
            return None, "engine_error"
        prefix.append(action)
    if in_auction(state):
        return None, "auction_not_finished"
    return prefix, None


_LOADED: dict = {}


def load_start_positions(path) -> dict:
    """``{num_players: [StartPosition, ...]}`` from a start-positions JSONL file
    (one ``{"id", "num_players", "actions"}`` object per line). Read once per
    process and cached."""
    key = str(Path(path).resolve())
    if key not in _LOADED:
        by_players: dict = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                start = StartPosition(f"{HUMAN_START_PREFIX}{record['id']}", tuple(record["actions"]))
                by_players.setdefault(int(record["num_players"]), []).append(start)
        _LOADED[key] = by_players
        LOGGER.info(
            "Loaded start positions from %s: %s",
            path,
            {n: len(starts) for n, starts in sorted(by_players.items())},
        )
    return _LOADED[key]


# ---------------------------------------------------------------- random starts
def _bid(player: int, company: str, price: int) -> dict:
    return {"type": "bid", "entity": player, "entity_type": "player", "company": company, "price": int(price)}


def _pass(player: int) -> dict:
    return {"type": "pass", "entity": player, "entity_type": "player"}


def random_allocation(
    companies: list[tuple[str, int]],
    cash: dict,
    rng: random.Random,
    max_price_multiple: float = 2.0,
    max_tries: int = 10_000,
    sv_discount_fraction: float = 0.25,
) -> dict:
    """``{company: (player id, price)}``: each private to a uniformly random
    player at a uniform multiple of $5 in ``[face, max_price_multiple * face]``,
    resampled until every player can pay for theirs out of ``cash``.

    ``companies`` is the auction's ``(sym, face value)`` list, cheapest first.
    The cheapest private (the SV) can't be bid on, only bought at its price,
    which an all-pass round lowers by $5: with probability
    ``sv_discount_fraction`` it goes $5-$20 below face (uniformly; at $0 the
    engine hands it to the next player, so its owner is ``None`` here and is
    filled in by :func:`_realize`), otherwise at face.
    """
    players = sorted(cash)
    for _ in range(max_tries):
        allocation = {}
        spend = dict.fromkeys(players, 0)
        for i, (sym, face) in enumerate(companies):
            assert face % 5 == 0, f"{sym} face value {face} is not a multiple of $5"
            top = max(face, int(face * max_price_multiple) // 5 * 5)
            if i == 0:
                discounted = rng.random() < sv_discount_fraction
                price = face - 5 * rng.randint(1, face // 5) if discounted else face
            else:
                price = rng.randrange(face, top + 5, 5)
            owner = rng.choice(players) if price > 0 else None
            allocation[sym] = (owner, price)
            if owner is not None:
                spend[owner] += price
        if all(spend[p] <= cash[p] for p in players):
            return allocation
    raise RuntimeError(f"no affordable allocation of {companies} in {max_tries} tries (cash {cash})")


def _realize(game: RustGameAdapter, allocation: dict, faces: dict, rng: random.Random) -> list[dict]:
    """Legal actions that end ``game``'s auction with exactly ``allocation``.

    Every owner places a bid of exactly its price on each of its privates
    priced above face (one bid per turn; nobody else bids; the rest pass).
    Then the cheapest private, which has no bid, is bought by its owner on
    their turn while the others pass, and each purchase resolves the
    sole-bidder privates behind it. The B&O owner pars at a uniformly random
    legal price. An active action (bid or buy) comes at least once every N
    turns, so those passes never complete an all-pass round.

    A discounted SV takes one all-pass round per $5 below face, played after a
    random number of the bids: who acts last before the passes decides who
    gets the first chance at the discounted SV -- or, at $0, who the engine
    hands it to (recorded into ``allocation``).
    """
    rs = game._game
    actions = []

    def play(action):
        game.process_action(dict(action))
        actions.append(action)

    sv = rs.auction_companies()[0]
    pending = {p.id: [] for p in rs.players}
    for sym, (owner, price) in allocation.items():
        if owner is not None and price > faces[sym]:
            pending[owner].append((sym, price))
    discount_rounds = (faces[sv] - allocation[sv][1]) // 5
    if discount_rounds:
        # Someone with a bid still to place is reached every N turns, so these
        # passes can't complete an all-pass round early.
        early_bids = rng.randint(0, sum(len(bids) for bids in pending.values()))
        while early_bids:
            player = rs.current_player.id
            if pending[player]:
                play(_bid(player, *pending[player].pop()))
                early_bids -= 1
            else:
                play(_pass(player))
        for _ in range(discount_rounds):
            for _ in rs.players:
                play(_pass(rs.current_player.id))
        if allocation[sv][0] is None:
            allocation[sv] = (position_summary(game)["owners"][sv], 0)
    for _ in range(50 * len(allocation) * len(pending)):
        if not in_auction(game):
            return actions
        pending_par = rs.auction_pending_par()
        if pending_par:
            corporation, player = pending_par
            prices = [c["params"]["par_price"] for c in rs.get_factored_choices() if c["type"] == "Par"]
            price = rng.choice(sorted(prices))
            row, col = next((r, c) for p, r, c in rs.par_prices_with_coords() if p == price)
            action = {
                "type": "par",
                "entity": int(player),
                "entity_type": "player",
                "corporation": corporation,
                "share_price": f"{price},{row},{col}",
            }
        else:
            player = rs.current_player.id
            cheapest = rs.auction_companies()[0]
            if any(pending.values()):
                action = _bid(player, *pending[player].pop()) if pending[player] else _pass(player)
            elif allocation[cheapest][0] == player:
                action = _bid(player, cheapest, rs.auction_min_bid(cheapest))
            else:
                action = _pass(player)
        play(action)
    raise RuntimeError(f"auction for {allocation} did not end; actions so far: {actions}")


def random_start_actions(
    num_players: int,
    rng: Optional[random.Random] = None,
    max_price_multiple: float = 2.0,
    sv_discount_fraction: float = 0.25,
) -> list[dict]:
    """Actions that end the auction with a random allocation of the privates
    (:func:`random_allocation`) and a random B&O par, leaving the game at its
    first Stock Round. The result is checked against the sample."""
    rng = rng or random.Random()
    game = new_game(num_players)
    rs = game._game
    faces = {sym: rs.company_by_id(sym).value for sym in rs.auction_companies()}
    cash = {p.id: p.cash for p in rs.players}
    allocation = random_allocation(
        list(faces.items()), cash, rng, max_price_multiple, sv_discount_fraction=sv_discount_fraction
    )
    actions = _realize(game, allocation, faces, rng)

    # The engine's position must match the sample exactly.
    summary = position_summary(game)
    assert summary["round"] == "Stock", summary
    bought = {}
    for action in actions:
        if action["type"] == "bid":
            assert action["company"] not in bought, f"two bids on {action['company']}: {actions}"
            bought[action["company"]] = (action["entity"], action["price"])
    # A $0 SV is handed over by the engine, not bought.
    assert bought == {sym: sale for sym, sale in allocation.items() if sale[1] > 0}, (bought, allocation)
    assert summary["owners"] == {sym: owner for sym, (owner, _) in allocation.items()}, (summary, allocation)
    spent = dict.fromkeys(cash, 0)
    for owner, price in allocation.values():
        spent[owner] += price
    assert summary["cash"] == {p: cash[p] - spent[p] for p in cash}, (summary, allocation)
    assert summary["bo_par"] is not None and summary["bo_president"] == allocation["BO"][0], summary
    return actions


# --------------------------------------------------------------------- sampling
def sample_start_position(
    num_players: int,
    path,
    random_fraction: float = 0.2,
    max_price_multiple: float = 2.0,
    rng: Optional[random.Random] = None,
) -> StartPosition:
    """A start for a new ``num_players`` self-play game: a random start with
    probability ``random_fraction`` (always, when ``path`` has no starts for
    this player count), else a uniformly chosen human start from ``path``."""
    rng = rng or random.Random()
    human = load_start_positions(path).get(num_players, []) if path else []
    if human and rng.random() >= random_fraction:
        return rng.choice(human)
    return StartPosition(RANDOM_START, tuple(random_start_actions(num_players, rng, max_price_multiple)))
