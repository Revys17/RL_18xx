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
  allocation of the privates (:func:`random_start_actions`); or
* a **mid-game start** (``midgame_fraction`` of the games, when a mid-game file
  is given) — a recorded human game up to the beginning of one of its later
  Stock Rounds (``scripts/build_midgame_positions.py``). Games starting at Stock
  Round 1 rarely reach the positions that drive most of 1830's strategy -- trains
  about to rust, a company left without trains, a dump or a bankruptcy in the
  air -- because self-play settled into slow games where they seldom arise
  (2026-10-07: trained agents' dumps hit a trainless company 10% of the time
  against humans' 38%, and bankruptcies were ~9x rarer); mid-game starts put the
  agents into such positions without rewarding any strategy.
* a **practice start** (``practice_fraction`` of the games, drawn before everything
  else, when a practice file is given; off by default) -- a recorded game up to the
  beginning of a Stock Round shortly before a tactic the bots fall for (the robustness
  plan's loot-and-dump situations: ``scripts/robustness/build_practice_positions.py``).
  Same file format as the mid-game starts (``cuts`` per game); labels
  ``practice:<id>:sr<k>``.

Start files hold one player count's games (``human_games/start_positions_1830_<N>p_v2.jsonl``,
``human_games/midgame_positions_1830_<N>p_v2.jsonl``: :func:`default_start_positions` /
:func:`default_midgame_positions`); a game draws only starts of its own count, so the generators
check that a file they will draw from has some (:func:`check_start_files`) -- a 4-player file in a
2-player run would otherwise silently give random starts only. Random auction endings work for any
count.

A start is a list of action dicts applied to a fresh game, so both engines and
the game's action log stay valid, and ``RustMCTSPlayer.extract_data`` rebuilds
the positions by replaying it on an empty game. The auction is real 1830 in
both cases: start games are built with ``auction_unlock`` off.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from rl18xx.rust_adapter import RustGameAdapter

LOGGER = logging.getLogger(__name__)

HUMAN_START_PREFIX = "human:"
RANDOM_START = "random"
AUCTION_START = "auction"  # a fresh game: the waterfall auction is played too
MIDGAME_PREFIX = "midgame:"
PRACTICE_PREFIX = "practice:"  # a practice start (a mid-game-format file of hand-picked situations)

# What ``process_action`` reads from an auction action (bid / pass / par).
# Everything else in a recorded action — ``user``, ``created_at``, ``id`` — is
# dropped from start positions.
ACTION_KEYS = ("type", "entity", "entity_type", "company", "price", "corporation", "share_price")
START_ACTION_TYPES = ("bid", "pass", "par")


def default_start_positions(num_players: int) -> str:
    """The post-auction start file of a ``num_players`` game (relative to the working directory)."""
    return f"human_games/start_positions_1830_{int(num_players)}p_v2.jsonl"


def default_midgame_positions(num_players: int) -> str:
    """The mid-game start file of a ``num_players`` game (relative to the working directory)."""
    return f"human_games/midgame_positions_1830_{int(num_players)}p_v2.jsonl"


#: The practice-position template (``{n}``: the player count): positions one and two Stock Rounds before the
#: looted-broke dumps of the human corpus and Caleb's games (``scripts/robustness/build_practice_positions.py``).
PRACTICE_TEMPLATE = "human_games/practice_positions_1830_{n}p_v1.jsonl"


_PLAYERS_FIELD = re.compile(r'"num_players"\s*:\s*(\d+)')


def file_player_counts(path) -> Counter:
    """``{num_players: games}`` of a start or mid-game JSONL file, read from each line's
    ``num_players`` field without parsing the actions (the mid-game file is ~70 MB)."""
    counts: Counter = Counter()
    with open(path) as f:
        for line in f:
            match = _PLAYERS_FIELD.search(line)
            if match:
                counts[int(match.group(1))] += 1
    return counts


def check_start_files(
    num_players: int, path, random_fraction: float = 0.2, midgame_path=None, midgame_fraction: float = 0.0
) -> None:
    """Raise SystemExit if a start file a ``num_players`` generator will draw from has no game of
    that count: ``path`` when ``random_fraction`` < 1, ``midgame_path`` when ``midgame_fraction`` > 0.
    (:func:`sample_start_position` falls back to random starts / Stock Round 1 starts without games of
    the count, which is right for a count no human file covers but silently wrong for a wrong file.)
    A file that isn't there isn't checked: the generator fails on it as it always did."""
    checks = (
        (path, random_fraction < 1.0, "--start-positions", default_start_positions),
        (midgame_path, midgame_fraction > 0, "--midgame-positions", default_midgame_positions),
    )
    for file, used, flag, default in checks:
        if not file or not used or not Path(file).is_file():
            continue
        counts = file_player_counts(file)
        if not counts.get(int(num_players)):
            raise SystemExit(
                f"{flag} {file} has no {num_players}-player games ({dict(sorted(counts.items()))}); "
                f"the {num_players}-player file is {default(num_players)}"
            )


@dataclass(frozen=True)
class StartPosition:
    """Actions that take a fresh game to its first Stock Round."""

    label: str  # "human:<game id>" or "random"
    actions: tuple


def new_game(num_players: int, auction_unlock: bool = False, no_buyback_rule: bool = True) -> RustGameAdapter:
    """An empty Rust game with players "Player 1".."Player N" (self-play's naming), with the self-play rule
    against train shuffling on (``BaseGame.no_buyback``) unless ``no_buyback_rule`` is False."""
    from engine_rs import BaseGame as RustGame

    game = RustGame({i + 1: f"Player {i + 1}" for i in range(num_players)})
    game.set_auction_unlock(auction_unlock)
    game.set_no_buyback_rule(no_buyback_rule)
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


def human_midgame_cuts(game: dict, first_round: int = 2, last_round: int = 6) -> tuple:
    """A cleaned human game's actions and where its Stock Rounds ``first_round``..
    ``last_round`` begin.

    The game is replayed through the Rust engine (base rules) with the
    pretraining importer's leniency -- a pass the engine rejects is dropped --
    and its applied actions are read back from the engine (``raw_actions``), so
    every prefix replays strictly. Returns ``(actions, cuts, None)`` -- the
    actions up to the last cut and ``{round: prefix length}``, each prefix ending
    where that Stock Round's first player is about to act -- or
    ``(None, None, reason)``.
    """
    from rl18xx.agent.alphazero.pretraining import _process_pass_leniently

    players = game.get("players") or []
    ids = [p.get("id") if isinstance(p, dict) else None for p in players]
    if not 2 <= len(players) <= 6 or ids != list(range(1, len(players) + 1)):
        return None, None, "player_ids"
    state = new_game(len(players))
    stock_round, previous, cuts = 0, None, {}
    for raw in game.get("actions") or []:
        action = {k: v for k, v in raw.items() if k not in ("user", "created_at", "id")}
        if action.get("type") == "message":
            continue
        if action.get("type") == "pass":
            _process_pass_leniently(state, dict(action), use_rust=True)
        else:
            try:
                state.process_action(dict(action))
            except Exception as e:
                LOGGER.debug("Game %s: engine rejected %s: %s", game.get("id"), action, e)
                return None, None, "engine_error"
        if state.finished:
            break
        round_type = state._game.round.round_type
        if round_type == "Stock" and previous != "Stock":
            stock_round += 1
            if first_round <= stock_round <= last_round:
                cuts[stock_round] = len(state.raw_actions)
            if stock_round >= last_round:
                break
        previous = round_type
    if not cuts:
        return None, None, "no_later_stock_round"
    actions = [dict(a) for a in list(state.raw_actions)[: max(cuts.values())]]
    return actions, cuts, None


_LOADED_MIDGAME: dict = {}


def load_midgame_positions(path) -> dict:
    """``{num_players: [(game id, actions, {round: prefix length}), ...]}`` from a
    mid-game JSONL file (``scripts/build_midgame_positions.py``). Read once per
    process and cached."""
    key = str(Path(path).resolve())
    if key not in _LOADED_MIDGAME:
        by_players: dict = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                cuts = {int(k): int(v) for k, v in record["cuts"].items()}
                by_players.setdefault(int(record["num_players"]), []).append(
                    (str(record["id"]), tuple(record["actions"]), cuts)
                )
        _LOADED_MIDGAME[key] = by_players
        LOGGER.info(
            "Loaded mid-game positions from %s: %s",
            path,
            {n: len(games) for n, games in sorted(by_players.items())},
        )
    return _LOADED_MIDGAME[key]


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
    midgame_path=None,
    midgame_fraction: float = 0.0,
    auction_fraction: float = 0.0,
    picker=None,
    practice_path=None,
    practice_fraction: float = 0.0,
) -> StartPosition:
    """A start for a new ``num_players`` self-play game: with probability ``practice_fraction`` (when
    ``practice_path`` has games for this player count; drawn first, so it scales the other kinds' shares by
    ``1 - practice_fraction``; no draw at all at 0) a practice start -- a uniformly chosen game and one of its
    cuts, or the picker's; else with probability
    ``auction_fraction`` the beginning of the game (the auction is played; the
    caller turns on ``auction_unlock``); else with probability ``midgame_fraction``
    (when ``midgame_path`` has games for this player count) a mid-game start -- a
    uniformly chosen game and one of its Stock Rounds; otherwise a random start with
    probability ``random_fraction`` (always, when ``path`` has no starts for this
    player count), else a uniformly chosen human start from ``path``.

    ``picker`` (``generic_play.StartPicker``: a generic run's prioritized starts) chooses which mid-game cut or
    human start of the drawn kind (None from it: the uniform choice); the kinds' shares are unchanged."""
    rng = rng or random.Random()
    if practice_path and practice_fraction > 0 and rng.random() < practice_fraction:
        games = load_midgame_positions(practice_path).get(num_players, [])
        if games:
            picked = picker.practice(games) if picker is not None else None
            if picked is not None:
                (game_id, actions, cuts), stock_round = picked
            else:
                game_id, actions, cuts = rng.choice(games)
                stock_round = rng.choice(sorted(cuts))
            return StartPosition(f"{PRACTICE_PREFIX}{game_id}:sr{stock_round}", actions[: cuts[stock_round]])
    if auction_fraction > 0 and rng.random() < auction_fraction:
        return StartPosition(AUCTION_START, ())
    if midgame_path and midgame_fraction > 0 and rng.random() < midgame_fraction:
        games = load_midgame_positions(midgame_path).get(num_players, [])
        if games:
            picked = picker.midgame(games) if picker is not None else None
            if picked is not None:
                (game_id, actions, cuts), stock_round = picked
            else:
                game_id, actions, cuts = rng.choice(games)
                stock_round = rng.choice(sorted(cuts))
            return StartPosition(f"{MIDGAME_PREFIX}{game_id}:sr{stock_round}", actions[: cuts[stock_round]])
    human = load_start_positions(path).get(num_players, []) if path else []
    if human and rng.random() >= random_fraction:
        picked = picker.human(human) if picker is not None else None
        return picked if picked is not None else rng.choice(human)
    return StartPosition(RANDOM_START, tuple(random_start_actions(num_players, rng, max_price_multiple)))


_STALE_WARNED: set = set()


def replayed_start(
    num_players: int,
    path,
    random_fraction: float = 0.2,
    rng: Optional[random.Random] = None,
    midgame_path=None,
    midgame_fraction: float = 0.0,
    auction_fraction: float = 0.0,
    max_draws: int = 100,
    auction_unlock: bool = True,
    picker=None,
    practice_path=None,
    practice_fraction: float = 0.0,
) -> tuple:
    """``(start, game, skipped)``: a :func:`sample_start_position` start applied to a
    new game, with ``auction_unlock`` on for an auction start (as the generators play
    those) unless ``auction_unlock`` is False: the real rules, where an all-pass round
    after the SV is sold pays the privates out and the auction goes on.

    A recorded start whose actions the engine now rejects -- a position file built
    before a rules fix (2026-10-19: one stale mid-game position killed a
    policy-gradient run) -- is skipped and another start drawn; ``skipped`` counts the
    draws skipped, for the caller's history. Only the start's replay is guarded: an
    engine error later in the game is the caller's to see. Raises RuntimeError after
    ``max_draws`` draws in a row fail to replay.
    """
    for skipped in range(max_draws):
        start = sample_start_position(
            num_players,
            path,
            random_fraction,
            rng=rng,
            midgame_path=midgame_path,
            midgame_fraction=midgame_fraction,
            auction_fraction=auction_fraction,
            picker=picker,
            practice_path=practice_path,
            practice_fraction=practice_fraction,
        )
        try:
            game = apply_actions(new_game(num_players), start.actions)
        except RuntimeError as e:  # the engine's GameError: the file predates a rules change
            if start.label not in _STALE_WARNED:
                _STALE_WARNED.add(start.label)
                LOGGER.warning(
                    "Start %s no longer replays (%s); skipped -- rebuild its file "
                    "(scripts/build_start_positions.py / build_midgame_positions.py)",
                    start.label,
                    e,
                )
            continue
        if start.label == AUCTION_START and auction_unlock:
            game._game.set_auction_unlock(True)
        return start, game, skipped
    raise RuntimeError(f"no start position replayed in {max_draws} draws ({path}, {midgame_path})")
