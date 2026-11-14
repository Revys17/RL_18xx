"""How the title-generic pipeline plays a title: legal moves, applying them, and where games start
(RL bridge milestone C2; ``docs/rl_model.md`` "Policy gradient for the generic model").

**Play** (:class:`TitlePlay`, one per title, :func:`title_play`): a position's legal moves as the generic
model's :class:`model_generic.LegalActions` (descriptor rows, legal price ranges, layout indices), whether a
position is a decision (several legal moves, or one whose price is still open), the engine price of a
fixed-price move, and applying a legal row at a price. The default reads the title's RL layout
(``model_generic.legal_actions``; 1830's factored layout through ``apply_action_index`` /
``price_range_for_index``, a generator layout -- 1867 -- through ``rl_apply_index`` / ``rl_price_range``);
:func:`register_title_play` is the hook for a title that needs something else.

**Starts** (design §5.6): as 1830's ``start_positions``, a game starts at the opening (``auction``), at the
first Stock Round from a recorded human game (``human:<id>``) or from a random opening played through legal
moves (``random``), or at the start of a later Stock Round of a recorded human game (``midgame:<id>:sr<k>``).
A title whose opening auction *is* its first Stock Round (the 1822 family's bid boxes,
:data:`OPENS_WITH_STOCK_ROUND`) starts at the fresh game (``auction``) or where that Stock Round ends
(``human:<id>``, ``random``).
Start files hold one title's games, one per line::

    {"id": "...", "title": "1867", "num_players": 4, "seed": 1234, "actions": [...]}                (SR1 starts)
    {"id": "...", "title": "1867", "num_players": 4, "seed": 1234, "actions": [...], "cuts": {"2": 151, ...}}

``actions`` is the engine's own log of the replayed game (``raw_actions``), ``seed`` the engine seed the
game was replayed with (titles with randomness; 1830: none), ``cuts`` where each Stock Round k >= 2 begins.
``scripts/build_generic_start_positions.py`` writes them for any title with a replay loader. 1830's legacy
files (no ``title`` / ``seed`` fields) load as 1830 files, and **1830 games are started by the legacy code**
(``start_positions.replayed_start``: its random openings know the SV and the B&O par, its auction starts
take the ``auction_unlock`` variant), so a generic 1830 run starts exactly as the legacy runs do. Other titles'
random openings play the opening rounds through the title's own legal-move generator (uniform over moves,
prices uniform in ``[lo, max_price_multiple * lo]``: intentionally un-human, as 1830's are), or through its
layout when it has no generator.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

AUCTION_START = "auction"
RANDOM_START = "random"
HUMAN_START_PREFIX = "human:"
MIDGAME_PREFIX = "midgame:"
STOCK_ROUND = "Stock"
AUCTION_ROUND = "Auction"
#: Titles whose game opens with a Stock Round: the 1822 family's first Stock Round is its opening auction (bids
#: on the bid boxes' minors, concessions and privates, resolved as the round ends). Their "auction" start is
#: the fresh game, and their human / random starts are the position where that Stock Round has just ended
#: (the first position of another round: OR1, or the Choices round before it) -- what 1830's and 1867's
#: first-Stock-Round starts are to their auctions.
OPENS_WITH_STOCK_ROUND = frozenset({"1822", "1822PNW", "1822MX", "1822CA"})


def raw_game(game):
    """The ``engine_rs.BaseGame`` behind a game (an adapter's, or itself)."""
    return getattr(game, "_game", game)


def price_field(wire: dict) -> str:
    """The field of a generated action dict that holds its price: ``price`` (bids, trains, companies), or
    ``amount`` for the 1822 family's partial debt payoff (``payoff_player_debt_partial``)."""
    return "amount" if wire.get("type") == "payoff_player_debt_partial" else "price"


def opens_with_stock_round(title: str) -> bool:
    return str(title) in OPENS_WITH_STOCK_ROUND


def opening_done(title: str, raw) -> bool:
    """Whether a game that began at the fresh position has played its opening: 1830 / 1867 are at a Stock
    Round (the auction is over); a title that opens with a Stock Round (:data:`OPENS_WITH_STOCK_ROUND`) is
    past it (another round)."""
    is_stock = raw.round.round_type == STOCK_ROUND
    return not is_stock if opens_with_stock_round(title) else is_stock


#: A redrawn all-pass opening's termination when a game slot keeps drawing them (:func:`all_pass_opening`).
ALL_PASS = "all_pass"
#: How many all-pass openings in a row a self-play / eval game slot redraws before it keeps one (termination
#: :data:`ALL_PASS`): a policy that always passes must not spin a worker forever.
MAX_ALL_PASS_REDRAWS = 20


def all_pass_opening(title: str, raw) -> bool:
    """Whether a game of a title that opens with a Stock Round (the 1822 family) ended in that Stock Round with
    nothing bought, sold or bid -- Ruby's ``GAME_END_ON_NOTHING_SOLD_IN_SR1`` (``nothing_sold_in_sr``, end reason
    ``dnf``; the encoder's ``sr_idle``): every seat tied after a few passes, no learning signal. Self-play
    (``generic_pg``) and evals (``scripts/eval_generic.py``) redraw such games (the user, 2026-11-14); the rule
    itself is unchanged (replay, human import and the engine stay Ruby-faithful). Only the game's first Stock Round
    can end this way: any purchase, sale or bid clears the flag for the rest of the game."""
    if not opens_with_stock_round(title) or not raw.finished:
        return False
    snap = json.loads(raw.g1822_snapshot())
    return snap.get("game_end_reason") == "dnf" and snap.get("round") == STOCK_ROUND and int(snap.get("turn") or 0) <= 1


# ---------------------------------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------------------------------


class TitlePlay:
    """Legal moves and moves of one title for the generic pipeline (default: the title's RL layout)."""

    def __init__(self, title: str):
        self.title = str(title)

    def legal(self, raw):
        """The position's legal moves (``model_generic.LegalActions`` with layout indices)."""
        from rl18xx.agent.alphazero import model_generic as mg

        return mg.legal_actions(raw)

    @staticmethod
    def open_price(legal, k: int) -> bool:
        """Whether legal row ``k`` still has a price to choose (a price kind and a range wider than one price)."""
        from rl18xx.agent.alphazero.model_generic import C_PRICE_KIND

        if legal.rows[k, C_PRICE_KIND] < 0:
            return False
        lo, hi = int(legal.price_range[k, 0]), int(legal.price_range[k, 1])
        return hi > lo

    def is_decision(self, legal) -> bool:
        """Several legal moves, or one whose price is open (a decision for the mover's network; legacy
        ``policy_selfplay._advance``)."""
        return len(legal) > 1 or (len(legal) == 1 and self.open_price(legal, 0))

    @property
    def generator_layout(self) -> bool:
        """Whether the title's layout is a generator layout (1867: ``rl_price_range`` / ``rl_apply_index``)
        rather than 1830's factored one (``price_range_for_index`` / ``apply_action_index``)."""
        if not hasattr(self, "_generator_layout"):
            from rl18xx.agent.alphazero.action_descriptors import action_descriptors

            self._generator_layout = action_descriptors(self.title).kind == "generator"
        return self._generator_layout

    def fixed_price(self, raw, legal, k: int) -> Optional[int]:
        """The price of legal row ``k`` when it has no open price: None for a move without one, the engine's
        price for a fixed-price move (a depot train), the range's low end otherwise (as the legacy samplers).

        A generator layout's list (``LegalActions.wires`` set) already holds every priced move's range
        (``rl_legal_moves_json``'s "range"; -1 for a move without one -- every priced move's block has a price
        kind, ``test_generic_1867``): its low end is ``rl_price_range``'s, without enumerating the moves again
        (which cost a second generator run per position)."""
        index = int(legal.indices[k])
        if self.generator_layout and getattr(legal, "wires", None) is not None:
            lo = int(legal.price_range[k, 0])
            return None if lo < 0 else lo
        price_range = raw.rl_price_range(index) if self.generator_layout else raw.price_range_for_index(index)
        return None if price_range is None else int(price_range[0])

    def apply(self, raw, legal, k: int, price: Optional[int]) -> None:
        """Apply legal row ``k`` at ``price`` (None: the move's own price, its range's low end). A generator
        layout's row is applied from its action dict when the legal list carries them (``LegalActions.wires``:
        the same state and logged action as ``rl_apply_index``, without enumerating the moves again), the price
        checked against the row's range and step first."""
        if not self.generator_layout:
            raw.apply_action_index(int(legal.indices[k]), price)
            return
        wires = getattr(legal, "wires", None)
        if wires is None:
            raw.rl_apply_index(int(legal.indices[k]), price)
            return
        wire = dict(wires[k])
        if price is not None:
            lo, hi, step = (int(x) for x in legal.price_range[k])
            if lo < 0 or not (lo <= int(price) <= hi and (int(price) - lo) % max(step, 1) == 0):
                raise ValueError(f"price {price} is not legal for index {int(legal.indices[k])}: ({lo}, {hi}, {step})")
            wire[price_field(wire)] = int(price)
        raw.process_action(wire)

    def generator_moves(self, raw) -> Optional[list]:
        """The title's own legal-move generator (``[{"wire": action, "range": [lo, hi, step] or None}]``) for
        random openings, or None (then the layout's legal rows are used)."""
        return None


class GeneratorTitlePlay(TitlePlay):
    """A title whose rules side has a JSON legal-move generator (1867 ``g1867_legal_moves_json``, the 1822
    family ``g1822_legal_moves_json``). Its RL layout (A2) gives :meth:`legal`; random openings use the
    generator, so they work before the layout exists."""

    def __init__(self, title: str, method: str):
        super().__init__(title)
        self.method = method

    def generator_moves(self, raw) -> Optional[list]:
        moves = json.loads(getattr(raw, self.method)())
        if isinstance(moves, dict):
            moves = moves.get("moves") or moves.get("actions") or []
        return list(moves)


_PLAY: Dict[str, Callable[[], TitlePlay]] = {
    "1830": lambda: TitlePlay("1830"),
    "1867": lambda: GeneratorTitlePlay("1867", "g1867_legal_moves_json"),
    **{
        t: (lambda t=t: GeneratorTitlePlay(t, "g1822_legal_moves_json"))
        for t in ("1822", "1822PNW", "1822MX", "1822CA")
    },
}
_PLAY_CACHE: Dict[str, TitlePlay] = {}


def register_title_play(title: str, factory: Callable[[], TitlePlay]) -> None:
    """Replace a title's play adapter (the hook for a title whose layout's legal path differs, e.g. 1867 if
    package A2 serves its legal rows from the generator rather than ``factored_legal_indices``)."""
    _PLAY[str(title)] = factory
    _PLAY_CACHE.pop(str(title), None)


def title_play(title: str) -> TitlePlay:
    title = str(title)
    if title not in _PLAY_CACHE:
        if title not in _PLAY:
            raise ValueError(f"no play adapter for title {title!r} (have {sorted(_PLAY)})")
        _PLAY_CACHE[title] = _PLAY[title]()
    return _PLAY_CACHE[title]


# ---------------------------------------------------------------------------------------------------
# New games
# ---------------------------------------------------------------------------------------------------


def new_title_game(
    title: str,
    num_players: int,
    seed: Optional[int] = None,
    auction_unlock: bool = False,
    optional_rules: Sequence[str] = (),
):
    """An empty game of ``title`` as the generators build it (``titles.new_game``: 1830 with the rule
    against train shuffling on, an adapter; other titles the raw engine game, ``seed`` its RNG, under
    ``optional_rules`` -- 1822's rules combination)."""
    from rl18xx.agent.alphazero.titles import new_game

    if str(title) == "1830":
        return new_game("1830", num_players, auction_unlock=auction_unlock, optional_rules=optional_rules or None)
    return new_game(title, num_players, seed=seed, optional_rules=optional_rules or None)


def apply_actions(game, actions: Sequence[dict]):
    for action in actions:
        game.process_action(dict(action))
    return game


# ---------------------------------------------------------------------------------------------------
# Random openings
# ---------------------------------------------------------------------------------------------------


def _random_price(
    lo: int, hi: int, step: int, rng: random.Random, max_price_multiple: float, zero_floor_steps: int = 0
) -> int:
    step = max(1, int(step or 1))
    top = min(int(hi), max(int(lo), int(lo * max_price_multiple)))
    if zero_floor_steps and int(lo) <= 0:  # [0, 2 * 0] would always bid $0
        top = min(int(hi), int(lo) + zero_floor_steps * step)
    count = (top - int(lo)) // step + 1
    return int(lo) + step * rng.randrange(count)


def random_opening_actions(
    title: str,
    num_players: int,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
    max_price_multiple: float = 2.0,
    max_actions: int = 5000,
    optional_rules: Sequence[str] = (),
) -> List[dict]:
    """Actions that take a fresh ``title`` game through its opening to the first Stock Round by uniformly
    random legal moves (1830: ``start_positions.random_start_actions``, which realizes a random allocation of
    the privates). Ranged prices are uniform multiples of the range's step in ``[lo, max_price_multiple * lo]``
    (clipped to the range; a range from $0 -- a private's opening bid in the 1822 family -- takes
    ``[lo, lo + 20 steps]``). A title that opens with a Stock Round (the 1822 family) plays that Stock Round,
    its opening auction, to its end (:func:`opening_done`). ``optional_rules``: the game's rules combination
    (1822's)."""
    rng = rng or random.Random()
    if str(title) == "1830":
        from rl18xx.agent.alphazero.start_positions import random_start_actions

        return random_start_actions(num_players, rng, max_price_multiple)
    play = title_play(title)
    raw = raw_game(new_title_game(title, num_players, seed=seed, optional_rules=optional_rules))
    zero_steps = 20 if opens_with_stock_round(title) else 0
    for _ in range(max_actions):
        if raw.finished or opening_done(title, raw):
            return [dict(a) for a in raw.raw_actions]
        moves = play.generator_moves(raw)
        if moves is not None:
            if not moves:
                raise RuntimeError(f"{title}: no legal move in the opening")
            move = rng.choice(moves)
            action = dict(move.get("wire", move))
            span = move.get("range")
            if span and price_field(action) in action:
                step = span[2] if len(span) > 2 else 1
                action[price_field(action)] = _random_price(
                    span[0], span[1], step, rng, max_price_multiple, zero_steps
                )
            raw.process_action(action)
            continue
        legal = play.legal(raw)
        k = rng.randrange(len(legal))
        price = None
        if play.open_price(legal, k):
            lo, hi, step = (int(x) for x in legal.price_range[k])
            price = _random_price(lo, hi, step, rng, max_price_multiple, zero_steps)
        elif len(legal):
            price = play.fixed_price(raw, legal, k)
        play.apply(raw, legal, k, price)
    raise RuntimeError(f"{title}: random opening did not reach a Stock Round in {max_actions} actions")


# ---------------------------------------------------------------------------------------------------
# Start files
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GenericStart:
    """Actions that take a fresh game (under ``optional_rules``: 1822's rules combination) to where it starts."""

    label: str
    actions: tuple
    seed: Optional[int] = None
    optional_rules: tuple = ()


_LOADED: dict = {}


def _check_title(record: dict, title: str, path) -> None:
    found = record.get("title", "1830")  # 1830's legacy files carry no title
    if str(found) != str(title):
        raise ValueError(f"{path}: a {found} start file, not {title}")


def _record_rules(record: dict, title: str) -> tuple:
    """A start line's optional rules, canonical (``titles.TitleSpec.canonical_rules``; lines without the field --
    every file before 1822's rules mix -- are base rules)."""
    rules = record.get("optional_rules") or ()
    if not rules:
        return ()
    from rl18xx.agent.alphazero.titles import title_spec

    return title_spec(title).canonical_rules(rules, strict=False)


def load_start_file(path, title: str) -> Dict[int, List[GenericStart]]:
    """``{num_players: [GenericStart]}`` of an SR1 start file (cached per process); each start carries its line's
    ``optional_rules`` (canonical; none = base rules)."""
    key = ("start", str(Path(path).resolve()), str(title))
    if key not in _LOADED:
        by_players: dict = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                _check_title(record, title, path)
                label = f"{HUMAN_START_PREFIX}{record['id']}"
                rules = _record_rules(record, title)
                start = GenericStart(label, tuple(record["actions"]), record.get("seed"), rules)
                by_players.setdefault(int(record["num_players"]), []).append(start)
        _LOADED[key] = by_players
    return _LOADED[key]


def load_midgame_file(path, title: str) -> Dict[int, list]:
    """``{num_players: [(id, actions, {round: prefix length}, seed, optional rules)]}`` of a mid-game file
    (cached; the rules canonical, none = base rules)."""
    key = ("midgame", str(Path(path).resolve()), str(title))
    if key not in _LOADED:
        by_players: dict = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                _check_title(record, title, path)
                cuts = {int(k): int(v) for k, v in record["cuts"].items()}
                rules = _record_rules(record, title)
                by_players.setdefault(int(record["num_players"]), []).append(
                    (str(record["id"]), tuple(record["actions"]), cuts, record.get("seed"), rules)
                )
        _LOADED[key] = by_players
    return _LOADED[key]


def sample_start(
    title: str,
    num_players: int,
    path=None,
    random_fraction: float = 0.2,
    rng: Optional[random.Random] = None,
    midgame_path=None,
    midgame_fraction: float = 0.0,
    auction_fraction: float = 0.0,
    max_price_multiple: float = 2.0,
    optional_rules: Sequence[str] = (),
) -> GenericStart:
    """A start for a new ``title`` game (``start_positions.sample_start_position``'s mix for any title):
    the opening with probability ``auction_fraction``; else a mid-game cut with probability
    ``midgame_fraction`` (when ``midgame_path`` has games of this count); else a random opening with
    probability ``random_fraction`` (always without human starts of this count), else a human SR1 start.

    ``optional_rules`` (canonical: ``titles.draw_rules``) is the game's rules combination: human and mid-game
    starts are drawn from the lines played under exactly those rules (a combination without lines of this count
    falls through to the next kind, as a count without lines does), auction and random starts are fresh games
    under them."""
    rng = rng or random.Random()
    rules = tuple(optional_rules or ())
    if auction_fraction > 0 and rng.random() < auction_fraction:
        if opens_with_stock_round(title):
            # the 1822 family's setup shuffles the company stacks by the engine seed (None = 0: every opening
            # would deal the same bid boxes)
            return GenericStart(AUCTION_START, (), rng.randrange(1 << 31), rules)
        return GenericStart(AUCTION_START, (), None, rules)
    if midgame_path and midgame_fraction > 0 and rng.random() < midgame_fraction:
        games = [g for g in load_midgame_file(midgame_path, title).get(int(num_players), []) if g[4] == rules]
        if games:
            game_id, actions, cuts, seed, _ = rng.choice(games)
            k = rng.choice(sorted(cuts))
            return GenericStart(f"{MIDGAME_PREFIX}{game_id}:sr{k}", actions[: cuts[k]], seed, rules)
    human = load_start_file(path, title).get(int(num_players), []) if path else []
    human = [s for s in human if s.optional_rules == rules]
    if human and rng.random() >= random_fraction:
        return rng.choice(human)
    seed = rng.randrange(1 << 31)
    actions = random_opening_actions(
        title, num_players, rng, seed=seed, max_price_multiple=max_price_multiple, optional_rules=rules
    )
    return GenericStart(RANDOM_START, tuple(actions), None if str(title) == "1830" else seed, rules)


_STALE_WARNED: set = set()


def replayed_start(
    title: str,
    num_players: int,
    path=None,
    random_fraction: float = 0.2,
    rng: Optional[random.Random] = None,
    midgame_path=None,
    midgame_fraction: float = 0.0,
    auction_fraction: float = 0.0,
    auction_unlock: bool = True,
    max_draws: int = 100,
    optional_rules: Sequence[str] = (),
):
    """``(start, raw game, skipped)``: a start applied to a new game (under ``optional_rules``: the game's rules
    combination, :func:`sample_start`); recorded starts the engine no longer replays are skipped and counted
    (``start_positions.replayed_start``). 1830 delegates to the legacy function (the same starts, random openings
    and ``auction_unlock`` handling as the legacy runs; base rules)."""
    if str(title) == "1830":
        if optional_rules:
            raise ValueError("1830 games start through the legacy code, under the base rules")
        from rl18xx.agent.alphazero.start_positions import replayed_start as legacy_replayed_start

        start, game, skipped = legacy_replayed_start(
            num_players,
            path,
            random_fraction,
            rng=rng,
            midgame_path=midgame_path,
            midgame_fraction=midgame_fraction,
            auction_fraction=auction_fraction,
            auction_unlock=auction_unlock,
        )
        return GenericStart(start.label, tuple(start.actions)), raw_game(game), skipped
    for skipped in range(max_draws):
        start = sample_start(
            title,
            num_players,
            path,
            random_fraction,
            rng=rng,
            midgame_path=midgame_path,
            midgame_fraction=midgame_fraction,
            auction_fraction=auction_fraction,
            optional_rules=optional_rules,
        )
        try:
            game = new_title_game(title, num_players, seed=start.seed, optional_rules=start.optional_rules)
            raw = raw_game(apply_actions(game, start.actions))
        except (RuntimeError, ValueError) as e:
            if start.label not in _STALE_WARNED:
                _STALE_WARNED.add(start.label)
                LOGGER.warning("Start %s no longer replays (%s); skipped -- rebuild its file", start.label, e)
            continue
        return start, raw, skipped
    raise RuntimeError(f"no {title} start replayed in {max_draws} draws ({path}, {midgame_path})")


# ---------------------------------------------------------------------------------------------------
# Building start files from human games
# ---------------------------------------------------------------------------------------------------


def stock_round_cuts(
    title: str,
    num_players: int,
    actions: Sequence[dict],
    seed: Optional[int] = None,
    last_round: int = 6,
    optional_rules: Sequence[str] = (),
) -> Dict[int, int]:
    """``{k: prefix length}``: where Stock Round ``k`` (1-based) begins in a replayed action log -- the
    prefix after which the round is a Stock Round and the previous position's was not (SR1 of a title that
    opens with a Stock Round: 0). Replays the log on a fresh game (``raw_actions`` replay strictly)."""
    raw = raw_game(new_title_game(title, num_players, seed=seed, optional_rules=optional_rules))
    cuts: Dict[int, int] = {}
    k, previous = 0, None
    for i in range(len(actions) + 1):
        if raw.finished:
            break
        round_type = raw.round.round_type
        if round_type == STOCK_ROUND and previous != STOCK_ROUND:
            k += 1
            cuts[k] = i
            if k >= last_round:
                break
        previous = round_type
        if i < len(actions):
            raw.process_action(dict(actions[i]))
    return cuts


def opening_end_cut(
    title: str,
    num_players: int,
    actions: Sequence[dict],
    seed: Optional[int] = None,
    optional_rules: Sequence[str] = (),
) -> Optional[int]:
    """The prefix length after which a title that opens with a Stock Round (:data:`OPENS_WITH_STOCK_ROUND`) has
    played that Stock Round -- the first position whose round is not a Stock Round (OR1, or the Choices round
    before it) --, or None if the log never gets there. Replays the log on a fresh game."""
    raw = raw_game(new_title_game(title, num_players, seed=seed, optional_rules=optional_rules))
    for i in range(len(actions) + 1):
        if raw.finished:
            return None
        if raw.round.round_type != STOCK_ROUND:
            return i
        if i < len(actions):
            raw.process_action(dict(actions[i]))
    return None


def check_cut(
    title: str,
    num_players: int,
    actions: Sequence[dict],
    seed: Optional[int] = None,
    opening_end: bool = False,
    optional_rules: Sequence[str] = (),
) -> Optional[str]:
    """None if ``actions`` replay strictly on a fresh game to an unfinished Stock Round (``opening_end``: to an
    unfinished round that is not a Stock Round, the end of an opening Stock Round)."""
    try:
        game = new_title_game(title, num_players, seed=seed, optional_rules=optional_rules)
        raw = raw_game(apply_actions(game, actions))
    except Exception:
        return "strict_replay_error"
    if opening_end:
        if raw.finished:
            return "cut_at_finished_game"
        return "cut_at_stock_round" if raw.round.round_type == STOCK_ROUND else None
    if raw.finished or raw.round.round_type != STOCK_ROUND:
        return "cut_not_at_stock_round"
    return None


@dataclass
class ReplayedGame:
    """A human game replayed by the engine: its id, seat count, seed and the engine's own action log."""

    id: str
    num_players: int
    seed: Optional[int]
    actions: list
    optional_rules: tuple = ()


def _load_1830(game: dict) -> Tuple[Optional[ReplayedGame], Optional[str]]:
    """1830: a cleaned export (``main.py clean``) replayed as the generic importer replays it."""
    from rl18xx.agent.alphazero.generic_data import importer_for

    importer = importer_for("1830")
    actions, reason = importer.load(game)
    if actions is None:
        return None, reason
    rules = tuple(importer.optional_rules(game))
    return ReplayedGame(str(game["id"]), len(importer.players(game)), None, [dict(a) for a in actions], rules), None


def _load_titled(game: dict) -> Tuple[Optional[ReplayedGame], Optional[str]]:
    """1867 / the 1822 family: a raw 18xx.games export replayed as the corpus audits replay it
    (``tests/title_replay_harness.replay_game``: Ruby's filtered actions, auto-actions, seats renumbered)."""
    try:
        from tests.title_replay_harness import ReplayError, TitleUnsupported, game_seed, replay_game
    except ImportError as e:  # the harness lives in the repo's tests/ (run from the repository root)
        return None, f"no_replay_harness: {e}"

    captured = {}  # the harness's engine game (handed to on_action before every action)
    rules = tuple((game.get("settings") or {}).get("optional_rules") or ())
    try:
        replay_game(game, on_action=lambda rust, i, a: captured.setdefault("game", rust))
    except (ReplayError, TitleUnsupported, ValueError) as e:
        # A record the engine rejects part-way gives no start at all, not even before the rejection (A3, the
        # user's call "for correctness": 1867's 24 such old games are records the current rules refuse).
        return None, f"replay_failed: {str(e)[:120]}"
    rust = captured.get("game")
    if rust is None:
        return None, "no_actions"
    actions = [dict(a) for a in rust.raw_actions]
    return ReplayedGame(str(game.get("id")), len(game["players"]), game_seed(game), actions, rules), None


_LOADERS: Dict[str, Callable[[dict], Tuple[Optional[ReplayedGame], Optional[str]]]] = {
    "1830": _load_1830,
    "1867": _load_titled,
    "1822": _load_titled,
    "1822PNW": _load_titled,
    "1822MX": _load_titled,
    "1822CA": _load_titled,
}


def register_replay_loader(title: str, loader) -> None:
    """A title's human-game replay for start files (``game dict -> (ReplayedGame or None, reason)``)."""
    _LOADERS[str(title)] = loader


def replay_human_game(title: str, game: dict) -> Tuple[Optional[ReplayedGame], Optional[str]]:
    try:
        loader = _LOADERS[str(title)]
    except KeyError:
        raise ValueError(f"no human-game replay for {title!r} (have {sorted(_LOADERS)})") from None
    if str(game.get("title", title)) != str(title):
        return None, f"title {game.get('title')}"
    return loader(game)


def start_records(
    title: str, replayed: ReplayedGame, first_round: int = 2, last_round: int = 6
) -> Tuple[Optional[dict], Optional[dict], Optional[str]]:
    """The start record and the mid-game record (Stock Rounds ``first_round``..``last_round``) of a replayed
    human game, each prefix checked to replay strictly to a Stock Round; ``(start, midgame, reason)`` with
    None for a part the game has not got. The start is SR1's beginning (1830, 1867: after the auction) or, for
    a title that opens with a Stock Round (the 1822 family), SR1's end (:func:`opening_end_cut`; such a record
    carries ``"start": "sr1_end"``). A title that plays optional rules (``titles.TitleSpec.rules_mix``: 1822) records
    each line's canonical ``optional_rules`` (``[]`` for base rules), and the prefixes replay under them."""
    from rl18xx.agent.alphazero.titles import title_spec

    n = replayed.num_players
    spec = title_spec(title)
    rules = spec.canonical_rules(replayed.optional_rules, strict=False) if replayed.optional_rules else ()
    try:
        cuts = stock_round_cuts(title, n, replayed.actions, replayed.seed, last_round, optional_rules=rules)
        first = (
            opening_end_cut(title, n, replayed.actions, replayed.seed, optional_rules=rules)
            if opens_with_stock_round(title)
            else cuts.get(1)
        )
    except Exception as e:  # a log that does not replay on a fresh game
        return None, None, f"log_replay_error: {type(e).__name__}"
    if 1 not in cuts:
        return None, None, "no_stock_round"
    base = {"id": replayed.id, "title": str(title), "num_players": n}
    if replayed.seed is not None:
        base["seed"] = int(replayed.seed)
    if spec.rules_mix or rules:
        base["optional_rules"] = list(rules)
    start = None
    if first is None:
        reason = "no_sr1_end"
    else:
        reason = check_cut(
            title,
            n,
            replayed.actions[:first],
            replayed.seed,
            opening_end=opens_with_stock_round(title),
            optional_rules=rules,
        )
    if reason is None:
        start = {**base, "actions": replayed.actions[:first]}
        if opens_with_stock_round(title):
            start["start"] = "sr1_end"
    later = {k: v for k, v in cuts.items() if first_round <= k <= last_round}
    for k, length in sorted(later.items()):
        if check_cut(title, n, replayed.actions[:length], replayed.seed, optional_rules=rules) is not None:
            later = {kk: vv for kk, vv in later.items() if kk < k}
            break
    midgame = None
    if later:
        cuts_out = {str(k): v for k, v in later.items()}
        midgame = {**base, "actions": replayed.actions[: max(later.values())], "cuts": cuts_out}
    return start, midgame, reason if start is None else None


__all__ = [
    "ALL_PASS",
    "AUCTION_START",
    "MAX_ALL_PASS_REDRAWS",
    "all_pass_opening",
    "GenericStart",
    "GeneratorTitlePlay",
    "ReplayedGame",
    "TitlePlay",
    "OPENS_WITH_STOCK_ROUND",
    "load_midgame_file",
    "load_start_file",
    "new_title_game",
    "opening_done",
    "opening_end_cut",
    "opens_with_stock_round",
    "random_opening_actions",
    "register_replay_loader",
    "register_title_play",
    "replay_human_game",
    "replayed_start",
    "sample_start",
    "start_records",
    "stock_round_cuts",
    "title_play",
]
