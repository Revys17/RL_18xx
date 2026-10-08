"""Follow a game on an 18xx.games server in the Rust engine.

A server game is its whole action list: undos, redos and programmed actions
included. Every update filters that list as Ruby's
``Game::Base.filtered_actions`` does (``pretraining.filter_actions``) and
replays it into the Rust engine (``RustGameAdapter``, which the model reads)
with the human-game importer (``pretraining.replay_human_action``: player ids
mapped to seats, stray passes and skipped steps handled) -- its training-data
drop rules off: here every action has to be tried, and the strict engine
decides.

Updates are incremental. The importer decides some actions by looking at the
next two, so the follower keeps a *settled* game after all but the last two
actions and replays those two on a copy (the *tip*). When the new action list
extends the old one, only the new actions (and the last two again) are
replayed; when it doesn't -- an undo -- the game is replayed from the start
(well under a second for a whole game).

The positions before the last few actions are kept (:class:`MoveRecord`) so
the advisor can say how likely the model found the moves actually played.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

LOGGER = logging.getLogger(__name__)

SUPPORTED_TITLE = "1830"
TRAINED_PLAYER_COUNT = 4  # the model was trained on 4-player games only
PLAYER_RANGE = (2, 6)
# replay_human_action looks this many actions ahead.
LOOKAHEAD = 2
# Keep the positions before this many of the latest actions.
RECENT_WINDOW = 16


class UnsupportedGame(ValueError):
    """A game the engines can't follow (another title, an unimplemented optional rule, ...)."""


def support_problem(game: dict) -> Optional[str]:
    """Why a server game can't be followed, or None."""
    from rl18xx.agent.alphazero.pretraining import ENGINE_OPTIONAL_RULES

    title = str(game.get("title") or "")
    if title != SUPPORTED_TITLE:
        return f"{title or 'This title'} isn't supported: the advisor's engines and model only know 1830."
    rules = (game.get("settings") or {}).get("optional_rules") or []
    unsupported = [str(rule) for rule in rules if rule not in ENGINE_OPTIONAL_RULES]
    if unsupported:
        return (
            f"Optional rule(s) {', '.join(unsupported)} aren't supported: the engines implement only "
            f"{', '.join(ENGINE_OPTIONAL_RULES)}."
        )
    count = len(game.get("players") or [])
    if not PLAYER_RANGE[0] <= count <= PLAYER_RANGE[1]:
        return f"1830 is for {PLAYER_RANGE[0]}-{PLAYER_RANGE[1]} players; this game has {count}."
    return None


def _key(action: dict, drop_id: bool = False) -> str:
    """A canonical string of an action, to compare action lists."""
    if drop_id:
        action = {k: v for k, v in action.items() if k != "id"}
    return json.dumps(action, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class MoveRecord:
    """A recorded action as the engine got it, and the position it was played in."""

    index: int  # in the filtered action list (0-based)
    before: object  # RustGameAdapter
    action: dict  # as applied: engine player ids, after any substitution
    applied: bool = False  # False for a stray pass the importer dropped
    analysis: dict = field(default_factory=dict)  # the advisor's cache


class _Stopped(Exception):
    def __init__(self, index: int, action: dict, error: Exception):
        super().__init__(str(error))
        self.index = index
        self.action = action
        self.error = error


class LiveGame:
    """One server game replayed in the Rust engine (see the module docstring).

    ``rust`` is the current position (a ``RustGameAdapter``), ``error`` where
    the replay stopped (the engine rejected an action: ``rust`` is the position
    before it), ``names`` the player names by engine player id (seat order, as
    on the server), ``version`` counts the position's changes and
    ``position_key`` identifies it.
    """

    def __init__(self, game: dict):
        problem = support_problem(game)
        if problem:
            raise UnsupportedGame(problem)
        self.game_id = game.get("id")
        self.version = 0
        self._set_players(game)
        self._reset()
        self.update(game)

    # ------------------------------------------------------------- players
    def _set_players(self, game: dict) -> None:
        from rl18xx.agent.alphazero.pretraining import engine_optional_rules

        self.players = [
            {"id": p.get("id"), "name": str(p.get("name") or f"Player {i + 1}")} for i, p in enumerate(game["players"])
        ]
        self.num_players = len(self.players)
        # The server lists players in seat order; the importer's mapping.
        self.player_mapping = {p["id"]: i + 1 for i, p in enumerate(self.players)}
        self.names = {i + 1: p["name"] for i, p in enumerate(self.players)}
        self.optional_rules = list(engine_optional_rules(game))

    @property
    def untested(self) -> bool:
        """The model was trained on 4-player games only."""
        return self.num_players != TRAINED_PLAYER_COUNT

    def user_id(self, player_id: int):
        """The server user id of engine player ``player_id``."""
        return self.players[int(player_id) - 1]["id"]

    # ------------------------------------------------------------- engines
    def _reset(self) -> None:
        self._keys: list = []
        self._settled = None  # Rust game after the first _settled_count filtered actions
        self._settled_count = 0
        self.rust = None
        self.error: Optional[dict] = None
        self.records: dict = {}
        self.position_key: Optional[str] = None

    def _new_rust_game(self):
        from engine_rs import BaseGame as RustGame

        from rl18xx.rust_adapter import RustGameAdapter

        # Seats are "Player N" as in every engine game the model saw.
        players = {i: f"Player {i}" for i in range(1, self.num_players + 1)}
        return RustGameAdapter(RustGame(players, optional_rules=list(self.optional_rules)))

    # -------------------------------------------------------------- update
    def update(self, game: dict) -> bool:
        """Catch up with the server's copy of the game; returns whether the position changed."""
        from rl18xx.agent.alphazero.pretraining import filter_actions

        problem = support_problem(game)
        if problem:
            raise UnsupportedGame(problem)
        if [p.get("id") for p in game["players"]] != [p["id"] for p in self.players]:
            self._set_players(game)
            self._reset()
        actions = game.get("actions") or []
        filtered = filter_actions(actions)
        keys = [_key(a) for a in filtered]
        if self.rust is not None and keys == self._keys:
            return False

        extends = self._settled is not None and self.error is None and keys[: len(self._keys)] == self._keys
        if not extends:
            self._settled = self._new_rust_game()
            self._settled_count = 0
            self.records = {}
        count = len(filtered)
        start = self._settled_count
        settle_to = max(start, count - LOOKAHEAD)
        window = count - RECENT_WINDOW
        self.records = {i: r for i, r in self.records.items() if window <= i < start}
        self.error = None
        try:
            self._replay_range(self._settled, filtered, start, settle_to, window)
            self._settled_count = settle_to
            tip = self._settled.pickle_clone()
            self._replay_range(tip, filtered, settle_to, count, window)
        except _Stopped as stopped:
            self.error = {
                "action": stopped.index + 1,
                "type": stopped.action.get("type"),
                "message": f"{type(stopped.error).__name__}: {stopped.error}",
            }
            LOGGER.warning(
                "game %s: the engine stopped at action %d of %d (%s)",
                self.game_id,
                stopped.index + 1,
                count,
                stopped.action.get("type"),
            )
            # The rejected action may have been half-applied: replay up to it
            # afresh, and start from scratch at the next update.
            tip = self._new_rust_game()
            self.records = {}
            try:
                self._replay_range(tip, filter_actions(actions), 0, stopped.index, window)
            except _Stopped:  # can't happen: the same actions went through before
                tip = self._new_rust_game()
            self._settled = None
        self._keys = keys
        self.rust = tip
        self.version += 1
        self._update_position_key()
        return True

    def _replay_range(self, game, filtered: list, start: int, stop: int, window: int) -> None:
        from rl18xx.agent.alphazero.pretraining import HumanActionRejected, replay_human_action

        for i in range(start, stop):
            captured: dict = {}

            def before_apply(state, action, i=i):
                captured["moves"] = state._game.move_number
                if i >= window:
                    captured["before"] = state.pickle_clone()
                    captured["action"] = copy.deepcopy(action)

            try:
                replay_human_action(
                    game, filtered, i, self.player_mapping, use_rust=True, drop_rules=False, before_apply=before_apply
                )
            except HumanActionRejected as rejected:
                raise _Stopped(i, rejected.action, rejected.error) from None
            except Exception as e:  # the importer's own lookups (an unknown entity, ...)
                raise _Stopped(i, filtered[i], e) from None
            if "before" in captured:
                applied = game._game.move_number > captured["moves"]
                self.records[i] = MoveRecord(i, captured["before"], captured["action"], applied=applied)

    def _update_position_key(self) -> None:
        """A digest of the engine's action log: equal keys, equal positions."""
        keys = [_key(a, drop_id=True) for a in self.rust.raw_actions]
        self.position_key = hashlib.sha1("\n".join(keys).encode()).hexdigest()

    # -------------------------------------------------------------- reading
    @property
    def finished(self) -> bool:
        return bool(self.rust is not None and self.rust.finished)

    def acting(self) -> Optional[dict]:
        """Who is to move: ``kind`` (player / corporation / company), ``id``,
        ``name``, and ``player`` / ``player_id`` (the player behind it)."""
        game = self.rust
        if game is None or game.finished:
            return None
        entity = game.current_entity
        if entity is None:
            return None
        try:
            player = entity.player()
        except Exception:
            player = None
        player_id = getattr(player, "id", None)
        player_name = self.names.get(player_id) if player_id is not None else None
        if entity.is_player():
            return {
                "kind": "player",
                "id": entity.id,
                "name": player_name,
                "player": player_name,
                "player_id": player_id,
            }
        kind = "company" if entity.is_company() else "corporation"
        sym = getattr(entity, "sym", None) or entity.id
        return {
            "kind": kind,
            "id": sym,
            "name": f"{sym} ({entity.name})" if getattr(entity, "name", None) and entity.name != sym else sym,
            "player": player_name,
            "player_id": player_id,
        }

    def acting_mismatch(self, server_acting) -> Optional[str]:
        """A warning when the server's list of acting users doesn't include the
        player the engine has to move (the replay may have gone astray)."""
        acting = self.acting()
        if not acting or acting.get("player_id") is None or not server_acting:
            return None
        user = self.user_id(acting["player_id"])
        if user in server_acting:
            return None
        names = [p["name"] for p in self.players if p["id"] in server_acting]
        return (
            f"The engine has {acting['player'] or acting['name']} to move, but the server is waiting for "
            f"{', '.join(names) or 'someone else'}: its replay may have diverged from the server's."
        )

    def recent_records(self) -> list:
        """The kept positions before the latest actions, newest first."""
        return [self.records[i] for i in sorted(self.records, reverse=True)]
