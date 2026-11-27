"""Bot seats of titles other than 1830 (1867 today): follow the game in the Rust engine, play a title-generic
policy's own move or (``--title-search``) the move a search of that policy with a title-generic critic chooses. Only
the local site's connector plays them (``main.py advisor --bots --title-policy 1867=<checkpoint>[@T] [--title-search
1867=<critic>[#head][@readouts][/c_puct][:options]]``); the advice, the extension and 18xx.games stay 1830-only.

**Following** (:class:`TitledGame`): the site's action list is filtered as Ruby's ``Game.load`` processes it --
undo / redo, then each kept action followed by its recorded auto actions
(``tests.title_replay_harness.ruby_filtered_actions``, the 1867 importer's replay, ``generic_import_1867``) -- and
replayed into an ``engine_rs`` game of the title (seeded as Ruby seeds it: ``settings.seed``, else the game id's
digits), the site's players as seats 1..N in the site's order. Updates are incremental: an action list that extends
the last one replays only its new actions; anything else (an undo) replays the game from the start. An action the
engine refuses stops the replay there (``error``; the position is the one before it).

**Moves** (:class:`TitledPolicy`): the legal moves are the title's cross-player policy's
(``model_generic.legal_actions``: what the policy-gradient runs trained and were evaluated on), so where Ruby offers
a purchase or a merger across players the bot never takes it -- it passes or declines among its own moves. A
decision is the policy's move sampled from softmax(log p / T) (T: the run's ``learner_temperature`` unless given) and,
for an open price, a price from its price head's sampled cell -- exactly as the generic policy-gradient learner plays
(``generic_pg``, ``scripts/eval_generic.py``); a position with one move and no price to choose just plays it. Two
of Ruby HEAD's dead ends are avoided as the 1867 wire fuzzer avoids them (``scripts/g1867_fuzz.py``; they are moves
Ruby accepts but then cannot go on from): a MajorTrainless choice that leaves Ruby's operator stale
(``g1867_ruby_dead_end_choice``) is never sampled while another move exists, and an auction bid keeps $5 back once
every other player is below $5. The move goes to the site as the engine logged it, a player entity as the site's
user id. Where the site's engine (Ruby) and ours disagree about who acts, the bot stops and says so: Ruby is the
source of truth for rules, a disagreement is a rules difference to fix, not to paper over.

**Searched moves** (``--title-search``, :class:`TitledPolicy` ``search``; ``docs/rl_model.md`` "Searching 1867"): every
decision is searched by the Rust tree over the same legal moves (``generic_search``: the policy's priors at the run's
``learner_temperature`` unless the spec's ``ptemp`` says otherwise, its price cells, the critic's ``#score`` head by
default) with the 1830 bots' settings unless the spec overrides them (``generic_search.BOT_SEARCH``: mean Q, c_puct
0.05, 8 leaves per batch, each distinct leaf once; ``BOT_READOUTS`` readouts), in-process on the bots' CPU. The move
is the most visited one whose Ruby continuation is safe (the dead-end choice above is never played while another
move exists); an open price is the most visited price grandchild's, and an auction bid still keeps $5 back as above.
A position with one move and no price is played as is; one move with an open price is priced by the root's own
evaluation (no further search).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from rl18xx.agent.advisor.autoplay import AutoplayError
from rl18xx.agent.advisor.live_game import UnsupportedGame

LOGGER = logging.getLogger(__name__)

#: The titles whose seats the bots can play this way, and their player counts (1867: as the generic runs, 3-6).
TITLES = {"1867": (3, 6)}
#: Action fields the site adds or numbers itself.
SITE_FIELDS = ("id", "created_at", "user")
#: Searched games whose tree the bots keep after their move (LRU): the next search of that game starts from the
#: chosen move's subtree when the game is still at its root (the same bot moving again).
KEPT_TREES = 4


def titled_problem(game: dict) -> Optional[str]:
    """Why a site game of a :data:`TITLES` title can't be followed, or None."""
    title = str(game.get("title") or "")
    if title not in TITLES:
        return f"{title or 'This title'} isn't one the bots play here (they play {', '.join(sorted(TITLES))} and 1830)."
    rules = (game.get("settings") or {}).get("optional_rules") or []
    if rules:
        return f"{title} with optional rules ({', '.join(map(str, rules))}) isn't supported: the bots play base rules."
    count = len(game.get("players") or [])
    low, high = TITLES[title]
    if not low <= count <= high:
        return f"The {title} bots play {low}-{high} players; this game has {count}."
    return None


def _key(action: dict) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), default=str)


class _Stopped(Exception):
    def __init__(self, index: int, action: dict, error: BaseException):
        super().__init__(str(error))
        self.index, self.action, self.error = index, action, error


class TitledGame:
    """One site game of a :data:`TITLES` title replayed in the Rust engine (module docstring).

    ``raw`` is the current position (an ``engine_rs.BaseGame``), ``error`` where the replay stopped, ``names`` the
    player names by seat (1..N, the site's order) and ``player_mapping`` site user id -> seat."""

    def __init__(self, game: dict):
        problem = titled_problem(game)
        if problem:
            raise UnsupportedGame(problem)
        self.title = str(game["title"])
        self.game_id = game.get("id")
        self.version = 0
        self._set_players(game)
        self._reset()
        self.update(game)

    def _set_players(self, game: dict) -> None:
        from tests.title_replay_harness import game_seed

        self.players = [
            {"id": p.get("id"), "name": str(p.get("name") or f"Player {i + 1}")} for i, p in enumerate(game["players"])
        ]
        self.num_players = len(self.players)
        self.player_mapping = {p["id"]: i + 1 for i, p in enumerate(self.players)}
        self.names = {i + 1: p["name"] for i, p in enumerate(self.players)}
        self.seed = int(game_seed(game))

    def user_id(self, seat: int):
        """The site user id of seat ``seat`` (1-based)."""
        return self.players[int(seat) - 1]["id"]

    def _reset(self) -> None:
        self._keys: list = []
        self.raw = None
        self.error: Optional[dict] = None
        self.position_key: Optional[str] = None

    def _new_game(self):
        from rl18xx.agent.alphazero.generic_data import new_engine_game

        # Seats are "Player N" as in every engine game the model saw.
        players = {i: f"Player {i}" for i in range(1, self.num_players + 1)}
        return new_engine_game(self.title, players, (), self.seed)

    def _filtered(self, game: dict) -> list:
        """The site's actions as Ruby processes them, players as seats."""
        from tests.title_replay_harness import ruby_filtered_actions

        out = []
        for action in ruby_filtered_actions(copy.deepcopy(game.get("actions") or [])):
            if action.get("entity_type") == "player":
                seat = self.player_mapping.get(action.get("entity"))
                if seat is None:
                    raise UnsupportedGame(f"action {action.get('id')}: {action.get('entity')} is not a player")
                action["entity"] = action["user"] = seat
            out.append(action)
        return out

    def update(self, game: dict) -> bool:
        """Catch up with the site's copy of the game; returns whether the position changed."""
        problem = titled_problem(game)
        if problem:
            raise UnsupportedGame(problem)
        if [p.get("id") for p in game["players"]] != [p["id"] for p in self.players]:
            self._set_players(game)
            self._reset()
        filtered = self._filtered(game)
        keys = [_key(a) for a in filtered]
        if self.raw is not None and keys == self._keys:
            return False
        extends = self.raw is not None and self.error is None and keys[: len(self._keys)] == self._keys
        start = len(self._keys) if extends else 0
        raw = self.raw if extends else self._new_game()
        self.error = None
        try:
            self._replay(raw, filtered, start, len(filtered))
        except _Stopped as stopped:
            self.error = {
                "action": stopped.index + 1,
                "type": stopped.action.get("type"),
                "message": f"{type(stopped.error).__name__}: {stopped.error}"[:500],
            }
            LOGGER.warning(
                "%s game %s: the engine stopped at action %d of %d (%s): %s",
                self.title,
                self.game_id,
                stopped.index + 1,
                len(filtered),
                stopped.action.get("type"),
                self.error["message"],
            )
            # The refused action may have been half-applied: replay up to it afresh.
            raw = self._new_game()
            self._replay(raw, filtered, 0, stopped.index)
        self.raw = raw
        self._keys = keys
        self.version += 1
        self.position_key = hashlib.sha1("\n".join(keys[: len(raw.raw_actions)]).encode()).hexdigest()
        return True

    @staticmethod
    def _replay(raw, filtered: list, start: int, stop: int) -> None:
        for i in range(start, stop):
            try:
                raw.process_action(dict(filtered[i]))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # noqa: BLE001 -- pyo3's PanicException is a BaseException
                raise _Stopped(i, filtered[i], e) from None

    @property
    def finished(self) -> bool:
        return bool(self.raw is not None and self.raw.finished)

    def acting(self) -> Optional[tuple]:
        """``(type, id)`` of the entity the engine has to move: ``("player", seat)``, ``("corporation", sym)`` or
        ``("company", sym)``; None for nobody (a finished game)."""
        if self.raw is None or self.raw.finished:
            return None
        offers = self.raw.g1867_step_offers()
        if not offers:
            return None
        entity = str(offers[0][1])
        if entity.startswith("company:"):
            return "company", entity[len("company:") :]
        if entity.isdigit():
            return "player", int(entity)
        return "corporation", entity

    def step(self) -> Optional[str]:
        """The engine's step kind for the acting entity (``HomeToken``, ``PostMergerShares``, ...)."""
        if self.raw is None or self.raw.finished:
            return None
        offers = self.raw.g1867_step_offers()
        return str(offers[0][0]) if offers else None

    def describe_entity(self, kind: str, entity) -> str:
        if kind == "player":
            return self.names.get(int(entity), str(entity))
        return str(entity)


def to_site_action(action: dict, live: TitledGame) -> dict:
    """An engine action as the site stores it: a player entity as the site's user id, without the fields the
    site sets itself (it numbers the action and adds its auto actions)."""
    out = {k: v for k, v in action.items() if k not in SITE_FIELDS}
    if out.get("entity_type") == "player":
        out["entity"] = live.user_id(int(out["entity"]))
    return out


def describe_action(action: dict) -> str:
    """A short description of an engine action dict for the bots' event log."""
    kind = str(action.get("type", "?"))
    parts = []
    for key in ("company", "corporation", "shares", "hex", "tile", "rotation", "city", "train", "exchange", "kind",
                "choice", "share_price"):  # fmt: skip
        value = action.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        parts.append(f"{key} {value}")
    if action.get("price") is not None:
        parts.append(f"${action['price']}")
    if kind == "run_routes":
        parts.append(f"${sum(int(r.get('revenue') or 0) for r in action.get('routes') or [])}")
    text = kind.replace("_", " ")
    return f"{text}: {', '.join(parts)}" if parts else text


class TitledPolicy:
    """A title-generic checkpoint playing one title's bot seats policy-only (module docstring)."""

    def __init__(self, title: str, path, device="cpu", temperature: Optional[float] = None, seed=None, search=None):
        from rl18xx.agent.advisor.advisor import checkpoint_label, training_temperature
        from rl18xx.agent.alphazero import model_generic as mg
        from rl18xx.agent.alphazero.generic_inference import LocalGenericServer
        from rl18xx.agent.alphazero.generic_play import title_play

        self.title = str(title)
        if self.title not in TITLES:
            raise ValueError(f"title {title!r}: the bots play {', '.join(sorted(TITLES))} this way")
        self.path = str(path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(f"No generic checkpoint at {Path(self.path).resolve()}.")
        model, meta = mg.load_checkpoint(self.path, device)
        if self.title not in (meta.get("titles") or []):
            raise mg.CheckpointMismatch(f"{self.path}: not trained on {self.title} (titles {meta.get('titles')})")
        self.temperature = float(training_temperature(self.path) if temperature is None else temperature)
        if not self.temperature > 0:
            raise ValueError(f"{self.title}: the play temperature must be > 0")
        self.label = f"generic:{checkpoint_label(self.path)}@{self.temperature:g}"
        self.server = LocalGenericServer(model.eval(), seed=seed)
        self.play = title_play(self.title)
        self.rng = np.random.default_rng(seed)
        self._lock = threading.Lock()
        self._trees: "OrderedDict" = OrderedDict()  # game key -> GenericSearchPlayer (--title-search)
        self.kept_trees = 0  # searches that started from a kept tree
        # --title-search: a generic_search.SearchSpec (its ``policy`` this checkpoint), searched in-process.
        self.search = search
        self.evaluator = None
        if search is not None:
            from rl18xx.agent.alphazero.generic_search import local_evaluator

            critic = None
            if search.value:
                value_model, value_meta = mg.load_checkpoint(search.value, device)
                if self.title not in (value_meta.get("titles") or []):
                    raise mg.CheckpointMismatch(
                        f"{search.value}: not trained on {self.title} (titles {value_meta.get('titles')})"
                    )
                critic = LocalGenericServer(value_model.eval(), seed=seed)
            self.evaluator = local_evaluator(self.server, critic, search.value_head)
            value = f"generic:{checkpoint_label(search.value)}" if search.value else "own heads"
            ptemp = search.overrides.get("prior_temperature", self.temperature)
            self.label += f" + search {value}#{search.value_head}@{search.readouts} (prior T {ptemp:g})"

    # ------------------------------------------------------------------ choice
    def _ruby_safe(self, raw, legal) -> np.ndarray:
        """Per legal row, whether Ruby HEAD can go on after it (module docstring): False for a MajorTrainless
        choice into Ruby's stale-operator dead end (unless every move is one)."""
        safe = np.array(
            [
                not (w.get("type") == "choose" and raw.g1867_ruby_dead_end_choice(str(w.get("entity"))))
                for w in legal.wires
            ]
        )
        return safe if safe.any() else np.ones(len(legal), dtype=bool)

    def _keep_back(self, raw, legal, k: int, price):
        """``(k, price)`` with an auction bid keeping $5 back once every other player is below $5 (Ruby HEAD's
        SingleItemAuction crashes when a lot comes up that nobody can bid on): the bid lowered on its grid, or a
        pass when it can't be."""
        wire = legal.wires[k]
        if wire.get("type") != "bid" or "company" not in wire or price is None:
            return k, price
        cash = {int(p.id): int(p.cash) for p in raw.players}
        me = int(wire["entity"])
        others = [c for pid, c in cash.items() if pid != me]
        if not others or max(others) >= 5 or int(price) <= cash[me] - 5:
            return k, price
        lo, _, step = (int(x) for x in legal.price_range[k])
        if lo <= cash[me] - 5:
            return k, lo + (cash[me] - 5 - lo) // max(step, 1) * max(step, 1)
        passes = [j for j, w in enumerate(legal.wires) if w.get("type") == "pass"]
        return (passes[0], None) if passes else (k, price)

    def choose(self, raw, legal, key=None) -> dict:
        """The move at ``raw`` among ``legal`` (``model_generic.LegalActions`` of the title's cross-player policy):
        ``{k, price, probability, kind}`` (``kind``: policy / priced / forced)."""
        from rl18xx.agent.alphazero import model_generic as mg
        from rl18xx.agent.alphazero.generic_pg import tempered_logp
        from rl18xx.agent.alphazero.generic_tokens import encode_batch

        if not self.play.is_decision(legal):
            return {"k": 0, "price": self.play.fixed_price(raw, legal, 0), "probability": 1.0, "kind": "forced"}
        if self.search is not None:
            return self._searched(raw, legal, key)
        arrays = mg.add_legal_inputs(encode_batch([raw]), [legal])
        with self._lock:
            reply = self.server.run(arrays, price_eps=0.0)
        logp = np.asarray(reply.logp[: len(legal)], dtype=np.float64)
        p = np.exp(tempered_logp(logp, self.temperature)) * self._ruby_safe(raw, legal)
        k = int(self.rng.choice(len(p), p=p / p.sum()))
        if self.play.open_price(legal, k):
            row = {int(r): j for j, r in enumerate(arrays["price_rows"])}[k]
            price = int(legal.price_cells(k).sample_price(int(reply.price_cell[row]), self.rng))
        else:
            price = self.play.fixed_price(raw, legal, k)
        chosen, price = self._keep_back(raw, legal, k, price)
        if chosen != k:
            k, price = chosen, self.play.fixed_price(raw, legal, chosen)
        return {
            "k": k,
            "price": price,
            "probability": float(np.exp(logp[k])),
            "kind": "policy" if len(legal) > 1 else "priced",
        }

    def search_config(self, num_players: int):
        """The tree's settings for a ``num_players`` game: the spec's, priors at this policy's temperature unless
        the spec gives ``ptemp``, finished games backed up as the critic's run scored them."""
        extra = {} if "prior_temperature" in self.search.overrides else {"prior_temperature": self.temperature}
        return self.search.config(num_players, **extra)

    def _searched(self, raw, legal, key=None) -> dict:
        """:meth:`choose` with ``--title-search``: the most visited Ruby-safe move of a search from ``raw``. ``key``
        (a site game's): its tree is kept after the move and searched on from if the game is still at its root
        (:data:`KEPT_TREES`; the action dicts compared without the site's own fields)."""
        from rl18xx.agent.alphazero.generic_play import price_field
        from rl18xx.agent.alphazero.generic_search import GenericSearchPlayer, search_position

        indices = [int(i) for i in legal.indices]
        safe = self._ruby_safe(raw, legal)
        with self._lock:
            player = self._trees.pop(key, None) if key is not None else None
            if player is None:
                player = GenericSearchPlayer(self.search_config(len(raw.players)))
            kept = player.kept
            search_position(player, raw, self.evaluator, ignore=SITE_FIELDS)
            self.kept_trees += player.kept - kept
            if [int(i) for i in player.legal()] != indices:
                raise RuntimeError("the search tree's legal moves are not the policy's")
            index = player.pick_move(allowed={i for i, ok in zip(indices, safe) if ok})
            visits = player.visits()
            readouts = float(player.tree.n_at_root())
            actions = player.play(index)
            if key is not None:
                self._trees[key] = player
                while len(self._trees) > KEPT_TREES:
                    self._trees.popitem(last=False)
        k = indices.index(index)
        if self.play.open_price(legal, k):
            price = int(actions[0][price_field(actions[0])])
        else:
            price = self.play.fixed_price(raw, legal, k)
        chosen, price = self._keep_back(raw, legal, k, price)
        if chosen != k:
            k, price = chosen, self.play.fixed_price(raw, legal, chosen)
        total = sum(visits.values())
        return {
            "k": k,
            "price": price,
            "probability": float(visits.get(indices[k], 0.0) / total) if total > 0 else 1.0,
            "kind": "search" if len(legal) > 1 else "priced",
            "visits": int(readouts),
        }

    # -------------------------------------------------------------------- move
    def move(self, live: TitledGame, site, seats) -> dict:
        """The model's move for one of ``seats`` (1-based, the site's order) where the site's engine waits for a
        bot (``site``: an ``autoplay.SiteView``): ``action`` (the site's format), ``actor``, ``description`` and
        what chose it. Raises :class:`~rl18xx.agent.advisor.autoplay.AutoplayError` when the bots have no move to
        make there."""
        seats = {int(s) for s in seats}
        if site.finished:
            raise AutoplayError("The game is over.")
        seat = live.player_mapping.get(site.player)
        if seat is None:
            raise AutoplayError(f"The site waits for {site.entity_id}, which no player controls.")
        if seat not in seats:
            raise AutoplayError(f"It is {live.names.get(seat, seat)}'s turn, a seat the bots don't play.")
        if live.error:
            raise AutoplayError(
                f"The engine could not follow action {live.error['action']} ({live.error['type']}: "
                f"{live.error['message']})."
            )
        if live.finished:
            raise AutoplayError("The engine has the game over while the site doesn't.")
        site_entity = ("player", seat) if site.entity_type == "player" else (site.entity_type, str(site.entity_id))
        engine = live.acting()
        # Ruby's view can name a player with no actions of their own while the step's move is another entity's:
        # 1867's HomeToken step waits for the Stock Round's player (``active_entities``) while the won minor places
        # its home token (``pending_entity``), and that player posts it. The same step on both sides is agreement.
        for_entity = (
            engine is not None and site_entity[0] == "player" and not site.actions and live.step() == site.step
        )
        if engine != site_entity and not for_entity:
            moving = live.describe_entity(*engine) if engine else "nobody"
            raise AutoplayError(
                f"The site's engine waits for {live.describe_entity(*site_entity)} ({site.step or '?'}) but ours has "
                f"{moving} to move: the engines disagree here."
            )
        raw = live.raw
        legal = self.play.legal(raw)
        if len(legal) == 0:
            raise AutoplayError(
                f"Our engine has no legal move for {live.describe_entity(*engine)} ({site.step or '?'})."
            )
        choice = self.choose(raw, legal, key=(live.game_id, live.seed))
        after = raw.pickle_clone()
        before = len(raw.raw_actions)
        self.play.apply(after, legal, choice["k"], choice["price"])
        logged = list(after.raw_actions)
        if len(logged) <= before:
            raise RuntimeError(f"the engine logged no action for move {int(legal.indices[choice['k']])}")
        action = dict(logged[before])
        actor = live.names[seat] if engine[0] == "player" else f"{engine[1]} ({live.names[seat]})"
        return {
            "kind": choice["kind"],
            "title": live.title,
            "action": to_site_action(action, live),
            "index": int(legal.indices[choice["k"]]),
            "num_legal": len(legal),
            # the policy's probability of the move (temperature 1); a searched move: its share of the root's visits
            "share": choice["probability"],
            "temperature": self.temperature,
            **({"visits": choice["visits"]} if "visits" in choice else {}),
            "seat": seat,
            "player": live.names[seat],
            "actor": actor,
            "description": describe_action(action),
        }


__all__ = ["TITLES", "TitledGame", "TitledPolicy", "describe_action", "titled_problem", "to_site_action"]
