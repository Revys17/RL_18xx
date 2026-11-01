"""The model playing seats of a hotseat game (``POST /api/move``): what to play next.

The extension's page bridge (``extension/src/bridge.js``) reads the game with the
site's own Ruby engine and says who it waits for (:class:`SiteView`); the
backend follows the same actions in the Rust engine (``LiveGame``). Usually both
name the same entity, and the model's move is searched there. They differ where
our 1830 deliberately forbids what Ruby still offers -- a train bought from
another player's corporation, a private bought from another player: the Rust
engine moves on while Ruby waits in a step that offers "pass" (the same
*implicit passes* ``scripts/ruby_replay_dump.rb`` inserts). If that entity
belongs to a seat the model plays, its move is that pass; anyone else's turn is
the human's.

Moves go back in the site's format: the Rust engine's action log is
18xx.games' export format with players numbered 1..N, so only a player entity
needs mapping to the site's player id. The bridge numbers the action, runs it
through Ruby before storing it, and refuses it if Ruby does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

LOGGER = logging.getLogger(__name__)

ENTITY_TYPES = ("player", "corporation", "company")


class AutoplayError(ValueError):
    """The model can't move here (not its seat, the engines disagree, ...)."""


@dataclass
class SiteView:
    """Who the site's Ruby engine waits for, as the page bridge reads it."""

    last_id: int  # the id of the game's last action (0 for none)
    entity_id: object  # a player's site id, or a corporation's / company's sym
    entity_type: str  # player / corporation / company
    player: object  # the site id of the player behind the entity (None: nobody)
    step: str = ""
    actions: list = field(default_factory=list)  # action types Ruby offers the entity
    finished: bool = False
    # Which train is whose on the site: {corporation: [{id, name}]} and the depot's buyable trains
    # (None: the page didn't say -- an older extension).
    trains: Optional[dict] = None
    depot: Optional[list] = None

    @classmethod
    def parse(cls, body) -> "SiteView":
        if not isinstance(body, dict):
            raise AutoplayError("site: expected the page bridge's view of the game")
        entity = body.get("entity") or {}
        entity_type = entity.get("type")
        if not body.get("finished") and entity_type not in ENTITY_TYPES:
            raise AutoplayError(f"site: unknown entity type {entity_type!r}")
        try:
            last_id = int(body.get("last_id") or 0)
        except (TypeError, ValueError):
            raise AutoplayError("site: last_id must be a number") from None
        return cls(
            last_id=last_id,
            entity_id=entity.get("id"),
            entity_type=entity_type or "",
            player=entity.get("player"),
            step=str(body.get("step") or ""),
            actions=[str(a) for a in body.get("actions") or []],
            finished=bool(body.get("finished")),
            trains=body.get("trains") if isinstance(body.get("trains"), dict) else None,
            depot=body.get("depot") if isinstance(body.get("depot"), list) else None,
        )


@dataclass
class Plan:
    """What the model plays: ``search`` (a searched or forced move at the engine's
    position) or ``pass`` (an implicit pass Ruby waits for; ``action`` is it)."""

    kind: str
    seat: int
    action: Optional[dict] = None
    reason: str = ""


def _engine_entity(live) -> Optional[tuple]:
    """``(type, id, seat)`` of the entity the Rust engine has to move (players by seat)."""
    acting = live.acting()
    if not acting:
        return None
    kind = acting["kind"]
    if kind == "player":
        return "player", int(acting["id"]), acting.get("player_id")
    return kind, str(acting["id"]), acting.get("player_id")


def _site_seat(live, site_player) -> Optional[int]:
    if site_player is None:
        return None
    return live.player_mapping.get(site_player)


def plan_move(live, site: SiteView, seats) -> Plan:
    """The model's next move for one of ``seats`` (1-based, the site's player order),
    or :class:`AutoplayError` when it has none to make."""
    seats = {int(s) for s in seats}
    if site.finished:
        raise AutoplayError("The game is over.")
    seat = _site_seat(live, site.player)
    if seat is None:
        raise AutoplayError(f"The site waits for {site.entity_id}, which no player controls.")
    if seat not in seats:
        raise AutoplayError(f"It is {live.names.get(seat, seat)}'s turn, a seat the model doesn't play.")
    if live.error:
        raise AutoplayError(
            f"The engine could not follow action {live.error['action']} ({live.error['type']}: "
            f"{live.error['message']}); undo it on the site to continue."
        )
    if live.finished:
        raise AutoplayError("The engine has the game over while the site doesn't.")
    site_entity = ("player", seat) if site.entity_type == "player" else (site.entity_type, str(site.entity_id))
    engine = _engine_entity(live)
    if engine is not None and engine[:2] == site_entity:
        return Plan("search", seat)
    if "pass" in site.actions:
        action = {"type": "pass", "entity": site.entity_id, "entity_type": site.entity_type}
        moving = f"{engine[1]}" if engine else "nobody"
        reason = f"Ruby waits for {site.entity_id} to pass ({site.step or 'a step'}); the engine has {moving} to move"
        return Plan("pass", seat, action=action, reason=reason)
    raise AutoplayError(
        f"The site waits for {site.entity_id} ({site.step or '?'}) but the engine has "
        f"{engine[1] if engine else 'nobody'} to move: their replays disagree."
    )


# Preprinted hexes whose two cities our engines number the other way round from Ruby
# (scripts/ruby_replay_check.py's SWAPPED_PREPRINTED_CITIES): until the hex gets a
# tile, a city reference crossing between the engines has its index swapped.
SWAPPED_PREPRINTED_CITIES = ("G19",)


def _swap_index(ref, hex_id: str, sep_count: int):
    parts = str(ref).split("-")
    if parts[0] != hex_id or len(parts) != sep_count + 1 or parts[-1] not in ("0", "1"):
        return ref
    parts[-1] = "1" if parts[-1] == "0" else "0"
    return "-".join(parts)


def swap_city_refs(action: dict, hexes) -> dict:
    """``action`` with its references to the cities of ``hexes`` (still preprinted)
    in the other engine's numbering: a token's city (``G19-0-<i>``, the preprinted
    tile's id) and routes' nodes (``G19-<i>``). Its own inverse; a new dict when
    anything changes."""
    out = action
    for hx in hexes:
        if action.get("type") == "place_token" and str(action.get("city", "")).startswith(f"{hx}-0-"):
            out = {**out, "city": _swap_index(out["city"], hx, 2)}
        if action.get("type") == "run_routes" and action.get("routes"):
            routes = []
            for route in out["routes"]:
                if isinstance(route, dict) and route.get("nodes"):
                    route = {**route, "nodes": [_swap_index(n, hx, 1) for n in route["nodes"]]}
                routes.append(route)
            out = {**out, "routes": routes}
    return out


def swap_city_refs_in_log(actions: list) -> list:
    """An action list translated between the engines' city numberings, each action
    as of its own position (a hex stops being swapped once a tile is laid on it)."""
    pending = list(SWAPPED_PREPRINTED_CITIES)
    out = []
    for action in actions:
        out.append(swap_city_refs(action, pending) if pending and isinstance(action, dict) else action)
        if isinstance(action, dict) and action.get("type") == "lay_tile" and action.get("hex") in pending:
            pending.remove(action["hex"])
    return out


def to_site_action(action: dict, live, log) -> dict:
    """A Rust engine action dict as the site stores it: players by their site id,
    cities in Ruby's numbering (``log``: the engine's actions before it), without
    the engine's action id (the site numbers it)."""
    out = {k: v for k, v in action.items() if k not in ("id", "created_at", "user")}
    if out.get("entity_type") == "player":
        out["entity"] = live.user_id(int(out["entity"]))
    laid = {a.get("hex") for a in log if a.get("type") == "lay_tile"}
    return swap_city_refs(out, [hx for hx in SWAPPED_PREPRINTED_CITIES if hx not in laid])


def match_train_ids(action: dict, game, site: Optional[SiteView]) -> dict:
    """``action`` (the site's format) with every train it names -- a route's, the one bought, the one
    traded in, the one discarded -- replaced by the site's train of the same name and owner, when
    the engines' ids for them differ. Trains of one name and owner are interchangeable, but if the
    engines ever disagree about which id is which (an import that swapped two), the site refuses a
    move naming "another corporation's train". ``game``: the engine's position before the move.
    Raises :class:`AutoplayError` naming the difference when the site has no such train."""
    if site is None or site.trains is None:
        return action
    out = dict(action)
    kind = out.get("type")
    entity = str(out.get("entity"))

    def name_of(train_id):
        train = game.train_by_id(train_id)
        if train is None:
            raise AutoplayError(f"The engine has no train {train_id}.")
        return train, str(train.name)

    def owned(owner_id):
        return list(site.trains.get(owner_id, []))

    def pick(candidates, train_id, name, where, taken=()):
        ids = [c["id"] for c in candidates if c.get("name") == name and c["id"] not in taken]
        if train_id in ids:
            return train_id
        if not ids:
            have = ", ".join(f"{c['id']}" for c in candidates) or "none"
            raise AutoplayError(
                f"The engines disagree about trains: the engine moves {train_id} ({name}) of {where}, "
                f"but on the site {where} has no {name}-train to use (it has: {have})."
            )
        LOGGER.warning("autoplay: train %s of %s is %s on the site", train_id, where, ids[0])
        return ids[0]

    if kind == "run_routes" and out.get("routes"):
        taken, routes = [], []
        for route in out["routes"]:
            route = dict(route)
            _, name = name_of(route["train"])
            route["train"] = pick(owned(entity), route["train"], name, entity, taken)
            taken.append(route["train"])
            routes.append(route)
        out["routes"] = routes
    elif kind == "discard_train" and out.get("train"):
        _, name = name_of(out["train"])
        out["train"] = pick(owned(entity), out["train"], name, entity)
    elif kind == "buy_train" and out.get("train"):
        train, name = name_of(out["train"])
        owner = getattr(train, "owner", None)
        is_corporation = getattr(owner, "is_corporation", None)
        if callable(is_corporation) and is_corporation():
            out["train"] = pick(owned(str(owner.id)), out["train"], name, str(owner.id))
        elif site.depot is not None:
            out["train"] = pick(site.depot, out["train"], name, "the depot")
        if out.get("exchange"):
            _, exchanged = name_of(out["exchange"])
            out["exchange"] = pick(owned(entity), out["exchange"], exchanged, entity)
    return out
