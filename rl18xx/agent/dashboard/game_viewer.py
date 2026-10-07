"""Step through saved games in the Python reference engine (the dashboard's game viewer).

A saved game (``rl18xx.agent.alphazero.game_records``) is its action log.
The viewer replays it in the Python engine -- the readable port of the
18xx.games rules, whose game log reads like the site's -- and serves:

- :func:`game_summary`: the whole game once -- its log (each line tagged with
  the action that wrote it), the round headings to jump between, the actions
  themselves, and the final standings.
- :func:`state_at`: the position after the first ``step`` actions, as plain
  data the page draws: the map (tiles, track, tokens), players, corporations,
  privates, the train depot and the stock market.

Every game here is a Rust-engine game; its log replays exactly in the Python
engine (``tests/test_rust_raw_actions_python_replay.py``). A game that
doesn't (a parity bug) is shown up to the action the Python engine rejects,
with the error.

Replays are cached per process: the summary per file version, and the last
few games at the step they were left at, so stepping forward applies only
the new actions (a full replay of a ~800-action game takes ~0.2 s).
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional

from rl18xx.agent.alphazero.game_records import LoadedGame, file_mtime_ns, load_game_file

LOGGER = logging.getLogger(__name__)

MAX_CACHED_SUMMARIES = 32
MAX_CACHED_GAMES = 4
# Python-engine debug lines that land in the game log at setup.
_DEBUG_LOG_LINE = re.compile(r"^(adding share|setting <)| object at 0x")
_ROUND_HEADING = re.compile(r"^-- (.*?)( --)?$")

_LOCK = threading.RLock()
_SUMMARIES: OrderedDict = OrderedDict()
_GAMES: OrderedDict = OrderedDict()


def new_python_game(num_players: int, auction_unlock: bool = False):
    """An empty Python-engine 1830 game named as every engine game here is."""
    from rl18xx.game.gamemap import GameMap

    game = GameMap().game_by_title("1830")({i: f"Player {i}" for i in range(1, num_players + 1)})
    game.auction_unlock = bool(auction_unlock)
    return game


def _apply(game, action: dict) -> None:
    from rl18xx.game.engine.actions import BaseAction

    game.process_action(BaseAction.action_from_dict(dict(action), game))


def _replay(loaded: LoadedGame, auction_unlock: bool, upto: Optional[int] = None):
    """``(game, actions applied, error or None)`` replaying ``upto`` actions (all by default)."""
    game = new_python_game(loaded.num_players, auction_unlock)
    actions = loaded.actions if upto is None else loaded.actions[:upto]
    for i, action in enumerate(actions):
        try:
            _apply(game, action)
        except Exception as e:  # the engine raises GameError and assorted lookup errors
            return game, i, f"action {i + 1} ({action.get('type')}): {type(e).__name__}: {e}"
    return game, len(actions), None


def _unlock_candidates(loaded: LoadedGame, hint: Optional[bool]) -> list:
    known = loaded.auction_unlock if loaded.auction_unlock is not None else hint
    return [bool(known)] if known is not None else [False, True]


def _log_lines(game) -> list:
    return [
        [int(entry.action_id or 0), str(entry.message)]
        for entry in game.log
        if not _DEBUG_LOG_LINE.search(str(entry.message))
    ]


def _round_markers(log: list) -> list:
    markers = []
    for step, message in log:
        match = _ROUND_HEADING.match(message)
        if not match:
            continue
        label = match.group(1).strip()
        if label.startswith("Stock Round"):
            kind = "stock"
        elif label.startswith("Operating Round"):
            kind = "operating"
        elif label.startswith("Phase"):
            kind = "phase"
            label = label.split("(")[0].strip()
        else:
            kind = "other"
        markers.append({"step": step, "label": label, "kind": kind})
    return markers


def _cache_put(cache: OrderedDict, key, value, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def game_summary(path, auction_unlock_hint: Optional[bool] = None) -> dict:
    """The whole game: ``log`` (``[step, message]``, a line written by action
    ``step``), ``rounds`` (headings with their step), ``actions``, ``steps``,
    ``playable_steps`` / ``error`` (where the Python engine stopped, if it
    did), ``auction_unlock`` (the rule variant the replay used), ``result``
    (net worth by player name) and the file's ``meta``."""
    path = Path(path)
    key = (str(path), file_mtime_ns(path), auction_unlock_hint)
    with _LOCK:
        if key in _SUMMARIES:
            _SUMMARIES.move_to_end(key)
            return _SUMMARIES[key]
    loaded = load_game_file(path)
    best = None
    for unlock in _unlock_candidates(loaded, auction_unlock_hint):
        game, applied, error = _replay(loaded, unlock)
        if best is None or applied > best[2]:
            best = (unlock, game, applied, error)
        if error is None:
            break
    unlock, game, applied, error = best
    log = _log_lines(game)
    names = {player.id: player.name for player in sorted(game.players, key=lambda p: p.id)}
    summary = {
        "num_players": loaded.num_players,
        "players": [{"id": pid, "name": name} for pid, name in names.items()],
        "auction_unlock": unlock,
        "steps": len(loaded.actions),
        "playable_steps": applied,
        "error": error,
        "log": log,
        "rounds": _round_markers(log),
        "actions": loaded.actions,
        "finished": bool(game.finished),
        "result": {names.get(pid, str(pid)): value for pid, value in game.result().items()},
        "meta": loaded.meta,
    }
    with _LOCK:
        _cache_put(_SUMMARIES, key, summary, MAX_CACHED_SUMMARIES)
    return summary


def _game_at(path: Path, step: int, auction_unlock: bool):
    """The Python game after ``step`` actions, reusing a cached game at or before ``step``."""
    key = (str(path), file_mtime_ns(path), auction_unlock)
    with _LOCK:
        cached = _GAMES.pop(key, None)
    loaded = load_game_file(path)
    step = max(0, min(step, len(loaded.actions)))
    if cached is not None and cached[0] <= step:
        at, game = cached
    else:
        at, game = 0, new_python_game(loaded.num_players, auction_unlock)
    error = None
    for i in range(at, step):
        try:
            _apply(game, loaded.actions[i])
        except Exception as e:
            error = f"action {i + 1}: {type(e).__name__}: {e}"
            step = i
            # The game may be half-way through the bad action; don't reuse it.
            game, _, _ = _replay(loaded, auction_unlock, upto=i)
            break
    with _LOCK:
        _cache_put(_GAMES, key, (step, game), MAX_CACHED_GAMES)
    return game, step, error, loaded


def state_at(path, step: int, auction_unlock: bool = False) -> dict:
    """The position after the first ``step`` actions (see :func:`snapshot`)."""
    path = Path(path)
    game, step, error, loaded = _game_at(path, int(step), auction_unlock)
    state = snapshot(game)
    state["step"] = step
    state["error"] = error
    state["last_action"] = loaded.actions[step - 1] if step > 0 else None
    state["highlight"] = _action_hexes(game, state["last_action"])
    return state


# -------------------------------------------------------------------- snapshot
def _name(entity) -> Optional[str]:
    return getattr(entity, "name", None) if entity is not None else None


def _call(value):
    return value() if callable(value) else value


def _revenue(revenue) -> list:
    """``[[phase color, revenue], ...]`` with repeats folded, so a flat
    revenue is one entry and an offboard's yellow/brown values are two."""
    if isinstance(revenue, dict):
        out = []
        for color, value in revenue.items():
            if not out or out[-1][1] != value:
                out.append([color, value])
        return out
    return [["", revenue]] if revenue is not None else []


def _node_ref(tile, node) -> Optional[str]:
    for prefix, nodes in (("c", tile.cities), ("t", tile.towns), ("o", tile.offboards)):
        for i, candidate in enumerate(nodes):
            if candidate is node:
                return f"{prefix}{i}"
    return None


def _path_end(tile, end):
    if end is None:
        return None
    if type(end).__name__ == "Edge":
        return {"edge": int(end.num)}
    ref = _node_ref(tile, end)
    return {"node": ref} if ref else {"node": "center"}


def _loc(node, rotation: int) -> Optional[float]:
    """A node's drawing position as a (fractional) edge number on the placed
    tile -- the tile definition's ``loc`` turned by the tile's rotation, as
    its paths' edges are -- or None where the definition gives none (the node
    is placed from its exits). Altoona (H12) needs it: its city sits beside
    the 1-4 track that bypasses it."""
    loc = getattr(node, "loc", None)
    try:
        return (float(loc) + rotation) % 6 if loc is not None else None
    except (TypeError, ValueError):
        return None


def _hex(hex_) -> dict:
    tile = hex_.tile
    rotation = int(tile.rotation or 0)
    cities = []
    for city in tile.cities:
        cities.append(
            {
                "slots": int(city.slots),
                "tokens": [_name(getattr(token, "corporation", None)) if token else None for token in city.tokens],
                "reservations": [_name(r) for r in (city.reservations or []) if r is not None],
                "revenue": _revenue(city.revenue),
                "loc": _loc(city, rotation),
            }
        )
    upgrades = [
        {"cost": int(getattr(u, "cost", 0) or 0), "terrains": list(getattr(u, "terrains", None) or [])}
        for u in getattr(tile, "upgrades", []) or []
    ]
    label = getattr(tile, "label", None)
    return {
        "id": hex_.coordinates,
        "x": hex_.x,
        "y": hex_.y,
        "tile": tile.name,
        "color": tile.color,
        "rotation": rotation,
        "location": hex_.location_name if not getattr(hex_, "hide_location_name", False) else None,
        "label": str(label) if label is not None and str(label) != "None" else None,
        "paths": [[_path_end(tile, p.a), _path_end(tile, p.b)] for p in tile.paths],
        "cities": cities,
        "towns": [{"revenue": _revenue(town.revenue), "loc": _loc(town, rotation)} for town in tile.towns],
        "offboards": [{"revenue": _revenue(off.revenue)} for off in tile.offboards],
        "upgrades": [u for u in upgrades if u["cost"]],
    }


def _grouped(trains) -> list:
    counts = Counter()
    first = {}
    for train in trains:
        counts[train.name] += 1
        first.setdefault(train.name, train)
    return [{"name": name, "count": n, "price": first[name].price} for name, n in counts.items()]


def _round_info(game) -> dict:
    round_ = game.round
    step = round_.active_step() if round_ else None
    entity = None
    try:
        entity = game.current_entity
    except Exception:
        pass
    owner = None
    if entity is not None:
        try:
            owner = entity.player() if hasattr(entity, "player") else None
        except Exception:
            owner = None
    kind = type(round_).__name__ if round_ else None
    description = None
    try:
        if kind in ("Stock", "Operating", "Auction"):
            description = game.round_description(kind)
    except Exception:
        description = None
    operating_order = []
    if kind == "Operating":
        try:
            operating_order = [c.name for c in _call(game.operating_order)]
        except Exception:
            operating_order = []
    return {
        "kind": kind,
        "description": description,
        "step": type(step).__name__ if step is not None else None,
        "entity": _name(entity),
        "entity_player": _name(owner),
        "operating_order": operating_order,
    }


def snapshot(game) -> dict:
    """Plain data for the viewer: ``round``, ``phase``, ``bank``, ``players``,
    ``corporations``, ``companies``, ``depot``, ``market`` and ``hexes``."""
    finished = bool(game.finished)
    try:
        priority = _name(game.priority_deal_player()) if not finished else None
    except Exception:
        priority = None
    try:
        active = {p.name for p in game.active_players()} if not finished else set()
    except Exception:
        active = set()
    corporations = list(game.corporations)
    players = []
    for player in sorted(game.players, key=lambda p: p.id):  # seat order, not turn order
        try:
            cert_limit = game.cert_limit(player)
        except Exception:
            cert_limit = None
        players.append(
            {
                "id": player.id,
                "name": player.name,
                "cash": player.cash,
                "value": game.player_value(player),
                "certs": game.num_certs(player),
                "cert_limit": cert_limit,
                "shares": {c.name: player.percent_of(c) for c in corporations if player.percent_of(c)},
                "presidencies": [c.name for c in corporations if c.owner is player],
                "companies": [c.sym for c in player.companies],
                "priority": player.name == priority,
                "active": player.name in active,
                "bankrupt": bool(getattr(player, "bankrupt", False)),
            }
        )
    corp_rows = []
    for corp in corporations:
        par = corp.par_price()
        price = corp.share_price
        try:
            ipo_pct = int(corp.num_ipo_shares() * corp.share_percent)
        except Exception:
            ipo_pct = None
        try:
            market_pct = int(corp.num_market_shares() * corp.share_percent)
        except Exception:
            market_pct = None
        corp_rows.append(
            {
                "name": corp.name,
                "full_name": corp.full_name,
                "color": corp.color,
                "text_color": getattr(corp, "text_color", None) or "#ffffff",
                "ipoed": bool(corp.ipoed),
                "floated": bool(_call(corp.floated)),
                "closed": bool(_call(getattr(corp, "closed", False))),
                "president": _name(corp.owner) if corp.owner is not None and corp.owner.is_player() else None,
                "par": par.price if par else None,
                "price": price.price if price else None,
                "market_cell": list(price.coordinates) if price else None,
                "cash": corp.cash,
                "ipo_pct": ipo_pct,
                "market_pct": market_pct,
                "trains": [t.name for t in corp.trains],
                "tokens_left": sum(1 for t in corp.tokens if not t.used),
                "tokens_total": len(corp.tokens),
                "companies": [c.sym for c in corp.companies],
                "operated": bool(_call(corp.operated)),
            }
        )
    companies = [
        {
            "sym": c.sym,
            "name": c.name,
            "value": c.value,
            "revenue": c.revenue,
            "owner": _name(c.owner),
            "closed": bool(c.closed),
        }
        for c in game.companies
    ]
    depot = game.depot
    market = [
        [
            (
                {
                    "price": cell.price,
                    "types": list(getattr(cell, "types", None) or []),
                    "corporations": [c.name for c in cell.corporations],
                }
                if cell
                else None
            )
            for cell in row
        ]
        for row in game.stock_market.market
    ]
    phase = game.phase
    try:
        train_limit = phase.train_limit(corporations[0])
    except Exception:
        train_limit = None
    return {
        "finished": finished,
        "round": _round_info(game),
        "phase": {"name": phase.name, "tiles": list(phase.tiles), "train_limit": train_limit},
        "bank": game.bank.cash,
        "players": players,
        "corporations": corp_rows,
        "companies": companies,
        "depot": {
            "available": [{"name": t.name, "price": t.price} for t in depot.depot_trains()],
            "upcoming": _grouped(depot.upcoming),
            "discarded": _grouped(depot.discarded),
        },
        "market": market,
        "hexes": [_hex(h) for h in game.hexes],
    }


def _action_hexes(game, action: Optional[dict]) -> dict:
    """Hexes the last action touched (``hexes``) and the routes it ran
    (``routes``: hex chains), for the map to outline."""
    if not action:
        return {"hexes": [], "routes": []}
    hexes, routes = [], []
    if action.get("hex"):
        hexes.append(action["hex"])
    city_id = action.get("city")
    if city_id:
        city = None
        try:
            city = game.city_by_id(city_id)
        except Exception:
            city = None
        hex_ = getattr(getattr(city, "tile", None), "hex", None)
        if hex_ is not None:
            hexes.append(hex_.coordinates)
    for route in action.get("routes") or []:
        chains = [list(c) for c in route.get("connections") or [] if c]
        routes.append({"train": route.get("train"), "connections": chains, "revenue": route.get("revenue")})
        hexes.extend(h for chain in chains for h in chain)
    return {"hexes": sorted(set(hexes)), "routes": routes}
