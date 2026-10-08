"""Human-readable descriptions of 1830 moves, for the advisor.

:func:`describe_action` reads an action as the Rust engine decodes a policy
index (``BaseGame.decode_index_to_map``: ``type``, ``entity`` and the action's
fields, all strings) -- the same shape for the model's recommendations and for
the moves actually played (whose policy index is decoded the same way) -- and
a :class:`DescribeContext` with the names and the position's step, built from
the game by :meth:`DescribeContext.from_game`. Two optional keys the caller
adds: ``fixed_price`` (the move's legal price range is a single price) and
``trade_in`` (a D-train bought by trading in a train).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

# What a pass declines, by the Rust engine's active step (``active_step_type()``).
_PASS_BY_STEP = {
    "WaterfallAuction": "Pass (no bid)",
    "BuySellParShares": "Pass",
    "LayTile": "Pass: lay no tile",
    "Track": "Pass: lay no tile",
    "PlaceToken": "Pass: place no token",
    "Token": "Pass: place no token",
    "BuyTrain": "Pass: buy no train",
    "BuyCompany": "Pass: buy no private",
    "DiscardTrain": "Pass",
    "SpecialToken": "Pass: decline the token",
}


@lru_cache(maxsize=1)
def _static_names() -> tuple:
    """``(hex id -> location name, hex id -> number of cities)`` of the 1830 map
    (fixed for the game: read once from a fresh Python game)."""
    from rl18xx.game.gamemap import GameMap

    game = GameMap().game_by_title("1830")({1: "a", 2: "b", 3: "c", 4: "d"})
    locations, cities = {}, {}
    for hex_ in game.hexes:
        name = getattr(hex_, "location_name", None)
        if name:
            locations[hex_.coordinates] = str(name)
        cities[hex_.coordinates] = len(hex_.tile.cities)
    return locations, cities


@lru_cache(maxsize=1)
def map_positions() -> dict:
    """``{"layout", "hexes": {hex id: [x, y]}}``: where the 18xx.games map draws
    each 1830 hex -- the ``translate(x, y)`` of its ``<g>`` inside ``#map-hexes``
    (upstream ``assets/app/view/game/hex.rb`` ``Hex.coordinates`` with
    ``map.rb``'s ``start_pos`` = the smallest x and y plus one), so the browser
    extension can outline a recommended hex."""
    import math

    from rl18xx.game.gamemap import GameMap

    game = GameMap().game_by_title("1830")({1: "a", 2: "b", 3: "c", 4: "d"})
    layout = str(game.layout)
    size = 100
    narrow, wide = size * math.sqrt(3) / 2, size * 3 / 2
    step_x, step_y = (narrow, wide) if layout == "pointy" else (wide, narrow)
    hexes = [h for h in game.hexes if not getattr(h, "ignore_for_axes", False)]
    min_x, min_y = min(h.x for h in hexes), min(h.y for h in hexes)
    return {
        "layout": layout,
        "hexes": {
            h.coordinates: [round(step_x * (h.x - min_x) + size, 2), round(step_y * (h.y - min_y) + size, 2)]
            for h in game.hexes
        },
    }


@dataclass
class DescribeContext:
    """What a description names: players, privates, corporations, map locations,
    and the position's round and step."""

    player_names: dict = field(default_factory=dict)  # engine player id -> name
    company_names: dict = field(default_factory=dict)  # private sym -> full name
    company_owners: dict = field(default_factory=dict)  # private sym -> owner label
    corporation_presidents: dict = field(default_factory=dict)  # corporation sym -> president's name
    hex_names: dict = field(default_factory=dict)  # hex id -> location name
    city_counts: dict = field(default_factory=dict)  # hex id -> cities on its tile
    round_type: Optional[str] = None  # "Auction", "Stock" or "Operating"
    step: Optional[str] = None  # the Rust engine's active step type
    first_unsold_private: Optional[str] = None  # the auction's cheapest private: bought, not bid on

    @classmethod
    def from_game(cls, game, player_names: dict) -> "DescribeContext":
        """The context of a ``RustGameAdapter`` position; ``player_names`` maps
        engine player ids to the names to show."""
        rs = game._game
        locations, _ = _static_names()
        names = {int(pid): str(name) for pid, name in player_names.items()}

        def owner_label(owner_id: Optional[str]) -> Optional[str]:
            # "player:<id>" / "corp:<sym>" (the Rust engine's owner ids; "" for none).
            if not owner_id or ":" not in owner_id:
                return None
            kind, ident = owner_id.split(":", 1)
            if kind == "player":
                return names.get(int(ident), f"Player {ident}")
            return ident

        companies, owners = {}, {}
        unsold = []
        for company in rs.companies:
            companies[company.sym] = company.name
            owner = owner_label(company.owner) if not company.closed else None
            if owner:
                owners[company.sym] = owner
            elif not company.closed:
                unsold.append((company.value, company.sym))
        presidents = {}
        for corp in rs.corporations:
            president = owner_label(corp.owner_id_str)
            if president:
                presidents[corp.sym] = president
        try:
            step = rs.active_step_type()
        except Exception:  # a finished game has no step
            step = None
        round_type = rs.round.round_type if not game.finished else None
        return cls(
            player_names=names,
            company_names=companies,
            company_owners=owners,
            corporation_presidents=presidents,
            hex_names=dict(locations),
            city_counts=dict(_static_names()[1]),
            round_type=round_type,
            step=step,
            first_unsold_private=min(unsold)[1] if unsold and round_type == "Auction" else None,
        )

    def player(self, entity) -> str:
        try:
            pid = int(entity)
        except (TypeError, ValueError):
            return str(entity)
        return self.player_names.get(pid, f"Player {pid}")

    def company(self, sym) -> str:
        name = self.company_names.get(str(sym))
        return f"{name} ({sym})" if name else str(sym)

    def hex(self, hex_id) -> str:
        name = self.hex_names.get(str(hex_id))
        return f"{hex_id} ({name})" if name else str(hex_id)


def _int(value, default=None):
    try:
        return int(str(value).split(",")[0])
    except (TypeError, ValueError):
        return default


def _train_name(action: dict) -> str:
    variant = action.get("variant")
    if variant:
        return str(variant)
    return str(action.get("train") or "?").split("-")[0]


def _tile_name(tile) -> str:
    return str(tile or "?").split("-")[0]


def actor_kind(action: dict, ctx: DescribeContext) -> str:
    """``"player"``, ``"corporation"`` or ``"company"``: who plays ``action``."""
    entity = str(action.get("entity", ""))
    if entity.isdigit():
        return "player"
    if entity in ctx.company_names:
        return "company"
    return "corporation"


def describe_actor(action: dict, ctx: DescribeContext) -> str:
    """Who plays ``action``: a player's name, ``"PRR (alice)"`` for a corporation
    and its president, ``"MH (bob)"`` for a private and its owner."""
    entity = action.get("entity")
    kind = actor_kind(action, ctx)
    if kind == "player":
        return ctx.player(entity)
    if kind == "company":
        owner = ctx.company_owners.get(str(entity))
        return f"{entity} ({owner})" if owner else str(entity)
    president = ctx.corporation_presidents.get(str(entity))
    return f"{entity} ({president})" if president else str(entity)


def describe_action(action: dict, ctx: DescribeContext) -> str:
    """One line for a decoded action, e.g. "Lay tile #57 on H12 (Altoona), rotation 2"."""
    kind = str(action.get("type", ""))
    entity = action.get("entity")
    by_company = actor_kind(action, ctx) == "company"
    price = _int(action.get("price"))

    if kind == "pass":
        if by_company:
            return f"Pass with {ctx.company(entity)}"
        return _PASS_BY_STEP.get(ctx.step or "", "Pass")
    if kind == "bid":
        company = str(action.get("company"))
        # The waterfall's cheapest private sells at a set price ("fixed_price":
        # its legal price range is one price); the others are bid on.
        if company == ctx.first_unsold_private and action.get("fixed_price"):
            return f"Buy {ctx.company(company)} for ${price}"
        return f"Bid ${price} on {ctx.company(company)}"
    if kind == "par":
        return f"Par {action.get('corporation')} at ${_int(action.get('share_price'))}"
    if kind == "buy_shares":
        source = "the market" if str(action.get("source")) == "market" else "the IPO"
        share = f"a {action.get('percent', 10)}% {action.get('corporation')} share from {source}"
        if by_company:
            return f"Exchange {ctx.company(entity)} for {share}"
        return f"Buy {share}"
    if kind == "sell_shares":
        percent = _int(action.get("percent"), 0)
        count = percent // 10
        return f"Sell {percent}% of {action.get('corporation')} ({count} share{'s' if count != 1 else ''})"
    if kind == "lay_tile":
        tile, where = _tile_name(action.get("tile")), ctx.hex(action.get("hex"))
        lay = f"lay tile #{tile} on {where}, rotation {action.get('rotation')}"
        if by_company:
            return f"Use {ctx.company(entity)} to {lay}"
        return lay[0].upper() + lay[1:]
    if kind == "place_token":
        hex_id = str(action.get("hex"))
        where = ctx.hex(hex_id)
        city = _int(action.get("city_index"), 0)
        if ctx.city_counts.get(hex_id, 1) > 1:
            where += f", city {city + 1}"
        if by_company:
            return f"Use {ctx.company(entity)} to place a token on {where}"
        return f"Place a token on {where}"
    if kind == "run_routes":
        revenue = action.get("revenue")
        return "Run trains (the engine's best routes)" + (f" for ${revenue}" if revenue is not None else "")
    if kind == "dividend":
        return {"payout": "Pay out", "withhold": "Withhold", "half": "Pay half"}.get(
            str(action.get("kind")), f"Dividend: {action.get('kind')}"
        )
    if kind == "buy_train":
        train = _train_name(action)
        seller = str(action.get("from") or "depot")
        if action.get("trade_in") or action.get("exchange"):
            traded = action.get("exchange")
            what = f"a {str(traded).split('-')[0]}-train" if traded else "a 4, 5 or 6-train"
            return f"Buy a {train}-train for ${price}, trading in {what}"
        if seller == "depot":
            return f"Buy a {train}-train from the bank for ${price}"
        president = ctx.corporation_presidents.get(seller)
        return f"Buy a {train}-train from {seller}{f' ({president})' if president else ''} for ${price}"
    if kind == "discard_train":
        return f"Discard a {_train_name(action)}-train"
    if kind == "buy_company":
        company = str(action.get("company"))
        owner = ctx.company_owners.get(company)
        return f"Buy {ctx.company(company)}{f' from {owner}' if owner else ''} for ${price}"
    if kind == "bankrupt":
        return "Declare bankruptcy"
    fields = ", ".join(f"{k}={v}" for k, v in sorted(action.items()) if k not in ("type", "entity"))
    return f"{kind}{f' ({fields})' if fields else ''}"
