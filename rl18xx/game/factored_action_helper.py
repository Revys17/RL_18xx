"""Factored legal-action enumeration for the AlphaZero pipeline.

This module is a clean-room helper that lives alongside ``ActionHelper``
(``rl18xx/game/action_helper.py``). It returns categorical-only legal actions
plus a ``price_range`` field for price-bearing types instead of expanding every
discrete price into its own entry. The factored output is the canonical
representation that the new policy head, MCTS, and pretraining consume.

The existing :class:`ActionHelper` is unchanged: it still emits one action per
discrete price option (or its ``_limited`` subset) and remains the reference
for game engine tests. ``FactoredActionHelper`` defers to the same engine
primitives but collapses the price dimension.

See ``docs/step1_review.md`` -> "FactoredActionHelper: new helper class
alongside the existing one" for the design context.
"""

__all__ = ["LegalAction", "FactoredActionHelper", "categorical_parity_test"]

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rl18xx.game.engine.abilities import Shares as SharesAbility
from rl18xx.game.engine.actions import (
    Bankrupt,
    Bid,
    BuyCompany,
    BuyShares,
    BuyTrain,
    DiscardTrain,
    Dividend,
    LayTile,
    Par,
    Pass,
    PlaceToken,
    RunRoutes,
    SellShares,
)
from rl18xx.game.engine.entities import Company
from rl18xx.game.engine.round import (
    BuyTrain as BuyTrainStep,
    Exchange as ExchangeStep,
    SpecialToken as SpecialTokenStep,
    SpecialTrack as SpecialTrackStep,
    WaterfallAuction as WaterfallAuctionStep,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LegalAction:
    """A single legal-action entry in factored form.

    Attributes:
        type: Action class name, e.g. ``"Pass"``, ``"LayTile"``, ``"Bid"``,
            ``"BuyTrain"``.
        entity: Dict describing the acting entity / target of the action. Keys
            are descriptive (``"corp"``, ``"private"``, ``"company"``,
            ``"source"``, ``"train"``, etc.). Empty when the action has no
            meaningful sub-target (e.g. ``Pass``).
        params: Categorical parameters that fully describe the action modulo
            its price dimension (e.g. ``{"hex": "E14", "tile": "59",
            "rotation": 3}`` for a tile lay).
        price_range: ``(min_legal_price, max_legal_price)`` inclusive for
            price-bearing types (``Bid``, ``BuyCompany``, ``BuyTrain``).
            ``None`` for purely categorical types. For depot trains the min
            equals the max (fixed face price).
    """

    type: str
    entity: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    price_range: Optional[Tuple[int, int]] = None


# ---------------------------------------------------------------------------
# helpers shared with the existing ActionHelper. We intentionally don't import
# from ActionHelper to keep the new class self-contained, but the implementation
# mirrors ActionHelper closely for any logic we need to reproduce.
# ---------------------------------------------------------------------------


def _corp_for_company(company) -> Optional[Any]:
    """Return the corporation associated with a private company, if any."""
    for ability in company.abilities:
        if isinstance(ability, SharesAbility):
            return ability.shares[0].corporation()
    return None


def _corp_sym(corp) -> str:
    """Symbol of a corporation (Corporation.name doubles as its symbol in 1830)."""
    return getattr(corp, "sym", None) or getattr(corp, "name", None) or str(corp)


def _entity_descriptor(entity) -> dict:
    """Build a short {kind: id} descriptor for any acting entity."""
    if entity is None:
        return {}
    if isinstance(entity, Company):
        return {"private": entity.sym}
    if hasattr(entity, "is_company") and entity.is_company():
        return {"private": getattr(entity, "sym", entity.id)}
    if hasattr(entity, "is_player") and entity.is_player():
        return {"player": entity.name}
    if hasattr(entity, "is_corporation") and entity.is_corporation():
        return {"corp": _corp_sym(entity)}
    # Fallback: use repr id.
    return {"id": getattr(entity, "id", str(entity))}


def _unique_trains(trains):
    seen = {}
    for train in trains:
        key = (train.name, train.owner)
        seen.setdefault(key, train)
    return list(seen.values())


def _train_source_descriptor(train) -> dict:
    """Build entity descriptor for a BuyTrain, identifying the source."""
    if train.from_depot():
        return {"source": "depot", "train": train.name}
    owner = train.owner
    owner_sym = getattr(owner, "sym", None) or getattr(owner, "name", None) or getattr(owner, "id", str(owner))
    return {"source": owner_sym, "train": train.name}


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------


class FactoredActionHelper:
    """Returns legal actions in factored (categorical + price_range) form.

    This is the AlphaZero-side enumeration. The Python engine and the existing
    :class:`ActionHelper` continue to be the source of truth for what's legal;
    this class just reshapes that information into the schema the policy head
    and MCTS consume.
    """

    # ----- top-level entry point -------------------------------------------------
    def get_choices(self, game) -> List[LegalAction]:
        """Return all legal actions for the current game state in factored form."""
        if game.finished:
            return []

        actions: List[LegalAction] = []
        for action_cls in self._current_actions(game):
            actions.extend(self._choices_for_action(game, action_cls))
        actions.extend(self._company_choices(game))

        if not actions:
            entity = game.current_entity
            owner = getattr(entity, "owner", None)
            if owner is not None and game.can_go_bankrupt(owner, entity):
                actions.append(LegalAction(type="Bankrupt", entity=_entity_descriptor(entity)))
        return actions

    # ----- per-action-type emitters ---------------------------------------------
    def _current_actions(self, game):
        if isinstance(game.current_entity, Company):
            if Pass in game.round.actions_for(game.current_entity):
                return [Pass]
            return []
        return game.round.actions_for(game.current_entity)

    def _choices_for_action(self, game, action_cls) -> List[LegalAction]:
        if action_cls is Pass:
            return self._pass_choices(game)
        if action_cls is Bid:
            return self._bid_choices(game)
        if action_cls is Par:
            return self._par_choices(game)
        if action_cls is BuyShares:
            return self._buy_shares_choices(game)
        if action_cls is SellShares:
            return self._sell_shares_choices(game)
        if action_cls is PlaceToken:
            return self._place_token_choices(game)
        if action_cls is LayTile:
            return self._lay_tile_choices(game)
        if action_cls is BuyTrain:
            return self._buy_train_choices(game)
        if action_cls is DiscardTrain:
            return self._discard_train_choices(game)
        if action_cls is RunRoutes:
            return self._run_routes_choices(game)
        if action_cls is Dividend:
            return self._dividend_choices(game)
        if action_cls is BuyCompany:
            return self._buy_company_choices(game)
        # Unknown / not modelled: emit nothing rather than crash.
        return []

    # ----- categorical types -----------------------------------------------------
    def _pass_choices(self, game) -> List[LegalAction]:
        return [LegalAction(type="Pass", entity=_entity_descriptor(game.current_entity))]

    def _par_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        par_values = game.share_prices
        out: List[LegalAction] = []
        entity_desc = _entity_descriptor(game.current_entity)

        if hasattr(step, "companies_pending_par"):
            for company in step.companies_pending_par:
                corp = _corp_for_company(company)
                if corp is None:
                    continue
                for price in par_values:
                    out.append(
                        LegalAction(
                            type="Par",
                            entity={**entity_desc, "corp": _corp_sym(corp)},
                            params={"par_price": price.price, "private": company.sym},
                        )
                    )
            return out

        parable = sorted(
            [corp for corp in game.corporations if game.can_par(corp, game.current_entity)],
            key=lambda c: c.name,
        )
        buying_power = game.buying_power(game.current_entity)
        for corp in parable:
            for price in par_values:
                if 2 * price.price > buying_power:
                    continue
                out.append(
                    LegalAction(
                        type="Par",
                        entity={**entity_desc, "corp": _corp_sym(corp)},
                        params={"par_price": price.price},
                    )
                )
        return out

    def _buy_shares_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        buyable = step.buyable_shares(game.current_entity)
        # buyable_shares returns groups; pick the lowest-index share from each.
        unique = sorted(
            [min(group, key=lambda s: s.index) for group in buyable],
            key=lambda share: (share.corporation().name, share.owner.__class__.__name__),
        )
        out: List[LegalAction] = []
        entity_desc = _entity_descriptor(game.current_entity)
        for share in unique:
            corp = share.corporation()
            # IPO shares can be owned by either the corporation itself (the
            # default) or, for 1830's B&O and similar configurations, the Bank
            # — whichever entity is set as the corp's ipo_owner. The market
            # branch is everything else (the SharePool, primarily).
            source = "ipo" if share.owner is corp.ipo_owner else "market"
            out.append(
                LegalAction(
                    type="BuyShares",
                    entity={**entity_desc, "corp": _corp_sym(corp)},
                    params={"source": source, "percent": share.percent},
                )
            )
        return out

    def _sell_shares_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        bundles = step.sellable_shares(game.current_entity)
        if isinstance(step, BuyTrainStep):
            bundles = [b for b in bundles if step.sellable_bundle(b)]

        out: List[LegalAction] = []
        for bundle in bundles:
            corp = bundle.corporation
            owner_desc = _entity_descriptor(bundle.owner)
            count = bundle.num_shares()
            out.append(
                LegalAction(
                    type="SellShares",
                    entity={**owner_desc, "corp": _corp_sym(corp)},
                    params={"count": int(count), "percent": int(bundle.percent)},
                )
            )
        return out

    def _place_token_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        entity_desc = _entity_descriptor(game.current_entity)
        out: List[LegalAction] = []

        if hasattr(step, "pending_token") and step.pending_token:
            hexes = step.pending_token.get("hexes")
            for hex_ in hexes:
                for city in hex_._tile.cities:
                    slot = city.get_slot(game.current_entity)
                    # A city with no open slot and no reservation for this
                    # corporation cannot accept the token — the engine rejects
                    # it at process time ("no token slots available"), so
                    # emitting it would mark an unapplyable policy index as
                    # legal (and a self-play playout that samples it dies).
                    # The home city always carries the corp's reservation, so
                    # at least one entry survives this filter.
                    if slot is None:
                        continue
                    out.append(
                        LegalAction(
                            type="PlaceToken",
                            entity=entity_desc,
                            params={
                                "hex": hex_.id,
                                "city": city.index,
                                "slot": slot,
                            },
                        )
                    )
            return out

        for city in game.graph.connected_nodes(game.current_entity):
            if not city.tokenable(game.current_entity):
                continue
            hex_id = city.hex.id if hasattr(city, "hex") else None
            out.append(
                LegalAction(
                    type="PlaceToken",
                    entity=entity_desc,
                    params={
                        "hex": hex_id,
                        "city": getattr(city, "index", None),
                        "slot": city.get_slot(game.current_entity),
                    },
                )
            )
        return out

    def _lay_tile_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        entity = game.current_entity
        entity_desc = _entity_descriptor(entity)
        out: List[LegalAction] = []

        moves: Dict[Any, Dict[Any, List[int]]] = defaultdict(dict)
        for hex_ in game.graph.connected_hexes(entity):
            layable_tiles = [
                tile
                for tile in step.upgradeable_tiles(entity, hex_)
                if game.upgrade_cost(hex_.tile, hex_, entity, entity) <= game.buying_power(entity)
                and not step.ability_blocking_hex(entity, hex_)
            ]
            for tile in layable_tiles:
                rotations = step.legal_tile_rotations(entity, hex_, tile)
                moves[hex_][tile] = rotations

        for hex_ in moves:
            for tile in moves[hex_]:
                for rotation in moves[hex_][tile]:
                    out.append(
                        LegalAction(
                            type="LayTile",
                            entity=entity_desc,
                            params={
                                "hex": hex_.id,
                                "tile": tile.name,
                                "rotation": rotation,
                            },
                        )
                    )
        out.sort(key=lambda la: (la.params["hex"], la.params["tile"], la.params["rotation"]))
        return out

    def _discard_train_choices(self, game) -> List[LegalAction]:
        entity_desc = _entity_descriptor(game.current_entity)
        return [
            LegalAction(
                type="DiscardTrain",
                entity=entity_desc,
                params={"train": train.name, "train_id": train.id},
            )
            for train in _unique_trains(game.current_entity.trains)
        ]

    def _run_routes_choices(self, game) -> List[LegalAction]:
        # The auto-router is non-categorical (it picks routes for you). We emit
        # a single opaque entry; the policy head treats this as a no-param type.
        return [LegalAction(type="RunRoutes", entity=_entity_descriptor(game.current_entity))]

    def _dividend_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        entity_desc = _entity_descriptor(game.current_entity)
        return [
            LegalAction(type="Dividend", entity=entity_desc, params={"kind": option})
            for option in step.dividend_options(game.current_entity).keys()
        ]

    # ----- price-bearing types ---------------------------------------------------
    def _bid_choices(self, game) -> List[LegalAction]:
        step = game.active_step()
        entity = game.current_entity
        entity_desc = _entity_descriptor(entity)
        out: List[LegalAction] = []

        auctioning = step.auctioning_company()
        if auctioning is not None:
            companies = [auctioning]
        else:
            companies = list(step.companies)

        for company in companies:
            min_bid = step.min_bid(company)
            max_bid = step.max_bid(entity, company)
            if min_bid is None or max_bid is None or max_bid < min_bid:
                continue
            # Buy-it-now (WaterfallAuction.may_purchase): price is fixed at the
            # min_bid for the head-of-queue private when no auction is active.
            if (
                isinstance(step, WaterfallAuctionStep)
                and auctioning is None
                and hasattr(step, "may_purchase")
                and step.may_purchase(company)
            ):
                max_bid = min_bid
            out.append(
                LegalAction(
                    type="Bid",
                    entity={**entity_desc, "private": company.sym},
                    price_range=(int(min_bid), int(max_bid)),
                )
            )
        return out

    def _buy_company_choices(self, game) -> List[LegalAction]:
        # 1830: a corporation buys a private from its president during operating.
        buyer = game.current_entity
        owner = getattr(buyer, "owner", None)
        if owner is None:
            return []
        companies = owner.companies
        buying_power = game.buying_power(buyer)
        entity_desc = _entity_descriptor(buyer)
        out: List[LegalAction] = []
        for company in companies:
            if game.abilities(company, "no_buy"):
                continue
            min_price = company.min_price
            max_price = min(company.max_price, buying_power)
            if max_price < min_price:
                continue
            out.append(
                LegalAction(
                    type="BuyCompany",
                    entity={**entity_desc, "private": company.sym},
                    price_range=(int(min_price), int(max_price)),
                )
            )
        return out

    def _buy_train_choices(self, game) -> List[LegalAction]:
        step = game.round.active_step()
        entity = game.current_entity
        entity_desc = _entity_descriptor(entity)
        buyable_trains = step.buyable_trains(entity)
        unique = _unique_trains(buyable_trains)

        pres_may_contribute = step.president_may_contribute(entity)
        # Cheapest-only restriction: per the engine's
        # ``check_for_cheapest_train``, this only applies when the corp's
        # cash is less than the train price (i.e. president must actually
        # contribute for this particular train). When the corp can afford
        # the train on its own, any depot train is buyable.
        cheapest_depot_names = None
        if pres_may_contribute and game.EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST:
            cheapest_depot_train = game.depot.min_depot_train
            if cheapest_depot_train:
                cheapest_depot_names = step.names_of_cheapest_variants(cheapest_depot_train)

        entity_buying_power = game.buying_power(entity)

        out: List[LegalAction] = []

        for train in unique:
            if train.from_depot():
                min_price = train.min_price()
                requires_pres_help = min_price > entity_buying_power
                if (
                    requires_pres_help
                    and cheapest_depot_names
                    and train.name not in cheapest_depot_names
                ):
                    continue
                affordable = min_price <= entity_buying_power
                if not affordable and pres_may_contribute:
                    owner = getattr(entity, "owner", None)
                    owner_buying_power = game.buying_power(owner) if owner is not None else 0
                    affordable = min_price <= (owner_buying_power + entity_buying_power)
                if not affordable:
                    continue
                out.append(
                    LegalAction(
                        type="BuyTrain",
                        entity={**entity_desc, **_train_source_descriptor(train)},
                        price_range=(int(min_price), int(min_price)),
                    )
                )
            else:
                min_p, max_p = step.spend_minmax(entity, train)
                if not pres_may_contribute:
                    entity_buying_power = game.buying_power(entity)
                    if max_p > entity_buying_power:
                        max_p = entity_buying_power
                if max_p < min_p:
                    continue
                out.append(
                    LegalAction(
                        type="BuyTrain",
                        entity={**entity_desc, **_train_source_descriptor(train)},
                        price_range=(int(min_p), int(max_p)),
                    )
                )

        # Exchange-discounted trains (e.g. the optional $800 D in 1830).
        out.extend(self._exchange_train_choices(game))
        return out

    def _exchange_train_choices(self, game) -> List[LegalAction]:
        discountable = game.discountable_trains_for(game.current_entity)
        if not discountable:
            return []

        unique = _unique_trains([d[1] for d in discountable])
        unique_discounts = [d for d in discountable if d[1] in unique]

        entity = game.current_entity
        entity_desc = _entity_descriptor(entity)
        entity_cash = game.buying_power(entity)
        can_president_help = game.round.active_step().president_may_contribute(entity)
        available_funds = entity_cash
        if can_president_help and getattr(entity, "owner", None) is not None:
            available_funds = entity_cash + game.buying_power(entity.owner)

        out: List[LegalAction] = []
        for discount in unique_discounts:
            old_train, new_train, _, exchange_price = discount[0], discount[1], discount[2], discount[3]
            if exchange_price > available_funds:
                continue
            descriptor = {**entity_desc, **_train_source_descriptor(new_train), "exchange": old_train.name}
            out.append(
                LegalAction(
                    type="BuyTrain",
                    entity=descriptor,
                    price_range=(int(exchange_price), int(exchange_price)),
                )
            )
        return out

    # ----- private-company-as-actor branch (e.g. MH exchange, special track) -----
    def _company_choices(self, game) -> List[LegalAction]:
        company_actions = self._company_actions(game)
        if not company_actions:
            return []

        out: List[LegalAction] = []
        for company, actions in company_actions.items():
            for action_cls in actions:
                if action_cls is BuyShares:
                    out.extend(self._company_buy_shares_choices(game, company))
                elif action_cls is LayTile:
                    out.extend(self._company_lay_tile_choices(game, company))
                elif action_cls is PlaceToken:
                    out.extend(self._company_place_token_choices(game, company))
        return out

    def _company_actions(self, game) -> Dict[Any, List[Any]]:
        company_actions: Dict[Any, List[Any]] = {}
        if isinstance(game.current_entity, Company):
            company_actions[game.current_entity] = list(game.round.actions_for(game.current_entity))
        elif hasattr(game.current_entity, "companies"):
            for company in game.current_entity.companies:
                company_actions[company] = list(game.round.actions_for(company))

        # MH-special: surface the MH exchange any time it's available to the
        # current player. This mirrors ActionHelper.get_company_actions.
        mh = game.company_by_id("MH")
        if mh is not None and mh.owner is not None and mh.owner.is_player():
            current_player = game.current_entity.player() if hasattr(game.current_entity, "player") else None
            if current_player is None or mh.player() == current_player:
                company_actions[mh] = list(game.round.actions_for(mh))

        return company_actions

    def _company_buy_shares_choices(self, game, company) -> List[LegalAction]:
        exchange_step = next(step for step in game.round.steps if isinstance(step, ExchangeStep))
        shares = exchange_step.exchangeable_shares(company)
        out: List[LegalAction] = []
        for share in shares:
            corp = share.corporation()
            # Match the IPO-owner-aware logic in _buy_shares_choices.
            source = "ipo" if share.owner is corp.ipo_owner else "market"
            out.append(
                LegalAction(
                    type="CompanyBuyShares",
                    entity={"private": company.sym, "corp": _corp_sym(corp)},
                    params={"source": source, "percent": share.percent},
                )
            )
        return out

    def _company_lay_tile_choices(self, game, company) -> List[LegalAction]:
        special_track_step = next(step for step in game.round.steps if isinstance(step, SpecialTrackStep))
        abilities = special_track_step.abilities(company)

        out: List[LegalAction] = []
        for ability in abilities:
            hexes = [game.hex_by_id(hex_id) for hex_id in ability.hexes]
            for hex_ in hexes:
                tiles = special_track_step.potential_tiles(company, hex_) or []
                for tile in tiles:
                    rotations = special_track_step.legal_tile_rotations(company, hex_, tile)
                    upgrade_cost = game.upgrade_cost(hex_.tile, hex_, company.owner, company.owner)
                    if upgrade_cost > game.buying_power(company.owner):
                        continue
                    for rotation in rotations:
                        out.append(
                            LegalAction(
                                type="LayTile",
                                entity={"private": company.sym},
                                params={
                                    "hex": hex_.id,
                                    "tile": tile.name,
                                    "rotation": rotation,
                                },
                            )
                        )
        return out

    def _company_place_token_choices(self, game, company) -> List[LegalAction]:
        special_token_step = next(step for step in game.round.steps if isinstance(step, SpecialTokenStep))
        ability = special_token_step.ability(company)
        if ability is None:
            return []

        out: List[LegalAction] = []
        for hex_id in ability.hexes:
            hex_ = game.hex_by_id(hex_id)
            for city in hex_.tile.cities:
                out.append(
                    LegalAction(
                        type="PlaceToken",
                        entity={"private": company.sym},
                        params={
                            "hex": hex_id,
                            "city": getattr(city, "index", None),
                            "slot": city.get_slot(game.current_entity),
                        },
                    )
                )
        return out


# ---------------------------------------------------------------------------
# parity-test helper
# ---------------------------------------------------------------------------


def _categorical_key(action_obj) -> Optional[Tuple]:
    """Categorical fingerprint of an :class:`ActionHelper` action instance.

    Returns ``None`` for action types the factored helper does not model.
    For price-bearing types the price field is dropped: every concrete price
    collapses to the same fingerprint.
    """
    name = type(action_obj).__name__

    if name == "Pass":
        return ("Pass",)
    if name == "Bid":
        company_sym = action_obj.company.sym if action_obj.company else None
        return ("Bid", company_sym)
    if name == "Par":
        return ("Par", _corp_sym(action_obj.corporation), action_obj.share_price.price)
    if name == "BuyShares":
        bundle = action_obj.bundle
        corp = bundle.corporation
        # Match _buy_shares_choices: ipo_owner may be either the corp itself
        # (default) or the Bank (1830's B&O), so compare against ipo_owner.
        source = "ipo" if bundle.owner is corp.ipo_owner else "market"
        return ("BuyShares", _corp_sym(corp), source, int(bundle.percent))
    if name == "SellShares":
        bundle = action_obj.bundle
        corp = bundle.corporation
        return ("SellShares", _corp_sym(corp), int(bundle.num_shares()), int(bundle.percent))
    if name == "PlaceToken":
        hex_id = action_obj.city.hex.id if hasattr(action_obj.city, "hex") else None
        return ("PlaceToken", hex_id, getattr(action_obj.city, "index", None), action_obj.slot)
    if name == "LayTile":
        return ("LayTile", action_obj.hex.id, action_obj.tile.name, action_obj.rotation)
    if name == "BuyTrain":
        train = action_obj.train
        if train.from_depot():
            source = "depot"
        else:
            owner = train.owner
            source = getattr(owner, "sym", None) or getattr(owner, "name", None) or str(owner)
        exchange_name = action_obj.exchange.name if action_obj.exchange else None
        return ("BuyTrain", source, train.name, exchange_name)
    if name == "DiscardTrain":
        return ("DiscardTrain", action_obj.train.name)
    if name == "Dividend":
        return ("Dividend", action_obj.kind)
    if name == "BuyCompany":
        return ("BuyCompany", action_obj.company.sym)
    if name == "RunRoutes":
        return ("RunRoutes",)
    if name == "Bankrupt":
        return ("Bankrupt",)
    return None


def _factored_key(la: LegalAction) -> Optional[Tuple]:
    """Categorical fingerprint of a :class:`LegalAction`, matched to
    :func:`_categorical_key` so the two can be compared as sets."""
    t = la.type
    p = la.params
    e = la.entity

    if t == "Pass":
        return ("Pass",)
    if t == "Bid":
        return ("Bid", e.get("private"))
    if t == "Par":
        return ("Par", e.get("corp"), p.get("par_price"))
    if t == "BuyShares":
        return ("BuyShares", e.get("corp"), p.get("source"), int(p.get("percent")))
    if t == "SellShares":
        return ("SellShares", e.get("corp"), int(p.get("count")), int(p.get("percent")))
    if t == "PlaceToken":
        return ("PlaceToken", p.get("hex"), p.get("city"), p.get("slot"))
    if t == "LayTile":
        return ("LayTile", p.get("hex"), p.get("tile"), p.get("rotation"))
    if t == "BuyTrain":
        return ("BuyTrain", e.get("source"), e.get("train"), e.get("exchange"))
    if t == "DiscardTrain":
        return ("DiscardTrain", p.get("train"))
    if t == "Dividend":
        return ("Dividend", p.get("kind"))
    if t == "BuyCompany":
        return ("BuyCompany", e.get("private"))
    if t == "RunRoutes":
        return ("RunRoutes",)
    if t == "Bankrupt":
        return ("Bankrupt",)
    if t == "CompanyBuyShares":
        # ActionHelper emits BuyShares for the company branch too — match that.
        return ("BuyShares", e.get("corp"), p.get("source"), int(p.get("percent")))
    return None


def categorical_parity_test(game) -> bool:
    """Verify that :class:`FactoredActionHelper` covers every categorical option
    in :meth:`ActionHelper.get_all_choices_limited`, modulo price collapsing.

    Returns ``True`` if every categorical fingerprint present in the existing
    helper's output is also emitted by ``FactoredActionHelper``. Mismatches are
    logged at WARNING level. Useful as a smoke check during development; it is
    not a substitute for the engine's own validation.
    """
    # Local import to avoid a hard dependency at module-import time.
    from rl18xx.game.action_helper import ActionHelper

    factored = FactoredActionHelper().get_choices(game)
    legacy = ActionHelper().get_all_choices_limited(game)

    legacy_keys = {k for k in (_categorical_key(a) for a in legacy) if k is not None}
    factored_keys = {k for k in (_factored_key(la) for la in factored) if k is not None}

    missing = legacy_keys - factored_keys
    if missing:
        LOGGER.warning(
            "categorical_parity_test: %d legacy categorical options missing from factored output: %s",
            len(missing),
            sorted(str(m) for m in missing),
        )
        return False
    return True
