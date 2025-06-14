__all__ = [
    "BaseStep",
    "Auctioner",
    "Programmer",
    "ShareBuying",
    "EmergencyMoney",
    "Train",
    "BuyTrain",
    "PassableAuction",
    "Tokener",
    "TokenMerger",
    "AcquireCompany",
    "Assign",
    "AutomaticLoan",
    "Bankrupt",
    "BuyCompany",
    "BuySellParShares",
    "BuySellParSharesCompanies",
    "BuySellParSharesViaBid",
    "BuySingleTrainOfType",
    "CompanyPendingPar",
    "ConcessionAuction",
    "CorporateBuyShares",
    "CorporateSellShares",
    "DiscardTrain",
    "Dividend",
    "EndGame",
    "Exchange",
    "HalfPay",
    "HomeToken",
    "IssueShares",
    "Message",
    "MinorHalfPay",
    "MinorWithhold",
    "Program",
    "ProgrammerAuctionBid",
    "ProgrammerMergerPass",
    "ReduceTokens",
    "ReturnToken",
    "Route",
    "SelectionAuction",
    "SimpleDraft",
    "SingleDepotTrainBuy",
    "SpecialBuyTrain",
    "SpecialBuy",
    "SpecialChoose",
    "SpecialToken",
    "Tracker",
    "SpecialTrack",
    "Token",
    "Track",
    "TrackAndToken",
    "TrackLayWhenCompanySold",
    "WaterfallAuction",
    "BaseRound",
    "Auction",
    "Choices",
    "Draft",
    "Merger",
    "Operating",
    "Stock",
]


from .core import Assignable, GameError, Passer, pascal_to_snake, ShareHolder
from .entities import Corporation, Minor, Share
from rl18xx.game.engine.actions import (
    BuyTrain as BuyTrainAction,
    Bid,
    OperatingInfo,
    Pass,
    RunRoutes,
    SellShares,
    Par,
    BuyShares,
    AcquireCompany,
    Assign as AssignAction,
    Bankrupt as BankruptAction,
    BuyCompany as BuyCompanyAction,
    SellCompany,
    CorporateBuyShares as CorporateBuySharesAction,
    CorporateSellShares as CorporateSellSharesAction,
    DiscardTrain as DiscardTrainAction,
    Dividend as DividendAction,
    EndGame as EndGameAction,
    PlaceToken,
    LayTile,
    Log,
    Message as MessageAction,
    ProgramAuctionBid,
    ProgramBuyShares,
    ProgramIndependentMines,
    ProgramMergerPass,
    ProgramHarzbahnDraftPass,
    ProgramSharePass,
    ProgramClosePass,
    ProgramDisable,
    RemoveToken,
    SpecialBuy as SpecialBuyAction,
    ChooseAbility,
    TakeLoan,
)
from .graph import Hex, Tile, Token as TokenPiece
import logging
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


class BaseStep(Passer):
    def __init__(self, game, round, **opts):
        Passer.__init__(self)
        self.game = game
        self.log = game.log
        self.round = round
        self.opts = opts
        self.acted = False

    @property
    def description(self):
        raise NotImplementedError

    def setup(self):
        pass

    def pass_description(self):
        return "Pass"

    def actions(self, entity):
        return []

    def auto_actions(self, entity):
        pass

    def available_hex(self, entity, hex):
        pass

    def did_sell(self, corporation, entity):
        return False

    def last_acted_upon(self, corporation, entity):
        return False

    def log_pass(self, entity):
        self.log.append(f"{entity.name} passes {self.description.lower()}")

    def log_skip(self, entity):
        self.log.append(f"{entity.name} skips {self.description.lower()}")

    def process_pass(self, action):
        self.log_pass(action.entity)
        self.pass_()

    def skip(self):
        if not self.acted and self.current_entity:
            self.log_skip(self.current_entity)
        self.pass_()

    @property
    def current_actions(self):
        entity = self.current_entity
        if not entity or entity.is_closed():
            return []
        return self.actions(entity)

    @property
    def current_entity(self):
        return self.active_entities[0] if self.active_entities else None

    @property
    def active_entities(self):
        if not self.entities:
            return []
        return [self.entities[self.entity_index]]

    @property
    def round_state(self):
        return {}

    @property
    def blocking(self):
        return bool(self.blocks and self.current_actions)

    @property
    def blocks(self):
        return True

    def unpass(self):
        super().unpass()
        self.acted = False

    def help(self):
        return ""

    @property
    def auctioneer(self):
        return False

    @property
    def entities(self):
        return self.round.entities

    @property
    def entity_index(self):
        return self.round.entity_index

    def buying_power(self, entity):
        return self.game.buying_power(entity)

    def try_take_loan(self, entity, price):
        pass

    def try_take_player_loan(self, entity, price):
        pass

    def __str__(self):
        return f"<{self.__class__.__name__}>"


class Auctioner:
    def __init__(self):
        self.setup_auction()

    def setup_auction(self):
        self.bids = defaultdict(list)

    @property
    def auctioneer(self):
        return True

    def pass_description(self):
        if self.auctioning:
            return f"Pass (on {self.auctioning.id})"
        else:
            return "Pass"

    @property
    def visible(self):
        return True

    @property
    def players_visible(self):
        return True

    def remove_from_auction(self, entity):
        if self.auctioning in self.bids:
            self.bids[self.auctioning] = [bid for bid in self.bids[self.auctioning] if bid.entity != entity]
            self.resolve_bids()

    def pass_auction(self, entity):
        self.log.append(f"{entity.name} passes on {self.auctioning.name}")
        self.remove_from_auction(entity)

    def min_increment(self):
        return self.game.MIN_BID_INCREMENT

    def must_bid_increment_multiple(self):
        return self.game.MUST_BID_INCREMENT_MULTIPLE

    def may_choose(self, company):
        return False

    def may_offer(self, company):
        return False

    def current_bid_amount(self, player, company):
        for bid in self.bids[company]:
            if bid.entity == player:
                return bid.price
        return 0

    def may_bid(self, company):
        return True

    def min_bid(self, company):
        raise NotImplementedError

    def max_place_bid(self, entity, company):
        return self.max_bid(entity, company)

    def max_bid(self, entity, company):
        raise NotImplementedError

    def bid_target(self, bid):
        if bid.company:
            return bid.company
        elif bid.corporation:
            return bid.corporation
        else:
            return bid.minor

    def auctioning_company(self):
        company, _ = self.active_auction()
        return company

    def highest_bid(self, company):
        return max(self.bids[company], key=lambda bid: bid.price, default=None)

    def add_bid(self, bid):
        company = self.bid_target(bid)
        entity = bid.entity
        price = bid.price
        min_bid = self.min_bid(company)
        if price < min_bid:
            raise GameError(f"Minimum bid is {self.game.format_currency(min_bid)} for {company.name}")
        if self.must_bid_increment_multiple() and (price - min_bid) % self.game.MIN_BID_INCREMENT != 0:
            raise GameError(f"Must increase bid by a multiple of {self.game.MIN_BID_INCREMENT}")
        if price > self.max_bid(entity, company):
            raise GameError(f"Cannot afford bid. Maximum possible bid is {self.max_bid(entity, company)}")

        bids = self.bids[company]
        self.bids[company] = [b for b in bids if b.entity != entity]
        self.bids[company].append(bid)

    def replace_bid(self, bid):
        company = self.bid_target(bid)
        entity = bid.entity
        price = bid.price
        min_bid = self.min_bid(company)
        if price < min_bid:
            raise GameError(f"Minimum bid is {self.game.format_currency(min_bid)} for {company.name}")
        if self.must_bid_increment_multiple() and (price - min_bid) % self.game.MIN_BID_INCREMENT != 0:
            raise GameError(f"Must increase bid by a multiple of {self.game.MIN_BID_INCREMENT}")
        if price > self.max_bid(entity, company):
            raise GameError(f"Cannot afford bid. Maximum possible bid is {self.max_bid(entity, company)}")

        bids = self.bids[company]
        bids.clear()
        bids.append(bid)

    def reset_bids(self):
        self.bids.clear()

    def bids_for_player(self, player):
        player_bids = []

        for bids in self.bids.values():
            if self.game.ONLY_HIGHEST_BID_COMMITTED:
                highest_bid = max(bids, key=lambda bid: bid.price, default=None)
                if highest_bid and highest_bid.entity == player:
                    player_bids.append(highest_bid)
            else:
                player_bid = next((bid for bid in bids if bid.entity == player), None)
                if player_bid:
                    player_bids.append(player_bid)

        return player_bids


class Programmer:
    def programmed_auto_actions(self, entity):
        """
        Execute programmed automatic actions for the given entity.

        Args:
            entity: The entity for which to execute programmed actions.

        Returns:
            A list of new actions generated by the programmed actions.
        """
        # Assuming self.game.programmed_actions is a dictionary that maps
        # player objects to their list of programmed actions
        p_list = self.game.programmed_actions.get(entity.player, [])
        if not p_list:
            return []

        a_list = []
        for program in p_list:
            # Constructing the method name based on the program type
            method_name = f"activate_{program.type}"
            # Getting the method from the class based on the constructed name
            method = getattr(self, method_name, None)
            # If the method exists, call it with entity and program as arguments
            if method:
                new_actions = method(entity, program)
                if new_actions:
                    a_list.extend(new_actions)
        return a_list


class ShareBuying:
    def buy_shares(
        self,
        entity,
        shares,
        exchange=None,
        exchange_price=None,
        swap=None,
        allow_president_change=True,
        borrow_from=None,
        silent=None,
    ):
        self.check_legal_buy(
            entity,
            shares,
            exchange=exchange,
            swap=swap,
            allow_president_change=allow_president_change,
        )

        self.game.share_pool.buy_shares(
            entity,
            shares,
            exchange=exchange,
            exchange_price=exchange_price,
            swap=swap,
            borrow_from=borrow_from,
            allow_president_change=allow_president_change,
            silent=silent,
        )

        self.maybe_place_home_token(shares.corporation)

    def check_legal_buy(self, entity, shares, exchange=None, swap=None, allow_president_change=True):
        if not self.can_buy(entity, shares.to_bundle()) and not swap:
            raise Exception(
                f"Cannot buy a share of {shares.corporation.name if shares and shares.corporation else 'None'}"
            )

    def maybe_place_home_token(self, corporation):
        if (self.game.HOME_TOKEN_TIMING == "float" and corporation.floated) or (
            self.game.HOME_TOKEN_TIMING == "par" and corporation.ipoed
        ):
            self.game.place_home_token(corporation)

    def can_gain(self, entity, bundle, exchange=False):
        if not bundle or not entity:
            return
        if (
            bundle.owner.is_player()
            and not self.game.BUY_SHARE_FROM_OTHER_PLAYER
            and (not self.game.CORPORATE_BUY_SHARE_ALLOW_BUY_FROM_PRESIDENT or not entity.corporation)
        ):
            return False

        corporation = bundle.corporation

        return corporation.holding_ok(entity, bundle.common_percent) and (
            not corporation.counts_for_limit() or exchange or self.game.num_certs(entity) < self.game.cert_limit(entity)
        )

    def swap_buy(self, player, corporation, ipo_or_pool_share):
        pass

    def swap_sell(self, player, corporation, bundle, pool_share):
        pass


class EmergencyMoney:
    def process_sell_shares(self, action):
        if not self.can_sell(action.entity, action.bundle):
            raise GameError(f"Cannot sell shares of {action.bundle.corporation.name}")

        self.game.sell_shares_and_change_price(action.bundle)

        if hasattr(self.round, "recalculate_order"):
            self.round.recalculate_order()

    def can_sell(self, entity, bundle):
        if entity != bundle.owner:
            return False

        if not self.game.check_sale_timing(entity, bundle):
            return False

        if not self.sellable_bundle(bundle):
            return False

        if self.game.EBUY_SELL_MORE_THAN_NEEDED:
            return True

        return self.selling_minimum_shares(bundle)

    def selling_minimum_shares(self, bundle):
        seller = bundle.owner
        additional_cash_needed = self.needed_cash(seller) - self.available_cash(seller)
        next_smaller_bundle_price = bundle.price - min(share.price for share in bundle.shares)
        return next_smaller_bundle_price < additional_cash_needed

    def sellable_bundle(self, bundle):
        seller = bundle.owner
        if not bundle.can_dump(seller):
            return False

        if not self.game.share_pool.fit_in_bank(bundle):
            return False

        corporation = bundle.corporation
        if not corporation.president(seller):
            return True

        return not self.causes_president_swap(corporation, bundle)

    def president_swap_concern(self, corporation):
        return not self.game.EBUY_PRES_SWAP or corporation == self.current_entity

    def causes_president_swap(self, corporation, bundle):
        seller = bundle.owner
        share_holders = corporation.player_share_holders(corporate=True)
        remaining = share_holders.get(seller, 0) - bundle.percent
        next_highest = max((value for key, value in share_holders.items() if key != seller), default=0)
        return remaining < next_highest

    def issuable_shares(self, entity):
        if not entity.corporation:
            return []

        return self.game.emergency_issuable_bundles(entity)


from pdb import set_trace


class Train(EmergencyMoney):
    def __init__(self, game):
        self.game = game
        self.setup()

    def setup(self):
        self.depot = self.game.depot
        self.last_share_sold_price = None
        self.last_share_issued_price = None
        self.corporations_sold = []

    def can_buy_train(self, entity=None, shell=None):
        entity = entity or self.round.current_entity
        can_buy_normal = self.room(entity) and self.buying_power(entity) >= self.depot.min_price(
            entity,
            ability=self.game.abilities(entity, "train_discount", time=self.ability_timing()),
        )

        discountable_trains_allowed = self.discountable_trains_allowed(entity) and any(
            self.buying_power(entity) >= price for _, _, _, price in self.game.discountable_trains_for(entity)
        )

        return can_buy_normal or discountable_trains_allowed

    def ability_timing(self):
        return ["%current_step%", "buying_train", "owning_corp_or_turn"]

    def room(self, entity, shell=None):
        return self.game.num_corp_trains(entity) < self.game.train_limit(entity)

    def can_entity_buy_train(self, entity):
        return not entity.is_minor()

    def must_buy_train(self, entity):
        return self.game.must_buy_train(entity)

    def president_may_contribute(self, entity, shell=None):
        return self.must_buy_train(entity)

    def should_buy_train(self, entity):
        pass

    def discountable_trains_allowed(self, entity):
        return True

    def buy_train_action(self, action, entity=None, borrow_from=None):
        entity = entity or action.entity
        train = action.train
        train.variant = action.variant
        price = action.price
        exchange = action.exchange

        # Check if the train is actually buyable in the current situation
        if train.variant not in self.buyable_exchangeable_train_variants(train, entity, exchange):
            raise Exception("Not a buyable train")
        if self.must_pay_face_value(train, entity, price):
            raise Exception("Must pay face value")
        if train.owner == entity:
            raise Exception("An entity cannot buy a train from itself")

        remaining = price - self.buying_power(entity)
        if remaining > 0 and self.president_may_contribute(entity, action.shell):
            self.check_for_cheapest_train(train)

            if exchange:
                raise Exception("Cannot contribute funds when exchanging")
            if price > train.price:
                raise Exception("Cannot buy for more than cost")

            self.try_take_player_loan(entity.owner, remaining)

            player = entity.owner

            if borrow_from and player.cash < remaining:
                current_cash = player.cash
                extra_needed = remaining - current_cash
                player.spend(current_cash, entity)
                self.log.append(f"{player.name} contributes {self.game.format_currency(current_cash)}")
                borrow_from.spend(extra_needed, entity)
                self.log.append(f"{borrow_from.name} contributes {self.game.format_currency(extra_needed)}")
            else:
                player.spend(remaining, entity)
                self.log.append(f"{player.name} contributes {self.game.format_currency(remaining)}")

        self.try_take_loan(entity, price)

        if exchange:
            verb = f"exchanges a {exchange.name} for"
            self.depot.reclaim_train(exchange)
        else:
            verb = "buys"

        source = train.owner
        source_name = "The Discard" if train in self.depot.discarded else train.owner.name

        self.log.append(
            f"{entity.name} {verb} a {train.name} train for {self.game.format_currency(price)} from {source_name}"
        )

        self.game.buy_train(entity, train, price)
        self.game.phase.buying_train(entity, train, source)
        if not self.can_buy_train(entity) and self.pass_if_cannot_buy_train(entity):
            self.pass_()

    def pass_if_cannot_buy_train(self, entity):
        return True

    def can_ebuy_sell_shares(self, entity):
        return self.game.EBUY_CAN_SELL_SHARES

    def can_sell(self, entity, bundle):
        if self.game.MUST_SELL_IN_BLOCKS and bundle.corporation in self.corporations_sold:
            return False
        if self.current_entity != entity and self.must_issue_before_ebuy(self.current_entity):
            return False
        return super().can_sell(entity, bundle)

    def sellable_shares(self, entity):
        shares = []
        if entity.owner and entity.owner.is_player():
            shares.extend(
                [
                    bundle
                    for corporation in self.game.corporations
                    for bundle in self.game.bundles_for_corporation(entity.owner, corporation)
                    if self.can_sell(entity.owner, bundle)
                ]
            )

        if isinstance(entity, ShareHolder):
            shares.extend(
                [
                    bundle
                    for corporation in self.game.corporations
                    for bundle in self.game.bundles_for_corporation(entity, corporation)
                    if self.can_sell(entity, bundle)
                ]
            )

        return shares

    def process_sell_shares(self, action):
        if action.entity != self.current_entity:
            self.last_share_sold_price = action.bundle.price_per_share()
        super().process_sell_shares(action)
        if action.entity != self.current_entity:
            self.corporations_sold.append(action.bundle.corporation)

    def needed_cash(self, entity):
        return self.depot.min_depot_price if self.game.EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST else self.depot.max_depot_price

    def available_cash(self, entity):
        return self.current_entity.cash if entity == self.current_entity else entity.cash + self.current_entity.cash

    def ebuy_offer_only_cheapest_depot_train(self):
        return self.game.EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST

    def buyable_trains(self, entity):
        depot_trains = self.depot.depot_trains()
        other_trains = self.game.can_buy_train_from_others() and self.other_trains(entity) or []

        if entity.cash < self.depot.min_depot_price:
            if self.ebuy_offer_only_cheapest_depot_train():
                depot_trains = [self.depot.min_depot_train]

            if self.game.EBUY_SELL_MORE_THAN_NEEDED_LIMITS_DEPOT_TRAIN:
                depot_trains = depot_trains[:]
                depot_trains = [t for t in depot_trains if t.price >= self.spend_minmax(entity, t)[0]]

            if self.last_share_sold_price is not None:
                if self.game.EBUY_OTHER_VALUE:
                    other_trains = [t for t in other_trains if t.price >= self.spend_minmax(entity, t)[0]]
                else:
                    other_trains = []

        if entity.cash == 0 and not self.game.EBUY_OTHER_VALUE:
            other_trains = []

        other_trains = [
            t for t in other_trains if not (entity.cash < t.price and self.must_buy_at_face_value(t, entity))
        ]

        return depot_trains + other_trains

    def other_trains(self, entity):
        return self.depot.other_trains(entity)

    def buyable_exchangeable_train_variants(self, train, entity, exchange):
        if exchange:
            return self.exchangeable_train_variants(train, entity)
        else:
            return self.buyable_train_variants(train, entity)

    def buyable_train_variants(self, train, entity):
        if not any(bt.variants[bt.name] for bt in self.buyable_trains(entity)):
            return []
        return self.train_variant_helper(train, entity)

    def exchangeable_train_variants(self, train, entity):
        discount_info = self.game.discountable_trains_for(entity)
        if not any(discount_train.variants[discount_train.name] for _, discount_train, _, _ in discount_info):
            return []
        return self.train_variant_helper(train, entity)

    def train_variant_helper(self, train, entity):
        variants = [v for v in train.variants.values()]
        if train.owned_by_corporation:
            return variants

        if self.must_issue_before_ebuy(entity):
            variants = [v for v in variants if entity.cash >= v["price"]]
        return variants

    def must_issue_before_ebuy(self, corporation):
        return (
            self.game.MUST_EMERGENCY_ISSUE_BEFORE_EBUY
            and not self.last_share_issued_price
            and any(self.game.emergency_issuable_bundles(corporation))
        )

    def ebuy_president_can_contribute(self, corporation):
        if corporation.cash < self.game.depot.min_depot_price:
            return not self.must_issue_before_ebuy(corporation)
        return False

    def must_pay_face_value(self, train, entity, price):
        if train.from_depot() or not self.must_buy_at_face_value(train, entity):
            return False
        return train.price != price

    def must_buy_at_face_value(self, train, entity):
        return self.face_value_ability(entity) or self.face_value_ability(train.owner)

    def spend_minmax(self, entity, train):
        if self.game.EBUY_OTHER_VALUE and (self.buying_power(entity) < train.price):
            min_price = 1
            if self.last_share_sold_price is not None:
                min_price = self.buying_power(entity) + entity.owner.cash - self.last_share_sold_price + 1
            max_price = min(train.price, self.buying_power(entity) + entity.owner.cash)
            return [min_price, max_price]
        else:
            return [1, self.buying_power(entity)]

    def face_value_ability(self, entity):
        ability = self.game.abilities(entity, "train_buy", time="current")
        if ability:
            return ability.get("face_value", False)
        return False

    def check_for_cheapest_train(self, train):
        cheapest = self.depot.min_depot_train
        cheapest_names = self.names_of_cheapest_variants(cheapest)
        if (
            train.name not in cheapest_names
            and self.game.EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST
            and (not self.game.EBUY_OTHER_VALUE or train.from_depot)
        ):
            raise Exception(f"Cannot purchase {train.name} train: cheaper train available ({cheapest_names[0]})")

    def names_of_cheapest_variants(self, train):
        variants = train.variants
        # Sorting variants by price and getting the names of the cheapest variants
        cheapest_price = min(variants.values(), key=lambda v: v["price"])["price"]
        cheapest_variants = [name for name, variant in variants.items() if variant["price"] == cheapest_price]
        return cheapest_variants


class BuyTrain(BaseStep, Train):
    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        Train.__init__(self, game)

    def setup(self):
        BaseStep.setup(self)
        Train.setup(self)

    def actions(self, entity):
        # 1846 and a few others minors can't buy trains
        if not self.can_entity_buy_train(entity):
            return []

        # TODO: This needs to check it actually needs to sell shares.
        if entity == self.current_entity.owner and self.can_ebuy_sell_shares(self.current_entity):
            return [SellShares]

        if entity != self.current_entity:
            return []

        # TODO: Not sure this is right
        if self.president_may_contribute(entity):
            return [SellShares, BuyTrainAction]
        elif self.can_buy_train(entity):
            return [BuyTrainAction, Pass]

        return []

    @property
    def description(self):
        return "Buy Trains"

    def pass_description(self):
        return "Done (Trains)" if self.acted else "Skip (Trains)"

    def pass_(self):
        self.last_share_sold_price = None
        self.last_share_issued_price = None
        super().pass_()

    def check_spend(self, action):
        if not action.train.owned_by_corporation():
            return

        min_spend, max_spend = self.spend_minmax(action.entity, action.train)
        if min_spend <= action.price <= max_spend:
            return

        if max_spend == 0 and not self.game.EBUY_OTHER_VALUE:
            raise GameError(f"{action.entity.name} may not buy a train from another corporation.")
        else:
            raise GameError(
                f"{action.entity.name} may not spend {self.game.format_currency(action.price)} on "
                f"{action.train.owner.name}'s {action.train.name} train; may only spend between "
                f"{self.game.format_currency(min_spend)} and {self.game.format_currency(max_spend)}."
            )

    def process_buy_train(self, action):
        self.check_spend(action)
        self.buy_train_action(action)
        if not self.can_buy_train(action.entity) and self.pass_if_cannot_buy_train(action.entity):
            self.pass_()

    def swap_sell(self, player, corporation, bundle, pool_share):
        pass  # Your implementation for swap_sell goes here if needed


class PassableAuction(Auctioner):
    def __init__(self):
        super().__init__()
        self.setup_auction()

    def setup_auction(self):
        super().setup_auction()
        self.auctioning = None
        self.active_bidders = []
        self.auction_triggerer = None

    def remove_from_auction(self, entity):
        self.active_bidders.remove(entity)
        super().remove_from_auction(entity)

    def committed_cash(self, player, show_hidden=False):
        return 0

    def active_auction(self):
        company = self.auctioning
        bids = self.bids[company]
        if not bids:
            return None, None
        return company, bids

    def initial_auction_entities(self):
        return self.entities

    def auction_entity(self, entity):
        self.auctioning = entity
        min_bid_amount = self.min_bid(self.auctioning)
        self.active_bidders = [
            player
            for player in self.initial_auction_entities()
            if player == self.auction_triggerer or self.max_bid(player, self.auctioning) >= min_bid_amount
        ]
        cannot_bid = [
            player
            for player in self.initial_auction_entities()
            if not (player == self.auction_triggerer or self.max_bid(player, self.auctioning) >= min_bid_amount)
        ]
        for player in cannot_bid:
            self.game.log.append(
                f"{player.name} cannot bid {self.game.format_currency(min_bid_amount)} "
                f"and is out of the auction for {self.auctioning.name}"
            )
        self.resolve_bids()

    def selection_bid(self, bid):
        self.add_bid(bid)
        self.auction_triggerer = bid.entity
        self.auction_entity(self.bid_target(bid))

    def add_bid(self, bid):
        super().add_bid(bid)
        if self.auctioning:
            min_bid_amount = self.min_bid(self.auctioning)
            passing = [
                player
                for player in self.active_bidders
                if player != bid.entity and self.max_bid(player, self.auctioning) < min_bid_amount
            ]
            for player in passing:
                self.game.log.append(
                    f"{player.name} cannot bid {self.game.format_currency(min_bid_amount)} "
                    f"and is out of the auction for {self.auctioning.name}"
                )
                self.remove_from_auction(player)

    def win_bid(self, winner, company):
        # Don't modify @auctioning here do it in post_win_bid
        pass

    def post_win_bid(self, winner, company):
        # Anything modifying @auctioning should be done here rather than win_bid
        pass

    def resolve_bids(self):
        if not self.auctioning:
            return

        company = self.auctioning

        if not self.active_bidders:
            self.win_bid(None, company)
        else:
            if len(self.active_bidders) != 1:
                return
            if not self.bids[self.auctioning]:
                return

            winner = self.bids[self.auctioning][0]
            self.win_bid(winner, company)

        self.bids.clear()
        self.active_bidders.clear()
        self.auctioning = None
        self.post_win_bid(winner, company)


class Tokener:
    def __init__(self):
        self.setup()

    def setup(self):
        self.round.tokened = False

    @property
    def round_state(self):
        return {
            "tokened": False,
        }

    def can_place_token(self, entity):
        if self.current_entity != entity:
            return False

        if self.round.tokened:
            return False

        tokens = self.available_tokens(entity)
        if not tokens:
            return False

        min_price = self.min_token_price(tokens)
        if min_price > self.buying_power(entity):
            return False

        return self.game.token_graph_for_entity(entity).can_token(entity)

    def token_cost_override(self, entity, city_hex, slot, token):
        return None

    def available_tokens(self, entity):
        token_holder = entity.owner if entity.is_company() else entity
        return token_holder.tokens_by_type

    def can_replace_token(self, entity, token):
        return False

    def place_token(
        self,
        entity,
        city,
        token,
        connected=True,
        extra_action=False,
        special_ability=None,
        check_tokenable=True,
        spender=None,
        same_hex_allowed=False,
    ):
        hex = city.hex
        extra_action = extra_action or (special_ability and special_ability.type in ["teleport", "token"])

        if connected:
            self.check_connected(entity, city, hex)

        if (
            special_ability
            and special_ability.type == "token"
            and special_ability.city
            and special_ability.city != city.index
        ):
            raise GameError(
                f"{special_ability.owner.name} can only place token on {hex.name} city {special_ability.city}, not on city {city.index}"
            )

        if (
            special_ability
            and special_ability.type == "teleport"
            and special_ability.hexes
            and hex.id not in special_ability.hexes
        ):
            raise GameError(
                f"{special_ability.owner.name} cannot place token in {hex.name} ({hex.location_name}) with teleport"
            )

        if not extra_action and self.round.tokened:
            raise GameError("Token already placed this turn")

        token, ability = self.adjust_token_price_ability(entity, token, hex, city, special_ability=special_ability)
        tokener = entity.name
        if ability:
            if ability.owner != entity:
                tokener += f" ({ability.owner.sym})"
            entity.remove_ability(ability)

        if token.used:
            raise GameError("Token is already used")

        free = token.price == 0
        cheater = None
        extra_slot = None
        if ability and ability.type == "token":
            cheater = ability.cheater
            extra_slot = ability.extra_slot

        city.place_token(
            entity,
            token,
            free=free,
            check_tokenable=check_tokenable,
            cheater=cheater,
            extra_slot=extra_slot,
            spender=spender,
            same_hex_allowed=same_hex_allowed,
        )

        if not free:
            self.pay_token_cost(spender or entity, token.price, city)
            price_log = f" for {self.game.format_currency(token.price)}"
        else:
            price_log = ""

        hex_description = hex.location_name if hex.location_name else ""
        hex_description = f" ({hex_description})" if hex_description else ""
        self.log.append(f"{tokener} places a token on {hex.name}{hex_description}{price_log}")

        if not extra_action:
            self.round.tokened = True

        self.game.clear_token_graph_for_entity(entity)

    def pay_token_cost(self, entity, cost, city):
        entity.spend(cost, self.game.bank)

    def min_token_price(self, tokens):
        token = tokens[0]
        prices = [t.price for t in tokens]

        for ability in self.game.abilities(token.corporation, "token"):
            prices.append(ability.price(token))
            prices.append(ability.teleport_price)

        return min(price for price in prices if price is not None)

    def adjust_token_price_ability(self, entity, token, hex, city, special_ability=None):
        if special_ability and special_ability.type == "teleport":
            if not special_ability.from_owner:
                token = TokenPiece(entity)
                entity.tokens.append(token)
            token.price = 0
            return token, special_ability

        for ability in self.game.abilities(entity, "token"):
            if ability.special_only and ability != special_ability:
                continue
            if ability.hexes and hex.id not in ability.hexes:
                continue
            if ability.city and ability.city != city.index:
                continue

            if ability.neutral:
                neutral_corp = Corporation(
                    sym="N",
                    name="Neutral",
                    logo="open_city",
                    tokens=[0],
                )
                token = TokenPiece(neutral_corp, type="neutral")
            elif ability.owner.is_company() and not ability.from_owner:
                token = TokenPiece(entity)
                entity.tokens.append(token)

            if ability.teleport_price:
                token.price = ability.teleport_price
            if self.game.token_graph_for_entity(entity).reachable_hexes(entity).get(hex):
                token.price = ability.price(token)
            return token, ability

        return token, None

    def check_connected(self, entity, city, hex):
        if self.game.loading or self.game.token_graph_for_entity(entity).connected_nodes(entity).get(city):
            return
        else:
            city_string = " city {}".format(city.index) if len(hex.tile.cities) > 1 else ""
            raise GameError("Cannot place token on {}{} because it is not connected".format(hex.name, city_string))

    def tokener_available_hex(self, entity, hex):
        return self.game.token_graph_for_entity(entity).reachable_hexes(entity).get(hex)


class TokenMerger:
    def tokens_in_same_hex(self, surviving, others):
        # Are there tokens in the same hex?
        surviving_hexes = set(token.hex for token in surviving.tokens if token.hex)
        others_hexes = set(token.hex for token in self.others_tokens(others) if token.hex)
        return bool(surviving_hexes & others_hexes)

    def tokens_above_limits(self, surviving, others):
        tokens = [token.hex for token in surviving.tokens if token.hex]
        return (
            len(set(tokens)) != len(tokens)
            or self.tokens_in_same_hex(surviving, others)
            or sum(token.used for token in surviving.tokens + self.others_tokens(others))
            > self.game.LIMIT_TOKENS_AFTER_MERGER
        )

    def others_tokens(self, others):
        others = others if isinstance(others, list) else [others]
        return [token for corp in others for token in corp.tokens]

    def remove_duplicate_tokens(self, surviving, others):
        others_cities = set(token.city for token in self.others_tokens(others) if token.city)
        for token in surviving.tokens:
            if token.city in others_cities:
                token.remove()

    @property
    def round_state(self):
        return {"corporations_removing_tokens": None}

    def move_tokens_to_surviving(self, surviving, others, price_for_new_token=0, check_tokenable=True):
        used, unused = [], []
        for token in surviving.tokens:
            if token.used:
                used.append(token)
            else:
                unused.append(token)

        tokens = []
        for token in self.others_tokens(others):
            new_token = TokenPiece(surviving, price=price_for_new_token)
            if token.hex:
                used.append(new_token)
                token.swap(new_token, check_tokenable=check_tokenable)
            else:
                unused.append(new_token)
            tokens.append(new_token.hex.id if new_token.hex else None)

        if len(used) > self.game.LIMIT_TOKENS_AFTER_MERGER:
            raise Exception("Used token above limit")

        surviving.tokens.clear()
        surviving_tokens = used + sorted(unused, key=lambda x: x.price)
        surviving.tokens.extend(surviving_tokens[: self.game.LIMIT_TOKENS_AFTER_MERGER])

        # Assuming self.game.graph.clear_graph_for(surviving) clears graph for the surviving corporation
        self.game.graph.clear_graph_for(surviving)

        return tokens


class AcquireCompany(BaseStep):
    ACTIONS = [AcquireCompany, Pass]

    def actions(self, entity):
        if entity != self.current_entity:
            return []

        if self.can_acquire_company(entity):
            return self.ACTIONS

        return []

    def can_acquire_company(self, entity):
        return not self.game.purchasable_companies(entity)

    @property
    def description(self):
        return "Acquire private companies"

    @property
    def pass_description(self):
        return "Done (Acquire companies)" if self.acted else "Skip (Acquire companies)"

    def process_acquire_company(self, action):
        entity = action.entity
        company = action.company

        owner = company.owner
        if owner:
            owner.companies.remove(company)

        company.owner = entity
        entity.companies.append(company)

        self.round_state["acquired_companies"].append(company)

        self.log.append(f"{entity.name} acquires {company.name} from {owner.name}")
        self.game.company_bought(company, entity)

        if not self.game.purchasable_companies(entity):
            self.pass_()

    @property
    def round_state(self):
        return {"acquired_companies": []}


class Assign(BaseStep):
    ACTIONS = [AssignAction]

    def actions(self, entity):
        if not entity.is_company():
            return []

        assign_hexes_ability = self.game.abilities(entity, "assign_hexes")
        assign_corporation_ability = self.game.abilities(entity, "assign_corporation")

        if assign_hexes_ability or assign_corporation_ability:
            return self.ACTIONS

        return []

    def process_assign(self, action):
        company = action.entity
        target = action.target

        if target.assigned(company.id):
            raise GameError(f"{company.name} is already assigned to {target.name}")

        if isinstance(target, Hex):
            ability = self.game.abilities(company, "assign_hexes")
            if not ability:
                raise GameError(f"Could not assign {company.name} to {target.name}; :assign_hexes ability not found")

            assignable_hexes = [self.game.hex_by_id(h) for h in ability.hexes if h in self.game.hexes]
            Assignable.remove_from_all(
                assignable_hexes,
                company.id,
                lambda unassigned: self.log.append(f"{company.name} is unassigned from {unassigned.name}"),
            )
            target.assign(company.id)
            ability.use()
            self.log.append(f"{company.name} is assigned to {target.name}")
        elif isinstance(target, (Corporation, Minor)):
            assignable_corporations = [c for c in self.assignable_corporations(company) if not c.assigned(company.id)]
            if target in assignable_corporations:
                ability = self.game.abilities(company, "assign_corporation")
                if not ability:
                    raise GameError(
                        f"Could not assign {company.name} to {target.name}; no assignable corporations found"
                    )

                Assignable.remove_from_all(
                    assignable_corporations,
                    company.id,
                    lambda unassigned: self.log.append(f"{company.name} is unassigned from {unassigned.name}"),
                )
                target.assign(company.id)
                ability.use()
                self.log.append(f"{company.name} is assigned to {target.name}")
            else:
                raise GameError(f"Could not assign {company.name} to {target.name}; no assignable corporations found")
        else:
            raise GameError(f"Invalid target {target} for assigning company {company.name}")

        ability_count = ability.count if ability and hasattr(ability, "count") else None
        if ability_count is None or ability_count == 0 or not ability.closed_when_used_up:
            action.entity.close()
            self.log.append(f"{company.name} closes")

    def assignable_corporations(self, company=None):
        return [c for c in self.game.corporations if c.floated and not c.assigned(company.id if company else None)]

    def available_hex(self, entity, hex):
        if not entity.is_company():
            return None

        assign_hexes_ability = self.game.abilities(entity, "assign_hexes")
        if not assign_hexes_ability or hex.id not in assign_hexes_ability.hexes:
            return None

        if hex.assigned(entity.id):
            return None

        return list(self.game.hex_by_id(hex.id).neighbors.keys())

    @property
    def blocks(self):
        return False


class AutomaticLoan:
    def buying_power(self, entity):
        return self.game.buying_power(entity, full=True)

    def try_take_loan(self, entity, cost, **kwargs):
        if cost <= 0:
            return

        while entity.cash < cost and self.game.can_take_loan(entity):
            self.game.take_loan(entity, self.game.loans.first, **kwargs)


class Bankrupt(BaseStep):
    ACTIONS = [BankruptAction]

    def actions(self, entity):
        if entity != self.current_entity:
            return []
        return self.ACTIONS

    @property
    def description(self):
        return "Bankrupt"

    @property
    def blocks(self):
        return False

    def process_bankrupt(self, action):
        corp = action.entity
        player = self.game.acting_for_entity(corp.owner)

        if not self.game.can_go_bankrupt(player, corp):
            buying_power = self.game.format_currency(self.game.total_emr_buying_power(player, corp))
            price = self.game.format_currency(self.game.depot.min_depot_price)

            msg = (
                f"Cannot go bankrupt. {corp.name}'s cash plus {player.name}'s cash and "
                f"sellable shares total {buying_power}, and the cheapest train in the "
                f"Depot costs {price}."
            )
            raise GameError(msg)

        self.sell_bankrupt_shares(player, corp)
        if hasattr(self.round, "recalculate_order"):
            self.round.recalculate_order()

        if player.cash > 0:
            player.spend(player.cash, self.game.bank)

        self.game.declare_bankrupt(player)

    def sell_bankrupt_shares(self, player, corp):
        self.log.append(f"-- {player.name} goes bankrupt and sells remaining shares --")

        for corporation, _ in player.shares_by_corporation_sorted.items():
            if not corporation.share_price:
                continue  # Skip corporations that have not parred

            while True:
                bundles = self.game.sellable_bundles(player, corporation)
                if not bundles:
                    break  # No more sellable bundles
                bundle = max(bundles, key=lambda x: x.price)
                self.game.sell_shares_and_change_price(bundle)


class BuyCompany(BaseStep):
    ACTIONS = [BuyCompanyAction, Pass]
    ACTIONS_NO_PASS = [BuyCompanyAction]
    PASS = [Pass]

    def __init__(self, game, round, **opts):
        BaseStep.__init__(self, game, round, **opts)
        self.setup()

    def setup(self):
        self._blocks = self.opts.get("blocks", False)

    def actions(self, entity):
        if entity.is_minor():
            return []
        if self.can_buy_company(entity):
            return self.ACTIONS if self.blocks else self.ACTIONS_NO_PASS
        if self.blocks and entity.corporation and self.game.abilities(entity, passive_ok=False):
            return self.PASS
        return []

    def can_buy_company(self, entity):
        companies = self.game.purchasable_companies(entity)
        return (
            entity == self.current_entity
            and "can_buy_companies" in self.game.phase.status
            and companies
            and min(company.min_price for company in companies) <= self.buying_power(entity)
        )

    @property
    def blocks(self):
        return self._blocks

    @property
    def description(self):
        return "Buy Companies"

    def pass_description(self):
        return "Done (Companies)" if self.acted else "Skip (Companies)"

    def process_buy_company(self, action):
        entity = action.entity
        company = action.company
        price = action.price
        owner = company.owner

        self.buy_company(entity, company, price, owner)

    def buy_company(self, entity, company, price, owner):
        if not self.game.company_sellable(company):
            raise GameError(f"Cannot buy {company.name} from {owner.name}")

        min_price = company.min_price
        max_price = company.get_max_price(entity)
        if not min_price <= price <= max_price:
            raise GameError(
                f"Price must be between {self.game.format_currency(min_price)} and {self.game.format_currency(max_price)}"
            )

        log_later = []
        company.owner = entity
        if owner:
            owner.companies.remove(company)

        for ability in self.game.abilities(company, "assign_corporation", time="sold"):
            for unassigned in Assignable.remove_from_all(self.assignable_corporations, company.id):
                if unassigned.name != entity.name:
                    log_later.append(f"{company.name} is unassigned from {unassigned.name}")
            entity.assign(company.id)
            ability.use()
            log_later.append(f"{company.name} is assigned to {entity.name}")

            assigned_hex = next((h for h in self.game.hexes if h.assigned(company.id)), None)
            log_later.append(
                f"{company.name} is still assigned to {assigned_hex.name}"
                if assigned_hex
                else f"{company.name} is not assigned to a hex"
            )

        for ability in self.game.abilities(company, "revenue_change", time="sold"):
            company.revenue = ability.revenue

        company.remove_ability_when("sold")

        self.round.just_sold_company = company
        self.round.company_sellers[company] = owner

        entity.companies.append(company)
        self.pay(entity, owner, price, company)

        for log in log_later:
            self.log.append(log)

        self.game.after_sell_company(entity, company, price, owner)

    def assignable_corporations(self, company=None):
        return self.game.corporations

    @property
    def round_state(self):
        return {"just_sold_company": None, "company_sellers": {}}

    def pay(self, entity, owner, price, company):
        entity.spend(price, owner or self.game.bank)

        self.game.company_bought(company, entity)

        self.log.append(
            f"{entity.name} buys {company.name} from "
            f"{'the market' if owner is None else owner.name} for "
            f"{self.game.format_currency(price)}"
        )


class BuySellParShares(BaseStep, ShareBuying, Programmer):
    PURCHASE_ACTIONS = [BuyCompanyAction, BuyShares, Par]

    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)

    def setup(self):
        for player, corps in self.round.players_sold.items():
            for corp in list(corps.keys()):
                corps[corp] = "prev"

        self.round.players_history.setdefault(self.current_entity, {}).clear()
        self.round.current_actions.clear()
        self.round.bought_from_ipo = False

    def actions(self, entity):
        if entity != self.current_entity:
            return []
        if self.must_sell(entity):
            return [SellShares]
        actions = []
        if self.can_buy_any(entity):
            actions.append(BuyShares)
        if self.can_ipo_any(entity):
            actions.append(Par)
        if self.purchasable_companies(entity) or self.buyable_bank_owned_companies(entity):
            actions.append(BuyCompanyAction)
        if self.can_sell_any(entity):
            actions.append(SellShares)

        if actions:
            actions.append(Pass)
        return actions

    def log_pass(self, entity):
        if not self.round.current_actions:
            self.log.append(f"{entity.name} passes")
        elif self.bought() or self.sold():
            action = "to sell" if self.bought() else "to buy"
            self.log.append(f"{entity.name} declines {action} shares")

    def log_skip(self, entity):
        self.log.append(f"{entity.name} has no valid actions and passes")

    @property
    def description(self):
        sell_buy_order = self.game.SELL_BUY_ORDER
        if sell_buy_order == "sell_buy_or_buy_sell":
            return "Buy or Sell Shares"
        elif sell_buy_order == "sell_buy":
            return "Sell then Buy Shares"
        elif sell_buy_order == "sell_buy_sell":
            return "Sell/Buy/Sell Shares"

    def pass_description(self):
        if not self.round.current_actions:
            return "Pass (Share)"
        else:
            return "Done (Share)"

    @property
    def round_state(self):
        return {
            "players_sold": {},
            "players_bought": {},
            "current_actions": [],
            "bought_from_ipo": False,
            "players_history": {},
        }

    def can_buy(self, entity, bundle):
        if not bundle or not bundle.buyable:
            return False
        if entity == bundle.owner:
            return False
        corp = bundle.corporation
        return (
            self.available_cash(entity) >= self.modify_purchase_price(bundle)
            and not self.round.players_sold.get(entity, {}).get(corp, None)
            and (self.can_buy_multiple(entity, corp, bundle.owner) or not self.bought())
            and self.can_gain(entity, bundle)
        )

    def must_sell(self, entity):
        if not self.can_sell_any(entity):
            return False
        if self.game.num_certs(entity) > self.game.cert_limit(entity):
            return True
        if not self.game.can_hold_above_corp_limit(entity):
            return any(not corp.holding_ok(entity) for corp in self.game.corporations)
        return False

    def can_sell(self, entity, bundle):
        if not bundle:
            return False
        if entity != bundle.owner:
            return False
        corporation = bundle.corporation
        timing = self.game.check_sale_timing(entity, bundle)
        return (
            timing
            and not (
                self.game.TURN_SELL_LIMIT
                and (bundle.percent + self.sold_this_turn(corporation)) > self.game.TURN_SELL_LIMIT
            )
            and not (
                self.game.MUST_SELL_IN_BLOCKS
                and self.round.players_sold.get(entity, {}).get(corporation, None) == "now"
            )
            and self.can_sell_order()
            and self.game.share_pool.fit_in_bank(bundle)
            and self.can_dump(entity, bundle)
        )

    def can_dump(self, entity, bundle):
        return bundle.can_dump(entity)

    def can_sell_order(self):
        sell_buy_order = self.game.SELL_BUY_ORDER
        if sell_buy_order == "sell_buy_or_buy_sell":
            return not (
                len(set([type(a) for a in self.round.current_actions])) == 2
                and isinstance(self.round.current_actions[-1], tuple(self.PURCHASE_ACTIONS))
            )
        elif sell_buy_order == "sell_buy":
            return not self.bought()
        elif sell_buy_order == "sell_buy_sell":
            return True

    def sold_this_turn(self, corporation):
        sell_actions = [
            a for a in self.round.current_actions if isinstance(a, SellShares) and a.bundle.corporation == corporation
        ]
        return sum(a.bundle.percent for a in sell_actions)

    def did_sell(self, corporation, entity):
        return self.round.players_sold.get(entity, {}).get(corporation, None)

    def last_acted_upon(self, corporation, entity):
        return self.round.players_history[entity][corporation]

    def track_action(self, action, corporation, player_action=True):
        self.round.last_to_act = action.entity.player()
        if player_action:
            self.round.current_actions.append(action)
        self.round.players_history.setdefault(action.entity.player, {}).setdefault(corporation, []).append(action)

    def process_buy_shares(self, action):
        self.round.players_bought.setdefault(action.entity, {}).setdefault(action.bundle.corporation, 0)
        self.round.players_bought[action.entity][action.bundle.corporation] += action.bundle.percent
        if action.bundle.owner.is_corporation():
            self.round.bought_from_ipo = True
        self.buy_shares(
            action.purchase_for or action.entity,
            action.bundle,
            swap=action.swap,
            borrow_from=action.borrow_from,
            allow_president_change=self.allow_president_change(action.bundle.corporation),
        )
        self.track_action(action, action.bundle.corporation)

    def process_sell_shares(self, action):
        self.sell_shares(action.entity, action.bundle, swap=action.swap)
        self.track_action(action, action.bundle.corporation)

    def process_par(self, action):
        if action.purchase_for:
            raise GameError("Cannot par on behalf of other entities")
        share_price = action.share_price
        corporation = action.corporation
        entity = action.entity
        if not self.game.can_par(corporation, entity):
            raise GameError(f"{corporation.name} cannot be parred")
        self.game.stock_market.set_par(corporation, share_price)
        share = corporation.ipo_shares[0]
        self.round.players_bought.setdefault(entity, {}).setdefault(corporation, 0)
        self.round.players_bought[entity][corporation] += share.percent
        self.buy_shares(entity, share.to_bundle())
        self.game.after_par(corporation)
        self.track_action(action, action.corporation)

    def pass_(self):
        super().pass_()
        if self.round.current_actions:
            if self.current_entity in self.round.pass_order:
                self.round.pass_order.remove(self.current_entity)
            self.current_entity.unpass()
        else:
            self.round.pass_order.append(self.current_entity)
            self.current_entity.pass_()

    def available_cash(self, entity):
        return entity.cash

    def can_buy_multiple(self, entity, corporation, owner):
        if self.game.multiple_buy_only_from_market:
            if not owner.is_share_pool():
                return False
            if self.round.bought_from_ipo:
                return False
        return (
            corporation.buy_multiple()
            and not any(isinstance(x, Par) for x in self.round.current_actions)
            and not any(
                isinstance(x, BuyShares) and x.bundle.corporation != corporation for x in self.round.current_actions
            )
        )

    def can_sell_any(self, entity):
        return any(
            any(self.can_sell(entity, bundle) for bundle in self.game.bundles_for_corporation(entity, corporation))
            for corporation in self.game.corporations
        )

    def buyable_shares(self, entity):
        buyable_shares = []
        for corporation, shares in self.game.share_pool.shares_by_corporation.items():
            if self.can_buy_shares(entity, shares):
                buyable_shares.append(shares)

        for corporation in self.game.corporations:
            if corporation.ipoed and self.can_buy_shares(entity, corporation.shares):
                buyable_shares.append(corporation.shares)

        if self.game.BUY_SHARE_FROM_OTHER_PLAYER:
            for player in self.game.players:
                if player != entity and self.can_buy_shares(entity, player.shares):
                    buyable_shares.append(player.shares)

        return buyable_shares

    def sellable_shares(self, entity):
        return [
            bundle
            for corporation in self.game.corporations
            for bundle in self.game.bundles_for_corporation(entity, corporation)
            if self.can_sell(entity, bundle)
        ]

    def can_buy_shares(self, entity, shares):
        # set_trace()
        if not shares:
            return False

        sample_share = shares[0]
        corporation = sample_share.corporation()
        owner = sample_share.owner
        if self.round.players_sold.get(entity, {}).get(corporation, None) or (
            self.bought() and not self.can_buy_multiple(entity, corporation, owner)
        ):
            return False

        min_share = None
        for share in shares:
            if not share.buyable:
                continue
            if not min_share or share.percent < min_share.percent:
                min_share = share

        bundle = min_share.to_bundle()
        if not bundle:
            return False

        return self.available_cash(entity) >= self.modify_purchase_price(bundle) and self.can_gain(entity, bundle)

    def can_buy_any_from_market(self, entity):
        for corporation, shares in self.game.share_pool.shares_by_corporation.items():
            if self.can_buy_shares(entity, shares):
                return True
        return False

    def can_buy_any_from_ipo(self, entity):
        for corporation in self.game.corporations:
            if corporation.ipoed and self.can_buy_shares(entity, corporation.shares):
                return True
        return False

    def can_buy_any_from_player(self, entity):
        if not self.game.BUY_SHARE_FROM_OTHER_PLAYER:
            return False
        for player in self.game.players:
            if player != entity and self.can_buy_shares(entity, player.shares):
                return True
        return False

    def can_buy_any(self, entity):
        return (
            self.can_buy_any_from_market(entity)
            or self.can_buy_any_from_ipo(entity)
            or self.can_buy_any_from_player(entity)
        )

    def can_ipo_any(self, entity):
        if self.bought():
            return False
        return any(
            self.game.can_par(c, entity) and self.can_buy(entity, c.shares[0].to_bundle())
            for c in self.game.corporations
        )

    def ipo_type(self, entity):
        return "par"

    def purchasable_companies(self, entity):
        if (
            self.bought()
            or not self.available_cash(entity)
            or not self.game.phase
            or "can_buy_companies_from_other_players" not in self.game.phase.status
        ):
            return []

        return self.game.purchasable_companies(entity)

    def buyable_bank_owned_companies(self, entity):
        if not entity.is_player() or self.bought():
            return []

        return [c for c in self.game.buyable_bank_owned_companies if self.can_buy_company(entity, c)]

    def can_buy_company(self, player, company):
        return company in self.game.buyable_bank_owned_companies and self.available_cash(player) >= company.value

    def get_par_prices(self, entity, corp):
        return [p for p in self.game.stock_market.par_prices if p.price * 2 <= self.available_cash(entity)]

    def sell_shares(self, entity, shares, swap=None):
        if not self.can_sell(entity, shares) and not swap:
            raise GameError(f"Cannot sell shares of {shares.corporation.name}")

        self.round.players_sold.setdefault(shares.owner, {})[shares.corporation] = "now"
        self.game.sell_shares_and_change_price(shares, swap=swap)

    def bought(self):
        return any(x.__class__ in self.PURCHASE_ACTIONS for x in self.round.current_actions)

    def sold(self):
        return any(isinstance(x, SellShares) for x in self.round.current_actions)

    def process_buy_company(self, action):
        entity = action.entity
        company = action.company
        price = action.price
        owner = company.owner

        if owner and owner.corporation:
            raise GameError(f"Cannot buy {company.name} from {owner.name}")

        company.owner = entity
        if owner:
            owner.companies.remove(company)

        entity.companies.append(company)
        entity.spend(price, self.game.bank if not owner else owner)
        self.round.current_actions.append(action)
        self.log.append(
            f"{'-- ' if owner else ''}{entity.name} buys {company.name} from "
            f"{'the market' if not owner else owner.name} for {self.game.format_currency(price)}"
        )
        if entity.is_player():
            self.game.after_buy_company(entity, company, price)

    def auto_actions(self, entity):
        return self.programmed_auto_actions(entity)

    def corporation_secure_percent(self):
        # Most games 50% is fine, those where it's not (e.g. 1817) should subclass
        return 50

    def corporation_secure(self, corporation):
        # Can any other player steal the corporation?
        return (corporation.owner.percent_of(corporation)) >= self.corporation_secure_percent()

    def action_is_shenanigan(self, entity, other_entity, action, corporation, share_to_buy):
        corp_buying = share_to_buy.corporation if share_to_buy else None

        if isinstance(action, Par):
            if not corp_buying or self.game.check_sale_timing(entity, Share(corporation).to_bundle):
                return f"Corporation {corporation.name} parred"

        elif isinstance(action, BuyShares):
            if action.entity == corporation:
                return f"{corporation.name} redeemed a share."

            if corporation.owner == entity:
                if not self.corporation_secure(corporation):
                    return f"{other_entity.name} bought on corporation {corporation.name} and is unsecure"

                if corporation != corp_buying:
                    return f"{other_entity.name} bought on corporation {corporation.name} and is unsecure"

                percentage = corporation.owner.percent_of(corporation) + share_to_buy.percent

                if percentage <= self.corporation_secure_percent():
                    return None

                bigger_share = max(
                    [
                        s
                        for s in self.game.shares_for_corporation(corporation)
                        if s.percent > share_to_buy.percent and (s.owner != entity or s.owner != corporation.owner)
                    ],
                    key=lambda s: s.percent,
                )

                if bigger_share:
                    other_percent = action.entity.percent_of(corporation) + bigger_share.percent

                    if percentage < other_percent:
                        return f"{action.entity.player.name} has bought, shares exist that could allow them to gain presidency"

        elif isinstance(action, SellShares):
            return "Shares were sold"

        elif isinstance(action, TakeLoan):
            return f"{corporation.name} took a loan"

        else:
            return f"Unknown action {action.type} disabling for safety"

        return None

    def should_stop_applying_program(self, entity, program, share_to_buy):
        if self.must_sell(entity):
            return f"{entity.name} must sell shares"

        for other_entity, corporations in self.round.players_history.items():
            if other_entity == entity:
                continue

            for corporation, actions in corporations.items():
                for action in actions:
                    if action < program:
                        continue

                    reason = self.action_is_shenanigan(entity, other_entity, action, corporation, share_to_buy)
                    if reason:
                        return reason

        return None

    def normal_pass(self, entity):
        return True

    def activate_program_share_pass(self, entity, program):
        available_actions = self.actions(entity)

        if Pass not in available_actions:
            return None

        if not self.normal_pass(entity):
            return None

        reason = self.should_stop_applying_program(entity, program, None) if not program.unconditional else None

        if reason:
            return [ProgramDisable(entity, reason=reason)]

        return [Pass(entity)]

    def activate_program_buy_shares(self, entity, program):
        corporation = program.corporation

        # Check if end condition is met
        finished_reason = None

        if program.until_condition == "float":
            if corporation.floated():
                finished_reason = f"{corporation.name} is floated"
        elif entity.num_shares_of(corporation, ceil=False) >= program.until_condition:
            finished_reason = f"{program.until_condition} share(s) bought in {corporation.name}, end condition met"

        if finished_reason:
            actions = [ProgramDisable(entity, reason=finished_reason)]
            if program.auto_pass_after:
                actions.append(ProgramSharePass(entity))
            return actions

        available_actions = self.actions(entity)

        if BuyShares in available_actions:
            source = "market" if self.from_market(program) else self.game.ipo_name(corporation)

            shares_by_percent = [
                share
                for share in (
                    self.game.share_pool.shares_by_corporation[corporation]
                    if self.from_market(program)
                    else corporation.ipo_shares
                )
                if self.can_buy(entity, share.to_bundle())
            ]

            if not shares_by_percent:
                return [ProgramDisable(entity, reason=f"Cannot buy {corporation.name} from {source}")]

            if len(set(share.percent for share in shares_by_percent)) != 1:
                return [
                    ProgramDisable(
                        entity,
                        reason=f"Shares of different sizes exist, cannot auto buy {corporation.name} from {source}",
                    )
                ]

            share = shares_by_percent[0]

            reason = self.should_stop_applying_program(entity, program, share)
            if reason:
                return [ProgramDisable(entity, reason=reason)]

            return [BuyShares(entity, shares=share)]

        elif self.bought() and Pass in available_actions:
            if program.until_condition == "float" and corporation.floated():
                return None

            return [Pass(entity)]

    def from_market(self, program):
        return program.from_market

    def modify_purchase_price(self, bundle):
        return bundle.price

    def allow_president_change(self, corporation):
        return True


class BuySellParSharesCompanies(BuySellParShares):
    def actions(self, entity):
        if entity != self.current_entity:
            return []
        actions = []
        if self.must_sell(entity):
            actions.append(SellShares)
        if self.can_buy_any(entity):
            actions.append(BuyShares)
        if self.can_ipo_any(entity):
            actions.append(Par)
        if self.can_buy_any_companies(entity):
            actions.append(BuyCompanyAction)
        if self.can_sell_any(entity):
            actions.append(SellShares)
        if self.can_sell_any_companies(entity):
            actions.append(SellCompany)
        if actions:
            actions.append(Pass)
        return actions

    @property
    def description(self):
        sell_buy_order = self.game.SELL_BUY_ORDER
        if sell_buy_order == "sell_buy_or_buy_sell":
            return "Buy or Sell Certificates"
        elif sell_buy_order == "sell_buy":
            return "Sell then Buy Certificates"
        elif sell_buy_order == "sell_buy_sell":
            return "Sell/Buy/Sell Certificates"

    def pass_description(self):
        if not self.round.current_actions:
            return "Pass (Certificates)"
        return "Done (Certificates)"

    def purchasable_companies(self, entity):
        return []

    def can_buy_company(self, player, company):
        return not self.did_sell(company, player)

    def can_buy_any_companies(self, entity):
        if self.bought or not entity.cash > 0 or self.game.num_certs(entity) >= self.game.cert_limit(entity):
            return False
        return any(c.owner == self.game.bank and not self.did_sell(c, entity) for c in self.game.companies)

    def get_par_prices(self, entity, corp):
        return self.game.par_prices(corp)

    def process_buy_shares(self, action):
        super().process_buy_shares(action)
        self.game.check_new_layer()

    def process_buy_company(self, action):
        player = action.entity
        company = action.company
        price = action.price
        owner = company.owner

        if owner != self.game.bank:
            raise GameError(f"Cannot buy {company.name} from {owner.name}")

        company.owner = player

        player.companies.append(company)
        player.spend(price, owner)
        self.track_action(action, company)
        self.log.append(f"{player.name} buys {company.name} from {owner.name} for {self.game.format_currency(price)}")

    def can_buy(self, entity, bundle):
        if not self.game.PRESIDENT_SALES_TO_MARKET:
            return super().can_buy(entity, bundle)
        if not bundle or not bundle.buyable:
            return False
        corporation = bundle.corporation
        if (
            not entity.cash >= bundle.price
            or not self.can_gain(entity, bundle)
            or self.round.players_sold.get(entity, {}).get(corporation, None)
            or (not self.can_buy_multiple(entity, corporation, bundle.owner) and self.bought)
        ):
            return False
        return self.can_buy_presidents_share(entity, bundle, corporation)

    def can_buy_presidents_share(self, entity, share, corporation):
        if share.percent != corporation.presidents_percent or share.owner != self.game.share_pool:
            return True
        difference = share.percent - corporation.share_percent
        num_shares_needed = difference / corporation.share_percent
        existing_shares = entity.percent_of(corporation) or 0
        return existing_shares > num_shares_needed

    def can_sell(self, entity, bundle):
        if not self.game.PRESIDENT_SALES_TO_MARKET:
            return super().can_sell(entity, bundle)
        if not bundle:
            return False
        corporation = bundle.corporation
        timing = self.game.check_sale_timing(entity, bundle)
        return (
            timing
            and (
                not self.game.MUST_SELL_IN_BLOCKS
                or self.round.players_sold.get(entity, {}).get(corporation, None) != "now"
            )
            and self.can_sell_order()
            and self.game.share_pool.fit_in_bank(bundle)
            and self.can_dump(entity, bundle)
        )

    def can_dump(self, entity, bundle):
        corp = bundle.corporation
        if not bundle.presidents_share or bundle.percent >= corp.presidents_percent:
            return True
        max_shares = max(v for p, v in corp.player_share_holders.items() if p != entity)
        if max_shares > 10:
            return True
        pool_shares = self.game.share_pool.percent_of(corp) or 0
        return pool_shares > 0

    def process_sell_company(self, action):
        company = action.company
        player = action.entity
        if not self.can_sell_company(company):
            raise GameError(f"Cannot sell {company.id}")
        self.sell_company(player, company, action.price)
        self.track_action(action, company)

    def sell_price(self, entity):
        if not self.can_sell_company(entity):
            return 0
        return entity.value - self.game.COMPANY_SALE_FEE

    def can_sell_any_companies(self, entity):
        return not self.bought and self.sellable_companies(entity)

    def sellable_companies(self, entity):
        if self.game.turn <= 1 or not entity.player:
            return []
        return entity.companies

    def can_sell_company(self, entity):
        if not entity.is_company():
            return False
        if entity.owner == self.game.bank:
            return False
        if self.game.turn <= 1:
            return False
        return True

    def sell_company(self, player, company, price):
        company.owner = self.game.bank
        player.companies.remove(company)
        if price > 0:
            self.game.bank.spend(price, player)
        self.log.append(f"{player.name} sells {company.name} to bank for {self.game.format_currency(price)}")
        self.round.players_sold.setdefault(player, {})[company] = "now"


class BuySellParSharesViaBid(BuySellParShares, PassableAuction):
    def __init__(self, game, round, **kwargs):
        BuySellParShares.__init__(self, game, round, **kwargs)
        PassableAuction.__init__(self)

    def setup(self):
        self.setup_auction()
        super().setup()

    def actions(self, entity):
        if entity != self.current_entity:
            return []
        if self.auctioning:
            return [Bid, Pass]
        actions = super().actions(entity)
        if not self.bought and self.can_bid_any(entity):
            actions.append(Bid)
        if actions and Pass not in actions and not self.must_sell(entity):
            actions.append(Pass)
        return actions

    def auctioning_company(self):
        return self.auctioning

    def auctioning_corporation(self):
        if self.winning_bid:
            return self.winning_bid.corporation
        return self.auctioning

    def normal_pass(self, entity):
        return not self.auctioning

    @property
    def active_entities(self):
        if not self.auctioning:
            return super().active_entities
        index = (self.active_bidders.index(self.highest_bid(self.auctioning).entity) + 1) % len(self.active_bidders)
        return [self.active_bidders[index]]

    def log_pass(self, entity):
        if not self.auctioning:
            super().log_pass(entity)

    def pass_(self):
        if not self.auctioning:
            super().pass_()
        else:
            self.pass_auction(self.current_entity)
            self.resolve_bids()

    def process_bid(self, action):
        if self.auctioning_company():
            self.add_bid(action)
        else:
            self.selection_bid(action)

    def add_bid(self, action):
        player = action.entity
        entity = action.corporation or action.company
        price = action.price

        if self.auctioning:
            self.log.append(f"{player.name} bids {self.game.format_currency(price)} for {entity.name}")
        else:
            self.log.append(f"{player.name} auctions {entity.name} for {self.game.format_currency(price)}")
            if self.game.HOME_TOKEN_TIMING == "par" and not entity.is_company():
                self.game.place_home_token(entity)
        super().add_bid(action)
        self.resolve_bids()

    def min_bid(self, corporation):
        if self.auctioning:
            return self.highest_bid(corporation).price + self.min_increment()
        return self.MIN_BID

    def max_bid(self, player, corporation=None):
        if not corporation:
            return player.cash
        if not self.can_gain(player, corporation.shares[0].to_bundle()):
            return 0
        return player.cash

    def pass_description(self):
        if self.auctioning:
            return "Pass (Bid)"
        elif not self.round.current_actions:
            return "Pass (Share)"
        return "Done (Share)"


class BuySingleTrainOfType(BuyTrain):
    def __init__(self, game, round, **kwargs):
        BuyTrain.__init__(self, game, round, **kwargs)
        self.depot_trains_bought = []

    def setup(self):
        super().setup()
        self.depot_trains_bought = []

    def buyable_trains(self, entity):
        return [x for x in super().buyable_trains(entity) if not (x.from_depot() and x.sym in self.depot_trains_bought)]

    def process_buy_train(self, action):
        # Since the train won't be in the depot after being bought, store the state now.
        from_depot = action.train.from_depot()

        super().process_buy_train(action)

        if from_depot:
            self.depot_trains_bought.append(action.train.sym)

            if not self.buyable_trains(action.entity):
                self.pass_()


class CompanyPendingPar(BaseStep, Auctioner):
    ACTIONS = [Par]

    @property
    def description(self):
        return "Choose Corporation Par Value"

    def actions(self, entity):
        if self.current_entity == entity:
            return self.ACTIONS
        return []

    @property
    def active(self):
        return bool(self.companies_pending_par)

    @property
    def active_entities(self):
        return [self.round.companies_pending_par[0].owner] if self.round.companies_pending_par else []

    def process_par(self, action):
        share_price = action.share_price
        corporation = action.corporation
        self.game.stock_market.set_par(corporation, share_price)
        self.game.share_pool.buy_shares(action.entity, corporation.shares[0], exchange="free")
        self.game.after_par(corporation)
        self.round.companies_pending_par.pop(0)

    @property
    def companies_pending_par(self):
        return self.round.companies_pending_par

    def get_par_prices(self, entity, corp):
        return self.game.stock_market.par_prices

    @property
    def round_state(self):
        return {
            "companies_pending_par": [],
        }


class ConcessionAuction(BaseStep, Auctioner):
    ACTIONS = [Bid, Pass]

    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        Auctioner.__init__(self)
        self.setup()

    def setup(self):
        self.setup_auction()
        self.companies = self.game.initial_auction_companies

    @property
    def description(self):
        if self.auctioning:
            return "Bid on Selected Concession or Purchase Option"
        else:
            return "Bid on Concession or Purchase Option"

    def available(self):
        return [self.auctioning_company()] if self.auctioning_company() else self.companies

    def finished(self):
        return not self.companies or all(entity.passed for entity in self.entities)

    def process_pass(self, action):
        entity = action.entity

        if self.auctioning_company():
            self.pass_auction(action.entity)
        else:
            self.log.append(f"{entity.name} passes bidding")
            entity.pass_()
            self.round.next_entity_index()

    def process_bid(self, action):
        action.entity.unpass()

        if self.auctioning_company():
            self.add_bid(action)
        else:
            self.start_auction(action)

    @property
    def active_entities(self):
        if self.auctioning:
            for _, bids in self.active_auction():
                return [min(bids, key=lambda bid: bid.price).entity]

        return super().active_entities

    def actions(self, entity):
        if self.finished():
            return []

        correct = False

        for _, bids in self.active_auction():
            correct = min(bids, key=lambda bid: bid.price).entity == entity

        return self.ACTIONS if correct or entity == self.current_entity else []

    def min_bid(self, company):
        if not company:
            return None

        high_bid = self.highest_bid(company)
        if high_bid:
            return high_bid.price + self.min_increment()
        else:
            return company.min_bid

    def may_purchase(self, company):
        return False

    def committed_cash(self, player, show_hidden=False):
        return sum(bid.price for bid in self.bids_for_player(player))

    def max_bid(self, player, company):
        return player.cash

    def resolve_bids(self):
        if len(self.bids[self.auctioning]) == 1:
            bid = self.bids[self.auctioning][0]
            self.auctioning = None
            price = bid.price
            company = bid.company
            player = bid.entity
            self.bids.pop(company)
            self.buy_company(player, company, price)
            self.round.next_entity_index()

    def active_auction(self):
        company = self.auctioning
        bids = self.bids[company]
        if len(bids) > 0:
            return company, bids
        return None, None

    def can_auction(self, company):
        return company == self.companies[0] and len(self.bids[company]) > 1

    def buy_company(self, player, company, price):
        available = self.max_bid(player, company)
        if available < price:
            raise GameError(
                f"{player.name} has {self.game.format_currency(available)} "
                f"available and cannot spend {self.game.format_currency(price)}"
            )

        company.owner = player
        player.companies.append(company)
        if price > 0:
            player.spend(price, self.game.bank)
        self.companies.remove(company)
        self.log.append(
            f"{player.name} wins the auction for {company.name} " f"with a bid of {self.game.format_currency(price)}"
        )

    def start_auction(self, bid):
        self.auctioning = bid.company
        self.log.append(f"-- {bid.entity.name} nominates {self.auctioning.name} for auction --")
        self.add_bid(bid)
        starter = bid.entity
        start_price = bid.price

        bids = self.bids[self.auctioning]

        entity_idx = self.entities.index(starter)
        entities_rotated = self.entities[entity_idx:] + self.entities[:entity_idx]
        for idx, player in enumerate(entities_rotated):
            if player != starter and self.max_bid(player, self.auctioning) > start_price:
                bids.append(
                    Bid(
                        player,
                        corporation=self.auctioning,
                        price=idx - len(self.entities),
                    )
                )

    def add_bid(self, bid):
        super().add_bid(bid)
        self.log.append(f"{bid.entity.name} bids {self.game.format_currency(bid.price)} for {bid.company.name}")


class CorporateBuyShares(BaseStep, ShareBuying):
    @property
    def description(self):
        return "Corporate Share Buying"

    @property
    def round_state(self):
        return {"corporations_bought": {}}

    def actions(self, entity):
        if entity != self.current_entity:
            return []

        actions = []
        if self.can_buy_any(entity):
            actions.append(CorporateBuySharesAction)
        if actions:
            actions.append(Pass)

        return actions

    def pass_description(self):
        return "Pass (Share Buy)"

    def log_pass(self, entity):
        self.log.append(f"{entity.name} passes buying shares")

    def log_skip(self, entity):
        self.log.append(f"{entity.name} skips corporate share buy")

    def can_buy_any(self, entity):
        return self.can_buy_any_from_market(entity) or self.can_buy_any_from_president(entity)

    def can_buy_any_from_market(self, entity):
        return any(self.can_buy(entity, s.to_bundle()) for s in self.game.share_pool.shares)

    def can_buy_corp_from_market(self, entity, corporation):
        shares = self.game.share_pool.shares_by_corporation.get(corporation, [])
        return any(self.can_buy(entity, s.to_bundle()) for s in shares)

    def can_buy_any_from_president(self, entity):
        if not self.game.CORPORATE_BUY_SHARE_ALLOW_BUY_FROM_PRESIDENT:
            return False

        return any(self.can_buy(entity, s.to_bundle()) for s in entity.owner.shares)

    def can_buy(self, entity, bundle):
        if not bundle:
            return False
        if not bundle.buyable:
            return False
        if not bundle.corporation.ipoed:
            return False
        if bundle.presidents_share:
            return False
        if entity == bundle.corporation:
            return False

        if (
            self.game.CORPORATE_BUY_SHARE_SINGLE_CORP_ONLY
            and self.bought(entity)
            and bundle.corporation != self.last_bought(entity)
        ):
            return False

        return entity.cash >= bundle.price

    def process_corporate_buy_shares(self, action):
        self.buy_shares(action.entity, action.bundle)
        if action.entity not in self.round.corporations_bought:
            self.round.corporations_bought[action.entity] = []
        self.round.corporations_bought[action.entity].append(action.bundle.corporation)
        if not self.can_buy_any(action.entity):
            self.pass_()

    def source_list(self, entity):
        source = []

        if self.game.CORPORATE_BUY_SHARE_SINGLE_CORP_ONLY and self.bought(entity):
            source = [
                corp
                for corp in self.game.sorted_corporations
                if corp == self.last_bought(entity)
                and corp.num_market_shares > 0
                and self.can_buy_corp_from_market(entity, corp)
            ]
        else:
            source = [
                corp
                for corp in self.game.sorted_corporations
                if corp != entity
                and corp.floated
                and not corp.is_closed()
                and corp.num_market_shares > 0
                and self.can_buy_corp_from_market(entity, corp)
            ]

        if self.game.CORPORATE_BUY_SHARE_ALLOW_BUY_FROM_PRESIDENT and self.can_buy_any_from_president(entity):
            source.append(entity.owner)

        return source

    def bought(self, entity):
        return entity in self.round.corporations_bought

    def last_bought(self, entity):
        if entity in self.round.corporations_bought:
            return self.round.corporations_bought[entity][-1]
        return None


class CorporateSellShares(BaseStep, ShareBuying):
    @property
    def description(self):
        return "Corporate Share Sales"

    def actions(self, entity):
        if entity != self.current_entity:
            return []

        actions = []
        if self.can_sell_any(entity):
            actions.append(CorporateSellSharesAction)

        if actions:
            actions.append(Pass)

        return actions

    def pass_description(self):
        return "Pass (Share Sale)"

    def log_pass(self, entity):
        self.log.append(f"{entity.name} passes selling shares")

    def log_skip(self, entity):
        self.log.append(f"{entity.name} skips corporate share sales")

    def can_sell_any(self, entity):
        return any(self.can_sell(entity, share.to_bundle()) for share in entity.corporate_shares)

    def can_sell(self, entity, bundle):
        if not bundle:
            return False
        if entity != bundle.owner:
            return False
        if entity == bundle.corporation:
            return False
        if self.bought(entity, bundle.corporation):
            return False
        return True

    def process_corporate_sell_shares(self, action):
        self.sell_shares(action.entity, action.bundle, swap=action.swap)

        if hasattr(self.round, "recalculate_order"):
            self.round.recalculate_order()

        if not self.can_sell_any(action.entity):
            self.pass_()

    def sell_shares(self, entity, shares, swap=None):
        if not self.can_sell(entity, shares) and not swap:
            raise GameError(f"Cannot sell shares of {shares.corporation.name}")

        self.game.sell_shares_and_change_price(shares, swap=swap)

    def source_list(self, entity):
        source = []
        for share in entity.corporate_shares:
            if not self.bought(entity, share.corporation):
                source.append(share.corporation)

        return list(set(source))

    def bought(self, entity, corporation):
        return corporation in self.round.corporations_bought[entity]


class DiscardTrain(BaseStep):
    ACTIONS = [DiscardTrainAction]

    def actions(self, entity):
        return [] if entity not in self.crowded_corps else self.ACTIONS

    @property
    def active_entities(self):
        return [self.crowded_corps[0]] if self.crowded_corps else []

    def active(self):
        return bool(self.crowded_corps)

    @property
    def description(self):
        return "Discard Train"

    def process_discard_train(self, action):
        train = action.train
        self.game.depot.reclaim_train(train)
        for step in self.round.steps:
            if isinstance(step, BuyTrain):
                LOGGER.debug(f"Unpassing BuyTrain step: {step}")
                step.unpass()
        self.log.append(f"{action.entity.name} discards {train.name}")

    @property
    def crowded_corps(self):
        return self.game.crowded_corps

    def trains(self, corporation):
        return corporation.trains


class Dividend(BaseStep):
    ACTIONS = [DividendAction]

    def actions(self, entity):
        if entity.is_company() or self.total_revenue() == 0:
            return []
        return self.ACTIONS

    @property
    def round_state(self):
        super_round_state = super().round_state if hasattr(super(), "round_state") else {}
        return {
            **super_round_state,
            "extra_revenue": 0,
        }

    DIVIDEND_TYPES = ["payout", "withhold"]

    @property
    def dividend_types(self):
        return self.DIVIDEND_TYPES

    @property
    def description(self):
        return "Pay or Withhold Dividends"

    def skip(self):
        action = DividendAction(self.current_entity, kind="withhold")
        if self.game.actions:
            action.id = self.game.actions[-1].id
        self.process_dividend(action)

    def dividend_options(self, entity):
        revenue = self.total_revenue()
        options = {}
        for dividend_type in self.dividend_types:
            payout = getattr(self, dividend_type)(entity, revenue)
            payout["divs_to_corporation"] = self.corporation_dividends(entity, payout["per_share"])
            payout.update(self.share_price_change(entity, revenue - payout["corporation"]))
            options[dividend_type] = payout
        return options

    def variable_share_multiplier(self, corporation):
        return 1

    def variable_input_step(self):
        return 1

    def variable_max(self):
        return 1

    def process_dividend(self, action):
        entity = action.entity
        revenue = self.total_revenue()
        kind = action.kind
        payout = self.dividend_options(entity)[kind]

        operating_history = OperatingInfo(
            self.routes,
            action,
            revenue,
            self.round.laid_hexes,
        )

        entity.operating_history[(self.game.turn, self.round.round_num)] = operating_history

        if not isinstance(entity, str) and entity.is_company():
            self.game.close_companies_on_event(entity, "ran_train", [])
        for train in entity.trains:
            train.operated = True

        self.rust_obsolete_trains(entity)
        self.round.routes = []
        self.round.extra_revenue = 0
        self.log_run_payout(entity, kind, revenue, action, payout)
        self.payout_corporation(payout["corporation"], entity)
        if payout["per_share"] > 0:
            self.payout_shares(entity, revenue - payout["corporation"])
        self.change_share_price(entity, payout)
        self.pass_()

    def payout_corporation(self, amount, entity):
        if amount > 0:
            self.game.bank.spend(amount, entity)

    def log_run_payout(self, entity, kind, revenue, action, payout):
        if kind not in self.DIVIDEND_TYPES:
            self.log.append(f"{entity.name} runs for {self.game.format_currency(revenue)} and pays {action.kind}")

        if payout["corporation"] > 0:
            self.log.append(f"{entity.name} withholds {self.game.format_currency(payout['corporation'])}")
        elif payout["per_share"] == 0:
            self.log.append(f"{entity.name} does not run")

    def share_price_change(self, entity, revenue):
        if revenue > 0:
            return {"share_direction": "right", "share_times": 1}
        return {"share_direction": "left", "share_times": 1}

    def withhold(self, entity, revenue):
        return {"corporation": revenue, "per_share": 0}

    def payout(self, entity, revenue):
        per_share = self.payout_per_share(entity, revenue)
        return {"corporation": 0, "per_share": per_share}

    def dividends_for_entity(self, entity, holder, per_share):
        return int(holder.num_shares_of(entity, ceil=False) * per_share)

    def corporation_dividends(self, entity, per_share):
        if entity.is_minor():
            return 0
        return self.dividends_for_entity(entity, self.holder_for_corporation(entity), per_share)

    def payout_per_share(self, entity, revenue):
        return revenue / entity.total_shares

    def holder_for_corporation(self, entity):
        return entity if entity.capitalization == "incremental" else self.game.share_pool

    def payout_shares(self, entity, revenue):
        per_share = self.payout_per_share(entity, revenue)
        payouts = {}
        for payee in self.game.players + self.game.corporations:
            self.payout_entity(entity, payee, per_share, payouts)

        receivers = ", ".join(
            [f"{self.game.format_currency(cash)} to {receiver.name}" for receiver, cash in payouts.items()]
        )

        self.log_payout_shares(entity, revenue, per_share, receivers)

    def payout_entity(self, entity, holder, per_share, payouts):
        amount = 0
        if entity == holder:
            amount = self.corporation_dividends(entity, per_share)
        else:
            amount = self.dividends_for_entity(entity, holder, per_share)

        if amount > 0:
            receiver = holder if holder else None
            payouts[receiver] = amount
            self.game.bank.spend(amount, receiver, check_positive=False)

    def change_share_price(self, entity, payout):
        if not payout["share_direction"]:
            return

        if not entity.share_price:
            return

        old_price = entity.share_price

        right_times = 0
        for share_times, direction in zip([payout["share_times"]], [payout["share_direction"]]):
            for _ in range(share_times):
                if direction == "left":
                    self.game.stock_market.move_left(entity)
                elif direction == "right":
                    self.game.stock_market.move_right(entity)
                    right_times += 1
                elif direction == "up":
                    self.game.stock_market.move_up(entity)
                elif direction == "down":
                    self.game.stock_market.move_down(entity)

        self.game.log_share_price(entity, old_price, right_times)

    @property
    def routes(self):
        return self.round.routes

    def extra_revenue(self):
        return self.round.extra_revenue or 0

    def total_revenue(self):
        return self.game.routes_revenue(self.routes) + self.extra_revenue()

    def rust_obsolete_trains(self, entity):
        rusted_trains = [train for train in entity.trains if train.obsolete]
        if rusted_trains:
            self.game.rust(rusted_trains)
            self.log.append("-- Event: Obsolete trains rust --")

    def pass_(self):
        entity = self.current_entity
        if entity:
            if len(entity.operating_history) == 1:
                self.game.close_companies_on_event(entity, "operated")
            super().pass_()

    def log_payout_shares(self, entity, revenue, per_share, receivers):
        msg = f"{entity.name} pays out {self.game.format_currency(revenue)} = "
        msg += f"{self.game.format_currency(per_share)} per share"
        if receivers:
            msg += f" ({receivers})"
        self.log.append(msg)


class EndGame(BaseStep):
    ACTIONS = [EndGameAction]

    def actions(self, entity):
        if entity.is_company():
            return []

        return self.ACTIONS

    def process_end_game(self, action):
        self.game.end_game(player_initiated=True)
        self.log.append(f"Game ended manually by {action.entity.name}")

    @property
    def blocks(self):
        return False


class Exchange(BaseStep, ShareBuying):
    ACTIONS = [BuyShares]

    def actions(self, entity):
        if self.can_exchange(entity):
            return self.ACTIONS

        return []

    @property
    def blocks(self):
        return False

    def process_buy_shares(self, action):
        company = action.entity
        bundle = action.bundle

        if not self.can_exchange(company, bundle):
            raise GameError(f"Cannot exchange {company.id} for {bundle.corporation.id}")

        owner = company.owner
        self.buy_shares(owner, bundle, exchange=company)

        if hasattr(self.round, "players_history"):
            self.round.players_history.setdefault(owner, {}).setdefault(bundle.corporation, []).append(action)

        company.close()

    def can_buy(self, entity, bundle):
        return self.can_gain(entity, bundle, exchange=True)

    def exchangeable_shares(self, entity, bundle=None):
        if not entity.is_company():
            return []

        ability = self.game.abilities(entity, "exchange")
        if not ability:
            return []

        ability = ability[0]
        owner = entity.owner

        if bundle:
            return [bundle] if self.can_gain(owner, bundle, exchange=True) else []

        shares = []
        for corporation in self.game.exchange_corporations(ability):
            if "reserved" in ability.from_:
                shares.append(corporation.reserved_shares[0])
            if "ipo" in ability.from_:
                shares.append(corporation.available_share)
            if "market" in ability.from_:
                if self.game.share_pool.shares_by_corporation[corporation]:
                    shares.append(self.game.share_pool.shares_by_corporation[corporation][0])

        # set_trace()
        return [share for share in shares if share and self.can_gain(owner, share.to_bundle(), exchange=True)]

    def can_exchange(self, entity, bundle=None):
        if bundle and bundle.presidents_share:
            return False
        return any(self.exchangeable_shares(entity, bundle))

    def can_gain(self, entity, bundle, exchange=False):
        return super().can_gain(entity, bundle, exchange=exchange)


class HalfPay:
    def half(self, entity, revenue):
        withheld = self.half_pay_withhold_amount(entity, revenue)
        return {
            "corporation": withheld,
            "per_share": self.payout_per_share(entity, revenue - withheld),
        }

    def half_pay_withhold_amount(self, entity, revenue):
        return (revenue // (2 * entity.total_shares)) * entity.total_shares


class HomeToken(BaseStep, Tokener):
    ACTIONS = [PlaceToken]

    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        Tokener.__init__(self)

    def setup(self):
        Tokener.setup(self)

    def actions(self, entity):
        if entity == self.pending_entity:
            return [PlaceToken]
        return []

    @property
    def round_state(self):
        return {
            **BaseStep.round_state.fget(self),
            **Tokener.round_state.fget(self),
            "pending_tokens": [],
        }

    def active(self):
        return bool(self.pending_entity)

    @property
    def current_entity(self):
        return self.pending_entity

    @property
    def pending_entity(self):
        return self.pending_token.get("entity", None)

    def token(self):
        return self.pending_token.get("token", None)

    @property
    def pending_token(self):
        if self.round.pending_tokens:
            return self.round.pending_tokens[0]
        return {}

    @property
    def description(self):
        if self.current_entity != self.token().corporation:
            return f"Place {self.token().corporation.name} Home Token"
        elif self.token().corporation.tokens[0] == self.token():
            return "Place Home Token"
        else:
            return "Place Token"

    def available_hex(self, entity, hex):
        return hex in self.pending_token.get("hexes", [])

    def available_tokens(self, entity):
        return [self.token()]

    def process_place_token(self, action):
        hex = action.city.hex
        if not self.available_hex(action.entity, hex):
            raise GameError(f"Cannot place token on {hex.name} as the hex is not available")

        self.place_token(
            self.token().corporation,
            action.city,
            self.token(),
            connected=False,
            extra_action=True,
        )
        self.round.pending_tokens.pop(0)


class IssueShares(BaseStep):
    def actions(self, entity):
        available_actions = []
        if entity.is_corporation():
            if entity == self.current_entity:
                if self.redeemable_shares(entity):
                    available_actions.append(BuyShares)
                if self.issuable_shares(entity):
                    available_actions.append(SellShares)
                if self.blocks() and available_actions:
                    available_actions.append(Pass)
        return available_actions

    @property
    def description(self):
        return "Issue or Redeem Shares"

    def pass_description(self):
        return "Skip (Issue/Redeem)"

    def process_sell_shares(self, action):
        self.game.share_pool.sell_shares(action.bundle)
        self.pass_()

    def process_buy_shares(self, action):
        self.game.share_pool.buy_shares(action.entity, action.bundle)
        self.pass_()

    def issuable_shares(self, entity):
        # Done via Sell Shares
        return self.game.issuable_shares(entity)

    def redeemable_shares(self, entity):
        # Done via Buy Shares
        return self.game.redeemable_shares(entity)


class Message(BaseStep):
    ACTIONS = [Log, MessageAction]

    def actions(self, entity):
        if entity.is_player():
            return self.ACTIONS
        return []

    def process_log(self, action):
        self.log.append(action)

    def process_message(self, action):
        self.log.append(action)

    def skip(self):
        pass

    def pass_(self):
        pass

    def unpass(self):
        pass

    @property
    def blocks(self):
        return self.game.finished


class MinorHalfPay:
    def actions(self, entity):
        if entity.is_minor():
            return []
        if entity.is_corporation() and entity.type == "minor":
            return []
        return super().actions(entity)

    def skip(self):
        if self.current_entity.is_corporation() and self.current_entity.type != "minor":
            revenue = self.game.routes_revenue(self.routes)
            self.process_dividend(Dividend(self.current_entity, kind="payout" if revenue > 0 else "withhold"))

    def share_price_change(self, entity, revenue=0):
        if entity.is_corporation() and entity.type != "minor":
            return super().share_price_change(entity, revenue)
        return {}

    def payout(self, entity, revenue):
        if entity.is_corporation() and entity.type != "minor":
            return super().payout(entity, revenue)

        amount = revenue // 2
        return {"corporation": amount, "per_share": amount}

    def payout_shares(self, entity, revenue):
        if entity.is_corporation() and entity.type != "minor":
            return super().payout_shares(entity, revenue)

        self.log.append(f"{entity.owner.name} receives {self.game.format_currency(revenue)}")
        self.game.bank.spend(revenue, entity.owner)


class MinorWithhold:
    def actions(self, entity):
        if entity.is_minor():
            return []
        if entity.is_corporation() and entity.type == "minor":
            return []
        return super().actions(entity)

    def skip(self):
        if self.current_entity.is_corporation() and not self.current_entity.is_minor():
            self.process_dividend(Dividend(self.current_entity, kind="withhold"))
        return super().skip()


class Program(BaseStep):
    ACTIONS = [
        ProgramAuctionBid,
        ProgramBuyShares,
        ProgramIndependentMines,
        ProgramMergerPass,
        ProgramHarzbahnDraftPass,
        ProgramSharePass,
        ProgramClosePass,
        ProgramDisable,
    ]

    def actions(self, entity):
        if not entity.is_player():
            return []
        return self.ACTIONS

    def process_program_auction_bid(self, action):
        self.process_program_enable(action)

    def process_program_buy_shares(self, action):
        if not self.game.loading and not action.until_condition:
            raise Exception("Until condition is unset")
        self.process_program_enable(action)

    def process_program_independent_mines(self, action):
        self.process_program_enable(action)

    def process_program_merger_pass(self, action):
        self.process_program_enable(action)

    def process_program_harzbahn_draft_pass(self, action):
        self.process_program_enable(action)

    def process_program_share_pass(self, action):
        self.process_program_enable(action)

    def process_program_close_pass(self, action):
        self.process_program_enable(action)

    def process_program_enable(self, action):
        self.remove_programmed_action(action.entity, action.type)
        self.game.player_log(action.entity, f"Enabled programmed action '{action}'")
        self.game.programmed_actions[action.entity].append(action)
        if hasattr(self.round, "player_enabled_program"):
            self.round.player_enabled_program(action.entity)

    def process_program_disable(self, action):
        program = self.remove_programmed_action(action.entity, action.original_type)
        if not program:
            return
        reason = action.reason if action.reason else "unknown reason"
        self.game.player_log(action.entity, f"Disabled programmed action '{program}' due to '{reason}'")

    def remove_programmed_action(self, entity, type_):
        if type_ and self.game.ALLOW_MULTIPLE_PROGRAMS:
            existing = next(
                (a for a in self.game.programmed_actions[entity] if a.type == type_),
                None,
            )
        else:
            existing = self.game.programmed_actions[entity][-1] if self.game.programmed_actions[entity] else None
        if existing:
            self.game.programmed_actions[entity].remove(existing)
        return existing

    def skip(self):
        pass

    @property
    def blocks(self):
        return False


class ProgrammerAuctionBid(Programmer):
    def auto_actions(self, entity):
        return self.programmed_auto_actions(entity)

    def activate_program_auction_bid(self, entity, program):
        target = program.bid_target

        if target and target.owner and target.owner.is_player():
            return [ProgramDisable(entity, reason=f"{target.name} is owned by {target.owner.name}")]

        if self.auto_requires_auctioning(entity, program):
            return [
                ProgramDisable(
                    entity,
                    reason=f"{self.auctioning.name} chosen instead of {target.name}",
                )
            ]

        if target not in self.available:
            return [ProgramDisable(entity, reason=f"{target.name} is no longer available")]

        high_bid = self.highest_bid(target)
        if high_bid and high_bid.entity == entity:
            return [
                ProgramDisable(
                    entity,
                    reason=f"{entity.name} is already the high bid on {target.name}",
                )
            ]

        bid_params = {"price": self.min_bid(target)}
        if target.is_corporation():
            bid_params["corporation"] = target
        if target.is_company():
            bid_params["company"] = target
        if target.is_minor():
            bid_params["minor"] = target

        if self.auto_buy(entity, program):
            return [Bid(entity, **bid_params)]
        if self.auto_bid(entity, program):
            return [Bid(entity, **bid_params)]

        if self.auto_disable_if_bids(entity, program):
            return [ProgramDisable(entity, reason=f"Bids submitted for {target.name}")]

        if self.auto_disable_if_exceeded_price(entity, program):
            return [ProgramDisable(entity, reason=f"Price for {target.name} exceeded maximum bid")]

        return [Pass(entity)] if Pass in self.actions(entity) else []

    def auto_buy(self, entity, program):
        return (
            program.enable_buy_price
            and self.min_bid(program.bid_target) <= int(program.buy_price)
            and self.may_purchase(program.bid_target)
        )

    def auto_bid(self, entity, program):
        if not program.enable_maximum_bid:
            return False
        if self.auto_bid_on_empty(entity, program):
            return False

        return self.min_bid(program.bid_target) <= int(program.maximum_bid)

    def auto_disable_if_bids(self, entity, program):
        return (
            not program.auto_pass_after
            and program.enable_buy_price
            and not program.enable_maximum_bid
            and bool(self.bids.get(program.bid_target))
        )

    def auto_disable_if_exceeded_price(self, entity, program):
        return (
            not program.auto_pass_after
            and program.enable_maximum_bid
            and self.min_bid(program.bid_target) > int(program.maximum_bid)
        )

    def auto_requires_auctioning(self, entity, program):
        return False

    def auto_bid_on_empty(self, entity, program):
        return program.enable_buy_price


class ProgrammerMergerPass(Programmer):
    def auto_actions(self, entity):
        return self.programmed_auto_actions(entity)

    def activate_program_merger_pass(self, entity, program):
        if (
            self.game.actions[-1] != program
            and program.options
            and "disable_others" in program.options
            and self.others_acted()
        ):
            return [
                ProgramDisable(
                    entity.player,
                    reason="Other players have acted and requested to stop",
                )
            ]

        pass_entity = self.merger_auto_pass_entity()
        if pass_entity is None:
            return None

        # Check to see if the round and corps include the current one
        if pass_entity not in program.corporations_by_round.get(self.round.__class__.__name__, []):
            return None

        # Corporation and round match, pass!
        return [Pass(entity)]


class ReduceTokens(BaseStep, TokenMerger):
    REMOVE_TOKEN_ACTIONS = [RemoveToken]

    @property
    def description(self):
        return f"Choose tokens to remove to drop below limit of {self.game.LIMIT_TOKENS_AFTER_MERGER} tokens"

    def actions(self, entity):
        if self.current_entity == entity:
            return self.REMOVE_TOKEN_ACTIONS
        return []

    def active(self):
        return bool(self.round.corporations_removing_tokens)

    @property
    def active_entities(self):
        if self.round.corporations_removing_tokens:
            return [self.round.corporations_removing_tokens[0]]
        return []

    def surviving(self):
        if self.round.corporations_removing_tokens:
            return self.round.corporations_removing_tokens[0]
        return None

    def acquired_corps(self):
        if self.round.corporations_removing_tokens:
            return self.round.corporations_removing_tokens[1:]
        return []

    def can_replace_token(self, entity, token):
        if not token:
            return False
        return token.corporation in self.round.corporations_removing_tokens

    def process_remove_token(self, action):
        entity = action.entity
        slot = action.slot
        city_tokens = len(action.city.tokens)
        token = action.city.tokens[slot] if slot < city_tokens else action.city.extra_tokens[slot - city_tokens]
        if not self.available_hex(entity, token.city.hex):
            raise Exception(f"Cannot remove {token.corporation.name} token")

        token.remove()
        self.log.append(f"{entity.name} removes token from {action.city.hex.name}")

        if not self.tokens_above_limits(entity, self.acquired_corps()):
            self.move_tokens_to_surviving(entity, self.acquired_corps())
            self.round.corporations_removing_tokens = None

    def available_hex(self, entity, hex_):
        if entity != self.surviving():
            return False

        surviving_token = next((t for t in entity.tokens if t.used and t.city and t.hex == hex_), None)
        acquired_token = next(
            (t for t in self.others_tokens(self.acquired_corps()) if t.used and t.city and t.hex == hex_),
            None,
        )

        if self.tokens_in_same_hex(entity, self.acquired_corps()):
            return surviving_token and acquired_token
        else:
            return surviving_token or acquired_token


class ReturnToken(BaseStep):
    ACTIONS = [RemoveToken]

    def actions(self, entity):
        if self.ability(entity):
            return self.ACTIONS
        return []

    @property
    def blocks(self):
        return False

    def process_remove_token(self, action):
        company = action.entity
        corporation = company.owner

        if not corporation.is_corporation():
            raise Exception(f"{company.name} must be owned by a corporation")

        last_used_token = self.available_tokens(corporation).first

        if not last_used_token:
            raise Exception(f"{corporation.name} cannot return its only placed token")

        selected_city = action.city
        hex_ = selected_city.hex

        city_string = " city {}".format(selected_city.index) if len(hex_.tile.cities) > 1 else ""
        if not self.available_city(corporation, selected_city):
            raise Exception(f"Cannot return token from {hex_.name}{city_string} to {corporation.name}")

        last_city = last_used_token.city
        return_ability = self.ability(company)
        selected_token = selected_city.tokens[action.slot]

        selected_token.remove()
        if selected_city:
            selected_city.remove_reservation(corporation)
        if selected_token != last_used_token:
            last_used_token.remove()
            last_city.place_token(corporation, selected_token)

        if return_ability.reimburse:
            self.game.bank.spend(last_used_token.price, corporation)

        return_ability.use()

        log_msg = f"{corporation.name} returns the token from {hex_.name}{city_string} using {company.name}"
        if return_ability.reimburse:
            log_msg += f" and is reimbursed {self.game.format_currency(last_used_token.price)}"
        self.log.append(log_msg)

    def can_replace_token(self, company, token):
        corporation = company.owner
        if not corporation.is_corporation():
            return False

        return any(self.available_tokens(corporation)) and any(t for t in corporation.tokens if t.city == token.city)

    def available_hex(self, company, hex_):
        corporation = company.owner
        if not corporation.is_corporation():
            return False

        return hex_ in [t.city.hex for t in corporation.tokens if t.city]

    def available_city(self, corporation, city):
        if not corporation.is_corporation():
            return False

        return city in [t.city for t in corporation.tokens]

    def available_tokens(self, corporation):
        if not corporation.is_corporation():
            return []

        used_tokens = [t for t in corporation.tokens if t.used]

        # You cannot return your last token
        if len(used_tokens) <= 1:
            return []

        return [used_tokens[-1]]

    def ability(self, entity):
        if not entity.is_company():
            return None

        return self.game.abilities(entity, "return_token")


class Route(BaseStep):
    ACTIONS = [RunRoutes]

    def actions(self, entity):
        if not entity.is_operator() or not self.game.route_trains(entity) or not self.game.can_run_route(entity):
            return []
        return self.ACTIONS

    @property
    def description(self):
        return "Run Routes"

    def help(self):
        if self.current_entity.is_receivership():
            return (
                f"{self.current_entity.name} is in receivership (it has no president). Most of its "
                "actions are automated, but it must have a player manually run its trains. "
                f"Please enter the best route you see for {self.current_entity.name}."
            )
        else:
            return super().help()

    def process_run_routes(self, action):
        entity = action.entity
        self.round.routes = action.routes
        self.round.extra_revenue = action.extra_revenue
        trains = {}
        abilities = []

        for route in self.round.routes:
            train = route.train
            if train.owner and self.game.train_owner(train) != entity:
                raise Exception("Cannot run another corporation's train. refresh")
            if train in trains:
                raise Exception("Cannot run train twice")
            if train.operated:
                raise Exception("Cannot run train that operated")

            trains[train] = True
            revenue = self.game.format_revenue_currency(route.revenue())
            self.log.append(f"{entity.name} runs a {train.name} train for {revenue}: {route.revenue_str}")
            if route.abilities:
                abilities.extend(route.abilities)
        self.log_extra_revenue(entity, action.extra_revenue)
        self.pass_()

        for ability_type in set(abilities):
            self.game.abilities(action.entity, ability_type, time="route").use()

    def log_extra_revenue(self, entity, extra_revenue):
        if extra_revenue and extra_revenue != 0:
            revenue_str = self.game.format_revenue_currency(extra_revenue)
            self.log.append(f"{entity.name} receives {revenue_str} additional revenue")

    def conversion(self):
        return False

    def available_hex(self, entity, hex_):
        return self.game.graph_for_entity(entity).reachable_hexes(entity).get(hex_)

    @property
    def round_state(self):
        return {
            "routes": [],
        }


class SelectionAuction(BaseStep, PassableAuction):
    ACTIONS = [Bid, Pass]

    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        PassableAuction.__init__(self)
        self.setup()

    def setup(self):
        self.setup_auction()
        self.companies = self.game.initial_auction_companies
        self.cheapest = self.companies[0]
        self.auction_entity(self.companies[0])
        self.auction_triggerer = self.current_entity

    @property
    def description(self):
        return "Bid on Companies"

    def available(self):
        return self.companies

    def may_bid(self, company):
        if not self.companies or self.companies[0] != company:
            return False
        return super().may_bid(company)

    @property
    def active_entities(self):
        if self.auctioning:
            winning_bid = self.highest_bid(self.auctioning)
            if winning_bid:
                next_index = (self.active_bidders.index(winning_bid.entity) + 1) % len(self.active_bidders)
                return [self.active_bidders[next_index]]
        return super().active_entities

    def process_pass(self, action):
        entity = action.entity

        if self.auctioning_company():
            self.pass_auction(entity)
            self.resolve_bids()
        else:
            self.log.append(f"{entity.name} passes bidding")
            self.active_bidders.remove(entity)
            entity.pass_()
            if all(entity.passed for entity in self.entities):
                self.all_passed()
            if not self.all_passed_win:
                self.next_entity()
            self.all_passed_win = False

    def next_entity(self):
        self.round.next_entity_index()
        entity = self.entities[self.entity_index]
        if self.auctioning and self.max_bid(entity, self.auctioning) < self.min_bid(self.auctioning):
            entity.pass_action()
        if entity.passed:
            self.next_entity()

    def process_bid(self, action):
        action.entity.unpass()
        if self.auctioning_company():
            self.add_bid(action)
        elif len(self.active_bidders) == 1:
            self.add_bid(action)
            self.resolve_bids()
        else:
            self.selection_bid(action)
            if self.auctioning_company():
                self.next_entity()

    def actions(self, entity):
        if not self.companies:
            return []
        return self.ACTIONS if entity == self.current_entity else []

    def min_increment(self):
        return self.game.MIN_BID_INCREMENT

    def selection_bid(self, bid):
        self.add_bid(bid)

    def starting_bid(self, company):
        return company.min_bid

    def min_bid(self, company):
        if not company:
            return None
        if not self.bids[company]:
            return self.starting_bid(company)
        high_bid = self.highest_bid(company)
        return (high_bid.price or company.min_bid) + self.min_increment()

    def may_purchase(self, _company):
        return False

    def max_bid(self, player, _company):
        return player.cash

    def add_bid(self, bid):
        super().add_bid(bid)
        company = bid.company
        entity = bid.entity
        price = bid.price

        self.log.append(f"{entity.name} bids {self.game.format_currency(price)} for {company.name}")

    def win_bid(self, winner, _company):
        player = winner.entity
        company = winner.company
        price = winner.price
        self.assign_company(company, player)

        if price > 0:
            player.spend(price, self.game.bank)
        self.game.after_buy_company(player, company, price)

        self.companies.remove(company)
        self.log.append(
            f"{player.name} wins the auction for {company.name} with a bid of {self.game.format_currency(price)}"
        )

    def forced_win(self, player, company):
        self.active_bidders = [player]
        self.process_bid(Bid(player, price=0, company=company))

    def assign_company(self, company, player):
        company.owner = player
        player.companies.append(company)

    def all_passed(self):
        if self.cheapest in self.companies:
            value = self.cheapest.min_bid
            self.cheapest.discount += 5
            new_value = self.cheapest.min_bid
            self.log.append(
                f"{self.cheapest.name} minimum bid decreases from {self.game.format_currency(value)} to {self.game.format_currency(new_value)}"
            )
            self.auction_entity(self.cheapest)
            if new_value <= 0:
                self.round.next_entity_index()
                self.forced_win(self.current_entity, self.cheapest)
        else:
            self.game.payout_companies()
            self.game.or_set_finished()
            if self.companies:
                self.auction_entity(self.companies[0])

        for entity in self.entities:
            entity.unpass()

    def post_win_bid(self, _winner, _company):
        self.round.goto_entity(self.auction_triggerer)
        for entity in self.entities:
            entity.unpass()
        self.next_entity_()
        self.auction_triggerer = self.current_entity
        if self.companies:
            self.auction_entity(self.companies[0])


class SimpleDraft(BaseStep):
    ACTIONS = [Bid]

    def __init__(self, game, round, **kwargs):
        super().__init__(game, round, **kwargs)
        self.setup()

    def setup(self):
        self.companies = sorted(self.game.companies, key=lambda x: x.sort_order)
        self.choices = []

    def available(self):
        return self.companies

    def may_purchase(self, _company):
        return True

    def may_choose(self, _company):
        return True

    def auctioning_company(self):
        pass

    def bids(self):
        return {}

    def visible(self):
        return True

    def players_visible(self):
        return True

    def name(self):
        return "Draft"

    @property
    def description(self):
        return "Draft One Company Each"

    def finished(self):
        return all(p.companies for p in self.game.players)

    def actions(self, entity):
        if self.finished():
            return []
        return self.ACTIONS if entity == self.current_entity else []

    def process_bid(self, action):
        company = action.company
        player = action.entity
        price = action.price

        company.owner = player
        player.companies.append(company)
        player.spend(price, self.game.bank)

        self.companies.remove(company)

        self.log.append(f"{player.name} buys {company.name} for {self.game.format_currency(price)}")

        self.round.next_entity_index()
        self.action_finalized()

    def action_finalized(self):
        if not self.finished():
            return

        for c in self.companies:
            self.log.append(f"{c.name} is removed from the game")
            self.game.companies.remove(c)
        self.round.reset_entity_index()

    def committed_cash(self, _player, _show_hidden=False):
        return 0

    def min_bid(self, company):
        if not company:
            return None
        return company.value


class SingleDepotTrainBuy(BuyTrain):
    STATUS_TEXT = {
        "limited_train_buy": [
            "Limited Train Buy",
            "Corporations can only buy one train from the bank per OR",
        ],
    }

    def __init__(self, game, round, **kwargs):
        super().__init__(game, round, **kwargs)
        self.round_state = {"bought_trains": []}

    def buyable_trains(self, entity):
        trains = super().buyable_trains(entity)
        if self.game.phase.status.include("limited_train_buy") and entity in self.round.bought_trains:
            trains = [train for train in trains if not train.from_depot]
        return trains

    def process_buy_train(self, action):
        from_depot = action.train.from_depot
        super().process_buy_train(action)
        if from_depot:
            entity = action.entity
            self.round.bought_trains.append(entity)
            if not self.buyable_trains(entity):
                self.pass_()

    @property
    def round_state(self):
        return {"bought_trains": []}


class SpecialBuyTrain(BaseStep, Train):
    ACTIONS = [BuyTrain]

    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        Train.__init__(self, game)

    def actions(self, entity):
        if self.ability(entity):
            return self.ACTIONS
        return []

    @property
    def blocks(self):
        return False

    def process_buy_train(self, action):
        company = action.entity
        corporation = self.round.current_operator
        ability = self.ability(company, train=action.train)
        from_depot = action.train.from_depot
        self.buy_train_action(action, corporation)

        if from_depot and hasattr(self.round, "bought_trains"):
            self.round.bought_trains.append(corporation)

        closes_company = ability and ability["count"] and (ability["count"] - 1) == 0 and ability["closed_when_used_up"]

        if (
            action.price < action.train.price
            and ability
            and ability["discounted_price"](action.train, action.train.price) == action.price
        ):
            ability["use"]()
        if closes_company and not action.entity.is_closed():
            action.entity.close()
            self.log.append(f"{company.name} closes")

        if not self.can_buy_train(corporation):
            self.pass_()

    def ability_timing(self):
        return [
            "%current_step%",
            "buying_train",
            "owning_corp_or_turn",
            "owning_player_or_turn",
        ]

    def ability(self, entity, train=None):
        if not entity or not entity.is_company():
            return None

        ability = self.game.abilities(entity, "train_discount", time=self.ability_timing())
        if ability and (not train or not ability["trains"] or train.name in ability["trains"]):
            return ability

        return None


class SpecialBuy(BaseStep):
    ACTIONS = [SpecialBuyAction, Pass]
    ACTIONS_NO_PASS = [SpecialBuyAction]

    def __init__(self, game, round, **opts):
        super().__init__(game, round, **opts)
        self.setup()

    def setup(self):
        self.blocks = self.opts.get("blocks", False)

    def actions(self, entity):
        if self.buyable_items(entity):
            return self.ACTIONS if self.blocks else self.ACTIONS_NO_PASS
        return []

    @property
    def blocks(self):
        return self.blocks

    def buyable_items(self, entity):
        # Override this method in subclasses to return the items buyable by the entity
        return []

    @property
    def description(self):
        return "Special Buy"

    def short_description(self):
        # Implement this method if a short description is needed
        pass

    def pass_description(self):
        return (
            "Done ({})".format(self.short_description()) if self.acted else "Skip ({})".format(self.short_description())
        )

    def process_special_buy(self, action):
        # Implement the logic for processing a special buy action here
        pass


class SpecialChoose(BaseStep):
    ACTIONS = [ChooseAbility]

    def actions(self, entity):
        if not entity.is_company():
            return []

        action = self.abilities(entity)
        if not action:
            return []

        return self.ACTIONS if action.type == "choose_ability" else []

    @property
    def blocks(self):
        return False

    def choices_ability(self, entity):
        return self.abilities(entity).choices

    def abilities(self, entity, **kwargs):
        return self.game.abilities(entity, "choose_ability", **kwargs)

    @property
    def description(self):
        return "Choose"

    def process_choose_ability(self, action):
        raise NotImplementedError("process_choose_ability method not implemented.")

    def skip(self):
        self.pass_()


class SpecialToken(BaseStep, Tokener):
    def __init__(self, game, round, **kwargs):
        super().__init__(game, round, **kwargs)
        if not hasattr(self.round, "teleported"):
            self.round.teleported = None
            self.round.teleport_tokener = None

    def actions(self, entity):
        if not self.ability(entity) or not self.available_tokens(entity):
            return []

        actions = [PlaceToken]
        if entity == self.round.teleported:
            actions.append(Pass)
        return actions

    @property
    def description(self):
        return "Place teleport token"

    def pass_description(self):
        return "Pass (Token)"

    @property
    def blocks(self):
        return self.can_token_after_teleport()

    @property
    def blocking(self):
        return self.can_token_after_teleport()

    @property
    def round_state(self):
        super_round_state = super().round_state if hasattr(super(), "round_state") else {}

        state = {} if hasattr(self.round, "teleported") else {"teleported": None, "teleport_tokener": None}

        return {**super_round_state, **state}

    def can_token_after_teleport(self):
        return self.round.teleported and self.available_tokens(self.round.teleported)

    @property
    def active_entities(self):
        return [self.round.teleported] if self.round.teleported else super().active_entities

    def process_place_token(self, action):
        entity = action.entity
        hex_ = action.city.hex
        city_string = f" city {action.city.index}" if len(hex_.tile.cities) > 1 else ""
        if not self.available_hex(entity, hex_):
            raise Exception(f"Cannot place token on {hex_.name}{city_string}")

        special_ability = self.ability(entity)
        check_tokenable = getattr(special_ability, "check_tokenable", True)

        connected = special_ability.type == "token" and special_ability.connected
        self.place_token(
            self.game.token_owner(entity),
            action.city,
            action.token,
            connected=connected,
            special_ability=special_ability,
            check_tokenable=check_tokenable,
        )

        if special_ability.type == "token":
            special_ability.use()
            if special_ability.count == 0 and special_ability.closed_when_used_up:
                company = special_ability.owner
                self.log.append(f"{company.name} closes")
                company.close()
        if self.round.teleported:
            self.teleport_complete()

    def process_pass(self, action):
        self.log.append(f"{action.entity.owner.name} ({action.entity.id}) declines to place token")
        self.teleport_complete()

    def teleport_complete(self):
        ability = self.ability(self.round.teleported)
        if ability:
            self.round.teleported.remove_ability(ability)
        self.round.teleported = None

    def available_hex(self, entity, hex):
        ability = self.ability(entity)
        if ability.hexes and not (hex.id in ability.hexes):
            return None

        if ability.type == "token" and ability.connected:
            return self.game.token_graph_for_entity(entity.owner).reachable_hexes(entity.owner).get(hex)

        return self.game.hex_by_id(hex.id).neighbors.keys

    def available_tokens(self, entity):
        ability = self.ability(entity)
        if ability and ability.type in ["teleport", "token"] and not ability.from_owner:
            return [TokenPiece(entity.owner)]

        return super().available_tokens(self.game.token_owner(entity))

    def min_token_price(self, tokens):
        if self.round.teleported:
            return 0

        return super().min_token_price(tokens)

    def ability(self, entity):
        if not entity or not entity.is_company():
            return None

        abilities = self.game.abilities(entity, "token")
        tp_abilities = self.game.abilities(entity, "teleport")
        abilities.extend([a for a in tp_abilities if a.used])

        if abilities:
            return abilities[0]
        return None


class Tracker:
    def __init__(self):
        # Assume anyone using this is also running the BaseStep constructor first
        self.setup()

    def setup(self):
        self.round.num_laid_track = 0
        self.round.upgraded_track = False
        self.round.num_upgraded_track = 0
        self.round.laid_hexes = []

    def can_lay_tile(self, entity):
        if self.tile_lay_abilities_should_block(entity):
            return True
        if self.can_buy_tile_laying_company(entity, self.__class__.__name__):
            return True

        action = self.get_tile_lay(entity)
        if not action:
            return False

        return (
            entity.tokens
            and (self.buying_power(entity) >= action.get("cost", 0))
            and (action.get("lay") or action.get("upgrade"))
        )

    def get_tile_lay(self, entity):
        corporation = self.get_tile_lay_corporation(entity)
        tile_lays = self.game.tile_lays(corporation)
        if self.tile_lay_index() >= len(tile_lays):
            return None
        action = self.game.tile_lays(corporation)[self.tile_lay_index()]
        if not action:
            return None
        action = action.copy()

        if action.get("lay") == "not_if_upgraded":
            action["lay"] = not self.round.upgraded_track
        if action.get("upgrade") == "not_if_upgraded":
            action["upgrade"] = not self.round.upgraded_track

        action["cost"] = action.get("cost", 0)
        action["upgrade_cost"] = action.get("upgrade_cost", action["cost"])
        action["cannot_reuse_same_hex"] = action.get("cannot_reuse_same_hex", False)
        return action

    def tile_lay_index(self):
        return self.round.num_laid_track

    def get_tile_lay_corporation(self, entity):
        return entity.owner if entity.is_company() else entity

    def lay_tile_action(self, action, entity=None, spender=None):
        tile = action.tile
        hex = action.hex

        old_tile = hex.tile
        tile_lay = self.get_tile_lay(action.entity)
        if self.track_upgrade(old_tile, tile, hex) and not (tile_lay and tile_lay.get("upgrade", False)):
            raise Exception("Cannot lay an upgrade now")
        if tile.color == "yellow" and not (tile_lay and tile_lay.get("lay", False)):
            raise Exception("Cannot lay a yellow now")
        if tile_lay.get("cannot_reuse_same_hex", False) and hex in self.round.laid_hexes:
            raise Exception(f"{hex.id} cannot be laid as this hex was already laid on this turn")

        extra_cost = self.extra_cost(tile, tile_lay, hex)

        self.lay_tile(action, extra_cost=extra_cost, entity=entity, spender=spender)
        if self.track_upgrade(old_tile, tile, hex):
            self.round.upgraded_track = True
            self.round.num_upgraded_track += 1
        self.round.num_laid_track += 1
        self.round.laid_hexes.append(hex)

    def extra_cost(self, tile, tile_lay, hex):
        return tile_lay["cost"] if tile.color == "yellow" else tile_lay["upgrade_cost"]

    def track_upgrade(self, from_tile, to_tile, hex):
        return from_tile.color != "white"

    def tile_lay_abilities_should_block(self, entity):
        times = [
            "current_step",
            "owning_player_track",
        ]
        abilities = []
        for time in times:
            ability = self.abilities(entity, time=time, passive_ok=False)
            if ability:
                abilities.extend(ability if isinstance(ability, list) else [ability])
        return any(not a.consume_tile_lay for a in abilities)

    def abilities(self, entity, **kwargs):
        if not kwargs.get("time"):
            kwargs["time"] = ["current_step"]
        return self.game.abilities(entity, "tile_lay", **kwargs)

    def lay_tile(self, action, extra_cost=0, entity=None, spender=None):
        entity = entity or action.entity
        entities = [entity] + action.combo_entities

        entity_or_entities = entity if action.combo_entities == [] else entities

        spender = spender or entity
        tile = action.tile
        hex = action.hex
        rotation = action.rotation
        old_tile = hex.tile
        graph = self.game.graph_for_entity(spender)

        if not self.game.loading and (blocking_ability := self.ability_blocking_hex(entity, hex)):
            raise GameError(f"{hex.id} is blocked by {blocking_ability.owner.name}")

        tile.rotate_absolute(rotation)

        if not self.game.upgrades_to(
            old_tile,
            tile,
            entity.is_company(),
            selected_company=(entity.is_company() and entity) or None,
        ):
            raise GameError(f"{old_tile.name} is not upgradeable to {tile.name}")
        if not self.game.loading and not self.legal_tile_rotation(entity_or_entities, hex, tile):
            raise GameError(f"{old_tile.name} is not legally rotated for {tile.name}")

        self.update_tile_lists(tile, old_tile)

        hex.lay(tile)

        if old_tile.color in self.game.IMPASSABLE_HEX_COLORS:
            for direction, neighbor in hex.all_neighbors.items():
                if any(border.edge == direction and border.type == "impassable" for border in hex.tile.borders):
                    continue
                if direction in tile.exits:
                    neighbor.neighbors[neighbor.neighbor_direction(hex)] = hex
                    hex.neighbors[direction] = neighbor

        self.game.clear_graph_for_entity(entity)
        free = False
        discount = 0
        teleport = False
        ability_found = False
        discount_abilities = []

        for entity_ in entities:
            for ability in self.abilities(entity_):
                if ability.owner != entity_:
                    continue
                if ability.hexes and hex.id not in ability.hexes:
                    continue
                if ability.tiles and tile.name not in ability.tiles:
                    continue

                ability_found = True
                if ability.type == "teleport":
                    teleport = True
                    free = free or ability.free_tile_lay
                    if ability.cost and ability.cost > 0:
                        spender.spend(ability.cost, self.game.bank)
                        self.log.append(
                            f"{spender.name} ({ability.owner.sym}) spends {self.game.format_currency(ability.cost)} "
                            f"and teleports to {hex.name} ({hex.location_name})"
                        )
                else:
                    if (
                        ability.reachable
                        and hex.name != spender.coordinates
                        and not self.game.loading
                        and not graph.reachable_hexes(spender).get(hex, False)
                    ):
                        raise GameError(f"Track laid must be connected to one of {spender.id}'s stations")

                    free = free or ability.free
                    discount += ability.discount
                    if ability.discount > 0:
                        discount_abilities.append(ability)
                    extra_cost += ability.cost

        if entity.is_company() and not ability_found:
            raise GameError(f"{entity.name} does not have an ability that allows them to lay this tile")

        if not teleport:
            self.check_track_restrictions(entity, old_tile, tile)

        terrain = old_tile.terrain
        if free:
            self.remove_border_calculate_cost(tile, entity_or_entities, spender)
            cost = extra_cost
        else:
            border, border_types = self.remove_border_calculate_cost(tile, entity_or_entities, spender)
            if border > 0:
                terrain += border_types

            base_cost = self.game.upgrade_cost(old_tile, hex, entity, spender) + border + extra_cost

            if discount_abilities:
                discount = min(base_cost, discount)
                self.game.log_cost_discount(spender, discount_abilities, discount)

            cost = self.game.tile_cost_with_discount(tile, hex, entity, spender, base_cost - discount)

        self.pay_tile_cost(entity_or_entities, tile, rotation, hex, spender, cost, extra_cost)

        self.update_token(action, entity, tile, old_tile)

        for company, ability in self.game.all_companies_with_ability("tile_income"):
            if not ability.terrain:
                self.pay_all_tile_income(company, ability)
            else:
                self.pay_terrain_tile_income(company, ability, terrain, entity, spender)

    def pay_all_tile_income(self, company, ability):
        income = ability.income
        self.game.bank.spend(income, company.owner)
        self.log.append(
            f"{company.owner.name} earns {self.game.format_currency(income)}" f" for the tile built by {company.name}"
        )

    def pay_terrain_tile_income(self, company, ability, terrain, entity, spender):
        if ability.terrain not in terrain:
            return
        if ability.owner_only and company.owner not in [entity, spender]:
            return

        income = ability.income * terrain.count(ability.terrain)
        self.game.bank.spend(income, company.owner)
        self.log.append(
            f"{company.owner.name} earns {self.game.format_currency(income)}"
            f" for the {ability.terrain} tile built by {company.name}"
        )

    def update_tile_lists(self, tile, old_tile):
        self.game.update_tile_lists(tile, old_tile)

    def pay_tile_cost(self, entity_or_entities, tile, rotation, hex, spender, cost, extra_cost):
        entities = [entity_or_entities] if isinstance(entity_or_entities, (list, tuple)) else [entity_or_entities]
        entity, *_combo_entities = entities

        self.try_take_loan(spender, cost)
        if cost > 0:
            spender.spend(cost, self.game.bank)

        log_string = f"{spender.name}"
        if spender != entity and entity.is_company:
            log_string += "+".join([entity.sym for entity in entities])
        if cost != 0:
            log_string += f" spends {self.game.format_currency(cost)} and"
        log_string += f" lays tile #{tile.name} with rotation {rotation} on {hex.name}"
        if tile.location_name:
            log_string += f" ({tile.location_name})"
        self.log.append(log_string)

    def update_token(self, action, entity, tile, old_tile):
        cities = tile.cities
        if not old_tile.paths and tile.paths and len(cities) > 1:
            tokens = [token for city in cities for token in city.tokens if token]
            if tokens:
                actor = entity.owner if entity.is_company() else entity
                for token in tokens:
                    self.round.pending_tokens.append(
                        {
                            "entity": actor,
                            "hexes": [action.hex],
                            "token": token,
                        }
                    )
                    self.log.append(f"{actor.name} must choose city for token")
                    token.remove()

    def remove_border_calculate_cost(self, tile, entity_or_entities, spender):
        entities = [entity_or_entities] if isinstance(entity_or_entities, (list, tuple)) else [entity_or_entities]
        entity, *_combo_entities = entities

        hex = tile.hex
        types = []

        total_cost = sum(
            [
                border.cost - self.border_cost_discount(entity, spender, border, border.cost, hex)
                for border in tile.borders
                if border.cost
                and hex.targeting(hex.neighbors[border.edge])
                and hex.neighbors[border.edge].targeting(hex)
            ]
        )
        types = [
            border.type
            for border in tile.borders
            if border.cost and hex.targeting(hex.neighbors[border.edge]) and hex.neighbors[border.edge].targeting(hex)
        ]
        # Assuming border removal and cost discount logic is handled in border_cost_discount
        return total_cost, types

    def border_cost_discount(self, entity, spender, border, cost, hex):
        if not entity.is_corporation() and entity.owner and entity.owner.corporation:
            entity = entity.owner
        for ability in entity.all_abilities():
            if (
                ability.type != "tile_discount"
                or not ability.terrain
                or border.type != ability.terrain
                or (ability.hexes and hex.name not in ability.hexes)
            ):
                continue
            discount = min(ability.discount, cost)
            if discount > 0:
                self.log.append(
                    f"{spender.name} receives a discount of {self.game.format_currency(discount)} from {ability.owner.name}"
                )
            return discount
        return 0

    def check_track_restrictions(self, entity, old_tile, new_tile):
        if self.game.loading or not entity.is_operator():
            return

        graph = self.game.graph_for_entity(entity)

        if not self.game.ALLOW_REMOVING_TOWNS:
            for old_city in old_tile.city_towns:
                old_exits = set(old_city.exits if old_city.exits else [])
                if all(old_exits - set(new_city.exits if new_city.exits else []) for new_city in new_tile.city_towns):
                    raise GameError("New track must override old one")

        old_paths = old_tile.paths
        changed_city = False
        used_new_track = not old_paths

        for np in new_tile.paths:
            if not graph.connected_paths(entity).get(np):
                continue
            op = next((path for path in old_paths if np <= path), None)
            used_new_track |= op is None
            old_revenues = sorted(node.max_revenue for node in op.nodes) if op and op.nodes else []
            new_revenues = sorted(node.max_revenue for node in np.nodes) if np and np.nodes else []
            changed_city |= old_revenues != new_revenues

        track_restriction = self.game.TRACK_RESTRICTION
        if track_restriction == "permissive":
            return True
        elif track_restriction == "city_permissive":
            if not new_tile.cities and not used_new_track:
                raise GameError("Must be city tile or use new track")
        elif track_restriction == "restrictive":
            if not used_new_track:
                raise GameError("Must use new track")
        elif track_restriction == "semi_restrictive":
            if not used_new_track and not changed_city:
                raise GameError("Must use new track or change city value")
        elif track_restriction == "station_restrictive":
            if not used_new_track and not new_tile.nodes:
                raise GameError("Must use new track")
        else:
            raise Exception("Unknown track restriction")

    def potential_tile_colors(self, entity, hex):
        return self.game.phase.tiles.copy()

    def potential_tiles(self, entity_or_entities, hex):
        entities = [entity_or_entities] if not isinstance(entity_or_entities, list) else entity_or_entities
        entity = entities[0]

        colors = self.potential_tile_colors(entity, hex)
        tile_names = {
            tile.name: tile
            for tile in sorted(self.game.tiles, key=lambda x: x.index, reverse=True)
            if self.game.tile_valid_for_phase(tile, hex=hex, phase_color_cache=colors)
            and self.game.upgrades_to(hex.tile, tile)
        }
        return list(tile_names.values())

    def upgradeable_tiles(self, entity_or_entities, ui_hex):
        hex = self.game.hex_by_id(ui_hex.id)
        tiles = self.potential_tiles(entity_or_entities, hex)
        for tile in tiles:
            tile.rotate_absolute(0)  # Reset tile to no rotation
            tile.legal_rotations = self.legal_tile_rotations(entity_or_entities, hex, tile)
            if tile.legal_rotations:
                tile.rotate_absolute()  # Rotate to the first legal rotation

        tiles = [tile for tile in tiles if tile.legal_rotations]

        if (
            (hex.tile.cities and self.game.TILE_UPGRADES_MUST_USE_MAX_EXITS in ["cities"])
            or (
                hex.tile.cities
                and not hex.tile.labels
                and self.game.TILE_UPGRADES_MUST_USE_MAX_EXITS in ["unlabeled_cities"]
            )
            or (not hex.tile.cities and not hex.tile.towns and self.game.TILE_UPGRADES_MUST_USE_MAX_EXITS in ["track"])
        ):
            return self.max_exits(tiles)
        else:
            return tiles

    def max_exits(self, tiles):
        grouped_by_color = {}
        for tile in tiles:
            grouped_by_color.setdefault(tile.color, []).append(tile)
        result = []
        for color_group in grouped_by_color.values():
            max_edges = max(tile.edges.size for tile in color_group)
            result.extend([tile for tile in color_group if tile.edges.size == max_edges])
        return result

    def legal_tile_rotation(self, entity_or_entities, hex, tile):
        entities = [entity_or_entities] if not isinstance(entity_or_entities, list) else entity_or_entities
        entity = entities[0]

        if not self.game.legal_tile_rotation(entity, hex, tile):
            return False

        old_ctedges = hex.tile.city_town_edges
        new_exits = tile.exits
        new_ctedges = tile.city_town_edges
        added_cities = max(0, len(new_ctedges) - len(old_ctedges))
        multi_city_upgrade = len(tile.cities) > 1 and len(hex.tile.cities) > 1

        all_new_exits_valid = all(edge in hex.neighbors for edge in new_exits)
        if not all_new_exits_valid:
            return False

        entity_reaches_a_new_exit = any(exit in self.hex_neighbors(entity, hex) for exit in new_exits)
        if not entity_reaches_a_new_exit:
            return False

        if not self.old_paths_maintained(hex, tile):
            return False

        valid_added_city_count = added_cities >= sum(
            1 for newct in new_ctedges if all(len(set(newct) & set(oldct)) == 0 for oldct in old_ctedges)
        )
        if not valid_added_city_count:
            return False

        old_cities_map_to_new = not multi_city_upgrade or (
            all(sum(1 for newct in new_ctedges if set(oldct) <= set(newct)) == 1 for oldct in old_ctedges)
        )
        if not old_cities_map_to_new:
            return False

        if not self.city_sizes_maintained(hex, tile):
            return False

        return True

    def old_paths_maintained(self, hex, tile):
        old_paths = hex.tile.paths
        new_paths = tile.paths
        return all(any(path <= p for p in new_paths) for path in old_paths)

    def city_sizes_maintained(self, hex, tile):
        if len(set(city.normal_slots() for city in hex.tile.cities)) <= 1:
            return True
        hex_city_map = hex.city_map_for(tile)
        return all(new_c.normal_slots() >= old_c.normal_slots() for old_c, new_c in hex_city_map.items())

    def legal_tile_rotations(self, entity_or_entities, hex, tile):
        return [
            rotation
            for rotation in Tile.ALL_EDGES
            if self.legal_tile_rotation(entity_or_entities, hex, tile.rotate_absolute(rotation))
        ]

    def hex_neighbors(self, entity, hex):
        # set_trace()
        return self.game.graph_for_entity(entity).connected_hexes(entity).get(hex, set())

    def can_buy_tile_laying_company(self, entity, time):
        if entity != self.current_entity:
            return False
        if "can_buy_companies" not in self.game.phase.status:
            return False
        return any(
            company.min_price <= self.buying_power(entity)
            and any(a.type == "tile_lay" and a.matches_when(time) for a in company.all_abilities)
            for company in self.game.purchasable_companies(entity)
        )

    def ability_blocking_hex(self, entity, hex):
        for company in self.game.companies + self.game.minors + self.game.corporations:
            if company.is_closed() or company == entity:
                continue
            abilities = self.game.abilities(company, "blocks_hexes")
            for ability in abilities:
                if self.game.hex_blocked_by_ability(entity, ability, hex):
                    return ability
        return None

    def tracker_available_hex(self, entity, hex):
        connected = self.hex_neighbors(entity, hex)
        if not connected:
            return None
        tile_lay = self.get_tile_lay(entity)
        if not tile_lay:
            return None
        color = hex.tile.color
        if color == "white" and not tile_lay["lay"]:
            return None
        if color != "white" and not tile_lay["upgrade"]:
            return None
        if color != "white" and tile_lay["cannot_reuse_same_hex"] and hex in self.round.laid_hexes:
            return None
        if self.ability_blocking_hex(entity, hex):
            return None
        return connected


class SpecialTrack(BaseStep, Tracker):
    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        Tracker.__init__(self)
        self.company = None

    ACTIONS = [LayTile]
    ACTIONS_WITH_PASS = [LayTile, Pass]

    def actions(self, entity):
        actions = self.abilities(entity)
        if not actions:
            return []
        return (
            self.ACTIONS
            if any(action.type == "tile_lay" and action.blocks for action in actions)
            else self.ACTIONS_WITH_PASS
        )

    @property
    def description(self):
        return f"Lay Track for {self.company.name}"

    @property
    def active_entities(self):
        return [self.company] if self.company else super().active_entities

    @property
    def blocks(self):
        return bool(self.company)

    @property
    def round_state(self):
        state = {} if hasattr(self.round, "teleported") else {"teleported": None, "teleport_tokener": None}
        state.update(super().round_state)
        return state

    def process_lay_tile(self, action):
        if (
            self.company
            and (self.company != action.entity)
            and (abilities := self.game.abilities(self.company, "tile_lay", time="track"))
            and abilities[0].must_lay_together
            and abilities[0].must_lay_all
        ):
            raise GameError(f"Cannot interrupt {self.company.name}'s tile lays")

        ability = self.abilities(action.entity)[0]

        owner = (
            action.entity.owner if action.entity.owner and action.entity.owner.corporation else self.game.current_entity
        )
        if ability.type == "teleport" or (ability.type == "tile_lay" and ability.consume_tile_lay):
            self.lay_tile_action(action, spender=owner)
        else:
            self.lay_tile(action, spender=owner)
            ability.laid_hexes.append(action.hex.id)
            self.round.laid_hexes.append(action.hex)
            self.check_connect(action, ability)
        ability.use(
            upgrade="green" in ["green", "brown", "gray"]
            or "brown" in ["green", "brown", "gray"]
            or "gray" in ["green", "brown", "gray"]
        )

        if (
            owner
            and owner.corporation
            and (operating_info := owner.operating_history.get((self.game.turn, self.round.round_num)))
        ):
            operating_info.laid_hexes = self.round.laid_hexes

        if ability.type == "tile_lay":
            if ability.count is not None and ability.count == 0 and ability.closed_when_used_up:
                company = ability.owner
                self.log.append(f"{company.name} closes")
                company.close()
            self.company = action.entity if ability.count > 0 and ability.must_lay_together else None

        if ability.type == "teleport":
            company = ability.owner
            tokener = company.owner if not company.owner.player else self.game.current_entity
            if not tokener.tokens_by_type:
                company.remove_ability(ability)
            else:
                self.round.teleported = company
                self.round.teleport_tokener = tokener

    def process_pass(self, action):
        entity = action.entity
        ability = self.abilities(entity)
        if entity != self.company:
            raise GameError(f"Not {entity.name}'s turn: {action.to_dict()}")

        if ability.must_lay_all and ability.count > 0:
            raise GameError(f"{entity.name} must use all its tile lays")

        entity.remove_ability(ability)
        self.log.append(f"{entity.owner.name} passes laying additional track with {entity.name}")
        self.company = None

    def available_hex(self, entity, hex):
        abilities = self.abilities(entity)
        if not abilities:
            return None
        ability = abilities[0]
        if not ability.hexes or ability.consume_tile_lay:
            return self.tracker_available_hex(entity, hex)
        return self.hex_neighbors(entity, hex)

    def hex_neighbors(self, entity, hex):
        abilities = self.abilities(entity)
        if not abilities:
            return None
        ability = abilities[0]
        if not ability or (ability.hexes and hex.id not in ability.hexes):
            return None
        operator = entity.owner if entity.owner.is_corporation() else self.game.current_entity
        if ability.type == "tile_lay" and ability.reachable and hex not in self.game.graph.connected_hexes(operator):
            return None
        return list(self.game.hex_by_id(hex.id).neighbors.keys())

    def potential_tiles(self, entity_or_entities, hex):
        entities = [entity_or_entities] if not isinstance(entity_or_entities, list) else entity_or_entities
        entity = entities[0]
        tile_abilities = self.abilities(entity)
        if not tile_abilities:
            return []
        tile_ability = tile_abilities[0]

        if not tile_ability.tiles:
            tiles = self.game.tiles.unique(lambda t: t.name)
        else:
            tiles = [next((t for t in self.game.tiles if t.name == name), None) for name in tile_ability.tiles]

        special = tile_ability.special if tile_ability.type == "tile_lay" else None
        return [
            t
            for t in tiles
            if t
            and self.game.tile_valid_for_phase(t, hex=hex, phase_color_cache=self.potential_tile_colors(entity, hex))
            and self.game.upgrades_to(hex.tile, t, special, selected_company=entity)
        ]

    def abilities(self, entity, **kwargs):
        if not entity or not entity.is_company():
            return []
        if hasattr(self.round, "just_sold_company") and entity == self.round.just_sold_company:
            ability = self.game.abilities(entity, "tile_lay", time="sold", **kwargs)
            if ability:
                return ability
        possible_times = [
            "%current_step%",
            "owning_corp_or_turn",
            "owning_player_or_turn",
            "owning_player_track",
            "or_between_turns",
            "stock_round",
        ]
        valid_abilities = []
        for type in ["tile_lay", "teleport"]:
            abilities = self.game.abilities(entity, type, time=possible_times, **kwargs)
            for ability in abilities:
                if ability and (type != "teleport" or not ability.used):
                    valid_abilities.append(ability)
        return valid_abilities

    def check_connect(self, action, ability):
        if (
            self.game.loading
            or ability.type == "teleport"
            or not ability.connect
            or len(ability.hexes) < 2
            or not ability.start_count
            or ability.start_count < 2
            or ability.start_count == ability.count
        ):
            return
        connected = {}
        laid_hexes = [self.game.hex_by_id(h) for h in ability.laid_hexes]
        for hex in laid_hexes:
            if hex in connected:
                continue
            for other in laid_hexes:
                if hex == other:
                    continue
                if any(a.connects_to(b, None) for a in hex.tile.paths for b in other.tile.paths):
                    connected[hex] = True
                    connected[other] = True
        if len(connected.keys()) != len(laid_hexes):
            raise GameError("Paths must be connected")


class Token(BaseStep, Tokener):
    ACTIONS = [PlaceToken, Pass]

    def __init__(self, game, round, **opts):
        BaseStep.__init__(self, game, round, **opts)
        Tokener.__init__(self)

    def actions(self, entity):
        if entity != self.current_entity:
            return []
        if not self.can_place_token(entity):
            return []

        return self.ACTIONS

    @property
    def description(self):
        return "Place a Token"

    def pass_description(self):
        return "Skip (Token)"

    def available_hex(self, entity, hex):
        return self.tokener_available_hex(entity, hex)

    @property
    def round_state(self):
        return {**BaseStep.round_state.fget(self), **Tokener.round_state.fget(self)}

    def process_place_token(self, action):
        entity = action.entity

        self.place_token(entity, action.city, action.token)
        self.pass_()


class Track(BaseStep, Tracker):
    ACTIONS = [LayTile, Pass]

    def __init__(self, game, round, **opts):
        BaseStep.__init__(self, game, round, **opts)
        Tracker.__init__(self)
        self.setup()

    def setup(self):
        BaseStep.setup(self)
        Tracker.setup(self)

    def actions(self, entity):
        if entity != self.current_entity:
            return []
        if entity.is_company() or not self.can_lay_tile(entity):
            return []

        return self.ACTIONS

    @property
    def description(self):
        tile_lay = self.get_tile_lay(self.current_entity)
        if not tile_lay:
            return "Lay Track"

        if tile_lay.get("lay") and tile_lay.get("upgrade"):
            return "Lay/Upgrade Track"
        elif tile_lay.get("lay"):
            return "Lay Track"
        else:
            return "Upgrade Track"

    def pass_description(self):
        return "Done (Track)" if self.acted else "Skip (Track)"

    def process_lay_tile(self, action):
        self.lay_tile_action(action)
        if not self.can_lay_tile(action.entity):
            self.pass_()

    def available_hex(self, entity_or_entities, hex):
        entities = [entity_or_entities] if not isinstance(entity_or_entities, list) else entity_or_entities
        entity = entities[0]

        return self.tracker_available_hex(entity, hex)


class TrackAndToken(Track, Tokener):
    ACTIONS = [LayTile, PlaceToken, Pass]

    def __init__(self, game, round, **opts):
        Track.__init__(self, game, round, **opts)
        Tokener.__init__(self)
        self.tokened = False

    def setup(self):
        super().setup()
        self.tokened = False

    def actions(self, entity):
        actions = []
        if entity != self.current_entity:
            return actions

        if self.can_lay_tile(entity):
            actions.append(LayTile)
        if self.can_place_token(entity):
            actions.append(PlaceToken)
        if actions:
            actions.append(Pass)
        return actions

    @property
    def description(self):
        return "Place a Token or Lay Track"

    def pass_description(self):
        return "Done (Token/Track)" if self.acted else "Skip (Token/Track)"

    def unpass(self):
        super().unpass()
        self.setup()

    def can_place_token(self, entity):
        return super().can_place_token(entity) and not self.tokened

    def can_lay_tile(self, entity):
        free = False
        tile_cost = self.game.TILE_COST  # Assuming TILE_COST is defined in the game class

        ability = self.game.abilities(entity, "tile_lay")
        if ability:
            for hex_id in ability.hexes:
                hex_tile = self.game.hex_by_id(hex_id).tile
                if (ability.free or ability.discount >= tile_cost) and hex_tile.preprinted:
                    free = True

        return (free or self.buying_power(entity) >= tile_cost) and super().can_lay_tile(entity)

    def process_place_token(self, action):
        entity = action.entity

        self.place_token(entity, action.city, action.token)
        self.tokened = True
        if not self.can_lay_tile(entity):
            self.pass_()

    def process_lay_tile(self, action):
        self.lay_tile_action(action)
        if not self.can_lay_tile(action.entity) and self.tokened:
            self.pass_()

    def available_hex(self, entity, hex):
        if self.can_lay_tile(entity) and self.tracker_available_hex(entity, hex):
            return True
        if self.can_place_token(entity) and self.tokener_available_hex(entity, hex):
            return True

        return False


class TrackLayWhenCompanySold:
    ACTIONS = [LayTile]
    ACTIONS_WITH_PASS = [LayTile, Pass]

    def actions(self, entity):
        if self.blocking_for_sold_company():
            ability = self.game.abilities(self.company, "tile_lay", time="sold")
            return self.ACTIONS if ability.blocks else self.ACTIONS_WITH_PASS
        else:
            return super().actions(entity)

    @property
    def blocking(self):
        return self.blocking_for_sold_company() or super().blocking()

    def process_lay_tile(self, action):
        if action.entity == self.company:
            entity = action.entity
            ability = self.game.abilities(self.company, "tile_lay", time="sold")
            if entity != self.company:
                raise Exception(f"Not {entity.name}'s turn: {action}")  # Adjusted exception type

            self.lay_tile(action, spender=entity.owner)
            self.round.laid_hexes.append(action.hex)
            self.check_connect(action, ability)
            if action.tile.color in ["green", "brown", "gray"]:
                ability.use(upgrade=True)

            self.company = None
        else:
            super().process_lay_tile(action)

    def process_pass(self, action):
        entity = action.entity
        ability = self.game.abilities(self.company, "tile_lay", time="sold")
        if entity != self.company:
            raise Exception(f"Not {entity.name}'s turn: {action}")  # Adjusted exception type

        self.company.remove_ability(ability)
        self.log.append(f"{entity.name} passes lay track")
        self.pass_()

        self.company = None

    def blocking_for_sold_company(self):
        self.company = None
        just_sold_company = self.round.just_sold_company if hasattr(self.round, "just_sold_company") else None

        if self.game.abilities(just_sold_company, "tile_lay", time="sold"):
            self.company = just_sold_company
            return True

        return False


class WaterfallAuction(BaseStep, Auctioner, ProgrammerAuctionBid):
    ACTIONS = [Bid, Pass]

    def __init__(self, game, round, **kwargs):
        BaseStep.__init__(self, game, round, **kwargs)
        Auctioner.__init__(self)
        self.setup()

    def setup(self):
        self.setup_auction()
        self.auctioning = None
        self.companies = sorted(
            self.game.initial_auction_companies,
            key=lambda x: x.value,
        )
        self.cheapest = self.companies[0]
        self.bidders = defaultdict(list)

    @property
    def description(self):
        return "Bid on Companies"

    def available(self):
        return self.companies

    def process_pass(self, action):
        entity = action.entity

        if self.auctioning_company():
            self.pass_auction(entity)
        else:
            self.log.append(f"{entity.name} passes bidding")
            entity.pass_()
            if all(e.passed for e in self.entities):
                self.all_passed()
            self.round.next_entity_index()

    def process_bid(self, action):
        entity = action.entity
        entity.unpass()

        if self.auctioning_company():
            if action.company != self.auctioning_company():
                raise GameError(
                    f"{entity.name} cannot bid on {action.company.name} because {self.auctioning_company().name} is up for auction"
                )
            self.add_bid(action)
        else:
            self.placement_bid(action)
            self.round.next_entity_index()

    @property
    def active_entities(self):
        _, bids = self.active_auction()
        if bids:
            return [min(bids, key=lambda x: x.price).entity]
        return super().active_entities

    def actions(self, entity):
        if not self.companies:
            return []
        correct = False
        _, bids = self.active_auction()
        if bids:
            correct = min(bids, key=lambda x: x.price).entity == entity
        return self.ACTIONS if correct or entity == self.current_entity else []

    @property
    def round_state(self):
        return {"companies_pending_par": []}

    def min_bid(self, company):
        if not company:
            return None
        if self.may_purchase(company):
            return company.min_bid

        high_bid = self.highest_bid(company)
        return (high_bid.price if high_bid else company.min_bid) + self.min_increment()

    def may_purchase(self, company):
        is_active, _ = self.active_auction()
        return is_active is None and company is not None and company == self.companies[0]

    def committed_cash(self, player):
        return sum(bid.price for bid in self.bids_for_player(player))

    def max_bid(self, player, company):
        return player.cash - self.committed_cash(player) + self.current_bid_amount(player, company)

    def resolve_bids(self):
        company = self.companies[0] if self.companies else None
        while company:
            if not self.resolve_bids_for_company(company):
                break
            company = self.companies[0] if self.companies else None

    def resolve_bids_for_company(self, company):
        resolved = False
        is_new_auction = company != self.auctioning
        self.auctioning = None
        bids = self.bids[company]

        if len(bids) == 1:
            self.accept_bid(bids[0])
            resolved = True
        elif self.can_auction(company):
            self.auctioning = company
            if is_new_auction:
                self.log.append(f"{self.auctioning.name} goes up for auction")

        return resolved

    def active_auction(self):
        company = self.auctioning
        bids = self.bids[company]
        if bids and len(bids) > 1:
            return (company, bids)
        return None, None

    def can_auction(self, company):
        return company == self.companies[0] and len(self.bids[company]) > 1

    def all_passed(self):
        if self.companies.count(self.cheapest):
            self.increase_discount(self.cheapest, 5)
        else:
            self.game.payout_companies()
            self.game.or_set_finished()

        for entity in self.entities:
            entity.unpass()

    def increase_discount(self, company, discount):
        value = company.min_bid
        company.discount += discount
        new_value = company.min_bid
        self.log.append(
            f"{company.name} minimum bid decreases from {self.game.format_currency(value)} to {self.game.format_currency(new_value)}"
        )

        if new_value <= 0:
            self.round.next_entity_index()
            self.buy_company(self.current_entity, company, 0)
            self.resolve_bids()

    def placement_bid(self, bid):
        if self.may_purchase(bid.company):
            self.round.last_to_act = bid.entity
            self.auction_triggerer = bid.entity
            self.accept_bid(bid)
            self.resolve_bids()
        else:
            self.add_bid(bid)

    def buy_company(self, player, company, price):
        available = self.max_bid(player, company)
        if available < price:
            raise ValueError(
                f"{player.name} has {self.game.format_currency(available)} available and cannot spend {self.game.format_currency(price)}"
            )

        # set_trace()
        company.owner = player
        player.companies.append(company)
        if price > 0:
            player.spend(price, self.game.bank)
        self.companies.remove(company)
        num_bidders = len(self.bidders[company])
        if num_bidders == 0:
            self.log.append(f"{player.name} buys {company.name} for {self.game.format_currency(price)}")
        elif num_bidders == 1:
            self.log.append(
                f"{player.name} wins the auction for {company.name} with the only bid of {self.game.format_currency(price)}"
            )
        else:
            self.log.append(
                f"{player.name} wins the auction for {company.name} with a bid of {self.game.format_currency(price)}"
            )

        self.game.after_buy_company(player, company, price)

    def accept_bid(self, bid):
        price = bid.price
        company = bid.company
        player = bid.entity
        min_bid = self.min_bid(company)
        if price < min_bid:
            raise GameError(f"Minimum bid is {self.game.format_currency(min_bid)} for {company.name}")
        self.bids[company] = []
        self.buy_company(player, company, price)

    def add_bid(self, bid):
        super().add_bid(bid)
        company = bid.company
        price = bid.price
        entity = bid.entity
        self.bidders[company].append(entity)
        self.log.append(f"{entity.name} bids {self.game.format_currency(price)} for {bid.company.name}")


class BaseRound:
    DEFAULT_STEPS = []  # [EndGame, Message, Program] disabled for agents

    def __init__(self, game, steps, round_num=1, **opts):
        self.game = game
        self.log = game.log
        self.entity_index = 0
        self.round_num = round_num
        self.entities = self.select_entities()
        self.last_to_act = None
        self.pass_order = []
        self.at_start = None
        self._active_step = None

        self.steps = []
        self.steps_to_do = self.DEFAULT_STEPS + steps
        for step_to_do in self.steps_to_do:
            step_opts = {}
            if isinstance(step_to_do, list):
                step_opts = step_to_do[1]
                step_to_do = step_to_do[0]
            step = step_to_do(game, self, **step_opts)
            self._set_dynamic_properties(step.round_state)
            game.next_turn()
            step.setup()
            self.steps.append(step)

    def _set_dynamic_properties(self, round_state):
        for key, value in round_state.items():
            self._create_property(key, value)

    def _create_property(self, key, initial_value):
        private_key = f"_{key}"

        def getter(instance):
            return getattr(instance, private_key, initial_value)

        def setter(instance, value):
            setattr(instance, private_key, value)

        setattr(self.__class__, key, property(getter, setter))
        setattr(self, private_key, initial_value)

    def setup(self):
        pass

    def name(self):
        raise NotImplementedError

    def select_entities(self):
        raise NotImplementedError

    def round_description(self):
        return f"{self.name()} {self.round_num}"

    @property
    def current_entity(self):
        return self.active_entities[0] if self.active_entities else None

    @property
    def description(self):
        return self.active_step().description

    @property
    def active_entities(self):
        return self.active_step().active_entities if self.active_step() else []

    def can_act(self, entity):
        return self.active_step().current_entity == entity if self.active_step() else None

    def pass_description(self):
        return self.active_step().pass_description

    def process_action(self, action):
        type = action.__class__
        self.clear_cache()

        self.before_process(action)

        for step in self.steps:
            if not step.active:
                continue

            process = type in step.actions(action.entity)
            blocking = step.blocking
            if blocking and not process:
                raise GameError(f"Blocking step {step.description} cannot process action {action}")

            if blocking or process:
                step.acted = True
                getattr(step, f"process_{pascal_to_snake(action.__class__.__name__)}")(action)

                self.at_start = False

                self.after_process_before_skip(action)
                self.skip_steps()
                self.clear_cache()
                self.after_process(action)
                return

        e = GameError(
            f"No step found for action {type.__name__} at {action.id}: {action.to_dict()}. Game Actions: {self.game.raw_actions}"
        )
        LOGGER.exception(e)
        raise e

    def actions_for(self, entity):
        actions = []
        if not entity:
            return actions

        for step in self.steps:
            if not step.active:
                continue

            available_actions = step.actions(entity)
            actions.extend(available_actions)
            if step.blocking:
                break

        return list(set(actions))

    def step_for(self, entity, action):
        if not entity:
            return None

        for step in self.steps:
            if not step.active():
                continue

            if action in step.actions(entity):
                return step
            if step.blocking:
                break

        return None

    def step_passed(self, action_klass):
        return any(step.passed and isinstance(step, action_klass) for step in self.steps)

    def active_step(self, entity=None):
        if entity:
            return next(
                (
                    step
                    for step in self.steps
                    if step.active and (entity.is_company() or step.blocking) and step.actions(entity)
                ),
                None,
            )
        if not self._active_step:
            steps = [step for step in self.steps if step.active and step.blocking]
            self._active_step = steps[0] if steps else None
        return self._active_step

    def auto_actions(self):
        return (
            self.active_step(self.current_entity).auto_actions(self.current_entity)
            if self.active_step(self.current_entity)
            else None
        )

    def finished(self):
        return not self.active_step()

    def goto_entity(self, entity):
        self.game.next_turn()
        self.entity_index = self.entities.index(entity)

    def next_entity_index(self):
        self.game.next_turn()
        self.entity_index = (self.entity_index + 1) % len(self.entities)

    def reset_entity_index(self):
        self.game.next_turn()
        self.entity_index = 0

    def clear_cache(self):
        self._active_step = None

    @property
    def operating(self):
        return False

    @property
    def stock(self):
        return False

    @property
    def merger(self):
        return False

    @property
    def auction(self):
        return False

    def unordered(self):
        return False

    def show_auto(self):
        return False

    def show_in_history(self):
        return True

    def skip_steps(self):
        for step in self.steps:
            if (
                not step.active
                or not step.blocks
                or (self.entities[self.entity_index] is not None and self.entities[self.entity_index].is_closed())
            ):
                continue
            if step.blocking:
                break
            step.skip()

    def before_process(self, action):
        pass

    def after_process_before_skip(self, action):
        pass

    def after_process(self, action):
        pass


class Auction(BaseRound):
    def name(self):
        return "Auction Round"

    @classmethod
    def short_name(cls):
        return "ISR"

    @property
    def auction(self):
        return True

    def select_entities(self):
        return self.game.players


class Choices(BaseRound):
    def name(self):
        return "Choices"

    @classmethod
    def short_name(cls):
        return "Choices"

    def show_in_history(self):
        return False


class Draft(BaseRound):
    def __init__(
        self,
        game,
        steps,
        reverse_order=False,
        snake_order=False,
        rotating_order=False,
        **opts,
    ):
        self.reverse_order = reverse_order
        self.snake_order = snake_order
        self.rotating_order = rotating_order
        self.snaking_up = True
        super().__init__(game, steps, **opts)

    @classmethod
    def short_name(cls):
        return "DR"

    def name(self):
        return "Draft Round"

    def select_entities(self):
        return list(reversed(self.game.players)) if self.reverse_order else self.game.players

    def next_entity_index(self):
        if self.rotating_order and self.entity_index == (len(self.entities) - 1):
            self.entities.append(self.entities.pop(0))

        if self.snake_order:
            if (self.snaking_up and self.entity_index == (len(self.entities) - 1)) or (
                not self.snaking_up and self.entity_index == 0
            ):
                self.snaking_up = not self.snaking_up
            else:
                plus_or_minus = 1 if self.snaking_up else -1
                self.game.next_turn()
                self.entity_index = (self.entity_index + plus_or_minus) % len(self.entities)
        else:
            super().next_entity_index()


class Merger(BaseRound):
    def name(self):
        return self.round_name()

    @classmethod
    def round_name(cls):
        raise NotImplementedError

    @property
    def merger(self):
        return True


class Operating(BaseRound):
    def __init__(self, game, steps, **opts):
        super().__init__(game, steps, **opts)
        self.current_operator = None
        self.current_operator_acted = False

    @classmethod
    def short_name(cls):
        return "OR"

    def name(self):
        return "Operating Round"

    def select_entities(self):
        return self.game.operating_order

    def setup(self):
        self.current_operator = None
        self.home_token_timing = self.game.HOME_TOKEN_TIMING
        self.game.payout_companies()
        if self.home_token_timing == "operating_round":
            for entity in self.entities:
                self.game.place_home_token(entity)
        for entity in self.game.corporations + self.game.minors + self.game.companies:
            entity.reset_ability_count_this_or()
        self.after_setup()

    def any_to_act(self):
        return any(not self.skip_entity(entity) for entity in self.entities)

    def after_setup(self):
        if self.any_to_act():
            self.start_operating()

    def after_process(self, action):
        if isinstance(action, MessageAction):
            return

        self.current_operator_acted = action.entity.corporation == self.current_operator

        if self.active_step():
            entity = self.entities[self.entity_index]
            if entity.owner and entity.owner.is_player() or entity.receivership:
                return

        self.after_end_of_turn(self.current_operator)

        if not self.game.finished:
            self.next_entity()

    def after_end_of_turn(self, operator):
        pass

    def force_next_entity(self):
        for step in self.steps:
            step.pass_()
        self.next_entity()
        self.clear_cache()

    def skip_entity(self, entity):
        return entity.is_closed()

    def next_entity(self):
        if self.entity_index == len(self.entities) - 1:
            return

        self.next_entity_index()

        if self.skip_entity(self.entities[self.entity_index]):
            return self.next_entity()

        for step in self.steps:
            step.unpass()
        for step in self.steps:
            step.setup()
        self.start_operating()

    def start_operating(self):
        entity = self.entities[self.entity_index]
        if self.skip_entity(entity):
            self.next_entity()

        self.current_operator = entity
        self.current_operator_acted = False
        for train in entity.trains:
            train.operated = False
        if not self.finished():
            self.log.append(f"{self.game.acting_for_entity(entity).name} operates {entity.name}")
        self.game.place_home_token(entity)
        self.skip_steps()
        if self.finished():
            self.after_end_of_turn(entity)
            self.next_entity()

    def recalculate_order(self):
        unsorted_corps = self.entities[self.entity_index + 1 :]
        self.entities[self.entity_index + 1 :] = [e for e in self.game.operating_order if e in unsorted_corps]

    @property
    def operating(self):
        return True

    def finished(self):
        finished = super().finished() or not self.any_to_act()
        if finished:
            self.current_operator = None
        return finished


class Stock(BaseRound):
    def select_entities(self):
        return [player for player in self.game.players if not player.bankrupt]

    @classmethod
    def short_name(cls):
        return "SR"

    def name(self):
        return "Stock Round"

    def setup(self):
        # set_trace()
        self.skip_steps()
        if not self.active_step():
            self.next_entity()

    def after_process(self, action):
        if not self.active_step():
            self.next_entity()

    def next_entity(self):
        if self.finished():
            self.next_entity_index()
            self.finish_round()
            return

        self.next_entity_index()
        self.start_entity()

    def start_entity(self):
        # set_trace()
        for step in self.steps:
            step.unpass()
            step.setup()

        self.skip_steps()
        if not self.active_step():
            self.next_entity()

    def finished(self):
        return self.game.finished or all(entity.passed for entity in self.entities)

    @property
    def stock(self):
        return True

    def show_auto(self):
        return True

    def finish_round(self):
        # set_trace()
        corporations_to_move_price = sorted(
            [corp for corp in self.game.corporations if corp.floated() and corp.type != "minor"]
        )

        for corp in corporations_to_move_price:
            if corp.share_price:
                old_price = corp.share_price
                if self.sold_out(corp) and self.game.sold_out_increase(corp):
                    self.sold_out_stock_movement(corp)

                pool_share_drop = self.game.POOL_SHARE_DROP
                if pool_share_drop != "none" and corp.num_market_shares > 0:
                    if pool_share_drop == "down_block":
                        self.game.stock_market.move_down(corp)
                    elif pool_share_drop == "down_share":
                        for _ in range(corp.num_market_shares):
                            self.game.stock_market.move_down(corp)
                    elif pool_share_drop == "left_block":
                        self.game.stock_market.move_left(corp)

                self.game.log_share_price(corp, old_price)

    def corporations_to_move_price(self):
        return [corp for corp in self.game.corporations if corp.floated() and corp.type != "minor"]

    def sold_out_stock_movement(self, corp):
        self.game.stock_market.move_up(corp)

    def sold_out(self, corporation):
        return sum(corporation.player_share_holders().values()) == 100
