__all__ = [
    "BaseAction",
    "AcquireCompany",
    "Assign",
    "Bankrupt",
    "Bid",
    "BlindBid",
    "BorrowTrain",
    "BuyCompany",
    "BuyCorporation",
    "BuyPower",
    "BuyShares",
    "SellShares",
    "BuyToken",
    "BuyTrain",
    "Choose",
    "ChooseAbility",
    "ClaimHexToken",
    "CombinedTrains",
    "Convert",
    "CorporateBuyShares",
    "CorporateSellShares",
    "CreditMobilier",
    "DestinationConnection",
    "DiscardTrain",
    "Dividend",
    "DoubleHeadTrains",
    "EndGame",
    "FailedMerge",
    "HexToken",
    "LayTile",
    "Message",
    "Log",
    "ManualCloseCompany",
    "Merge",
    "MoveBid",
    "Offer",
    "Par",
    "Pass",
    "PayoffDebt",
    "TakeLoan",
    "PayoffPlayerDebt",
    "PlaceToken",
    "ProgramEnable",
    "ProgramDisable",
    "ProgramAuctionBid",
    "ProgramBuyShares",
    "ProgramClosePass",
    "ProgramHarzbahnDraftPass",
    "ProgramIndependentMines",
    "ProgramMergerPass",
    "ProgramSharePass",
    "PurchaseTrain",
    "ReassignTrains",
    "Redo",
    "RemoveHexToken",
    "RemoveToken",
    "Respond",
    "RunRoutes",
    "ScrapTrain",
    "SellCompany",
    "Short",
    "SpecialBuy",
    "Split",
    "SwapTrain",
    "SwitchTrains",
    "Undo",
    "UseGraph",
    "ViewMergeOptions",
    "OperatingInfo",
]


from .core import Item, pascal_to_snake, snake_to_pascal
from .entities import Player, ShareBundle
from .graph import Route


import time
from typing import Any, Dict


class BaseAction:
    def __init__(self, entity):
        self.entity = entity
        self.id = None
        self.user = None
        self.created_at = time.time()
        self.auto_actions = []

    @classmethod
    def action_from_dict(cls, data, game):
        entity_type = data.get("entity_type")
        entity_id = data.get("entity")
        entity = game.get(entity_type, entity_id) or Player(None, entity_id)

        class_name = f"{snake_to_pascal(data.get('type'))}"

        # Dynamically access the class from globals
        if class_name not in globals():
            raise ValueError(f"Action class not found: {class_name}")

        action_class = globals()[class_name]

        obj = action_class(entity, **action_class.dict_to_args(data, game))

        obj.id = data.get("id")
        obj.user = (
            data.get("user")
            if hasattr(entity, "player") and data.get("user") != getattr(entity, "player", None)
            else None
        )
        obj.created_at = data.get("created_at", time.time())
        obj.auto_actions = [BaseAction.action_from_dict(auto_data, game) for auto_data in data.get("auto_actions", [])]

        return obj

    def to_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_dict_cache"):
            self._dict_cache = {
                "type": pascal_to_snake(self.__class__.__name__),
                "entity": self.entity.id,
                "entity_type": pascal_to_snake(self.entity.__class__.__name__),
                "id": self.id,
                "user": self.user,
                "created_at": int(self.created_at),
                "auto_actions": [action.to_dict() for action in self.auto_actions] if self.auto_actions else None,
                **self.args_to_dict(),
            }
            self._dict_cache = {k: v for k, v in self._dict_cache.items() if v is not None}
        return self._dict_cache

    @staticmethod
    def dict_to_args(data: Dict[str, Any], game) -> Dict[str, Any]:
        return {}

    def args_to_dict(self):
        return {}

    def clear_cache(self):
        self._dict_cache = None

    def pass_(self):
        return False

    def copy(self, game):
        return self.from_dict(self.to_h(), game)

    def free(self):
        return False

    def __lt__(self, other):
        # Compare based on id if both have one, otherwise compare based on created_at timestamp
        return (self.id < other.id) if self.id and other.id else (self.created_at < other.created_at)

    # Implementing the rest of the comparison methods if needed
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.id == other.id if self.id and other.id else self.created_at == other.created_at

    def __str__(self):
        return f"Type: {self.__class__.__name__}, id: {self.id}, entity: {self.entity}"

    def __repr__(self):
        return self.__str__()


class AcquireCompany(BaseAction):
    def __init__(self, entity, company):
        super().__init__(entity)
        self.company = company

    @staticmethod
    def dict_to_args(args, game):
        return {"company": game.company_by_id(args["company"])}

    def args_to_dict(self):
        return {
            "company": self.company.id,
        }


class Assign(BaseAction):
    def __init__(self, entity, target):
        super().__init__(entity)
        self.target = target

    @staticmethod
    def dict_to_args(args, game):
        return {
            "target": game.get(args["target_type"], args["target"]),
        }

    def args_to_dict(self):
        return {
            "target": self.target.id,
            "target_type": self.target.__class__.__name__,
        }


class Bankrupt(BaseAction):
    def __init__(self, entity, option=None):
        super().__init__(entity)
        self.option = option

    @staticmethod
    def dict_to_args(args, game):
        return {"option": args["option"]}

    def args_to_dict(self):
        return {
            "option": self.option,
        }


class Bid(BaseAction):
    def __init__(self, entity, price, company=None, corporation=None, minor=None):
        super().__init__(entity)
        self.company = company
        self.corporation = corporation
        self.minor = minor
        self.price = price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "company": game.company_by_id(args["company"]) if args.get("company") else None,
            "corporation": game.corporation_by_id(args["corporation"]) if args.get("corporation") else None,
            "minor": game.minor_by_id(args["minor"]) if args.get("minor") else None,
            "price": args["price"],
        }

    def args_to_dict(self):
        return {
            "company": self.company.id if self.company else None,
            "corporation": self.corporation.id if self.corporation else None,
            "minor": self.minor.id if self.minor else None,
            "price": self.price,
        }

    def __str__(self):
        if self.company:
            string = f", company: {self.company}, "
        elif self.corporation:
            string = f", corporation: {self.corporation}, "
        elif self.minor:
            string = f", minor: {self.minor}, "

        return super().__str__() + string + f"price: {self.price}"


class BlindBid(BaseAction):
    def __init__(self, entity, bids=None):
        super().__init__(entity)
        self.bids = bids or []

    @staticmethod
    def dict_to_args(args, game):
        return {
            "bids": [int(bid) for bid in args.get["bids"]],
        }

    def args_to_dict(self):
        return {"bids": [str(bid) for bid in self.bids]}


class BorrowTrain(BaseAction):
    def __init__(self, entity, train):
        super().__init__(entity)
        self.train = train

    @staticmethod
    def dict_to_args(args, game):
        return {
            "train": game.train_by_id(args["train"]),
        }

    def args_to_dict(self):
        return {"train": self.train.id}


class BuyCompany(BaseAction):
    def __init__(self, entity, company, price):
        super().__init__(entity)
        self.company = company
        self.price = price

    @staticmethod
    def dict_to_args(args, game):
        return {"company": game.company_by_id(args["company"]), "price": args["price"]}

    def args_to_dict(self):
        return {"company": self.company.id, "price": self.price}

    def __str__(self):
        return super().__str__() + f", company: [{self.company}], price: [{self.price}]"


class BuyCorporation(BaseAction):
    def __init__(self, entity, price, corporation=None, minor=None):
        super().__init__(entity)
        self.corporation = corporation
        self.minor = minor
        self.price = price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
            "minor": game.minor_by_id(args["minor"]),
            "price": args["price"],
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id if self.corporation else None,
            "minor": self.minor.id if self.minor else None,
            "price": self.price,
        }


class BuyPower(BaseAction):
    def __init__(self, entity, power):
        super().__init__(entity)
        self.power = power

    @staticmethod
    def dict_to_args(args, game):
        return {"power": args["power"]}

    def args_to_dict(self):
        return {"power": self.power}


class BuyShares(BaseAction):
    def __init__(
        self,
        entity,
        shares,
        share_price=None,
        percent=None,
        swap=None,
        purchase_for=None,
        borrow_from=None,
        total_price=None,
    ):
        super().__init__(entity)
        self.bundle = ShareBundle(shares if isinstance(shares, list) else [shares], percent)
        self.bundle.share_price = share_price
        self.swap = swap
        self.purchase_for = purchase_for
        self.borrow_from = borrow_from
        self.total_price = total_price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "shares": [game.share_by_id(id) for id in args["shares"]],
            "share_price": args.get("share_price"),
            "percent": args.get("percent"),
            "swap": game.share_by_id(args["swap"]) if args.get("swap") else None,
            "purchase_for": game.get(args["purchase_for_type"], args["purchase_for"])
            if args.get("purchase_for")
            else None,
            "borrow_from": game.get(args["borrow_from_type"], args["borrow_from"]) if args.get("borrow_from") else None,
            "total_price": args.get("total_price"),
        }

    def args_to_dict(self):
        return {
            "shares": [share.id for share in self.bundle.shares],
            "percent": self.bundle.percent,
            "share_price": self.bundle.share_price,
            "swap": self.swap.id if self.swap else None,
            "purchase_for_type": self.purchase_for.__class__.__name__ if self.purchase_for else None,
            "purchase_for": self.purchase_for.id if self.purchase_for else None,
            "borrow_from_type": self.borrow_from.__class__.__name__ if self.borrow_from else None,
            "borrow_from": self.borrow_from.id if self.borrow_from else None,
            "total_price": self.total_price,
        }

    def __str__(self):
        return super().__str__() + f", bundle: [{self.bundle}]"


class SellShares(BaseAction):
    def __init__(self, entity, shares, share_price=None, percent=None, swap=None):
        super().__init__(entity)
        if isinstance(shares, ShareBundle):
            self.bundle = shares
        else:
            self.bundle = ShareBundle(shares if isinstance(shares, list) else [shares], percent)
        self.bundle.share_price = share_price
        self.swap = swap

    @staticmethod
    def dict_to_args(args, game):
        return {
            "shares": [game.share_by_id(id) for id in args["shares"]],
            "share_price": args.get("share_price"),
            "percent": args.get("percent"),
            "swap": game.share_by_id(args["swap"]) if args.get("swap") else None,
        }

    def args_to_dict(self):
        return {
            "shares": [share.id for share in self.bundle.shares],
            "share_price": self.bundle.share_price,
            "percent": self.bundle.percent,
            "swap": self.swap.id if self.swap else None,
        }

    def __str__(self):
        return super().__str__() + f", bundle: [{self.bundle}]"


class BuyToken(BaseAction):
    def __init__(self, entity, city, slot, price):
        super().__init__(entity)
        self.city = city
        self.slot = slot
        self.price = price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "city": game.city_by_id(args["city"]),
            "slot": args["slot"],
            "price": args["price"],
        }

    def args_to_dict(self):
        return {"city": self.city.id, "slot": self.slot, "price": self.price}


class BuyTrain(BaseAction):
    def __init__(
        self,
        entity,
        train,
        price,
        variant=None,
        exchange=None,
        shell=None,
        slots=None,
        extra_due=None,
        warranties=None,
    ):
        super().__init__(entity)
        self.train = train
        self.price = price
        self.variant = variant
        self.exchange = exchange
        self.shell = shell
        self.slots = slots if slots is not None else []
        self.extra_due = extra_due
        self.warranties = warranties

    @staticmethod
    def shell_by_name(name, game):
        if not name:
            return None

        for system in [corp for corp in game.corporations if corp.system]:
            for shell in system.shells:
                if shell.name == name:
                    return shell

        return None

    @staticmethod
    def dict_to_args(args, game):
        return {
            "train": game.train_by_id(args["train"]),
            "price": args["price"],
            "variant": args.get("variant"),
            "exchange": game.train_by_id(args["exchange"]) if args.get("exchange") else None,
            "shell": BuyTrain.shell_by_name(args.get("shell"), game),
            "slots": args.get("slots", []),
            "extra_due": args.get("extra_due"),
            "warranties": args.get("warranties"),
        }

    def args_to_dict(self):
        return {
            "train": self.train.id,
            "price": self.price,
            "variant": self.variant,
            "exchange": self.exchange.id if self.exchange else None,
            "shell": self.shell.name if self.shell else None,
            "slots": self.slots,
            "extra_due": self.extra_due,
            "warranties": self.warranties,
        }

    def __str__(self):
        return super().__str__() + f", train: [{self.train}], price: {self.price}"


class Choose(BaseAction):
    def __init__(self, entity, choice):
        super().__init__(entity)
        self.choice = choice

    @staticmethod
    def dict_to_args(args, game):
        return {"choice": args["choice"]}

    def args_to_dict(self):
        return {
            "choice": self.choice,
        }


class ChooseAbility(Choose):
    pass


class ClaimHexToken(BaseAction):
    def __init__(self, entity, hex, token_type=None):
        super().__init__(entity)
        self.hex = hex
        self.token_type = token_type

    @staticmethod
    def dict_to_args(args, game):
        return {
            "hex": game.hex_by_id(args["hex"]),
            "token_type": args["token_type"],
        }

    def args_to_dict(self):
        return {
            "hex": self.hex.id,
            "token_type": self.token_type,
        }


class CombinedTrains(BaseAction):
    def __init__(self, entity, base, additional_train, additional_train_variant):
        super().__init__(entity)
        self.base = base
        self.additional_train = additional_train
        self.additional_train_variant = additional_train_variant

    @staticmethod
    def dict_to_args(args, game):
        return {
            "base": game.train_by_id(args["base"]),
            "additional_train": game.train_by_id(args["additional_train"]),
            "additional_train_variant": args["additional_train_variant"],
        }

    def args_to_dict(self):
        return {
            "base": self.base.id,
            "additional_train": self.additional_train.id,
            "additional_train_variant": self.additional_train_variant,
        }


class Convert(BaseAction):
    pass


class CorporateBuyShares(BuyShares):
    pass


class CorporateSellShares(SellShares):
    pass


class CreditMobilier(BaseAction):
    def __init__(self, entity, hex, amount):
        super().__init__(entity)
        self.hex = hex
        self.amount = amount

    @staticmethod
    def dict_to_args(args, game):
        return {"hex": game.hex_by_id(args["hex"]), "amount": args["amount"]}

    def args_to_dict(self):
        return {"hex": self.hex.id, "amount": self.amount}


class DestinationConnection(BaseAction):
    def __init__(self, entity, corporations=None, minors=None, hexes=None):
        super().__init__(entity)
        self.corporations = corporations if corporations else []
        self.minors = minors if minors else []
        self.hexes = hexes if hexes else []

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporations": [game.corporation_by_id(c) for c in args.get("corporations", [])],
            "minors": [game.minor_by_id(m) for m in args.get("minors", [])],
            "hexes": [game.hex_by_id(h) for h in args.get("hexes", [])],
        }

    def args_to_dict(self):
        return {
            "corporations": [corp.id for corp in self.corporations] if self.corporations else [],
            "minors": [minor.id for minor in self.minors] if self.minors else [],
            "hexes": [hex.id for hex in self.hexes] if self.hexes else [],
        }


class DiscardTrain(BaseAction):
    def __init__(self, entity, train):
        super().__init__(entity)
        self.train = train

    @staticmethod
    def dict_to_args(args, game):
        return {
            "train": game.train_by_id(args["train"]),
        }

    def args_to_dict(self):
        return {
            "train": self.train.id,
        }


class Dividend(BaseAction):
    def __init__(self, entity, kind, amount=None):
        super().__init__(entity)
        self.kind = kind
        self.amount = amount

    @staticmethod
    def dict_to_args(args, _game):
        return {
            "kind": args.get("kind"),
            "amount": args.get("amount"),
        }

    def args_to_dict(self):
        return {
            "kind": self.kind,
            "amount": self.amount,
        }

    def __str__(self):
        return super().__str__() + f", kind: [{self.kind}], amount: {self.amount}"


class DoubleHeadTrains(BaseAction):
    def __init__(self, entity, trains):
        super().__init__(entity)
        self.trains = trains

    @staticmethod
    def dict_to_args(args, game):
        return {
            "trains": [game.train_by_id(t) for t in args.get("trains", [])],
        }

    def args_to_dict(self):
        return {
            "trains": [train.id for train in self.trains],
        }


class EndGame(BaseAction):
    def free(self):
        return True


class FailedMerge(BaseAction):
    def __init__(self, entity, corporations):
        super().__init__(entity)
        self.corporations = corporations

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporations": [game.corporation_by_id(c_id) for c_id in args.get("corporations", [])],
        }

    def args_to_dict(self):
        return {
            "corporations": [corporation.id for corporation in self.corporations],
        }


class HexToken(BaseAction):
    def __init__(self, entity, hex, cost=None, token_type=None, token=None):
        super().__init__(entity)
        self.hex = hex
        self.cost = cost
        self.token = token or entity.find_token_by_type(token_type)

    @staticmethod
    def dict_to_args(args, game):
        return {
            "hex": game.hex_by_id(args["hex"]),
            "cost": args["cost"],
            "token_type": args.get("token_type"),
        }

    def args_to_dict(self):
        return {
            "hex": self.hex.id,
            "cost": self.cost,
            "token_type": None if self.token.type == "normal" else self.token.type,
        }


class LayTile(BaseAction):
    def __init__(self, entity, tile, hex, rotation, combo_entities=None):
        super().__init__(entity)
        self.hex = hex
        self.tile = tile
        self.rotation = rotation
        self.combo_entities = combo_entities or []

    @staticmethod
    def dict_to_args(args, game):
        return {
            "tile": game.tile_by_id(args["tile"]),
            "hex": game.hex_by_id(args["hex"]),
            "rotation": args["rotation"],
            "combo_entities": [game.company_by_id(id) for id in args.get("combo_entities", [])],
        }

    def args_to_dict(self):
        return {
            "hex": self.hex.id,
            "tile": self.tile.id,
            "rotation": self.rotation,
            "combo_entities": [entity.id for entity in self.combo_entities] if self.combo_entities else None,
        }

    def __str__(self):
        return super().__str__() + f", hex: [{self.hex}], tile: [{self.tile}, rotation: [{self.rotation}]"


class Message(BaseAction):
    def __init__(self, entity, message):
        super().__init__(entity)
        self.message = message

    def free(self):
        return True

    @staticmethod
    def dict_to_args(args, _):
        return {
            "message": args["message"],
        }

    def args_to_dict(self):
        return {
            "message": self.message,
        }


class Log(Message):
    pass


class ManualCloseCompany(BaseAction):
    pass


class Merge(BaseAction):
    def __init__(self, entity, corporation=None, minor=None):
        super().__init__(entity)
        self.corporation = corporation
        self.minor = minor

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
            "minor": game.minor_by_id(args["minor"]),
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id if self.corporation else None,
            "minor": self.minor.id if self.minor else None,
        }


class MoveBid(BaseAction):
    def __init__(self, entity, price, from_company, from_price, company=None, corporation=None):
        super().__init__(entity)
        self.company = company
        self.corporation = corporation
        self.price = price
        self.from_company = from_company
        self.from_price = from_price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "company": game.company_by_id(args["company"]),
            "corporation": game.corporation_by_id(args["corporation"]),
            "from_company": game.company_by_id(args.get("from_company")),
            "price": args["price"],
            "from_price": args.get("from_price"),
        }

    def args_to_dict(self):
        return {
            "company": self.company.id if self.company else None,
            "corporation": self.corporation.id if self.corporation else None,
            "from_company": self.from_company.id if self.from_company else None,
            "price": self.price,
            "from_price": self.from_price,
        }


class Offer(BaseAction):
    def __init__(self, entity, corporation=None, company=None, price=None):
        super().__init__(entity)
        self.corporation = corporation
        self.company = company
        self.price = price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
            "company": game.company_by_id(args["company"]),
            "price": args["price"],
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id if self.corporation else None,
            "company": self.company.id if self.company else None,
            "price": self.price,
        }


class Par(BaseAction):
    def __init__(
        self,
        entity,
        corporation,
        share_price,
        slot=None,
        purchase_for=None,
        borrow_from=None,
    ):
        super().__init__(entity)
        self.corporation = corporation
        self.share_price = share_price
        self.slot = slot
        self.purchase_for = purchase_for
        self.borrow_from = borrow_from

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
            "share_price": game.share_price_by_id(args["share_price"]),
            "slot": args.get("slot"),
            "purchase_for": game.get(args.get("purchase_for_type"), args["purchase_for"])
            if "purchase_for" in args
            else None,
            "borrow_from": game.get(args.get("borrow_from_type"), args["borrow_from"])
            if "borrow_from" in args
            else None,
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id,
            "share_price": self.share_price.id,
            "slot": self.slot,
            "purchase_for_type": self.purchase_for.__class__.__name__ if self.purchase_for else None,
            "purchase_for": self.purchase_for.id if self.purchase_for else None,
            "borrow_from_type": self.borrow_from.__class__.__name__ if self.borrow_from else None,
            "borrow_from": self.borrow_from.id if self.borrow_from else None,
        }

    def __str__(self):
        return super().__str__() + f", corporation: [{self.corporation}], par price: [{self.share_price}]"


class Pass(BaseAction):
    def pass_(self):
        return True


class PayoffDebt(BaseAction):
    pass


class TakeLoan(BaseAction):
    def __init__(self, entity, loan):
        super().__init__(entity)
        self.loan = loan

    @classmethod
    def from_dict(cls, data, game):
        return cls(
            entity=game.get_entity(data["entity_type"], data["entity"]),
            loan=game.get_loan(data["loan"]),
        )

    def to_dict(self):
        return {
            "type": "take_loan",
            "entity": self.entity.id,
            "entity_type": self.entity.type,
            "loan": self.loan.id if self.loan else None,
        }


class PayoffDebt(TakeLoan):
    pass


class PayoffPlayerDebt(BaseAction):
    pass


class PlaceToken(BaseAction):
    def __init__(self, entity, city, slot, cost=None, tokener=None, token_type=None):
        super().__init__(entity)
        self.city = city
        self.slot = slot
        self.cost = cost
        self.tokener = tokener
        token_owner = tokener or (entity.owner if entity.is_company() else entity)
        self.token = token_owner.find_token_by_type(token_type)

    @staticmethod
    def dict_to_args(args, game):
        return {
            "city": game.city_by_id(args.get("city")),
            "slot": args.get("slot"),
            "cost": args.get("cost"),
            "tokener": game.corporation_by_id(args.get("tokener")) or game.minor_by_id(args.get("tokener")),
            "token_type": args.get("token_type"),
        }

    def args_to_dict(self):
        return {
            "city": self.city.id,
            "slot": self.slot,
            "cost": self.cost,
            "tokener": self.tokener.id if self.tokener else None,
            "token_type": self.token.type if self.token and self.token.type != "normal" else None,
        }

    def __str__(self):
        return (
            super().__str__()
            + f", city: [{self.city}], slot: [{self.slot}, cost: [{self.cost}, tokener: [{self.tokener}]"
        )


class ProgramEnable(BaseAction):
    def disable(self, game):
        return True


class ProgramDisable(BaseAction):
    def __init__(self, entity, reason, original_type=None):
        super().__init__(entity)
        self.reason = reason
        self.original_type = original_type

    @staticmethod
    def dict_to_args(args, _game):
        return {
            "reason": args["reason"],
            "original_type": args["original_type"],
        }

    def args_to_dict(self):
        return {
            "reason": self.reason,
            "original_type": self.original_type,
        }


class ProgramAuctionBid(ProgramEnable):
    def __init__(
        self,
        entity,
        bid_target,
        maximum_bid,
        buy_price,
        enable_maximum_bid=False,
        enable_buy_price=False,
        auto_pass_after=False,
    ):
        super().__init__(entity)
        self.bid_target = bid_target
        self.enable_maximum_bid = enable_maximum_bid
        self.maximum_bid = maximum_bid
        self.enable_buy_price = enable_buy_price
        self.buy_price = buy_price
        self.auto_pass_after = auto_pass_after

    def disable(self, game):
        return not game.round.auction

    @staticmethod
    def dict_to_args(args, game):
        bid_target = (
            game.corporation_by_id(args.get("bid_target"))
            or game.company_by_id(args.get("bid_target"))
            or game.minor_by_id(args.get("bid_target"))
        )
        return {
            "bid_target": bid_target,
            "enable_maximum_bid": args.get("enable_maximum_bid"),
            "maximum_bid": args.get("maximum_bid"),
            "enable_buy_price": args.get("enable_buy_price"),
            "buy_price": args.get("buy_price"),
            "auto_pass_after": args.get("auto_pass_after"),
        }

    def args_to_dict(self):
        return {
            "bid_target": self.bid_target.id,
            "enable_maximum_bid": self.enable_maximum_bid,
            "maximum_bid": self.maximum_bid,
            "enable_buy_price": self.enable_buy_price,
            "buy_price": self.buy_price,
            "auto_pass_after": self.auto_pass_after,
        }

    def __str__(self):
        buy = f"Buy if price at {self.buy_price}. " if self.enable_buy_price else ""
        bid = f"Bid on {self.bid_target.name} up to {self.maximum_bid}. " if self.enable_maximum_bid else ""
        suffix = "Otherwise auto pass." if self.auto_pass_after else ""

        return f"{buy}{bid}{suffix}"


class ProgramBuyShares(ProgramEnable):
    def __init__(
        self,
        entity,
        corporation,
        until_condition,
        from_market=False,
        auto_pass_after=False,
    ):
        super().__init__(entity)
        self.corporation = corporation
        self.until_condition = until_condition
        self.from_market = from_market
        self.auto_pass_after = auto_pass_after

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
            "until_condition": args.get("until_condition"),
            "from_market": args.get("from_market"),
            "auto_pass_after": args.get("auto_pass_after"),
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id,
            "until_condition": self.until_condition,
            "from_market": self.from_market,
            "auto_pass_after": self.auto_pass_after,
        }

    def __str__(self):
        source = "market" if self.from_market else "IPO"
        condition = "floated" if self.until_condition == "float" else f"{self.until_condition} shares"
        suffix = ", then auto pass" if self.auto_pass_after else ""

        return f"Buy {self.corporation.name} from {source} until {condition}{suffix}"

    def disable(self, game):
        return not game.round.stock


class ProgramClosePass(ProgramEnable):
    def __init__(self, entity, unconditional=False):
        super().__init__(entity)
        self.unconditional = unconditional

    @staticmethod
    def dict_to_args(args, game):
        return {
            "unconditional": args["unconditional"],
        }

    def args_to_dict(self):
        return {
            "unconditional": self.unconditional,
        }

    def __str__(self):
        unconditionally = ", unconditionally" if self.unconditional else ""
        return f"Pass in Closing Round{unconditionally}"

    def disable(self, _game):
        return False


class ProgramHarzbahnDraftPass(ProgramEnable):
    def __init__(self, entity, until_premium, unconditional):
        super().__init__(entity)
        self.until_premium = until_premium
        self.unconditional = unconditional

    @staticmethod
    def dict_to_args(args, game):
        return {
            "until_premium": args["until_premium"],
            "unconditional": args["unconditional"],
        }

    def args_to_dict(self):
        return {
            "until_premium": self.until_premium,
            "unconditional": self.unconditional,
        }

    def __str__(self):
        until_premium = f", until premium {self.until_premium}" if self.until_premium else ""
        unconditionally = ", unconditionally" if self.unconditional else ""
        return f"Pass in Draft{until_premium}{unconditionally}"

    def disable(self, game):
        return not game.round.auction


class ProgramIndependentMines(ProgramEnable):
    def __init__(self, entity, skip_track, skip_buy, skip_close, indefinite):
        super().__init__(entity)
        self.skip_track = skip_track
        self.skip_buy = skip_buy
        self.skip_close = skip_close
        self.indefinite = indefinite

    @staticmethod
    def dict_to_args(args, game):
        return {
            "skip_track": args["skip_track"],
            "skip_buy": args["skip_buy"],
            "skip_close": args["skip_close"],
            "indefinite": args["indefinite"],
        }

    def args_to_dict(self):
        return {
            "skip_track": self.skip_track,
            "skip_buy": self.skip_buy,
            "skip_close": self.skip_close,
            "indefinite": self.indefinite,
        }

    def __str__(self):
        steps = []
        if self.skip_track:
            steps.append("track")
        if self.skip_buy:
            steps.append("buy trains")
        if self.skip_close:
            steps.append("close")
        if not steps:
            steps.append("nothing?!")
        condition = "turned off" if self.indefinite else "next SR"
        return f"Pass ({','.join(steps)}) for independent mines until {condition}"

    def disable(self, game):
        return not (game.round.operating or self.indefinite)


class ProgramMergerPass(ProgramEnable):
    def __init__(self, entity, corporations_by_round, options):
        super().__init__(entity)
        self.corporations_by_round = corporations_by_round
        self.options = options

    @staticmethod
    def dict_to_args(args, game):
        corporations_by_round = args.get("corporations_by_round", {})
        transformed_corps = {}
        for phase, corps_ids in corporations_by_round.items():
            corps_list = [game.corporation_by_id(c_id) for c_id in corps_ids]
            transformed_corps[phase] = corps_list
        return {
            "corporations_by_round": transformed_corps,
            "options": args.get("options", []),
        }

    def args_to_dict(self):
        transformed_corps = {}
        for phase, corps_list in self.corporations_by_round.items():
            corps_ids = [corp.id for corp in corps_list]
            transformed_corps[phase] = corps_ids
        return {
            "corporations_by_round": transformed_corps,
            "options": self.options,
        }

    def __str__(self):
        phases = [
            f"{phase} ({', '.join(corp.name for corp in corps)})" if corps else f"{phase} (none)"
            for phase, corps in self.corporations_by_round.items()
        ]
        phases_str = " and ".join(phases)
        suffix = ", unless someone else acts" if "disable_others" in self.options else ""
        return f"Pass on mergers in {phases_str}{suffix}"

    def disable(self, game):
        return not game.round.merger


class ProgramSharePass(ProgramEnable):
    def __init__(self, entity, unconditional=False, indefinite=False):
        super().__init__(entity)
        self.unconditional = unconditional
        self.indefinite = indefinite

    @staticmethod
    def dict_to_args(args, game):
        return {
            "unconditional": args.get("unconditional"),
            "indefinite": args.get("indefinite"),
        }

    def args_to_dict(self):
        return {
            "unconditional": self.unconditional,
            "indefinite": self.indefinite,
        }

    def to_s(self):
        unconditionally = ", unconditionally" if self.unconditional else ""
        indefinitely = ", indefinitely" if self.indefinite else ""
        return f"Pass in Stock Round{unconditionally}{indefinitely}"

    def disable(self, game):
        return not game.round.stock and not self.indefinite


class PurchaseTrain(BaseAction):
    pass


class ReassignTrains(BaseAction):
    def __init__(self, entity, assignments=None):
        super().__init__(entity)
        self.assignments = assignments if assignments else []

    @staticmethod
    def dict_to_args(args, game):
        assignments = [
            {
                "train": game.train_by_id(assignment["train"]),
                "corporation": game.corporation_by_id(assignment["corporation"]),
            }
            for assignment in args.get("assignments", [])
        ]
        return {
            "assignments": assignments,
        }

    def args_to_dict(self):
        assignments = [
            {
                "train": item["train"].id,
                "corporation": item["corporation"].id,
            }
            for item in self.assignments
        ]

        return {
            "assignments": assignments,
        }


class Redo(BaseAction):
    def free(self):
        return True


class RemoveHexToken(BaseAction):
    def __init__(self, entity, hex):
        super().__init__(entity)
        self.hex = hex

    @staticmethod
    def dict_to_args(args, game):
        return {
            "hex": game.hex_by_id(args["hex"]),
        }

    def args_to_dict(self):
        return {
            "hex": self.hex.id,
        }


class RemoveToken(BaseAction):
    def __init__(self, entity, city, slot):
        super().__init__(entity)
        self.city = city
        self.slot = slot

    @staticmethod
    def dict_to_args(args, game):
        return {
            "city": game.city_by_id(args["city"]),
            "slot": args["slot"],
        }

    def args_to_dict(self):
        return {
            "city": self.city.id,
            "slot": self.slot,
        }


class Respond(BaseAction):
    def __init__(self, entity, corporation, company, accept):
        super().__init__(entity)
        self.corporation = corporation
        self.company = company
        self.accept = accept

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
            "company": game.company_by_id(args["company"]),
            "accept": args["accept"] == "true",
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id,
            "company": self.company.id,
            "accept": "true" if self.accept else "false",
        }


class RunRoutes(BaseAction):
    def __init__(self, entity, routes, extra_revenue=0):
        super().__init__(entity)
        self.routes = routes
        self.extra_revenue = extra_revenue

    @staticmethod
    def dict_to_args(args, game):
        routes = []
        for route in args.get("routes", []):
            opts = {
                "connection_hexes": route["connections"],
                "hexes": [game.hex_by_id(id) for id in route.get("hexes", []) if id],
                "revenue": route.get("revenue"),
                "revenue_str": route.get("revenue_str"),
                "subsidy": route.get("subsidy"),
                "halts": route.get("halts"),
                "abilities": route.get("abilities"),
                "nodes": route.get("nodes"),
            }
            # Filter out keys with None values
            opts = {k: v for k, v in opts.items() if v is not None}

            routes.append(
                Route(
                    game,
                    game.phase,
                    game.train_by_id(route["train"]),
                    routes=routes,
                    **opts,
                )
            )

        return {
            "routes": routes,
            "extra_revenue": args.get("extra_revenue"),
        }

    def args_to_dict(self):
        routes = [
            {
                "train": route.train.id,
                "connections": route.connection_hexes,
                "hexes": [hex.id for hex in route.hexes],
                "revenue": route.revenue,
                "revenue_str": route.revenue_str,
                "subsidy": route.subsidy,
                "halts": route.halts,
                "abilities": route.abilities,
                "nodes": route.node_signatures,
            }
            for route in self.routes
            if route is not None
        ]

        return {
            "routes": routes,
            "extra_revenue": self.extra_revenue,
        }


class ScrapTrain(BaseAction):
    def __init__(self, entity, train):
        super().__init__(entity)
        self.train = train

    @staticmethod
    def dict_to_args(args, game):
        return {
            "train": game.train_by_id(args["train"]),
        }

    def args_to_dict(self):
        return {
            "train": self.train.id,
        }


class SellCompany(BaseAction):
    def __init__(self, entity, company, price):
        super().__init__(entity)
        self.company = company
        self.price = price

    @staticmethod
    def dict_to_args(args, game):
        return {
            "company": game.company_by_id(args["company"]),
            "price": args["price"],
        }

    def args_to_dict(self):
        return {
            "company": self.company.id,
            "price": self.price,
        }


class Short(BaseAction):
    def __init__(self, entity, corporation):
        super().__init__(entity)
        self.corporation = corporation

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id,
        }


class SpecialBuy(BaseAction):
    def __init__(self, entity, item):
        super().__init__(entity)
        self.item = item

    @staticmethod
    def dict_to_args(args, _game):
        return {
            "item": Item(description=args["description"], cost=args["cost"]),
        }

    def args_to_dict(self):
        return {
            "description": self.item.description,
            "cost": self.item.cost,
        }


class Split(BaseAction):
    def __init__(self, entity, corporation):
        super().__init__(entity)
        self.corporation = corporation

    @staticmethod
    def dict_to_args(args, game):
        return {
            "corporation": game.corporation_by_id(args["corporation"]),
        }

    def args_to_dict(self):
        return {
            "corporation": self.corporation.id,
        }


class SwapTrain(BaseAction):
    def __init__(self, entity, train):
        super().__init__(entity)
        self.train = train

    @staticmethod
    def dict_to_args(args, game):
        return {
            "train": game.train_by_id(args["train"]),
        }

    def args_to_dict(self):
        return {
            "train": self.train.id,
        }


class SwitchTrains(BaseAction):
    def __init__(self, entity, slots=None):
        super().__init__(entity)
        self.slots = slots

    @staticmethod
    def dict_to_args(args, _game):
        return {
            "slots": [int(m) for m in args.get("slots", [])] if "slots" in args else None,
        }

    def args_to_dict(self):
        return {
            "slots": self.slots,
        }


class Undo(BaseAction):
    def __init__(self, entity, action_id=None):
        super().__init__(entity)
        self.action_id = action_id

    @staticmethod
    def dict_to_args(args, _game):
        return {
            "action_id": args.get("action_id"),
        }

    def args_to_dict(self):
        return {
            "action_id": self.action_id,
        }


class UseGraph(BaseAction):
    def __init__(self, entity, graph_id):
        super().__init__(entity)
        self.graph_id = graph_id

    @staticmethod
    def dict_to_args(args, _game):
        return {
            "graph_id": args["graph_id"],
        }

    def args_to_dict(self):
        return {
            "graph_id": self.graph_id,
        }


class ViewMergeOptions(BaseAction):
    pass


class OperatingInfo:
    def __init__(self, runs, dividend, revenue, laid_hexes, dividend_kind=None):
        self.routes = {run.train: run.connection_hexes for run in runs}
        self.halts = {run.train: run.halts for run in runs}
        self.nodes = {run.train: run.node_signatures for run in runs}
        self.revenue = revenue
        self.dividend = dividend
        self.laid_hexes = laid_hexes
        self.dividend_kind = dividend_kind or (dividend.kind if isinstance(dividend, Dividend) else "withhold")
