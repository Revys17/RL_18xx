from rl18xx.game.engine.game import BaseGame
from rl18xx.game import ActionHelper
from rl18xx.game.engine.actions import BaseAction
import torch

from rl18xx.game.engine.actions import (
    Bankrupt,
    Bid,
    BuyCompany,
    BuyShares,
    BuyTrain,
    Dividend,
    LayTile,
    Par,
    Pass,
    PlaceToken,
    RunRoutes,
    SellShares,
)

from rl18xx.game.engine.abilities import Shares as SharesAbility
from rl18xx.game.engine.autorouter import AutoRouter
from rl18xx.game.engine.round import (
    BuyTrain as BuyTrainStep,
    Exchange as ExchangeStep,
    SpecialTrack as SpecialTrackStep,
    SpecialToken as SpecialTokenStep,
)

import logging

LOGGER = logging.getLogger(__name__)

# Action Encoding
# 0: Pass
# 1: MinBid on Private 1 (aka purchase)
# 2: MinBid on Private 2
# 3. MinBid on Private 3
# 4. MinBid on Private 4
# 5. MinBid on Private 5
# 6. MinBid on Private 6
# 7. Par the B&O at 67
# 8. Par the B&O at 71
# 9. Par the B&O at 76
# 10. Par the B&O at 82
# 11. Par the B&O at 90
# 12. Par the B&O at 100

ACTION_DESCRIPTIONS = [
    (Pass, None, None),
    (Bid, "SV", None),
    (Bid, "CS", None),
    (Bid, "DH", None),
    (Bid, "MH", None),
    (Bid, "CA", None),
    (Bid, "BO", None),
    (Par, "B&O", 67),
    (Par, "B&O", 71),
    (Par, "B&O", 76),
    (Par, "B&O", 82),
    (Par, "B&O", 90),
    (Par, "B&O", 100),
]
ACTION_ENCODING_SIZE = len(ACTION_DESCRIPTIONS)
MASK_SIZE = torch.tensor(len(ACTION_DESCRIPTIONS), dtype=torch.float32)


class ActionMapper:
    def get_legal_action_mask(self, state: BaseGame) -> torch.Tensor:
        # Assuming ActionHelper is available via state or globally
        # Replace with actual way to get ActionHelper if different
        helper = ActionHelper(state)
        legal_actions = helper.get_all_choices_limited()

        LOGGER.debug(f"Legal actions: {legal_actions}")

        indices = []
        for action in legal_actions:
            try:
                indices.append(self._get_index_for_action(action))
            except ValueError as e:
                # Should not happen if ActionHelper is correct, but good for debugging
                LOGGER.warning(f"Warning: Skipping unknown action from ActionHelper: {action} ({e})")
                continue

        LOGGER.debug(f"Indices: {indices}")

        mask = torch.zeros(ACTION_ENCODING_SIZE, dtype=torch.float32)
        if indices:
            # Ensure indices are long and src value is float
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            # Use float 1.0, PyTorch should handle dtype matching for scatter_ with scalar src
            mask.scatter_(0, indices_tensor, 1.0)
        return mask

    def _get_index_for_action(self, action: BaseAction) -> int:
        if isinstance(action, Pass):
            return 0
        elif isinstance(action, Bid):
            if not action.company:
                raise ValueError(f"Company is None for bid action: {action}")

            if action.company.id == "SV":
                return 1
            elif action.company.id == "CS":
                return 2
            elif action.company.id == "DH":
                return 3
            elif action.company.id == "MH":
                return 4
            elif action.company.id == "CA":
                return 5
            elif action.company.id == "BO":
                return 6
            else:
                raise ValueError(f"Unknown company for bid action: {action.company}")
        elif isinstance(action, Par):
            if not action.corporation:
                raise ValueError(f"Corporation is None for par action: {action}")

            if action.corporation.id == "B&O":
                if action.share_price.price == 67:
                    return 7
                elif action.share_price.price == 71:
                    return 8
                elif action.share_price.price == 76:
                    return 9
                elif action.share_price.price == 82:
                    return 10
                elif action.share_price.price == 90:
                    return 11
                elif action.share_price.price == 100:
                    return 12
                else:
                    raise ValueError(f"Unknown share price for par action: {action.share_price}")
            else:
                raise ValueError(f"Unknown corporation for par action: {action.corporation}")
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def map_index_to_action(self, index: int, state: BaseGame) -> BaseAction:
        if not (0 <= index < ACTION_ENCODING_SIZE):
            raise IndexError(f"Action index {index} out of bounds (0-{ACTION_ENCODING_SIZE-1})")

        action_type, target_id, price = ACTION_DESCRIPTIONS[index]
        entity = state.current_entity

        if action_type is Pass:
            return Pass(entity)
        elif action_type is Bid:
            company = state.company_by_id(target_id)
            if company is None:
                raise ValueError(f"Company '{target_id}' not found in state for Bid action")
            bid_price = state.active_step().min_bid(company)
            return Bid(entity, bid_price, company=company)
        elif action_type is Par:
            # Find the actual corporation object from the state
            # Assuming state.corporations is a list or dict
            corporation = state.corporation_by_id(target_id)
            if corporation is None:
                raise ValueError(f"Corporation '{target_id}' not found in state for Par action")
            # Assuming Par constructor is Par(entity, corporation, share_price)
            for par_price in state.stock_market.par_prices:
                if par_price.price == price:
                    return Par(entity, corporation, par_price)
            raise ValueError(f"Share price {price} not found in state for Par action")
        else:
            # Should not happen if ACTION_DESCRIPTIONS is correct
            raise TypeError(f"Cannot create action for type: {action_type}")
