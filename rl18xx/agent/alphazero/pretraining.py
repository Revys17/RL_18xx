from pathlib import Path
import json
import random
from typing import Any, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.checkpointer import get_model_from_path
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.encoder import Encoder_1830, Encoder_GNN, Encoder_SSME
from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.train import TrainingMetrics, train_model
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.engine.actions import BaseAction, Bid, BuyShares, BuyTrain, DiscardTrain, LayTile, PlaceToken, RunRoutes
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.dataset import HumanPlayDataset, TrainingExampleProcessor
from rl18xx.game.engine.round import BuyCompany as BuyCompanyStep, BuyTrain as BuyTrainStep, Track as TrackStep, BuySellParShares
import logging

LOGGER = logging.getLogger(__name__)

def load_games_from_json(database_path: str):
    # Games are in json format
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")
    
    game_files = list(path.glob("*.json"))
    if not game_files:
        raise FileNotFoundError(f"No game files found in {database_path}")
    
    games = []
    for game_file in game_files:
        with open(game_file, "r") as f:
            game = json.load(f)
            game["id"] = game_file.stem
            games.append(game)
    
    return games

def filter_to_completed_4_player_games(games: list[dict]):
    filtered_games = []
    for game in games:
        if game["status"] != "finished":
            continue

        if len(game["players"]) != 4:
            continue

        should_skip = False
        for action in game["actions"]:
            if action["type"] == "end_game":
                should_skip = True
                break

        if should_skip:
            continue

        filtered_games.append(game)
    return filtered_games

def filter_actions(actions: list[dict]) -> list[dict]:
    filtered_actions_history = []
    filtered_actions = []
    # First, remove the actions that should be removed by undo/redo along with all messages
    for action in actions:
        if action["type"] == "message":
            continue
        if action["type"] == "undo":
            filtered_actions_history.append(filtered_actions.copy())
            if action.get("action_id", None):
                filtered_actions = [x for x in filtered_actions if x["id"] <= action["action_id"]]
            else:
                filtered_actions = filtered_actions[:-1]
        elif action["type"] == "redo":
            filtered_actions = filtered_actions_history.pop()
        else:
            filtered_actions.append(action.copy())

    # Next, flatten auto_actions
    flattened_actions = []
    for action in filtered_actions:
        flattened_actions.append(action)
        if action.get("auto_actions", None):
            flattened_actions.extend(action["auto_actions"].copy())
            del action["auto_actions"]
    filtered_actions = flattened_actions

    # Next, remove "program_" actions
    filtered_actions = [x for x in filtered_actions if not x["type"].startswith("program_")]

    # Next, reset the id on all actions
    for i, action in enumerate(filtered_actions):
        action["id"] = i + 1

    #LOGGER.debug(f"actions: {json.dumps(actions, indent=4)}")
    #LOGGER.debug(f"filtered_actions: {json.dumps(filtered_actions, indent=4)}")
    return filtered_actions

def check_action_in_all_actions(action, all_actions):
    # Don't check runroutes
    if isinstance(action, RunRoutes):
        return None

    assertion = False
    for a in all_actions:
        if action.__class__ != a.__class__:
            continue
        if action.entity.id != a.entity.id:
            continue

        action_args = action.args_to_dict()
        expected_args = a.args_to_dict()
        if isinstance(action, BuyTrain):
            action_args["variant"] = None
            expected_args["variant"] = None
        elif isinstance(action, BuyShares):
            action_args["shares"] = [share.split("_")[0] for share in action_args["shares"]]
            expected_args["shares"] = [share.split("_")[0] for share in expected_args["shares"]]
            action_args["share_price"] = None
        elif isinstance(action, PlaceToken):
            expected_args["tokener"] = action_args["tokener"]
            expected_args["slot"] = action_args["slot"]
        elif isinstance(action, LayTile):
            action_args["tile"] = action_args["tile"].split("-")[0]
            expected_args["tile"] = expected_args["tile"].split("-")[0]
        elif isinstance(action, Bid):
            if action_args["price"] % 5 != 0:
                # Round down to the nearest multiple of 5
                action_args["price"] = action_args["price"] - (action_args["price"] % 5)
                expected_args["price"] = expected_args["price"] - (expected_args["price"] % 5)
        elif isinstance(action, DiscardTrain):
            action_args['train'] = action_args['train'].split("-")[0]
            expected_args['train'] = expected_args['train'].split("-")[0]

        if action_args != expected_args:
            continue
        assertion = True
    if not assertion:
        LOGGER.debug(f"action class: {action.__class__}")
        LOGGER.debug(f"action entity: {action.entity}")
        LOGGER.debug(f"action args: {action.args_to_dict()}")
        for a in all_actions:
            LOGGER.debug(f"a class: {a.__class__}")
            LOGGER.debug(f"a entity: {a.entity}")
            LOGGER.debug(f"a args: {a.args_to_dict()}")

        if isinstance(action, PlaceToken):
            # Find if all_actions contains a PlaceToken with the same tile
            for a in all_actions:
                if isinstance(a, PlaceToken) and a.city.tile.id == action.city.tile.id:
                    return a.to_dict()

    return None
    #assert assertion, f"Action {action} not in all_actions"


def check_action_in_action_helper(action_dict, game_state):
    action_helper = ActionHelper()
    stored_action = BaseAction.action_from_dict(action_dict, game_state)
    action_helper_actions = action_helper.get_all_choices(game_state)
    updated_action = check_action_in_all_actions(stored_action, action_helper_actions)
    return updated_action

    #if not present:
    #    LOGGER.debug(f"Expected action {action_dict} to be in action_helper choices: {action_helper.get_all_choices(game_state, as_dict=True)}")

# Online game implementations require an extra pass during BuyTrains because they allow cross-company purchases
# Therefore, sometimes, we have an extra skip to process that would normally happen during BuyTrains, but is instead
# happening during BuyCompany.
def should_skip_action(filtered_actions, action, game_state, i):
    if i + 2 < len(filtered_actions) and isinstance(game_state.round.active_step(), BuyTrainStep) and action["type"] == "pass":
        #LOGGER.debug("checking buytrain pass 2")
        current_action_entity = game_state.get(action["entity_type"], action["entity"])
        next_action_type = filtered_actions[i+1]["type"]
        if filtered_actions[i+1]["entity_type"] == "player":
            return False
        next_action_entity = game_state.get(filtered_actions[i+1]["entity_type"], filtered_actions[i+1]["entity"])
        if next_action_entity.is_company():
            next_action_entity = next_action_entity.owner
        next_next_action_entity = game_state.get(filtered_actions[i+2]["entity_type"], filtered_actions[i+2]["entity"])
        next_next_action_type = filtered_actions[i+2]["type"]
        # LOGGER.debug(f"current action entity: {current_action_entity}")
        # LOGGER.debug(f"next action entity: {next_action_entity}")
        # LOGGER.debug(f"next action type: {next_action_type}")
        # LOGGER.debug(f"next next action entity: {next_next_action_entity}")
        # LOGGER.debug(f"next next action type: {next_next_action_type}")
        # Two conditions to skip this pass:
        # 1. IGNORE FOR NOW: Either the next action is a pass by the same entity and the action after that is by a different entity
        # 2. Or the next action is a non-pass by the same entity and the action after that is a pass by the same entity
        # if (
        #     next_action_type == "pass" and
        #     next_action_entity == current_action_entity and
        #     next_next_action_entity != current_action_entity
        # ):
        #     LOGGER.debug(f"Skipping pass action during BuyTrain from condition 1")
        #     LOGGER.debug(f"Current action entity: {current_action_entity}, next action entity: {next_action_entity}, next next action entity: {next_next_action_entity}")
        #     continue
        if (
            next_action_type != "pass" and
            current_action_entity == next_action_entity and
            next_next_action_type == "pass" and
            next_action_entity == next_next_action_entity
        ):
            LOGGER.debug(f"Skipping pass action during BuyTrain from condition 2")
            LOGGER.debug(f"Current action entity: {current_action_entity}, next action entity: {next_action_entity}, next next action entity: {next_next_action_entity}")
            return True
        return False
    
    if isinstance(game_state.round.active_step(), BuyCompanyStep) and action["type"] == "pass":
        #LOGGER.debug("checking buycompany pass")
        # We skip buycompany passes if its already a different player's turn or if the next action is by the same corp
        current_action_entity = game_state.get(action["entity_type"], action["entity"])
        current_active_entity = game_state.current_entity
        if action["entity_type"] == "company":
            current_action_entity = current_action_entity.owner
        LOGGER.debug(f"current action entity: {current_action_entity}, current active entity: {current_active_entity}")
        if current_action_entity != current_active_entity:
            LOGGER.debug(f"Skipping pass action during BuyCompany because it's already a new corp or player's turn")
            LOGGER.debug(f"Current action entity: {current_action_entity}, current active entity: {current_active_entity}")
            return True
        if i + 1 >= len(filtered_actions):
            return False
        next_action_entity = game_state.get(filtered_actions[i+1]["entity_type"], filtered_actions[i+1]["entity"])
        if filtered_actions[i+1]["entity_type"] == "company":
            next_action_entity = next_action_entity.owner
        next_action_type = filtered_actions[i+1]["type"]
        LOGGER.debug(f"next action entity: {next_action_entity}, next action type: {next_action_type}")

        if next_action_entity == current_action_entity or (
            filtered_actions[i+1]["entity"] == "MH" and next_action_entity.player() == current_action_entity.player()
        ):
            LOGGER.debug(f"Skipping pass action during BuyCompany because it's followed by an action by the same corp")
            LOGGER.debug(f"Current action entity: {current_action_entity}, next action entity: {next_action_entity}")
            return True
        return False
    
    if isinstance(game_state.round.active_step(), TrackStep) and action["type"] == "pass":
        #LOGGER.debug("checking laytile pass")
        # We skip laytile passes if it is a different corporation's turn or if the next action is a laytile by the same corp
        current_action_entity = game_state.get(action["entity_type"], action["entity"])
        current_active_entity = game_state.current_entity
        if current_action_entity.is_company():
            current_action_entity = current_action_entity.owner
            
        #LOGGER.debug(f"current action entity: {current_action_entity}, current active entity: {current_active_entity}")
        if current_action_entity != current_active_entity:
            LOGGER.debug(f"Skipping pass action during LayTile because it's already a new corp or player's turn")
            LOGGER.debug(f"Current active entity: {current_active_entity}, current action entity: {current_action_entity}")
            return True
        if i + 1 >= len(filtered_actions):
            return False

        next_action_entity = game_state.get(filtered_actions[i+1]["entity_type"], filtered_actions[i+1]["entity"])
        next_action_type = filtered_actions[i+1]["type"]
        #LOGGER.debug(f"next action entity: {next_action_entity}, next action type: {next_action_type}")
        if next_action_type == "lay_tile" and next_action_entity == current_action_entity:
            LOGGER.debug(f"Skipping pass action during LayTile because it's followed by a laytile by the same corp")
            LOGGER.debug(f"Current action entity: {current_action_entity}, next action entity: {next_action_entity}")
            return True
        return False
    return False

def should_add_pass(action, game_state):
    if not isinstance(game_state.round.active_step(), BuyCompanyStep):
        return False
    
    current_action_entity = game_state.get(action["entity_type"], action["entity"])
    current_action_type = action["type"]
    current_active_entity = game_state.current_entity
    if current_action_entity.is_company():
        current_action_entity = current_action_entity.owner
    
    LOGGER.debug(f"current action entity: {current_action_entity}, current active entity: {current_active_entity}")
    LOGGER.debug(f"current action type: {current_action_type}")
    if (
        not (action["entity"] == "MH" and current_action_entity.player() == current_active_entity.player()) and
        current_action_entity != current_active_entity and
        current_action_type in ["lay_tile", "buy_shares", "sell_shares", "pass", "buy_company", "par", "place_token"]
    ):
        LOGGER.debug(f"Adding pass action during BuyCompany because it's followed by an action by a different corp")
        return True
    return False


def get_game_object_for_game(game: dict) -> BaseGame:
    LOGGER.debug(f"Processing game {game['id']}")
    optional_rules = False
    if game["settings"]["optional_rules"]:
        LOGGER.debug(f"Warning: Game {game['id']} has optional rules {game['settings']['optional_rules']}. If we hit an error, we will skip this game.")
        optional_rules = True
    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"} # Need to use this for consistency in training
    game_state = game_class(players)

    player_mapping = {
        player["id"]: i + 1
        for i, player in enumerate(game["players"])
    }

    filtered_actions = filter_actions(game["actions"])

    for i, action in enumerate(filtered_actions):
        LOGGER.debug(f"Processing action: {action}")
        if action["entity_type"] == "player":
            if action.get("user", None) and action["entity"] != action["user"]:
                LOGGER.debug(f"Master mode or bug!!")
            action["entity"] = player_mapping[action["entity"]]
            action["user"] = action["entity"]
        else:
            entity_owner = game_state.get(action['entity_type'], action['entity']).player()
            if action.get("user", None):
                if action["user"] not in player_mapping:
                    LOGGER.debug(f"Non-playing user played via master mode. Changing to entity owner.")
                elif entity_owner.id != player_mapping[action["user"]]:
                    LOGGER.debug(f"In-game user played via master mode. Changing to entity owner.")
                action["user"] = entity_owner.id
        
        # If a player buys a company from a different player, we don't want to use this game.
        if action["type"] == "buy_company":
            company_purchaser = game_state.get(action["entity_type"], action["entity"]).player()
            company_owner = game_state.company_by_id(action["company"]).player()
            if company_purchaser.id != company_owner.id:
                LOGGER.debug(f"Skipping game because there's a cross-player company purchase")
                LOGGER.debug(f"Company purchaser: {company_purchaser}, company owner: {company_owner}")
                LOGGER.debug(f"Game actions: {game_state.raw_actions}")
                return None

        # If a player buys a train from a different player, we don't want to use this game.
        if action["type"] == "buy_train":
            train_purchaser = game_state.get(action["entity_type"], action["entity"])
            train_owner = game_state.train_by_id(action["train"]).owner
            if train_owner.is_corporation():
                if train_purchaser.player() != train_owner.player():
                    LOGGER.debug(f"Skipping game because there's a cross-player train purchase")
                    LOGGER.debug(f"Train purchaser: {train_purchaser.player()}")
                    LOGGER.debug(f"Train owner: {train_owner.player()}")
                    LOGGER.debug(f"Game actions: {game_state.raw_actions}")
                    return None
        
        # If the MH takes an action out of turn, we don't want to use this game.
        if action["entity_type"] == "company" and action["entity"] == "MH":
            mh_owner = game_state.company_by_id("MH").player()
            current_player = game_state.current_entity.player()
            if mh_owner != current_player:
                LOGGER.debug(f"Skipping game because the MH took an action out of turn")
                LOGGER.debug(f"MH owner: {mh_owner}, current player: {current_player}")
                LOGGER.debug(f"Game actions: {game_state.raw_actions}")
                return None

        if should_add_pass(action, game_state):
            pass_action = {
                "type": "pass",
                "entity": game_state.current_entity.id,
                "entity_type": game_state.current_entity.__class__.__name__.lower(),
                "user": game_state.current_entity.player().id
            }
            LOGGER.debug(f"Adding pass action {pass_action} before action: {action}")
            game_state.process_action(pass_action)

        if should_skip_action(filtered_actions, action, game_state, i):
            LOGGER.debug(f"Skipping action: {action}")
            continue

        # If the player is trying to buy a share from the ipo after buying from the market when the company is in brown (or vice versa), we can't use this game.
        if isinstance(game_state.round.active_step(), BuySellParShares) and action["type"] == "buy_shares":
            shares = [game_state.share_by_id(share) for share in action["shares"]]
            if not game_state.round.active_step().can_buy_shares(game_state.current_entity, shares):
                LOGGER.debug(f"Skipping game because the player is trying to buy an illegal share.")
                LOGGER.debug(f"Game actions: {game_state.raw_actions}")
                return None

        if action["type"] not in ["pass", "bankrupt"]:
            # Don't check passes because we have a lot of extra passes
            # Don't check bankrupt because we don't allow bankruptcies as often
            updated_action = check_action_in_action_helper(action, game_state)
            if updated_action is not None:
                LOGGER.debug(f"Updating action based on action helper actions.")
                LOGGER.debug(f"Original action: {action}")
                LOGGER.debug(f"Updated action: {updated_action}")
                action = updated_action

        try:
            game_state.process_action(action)
        except Exception as e:
            LOGGER.debug(f"Error processing action: {e}")
            if optional_rules:
                LOGGER.info(f"Skipping game {game['id']} because of optional rules")
                LOGGER.info(f"Game actions: {game_state.raw_actions}")
                return None

            LOGGER.error(f"Game id: {game['id']}")
            LOGGER.error(f"Actions processed so far: {game_state.raw_actions}")
            LOGGER.error(f"Action being processed: {action}")
            LOGGER.error(f"All actions: {game['actions']}")
            raise e

    LOGGER.debug(f"Finished processing game {game['id']}")
    LOGGER.debug(f"Game actions: {game_state.raw_actions}")
    return game_state

def fix_online_games(game_data_dir: str, output_dir: str, overwrite: bool = False):
    games = load_games_from_json(game_data_dir)
    games = filter_to_completed_4_player_games(games)
    fixed_games = [x.stem for x in Path(output_dir).glob("*.json")]
    for game in tqdm(games, desc="Fixing games"):
        if str(game["id"]) in fixed_games and not overwrite:
            LOGGER.debug(f"Skipping game {game['id']} because it has already been fixed")
            continue
        fixed_game = get_game_object_for_game(game)
        if fixed_game is None:
            with open(Path(output_dir) / f"{game['id']}.json", "w") as f: 
                json.dump({"id": game["id"], "status": "error", "reason": "unusable due to optional rules", "actions": []}, f, indent=4)
            continue
        with open(Path(output_dir) / f"{game['id']}.json", "w") as f: 
            json.dump(fixed_game.to_dict(), f, indent=4)


def make_action_model_friendly(game_state: BaseGame, action: dict) -> dict:
    if action["type"] not in ["bid", "buy_train", "buy_company"]:
        return action
    
    if action["type"] == "bid":
        LOGGER.debug(f"Checking bid action: {action}")
        original_price = action["price"]
        # Models are only allowed to make the minimum possible bid on a private
        # Therefore, we need to convert player actions such that they are making the minimum possible bid
        # from the model's perspective
        # Concretely, this looks like the following:
        # 1. If the player is bidding on a company with no pre-existing bids, the player's bid amount should be updated to be 
        #    the minimum possible bid for that company.
        # 2. If the player is bidding on a company with at least one pre-existing bid, the pre-existing bid amounts should be
        #    updated such that the player's bid is the minimum possible bid for that company.
        # 3. In both #1 and #2, the player's bid amount must be updated to be a multiple of 5.
        # 4. Additionally, all existing bids on other companies should be updated to be a multiple of 5.
        #
        # Importantly, these updates only happen in the encoded game state and action index - the actual game state is not updated
        # with these new values.
        # With all of this in mind, the changes to the action object are as follows:
        # If the player is bidding on a company with bids, make no change to the action object. Instead, the game state encoding
        # will be updated to handle the bid amount.
        company = game_state.company_by_id(action["company"])
        if len(game_state.round.active_step().bidders[company]) > 0:
            return action
        
        # Otherwise, we need to update the bid amount to be the minimum possible bid for that company that is a multiple of 5.
        min_bid = game_state.round.active_step().min_bid(company)
        # Round min_bid up to the nearest multiple of 5
        if min_bid % 5 != 0:
            LOGGER.info(f"Minimum bid amount {min_bid} for {company.id} is not a multiple of 5. Updating to {min_bid + (5 - (min_bid % 5))}")
            min_bid = min_bid + (5 - (min_bid % 5))

        action["price"] = min_bid
        LOGGER.debug(f"Original price: {original_price}, new price: {min_bid}")

        # The remaining changes will happen in `make_encoded_game_state_model_friendly`
        return action
    
    if action["type"] == "buy_train":
        LOGGER.debug(f"Checking buy train action: {action}")
        original_price = action["price"]
        # Models are only allowed to purchase trains for a subset of cash values
        # Therefore, we need to convert the amount the player spends to the nearest legal value
        # Concretely, this looks like the following:
        action_helper = ActionHelper(game_state)
        all_legal_buy_train_actions = action_helper.get_buy_train_actions(game_state, limited_price_options=True)
        legal_buy_train_actions = [x for x in all_legal_buy_train_actions if isinstance(x, BuyTrain) and x.train.id == action["train"]]
        LOGGER.debug(f"Attempted price {original_price}, legal prices: {[x.price for x in legal_buy_train_actions]}")
        min_diff_amount = 2000
        min_diff_action = None
        for legal_action in legal_buy_train_actions:
            legal_action = legal_action.to_dict()
            if legal_action["price"] == action["price"]:
                LOGGER.debug(f"Buy train action is already legal")
                return action
            diff = abs(legal_action["price"] - action["price"])
            if diff < min_diff_amount:
                min_diff_amount = diff
                min_diff_action = legal_action
        LOGGER.debug(f"Min diff action: {min_diff_action}")
        if min_diff_action is None:
            LOGGER.error(f"No legal buy train action found for {action['train']} with price {action['price']}")
            LOGGER.error(f"All legal buy train actions: {[x.to_dict() for x in all_legal_buy_train_actions]}")
            LOGGER.error(f"Attempted action: {action}")
            LOGGER.error(f"Train details: {game_state.train_by_id(action['train'])}, {game_state.train_by_id(action['train']).owner}")
            LOGGER.error(f"Players: {game_state.corporation_by_id(action['entity']).player(), game_state.train_by_id(action['train']).owner.player()}")
            raise ValueError(f"No legal buy train action found for {action['train']} with price {action['price']}")
        
        LOGGER.info(f"Original price: {original_price}, new price: {min_diff_action['price']}")
        action["price"] = min_diff_action["price"]
        return action

    if action["type"] == "buy_company":
        LOGGER.debug(f"Checking buy company action: {action}")
        # Models are only allowed to purchase companies for 3 prices: the minimum, the maximum, or their full cash stack.
        # Therefore, we need to convert the amount the player spends to the nearest legal value
        company = game_state.company_by_id(action["company"])
        corporation = game_state.corporation_by_id(action["entity"])
        if action["price"] == company.min_price or action["price"] == company.max_price or action["price"] == corporation.cash:
            LOGGER.debug(f"Buy company action {action} is already legal")
            return action
        
        LOGGER.debug(f"Updating buy company action {action} to the nearest legal price")
        min_diff_amount = 2000
        min_diff_price = None
        for price in [company.min_price, company.max_price, corporation.cash]:
            if abs(price - action["price"]) < min_diff_amount:
                min_diff_amount = abs(price - action["price"])
                min_diff_price = price

        LOGGER.info(f"Updating buy company action {action} to the nearest legal price: {min_diff_price}")
        action["price"] = min_diff_price
        return action

    raise ValueError(f"Unknown action type: {action['type']}")

def make_encoded_game_state_model_friendly(encoder: Encoder_1830, encoded_game_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], action: dict, game_state: BaseGame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if action["type"] not in ["bid"]:
        return encoded_game_state
    
    if isinstance(encoder, Encoder_GNN):
        return make_encoded_gnn_game_state_model_friendly(encoder, encoded_game_state, action, game_state)
    if isinstance(encoder, Encoder_SSME):
        return make_encoded_ssme_game_state_model_friendly(encoder, encoded_game_state, action, game_state)
    
    raise ValueError(f"Unknown encoder type: {type(encoder)}")

def make_encoded_gnn_game_state_model_friendly(encoder: Encoder_GNN, encoded_game_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], action: dict, game_state: BaseGame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if action["type"] == "bid":
        # Models are only allowed to make the minimum possible bid on a private
        # Therefore, we need to convert player actions such that they are making the minimum possible bid
        # from the model's perspective
        # Concretely, this looks like the following:
        # 1. If the player is bidding on a company with no pre-existing bids, the player's bid amount should be updated to be 
        #    the minimum possible bid for that company.
        # 2. If the player is bidding on a company with at least one pre-existing bid, the pre-existing bid amounts should be
        #    updated such that the player's bid is the minimum possible bid for that company.
        # 3. In both #1 and #2, the player's bid amount must be updated to be a multiple of 5.
        # 4. Additionally, all existing bids on other companies should be updated to be a multiple of 5.
        #
        # Importantly, these updates only happen in the encoded game state and action index - the actual game state is not updated
        # with these new values.
        # With all of this in mind, the changes to the game state encoding object are as follows:
        # First, all existing company bids must be consecutive multiples of 5.
        encoded_game_data, a, b, c = encoded_game_state
        bids_offset = 335 # Magic number
        min_bid_offset = 359
        encoded_game_data = encoded_game_data.squeeze(0)
        for priv_id, priv_idx in encoder.private_id_to_idx.items():
            company = game_state.company_by_id(priv_id)
            if not company:
                raise ValueError(f"Company {priv_id} not found in game.")
            if company.owner:
                continue
            bids = sorted(game_state.round.active_step().bids.get(company, []), key=lambda x: x.price, reverse=True)
            bids = [bid.copy(game_state) for bid in bids]
            if len(bids) == 0:
                continue

            LOGGER.debug(f"Bids: {bids}")

            highest_bid = bids[0]
            if highest_bid.price % 5 != 0:
                LOGGER.info(f"Highest bid amount {highest_bid.price} for {highest_bid.entity.id} on {company.id} is not a multiple of 5. Updating to {highest_bid.price + (5 - (highest_bid.price % 5))}")
                highest_bid.price = highest_bid.price + (5 - (highest_bid.price % 5))
            
            prev_price = highest_bid.price
            for bid in bids[1:]:
                bid.price = prev_price - 5
                prev_price = bid.price

            for bid in bids:
                bid_amount = bid.price
                bid_amount = float(bid_amount) / encoder.starting_cash
                player_idx = encoder.player_id_to_idx[bid.entity.id]
                LOGGER.debug(f"Encoding bid: Bid amount: {bid.price}, normalized: {bid_amount}, player: {bid.entity.id}, player idx: {player_idx}")
                encoded_game_data[bids_offset + priv_idx * encoder.num_players + player_idx] = bid_amount
        
        # Then, update the minimum bid for the company
        for priv_id, priv_idx in encoder.private_id_to_idx.items():
            company = game_state.company_by_id(priv_id)
            if not company:
                raise ValueError(f"Company {priv_id} not found in game.")
            if company.owner:
                encoded_game_data[min_bid_offset + priv_idx] = -1.0  # Indicate owned
                continue

            min_bid_val = game_state.round.active_step().min_bid(company)
            if min_bid_val % 5 != 0:
                LOGGER.info(f"Minimum bid amount {min_bid_val} for {company.id} is not a multiple of 5. Updating to {min_bid_val + (5 - (min_bid_val % 5))}")
                min_bid_val = min_bid_val + (5 - (min_bid_val % 5))
            encoded_game_data[min_bid_offset + priv_idx] = float(min_bid_val) / encoder.starting_cash

        encoded_game_data = encoded_game_data.unsqueeze(0)
        return encoded_game_data, a, b, c


def make_encoded_ssme_game_state_model_friendly(encoded_game_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], action: dict, game_state: BaseGame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass


def convert_game_to_training_data(game: BaseGame, encoder: Encoder_1830) -> Tuple[list[Any], list[Any]]:
    training_data = []
    validation_data = []
    action_mapper = ActionMapper()
    validation_percentage = 0.05
    if random.random() < validation_percentage:
        save_array = validation_data
        LOGGER.debug(f"Adding to validation data")
    else:
        save_array = training_data
        LOGGER.debug(f"Adding to training data")

    game_map = GameMap()
    game_class = game_map.game_by_title("1830")
    players = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    fresh_game_state = game_class(players)

    winning_score = max(game.result().values())
    actual_value = torch.full((4,), -1.0, dtype=torch.float32)
    player_mapping = {p.id: i for i, p in enumerate(sorted(game.players, key=lambda x: x.id))}
    winners = [player_mapping[pid] for pid, score in game.result().items() if score == winning_score]
    if len(winners) > 1:
        actual_value[winners] = 0.0
    else:
        actual_value[winners] = 1.0

    LOGGER.debug(f"Game result: {game.result()}")
    LOGGER.debug(f"Game value: {actual_value}")
    for action in game.raw_actions:
        LOGGER.debug(f"Processing action: {action}")
        updated_action = make_action_model_friendly(fresh_game_state, action.copy())
        encoded_game_state = encoder.encode(fresh_game_state)
        updated_encoded_game_state = make_encoded_game_state_model_friendly(encoder, encoded_game_state, updated_action, fresh_game_state)
        action_index = action_mapper.get_index_for_action(BaseAction.action_from_dict(updated_action, fresh_game_state), fresh_game_state)
        legal_action_indices = action_mapper.get_legal_action_indices(fresh_game_state)
        epsilon = 0.03
        pi = torch.zeros(action_mapper.action_encoding_size)
        pi[legal_action_indices] += epsilon / len(legal_action_indices)
        pi[action_index] = 1.0 - epsilon
        save_array.append((updated_encoded_game_state, legal_action_indices, pi, actual_value))
        fresh_game_state.process_action(action)

    return training_data, validation_data


def convert_games_to_training_dataset(game_data_dir: str, encoder: Encoder_1830, save_path: Union[str, Path]) -> HumanPlayDataset:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    games = load_games_from_json(game_data_dir)
    processor = TrainingExampleProcessor(encoder)
    for game in tqdm(games, desc="Converting games to training data"):
        if game.get("status") == "error":
            LOGGER.debug(f"Skipping game {game['id']} because it has status error")
            continue
        with open(save_path / "progress.json", "r") as f:
            progress = json.load(f)
        if game["id"] in progress:
            LOGGER.debug(f"Skipping game {game['id']} because it has already been converted")
            continue

        LOGGER.info(f"Converting game {game['id']}")
        game_obj = BaseGame.load(game)
        train, val = convert_game_to_training_data(game_obj, encoder)
        if train:
            dest = save_path / "training"
            processor.write_samples(train, dest)
        if val:
            dest = save_path / "validation"
            processor.write_samples(val, dest)

        progress[game["id"]] = True
        with open(save_path / "progress.json", "w") as f:
            json.dump(progress, f, indent=4)
        LOGGER.info(f"Converted game {game['id']}")


def do_pretraining(model_dir: str, game_data_dir: str, config: TrainingConfig) -> TrainingMetrics:
    # Assume the data has been pre-cleaned using the other methods in this file.
    model = get_model_from_path(model_dir)
    games = load_games_from_json(game_data_dir)
    games = [BaseGame.load(game) for game in games if game.get("status") != "error"]
    training_data, validation_data = [], []
    for game in tqdm(games, desc="Converting games to training data"):
        train, val = convert_game_to_training_data(game, model)
        training_data.extend(train)
        validation_data.extend(val)

    train_dataset = HumanPlayDataset(training_data)
    val_dataset = HumanPlayDataset(validation_data)

    return train_model(model, train_dataset, val_dataset, config)

