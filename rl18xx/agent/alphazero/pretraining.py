from pathlib import Path
import copy
import json
import random
from datetime import datetime
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.checkpointer import (
    get_latest_model,
    save_model,
    session_name_for,
    set_current_best,
)
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.agent.alphazero.model import AlphaZeroModel
from rl18xx.agent.alphazero.train import (
    TrainingMetrics,
    compute_losses,
    move_batch_to_device,
)
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.engine.actions import (
    BaseAction,
    Bid,
    BuyShares,
    BuyTrain,
    DiscardTrain,
    LayTile,
    PlaceToken,
    RunRoutes,
)
from rl18xx.game.engine.game import BaseGame
from rl18xx.game.gamemap import GameMap
from rl18xx.agent.alphazero.dataset import HumanPlayDataset, TrainingExampleProcessor
from rl18xx.game.engine.round import (
    BuyCompany as BuyCompanyStep,
    BuyTrain as BuyTrainStep,
    Track as TrackStep,
    Token as TokenStep,
    Route as RouteStep,
    BuySellParShares,
)
from rl18xx.shared.atomic_io import atomic_write_json
import logging

LOGGER = logging.getLogger(__name__)


class PlayerCountBatchSampler:
    """Yields batches where every example has the same ``num_players``.

    Bucket selection is uniform across observed player counts (2..6), which
    counteracts the natural skew of the human game corpus (mostly 4-player
    games) and keeps rare player counts from being starved of gradient signal.
    Within a bucket, examples are shuffled and then partitioned into
    fixed-size batches. Each yielded batch is a homogeneous-N batch that the
    variable-N transformer model (Task #38) can run through its attention
    key-padding mask without further changes.

    ``num_players_fn`` extracts the player count from a single example. With
    the encoder 8-tuple shape in use today, the encoded game_state vector
    carries ``num_players`` at index 7, so callers typically pass
    ``lambda ex: ex[0][7]``.

    Notes:
        - ``__len__`` reports the number of *full* batches per bucket
          (matching the standard ``BatchSampler(drop_last=True)`` semantics).
          ``__iter__`` follows the same convention so DataLoader progress
          bars stay accurate.
        - ``random.shuffle`` uses Python's module-level RNG; seed it once
          (e.g. ``random.seed(...)``) for deterministic shuffling.
    """

    def __init__(self, dataset, batch_size: int, num_players_fn):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.num_players_fn = num_players_fn
        # Build buckets: {num_players: [example_idx, ...]} by walking the
        # dataset once at construction. We extract the player count via
        # ``num_players_fn`` so the sampler stays decoupled from the precise
        # example layout (which differs between SelfPlayDataset and
        # HumanPlayDataset in subtle ways).
        self.buckets: dict = {}
        for idx in range(len(dataset)):
            n = int(num_players_fn(dataset[idx]))
            self.buckets.setdefault(n, []).append(idx)

    def _batched(self, indices):
        # Shuffle in place each pass so successive epochs see different
        # batch boundaries (important because batches are homogeneous-N —
        # otherwise the same examples always co-train together).
        random.shuffle(indices)
        n = self.batch_size
        # ``drop_last=True`` semantics: skip the trailing partial batch so
        # ``__len__`` matches what we actually yield (DataLoader otherwise
        # gets confused).
        full = (len(indices) // n) * n
        for i in range(0, full, n):
            yield indices[i : i + n]

    def __iter__(self):
        bucket_iters = {
            n: iter(self._batched(list(indices)))
            for n, indices in self.buckets.items()
        }
        while bucket_iters:
            # Uniform pick over remaining buckets so a tiny 6-player bucket
            # gets sampled roughly as often as a large 4-player one (the
            # whole point of the bucketing).
            n = random.choice(list(bucket_iters.keys()))
            try:
                yield next(bucket_iters[n])
            except StopIteration:
                del bucket_iters[n]

    def __len__(self):
        return sum(len(b) // self.batch_size for b in self.buckets.values())


def _infer_num_players_from_state_size(size: int) -> int:
    """Reverse the encoder layout to recover ``num_players`` from a flat state size.

    Mirrors ``model_transformer._infer_num_players_from_state_size`` but
    duplicated here so pretraining stays self-contained (the model module is
    off-limits to this migration). ``compute_section_layout`` lives on
    ``Encoder_1830Graph`` (the Encoder_1830 subclass that actually owns the layout
    machinery — Encoder_1830 itself is a thin base). Falls back to 4 if the
    size can't be matched; practically every well-formed example matches one
    of the 2..6 layouts.
    """
    from rl18xx.agent.alphazero.encoder import Encoder_1830Graph

    for n in range(2, 7):
        _, total = Encoder_1830Graph.compute_section_layout(n)
        if total == size:
            return n
    LOGGER.warning(
        f"State of length {size} doesn't match any layout for 2..6 players; "
        f"defaulting to 4 for bucket sampling."
    )
    return 4


def _make_num_players_fn(dataset):
    """Return a callable that extracts ``num_players`` from a decoded dataset row.

    ``PlayerCountBatchSampler`` invokes this once per example at construction
    time, passing the result of ``dataset[idx]`` (the decoded 6-tuple
    ``(game_state_data, data, legal_action_mask, pi, value, price_targets)``).
    The shape of ``game_state_data`` (a flat 1D state vector whose layout
    depends on the actual ``num_players``) tells us which bucket the example
    belongs to — we reverse the encoder layout via
    ``_infer_num_players_from_state_size``.

    This returns the same function for every Dataset_1830 subclass so the
    sampler stays decoupled from dataset internals; the dataset arg is kept
    in the signature for future fast-paths (e.g. reading num_players directly
    off ``HumanPlayDataset.examples`` without invoking ``__getitem__``).
    """

    def from_row(row):
        game_state_data = row[0]
        size = int(game_state_data.shape[-1])
        return _infer_num_players_from_state_size(size)

    return from_row


def load_games_from_json(database_path: str):
    # Games are in json format
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")

    game_files = list(path.glob("*.json"))
    if not game_files:
        raise FileNotFoundError(f"No game files found in {database_path}")

    games = []
    seen = set()
    for game_file in game_files:
        with open(game_file, "r") as f:
            game = json.load(f)
            game["id"] = game_file.stem
            if game["id"] in seen:
                LOGGER.warning(
                    f"Duplicate game id {game['id']} encountered at {game_file}; skipping subsequent occurrence."
                )
                continue
            seen.add(game["id"])
            games.append(game)

    return games


def filter_to_completed_4_player_games(games: list[dict]):
    """Backwards-compatible filter that keeps only finished 4-player games.

    Kept for callers that still want a strict 4-player slice. New code should
    prefer :func:`filter_to_completed_games`, which accepts any legal 1830
    player count (2..6) so that variable-player pretraining works end-to-end.
    """
    return filter_to_completed_games(games, allowed_player_counts=(4,))


def filter_to_completed_games(
    games: list[dict],
    allowed_player_counts: tuple[int, ...] = (2, 3, 4, 5, 6),
):
    """Filter to finished games whose player count is allowed.

    Variable player count support: the human game dataset is a mix of 2..6
    player counts; pretraining must accept all of them (per
    ``docs/step1_review.md`` "Pretraining supports variable player count
    (2-6)"). Pass ``allowed_player_counts=(4,)`` to recover the old
    4-player-only behaviour.
    """
    filtered_games = []
    for game in games:
        if game["status"] != "finished":
            continue

        if len(game["players"]) not in allowed_player_counts:
            continue

        # NOTE: games containing a manual ``end_game`` action are still
        # acceptable — the cleaning pipeline's :func:`filter_actions` strips
        # those out before the engine sees them. (Both engines reject
        # ``EndGame`` through the blocking-step pipeline, so feeding it
        # through would crash regardless.)
        filtered_games.append(game)
    return filtered_games


def filter_actions(actions: list[dict]) -> list[dict]:
    # Deepcopy so the function is pure — we mutate action dicts below (e.g.
    # stripping `auto_actions` and reassigning `id`), and callers reuse the
    # input list elsewhere.
    actions = copy.deepcopy(actions)
    filtered_actions_history = []
    filtered_actions = []
    # First, remove the actions that should be removed by undo/redo along with all messages
    for action in actions:
        if action["type"] == "message":
            continue
        if action["type"] == "undo":
            filtered_actions_history.append(filtered_actions.copy())
            # ``action_id`` may be 0, meaning "undo everything up to and
            # including action 0" — i.e. drop ALL prior actions. The legacy
            # truthy check incorrectly fell through to the "drop last action"
            # branch for action_id=0. Use ``is not None`` so a literal 0
            # keeps the same semantics as any other action id.
            action_id = action.get("action_id")
            if action_id is not None:
                filtered_actions = [x for x in filtered_actions if x["id"] <= action_id]
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

    # ``end_game`` is a manual game-end trigger that human players sometimes
    # use to declare the game finished without the engine recognising the
    # final-train / bankruptcy / stock-market end condition. Neither the
    # Python nor Rust engines accept an ``EndGame`` action through their
    # blocking step pipeline — both raise. Strip it from the cleaning stream
    # so both engines reach the same terminal state (typically the action
    # right before is enough to wrap the game; if not, both engines simply
    # stop where the human action stream ends, which is what we want).
    filtered_actions = [x for x in filtered_actions if x["type"] != "end_game"]

    # Next, reset the id on all actions
    for i, action in enumerate(filtered_actions):
        action["id"] = i + 1

    # LOGGER.debug(f"actions: {json.dumps(actions, indent=4)}")
    # LOGGER.debug(f"filtered_actions: {json.dumps(filtered_actions, indent=4)}")
    return filtered_actions


def check_action_in_all_actions(action, all_actions) -> Optional[dict]:
    """Verify ``action`` is among the legal ``all_actions`` and decide whether
    to substitute a replacement.

    Returns:
        ``None`` when no substitution is needed — either the action matches a
        legal choice cleanly, or no acceptable replacement was found (the
        caller should keep the original action).
        ``dict`` when the caller should substitute the original action with
        this serialized replacement (currently only triggered for
        ``PlaceToken`` actions, where we fall back to any legal PlaceToken on
        the same tile).
    """
    # Don't check runroutes
    if isinstance(action, RunRoutes):
        return None

    matched = False
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
            action_args["train"] = action_args["train"].split("-")[0]
            expected_args["train"] = expected_args["train"].split("-")[0]

        if action_args != expected_args:
            continue
        matched = True
        break

    if matched:
        return None

    LOGGER.debug(f"action class: {action.__class__}")
    LOGGER.debug(f"action entity: {action.entity}")
    LOGGER.debug(f"action args: {action.args_to_dict()}")
    for a in all_actions:
        LOGGER.debug(f"a class: {a.__class__}")
        LOGGER.debug(f"a entity: {a.entity}")
        LOGGER.debug(f"a args: {a.args_to_dict()}")

    if isinstance(action, PlaceToken):
        # Find if all_actions contains a PlaceToken with the same tile and
        # propose it as a substitute.
        for a in all_actions:
            if isinstance(a, PlaceToken) and a.city.tile.id == action.city.tile.id:
                return a.to_dict()

    if isinstance(action, BuyTrain):
        # Ruby's older engine assigned depot train IDs in a slightly different
        # order than our Python port, so an action like ``buy_train train=2-3``
        # may not literally match any legal BuyTrain in our engine, even
        # though a same-named depot train (e.g. ``2-2``) is buyable at the
        # same price. Substitute the legal BuyTrain with the same train
        # name, price, variant, and exchange train name (treating IDs as
        # interchangeable for trains of the same family).
        action_train_name = action.train.name
        action_train_id = getattr(action.train, "id", None)
        action_exchange_name = action.exchange.name if action.exchange else None

        # Two-pass match: prefer a legal BuyTrain whose train ID is identical
        # to the action's train ID (preserves cross-corp source identity when
        # multiple same-named trains exist on the seller, e.g. B&O holding two
        # 5-trains '5-1' and '5-2' — see game 28908). Fall back to the legacy
        # name-only match when no exact ID match exists.
        def _candidate_matches(a):
            if not isinstance(a, BuyTrain):
                return False
            if a.entity.id != action.entity.id:
                return False
            if a.train.name != action_train_name:
                return False
            if a.price != action.price:
                return False
            if (a.variant or a.train.name) != (action.variant or action_train_name):
                return False
            a_exchange_name = a.exchange.name if a.exchange else None
            if a_exchange_name != action_exchange_name:
                return False
            return True

        # Pass 1: exact train ID match.
        if action_train_id is not None:
            for a in all_actions:
                if _candidate_matches(a) and getattr(a.train, "id", None) == action_train_id:
                    return a.to_dict()
        # Pass 2: any name+price match (legacy behavior).
        for a in all_actions:
            if _candidate_matches(a):
                return a.to_dict()

    return None


def check_action_in_action_helper(action_dict, game_state):
    action_helper = ActionHelper()
    stored_action = BaseAction.action_from_dict(action_dict, game_state)
    action_helper_actions = action_helper.get_all_choices(game_state)
    updated_action = check_action_in_all_actions(stored_action, action_helper_actions)
    return updated_action

    # if not present:
    #    LOGGER.debug(f"Expected action {action_dict} to be in action_helper choices: {action_helper.get_all_choices(game_state, as_dict=True)}")


# Online game implementations require an extra pass during BuyTrains because they allow cross-company purchases
# Therefore, sometimes, we have an extra skip to process that would normally happen during BuyTrains, but is instead
# happening during BuyCompany.
def should_skip_action(filtered_actions, action, game_state, i):
    # General stray corp-typed pass: Ruby's action log occasionally records
    # a ``pass entity=CorpX`` after CorpX has already finished its turn (or
    # the round has transitioned away from OR). Skip whenever the action's
    # entity doesn't match the engine's current entity. This generalizes
    # the per-step checks below (BuyTrain / BuyCompany / TrackStep) to
    # operating-step states they don't cover (e.g., between OR and SR).
    if action["type"] == "pass" and action["entity_type"] == "corporation":
        try:
            current_action_entity = game_state.get("corporation", action["entity"])
            current_active = game_state.current_entity
            if (
                current_action_entity is not None
                and current_active is not None
                and current_action_entity != current_active
            ):
                LOGGER.debug(
                    f"Skipping stray corp pass entity={action['entity']} "
                    f"(current operator is {current_active})"
                )
                return True
        except Exception:
            pass

    if (
        i + 2 < len(filtered_actions)
        and isinstance(game_state.round.active_step(), BuyTrainStep)
        and action["type"] == "pass"
    ):
        # LOGGER.debug("checking buytrain pass 2")
        current_action_entity = game_state.get(action["entity_type"], action["entity"])
        next_action_type = filtered_actions[i + 1]["type"]
        if filtered_actions[i + 1]["entity_type"] == "player":
            return False
        next_action_entity = game_state.get(filtered_actions[i + 1]["entity_type"], filtered_actions[i + 1]["entity"])
        if next_action_entity.is_company():
            next_action_entity = next_action_entity.owner
        next_next_action_entity = game_state.get(
            filtered_actions[i + 2]["entity_type"], filtered_actions[i + 2]["entity"]
        )
        next_next_action_type = filtered_actions[i + 2]["type"]
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
            next_action_type != "pass"
            and current_action_entity == next_action_entity
            and next_next_action_type == "pass"
            and next_action_entity == next_next_action_entity
        ):
            LOGGER.debug(f"Skipping pass action during BuyTrain from condition 2")
            LOGGER.debug(
                f"Current action entity: {current_action_entity}, next action entity: {next_action_entity}, next next action entity: {next_next_action_entity}"
            )
            return True
        return False

    if isinstance(game_state.round.active_step(), BuyCompanyStep) and action["type"] == "pass":
        # LOGGER.debug("checking buycompany pass")
        # We skip buycompany passes if its already a different player's turn or if the next action is by the same corp
        current_action_entity = game_state.get(action["entity_type"], action["entity"])
        current_active_entity = game_state.current_entity
        if action["entity_type"] == "company":
            current_action_entity = current_action_entity.owner
        LOGGER.debug(f"current action entity: {current_action_entity}, current active entity: {current_active_entity}")
        if current_action_entity != current_active_entity:
            LOGGER.debug(f"Skipping pass action during BuyCompany because it's already a new corp or player's turn")
            LOGGER.debug(
                f"Current action entity: {current_action_entity}, current active entity: {current_active_entity}"
            )
            return True
        if i + 1 >= len(filtered_actions):
            return False
        next_action_entity = game_state.get(filtered_actions[i + 1]["entity_type"], filtered_actions[i + 1]["entity"])
        if filtered_actions[i + 1]["entity_type"] == "company":
            next_action_entity = next_action_entity.owner
        next_action_type = filtered_actions[i + 1]["type"]
        LOGGER.debug(f"next action entity: {next_action_entity}, next action type: {next_action_type}")

        # Decide whether the next action is something the CURRENT corp can still
        # do within its own OR turn (i.e., we should skip this redundant pass
        # so BuyCompany stays alive) versus the start of the next OR turn (in
        # which case we MUST process this pass so the engine advances past
        # BuyCompany before the next turn's actions arrive).
        #
        # Within-OR-turn next actions for the current corp:
        #   - another pass by the same corp (redundant first pass / Ruby
        #     buytrain pass our engine auto-skipped — the second pass is what
        #     actually clears BuyCompany). See games 27604, 30795, 58218.
        #   - a pass by an MH company owned by the same player (MH special
        #     ability still being exercised).
        #   - a buy_company action by the same corp (more BC business — see
        #     game 27604 where PRR buys DH after a stray pass).
        #   - an action by a private company owned by the same corp (e.g.
        #     CS / DH / MH using their tile-lay / token specials — see game
        #     30795 where PRR's CS lays a tile after a stray pass).
        #
        # Start-of-next-OR-turn next actions look like:
        #   - lay_tile / place_token / run_routes / dividend / buy_train by the
        #     same corp with entity_type=corporation. This is the corp's NEXT
        #     OR turn after the round wraps back around — see games 62926/58218
        #     where the same corp operates again two-corp rotations later.
        next_entity_type = filtered_actions[i + 1]["entity_type"]
        next_is_within_or_turn = False
        if next_action_type == "pass" and (
            next_action_entity == current_action_entity
            or (
                filtered_actions[i + 1]["entity"] == "MH"
                and next_action_entity.player() == current_action_entity.player()
            )
        ):
            next_is_within_or_turn = True
        elif next_action_type == "buy_company" and next_action_entity == current_action_entity:
            next_is_within_or_turn = True
        elif next_entity_type == "company" and next_action_entity == current_action_entity:
            # Private-company-driven action (special ability of a private owned
            # by the current corp). The OR turn hasn't ended yet.
            next_is_within_or_turn = True

        if next_is_within_or_turn:
            LOGGER.debug(
                f"Skipping pass action during BuyCompany because next action "
                f"(type={next_action_type}, entity_type={next_entity_type}, "
                f"entity={filtered_actions[i + 1].get('entity')}) is still "
                f"within {current_action_entity}'s OR turn"
            )
            return True
        return False

    if isinstance(game_state.round.active_step(), TrackStep) and action["type"] == "pass":
        # LOGGER.debug("checking laytile pass")
        # We skip laytile passes if it is a different corporation's turn or if the next action is a laytile by the same corp
        current_action_entity = game_state.get(action["entity_type"], action["entity"])
        current_active_entity = game_state.current_entity
        if current_action_entity.is_company():
            current_action_entity = current_action_entity.owner

        # LOGGER.debug(f"current action entity: {current_action_entity}, current active entity: {current_active_entity}")
        if current_action_entity != current_active_entity:
            LOGGER.debug(f"Skipping pass action during LayTile because it's already a new corp or player's turn")
            LOGGER.debug(
                f"Current active entity: {current_active_entity}, current action entity: {current_action_entity}"
            )
            return True
        if i + 1 >= len(filtered_actions):
            return False

        next_action_entity = game_state.get(filtered_actions[i + 1]["entity_type"], filtered_actions[i + 1]["entity"])
        next_action_type = filtered_actions[i + 1]["type"]
        # LOGGER.debug(f"next action entity: {next_action_entity}, next action type: {next_action_type}")
        if next_action_type == "lay_tile" and next_action_entity == current_action_entity:
            LOGGER.debug(f"Skipping pass action during LayTile because it's followed by a laytile by the same corp")
            LOGGER.debug(f"Current action entity: {current_action_entity}, next action entity: {next_action_entity}")
            return True
        return False
    return False


def should_add_pass(action, game_state):
    active_step = game_state.round.active_step()
    current_action_entity = game_state.get(action["entity_type"], action["entity"])
    current_action_type = action["type"]
    current_active_entity = game_state.current_entity
    if current_action_entity is not None and current_action_entity.is_company():
        current_action_entity = current_action_entity.owner

    # Token step (PlaceToken) is blocking when the operating corp can place
    # a token, but human Ruby action streams often skip the placement without
    # recording an explicit pass — the next action in the stream is the
    # corp's run_routes/dividend (or, less commonly, an action by a later
    # corp). Insert a pass when the next action is anything other than a
    # PlaceToken by the same corp.
    if isinstance(active_step, TokenStep):
        if current_action_type == "place_token" and current_action_entity == current_active_entity:
            return False
        # Pass entered explicitly — let the engine handle it (no insertion).
        if current_action_type == "pass" and current_action_entity == current_active_entity:
            return False
        # A stray corp pass for a DIFFERENT corp (e.g. previous operator's
        # leftover pass) will be filtered by ``should_skip_action`` later in
        # the pipeline. Don't insert a Pass on Token for the current operator
        # — that would prematurely advance Token, breaking subsequent actions
        # by the current corp (see game 63012 with NYC's stray pass landing
        # while ERIE is the operator).
        if current_action_type == "pass" and current_action_entity != current_active_entity:
            return False
        # ``buy_company`` is handled by the non-blocking BuyCompany step which
        # appears BEFORE the Token step in the OR step order. The engine's
        # ``process_action`` loop reaches BuyCompany first and processes the
        # action without disturbing Token — no pass needed. Without this
        # exemption, we'd inject a Pass and prematurely close the Token step,
        # so a subsequent ``place_token`` by the same corp (after the
        # buy_company) crashes with "Blocking step Run Routes cannot process
        # action Type: PlaceToken". See games 30582, 30593, 41788, 42571, ...
        if current_action_type == "buy_company" and current_action_entity == current_active_entity:
            return False
        # Company-owned special abilities (e.g. CS lay_tile, DH lay_tile)
        # also flow through SpecialTrack/SpecialToken which appear BEFORE
        # the Token step. As long as the company's owner is the current
        # operator, the engine will route the action through the special
        # step without disturbing Token. See games 48362, 63322, 66579,
        # 70729 (CS/DH lay_tile sandwiched between lay_tile and place_token).
        if (
            current_action_type in ("lay_tile", "place_token")
            and action.get("entity_type") == "company"
            and current_action_entity == current_active_entity
        ):
            return False
        # MH company exchange (``buy_shares entity=MH``) flows through the
        # Exchange step, which is active alongside Token. Don't insert a
        # Token-pass — the exchange doesn't disturb Token state. See game
        # 28034.
        if (
            current_action_type == "buy_shares"
            and action.get("entity_type") == "company"
            and action.get("entity") == "MH"
        ):
            return False
        LOGGER.debug(
            f"Adding pass action during Token because next action "
            f"(type={current_action_type}, entity={action['entity']}) is not a "
            f"PlaceToken/Pass by current operator {current_active_entity}"
        )
        return True

    # BuyTrain step is blocking when the corp can buy a train. Some human
    # streams skip the step (no buy_train, no pass) — typically because the
    # corp had no buyable train under Ruby's rules but Python sees a
    # purchase option. Insert a pass so the engine advances.
    if isinstance(active_step, BuyTrainStep):
        if current_action_type == "buy_train" and current_action_entity == current_active_entity:
            return False
        if current_action_type == "discard_train" and current_action_entity == current_active_entity:
            return False
        if current_action_type == "pass" and current_action_entity == current_active_entity:
            return False
        # Sell shares during BuyTrain step = emergency sell, handled by step.
        if current_action_type == "sell_shares":
            return False
        # buy_company sometimes appears during BuyTrain (cross-step purchase).
        if current_action_type == "buy_company":
            return False
        # MH company exchange (``buy_shares entity=MH``) flows through the
        # Exchange step, which is active alongside BuyTrain. Inserting a
        # BuyTrain pass would prematurely advance the operator past BuyTrain
        # and break downstream actions for the same corp (see game 26990).
        if (
            current_action_type == "buy_shares"
            and action.get("entity_type") == "company"
            and action.get("entity") == "MH"
        ):
            return False
        # CS / DH any-time tile-lay company abilities flow through SpecialTrack
        # which is active alongside BuyTrain. Skip the pass insertion when the
        # company action belongs to the same player as the current operator —
        # the engine routes it through the special step. See games 56454, ...
        if (
            current_action_type == "lay_tile"
            and action.get("entity_type") == "company"
            and current_action_entity == current_active_entity
        ):
            return False
        # DH place_token (homeless DH special) likewise flows through
        # SpecialToken which runs during the operator's whole OR turn.
        if (
            current_action_type == "place_token"
            and action.get("entity_type") == "company"
            and current_action_entity == current_active_entity
        ):
            return False
        LOGGER.debug(
            f"Adding pass action during BuyTrain because next action "
            f"(type={current_action_type}, entity={action['entity']}) is not a "
            f"BuyTrain/DiscardTrain/Pass by current operator {current_active_entity}"
        )
        return True

    if not isinstance(active_step, BuyCompanyStep):
        return False

    LOGGER.debug(f"current action entity: {current_action_entity}, current active entity: {current_active_entity}")
    LOGGER.debug(f"current action type: {current_action_type}")
    if (
        not (action["entity"] == "MH" and current_action_entity.player() == current_active_entity.player())
        and current_action_entity != current_active_entity
        and current_action_type
        in ["lay_tile", "buy_shares", "sell_shares", "pass", "buy_company", "par", "place_token"]
    ):
        LOGGER.debug(f"Adding pass action during BuyCompany because it's followed by an action by a different corp")
        return True
    return False


def _share_source(state, share_id: str) -> Optional[str]:
    """Return ``"ipo"`` or ``"market"`` for ``share_id`` at the current state.

    Reads the share's underlying owner string from the Rust engine (via the
    adapter's share lookup) and maps it to the source bucket the factored
    helper uses. Returns ``None`` for player-owned shares (which never
    appear in a legal BuyShares action — that'd be a player-to-player swap).
    """
    try:
        share = state.share_by_id(share_id)
    except Exception:
        return None
    raw = getattr(getattr(share, "_share", None), "owner", "") or ""
    if raw == "market":
        return "market"
    if raw.startswith("ipo:") or raw.startswith("corp:"):
        return "ipo"
    return None


def _normalize_tile_name(raw: Optional[str]) -> Optional[str]:
    """Strip the ``-N`` suffix the human format uses on tile names.

    Human dicts encode tiles as ``"57-0"`` (tile-instance suffix); the
    factored helper emits just ``"57"`` in the canonical ``params.tile``.
    """
    if raw is None:
        return None
    return raw.split("-", 1)[0]


def _normalize_train_name(raw: Optional[str]) -> Optional[str]:
    """Strip the ``-N`` suffix from human train ids like ``"2-0"``."""
    if raw is None:
        return None
    return raw.split("-", 1)[0]


def _action_dict_to_factored_index(
    action_dict: dict,
    state,
    factored_choices: list,
    action_mapper,
) -> Optional[int]:
    """Map a human action dict directly to its canonical flat-policy index
    via the factored helper output, bypassing BaseAction reconstruction.

    Returns the int index, or ``None`` when no matching ``LegalAction``
    is found (the caller logs + skips). Encoding goes through
    ``ActionMapper.index_for_factored`` so the path stays consistent with
    the production self-play encoding.

    ``factored_choices`` is the list of dicts returned by
    ``state.get_factored_choices()``; passing it in lets the caller
    enumerate once per step and share between encode + legal-index
    extraction.
    """
    raw_type = action_dict.get("type")
    if raw_type is None:
        return None
    target_type = "".join(p.title() for p in raw_type.split("_"))

    # CompanyBuyShares appears in the human dict as ``type="buy_shares"``
    # with ``entity_type="company"`` (the acting entity is a private, e.g.
    # MH exchanging for NYC); the factored helper emits ``type="CompanyBuyShares"``.
    # Same idea for Company-level LayTile / PlaceToken.
    if action_dict.get("entity_type") == "company":
        type_overrides = {
            "BuyShares": "CompanyBuyShares",
            "LayTile": "CompanyLayTile",
            "PlaceToken": "CompanyPlaceToken",
        }
        target_type = type_overrides.get(target_type, target_type)

    candidates = [la for la in factored_choices if la.type == target_type]

    if not candidates:
        return None

    for la in candidates:
        if not _factored_choice_matches(la, action_dict, state):
            continue
        try:
            return action_mapper.index_for_factored(la, state)
        except (KeyError, ValueError):
            continue
    return None


def _factored_choice_matches(la, action_dict: dict, state) -> bool:
    """Type-by-type matcher between a ``LegalAction`` and a human action dict.

    Returns True iff the ``LegalAction`` corresponds to the action the
    human played. Used by :func:`_action_dict_to_factored_index`.
    """
    t = la.type
    entity = la.entity or {}
    params = la.params or {}

    if t in ("Pass", "Bankrupt", "RunRoutes"):
        return True

    if t == "Bid":
        return entity.get("private") == action_dict.get("company")

    if t == "Par":
        if entity.get("corp") != action_dict.get("corporation"):
            return False
        sp = action_dict.get("share_price")
        if isinstance(sp, str):
            try:
                sp = int(sp.split(",")[0])
            except (ValueError, IndexError):
                return False
        return params.get("par_price") == sp

    if t == "BuyShares":
        shares = action_dict.get("shares") or []
        if not shares:
            return False
        first_id = shares[0]
        corp_sym = first_id.split("_", 1)[0]
        if entity.get("corp") != corp_sym:
            return False
        source = _share_source(state, first_id)
        return params.get("source") == source

    if t == "CompanyBuyShares":
        # MH → NYC exchange only in 1830.
        if entity.get("private") != "MH":
            return False
        shares = action_dict.get("shares") or []
        if not shares:
            return False
        source = _share_source(state, shares[0])
        return params.get("source") == source

    if t == "SellShares":
        shares = action_dict.get("shares") or []
        if not shares:
            return False
        corp_sym = shares[0].split("_", 1)[0]
        if entity.get("corp") != corp_sym:
            return False
        return int(params.get("count", 0)) == len(shares)

    if t == "PlaceToken":
        # The factored helper's PlaceToken slot identifies the placement
        # site via ``params.hex`` + ``params.city`` (city_idx). The human
        # dict gives us ``city`` (e.g. ``"F20-0"`` or ``"57-1-0"``) plus
        # the operating ``entity`` (the placing corp). PlaceToken legal
        # sets are usually small (a corp places on its operating hex); if
        # the candidate set is unique, accept; otherwise compare by hex
        # id parsed from the city string.
        target_hex = _hex_from_city_id(action_dict.get("city"))
        if target_hex is None:
            # Fall back to "any" when the human dict's city doesn't pin a
            # hex unambiguously — in practice PlaceToken legal sets are
            # length-1 in 1830 OR cases.
            return True
        return params.get("hex") == target_hex

    if t == "LayTile":
        if entity.get("private"):
            # Company tile-lays (DH/CS) — the entity in the human dict
            # is the *corporation* but the cleaning pipeline tags these
            # as company lays; rely on the hex+tile+rotation triple alone.
            pass
        return (
            params.get("hex") == action_dict.get("hex")
            and params.get("tile") == _normalize_tile_name(action_dict.get("tile"))
            and int(params.get("rotation", -1)) == int(action_dict.get("rotation", -2))
        )

    if t == "BuyTrain":
        # Match on the train's source (depot / discard / cross-corp) and
        # the train type. The human dict tags the train as ``"2-0"``,
        # ``"3-1"``, etc. — strip the suffix.
        target_train = _normalize_train_name(action_dict.get("train"))
        if entity.get("train") != target_train:
            return False
        # Discriminate depot / discard / cross-corp:
        # - factored ``entity={"source": "depot", "train": "X"}`` → depot
        # - factored ``entity={"source": "discard", "train": "X"}`` → discard
        # - factored ``entity={"corp": "OWNER", "train": "X"}`` → cross-corp
        return True  # train-type + corp-owner check via entity above is sufficient

    if t == "DiscardTrain":
        return params.get("train") == _normalize_train_name(action_dict.get("train"))

    if t == "Dividend":
        return params.get("kind") == action_dict.get("kind")

    if t == "BuyCompany":
        return entity.get("private") == action_dict.get("company")

    return False


def _factored_price_target(
    action_dict: dict,
    state,
    factored_choices: list,
    action_mapper,
    chosen_index: int,
) -> list:
    """Build the ContinuousPriceHead training target for a price-bearing
    human action via the factored-helper path.

    Returns a list of ``(slot_idx, price, weight, price_min, price_max)``
    tuples — the same shape ``_compute_price_nll_loss`` consumes — or an
    empty list for categorical-only actions. The price head slot is
    resolved from the factored ``(action_type, entity_key)`` tuple, so
    this path doesn't need a Python ``BaseAction``.
    """
    raw_type = action_dict.get("type")
    if raw_type not in ("bid", "buy_company", "buy_train"):
        return []
    observed_price = action_dict.get("price")
    if observed_price is None:
        return []
    # Find the LegalAction whose canonical index matches chosen_index.
    target_la = None
    for la in factored_choices:
        try:
            if int(action_mapper.index_for_factored(la, state)) == int(chosen_index):
                target_la = la
                break
        except (KeyError, ValueError):
            continue
    if target_la is None:
        return []
    pr = target_la.price_range
    if pr is None:
        return []
    price_min, price_max = int(pr[0]), int(pr[1])
    if price_min == price_max:
        return []  # fixed-price slot (depot trains, exchange trains)

    # Resolve the ContinuousPriceHead slot index from (action_type, entity_key).
    entity = target_la.entity or {}
    action_type = target_la.type
    slot = None
    if action_type == "Bid":
        company = entity.get("private")
        if company is not None and company in action_mapper._PRICE_HEAD_COMPANIES:
            slot = action_mapper._PRICE_HEAD_COMPANIES.index(company)
    elif action_type == "BuyCompany":
        company = entity.get("private")
        if company is not None and company in action_mapper._PRICE_HEAD_COMPANIES:
            slot = (
                len(action_mapper._PRICE_HEAD_COMPANIES)
                + len(action_mapper._PRICE_HEAD_CORPORATIONS) * len(action_mapper._PRICE_HEAD_TRAIN_TYPES)
                + action_mapper._PRICE_HEAD_COMPANIES.index(company)
            )
    elif action_type == "BuyTrain":
        corp_sym = entity.get("corp")
        train_type = entity.get("train")
        if (corp_sym in action_mapper._PRICE_HEAD_CORPORATIONS
                and train_type in action_mapper._PRICE_HEAD_TRAIN_TYPES):
            slot = (
                len(action_mapper._PRICE_HEAD_COMPANIES)
                + action_mapper._PRICE_HEAD_CORPORATIONS.index(corp_sym)
                * len(action_mapper._PRICE_HEAD_TRAIN_TYPES)
                + action_mapper._PRICE_HEAD_TRAIN_TYPES.index(train_type)
            )
    if slot is None:
        return []
    return [(int(slot), float(observed_price), 1.0, float(price_min), float(price_max))]


def _hex_from_city_id(city_str: Optional[str]) -> Optional[str]:
    """Extract the hex id from a city descriptor like ``"F20-0"``.

    18xx.games's serialized city ids fall into two shapes:
      - ``"<hex_id>-<city_idx>"`` (preprinted): e.g. ``"F20-0"``.
      - ``"<tile_name>-<tile_instance>-<city_idx>"`` (laid tile):
        e.g. ``"57-1-0"`` — does NOT pin the hex.

    We can only recover the hex id directly from the first form. Returning
    ``None`` lets the caller fall back to accepting any candidate at this
    state (PlaceToken legal sets are typically singleton in 1830 OR
    operating phases).
    """
    if not city_str or "-" not in city_str:
        return None
    parts = city_str.split("-")
    if len(parts) == 2:
        return parts[0]
    # 3+ parts: tile-form, hex isn't directly extractable.
    return None


def _load_cleaned_game_via_rust(game: dict):
    """Replay a *cleaned* game dict through the Rust engine and return the
    resulting ``RustGameAdapter`` (or ``None`` if replay fails).

    ``BaseGame.load(game)`` is the legacy Python-engine constructor that
    triggers the ``FactoredActionHelper`` fallback warning on every legal-
    action lookup later in the pretrain pipeline. Cleaned games have
    already passed cleaning (so we trust the action stream), so we can
    just construct a fresh Rust game and replay the actions directly —
    same engine pretrain self-play uses, no Python-side enumeration.

    Returns ``None`` for malformed game records so the caller can skip
    them gracefully.
    """
    try:
        from engine_rs import BaseGame as RustGame
        from rl18xx.rust_adapter import RustGameAdapter

        players_field = game.get("players") or []
        if not players_field:
            return None
        # Normalize the player dict to ``{int_id: name}`` matching the
        # constructor signature both engines share. Cleaned games store
        # players as ``[{"id": int, "name": str}, ...]``.
        players = {}
        for i, p in enumerate(players_field):
            if isinstance(p, dict):
                pid = int(p.get("id", i + 1))
                name = str(p.get("name", f"Player {pid}"))
            else:
                pid = i + 1
                name = str(p)
            players[pid] = name
        rust_game = RustGame(players)
        adapter = RustGameAdapter(rust_game)
        for action in game.get("actions", []):
            adapter.process_action(action)
        return adapter
    except Exception as e:
        LOGGER.warning(
            "Failed to load cleaned game %s via Rust: %s",
            game.get("id", "<unknown>"), e,
        )
        return None


def get_game_object_for_game(game: dict, use_rust: bool = True) -> BaseGame:
    """Replay a recorded human game through the engine, filtering out
    illegal/redundant actions where possible.

    When ``use_rust=True`` (the default), the underlying engine is the Rust
    ``BaseGame`` wrapped by :class:`RustGameAdapter`. Set ``use_rust=False``
    to fall back to the legacy Python engine (kept only as a parity
    reference — production cleaning has run on Rust since the engine
    migration). The cleaning logic itself is unchanged — both engines
    expose the same Python-shaped API.

    Returns the final game state, or ``None`` if the cleaning logic
    determined the game must be dropped.
    """
    game_or_none, _reason = _get_game_object_for_game_with_reason(game, use_rust=use_rust)
    return game_or_none


def _get_game_object_for_game_with_reason(game: dict, use_rust: bool = True):
    """Internal: same as :func:`get_game_object_for_game` but additionally
    returns the drop reason (or ``None`` if the game cleaned successfully).

    The drop reason is a short stable string suitable for parity audits
    (e.g. ``"cross_player_company_purchase"``,
    ``"cross_player_train_purchase"``, ``"ruby_depot_phase_skip"``,
    ``"mh_out_of_turn"``, ``"company_tile_lay_outside_or"``,
    ``"illegal_share_buy"``, ``"entity_mismatch"``, ``"engine_error"``,
    ``"optional_rules_engine_error"``).
    """
    LOGGER.debug(f"Processing game {game['id']}")
    optional_rules = False
    if game["settings"]["optional_rules"]:
        LOGGER.debug(
            f"Warning: Game {game['id']} has optional rules {game['settings']['optional_rules']}. If we hit an error, we will skip this game."
        )
        optional_rules = True
    num_players = len(game["players"])
    players = {
        i + 1: f"Player {i + 1}" for i in range(num_players)
    }  # Need to use this for consistency in training
    if use_rust:
        from engine_rs import BaseGame as RustGame
        from rl18xx.rust_adapter import RustGameAdapter
        rust_game = RustGame(players)
        game_state = RustGameAdapter(rust_game)
    else:
        game_map = GameMap()
        game_class = game_map.game_by_title("1830")
        game_state = game_class(players)

    player_mapping = {p["id"]: i + 1 for i, p in enumerate(game["players"])}

    filtered_actions = filter_actions(game["actions"])

    for i, action in enumerate(filtered_actions):
        LOGGER.debug(f"Processing action: {action}")
        if action["entity_type"] == "player":
            if action.get("user", None) and action["entity"] != action["user"]:
                LOGGER.debug(f"Master mode or bug!!")
            action["entity"] = player_mapping[action["entity"]]
            action["user"] = action["entity"]
        else:
            entity_owner = game_state.get(action["entity_type"], action["entity"]).player()
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
                return None, "cross_player_company_purchase"

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
                    return None, "cross_player_train_purchase"

            # Detect "phase-skip" depot buys: in some 2021-era online games
            # (e.g. 58217, 78203, 64349), the user bought a higher-grade
            # train (e.g. a 3-train) from the depot while a lower-grade
            # train of the previous tier was still at the front of the
            # depot queue. The older Ruby 18xx engine only validated the
            # train variant, not depot containment, so the action was
            # accepted at the time; current Ruby and our Python port both
            # reject it. Drop such games rather than try to fudge the
            # depot state.
            if not action.get("exchange"):
                bought = game_state.train_by_id(action["train"])
                # Only check depot buys (where the train is currently owned
                # by the depot, not by another corp).
                if bought is not None and not bought.owner.is_corporation():
                    depot_trains = game_state.depot.depot_trains()
                    # The legitimate depot-buyable names at this moment.
                    buyable_names = {t.name for t in depot_trains}
                    if bought.name not in buyable_names and bought in game_state.depot.upcoming:
                        LOGGER.debug(
                            f"Skipping game because of a Ruby-quirky depot phase-skip: "
                            f"action bought train {bought.id} (name={bought.name}) "
                            f"while depot front exposes only {sorted(buyable_names)}"
                        )
                        return None, "ruby_depot_phase_skip"

        # If the MH takes an action out of turn, we don't want to use this game.
        if action["entity_type"] == "company" and action["entity"] == "MH":
            mh_owner = game_state.company_by_id("MH").player()
            current_player = game_state.current_entity.player()
            if mh_owner != current_player:
                LOGGER.debug(f"Skipping game because the MH took an action out of turn")
                LOGGER.debug(f"MH owner: {mh_owner}, current player: {current_player}")
                LOGGER.debug(f"Game actions: {game_state.raw_actions}")
                return None, "mh_out_of_turn"

        # CS / DH any-time tile-lay abilities are only legal during an
        # Operating Round (CS: any step of the owning corp's OR turn via the
        # SpecialTrack step; DH: only in lieu of the owning corp's LayTile).
        # A handful of human games (e.g. 26846 with CS, 27514 with DH) attempt
        # to use these powers during a Stock Round — Ruby's online engine
        # accepted it via a looser interpretation, but our engine (and the
        # printed rules) reject it. Drop these games so they don't break
        # cleaning. Note: legal uses during the corp's OR turn fall through
        # to SpecialTrack and are handled by the engine without any filter.
        if (
            action["entity_type"] == "company"
            and action["entity"] in ("CS", "DH")
            and action["type"] == "lay_tile"
            and not game_state.round.operating
        ):
            LOGGER.debug(
                f"Skipping game because {action['entity']} tile-lay attempted "
                f"outside an Operating Round (round={type(game_state.round).__name__})"
            )
            return None, "company_tile_lay_outside_or"

        if should_add_pass(action, game_state):
            pass_action = {
                "type": "pass",
                "entity": game_state.current_entity.id,
                "entity_type": game_state.current_entity.__class__.__name__.lower(),
                "user": game_state.current_entity.player().id,
            }
            LOGGER.debug(f"Adding pass action {pass_action} before action: {action}")
            game_state.process_action(pass_action)

        # Some human Ruby game streams skip ``run_routes`` entirely for a
        # corp that has a train but no profitable route — Ruby's online
        # engine allowed it, our engine (and current Ruby) requires an
        # explicit empty run_routes to advance past the blocking Route step.
        # Synthesize one when the action belongs to a step that comes
        # AFTER Route in the OR sequence (BuyTrain, Dividend, DiscardTrain)
        # so the only way to reach it is to clear Route first. Actions
        # belonging to earlier steps (buy_company, lay_tile, etc.) flow
        # through their own non-blocking step without disturbing Route.
        # See game 86319 where ERIE skips straight to buy_train.
        active_step = game_state.round.active_step()
        if (
            isinstance(active_step, RouteStep)
            and action["type"] in ("buy_train", "dividend", "discard_train")
            and action.get("entity_type") == "corporation"
            and game_state.current_entity.id == action.get("entity")
        ):
            run_routes_action = {
                "type": "run_routes",
                "entity": game_state.current_entity.id,
                "entity_type": "corporation",
                "user": game_state.current_entity.player().id,
                "routes": [],
            }
            LOGGER.debug(
                f"Inserting empty run_routes for {game_state.current_entity.id} "
                f"before action {action} (Route step blocking but stream skips it)"
            )
            game_state.process_action(run_routes_action)

        if should_skip_action(filtered_actions, action, game_state, i):
            LOGGER.debug(f"Skipping action: {action}")
            continue

        # If the player is trying to buy a share from the ipo after buying from the market when the company is in brown (or vice versa), we can't use this game.
        if isinstance(game_state.round.active_step(), BuySellParShares) and action["type"] == "buy_shares":
            # Company-driven share buys (e.g. MH exchanging for an NYC IPO
            # share) flow through ExchangeStep, not through
            # ``BuySellParShares.can_buy_shares``. Skip the per-player
            # legality check for those — the filter is for normal player
            # buys only. (Ruby's online engine also routes MH exchange
            # through a separate ability path; Python's port mirrors that.)
            if action["entity_type"] != "company":
                shares = [game_state.share_by_id(share) for share in action["shares"]]
                if not game_state.round.active_step().can_buy_shares(game_state.current_entity, shares):
                    LOGGER.debug(f"Skipping game because the player is trying to buy an illegal share.")
                    LOGGER.debug(f"Game actions: {game_state.raw_actions}")
                    return None, "illegal_share_buy"

        if action["type"] not in ["pass", "bankrupt"]:
            # Don't check passes because we have a lot of extra passes
            # Don't check bankrupt because we don't allow bankruptcies as often
            # `check_action_in_action_helper` returns None when no substitution
            # is needed (either matched cleanly or no acceptable replacement
            # was found) and a serialized action dict when the caller should
            # swap in the replacement.
            replacement_action = check_action_in_action_helper(action, game_state)
            if replacement_action is not None:
                LOGGER.debug(f"Substituting action based on action helper actions.")
                LOGGER.debug(f"Original action: {action}")
                LOGGER.debug(f"Replacement action: {replacement_action}")
                action = replacement_action

        # Mis-attributed corp action (e.g., game 54156's master-mode dividend
        # re-do). After should_add_pass / should_skip_action / action-helper
        # substitution have applied their corrections, if action.entity still
        # doesn't match the current operator, this is a genuine Ruby
        # mis-attribution that the engines (both strict now) would reject.
        # Drop the game.
        if (
            action["entity_type"] == "corporation"
            and action["type"] in ("lay_tile", "place_token", "run_routes", "dividend", "buy_train", "buy_company", "pass")
            and action["entity"] != game_state.current_entity.id
        ):
            LOGGER.debug(f"Skipping game because action entity does not match current operator")
            LOGGER.debug(f"Action: {action}")
            LOGGER.debug(f"Current operator: {game_state.current_entity.id}")
            return None, "entity_mismatch"

        try:
            game_state.process_action(action)
        except Exception as e:
            LOGGER.debug(f"Error processing action: {e}")
            if optional_rules:
                LOGGER.info(f"Skipping game {game['id']} because of optional rules")
                LOGGER.info(f"Game actions: {game_state.raw_actions}")
                return None, "optional_rules_engine_error"

            LOGGER.error(f"Game id: {game['id']}")
            LOGGER.error(f"Actions processed so far: {game_state.raw_actions}")
            LOGGER.error(f"Action being processed: {action}")
            LOGGER.error(f"All actions: {game['actions']}")
            raise e

    LOGGER.debug(f"Finished processing game {game['id']}")
    LOGGER.debug(f"Game actions: {game_state.raw_actions}")
    return game_state, None


def fix_online_games(game_data_dir: str, output_dir: str, overwrite: bool = False):
    games = load_games_from_json(game_data_dir)
    # Variable player count (2..6): accept the full 1830 player range so
    # pretraining can train on the entire human dataset, not just 4-player
    # games. Use ``filter_to_completed_4_player_games`` if you specifically
    # want to restrict to 4-player.
    games = filter_to_completed_games(games)
    fixed_games = [x.stem for x in Path(output_dir).glob("*.json")]
    for game in tqdm(games, desc="Fixing games"):
        if str(game["id"]) in fixed_games and not overwrite:
            LOGGER.debug(f"Skipping game {game['id']} because it has already been fixed")
            continue
        fixed_game = get_game_object_for_game(game)
        if fixed_game is None:
            with open(Path(output_dir) / f"{game['id']}.json", "w") as f:
                json.dump(
                    {"id": game["id"], "status": "error", "reason": "unusable due to optional rules", "actions": []},
                    f,
                    indent=4,
                )
            continue
        with open(Path(output_dir) / f"{game['id']}.json", "w") as f:
            json.dump(fixed_game.to_dict(), f, indent=4)


# NOTE: ``make_action_model_friendly`` and ``make_encoded_game_state_model_friendly``
# were removed as part of the FactoredActionHelper / ContinuousPriceHead migration
# (see docs/step1_review.md "Continuous-price action space via progressive
# widening"). They existed to fudge human prices into the network's old discrete
# price-slot enumeration and to rewrite the encoded bid ladder so the model only
# ever saw "minimum possible bid + multiples of 5". With prices now handled
# continuously by the ContinuousPriceHead (NLL-trained against the human's raw
# observed price), neither transformation is needed.


def convert_game_to_training_data(
    game: BaseGame,
    encoder: Encoder_1830,
    config: Optional[TrainingConfig] = None,
) -> Tuple[list[Any], list[Any]]:
    if config is None:
        config = TrainingConfig()
    training_data = []
    validation_data = []
    action_mapper = ActionMapper()
    validation_percentage = config.pretrain_validation_percentage
    if random.random() < validation_percentage:
        save_array = validation_data
        LOGGER.debug(f"Adding to validation data")
    else:
        save_array = training_data
        LOGGER.debug(f"Adding to training data")

    # Replay state runs on the Rust engine so every per-action
    # ``action_mapper.get_legal_action_indices`` call hits the native
    # ``state.get_factored_choices`` path (no Python ``FactoredActionHelper``
    # fallback). ``ActionMapper.canonical_index_for_action`` (the encode
    # path) only needs ``action.bundle.owner.name`` to discriminate IPO /
    # Market for ``BuyShares``; the Rust adapter's ``_ShareProxy.owner``
    # now returns the appropriate ``_BankProxy`` / ``_MarketProxy`` based
    # on the share's underlying Rust owner string, so this works without
    # falling back to the Python engine.
    from engine_rs import BaseGame as _RustGame
    from rl18xx.rust_adapter import RustGameAdapter as _RustGameAdapter
    num_players = len(game.players)
    players = {i + 1: f"Player {i + 1}" for i in range(num_players)}
    fresh_game_state = _RustGameAdapter(_RustGame(players))

    # KataGo-style dual value head training target.
    #
    # We store a single per-example value tensor (the existing
    # ``actual_value``) that lives in **normalized net-worth fractions**
    # space — the same encoding self-play uses when
    # ``SelfPlayConfig.use_score_values=True``. ``_derive_dual_value_targets``
    # in train.py then splits this into:
    #   - win_loss_target: share-of-winners (argmax over fractions, equal
    #     mass on ties). KL-trained.
    #   - score_target:    the stored fractions themselves. MSE-trained.
    # so the score head gets the dense continuous signal while the win-loss
    # head still receives the binary outcome target it's optimized for.
    result = game.result()
    player_mapping = {p.id: i for i, p in enumerate(sorted(game.players, key=lambda x: x.id))}
    # Variable player count: size the per-example value tensor to the actual
    # number of players in this game (2..6). Downstream consumers
    # (``_derive_dual_value_targets``) treat the tensor length as the player
    # count so this flows through without further changes.
    scores = torch.zeros(num_players, dtype=torch.float32)
    for pid, nw in result.items():
        scores[player_mapping[pid]] = float(nw)
    total = scores.sum().item()
    if total > 0:
        actual_value = scores / total
    else:
        # Degenerate case (all zero net worth) — fall back to uniform so
        # downstream losses stay finite.
        actual_value = torch.full((num_players,), 1.0 / num_players, dtype=torch.float32)

    LOGGER.debug(f"Game result: {result}")
    LOGGER.debug(f"Game value (normalized net-worth fractions): {actual_value}")
    skipped_actions = 0
    _skipped_examples_by_type: dict = {}
    for action in game.raw_actions:
        LOGGER.debug(f"Processing action: {action}")
        # Encode the game state exactly as the engine sees it — no
        # bid-ladder rewriting. With the ContinuousPriceHead, the model
        # learns from raw observed prices and the encoder no longer needs to
        # massage bids into a "minimum + multiples of 5" canonical form.
        encoded_game_state = encoder.encode(fresh_game_state)

        # Enumerate legal actions via the Rust factored helper (native, fast)
        # and use the same list for both:
        #   1. the categorical pi target (one-hot on the matching slot)
        #   2. legal_action_indices (for the masked-policy loss)
        # This replaces the legacy ``BaseAction.action_from_dict`` + Python-
        # state-dependent ``canonical_index_for_action`` path that bottlenecked
        # pretraining on the Python ``FactoredActionHelper`` fallback.
        factored = fresh_game_state.get_factored_choices()
        action_index = _action_dict_to_factored_index(action, fresh_game_state, factored, action_mapper)
        if action_index is None:
            skipped_actions += 1
            _t = action.get("type", "?")
            _skipped_examples_by_type[_t] = _skipped_examples_by_type.get(_t, 0) + 1
            LOGGER.debug(
                "Skipping action that doesn't match any factored LegalAction: %s",
                action,
            )
            fresh_game_state.process_action(action)
            continue

        # Build legal_action_indices from the factored output (Rust-native).
        legal_set = set()
        for la in factored:
            try:
                legal_set.add(int(action_mapper.index_for_factored(la, fresh_game_state)))
            except (KeyError, ValueError):
                continue
        legal_action_indices = sorted(legal_set)

        epsilon = config.pretrain_label_smoothing
        pi = torch.zeros(action_mapper.action_encoding_size)
        if legal_action_indices:
            pi[legal_action_indices] += epsilon / len(legal_action_indices)
        pi[action_index] = 1.0 - epsilon

        # Price target for the ContinuousPriceHead's NLL loss. The factored
        # ``LegalAction`` for a price-bearing slot carries ``price_range``;
        # we derive the head slot directly from ``(action_type, entity)``
        # instead of round-tripping through a Python ``BaseAction``.
        price_targets = _factored_price_target(
            action, fresh_game_state, factored, action_mapper, action_index
        )

        save_array.append((encoded_game_state, legal_action_indices, pi, actual_value, price_targets))
        fresh_game_state.process_action(action)

    if skipped_actions:
        LOGGER.warning(
            "convert_game_to_training_data: skipped %d/%d actions that could "
            "not be matched against the factored legal set; by type=%s",
            skipped_actions, len(game.raw_actions),
            _skipped_examples_by_type,
        )

    return training_data, validation_data


def convert_games_to_training_dataset(
    game_data_dir: str,
    encoder: Encoder_1830,
    save_path: Union[str, Path],
    config: Optional[TrainingConfig] = None,
):
    if config is None:
        config = TrainingConfig()
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    progress_file = save_path / "progress.json"

    # Ensure progress file exists on first run so downstream readers (and
    # this function's own resume logic) never hit FileNotFoundError. Also
    # tolerate an empty / corrupt file by falling back to an empty tracker.
    if not progress_file.exists():
        atomic_write_json(progress_file, {})
        progress = {}
    else:
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            LOGGER.warning(
                f"Progress file {progress_file} is empty or corrupt ({e}); resetting to empty tracker."
            )
            progress = {}
            atomic_write_json(progress_file, progress)

    games = load_games_from_json(game_data_dir)
    games = [g for g in games if g.get("status") != "error" and "title" in g]
    LOGGER.info(f"Found {len(games)} valid games to convert")

    processor = TrainingExampleProcessor(encoder)
    converted = 0
    skipped = 0
    errors = 0
    for game in tqdm(games, desc="Converting games to training data"):
        if game["id"] in progress:
            skipped += 1
            continue

        try:
            game_obj = _load_cleaned_game_via_rust(game)
            if game_obj is None:
                skipped += 1
                progress[game["id"]] = True
                atomic_write_json(progress_file, progress)
                continue
            train, val = convert_game_to_training_data(game_obj, encoder, config=config)
            if train:
                dest = save_path / "training"
                processor.write_samples(train, dest)
            if val:
                dest = save_path / "validation"
                processor.write_samples(val, dest)
            converted += 1
        except Exception as e:
            LOGGER.warning(f"Error converting game {game['id']}: {e}")
            errors += 1

        progress[game["id"]] = True
        atomic_write_json(progress_file, progress)

    LOGGER.info(f"Conversion complete: {converted} converted, {skipped} already done, {errors} errors")


def pretrain_model(
    model: AlphaZeroModel,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    config: TrainingConfig,
    model_dir: str,
) -> TrainingMetrics:
    """Supervised pretraining on human game data.

    Key differences from ``train_model``:
    - Cross-entropy-style loss on the policy target (the existing loss
      computation already implements masked KL with a near-one-hot label-smoothed
      pi, which is equivalent to CE with label smoothing on the legal action set).
      ``config.pretrain_label_smoothing`` controls smoothing at example-construction
      time (see ``convert_game_to_training_data``).
    - Runs a validation pass each epoch (``model.eval()``, ``torch.no_grad()``).
    - Cosine LR schedule (not constant after warmup).
    - Does NOT call ``load_optimizer_state`` — pretraining always starts fresh.
    - Saves the best-val-loss checkpoint to ``model_dir`` via ``save_model``.
    - Writes per-batch/per-epoch losses and policy accuracy to TensorBoard
      under ``runs/pretrain_<timestamp>/``.

    The full feature (autoregressive policy targets, dual value targets,
    variable player count) is being implemented separately — this wrapper
    just exists so the SL pipeline persists checkpoints, validates each
    epoch, and uses SL-appropriate optimization hyperparameters.
    """
    metrics = TrainingMetrics()
    if len(train_dataset) == 0:
        LOGGER.warning("Training dataset is empty. Skipping pretraining.")
        return metrics

    device = model.device

    # --- Optimizer + cosine LR (fresh; no load_optimizer_state) ---
    # Dual value head: both ``win_loss_head`` and ``score_head`` get the
    # elevated LR (same rationale as the legacy single ``value_head``).
    value_head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "win_loss_head" in name or "score_head" in name:
            value_head_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.Adam(
        [
            {"params": other_params, "lr": config.lr},
            {"params": value_head_params, "lr": config.lr * config.value_lr_multiplier},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Variable-N pretraining bucket sampler: each yielded batch has uniform
    # ``num_players``, and bucket selection is uniform across observed player
    # counts (so the rare 2- and 6-player buckets get sampled roughly as
    # often as the common 4-player one). Without this the dataset's natural
    # skew toward 4-player games would starve the other counts of gradient
    # signal even though the variable-N model (Task #38) supports them.
    num_players_fn = _make_num_players_fn(train_dataset)
    train_sampler = PlayerCountBatchSampler(
        train_dataset, batch_size=config.batch_size, num_players_fn=num_players_fn
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_num_players_fn = _make_num_players_fn(val_dataset)
        val_sampler = PlayerCountBatchSampler(
            val_dataset, batch_size=config.batch_size, num_players_fn=val_num_players_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
        )
    else:
        LOGGER.warning("Validation dataset is empty or None; pretraining will skip val pass.")

    total_steps = max(1, len(train_loader) * config.num_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Pretraining TensorBoard logs land under the same root TensorBoard
    # is configured to watch in ``startup.sh`` (``runs/alphazero_runs``).
    # That makes pretrain curves visible in the same TB instance the
    # self-play loop populates.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = Path("runs") / "alphazero_runs" / f"pretrain_{timestamp}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(str(tb_dir))
    LOGGER.info(f"Pretraining TensorBoard logs: {tb_dir}")

    metrics.training_examples = len(train_dataset)
    best_val_loss = float("inf")
    best_checkpoint_num: Optional[int] = None
    global_step = 0

    for epoch in range(config.num_epochs):
        # ---------------------------- TRAIN PASS ----------------------------
        model.train()
        train_losses = []
        train_policy_losses = []
        train_value_losses = []
        train_score_losses = []
        train_aux_losses = []
        train_entropies = []
        epoch_top1_correct = 0
        epoch_top5_correct = 0
        epoch_total_samples = 0

        optimizer.zero_grad()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Pretrain]", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            global_step += 1
            game_state_data, batch_data, legal_action_mask, pi, value, price_targets = move_batch_to_device(batch, device)

            outputs = compute_losses(
                model, game_state_data, batch_data, legal_action_mask, pi, value, config,
                price_targets=price_targets,
            )
            total_loss = outputs["total_loss"]
            if not torch.isfinite(total_loss):
                LOGGER.warning(
                    f"Non-finite loss at batch {batch_idx}: total={total_loss.item():.4f}; skipping."
                )
                optimizer.zero_grad()
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_losses.append(total_loss.item())
            train_policy_losses.append(outputs["policy_loss"].item())
            train_value_losses.append(outputs["value_loss"].item())
            train_score_losses.append(outputs["score_loss"].item())
            train_aux_losses.append(outputs["aux_loss"].item())
            train_entropies.append(outputs["entropy"].item())

            metrics.batch_losses.append(total_loss.item())
            metrics.batch_policy_losses.append(outputs["policy_loss"].item())
            metrics.batch_value_losses.append(outputs["value_loss"].item())
            metrics.batch_numbers.append(global_step)

            summary_writer.add_scalar("train/loss_total", total_loss.item(), global_step)
            summary_writer.add_scalar("train/loss_policy", outputs["policy_loss"].item(), global_step)
            summary_writer.add_scalar("train/loss_value", outputs["value_loss"].item(), global_step)
            summary_writer.add_scalar("train/loss_score", outputs["score_loss"].item(), global_step)
            summary_writer.add_scalar("train/loss_aux", outputs["aux_loss"].item(), global_step)
            summary_writer.add_scalar("train/entropy", outputs["entropy"].item(), global_step)
            summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            with torch.no_grad():
                policy_probs = outputs["policy_probs"]
                target_top1 = pi.argmax(dim=1)
                pred_top1 = policy_probs.argmax(dim=1)
                epoch_top1_correct += (target_top1 == pred_top1).sum().item()
                pred_top5 = policy_probs.topk(min(5, policy_probs.size(1)), dim=1).indices
                target_expanded = target_top1.unsqueeze(1).expand_as(pred_top5)
                epoch_top5_correct += (pred_top5 == target_expanded).any(dim=1).sum().item()
                epoch_total_samples += pi.size(0)

            train_pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_train_policy = float(np.mean(train_policy_losses)) if train_policy_losses else 0.0
        avg_train_value = float(np.mean(train_value_losses)) if train_value_losses else 0.0
        avg_train_score = float(np.mean(train_score_losses)) if train_score_losses else 0.0
        avg_train_aux = float(np.mean(train_aux_losses)) if train_aux_losses else 0.0
        avg_train_entropy = float(np.mean(train_entropies)) if train_entropies else 0.0
        top1_acc = epoch_top1_correct / max(epoch_total_samples, 1)
        top5_acc = epoch_top5_correct / max(epoch_total_samples, 1)

        metrics.epoch_losses.append(avg_train_loss)
        metrics.epoch_policy_losses.append(avg_train_policy)
        metrics.epoch_value_losses.append(avg_train_value)
        metrics.epoch_top1_accuracy.append(top1_acc)
        metrics.epoch_top5_accuracy.append(top5_acc)
        metrics.epoch_lr.append(optimizer.param_groups[0]["lr"])

        # In-flight sidecar so the Flask dashboard can show per-epoch
        # progress while pretrain is still running. Overwrites at each
        # epoch boundary. The final summary block at the end of training
        # rewrites the same file with the complete record.
        try:
            with open(tb_dir / "pretrain_summary.json", "w") as _f:
                json.dump(
                    {
                        "kind": "pretrain",
                        "timestamp": timestamp,
                        "in_progress": True,
                        "epochs_planned": config.num_epochs,
                        "epochs_trained": epoch + 1,
                        "training_examples": metrics.training_examples,
                        "epoch_losses": list(metrics.epoch_losses),
                        "epoch_policy_losses": list(metrics.epoch_policy_losses),
                        "epoch_value_losses": list(metrics.epoch_value_losses),
                        "epoch_top1_accuracy": list(metrics.epoch_top1_accuracy),
                        "epoch_top5_accuracy": list(metrics.epoch_top5_accuracy),
                        "epoch_lr": list(metrics.epoch_lr),
                    },
                    _f,
                    indent=2,
                )
        except Exception as _e:
            LOGGER.debug(f"Could not write in-flight pretrain_summary.json: {_e}")

        summary_writer.add_scalar("train_epoch/loss_total", avg_train_loss, epoch)
        summary_writer.add_scalar("train_epoch/loss_policy", avg_train_policy, epoch)
        summary_writer.add_scalar("train_epoch/loss_value", avg_train_value, epoch)
        summary_writer.add_scalar("train_epoch/loss_score", avg_train_score, epoch)
        summary_writer.add_scalar("train_epoch/loss_aux", avg_train_aux, epoch)
        summary_writer.add_scalar("train_epoch/entropy", avg_train_entropy, epoch)
        summary_writer.add_scalar("train_epoch/top1_acc", top1_acc, epoch)
        summary_writer.add_scalar("train_epoch/top5_acc", top5_acc, epoch)

        # ------------------------- VALIDATION PASS --------------------------
        val_loss = float("inf")
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_policy_losses = []
            val_value_losses = []
            val_aux_losses = []
            val_entropies = []
            val_top1_correct = 0
            val_top5_correct = 0
            val_total_samples = 0
            val_value_se_sum = torch.zeros(model.config.value_size, dtype=torch.float64)
            val_score_se_sum = torch.zeros(model.config.value_size, dtype=torch.float64)
            val_value_count = 0
            val_score_loss_sum = 0.0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]", leave=False)
                for batch in val_pbar:
                    game_state_data, batch_data, legal_action_mask, pi, value, price_targets = move_batch_to_device(batch, device)
                    outputs = compute_losses(
                        model, game_state_data, batch_data, legal_action_mask, pi, value, config,
                        price_targets=price_targets,
                    )
                    val_losses.append(outputs["total_loss"].item())
                    val_policy_losses.append(outputs["policy_loss"].item())
                    val_value_losses.append(outputs["value_loss"].item())
                    val_aux_losses.append(outputs["aux_loss"].item())
                    val_entropies.append(outputs["entropy"].item())
                    val_score_loss_sum += outputs["score_loss"].item()

                    policy_probs = outputs["policy_probs"]
                    target_top1 = pi.argmax(dim=1)
                    pred_top1 = policy_probs.argmax(dim=1)
                    val_top1_correct += (target_top1 == pred_top1).sum().item()
                    pred_top5 = policy_probs.topk(min(5, policy_probs.size(1)), dim=1).indices
                    target_expanded = target_top1.unsqueeze(1).expand_as(pred_top5)
                    val_top5_correct += (pred_top5 == target_expanded).any(dim=1).sum().item()
                    val_total_samples += pi.size(0)

                    # Win-loss head: per-player MSE between softmax of head output
                    # and the derived share-of-winners target.
                    win_loss_probs = F.softmax(outputs["win_loss_logits"], dim=1)
                    win_loss_target = outputs["win_loss_target"]
                    se = ((win_loss_probs - win_loss_target) ** 2).sum(dim=0).double().cpu()
                    val_value_se_sum += se

                    # Score head: per-player MSE of raw prediction vs. normalized
                    # net-worth target. This is the head's actual training loss
                    # decomposed per-player, exposed as a TB scalar below.
                    score_pred = outputs["score_pred"]
                    score_target = outputs["score_target"]
                    score_se = ((score_pred - score_target) ** 2).sum(dim=0).double().cpu()
                    val_score_se_sum += score_se

                    val_value_count += pi.size(0)

            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            val_policy = float(np.mean(val_policy_losses)) if val_policy_losses else 0.0
            val_value = float(np.mean(val_value_losses)) if val_value_losses else 0.0
            val_aux = float(np.mean(val_aux_losses)) if val_aux_losses else 0.0
            val_entropy = float(np.mean(val_entropies)) if val_entropies else 0.0
            val_score = val_score_loss_sum / max(len(val_losses), 1)
            val_top1_acc = val_top1_correct / max(val_total_samples, 1)
            val_top5_acc = val_top5_correct / max(val_total_samples, 1)
            val_value_mse = (val_value_se_sum / max(val_value_count, 1)).tolist()
            val_score_mse = (val_score_se_sum / max(val_value_count, 1)).tolist()

            summary_writer.add_scalar("val/loss_total", val_loss, epoch)
            summary_writer.add_scalar("val/loss_policy", val_policy, epoch)
            summary_writer.add_scalar("val/loss_value", val_value, epoch)
            summary_writer.add_scalar("val/loss_score", val_score, epoch)
            summary_writer.add_scalar("val/loss_aux", val_aux, epoch)
            summary_writer.add_scalar("val/entropy", val_entropy, epoch)
            summary_writer.add_scalar("val/top1_acc", val_top1_acc, epoch)
            summary_writer.add_scalar("val/top5_acc", val_top5_acc, epoch)
            for p_idx, p_mse in enumerate(val_value_mse):
                summary_writer.add_scalar(f"val/value_mse_p{p_idx}", p_mse, epoch)
            for p_idx, p_mse in enumerate(val_score_mse):
                summary_writer.add_scalar(f"val/score_mse_p{p_idx}", p_mse, epoch)

            LOGGER.info(
                f"Pretrain epoch {epoch+1}/{config.num_epochs}: "
                f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_top1={val_top1_acc:.3f} val_top5={val_top5_acc:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            LOGGER.info(
                f"Pretrain epoch {epoch+1}/{config.num_epochs}: "
                f"train_loss={avg_train_loss:.4f} (no val) "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # -------- Checkpoint best (or last, if no val available) --------
        improved = val_loss < best_val_loss if val_loader is not None else True
        if improved:
            best_val_loss = val_loss if val_loader is not None else best_val_loss
            try:
                best_checkpoint_num = save_model(model, model_dir)
                metrics.checkpoint_num = best_checkpoint_num
                LOGGER.info(
                    f"Saved pretraining checkpoint {best_checkpoint_num} "
                    f"(val_loss={val_loss:.4f})"
                )
            except Exception as e:
                LOGGER.warning(f"Failed to save pretraining checkpoint: {e}")

    metrics.epochs_trained = config.num_epochs
    if metrics.epoch_losses:
        metrics.avg_total_loss = float(np.mean(metrics.epoch_losses))
        metrics.avg_policy_loss = float(np.mean(metrics.epoch_policy_losses))
        metrics.avg_value_loss = float(np.mean(metrics.epoch_value_losses))

    # Write a richer JSON sidecar so the dashboard (and external tooling)
    # can plot pretrain curves without parsing TensorBoard event files.
    # ``best_val_loss``/``best_checkpoint_num`` are kept under the same
    # keys as before for backward compatibility.
    try:
        with open(tb_dir / "pretrain_summary.json", "w") as f:
            json.dump(
                {
                    "kind": "pretrain",
                    "timestamp": timestamp,
                    "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                    "best_checkpoint_num": best_checkpoint_num,
                    "epochs_trained": metrics.epochs_trained,
                    "training_examples": metrics.training_examples,
                    "avg_total_loss": metrics.avg_total_loss,
                    "avg_policy_loss": metrics.avg_policy_loss,
                    "avg_value_loss": metrics.avg_value_loss,
                    "top1_accuracy": metrics.epoch_top1_accuracy[-1] if metrics.epoch_top1_accuracy else None,
                    "top5_accuracy": metrics.epoch_top5_accuracy[-1] if metrics.epoch_top5_accuracy else None,
                    # Per-epoch arrays for dashboard plotting.
                    "epoch_losses": list(metrics.epoch_losses),
                    "epoch_policy_losses": list(metrics.epoch_policy_losses),
                    "epoch_value_losses": list(metrics.epoch_value_losses),
                    "epoch_top1_accuracy": list(metrics.epoch_top1_accuracy),
                    "epoch_top5_accuracy": list(metrics.epoch_top5_accuracy),
                    "epoch_lr": list(metrics.epoch_lr),
                },
                f,
                indent=2,
            )
    except Exception as e:
        LOGGER.warning(f"Could not write pretrain_summary.json: {e}")

    # Promote the best-val checkpoint by writing the ``current_best.json``
    # pointer for this architecture. Without this, the next ``main.py train``
    # invocation has to fall back to a lexicographic scan to locate the
    # pretrained weights — which still works in practice but is fragile
    # (e.g. coexisting older sessions can shadow the newly pretrained one).
    if best_checkpoint_num is not None:
        try:
            set_current_best(
                model_dir,
                arch=model.architecture_name(),
                session=session_name_for(model),
                checkpoint_num=best_checkpoint_num,
            )
        except Exception as e:
            LOGGER.warning(f"Could not update current_best pointer: {e}")

    summary_writer.close()
    return metrics


def _split_dataset(
    dataset: Dataset, validation_percentage: float, seed: int = 0
) -> Tuple[Dataset, Optional[Dataset]]:
    """Deterministic random train/val split of an arbitrary indexable dataset.

    Returns (train_subset, val_subset). val_subset is None if validation_percentage
    is 0 or if the resulting validation slice would be empty.
    """
    from torch.utils.data import Subset

    n = len(dataset)
    if n == 0:
        return dataset, None

    validation_percentage = max(0.0, min(1.0, float(validation_percentage)))
    val_size = int(round(n * validation_percentage))
    if val_size <= 0 or val_size >= n:
        return dataset, None

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def do_pretraining(model_dir: str, game_data_dir: str, config: TrainingConfig) -> TrainingMetrics:
    """Pre-train from human game data.

    game_data_dir can be either:
      - A directory of raw game JSON files (will be converted on the fly)
      - A path to pre-converted LMDB data (e.g. human_games/lmdb/training)

    Splits data into train/val using ``config.pretrain_validation_percentage``,
    then delegates to ``pretrain_model`` which owns the SL training loop,
    validation, and best-val checkpoint persistence.
    """
    from rl18xx.agent.alphazero.dataset import SelfPlayDataset

    model = get_latest_model(model_dir)

    data_path = Path(game_data_dir)

    def _try_load_lmdb(lmdb_dir: Path):
        """Load LMDB dataset, validating shape compatibility with the model."""
        ds = SelfPlayDataset(lmdb_dir)
        if len(ds) == 0:
            LOGGER.warning(f"LMDB at {lmdb_dir} is empty, skipping")
            ds.env.close()
            return None
        sample = ds[0]
        gs_dim = sample[0].shape[-1]
        expected_dim = model.config.game_state_size
        if gs_dim != expected_dim:
            LOGGER.warning(
                f"LMDB data has game_state_size={gs_dim} but model expects {expected_dim}. "
                f"Skipping stale LMDB; will re-convert from raw JSON."
            )
            ds.env.close()
            return None
        return ds

    def _run(train_ds: Dataset, val_ds: Optional[Dataset]) -> TrainingMetrics:
        return pretrain_model(model, train_ds, val_ds, config, model_dir)

    # ---- Case 1: path itself is an LMDB env (data.mdb sits directly) ----
    if (data_path / "data.mdb").exists():
        LOGGER.info(f"Loading pre-converted LMDB data from {data_path}")
        train_dataset = _try_load_lmdb(data_path)
        if train_dataset is not None:
            LOGGER.info(f"Loaded {len(train_dataset)} examples from LMDB")
            # Look for a sibling validation env.
            val_dataset: Optional[Dataset] = None
            sibling_val = data_path.parent / "validation"
            if (sibling_val / "data.mdb").exists():
                val_dataset = _try_load_lmdb(sibling_val)
                if val_dataset is not None:
                    LOGGER.info(f"Loaded {len(val_dataset)} validation examples from {sibling_val}")
            if val_dataset is None:
                train_dataset, val_dataset = _split_dataset(
                    train_dataset, config.pretrain_validation_percentage
                )
            return _run(train_dataset, val_dataset)

    # ---- Case 2: known LMDB subdirectory layouts ----
    for lmdb_subdir in ["lmdb_transformer", "lmdb"]:
        lmdb_root = data_path / lmdb_subdir
        lmdb_training = lmdb_root / "training"
        if (lmdb_training / "data.mdb").exists():
            LOGGER.info(f"Loading pre-converted LMDB data from {lmdb_training}")
            train_dataset = _try_load_lmdb(lmdb_training)
            if train_dataset is not None:
                LOGGER.info(f"Loaded {len(train_dataset)} examples from LMDB")
                val_dataset = None
                lmdb_validation = lmdb_root / "validation"
                if (lmdb_validation / "data.mdb").exists():
                    val_dataset = _try_load_lmdb(lmdb_validation)
                    if val_dataset is not None:
                        LOGGER.info(
                            f"Loaded {len(val_dataset)} validation examples from {lmdb_validation}"
                        )
                if val_dataset is None:
                    train_dataset, val_dataset = _split_dataset(
                        train_dataset, config.pretrain_validation_percentage
                    )
                return _run(train_dataset, val_dataset)

    # ---- Case 3: raw JSON conversion (in-memory HumanPlayDataset) ----
    json_dir = data_path
    if not list(json_dir.glob("*.json")):
        for subdir in ["1830_clean", "1830"]:
            candidate = data_path / subdir
            if candidate.exists() and list(candidate.glob("*.json")):
                json_dir = candidate
                break

    LOGGER.info(f"No compatible LMDB data found, converting from raw game JSON files in {json_dir}")
    games = load_games_from_json(str(json_dir))
    games = [g for g in games if g.get("status") != "error" and "title" in g]
    LOGGER.info(f"Filtered to {len(games)} valid games")
    games = [_load_cleaned_game_via_rust(game) for game in games]
    games = [g for g in games if g is not None]
    LOGGER.info(f"Loaded {len(games)} games via the Rust engine")
    encoder = Encoder_1830.get_encoder_for_model(model)
    training_data, validation_data = [], []
    for game in tqdm(games, desc="Converting games to training data"):
        train, val = convert_game_to_training_data(game, encoder, config=config)
        training_data.extend(train)
        validation_data.extend(val)

    train_dataset = HumanPlayDataset(training_data) if training_data else HumanPlayDataset([])
    val_dataset = HumanPlayDataset(validation_data) if validation_data else None

    if val_dataset is None and len(train_dataset) > 0:
        # Per-game split produced no val games; fall back to random index split.
        train_dataset, val_dataset = _split_dataset(
            train_dataset, config.pretrain_validation_percentage
        )

    return _run(train_dataset, val_dataset)
