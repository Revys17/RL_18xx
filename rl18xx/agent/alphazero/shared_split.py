"""One shared network, two learners: the policy path from policy gradient, the value
side from search games (docs/search_learning_design.md, "AlphaGo-style split").

A shared value-encoder network (``ModelTransformerConfig.value_entity_encoder``)
has two parts:

- the **policy path** -- entity encoder, map encoder, fusion, trunk, policy and
  price heads (everything a policy-only game uses);
- the **value side** -- the value-only entity encoder (``value_econ_transformer``)
  and the win / score (and training-only auxiliary) heads. It reads the policy
  path's trunk, but with ``value_stop_grad`` never trains it.

``policy-gradient --shared --value-follow <value run>`` trains the policy path and,
before each update, takes the value side from the value run's latest checkpoint;
``search-learning --shared --freeze-policy --policy-follow <PG run>`` trains only
the value side on search games and, before each update, takes the policy path from
the PG run's latest checkpoint. Each run's checkpoints are then whole networks: the
latest of both learners.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

VALUE_MODULES = ("value_econ_transformer", "win_loss_head", "score_head", "aux_seat_head", "aux_global_head")


def is_value_key(key: str) -> bool:
    """Whether a state-dict key belongs to the value side."""
    return key.split(".", 1)[0] in VALUE_MODULES


def value_parameters(model) -> list:
    return [p for name, p in model.named_parameters() if is_value_key(name)]


def policy_parameters(model) -> list:
    return [p for name, p in model.named_parameters() if not is_value_key(name)]


def checkpoint_state(path) -> dict:
    """A checkpoint's state dict (``checkpointer``'s self-describing format or a bare one)."""
    import torch

    from rl18xx.agent.alphazero.checkpointer import STATE_DICT_KEY

    data = torch.load(str(path), map_location="cpu", weights_only=False)
    return data[STATE_DICT_KEY] if isinstance(data, dict) and STATE_DICT_KEY in data else data


def load_part(model, state: dict, part: str) -> int:
    """Copy ``part`` ("value" or "policy") of ``state`` into ``model`` in place; returns
    the number of tensors copied. Keys of the other part are left alone; a key the
    model has but ``state`` lacks (or a shape mismatch) is an error, except the
    training-only auxiliary heads, which a checkpoint may not carry."""
    import torch

    want_value = part == "value"
    if part not in ("value", "policy"):
        raise ValueError(f"part must be value or policy, got {part!r}")
    own = model.state_dict()
    copied = 0
    with torch.no_grad():
        for key, tensor in own.items():
            if is_value_key(key) != want_value:
                continue
            if key not in state:
                if key.split(".", 1)[0] in ("aux_seat_head", "aux_global_head"):
                    continue
                raise KeyError(f"{part} key {key} missing from the source state")
            src = state[key]
            if tuple(src.shape) != tuple(tensor.shape):
                raise ValueError(f"{key}: shape {tuple(src.shape)} vs {tuple(tensor.shape)}")
            tensor.copy_(src.to(tensor.device, tensor.dtype))
            copied += 1
    if want_value and getattr(model, "value_econ_transformer", None) is None:
        raise ValueError("the model has no value entity encoder (value_entity_encoder): not a shared value network")
    return copied


def latest_checkpoint(run_dir, sub: str = "learner") -> Optional[tuple]:
    """``(update, path)`` of the newest ``<run_dir>/<sub>/<n>.pth``, or None."""
    directory = Path(run_dir) / sub
    if not directory.is_dir():
        return None
    best = None
    for path in directory.glob("*.pth"):
        try:
            n = int(path.stem)
        except ValueError:
            continue
        if best is None or n > best[0]:
            best = (n, path)
    return best


class Follower:
    """Loads one part of another run's latest checkpoint into a model when the run
    has written a newer one than last time (``poll``). A checkpoint deleted between
    listing and reading (runs keep only every N-th) is retried on the next poll."""

    def __init__(self, run_dir, part: str, sub: str = "learner"):
        self.run_dir, self.part, self.sub = Path(run_dir), part, sub
        self.update: Optional[int] = None
        self.loads = 0

    def poll(self, model) -> Optional[int]:
        """The source update loaded now, or None (nothing newer, or not readable yet)."""
        found = latest_checkpoint(self.run_dir, self.sub)
        if found is None or (self.update is not None and found[0] <= self.update):
            return None
        update, path = found
        try:
            state = checkpoint_state(path)
        except (FileNotFoundError, EOFError, RuntimeError) as e:  # deleted or still being renamed in
            LOGGER.warning(f"follow {self.run_dir}: could not read {path} ({e!r}); retrying later")
            return None
        load_part(model, state, self.part)
        self.update = update
        self.loads += 1
        return update
