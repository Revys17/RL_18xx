"""Search with one network's policy and another's value (AlphaGo's split).

AlphaGo searched with its supervised policy for priors and a separately
trained value network for leaf evaluation. ``PolicyValueComposite`` gives the
inference server and MCTS that split: priors and price-head output from
``policy_model``, values from ``value_model`` (both in the encoder's canonical
frame, so callers unrotate as usual). It quacks like a model wherever the
search or the inference server uses one (``run_many_encoded``,
``run_encoded``, ``last_price_components``, ``eval``, ``encoder_type``).
"""

from pathlib import Path


class PolicyValueComposite:
    def __init__(self, policy_model, value_model, value_head: str = "win"):
        self.policy_model = policy_model
        self.value_model = value_model
        # "win": per-seat win probabilities; "score": the score head's per-seat output
        # (a policy-gradient exploiter's critic regresses its shaped reward there);
        # "blend=<w>": (1 - w) * win + w * score -- KataGo-style, the critic's predicted
        # outcome (gen4's: placement points 1 / 0.5 / 0.2 / 0) mixed into the win estimate.
        _, self.blend = parse_value_head(value_head)
        self.value_head = value_head
        self.last_price_components = None

    @property
    def device(self):
        return self.policy_model.device

    def eval(self):
        self.policy_model.eval()
        self.value_model.eval()
        return self

    def encoder_type(self):
        return self.policy_model.encoder_type()

    def get_name(self):
        return f"{self.policy_model.get_name()}+value:{self.value_model.get_name()}"

    def run_many_encoded(self, encoded_game_states, *args, **kwargs):
        probs, log_probs, _ = self.policy_model.run_many_encoded(encoded_game_states, *args, **kwargs)
        kwargs.pop("legal_indices", None)  # only the policy network uses them
        self.last_price_components = getattr(self.policy_model, "last_price_components", None)
        if self.value_head == "score":
            return probs, log_probs, self.value_model.run_scores_encoded(encoded_game_states)
        if self.blend is not None:
            win, score = self.value_model.run_values_and_scores_encoded(encoded_game_states)
            return probs, log_probs, (1.0 - self.blend) * win + self.blend * score
        # A value-only forward skips the value network's policy head (most of its cost).
        run_values = getattr(self.value_model, "run_values_encoded", None)
        if run_values is not None:
            values = run_values(encoded_game_states)
        else:
            _, _, values = self.value_model.run_many_encoded(encoded_game_states, *args, **kwargs)
        return probs, log_probs, values

    @property
    def supports_packed(self) -> bool:
        return all(hasattr(m, "run_packed") for m in (self.policy_model, self.value_model))

    def packed_inputs(self, parts: list) -> dict:
        """One set of device inputs for both networks (``AlphaZeroTransformerModel.packed_inputs``)."""
        return self.policy_model.packed_inputs(parts)

    def run_packed(self, inputs: dict, legal_indices=None):
        """``run_many_encoded`` on ``packed_inputs``: the policy network's forward
        and the value network's value-only forward read the same device tensors."""
        probs, log_probs, _ = self.policy_model.run_packed(inputs, legal_indices)
        self.last_price_components = getattr(self.policy_model, "last_price_components", None)
        if self.blend is not None:
            win, score = self.value_model.run_packed(inputs, head="win+score")
            return probs, log_probs, (1.0 - self.blend) * win + self.blend * score
        return probs, log_probs, self.value_model.run_packed(inputs, head=self.value_head)

    def run_encoded(self, encoded_game_state):
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0] if log_probs is not None else None, values[0]


def parse_value_head(value_head: str):
    """``(kind, blend weight or None)`` of a value-head spec: ``win``, ``score`` or ``blend=<w>``."""
    if value_head.startswith("blend="):
        blend = float(value_head.split("=", 1)[1])
        if not 0.0 <= blend <= 1.0:
            raise ValueError(f"blend weight {blend} outside [0, 1]")
        return "blend", blend
    if value_head not in ("win", "score"):
        raise ValueError(f"value head {value_head!r}: 'win', 'score' or 'blend=<w>'")
    return value_head, None


class SharedValueModel:
    """One network (AlphaZero's shared trunk) whose search value is read from its own
    heads as ``value_head`` says -- ``score`` or ``blend=<w>`` ((1 - w) * win + w *
    score) -- from the same forward as its policy (a plain checkpoint already gives
    its win head). Quacks like a model where the search and the inference server use
    one, as :class:`PolicyValueComposite` does."""

    def __init__(self, model, value_head: str):
        self.kind, self.blend = parse_value_head(value_head)
        self.model = model
        self.value_head = value_head
        self.last_price_components = None

    @property
    def device(self):
        return self.model.device

    def eval(self):
        self.model.eval()
        return self

    def encoder_type(self):
        return self.model.encoder_type()

    def get_name(self):
        return f"{self.model.get_name()}#{self.value_head}"

    def _values(self, win_loss_logits, score_pred):
        import torch.nn.functional as F

        win = F.softmax(win_loss_logits, dim=-1)
        if self.kind == "win":
            return win
        if self.kind == "score":
            return score_pred
        return (1.0 - self.blend) * win + self.blend * score_pred

    def run_many_encoded(self, encoded_game_states, legal_indices=None):
        """``(probabilities, log_probs, values)`` as ``AlphaZeroModel.run_many_encoded``,
        the values from this wrapper's heads."""
        import torch.nn.functional as F

        if len(encoded_game_states) == 0:
            raise ValueError("Received no game states to run.")
        m = self.model
        hints = m._policy_row_hints(legal_indices) if legal_indices is not None else {}
        policy_logits, win_loss_logits, score_pred, _ = m._forward_encoded_batch(encoded_game_states, **hints)
        self.last_price_components = getattr(m, "last_price_components", None)
        probs = F.softmax(policy_logits, dim=1)
        log_probs = None if legal_indices is not None else F.log_softmax(policy_logits, dim=1)
        return probs, log_probs, self._values(win_loss_logits, score_pred)

    def run_encoded(self, encoded_game_state):
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0] if log_probs is not None else None, values[0]

    @property
    def supports_packed(self) -> bool:
        return hasattr(self.model, "run_packed")

    def packed_inputs(self, parts):
        return self.model.packed_inputs(parts)

    def run_packed(self, inputs: dict, legal_indices=None, head: str = "policy"):
        """``run_many_encoded`` on ``packed_inputs`` (the inference server's fast path):
        one forward gives the policy and this wrapper's values."""
        import torch.nn.functional as F

        m = self.model
        if head != "policy":
            _, win_loss_logits, score_pred, _ = m.forward_packed(inputs, value_only=True)
            return self._values(win_loss_logits, score_pred)
        hints = m._policy_row_hints(legal_indices) if legal_indices is not None else {}
        policy_logits, win_loss_logits, score_pred, _ = m.forward_packed(inputs, **hints)
        self.last_price_components = getattr(m, "last_price_components", None)
        probs = F.softmax(policy_logits, dim=1)
        log_probs = None if legal_indices is not None else F.log_softmax(policy_logits, dim=1)
        return probs, log_probs, self._values(win_loss_logits, score_pred)

    def run_values_encoded(self, encoded_game_states):
        _, win_loss_logits, score_pred, _ = self.model._forward_encoded_batch(encoded_game_states, value_only=True)
        return self._values(win_loss_logits, score_pred)


def load_policy_value(checkpoint_paths: str):
    """``"<policy.pth>+<value.pth>"`` -> a :class:`PolicyValueComposite` (``...#score``:
    values from the value network's score head; ``#blend=<w>``: win and score mixed); a single path -> that
    checkpoint's model (``<path>#score`` / ``#blend=<w>``: a :class:`SharedValueModel`, the
    search value from its own heads). Module-level so it pickles into the inference server process."""
    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint

    def load(path):
        path = Path(path)
        model = _load_model_from_session_checkpoint(path.parent, path)
        model.eval()
        return model

    paths, _, value_head = str(checkpoint_paths).partition("#")
    parts = paths.split("+")
    if len(parts) == 1 and value_head in ("", "win"):
        return load(parts[0])
    if len(parts) == 1:
        return SharedValueModel(load(parts[0]), value_head).eval()
    if len(parts) != 2:
        raise ValueError(f"expected '<policy>+<value>[#score]', got {checkpoint_paths!r}")
    return PolicyValueComposite(load(parts[0]), load(parts[1]), value_head or "win").eval()
