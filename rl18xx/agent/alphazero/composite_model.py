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
    def __init__(self, policy_model, value_model):
        self.policy_model = policy_model
        self.value_model = value_model
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
        self.last_price_components = getattr(self.policy_model, "last_price_components", None)
        _, _, values = self.value_model.run_many_encoded(encoded_game_states, *args, **kwargs)
        return probs, log_probs, values

    def run_encoded(self, encoded_game_state):
        probs, log_probs, values = self.run_many_encoded([encoded_game_state])
        return probs[0], log_probs[0] if log_probs is not None else None, values[0]


def load_policy_value(checkpoint_paths: str):
    """``"<policy.pth>+<value.pth>"`` -> a :class:`PolicyValueComposite`; a single
    path -> that checkpoint's model. Module-level so it pickles into the
    inference server process."""
    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint

    def load(path):
        path = Path(path)
        model = _load_model_from_session_checkpoint(path.parent, path)
        model.eval()
        return model

    parts = str(checkpoint_paths).split("+")
    if len(parts) == 1:
        return load(parts[0])
    if len(parts) != 2:
        raise ValueError(f"expected '<policy>+<value>', got {checkpoint_paths!r}")
    return PolicyValueComposite(load(parts[0]), load(parts[1])).eval()
