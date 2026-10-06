"""PolicyValueComposite: priors and price head from one network, value from another."""
import torch

from rl18xx.agent.alphazero.composite_model import PolicyValueComposite


class _Net:
    def __init__(self, prob, value, price):
        self.prob, self.value, self.last_price_components = prob, value, None
        self._price = price
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def encoder_type(self):
        return "Transformer"

    def get_name(self):
        return f"net{self.prob}"

    def run_many_encoded(self, states):
        self.last_price_components = {"price_logits": torch.full((len(states), 2), self._price)}
        n = len(states)
        return [torch.full((3,), self.prob)] * n, None, [torch.full((4,), self.value)] * n


def test_composite_takes_policy_and_price_from_one_net_and_value_from_the_other():
    policy, value = _Net(prob=0.1, value=0.9, price=1.0), _Net(prob=0.7, value=0.2, price=-1.0)
    composite = PolicyValueComposite(policy, value).eval()
    probs, _, values = composite.run_many_encoded(["s1", "s2"])
    assert torch.allclose(probs[1], torch.full((3,), 0.1))
    assert torch.allclose(values[0], torch.full((4,), 0.2))
    assert torch.all(composite.last_price_components["price_logits"] == 1.0)
    p, _, v = composite.run_encoded("s")
    assert torch.allclose(p, torch.full((3,), 0.1)) and torch.allclose(v, torch.full((4,), 0.2))
    assert composite.get_name() == "net0.1+value:net0.7"
