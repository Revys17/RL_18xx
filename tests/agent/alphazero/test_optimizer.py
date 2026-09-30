"""The shared optimizer must not collapse weakly-gradiented parameters.

Plain ``Adam(weight_decay=...)`` adds the L2 term to the gradient and then
normalizes it, so a parameter whose loss gradient is ~0 moves toward zero by
~``lr`` every step. In pretraining that zeroed the whole economic transformer
within one epoch. ``build_optimizer`` uses decoupled decay (AdamW) and exempts
biases, norms and embeddings.
"""

import torch
from torch import nn

from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.agent.alphazero.train import build_optimizer


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 8)
        self.trunk = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)
        self.idle = nn.Linear(8, 8)  # in the graph, but its gradient is ~0
        self.win_loss_head = nn.Linear(8, 3)

    def forward(self, idx):
        x = self.norm(self.trunk(self.embedding(idx)))
        return self.win_loss_head(x) + 1e-9 * self.idle(x).sum()


def test_param_groups():
    model = _Toy()
    config = TrainingConfig(lr=1e-3, value_lr_multiplier=3.0, weight_decay=1e-4)
    opt = build_optimizer(model, config)
    assert isinstance(opt, torch.optim.AdamW)

    group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
    assert opt.param_groups[0]["lr"] == 1e-3 and opt.param_groups[0]["weight_decay"] == 1e-4
    for p in (model.embedding.weight, model.norm.weight, model.norm.bias, model.trunk.bias):
        assert group_of[id(p)]["weight_decay"] == 0.0
    assert group_of[id(model.trunk.weight)]["weight_decay"] == 1e-4
    assert group_of[id(model.win_loss_head.weight)]["lr"] == 3e-3
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_weakly_gradiented_params_survive():
    torch.manual_seed(0)
    model = _Toy()
    opt = build_optimizer(model, TrainingConfig(lr=1e-3, weight_decay=1e-4))
    idle_before = model.idle.weight.detach().norm().item()
    idx = torch.randint(0, 4, (32,))
    target = torch.randint(0, 3, (32,))
    for _ in range(2000):
        loss = nn.functional.cross_entropy(model(idx), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Plain Adam + L2 drives this to ~0 (it moves ~lr per step for 2000 steps).
    assert model.idle.weight.detach().norm().item() > 0.9 * idle_before
