"""``collate_examples`` keeps price targets per example.

PyG's default collater zips per-example lists together, truncating to the
shortest: one example without a price target erased the whole batch's
targets, so the continuous price head never received a gradient.
"""

import torch
from torch_geometric.data import Data

from rl18xx.agent.alphazero.dataset import collate_examples


def _row(price_targets):
    data = Data(x=torch.zeros(3, 2), edge_index=torch.zeros(2, 1, dtype=torch.long), edge_attr=torch.zeros(1))
    return torch.zeros(1, 5), data, torch.ones(7), torch.zeros(7), torch.tensor([0.5, 0.5]), price_targets


def test_mixed_batch_keeps_per_example_price_targets():
    bid = (4, 165.0, 1.0, 165.0, 800.0)
    game_state, data, mask, pi, value, price_targets = collate_examples([_row([]), _row([bid]), _row(None)])

    assert price_targets == [[], [bid], []]
    assert game_state.shape == (3, 1, 5)
    assert data.x.shape == (9, 2) and data.num_graphs == 3
    assert mask.shape == pi.shape == (3, 7)
    assert value.shape == (3, 2)
