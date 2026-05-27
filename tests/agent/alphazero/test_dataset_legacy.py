"""SelfPlayDataset reads back legacy 4-tuple LMDB rows without crashing.

LMDB rows pre-dating the continuous-price head were 4-tuples
``(state, legal_actions, pi, value)``. New rows are 5-tuples that carry an
extra ``price_targets`` field. ``SelfPlayDataset.__getitem__`` is supposed to
tolerate both layouts (per the doc-comment in ``dataset.py``); this test pins
that contract — a legacy 4-tuple row decodes cleanly and the ``price_targets``
slot of the returned 6-tuple defaults to an empty list.
"""

import io
from pathlib import Path

import lmdb
import lz4.frame
import pytest
import torch

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.dataset import SelfPlayDataset


def _write_legacy_row(lmdb_path: Path, state, legal_actions, pi, value):
    """Write a single legacy 4-tuple row to a fresh LMDB env at ``lmdb_path``."""
    buffer = io.BytesIO()
    torch.save((state, legal_actions, pi, value), buffer)
    compressed = lz4.frame.compress(buffer.getvalue())
    env = lmdb.open(str(lmdb_path), map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"00000000", compressed)
    finally:
        env.close()


def test_self_play_dataset_decodes_legacy_4_tuple(tmp_path):
    """A legacy 4-tuple row roundtrips cleanly via SelfPlayDataset and the
    returned ``price_targets`` defaults to ``[]``."""
    mapper = ActionMapper()
    action_size = mapper.action_encoding_size

    # Minimal encoded state: shape matches the dataset unpack
    # ``state[0], state[1], state[2], state[3]`` → (game_state, node_data,
    # edge_index, edge_attr). Tiny dummies are fine because the dataset only
    # routes them through; nothing here exercises the model.
    game_state = torch.zeros(1, 16, dtype=torch.float32)
    node_data = torch.zeros(2, 4, dtype=torch.float32)
    edge_index = torch.zeros(2, 1, dtype=torch.long)
    edge_attr = torch.zeros(1, dtype=torch.long)
    state = (game_state, node_data, edge_index, edge_attr)

    # Store legal_actions as a torch tensor; PyTorch 2.6's default
    # ``weights_only=True`` rejects numpy reconstruction without allow-listing,
    # and the production write path (in self-play) passes a torch tensor here.
    legal_actions = torch.tensor([0, 1, 2], dtype=torch.long)
    pi = torch.zeros(action_size, dtype=torch.float32)
    pi[0] = 1.0
    value = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    lmdb_dir = tmp_path / "legacy_db"
    _write_legacy_row(lmdb_dir, state, legal_actions, pi, value)

    ds = SelfPlayDataset(lmdb_dir)
    assert len(ds) == 1

    gs, data, legal_mask, pi_out, value_out, price_targets = ds[0]

    # The state passes through unchanged.
    assert gs.shape == game_state.shape
    # data wraps the node/edge tensors.
    assert hasattr(data, "x") and hasattr(data, "edge_index")
    # Legal mask reflects the input legal indices.
    assert legal_mask.shape == (action_size,)
    assert legal_mask[0].item() == 1.0
    assert legal_mask[1].item() == 1.0
    assert legal_mask[2].item() == 1.0
    assert legal_mask[3].item() == 0.0
    # pi and value pass through unchanged.
    assert torch.allclose(pi_out, pi)
    assert torch.allclose(value_out, value)
    # Legacy fallback contract: missing price_targets becomes an empty list.
    assert price_targets == []
