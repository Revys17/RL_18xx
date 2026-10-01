"""Value targets must live in the same canonical player frame as the state.

The encoder rotates player-indexed state sections so the active player sits
at slot 0, and MCTS unrotates the network's value output assuming the value
heads learned in that frame. Human-game rows (pretraining, autoresearch) keep
the full 8-tuple ``encoded_state`` with its ``rotation`` and store the value
in absolute seat order; self-play rows keep ``encoded_state[:6]`` with the
value already rotated. The datasets must rotate the former and leave the
latter alone — otherwise ~75% of pretraining rows train the value head
against the wrong seats.

The win-loss head must also keep nonexistent seats (beyond the game's player
count) out of its softmax.
"""

import io

import lmdb
import lz4.frame
import torch
import torch.nn.functional as F

from rl18xx.agent.alphazero.action_mapper import ActionMapper
from rl18xx.agent.alphazero.config import ModelTransformerConfig
from rl18xx.agent.alphazero.dataset import HumanPlayDataset, SelfPlayDataset, canonicalize_value_target
from rl18xx.agent.alphazero.encoder import Encoder_1830Graph
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.game.action_helper import ActionHelper
from rl18xx.game.gamemap import GameMap


def _dummy_state(rotation=None, num_players=4):
    state = (
        torch.zeros(1, 16),
        torch.zeros(2, 4),
        torch.zeros(2, 1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        0,
        0,
    )
    if rotation is None:
        return state
    return state + (rotation, num_players)


def _write_row(lmdb_path, state, value):
    pi = torch.zeros(ActionMapper().action_encoding_size)
    pi[0] = 1.0
    buffer = io.BytesIO()
    torch.save((state, torch.tensor([0, 1]), pi, value, []), buffer)
    env = lmdb.open(str(lmdb_path), map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"00000000", lz4.frame.compress(buffer.getvalue()))
    finally:
        env.close()


def test_canonicalize_rotates_rows_that_carry_rotation():
    value = torch.tensor([0.1, 0.2, 0.3, 0.4])
    rotated = canonicalize_value_target(_dummy_state(rotation=2), value)
    assert torch.equal(rotated, torch.tensor([0.3, 0.4, 0.1, 0.2]))


def test_canonicalize_leaves_self_play_rows_alone():
    value = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert torch.equal(canonicalize_value_target(_dummy_state(), value), value)
    assert torch.equal(canonicalize_value_target(_dummy_state(rotation=0), value), value)


def test_lmdb_dataset_rotates_human_rows_only(tmp_path):
    value = torch.tensor([0.1, 0.2, 0.3, 0.4])

    human_db = tmp_path / "human"
    _write_row(human_db, _dummy_state(rotation=1), value)
    assert torch.equal(SelfPlayDataset(human_db)[0][4], torch.tensor([0.2, 0.3, 0.4, 0.1]))

    self_play_db = tmp_path / "self_play"
    _write_row(self_play_db, _dummy_state(), value)
    assert torch.equal(SelfPlayDataset(self_play_db)[0][4], value)


def test_in_memory_dataset_rotates_human_rows():
    pi = torch.zeros(ActionMapper().action_encoding_size)
    pi[0] = 1.0
    value = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ds = HumanPlayDataset([(_dummy_state(rotation=3), [0, 1], pi, value, [])])
    assert torch.equal(ds[0][4], torch.tensor([0.4, 0.1, 0.2, 0.3]))


def test_canonical_slot_zero_is_the_active_player():
    """End to end on a real game: after P1's opening bid P2 is active, so the
    canonical value's slot 0 must hold P2's absolute entry."""
    game = GameMap().game_by_title("1830")({1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"})
    game.process_action(ActionHelper().get_all_choices(game)[0])
    assert game.active_players()[0].id == 2

    state = Encoder_1830Graph().encode(game)
    assert state[6] == 1

    absolute_value = torch.tensor([0.1, 0.4, 0.2, 0.3])  # P2 (index 1) has the largest share
    canonical = canonicalize_value_target(state, absolute_value)
    assert canonical[0] == absolute_value[1]


def test_win_loss_head_masks_nonexistent_seats():
    model = AlphaZeroTransformerModel(ModelTransformerConfig())
    model.eval()
    game = GameMap().game_by_title("1830")({i + 1: f"Player {i + 1}" for i in range(3)})
    model._compute_structural_matrices(game)

    with torch.no_grad():
        _, win_loss_logits, score_pred, _ = model._forward_encoded_batch([model.encoder.encode(game)])

    probs = F.softmax(win_loss_logits, dim=1).cpu()
    score_pred = score_pred.cpu()
    assert torch.allclose(probs[:, :3].sum(dim=1), torch.ones(1), atol=1e-5)
    assert torch.all(probs[:, 3:] < 1e-6)
    assert torch.all(score_pred[:, 3:] == 0)


def test_value_stop_grad_keeps_value_gradients_in_the_heads():
    model = AlphaZeroTransformerModel(ModelTransformerConfig())
    model.train()
    game = GameMap().game_by_title("1830")({i + 1: f"Player {i + 1}" for i in range(4)})
    model._compute_structural_matrices(game)
    encoded = [model.encoder.encode(game)]
    # The heads' output layers start at zero, which blocks upstream gradient
    # regardless of stop-grad; give them weights so the comparison means something.
    with torch.no_grad():
        for head in (model.win_loss_head, model.score_head):
            head[-1].weight.normal_(0, 0.1)

    def value_grads(stop_grad):
        model.zero_grad()
        model.value_stop_grad = stop_grad
        _, win_loss_logits, score_pred, _ = model._forward_encoded_batch(encoded)
        (win_loss_logits[:, :4].logsumexp(1).sum() + score_pred.sum()).backward()
        trunk = sum(p.grad.abs().sum() for n, p in model.named_parameters()
                    if p.grad is not None and n.startswith(("econ_transformer.", "res_blocks.")))
        head = sum(p.grad.abs().sum() for n, p in model.named_parameters()
                   if p.grad is not None and n.startswith(("win_loss_head.", "score_head.")))
        return float(trunk), float(head)

    trunk, head = value_grads(stop_grad=True)
    assert trunk == 0.0 and head > 0.0
    trunk, _ = value_grads(stop_grad=False)
    assert trunk > 0.0
