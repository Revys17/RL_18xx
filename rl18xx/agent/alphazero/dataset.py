from rl18xx.agent.alphazero.encoder import Encoder_1830
from rl18xx.game.engine.game import BaseGame
import lmdb
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
import logging
import io
from pathlib import Path
from typing import Any, Union, List, Tuple
import lz4.frame
from rl18xx.agent.alphazero.action_mapper import ActionMapper

LOGGER = logging.getLogger(__name__)


class Dataset_1830(Dataset):
    pass

class SelfPlayDataset(Dataset_1830):
    def __init__(self, lmdb_path: Union[Path, str], start_index: int = 0):
        self.action_mapper = ActionMapper()
        if isinstance(lmdb_path, Path):
            lmdb_path = lmdb_path.as_posix()

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.start_index = start_index
        with self.env.begin() as txn:
            total_entries = txn.stat()["entries"]
        self.length = total_entries - self.start_index

    def _maybe_pad_pi(self, pi):
        """Zero-pad legacy 26535-wide pi vectors to the current action space."""
        expected = self.action_mapper.action_encoding_size
        if isinstance(pi, torch.Tensor):
            if pi.shape[-1] < expected:
                pad_amount = expected - pi.shape[-1]
                pad = torch.zeros(*pi.shape[:-1], pad_amount, dtype=pi.dtype)
                pi = torch.cat([pi, pad], dim=-1)
            return pi
        if isinstance(pi, np.ndarray):
            if pi.shape[-1] < expected:
                pad_amount = expected - pi.shape[-1]
                pad_shape = list(pi.shape)
                pad_shape[-1] = pad_amount
                pad = np.zeros(pad_shape, dtype=pi.dtype)
                pi = np.concatenate([pi, pad], axis=-1)
            return pi
        return pi

    def __getitem__(self, index):
        key = f"{self.start_index + index:08}".encode("ascii")
        with self.env.begin() as txn:
            compressed_data = txn.get(key)
        data = lz4.frame.decompress(compressed_data)
        buffer = io.BytesIO(data)

        # Backward-compat: legacy LMDB rows were 4-tuples
        # ``(state, legal_actions, pi, value)``. New rows are 5-tuples that
        # carry an extra ``price_targets`` field (the continuous-price head's
        # NLL targets — a list of ``(slot, price, weight, price_min,
        # price_max)`` tuples for the chosen action, or ``None`` when the
        # action wasn't price-bearing). Older datasets read back with
        # ``price_targets=None`` so the price head stays inactive on legacy
        # data without forcing a re-conversion.
        loaded = torch.load(buffer, map_location=torch.device('cpu'))
        if len(loaded) == 5:
            state, legal_actions, pi, value, price_targets = loaded
        else:
            state, legal_actions, pi, value = loaded
            price_targets = []
        # Legacy data was written when the action space was 26535-wide; the
        # current mapper is 26537 (two new D-train slots at the tail). Pad with
        # zero so old rows align with the new policy_size without forcing a
        # re-conversion. The new tail slots get zero target mass.
        pi = self._maybe_pad_pi(pi)
        legal_action_mask = torch.from_numpy(self.action_mapper.convert_indices_to_mask(legal_actions))
        # Encoder returns (game_state, node_data, edge_index, edge_attr, [round_type_idx, active_player_idx])
        game_state_data, node_data, edge_index, edge_attr = state[0], state[1], state[2], state[3]

        data = Data(x=node_data, edge_index=edge_index, edge_attr=edge_attr)
        return game_state_data, data, legal_action_mask, pi, value, price_targets

    def __len__(self):
        return self.length

class HumanPlayDataset(Dataset_1830):
    def __init__(self, examples: List[Any], price_targets: List[Any] = None):
        """In-memory pretraining dataset.

        ``examples`` items are 4-tuples ``(encoded_game_state, legal_actions,
        pi, value)`` or 5-tuples that also carry the per-example
        ``price_targets`` payload.  When the caller supplies a parallel
        ``price_targets`` list, those entries take precedence; otherwise we
        read the 5th tuple element when present and default to ``None``.

        ``price_targets[i]`` is either ``None`` (chosen action wasn't
        price-bearing) or a list of ``(slot_idx, price, weight, price_min,
        price_max)`` tuples — the format consumed by
        ``_compute_price_nll_loss``.
        """
        self.examples = examples
        self.action_mapper = ActionMapper()
        # Cache an aligned price_targets list so __getitem__ stays O(1) and
        # callers can pass targets either inline (5-tuple) or as a sidecar
        # list (parallel to ``examples``) without forking the read path.
        if price_targets is not None:
            if len(price_targets) != len(examples):
                raise ValueError(
                    f"price_targets length {len(price_targets)} != examples length {len(examples)}"
                )
            self._price_targets = list(price_targets)
        else:
            self._price_targets = [
                ex[4] if len(ex) >= 5 and ex[4] is not None else [] for ex in examples
            ]

    def _maybe_pad_pi(self, pi):
        """Zero-pad legacy 26535-wide pi vectors to the current action space."""
        expected = self.action_mapper.action_encoding_size
        if isinstance(pi, torch.Tensor):
            if pi.shape[-1] < expected:
                pad_amount = expected - pi.shape[-1]
                pad = torch.zeros(*pi.shape[:-1], pad_amount, dtype=pi.dtype)
                pi = torch.cat([pi, pad], dim=-1)
            return pi
        if isinstance(pi, np.ndarray):
            if pi.shape[-1] < expected:
                pad_amount = expected - pi.shape[-1]
                pad_shape = list(pi.shape)
                pad_shape[-1] = pad_amount
                pad = np.zeros(pad_shape, dtype=pi.dtype)
                pi = np.concatenate([pi, pad], axis=-1)
            return pi
        return pi

    def __getitem__(self, index):
        example = self.examples[index]
        # Tolerate both 4-tuple (legacy) and 5-tuple (with price_targets)
        # rows: cached ``_price_targets`` is the source of truth.
        game_state, legal_actions, pi, value = example[0], example[1], example[2], example[3]
        price_targets = self._price_targets[index]
        pi = self._maybe_pad_pi(pi)
        legal_action_mask = torch.from_numpy(self.action_mapper.convert_indices_to_mask(legal_actions))
        # Encoder now returns an 8-tuple ``(game_state, node_data, edge_index,
        # edge_attr, round_type_idx, active_player_idx, rotation,
        # num_players)``. Only the first four slots feed the network; the
        # rest are bookkeeping kept on the example so the bucket sampler can
        # read ``num_players`` directly from ``game_state[7]``.
        game_state_data, node_data, edge_index, edge_attr = (
            game_state[0],
            game_state[1],
            game_state[2],
            game_state[3],
        )
        data = Data(x=node_data, edge_index=edge_index, edge_attr=edge_attr)
        return game_state_data, data, legal_action_mask, pi, value, price_targets

    def __len__(self):
        return len(self.examples)


class TrainingExampleProcessor:
    def __init__(self, encoder: Encoder_1830):
        self.encoder = encoder

    def write_lmdb(self, game_data, lmdb_path: Union[Path, str], map_size=1e12):
        samples = self.make_dataset_from_selfplay(game_data)
        self.write_samples(samples, lmdb_path, map_size)

    def make_dataset_from_selfplay(
        self, data_extracts,
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Any]]:
        """Convert self-play ``extract_data`` rows into LMDB-ready 5-tuples.

        Accepts the 5-tuple form ``(game_state, legal_actions, searches_pi,
        result, price_targets)`` from ``MCTSPlayer.extract_data`` and the
        legacy 4-tuple form (older callers / fixtures); the legacy form gets
        an empty ``price_targets`` list to preserve schema uniformity.
        ``price_targets`` is a list of
        ``(slot_idx, price, weight, price_min, price_max)`` tuples,
        visit-weighted across MCTS price grandchildren at the chosen slot.
        """
        examples = []
        for row in data_extracts:
            if len(row) == 5:
                game_state, legal_actions, searches_pi, result, price_targets = row
            else:
                game_state, legal_actions, searches_pi, result = row
                price_targets = []
            encoded_state = self.encoder.encode(game_state)
            # encoded_state = (game_state_tensor, node_features, edge_index, edge_attr,
            #                  round_type_idx, active_player_idx, rotation, num_players)
            # The encoder now canonicalizes the game-state vector itself so that the
            # active player sits at slot 0 (active_player_idx == 0). We still need to
            # rotate the per-player value target into the same canonical frame.
            rotation = encoded_state[6]

            value = result
            if rotation != 0:
                value = torch.roll(result, shifts=-rotation, dims=0)

            # ``price_targets`` are populated by ``MCTSPlayer.play_move`` for
            # price-bearing slots (Bid / cross-corp BuyTrain / BuyCompany) by
            # aggregating MCTS visit counts across the price grandchildren.
            # Pretraining writes its own ``price_targets`` for the same LMDB
            # schema via the human-data converter.
            examples.append(
                (encoded_state[:6], legal_actions, searches_pi, value, price_targets)
            )
        return examples

    def write_samples(self, samples, lmdb_path: Union[Path, str], map_size=1e12):
        """Write training samples to an LMDB env.

        Each sample is a 5-tuple ``(state, legal_actions, pi, value,
        price_targets)``. Legacy 4-tuples are accepted for backward
        compatibility — they're padded out with ``price_targets=None`` so
        readers see a uniform schema.
        """
        if isinstance(lmdb_path, Path):
            lmdb_path = lmdb_path.as_posix()
        env = lmdb.open(lmdb_path, map_size=int(map_size))

        start_index = 0
        with env.begin() as txn:
            start_index = txn.stat()["entries"]

        with env.begin(write=True) as txn:
            for i, sample in enumerate(tqdm(samples)):
                if len(sample) == 5:
                    state, legal_actions, pi, value, price_targets = sample
                else:
                    state, legal_actions, pi, value = sample
                    price_targets = None
                buffer = io.BytesIO()
                torch.save(
                    (state, legal_actions, pi, value, price_targets), buffer
                )
                serialized_data = buffer.getvalue()
                compressed_data = lz4.frame.compress(serialized_data)

                key = f"{(start_index + i):08}".encode("ascii")
                txn.put(key, compressed_data)
        LOGGER.info(f"Wrote {len(samples)} samples to {lmdb_path}")

