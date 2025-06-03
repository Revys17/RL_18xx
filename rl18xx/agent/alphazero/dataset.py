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
from typing import Union, List, Tuple
import lz4.frame
from rl18xx.agent.alphazero.action_mapper import ActionMapper

LOGGER = logging.getLogger(__name__)


class MCTSDataset(Dataset):
    def __init__(self, lmdb_path: Union[Path, str]):
        self.action_mapper = ActionMapper()
        if isinstance(lmdb_path, Path):
            lmdb_path = lmdb_path.as_posix()

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]

    def __getitem__(self, index):
        key = f"{index:08}".encode("ascii")
        with self.env.begin() as txn:
            compressed_data = txn.get(key)
        data = lz4.frame.decompress(compressed_data)
        buffer = io.BytesIO(data)

        state, legal_actions, pi, value = torch.load(buffer)
        legal_action_mask = torch.from_numpy(self.action_mapper.convert_indices_to_mask(legal_actions))
        game_state_data, node_data, edge_index, edge_attr = state

        data = Data(x=node_data, edge_index=edge_index, edge_attr=edge_attr)
        return game_state_data, data, legal_action_mask, pi, value

    def __len__(self):
        return self.length


class TrainingExampleProcessor:
    def __init__(self):
        self.encoder = Encoder_1830()

    def write_lmdb(self, game_data, lmdb_path: Union[Path, str], map_size=1e12):
        if isinstance(lmdb_path, Path):
            lmdb_path = lmdb_path.as_posix()

        samples = self.make_dataset_from_selfplay(game_data)

        env = lmdb.open(lmdb_path, map_size=int(map_size))
        
        start_index = 0
        with env.begin() as txn:
            start_index = txn.stat()["entries"]

        with env.begin(write=True) as txn:
            for i, (state, legal_actions, pi, value) in enumerate(tqdm(samples)):
                buffer = io.BytesIO()
                torch.save((state, legal_actions, pi, value), buffer)
                serialized_data = buffer.getvalue()
                compressed_data = lz4.frame.compress(serialized_data)

                key = f"{(start_index + i):08}".encode("ascii")
                txn.put(key, compressed_data)
        LOGGER.info(f"Wrote {len(samples)} samples to {lmdb_path}")

    def make_dataset_from_selfplay(
        self, data_extracts: List[Tuple[BaseGame, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]]:
        examples = []
        for game_state, legal_actions, searches_pi, result in data_extracts:
            encoded_state = self.encoder.encode(game_state)
            pi = searches_pi
            value = result
            examples.append((encoded_state, legal_actions, pi, value))
        return examples
