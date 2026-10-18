"""Do the values a search reads match what the value refit evaluated? (shared-network diagnosis)

For positions of saved games, every value source is evaluated through each path
between the checkpoint and the search tree, and the paths are compared with each
other (max abs difference, canonical frame) and against the games' outcomes (win CE):

- ``refit``: ``value_refit.model_predictions`` (``model(gs, nodes, value_only=True)``,
  the round type read back from the state) -- what the refit scored;
- ``full``: the same with the policy head run (``model(gs, nodes)``);
- ``rt``: ``value_only`` with the encoder's round type and player count;
- ``packed``: ``inference_server.run_packed_states`` on ``engine_rs.encode_games``
  (the server's fast path, in process, with legal indices);
- ``server``: a real inference server process (``eval_head_to_head.model_factory``,
  2 replicas) through ``LockstepSearcher`` policy requests with ``dedupe_maps``;
- ``tree``: a ``RustMCTSPlayer`` built as ``eval_head_to_head`` builds one, its root
  expanded by one lockstep round; its root value (absolute seats) rotated back.

Usage::

    uv run python scripts/diag_value_paths.py --games 40 --every 9 \\
        --source shared=model_checkpoints_pg/az1u280_shared_20261018/learner/280.pth \\
        --source critic=model_checkpoints_search/az1_20261017/learner/280.pth+model_checkpoints_search/az1_20261017/critic/280.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("diag_value_paths")
NUM_PLAYERS = 4


def gather(collection: Path, games: int, every: int):
    """``(positions, clones)``: value_refit Positions and an engine clone of each."""
    from rl18xx.agent.alphazero import value_refit as vr
    from rl18xx.agent.alphazero.policy_gradient import placement_points
    from rl18xx.agent.alphazero.start_positions import new_game

    positions, clones = [], []
    for game_id, actions, record in vr.collection_games(collection)[:games]:
        win = np.asarray(record["win_share"], dtype=np.float32)
        placement = placement_points(np.asarray(record["net_worth"], dtype=np.float64), (1.0, 0.5, 0.2, 0.0))
        g = new_game(NUM_PLAYERS)
        rs = g._game
        n = len(actions)
        for i, action in enumerate(actions):
            if not g.finished and i > 20 and i % every == 0:
                legal = np.asarray(rs.factored_legal_indices(), dtype=np.int32)
                if len(legal) > 1:
                    state, nodes, _, rotation, _ = rs.encode_canonical_py()
                    positions.append(
                        vr.Position(state, nodes, int(rotation), win, placement.astype(np.float32),
                                    min(2, 3 * i // max(n, 1)), game_id, legal)
                    )
                    clones.append(rs.pickle_clone())
            rs.process_action(action)
    return positions, clones


def path_refit(model, positions, device):
    from rl18xx.agent.alphazero import value_refit as vr

    return vr.model_predictions(model, positions, device)["win"]


def _torch_batches(positions, device, bs=256):
    import torch

    for b in range(0, len(positions), bs):
        chunk = positions[b : b + bs]
        gs = torch.from_numpy(np.stack([p.game_state for p in chunk])).float().to(device)
        nodes = torch.from_numpy(np.stack([p.node_features for p in chunk])).float().to(device)
        yield b, chunk, gs, nodes


def path_full(model, positions, device):
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for _, _, gs, nodes in _torch_batches(positions, device):
            _, w, _, _ = model(gs, nodes)
            out.append(F.softmax(w.float(), -1)[:, :NUM_PLAYERS].cpu().numpy())
    return np.concatenate(out)


def path_rt(model, positions, clones, device):
    import engine_rs
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for b, chunk, gs, nodes in _torch_batches(positions, device):
            enc = engine_rs.encode_games(clones[b : b + len(chunk)], False)
            rt = torch.as_tensor(np.asarray(enc["round_types"]), dtype=torch.long, device=device)
            npl = torch.full((len(chunk),), NUM_PLAYERS, dtype=torch.long, device=device)
            _, w, _, _ = model(gs, nodes, rt, None, npl, value_only=True)
            out.append(F.softmax(w.float(), -1)[:, :NUM_PLAYERS].cpu().numpy())
    return np.concatenate(out)


def round_type_agreement(model, positions, clones) -> float:
    import engine_rs
    import torch

    from rl18xx.agent.alphazero.model_transformer import MAX_PLAYERS, _pad_state_to_max_players

    gs = torch.from_numpy(np.stack([p.game_state for p in positions])).float()
    padded = _pad_state_to_max_players(gs, NUM_PLAYERS, MAX_PLAYERS)
    extracted = model._extract_round_type(padded).numpy()
    enc = np.concatenate(
        [np.asarray(engine_rs.encode_games(clones[b : b + 256], False)["round_types"]) for b in range(0, len(clones), 256)]
    )
    return float((extracted == enc).mean()), np.unique(extracted).tolist(), np.unique(enc).tolist()


def path_packed(model, positions, clones):
    import engine_rs
    import torch

    from rl18xx.agent.alphazero.lockstep import pack
    from rl18xx.agent.alphazero.inference_server import run_packed_states

    out = []
    with torch.no_grad():
        for b in range(0, len(clones), 256):
            chunk = clones[b : b + 256]
            batch = engine_rs.encode_games(chunk, False)
            legal = [np.asarray(p.legal, dtype=np.int32) for p in positions[b : b + 256]]
            _, _, values = run_packed_states(model, [pack(batch, legal)])
            out.append(values.float()[:, :NUM_PLAYERS].cpu().numpy())
    return np.concatenate(out)


def start_server(spec: str):
    import importlib.util

    from rl18xx.agent.alphazero.inference_server import InferenceClient, start_inference_server

    module_spec = importlib.util.spec_from_file_location("eval_head_to_head", REPO_ROOT / "scripts" / "eval_head_to_head.py")
    h2h = importlib.util.module_from_spec(module_spec)
    sys.modules["eval_head_to_head"] = h2h
    module_spec.loader.exec_module(h2h)
    path, _ = h2h.parse_player(spec + "@200", 200)
    handle = start_inference_server(
        num_workers=2, model_factory=h2h.model_factory, checkpoint_path=path, batch_size=512,
        batch_timeout_ms=2.0, autocast_device=None, replicas=2,
    )
    worker_id = handle.ticket_q.get()
    client = InferenceClient(request_q=handle.request_q, reply_q=handle.reply_qs[worker_id], worker_id=worker_id)
    return handle, client


def path_server(client, positions, clones):
    from rl18xx.agent.alphazero.lockstep import ClientEvaluator, LockstepSearcher, PolicyRequest

    searcher = LockstepSearcher({"m": ClientEvaluator(client)}, dedupe_maps=True)
    out = []
    for b in range(0, len(clones), 128):
        reqs = [PolicyRequest(c, [int(i) for i in p.legal], key="m") for c, p in zip(clones[b : b + 128], positions[b : b + 128])]
        searcher.run([], reqs)
        out.extend(np.asarray(r.value, dtype=np.float32)[:NUM_PLAYERS] for r in reqs)
    return np.stack(out)


def path_tree(client, positions, clones):
    """Root values of eval-configured trees after their root expansion, rotated to the canonical frame."""
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.lockstep import ClientEvaluator, LockstepSearcher
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
    from rl18xx.agent.alphazero.search_learning import root_value
    from rl18xx.rust_adapter import RustGameAdapter

    overrides = dict(num_readouts=200, min_readouts=200, c_puct_init=0.05, c_puct_by_round={}, mcts_mean_q=True,
                     fpu_reduction=0.0, dedupe_leaves=True, parallel_readouts=8, prior_temperature=0.5)
    searcher = LockstepSearcher({"m": ClientEvaluator(client)}, dedupe_maps=True)
    out = np.full((len(clones), NUM_PLAYERS), np.nan, dtype=np.float32)
    for b in range(0, len(clones), 64):
        players, searches = [], []
        for j in range(b, min(b + 64, len(clones))):
            config = SelfPlayConfig(
                network=None, use_inference_server=True, inference_client=client, **overrides, softpick_move_cutoff=0,
                dirichlet_noise_weight=0.0, enable_resign=False, noresign_holdout_rate=0.0, max_game_length=1000,
                player_count_distribution={NUM_PLAYERS: 1.0}, auction_unlock=True,
            )
            player = RustMCTSPlayer(config)
            player.initialize_game(RustGameAdapter(clones[j].pickle_clone()))
            s = player.start_search(key="m")
            if s is None:
                continue
            players.append((j, s))
            searches.append(s)
        # One round: the roots' expansion only (each tree's first round is a one-leaf root expansion).
        searcher.round(searches)
        for j, s in players:
            v = root_value(s.tree)[:NUM_PLAYERS]
            out[j] = np.roll(v, -positions[j].rotation)
    return out


def win_ce(pred, positions):
    win = np.stack([np.roll(p.win, -p.rotation) for p in positions])
    p = np.clip(pred / pred.sum(1, keepdims=True), 1e-9, 1)
    return float(-(win * np.log(p)).sum(1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", required=True, help="An eval collection (games/*.json + games.jsonl)")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--every", type=int, default=9)
    parser.add_argument("--source", action="append", required=True, help="name=<spec> (eval player spec, no @)")
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("logs/shared_diag/value_paths.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    from rl18xx.agent.alphazero.composite_model import load_policy_value

    t0 = time.time()
    positions, clones = gather(Path(args.collection), args.games, args.every)
    LOGGER.info(f"{len(positions)} positions from {args.games} games in {time.time() - t0:.0f}s")
    report = {"collection": args.collection, "positions": len(positions), "sources": {}}
    for item in args.source:
        name, _, spec = item.partition("=")
        model = load_policy_value(spec)
        device = model.device
        paths = {}
        if hasattr(model, "forward"):  # a plain network: its own value heads
            agree = round_type_agreement(model, positions, clones)
            report.setdefault("round_type_agreement", agree)
            paths["refit"] = path_refit(model, positions, device)
            paths["full"] = path_full(model, positions, device)
            paths["rt"] = path_rt(model, positions, clones, device)
        else:  # a composite: the value network's paths
            vm = model.value_model
            paths["refit"] = path_refit(vm, positions, device)
            paths["rt"] = path_rt(vm, positions, clones, device)
        paths["packed"] = path_packed(model, positions, clones)
        del model
        torch.cuda.empty_cache()
        if not args.no_server:
            handle, client = start_server(spec)
            try:
                paths["server"] = path_server(client, positions, clones)
                paths["tree"] = path_tree(client, positions, clones)
            finally:
                handle.shutdown()
        base = paths["refit"]
        entry = {}
        for k, v in paths.items():
            ok = ~np.isnan(v).any(1)
            entry[k] = {
                "max_abs_vs_refit": float(np.abs(v[ok] - base[ok]).max()),
                "mean_abs_vs_refit": float(np.abs(v[ok] - base[ok]).mean()),
                "win_ce": win_ce(v[ok], [p for p, o in zip(positions, ok) if o]),
                "n": int(ok.sum()),
            }
            LOGGER.info(f"{name} {k}: {entry[k]}")
        report["sources"][name] = {"spec": spec, "paths": entry}
        np.savez_compressed(args.out.with_suffix(f".{name}.npz"), **paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
