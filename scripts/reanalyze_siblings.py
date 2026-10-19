"""Reanalyze value targets on sibling positions (MuZero Reanalyze / search-value targets).

A search compares a position's children; the shared value-encoder network ranks them
worst in track lays (``diag_sibling_values.py``: Spearman 0.75 against az1's critic,
0.97 in stock rounds). Here the value is taught to rank children the way a deeper
search of the *same* network does -- no teacher network:

``build``: decision positions of saved games (``eval_head_to_head`` collections, as
``diag_sibling_values.build`` samples them); each position's root is expanded by
``--expander`` (default ``--model``), up to ``--max-children`` children are built the
way the search builds them (``maybe_add_child``: forced follow-ups collapsed, an open
price from the price head's proposal; half the top moves by prior, half drawn at
random from the rest), and every non-terminal child is searched as a root by
``--model`` with ``--readouts`` readouts (the search-learning generator's settings:
mean Q, c_puct 0.05, FPU 0, dedupe, 8 leaves, priors at the run's temperature). A
child's target is that search's root value -- the mean of every value backed up
through it, per seat -- in the child's canonical frame (its mover first), i.e. the
value head's own frame. The positions themselves are stored with their games'
outcomes. Writes ``diag_sibling_values.Store`` arrays + ``targets.npy`` + ``meta.json``
(+ the cost: evaluations, forward seconds, wall seconds).

``check``: before any training -- do the reanalyzed values rank siblings closer to a
reference critic than the network's own raw values (per step type)? If not, training
toward them can't help.

``train``: fine-tune the value side only -- the value entity encoder
(``value_econ_transformer``) and the win / score heads, the shared trunk and the
policy frozen bit for bit, as ``search-learning --shared --value-stop-grad`` trains it
-- on the children's reanalyzed targets (win CE, ``--sibling-weight``) and the
positions' outcomes (win CE, ``--outcome-weight``), the score head anchored to its
starting outputs. The trunk features are computed once (frozen), so a step costs
only the value encoder and heads. Evaluates on a held-out ``diag_sibling_values``
set (``--test``: sibling Spearman / top-1 / regret against its reference critic per
step type, and win CE of its positions against their outcomes) and on held-out
games of the build sets (win CE vs outcome, CE to the reanalyzed targets).

Usage::

    uv run python scripts/reanalyze_siblings.py build --model <ckpt.pth> --out logs/reanalyze/train1 \\
        --collection logs_shared/eval/az6_u100_vs_az4_u100 --games 60 --per-game 8 --readouts 64
    uv run python scripts/reanalyze_siblings.py check --data logs/reanalyze/train1 --model <ckpt.pth> \\
        --reference critic280=model_checkpoints_search/az1_20261017/critic/280.pth
    uv run python scripts/reanalyze_siblings.py train --data logs/reanalyze/train1 --init <ckpt.pth> \\
        --test-set critic=logs/shared_diag/siblings_v2:values.critic280 \\
        --out model_checkpoints_search/<run>/learner/<n>.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import diag_sibling_values as dsv  # noqa: E402

LOGGER = logging.getLogger("reanalyze_siblings")
NUM_PLAYERS = 4


# ---------------------------------------------------------------------------
# Build: children and their reanalyzed values
# ---------------------------------------------------------------------------


def choose_children(prior: np.ndarray, max_children: int, rng: np.random.Generator) -> list:
    """Indices of the children to build: all when there are at most ``max_children``,
    else the top half by prior and the rest drawn uniformly from the others."""
    n = len(prior)
    if n <= max_children:
        return list(range(n))
    top = [int(k) for k in np.argsort(-prior)[: max_children // 2]]
    rest = [k for k in range(n) if k not in set(top)]
    extra = [int(k) for k in rng.choice(rest, size=max_children - len(top), replace=False)]
    return sorted(set(top) | set(extra))


TARGET_KINDS = ("mean", "best")
TARGET_FILES = {"mean": "targets.npy", "best": "targets_best.npy"}


def _distribution(value) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64)[:NUM_PLAYERS], 0.0, None)
    return value / value.sum() if value.sum() > 0 else np.full(NUM_PLAYERS, 1.0 / NUM_PLAYERS)


def search_values(tree) -> dict:
    """A searched root's value per seat (absolute order) two ways: ``mean`` -- the mean of
    every value backed up through the root (MuZero's search value); ``best`` -- the mean
    value of the root's most visited move (the line the search would play: the root
    mover's exploration of weaker moves does not pull it down)."""
    from rl18xx.agent.alphazero.search_learning import root_value

    out = {"mean": _distribution(root_value(tree))}
    stats = tree.node_stats(tree.root_idx)
    n = np.asarray(stats["child_n"], dtype=np.float64)
    if n.sum() > 0:
        a = int(np.argmax(n))
        out["best"] = _distribution(np.asarray(stats["child_w"], dtype=np.float64)[a] / n[a])
    else:
        out["best"] = out["mean"]
    return out


def reanalysis_settings(readouts: int, c_puct: float, temperature: float) -> dict:
    """The search-learning generator's tree settings (``search_target_quality.generator_settings``)."""
    from search_target_quality import generator_settings

    return generator_settings({"readouts": readouts, "c_puct": c_puct, "qnorm": False}, temperature)


def build(args) -> None:
    import engine_rs
    import torch

    from rl18xx.agent.alphazero import search_learning as sl
    from rl18xx.agent.alphazero.composite_model import load_policy_value
    from rl18xx.agent.alphazero.config import SelfPlayConfig
    from rl18xx.agent.alphazero.lockstep import LockstepSearcher, ModelEvaluator, TreeSearch
    from rl18xx.agent.alphazero.policy_gradient import use_tf32
    from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer
    from rl18xx.rust_adapter import RustGameAdapter

    use_tf32()
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_policy_value(args.model)
    expander = net if not args.expander else load_policy_value(args.expander)
    evaluator = ModelEvaluator(net)
    expand_eval = evaluator if expander is net else ModelEvaluator(expander)
    temperature = args.temperature or sl.run_temperature(args.model.split("+")[0].split("#")[0])
    settings = reanalysis_settings(args.readouts, args.c_puct, temperature)
    gen = sl.Generator(settings, {"m": evaluator})
    searcher = LockstepSearcher({"m": evaluator, "x": expand_eval}, dedupe_maps=True)

    positions = []
    for collection in args.collection:
        positions.extend(dsv.sample_positions(Path(collection), args.games, args.per_game, rng, args.game_offset))
        LOGGER.info(f"{collection}: {len(positions)} positions so far")
    if args.limit:
        positions = positions[: args.limit]

    store = dsv.Store()
    meta, child_meta, terminal_values, targets = [], [], [], []
    t0, states0, fwd0 = time.time(), evaluator.states, evaluator.forward_seconds
    reanalysis_states, reanalysis_fwd, searched_children = 0, 0.0, 0
    chunk = max(1, args.trees // max(args.max_children, 1))
    for b in range(0, len(positions), chunk):
        part = positions[b : b + chunk]
        players, roots = [], []
        for _, _, _, rs, _, _ in part:
            config = SelfPlayConfig(
                network=expander, use_inference_server=False,
                **{**dsv.EVAL_OVERRIDES, "prior_temperature": temperature},
                softpick_move_cutoff=0, dirichlet_noise_weight=0.0, enable_resign=False, noresign_holdout_rate=0.0,
                max_game_length=1000, player_count_distribution={NUM_PLAYERS: 1.0}, auction_unlock=True,
            )
            player = RustMCTSPlayer(config)
            player.initialize_game(RustGameAdapter(rs.pickle_clone()))
            players.append(player)
            s = player.start_search(key="x")
            if s is not None:
                roots.append(s)
        with torch.no_grad():
            searcher.round(roots)  # each root's expansion (priors, price head)
        jobs = []  # (child_meta index, TreeSearch)
        for (game_id, i, n, rs, action, record), player in zip(part, players):
            tree = player._rust_player
            stats = tree.node_stats(tree.root_idx)
            legal = [int(x) for x in stats["legal"]]
            prior = np.asarray(stats["child_prior"], dtype=np.float64)[: len(legal)]
            mover = int(rs.encode_canonical_py()[3])
            first_child = len(child_meta)
            games = []
            for k in choose_children(prior, args.max_children, rng):
                idx = tree.maybe_add_child(tree.root_idx, legal[k], None)
                game = tree.get_game_for_idx(idx)
                finished = bool(game.finished)
                if finished:
                    terminal_values.append(np.asarray(engine_rs.mcts_terminal_value(game, True), dtype=np.float32))
                else:
                    terminal_values.append(np.full(6, np.nan, dtype=np.float32))
                    games.append((len(child_meta), game))
                child_meta.append({"move": legal[k], "prior": float(prior[k]), "terminal": finished, "played": False})
            for (ci, game), (row, rot) in zip(games, store.add([g for _, g in games]) if games else []):
                child_meta[ci]["rotation"] = rot
                child_meta[ci]["enc"] = row
                jobs.append((ci, TreeSearch(gen._tree(game), args.readouts, key="m", budget="total")))
            (parent_row, parent_rot), = store.add([rs])
            meta.append({
                "game": game_id, "action": i, "n_actions": n, "stage": dsv.STAGES[min(2, 3 * i // max(n, 1))],
                "step": dsv.step_type(rs, legal), "mover": mover, "side": "?", "legal": len(legal),
                "children": [first_child, len(child_meta)], "played": -1, "enc": parent_row,
                "win": [float(x) for x in record["win_share"]],
            })
        s0, f0 = evaluator.states, evaluator.forward_seconds
        with torch.no_grad():
            searcher.run([s for _, s in jobs])
        reanalysis_states += evaluator.states - s0
        reanalysis_fwd += evaluator.forward_seconds - f0
        searched_children += len(jobs)
        for ci, s in jobs:
            for kind, value in search_values(s.tree).items():
                child_meta[ci][f"target_{kind}"] = np.roll(value, -child_meta[ci]["rotation"]).tolist()
            child_meta[ci]["visits"] = float(s.tree.n_at_root())
        del jobs, players, roots
        LOGGER.info(
            f"{b + len(part)} / {len(positions)} positions, {len(child_meta)} children "
            f"({searched_children} reanalyzed), {len(store.maps)} maps, {time.time() - t0:.0f}s, "
            f"{reanalysis_states / max(searched_children, 1):.1f} evaluations per child"
        )
    arrays = store.arrays()
    for kind in TARGET_KINDS:
        target = np.full((len(arrays["game_states"]), NUM_PLAYERS), np.nan, dtype=np.float32)
        for ch in child_meta:
            if f"target_{kind}" in ch:
                target[ch["enc"]] = ch.pop(f"target_{kind}")
        np.save(out / TARGET_FILES[kind], target)
    for key, array in {"terminal_values": np.stack(terminal_values), **arrays}.items():
        np.save(out / f"{key}.npy", array)
    cost = {
        "positions": len(meta),
        "children": len(child_meta),
        "reanalyzed_children": searched_children,
        "evaluations": int(evaluator.states - states0),
        "reanalysis_evaluations": int(reanalysis_states),
        "forward_seconds": float(evaluator.forward_seconds - fwd0),
        "reanalysis_forward_seconds": float(reanalysis_fwd),
        "wall_seconds": time.time() - t0,
    }
    cost["evaluations_per_reanalyzed_child"] = reanalysis_states / max(searched_children, 1)
    cost["forward_seconds_per_reanalyzed_child"] = reanalysis_fwd / max(searched_children, 1)
    cost["wall_seconds_per_position"] = cost["wall_seconds"] / max(len(meta), 1)
    (out / "meta.json").write_text(
        json.dumps({"positions": meta, "children": child_meta, "args": vars(args), "cost": cost})
    )
    LOGGER.info(f"wrote {out}: {len(meta)} positions, {len(child_meta)} children; cost {cost}")


# ---------------------------------------------------------------------------
# Check: do the reanalyzed values rank siblings closer to a reference?
# ---------------------------------------------------------------------------


def target_mover_values(data: dict, meta: dict, kind: str = "mean") -> np.ndarray:
    """Per child, the parent mover's reanalyzed value (terminal children: their result)."""
    canon = np.nan_to_num(np.asarray(data[f"targets_{kind}"], dtype=np.float64), nan=0.25)
    return dsv.mover_values(canon, data, meta)


def check(args) -> None:
    from rl18xx.agent.alphazero.composite_model import load_policy_value

    data_dir = Path(args.data)
    data = load_build(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    values = {}
    for name, path in [tuple(args.reference.split("=", 1)), ("raw", args.model)]:
        cache = data_dir / f"values.{name}.npy"
        if cache.exists():
            canon = np.load(cache)
        else:
            model = load_policy_value(path)
            canon = dsv.child_values(model, data, model.device)
            np.save(cache, canon)
            del model
        values[name] = dsv.mover_values(canon, data, meta)
    for kind in TARGET_KINDS:
        if f"targets_{kind}" in data:
            values[f"reanalyzed_{kind}"] = target_mover_values(data, meta, kind)
    for spec in args.compare or []:
        # Another build of the same positions and children (e.g. another network's searches).
        name, _, other = spec.partition("=")
        other, _, kind = other.partition(":")
        other_data = load_build(Path(other))
        same = np.all(np.asarray(other_data["game_states"]) == np.asarray(data["game_states"]), axis=1)
        LOGGER.info(f"{name}: {int(same.sum())} / {len(same)} states encode alike")
        values[name] = target_mover_values(other_data, meta, kind or "mean")
    ref = args.reference_name or args.reference.split("=", 1)[0]
    summary = dsv.summarize(dsv.position_metrics(values, meta, ref), list(values))
    text = dsv.format_table(summary, list(values))
    print(text)
    (data_dir / f"check.{ref}.txt").write_text(text)
    (data_dir / f"check.{ref}.json").write_text(json.dumps(summary, indent=2))


def load_build(data_dir: Path, kind: str = "mean") -> dict:
    """A build set's arrays, its reanalyzed targets as ``targets_<kind>`` per kind stored
    and ``targets`` = the ``kind`` one (training)."""
    data = dsv.load_data(data_dir)
    for k in TARGET_KINDS:
        if (data_dir / TARGET_FILES[k]).exists():
            data[f"targets_{k}"] = np.load(data_dir / TARGET_FILES[k])
    if f"targets_{kind}" in data:
        data["targets"] = data[f"targets_{kind}"]
    return data


# ---------------------------------------------------------------------------
# Train: the value side on reanalyzed siblings + outcomes
# ---------------------------------------------------------------------------


VALUE_MODULES = ("value_econ_transformer", "win_loss_head", "score_head")


def value_modules(model) -> list:
    return [getattr(model, name) for name in VALUE_MODULES if getattr(model, name, None) is not None]


def trunk_features(model, data: dict, device, batch_size: int = 256) -> np.ndarray:
    """(rows, d_model) float32: the frozen trunk's output (the last FiLM block) per stored state."""
    import torch

    store = {}
    handle = model.res_blocks[-1].register_forward_hook(lambda _m, _a, out: store.__setitem__("x", out))
    out = []
    try:
        with torch.no_grad():
            for b in range(0, len(data["game_states"]), batch_size):
                gs, nodes, rt, npl = _inputs(data, slice(b, b + batch_size), device)
                model(gs, nodes, rt, None, npl, value_only=True)
                out.append(store["x"].float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(out)


def _inputs(data: dict, rows, device) -> tuple:
    import torch

    gs = torch.from_numpy(np.array(data["game_states"][rows], dtype=np.float32)).to(device)
    nodes = torch.from_numpy(np.asarray(data["maps"][np.asarray(data["map_index"][rows])])).float().to(device)
    rt = torch.from_numpy(np.asarray(data["round_types"][rows])).long().to(device)
    npl = torch.full((len(gs),), NUM_PLAYERS, dtype=torch.long, device=device)
    return gs, nodes, rt, npl


def map_embeddings(model, data: dict, device, batch_size: int = 128):
    """(maps, hexes, d_map) float16 on ``device``: the frozen map encoder's per-hex
    embeddings of each distinct stored map (the forward's step 2)."""
    import torch

    out = []
    with torch.no_grad():
        for b in range(0, len(data["maps"]), batch_size):
            nodes = torch.from_numpy(np.array(data["maps"][b : b + batch_size], dtype=np.float32)).to(device)
            track = model.track_conn_computer(nodes, model.direction_matrix)
            emb, _ = model.hex_transformer(nodes, model.axial_coords, model.distance_matrix, model.direction_matrix, track)
            out.append(emb.half())
    return torch.cat(out)


def make_map_readout(d_entity: int, d_map: int, d_trunk: int, heads: int = 4, d_head: int = 32):
    """A value-only read-out of the map (a probe for a value map pathway): each seat's
    value player token attends over the map encoder's per-hex embeddings; the result,
    projected to the trunk's width by a zero-initialized layer (the starting network's
    values unchanged), is added to the trunk features the value heads read."""
    import torch

    class MapReadout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heads, self.d_head = heads, d_head
            self.q = torch.nn.Linear(d_entity, heads * d_head)
            self.k = torch.nn.Linear(d_map, heads * d_head)
            self.v = torch.nn.Linear(d_map, heads * d_head)
            self.out = torch.nn.Linear(heads * d_head, d_trunk)
            torch.nn.init.zeros_(self.out.weight)
            torch.nn.init.zeros_(self.out.bias)

        def forward(self, tokens, nodes):
            b, p, _ = tokens.shape
            n = nodes.shape[1]
            q = self.q(tokens).view(b, p, self.heads, self.d_head).transpose(1, 2)  # (B, H, P, dh)
            k = self.k(nodes).view(b, n, self.heads, self.d_head).transpose(1, 2)  # (B, H, N, dh)
            v = self.v(nodes).view(b, n, self.heads, self.d_head).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-1, -2) / self.d_head**0.5, dim=-1)  # (B, H, P, N)
            o = (att @ v).transpose(1, 2).reshape(b, p, self.heads * self.d_head)
            return self.out(o)

    return MapReadout()


def value_from_trunk(model, game_state, trunk, num_players, readout=None, nodes=None) -> tuple:
    """``(win_logits, score)`` of the value heads from cached trunk features: the
    forward's value section (``AlphaZeroTransformerModel.forward`` step 6); with a
    ``readout`` (:func:`make_map_readout`), its map read-out of ``nodes`` (the map
    encoder's per-hex embeddings) joins each seat's trunk features."""
    import torch

    from rl18xx.agent.alphazero.model_transformer import MAX_PLAYERS, _pad_state_to_max_players

    if game_state.shape[1] != model.config.game_state_size:
        inferred = model._infer_num_players_from_state_size(game_state.shape[1])
        game_state = _pad_state_to_max_players(game_state, inferred, MAX_PLAYERS)
    econ = model.value_econ_transformer if model.value_econ_transformer is not None else model.econ_transformer
    _, entity_embeds, _ = econ(game_state, num_players=num_players)
    tokens = entity_embeds[:, econ.player_token_start : econ.player_token_end]
    seat_trunk = trunk.unsqueeze(1).expand(-1, tokens.shape[1], -1)
    if readout is not None:
        seat_trunk = seat_trunk + readout(tokens, nodes.float())
    seat_input = torch.cat([tokens, seat_trunk], dim=-1)
    win = model.win_loss_head(seat_input).squeeze(-1)
    score = model.score_head(seat_input).squeeze(-1)
    seat_idx = torch.arange(win.shape[1], device=win.device)
    padded = seat_idx.unsqueeze(0) >= num_players.unsqueeze(1)
    return win.masked_fill(padded, -1e4), score.masked_fill(padded, 0.0)


def rank_loss(groups: list, forward, target_all, temperature: float, device):
    """ListNet over each position's children: KL(softmax(t / tau) || softmax(v / tau)) of
    the parent mover's target values t and predicted values v (win probabilities), the
    mean over the groups -- the siblings' order and gaps, not the position's level."""
    import torch

    rows = np.concatenate([g[0] for g in groups])
    slots = torch.as_tensor(np.concatenate([g[1] for g in groups]), device=device, dtype=torch.long)
    log_v, _, ids_t = forward(rows)
    pred = log_v.exp().gather(1, slots.unsqueeze(1)).squeeze(1)
    target = target_all[ids_t].gather(1, slots.unsqueeze(1)).squeeze(1)
    total, start = torch.zeros((), device=device), 0
    for g in groups:
        n = len(g[0])
        tau = temperature if temperature > 0 else float(target[start : start + n].std().clamp_min(1e-3))
        t = torch.log_softmax(target[start : start + n] / tau, dim=0)
        v = torch.log_softmax(pred[start : start + n] / tau, dim=0)
        total = total + (t.exp() * (t - v)).sum()
        start += n
    return total / len(groups)


def rank_eval(groups: list, forward, target_all, temperature: float, device) -> float:
    import torch

    with torch.no_grad():
        total = sum(
            float(rank_loss(groups[b : b + 64], forward, target_all, temperature, device)) * len(groups[b : b + 64])
            for b in range(0, len(groups), 64)
        )
        return float(total) / max(len(groups), 1)


class TrainSet:
    """Every encoded state of the build sets: children with reanalyzed targets,
    positions with their games' outcomes; held-out games by a hash of the game id."""

    def __init__(
        self, dirs: list, model, device, val_fraction: float, kind: str = "mean", teacher: str = None,
        exclude: frozenset = frozenset(), with_maps: bool = False,
    ):
        """``teacher``: the children's targets are a network's own values instead
        (``values.<teacher>.npy`` in each build dir, as ``check`` caches them) -- the
        distillation baseline. ``exclude``: game ids whose states are left out (a test
        set's games)."""
        self.parts = []
        for d in dirs:
            d = Path(d)
            data = load_build(d, kind)
            meta = json.loads((d / "meta.json").read_text())
            n = len(data["game_states"])
            game = np.empty(n, dtype=object)
            outcome = np.full((n, NUM_PLAYERS), np.nan, dtype=np.float32)
            parent = np.zeros(n, dtype=bool)
            for p in meta["positions"]:
                game[p["enc"]] = p["game"]
                parent[p["enc"]] = True
                outcome[p["enc"]] = np.roll(np.asarray(p["win"], dtype=np.float32), -int(data["rotations"][p["enc"]]))
                for c in range(*p["children"]):
                    ch = meta["children"][c]
                    if not ch["terminal"]:
                        game[ch["enc"]] = p["game"]
            target = np.asarray(data["targets"], dtype=np.float32)
            if teacher:
                values = np.load(d / f"values.{teacher}.npy").astype(np.float32)
                target = np.where(np.isnan(target).any(axis=1, keepdims=True), np.nan, values[:, :NUM_PLAYERS])
            trunk = trunk_features(model, data, device)
            val = np.array([(zlib.crc32(str(g).encode()) % 10_000) < val_fraction * 10_000 for g in game])
            dropped = np.array([g in exclude for g in game])
            target[dropped] = np.nan
            outcome[dropped] = np.nan
            parent = parent & ~dropped
            self.parts.append({
                "dir": str(d), "data": data, "meta": meta, "trunk": trunk, "game": game, "outcome": outcome,
                "parent": parent, "target": target, "val": val,
            })
        self.gs = np.concatenate([np.asarray(p["data"]["game_states"]) for p in self.parts]).astype(np.float32)
        self.trunk = np.concatenate([p["trunk"] for p in self.parts])
        self.target = np.concatenate([p["target"] for p in self.parts])
        self.outcome = np.concatenate([p["outcome"] for p in self.parts])
        self.val = np.concatenate([p["val"] for p in self.parts])
        self.parent = np.concatenate([p["parent"] for p in self.parts])
        offsets = np.cumsum([0] + [len(p["data"]["maps"]) for p in self.parts])
        self.row_map = np.concatenate(
            [np.asarray(p["data"]["map_index"]) + off for p, off in zip(self.parts, offsets[:-1])]
        )
        self.map_emb = None
        if with_maps:
            import torch

            self.map_emb = torch.cat([map_embeddings(model, p["data"], device) for p in self.parts])
        sib = ~np.isnan(self.target).any(axis=1)
        self.sib_train = np.flatnonzero(sib & ~self.val)
        self.sib_val = np.flatnonzero(sib & self.val)
        self.par_train = np.flatnonzero(self.parent & ~self.val)
        self.par_val = np.flatnonzero(self.parent & self.val)
        # Each position's reanalyzed children (global rows) and, per child, the parent's
        # mover's slot in the child's canonical frame: the sibling-ranking loss's groups.
        self.groups_train, self.groups_val = [], []
        offset = 0
        for part in self.parts:
            rotations = np.asarray(part["data"]["rotations"])
            for pos in part["meta"]["positions"]:
                rows, slots = [], []
                for c in range(*pos["children"]):
                    ch = part["meta"]["children"][c]
                    if ch["terminal"] or np.isnan(part["target"][ch["enc"]]).any():
                        continue
                    rows.append(offset + ch["enc"])
                    slots.append((int(pos["mover"]) - int(rotations[ch["enc"]])) % NUM_PLAYERS)
                if len(rows) >= 2:
                    group = (np.asarray(rows), np.asarray(slots))
                    (self.groups_val if part["val"][pos["enc"]] else self.groups_train).append(group)
            offset += len(part["target"])


def train(args) -> None:
    import torch

    from rl18xx.agent.alphazero import value_refit as vr
    from rl18xx.agent.alphazero.policy_gradient import _load, use_tf32

    use_tf32()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tuned = _load(args.init).eval()
    device = next(tuned.parameters()).device
    for p in tuned.parameters():
        p.requires_grad_(False)
    params = [p for m in value_modules(tuned) for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    exclude = set()
    for spec in args.test_set or []:
        exclude |= {p["game"] for p in json.loads((Path(spec.split("=", 1)[1].split(":", 1)[0]) / "meta.json")
                                                  .read_text())["positions"]}
    rows = TrainSet(
        args.data, tuned, device, args.val_fraction, args.target_kind, args.teacher, frozenset(exclude),
        with_maps=bool(getattr(args, "map_readout", False)),
    )
    readout = None
    if getattr(args, "map_readout", False):
        c = tuned.config
        readout = make_map_readout(c.d_entity, c.d_map, c.d_trunk).to(device)
        params += list(readout.parameters())
    row_map_all = torch.as_tensor(rows.row_map, device=device, dtype=torch.long)
    LOGGER.info(
        f"{len(rows.sib_train)} sibling rows + {len(rows.par_train)} positions to train on; held out "
        f"{len(rows.sib_val)} siblings, {len(rows.par_val)} positions"
    )
    gs_all = torch.from_numpy(rows.gs).to(device)
    trunk_all = torch.from_numpy(rows.trunk).to(device)
    target_all = torch.from_numpy(np.nan_to_num(rows.target, nan=0.25)).to(device)
    outcome_all = torch.from_numpy(np.nan_to_num(rows.outcome, nan=0.25)).to(device)
    with torch.no_grad():
        score0 = torch.cat([
            value_from_trunk(tuned, gs_all[b : b + 1024], trunk_all[b : b + 1024],
                             torch.full((len(gs_all[b : b + 1024]),), NUM_PLAYERS, device=device))[1][:, :NUM_PLAYERS]
            for b in range(0, len(gs_all), 1024)
        ])

    tests = {}
    for spec in args.test_set or []:
        # <name>=<dir>:values.<net>  (a network's cached values on the set's children: e.g. az1's critic)
        # <name>=<dir>:build=<other build dir>[:<kind>]  (another build's reanalyzed values: a searched reference)
        name, _, rest = spec.partition("=")
        tdir, _, ref = rest.partition(":")
        tdir = Path(tdir)
        tdata = load_build(tdir)
        tmeta = json.loads((tdir / "meta.json").read_text())
        if ref.startswith("values."):
            ref_values = dsv.mover_values(np.load(tdir / f"{ref}.npy"), tdata, tmeta)
        elif ref.startswith("build="):
            other, _, kind = ref[len("build="):].partition(":")
            ref_values = target_mover_values(load_build(Path(other)), tmeta, kind or "mean")
        else:
            raise SystemExit(f"--test-set {spec!r}: the reference must be values.<name> or build=<dir>[:kind]")
        tmaps = None
        if readout is not None:
            tmaps = (map_embeddings(tuned, tdata, device),
                     torch.as_tensor(np.asarray(tdata["map_index"]), device=device, dtype=torch.long))
        tests[name] = (tdata, tmeta, ref_values, trunk_features(tuned, tdata, device), tmaps)

    def forward(ids):
        ids_t = torch.as_tensor(ids, device=device, dtype=torch.long)
        npl = torch.full((len(ids),), NUM_PLAYERS, dtype=torch.long, device=device)
        nodes = rows.map_emb[row_map_all[ids_t]] if readout is not None else None
        win, score = value_from_trunk(tuned, gs_all[ids_t], trunk_all[ids_t], npl, readout, nodes)
        return torch.log_softmax(win.float()[:, :NUM_PLAYERS], dim=1), score.float()[:, :NUM_PLAYERS], ids_t

    def evaluate() -> dict:
        tuned.eval()
        out = {}
        with torch.no_grad():
            for name, ids, tgt in (("val_sib", rows.sib_val, target_all), ("val_pos", rows.par_val, outcome_all),
                                   ("train_sib", rows.sib_train[:8192], target_all)):
                if not len(ids):
                    continue
                ce = 0.0
                for b in range(0, len(ids), 1024):
                    log_v, _, ids_t = forward(ids[b : b + 1024])
                    ce += float(-(tgt[ids_t] * log_v).sum())
                out[f"{name}_ce"] = ce / len(ids)
            if rows.groups_val:
                out["val_rank_kl"] = rank_eval(rows.groups_val, forward, target_all, args.rank_temperature, device)
            for test_name, (tdata, tmeta, ref_values, ttrunk, tmaps) in tests.items():
                canon = []
                tg = torch.from_numpy(np.asarray(tdata["game_states"])).float().to(device)
                tt = torch.from_numpy(ttrunk).to(device)
                for b in range(0, len(tg), 1024):
                    npl = torch.full((len(tg[b : b + 1024]),), NUM_PLAYERS, dtype=torch.long, device=device)
                    nodes = tmaps[0][tmaps[1][b : b + 1024]] if tmaps is not None else None
                    win, _ = value_from_trunk(tuned, tg[b : b + 1024], tt[b : b + 1024], npl, readout, nodes)
                    canon.append(torch.softmax(win.float()[:, :NUM_PLAYERS], dim=1).cpu().numpy())
                canon = np.concatenate(canon)
                values = {"ref": ref_values, "tuned": dsv.mover_values(canon, tdata, tmeta)}
                summary = dsv.summarize(dsv.position_metrics(values, tmeta, "ref"), list(values))
                for g in ("all", "step:stock", "step:track", "step:train", "step:dividend", "step:token"):
                    if g in summary:
                        for key in ("spearman", "top1", "regret"):
                            out[f"{test_name}_{g}_{key}"] = summary[g]["tuned"][key]
                parents = np.array([p["enc"] for p in tmeta["positions"]])
                rot = np.asarray(tdata["rotations"])[parents]
                win_t = np.stack([np.roll(np.asarray(p["win"]), -r) for p, r in zip(tmeta["positions"], rot)])
                pr = np.clip(canon[parents], 1e-9, 1)
                out[f"{test_name}_outcome_ce"] = float(-(win_t * np.log(pr / pr.sum(1, keepdims=True))).sum(1).mean())
        return out

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    history = [{"step": 0, **evaluate()}]
    LOGGER.info(f"step 0: {history[-1]}")
    best = (history[0].get("val_sib_ce", 0.0) + history[0].get("val_pos_ce", 0.0), copy.deepcopy(tuned.state_dict()), 0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        frac = min(1.0, step / max(args.warmup, 1)) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = args.lr * frac
        loss = torch.zeros((), device=device)
        stats = {}
        if args.sibling_weight > 0 and len(rows.sib_train):
            ids = rng.choice(rows.sib_train, size=min(args.batch_size, len(rows.sib_train)), replace=False)
            log_v, score, ids_t = forward(ids)
            ce = -(target_all[ids_t] * log_v).sum(1).mean()
            anchor = ((score - score0[ids_t]) ** 2).mean()
            loss = loss + args.sibling_weight * ce + args.score_anchor * anchor
            stats.update(sib_ce=float(ce.detach()), sib_anchor=float(anchor.detach()))
        if args.rank_weight > 0 and rows.groups_train:
            k = min(args.rank_groups, len(rows.groups_train))
            picks = rng.choice(len(rows.groups_train), size=k, replace=False)
            rank = rank_loss([rows.groups_train[k] for k in picks], forward, target_all, args.rank_temperature, device)
            loss = loss + args.rank_weight * rank
            stats.update(rank=float(rank.detach()))
        if args.outcome_weight > 0 and len(rows.par_train):
            ids = rng.choice(rows.par_train, size=min(args.outcome_batch, len(rows.par_train)), replace=False)
            log_v, score, ids_t = forward(ids)
            ce = -(outcome_all[ids_t] * log_v).sum(1).mean()
            anchor = ((score - score0[ids_t]) ** 2).mean()
            loss = loss + args.outcome_weight * ce + args.score_anchor * anchor
            stats.update(pos_ce=float(ce.detach()))
        if not torch.isfinite(loss):
            raise RuntimeError("reanalyze value training diverged (non-finite loss)")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        if step % 100 == 0:
            LOGGER.info(f"step {step}/{args.steps}: {stats} ({time.time() - t0:.0f}s)")
        if step % args.eval_every == 0 or step == args.steps:
            history.append({"step": step, **evaluate()})
            LOGGER.info(f"step {step}: {history[-1]}")
            crit = history[-1].get("val_sib_ce", 0.0) * (args.sibling_weight > 0) + history[-1].get("val_pos_ce", 0.0)
            if crit < best[0]:
                best = (crit, copy.deepcopy(tuned.state_dict()), step)
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    if args.out:
        if not args.keep_last:
            tuned.load_state_dict(best[1])
        provenance = {"reanalyze": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                    "best_step": best[2] if not args.keep_last else args.steps, "history": history}}
        vr.write_model_checkpoint(tuned, args.init, args.out, provenance)
        LOGGER.info(f"saved {args.out} (step {best[2] if not args.keep_last else args.steps})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--collection", action="append", required=True)
    b.add_argument("--games", type=int, default=40, help="games per collection")
    b.add_argument("--per-game", type=int, default=8)
    b.add_argument("--game-offset", type=int, default=0, help="skip each collection's first games")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--max-children", type=int, default=16)
    b.add_argument("--model", required=True,
                   help="the network that reanalyzes (a shared checkpoint or <policy>+<value>)")
    b.add_argument("--expander", default=None, help="the network that expands the positions (default --model)")
    b.add_argument("--readouts", type=int, default=64, help="readouts per child search")
    b.add_argument("--c-puct", type=float, default=0.05)
    b.add_argument("--temperature", type=float, default=0.0, help="priors' temperature (0: the checkpoint run's)")
    b.add_argument("--trees", type=int, default=96, help="child searches in lockstep")
    b.add_argument("--torch-threads", type=int, default=2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--data", required=True)
    c.add_argument("--model", required=True, help="the network whose raw values to compare")
    c.add_argument("--reference", required=True, help="name=<critic checkpoint>")
    c.add_argument("--compare", nargs="*", default=None,
                   help="name=<build dir>[:mean|best]: another build's reanalyzed values on the same children")
    c.add_argument("--reference-name", default=None, help="rank against this source instead (e.g. a --compare one)")
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--init", required=True)
    t.add_argument("--test-set", action="append", default=None,
                   help="<name>=<build dir>:values.<net> (cached raw values of a reference network) or "
                   "<name>=<build dir>:build=<other build>[:mean|best] (another network's reanalyzed values); its "
                   "games are kept out of training")
    t.add_argument("--out", default=None)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--outcome-batch", type=int, default=64)
    t.add_argument("--target-kind", choices=TARGET_KINDS, default="mean",
                   help="the children's targets: the reanalysis root's mean value or its most visited move's")
    t.add_argument("--teacher", default=None,
                   help="train the children toward a network's cached values (values.<name>.npy) instead: the "
                   "distillation baseline")
    t.add_argument("--sibling-weight", type=float, default=1.0)
    t.add_argument("--rank-weight", type=float, default=0.0,
                   help="ListNet sibling-ranking loss over each position's children (the parent mover's values)")
    t.add_argument("--rank-temperature", type=float, default=0.01,
                   help="its softmax temperature (win share); 0: each position's target spread (std)")
    t.add_argument("--rank-groups", type=int, default=32, help="positions per step for the ranking loss")
    t.add_argument("--outcome-weight", type=float, default=1.0)
    t.add_argument("--score-anchor", type=float, default=1.0)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--keep-last", action="store_true", help="save the last step, not the best on held-out CE")
    t.add_argument("--map-readout", action="store_true",
                   help="probe: add a value-only read-out of the map encoder's per-hex embeddings (zero-initialized, "
                   "trained with the value side; not saved in the checkpoint)")
    t.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("rl18xx.agent.alphazero.rust_mcts_player").setLevel(logging.WARNING)
    {"build": build, "check": check, "train": train}[args.cmd](args)


if __name__ == "__main__":
    main()
