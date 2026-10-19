"""``scripts/reanalyze_siblings.py``: reanalyzed value targets on sibling positions -- the
search values it reads from a tree, the cached-trunk value path its training uses (equal to
the network's own value forward), and a fine-tune that trains the value side only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import search_learning as sl  # noqa: E402

from .search_test_utils import HashNet, expand_root, make_player, random_positions  # noqa: E402
from .test_search_learning import _tiny_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
PUCT = dict(mcts_mean_q=True, c_puct_init=0.05, c_puct_by_round={}, dedupe_leaves=True, prior_temperature=0.5)


def _module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("reanalyze_siblings", REPO_ROOT / "scripts" / "reanalyze_siblings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value_encoder_model(seed: int):
    """A tiny shared network with a value-only entity encoder (as az6's)."""
    from rl18xx.agent.alphazero.config import ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel

    torch.manual_seed(seed)
    config = ModelTransformerConfig(
        device=torch.device("cpu"), d_entity=32, econ_transformer_layers=1, econ_transformer_heads=2,
        econ_transformer_ff_dim=64, d_map=32, hex_transformer_layers=1, hex_transformer_heads=2,
        hex_transformer_ff_dim=64, cross_attn_heads=2, d_trunk=64, num_res_blocks=1, value_head_layers=1,
        value_entity_encoder=True,
    )
    model = AlphaZeroTransformerModel(config).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():  # the heads' output layers start at zero: give them values to compare
        for head in (model.win_loss_head, model.score_head):
            head[-1].weight.copy_(torch.randn(head[-1].weight.shape, generator=g) * 0.5)
    return model


def _store(games):
    rs = _module()
    store = rs.dsv.Store()
    rows = store.add(games)
    return store, rows


def test_choose_children_takes_all_or_the_top_half_and_random_others():
    rs = _module()
    rng = np.random.default_rng(0)
    prior = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    assert rs.choose_children(prior, 8, rng) == list(range(6))
    for _ in range(20):
        kept = rs.choose_children(prior, 4, rng)
        assert len(kept) == 4 and {0, 1} <= set(kept) and kept == sorted(kept)


def test_search_values_are_the_roots_mean_and_its_most_visited_moves_value():
    rs = _module()
    net = HashNet()
    games = [g for g in random_positions(4, 6, steps=300, every=19) if len(g.factored_legal_indices()) > 2][:4]
    for seed, game in enumerate(games):
        player = make_player(net, game, seed=seed, parallel_readouts=8, num_readouts=48, min_readouts=48,
                             softpick_move_cutoff=0, **PUCT)
        expand_root(player)
        player.suggest_move()
        tree = player._rust_player
        values = rs.search_values(tree)
        mean = sl.root_value(tree)[:4].astype(np.float64)
        np.testing.assert_allclose(values["mean"], mean / mean.sum(), rtol=1e-6)
        stats = tree.node_stats(tree.root_idx)
        n = np.asarray(stats["child_n"], dtype=np.float64)
        a = int(np.argmax(n))
        best = np.asarray(stats["child_w"], dtype=np.float64)[a, :4] / n[a]
        np.testing.assert_allclose(values["best"], best / best.sum(), rtol=1e-6)
        assert int(tree.legal_action_indices_at_root()[a]) == int(tree.pick_best_action())


def test_the_cached_trunk_value_path_is_the_networks_value():
    rs = _module()
    model = _value_encoder_model(0)
    games = random_positions(4, 7, steps=200, every=23)[:6]
    store, _ = _store(games)
    data = store.arrays()
    device = torch.device("cpu")
    trunk = rs.trunk_features(model, data, device, batch_size=4)
    gs, nodes, rt, npl = rs._inputs(data, slice(0, len(games)), device)
    with torch.no_grad():
        _, win, score, _ = model(gs, nodes, rt, None, npl, value_only=True)
        win2, score2 = rs.value_from_trunk(model, gs, torch.from_numpy(trunk), npl)
    torch.testing.assert_close(win2[:, :4], win[:, :4], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(score2[:, :4], score[:, :4], rtol=1e-5, atol=1e-5)
    # The value modules are the value encoder and the heads, none of the shared layers.
    names = {name for name in rs.VALUE_MODULES if getattr(model, name, None) is not None}
    assert names == {"value_econ_transformer", "win_loss_head", "score_head"}
    # The map read-out probe starts as a no-op (zero-initialized output) and reads each state's map.
    c = model.config
    readout = rs.make_map_readout(c.d_entity, c.d_map, c.d_trunk)
    maps = rs.map_embeddings(model, data, device)
    nodes = maps[torch.as_tensor(np.asarray(data["map_index"]), dtype=torch.long)]
    assert maps.shape == (len(data["maps"]), 93, c.d_map)
    with torch.no_grad():
        win3, _ = rs.value_from_trunk(model, gs, torch.from_numpy(trunk), npl, readout, nodes)
        torch.testing.assert_close(win3, win2)
        torch.nn.init.normal_(readout.out.weight, std=0.1)
        win4, _ = rs.value_from_trunk(model, gs, torch.from_numpy(trunk), npl, readout, nodes)
    assert not torch.allclose(win4[:, :4], win2[:, :4])


def test_training_moves_the_value_toward_the_reanalyzed_targets_and_leaves_the_policy(tmp_path):
    from rl18xx.agent.alphazero.policy_gradient import _load, _save

    rs = _module()
    model = _value_encoder_model(1)
    init = tmp_path / "start" / "learner" / "1.pth"
    _save(model, init)
    (init.parent.parent / "config.json").write_text(json.dumps({"learner_temperature": 0.5}))
    # A build set by hand: 4 positions of 3 children each, targets favouring each position's first child.
    games = random_positions(4, 8, steps=320, every=9)
    games = [g for g in games if len(g.factored_legal_indices()) >= 3][:4]
    store = rs.dsv.Store()
    meta = {"positions": [], "children": []}
    targets = []
    rng = np.random.default_rng(0)
    for k, game in enumerate(games):
        legal = [int(x) for x in game.factored_legal_indices()][:3]
        kids = []
        for ix in legal:
            child = game.pickle_clone()
            child.apply_action_index(ix, None if child.price_range_for_index(ix) is None
                                     else int(child.price_range_for_index(ix)[0]))
            kids.append(child)
        first = len(meta["children"])
        for (row, rot), child in zip(store.add(kids), kids):
            meta["children"].append({"move": 0, "prior": 0.3, "terminal": False, "played": False,
                                     "rotation": rot, "enc": row})
            targets.append((row, rng.dirichlet(np.ones(4))))
        (parent_row, _), = store.add([game])
        meta["positions"].append({"game": f"g{k}", "action": k, "n_actions": 100, "stage": "mid", "step": "stock",
                                  "mover": 0, "side": "?", "legal": 3, "children": [first, len(meta["children"])],
                                  "played": -1, "enc": parent_row, "win": [0.25, 0.25, 0.25, 0.25]})
    arrays = store.arrays()
    target = np.full((len(arrays["game_states"]), 4), np.nan, dtype=np.float32)
    for row, t in targets:
        target[row] = t
    build = tmp_path / "build"
    build.mkdir()
    for key, array in {"terminal_values": np.full((len(meta["children"]), 6), np.nan, np.float32), **arrays}.items():
        np.save(build / f"{key}.npy", array)
    np.save(build / "targets.npy", target)
    (build / "meta.json").write_text(json.dumps(meta))

    out = tmp_path / "tuned" / "learner" / "1.pth"
    args = type("A", (), dict(
        data=[str(build)], init=str(init), test_set=None, out=str(out), lr=3e-3, steps=60, warmup=1,
        batch_size=8, outcome_batch=2, sibling_weight=1.0, outcome_weight=0.0, score_anchor=1.0, eval_every=30,
        val_fraction=0.0, keep_last=True, seed=0, target_kind="mean", teacher=None, rank_weight=0.5,
        rank_temperature=0.05, rank_groups=4,
    ))()
    rs.train(args)
    history = json.loads(out.with_suffix(".history.json").read_text())
    assert history[-1]["train_sib_ce"] < history[0]["train_sib_ce"] - 0.01
    tuned, start = _load(out).eval(), _load(init).eval()
    for name, p in start.named_parameters():
        q = dict(tuned.named_parameters())[name]
        if name.split(".")[0] in rs.VALUE_MODULES:
            continue
        assert torch.equal(p, q), name  # the shared layers and the policy are untouched
    assert any(not torch.equal(p, dict(tuned.named_parameters())[n]) for n, p in start.named_parameters()
               if n.startswith("value_econ_transformer"))


def test_the_ranking_loss_sees_the_siblings_order_not_the_positions_level():
    rs = _module()
    # Two positions of three children; the parent's mover sits in slot 1 of every child's frame.
    target = torch.tensor([[0.2, 0.30, 0.25, 0.25], [0.2, 0.32, 0.24, 0.24], [0.2, 0.28, 0.26, 0.26],
                           [0.1, 0.50, 0.20, 0.20], [0.1, 0.40, 0.25, 0.25], [0.1, 0.45, 0.25, 0.20]])
    groups = [(np.array([0, 1, 2]), np.array([1, 1, 1])), (np.array([3, 4, 5]), np.array([1, 1, 1]))]

    def forward_of(values):
        def forward(rows):
            ids = torch.as_tensor(rows, dtype=torch.long)
            return torch.log(values[ids]), None, ids

        return forward

    cpu = torch.device("cpu")
    assert float(rs.rank_loss(groups, forward_of(target), target, 0.01, cpu)) == pytest.approx(0.0, abs=1e-6)
    shifted = target.clone()
    shifted[:, 1] += 0.1  # every child's value up by about the same: the same order and gaps
    shifted = shifted / shifted.sum(1, keepdim=True)
    level = rs.rank_loss(groups, forward_of(shifted), target, 0.01, cpu)
    swapped = target[[1, 0, 2, 4, 3, 5]]  # each position's first two children swapped
    order = rs.rank_loss(groups, forward_of(swapped), target, 0.01, cpu)
    assert float(order) > 10 * float(level)
