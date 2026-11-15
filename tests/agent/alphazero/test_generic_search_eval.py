"""Generic systems searched in lockstep (lockstep.GenericEvaluator, engine_rs.mcts_collect_leaf_games) and in
``scripts/eval_head_to_head.py`` (``generic:<learner>+generic:<critic>`` players), with tiny untrained CPU models
and in-process "servers" (generic_inference.LocalGenericClient)."""

import importlib.util
import json
import queue
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero import price_pmf  # noqa: E402
from rl18xx.agent.alphazero.generic_inference import LocalGenericClient, LocalGenericServer  # noqa: E402
from rl18xx.agent.alphazero.lockstep import (  # noqa: E402
    EvalResult,
    GenericEvaluator,
    LockstepSearcher,
    ModelEvaluator,
    PolicyRequest,
    to_tree_frames,
)
from rl18xx.agent.alphazero.mcts import _rust_encode, unrotate_value  # noqa: E402
from rl18xx.rust_adapter import RustGameAdapter  # noqa: E402

from .search_test_utils import HashNet, make_player, random_positions, tree_summary  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip
VALUE_DRIVEN = dict(
    parallel_readouts=8, num_readouts=24, min_readouts=24, mcts_mean_q=True, c_puct_init=0.15, c_puct_by_round={},
    dedupe_leaves=True, prior_temperature=0.5,
)  # fmt: skip


@pytest.fixture(scope="module")
def h2h():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("_generic_eval_head_to_head", SCRIPTS / "eval_head_to_head.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generic_run(tmp_path_factory) -> Path:
    """A generic policy-gradient run's layout with tiny untrained checkpoints: learner/1.pth, critic/1.pth."""
    run = tmp_path_factory.mktemp("generic_run")
    (run / "config.json").write_text(json.dumps({"learner_temperature": 0.5, "reward": "placement"}))
    for seed, name in ((1, "learner"), (2, "critic")):
        torch.manual_seed(seed)
        model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)).eval()
        with torch.no_grad():  # non-trivial heads (zero-initialized outputs give equal odds / uniform cells)
            for p in model.parameters():
                if p.abs().sum() == 0:
                    p.normal_(0, 0.1)
        mg.save_checkpoint(model, run / name / "1.pth", titles=["1830"])
    return run


def local_client(path) -> LocalGenericClient:
    model, _ = mg.load_checkpoint(str(path), "cpu")
    return LocalGenericClient(LocalGenericServer(model.eval()))


def generic_evaluator(run: Path, value_head: str = "win", critic: bool = True) -> GenericEvaluator:
    value = local_client(run / "critic" / "1.pth") if critic else None
    return GenericEvaluator(local_client(run / "learner" / "1.pth"), value, value_head)


# ------------------------------------------------------------------ the lockstep games path
class HashGames:
    """``HashNet`` as a games evaluator: the same outputs from each position's legacy encoding, the values
    handed back in absolute order (rotation 0), so the searcher has to move them into the trees' frames."""

    takes_games = True

    def __init__(self, net: HashNet):
        self.net = net

    def send_games(self, games, legal):
        priors, values, prices = [], [], []
        for game, ix in zip(games, legal):
            encoded = _rust_encode(RustGameAdapter(game))
            probs, value, price = self.net.outputs(encoded)
            priors.append(probs[np.asarray(ix, dtype=np.int64)])
            values.append(unrotate_value(value, int(encoded[6]), int(encoded[7])))
            prices.append(price)
        return EvalResult(
            priors=np.concatenate(priors).astype(np.float32),
            values=np.stack(values).astype(np.float32),
            price_logits=np.stack(prices).astype(np.float32),
            slot_index=dict(price_pmf.SLOT_INDEX),
            rotations=np.zeros(len(games), dtype=np.int64),
        )

    def receive(self, pending):
        return pending


def test_the_games_path_searches_exactly_as_the_legacy_path():
    """The same network through mcts_collect_leaf_games (leaves as log-free games, values re-framed) and
    through mcts_collect_leaves (native legacy encodings) builds the same trees, bit for bit; policy requests
    get the same priors and price rows."""
    net = HashNet()
    positions = random_positions(4, 0, steps=260, every=13) + random_positions(3, 1, steps=200, every=25)
    legacy = [make_player(net, g, seed=k, **VALUE_DRIVEN) for k, g in enumerate(positions)]
    games = [make_player(net, g, seed=k, **VALUE_DRIVEN) for k, g in enumerate(positions)]
    requests = [PolicyRequest(g, [int(i) for i in g.factored_legal_indices()]) for g in positions[:6]]
    games_requests = [PolicyRequest(g, list(r.legal)) for g, r in zip(positions[:6], requests)]
    LockstepSearcher({None: ModelEvaluator(net)}).run([p.start_search() for p in legacy], requests)
    LockstepSearcher({None: HashGames(net)}).run([p.start_search() for p in games], games_requests)
    for a, b in zip(legacy, games):
        sa, sb = tree_summary(a), tree_summary(b)
        assert sa["n"] == sb["n"] and sa["arena"] == sb["arena"] and sa["prices"] == sb["prices"]
        assert np.array_equal(sa["child_n"], sb["child_n"]) and np.array_equal(sa["q"], sb["q"])
    for a, b in zip(requests, games_requests):
        assert np.array_equal(a.priors, b.priors) and np.array_equal(a.price_row, b.price_row)


def test_values_move_into_the_trees_frames():
    values = np.array([[0.1, 0.2, 0.3, 0.4, 0, 0], [0.5, 0.2, 0.2, 0.1, 0, 0]], dtype=np.float32)
    # Row 0: seat 0 = player 0 (absolute); the tree's frame has player 2 first. Row 1: already the tree's.
    out = to_tree_frames(values, np.array([0, 1]), np.array([2, 1]), 4)
    assert np.allclose(out[0, :4], [0.3, 0.4, 0.1, 0.2]) and np.allclose(out[1], values[1])
    assert np.allclose(unrotate_value(out[0], 2, 4), values[0])


def test_the_generic_evaluator_answers_as_the_game_network(generic_run):
    """A generic learner + critic through the (local) inference server path gives the in-process
    game_network.GameNetwork's priors, price rows and values (here the score head)."""
    from rl18xx.agent.alphazero.game_network import GameNetwork, GenericNetwork

    positions = random_positions(4, 3, steps=240, every=11)
    legal = [[int(i) for i in g.factored_legal_indices()] for g in positions]
    evaluator = generic_evaluator(generic_run, value_head="score")
    result = evaluator.receive(evaluator.send_games(positions, legal))
    network = GameNetwork(
        GenericNetwork(generic_run / "learner" / "1.pth"),
        GenericNetwork(generic_run / "critic" / "1.pth"),
        value_head="score",
    )
    expected = network.evaluate([RustGameAdapter(g) for g in positions], legal)
    offsets = np.concatenate([[0], np.cumsum([len(ix) for ix in legal])])
    priced = 0
    for b, e in enumerate(expected):
        assert np.allclose(result.priors[offsets[b] : offsets[b + 1]], e.probs, atol=1e-5)
        n = len(positions[b].players)
        absolute = unrotate_value(result.values[b], int(result.rotations[b]), n)
        assert np.allclose(absolute[:n], e.value[:n], atol=1e-5)
        assert np.allclose(result.price_logits[b], e.price_logits, atol=1e-4)
        priced += int(np.any(e.price_logits != 0))
    assert priced > 0


def test_a_generic_system_searches_in_lockstep(generic_run):
    positions = random_positions(4, 5, steps=200, every=17)
    players = [make_player(None, g, seed=k, **VALUE_DRIVEN) for k, g in enumerate(positions)]
    searches = [p.start_search(key="g") for p in players]
    searcher = LockstepSearcher({"g": generic_evaluator(generic_run, value_head="blend=0.5")})
    searcher.run([s for s in searches if s is not None])
    for player, search in zip(players, searches):
        if search is None:
            continue
        tree = player._rust_player
        assert search.done and tree.n_at_root() >= 1 + VALUE_DRIVEN["num_readouts"]
        assert player.pick_move() in [int(i) for i in tree.legal_action_indices_at_root()]
    assert searcher.stats.leaves > 0 and searcher.stats.requests < searcher.stats.rounds * len(players)


# ------------------------------------------------------------------ eval_head_to_head
def test_generic_player_specs(h2h, generic_run):
    learner, critic = generic_run / "learner" / "1.pth", generic_run / "critic" / "1.pth"
    key, overrides = h2h.parse_player(
        f"generic:{learner}+generic:{critic}#score@400/0.05:mean,dedupe,par=8,ptemp=0.5", 64
    )
    assert key == f"generic:{learner}+{critic}#score" and h2h.is_generic(key)
    assert h2h.generic_parts(key) == {"policy": str(learner), "value": str(critic), "value_head": "score"}
    assert overrides["num_readouts"] == 400 and overrides["c_puct_init"] == 0.05 and overrides["mcts_mean_q"]
    assert overrides["prior_temperature"] == 0.5 and overrides["dedupe_leaves"] and overrides["parallel_readouts"] == 8
    assert overrides["terminal_score_weight"] == 1.0  # the run's placement reward: finished games back up points
    key, overrides = h2h.parse_player(f"generic:{learner}:sample=0.5", 64)
    assert key == f"generic:{learner}" and overrides == {"policy_temperature": 0.5}
    assert h2h.generic_parts(key) == {"policy": str(learner), "value": None, "value_head": "win"}
    with pytest.raises(SystemExit):
        h2h.parse_player(f"generic:{learner}+{critic}@8", 64)  # a legacy part in a generic system
    with pytest.raises(SystemExit):
        h2h.parse_player(f"generic:{generic_run}/missing.pth@8", 64)


def play_match(h2h, monkeypatch, settings: dict, evaluators: dict, matches: list, concurrency: int) -> list:
    """``serve_games`` (one lockstep worker) over ``matches`` [(A, B)] in the 2v2 arrangements, with
    in-process evaluators instead of the inference servers' clients."""
    jobs, results = queue.Queue(), queue.Queue()
    count = 0
    for m, (a, b) in enumerate(matches):
        for g, arrangement in enumerate(settings["arrangements"]):
            jobs.put((m, arrangement, m * h2h.MATCH_STRIDE + g, tuple(a if s == "A" else b for s in arrangement)))
            count += 1
    jobs.put(None)
    monkeypatch.setattr(h2h, "_JOBS", jobs)
    monkeypatch.setattr(h2h, "_RESULTS", results)
    monkeypatch.setattr(h2h, "_CLIENTS", {"legacy": None})
    monkeypatch.setattr(h2h, "make_evaluators", lambda _settings: evaluators)
    assert h2h.serve_games(settings, concurrency) == count
    return [results.get_nowait() for _ in range(count)]


def test_generic_search_players_finish_eval_games(h2h, generic_run, monkeypatch, tmp_path):
    """A generic learner + critic search plays 2v2 games against a legacy search (a stand-in network) and
    against the same learner's policy alone, in one lockstep worker."""
    learner, critic = generic_run / "learner" / "1.pth", generic_run / "critic" / "1.pth"
    search = f"generic:{learner}+generic:{critic}#score@6:mean,dedupe,par=4,ptemp=0.5"
    policy = f"generic:{learner}:sample=0.5"
    legacy = "legacy@6:mean,dedupe,par=4"
    players = {}
    for spec in (search, policy):
        key, overrides = h2h.parse_player(spec, 6)
        players[spec] = {"server": key, "overrides": overrides}
    # The legacy side: a stand-in network (HashNet) on the "legacy" key; its options parsed as a .pth player's.
    _, legacy_overrides = h2h.parse_player(f"{tmp_path}/legacy.pth@6:mean,dedupe,par=4", 6)
    players[legacy] = {"server": "legacy", "overrides": legacy_overrides}
    starts = tmp_path / "starts.jsonl"
    starts.write_text("")  # every game from a random auction ending, at the first Stock Round
    settings = {
        "readouts": 6, "softpick": 4, "max_length": 40, "stall": 400, "auction_unlock": True, "seed": 3,
        "players": players, "start_positions": str(starts), "random_start_fraction": 1.0,
        "arrangements": h2h.arrangements(2)[:3], "dedupe_maps": False, "num_players": 4,
    }  # fmt: skip
    evaluators = {
        players[search]["server"]: generic_evaluator(generic_run, value_head="score"),
        players[policy]["server"]: generic_evaluator(generic_run, critic=False),
        "legacy": ModelEvaluator(HashNet()),
    }
    matches = [(search, legacy), (search, policy)]
    out = play_match(h2h, monkeypatch, settings, evaluators, matches, concurrency=3)
    assert len(out) == 6 and {m for m, _, _ in out} == {0, 1}
    for match, arrangement, record in out:
        assert record["termination"] in ("finished", "max_length") and record["decision_count"] > 20
        assert sorted(arrangement) == ["A", "A", "B", "B"] and set(record["seats"]) == set(matches[match])
        assert abs(sum(record["win_share"]) - 1) < 1e-6
    assert evaluators[players[search]["server"]].states > 0 and evaluators[players[policy]["server"]].states > 0
