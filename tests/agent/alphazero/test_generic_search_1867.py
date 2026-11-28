"""Searching a generator title (1867): the Rust tree over the title's generator moves (``mcts.rs``
``TreeMoves::Generator``), lockstep evaluation by title-generic networks (``lockstep.GenericEvaluator``: per-move price
rows, values moved into the tree's frame), ``generic_search.GenericSearchPlayer``, the player deciding a position, and
``scripts/eval_generic.py``'s searched sides -- tiny untrained 1867 checkpoints on CPU, positions from a Ruby fixture
game (``tests/fixtures/1867``)."""

import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero.game_network import absolute_values  # noqa: E402
from rl18xx.agent.alphazero.generic_data import new_engine_game  # noqa: E402
from rl18xx.agent.alphazero.generic_import_1867 import replay_export  # noqa: E402
from rl18xx.agent.alphazero.generic_inference import LocalGenericClient, LocalGenericServer  # noqa: E402
from rl18xx.agent.alphazero.generic_search import (  # noqa: E402
    BOT_SEARCH,
    GenericSearchPlayer,
    local_evaluator,
    parse_search_spec,
    search_config,
    search_position,
)
from rl18xx.agent.alphazero.generic_tokens import encode_batch  # noqa: E402
from rl18xx.agent.alphazero.lockstep import EvalResult, LockstepSearcher, to_tree_frames  # noqa: E402
from rl18xx.agent.alphazero.policy_selfplay import tempered  # noqa: E402
from tests.title_replay_harness import game_seed  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "1867" / "hs_ahjzadkh_19792.json"
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
SMALL = dict(
    d_model=32, num_heads=4, hex_layers=1, entity_layers=1, fusion_layers=1, d_trunk=64, trunk_blocks=1,
    d_head=32, d_edge=8, d_cond=16, d_id=8,
)  # fmt: skip
SEARCH = dict(BOT_SEARCH, num_readouts=24, min_readouts=24, prior_temperature=0.5)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    """A generic 1867 policy-gradient run's layout with tiny untrained checkpoints: learner/1.pth, critic/1.pth."""
    run = tmp_path_factory.mktemp("generic_1867_run")
    (run / "config.json").write_text(json.dumps({"learner_temperature": 0.5, "reward": "placement"}))
    for seed, name in ((1, "learner"), (2, "critic")):
        torch.manual_seed(seed)
        model = mg.GenericPolicyValueModel(mg.GenericModelConfig(**SMALL)).eval()
        with torch.no_grad():  # non-trivial heads (zero-initialized outputs give equal odds / uniform cells)
            for p in model.parameters():
                if p.abs().sum() == 0:
                    p.normal_(0, 0.1)
        mg.save_checkpoint(model, run / name / "1.pth", titles=["1867"])
    return run


@lru_cache(maxsize=1)
def _record():
    game = json.loads(FIXTURE.read_text())
    log, _, stopped, _ = replay_export(game)
    assert stopped is None
    return game, log


def position(count: int):
    """The fixture game after ``count`` engine actions."""
    game, log = _record()
    raw = new_engine_game("1867", {i + 1: f"Player {i + 1}" for i in range(len(game["players"]))}, (), game_seed(game))
    for action in log[:count]:
        raw.process_action(dict(action))
    return raw


def decisions(step: int = 1):
    """``(count, raw)`` at every ``step``-th decision of the fixture game (several moves or an open price)."""
    game, log = _record()
    raw = position(0)
    out = []
    for k in range(len(log) + 1):
        if raw.finished:
            break
        legal = mg.legal_actions(raw)
        if len(legal) > 1 or (len(legal) == 1 and legal.price_range[0, 1] > legal.price_range[0, 0]):
            out.append((k, raw.pickle_clone()))
        if k < len(log):
            raw.process_action(dict(log[k]))
    return out[::step]


def tree_for(raw, **overrides):
    player = GenericSearchPlayer(search_config(**{**SEARCH, **overrides}))
    player.reset(raw)
    player.tree.seed(5)
    return player


# ------------------------------------------------------------------ the decider


def _controller(raw, wire):
    kind, entity = wire.get("entity_type"), wire.get("entity")
    if kind == "player":
        return int(entity)
    owner = (raw.corporation_by_id if kind == "corporation" else raw.company_by_id)(str(entity)).owner_id_str
    if owner.startswith("corp"):
        owner = raw.corporation_by_id(owner.split(":")[1]).owner_id_str
    return int(owner.split(":")[1]) if owner.startswith("player") else None


def test_the_tree_backs_up_the_value_of_the_player_deciding():
    """At every position of the fixture game the tree's mover is the player who controls the moves' entity; at a
    stock-round home token (a minor's, chosen before anyone else bids and before it has an owner) the opening bidder;
    the encoder's seat 0 is the same player (until 2026-11-27 it was the first player at those home tokens)."""
    game, log = _record()
    raw = position(0)
    steps, home_tokens = set(), 0
    for k in range(len(log)):
        moves = json.loads(raw.rl_legal_moves_json())
        if moves:
            ids = sorted(p.id for p in raw.players)
            mover = ids[engine_rs.RustMCTSPlayer(raw).root_active_player_index()]
            assert engine_rs.mcts_decider_id(raw) == mover
            # the network's seat 0 (the encoder's mover; generic_pg.mover_seat credits the decision to it)
            assert int(raw.players[int(encode_batch([raw])["meta"][0, 5])].id) == mover, k
            if moves[0]["step"] == "HomeToken" and raw.round.round_type == "Stock":
                opening = [a for a in raw.raw_actions if a.get("type") in ("bid", "par")][-1]
                assert mover == int(opening["entity"]), k
                home_tokens += 1
            else:
                assert {_controller(raw, m["wire"]) for m in moves} == {mover}, (k, moves[0])
            steps.add(moves[0]["step"])
        raw.process_action(dict(log[k]))
    assert home_tokens >= 5
    assert {"SingleItemAuction", "BuySellParShares", "Track", "Route", "BuyTrain", "Merge"} <= steps


# ----------------------------------------------------------- moves and ranges


def test_the_trees_moves_are_the_policies_and_leaves_carry_their_ranges():
    for count, raw in decisions(step=7):
        legal = mg.legal_actions(raw)
        player = tree_for(raw)
        assert player.legal() == [int(i) for i in legal.indices], count
        batch = engine_rs.mcts_collect_leaf_games([player.tree], [1], [True], [True])
        assert batch["price_ranges"].shape == (len(legal), 3)
        assert list(batch["legal"]) == [int(i) for i in legal.indices]
        again = mg.legal_actions_with_ranges(batch["games"][0], batch["legal"], batch["price_ranges"])
        assert np.array_equal(again.rows, legal.rows) and np.array_equal(again.price_range, legal.price_range)
        assert again.price_grid and again.wires is None
        engine_rs.mcts_discard_pending([player.tree])


# -------------------------------------------------- the networks' answers


def test_the_root_reads_the_policy_the_price_row_and_the_critics_values(run):
    """One root expansion through the evaluator: priors = the policy at the search's prior temperature, the
    critic's score head in absolute seat order (root Q = W / (1 + N) after one backup), and the first price
    proposal of an open-price move in the policy's most likely cell of that move's row."""
    policy, _ = mg.load_checkpoint(str(run / "learner" / "1.pth"), "cpu")
    critic, _ = mg.load_checkpoint(str(run / "critic" / "1.pth"), "cpu")
    evaluator = local_evaluator(policy, critic, "score")
    checked_price = 0
    for count, raw in decisions(step=11):
        legal = mg.legal_actions(raw)
        arrays = mg.build_inputs([raw], [legal])
        p_out = LocalGenericServer(policy.eval()).run(arrays, want_price_logp=True)
        v_out = LocalGenericServer(critic.eval()).run(arrays)
        player = tree_for(raw)
        search = player.start_search(readouts=0, key="main")
        LockstepSearcher({"main": evaluator}).run([search])
        stats = player.tree.root_target_stats()
        expected = tempered(np.exp(np.asarray(p_out.logp, dtype=np.float64)), 0.5)
        assert np.allclose(stats["child_prior"], expected, atol=1e-5), count
        ids = [p.id for p in raw.players]
        n = len(ids)
        value = absolute_values(v_out.score[0], int(arrays["meta"][0, 5]), n, ids)
        assert np.allclose(player.tree.root_q_vector(), value[:n] / 2.0, atol=1e-5), count
        for k in range(len(legal)):
            if legal.price_range[k, 1] <= legal.price_range[k, 0] or legal.rows[k, mg.C_PRICE_KIND] < 0:
                continue
            index = int(legal.indices[k])
            row = int(np.flatnonzero(arrays["price_rows"] == k)[0])
            cells = legal.price_cells(k)
            logp = np.asarray(p_out.price_logp[row], dtype=np.float64)
            live = [c for c in range(mg.NUM_CELLS) if np.isfinite(logp[c]) and cells.counts[c] > 0]
            modal = max(live, key=lambda c: (logp[c], -c))
            player.tree.maybe_add_child(player.tree.root_idx, index)
            [price] = player.tree.price_grandchildren_at_root()[index]
            assert cells.cell_of(int(price)) == modal, (count, index)
            checked_price += 1
            break
    assert checked_price >= 3


def test_values_reach_the_tree_in_its_frame_whatever_the_seating():
    """``to_tree_frames``: a generic row (seat k = seating[(k + rotation) % n]) lands in the tree's frame (seat k =
    the (k + tree rotation)-th player by id), which ``mcts_apply_results`` unrotates to absolute order."""
    rng = np.random.default_rng(0)
    for ids in ([1, 2, 3, 4], [3, 1, 4, 2], [1004, 1001, 1003, 1002]):
        n = len(ids)
        order = sorted(ids)
        true = {pid: rng.random() for pid in ids}  # each player's value
        for rotation in range(n):
            for tree_rotation in range(n):
                generic = np.zeros((1, 6), dtype=np.float32)
                generic[0, :n] = [true[ids[(k + rotation) % n]] for k in range(n)]
                out = to_tree_frames(generic, np.array([rotation]), np.array([tree_rotation]), n, [ids])
                absolute = [out[0, (i - tree_rotation) % n] for i in range(n)]  # mcts_apply_results' unrotation
                assert np.allclose(absolute, [true[pid] for pid in order])


# ------------------------------------------------- lockstep: trees together


class StubGenericEvaluator:
    """A games evaluator whose answers depend on each position alone (its token encoding's hash): priors, values
    in the generic frame and a price row per priced legal move -- so a tree searched among others must end as
    when searched alone."""

    takes_games = True

    def send_games(self, games, legal, price_ranges=None):
        assert price_ranges is not None and all(r is not None for r in price_ranges)
        priors, values, rows, logits, rotations, seats = [], [], [], [], [], []
        offset = 0
        for g, ix, ranges in zip(games, legal, price_ranges):
            enc = encode_batch([g])
            blobs = [np.ascontiguousarray(v).tobytes() for v in enc.values() if hasattr(v, "tobytes")]
            digest = hashlib.sha1(b"".join(blobs))
            rng = np.random.default_rng(int.from_bytes(digest.digest()[:8], "little") ^ len(ix))
            p = rng.random(len(ix)) + 0.05
            priors.append(np.log(p / p.sum()))
            v = np.zeros(6, dtype=np.float32)
            v[: len(g.players)] = rng.random(len(g.players))
            values.append(v)
            rotations.append(int(enc["meta"][0, 5]))
            seats.append([int(q.id) for q in g.players])
            for k, r in enumerate(np.asarray(ranges).reshape(-1, 3)):
                if r[0] >= 0:
                    rows.append(offset + k)
                    logits.append(rng.normal(size=mg.NUM_CELLS).astype(np.float32))
            offset += len(ix)
        return EvalResult(
            priors=np.exp(np.concatenate(priors)).astype(np.float32),
            values=np.stack(values),
            rotations=np.array(rotations, dtype=np.int64),
            seat_ids=seats,
            price_rows=np.array(rows, dtype=np.int64),
            price_row_logits=np.array(logits, dtype=np.float32).reshape(-1, mg.NUM_CELLS),
        )

    def receive(self, pending):
        return pending


def _summary(player):
    stats = player.tree.root_target_stats()
    return (
        list(stats["legal"]),
        list(stats["child_n"]),
        np.round(stats["child_prior"], 6).tolist(),
        {k: dict(v) for k, v in player.tree.price_grandchildren_at_root().items()},
    )


def test_trees_searched_together_end_as_searched_alone():
    positions = [raw for count, raw in decisions(step=9)][:6]
    alone = []
    for raw in positions:
        player = tree_for(raw)
        LockstepSearcher({"main": StubGenericEvaluator()}).run([player.start_search(key="main")])
        alone.append(_summary(player))
    players = [tree_for(raw) for raw in positions]
    searcher = LockstepSearcher({"main": StubGenericEvaluator()})
    searcher.run([p.start_search(key="main") for p in players])
    assert [_summary(p) for p in players] == alone
    assert searcher.stats.requests < sum(p.tree.n_at_root() for p in players)
    assert any(s[3] for s in alone), "no open-price move searched"


# ------------------------------------------------------------- the player


def test_the_player_plays_the_trees_move_and_keeps_its_tree(run):
    evaluator = local_evaluator(run / "learner" / "1.pth", run / "critic" / "1.pth", "score")
    kept = 0
    for count, raw in decisions(step=13):
        player = GenericSearchPlayer(search_config(**SEARCH))
        search_position(player, raw, evaluator)
        assert player.tree.n_at_root() >= 1 + SEARCH["num_readouts"] or len(player.legal()) == 1
        index = player.pick_move()
        actions = player.play(index)
        game = raw.pickle_clone()
        for action in actions:  # the move at its price, then the forced moves: the engine takes them as they are
            game.process_action(dict(action))
        assert [dict(a) for a in game.raw_actions] == [dict(a) for a in player.tree.root_game_object().raw_actions]
        if not game.finished and player.follow(game):
            kept += 1
            assert player.legal() == [int(i) for i in mg.legal_actions(game).indices]
        assert not player.follow(raw)  # an earlier position: a fresh tree
    assert kept >= 2


def test_the_most_visited_allowed_move():
    raw = decisions()[0][1]
    player = tree_for(raw)
    LockstepSearcher({"main": StubGenericEvaluator()}).run([player.start_search(key="main")])
    visits = player.visits()
    best = max(visits, key=visits.get)
    others = set(visits) - {best}
    second = max(others, key=visits.get)
    assert player.pick_move(allowed=others) == second
    assert player.pick_move(allowed=set(visits)) == player.pick_move()


def test_search_specs(run):
    learner, critic = run / "learner" / "1.pth", run / "critic" / "1.pth"
    s = parse_search_spec(f"generic:{learner}+generic:{critic}#score@64/0.05:mean,dedupe,par=8,ptemp=0.5")
    assert (s.policy, s.value, s.value_head) == (str(learner), str(critic), "score")
    assert s.overrides == dict(num_readouts=64, min_readouts=64, c_puct_init=0.05, c_puct_by_round={},
                               mcts_mean_q=True, fpu_reduction=0.0, dedupe_leaves=True, parallel_readouts=8,
                               prior_temperature=0.5)  # fmt: skip
    config = s.config(4)
    assert config.terminal_score_weight == 1.0 and tuple(config.terminal_score_points) == (1.0, 0.5, 0.2, 0.0)
    assert config.softpick_move_cutoff == 0 and config.dirichlet_noise_weight == 0.0
    assert len(s.config(6).terminal_score_points) == 6
    assert parse_search_spec(f"{learner}:sample=0.5").policy_only
    own = parse_search_spec(f"generic:{learner}@32", default_head="score")
    assert own.value is None and own.readouts == 32 and own.value_head == "score"
    bot = parse_search_spec(f"generic:{critic}#win", policy=str(learner), default_readouts=400)
    assert (bot.policy, bot.value, bot.value_head, bot.readouts) == (str(learner), str(critic), "win", 400)
    assert parse_search_spec("self@8", policy=str(learner)).value is None
    with pytest.raises(FileNotFoundError):
        parse_search_spec("generic:/nonexistent.pth@8")
    with pytest.raises(ValueError):
        parse_search_spec(f"{learner}:bogus")


# -------------------------------------------------------- eval_generic.py


@pytest.fixture(scope="module")
def eval_generic():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("_eval_generic_1867", SCRIPTS / "eval_generic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_generic_plays_a_searched_side_against_the_policy(run, eval_generic):
    from eval_policy_only import arrangements

    learner, critic = run / "learner" / "1.pth", run / "critic" / "1.pth"
    a = eval_generic.parse_side(f"generic:{learner}+generic:{critic}#score@8/0.05:mean,dedupe,par=4,ptemp=0.5", 200)
    b = eval_generic.parse_side(f"{learner}@0.5", 200)
    assert a["kind"] == "search" and b == {"kind": "policy", "path": str(learner), "temperature": 0.5}
    assert eval_generic.parse_side(f"generic:{learner}:sample=0.5", 200)["kind"] == "policy"
    settings = {
        "title": "1867",
        "paths": {"B": b["path"]},
        "temperatures": {"B": 0.5},
        "search": {"A": a},
        "softpick": 0,
        "server": False,
        "seed": 3,
        "start_positions": None,
        "random_start_fraction": 0.0,
        "auction_start": True,
        "max_decisions": 40,
        "auction_stall_decisions": 0,
        "price_eps": 0.0,
        "save_games": 2,
        "arrangements": arrangements(2, 4),
        "rules_mix": {"base": 1.0},
        "redraw_all_pass": True,
    }
    records = eval_generic.play_games([0, 1], settings)
    assert len(records) == 2
    for r in records:
        assert r["termination"] in ("max_length", "finished") and r["decisions"] >= 30
        searched = r["search"]["A"]["decisions"]
        assert 0 < searched < r["decisions"]
        # the saved game replays in the engine, and every decision went to the side of the player deciding it:
        # side A searched each of its decisions
        from rl18xx.agent.alphazero.generic_play import title_play

        play = title_play("1867")
        raw = new_engine_game("1867", {pid: name for pid, name in r["players"]}, (), r["seed"])
        sides = []
        for action in r["raw_actions"]:
            if play.is_decision(mg.legal_actions(raw)):
                sides.append(r["seats"][eval_generic.decider_seat(raw, [pid for pid, _ in r["players"]])])
            raw.process_action(dict(action))
        assert len(sides) == r["decisions"] and sides.count("A") == searched
