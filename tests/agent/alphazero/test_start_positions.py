"""Self-play start positions: games that begin at the first Stock Round from a
human post-auction position or a random auction ending, instead of the
private auction (``rl18xx/agent/alphazero/start_positions.py``)."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import start_positions as sp  # noqa: E402
from rl18xx.agent.alphazero.config import SelfPlayConfig  # noqa: E402
from rl18xx.agent.alphazero.mcts import POLICY_SIZE, VALUE_SIZE, _rust_encode  # noqa: E402
from rl18xx.agent.alphazero.rust_mcts_player import RustMCTSPlayer  # noqa: E402

FACES = {"SV": 20, "CS": 40, "DH": 70, "MH": 110, "CA": 160, "BO": 220}
STARTING_CASH = {2: 1200, 3: 800, 4: 600, 5: 480, 6: 400}


def bid(player, company, price):
    return {"type": "bid", "entity": player, "entity_type": "player", "company": company, "price": price}


def pass_(player):
    return {"type": "pass", "entity": player, "entity_type": "player"}


BO_PAR = {"type": "par", "entity": 2, "entity_type": "player", "corporation": "B&O", "share_price": "100,0,6"}
# P1 buys the SV, P2 bids on the DH, everyone passes (real 1830: the SV pays
# P1 $5, nothing is discounted), then the rest go at face and P2 pars the B&O.
AUCTION = [
    bid(1, "SV", 20),
    bid(2, "DH", 75),
    pass_(2),  # out of turn: the engine rejects it, the importer drops it
    pass_(3),
    pass_(4),
    pass_(1),
    pass_(2),
    bid(3, "CS", 40),
    bid(4, "MH", 110),
    bid(1, "CA", 160),
    bid(2, "BO", 220),
    BO_PAR,
]
PREFIX = AUCTION[:2] + AUCTION[3:]
FIRST_SR = {
    "round": "Stock",
    "cash": {1: 425, 2: 305, 3: 560, 4: 490},
    "owners": {"SV": 1, "CS": 3, "DH": 2, "MH": 4, "CA": 1, "BO": 2},
    "bo_par": 100,
    "bo_president": 2,
    "priority": 3,
    "current": 3,
    "bank": 10220,
}


STOCK_ROUND = [
    {"type": "par", "entity": 3, "entity_type": "player", "corporation": "PRR", "share_price": "67,5,6"},
    pass_(4),
]


def _recorded_game(game_id="1001", actions=None):
    """A cleaned export as pretraining.load_games_from_json returns it."""
    actions = AUCTION + STOCK_ROUND if actions is None else actions
    recorded = [dict(a, id=i + 1, user=a["entity"], created_at=1_700_000_000 + i) for i, a in enumerate(actions)]
    return {
        "id": game_id,
        "title": "1830",
        "players": [{"id": i, "name": f"Player {i}"} for i in range(1, 5)],
        "settings": {"optional_rules": []},
        "actions": recorded,
    }


def _write_starts(path, starts):
    with open(path, "w") as f:
        for game_id, actions in starts:
            f.write(json.dumps({"id": game_id, "num_players": 4, "actions": actions}) + "\n")
    return str(path)


# ------------------------------------------------------------------ human starts
def test_human_prefix_stops_at_the_first_stock_round():
    prefix, reason = sp.human_start_prefix(_recorded_game())
    assert reason is None
    # The auction and the B&O par, nothing from the Stock Round, the rejected
    # pass dropped, and only what process_action reads.
    assert prefix == PREFIX
    assert all(set(a) <= set(sp.ACTION_KEYS) for a in prefix)

    game = sp.apply_actions(sp.new_game(4), prefix)
    assert sp.position_summary(game) == FIRST_SR
    # Same position in the Python reference engine.
    assert sp.position_summary(sp.python_game(4, prefix)) == FIRST_SR


def test_human_prefix_rejects_unfinished_auctions_and_other_actions():
    assert sp.human_start_prefix(_recorded_game(actions=AUCTION[:5])) == (None, "auction_not_finished")
    message = {"type": "message", "entity": 1, "entity_type": "player", "message": "hi"}
    assert sp.human_start_prefix(_recorded_game(actions=[message] + AUCTION)) == (None, "action_type_message")


def test_load_and_sample_start_positions(tmp_path):
    path = _write_starts(tmp_path / "starts.jsonl", [("1001", PREFIX), ("1002", PREFIX)])
    starts = sp.load_start_positions(path)
    assert sorted(starts) == [4]
    assert [s.label for s in starts[4]] == ["human:1001", "human:1002"]
    assert sp.load_start_positions(path) is starts  # read once per process

    rng = random.Random(0)
    assert sp.sample_start_position(4, path, random_fraction=0.0, rng=rng).label.startswith("human:")
    assert sp.sample_start_position(4, path, random_fraction=1.0, rng=rng).label == "random"
    # No human starts for 3 players: always a random ending.
    three = sp.sample_start_position(3, path, random_fraction=0.0, rng=rng)
    assert three.label == "random"
    assert sp.position_summary(sp.apply_actions(sp.new_game(3), three.actions))["round"] == "Stock"


# ----------------------------------------------------------------- random starts
@pytest.mark.parametrize(
    "num_players,seeds", [(4, range(200)), (2, range(40)), (3, range(40)), (5, range(40)), (6, range(40))]
)
def test_random_starts_are_legal_first_stock_round_positions(num_players, seeds):
    for seed in seeds:
        actions = sp.random_start_actions(num_players, random.Random(seed))
        summary = sp.position_summary(sp.apply_actions(sp.new_game(num_players), actions))
        prices = {a["company"]: (a["entity"], a["price"]) for a in actions if a["type"] == "bid"}
        if "SV" not in prices:  # discounted to $0 by four all-pass rounds and handed over
            prices["SV"] = (summary["owners"]["SV"], 0)
        assert sorted(prices) == sorted(FACES)
        assert prices["SV"][1] in (0, 5, 10, 15, 20)
        for sym, (_, price) in prices.items():
            if sym != "SV":
                assert FACES[sym] <= price <= 2 * FACES[sym] and price % 5 == 0, (seed, sym, price)
        spent = {}
        for owner, price in prices.values():
            spent[owner] = spent.get(owner, 0) + price
        assert all(total <= STARTING_CASH[num_players] for total in spent.values())

        assert summary["round"] == "Stock"
        assert summary["owners"] == {sym: owner for sym, (owner, _) in prices.items()}
        assert summary["cash"] == {p: STARTING_CASH[num_players] - spent.get(p, 0) for p in range(1, num_players + 1)}
        assert summary["bo_par"] in (67, 71, 76, 82, 90, 100) and summary["bo_president"] == prices["BO"][0]
        if seed < 10:
            assert sp.position_summary(sp.python_game(num_players, actions)) == summary


def test_random_starts_vary_and_respect_the_price_cap():
    endings = {json.dumps(sp.random_start_actions(4, random.Random(seed))) for seed in range(20)}
    assert len(endings) == 20
    at_face = sp.random_start_actions(4, random.Random(0), max_price_multiple=1.0)
    assert all(a["price"] == FACES[a["company"]] for a in at_face if a["type"] == "bid")
    bids = [a for s in range(20) for a in sp.random_start_actions(4, random.Random(s)) if a["type"] == "bid"]
    assert any(a["price"] > 1.5 * FACES[a["company"]] for a in bids)


def test_random_starts_sometimes_discount_the_sv():
    """An all-pass round while the SV is the cheapest private takes $5 off it
    (real 1830); four of them make it free to the next player."""
    sv_prices, free_sv_owners = [], set()
    for seed in range(300):
        actions = sp.random_start_actions(4, random.Random(seed))
        bids = [a["price"] for a in actions if a["type"] == "bid" and a["company"] == "SV"]
        sv_prices.append(bids[0] if bids else 0)
        if not bids:
            summary = sp.position_summary(sp.apply_actions(sp.new_game(4), actions))
            free_sv_owners.add(summary["owners"]["SV"])
    assert set(sv_prices) == {0, 5, 10, 15, 20}
    assert 0.1 < sum(p < 20 for p in sv_prices) / len(sv_prices) < 0.4
    # Bids placed before the all-pass rounds shift who passes first, so the
    # free SV doesn't always go to the player who opened the auction.
    assert len(free_sv_owners) > 1
    never = [a for s in range(20) for a in sp.random_start_actions(4, random.Random(s), sv_discount_fraction=0.0)]
    assert all(a["price"] == 20 for a in never if a["type"] == "bid" and a["company"] == "SV")


def test_random_allocation_is_seeded():
    companies = list(FACES.items())
    cash = dict.fromkeys(range(1, 5), 600)
    assert sp.random_allocation(companies, cash, random.Random(7)) == sp.random_allocation(
        companies, cash, random.Random(7)
    )


# ---------------------------------------------------------------- self-play path
class DummyNet:
    """Uniform prior, zero value."""

    def encoder_type(self) -> str:
        return "GNN"

    def eval(self):
        return self

    def run_encoded(self, encoded_game_state):
        priors = torch.ones(POLICY_SIZE, dtype=torch.float32) / POLICY_SIZE
        return priors, torch.log(priors), torch.zeros(VALUE_SIZE, dtype=torch.float32)

    def run_many_encoded(self, encoded_game_states):
        n = len(encoded_game_states)
        probs, log_probs, value = self.run_encoded(None)
        return [probs] * n, [log_probs] * n, [value] * n


def _config(**overrides) -> SelfPlayConfig:
    defaults = dict(
        network=DummyNet(),
        use_rust_mcts=True,
        num_readouts=4,
        parallel_readouts=2,
        min_readouts=2,
        use_score_values=False,
        player_count_distribution={4: 1.0},
        enable_resign=False,
        dirichlet_noise_weight=0.0,
    )
    defaults.update(overrides)
    return SelfPlayConfig(**defaults)


def _same_encoding(a, b) -> bool:
    for x, y in zip(a, b):
        if isinstance(x, (np.ndarray, torch.Tensor)):
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                return False
        elif x != y:
            return False
    return len(a) == len(b)


@pytest.mark.parametrize("random_start_fraction", [0.0, 1.0])
def test_self_play_starts_at_the_first_stock_round(tmp_path, monkeypatch, random_start_fraction):
    """The game starts at Stock Round 1; its training examples cover only the
    self-play decisions, rebuilt by replaying the start on an empty game — not
    a newly sampled start — and match the positions the search saw."""
    from rl18xx.agent.alphazero import self_play

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()
    # Two different human starts, so a resampled start would differ.
    alt_prefix = [bid(1, "SV", 20), bid(2, "CS", 40), bid(3, "DH", 70), bid(4, "MH", 110), bid(1, "CA", 160)]
    alt_prefix += [bid(2, "BO", 220), dict(BO_PAR, share_price="67,5,6")]
    path = _write_starts(tmp_path / "starts.jsonl", [("1001", PREFIX), ("1002", alt_prefix)])

    seen = []
    play_move = RustMCTSPlayer.play_move

    def recording_play_move(self, action_index):
        seen.append(_rust_encode(self.get_game_state()))
        return play_move(self, action_index)

    monkeypatch.setattr(RustMCTSPlayer, "play_move", recording_play_move)
    config = _config(
        game_id="start",
        start_positions_path=path,
        random_start_fraction=random_start_fraction,
        max_game_length=40,
    )
    player = self_play.SelfPlay(config).play()

    status = json.loads((tmp_path / "status" / "start.json").read_text())
    assert status["status"] == "Completed"
    assert status["start"] == player.start_label
    expected_labels = ("random",) if random_start_fraction else ("human:1001", "human:1002")
    assert player.start_label in expected_labels
    assert status["auction_unlock_discounts"] == 0

    examples = list(player.extract_data())
    assert len(examples) == len(player.searches_pi) == len(seen) > 0
    first = examples[0][0]
    assert first.round.__class__.__name__ == "Stock"
    n_start = len(player.played_actions) - len(player.searches_pi)
    assert [sp.clean_action(a) for a in first.raw_actions] == [
        sp.clean_action(a) for a in player.played_actions[:n_start]
    ]
    if player.start_label == "human:1001":
        assert sp.position_summary(first) == FIRST_SR
    for (game, legal, pi, _, _), expected in zip(examples, seen):
        assert _same_encoding(_rust_encode(game), expected)
        assert pi.shape[-1] == POLICY_SIZE and set(legal.tolist()) >= set(np.flatnonzero(pi.numpy()).tolist())


def test_defaults_still_start_at_the_auction(tmp_path, monkeypatch):
    from rl18xx.agent.alphazero import self_play

    config = SelfPlayConfig()
    assert config.start_positions_path is None
    assert config.random_start_fraction == 0.2 and config.random_start_max_price_multiple == 2.0

    player = RustMCTSPlayer(_config())
    assert player.start_label is None
    assert player.played_actions == []
    assert player.get_game_state().round.__class__.__name__ == "WaterfallAuction"
    assert player.get_game_state()._game.auction_unlock is True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(self_play, "SELF_PLAY_GAMES_STATUS_PATH", tmp_path / "status")
    (tmp_path / "status").mkdir()
    self_play.SelfPlay(_config(game_id="fresh", max_game_length=3)).play()
    assert "start" not in json.loads((tmp_path / "status" / "fresh.json").read_text())


def test_python_mcts_player_refuses_start_positions(tmp_path):
    from rl18xx.agent.alphazero.self_play import MCTSPlayer

    path = _write_starts(tmp_path / "starts.jsonl", [("1001", PREFIX)])
    with pytest.raises(ValueError, match="Rust MCTS"):
        MCTSPlayer(_config(use_rust_mcts=False, start_positions_path=path))


# ----------------------------------------------------------------------- loop
def test_loop_passes_start_position_keys_to_self_play(tmp_path, monkeypatch):
    from rl18xx.agent.alphazero import loop

    config_path = tmp_path / "loop_config.json"
    config_path.write_text(
        json.dumps(
            {"start_positions_path": "starts.jsonl", "random_start_fraction": 0.3, "player_count_distribution": None}
        )
    )
    monkeypatch.setattr(loop, "LOOP_CONFIG_PATH", config_path)
    monkeypatch.setattr(loop, "SELF_PLAY_LOGS_PATH", tmp_path)
    monkeypatch.setattr(loop, "get_latest_model", lambda *_: DummyNet())
    captured = {}

    class CapturingSelfPlay:
        def __init__(self, config):
            captured["config"] = config

        def run_game(self):
            pass

    monkeypatch.setattr(loop, "SelfPlay", CapturingSelfPlay)
    loop.run_self_play(0, str(tmp_path), "ts", 0)
    config = captured["config"]
    assert config.start_positions_path == "starts.jsonl"
    assert config.random_start_fraction == 0.3
    assert config.random_start_max_price_multiple == 2.0

    loop_config = loop.load_loop_config(1, 1, 1, loop.TrainingConfig(), 8)
    assert loop_config.start_positions_path == "starts.jsonl" and loop_config.random_start_fraction == 0.3
    assert json.loads(config_path.read_text())["random_start_max_price_multiple"] == 2.0
