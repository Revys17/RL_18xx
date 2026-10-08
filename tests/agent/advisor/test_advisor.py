"""The model's advice on a followed game (rl18xx.agent.advisor.advisor), with untrained CPU models."""

import time

import numpy as np
import pytest

from rl18xx.agent.advisor.advisor import Advisor, SearchBusy, checkpoint_label, price_options
from rl18xx.agent.advisor.live_game import LiveGame

from .conftest import truncated


@pytest.fixture(scope="module")
def advisor(advisor_models):
    return Advisor(advisor_models)


def test_checkpoint_labels():
    assert checkpoint_label("model_checkpoints_pg/pg4_20261007/learner/600.pth") == "pg4_20261007/learner/600"


def test_advice_on_an_operating_turn(advisor, game):
    live = LiveGame(game)
    advice = advisor.advise(live, game["acting"])
    assert advice["supported"] and not advice["finished"] and advice["warnings"] == []
    assert [p["name"] for p in advice["players"]] == ["Player 1", "Player 2", "Player 3", "Player 4"]
    assert advice["acting"]["kind"] == "corporation" and advice["acting"]["player"] == "Player 3"
    win = advice["win"]["players"]
    assert len(win) == 4 and sum(p["probability"] for p in win) == pytest.approx(1.0)
    assert advice["policy"]["role"] == "policy" and advice["policy"]["temperature"] == 1.0
    moves = advice["moves"]
    assert 1 <= len(moves) <= 5 and advice["num_legal"] >= len(moves)
    probabilities = [m["probability"] for m in moves]
    assert probabilities == sorted(probabilities, reverse=True) and sum(probabilities) <= 1 + 1e-9
    assert any(m["type"] == "lay_tile" and m["map"] and m["tile"] and "rotation" in m for m in moves)
    assert all(m["description"] and m["actor"] for m in moves)
    assert advice["recent"], "the last moves get the model's probabilities"
    for entry in advice["recent"]:
        assert 0 <= entry["probability"] <= 1 and 1 <= entry["rank"] <= entry["num_legal"]
        assert entry["label"] in ("expected", "plausible", "surprising")


def test_move_probabilities_sum_to_one_over_the_legal_moves(advisor, game):
    live = LiveGame(game)
    role, legal, probs, price_row = advisor._analyse([live.rust])[0]
    assert sorted(legal) == sorted(live.rust._game.factored_legal_indices())
    assert probs.sum() == pytest.approx(1.0) and np.all(probs >= 0)
    assert price_row is not None and price_row.ndim == 2


def test_the_auction_uses_the_auction_policy_and_prices_bids(advisor, advisor_models, game):
    live = LiveGame(truncated(game, 1))
    advice = advisor.advise(live)
    assert advice["policy"]["role"] == "auction_policy"
    assert advice["policy"]["checkpoint"] == advisor_models.labels["auction_policy"]
    bids = [m for m in advice["moves"] if m["type"] == "bid"]
    assert bids
    for move in bids:
        price = move["price"]
        assert price["range"][0] <= price["price"] <= price["range"][1]
        if not price["fixed"]:
            options = price["options"]
            assert options[0]["price"] == price["price"] and len(options) <= 3
            assert all(o["low"] <= o["price"] <= o["high"] for o in options)
            assert f"${price['price']}" in move["description"]
    assert advice["recent"][0]["policy"] == "auction_policy"


def test_price_options_for_fixed_and_open_prices(game):
    live = LiveGame(truncated(game, 0))
    rs = live.rust._game
    sv = 1  # Bid on the SV: the waterfall sells it at a set price
    assert price_options(rs, sv, None) == {"fixed": True, "price": 20, "range": [20, 20], "options": []}
    open_price = price_options(rs, 2, None)  # a bid on the CS: any price from $45
    assert not open_price["fixed"] and open_price["range"][0] == 45
    assert sum(o["probability"] for o in open_price["options"]) <= 1 + 1e-9
    assert price_options(rs, 0, None) is None  # a pass has no price


def test_a_three_player_game_is_flagged_untested(advisor, game):
    three = {**truncated(game, 0), "players": game["players"][:3]}
    advice = advisor.advise(LiveGame(three))
    assert advice["untested_player_count"]
    assert any(w.startswith("Untested") for w in advice["warnings"])
    assert len(advice["win"]["players"]) == 3


def test_advice_is_cached_per_position(advisor, game):
    live = LiveGame(game)
    first = advisor.advise(live)
    assert advisor.advise(live)["moves"] == first["moves"]
    assert advisor.advise(live, [101])["warnings"], "the server's acting list is checked every time"


def test_think_harder_runs_one_search_at_a_time(advisor, game):
    live = LiveGame(game)
    job = advisor.start_search(live, 16)
    with pytest.raises(SearchBusy):
        advisor.start_search(live, 16)
    deadline = time.time() + 120
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == "done", job.error
    status = advisor.job(job.id).to_dict()
    assert status["progress"] == 1.0 and status["position"] == live.position_key
    result = status["result"]
    assert result["visits"] >= 16
    assert result["moves"] and sum(m["visits"] for m in result["moves"]) <= result["visits"]
    assert all(m["description"] for m in result["moves"])
    assert len(result["win"]) == 4 and sum(p["probability"] for p in result["win"]) == pytest.approx(1.0)
    # Done: the next one may start.
    second = advisor.start_search(live, 8)
    while second.status == "running":
        time.sleep(0.05)
    assert second.status == "done"
