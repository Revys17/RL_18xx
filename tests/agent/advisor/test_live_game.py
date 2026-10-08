"""Following a server game in the Rust engine (rl18xx.agent.advisor.live_game)."""

import copy

import pytest

from rl18xx.agent.advisor import live_game as lg
from rl18xx.agent.advisor.live_game import LiveGame, UnsupportedGame, support_problem

from .conftest import truncated


def _replayed(game: dict):
    """The game replayed from scratch by a new follower: (position key, legal moves)."""
    live = LiveGame(game)
    return live.position_key, sorted(live.rust._game.factored_legal_indices())


def test_a_server_game_replays_to_the_position_it_was_saved_at(game):
    live = LiveGame(game)
    assert live.error is None
    assert live.num_players == 4 and not live.untested
    assert live.names == {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}
    # The fixture stops at a corporation's tile lay; the server says player 3 (user 103) is acting.
    acting = live.acting()
    assert acting["kind"] == "corporation" and acting["player"] == "Player 3"
    assert live.acting_mismatch(game["acting"]) is None
    assert live.acting_mismatch([101]) is not None
    assert live.rust._game.active_step_type() == "LayTile"
    assert len(live.rust.raw_actions) == len(game["actions"])


def test_following_action_by_action_matches_a_full_replay(game):
    live = LiveGame(truncated(game, 20))
    first = live.version
    for count in (21, 22, 60, 61, 150, len(game["actions"])):
        assert live.update(truncated(game, count))
        assert live.error is None
        assert (live.position_key, sorted(live.rust._game.factored_legal_indices())) == _replayed(
            truncated(game, count)
        )
    assert live.version == first + 6
    assert not live.update(truncated(game, len(game["actions"]))), "the same actions again: nothing to do"


def test_an_update_replays_only_the_new_actions(game, monkeypatch):
    live = LiveGame(truncated(game, 200))
    calls = []
    original = LiveGame._replay_range

    def counting(self, rust, filtered, start, stop, window):
        calls.append((start, stop))
        return original(self, rust, filtered, start, stop, window)

    monkeypatch.setattr(LiveGame, "_replay_range", counting)
    live.update(truncated(game, 203))
    # The settled game catches up from 198 (all but the last two of 200) to 201,
    # and the copy replays the last two.
    assert calls == [(198, 201), (201, 203)]


def test_an_undo_rebuilds_the_game(game):
    count = len(game["actions"]) - 5
    live = LiveGame(truncated(game, count))
    before = live.position_key
    extra = copy.deepcopy(game["actions"][count])
    undone = truncated(game, count)
    undone["actions"] += [extra, {"type": "undo", "entity": extra["user"], "entity_type": "player", "id": count + 2}]
    live.update({**undone, "actions": undone["actions"][:-1]})
    assert live.position_key != before
    assert live.update(undone)
    assert live.position_key == before
    assert live._settled_count == count - lg.LOOKAHEAD
    # A redo brings the move back.
    redo = {"type": "redo", "entity": extra["user"], "entity_type": "player", "id": count + 3}
    live.update({**undone, "actions": undone["actions"] + [redo]})
    assert live.position_key == _replayed(truncated(game, count + 1))[0]


def test_the_latest_moves_keep_their_positions(game):
    live = LiveGame(game)
    records = live.recent_records()
    assert records and len(records) <= lg.RECENT_WINDOW
    assert [r.index for r in records] == sorted((r.index for r in records), reverse=True)
    newest = records[0]
    assert newest.index == len(game["actions"]) - 1
    assert newest.applied
    # The position before the newest move plus that move is the current position.
    after = newest.before.pickle_clone()
    after.process_action(dict(newest.action))
    assert sorted(after._game.factored_legal_indices()) == sorted(live.rust._game.factored_legal_indices())


def test_a_rejected_action_stops_the_replay_before_it(game):
    count = 120
    bad = copy.deepcopy(game)
    no_such_train = {"type": "buy_train", "entity": "PRR", "entity_type": "corporation", "train": "D-9", "price": 1}
    bad["actions"] = copy.deepcopy(game["actions"][:count]) + [{**no_such_train, "id": count + 1}]
    live = LiveGame(bad)
    assert live.error is not None
    assert live.error["action"] == count + 1 and live.error["type"] == "buy_train"
    assert live.position_key == _replayed(truncated(game, count))[0]
    # The game goes on once the bad action is gone (an undo on the server).
    assert live.update(truncated(game, count + 3))
    assert live.error is None
    assert live.position_key == _replayed(truncated(game, count + 3))[0]


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"title": "1867"}, "isn't supported"),
        ({"settings": {"optional_rules": ["optional_6_train", "two_player_variant"]}}, "two_player_variant"),
        ({"players": [{"id": 1, "name": "solo"}]}, "2-6 players"),
    ],
)
def test_unsupported_games_are_refused_with_a_reason(game, change, reason):
    other = {**game, **change}
    assert reason in support_problem(other)
    with pytest.raises(UnsupportedGame, match=reason):
        LiveGame(other)


def test_the_six_train_variant_and_other_player_counts_are_followed(game):
    assert support_problem({**game, "settings": {"optional_rules": ["optional_6_train"]}}) is None
    three = {
        **truncated(game, 0),
        "players": game["players"][:3],
    }
    live = LiveGame(three)
    assert live.untested and live.num_players == 3
    assert live.acting()["kind"] == "player"
