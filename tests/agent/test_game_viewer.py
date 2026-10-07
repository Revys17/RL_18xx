"""Saved games (game_records), the dashboard's game viewer (game_viewer + /games
endpoints), and the writers and readers around them: eval_policy_only.py
--save-games and main.py replay."""

import json
import sys
from pathlib import Path

import pytest

from rl18xx.agent.alphazero import game_records
from rl18xx.agent.alphazero import policy_gradient as pg
from rl18xx.agent.dashboard import dashboard, game_viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "agent" / "alphazero"))

from test_policy_gradient import _Client, _settings  # noqa: E402

NUM_PLAYERS = 4


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    """Two short policy-gradient games (uniform fake policies) with their action logs."""
    start_file = tmp_path_factory.mktemp("starts") / "starts.jsonl"
    start_file.write_text("")
    clients = {"learner": _Client(), "sl": _Client(), "pool": _Client()}
    saved = dict(pg._CLIENTS)
    pg._CLIENTS.clear()
    pg._CLIENTS.update(clients)
    try:
        result = pg.play_pg_games(2, _settings(str(start_file), max_decisions=60, save_game_every=1))
    finally:
        pg._CLIENTS.clear()
        pg._CLIENTS.update(saved)
    return result["games"]


def _h2h_collection(root: Path, actions: list, start_positions=None) -> Path:
    """An eval_head_to_head.py output directory with one game."""
    directory = root / "logs" / "eval" / "h2h_test"
    (directory / "games").mkdir(parents=True)
    (directory / "settings.json").write_text(
        json.dumps({"readouts": 64, "matches": [["new@64", "old@64"]], "start_positions": start_positions})
    )
    record = {
        "game_idx": 7,
        "seats": ["B", "A", "A", "B"],
        "termination": "finished",
        "decision_count": 12,
        "decisions": {"stock": 12},
        "start": None,
        "net_worth": [1.0, 4.0, 2.0, 3.0],
        "win_share": [0.0, 1.0, 0.0, 0.0],
        "match": 0,
    }
    lost = {**record, "game_idx": 8}  # in the index, but its file is gone
    (directory / "games.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(lost) + "\n{partial")
    (directory / "games" / "7.json").write_text(json.dumps(actions))
    return directory


# ----------------------------------------------------------------- game_records
def test_a_game_record_round_trips_and_reads_as_an_18xx_export(tmp_path, played):
    actions = played[0]["saved_game"]["raw_actions"]
    record = game_records.make_game_record(actions, NUM_PLAYERS, meta={"name": "g1", "update": 3})
    assert [p["name"] for p in record["players"]] == [f"Player {i}" for i in range(1, 5)]
    relative = game_records.save_game_record(tmp_path, "g1", record)
    assert relative == "games/g1.json"
    loaded = game_records.load_game_file(tmp_path / relative)
    assert loaded.actions == actions and loaded.num_players == NUM_PLAYERS
    assert loaded.auction_unlock is False and loaded.meta["update"] == 3
    with pytest.raises(ValueError):
        game_records.save_game_record(tmp_path, "../escape", record)


def test_bare_action_lists_and_self_play_logs_load(tmp_path, played):
    actions = played[0]["saved_game"]["raw_actions"]
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(actions))
    loaded = game_records.load_game_file(bare)
    assert loaded.actions == actions and loaded.num_players == NUM_PLAYERS and loaded.auction_unlock is None
    log = tmp_path / "self_play.log"
    log.write_text(f"2026-10-07 INFO rl18xx.agent.alphazero.self_play - INFO - Game actions: {actions!r}\n")
    assert game_records.load_game_file(log).actions == actions
    junk = tmp_path / "junk.json"
    junk.write_text('{"not": "a game"}')
    with pytest.raises(ValueError):
        game_records.load_game_file(junk)


def test_head_to_head_games_are_listed_with_their_seats(tmp_path, played):
    directory = _h2h_collection(tmp_path, played[0]["saved_game"]["raw_actions"], start_positions="starts.jsonl")
    games = game_records.list_games(directory)
    assert [g["name"] for g in games] == ["7"]  # game 8's file is missing; the partial line is skipped
    game = games[0]
    assert game["seat_labels"] == ["old@64", "new@64", "new@64", "old@64"]
    assert game["sides"] == ["B", "A", "A", "B"] and game["match"] == "new@64 vs old@64"
    assert game["decisions"] == 12 and game["winners"] == [1]
    assert game["auction_unlock"] is False  # start-position games don't use the variant
    collections = game_records.find_collections(tmp_path)
    assert [(c["id"], c["kind"], c["games"]) for c in collections] == [("logs/eval/h2h_test", "head_to_head", 1)]


def test_unindexed_game_files_are_listed_from_their_own_metadata(tmp_path, played):
    meta = {"seat_labels": list("abcd"), "win_share": [0, 0, 1, 0]}
    record = game_records.make_game_record(played[0]["saved_game"]["raw_actions"], NUM_PLAYERS, meta=meta)
    game_records.save_game_record(tmp_path / "logs/games/mine", "x", record)
    games = game_records.list_games(tmp_path / "logs/games/mine")
    assert games[0]["name"] == "x" and games[0]["seat_labels"] == list("abcd") and games[0]["winners"] == [2]


def test_only_collections_under_the_roots_resolve(tmp_path, played):
    _h2h_collection(tmp_path, played[0]["saved_game"]["raw_actions"])
    (tmp_path / "elsewhere" / "games").mkdir(parents=True)
    expected = (tmp_path / "logs/eval/h2h_test").resolve()
    assert game_records.resolve_collection(tmp_path, "logs/eval/h2h_test") == expected
    for bad in ["elsewhere", "logs/eval/../../elsewhere", "/etc", "", "logs/eval/missing"]:
        assert game_records.resolve_collection(tmp_path, bad) is None, bad
    directory = tmp_path / "logs/eval/h2h_test"
    assert game_records.resolve_game_file(directory, "7") is not None
    for bad in ["../settings", "8", "", "7.json/.."]:
        assert game_records.resolve_game_file(directory, bad) is None, bad


# ------------------------------------------------------------------ game_viewer
def test_the_viewer_replays_a_saved_game_and_steps_through_it(tmp_path, played):
    actions = played[0]["saved_game"]["raw_actions"]
    path = tmp_path / "g.json"
    path.write_text(json.dumps(game_records.make_game_record(actions, NUM_PLAYERS)))
    summary = game_viewer.game_summary(path)
    assert summary["error"] is None and summary["playable_steps"] == summary["steps"] == len(actions)
    assert summary["auction_unlock"] is False
    assert any(m["label"] == "Stock Round 1" for m in summary["rounds"])
    assert all(isinstance(step, int) and 0 <= step <= len(actions) for step, _ in summary["log"])
    assert not any(" object at 0x" in message for _, message in summary["log"])
    # The Python replay ends where the Rust game did (net worth as both engines score it).
    expected = played[0]["saved_game"]["net_worth"]
    assert [summary["result"][f"Player {i + 1}"] for i in range(NUM_PLAYERS)] == expected

    # Stepping forward from a cached game matches a fresh replay to the same step.
    sr1 = next(m["step"] for m in summary["rounds"] if m["label"] == "Stock Round 1")
    later = min(len(actions), sr1 + 25)
    game_viewer.state_at(path, sr1, False)
    stepped = game_viewer.state_at(path, later, False)
    game_viewer._GAMES.clear()
    fresh = game_viewer.state_at(path, later, False)
    assert stepped == fresh
    assert fresh["step"] == later and fresh["last_action"] == actions[later - 1]
    assert {c["name"] for c in fresh["corporations"]} >= {"PRR", "B&O", "NYC"}
    assert len(fresh["players"]) == NUM_PLAYERS and len(fresh["hexes"]) > 90
    # Going back replays from the start.
    assert game_viewer.state_at(path, 0, False)["last_action"] is None
    # Steps are clamped to the game.
    assert game_viewer.state_at(path, 10**6, False)["step"] == len(actions)


def test_the_viewer_stops_where_the_engine_rejects_an_action(tmp_path, played):
    actions = list(played[0]["saved_game"]["raw_actions"])
    bad = len(actions) // 2
    actions.insert(bad, {"type": "par", "entity": 1, "entity_type": "player", "corporation": "NOPE"})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(actions))
    summary = game_viewer.game_summary(path)
    assert summary["playable_steps"] == bad and summary["error"].startswith(f"action {bad + 1} (par)")
    state = game_viewer.state_at(path, bad + 5, summary["auction_unlock"])
    assert state["step"] == bad and state["error"]


def test_action_highlights_name_the_hexes_an_action_touched():
    action = {
        "type": "run_routes",
        "routes": [{"train": "2-0", "connections": [["F24", "E23"]], "revenue": 40}],
    }
    assert game_viewer._action_hexes(None, action) == {
        "hexes": ["E23", "F24"],
        "routes": [{"train": "2-0", "connections": [["F24", "E23"]], "revenue": 40}],
    }
    assert game_viewer._action_hexes(None, {"type": "lay_tile", "hex": "D22"})["hexes"] == ["D22"]
    assert game_viewer._action_hexes(None, None) == {"hexes": [], "routes": []}


# ------------------------------------------------------------- dashboard routes
@pytest.fixture
def client(tmp_path, monkeypatch, played):
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)
    _h2h_collection(tmp_path, played[0]["saved_game"]["raw_actions"], start_positions="starts.jsonl")
    return dashboard.app.test_client()


def test_game_endpoints_list_and_step_through_games(client):
    collections = client.get("/api/game_collections").get_json()
    assert [c["id"] for c in collections] == ["logs/eval/h2h_test"]
    listing = client.get("/api/game_collection?collection=logs/eval/h2h_test").get_json()
    assert listing["kind"] == "head_to_head" and [g["name"] for g in listing["games"]] == ["7"]

    game = client.get("/api/game?collection=logs/eval/h2h_test&game=7").get_json()
    assert game["error"] is None and game["entry"]["seat_labels"][1] == "new@64"
    state = client.get("/api/game_state?collection=logs/eval/h2h_test&game=7&step=20").get_json()
    assert state["step"] == 20 and state["last_action"] == game["actions"][19]

    export = client.get("/api/game_export?collection=logs/eval/h2h_test&game=7")
    assert "attachment" in export.headers["Content-Disposition"]
    record = export.get_json()
    assert record["actions"] == game["actions"] and record["rl18xx"]["seat_labels"][1] == "new@64"

    for page in ["/games", "/games/view?collection=logs/eval/h2h_test&game=7"]:
        assert client.get(page).status_code == 200


@pytest.mark.parametrize(
    "url",
    [
        "/api/game_collection?collection=../../etc",
        "/api/game?collection=logs/eval/h2h_test&game=../settings",
        "/api/game?collection=/etc&game=passwd",
        "/api/game_state?collection=logs/eval/h2h_test&game=8",
        "/api/game_export?collection=logs/eval&game=7",
    ],
)
def test_game_endpoints_refuse_paths_outside_the_collections(client, url):
    assert client.get(url).status_code == 404


# -------------------------------------------------------------------- writers
def test_policy_gradient_games_are_saved_for_the_viewer(tmp_path, played):
    cfg = pg.PGConfig(policy_checkpoint="sl.pth", value_checkpoint="critic.pth")
    run_dir = tmp_path / "model_checkpoints_pg" / "pg_test"
    names = [pg.save_game(run_dir, record, 12, "learner/12.pth", "sl.pth", cfg) for record in played]
    games = game_records.list_games(run_dir)
    assert [g["name"] for g in games] == names
    for game, record in zip(games, played):
        assert game["update"] == 12 and game["opponent"] == "sl" and game["opponent_checkpoint"] == "sl.pth"
        learner = set(record["learner_seats"])
        assert game["sides"] == ["A" if s in learner else "B" for s in range(NUM_PLAYERS)]
        assert game["win_share"] == record["win_share"] and game["termination"] == "max_length"
        loaded = game_records.load_game_file(run_dir / game["game_file"])
        assert loaded.actions == record["saved_game"]["raw_actions"] and loaded.meta["update"] == 12
    assert game_records.describe_collection(run_dir)[0] == "games"  # no config.json in this bare test dir


def test_eval_policy_only_saves_the_first_games_of_a_match(tmp_path, monkeypatch, played):
    import eval_policy_only

    clients = {"a.pth": _Client(), "b.pth": _Client()}
    monkeypatch.setattr(pg, "_CLIENTS", clients)
    start_file = tmp_path / "starts.jsonl"
    start_file.write_text("")
    settings = {
        "servers": {"A": "a.pth", "B": "b.pth"},
        "temperatures": {"A": 1.0, "B": 1.0},
        "seed": 0,
        "start_positions": str(start_file),
        "random_start_fraction": 1.0,
        "max_decisions": 30,
        "price_eps": 0.0,
    }
    plain = eval_policy_only.play_games([0, 1], settings)
    assert not any("raw_actions" in r or "start" in r for r in plain)  # off by default: records unchanged
    records = eval_policy_only.play_games([0, 1, 2], {**settings, "save_games": 2})
    assert sorted(r["idx"] for r in records if "raw_actions" in r) == [0, 1]
    eval_policy_only.save_game_logs(tmp_path, 1, "new.pth", "old.pth", records)
    assert not any("raw_actions" in r for r in records)
    with (tmp_path / "games.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps({"match": "new.pth vs old.pth", **r}) + "\n")
    games = game_records.list_games(tmp_path)
    assert [g["name"] for g in games] == ["m1_0", "m1_1"]
    assert games[0]["seat_labels"] == ["new.pth" if s == "A" else "old.pth" for s in games[0]["sides"]]
    summary = game_viewer.game_summary(tmp_path / games[0]["game_file"])
    assert summary["error"] is None and summary["meta"]["match"] == "new.pth vs old.pth"


# --------------------------------------------------------------------- replay
def test_replay_prints_the_game_and_copies_outside_files_into_the_viewer(tmp_path, monkeypatch, capsys, played):
    from rl18xx.client import replay_game_from_log_file as replay

    actions = played[1]["saved_game"]["raw_actions"]
    outside = tmp_path / "somewhere" / "my game.json"
    outside.parent.mkdir()
    outside.write_text(json.dumps(actions))
    monkeypatch.setattr(replay, "REPO_ROOT", tmp_path)
    summary = replay.replay_locally(outside)
    collection, name = replay.add_to_viewer(outside, summary, tmp_path)
    assert (collection, name) == ("logs/games/replays", "my_game")
    assert game_records.load_game_file(tmp_path / collection / "games" / "my_game.json").actions == actions
    # A file already in a collection is viewed where it is.
    assert replay.add_to_viewer(tmp_path / collection / "games" / "my_game.json", summary, tmp_path) == (
        collection,
        "my_game",
    )
    url = replay.replay_game_from_log_file(str(tmp_path / collection / "games" / "my_game.json"), print_log=True)
    assert url.endswith("/games/view?collection=logs/games/replays&game=my_game")
    out = capsys.readouterr().out
    assert "-- Stock Round 1 --" in out and f"{len(actions)}/{len(actions)} actions replayed" in out


def test_replay_refuses_the_public_site():
    from rl18xx.client import replay_game_from_log_file as replay

    with pytest.raises(SystemExit):
        replay.replay_on_local_server("unused.json", "https://18xx.games")
