"""``titles.py``: per-title specs and the one game factory (1830 unchanged, new titles raw Rust)."""

import json

import pytest

import engine_rs
from rl18xx.agent.alphazero import start_positions
from rl18xx.agent.alphazero.titles import TITLE_SPECS, TitleSpec, layout_metadata, new_game, title_spec, titles
from rl18xx.rust_adapter import RustGameAdapter

NEW_TITLES = ("1867", "1822", "1822PNW", "1822MX", "1822CA")


def test_every_engine_title_has_a_spec():
    assert set(titles()) == set(engine_rs.supported_titles_py())
    assert titles()[0] == "1830"
    with pytest.raises(ValueError, match="unknown title"):
        title_spec("1899")


def test_player_ranges_follow_the_user_decisions():
    """3-6 players for new titles (1830: 2-6), within each title's own range: 1822PNW and 1822MX
    have no 6-player game in Ruby."""
    assert title_spec("1830").player_counts == (2, 3, 4, 5, 6)
    for title in NEW_TITLES:
        spec = title_spec(title)
        want = (3, 4, 5) if title in ("1822PNW", "1822MX") else (3, 4, 5, 6)
        assert spec.player_counts == want, title
        assert spec.eval_players == 4 and spec.eval_a_seats == (2, 1)
        with pytest.raises(ValueError, match="outside the pipeline's range"):
            new_game(title, 2)
        with pytest.raises(ValueError, match="outside the pipeline's range"):
            new_game(title, spec.max_players + 1)
        # The engine accepts every count the pipeline uses.
        for n in spec.player_counts:
            assert len(new_game(title, n).players) == n
    with pytest.raises(ValueError):
        TitleSpec(title="X", family="X", min_players=1, max_players=4)
    with pytest.raises(ValueError):
        TitleSpec(title="X", family="X", min_players=3, max_players=6, eval_players=2)


def test_1830_optional_rules_are_the_importers():
    from rl18xx.agent.alphazero.pretraining import ENGINE_OPTIONAL_RULES

    assert set(title_spec("1830").optional_rules) == set(ENGINE_OPTIONAL_RULES)
    for title in NEW_TITLES:
        assert title_spec(title).optional_rules == (), "new titles train on the base rules"


def test_data_paths():
    spec = title_spec("1830")
    for n in spec.player_counts:
        assert spec.start_positions(n) == start_positions.default_start_positions(n)
        assert spec.midgame_positions(n) == start_positions.default_midgame_positions(n)
    assert title_spec("1867").start_positions(5) == "human_games/start_positions_1867_5p.jsonl"
    assert "human_games/scrape_2026_10/1867" in title_spec("1867").human_games_dirs
    with pytest.raises(ValueError):
        title_spec("1867").start_positions(2)


def _state(game):
    """A comparable snapshot of a fresh game: seating, cash, round, legal moves, the rule switch."""
    return (
        [(p.id, p.name, p.cash) for p in game.players],
        game.round.round_type,
        game.factored_legal_indices(),
        game.no_buyback_rule,
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_1830_games_are_built_as_before(n):
    game = new_game("1830", n)
    old = start_positions.new_game(n)
    assert isinstance(game, RustGameAdapter)
    assert _state(game._game) == _state(old._game)
    assert game._game.no_buyback_rule and not game._game.auction_unlock
    # The 1830 knobs pass through.
    assert new_game("1830", n, no_buyback_rule=False)._game.no_buyback_rule is False
    assert new_game("1830", n, auction_unlock=True)._game.auction_unlock is True


def test_1830_optional_rules_reach_the_engine():
    game = new_game("1830", 4, optional_rules=["optional_6_train"])
    assert game._game.optional_rules == ["optional_6_train"]
    assert game._game.no_buyback_rule
    with pytest.raises(ValueError, match="not used by the pipeline"):
        new_game("1830", 4, optional_rules=["no_such_rule"])


@pytest.mark.parametrize("title", NEW_TITLES)
@pytest.mark.parametrize("n", [3, 5])
def test_new_titles_build_raw_rust_games(title, n):
    game = new_game(title, n, seed=7)
    assert isinstance(game, engine_rs.BaseGame)
    assert game.title == title
    spec = title_spec(title)
    assert [(p.id, p.name) for p in game.players] == list(spec.player_names(n).items())
    with pytest.raises(ValueError, match="1830 rule variants"):
        new_game(title, n, auction_unlock=True)
    with pytest.raises(ValueError, match="1830 rule variants"):
        new_game(title, n, no_buyback_rule=True)
    # 1867 has its generator layout; the 1822 family has none yet.
    if title == "1867":
        assert spec.layout_ready()
        assert spec.status() == {"title": "1867", "layout_ready": True, "layout_kind": "generator", "encoder_ready": False}
    else:
        assert not spec.layout_ready()
        assert spec.status()["layout_kind"] == "none"


@pytest.mark.parametrize("title", NEW_TITLES)
def test_new_title_games_step_through_their_generators(title):
    """A new title's game from the factory is driven by its own generator (wire dicts)."""
    game = new_game(title, 4, seed=3)
    moves_json = game.g1867_legal_moves_json if title == "1867" else game.g1822_legal_moves_json
    for _ in range(25):
        if game.finished:
            break
        moves = json.loads(moves_json())
        assert moves, f"{title}: no legal moves"
        game.process_action(moves[0]["wire"])
    assert len(game.raw_actions) >= 25 or game.finished


def test_1830_status():
    spec = title_spec("1830")
    assert spec.layout_ready()
    assert spec.status() == {"title": "1830", "layout_ready": True, "layout_kind": "factored", "encoder_ready": True}
    assert set(TITLE_SPECS) == set(titles())


def test_layout_metadata():
    """The layout a title's records are indexed with (design S1: written, not enforced)."""
    assert layout_metadata("1830") == {
        "title": "1830",
        "layout_kind": "factored",
        "layout_version": "1830-v1",
        "layout_fingerprint": "e721aa6520116e2a",
    }
    meta = layout_metadata("1867")
    assert meta["layout_kind"] == "generator" and meta["layout_version"] == "1867-v1"
    assert meta["layout_fingerprint"] == engine_rs.rl_layout_info("1867")["fingerprint"]
    for title in ("1822", "1822PNW", "1822MX", "1822CA"):
        assert layout_metadata(title) is None
    with pytest.raises(ValueError):
        layout_metadata("1899")


def test_game_records_carry_the_title_and_layout():
    from rl18xx.agent.alphazero.game_records import make_game_record

    # 1830 records are unchanged: no layout key, the 1830 title.
    r30 = make_game_record([{"type": "pass", "entity": 1}], 4, meta={"name": "x"})
    assert r30["title"] == "1830" and "layout" not in r30["rl18xx"]
    assert [p["id"] for p in r30["players"]] == [1, 2, 3, 4] and r30["settings"]["seed"] == 0
    r67 = make_game_record([], 5, title="1867", seed=9)
    assert r67["title"] == "1867" and r67["settings"]["seed"] == 9
    assert r67["rl18xx"]["layout"] == layout_metadata("1867")
    assert [p["name"] for p in r67["players"]] == [f"Player {i}" for i in range(1, 6)]
    r22 = make_game_record([], 3, title="1822")
    assert r22["rl18xx"]["layout"] is None and r22["players"][0]["id"] == 1001
