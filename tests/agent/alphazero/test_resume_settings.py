"""A resume changes only the settings passed on its command line (the rest stay the run's)."""

import importlib.util
from pathlib import Path

import pytest

from rl18xx.agent.alphazero import policy_gradient as pg
from rl18xx.agent.alphazero import search_learning as sl

MAIN = Path(__file__).resolve().parents[3] / "main.py"


def _sl(**kwargs):
    return sl.SearchLearningConfig(policy_checkpoint="p.pth", value_checkpoint="", **kwargs)


def _pg(**kwargs):
    return pg.PGConfig(policy_checkpoint="p.pth", value_checkpoint="", **kwargs)


@pytest.fixture(scope="module")
def main_module():
    spec = importlib.util.spec_from_file_location("rl18xx_main_cli", MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_options_lists_only_the_flags_given(main_module):
    argv = ["search-learning", "--resume", "runs/x", "--max-updates", "9", "--midgame-fraction=0.3", "--dedupe-maps"]
    assert main_module.explicit_options(argv) == {"resume", "max_updates", "midgame_fraction", "dedupe_maps"}
    # an option given at its default value still counts (a resume can set a value back to the default)
    assert main_module.explicit_options(["policy-gradient", "--resume", "r", "--auction-fraction", "0"]) == {
        "resume",
        "auction_fraction",
    }
    assert main_module.explicit_options(["search-learning", "--resume", "r"]) == {"resume"}


def test_a_flag_left_out_keeps_the_run_value():
    saved = {"max_updates": 100, "midgame_fraction": 0.3, "midgame_positions": "mid.jsonl", "auction_fraction": 0.15,
             "full_move_temperature": 1.0, "explore_decisions": 120, "dedupe_maps": True, "workers": 40}
    cfg = _sl(max_updates=500, midgame_fraction=0.0, midgame_positions=None, auction_fraction=0.0,
                                  full_move_temperature=0.0, explore_decisions=0, dedupe_maps=False, workers=24)
    assert pg.resume_overrides(cfg, saved, sl.RESUME_SETTINGS, explicit={"resume", "max_updates"}) == {
        "max_updates": 500
    }
    merged = {**saved, **pg.resume_overrides(cfg, saved, sl.RESUME_SETTINGS, explicit={"max_updates", "workers"})}
    assert merged["midgame_fraction"] == 0.3 and merged["explore_decisions"] == 120 and merged["dedupe_maps"]
    assert merged["workers"] == 24 and merged["max_updates"] == 500


def test_midgame_fraction_alone_keeps_or_finds_a_position_file():
    cfg = _sl(midgame_fraction=0.5, midgame_positions="default.jsonl")
    keep = pg.resume_overrides(cfg, {"midgame_positions": "mid.jsonl"}, sl.RESUME_SETTINGS, {"midgame_fraction"})
    assert keep == {"midgame_fraction": 0.5, "midgame_positions": "mid.jsonl"}
    fresh = pg.resume_overrides(cfg, {"midgame_positions": None}, sl.RESUME_SETTINGS, {"midgame_fraction"})
    assert fresh == {"midgame_fraction": 0.5, "midgame_positions": "default.jsonl"}
    off = _sl(midgame_fraction=0.0, midgame_positions="default.jsonl")
    assert pg.resume_overrides(off, {"midgame_positions": "mid.jsonl"}, sl.RESUME_SETTINGS, {"midgame_fraction"}) == {
        "midgame_fraction": 0.0,
        "midgame_positions": None,
    }


def test_without_a_command_line_every_setting_but_the_learning_rates_changes():
    cfg = _sl(max_updates=7, lr=1.0, critic_lr=2.0, policy_follow=None)
    overrides = pg.resume_overrides(cfg, {}, sl.RESUME_SETTINGS)
    assert overrides["max_updates"] == 7
    assert "lr" not in overrides and "critic_lr" not in overrides and "policy_follow" not in overrides
    pg_cfg = _pg(max_updates=3, population=None, value_follow="runs/v")
    pg_overrides = pg.resume_overrides(pg_cfg, {}, pg.RESUME_SETTINGS)
    assert pg_overrides["max_updates"] == 3 and pg_overrides["value_follow"] == "runs/v"
    assert "population" not in pg_overrides
