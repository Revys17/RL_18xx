"""1867 fixture-replay gate (Phase 1 validation strategy, leg 1).

Two layers:

1. ``test_fixture_prefix_frontier`` — the RATCHET. Each fixture must replay
   at least as far as the pinned watermark. All four fixtures now replay
   END TO END, so the watermarks sit at the full action counts — any
   rejection anywhere is red.
2. ``test_fixture_replays_to_recorded_result`` — the FINISH LINE: every
   action accepted and exact final scores. PASSING since the par-ladder +
   loan-valuation seams (2026-06-13); the former xfail marker is gone.
"""

import json
from pathlib import Path

import pytest

from tests.title_replay_harness import (
    ReplayError,
    TitleUnsupported,
    check_fixture,
    replay_game,
    supported_titles,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "1867"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))

# fixture stem -> minimum number of actions the engine must accept.
# ALL FOUR fixtures replay END TO END since the class-typed par ladder
# (majors par on the z+p cells $70-200 — get_all_par_prices, NOT
# phase-gated; pars pay the corp under incremental cap and queue the SR
# home-token choice) and the loan valuation/settlement (player_value
# values loan-carrying corps one price-step left per loan; end_game
# really moves the prices) landed. The watermarks are the FULL filtered
# action counts — any rejection anywhere is a regression.
PREFIX_WATERMARK = {
    "21268": 736,
    "hs_ahjzadkh_19792": 220,
    "hs_wuveadew_21268": 736,
    "nationalization_cash": 720,
}


def test_fixtures_are_vendored():
    assert len(FIXTURES) >= 4, "1867 fixtures missing from tests/fixtures/1867"


@pytest.mark.skipif(
    "1867" not in supported_titles(),
    reason="engine does not register the 1867 title yet (Phase 1 in progress)",
)
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_prefix_frontier(path):
    game = json.loads(path.read_text())
    try:
        report = replay_game(game)
        applied, first_rejection = report.actions_applied, None
    except ReplayError as exc:
        applied, first_rejection = exc.step, exc
    watermark = PREFIX_WATERMARK[path.stem]
    assert applied >= watermark, (
        f"replay frontier regressed: {applied} < watermark {watermark}; "
        f"first rejection: {first_rejection}"
    )


@pytest.mark.skipif(
    "1867" not in supported_titles(),
    reason="engine does not register the 1867 title yet (Phase 1 in progress)",
)
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_route_revenue_cross_check(path):
    """The engine prices every 1867 route itself (recorded run_routes may
    carry connections WITHOUT revenue) — so wherever the record DOES carry
    revenue, the engine's computation must reproduce it exactly. This is
    the per-route oracle for the revenue rules (towns fill spare capacity,
    offboard phase tiers, multipliers, hex_bonus, capitals+Timmins).

    Also holds 3668/3668 over the first 300 human-corpus games
    (2026-06-12 sweep) — re-run that sweep when revenue rules change.
    """
    game = json.loads(path.read_text())
    mismatches = []

    def check(rust, i, action):
        if action.get("type") != "run_routes":
            return
        for r in action.get("routes", []):
            if "revenue" in r and r.get("connections"):
                got = rust.route_revenue_py(
                    action["entity"], r["train"], r["connections"]
                )
                if got != r["revenue"]:
                    mismatches.append(
                        (i, r["train"], got, r["revenue"], r.get("revenue_str"))
                    )

    try:
        replay_game(game, on_action=check)
    except ReplayError:
        pass  # the frontier ratchet covers replay depth; we check what we reach
    assert not mismatches, f"computed route revenue diverges: {mismatches}"


@pytest.mark.skipif(
    "1867" not in supported_titles(),
    reason="engine does not register the 1867 title yet (Phase 1 in progress)",
)
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_replays_to_recorded_result(path):
    try:
        report = check_fixture(path)
    except TitleUnsupported as exc:  # pragma: no cover - guarded by skipif
        pytest.skip(str(exc))
    assert report.actions_applied == report.actions_total
