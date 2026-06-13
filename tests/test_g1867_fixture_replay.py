"""1867 fixture-replay gate (Phase 1 validation strategy, leg 1).

Two layers while Phase 1 lands mechanic by mechanic:

1. ``test_fixture_prefix_frontier`` — the RATCHET. Each fixture must replay
   at least as far as the pinned watermark (the action index where the
   engine currently stops, at the boundary of the next unimplemented
   mechanic). Landing a mechanic moves the frontier forward: bump the
   watermark in the same commit. A regression below the watermark is red.
2. ``test_fixture_replays_to_recorded_result`` — the FINISH LINE: every
   action accepted and exact final scores. xfail (non-strict) until the
   full ruleset lands; drop the marker when it first passes.
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
# Current frontier: REDEEM SHARES + the X3→X6 brown-Montreal token merge
# replay (RedeemShares pc before Track consuming corp buy_shares/pass;
# Ruby city_map_for exit-SUBSET token transfer; the 1-D sold-out bump
# up==right). The fixtures' MajorTrainless `choose` actions now replay.
# Verified exact against Ruby: 21268 cash/bank at file id 857, L12 city
# layout at 619; nationalization_cash cash/bank/CN tokens at 555.
# hs_ahjzadkh replays COMPLETELY (scores still diverge on the endgame loan
# valuation — the xfail below); the other three stop on a `par` at $200 —
# the phase-gated PAR_PRICE_GRID (Ruby G1867 par prices 135-200 unlock by
# phase) is the next seam.
PREFIX_WATERMARK = {
    "21268": 626,
    "hs_ahjzadkh_19792": 220,
    "hs_wuveadew_21268": 626,
    "nationalization_cash": 600,
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


@pytest.mark.xfail(
    reason="1867 Phase 1 in progress — mechanics landing seam by seam",
    strict=False,
)
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
