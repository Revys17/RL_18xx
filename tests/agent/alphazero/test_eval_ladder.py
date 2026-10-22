"""scripts/eval_ladder.py on synthetic ledgers: rating recovery, start-clustered SEs, side balancing, the
fixed-panel decision, ledger import, and the scheduler end to end with a fake eval (no GPU, no network)."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("eval_ladder", REPO_ROOT / "scripts" / "eval_ladder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _load()
SETTINGS = {
    "seed": 0,
    "start_positions": "human_games/start_positions_1830_4p_v2.jsonl",
    "random_start_fraction": 0.2,
    "softpick": 0,
    "max_length": 1000,
    "stall": 400,
}


def simulate(
    rng,
    a,
    b,
    mode,
    strength,
    n_starts,
    side_bonus=0.0,
    seat=(0.0, 0.0, 0.0, 0.0),
    start_effect=None,
    effect_player=None,
    kappa=40.0,
):
    """Game records of ``a`` (side A) vs ``b`` under the four-seat Luce model: seat i wins with probability
    proportional to exp(strength + seat effect + side bonus (+ a per-start effect on ``effect_player``'s seats));
    net-worth shares are Dirichlet with that mean."""
    games = []
    for s in range(n_starts):
        for seats in L.h2h.arrangements(L.MODES[mode]):
            logits = np.array(
                [
                    strength[a if x == "A" else b] + seat[i] + (side_bonus if x == "A" else 0.0)
                    for i, x in enumerate(seats)
                ]
            )
            if start_effect is not None:
                logits += [start_effect[s] if (a if x == "A" else b) == effect_player else 0.0 for x in seats]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            win = np.eye(4)[rng.choice(4, p=p)]
            games.append([f"human:{s}", "".join(seats), win.tolist(), rng.dirichlet(kappa * p).tolist()])
    return games


def record(candidate, opponent, mode, side, games, role="panel", updates=None):
    updates = updates or {}
    a, b = (candidate, opponent) if side == "A" else (opponent, candidate)
    a_scores = [sum(w for w, s in zip(g[2], g[1]) if s == "A") for g in games]
    return {
        "key": L.match_key(candidate, opponent, mode, a),
        "name": f"{candidate}_vs_{opponent}_{mode}_{side}",
        "mode": mode,
        "candidate": candidate,
        "candidate_update": updates.get(candidate),
        "candidate_side": side,
        "opponent": opponent,
        "opponent_update": updates.get(opponent),
        "opponent_role": role,
        "a": a,
        "b": b,
        "a_spec": f"spec:{a}",
        "b_spec": f"spec:{b}",
        "settings": dict(SETTINGS),
        "games": len(games),
        "a_score": float(np.mean(a_scores)),
        "game_records": games,
    }


def ledger(rng, strength, candidates, panel, modes=("2v2", "1v3"), n_starts=40, both_ways=True, **kw):
    """Every candidate against every panel member in every mode, both ways (or the candidate always side A)."""
    updates = {**candidates, **{p: (0 if p == "u0" else None) for p in panel}}
    out = []
    for c in candidates:
        for p in panel:
            for mode in modes:
                for side in ("A", "B") if both_ways else ("A",):
                    a, b = (c, p) if side == "A" else (p, c)
                    out.append(
                        record(c, p, mode, side, simulate(rng, a, b, mode, strength, n_starts, **kw), updates=updates)
                    )
    return out


STRENGTH = {"u0": 0.0, "ref": -0.6, "sl": 0.3, "u300": 0.2, "u600": 0.5, "u900": -0.1}
CANDIDATES = {"u300": 300, "u600": 600, "u900": 900}
PANEL = ["u0", "ref", "sl"]


def _assert_recovered(fit, strength, z=4.0):
    for name, true in strength.items():
        if name == fit.anchor:
            assert fit.rating[name] == 0.0
            continue
        est, se = fit.rating[name], fit.se(name)
        assert abs(est - (true - strength[fit.anchor])) < z * se, (fit.label, name, est, true, se)


def test_rating_recovery_all_three_fits():
    rng = np.random.default_rng(0)
    seat = (0.1, 0.0, -0.05, -0.05)
    records = ledger(rng, STRENGTH, CANDIDATES, PANEL, seat=seat)
    fits = L.fit_all(records, anchor="u0")
    assert set(fits) == {"2v2", "1v3", "worth"}
    for fit in fits.values():
        assert isinstance(fit, L.RatingFit), fit
        _assert_recovered(fit, STRENGTH)
        # seats 1-3 against seat 0
        for (est, se), true in zip(fit.seat, np.array(seat[1:]) - seat[0]):
            assert abs(est - true) < 4 * se
        assert fit.n_clusters == 40
    # the fits are separate: 2v2 and 1v3 see their own games, worth sees both modes
    for mode in ("2v2", "1v3"):
        assert fits[mode].n_games == sum(r["games"] for r in records if r["mode"] == mode)
    assert fits["worth"].n_games == sum(r["games"] for r in records)
    # net-worth shares are far less noisy than wins
    assert fits["worth"].se("u600") < fits["2v2"].se("u600") / 3
    report = L.format_report(records, fits)
    assert "u600" in report and "(anchor)" in report


def test_rating_scale_matches_two_by_two_and_one_by_three_odds():
    """One rating gap predicts both 2v2 (sigmoid(d)) and 1v3 (e^d / (e^d + 3)) scores."""
    rng = np.random.default_rng(1)
    strength = {"u0": 0.0, "u300": 0.7}
    records = ledger(rng, strength, {"u300": 300}, ["u0"], n_starts=150)
    for label, fit in L.fit_all(records, anchor="u0").items():
        if label != "worth":
            _assert_recovered(fit, strength)
    a_scores = {r["mode"]: r["a_score"] for r in records if r["candidate_side"] == "A"}
    assert abs(a_scores["2v2"] - 1 / (1 + math.exp(-0.7))) < 0.05
    assert abs(a_scores["1v3"] - math.exp(0.7) / (math.exp(0.7) + 3)) < 0.05


def test_disconnected_player_is_dropped():
    rng = np.random.default_rng(2)
    records = ledger(rng, STRENGTH, {"u300": 300}, ["u0"], modes=("2v2",), n_starts=10)
    records.append(record("x", "y", "2v2", "A", simulate(rng, "x", "y", "2v2", {"x": 0, "y": 0}, 10)))
    fit = L.fit_ratings(L.game_table(records, ("2v2",), "win"), anchor="u0")
    assert sorted(fit.dropped) == ["x", "y"]
    assert set(fit.rating) == {"u0", "u300"}


def test_clustered_se_tracks_the_spread_when_starts_are_correlated():
    """A per-start advantage shared by every arrangement and both directions of a start: games of one start are
    correlated, the start-clustered SE matches the estimate's spread across replicates, the iid SE is too small."""
    rng = np.random.default_rng(3)
    strength = {"ref": 0.0, "u300": 0.2}
    estimates, se, se_iid = [], [], []
    for _ in range(80):
        effect = rng.normal(0.0, 1.0, size=40)
        records = ledger(
            rng,
            strength,
            {"u300": 300},
            ["ref"],
            modes=("2v2",),
            n_starts=40,
            start_effect=effect,
            effect_player="u300",
        )
        fit = L.fit_ratings(L.game_table(records, ("2v2",), "win"), anchor="ref")
        estimates.append(fit.rating["u300"])
        se.append(fit.se("u300"))
        se_iid.append(fit.se("u300", iid=True))
    spread = np.std(estimates, ddof=1)
    assert 0.75 < np.mean(se) / spread < 1.35, (np.mean(se), spread)
    assert np.mean(se_iid) < 0.75 * spread, (np.mean(se_iid), spread)


def test_clustered_se_close_to_iid_without_start_effects():
    rng = np.random.default_rng(4)
    records = ledger(rng, {"ref": 0.0, "u300": 0.2}, {"u300": 300}, ["ref"], modes=("2v2",), n_starts=200)
    fit = L.fit_ratings(L.game_table(records, ("2v2",), "win"), anchor="ref")
    assert 0.8 < fit.se("u300") / fit.se("u300", iid=True) < 1.25


def test_start_cluster_merges_one_human_start_drawn_twice():
    assert L.start_cluster("human:228367", 0, 3) == L.start_cluster("human:228367", 0, 57)
    assert L.start_cluster("random", 0, 3) != L.start_cluster("random", 0, 4)
    assert L.start_cluster("random", 0, 3) != L.start_cluster("random", 1, 3)
    # a match's own clustered SE: identical values within each cluster -> SE of the cluster means
    values = [1.0] * 6 + [0.0] * 6
    clusters = ["a"] * 6 + ["b"] * 6
    assert abs(L.clustered_mean_se(values, clusters) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Side balancing


def _config(tmp_path, **overrides):
    cfg = {
        "name": "lad",
        "run": "run",
        "spec": "{checkpoint}@200/0.05:mean,ptemp=1",
        "every": 300,
        "games": 100,
        "modes": ["2v2", "1v3"],
        "panel": [
            {"name": "u0", "update": 0},
            {"name": "gen4critic", "spec": "g.pth+c.pth@200/0.05:mean,ptemp=0.5"},
            {"name": "solo", "spec": "s.pth@200", "modes": ["1v3"], "games": 40},
        ],
        **overrides,
    }
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(cfg))
    return L.load_config(path), path


def test_plan_plays_every_pairing_both_ways(tmp_path):
    cfg, _ = _config(tmp_path, self={"every": 1800, "count": 1, "modes": ["2v2"]})
    matches = L.plan_matches(cfg, 3600, available=range(0, 3601, 300))
    assert len({m.key for m in matches}) == len(matches)
    groups = {}
    for m in matches:
        groups.setdefault((m.opponent.name, m.mode), []).append(m)
        assert {m.a.name, m.b.name} == {"u3600", m.opponent.name}
        assert m.candidate.spec == "run/learner/3600.pth@200/0.05:mean,ptemp=1"
    assert set(groups) == {
        ("u0", "2v2"),
        ("u0", "1v3"),
        ("gen4critic", "2v2"),
        ("gen4critic", "1v3"),
        ("solo", "1v3"),
        ("u1800", "2v2"),
    }
    for (opp, mode), pair in groups.items():
        assert sorted(m.candidate_side for m in pair) == ["A", "B"]
        assert pair[0].games == pair[1].games
        assert pair[0].games == {
            ("solo", "1v3"): 40,
            ("u0", "2v2"): 102,
            ("gen4critic", "2v2"): 102,
            ("u1800", "2v2"): 102,
        }.get((opp, mode), 100)
    roles = {m.opponent.name: m.opponent.role for m in matches}
    assert roles == {"u0": "panel", "gen4critic": "panel", "solo": "panel", "u1800": "self"}
    # one self member, at least ``every`` behind: none yet at u3000, u1800 at u3600 and u4800
    assert not L.self_members(cfg, 3000, range(0, 3001, 300))
    assert [m.name for m in L.self_members(cfg, 4800, range(0, 4801, 300))] == ["u1800"]
    assert [m.name for m in L.self_members(cfg, 5400, range(0, 5401, 300))] == ["u3600"]


def test_both_ways_cancels_a_listing_bias():
    """A planted side-A bonus: with every pairing played both ways the ratings come out unbiased without modelling
    it, the listing-effect diagnostic measures it, and the optional side term recovers it."""
    rng = np.random.default_rng(5)
    bonus = 0.4
    records = ledger(rng, STRENGTH, CANDIDATES, PANEL, side_bonus=bonus)
    for fit in L.fit_all(records, anchor="u0").values():
        _assert_recovered(fit, STRENGTH)
    effect, se, pairings = L.listing_effect(records)
    assert pairings == len(CANDIDATES) * len(PANEL)
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    expected = np.mean(
        [
            (sig(STRENGTH[c] - STRENGTH[p] + bonus) + sig(STRENGTH[p] - STRENGTH[c] + bonus) - 1) / 2
            for c in CANDIDATES
            for p in PANEL
        ]
    )
    assert abs(effect - expected) < 4 * se
    fit = L.fit_ratings(L.game_table(records, ("2v2", "1v3"), "win"), anchor="u0", side_effect=True)
    assert abs(fit.side_a[0] - bonus) < 4 * fit.side_a[1]
    _assert_recovered(fit, STRENGTH)


def test_one_way_matches_carry_the_listing_bias():
    """The old ladder's design (candidate always side A): the same bonus lands in every candidate's rating."""
    rng = np.random.default_rng(6)
    bonus = 0.4
    records = ledger(rng, STRENGTH, CANDIDATES, PANEL, both_ways=False, side_bonus=bonus)
    fit = L.fit_all(records, anchor="u0")["2v2"]
    bias = [fit.rating[c] - STRENGTH[c] for c in CANDIDATES]
    assert all(b > 3 * fit.se(c) for b, c in zip(bias, CANDIDATES))
    assert abs(np.mean(bias) - bonus) < 0.15
    assert L.listing_effect(records)[2] == 0
    assert len(L.directions(records)) == len(CANDIDATES) * len(PANEL) * 2


# ---------------------------------------------------------------------------
# Decision


def _decision_ledger(rng, drop_1v3_ref=False, with_self=True):
    strength = {"u0": 0.0, "ref": -0.5, "u600": 0.0, "u1200": 0.35, "u1800": 0.35}
    records = ledger(rng, strength, {"u600": 600, "u1200": 1200, "u1800": 1800}, ["u0", "ref"], n_starts=60)
    if drop_1v3_ref:
        records = [
            r for r in records if not (r["candidate"] == "u1800" and r["opponent"] == "ref" and r["mode"] == "1v3")
        ]
    if with_self:
        # run-vs-run matches that say the opposite (u1200 loses every game to u600): the decision must ignore them
        crushed = {"u1200": -8.0, "u600": 0.0}
        for side in ("A", "B"):
            a, b = ("u1200", "u600") if side == "A" else ("u600", "u1200")
            records.append(
                record(
                    "u1200",
                    "u600",
                    "2v2",
                    side,
                    simulate(rng, a, b, "2v2", crushed, 60),
                    role="self",
                    updates={"u1200": 1200, "u600": 600},
                )
            )
    return records


def test_decide_uses_the_fixed_panel_only():
    rng = np.random.default_rng(7)
    records = _decision_ledger(rng)
    results = {r["fit"]: r for r in L.decide(records, back=1, latest=1200)}
    for label in ("2v2", "1v3", "worth"):
        r = results[label]
        assert (r["latest"], r["earlier"]) == (1200, 600)
        assert r["opponents"] == ["ref", "u0"]
        assert abs(r["diff"] - 0.35) < 4 * r["se"], r
        assert r["verdict"] == "above", r
    # the self matches changed nothing
    no_self = {
        r["fit"]: r for r in L.decide([r for r in records if r["opponent_role"] == "panel"], back=1, latest=1200)
    }
    for label in results:
        assert results[label]["diff"] == pytest.approx(no_self[label]["diff"], abs=1e-12)
    # ...though the full fit sees them
    full = L.fit_all(records, anchor="u0")["2v2"]
    assert full.rating["u1200"] < full.rating["u600"]
    # equal strength: no verdict
    same = {r["fit"]: r for r in L.decide(records, back=1)}
    assert same["2v2"]["latest"] == 1800 and same["2v2"]["earlier"] == 1200
    for r in same.values():
        assert abs(r["z"]) < 3
    text = L.format_decision(L.decide(records, back=1, latest=1200))
    assert "ABOVE" in text and "vs ref" in text


def test_decide_compares_on_common_panel_members():
    rng = np.random.default_rng(8)
    records = _decision_ledger(rng, drop_1v3_ref=True, with_self=False)
    results = {r["fit"]: r for r in L.decide(records, back=1)}
    assert results["1v3"]["opponents"] == ["u0"] and results["1v3"]["missing"] == ["ref"]
    assert results["2v2"]["opponents"] == ["ref", "u0"] and results["2v2"]["missing"] == []
    assert L.decide(records, back=5)[0]["error"].startswith("no candidate 5 back")
    explicit = L.decide(records, latest=1800, earlier=600)[0]
    assert (explicit["latest"], explicit["earlier"]) == (1800, 600)
    # a pairing played one way only is flagged
    one_way = [
        r
        for r in records
        if not (
            r["candidate"] == "u1800" and r["opponent"] == "u0" and r["mode"] == "2v2" and r["candidate_side"] == "B"
        )
    ]
    flagged = {r["fit"]: r for r in L.decide(one_way, back=1)}
    assert flagged["2v2"]["one_way"] == ["u1800 vs u0 2v2"]


# ---------------------------------------------------------------------------
# Ledger I/O and import


def _write_eval_dir(path, a_spec, b_spec, a_seats=2, n_games=24, settings=None, truncate=False):
    seatings = L.h2h.arrangements(a_seats)
    path.mkdir(parents=True)
    (path / "settings.json").write_text(
        json.dumps(
            {
                **SETTINGS,
                **(settings or {}),
                "readouts": 64,
                "players": {a_spec: {}, b_spec: {}},
                "arrangements": seatings,
                "matches": [[a_spec, b_spec]],
                "games_per_match": n_games,
            }
        )
    )
    rng = np.random.default_rng(9)
    lines = []
    for g in range(n_games):
        start = g // len(seatings)
        w = int(rng.integers(4))
        lines.append(
            json.dumps(
                {
                    "game_idx": g,
                    "seats": list(seatings[g % len(seatings)]),
                    "termination": "finished",
                    "engine_moves": 400,
                    "decision_count": 300,
                    "decisions": {"stock": 150},
                    "start": "human:77" if start in (0, 2) else "random",
                    "net_worth": [1000.0] * 4,
                    "worth_share": [0.25] * 4,
                    "win_share": [float(i == w) for i in range(4)],
                    "match": 0,
                }
            )
        )
    text = "\n".join(lines) + "\n"
    if truncate:
        text += '{"game_idx": 99, "seats": ["A"'
    (path / "games.jsonl").write_text(text)


def test_read_match_clusters_and_scores(tmp_path):
    _write_eval_dir(tmp_path / "m", "x@200", "y@200", a_seats=1, n_games=16, truncate=True)
    rec = L.read_match(tmp_path / "m")
    assert rec["mode"] == "1v3" and rec["games"] == 16 and rec["a_score_equal"] == 0.25
    clusters = [g[0] for g in rec["game_records"]]
    # start = game_idx // 4 in 1v3; starts 0 and 2 drew the same human game
    assert clusters[0:4] == ["human:77"] * 4 and clusters[8:12] == ["human:77"] * 4
    assert clusters[4] == "random:0:1" and clusters[12] == "random:0:3"
    a_scores = [sum(w for w, s in zip(g[2], g[1]) if s == "A") for g in rec["game_records"]]
    assert rec["a_score"] == pytest.approx(np.mean(a_scores))
    assert rec["settings"] == SETTINGS


def test_import_names_players_and_checks_settings(tmp_path):
    cfg, _ = _config(tmp_path)
    led = tmp_path / "ledger.jsonl"
    u600, u300, gen4 = L.run_spec(cfg, 600), L.run_spec(cfg, 300), cfg["panel"][1]["spec"]
    _write_eval_dir(tmp_path / "e" / "u600_vs_gen4", u600, gen4)
    _write_eval_dir(tmp_path / "e" / "u600_vs_u300", u600, u300)
    _write_eval_dir(tmp_path / "e" / "u0_first", L.run_spec(cfg, 0), u300, a_seats=1, n_games=8)
    _write_eval_dir(tmp_path / "e" / "softpick30", u600, gen4, settings={"softpick": 30})
    _write_eval_dir(tmp_path / "e" / "unknown", u600, "other.pth@200")
    dirs = sorted((tmp_path / "e").iterdir())
    assert L.import_dirs(cfg, led, dirs) == 3
    records = {r["name"]: r for r in L.read_ledger(led)}
    assert set(records) == {"u600_vs_gen4", "u600_vs_u300", "u0_first"}
    r = records["u600_vs_gen4"]
    assert (r["candidate"], r["candidate_update"], r["candidate_side"]) == ("u600", 600, "A")
    assert (r["opponent"], r["opponent_role"]) == ("gen4critic", "panel")
    r = records["u600_vs_u300"]
    assert (r["candidate"], r["opponent"], r["opponent_role"], r["opponent_update"]) == ("u600", "u300", "self", 300)
    r = records["u0_first"]  # the panel member listed first: the run checkpoint is the candidate, side B
    assert (r["candidate"], r["candidate_side"], r["opponent"], r["mode"]) == ("u300", "B", "u0", "1v3")
    assert L.import_dirs(cfg, led, dirs) == 0  # already in the ledger
    assert len(L.read_ledger(led)) == 3


def test_ledger_rejects_a_renamed_spec(tmp_path):
    led = tmp_path / "ledger.jsonl"
    rng = np.random.default_rng(10)
    rec = record("u300", "u0", "2v2", "A", simulate(rng, "u300", "u0", "2v2", {"u300": 0, "u0": 0}, 2))
    L.append_ledger(led, rec)
    with pytest.raises(ValueError, match="u300"):
        L.append_ledger(led, {**rec, "key": "other", "a_spec": "spec:something else"})
    with open(led, "a") as f:
        f.write('{"cut short')
    assert len(L.read_ledger(led)) == 1


# ---------------------------------------------------------------------------
# The scheduler, end to end, with a fake eval_head_to_head.py

FAKE_EVAL = """#!{python}
import argparse, itertools, json, os, random, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("script")
p.add_argument("--match", nargs=2)
for flag in ("--games", "--a-seats", "--seed", "--softpick", "--max-length", "--stall", "--workers"):
    p.add_argument(flag, type=int)
p.add_argument("--random-start-fraction", type=float)
p.add_argument("--start-positions")
p.add_argument("--out")
args, _ = p.parse_known_args()
out = Path(args.out)
out.mkdir(parents=True)
seatings = [tuple("A" if s in c else "B" for s in range(4)) for c in itertools.combinations(range(4), args.a_seats)]
(out / "settings.json").write_text(json.dumps({{
    "seed": args.seed, "start_positions": args.start_positions, "random_start_fraction": args.random_start_fraction,
    "softpick": args.softpick, "max_length": args.max_length, "stall": args.stall, "arrangements": seatings,
    "matches": [args.match], "workers": args.workers}}))
fail = Path(os.environ.get("FAKE_EVAL_FAIL", "/nonexistent"))
if os.environ.get("FAKE_EVAL_BROKEN", "/nonexistent") in args.out:
    sys.exit(4)  # this match always fails
rng = random.Random(args.out)
with open(out / "games.jsonl", "w") as f:
    for g in range(args.games):
        if g == 3 and fail.exists():
            fail.unlink()
            sys.exit(3)  # crash mid-match once
        w = rng.randrange(4)
        f.write(json.dumps({{"game_idx": g, "seats": list(seatings[g % len(seatings)]),
                            "start": "human:%d" % (g // len(seatings)), "termination": "finished",
                            "engine_moves": 300, "decisions": {{}}, "win_share": [float(i == w) for i in range(4)],
                            "worth_share": [0.1, 0.2, 0.3, 0.4]}}) + "\\n")
        f.flush()
"""


def test_scheduler_end_to_end(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_EVAL.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    for u in (0, 300, 600, 610):
        (tmp_path / "run" / "learner").mkdir(parents=True, exist_ok=True)
        (tmp_path / "run" / "learner" / f"{u}.pth").write_bytes(b"x")
    cfg, panel = _config(
        tmp_path,
        games=8,
        python=str(fake),
        eval_args=[],
        settle_seconds=0,
        retry_seconds=0,
        panel=[{"name": "u0", "update": 0}, {"name": "ref", "spec": "ref.pth@200"}],
    )
    (tmp_path / "fail_once").touch()
    monkeypatch.setenv("FAKE_EVAL_FAIL", str(tmp_path / "fail_once"))
    args = argparse.Namespace(
        panel=panel, workers=3, start_update=0, once=True, dry_run=False, results_log=tmp_path / "results.log"
    )
    L.run_ladder(args)

    root = tmp_path / "logs" / "eval" / "lad"
    records = L.read_ledger(root / "ledger.jsonl")
    assert len(records) == 8  # 2 panel members x 2 modes x 2 directions, for the newest multiple of 300
    assert {r["candidate"] for r in records} == {"u600"}
    sides = {}
    for r in records:
        sides.setdefault((r["opponent"], r["mode"]), set()).add(r["candidate_side"])
        assert r["games"] == {"2v2": 12, "1v3": 8}[r["mode"]]
        assert r["settings"] == SETTINGS
        assert r["source"] == "ladder"
    assert all(s == {"A", "B"} for s in sides.values()) and len(sides) == 4
    first = records[0]  # the match that crashed once and was rerun
    assert first["name"] == "u600_vs_u0_2v2_candA" and first["a"] == "u600" and first["b"] == "u0"
    assert list(root.glob("u600_vs_u0_2v2_candA.failed_*"))
    assert "--workers 3" in (root / "u600_vs_u0_2v2_candA.out").read_text()
    assert json.loads((root / "state.json").read_text()) == {"candidates": [600], "current": None}
    assert (root / "report.txt").exists() and (root / "ratings.json").exists()
    results = (tmp_path / "results.log").read_text().splitlines()
    assert len(results) == 9 and "lad/u600_vs_ref_1v3_oppA" in results[7] and "vs fixed panel" in results[8]

    L.run_ladder(args)  # nothing new is due
    assert len(L.read_ledger(root / "ledger.jsonl")) == 8
    assert "no candidate due" in capsys.readouterr().out

    (tmp_path / "run" / "learner" / "900.pth").write_bytes(b"x")
    L.run_ladder(argparse.Namespace(**{**vars(args), "dry_run": True}))
    commands = [line for line in capsys.readouterr().out.splitlines() if "eval_head_to_head.py" in line]
    assert len(commands) == 8
    assert all("run/learner/900.pth@200" in c for c in commands)
    assert sum("--a-seats 1" in c for c in commands) == 4
    assert all("--softpick 0" in c and "--seed 0" in c and "start_positions_1830_4p_v2.jsonl" in c for c in commands)
    assert len(L.read_ledger(root / "ledger.jsonl")) == 8


def test_resume_finishes_the_current_candidate(tmp_path, monkeypatch):
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_EVAL.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAKE_EVAL_FAIL", raising=False)
    (tmp_path / "run" / "learner").mkdir(parents=True)
    for u in (0, 300, 600):
        (tmp_path / "run" / "learner" / f"{u}.pth").write_bytes(b"x")
    cfg, panel = _config(
        tmp_path,
        games=4,
        modes=["1v3"],
        python=str(fake),
        eval_args=[],
        settle_seconds=0,
        retry_seconds=0,
        panel=[{"name": "u0", "update": 0}],
    )
    root = tmp_path / "logs" / "eval" / "lad"
    root.mkdir(parents=True)
    # a crash in the middle of u300's matches: one is in the ledger, the state still names u300
    L.save_state(root, {"candidates": [], "current": 300})
    match = L.plan_matches(cfg, 300)[0]
    L.append_ledger(
        L.ledger_path(cfg), {"source": "ladder", **match.fields(), **L.read_match(_played(tmp_path, cfg, match))}
    )
    L.run_ladder(
        argparse.Namespace(panel=panel, workers=None, start_update=0, once=True, dry_run=False, results_log=None)
    )
    records = L.read_ledger(L.ledger_path(cfg))
    assert [r["candidate"] for r in records] == ["u300", "u300"]
    assert json.loads((root / "state.json").read_text())["candidates"] == [300]


def _played(tmp_path, cfg, match):
    out = tmp_path / "played"
    _write_eval_dir(out, match.a.spec, match.b.spec, a_seats=L.MODES[match.mode], n_games=match.games)
    return out


def test_a_match_that_keeps_failing_is_given_up(tmp_path, monkeypatch):
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_EVAL.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAKE_EVAL_FAIL", raising=False)
    monkeypatch.setenv("FAKE_EVAL_BROKEN", "u300_vs_u0_1v3_oppA")
    (tmp_path / "run" / "learner").mkdir(parents=True)
    for u in (0, 300):
        (tmp_path / "run" / "learner" / f"{u}.pth").write_bytes(b"x")
    cfg, panel = _config(
        tmp_path,
        games=4,
        modes=["1v3"],
        python=str(fake),
        eval_args=[],
        settle_seconds=0,
        retry_seconds=0,
        max_attempts=3,
        panel=[{"name": "u0", "update": 0}],
    )
    L.run_ladder(
        argparse.Namespace(panel=panel, workers=None, start_update=0, once=True, dry_run=False, results_log=None)
    )
    root = tmp_path / "logs" / "eval" / "lad"
    assert [r["name"] for r in L.read_ledger(L.ledger_path(cfg))] == ["u300_vs_u0_1v3_candA"]
    assert len(list(root.glob("u300_vs_u0_1v3_oppA.failed_*"))) == 2  # three attempts, the last one left in place
    assert "gave up after 3 attempts: u300_vs_u0_1v3_oppA" in (root / "ladder.log").read_text()
    assert json.loads((root / "state.json").read_text()) == {"candidates": [300], "current": None}
