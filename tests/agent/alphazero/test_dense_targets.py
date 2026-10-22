"""Dense auxiliary value targets (rl18xx/agent/alphazero/dense_targets.py) and the value gate's replay."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import dense_targets as dt  # noqa: E402

from .search_test_utils import ending_positions  # noqa: E402
from .test_search_learning import _tiny_model  # noqa: E402


@pytest.fixture(scope="module")
def finished():
    game = ending_positions(4, 3)[-1]
    assert game.finished
    return game


def test_final_targets_describe_the_finished_game(finished):
    t = dt.final_targets(finished, list(finished.raw_actions))
    assert t["p_final"].shape == (4, 4) and t["pc_final"].shape == (4, 8) and t["c_final"].shape == (8, 4)
    assert t["h_color"].shape == (93,) and t["h_token"].shape == (93, 8) and t["h_route"].shape == (93, 8)
    assert t["p_final"][:, 0].sum() == pytest.approx(1.0, abs=1e-5)
    assert ((t["c_pres"] >= 0) & (t["c_pres"] <= 4)).all()
    assert (t["pc_final"].sum(axis=0) <= 1.0 + 1e-6).all()  # players hold at most 100% of a corporation
    corps = [c.sym for c in finished.corporations]
    players = sorted(finished.players, key=lambda p: int(p.id))
    for i, sym in enumerate(corps):
        c = next(x for x in finished.corporations if x.sym == sym)
        owner = c.owner_id_str
        expected = next((s for s, p in enumerate(players) if owner == f"player:{int(p.id)}"), 4)
        assert t["c_pres"][i] == expected
        assert t["c_final"][i, 0] == float(bool(c.floated))
    # presidencies counted per seat
    assert t["p_final"][:, 3].sum() == pytest.approx((t["c_pres"] < 4).sum() / 8)
    placed = sum(1 for h in finished.hexes for city in h.tile.cities for tok in city.tokens if tok is not None)
    assert int(t["h_token"].sum()) <= placed  # one flag per corporation and hex
    assert ((t["h_color"] >= 0) & (t["h_color"] < dt.NUM_COLORS)).all()


def test_canonical_rotates_seats_and_renumbers_presidents():
    t = {
        "p_final": np.arange(16, dtype=np.float32).reshape(4, 4),
        "p_short": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "c_pres": np.array([0, 1, 2, 3, 4, 4, 2, 1]),
        "h_color": np.zeros(93, dtype=np.int64),
    }
    c = dt.canonical(t, 2)
    np.testing.assert_array_equal(c["p_final"][0], t["p_final"][2])  # the mover (seat 2) first
    np.testing.assert_allclose(c["p_short"], [0.3, 0.4, 0.1, 0.2])
    np.testing.assert_array_equal(c["c_pres"], [2, 3, 0, 1, 4, 4, 0, 3])
    np.testing.assert_array_equal(c["h_color"], t["h_color"])  # hexes and corporations don't rotate


def test_next_run_targets_take_each_corporations_next_run():
    actions = [
        {"type": "run_routes", "entity": "PRR", "routes": [{"revenue": 80}, {"revenue": 40}]},
        {"type": "pass", "entity": "PRR"},
        {"type": "run_routes", "entity": "NYC", "routes": [{"revenue": 500}], "extra_revenue": 0},
        {"type": "run_routes", "entity": "PRR", "routes": [{"revenue": 200}]},
    ]
    revenues = dt.run_revenues(actions)
    nxt, mask = dt.next_run_targets(revenues, 1)
    corps = dt._corp_order()
    assert nxt[corps.index("PRR")] == pytest.approx(200 / 400) and mask[corps.index("PRR")] == 1
    assert nxt[corps.index("NYC")] == 1.0  # capped at 400
    assert mask[corps.index("CPR")] == 0
    nxt0, _ = dt.next_run_targets(revenues, 0)
    assert nxt0[corps.index("PRR")] == pytest.approx(120 / 400)


def test_dense_loss_trains_the_networks_encoders(finished):
    import torch

    model = _tiny_model(0).eval()
    heads = dt.build_dense_heads(model)
    tap = dt.ActivationTap(model)
    state, nodes, _, rotation, _ = finished.encode_canonical_py()
    t = dt.canonical({**dt.final_targets(finished, list(finished.raw_actions)), "p_short": np.full(4, 0.25)}, rotation)
    batch = {k: torch.from_numpy(np.asarray(v)[None]).to(torch.long if k in ("c_pres", "h_color") else torch.float32)
             for k, v in t.items()}
    batch["c_next"], batch["c_next_mask"] = torch.full((1, 8), 0.5), torch.ones(1, 8)
    gs = torch.from_numpy(np.asarray(state, np.float32)[None])
    nd = torch.from_numpy(np.asarray(nodes, np.float32)[None])
    model(gs, nd, value_only=True)
    total, parts = dt.dense_loss(heads, tap, batch)
    assert torch.isfinite(total) and set(parts) == set(dt.DenseWeights().__dict__)
    total.backward()
    for module in (model.econ_transformer, model.hex_transformer, model.res_blocks[-1]):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    tap.remove()


def test_value_gate_replays_a_game_into_positions(finished):
    path = Path(__file__).resolve().parents[3] / "scripts" / "value_gate.py"
    spec = importlib.util.spec_from_file_location("value_gate", path)
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    from rl18xx.agent.alphazero.self_play import _compute_net_worth
    from rl18xx.rust_adapter import RustGameAdapter

    worth = _compute_net_worth(RustGameAdapter(finished))
    w = np.array([float(worth[k]) for k in sorted(worth)])
    record = {"win_share": (w == w.max()) / (w == w.max()).sum(), "net_worth": w.tolist()}
    arrays = vg.replay_game("t/0", list(finished.raw_actions), record, True, per_game=8)
    n = len(arrays["game_states"])
    assert 0 < n <= 8
    assert arrays["nodes"].dtype == np.float16 and arrays["h_token"].dtype == np.uint8
    for k in range(n):
        r = int(arrays["rotation"][k])
        np.testing.assert_allclose(arrays["win"][k], np.roll(record["win_share"], -r))
