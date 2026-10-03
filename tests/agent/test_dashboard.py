"""The training dashboard's endpoints against the file layout a live loop leaves behind."""

import json
import math

import pytest
from torch.utils.tensorboard import SummaryWriter

from rl18xx.agent.dashboard import dashboard

SESSION = "20261002_121804_614841588"
ARCH = "AlphaZeroTransformer"


def _record(loop, timestamp, **extra):
    record = {
        "loop": loop,
        "timestamp": timestamp,
        "model_session": f"{ARCH}_{SESSION}",
        "model_architecture": ARCH,
        "total_loss": 2.0,
        "epoch_losses": [2.0],
        "games_played": 100,
        "experiences": 40000,
        "gate_win_rate": None,
    }
    record.update(extra)
    return record


def _write_jsonl(path, records, trailing=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + trailing)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Two runs in the history (the first diverged to NaN), a TensorBoard
    experiment for the second, self-play status files, and a pretrain sidecar."""
    for name, rel in [
        ("LOOP_CONFIG_FILE_PATH", "loop_config.json"),
        ("LOOP_STATUS_PATH", "loop_status.json"),
        ("SELF_PLAY_GAMES_STATUS_PATH", "self_play_games_status"),
        ("METRICS_HISTORY_PATH", "logs/loop/metrics_history.jsonl"),
        ("MODEL_HISTORY_PATH", "logs/loop/model_history.jsonl"),
        ("MODEL_CHECKPOINT_DIR", "model_checkpoints"),
        ("RUNS_ROOT", "runs/alphazero_runs"),
    ]:
        monkeypatch.setattr(dashboard, name, tmp_path / rel)
    monkeypatch.setattr(dashboard, "REPO_ROOT", tmp_path)
    dashboard._TTL_CACHE.clear()
    dashboard._TB_SCALARS_CACHE.clear()

    tb_dir = tmp_path / "runs/alphazero_runs" / SESSION / "experiment_20261002_160654"
    (tmp_path / "loop_status.json").write_text(
        json.dumps(
            {
                "current_loop": 3,
                "target_loop_count": 100,
                "total_games_this_iteration": 20,
                "status_message": "Self-play phase completed. Starting training.",
                "tensorboard_log_dir": f"runs/alphazero_runs/{SESSION}/experiment_20261002_160654",
                "loop_metrics": {"total_games_played": 200},
            }
        )
    )
    (tmp_path / "loop_config.json").write_text(
        json.dumps(
            {
                "num_loop_iterations": 100,
                "num_games_per_iteration": 20,
                "num_threads": 48,
                "num_readouts": 64,
                "target_experiences": 20000,
                "resign_high_threshold": 0.95,
                "inference_batch_size": 512,
                "training_config": {"batch_size": 256, "lr": 0.001, "use_fp16_training": True},
            }
        )
    )
    _write_jsonl(
        tmp_path / "logs/loop/metrics_history.jsonl",
        [
            _record(1, "2026-10-01T17:28:49", total_loss=math.nan, epoch_losses=[2.6, math.nan]),
            _record(1, "2026-10-02T16:11:44"),
            _record(2, "2026-10-02T16:16:48"),
        ],
        trailing='{"loop": 3, "timest',  # an append in progress
    )
    _write_jsonl(
        tmp_path / "logs/loop/model_history.jsonl",
        [
            {
                "loop": None,
                "timestamp": "2026-10-02T14:00:05",
                "checkpoint_num": 7,
                "session": SESSION,
                "promoted": True,
                "reason": "pretrain_value_refit",
            },
            {
                "loop": 1,
                "timestamp": "2026-10-01T17:28:49",
                "checkpoint_num": 8,
                "session": "other",
                "promoted": True,
                "reason": "first_iteration",
            },
            {
                "loop": 1,
                "timestamp": "2026-10-02T16:11:43",
                "checkpoint_num": 8,
                "session": SESSION,
                "promoted": True,
                "reason": "first_iteration",
            },
            {
                "loop": 2,
                "timestamp": "2026-10-02T16:16:48",
                "checkpoint_num": 9,
                "session": SESSION,
                "promoted": True,
                "reason": "gating_disabled",
            },
        ],
    )
    games = tmp_path / "self_play_games_status"
    games.mkdir()
    (games / "L0_G0.json").write_text(
        json.dumps({"status": "Completed", "termination": "resigned", "moves_played": 300, "start_time_unix": 1.0})
    )
    (games / "L1_G0.json").write_text(json.dumps({"status": "In Progress", "moves_played": 12, "start_time_unix": 2.0}))
    (games / "L1_G1.json").write_text('{"status": "In Pro')  # written by a non-atomic writer
    (games / "check_0.json").write_text(json.dumps({"status": "Completed", "moves_played": 10}))

    pointer = tmp_path / "model_checkpoints" / ARCH / "current_best.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"arch": ARCH, "session": SESSION, "checkpoint_num": 9}))

    writer = SummaryWriter(str(tb_dir))
    for step, (new, trained) in enumerate([(0.9, 0.6), (1.0, 0.65)]):
        writer.add_scalar("Value/CE_New_Games", new, step)
        writer.add_scalar("Value/CE_Trained_Window", trained, step)
        writer.add_scalar("SelfPlay/Auction_Unlock_Game_Rate", 0.5, step)
    writer.close()
    (tb_dir / "game_L0_G0").mkdir()  # per-game TensorBoard dirs sit alongside

    pretrain = tmp_path / "runs/alphazero_runs" / SESSION / "pretrain_20261002_125038"
    pretrain.mkdir(parents=True)
    (pretrain / "pretrain_summary.json").write_text(
        json.dumps({"model_session": SESSION, "training_examples": 2412602, "epoch_losses": [1.2, math.nan]})
    )
    return tmp_path


@pytest.fixture
def client(layout):
    return dashboard.app.test_client()


def _strict_json(response):
    """Parse like a browser's JSON.parse: NaN/Infinity tokens are errors."""

    def reject(token):
        raise ValueError(f"{token} is not valid JSON")

    return json.loads(response.get_data(as_text=True), parse_constant=reject)


ENDPOINTS = [
    "/api/current_status",
    "/api/loop_config",
    "/api/metrics_history",
    "/api/metrics_history?run=latest",
    "/api/metrics_history?run=0",
    "/api/metrics_runs",
    "/api/model_history",
    "/api/model_history?run=latest",
    "/api/games_status",
    "/api/games_status?loop=1",
    "/api/games_status?limit=2",
    "/api/system_metrics",
    "/api/pretrain_runs",
    "/api/pretrain_runs?brief=true",
    "/api/pretrain_runs?latest=true",
    f"/api/pretrain_runs?run={SESSION}/pretrain_20261002_125038",
]


@pytest.mark.parametrize("url", ENDPOINTS)
def test_endpoint_returns_browser_parseable_json(client, url):
    response = client.get(url)
    assert response.status_code == 200, response.get_data(as_text=True)
    _strict_json(response)


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "let currentLoop = 3;" in html
    assert "let targetExperiences = 20000;" in html


def test_metrics_history_is_split_into_runs_with_tensorboard_scalars(client):
    runs = _strict_json(client.get("/api/metrics_runs"))
    assert [(r["run"], r["first_loop"], r["last_loop"]) for r in runs] == [(0, 1, 1), (1, 1, 2)]
    assert runs[1]["tensorboard_dir"] == f"runs/alphazero_runs/{SESSION}/experiment_20261002_160654"

    latest = _strict_json(client.get("/api/metrics_history?run=latest"))
    assert [r["loop"] for r in latest] == [1, 2]
    # TensorBoard step N is loop N+1.
    assert [r["value_ce_new_games"] for r in latest] == pytest.approx([0.9, 1.0])
    assert [r["value_ce_trained_window"] for r in latest] == pytest.approx([0.6, 0.65])
    assert latest[0]["auction_unlock_game_rate"] == pytest.approx(0.5)

    diverged = _strict_json(client.get("/api/metrics_history?run=0"))
    assert diverged[0]["total_loss"] is None
    assert diverged[0]["epoch_losses"] == [2.6, None]

    assert client.get("/api/metrics_history?run=5").status_code == 404
    assert len(_strict_json(client.get("/api/metrics_history"))) == 3


def test_model_history_lineage_of_a_run(client):
    lineage = _strict_json(client.get("/api/model_history?run=latest"))
    assert [(e["loop"], e["checkpoint_num"]) for e in lineage] == [(None, 7), (1, 8), (2, 9)]


def test_games_status_filters_by_loop_and_skips_unreadable_files(client):
    games = _strict_json(client.get("/api/games_status?loop=1"))
    assert [g["game_id"] for g in games] == ["L1_G0"]
    assert games[0]["loop_number"] == 1
    assert len(_strict_json(client.get("/api/games_status?limit=2"))) == 2
    assert {g["game_id"] for g in _strict_json(client.get("/api/games_status"))} == {"L0_G0", "L1_G0", "check_0"}


def test_current_status_reports_liveness_and_current_best(client):
    status = _strict_json(client.get("/api/current_status"))
    assert status["current_loop"] == 3
    assert status["loop_process"] is None  # nothing runs main.py train from tmp_path
    assert status["current_best"]["checkpoint_num"] == 9
    assert status["status_updated_unix"] > 0


def test_training_param_form_keeps_the_rest_of_loop_config(client, layout):
    response = client.post("/", data={"lr": "0.002", "use_fp16_training": "false", "batch_size": "abc"})
    assert response.status_code == 302
    config = json.loads((layout / "loop_config.json").read_text())
    assert config["training_config"]["lr"] == 0.002
    assert config["training_config"]["use_fp16_training"] is False
    assert config["training_config"]["batch_size"] == 256  # invalid input left unchanged
    # Keys the form doesn't edit -- including the loop's auto-calibrated resign threshold -- survive.
    assert config["resign_high_threshold"] == 0.95
    assert config["target_experiences"] == 20000
    assert config["inference_batch_size"] == 512


def test_loop_cmdline_detection():
    assert dashboard._is_loop_cmdline(["/venv/bin/python3", "main.py", "train", "--iterations", "100"])
    assert dashboard._is_loop_cmdline(["python", "-m", "rl18xx.agent.alphazero.loop"])
    assert not dashboard._is_loop_cmdline(["uv", "run", "python", "main.py", "train"])
    assert not dashboard._is_loop_cmdline(["python", "main.py", "pretrain"])
    assert not dashboard._is_loop_cmdline(["python", "-c", "from multiprocessing.spawn import spawn_main"])
