from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask.json.provider import DefaultJSONProvider
import json
import math
import os
import subprocess
from dataclasses import fields
from rl18xx.agent.alphazero import game_records
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.shared.atomic_io import atomic_write_json
import time
from pathlib import Path
from datetime import datetime
import psutil

# Fields to exclude from the training parameters display. ``pretrain_*`` fields
# only configure ``main.py pretrain``; the self-play loop never reads them.
EXCLUDED_FIELDS = {"root_dir", "train_dir", "val_dir", "model_checkpoint_dir", "metrics", "global_step"}
EDITABLE_TRAINING_PARAMS = [
    field.name
    for field in fields(TrainingConfig)
    if field.name not in EXCLUDED_FIELDS and not field.name.startswith("pretrain_")
]
BOOL_TRAINING_PARAMS = {name for name in EDITABLE_TRAINING_PARAMS if isinstance(getattr(TrainingConfig(), name), bool)}
# The loop takes these from its command line and rewrites them into
# loop_config.json every iteration (load_loop_config), so editing them in the
# file has no effect; the dashboard shows them read-only.
CLI_OWNED_LOOP_KEYS = ("num_loop_iterations", "target_experiences", "num_threads", "num_readouts")


def _json_safe(obj):
    """Replace NaN/Infinity (which the loop writes when an iteration diverges) with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class _BrowserJSONProvider(DefaultJSONProvider):
    """Emit null for non-finite floats. Python's json writes bare ``NaN`` tokens,
    which the browser's JSON.parse rejects, so one diverged iteration in
    metrics_history.jsonl used to blank every self-play chart."""

    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)


app = Flask(__name__)
app.json = _BrowserJSONProvider(app)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key_change_me")  # Use env var for production

# Resolve paths relative to the project root.
# dashboard.py -> dashboard/ -> agent/ -> rl18xx/ -> repo root == parents[3].
# This works regardless of CWD (both `main.py dashboard` and gunicorn --chdir).
REPO_ROOT = Path(__file__).resolve().parents[3]

LOOP_CONFIG_FILE_PATH = REPO_ROOT / "loop_config.json"
LOOP_STATUS_PATH = REPO_ROOT / "loop_status.json"
SELF_PLAY_GAMES_STATUS_PATH = REPO_ROOT / "self_play_games_status"
METRICS_HISTORY_PATH = REPO_ROOT / "logs" / "loop" / "metrics_history.jsonl"
MODEL_HISTORY_PATH = REPO_ROOT / "logs" / "loop" / "model_history.jsonl"
MODEL_CHECKPOINT_DIR = REPO_ROOT / "model_checkpoints"
# TensorBoard logs: runs/alphazero_runs/<session>/experiment_{ts}/ for each
# self-play loop run and .../pretrain_{ts}/ for pretraining (with a JSON
# sidecar of per-epoch loss + accuracy arrays that the Pretraining tab plots).
RUNS_ROOT = REPO_ROOT / "runs" / "alphazero_runs"
TENSORBOARD_PORT = 6006
# Saved-game collections (game_records): directories under these, relative to
# REPO_ROOT, with a games/ dir. eval_head_to_head.py / eval_policy_only.py
# write under logs/eval, the policy-gradient stage under model_checkpoints_pg.
GAME_ROOTS = game_records.DEFAULT_ROOTS

# Per-iteration scalars the loop logs only to TensorBoard, merged into the
# metrics-history records under these keys. TensorBoard steps are 0-based loop
# indices; history records number loops from 1.
TENSORBOARD_EXTRA_SCALARS = {
    "Value/CE_New_Games": "value_ce_new_games",
    "Value/CE_Trained_Window": "value_ce_trained_window",
    "SelfPlay/Auction_Unlock_Game_Rate": "auction_unlock_game_rate",
    "Resign/Resigned_Games": "resigned_games",
    "Resign/High_Threshold": "resign_high_threshold",
    "Resign/Holdout_FP_Rate": "resign_holdout_fp_rate",
    "training/oldest_example_age_minutes": "oldest_example_age_minutes",
    "Eval/Score_vs_Start": "eval_score_vs_start",
    "Eval/Score_vs_Start_SE": "eval_score_vs_start_se",
}

_TTL_CACHE: dict = {}
_TB_SCALARS_CACHE: dict = {}


def _ttl_cached(key, ttl_seconds, compute):
    now = time.monotonic()
    hit = _TTL_CACHE.get(key)
    if hit is not None and now - hit[0] < ttl_seconds:
        return hit[1]
    value = compute()
    _TTL_CACHE[key] = (now, value)
    return value


def _read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path):
    """Records of a JSONL file the loop appends to. A line that doesn't parse
    (an append in progress, or a write cut short by a crash) is skipped rather
    than failing the whole read."""
    if not path.exists():
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _parse_iso(ts):
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def get_current_status():
    if not LOOP_STATUS_PATH.exists():
        return {
            "status_message": "Status file not yet created. Training loop may not be running or hasn't started a loop."
        }

    try:
        with open(LOOP_STATUS_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"error": "Error reading status file (invalid JSON)."}
    except Exception as e:
        return {"error": f"Status file unreadable: {e}"}


def _is_loop_cmdline(cmdline):
    if not cmdline or "python" not in os.path.basename(cmdline[0]):
        return False
    args = cmdline[1:]
    for i, arg in enumerate(args):
        if arg.endswith("main.py") and "train" in args[i + 1 : i + 2]:
            return True
        if arg == "rl18xx.agent.alphazero.loop" or arg.endswith(os.path.join("alphazero", "loop.py")):
            return True
    return False


def _find_loop_process():
    """The ``main.py train`` (or ``alphazero.loop``) process running from this
    checkout, or None. loop_status.json keeps its last message after the loop
    dies, so this is how the dashboard tells a live run from a stale file."""
    matches = {}
    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        try:
            if not _is_loop_cmdline(proc.info["cmdline"]):
                continue
            if Path(proc.cwd()).resolve() != REPO_ROOT:
                continue
        except (psutil.Error, OSError):
            continue
        matches[proc.info["pid"]] = proc.info
    # A forked child shares the parent's command line; report the root.
    roots = [info for info in matches.values() if info["ppid"] not in matches]
    if not roots:
        return None
    info = min(roots, key=lambda i: i["create_time"])
    return {"pid": info["pid"], "started_unix": info["create_time"]}


def get_current_best():
    """``{arch, session, checkpoint_num}`` of the model self-play is using, or None."""
    if not MODEL_CHECKPOINT_DIR.is_dir():
        return None
    pointers = []
    for arch_dir in MODEL_CHECKPOINT_DIR.iterdir():
        pointer = _read_json(arch_dir / "current_best.json") if arch_dir.is_dir() else None
        if isinstance(pointer, dict) and pointer.get("session"):
            pointers.append(pointer)
    return max(pointers, key=lambda p: str(p["session"])) if pointers else None


def get_current_loop_config():
    """Reads the loop_config.json file and provides defaults."""
    default_config = {
        "num_loop_iterations": None,
        "target_experiences": None,
        "num_threads": None,
        "num_readouts": None,
        "training_params": TrainingConfig().to_json(),
    }

    if not LOOP_CONFIG_FILE_PATH.exists():
        flash("Loop configuration file not found. Displaying default values. Save to create one.", "info")
        return default_config

    try:
        with open(LOOP_CONFIG_FILE_PATH, "r") as f:
            loaded_config_json = json.load(f)

        # Start with defaults and override with loaded values
        config_for_form = default_config.copy()
        for key in CLI_OWNED_LOOP_KEYS:
            config_for_form[key] = loaded_config_json.get(key)

        loaded_training_config = loaded_config_json.get("training_config", {})
        current_training_params = default_config["training_params"].copy()  # Start with defaults
        for param in EDITABLE_TRAINING_PARAMS:
            if param in loaded_training_config:
                current_training_params[param] = loaded_training_config[param]
        config_for_form["training_params"] = current_training_params

        return config_for_form
    except json.JSONDecodeError:
        flash(f"Error reading loop config file (invalid JSON). Displaying defaults.", "warning")
        return default_config
    except Exception as e:
        flash(f"Error loading loop config: {e}. Displaying defaults.", "warning")
        return default_config


def save_training_params(updates):
    """Merge ``updates`` into loop_config.json's training_config, which the loop
    hot-reloads at the start of each iteration. Every other key (resign
    calibration, inference settings, ...) is kept as the loop wrote it."""
    try:
        config = _read_json(LOOP_CONFIG_FILE_PATH) if LOOP_CONFIG_FILE_PATH.exists() else None
        if not isinstance(config, dict):
            config = {}
        training_config = config.get("training_config")
        if not isinstance(training_config, dict):
            training_config = TrainingConfig().to_json()
        training_config.update(updates)
        config["training_config"] = training_config
        atomic_write_json(LOOP_CONFIG_FILE_PATH, config, indent=4)
        flash("Training parameters updated. Changes will apply on the next loop iteration.", "success")
    except Exception as e:
        flash(f"Error saving loop config: {e}", "error")


def get_games_in_progress(loop=None, limit=None):
    """Self-play game status dicts, oldest first.

    ``loop`` (the 0-based index in the ``L{loop}_G{game}.json`` file names)
    restricts to one iteration; ``limit`` keeps only the most recently updated
    files. Files accumulate across iterations (and runs started with
    --keep-old-files), so the dashboard always passes one or the other.
    """
    if not SELF_PLAY_GAMES_STATUS_PATH.exists():
        return {"error": "Self-play games status file not found."}

    games_data = []
    try:
        if loop is None:
            game_files = list(SELF_PLAY_GAMES_STATUS_PATH.glob("*.json"))
        else:
            game_files = list(SELF_PLAY_GAMES_STATUS_PATH.glob(f"L{loop}_G*.json"))
        if limit is not None and limit > 0:

            def mtime(path):
                try:
                    return path.stat().st_mtime
                except OSError:
                    return 0.0

            game_files.sort(key=mtime, reverse=True)
        for game_file in game_files:
            if limit is not None and limit > 0 and len(games_data) >= limit:
                break
            # A file can vanish between the glob and the read (cleanup), and a
            # writer that doesn't go through atomic_write_json can leave a
            # partial one; skip it rather than failing the whole list.
            game_data = _read_json(game_file)
            if not isinstance(game_data, dict):
                continue
            game_data["start_time_str"] = datetime.fromtimestamp(game_data.get("start_time_unix") or 0).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            game_data["last_update_str"] = datetime.fromtimestamp(
                game_data.get("last_update_unix") or time.time()
            ).strftime("%Y-%m-%d %H:%M:%S")

            game_data["game_id"] = game_file.name[:-5]

            # Extract loop number from filename (format: L{loop}_G{game}.json)
            if game_file.name.startswith("L") and "_G" in game_file.name:
                try:
                    loop_num = int(game_file.name.split("_")[0][1:])
                    game_data["loop_number"] = loop_num
                except (ValueError, IndexError):
                    game_data["loop_number"] = None
            else:
                game_data["loop_number"] = None

            games_data.append(game_data)

    except Exception as e:
        return {"error": f"Error reading self-play games status: {e}"}
    return sorted(games_data, key=lambda x: x.get("start_time_unix") or 0, reverse=False)


def split_runs(records):
    """Group metrics-history records into training runs.

    Every ``main.py train`` invocation appends to the same history file and
    numbers its iterations from 1, so a record whose loop number doesn't
    exceed the previous record's starts a new run.
    """
    runs = []
    prev_loop = None
    for record in records:
        loop = record.get("loop")
        if not runs or not isinstance(loop, int) or prev_loop is None or loop <= prev_loop:
            runs.append([])
        runs[-1].append(record)
        prev_loop = loop if isinstance(loop, int) else None
    return runs


def _session_of(record):
    """Checkpoint session (``{date}_{time}_{rand}``) of a metrics record, whose
    ``model_session`` is prefixed with the architecture name."""
    session = str(record.get("model_session") or "")
    arch = str(record.get("model_architecture") or "")
    if arch and session.startswith(arch + "_"):
        session = session[len(arch) + 1 :]
    return session


def _tensorboard_dir_for_run(run_records):
    """The ``experiment_{ts}`` TensorBoard directory a run logged to: the
    latest one in the model session's directory created before the run's
    first record."""
    first = run_records[0]
    session = _session_of(first)
    first_record_time = _parse_iso(first.get("timestamp"))
    session_dir = RUNS_ROOT / session
    if not session or first_record_time is None or not session_dir.is_dir():
        return None
    best = None
    for candidate in session_dir.glob("experiment_*"):
        try:
            started = datetime.strptime(candidate.name[len("experiment_") :], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if started <= first_record_time and (best is None or started > best[0]):
            best = (started, candidate)
    return best[1] if best else None


def read_tensorboard_scalars(log_dir, tags):
    """``{tag: {step: value}}`` for the requested scalar tags in ``log_dir``'s
    own event files (not its per-game subdirectories). Cached until an event
    file changes."""
    try:
        with os.scandir(log_dir) as entries:
            event_files = sorted(
                (e.name, e.stat().st_size, e.stat().st_mtime_ns)
                for e in entries
                if "tfevents" in e.name and e.is_file()
            )
    except OSError:
        return {}
    if not event_files:
        return {}
    cached = _TB_SCALARS_CACHE.get(str(log_dir))
    if cached is not None and cached[0] == event_files:
        return cached[1]
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        scalars = {
            tag: {event.step: event.value for event in accumulator.Scalars(tag)} for tag in tags if tag in available
        }
    except Exception:
        return {}
    _TB_SCALARS_CACHE[str(log_dir)] = (event_files, scalars)
    return scalars


def _add_tensorboard_extras(run_records, tensorboard_dir):
    if tensorboard_dir is None:
        return
    scalars = read_tensorboard_scalars(tensorboard_dir, TENSORBOARD_EXTRA_SCALARS)
    for record in run_records:
        loop = record.get("loop")
        if not isinstance(loop, int):
            continue
        for tag, key in TENSORBOARD_EXTRA_SCALARS.items():
            value = scalars.get(tag, {}).get(loop - 1)
            if value is not None and key not in record:
                record[key] = value


def _run_summary(index, run_records, tensorboard_dir):
    first, last = run_records[0], run_records[-1]
    return {
        "run": index,
        "model_session": first.get("model_session"),
        "first_loop": first.get("loop"),
        "last_loop": last.get("loop"),
        "iterations": len(run_records),
        "start": first.get("timestamp"),
        "end": last.get("timestamp"),
        "games": sum(r.get("games_played") or 0 for r in run_records),
        "experiences": sum(r.get("experiences") or 0 for r in run_records),
        "tensorboard_dir": _display_path(tensorboard_dir) if tensorboard_dir else None,
    }


def _display_path(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_metrics_runs(history, selected=None):
    """``[(summary, records)]`` for every run in the metrics ``history``
    records, oldest first, each record tagged with its ``run`` index.
    TensorBoard-only scalars are merged into the records of the runs in
    ``selected`` (all when None)."""
    runs = []
    for index, run_records in enumerate(split_runs(history)):
        tensorboard_dir = _tensorboard_dir_for_run(run_records)
        if selected is None or index in selected:
            _add_tensorboard_extras(run_records, tensorboard_dir)
        for record in run_records:
            record["run"] = index
        runs.append((_run_summary(index, run_records, tensorboard_dir), run_records))
    return runs


def _resolve_run(run_arg, num_runs):
    """Index of the run selected by ``?run=`` (``latest`` or an index; negative counts from the end)."""
    if num_runs == 0:
        return None
    if run_arg == "latest":
        return num_runs - 1
    try:
        index = int(run_arg)
    except (TypeError, ValueError):
        return None
    if index < 0:
        index += num_runs
    return index if 0 <= index < num_runs else None


def list_pretrain_runs():
    """Return per-run pretrain summaries (newest first).

    Each run directory is ``runs/alphazero_runs/<session>/pretrain_{ts}/``
    (legacy runs sit directly under the root without a session dir) and (if
    pretraining wrote a sidecar) contains ``pretrain_summary.json`` with
    per-epoch loss/accuracy arrays. Returns a list of summary dicts with
    an extra ``run_name`` field. Runs without a sidecar are skipped.
    """
    if not RUNS_ROOT.exists():
        return []
    runs = []
    run_dirs = list(RUNS_ROOT.glob("pretrain_*")) + list(RUNS_ROOT.glob("*/pretrain_*"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        sidecar = run_dir / "pretrain_summary.json"
        if not sidecar.exists():
            continue
        try:
            with open(sidecar, "r") as f:
                summary = json.load(f)
            # ``in_progress`` stays true when a run is killed; the age lets the
            # page tell a live run from an abandoned one.
            summary["updated_unix"] = sidecar.stat().st_mtime
        except (json.JSONDecodeError, OSError):
            continue
        summary["run_name"] = str(run_dir.relative_to(RUNS_ROOT))
        runs.append(summary)
    # Newest first, by the pretrain_{ts} leaf name regardless of nesting.
    runs.sort(key=lambda r: r["run_name"].rsplit("/", 1)[-1], reverse=True)
    return runs


PRETRAIN_BRIEF_KEYS = (
    "run_name",
    "model_session",
    "timestamp",
    "kind",
    "in_progress",
    "updated_unix",
    "epochs_trained",
    "epochs_planned",
    "training_examples",
    "best_val_loss",
)


@app.route("/api/pretrain_runs")
def api_pretrain_runs():
    """List all pretraining runs found under ``runs/alphazero_runs/pretrain_*``.

    Returns the same per-epoch arrays the dashboard consumes for plotting.
    ``?latest=true`` returns just the most-recent run as a single dict,
    ``?run=<run_name>`` that run, and ``?brief=true`` the list without the
    per-epoch arrays.
    """
    runs = list_pretrain_runs()
    if request.args.get("latest", "").lower() in ("1", "true", "yes"):
        return jsonify(runs[0] if runs else {})
    run_name = request.args.get("run")
    if run_name:
        match = next((r for r in runs if r["run_name"] == run_name), None)
        return (jsonify(match), 200) if match else (jsonify({"error": f"No pretrain run {run_name!r}"}), 404)
    if request.args.get("brief", "").lower() in ("1", "true", "yes"):
        return jsonify([{k: r.get(k) for k in PRETRAIN_BRIEF_KEYS} for r in runs])
    return jsonify(runs)


@app.route("/api/loop_config", methods=["GET", "POST"])
def api_loop_config_handler():
    if request.method == "GET":
        if not LOOP_CONFIG_FILE_PATH.exists():
            return jsonify({"error": "Loop configuration file not found."}), 404
        try:
            with open(LOOP_CONFIG_FILE_PATH, "r") as f:
                data = json.load(f)
            return jsonify(data), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON in loop configuration file."}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to read loop configuration file: {e}"}), 500

    if request.method == "POST":
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Type checks (only for the fields actually present — partial updates allowed)
        type_specs = {
            "num_loop_iterations": int,
            "num_games_per_iteration": int,
            "num_threads": int,
            "training_config": dict,
            "num_readouts": int,
        }
        for key, expected_type in type_specs.items():
            if key in data and not isinstance(data[key], expected_type):
                return jsonify({"error": f"Invalid type for '{key}': expected {expected_type.__name__}."}), 400

        # Load existing on-disk config and merge the POST body on top, so the
        # caller can do a partial update without dropping fields like
        # ``target_experiences`` or ``endAfterCurrentLoop`` that aren't in the body.
        merged = {}
        if LOOP_CONFIG_FILE_PATH.exists():
            try:
                with open(LOOP_CONFIG_FILE_PATH, "r") as f:
                    merged = json.load(f)
            except json.JSONDecodeError:
                merged = {}
            except Exception as e:
                return jsonify({"error": f"Failed to read existing loop configuration: {e}"}), 500
        merged.update(data)

        try:
            atomic_write_json(LOOP_CONFIG_FILE_PATH, merged, indent=4)
            return jsonify({"message": "Loop configuration updated successfully."}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to write loop configuration: {e}"}), 500


@app.route("/api/games_status")
def api_games_status():
    """Self-play game statuses. ``?loop=N`` (0-based, as in the file names)
    for one iteration's games; ``?limit=N`` for the N most recently updated."""
    loop = request.args.get("loop", type=int)
    limit = request.args.get("limit", type=int)
    games_data = get_games_in_progress(loop=loop, limit=limit)
    if isinstance(games_data, dict) and "error" in games_data:
        return jsonify(games_data), 500
    return jsonify(games_data)


@app.route("/api/current_status")
def api_current_status():
    """loop_status.json plus what the dashboard derives about it: whether the
    loop process is still alive (``loop_process``, null when it isn't), how
    old the file is, and the current-best model pointer."""
    status = get_current_status()
    try:
        status["status_updated_unix"] = LOOP_STATUS_PATH.stat().st_mtime
    except OSError:
        status["status_updated_unix"] = None
    status["loop_process"] = _ttl_cached("loop_process", 5.0, _find_loop_process)
    status["current_best"] = get_current_best()
    return jsonify(status)


@app.route("/api/metrics_history")
def api_metrics_history():
    """Per-iteration metrics records from logs/loop/metrics_history.jsonl.

    Records carry a ``run`` index (one per ``main.py train`` invocation) and
    the TensorBoard-only scalars of TENSORBOARD_EXTRA_SCALARS. ``?run=latest``
    or ``?run=<index>`` restricts to one run; ``?last=N`` keeps the last N.
    """
    try:
        history = _read_jsonl(METRICS_HISTORY_PATH)
        run_arg = request.args.get("run")
        if run_arg is None:
            records = [record for _, run_records in load_metrics_runs(history) for record in run_records]
        else:
            num_runs = len(split_runs(history))
            index = _resolve_run(run_arg, num_runs)
            if index is None and num_runs:
                return jsonify({"error": f"Unknown run {run_arg!r}; there are {num_runs} runs."}), 404
            records = load_metrics_runs(history, selected={index})[index][1] if index is not None else []
        last_n = request.args.get("last", type=int)
        if last_n is not None and last_n > 0:
            records = records[-last_n:]
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": f"Failed to read metrics history: {e}"}), 500


@app.route("/api/metrics_runs")
def api_metrics_runs():
    """One summary per training run in the metrics history, oldest first."""
    try:
        history = _read_jsonl(METRICS_HISTORY_PATH)
        return jsonify([summary for summary, _ in load_metrics_runs(history, selected=set())])
    except Exception as e:
        return jsonify({"error": f"Failed to read metrics history: {e}"}), 500


@app.route("/api/model_history")
def api_model_history():
    """Model gating / promotion records from logs/loop/model_history.jsonl.

    ``?run=latest`` or ``?run=<index>`` (as in /api/metrics_history) keeps the
    lineage of that run: its own promotions plus the checkpoint it started from.
    """
    try:
        records = _read_jsonl(MODEL_HISTORY_PATH)
        run_arg = request.args.get("run")
        if run_arg is not None:
            records = _lineage_for_run(records, run_arg)
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": f"Failed to read model history: {e}"}), 500


def _lineage_for_run(records, run_arg):
    runs = split_runs(_read_jsonl(METRICS_HISTORY_PATH))
    index = _resolve_run(run_arg, len(runs))
    if index is None:
        return []
    run_records = runs[index]
    session = _session_of(run_records[0])
    # The run's window: after the previous run's last record, up to (a little
    # past) its own last one -- or open-ended for the newest run.
    start = _parse_iso(runs[index - 1][-1].get("timestamp")) if index > 0 else None
    end = _parse_iso(run_records[-1].get("timestamp")) if index < len(runs) - 1 else None
    in_window, seed = [], None
    for record in records:
        if record.get("session") != session:
            continue
        ts = _parse_iso(record.get("timestamp"))
        if ts is None:
            continue
        if start is not None and ts <= start:
            if record.get("loop") is None:
                seed = record  # the latest pretrain checkpoint before the run
            continue
        if end is not None and (ts - end).total_seconds() > 60:
            continue
        if record.get("loop") is None and not in_window:
            seed = record
            continue
        in_window.append(record)
    return ([seed] if seed else []) + in_window


def _gpu_stats():
    """GPU utilization and memory of the whole device, via nvidia-smi (the
    dashboard's own process holds no CUDA context, so torch can't see what
    training is using)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        try:
            util, used, total = (float(part) for part in line.split(","))
        except ValueError:
            continue
        gpus.append({"util_percent": util, "memory_used_mb": used, "memory_total_mb": total})
    return gpus or None


@app.route("/api/system_metrics")
def api_system_metrics():
    try:
        # Get CPU percentage (average across all cores)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        response = {
            "cpu_percent": round(cpu_percent, 1),
            "memory_percent": round(memory_percent, 1),
        }

        gpus = _ttl_cached("gpu_stats", 3.0, _gpu_stats)
        if gpus:
            response["gpus"] = gpus
            response["gpu_util_percent"] = round(sum(g["util_percent"] for g in gpus) / len(gpus), 1)
            response["gpu_memory_used_mb"] = sum(g["memory_used_mb"] for g in gpus)
            response["gpu_memory_total_mb"] = sum(g["memory_total_mb"] for g in gpus)

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Failed to get system metrics: {e}"}), 500


# ----------------------------------------------------------------- saved games
# The game viewer: /games lists the saved-game collections (game_records) and
# their games; /games/view steps through one, replayed in the Python engine
# (game_viewer).


def _requested_game():
    """``((collection dir, game file, listing entry), None)`` for ``?collection=&game=``,
    or ``(None, error response)``."""
    collection_id = request.args.get("collection", "")
    directory = game_records.resolve_collection(REPO_ROOT, collection_id, GAME_ROOTS)
    if directory is None:
        return None, (jsonify({"error": f"Unknown game collection {collection_id!r}"}), 404)
    name = request.args.get("game", "")
    path = game_records.resolve_game_file(directory, name)
    if path is None:
        return None, (jsonify({"error": f"No game {name!r} in {collection_id}"}), 404)
    return (directory, path, game_records.game_context(directory, name)), None


def _game_summary(path, entry):
    from rl18xx.agent.dashboard import game_viewer

    return game_viewer.game_summary(path, entry.get("auction_unlock"))


@app.route("/games")
def games_page():
    return render_template("games.html")


@app.route("/games/view")
def game_view_page():
    return render_template(
        "game_view.html", collection=request.args.get("collection", ""), game=request.args.get("game", "")
    )


@app.route("/api/game_collections")
def api_game_collections():
    """Saved-game collections under GAME_ROOTS, newest first."""
    return jsonify(game_records.find_collections(REPO_ROOT, GAME_ROOTS))


@app.route("/api/game_collection")
def api_game_collection():
    """``?collection=<id>``: the collection's saved games (its listing entries)."""
    collection_id = request.args.get("collection", "")
    directory = game_records.resolve_collection(REPO_ROOT, collection_id, GAME_ROOTS)
    if directory is None:
        return jsonify({"error": f"Unknown game collection {collection_id!r}"}), 404
    kind, title = game_records.describe_collection(directory)
    return jsonify({"id": collection_id, "kind": kind, "title": title, "games": game_records.list_games(directory)})


@app.route("/api/game")
def api_game():
    """``?collection=&game=``: the whole game (game_viewer.game_summary) plus its listing ``entry``."""
    found, error = _requested_game()
    if error:
        return error
    _, path, entry = found
    try:
        summary = _game_summary(path, entry)
    except (OSError, ValueError) as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({**summary, "entry": entry})


@app.route("/api/game_state")
def api_game_state():
    """``?collection=&game=&step=N``: the position after N actions (game_viewer.state_at).
    ``auction_unlock`` (0/1) should be the summary's, which it defaults to."""
    found, error = _requested_game()
    if error:
        return error
    _, path, entry = found
    from rl18xx.agent.dashboard import game_viewer

    unlock = request.args.get("auction_unlock")
    try:
        if unlock is None:
            unlock = _game_summary(path, entry)["auction_unlock"]
        else:
            unlock = unlock.lower() in ("1", "true", "yes")
        return jsonify(game_viewer.state_at(path, request.args.get("step", default=0, type=int), unlock))
    except (OSError, ValueError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/game_export")
def api_game_export():
    """``?collection=&game=``: the game as an 18xx.games-style JSON file
    (game_records.make_game_record, metadata included) to download."""
    found, error = _requested_game()
    if error:
        return error
    _, path, entry = found
    try:
        loaded = game_records.load_game_file(path)
        unlock = _game_summary(path, entry)["auction_unlock"]
    except (OSError, ValueError) as e:
        return jsonify({"error": str(e)}), 500
    collection_id = request.args.get("collection", "")
    name = f"{collection_id.replace('/', '_')}_{path.stem}"
    meta = {**{k: v for k, v in entry.items() if k != "game_file"}, **loaded.meta, "name": name}
    record = game_records.make_game_record(loaded.actions, loaded.num_players, auction_unlock=unlock, meta=meta)
    response = jsonify(record)
    response.headers["Content-Disposition"] = f'attachment; filename="{name}.json"'
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Only training params are editable: the loop hot-reloads
            # loop_config.json's training_config each iteration but takes the
            # loop-level settings from its command line.
            updates = {}
            for param_name in EDITABLE_TRAINING_PARAMS:
                value_str = request.form.get(param_name)
                if value_str is None or value_str.strip() == "":
                    continue
                value_str = value_str.strip()
                if param_name in BOOL_TRAINING_PARAMS:
                    if value_str.lower() in ("true", "false"):
                        updates[param_name] = value_str.lower() == "true"
                    else:
                        flash(f"Invalid value for '{param_name}': '{value_str}'. Left unchanged.", "warning")
                    continue
                try:
                    val = float(value_str)
                except ValueError:
                    val = math.nan
                if not math.isfinite(val):
                    flash(
                        f"Invalid numeric value for training parameter '{param_name}': '{value_str}'. Left unchanged.",
                        "warning",
                    )
                    continue
                updates[param_name] = int(val) if val.is_integer() else val  # Store as int if it's a whole number

            if updates:
                save_training_params(updates)
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for("index"))

    status_data = get_current_status()
    current_config_for_form = get_current_loop_config()

    return render_template(
        "index.html",
        status=status_data,
        config_form=current_config_for_form,
        tensorboard_port=TENSORBOARD_PORT,
        editable_training_params=EDITABLE_TRAINING_PARAMS,
        bool_training_params=BOOL_TRAINING_PARAMS,
    )  # Used for initial render


if __name__ == "__main__":
    # For development: flask run --debug (Flask CLI) or python dashboard.py
    # In production, use a WSGI server like Gunicorn.
    # Example: gunicorn -w 4 'dashboard:app' -b 0.0.0.0:5001
    app.run(debug=True, host="0.0.0.0", port=5001)
