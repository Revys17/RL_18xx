from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import json
import os
from dataclasses import fields
from rl18xx.agent.alphazero.config import TrainingConfig
from rl18xx.shared.atomic_io import atomic_write_json
import datetime
import time
from pathlib import Path
from datetime import datetime
import psutil

# Fields to exclude from the training parameters display
EXCLUDED_FIELDS = {"root_dir", "train_dir", "val_dir", "model_checkpoint_dir", "metrics", "global_step"}
EDITABLE_TRAINING_PARAMS = [field.name for field in fields(TrainingConfig) if field.name not in EXCLUDED_FIELDS]

app = Flask(__name__)
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
TENSORBOARD_URL_PATH = "http://localhost:6006"


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


def get_current_loop_config():
    """Reads the loop_config.json file and provides defaults."""
    default_config = {
        "num_loop_iterations": 5,
        "num_games_per_iteration": 25,
        "num_threads": 2,
        "training_params": TrainingConfig().to_json(),
        "num_readouts": 32,
    }

    if not LOOP_CONFIG_FILE_PATH.exists():
        flash("Loop configuration file not found. Displaying default values. Save to create one.", "info")
        return default_config

    try:
        with open(LOOP_CONFIG_FILE_PATH, "r") as f:
            loaded_config_json = json.load(f)

        # Start with defaults and override with loaded values
        config_for_form = default_config.copy()
        config_for_form["num_loop_iterations"] = loaded_config_json.get(
            "num_loop_iterations", default_config["num_loop_iterations"]
        )
        config_for_form["num_games_per_iteration"] = loaded_config_json.get(
            "num_games_per_iteration", default_config["num_games_per_iteration"]
        )
        config_for_form["num_threads"] = loaded_config_json.get("num_threads", default_config["num_threads"])
        config_for_form["num_readouts"] = loaded_config_json.get("num_readouts", default_config["num_readouts"])

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


def save_loop_config(config_data_to_save):
    """Saves the provided configuration data to loop_config.json."""
    try:
        atomic_write_json(LOOP_CONFIG_FILE_PATH, config_data_to_save, indent=4)
        flash("Loop configuration updated. Changes will apply on the next loop iteration.", "success")
    except Exception as e:
        flash(f"Error saving loop config: {e}", "error")


def get_games_in_progress(loop=None):
    if not SELF_PLAY_GAMES_STATUS_PATH.exists():
        return {"error": "Self-play games status file not found."}

    games_data = []
    try:
        if loop is None:
            all_games = SELF_PLAY_GAMES_STATUS_PATH.glob("*.json")
        else:
            all_games = SELF_PLAY_GAMES_STATUS_PATH.glob(f"L{loop}_G*.json")
        for game_file in all_games:
            with open(game_file, "r") as f:
                content = f.read()
                if content:  # Ensure file is not empty
                    # Use game_file.name (string) as key for JSON compatibility
                    game_data = json.loads(content)
                    game_data["start_time_str"] = datetime.fromtimestamp(game_data.get("start_time_unix", 0)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    game_data["last_update_str"] = datetime.fromtimestamp(
                        game_data.get("last_update_unix", time.time())
                    ).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    continue

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

    except json.JSONDecodeError as e:
        return {"error": f"Error reading self-play games status file: {e} (JSONDecodeError)."}
    except Exception as e:
        return {"error": f"Error reading self-play games status: {e}"}
    return sorted(games_data, key=lambda x: x["start_time_unix"], reverse=False)


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
        data = request.get_json()
        if not data:
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
    loop = request.args.get("loop", type=int)
    games_data = get_games_in_progress(loop=loop)
    if isinstance(games_data, dict) and "error" in games_data:
        return jsonify(games_data), 500
    return jsonify(games_data)


@app.route("/api/current_status")
def api_current_status():
    status = get_current_status()
    return jsonify(status)


@app.route("/api/metrics_history")
def api_metrics_history():
    """Return metrics history records. Optional ?last=N to limit to recent entries."""
    if not METRICS_HISTORY_PATH.exists():
        return jsonify([])
    try:
        records = []
        with open(METRICS_HISTORY_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        last_n = request.args.get("last", type=int)
        if last_n is not None and last_n > 0:
            records = records[-last_n:]
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": f"Failed to read metrics history: {e}"}), 500


@app.route("/api/model_history")
def api_model_history():
    """Return model gating history records."""
    if not MODEL_HISTORY_PATH.exists():
        return jsonify([])
    try:
        records = []
        with open(MODEL_HISTORY_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return jsonify(records)
    except Exception as e:
        return jsonify({"error": f"Failed to read model history: {e}"}), 500


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

        # GPU memory (only when CUDA is available in this process)
        try:
            import torch

            if torch.cuda.is_available():
                response["gpu_memory_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
                response["gpu_max_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        except ImportError:
            pass

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Failed to get system metrics: {e}"}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # --- General Loop Parameters ---
            num_loop_iterations_str = request.form.get("num_loop_iterations")
            num_games_str = request.form.get("num_games_per_iteration")
            num_threads_str = request.form.get("num_threads")
            num_readouts_str = request.form.get("num_readouts")

            if not num_loop_iterations_str or not num_games_str or not num_threads_str or not num_readouts_str:
                flash("Number of loop iterations, self-play games, and threads are required.", "error")
                return redirect(url_for("index"))

            num_loop_iterations = int(num_loop_iterations_str)
            num_games = int(num_games_str)
            num_threads = int(num_threads_str)
            num_readouts = int(num_readouts_str)

            if num_loop_iterations < 0 or num_games <= 0 or num_threads <= 0 or num_readouts <= 0:
                flash("Number of games, threads, and readouts must be positive integers. Loop iterations can be 0 (indefinite).", "error")
                return redirect(url_for("index"))

            # --- Training Parameters ---
            training_config_data = {}
            has_training_param_errors = False
            for param_name in EDITABLE_TRAINING_PARAMS:
                value_str = request.form.get(param_name)
                if value_str is not None and value_str.strip() != "":
                    try:
                        val = float(value_str)
                        if val.is_integer():  # Store as int if it's a whole number
                            val = int(val)
                        training_config_data[param_name] = val
                    except ValueError:
                        flash(
                            f"Invalid numeric value for training parameter '{param_name}': '{value_str}'. This parameter will use its current or default value.",
                            "warning",
                        )
                        has_training_param_errors = True

            # --- Construct the full config to save ---
            # Get current config to fill in any training params not successfully parsed from form
            current_config = get_current_loop_config()
            final_training_config = current_config["training_params"].copy()
            # Override with successfully parsed form values
            final_training_config.update(training_config_data)

            loop_config_to_save = {
                "num_loop_iterations": num_loop_iterations,
                "num_games_per_iteration": num_games,
                "num_threads": num_threads,
                "training_config": final_training_config,  # Use "training_config" as key for the file
                "num_readouts": num_readouts,
            }

            save_loop_config(loop_config_to_save)

        except ValueError:
            flash("Invalid input for numeric fields. Please enter valid integers.", "error")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for("index"))

    status_data = get_current_status()
    current_config_for_form = get_current_loop_config()

    # Initial load of games in progress
    games_in_progress_result = get_games_in_progress()
    games_in_progress_for_template = {}

    if "error" in games_in_progress_result:
        flash(f"Initial load error for self-play games: {games_in_progress_result['error']}", "warning")
        # Pass the error to the template for initial display if needed
        games_in_progress_for_template = {"error": games_in_progress_result["error"]}
    else:
        games_in_progress_for_template = games_in_progress_result

    return render_template(
        "index.html",
        status=status_data,
        config_form=current_config_for_form,
        tensorboard_url=TENSORBOARD_URL_PATH,
        editable_training_params=EDITABLE_TRAINING_PARAMS,
        games_in_progress=games_in_progress_for_template,
    )  # Used for initial render


if __name__ == "__main__":
    # For development: flask run --debug (Flask CLI) or python dashboard.py
    # In production, use a WSGI server like Gunicorn.
    # Example: gunicorn -w 4 'dashboard:app' -b 0.0.0.0:5001
    app.run(debug=True, host="0.0.0.0", port=5001)
