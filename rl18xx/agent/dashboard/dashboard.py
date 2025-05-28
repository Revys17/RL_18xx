from flask import Flask, render_template, request, redirect, url_for, flash
import json
import os
from dataclasses import fields
from rl18xx.agent.alphazero.config import TrainingConfig
EDITABLE_TRAINING_PARAMS = [field.name for field in fields(TrainingConfig)]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_secret_key_change_me") # Use env var for production

# These paths should match those in loop.py
RUNTIME_CONFIG_PATH = "runtime_config.json"
LOOP_STATUS_PATH = "loop_status.json"
TENSORBOARD_URL_PATH = "/tensorboard/"

def get_current_status():
    if os.path.exists(LOOP_STATUS_PATH):
        try:
            with open(LOOP_STATUS_PATH, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"error": "Error reading status file (invalid JSON)."}
        except Exception as e:
            return {"error": f"Status file unreadable: {e}"}
    return {"status_message": "Status file not yet created. Training loop may not be running or hasn't started a loop."}

def get_current_runtime_config():
    config_for_form = {
        "num_games_per_iteration": None, # Placeholder
        "training_params": {param: None for param in EDITABLE_TRAINING_PARAMS}
    }
    
    tc_defaults = TrainingConfig()
    for param_name in EDITABLE_TRAINING_PARAMS:
        if hasattr(tc_defaults, param_name):
            config_for_form["training_params"][param_name] = getattr(tc_defaults, param_name)

    try:
        with open(RUNTIME_CONFIG_PATH, 'r') as f:
            runtime_conf = json.load(f)
        config_for_form["num_games_per_iteration"] = runtime_conf.get("num_games_per_iteration", config_for_form["num_games_per_iteration"])
        
        loaded_training_params = runtime_conf.get("training_params", {})
        for param in EDITABLE_TRAINING_PARAMS:
            if param in loaded_training_params: # Override default if present in file
                config_for_form["training_params"][param] = loaded_training_params[param]
    except Exception as e:
        flash(f"Error loading runtime config: {e}. Displaying defaults/placeholders.", "warning")

    return config_for_form

def save_runtime_config(config_data):
    try:
        with open(RUNTIME_CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)
        flash("Configuration updated. Changes will apply on the next loop iteration.", "success")
    except Exception as e:
        flash(f"Error saving runtime config: {e}", "error")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            num_games_str = request.form.get('num_games_per_iteration')
            if not num_games_str:
                flash("Number of self-play games is required.", "error")
                return redirect(url_for('index'))
            
            num_games = int(num_games_str)
            if num_games <= 0:
                flash("Number of self-play games must be positive.", "error")
                return redirect(url_for('index'))

            new_config_data = {
                "num_games_per_iteration": num_games,
                "training_params": {}
            }
            
            has_training_param_errors = False
            for param_name in EDITABLE_TRAINING_PARAMS:
                value_str = request.form.get(param_name)
                if value_str is not None and value_str.strip() != "":
                    try:
                        # Attempt to convert to float, then int if it's a whole number float
                        val = float(value_str)
                        if val.is_integer():
                            val = int(val)
                        new_config_data["training_params"][param_name] = val
                    except ValueError:
                        flash(f"Invalid numeric value for '{param_name}': '{value_str}'. This parameter will not be saved.", "warning")
                        has_training_param_errors = True
            
            # Only include training_params if there are any valid ones
            if not new_config_data["training_params"]:
                del new_config_data["training_params"]
            
            if not has_training_param_errors or new_config_data.get("training_params"): # Save if no errors or some params are valid
                 save_runtime_config(new_config_data)
            else:
                flash("No valid training parameters were provided to save.", "warning")

        except ValueError:
            flash("Invalid input for number of games. Please enter a valid integer.", "error")
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('index'))

    status_data = get_current_status()
    current_config_for_form = get_current_runtime_config()
    
    return render_template('index.html', status=status_data, config_form=current_config_for_form, 
                           tensorboard_url=TENSORBOARD_URL_PATH, 
                           editable_training_params=EDITABLE_TRAINING_PARAMS)

if __name__ == '__main__':
    # For development: flask run --debug (Flask CLI) or python dashboard.py
    # In production, use a WSGI server like Gunicorn.
    # Example: gunicorn -w 4 'dashboard:app' -b 0.0.0.0:5001
    app.run(debug=True, host='0.0.0.0', port=5001)