uv run tensorboard --logdir ./runs/alphazero_runs
uv run gunicorn --chdir ./rl18xx/agent/dashboard -w 4 'dashboard:app' -b 0.0.0.0:5001
uv run python -m rl18xx.agent.alphazero.loop --num-loop-iterations 1 --num-games-per-iteration 100 --num-threads 10
