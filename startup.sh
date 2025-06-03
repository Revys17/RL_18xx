#!/bin/bash

# Function to handle cleanup on exit
cleanup() {
    echo "Shutting down processes..."
    # Kill all child processes
    jobs -p | xargs -r kill
    wait
    echo "All processes stopped."
    exit 0
}

# Set up trap to catch signals
trap cleanup SIGINT SIGTERM EXIT

# Start TensorBoard
echo "Starting TensorBoard on port 6006..."
uv run tensorboard --logdir ./runs/alphazero_runs --bind_all &
TENSORBOARD_PID=$!

# Start Dashboard
echo "Starting Dashboard on port 5001..."
uv run gunicorn --chdir ./rl18xx/agent/dashboard -w 4 'dashboard:app' -b 0.0.0.0:5001 &
DASHBOARD_PID=$!

# Give services time to start
sleep 3

echo "All processes started:"
echo "  - TensorBoard: http://localhost:6006 (PID: $TENSORBOARD_PID)"
echo "  - Dashboard: http://localhost:5001 (PID: $DASHBOARD_PID)"
echo ""
echo "Press Ctrl+C to stop all processes."

# Wait for all background processes
wait
