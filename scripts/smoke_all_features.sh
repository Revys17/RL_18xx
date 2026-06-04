#!/usr/bin/env bash
# Run all production-feature smokes end-to-end. Each subtest is independent
# and runs to completion before the next starts. Total wall ~10-15 min on CPU.
#
# Usage:
#   ./scripts/smoke_all_features.sh           # all smokes
#   ./scripts/smoke_all_features.sh fast      # skip the resign-calibration sweep (longest)
#
# What each smoke covers:
#   1. Python MCTS path: in-process inference, resign + trace ON
#   2. Rust MCTS path:   in-process inference, resign + trace ON
#   3. Phase 3 inference server unit tests (in-thread)
#   4. Phase 3 inference server LIVE: full training iteration (~1 game, GPU)
#   5. Phase 3 + Phase 4 LIVE: training iteration with both flags on
#   6. Resign calibration sweep: 3 short iterations to observe calibration logs
#   7. Rust vs Python parity: alphazero test suite
#   8. Trace renderer: scripts/view_trace.py on a fresh trace
#
# Each smoke creates its own tempdir + loop_config.json. Existing
# loop_config.json is backed up before any LIVE smoke and restored at the end.

set -e
cd "$(dirname "$0")/.."

MODE="${1:-full}"
PYTHONPATH=. export PYTHONPATH
BACKUP_CONFIG=""
TRACE_LAST=""

backup_loop_config() {
  if [ -f loop_config.json ]; then
    BACKUP_CONFIG="$(mktemp)"
    cp loop_config.json "$BACKUP_CONFIG"
    echo "  (backed up existing loop_config.json to $BACKUP_CONFIG)"
  fi
}

restore_loop_config() {
  if [ -n "$BACKUP_CONFIG" ] && [ -f "$BACKUP_CONFIG" ]; then
    mv "$BACKUP_CONFIG" loop_config.json
    echo "  (restored original loop_config.json)"
  elif [ -f loop_config.json ]; then
    rm -f loop_config.json
    echo "  (removed test loop_config.json)"
  fi
}

trap restore_loop_config EXIT

banner() {
  echo
  echo "==============================================================="
  echo "  $1"
  echo "==============================================================="
}

# ----------- Smoke 1: Python MCTS + resign + trace -------------------
banner "[1/8] Python MCTS path (resign+trace ON, use_rust_mcts=False)"
cat > /tmp/smoke_py_full.py << 'EOF'
import sys, tempfile, time, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
from rl18xx.agent.alphazero.config import SelfPlayConfig, ModelTransformerConfig, TraceConfig
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.self_play import SelfPlay
from rl18xx.agent.alphazero.checkpointer import save_model, get_latest_model
with tempfile.TemporaryDirectory() as td:
    md = Path(td) / "models"; md.mkdir()
    save_model(AlphaZeroTransformerModel(ModelTransformerConfig()), str(md))
    trace_dir = Path(td) / "traces"
    cfg = SelfPlayConfig(
        network=get_latest_model(str(md)),
        num_readouts=16, parallel_readouts=4, max_game_length=60,
        use_rust_mcts=False,
        enable_resign=True, resign_window=4,
        resign_high_threshold=0.40, resign_gap_threshold=0.05,
        noresign_holdout_rate=1.0,
        trace=TraceConfig(trace_game_rate=1.0, trace_every_n_moves=2,
                          traces_per_move=2, output_dir=trace_dir),
        game_id="smoke-py", game_idx_in_iteration=0, global_step=0,
        selfplay_dir=Path(td) / "selfplay",
    )
    t0 = time.time(); SelfPlay(cfg).run_game()
    print(f"  game in {time.time() - t0:.1f}s")
    tf = list((trace_dir / "0").glob("*.jsonl"))
    print(f"  trace files: {tf}")
    assert tf, "no trace files emitted"
    # Persist a copy for smoke 8.
    import shutil
    out = Path("/tmp/smoke_trace_last.jsonl")
    shutil.copy(tf[0], out)
    print(f"  copied trace to {out}")
print("  Python smoke OK")
EOF
uv run python /tmp/smoke_py_full.py
TRACE_LAST=/tmp/smoke_trace_last.jsonl

# ----------- Smoke 2: Rust MCTS + resign + trace ---------------------
banner "[2/8] Rust MCTS path (resign+trace ON, use_rust_mcts=True)"
cat > /tmp/smoke_rs_full.py << 'EOF'
import sys, tempfile, time, json, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
from rl18xx.agent.alphazero.config import SelfPlayConfig, ModelTransformerConfig, TraceConfig
from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
from rl18xx.agent.alphazero.self_play import SelfPlay
from rl18xx.agent.alphazero.checkpointer import save_model, get_latest_model
with tempfile.TemporaryDirectory() as td:
    md = Path(td) / "models"; md.mkdir()
    save_model(AlphaZeroTransformerModel(ModelTransformerConfig()), str(md))
    trace_dir = Path(td) / "traces"
    cfg = SelfPlayConfig(
        network=get_latest_model(str(md)),
        num_readouts=16, parallel_readouts=4, max_game_length=60,
        use_rust_mcts=True,
        enable_resign=True, resign_window=4,
        resign_high_threshold=0.40, resign_gap_threshold=0.05,
        noresign_holdout_rate=1.0,
        trace=TraceConfig(trace_game_rate=1.0, trace_every_n_moves=2,
                          traces_per_move=2, output_dir=trace_dir),
        game_id="smoke-rs", game_idx_in_iteration=0, global_step=0,
        selfplay_dir=Path(td) / "selfplay",
    )
    t0 = time.time(); SelfPlay(cfg).run_game()
    print(f"  game in {time.time() - t0:.1f}s")
    tf = list((trace_dir / "0").glob("*.jsonl"))
    assert tf, "no trace files emitted"
    header = json.loads(tf[0].read_text().split("\n")[0])
    assert header.get("engine") == "rust", f"expected engine=rust, got {header}"
    print(f"  trace header.engine=rust confirmed")
print("  Rust smoke OK")
EOF
uv run python /tmp/smoke_rs_full.py

# ----------- Smoke 3: Inference-server unit tests --------------------
banner "[3/8] Phase 3 inference server unit tests"
uv run pytest tests/agent/alphazero/test_inference_server.py -q

# ----------- Smoke 4: Inference server LIVE (real subprocess) --------
banner "[4/8] Phase 3 inference server LIVE: one short training iteration"
backup_loop_config
cat > loop_config.json << 'EOF'
{
  "use_inference_server": true,
  "inference_batch_size": 16,
  "inference_batch_timeout_ms": 5.0,
  "training_config": {"batch_size": 32, "num_epochs": 1, "lr": 0.001},
  "num_loop_iterations": 1,
  "num_games_per_iteration": 2,
  "num_threads": 2,
  "target_experiences": 100,
  "num_readouts": 16
}
EOF
echo "  Running 1 iteration with --threads 2 --readouts 16 --target-experiences 100 ..."
uv run python main.py train --threads 2 --readouts 16 --target-experiences 100 \
  --iterations 1 --game-length-schedule 30 30 1 --readout-schedule 16 16 1 \
  2>&1 | grep -E "Spawning inference server|paused before training|reloaded|server shut down|Self-play game L0_G[0-9]+ completed|Loop 1: Total experiences|ERROR" | head -30
restore_loop_config

# ----------- Smoke 5: Inference server + Rust MCTS combined ----------
banner "[5/8] Phase 3 + Phase 4 LIVE: server ON + Rust MCTS ON"
backup_loop_config
cat > loop_config.json << 'EOF'
{
  "use_inference_server": true,
  "use_rust_mcts": true,
  "inference_batch_size": 16,
  "inference_batch_timeout_ms": 5.0,
  "training_config": {"batch_size": 32, "num_epochs": 1, "lr": 0.001},
  "num_loop_iterations": 1,
  "num_games_per_iteration": 2,
  "num_threads": 2,
  "target_experiences": 100,
  "num_readouts": 16
}
EOF
uv run python main.py train --threads 2 --readouts 16 --target-experiences 100 \
  --iterations 1 --game-length-schedule 30 30 1 --readout-schedule 16 16 1 \
  2>&1 | grep -E "Spawning inference server|paused before training|reloaded|server shut down|RustMCTSPlayer|Self-play game L0_G[0-9]+ completed|Loop 1: Total experiences|ERROR" | head -30
restore_loop_config

# ----------- Smoke 6: Resign calibration sweep -----------------------
if [ "$MODE" != "fast" ]; then
banner "[6/8] Resign auto-calibration: 3 short iterations"
backup_loop_config
cat > loop_config.json << 'EOF'
{
  "use_rust_mcts": false,
  "noresign_holdout_rate": 0.5,
  "resign_window": 4,
  "resign_high_threshold": 0.45,
  "resign_gap_threshold": 0.10,
  "training_config": {"batch_size": 32, "num_epochs": 1, "lr": 0.001},
  "num_loop_iterations": 3,
  "num_games_per_iteration": 4,
  "num_threads": 2,
  "target_experiences": 200,
  "num_readouts": 16
}
EOF
uv run python main.py train --threads 2 --readouts 16 --target-experiences 200 \
  --iterations 3 --game-length-schedule 60 60 1 --readout-schedule 16 16 1 \
  2>&1 | grep -E "resign calibration|threshold:|holdout|Resign/High_Threshold|Loop [0-9]+: Total experiences" | head -40
restore_loop_config
else
echo
echo "[6/8] SKIPPED (fast mode)"
fi

# ----------- Smoke 7: Engine + MCTS parity tests ---------------------
banner "[7/8] Engine + MCTS parity tests"
uv run pytest tests/agent/alphazero/test_rust_mcts_parity.py \
              tests/agent/alphazero/test_rust_mcts_parity_pw.py \
              tests/agent/alphazero/test_rust_mcts_player_e2e.py \
              tests/agent/alphazero/test_rust_mcts_resign_trace.py \
              tests/agent/alphazero/test_action_mapper_thin_shim.py -q

# ----------- Smoke 8: Trace renderer ---------------------------------
banner "[8/8] Trace renderer on a fresh JSONL"
if [ -n "$TRACE_LAST" ] && [ -f "$TRACE_LAST" ]; then
  uv run python scripts/view_trace.py "$TRACE_LAST" | head -25
  echo "  ..."
else
  echo "  SKIPPED (no trace file from smoke 1)"
fi

banner "ALL SMOKES PASSED"
echo "If anything above flagged an error you missed, scroll up and look for"
echo "'ERROR' or stack traces. Run 'fast' to skip the resign calibration sweep:"
echo "    ./scripts/smoke_all_features.sh fast"
