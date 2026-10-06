"""Entry point for RL 18xx commands.

Usage:
    python main.py train              Run the AlphaZero training loop (self-play + training)
    python main.py pretrain            Pre-train from human game data
    python main.py convert             Encode cleaned game JSONs to LMDB for pretraining
    python main.py arena               Run an arena match between agents
    python main.py dashboard           Start the training dashboard web server
    python main.py replay <log_file>   Replay a game from a log file in the browser
"""
import argparse
import logging
import sys


def cmd_train(args):
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    from rl18xx.agent.alphazero.loop import main as loop_main

    loop_main(
        num_loop_iterations=args.iterations,
        num_threads=args.threads,
        cleanup=not args.keep_old_files,
        num_readouts=args.readouts,
        max_training_window=args.max_training_window,
        gate_games=args.gate_games,
        gate_threshold=args.gate_threshold,
        no_gate=args.no_gate,
        model_type=args.model_type,
        fresh=args.fresh,
        target_experiences=args.target_experiences,
        batch_size=args.batch_size,
        game_length_schedule=tuple(args.game_length_schedule),
        readout_schedule=tuple(args.readout_schedule),
        inference_server=args.inference_server,
    )


def cmd_pretrain(args):
    import logging

    # Surface the per-epoch summaries (train/val loss, accuracy, value-head
    # diagnostics, checkpoint saves) on stderr alongside the progress bars.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from rl18xx.agent.alphazero.pretraining import do_pretraining
    from rl18xx.agent.alphazero.config import TrainingConfig
    from rl18xx.agent.alphazero.loop import ensure_seed_model

    # Seed the model directory so do_pretraining's get_latest_model() call
    # works on a fresh checkout (it would otherwise FileNotFoundError).
    ensure_seed_model(model_type=args.model_type, force=args.fresh)

    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        value_lr_multiplier=args.value_lr_multiplier,
        value_loss_weight=args.value_loss_weight,
        policy_loss_weight=args.policy_loss_weight,
        price_loss_weight=args.price_loss_weight,
        pretrain_value_joint_epochs=None if args.value_joint_epochs < 0 else args.value_joint_epochs,
        pretrain_value_refit=not args.no_value_refit,
    )
    do_pretraining(
        model_dir=args.model_dir,
        game_data_dir=args.data_dir,
        config=config,
    )


def cmd_clean(args):
    """Filter raw 18xx.games JSONs through the Rust cleaning pipeline.

    Reads raw games from ``--data-dir`` (default ``human_games/1830``),
    runs ``fix_online_games`` over each — which routes through the Rust
    engine by default — and writes the cleaned JSON dicts to
    ``--output``. Games rejected by cleaning (unfinished, unusable
    optional rules, ...) get a stub ``{status: "error", reason: ...}``
    record instead, so downstream pretraining can skip them.
    """
    import os
    from rl18xx.agent.alphazero.pretraining import fix_online_games

    os.makedirs(args.output, exist_ok=True)
    fix_online_games(
        game_data_dir=args.data_dir,
        output_dir=args.output,
        overwrite=args.overwrite,
    )


def _resolve_checkpoint(spec, model_dir):
    """``<session>/<num>`` or a ``.pth`` path under ``model_dir``; None -> its current best."""
    from pathlib import Path

    from rl18xx.agent.alphazero.checkpointer import get_current_best

    if spec:
        if spec.endswith(".pth"):
            return Path(spec)
        session, num = spec.rsplit("/", 1)
        matches = list(Path(model_dir).glob(f"*/{session}/{int(num)}.pth"))
        if not matches:
            raise SystemExit(f"No checkpoint {spec} under {model_dir}")
        return matches[0]
    best = get_current_best(model_dir)
    if best is None:
        raise SystemExit(f"No current_best.json under {model_dir}")
    return Path(model_dir) / best["arch"] / best["session"] / f"{int(best['checkpoint_num'])}.pth"


def cmd_policy_selfplay(args):
    import json
    import multiprocessing
    from pathlib import Path

    multiprocessing.set_start_method("spawn", force=True)
    from rl18xx.agent.alphazero.policy_selfplay import generate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    checkpoint = _resolve_checkpoint(args.checkpoint, args.model_dir)
    stats = generate(
        checkpoint=checkpoint,
        out_dir=Path(args.output),
        num_games=args.games,
        workers=args.workers,
        games_per_task=args.games_per_task,
        positions_per_game=args.positions_per_game,
        temperature=args.temperature,
        opening_decisions=args.opening_decisions,
        opening_temperature=args.opening_temperature,
        max_decisions=args.max_decisions,
        start_positions=args.start_positions,
        random_start_fraction=args.random_start_fraction,
    )
    print(json.dumps({"checkpoint": str(checkpoint), **stats}, indent=2))


def cmd_policy_gradient(args):
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    from rl18xx.agent.alphazero.policy_gradient import PGConfig, run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = PGConfig(
        policy_checkpoint=str(_resolve_checkpoint(args.policy, args.model_dir)),
        value_checkpoint=str(_resolve_checkpoint(args.value, args.value_model_dir)),
        out_dir=args.out_dir,
        run_name=args.run_name,
        workers=args.workers,
        games_per_task=args.games_per_task,
        learner_seats=args.learner_seats,
        positions_per_seat=args.positions_per_seat,
        sl_opponent_fraction=args.sl_opponent_fraction,
        rows_per_update=args.rows_per_update,
        minibatch=args.minibatch,
        ppo_epochs=args.ppo_epochs,
        clip=args.clip,
        lr=args.lr,
        critic_lr=args.critic_lr,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        entropy_coef=args.entropy_coef,
        learner_temperature=args.learner_temperature,
        opponent_temperature=args.opponent_temperature,
        gae_lambda=args.gae_lambda,
        max_updates=args.max_updates,
        snapshot_every=args.snapshot_every,
        pool_refresh_every=args.pool_refresh_every,
        max_decisions=args.max_decisions,
        start_positions=args.start_positions,
        random_start_fraction=args.random_start_fraction,
    )
    print(run(config, resume=args.resume))


def cmd_convert(args):
    import logging

    # Surface the end-of-run summary (games converted, examples written,
    # forced positions skipped).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from rl18xx.agent.alphazero.pretraining import convert_games_to_training_dataset
    from rl18xx.agent.alphazero.encoder import Encoder_1830
    from rl18xx.agent.alphazero.checkpointer import get_latest_model

    model = get_latest_model(args.model_dir)
    encoder = Encoder_1830.get_encoder_for_model(model)
    convert_games_to_training_dataset(
        game_data_dir=args.data_dir,
        encoder=encoder,
        save_path=args.output,
        skip_forced=args.skip_forced,
    )


def cmd_refit_price_head(args):
    """Fit a fresh price head on a checkpoint's frozen trunk from human prices.

    For checkpoints saved with the legacy Gaussian price head (which doesn't
    carry over to the cell-logit head) or any checkpoint whose price head
    should be re-fit. Saves the result as the session's next checkpoint and
    only moves ``current_best`` with ``--promote``.
    """
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from rl18xx.agent.alphazero.checkpointer import get_latest_model, set_current_best
    from rl18xx.agent.alphazero.config import TrainingConfig
    from rl18xx.agent.alphazero.dataset import SelfPlayDataset
    from rl18xx.agent.alphazero.pretraining import refit_price_head

    model = get_latest_model(args.model_dir)
    data = Path(args.data_dir)
    config = TrainingConfig(batch_size=args.batch_size, lr=args.lr)
    result = refit_price_head(
        model,
        SelfPlayDataset(data / "training"),
        SelfPlayDataset(data / "validation"),
        config,
        args.model_dir,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        patience=args.patience,
    )
    print(
        f"Refit price head saved as {result['session']} checkpoint {result['checkpoint_num']}: "
        f"{result['val_bits_per_price']:.3f} bits/price, cell top-1 {result['val_cell_top1']:.3f}"
    )
    if args.promote:
        set_current_best(args.model_dir, model.architecture_name(), result["session"], result["checkpoint_num"])


def cmd_arena(args):
    from rl18xx.agent.arena import Arena
    from rl18xx.agent.alphazero.self_play import MCTSPlayer
    from rl18xx.agent.alphazero.config import SelfPlayConfig, ModelTransformerConfig
    from rl18xx.agent.alphazero.model_transformer import AlphaZeroTransformerModel
    from rl18xx.agent.alphazero.checkpointer import get_latest_model
    from rl18xx.agent.random.random_agent import RandomPlayer

    if args.model_dir:
        model = get_latest_model(args.model_dir)
    else:
        model = AlphaZeroTransformerModel(ModelTransformerConfig())
    model.eval()

    config = SelfPlayConfig(network=model, num_readouts=args.readouts)
    agents = []
    for spec in args.agents:
        if spec == "mcts":
            agents.append(MCTSPlayer(config))
        elif spec == "random":
            agents.append(RandomPlayer())
        else:
            raise ValueError(f"Unknown agent type: {spec}. Use 'mcts' or 'random'.")

    if len(agents) != 4:
        raise ValueError(f"Need exactly 4 agents, got {len(agents)}. Example: --agents mcts mcts random random")

    arena = Arena(*agents, browser=args.browser)
    arena.play()


def cmd_dashboard(args):
    from rl18xx.agent.dashboard.dashboard import app

    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_replay(args):
    from rl18xx.client.replay_game_from_log_file import replay_game_from_log_file

    replay_game_from_log_file(args.log_file)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rl18xx",
        description="RL 18xx - AlphaZero agent for the 1830 board game",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # train
    p = sub.add_parser("train", help="Run the AlphaZero training loop")
    p.add_argument("--iterations", type=int, default=0, help="Training loop iterations (0 = run indefinitely, default: 0)")
    p.add_argument(
        "--target-experiences", type=int, default=10000,
        help="Target experience count per iteration (default: 10000)"
    )
    p.add_argument("--threads", type=int, default=2, help="Parallel self-play threads (default: 2)")
    p.add_argument("--readouts", type=int, default=64, help="MCTS readouts per move (default: 64)")
    p.add_argument("--keep-old-files", action="store_true", help="Keep files from previous runs")
    p.add_argument(
        "--max-training-window", type=int, default=100000,
        help="Max training examples to use (0 = all data, default: 100000)"
    )
    p.add_argument("--gate-games", type=int, default=10, help="Arena games for model gating (default: 10)")
    p.add_argument(
        "--gate-threshold", type=float, default=0.55,
        help="Min win rate to promote, on the 2-player scale; scaled to the 4-player gate's 25%% fair share "
        "(default: 0.55 -> 27.5%%)",
    )
    p.add_argument("--no-gate", action="store_true", help="Disable model gating (always promote)")
    p.add_argument(
        "--model-type", type=str, default="transformer", choices=["gnn", "transformer"],
        help="Model architecture for initial checkpoint (default: v2)"
    )
    p.add_argument(
        "--fresh", action="store_true",
        help="Clear all model checkpoints and training data to start from scratch"
    )
    p.add_argument("--batch-size", type=int, default=256, help="Training batch size (default: 256)")
    p.add_argument(
        "--game-length-schedule", type=int, nargs=3, metavar=("START", "END", "RAMP"),
        default=[150, 1000, 150],
        help="Game length schedule: start end ramp_checkpoints (default: 150 1000 150)"
    )
    p.add_argument(
        "--readout-schedule", type=int, nargs=3, metavar=("START", "END", "RAMP"),
        default=[64, 200, 150],
        help="MCTS readout schedule: start end ramp_checkpoints (default: 64 200 150)"
    )
    p.add_argument(
        "--inference-server", action="store_true",
        help="Route self-play inference through one shared GPU server process (cross-game batching)",
    )

    # pretrain
    p = sub.add_parser("pretrain", help="Pre-train model from human game data")
    p.add_argument("--data-dir", type=str, default="human_games", help="Directory with game JSON files or LMDB path")
    p.add_argument("--model-dir", type=str, default="model_checkpoints", help="Model checkpoint directory")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    p.add_argument(
        "--fresh", action="store_true",
        help="Start from a newly initialized model instead of the latest checkpoint",
    )
    # Value-head defaults come from the 2026-09 pretraining sweep: with ~2,900
    # human games the value head generalizes for under an epoch and then
    # memorizes games, while the policy keeps improving for ~5. So stage 1
    # trains jointly with a small value weight (keeps value features in the
    # trunk without letting value overfitting drive checkpoint selection), and
    # stage 2 re-fits fresh value heads on the frozen best checkpoint with
    # sub-epoch early stopping.
    p.add_argument(
        "--value-joint-epochs", type=int, default=-1,
        help="Epochs during which value gradients reach the trunk; afterwards the value heads "
        "train on detached features (default: -1 = joint throughout; 0 = detached from the start)",
    )
    p.add_argument(
        "--value-lr-multiplier", type=float, default=1.0,
        help="Value-head LR multiplier relative to --lr (default: 1.0)"
    )
    p.add_argument(
        "--value-loss-weight", type=float, default=0.1,
        help="Weight of the value loss in the total (default: 0.1)"
    )
    p.add_argument(
        "--policy-loss-weight", type=float, default=1.0,
        help="Weight of the policy loss (default: 1.0; 0 trains a value network only)",
    )
    p.add_argument(
        "--price-loss-weight", type=float, default=0.1, help="Weight of the price-head loss (default: 0.1)"
    )
    p.add_argument(
        "--no-value-refit", action="store_true",
        help="Skip re-fitting the value heads on the frozen best checkpoint after training",
    )
    p.add_argument(
        "--model-type", type=str, default="transformer", choices=["gnn", "transformer"],
        help="Model architecture if no seed checkpoint exists (default: transformer)",
    )

    # clean (Rust-engine filtering + cleaning of raw 18xx.games JSONs)
    p = sub.add_parser("clean", help="Filter raw 18xx.games JSONs through the Rust cleaning pipeline")
    p.add_argument("--data-dir", type=str, default="human_games/1830", help="Directory of raw game JSONs (default: human_games/1830)")
    p.add_argument("--output", type=str, default="human_games/1830_clean", help="Output directory for cleaned JSONs (default: human_games/1830_clean)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite already-cleaned outputs (default: skip)")

    # policy-selfplay (fast search-free games -> value-network training data)
    p = sub.add_parser(
        "policy-selfplay", help="Play search-free self-play games with a checkpoint's policy and write value data"
    )
    p.add_argument("--checkpoint", type=str, default=None, help="<session>/<num> or a .pth (default: current best)")
    p.add_argument("--model-dir", type=str, default="model_checkpoints")
    p.add_argument("--output", type=str, required=True, help="LMDB root (training/ and validation/ inside)")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--games-per-task", type=int, default=32, help="Games each worker plays concurrently")
    p.add_argument("--positions-per-game", type=int, default=4, help="Positions kept per game (uniform sample)")
    p.add_argument("--temperature", type=float, default=1.0, help="Policy sampling temperature after the opening")
    p.add_argument("--opening-decisions", type=int, default=0, help="Decisions sampled at --opening-temperature")
    p.add_argument("--opening-temperature", type=float, default=1.0)
    p.add_argument("--max-decisions", type=int, default=1000)
    p.add_argument("--start-positions", type=str, default="human_games/start_positions_1830_4p.jsonl")
    p.add_argument("--random-start-fraction", type=float, default=0.2)

    # policy-gradient (AlphaGo's RL-policy stage: refine the supervised policy by self-play)
    p = sub.add_parser(
        "policy-gradient",
        help="Refine a policy by policy-gradient self-play against itself and an opponent pool (no search)",
    )
    p.add_argument("--policy", type=str, default=None, help="Policy <session>/<num> or .pth (default: current best)")
    p.add_argument("--model-dir", type=str, default="model_checkpoints")
    p.add_argument("--value", type=str, default=None, help="Critic <session>/<num> or .pth (default: its current best)")
    p.add_argument("--value-model-dir", type=str, default="model_checkpoints_value")
    p.add_argument("--out-dir", type=str, default="model_checkpoints_pg")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue from its last snapshot")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--games-per-task", type=int, default=32)
    p.add_argument("--learner-seats", type=int, default=2)
    p.add_argument("--positions-per-seat", type=int, default=32, help="Learner decisions kept per seat per game")
    p.add_argument("--sl-opponent-fraction", type=float, default=0.5, help="Games against the supervised policy")
    p.add_argument("--rows-per-update", type=int, default=65536)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=3e-5)
    p.add_argument("--kl-coef", type=float, default=0.02, help="KL(learner || supervised) penalty")
    p.add_argument("--kl-target", type=float, default=None, help="Adapt --kl-coef toward this KL per decision")
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument(
        "--learner-temperature", type=float, default=1.0,
        help="The learner plays and is trained as softmax(logits / T) (KL anchor at the same T)",
    )
    p.add_argument("--opponent-temperature", type=float, default=1.0, help="Sampling temperature of opponents")
    p.add_argument("--gae-lambda", type=float, default=1.0, help="1 = Monte Carlo returns")
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--pool-refresh-every", type=int, default=5)
    p.add_argument("--max-decisions", type=int, default=1000)
    p.add_argument("--start-positions", type=str, default="human_games/start_positions_1830_4p.jsonl")
    p.add_argument("--random-start-fraction", type=float, default=0.2)

    # convert (encode games to LMDB for pretraining)
    p = sub.add_parser("convert", help="Convert cleaned game JSONs to LMDB training data")
    p.add_argument("--data-dir", type=str, default="human_games/1830_clean", help="Directory with cleaned game JSON files")
    p.add_argument("--output", type=str, default="human_games/lmdb_v2", help="Output directory for LMDB data")
    p.add_argument("--model-dir", type=str, default="model_checkpoints", help="Model checkpoint (determines encoder)")
    p.add_argument(
        "--skip-forced", action="store_true",
        help="Don't record positions with a single legal action (as self-play doesn't); off by default because "
        "it made pretraining worse (see pretraining.convert_game_to_training_data)",
    )

    # refit-price-head (fit the price head on a frozen checkpoint)
    p = sub.add_parser("refit-price-head", help="Fit a fresh price head on the current-best checkpoint's frozen trunk")
    p.add_argument("--model-dir", type=str, default="model_checkpoints", help="Model checkpoint directory")
    p.add_argument(
        "--data-dir", type=str, default="human_games/lmdb_v3",
        help="Converted human-game LMDB root with training/ and validation/ (default: human_games/lmdb_v3)",
    )
    p.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    p.add_argument("--max-steps", type=int, default=4000, help="Maximum optimizer steps (default: 4000)")
    p.add_argument("--eval-every", type=int, default=200, help="Validate every N steps (default: 200)")
    p.add_argument("--patience", type=int, default=5, help="Stop after N validations without improvement (default: 5)")
    p.add_argument("--promote", action="store_true", help="Point current_best at the refit checkpoint")

    # arena
    p = sub.add_parser("arena", help="Run a match between agents")
    p.add_argument(
        "--agents",
        nargs=4,
        default=["mcts", "mcts", "mcts", "mcts"],
        help="4 agent types: 'mcts' or 'random' (default: mcts mcts mcts mcts)",
    )
    p.add_argument("--model-dir", type=str, default=None, help="Model checkpoint directory")
    p.add_argument("--readouts", type=int, default=200, help="MCTS readouts per move (default: 200)")
    p.add_argument("--browser", action="store_true", help="Show game in browser via 18xx.games")

    # dashboard
    p = sub.add_parser("dashboard", help="Start the training dashboard")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    # replay
    p = sub.add_parser("replay", help="Replay a game from a log file in the browser")
    p.add_argument("log_file", type=str, help="Path to the game log file")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "pretrain": cmd_pretrain,
        "clean": cmd_clean,
        "convert": cmd_convert,
        "policy-selfplay": cmd_policy_selfplay,
        "policy-gradient": cmd_policy_gradient,
        "refit-price-head": cmd_refit_price_head,
        "arena": cmd_arena,
        "dashboard": cmd_dashboard,
        "replay": cmd_replay,
    }
    commands[args.command](args)
