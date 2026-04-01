"""Entry point for RL 18xx commands.

Usage:
    python main.py train              Run the AlphaZero training loop (self-play + training)
    python main.py pretrain            Pre-train from human game data
    python main.py arena               Run an arena match between agents
    python main.py dashboard           Start the training dashboard web server
    python main.py replay <log_file>   Replay a game from a log file in the browser
"""
import argparse
import sys


def cmd_train(args):
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    from rl18xx.agent.alphazero.loop import main as loop_main
    loop_main(
        num_loop_iterations=args.iterations,
        num_games_per_iteration=args.games,
        num_threads=args.threads,
        cleanup=not args.keep_old_files,
        num_readouts=args.readouts,
        max_training_window=args.max_training_window,
    )


def cmd_pretrain(args):
    from rl18xx.agent.alphazero.pretraining import do_pretraining
    from rl18xx.agent.alphazero.config import TrainingConfig
    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    do_pretraining(
        model_dir=args.model_dir,
        game_data_dir=args.data_dir,
        config=config,
    )


def cmd_arena(args):
    from rl18xx.agent.arena import Arena
    from rl18xx.agent.alphazero.self_play import MCTSPlayer
    from rl18xx.agent.alphazero.config import SelfPlayConfig, ModelConfig
    from rl18xx.agent.alphazero.model import AlphaZeroGNNModel
    from rl18xx.agent.alphazero.checkpointer import get_latest_model
    from rl18xx.agent.random.random_agent import RandomPlayer

    if args.model_dir:
        model = get_latest_model(args.model_dir)
    else:
        model = AlphaZeroGNNModel(ModelConfig())
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
    p.add_argument("--iterations", type=int, default=5, help="Training loop iterations (default: 5)")
    p.add_argument("--games", type=int, default=25, help="Self-play games per iteration (default: 25)")
    p.add_argument("--threads", type=int, default=2, help="Parallel self-play threads (default: 2)")
    p.add_argument("--readouts", type=int, default=64, help="MCTS readouts per move (default: 64)")
    p.add_argument("--keep-old-files", action="store_true", help="Keep files from previous runs")
    p.add_argument(
        "--max-training-window", type=int, default=0,
        help="Max training examples to use (0 = all data, default: 0)"
    )

    # pretrain
    p = sub.add_parser("pretrain", help="Pre-train model from human game data")
    p.add_argument("--data-dir", type=str, default="human_games", help="Directory with game JSON files")
    p.add_argument("--model-dir", type=str, default="model_checkpoints", help="Model checkpoint directory")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")

    # arena
    p = sub.add_parser("arena", help="Run a match between agents")
    p.add_argument("--agents", nargs=4, default=["mcts", "mcts", "mcts", "mcts"],
                    help="4 agent types: 'mcts' or 'random' (default: mcts mcts mcts mcts)")
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
        "arena": cmd_arena,
        "dashboard": cmd_dashboard,
        "replay": cmd_replay,
    }
    commands[args.command](args)
