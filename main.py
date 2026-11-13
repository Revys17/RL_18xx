"""Entry point for RL 18xx commands.

Usage:
    python main.py train              Run the AlphaZero training loop (self-play + training)
    python main.py pretrain            Pre-train from human game data
    python main.py convert             Encode cleaned game JSONs to LMDB for pretraining
    python main.py convert-generic     Cleaned games -> supervised records for the title-generic model
    python main.py pretrain-generic    Two-stage supervised pretraining of the title-generic model
    python main.py compare-generic     A generic vs a legacy checkpoint on the same validation positions
    python main.py arena               Run an arena match between agents
    python main.py dashboard           Start the training dashboard web server
    python main.py replay <game_file>  Check a saved game in the Python engine and open it in the game viewer
    python main.py advisor             Serve the model advisor's backend for the browser extension (extension/)
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
    # In --model-dir: with --fresh and another directory (a value network's)
    # the seed used to land in model_checkpoints and repoint the policy's
    # current_best.json.
    ensure_seed_model(model_type=args.model_type, force=args.fresh, model_dir=args.model_dir)

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
    from rl18xx.agent.alphazero.policy_gradient import RESUME_SETTINGS, PGConfig, default_learner_seats, run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    player_mix = _by_players(args.player_mix, float, "--player-mix")
    if player_mix:  # the start files of every count: templates with {n}
        args.start_positions = args.start_positions or "human_games/start_positions_1830_{n}p_v2.jsonl"
        args.midgame_positions = args.midgame_positions or "human_games/midgame_positions_1830_{n}p_v2.jsonl"
    _player_count_defaults(args)
    config = PGConfig(
        policy_checkpoint=str(_resolve_checkpoint(args.policy, args.model_dir)),
        value_checkpoint=(
            "" if args.shared and not args.value else str(_resolve_checkpoint(args.value, args.value_model_dir))
        ),
        shared=args.shared,
        value_follow=args.value_follow,
        value_pg_weight=args.value_pg_weight,
        out_dir=args.out_dir,
        run_name=args.run_name,
        workers=args.workers,
        games_per_task=args.games_per_task,
        num_players=args.num_players,
        player_mix=player_mix,
        max_decisions_by_players=_by_players(args.max_decisions_by_players, int, "--max-decisions-by-players"),
        auction_stall_by_players=_by_players(args.auction_stall_by_players, int, "--auction-stall-by-players"),
        learner_seats=args.learner_seats if args.learner_seats is not None else default_learner_seats(args.num_players),
        positions_per_seat=args.positions_per_seat,
        sl_opponent_fraction=args.sl_opponent_fraction,
        opponent_checkpoint=(
            str(_resolve_checkpoint(args.opponent, args.model_dir)) if args.opponent else None
        ),
        anchor_every=args.anchor_every,
        population=args.population,
        opponent_slots=args.opponent_slots,
        learner_seat_weights=(
            {int(k): float(v) for k, v in (part.split(":") for part in args.learner_seat_weights.split(","))}
            if args.learner_seat_weights
            else None
        ),
        pfsp_power=args.pfsp_power,
        slot_refresh_every=args.slot_refresh_every,
        league_snapshot_every=args.league_snapshot_every,
        league_max_snapshots=args.league_max_snapshots,
        reward=args.reward,
        rows_per_update=args.rows_per_update,
        minibatch=args.minibatch,
        microbatch=args.microbatch,
        ppo_epochs=args.ppo_epochs,
        clip=args.clip,
        lr=args.lr,
        critic_lr=args.critic_lr,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        kl_coef_min=args.kl_coef_min,
        entropy_coef=args.entropy_coef,
        learner_temperature=args.learner_temperature,
        opponent_temperature=args.opponent_temperature,
        opponent_price_eps=args.opponent_price_eps,
        gae_lambda=args.gae_lambda,
        max_updates=args.max_updates,
        snapshot_every=args.snapshot_every,
        pool_refresh_every=args.pool_refresh_every,
        max_decisions=args.max_decisions,
        start_positions=args.start_positions,
        random_start_fraction=args.random_start_fraction,
        save_game_every=args.save_game_every,
        midgame_positions=args.midgame_positions if args.midgame_fraction > 0 or args.resume else None,
        midgame_fraction=args.midgame_fraction,
        auction_fraction=args.auction_fraction,
        auction_unlock=args.auction_unlock,
        auction_stall_decisions=args.auction_stall_decisions,
    )
    explicit = _resume_explicit(args, RESUME_SETTINGS)
    print(run(config, resume=args.resume, explicit=explicit))


def cmd_search_learning(args):
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    from rl18xx.agent.alphazero.search_learning import RESUME_SETTINGS, SearchLearningConfig, run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _player_count_defaults(args)
    resuming = args.resume and not (args.policy or args.value)  # a resumed run keeps its checkpoints
    no_critic = (args.shared or args.value_follow) and not args.value
    if not resuming and not args.shared and not args.value and not args.value_follow:
        raise SystemExit("--value (the critic) is needed unless --shared (its own value heads) or --value-follow")
    config = SearchLearningConfig(
        policy_checkpoint="" if resuming else str(_resolve_checkpoint(args.policy, args.model_dir)),
        value_checkpoint=(
            "" if resuming or no_critic else str(_resolve_checkpoint(args.value, args.value_model_dir))
        ),
        out_dir=args.out_dir,
        run_name=args.run_name,
        learner_temperature=args.learner_temperature,
        target=args.target,
        visit_target=args.visit_target,
        target_temperature=args.target_temperature,
        cq_rescale=args.cq_rescale,
        rpo_c=args.rpo_c,
        rpo_unvisited=args.rpo_unvisited,
        rpo_qnorm=args.rpo_qnorm,
        q_normalize=args.q_normalize,
        move_widening=not args.no_move_widening,
        fast_row_weight=args.fast_row_weight,
        fast_row_value=not args.no_fast_row_value,
        p_full=args.p_full,
        n_full=args.n_full,
        n_fast=args.n_fast,
        c_puct_init=args.c_puct,
        mean_q=not args.no_mean_q,
        fpu_reduction=args.fpu_reduction,
        dedupe_leaves=not args.no_dedupe,
        parallel_readouts=args.parallel_readouts,
        prior_temperature=args.prior_temperature,
        gumbel_k=args.gumbel_k,
        gumbel_c_visit=args.gumbel_c_visit,
        gumbel_c_scale=args.gumbel_c_scale,
        gumbel_noise_scale=args.gumbel_noise_scale,
        full_move_temperature=args.full_move_temperature,
        explore_decisions=args.explore_decisions,
        explore_from_game_start=args.explore_from_game_start,
        dirichlet_weight=args.dirichlet_weight,
        dirichlet_concentration=args.dirichlet_concentration,
        value_head=args.value_head,
        reuse_tree=args.reuse_tree,
        reuse_max_root_visits=args.reuse_max_root_visits,
        auction_only=args.auction_only,
        terminal_value=args.terminal_value,
        auction_max_decisions=args.auction_max_decisions,
        auction_prefix_decisions=args.auction_prefix_decisions,
        dedupe_maps=args.dedupe_maps,
        server_pipeline=args.server_pipeline,
        server_replicas=args.server_replicas,
        workers=args.workers,
        games_per_worker=args.games_per_worker,
        task_seconds=args.task_seconds,
        max_decisions=args.max_decisions,
        num_players=args.num_players,
        start_positions=args.start_positions,
        random_start_fraction=args.random_start_fraction,
        midgame_positions=args.midgame_positions if args.midgame_fraction > 0 or args.resume else None,
        midgame_fraction=args.midgame_fraction,
        auction_fraction=args.auction_fraction,
        auction_unlock=args.auction_unlock,
        auction_stall_decisions=args.auction_stall_decisions,
        inference_batch_size=args.inference_batch_size,
        critic_positions_per_game=args.critic_positions_per_game,
        reward=args.reward,
        window_rows=args.window_rows,
        rows_per_update=args.rows_per_update,
        min_window_rows=args.min_window_rows,
        steps_per_update=args.steps_per_update,
        minibatch=args.minibatch,
        microbatch=args.microbatch,
        lr=SearchLearningConfig.lr if args.lr is None else args.lr,
        critic_lr=SearchLearningConfig.critic_lr if args.critic_lr is None else args.critic_lr,
        price_loss_weight=args.price_loss_weight,
        score_loss_weight=args.score_loss_weight,
        kl_coef=args.kl_coef,
        kl_target=None if args.kl_target is not None and args.kl_target <= 0 else args.kl_target,
        kl_coef_min=args.kl_coef_min,
        anchor_every=args.anchor_every,
        max_updates=args.max_updates,
        snapshot_every=args.snapshot_every,
        shared=args.shared,
        value_loss_weight=args.value_loss_weight,
        value_stop_grad=args.value_stop_grad,
        value_nstep=args.value_nstep,
        value_bootstrap_weight=args.value_bootstrap_weight,
        value_lambda=args.value_lambda,
        aux_loss_weight=args.aux_loss_weight,
        surprise_weight=args.surprise_weight,
        surprise_cap=args.surprise_cap,
        target_confidence=args.target_confidence,
        target_confidence_mode=args.target_confidence_mode,
        target_confidence_prior_weight=args.target_confidence_prior_weight,
        target_confidence_signal=args.target_confidence_signal,
        target_distrust_steps=tuple(x for x in (args.target_distrust_steps or "").split(",") if x),
        save_data=args.save_data,
        save_data_shard_games=args.save_data_shard_games,
        freeze_policy=args.freeze_policy,
        policy_follow=args.policy_follow,
        value_follow=args.value_follow,
        population=args.population,
        league_share=args.league_share,
        league_seat_weights=(
            {int(k): float(v) for k, v in (x.split(":") for x in args.league_seat_weights.split(","))}
            if args.league_seat_weights
            else None
        ),
        league_readouts=args.league_readouts,
        pfsp_power=args.pfsp_power,
        league_snapshot_hours=args.league_snapshot_hours,
        league_snapshot_slots=args.league_snapshot_slots,
    )
    explicit = _resume_explicit(args, RESUME_SETTINGS)
    print(run(config, resume=args.resume, explicit=explicit))


def cmd_value_train(args):
    from dataclasses import fields

    from rl18xx.agent.alphazero.value_trainer import ValueTrainerConfig, serve

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    values = {f.name: getattr(args, f.name) for f in fields(ValueTrainerConfig) if hasattr(args, f.name)}
    if args.no_fixed:
        values["fixed_data"] = None
    print(f"value-train: {serve(ValueTrainerConfig(**values), max_cycles=args.max_cycles)} cycles run")


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


def cmd_convert_generic(args):
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from rl18xx.agent.alphazero.generic_data import convert_dataset

    meta = convert_dataset(
        sources=args.data_dir,
        out=args.output,
        title=args.title,
        workers=args.workers,
        limit=args.limit,
        validation_percentage=args.validation_percentage,
        skip_forced=args.skip_forced,
    )
    stats = {k: v for k, v in meta["stats"].items() if k != "dropped_detail"}
    print(json.dumps({"counts": meta["counts"], "stats": stats}, indent=2))


def cmd_pretrain_generic(args):
    import json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from rl18xx.agent.alphazero.generic_pretrain import MODEL_PRESETS, GenericPretrainConfig, pretrain

    model = dict(MODEL_PRESETS[args.model_preset])
    if args.model_config:
        model.update(json.loads(args.model_config))
    cfg = GenericPretrainConfig(
        data=args.data,
        out=args.out,
        model=model,
        init=args.init,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_lr_multiplier=args.value_lr_multiplier,
        epochs=args.epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        entropy_weight=args.entropy_weight,
        value_loss_weight=args.value_loss_weight,
        score_loss_weight=args.score_loss_weight,
        price_loss_weight=args.price_loss_weight,
        value_joint_epochs=None if args.value_joint_epochs < 0 else args.value_joint_epochs,
        value_refit=not args.no_value_refit,
        refit_eval_steps=args.refit_eval_steps,
        refit_patience=args.refit_patience,
        refit_max_steps=args.refit_max_steps,
        device=args.device,
        bf16=args.bf16,
        workers=args.workers,
        buffer_positions=args.buffer_positions,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        train_max_games=args.train_max_games,
        val_max_games=args.val_max_games,
        log_every=args.log_every,
        threads=args.threads,
    )
    pretrain(cfg)


def cmd_compare_generic(args):
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    from rl18xx.agent.alphazero.generic_pretrain import compare_with_legacy

    torch.set_num_threads(args.threads)
    result = compare_with_legacy(
        args.generic, args.legacy, args.data, split=args.split, batch_size=args.batch_size, device=args.device,
        max_games=args.max_games, max_positions=args.max_positions,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    for name in ("generic", "legacy"):
        m = result[name]
        fams = " ".join(f"{k}={v['top1']:.3f}" for k, v in sorted(m["top1_by_family"].items()))
        print(
            f"{name}: positions {m['positions']} top1 {m['top1']:.4f} top5 {m['top5']:.4f} "
            f"value CE {m['value_ce']:.4f} (equal odds {m['value_baseline']:.4f}) "
            f"winner {m['winner_acc']:.3f}\n  {fams}"
        )
    print(f"rotation mismatches: {result['rotation_mismatch']}")


def cmd_policy_gradient_generic(args):
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    from rl18xx.agent.alphazero.generic_pg import RESUME_SETTINGS, GenericPGConfig, run
    from rl18xx.agent.alphazero.policy_gradient import default_learner_seats

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.resume and not args.policy:
        raise SystemExit("--policy (a generic checkpoint) is required for a new run")
    config = GenericPGConfig(
        policy_checkpoint=args.policy or "",
        value_checkpoint=args.value,
        title=args.title,
        out_dir=args.out_dir,
        run_name=args.run_name,
        num_players=args.num_players,
        player_mix=_by_players(args.player_mix, float, "--player-mix"),
        max_decisions_by_players=_by_players(args.max_decisions_by_players, int, "--max-decisions-by-players"),
        auction_stall_by_players=_by_players(args.auction_stall_by_players, int, "--auction-stall-by-players"),
        workers=args.workers,
        games_per_task=args.games_per_task,
        learner_seats=args.learner_seats if args.learner_seats is not None else default_learner_seats(args.num_players),
        positions_per_seat=args.positions_per_seat,
        sl_opponent_fraction=args.sl_opponent_fraction,
        opponent_checkpoint=args.opponent,
        anchor_every=args.anchor_every,
        population=args.population,
        opponent_slots=args.opponent_slots,
        learner_seat_weights=(
            {int(k): float(v) for k, v in (part.split(":") for part in args.learner_seat_weights.split(","))}
            if args.learner_seat_weights
            else None
        ),
        pfsp_power=args.pfsp_power,
        slot_refresh_every=args.slot_refresh_every,
        league_snapshot_every=args.league_snapshot_every,
        league_max_snapshots=args.league_max_snapshots,
        learner_temperature=args.learner_temperature,
        opponent_temperature=args.opponent_temperature,
        opponent_price_eps=args.opponent_price_eps,
        max_decisions=args.max_decisions,
        start_positions=args.start_positions,
        random_start_fraction=args.random_start_fraction,
        midgame_positions=args.midgame_positions,
        midgame_fraction=args.midgame_fraction,
        auction_fraction=args.auction_fraction,
        auction_unlock=args.auction_unlock,
        auction_stall_decisions=args.auction_stall_decisions,
        inference_batch_size=args.inference_batch_size,
        server_replicas=args.server_replicas,
        server_pipeline=args.server_pipeline,
        server_threads=args.server_threads,
        server_cpu=args.server_cpu,
        compress_requests=not args.no_compress_requests,
        rows_per_update=args.rows_per_update,
        minibatch=args.minibatch,
        microbatch=args.microbatch,
        ppo_epochs=args.ppo_epochs,
        clip=args.clip,
        lr=args.lr,
        critic_lr=args.critic_lr,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        kl_coef_min=args.kl_coef_min,
        entropy_coef=args.entropy_coef,
        max_updates=args.max_updates,
        snapshot_every=args.snapshot_every,
        pool_refresh_every=args.pool_refresh_every,
        save_game_every=args.save_game_every,
        reward=args.reward,
        device=args.device,
        threads=args.threads,
    )
    explicit = _resume_explicit(args, RESUME_SETTINGS)
    print(run(config, resume=args.resume, explicit=explicit))


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
    from rl18xx.client.replay_game_from_log_file import replay_game_from_log_file, replay_on_local_server

    if args.local_server:
        replay_on_local_server(args.log_file, args.local_server)
    else:
        replay_game_from_log_file(args.log_file, print_log=args.log)


def cmd_advisor(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from rl18xx.agent.advisor.app import run

    run(args)


def explicit_options(argv=None) -> set:
    """The dests of the options given on the command line (``argv``, default ``sys.argv[1:]``), whatever
    their values -- a resume changes only these (``policy_gradient.resume_overrides``). Parsed again with
    every option's default suppressed, so abbreviations and ``--opt=value`` count as argparse reads them."""
    quiet = build_parser()
    parsers = [quiet]
    for action in quiet._actions:
        if isinstance(action, argparse._SubParsersAction):
            parsers.extend(action.choices.values())
    for parser in parsers:
        for action in parser._actions:
            if action.option_strings:
                action.default = argparse.SUPPRESS
    return set(vars(quiet.parse_args(argv))) - {"command"}


def _by_players(spec, kind, flag) -> dict | None:
    """``"2:1500,6:1800"`` -> ``{2: 1500, 6: 1800}`` (player count -> value); None or "" -> None."""
    if not spec:
        return None
    try:
        out = {int(k): kind(v) for k, v in (part.split(":") for part in spec.split(","))}
    except ValueError:
        raise SystemExit(f"{flag}: <players>:<value>,... e.g. 2:1,4:4,6:1, got {spec!r}") from None
    if any(not 2 <= k <= 6 for k in out):
        raise SystemExit(f"{flag}: player counts are 2-6, got {spec!r}")
    return out


def _player_count_defaults(args) -> None:
    """``--start-positions`` / ``--midgame-positions`` left out: the ``--num-players`` count's files
    (``human_games/{start,midgame}_positions_1830_<N>p_v2.jsonl``; 4 players: the files used so far)."""
    from rl18xx.agent.alphazero.start_positions import default_midgame_positions, default_start_positions

    if args.start_positions is None:
        args.start_positions = default_start_positions(args.num_players)
    if args.midgame_positions is None:
        args.midgame_positions = default_midgame_positions(args.num_players)


def _resume_explicit(args, settings) -> set | None:
    """A resume's explicitly passed options (None outside the command line), warning about the ones a
    resume ignores (the run keeps its own value)."""
    explicit = getattr(args, "explicit", None)
    if args.resume and explicit is not None:
        ignored = sorted(explicit - set(settings) - {"resume"})
        if ignored:
            flags = ", ".join("--" + k.replace("_", "-") for k in ignored)
            logging.getLogger(__name__).warning(f"--resume keeps the run's own settings; ignoring {flags}")
    return explicit


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
    p.add_argument(
        "--num-players", type=int, default=4,
        help="Players per game (default 4; 2: 1830's two-player game -- the learner holds 1 of 2 seats, the start "
        "files default to the 2-player ones; placement points 1/0)",
    )
    p.add_argument(
        "--player-mix", type=str, default=None,
        help="Mixed player counts, <players>:<weight>,... e.g. 2:1,3:2,4:4,5:2,6:1 (each worker task plays one count; "
        "the games come in these shares). The start files default to the per-count ones ({n} in a path is the count); "
        "the learner's league seats and placement points are the count's (policy_gradient's *_BY_PLAYERS tables)",
    )
    p.add_argument(
        "--max-decisions-by-players", type=str, default=None,
        help="Per-count --max-decisions, e.g. 2:1500,6:1500 (other counts: --max-decisions)",
    )
    p.add_argument(
        "--auction-stall-by-players", type=str, default=None,
        help="Per-count --auction-stall-decisions, e.g. 6:200 (other counts: --auction-stall-decisions)",
    )
    p.add_argument("--learner-seats", type=int, default=None, help="Learner seats per game (default: half the table)")
    p.add_argument("--positions-per-seat", type=int, default=32, help="Learner decisions kept per seat per game")
    p.add_argument("--sl-opponent-fraction", type=float, default=0.5, help="Games against the fixed opponent")
    p.add_argument(
        "--opponent", type=str, default=None,
        help="Fixed opponent <session>/<num> or .pth (default: the starting policy)",
    )
    p.add_argument(
        "--midgame-fraction", type=float, default=0.0,
        help="Share of games starting at a later Stock Round of a recorded human game (with --resume: from now on)",
    )
    p.add_argument(
        "--midgame-positions", type=str, default=None,
        help="default: human_games/midgame_positions_1830_<N>p_v2.jsonl for --num-players N",
    )
    p.add_argument(
        "--auction-fraction", type=float, default=0.0,
        help="Share of games starting at the waterfall auction, the learner's bids trained too (with --resume: from now on)",
    )
    p.add_argument(
        "--no-auction-unlock", dest="auction_unlock", action="store_false",
        help="Auction starts play the real rules: no auction_unlock variant (after the SV is sold an all-pass round "
        "pays the privates out and the auction goes on)",
    )
    p.add_argument(
        "--auction-stall-decisions", type=int, default=0,
        help="End a game still in the auction after this many decisions, scored on net worth as it stands (0: off)",
    )
    p.add_argument(
        "--population", type=str, default=None,
        help="League mode: JSON list of checkpoint paths (or {\"path\", \"min_share\"} objects: drawn at least that often); opponents are drawn per seat from it (prioritized by how often they beat the learner) instead of --opponent and the pool",
    )
    p.add_argument("--opponent-slots", type=int, default=3, help="League mode: opponent servers (members at once) per player count; a --player-mix run has this many for each count, drawn by PFSP on the learner's results at that count")
    p.add_argument(
        "--learner-seat-weights", type=str, default=None,
        help="League mode: seats the learner holds per game, e.g. 1:0.4,2:0.4,3:0.2 (the default; 2 players: 1:1)",
    )
    p.add_argument("--pfsp-power", type=float, default=2.0, help="League mode: weight (1 - win rate) ** power")
    p.add_argument("--slot-refresh-every", type=int, default=2, help="League mode: reload one slot (per player count) every N updates")
    p.add_argument(
        "--league-snapshot-every", type=int, default=50,
        help="League mode: add the learner's snapshot to the league every N updates (a multiple of the snapshot interval)",
    )
    p.add_argument("--league-max-snapshots", type=int, default=4, help="League mode: own snapshots kept in the league")
    p.add_argument(
        "--reward", type=str, default="win",
        help="The learner's goal: win (default), or an exploiter's -- placement (1st 1, 2nd 0.5, 3rd 0.2; or placement:1,0.3,0.1,0), "
        "bankrupt (+0.5 when another player goes bankrupt), president:<CORP> (+0.5 for its presidency at the end), "
        "rusher (+0.15 per phase change its purchases cause, up to 4), rust (+0.1 per opponent train its purchases rust, up to 6)",
    )
    p.add_argument(
        "--anchor-every", type=int, default=0,
        help="Move the KL anchor to the learner every N updates (0: anchor at the starting policy)",
    )
    p.add_argument("--rows-per-update", type=int, default=65536)
    p.add_argument("--minibatch", type=int, default=256, help="Rows per optimizer step")
    p.add_argument("--microbatch", type=int, default=256, help="Rows per forward/backward (accumulated to --minibatch)")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=3e-5)
    p.add_argument("--kl-coef", type=float, default=0.02, help="KL(learner || supervised) penalty")
    p.add_argument("--kl-target", type=float, default=None, help="Adapt --kl-coef toward this KL per decision")
    p.add_argument("--kl-coef-min", type=float, default=0.005, help="Floor for the adapted --kl-coef")
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument(
        "--learner-temperature", type=float, default=1.0,
        help="The learner plays and is trained as softmax(logits / T) (KL anchor at the same T)",
    )
    p.add_argument("--opponent-temperature", type=float, default=1.0, help="Sampling temperature of opponents")
    p.add_argument(
        "--opponent-price-eps", type=float, default=0.0,
        help="Uniform mixing into opponents' price draws (the learner's is always 0): random prices, a test option only",
    )
    p.add_argument("--gae-lambda", type=float, default=1.0, help="1 = Monte Carlo returns")
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--pool-refresh-every", type=int, default=5)
    p.add_argument("--max-decisions", type=int, default=1000)
    p.add_argument(
        "--start-positions", type=str, default=None,
        help="default: human_games/start_positions_1830_<N>p_v2.jsonl for --num-players N",
    )
    p.add_argument("--random-start-fraction", type=float, default=0.2)
    p.add_argument(
        "--save-game-every", type=int, default=0,
        help="Keep the action log of about one game in N in <run>/games/ for the dashboard's game viewer (0 = off)",
    )
    p.add_argument(
        "--shared", action="store_true",
        help="AlphaGo-style split: --policy is a shared value-encoder network whose own value heads give the "
        "baseline (no --value critic); PPO trains its policy path only",
    )
    p.add_argument(
        "--value-follow", type=str, default=None,
        help="--shared: a search-learning run (--freeze-policy) training the value side; before each update the "
        "learner takes its latest value side",
    )
    p.add_argument(
        "--value-pg-weight", type=float, default=0.0,
        help="--shared: also train the value side on these games' rows (trunk detached) with this weight, at --critic-lr",
    )

    # search-learning (expert iteration: train the policy toward search targets)
    from rl18xx.agent.alphazero.search_learning import SearchLearningConfig as _SL

    p = sub.add_parser(
        "search-learning",
        help="Train the policy toward its search's targets and the critic on the games (expert iteration)",
    )
    p.add_argument("--policy", type=str, default=None, help="Starting policy <session>/<num> or .pth")
    p.add_argument("--model-dir", type=str, default="model_checkpoints_pg")
    p.add_argument("--value", type=str, default=None, help="The search's value network (critic) <session>/<num> or .pth")
    p.add_argument("--value-model-dir", type=str, default="model_checkpoints_pg")
    p.add_argument("--out-dir", type=str, default=_SL.out_dir)
    p.add_argument(
        "--auction-only", action="store_true",
        help="Episodes are the waterfall auction alone (real rules), scored by --terminal-value's win "
        "estimate of the first Stock Round position (docs/search_learning_design.md 'Auction learning')",
    )
    p.add_argument("--terminal-value", type=str, default=None, help="--auction-only: the frozen scorer (.pth)")
    p.add_argument(
        "--auction-max-decisions", type=int, default=_SL.auction_max_decisions,
        help="--auction-only: an auction still running after this many decisions is dropped",
    )
    p.add_argument(
        "--auction-prefix-decisions", type=int, default=_SL.auction_prefix_decisions,
        help="--auction-only: each episode opens with 0..N decisions sampled from the policy (varied openings)",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue from its last snapshot")
    p.add_argument(
        "--learner-temperature", type=float, default=None,
        help="The policy is softmax(logits / T), trained and searched so (default: the --policy run's learner_temperature)",
    )
    p.add_argument("--target", choices=("visits", "gumbel", "cq", "rpo"), default=_SL.target,
                   help="Policy target and root selection: the Gumbel root's completed-Q policy (default; every "
                   "search, full and fast, uses the Gumbel root), or on the PUCT search: its visit distribution, "
                   "the completed-Q improved policy softmax(logits + sigma(completed q)) (cq; sigma from "
                   "--gumbel-c-visit / --gumbel-c-scale), or Grill et al. 2020's regularized policy (rpo)")
    p.add_argument("--cq-rescale", action="store_true",
                   help="--target cq: min-max rescale the completed q's to [0, 1] first (mctx's rescale_values)")
    p.add_argument("--rpo-c", type=float, default=_SL.rpo_c,
                   help="--target rpo: lambda_N's c (0: the tree's own effective c_puct at the root)")
    p.add_argument("--rpo-unvisited", choices=("vmix", "parent", "min"), default=_SL.rpo_unvisited,
                   help="--target rpo: q of unvisited moves (the root's v_mix, its mean value, the lowest visited q)")
    p.add_argument("--rpo-qnorm", action="store_true",
                   help="--target rpo: q min-max normalized over the root's value and visited children first")
    p.add_argument("--q-normalize", action="store_true",
                   help="PUCT selection on min-max normalized Q (MuZero); size --c-puct for a 0-1 range")
    p.add_argument("--no-move-widening", action="store_true",
                   help="Every legal move competes at nodes with > 20 of them (AlphaZero), not the top-k by prior")
    p.add_argument("--fast-row-weight", type=float, default=_SL.fast_row_weight,
                   help="Fast searches (--n-fast) also become rows with this weight in the policy loss (0: no rows)")
    p.add_argument("--no-fast-row-value", action="store_true",
                   help="--fast-row-weight: fast rows train the policy only, not the value")
    p.add_argument("--visit-target", choices=("pruned", "raw"), default=_SL.visit_target,
                   help="--target visits: KataGo's policy-target pruning of exploration visits, or the raw visits")
    p.add_argument("--target-temperature", type=float, default=_SL.target_temperature,
                   help="Sharpen the search target to target ** (1 / t); 0: one-hot on the move the search played")
    p.add_argument("--p-full", type=float, default=_SL.p_full, help="Share of decisions searched (and trained on)")
    p.add_argument("--n-full", type=int, default=_SL.n_full, help="Readouts of a searched decision")
    p.add_argument("--n-fast", type=int, default=_SL.n_fast,
                   help="Readouts of the other learner decisions (played, not trained on); 0: the policy plays them")
    p.add_argument("--c-puct", type=float, default=_SL.c_puct_init, help="c_puct_init in every round")
    p.add_argument("--no-mean-q", action="store_true", help="minigo's W / (1 + N) instead of mean Q")
    p.add_argument("--fpu-reduction", type=float, default=_SL.fpu_reduction)
    p.add_argument("--no-dedupe", action="store_true", help="Back up a batch's duplicate leaves as visits")
    p.add_argument("--parallel-readouts", type=int, default=_SL.parallel_readouts, help="Leaves per tree per batch")
    p.add_argument("--prior-temperature", type=float, default=None, help="Search priors' temperature (default: T)")
    p.add_argument("--gumbel-k", type=int, default=_SL.gumbel_k, help="--target gumbel: root moves kept")
    p.add_argument("--gumbel-c-visit", type=float, default=_SL.gumbel_c_visit)
    p.add_argument("--gumbel-c-scale", type=float, default=_SL.gumbel_c_scale)
    p.add_argument("--gumbel-noise-scale", type=float, default=_SL.gumbel_noise_scale)
    p.add_argument(
        "--full-move-temperature", type=float, default=_SL.full_move_temperature,
        help="--target visits: 0 plays the most-visited move, t > 0 samples it with weight N ** (1 / t)",
    )
    p.add_argument(
        "--explore-decisions", type=int, default=_SL.explore_decisions,
        help="--full-move-temperature only for each game's first N decisions (0: all; AlphaZero's first-30-moves "
        "temperature). Can change on --resume",
    )
    p.add_argument(
        "--explore-from-game-start", action="store_true",
        help="--explore-decisions counts from the game's first Stock Round (a mid-game start's recorded decisions "
        "count), not from the start position",
    )
    p.add_argument(
        "--dirichlet-weight", type=float, default=_SL.dirichlet_weight,
        help="AlphaZero's root noise on the full searches (0.25 in AlphaZero; 0: off; not with --target gumbel). "
        "Can change on --resume",
    )
    p.add_argument(
        "--dirichlet-concentration", type=float, default=_SL.dirichlet_concentration,
        help="The noise's alpha is this / the legal moves (10: ~AlphaZero's alpha for its games). "
        "Can change on --resume",
    )
    p.add_argument("--value-head", type=str, default=_SL.value_head,
                   help="The value head the search reads its leaf values from: win, score or blend=<w> "
                   "((1 - w) * win + w * score, KataGo's score utility); the critic's, or the learner's with --shared")
    p.add_argument(
        "--reuse-tree", choices=("off", "total", "additional"), default=_SL.reuse_tree,
        help="Keep each game's tree across decisions: 'total' counts the inherited root visits toward the "
        "budget (same visits per decision, fewer evaluations), 'additional' searches the budget on top of them",
    )
    p.add_argument("--reuse-max-root-visits", type=int, default=None,
                   help="--reuse-tree additional: rebuild a tree whose root passed this many visits (default 2 * n_full)")
    p.add_argument("--dedupe-maps", action="store_true",
                   help="Send each distinct hex map of a request once; the map encoder runs once per map")
    p.add_argument("--server-pipeline", action="store_true",
                   help="The inference server receives the next batch on a thread while the current one runs "
                   "(measured slower than --server-replicas 2: GIL contention)")
    p.add_argument("--server-replicas", type=int, default=_SL.server_replicas,
                   help="Inference server processes (one forward runs while another receives and replies)")
    p.add_argument("--workers", type=int, default=_SL.workers)
    p.add_argument("--games-per-worker", type=int, default=_SL.games_per_worker, help="Games each worker plays at once")
    p.add_argument("--task-seconds", type=float, default=_SL.task_seconds, help="A worker reports its finished games this often")
    p.add_argument("--max-decisions", type=int, default=_SL.max_decisions)
    p.add_argument(
        "--num-players", type=int, default=_SL.num_players,
        help="Players per game (default 4; 2: 1830's two-player game -- the start files default to the 2-player "
        "ones, a league member takes 1 of 2 seats, placement points 1/0). A resume keeps the run's",
    )
    p.add_argument(
        "--start-positions", type=str, default=None,
        help="default: human_games/start_positions_1830_<N>p_v2.jsonl for --num-players N",
    )
    p.add_argument("--random-start-fraction", type=float, default=_SL.random_start_fraction)
    p.add_argument("--midgame-fraction", type=float, default=_SL.midgame_fraction)
    p.add_argument(
        "--midgame-positions", type=str, default=None,
        help="default: human_games/midgame_positions_1830_<N>p_v2.jsonl for --num-players N",
    )
    p.add_argument("--auction-fraction", type=float, default=_SL.auction_fraction)
    p.add_argument(
        "--no-auction-unlock", dest="auction_unlock", action="store_false",
        help="Auction starts play the real rules: an all-pass round after the SV is sold pays the privates out "
        "and the auction goes on (default: the auction_unlock variant)",
    )
    p.add_argument(
        "--auction-stall-decisions", type=int, default=_SL.auction_stall_decisions,
        help="End a game still in the auction after this many decisions, scored on net worth (0: never)",
    )
    p.add_argument("--inference-batch-size", type=int, default=_SL.inference_batch_size)
    p.add_argument("--critic-positions-per-game", type=int, default=_SL.critic_positions_per_game,
                   help="Rows of each game the critic trains on (once each; 0 = all)")
    p.add_argument("--reward", type=str, default=_SL.reward,
                   help="The critic's score-head target: placement (gen4's critic; or placement:1,0.5,0.2,0) or win")
    p.add_argument("--window-rows", type=int, default=_SL.window_rows, help="Replay window: the latest rows")
    p.add_argument("--rows-per-update", type=int, default=_SL.rows_per_update, help="New rows between updates")
    p.add_argument("--min-window-rows", type=int, default=_SL.min_window_rows, help="No update before the window has these")
    p.add_argument("--steps-per-update", type=int, default=_SL.steps_per_update)
    p.add_argument("--minibatch", type=int, default=_SL.minibatch, help="Rows per optimizer step")
    p.add_argument("--microbatch", type=int, default=_SL.microbatch, help="Rows per forward/backward")
    p.add_argument("--lr", type=float, default=None, help=f"Trunk and policy (default {_SL.lr}; with --resume: from now on)")
    p.add_argument("--critic-lr", type=float, default=None,
                   help=f"Value heads (default {_SL.critic_lr}; with --resume: from now on)")
    p.add_argument("--price-loss-weight", type=float, default=_SL.price_loss_weight)
    p.add_argument("--score-loss-weight", type=float, default=_SL.score_loss_weight)
    p.add_argument("--kl-coef", type=float, default=_SL.kl_coef, help="KL(learner || anchor) at T")
    p.add_argument("--kl-target", type=float, default=_SL.kl_target, help="Adapt --kl-coef toward this KL (<= 0: fixed)")
    p.add_argument("--kl-coef-min", type=float, default=_SL.kl_coef_min)
    p.add_argument("--anchor-every", type=int, default=_SL.anchor_every,
                   help="Move the KL anchor to the learner every N updates (0: the starting policy)")
    p.add_argument("--max-updates", type=int, default=_SL.max_updates)
    p.add_argument("--snapshot-every", type=int, default=_SL.snapshot_every)
    p.add_argument("--shared", action="store_true",
                   help="One network: the search reads the learner's own value heads (no --value critic) and the "
                   "value loss trains the learner's trunk with the policy; its value heads train at --critic-lr")
    p.add_argument("--value-loss-weight", type=float, default=_SL.value_loss_weight,
                   help="--shared: the win CE's weight in the learner's loss (the trunk's share of the value signal)")
    p.add_argument("--value-stop-grad", action="store_true",
                   help="--shared: the value heads train on detached trunk features (no value gradient in the trunk)")
    p.add_argument("--value-nstep", type=int, default=_SL.value_nstep,
                   help="Win-head target: 0 the outcome; n > 0 mixes in the search's root value n searched decisions "
                   "later (MuZero's n-step target; the outcome where the game ends sooner)")
    p.add_argument("--value-bootstrap-weight", type=float, default=_SL.value_bootstrap_weight,
                   help="--value-nstep / --value-lambda: the bootstrap's weight in the target (the outcome's is 1 - w)")
    p.add_argument("--value-lambda", type=float, default=_SL.value_lambda,
                   help="Bootstrap from the lambda-return over the game's later search values instead (TD(lambda))")
    p.add_argument("--aux-loss-weight", type=float, default=_SL.aux_loss_weight,
                   help="Auxiliary outcome heads (final net-worth and cash shares, corporations floated); 0 = off")
    p.add_argument("--surprise-weight", type=float, default=_SL.surprise_weight,
                   help="KataGo's policy surprise weighting: a row's target CE weighs (1 - a) + a * KL / mean KL; 0 = off")
    p.add_argument("--surprise-cap", type=float, default=_SL.surprise_cap, help="Largest surprise weight")
    p.add_argument("--target-confidence", type=float, default=_SL.target_confidence,
                   help="Keep a target that departs from the policy's top move only where z = (q_t - q_p) / se >= this "
                   "(search_targets.departure_confidence); weaker departures shrink toward the policy. Default: off")
    p.add_argument("--target-confidence-mode", choices=("prior", "soft"), default=_SL.target_confidence_mode,
                   help="Below the threshold: prior = the policy's distribution; "
                   "soft = keep z / threshold of the departure")
    p.add_argument("--target-confidence-prior-weight", type=float, default=_SL.target_confidence_prior_weight,
                   help="Pseudo-visits of the root's pooled value variance in each move's variance estimate")
    p.add_argument("--target-confidence-signal", choices=("z", "share"), default=_SL.target_confidence_signal,
                   help="z: (q_t - q_p) / se; share: N_t / (N_t + N_p), the departure's share of the two moves' visits")
    p.add_argument("--target-distrust-steps", default="",
                   help="Comma-separated step types (stock, track, token, route, dividend, train, other) whose "
                   "departures from the policy's top move always shrink toward the policy (e.g. track)")
    p.add_argument("--freeze-policy", action="store_true",
                   help="--shared: train only the value side (value encoder + heads, trunk detached) on the search "
                   "games -- the AlphaGo-style split's value learner. --value-follow: train nothing (generation only: "
                   "games, --save-data, the history)")
    p.add_argument("--policy-follow", type=str, default=None,
                   help="--freeze-policy: a policy-gradient run (--shared) training the policy path; before each "
                   "update the learner takes its latest policy path")
    p.add_argument("--save-data", type=str, default=_SL.save_data,
                   help="Keep every finished game in this directory: action log, outcome, each search's targets and "
                   "root value (search_data: xz shards; rows are rebuilt by replaying the log). With --resume: the "
                   "run's own directory unless given. Default: off")
    p.add_argument("--save-data-shard-games", type=int, default=_SL.save_data_shard_games,
                   help="Games per --save-data shard (the open shard's games are journaled until it is written)")
    p.add_argument("--value-follow", type=str, default=_SL.value_follow,
                   help="A value-train run directory: the search's value is its current promoted value (promoted.json), "
                   "reloaded when it promotes a new one; no critic trains here (no --value needed)")
    p.add_argument("--population", type=str, default=_SL.population,
                   help="League seats: a JSON file of members ({\"members\": [{\"name\", \"spec\", \"mode\": search|policy, "
                   "\"readouts\", \"prior_temperature\", \"temperature\", \"min_share\"}]}), drawn by PFSP")
    p.add_argument("--league-share", type=float, default=_SL.league_share,
                   help="--population: the share of games in which one member takes 1-2 seats")
    p.add_argument("--league-seat-weights", type=str, default=None,
                   help="--population: seats:weight pairs for a member's seat count "
                   "(default 1:0.5,2:0.5; 2 players: 1:1)")
    p.add_argument("--league-readouts", type=int, default=_SL.league_readouts,
                   help="--population: a searching member's readouts per decision (the fast budget)")
    p.add_argument("--pfsp-power", type=float, default=_SL.pfsp_power,
                   help="--population: PFSP weight (1 - p) ** power, p the main system's pairwise rate against a member")
    p.add_argument("--league-snapshot-hours", type=float, default=_SL.league_snapshot_hours,
                   help="--population: the main system joins the league as a snapshot this often (wall-clock hours)")
    p.add_argument("--league-snapshot-slots", type=int, default=_SL.league_snapshot_slots,
                   help="--population: snapshot members kept (each an inference server; 0: no snapshots)")

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

    # convert-generic (title-generic supervised records for the generic model; docs/rl_model.md)
    p = sub.add_parser(
        "convert-generic",
        help="Cleaned human games -> title-generic supervised records (legal indices, labels, prices, outcomes)",
    )
    p.add_argument(
        "--data-dir", type=str, nargs="+", default=["human_games/1830_clean_all"],
        help="Cleaned game JSON directories or files (default: human_games/1830_clean_all, lmdb_v6k's source)",
    )
    p.add_argument("--output", type=str, required=True, help="New dataset directory")
    p.add_argument("--title", type=str, default="1830", help="Title of the games (needs an importer; default 1830)")
    p.add_argument("--workers", type=int, default=4, help="Conversion processes (default 4)")
    p.add_argument("--limit", type=int, default=None, help="Convert only the first N games by id")
    p.add_argument(
        "--validation-percentage", type=float, default=0.05,
        help="pretraining.in_validation_split's per-game share (default 0.05, as lmdb_v6k / SL ckpt 10)",
    )
    p.add_argument("--skip-forced", action="store_true", help="Drop positions with one legal index (off: kept)")

    # pretrain-generic (two-stage supervised pretraining of the generic model)
    p = sub.add_parser("pretrain-generic", help="Supervised two-stage pretraining of the title-generic model")
    p.add_argument("--data", type=str, required=True, help="Dataset directory from convert-generic")
    p.add_argument("--out", type=str, required=True, help="Run directory (checkpoints, tb/, history.jsonl)")
    p.add_argument("--model-preset", choices=["default", "small"], default="default")
    p.add_argument("--model-config", type=str, default=None, help="JSON GenericModelConfig overrides")
    p.add_argument("--init", type=str, default=None, help="Warm start from a generic checkpoint")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-lr-multiplier", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=4, help="Evaluations without improvement before stopping")
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--value-loss-weight", type=float, default=0.1)
    p.add_argument("--score-loss-weight", type=float, default=0.1)
    p.add_argument("--price-loss-weight", type=float, default=0.1)
    p.add_argument("--value-joint-epochs", type=int, default=-1, help="-1: value gradients reach the trunk throughout")
    p.add_argument("--no-value-refit", action="store_true", help="Skip stage 2 (value heads refit)")
    p.add_argument("--refit-eval-steps", type=int, default=250)
    p.add_argument("--refit-patience", type=int, default=4)
    p.add_argument("--refit-max-steps", type=int, default=20000)
    p.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:N")
    p.add_argument("--bf16", choices=["auto", "on", "off"], default="auto", help="auto: bf16 autocast on CUDA only")
    p.add_argument("--workers", type=int, default=4, help="Data-loader workers (replay + encode)")
    p.add_argument("--buffer-positions", type=int, default=32768, help="Per-worker shuffle buffer (positions)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="Cap on stage-1 steps (smoke runs)")
    p.add_argument("--eval-every", type=int, default=None, help="Validate every N steps (default: each epoch)")
    p.add_argument("--train-max-games", type=int, default=None)
    p.add_argument("--val-max-games", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=4, help="torch threads of the training process")

    # compare-generic (generic vs legacy 1830 checkpoint on the same positions)
    p = sub.add_parser(
        "compare-generic", help="Per-family top-1 and value CE of a generic and a legacy checkpoint on one split"
    )
    p.add_argument("--generic", type=str, required=True, help="Generic checkpoint (e.g. <run>/final.pth)")
    p.add_argument("--legacy", type=str, required=True, help="Legacy checkpoint (e.g. SL ckpt 10's 10.pth)")
    p.add_argument("--data", type=str, required=True, help="Dataset directory from convert-generic")
    p.add_argument("--split", choices=["validation", "train"], default="validation")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-games", type=int, default=None)
    p.add_argument("--max-positions", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=str, default=None, help="Write the full result JSON here")

    # policy-gradient-generic (the policy-gradient stage for the title-generic model; docs/rl_model.md)
    p = sub.add_parser(
        "policy-gradient-generic",
        help="Policy-only policy gradient of a title-generic checkpoint against a league (mixed player counts, PFSP)",
    )
    p.add_argument("--policy", type=str, default=None, help="Starting generic checkpoint (e.g. pretrain-generic's final.pth)")
    p.add_argument("--value", type=str, default=None, help="Critic's starting generic checkpoint (default: --policy)")
    p.add_argument("--title", type=str, default="1830", help="Title (needs an RL layout; default 1830)")
    p.add_argument("--out-dir", type=str, default="model_checkpoints_generic_pg")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Run directory to continue from its last snapshot")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--games-per-task", type=int, default=32)
    p.add_argument("--num-players", type=int, default=4)
    p.add_argument("--player-mix", type=str, default=None, help="<players>:<weight>,... (as policy-gradient)")
    p.add_argument("--max-decisions-by-players", type=str, default=None)
    p.add_argument("--auction-stall-by-players", type=str, default=None)
    p.add_argument("--learner-seats", type=int, default=None, help="Outside league mode (default: half the table)")
    p.add_argument("--positions-per-seat", type=int, default=32)
    p.add_argument("--sl-opponent-fraction", type=float, default=0.5)
    p.add_argument("--opponent", type=str, default=None, help="Fixed opponent (a generic checkpoint; default --policy)")
    p.add_argument("--anchor-every", type=int, default=0)
    p.add_argument("--population", type=str, default=None, help="League file of generic checkpoints (as policy-gradient)")
    p.add_argument("--opponent-slots", type=int, default=3)
    p.add_argument("--learner-seat-weights", type=str, default=None)
    p.add_argument("--pfsp-power", type=float, default=2.0)
    p.add_argument("--slot-refresh-every", type=int, default=2)
    p.add_argument("--league-snapshot-every", type=int, default=50)
    p.add_argument("--league-max-snapshots", type=int, default=4)
    p.add_argument("--learner-temperature", type=float, default=1.0)
    p.add_argument("--opponent-temperature", type=float, default=1.0)
    p.add_argument("--opponent-price-eps", type=float, default=0.0)
    p.add_argument("--max-decisions", type=int, default=1000)
    p.add_argument(
        "--start-positions", type=str, default=None,
        help="SR1 start file template ({n}: the player count; default: the title's, titles.TitleSpec)",
    )
    p.add_argument("--random-start-fraction", type=float, default=0.2)
    p.add_argument("--midgame-positions", type=str, default=None, help="Mid-game start template (default: the title's)")
    p.add_argument("--midgame-fraction", type=float, default=0.0)
    p.add_argument("--auction-fraction", type=float, default=0.0)
    p.add_argument("--no-auction-unlock", dest="auction_unlock", action="store_false",
                   help="1830 auction starts under the real rules")
    p.add_argument("--auction-stall-decisions", type=int, default=0)
    p.add_argument("--inference-batch-size", type=int, default=512, help="Positions per server batch")
    p.add_argument("--server-replicas", type=int, default=1, help="Learner server processes")
    p.add_argument("--server-pipeline", action="store_true", help="Servers merge the next batch on a collector thread")
    p.add_argument("--server-threads", type=int, default=None, help="Torch threads per server process")
    p.add_argument("--server-cpu", action="store_true", help="Inference servers on the CPU (smoke runs)")
    p.add_argument("--no-compress-requests", action="store_true", help="Send requests uncompressed (default LZ4)")
    p.add_argument("--rows-per-update", type=int, default=65536)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--microbatch", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--critic-lr", type=float, default=3e-5)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--kl-target", type=float, default=None)
    p.add_argument("--kl-coef-min", type=float, default=0.005)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--pool-refresh-every", type=int, default=5)
    p.add_argument("--save-game-every", type=int, default=0)
    p.add_argument("--reward", type=str, default="win", help="win or placement[:<1st>,<2nd>,...]")
    p.add_argument("--device", type=str, default=None, help="Trainer device (default: cuda if available)")
    p.add_argument("--threads", type=int, default=None, help="Torch threads of the trainer")

    # refit-price-head (fit the price head on a frozen checkpoint)
    p = sub.add_parser("refit-price-head", help="Fit a fresh price head on the current-best checkpoint's frozen trunk")
    p.add_argument("--model-dir", type=str, default="model_checkpoints", help="Model checkpoint directory")
    p.add_argument(
        "--data-dir", type=str, default="human_games/lmdb_v6k",
        help="Converted human-game LMDB root with training/ and validation/ (default: human_games/lmdb_v6k)",
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
    p.add_argument(
        "--browser", action="store_true",
        help="Mirror the game onto a self-hosted 18xx server at localhost:9292 (GameSync), not the public site",
    )

    # dashboard
    p = sub.add_parser("dashboard", help="Start the training dashboard")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    # replay
    p = sub.add_parser("replay", help="Check a saved game in the Python engine and open it in the game viewer")
    p.add_argument(
        "log_file", type=str,
        help="A saved game (<collection>/games/<name>.json), an 18xx.games-style export, or a self-play log",
    )
    p.add_argument("--log", action="store_true", help="Also print the game log")
    p.add_argument(
        "--local-server", type=str, default=None,
        help="Instead, re-create the game on this self-hosted 18xx server (e.g. http://localhost:9292)",
    )

    # advisor
    from rl18xx.agent.advisor import DEFAULT_AUCTION_POLICY, DEFAULT_POLICY, DEFAULT_SEARCH_VALUE, DEFAULT_VALUE

    p = sub.add_parser(
        "advisor",
        help="Serve the model advisor's backend for the browser extension in extension/",
    )
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Address to serve on (default 127.0.0.1; 0.0.0.0 for a browser on another machine of the LAN)",
    )
    p.add_argument("--port", type=int, default=5002, help="Port (default: 5002)")
    p.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="Policy checkpoint for recommended moves")
    p.add_argument(
        "--auction-policy", type=str, default=DEFAULT_AUCTION_POLICY,
        help="Policy checkpoint for the private auction (policy-gradient runs never trained on it)",
    )
    p.add_argument("--value", type=str, default=DEFAULT_VALUE, help="Value checkpoint for win estimates")
    p.add_argument(
        "--search-value", type=str, default=DEFAULT_SEARCH_VALUE,
        help="Value checkpoint the 'think harder' search uses (gen4's critic searches better than value_v2)",
    )
    p.add_argument(
        "--play-policy", type=str, default=None,
        help="Policy the model plays hotseat seats with (default: --policy); every decision is searched",
    )
    p.add_argument(
        "--play-value", type=str, default=None, help="Leaf value of its searches (default: --search-value)"
    )
    p.add_argument(
        "--play-auction-policy", type=str, default=None,
        help="Policy its private-auction searches use (default: --auction-policy)",
    )
    p.add_argument(
        "--play-auction-value", type=str, default=None,
        help="Leaf value of its private-auction searches (default: --play-value); most value nets never "
        "saw an auction position (gen4's critic did: its run started 15%% of games at the auction)",
    )
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--cpu-threads", type=int, default=4, help="Torch threads when on CPU (default: 4)")
    p.add_argument(
        "--debug-page", action="store_true",
        help="Also serve /debug: paste a game JSON and see the extension's panel for it",
    )
    p.add_argument(
        "--web-site-port", type=int, action="append", default=[],
        help="Let the pages of a site on this machine served on this port ask for advice (/api/advise, "
        "/api/health): the local copy of 18xx.games (9292) has an engine panel. Repeatable",
    )
    p.add_argument(
        "--bots", action="store_true",
        help="Also play bot seats: on the local copy of 18xx.games and/or 18xx.games, each switched on and off "
        "on the admin page /admin (rl18xx/agent/advisor/bot_hub.py)",
    )
    p.add_argument(
        "--bot-settings", type=str, default="~/.config/18xx/bot_hub.json",
        help="The bots' settings, as the admin page saves them (default: ~/.config/18xx/bot_hub.json)",
    )
    p.add_argument(
        "--admin-password-file", type=str, default="~/.config/18xx/bot_admin_password",
        help="The admin page's password (user admin), made on first start (default: ~/.config/18xx/bot_admin_password)",
    )

    # value-train (the long run's value trainer)
    from rl18xx.agent.alphazero.value_trainer import ValueTrainerConfig as V

    p = sub.add_parser(
        "value-train",
        help="Train a separate value network offline over saved search-learning games, in gated cycles",
        description="Long-lived: polls the --data shards, trains a candidate every cycle (warm start from the "
        "current promoted value) and promotes it through the paired-CE gate; resumes an existing --run-dir. "
        "rl18xx/agent/alphazero/value_trainer.py has the run directory's contract.",
    )
    p.add_argument("--run-dir", required=True, help="The trainer's run directory (resumed if it exists)")
    p.add_argument(
        "--data", nargs="+", action="extend", required=True,
        help="search-learning --save-data directories (repeatable); rescanned every poll",
    )
    p.add_argument("--init", default=V.init, help=f"Start value, promoted version 0 on a new run (default: {V.init})")
    p.add_argument(
        "--fixed-data", default=V.fixed_data,
        help=f"The value gate's positions; its test games are the fixed held-out set (default: {V.fixed_data})",
    )
    p.add_argument("--no-fixed", action="store_true", help="No fixed held-out set (tests)")
    p.add_argument(
        "--gate-data", nargs="+", action="extend", default=[],
        help="Value-gate data directories of eval-style games (their test split): the gate requires the gain "
        "there instead of on the rolling set, and a cycle trains without early stopping (default: none)",
    )
    p.add_argument(
        "--anchor-data", default=V.anchor_data,
        help="A value-gate data directory whose train split joins every batch (default: none)",
    )
    p.add_argument("--anchor-share", type=float, default=V.anchor_share,
                   help=f"Share of each batch from --anchor-data (default: {V.anchor_share})")
    p.add_argument("--positions-per-game", type=int, default=V.positions_per_game,
                   help=f"Training positions per game and pass (default: {V.positions_per_game})")
    p.add_argument("--num-players", type=int, default=V.num_players,
                   help=f"Train and hold out only games of this many players (default: {V.num_players})")
    p.add_argument("--held-out-per-mille", type=int, default=V.held_out_per_mille,
                   help=f"Games with crc32(uid) %% 1000 below this are held out (default: {V.held_out_per_mille})")
    p.add_argument("--rolling-games", type=int, default=V.rolling_games,
                   help=f"The rolling held-out set's games come from the newest N (default: {V.rolling_games})")
    p.add_argument("--rolling-positions-per-game", type=int, default=V.rolling_positions_per_game,
                   help=f"(default: {V.rolling_positions_per_game})")
    p.add_argument("--rolling-max-positions", type=int, default=V.rolling_max_positions,
                   help=f"(default: {V.rolling_max_positions})")
    p.add_argument("--select-share", type=float, default=V.select_share,
                   help="Share of the rolling games used for early stopping only, kept out of the gate (default: "
                   f"{V.select_share}; 0 = early-stop on the gate's own rolling set, an optimistic gate)")
    p.add_argument("--cycle-games", type=int, default=V.cycle_games,
                   help=f"A cycle starts once this many new trainable games arrived (default: {V.cycle_games})")
    p.add_argument("--cycle-hours", type=float, default=V.cycle_hours,
                   help=f"... or this long after the last cycle with --min-new-games new (default: {V.cycle_hours})")
    p.add_argument("--min-new-games", type=int, default=V.min_new_games, help=f"(default: {V.min_new_games})")
    p.add_argument("--poll-seconds", type=float, default=V.poll_seconds, help=f"(default: {V.poll_seconds})")
    p.add_argument("--recent-games", type=int, default=V.recent_games,
                   help=f"Every trainable game of the newest N, plus as many older ones (default: {V.recent_games})")
    p.add_argument("--sample-scale", type=float, default=V.sample_scale,
                   help=f"Scales the sample (doubled after failed cycles) (default: {V.sample_scale})")
    p.add_argument("--escalate-after", type=int, default=V.escalate_after,
                   help=f"Failed cycles in a row before the interval and sample double (default: {V.escalate_after})")
    p.add_argument("--lr", type=float, default=V.lr, help=f"AdamW learning rate (default: {V.lr})")
    p.add_argument("--batch-size", type=int, default=V.batch_size, help=f"(default: {V.batch_size})")
    p.add_argument("--score-weight", type=float, default=V.score_weight,
                   help=f"Weight of the placement-points MSE (default: {V.score_weight})")
    p.add_argument("--epochs", type=float, default=V.epochs, help=f"Passes over a cycle's sample (default: {V.epochs})")
    p.add_argument("--ema-decay", type=float, default=V.ema_decay,
                   help="> 0: evaluations, early stopping and the candidate use an exponential moving average of "
                   f"the weights with this decay per step, e.g. 0.9998 (default: {V.ema_decay} = the raw weights)")
    p.add_argument("--eval-every", type=int, default=V.eval_every,
                   help=f"Steps between rolling-set evaluations (default: {V.eval_every})")
    p.add_argument("--patience", type=int, default=V.patience,
                   help=f"Evaluations without a gain before stopping (default: {V.patience})")
    p.add_argument("--min-improvement", type=float, default=V.min_improvement, help=f"(default: {V.min_improvement})")
    p.add_argument("--max-steps", type=int, default=V.max_steps, help="Cap on a cycle's steps (default: 0 = none)")
    p.add_argument("--promote-z", type=float, default=V.promote_z,
                   help=f"Gate: rolling gain >= z paired SE, fixed set not worse by > z SE (default: {V.promote_z})")
    p.add_argument("--diverge-ce", type=float, default=V.diverge_ce,
                   help=f"Rolling CE above the current value's by more: diverged (default: {V.diverge_ce})")
    p.add_argument("--workers", type=int, default=V.workers,
                   help=f"Re-encoding processes (0: in the trainer's process) (default: {V.workers})")
    p.add_argument("--task-games", type=int, default=V.task_games, help=f"Games per encoding task ({V.task_games})")
    p.add_argument("--buffer-positions", type=int, default=V.buffer_positions,
                   help=f"Shuffle buffer size, ~11 KB a position (default: {V.buffer_positions})")
    p.add_argument("--eval-batch-size", type=int, default=V.eval_batch_size, help=f"(default: {V.eval_batch_size})")
    p.add_argument("--device", default=V.device, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--seed", type=int, default=V.seed, help=f"(default: {V.seed})")
    p.add_argument("--max-cycles", type=int, default=0, help="Exit after this many cycles (default: 0 = until STOP)")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.explicit = explicit_options()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "pretrain": cmd_pretrain,
        "clean": cmd_clean,
        "convert": cmd_convert,
        "convert-generic": cmd_convert_generic,
        "pretrain-generic": cmd_pretrain_generic,
        "compare-generic": cmd_compare_generic,
        "policy-selfplay": cmd_policy_selfplay,
        "policy-gradient": cmd_policy_gradient,
        "policy-gradient-generic": cmd_policy_gradient_generic,
        "search-learning": cmd_search_learning,
        "refit-price-head": cmd_refit_price_head,
        "arena": cmd_arena,
        "dashboard": cmd_dashboard,
        "replay": cmd_replay,
        "advisor": cmd_advisor,
        "value-train": cmd_value_train,
    }
    commands[args.command](args)
