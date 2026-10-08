"""Fast self-play driven by the policy alone: AlphaGo-style value-network data.

AlphaGo trained its value network on games its policy played against itself,
one position per game from millions of games: the positions of one game share
one outcome, and a value net fit to whole games memorizes them (on human games
its train MSE was 0.19 against 0.37 on new games). Here search-free self-play
costs one network evaluation per decision instead of a search's ~64, so the
same machine plays two orders of magnitude more games, and each game
contributes only ``positions_per_game`` positions (a uniform reservoir sample
of its decisions).

Every game starts at the first Stock Round (``start_positions``), each player
samples its moves from the policy and prices from the price head, forced moves
are applied without the network, and the game is scored on net worth when it
finishes or reaches ``max_decisions``. The first ``opening_decisions`` moves
are sampled at ``opening_temperature`` (diverse games) and the rest at
``temperature`` -- below 1 the outcome reflects sharper play, closer to how a
searching agent plays, as AlphaGo's value data used its strong policy after a
diverse opening. Rows are written in
the human-data layout -- ``(encoded_state, legal_indices, pi, value,
price_targets)`` with the full encoded state (its rotation at index 6) and the
value as net-worth fractions in player-id order -- to
``<out>/training`` and ``<out>/validation`` (a hash of the game's id), so the
pretraining loader reads them unchanged.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def load_checkpoint_model(checkpoint_path):
    """Inference-server model loader for one checkpoint (module-level so it pickles)."""
    from rl18xx.agent.alphazero.checkpointer import _load_model_from_session_checkpoint

    path = Path(checkpoint_path)
    model = _load_model_from_session_checkpoint(path.parent, path)
    model.eval()
    return model


@dataclass
class _Game:
    game: object  # RustGameAdapter
    uid: str
    start: str
    decisions: int = 0
    seen: int = 0
    reservoir: list = field(default_factory=list)
    price_row: Optional[np.ndarray] = None  # this game's last price-head output
    termination: Optional[str] = None


def _sample_price(game, index: int, price_row: Optional[np.ndarray], rng: np.random.Generator, eps: float):
    """Price for a legal ``index``: None for categorical moves, the engine price for
    fixed-price ones, and for price-head slots a cell drawn from the head (mixed
    with ``eps`` uniform) and a price inside it. Keep ``eps`` 0 for play: the
    Rust search's ``price_explore_eps`` only orders which price cells it
    expands -- it still plays the most-visited move -- but here a mixed draw
    is the price actually paid, i.e. a uniformly random price."""
    from rl18xx.agent.alphazero import price_pmf

    rs = game._game
    price_range = rs.price_range_for_index(int(index))
    if price_range is None:
        return None
    lo, hi = price_range
    slot = rs.price_head_slot_for_index(int(index))
    if lo == hi or slot is None:
        return int(lo)
    slot_index, lo, hi = slot
    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot_index], int(lo), int(hi))
    logits = price_row[slot_index] if price_row is not None else None
    probs = cells.cell_probs(logits)
    live = cells.nonempty
    mixed = np.where(live, (1.0 - eps) * probs + eps / live.sum(), 0.0)
    cell = int(rng.choice(len(mixed), p=mixed / mixed.sum()))
    return int(cells.sample_price(cell, rng))


def has_price_choice(rs, index: int) -> bool:
    """Whether a legal ``index`` still leaves its price open (a price-head slot
    with more than one legal price), so playing it is a decision."""
    price_range = rs.price_range_for_index(int(index))
    return price_range is not None and price_range[0] < price_range[1] and (
        rs.price_head_slot_for_index(int(index)) is not None
    )


def _advance(g: _Game, settings: dict, rng: np.random.Generator) -> Optional[list]:
    """Apply forced moves until a decision (returns its legal indices) or the end (None).

    A single legal index whose price is still open -- e.g. a corporation that
    must buy a train and can only buy it from another corporation, at any
    price from $1 to its cash -- is a decision, not a forced move: its price
    comes from the mover's network at this position (it used to come from
    the game's previous network output, a stale position and possibly another
    seat's network, and was never trained on)."""
    rs = g.game._game
    while True:
        if g.game.finished:
            g.termination = g.termination or "finished"
            return None
        if g.decisions >= settings["max_decisions"]:
            # Scored on net worth as it stands (the Rust engine has no end_game).
            g.game.end_game()
            g.termination = "max_length"
            return None
        legal = [int(i) for i in rs.factored_legal_indices()]
        if len(legal) > 1 or (len(legal) == 1 and has_price_choice(rs, legal[0])):
            return legal
        if not legal:
            raise RuntimeError(f"game {g.uid}: no legal action in an unfinished game")
        rs.apply_action_index(legal[0], _sample_price(g.game, legal[0], g.price_row, rng, settings["price_eps"]))


def _finish(g: _Game) -> list:
    """Training rows for a finished game's reservoir, valued by its final net worth."""
    from rl18xx.agent.alphazero.mcts import POLICY_SIZE
    from rl18xx.agent.alphazero.self_play import _compute_net_worth

    net_worth = _compute_net_worth(g.game)
    scores = np.array([float(net_worth[pid]) for pid in sorted(net_worth)], dtype=np.float32).clip(min=0)
    total = float(scores.sum())
    value = torch.from_numpy(scores / total if total > 0 else np.full_like(scores, 1.0 / len(scores)))
    rows = []
    for encoded, legal, probs in g.reservoir:
        pi = torch.zeros(POLICY_SIZE)
        pi[legal] = torch.from_numpy(probs.astype(np.float32))
        rows.append((encoded, legal, pi, value.clone(), []))
    return rows


def play_policy_games(num_games: int, settings: dict) -> dict:
    """Pool task: play ``num_games`` concurrent games on this worker's inference
    client and return their training rows (grouped by game) and stats."""
    from rl18xx.agent.alphazero.inference_server import get_worker_client
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.agent.alphazero.start_positions import apply_actions, new_game, sample_start_position

    client = get_worker_client()
    if client is None:
        raise RuntimeError("play_policy_games needs a worker inference client (worker_init_inference)")
    rng = np.random.default_rng()
    py_rng = random.Random()
    k = settings["positions_per_game"]
    start_time = time.time()

    games = []
    for _ in range(num_games):
        start = sample_start_position(
            4, settings["start_positions"], settings["random_start_fraction"], rng=py_rng
        )
        game = apply_actions(new_game(4), start.actions)
        games.append(_Game(game=game, uid=uuid.uuid4().hex, start=start.label))

    finished, active, decisions_total, requests = [], list(games), 0, 0
    while active:
        batch = []
        for g in active:
            legal = _advance(g, settings, rng)
            if legal is None:
                finished.append(g)
            else:
                batch.append((g, legal))
        active = [g for g, _ in batch]
        if not batch:
            break
        encoded = [_rust_encode(g.game) for g, _ in batch]
        probs, _, _ = client.run_many_encoded(encoded, legal_indices=[legal for _, legal in batch])
        requests += 1
        price = client.last_price_components
        price_logits = price["price_logits"].numpy() if price is not None else None
        for i, (g, legal) in enumerate(batch):
            p = probs[i].numpy()[legal].astype(np.float64)
            opening = g.decisions < settings.get("opening_decisions", 0)
            temperature = settings.get("opening_temperature", 1.0) if opening else settings["temperature"]
            if temperature != 1.0:
                p = np.power(np.clip(p, 1e-12, None), 1.0 / temperature)
            p = p / p.sum() if p.sum() > 0 else np.full(len(legal), 1.0 / len(legal))
            choice = legal[int(rng.choice(len(legal), p=p))]
            # Uniform reservoir sample of the game's decisions.
            g.seen += 1
            if len(g.reservoir) < k:
                g.reservoir.append((encoded[i], legal, p))
            else:
                j = int(rng.integers(0, g.seen))
                if j < k:
                    g.reservoir[j] = (encoded[i], legal, p)
            g.price_row = price_logits[i] if price_logits is not None else None
            price = _sample_price(g.game, choice, g.price_row, rng, settings["price_eps"])
            g.game._game.apply_action_index(choice, price)
            g.decisions += 1
            decisions_total += 1

    return {
        "games": [
            {"uid": g.uid, "start": g.start, "termination": g.termination, "decisions": g.decisions, "rows": _finish(g)}
            for g in finished
        ],
        "decisions": decisions_total,
        "requests": requests,
        "seconds": time.time() - start_time,
    }


def generate(
    checkpoint: Path,
    out_dir: Path,
    num_games: int,
    workers: int = 48,
    games_per_task: int = 32,
    positions_per_game: int = 4,
    temperature: float = 1.0,
    opening_decisions: int = 0,
    opening_temperature: float = 1.0,
    max_decisions: int = 1000,
    start_positions: str = "human_games/start_positions_1830_4p.jsonl",
    random_start_fraction: float = 0.2,
    price_eps: float = 0.0,
    validation_percentage: float = 0.05,
    batch_size: int = 512,
    write_every: int = 1000,
) -> dict:
    """Play ``num_games`` policy-only games with ``checkpoint`` and write their
    sampled positions to ``out_dir/{training,validation}``. Returns stats."""
    from rl18xx.agent.alphazero.dataset import TrainingExampleProcessor
    from rl18xx.agent.alphazero.inference_server import (
        start_inference_server,
        wait_checking_servers,
        worker_init_inference,
    )
    from rl18xx.agent.alphazero.pretraining import in_validation_split

    settings = {
        "positions_per_game": positions_per_game,
        "temperature": temperature,
        "opening_decisions": opening_decisions,
        "opening_temperature": opening_temperature,
        "max_decisions": max_decisions,
        "start_positions": start_positions,
        "random_start_fraction": random_start_fraction,
        "price_eps": price_eps,
    }
    out_dir = Path(out_dir)
    (out_dir / "training").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)
    writer = TrainingExampleProcessor(encoder=None)
    server = start_inference_server(
        num_workers=workers,
        model_factory=load_checkpoint_model,
        checkpoint_path=str(checkpoint),
        batch_size=batch_size,
        autocast_device=None,
    )
    stats = {"games": 0, "rows": 0, "decisions": 0, "terminations": {}, "starts": {}, "lengths": []}
    pending_train, pending_val = [], []
    started = time.time()

    def flush():
        if pending_train:
            writer.write_samples(pending_train, out_dir / "training")
            pending_train.clear()
        if pending_val:
            writer.write_samples(pending_val, out_dir / "validation")
            pending_val.clear()

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init_inference,
            initargs=(server.request_q, server.reply_qs, server.ticket_q),
        ) as pool:
            submitted = 0
            futures = set()
            while submitted < num_games and len(futures) < workers:
                n = min(games_per_task, num_games - submitted)
                futures.add(pool.submit(play_policy_games, n, settings))
                submitted += n
            while futures:
                done, futures = wait_checking_servers(futures, {"policy": server}, pool)
                for future in done:
                    result = future.result()
                    stats["decisions"] += result["decisions"]
                    for record in result["games"]:
                        stats["games"] += 1
                        stats["lengths"].append(record["decisions"])
                        start_kind = record["start"].split(":")[0]
                        for key, name in (("terminations", record["termination"]), ("starts", start_kind)):
                            stats[key][name] = stats[key].get(name, 0) + 1
                        held_out = in_validation_split(record["uid"], validation_percentage)
                        dest = pending_val if held_out else pending_train
                        dest.extend(record["rows"])
                        stats["rows"] += len(record["rows"])
                    if submitted < num_games:
                        n = min(games_per_task, num_games - submitted)
                        futures.add(pool.submit(play_policy_games, n, settings))
                        submitted += n
                if len(pending_train) + len(pending_val) >= write_every * positions_per_game:
                    flush()
                    elapsed = time.time() - started
                    LOGGER.info(
                        f"{stats['games']}/{num_games} games, {stats['decisions'] / elapsed:.0f} decisions/s, "
                        f"{stats['games'] / elapsed * 3600:.0f} games/hour"
                    )
            flush()
    finally:
        server.shutdown()
    elapsed = time.time() - started
    lengths = stats.pop("lengths")
    stats.update(
        seconds=round(elapsed, 1),
        games_per_hour=round(stats["games"] / elapsed * 3600),
        decisions_per_second=round(stats["decisions"] / elapsed),
        mean_decisions=round(float(np.mean(lengths)), 1) if lengths else 0.0,
    )
    return stats
