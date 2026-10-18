"""Learning from search (expert iteration), steps 3-4 of docs/search_learning_design.md.

The policy is trained toward what a search does with it, the critic (the
search's value network) toward the games' outcomes:

- **Policy.** The learner's policy *is* softmax(logits / T), T its training
  temperature (``learner_temperature``; by default the starting checkpoint's
  run's, 0.5 for gen4). The search's priors are that policy
  (``prior_temperature`` = T), and the loss is the cross-entropy of
  log_softmax(logits / T) over the legal moves to the search's target, plus
  ``kl_coef`` * KL(learner || anchor) at T (the anchor moves to the learner
  every ``anchor_every`` updates; ``kl_target`` adapts the coefficient), as in
  ``policy_gradient``. Targets (``target``):

  - ``visits``: the root's visit distribution from the value-driven PUCT search
    validated in play (mean Q, c_puct_init 0.15 in every round, dedupe, 8
    leaves per batch, priors at T; @200 beat gen4 0.933 2v2, 0.787 1v3), by
    default with KataGo's policy-target pruning (``visit_target``): with
    first-play urgency at the parent's value most moves get a visit or two
    of exploration, which the raw distribution would teach the sharp policy
    (``scripts/search_target_stats.py``). The move played is the most visited
    (``full_move_temperature`` > 0 samples from the visits instead).
  - ``gumbel``: the Gumbel root's improved policy softmax(logits +
    sigma(completed q)) (``engine-rs/src/gumbel.rs``), the move its
    sequential-halving choice. Its logits are the tree's priors, already
    softmax(logits / T), so the target improves on the policy as defined.

  The price head's target at a searched price decision is the searched price
  grandchildren's visit distribution over the slot's price cells.
- **Playout-cap randomization** (KataGo): each decision with several legal
  moves is searched with probability ``p_full`` (``n_full`` readouts) and
  becomes a training row; the others are played by the policy at T (a
  ``PolicyRequest`` in the same lockstep round) and are not trained on, nor
  are single-move decisions with an open price (played by the price head) or
  forced moves (applied as in policy-only play).
- **Critic.** ``policy_gradient.critic_losses`` on the outcome: the win-share
  CE on the win head and (``reward`` "placement", as gen4's critic) the score
  head on placement points, on ``critic_positions_per_game`` rows of each game,
  each used once.
- **Generation** runs continuously: every worker keeps ``games_per_worker``
  games going across tasks (``lockstep.LockstepSearcher``: one request per
  round carries every game's search leaves and policy moves), writes a row at
  each searched decision (``BaseGame.encode_canonical_py``) and fills in the
  game's outcome when it ends. All seats are the learner (``seat_servers``
  holds a hook for league opponents: other servers' seats play their policy
  and give no rows). Starts as in ``policy_gradient`` (human auction endings,
  random endings, ``midgame_fraction``, ``auction_fraction``); a start that no
  longer replays is skipped.
- **Training** samples minibatches from a replay window of the latest
  ``window_rows`` rows: the targets don't depend on the policy that played, so
  rows can be reused without importance ratios. Each update waits for
  ``rows_per_update`` new rows and takes ``steps_per_update`` steps of
  ``minibatch`` rows: a row is trained on ``steps_per_update * minibatch /
  rows_per_update`` times on average (``reuse`` in the history).

az2's options (docs/search_learning_design.md, "az2"; all off by default):

- ``shared``: one network. The search reads the learner's own value heads (the
  server loads the learner alone; ``value_head`` ``score`` / ``blend=<w>`` via
  ``composite_model.SharedValueModel``), and the value loss joins the policy loss
  on the same forward (``value_loss_weight``, ``score_loss_weight``), the value
  heads at ``critic_lr``, the trunk at ``lr``; ``value_stop_grad`` keeps the
  value gradient out of the trunk. Start from a learner whose heads were refit
  to its critic (``scripts/refit_value_heads.py``).
- ``value_nstep`` / ``value_bootstrap_weight``: every search records its root
  value (per seat, absolute order); a row's win-head target mixes the outcome
  with the root value ``value_nstep`` searches later (MuZero's n-step target).
- ``aux_loss_weight``: auxiliary outcome heads (final net-worth / cash shares,
  corporations floated) on the network that learns the value.
- ``surprise_weight`` / ``surprise_cap``: KataGo's policy surprise weighting.
- ``critic_positions_per_game`` 0 trains the value on every searched row (once).

A run directory as ``policy_gradient``'s: ``<out_dir>/<run>/{learner,critic}/
<update>.pth`` (every ``snapshot_every``-th kept), ``config.json``,
``state.json`` + ``optimizer.pth`` at snapshots (``--resume``), one JSON line
per update in ``history.jsonl``, TensorBoard ``SL/*`` in
``runs/alphazero_runs/<run>``; touch ``STOP`` to end after the current update.
The replay window is not saved: a resumed run refills it first.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

NUM_PLAYERS = 4
TARGETS = ("visits", "gumbel")


@dataclass
class SearchLearningConfig:
    policy_checkpoint: str
    value_checkpoint: str
    out_dir: str = "model_checkpoints_search"
    run_name: Optional[str] = None
    # The policy is softmax(logits / T); None: the starting checkpoint's run's
    # learner_temperature (1 for a supervised checkpoint).
    learner_temperature: Optional[float] = None
    # Targets and search. Gumbel (the default since 2026-10-18): against PUCT at the
    # same budget it scored 0.738 at 16 readouts, 0.596 at 32, 0.492 at 64, 0.525 at 200.
    target: str = "gumbel"
    # visits: KataGo's policy-target pruning (keep only the visits a move's PUCT
    # score needed against the most-visited move's; "raw": the whole distribution).
    # On gen4 the raw visits put 12-19% of the target on moves its policy gives
    # under 1% (FPU exploration), KL(target || policy) 1.6-2.4; pruned 0.2-0.35.
    visit_target: str = "pruned"
    # Sharpen the target: target ** (1 / t) renormalized; 0 is one-hot on the move the
    # search played. The 2026-10-17 pilot (t = 1, pruned visits @200) flattened gen4: its
    # targets carried more entropy than the sharp policy (0.19 vs 0.14 at the start), the
    # policy followed (0.14 -> 0.45 by update 100) and played worse (0.28 2v2 vs gen4).
    target_temperature: float = 1.0
    p_full: float = 0.25
    n_full: int = 200
    # Readouts of the other learner decisions (KataGo's fast searches: played, never
    # trained on); 0 lets the policy play them at T. With 0 a game is mostly policy
    # play, so the critic learns how the raw policy's games end, not search play's.
    n_fast: int = 0
    c_puct_init: float = 0.15
    mean_q: bool = True
    fpu_reduction: float = 0.0
    dedupe_leaves: bool = True
    parallel_readouts: int = 8
    # The search's priors: softmax(logits / prior_temperature); None: learner_temperature.
    prior_temperature: Optional[float] = None
    gumbel_k: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    gumbel_noise_scale: float = 1.0
    # visits: 0 plays the most-visited move; > 0 samples it with weight N ** (1 / t).
    full_move_temperature: float = 0.0
    # Exploration as AlphaZero's first-30-moves temperature: full_move_temperature applies only to
    # each game's first explore_decisions decisions (0: every decision). The targets stay the
    # search's visits; only which move is played (and so which positions the games reach) changes.
    explore_decisions: int = 0
    # The search's leaf value: the critic's win head ("win") or score head ("score").
    value_head: str = "win"
    # Tree reuse (KataGo / the single-tree Rust path's advance_root): each game keeps
    # its tree across decisions, advanced along every move played, so a search starts
    # from the chosen move's subtree. "total": a search stops once the root has the
    # visits a fresh search ends with (1 + readouts), counting the inherited ones --
    # the same per-decision visit budget, fewer evaluations; "additional": readouts
    # more on top of them (RustMCTSPlayer.suggest_move's and eval_head_to_head's
    # count) -- the same evaluations, more visits; "off": a fresh tree each decision.
    # A Gumbel root spends its budget either way (inherited visits inform q). A tree is
    # dropped when the network version changes or the game left the tree's line (an
    # open-price forced move the tree priced itself).
    reuse_tree: str = "off"
    # "additional" trees keep growing: one whose root passed this many visits is
    # rebuilt (None: 2 * n_full; a tree node holds a game clone, ~124 KB).
    reuse_max_root_visits: Optional[int] = None
    # Each request sends each distinct hex map once (lockstep dedupe_maps); the
    # server's map encoder runs once per map. Outputs within float rounding.
    dedupe_maps: bool = False
    # Inference server processes (start_inference_server(replicas=)): while one
    # runs a forward pass the others receive, unpickle and reply. Replies are
    # unchanged (each is one server's answer to its own batch).
    server_replicas: int = 1
    # The inference server: a collector thread receives the next batch while
    # the current one runs (InferenceServer.pipeline); replies are unchanged.
    # Measured slower than replicas (the threads contend for the GIL).
    server_pipeline: bool = False
    # Generation
    workers: int = 48
    games_per_worker: int = 24
    task_seconds: float = 30.0
    max_decisions: int = 1000
    start_positions: str = "human_games/start_positions_1830_4p.jsonl"
    random_start_fraction: float = 0.2
    midgame_positions: Optional[str] = None
    midgame_fraction: float = 0.0
    auction_fraction: float = 0.0
    inference_batch_size: int = 512
    # Critic rows: this many of each game's rows (0: all), each trained on once.
    critic_positions_per_game: int = 32
    # The critic's score-head target: "placement" (gen4's, 1/0.5/0.2/0 by place;
    # "placement:<points>") or "win" (the net-worth shares).
    reward: str = "placement"
    # Training
    window_rows: int = 131072
    rows_per_update: int = 16384
    min_window_rows: int = 16384
    steps_per_update: int = 16
    minibatch: int = 2048
    microbatch: int = 1024
    lr: float = 2e-5
    critic_lr: float = 3e-5
    price_loss_weight: float = 1.0
    score_loss_weight: float = 1.0
    kl_coef: float = 0.02
    kl_target: Optional[float] = 0.08
    kl_coef_min: float = 0.005
    anchor_every: int = 50
    max_updates: int = 1000
    snapshot_every: int = 10
    # -- az2 (docs/search_learning_design.md, "az2"), each off by default --
    # One network (AlphaZero / KataGo / MuZero): the search's leaf values come from the
    # learner's own value heads (``value_head`` of the learner; no critic, no second
    # forward) and the value loss trains the learner -- its trunk alongside the policy
    # loss (``value_loss_weight`` on the win CE, ``score_loss_weight`` on the score
    # MSE), its value heads at ``critic_lr``. ``value_stop_grad`` keeps the value
    # gradient out of the trunk (the heads then learn on the policy's features).
    shared: bool = False
    value_loss_weight: float = 1.0
    value_stop_grad: bool = False
    # Value targets. 0: the win head learns the game's outcome. n > 0 (MuZero's n-step
    # target, gamma 1, no intermediate rewards): (1 - w) * outcome + w * the search's
    # root value n searched decisions later in the game (full and fast searches, every
    # seat's), the outcome alone where fewer than n remain; w = value_bootstrap_weight.
    value_nstep: int = 0
    value_bootstrap_weight: float = 0.5
    # Or the lambda-return over the game's searches instead of the n-step value (TD(lambda),
    # gamma 1): G_i = (1 - lambda) * v_{i+1} + lambda * G_{i+1}, the outcome after the last
    # search; the target is then (1 - w) * outcome + w * G_i. None: off.
    value_lambda: Optional[float] = None
    # Auxiliary outcome targets (0: none): per seat the final net-worth share and cash
    # share (MSE), and which corporations floated (BCE), on the network that learns the
    # value (training-only heads: ``AlphaZeroTransformerModel.add_aux_heads``).
    aux_loss_weight: float = 0.0
    # KataGo's policy surprise weighting: a row's target CE weighs
    # (1 - a) + a * s / mean(s), s = KL(target || policy) at T (the policy being trained,
    # the microbatch's mean), capped at ``surprise_cap`` and renormalized to mean 1;
    # a = surprise_weight (KataGo's 0.5), 0 = off. The price and anchor terms are unweighted.
    surprise_weight: float = 0.0
    surprise_cap: float = 8.0


AUX_SEAT_TARGETS = 2  # final net-worth share, final cash share
AUX_GLOBAL_TARGETS = 8  # floated, per corporation (the engine's order)


def sharpen_target(target: np.ndarray, legal: np.ndarray, move: int, temperature: float) -> np.ndarray:
    """``target`` over ``legal`` normalized and sharpened to ``target ** (1 / temperature)``;
    temperature 0 (or an all-zero target) puts everything on ``move``, the move played."""
    target = np.asarray(target, dtype=np.float64)
    if temperature <= 0 or target.sum() <= 0:
        out = np.zeros(len(legal), dtype=np.float64)
        out[int(np.flatnonzero(legal == move)[0])] = 1.0
        return out
    if temperature != 1.0:
        target = np.power(target / target.max(), 1.0 / temperature)
    return target / target.sum()


def run_temperature(path) -> float:
    """The ``learner_temperature`` of the run a ``<run>/learner/<n>.pth`` checkpoint
    comes from (its ``config.json``), else 1 -- as the advisor's
    ``training_temperature`` reads it."""
    path = Path(path)
    config = path.parent.parent / "config.json"
    if path.parent.name != "learner" or not config.is_file():
        return 1.0
    try:
        return float(json.loads(config.read_text()).get("learner_temperature", 1.0) or 1.0)
    except (ValueError, OSError):
        return 1.0


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class SearchRow:
    """One searched decision. The state is in the encoder's canonical frame
    (the mover first); ``fractions`` / ``score_target`` are per seat in
    player-id order and rotated into that frame when batched."""

    game_state: np.ndarray  # (D,) float32
    node_features: np.ndarray  # (hexes, features) float32
    rotation: int  # the mover's seat
    legal: np.ndarray  # int32 policy indices, sorted
    target: np.ndarray  # float32 over ``legal``, sums to 1
    # At a searched price decision: (slot, lo, hi, float32 target over the slot's price cells).
    price: Optional[tuple]
    version: int  # learner update the generating policy came from
    fractions: Optional[np.ndarray] = None  # net-worth shares at the game's end
    score_target: Optional[np.ndarray] = None  # placement points (None: the score head learns the shares)
    critic: bool = False  # still to be trained on by the critic
    # The search's root value per seat (player-id order) at decision time, and this
    # decision's place among the game's searches (full and fast).
    root_value: Optional[np.ndarray] = None
    search_index: int = -1
    win_target: Optional[np.ndarray] = None  # the win head's target (player-id order); None: the outcome
    aux_seat: Optional[np.ndarray] = None  # (players, AUX_SEAT_TARGETS) at the game's end, player-id order
    aux_global: Optional[np.ndarray] = None  # (AUX_GLOBAL_TARGETS,) corporations floated at the end


def root_value(tree) -> np.ndarray:
    """The search's value of the root per seat (absolute order): the mean of every
    value backed up through it (``root_q_vector`` is minigo's W / (1 + N))."""
    n = float(tree.n_at_root())
    q = np.asarray(tree.root_q_vector(), dtype=np.float64)
    return (q * (1.0 + n) / max(n, 1.0)).astype(np.float32)


def nstep_win_target(outcome, search_values: list, index: int, n: int, weight: float) -> np.ndarray:
    """``(1 - weight) * outcome + weight * search_values[index + n]`` (normalized to a
    distribution over the seats), or the outcome when ``n`` is 0, the row is not a
    search (index < 0) or the game has fewer than ``n`` searches after it."""
    outcome = np.asarray(outcome, dtype=np.float64)
    if n <= 0 or weight <= 0 or index < 0 or index + n >= len(search_values):
        return outcome.astype(np.float32)
    later = np.clip(np.asarray(search_values[index + n], dtype=np.float64)[: len(outcome)], 0.0, None)
    if later.sum() <= 0:
        return outcome.astype(np.float32)
    return ((1.0 - weight) * outcome + weight * later / later.sum()).astype(np.float32)


def _as_distribution(value, like: np.ndarray) -> Optional[np.ndarray]:
    v = np.clip(np.asarray(value, dtype=np.float64)[: len(like)], 0.0, None)
    return v / v.sum() if v.sum() > 0 else None


def lambda_returns(outcome, search_values: list, lam: float) -> list:
    """TD(lambda) returns (gamma 1, no intermediate rewards) over a game's searches:
    ``G_i = (1 - lam) * v_{i+1} + lam * G_{i+1}``, ``G_last`` = the outcome, each
    root value ``v`` normalized over the seats (lam 1: the outcome; 0: the next
    search's value)."""
    outcome = np.asarray(outcome, dtype=np.float64)
    returns = [None] * len(search_values)
    following = outcome
    for i in range(len(search_values) - 1, -1, -1):
        returns[i] = following
        v = _as_distribution(search_values[i], outcome)
        following = following if v is None else (1.0 - lam) * v + lam * following
    return returns


def lambda_win_target(outcome, returns: list, index: int, weight: float) -> np.ndarray:
    """``(1 - weight) * outcome + weight * returns[index]`` (the outcome for a row
    that is not a search)."""
    outcome = np.asarray(outcome, dtype=np.float64)
    if index < 0 or index >= len(returns) or weight <= 0:
        return outcome.astype(np.float32)
    return ((1.0 - weight) * outcome + weight * returns[index]).astype(np.float32)


def root_value_ce(rows: list, win: np.ndarray) -> tuple:
    """``(sum, count)`` over ``rows`` of the win CE of each row's search root value
    (normalized over the seats) against the game's win share ``win``."""
    total, count = 0.0, 0
    for row in rows:
        v = row.root_value
        if v is None or float(np.sum(v)) <= 0:
            continue
        p = np.clip(np.asarray(v, dtype=np.float64)[: len(win)] / float(np.sum(v[: len(win)])), 1e-9, 1.0)
        total += float(-(np.asarray(win, dtype=np.float64) * np.log(p)).sum())
        count += 1
    return total, count


def aux_outcomes(game, fractions: np.ndarray) -> tuple:
    """``(per seat [net-worth share, cash share], floated per corporation)`` at the
    end of ``game`` (a RustGameAdapter), player-id order; the cash share is a seat's
    cash over the players' total (clipped) net worth."""
    from rl18xx.agent.alphazero.self_play import _compute_net_worth

    rs = game._game
    net_worth = _compute_net_worth(game)
    total = sum(max(float(v), 0.0) for v in net_worth.values())
    cash = {int(p.id): float(p.cash) for p in rs.players}
    cash_share = np.array([cash.get(int(pid), 0.0) / total if total > 0 else 0.0 for pid in sorted(net_worth)])
    seat = np.stack([np.asarray(fractions, dtype=np.float64), cash_share], axis=1).astype(np.float32)
    floated = np.zeros(AUX_GLOBAL_TARGETS, dtype=np.float32)
    for i, corp in enumerate(list(rs.corporations)[:AUX_GLOBAL_TARGETS]):
        floated[i] = float(bool(corp.floated))
    return seat, floated


def price_cell_target(price_targets: list) -> Optional[tuple]:
    """``(slot, lo, hi, cell distribution)`` from a move's searched price
    grandchildren (``RustMCTSPlayer.price_targets`` entries ``(slot, price,
    weight, lo, hi)``), or None."""
    from rl18xx.agent.alphazero import price_pmf

    if not price_targets:
        return None
    slot, _, _, lo, hi = price_targets[0]
    slot, lo, hi = int(slot), int(lo), int(hi)
    cells = price_pmf.price_cells(price_pmf.SLOT_TYPES[slot], lo, hi)
    target = np.zeros(price_pmf.NUM_CELLS, dtype=np.float64)
    for _, price, weight, _, _ in price_targets:
        target[cells.cell_of(int(round(price)))] += float(weight)
    if target.sum() <= 0:
        return None
    return slot, lo, hi, (target / target.sum()).astype(np.float32)


def search_batch(rows: list, device) -> dict:
    """Stack :class:`SearchRow` s into device tensors: states, the legal mask,
    the targets as (row, column, probability) triples, the price rows with their
    cell targets, and the critic rows with their outcome targets in the mover's
    frame (as ``policy_gradient.batch_tensors`` rotates them)."""
    import torch

    from rl18xx.agent.alphazero import price_pmf
    from rl18xx.agent.alphazero.mcts import POLICY_SIZE
    from rl18xx.agent.alphazero.policy_gradient import _lay_tile_range

    n = len(rows)
    game_state = torch.from_numpy(np.stack([r.game_state for r in rows])).float()
    nodes = torch.from_numpy(np.stack([r.node_features for r in rows])).float()
    lengths = torch.tensor([len(r.legal) for r in rows])
    row_ids = torch.repeat_interleave(torch.arange(n), lengths)
    col_ids = torch.from_numpy(np.concatenate([r.legal for r in rows]).astype(np.int64))
    target = np.concatenate([r.target for r in rows]).astype(np.float32)
    legal = torch.zeros(n, POLICY_SIZE, dtype=torch.bool)
    legal[row_ids, col_ids] = True
    lo, hi = _lay_tile_range()
    lay_tile_rows = legal[:, lo:hi].any(dim=1)
    best = torch.tensor([int(r.legal[int(np.argmax(r.target))]) for r in rows], dtype=torch.long)
    entropy = np.array([float(-(r.target * np.log(np.clip(r.target, 1e-12, None))).sum()) for r in rows])

    price_rows, price_slots, price_masks, price_targets = [], [], [], []
    for b, r in enumerate(rows):
        if r.price is not None:
            slot, plo, phi, cells = r.price
            price_rows.append(b)
            price_slots.append(slot)
            price_masks.append(price_pmf.price_cells(price_pmf.SLOT_TYPES[slot], plo, phi).nonempty)
            price_targets.append(cells)
    critic_rows = [b for b, r in enumerate(rows) if r.critic and r.fractions is not None]

    def frame(values: list) -> torch.Tensor:
        return torch.stack([torch.roll(torch.as_tensor(v), shifts=-int(r.rotation), dims=0) for v, r in values]).float()

    fractions = score_target = win_target = outcome_win = aux_seat = aux_global = None
    if critic_rows:
        fractions = frame([(rows[b].fractions, rows[b]) for b in critic_rows]).to(device)
        if rows[critic_rows[0]].score_target is not None:
            score_target = frame([(rows[b].score_target, rows[b]) for b in critic_rows]).to(device)
        outcome_win = frame([(share_of_winners(rows[b].fractions), rows[b]) for b in critic_rows]).to(device)
        win_target = frame(
            [
                (rows[b].win_target if rows[b].win_target is not None else share_of_winners(rows[b].fractions), rows[b])
                for b in critic_rows
            ]
        ).to(device)
        if all(rows[b].aux_seat is not None for b in critic_rows):
            aux_seat = frame([(rows[b].aux_seat, rows[b]) for b in critic_rows]).to(device)
            aux_global = torch.from_numpy(np.stack([rows[b].aux_global for b in critic_rows])).float().to(device)
    return {
        "game_state": game_state.to(device, non_blocking=True),
        "nodes": nodes.to(device, non_blocking=True),
        "legal": legal.to(device, non_blocking=True),
        "lay_tile_rows": lay_tile_rows,
        "target_rows": row_ids.to(device),
        "target_cols": col_ids.to(device),
        "target_vals": torch.from_numpy(target).to(device),
        "target_best": best.to(device),
        "target_entropy": torch.from_numpy(entropy).float().to(device),
        "price_rows": torch.tensor(price_rows, dtype=torch.long, device=device),
        "price_slots": torch.tensor(price_slots, dtype=torch.long, device=device),
        "price_masks": (
            torch.from_numpy(np.stack(price_masks)).to(device)
            if price_masks
            else torch.zeros(0, price_pmf.NUM_CELLS, dtype=torch.bool, device=device)
        ),
        "price_targets": (
            torch.from_numpy(np.stack(price_targets)).to(device)
            if price_targets
            else torch.zeros(0, price_pmf.NUM_CELLS, device=device)
        ),
        "critic_rows": torch.tensor(critic_rows, dtype=torch.long, device=device),
        "fractions": fractions,
        "score_target": score_target,
        "win_target": win_target,
        "outcome_win": outcome_win,
        "aux_seat": aux_seat,
        "aux_global": aux_global,
    }


def share_of_winners(fractions) -> np.ndarray:
    """The win share of net-worth shares: the leaders split 1 (``train._derive_dual_value_targets``)."""
    f = np.asarray(fractions, dtype=np.float64)
    winners = f >= f.max() - 1e-6
    return (winners / winners.sum()).astype(np.float32)


def value_losses(win_logits, score_pred, win_target, score_target, outcome_win=None) -> dict:
    """The value network's losses on the value rows (mover's frame): ``ce`` the win
    head's cross-entropy to ``win_target`` (the outcome's win share, or an n-step
    mix), ``mse`` the score head's to ``score_target``, ``outcome_ce`` / ``winner_hit``
    against the outcome (``outcome_win``; default ``win_target``) -- comparable across
    value targets. Means over the rows, the targets padded to the heads' seats as
    ``policy_gradient.critic_losses`` pads them (with the outcome as ``win_target``
    the two agree exactly)."""
    import torch
    import torch.nn.functional as F

    outcome = win_target if outcome_win is None else outcome_win
    pad = win_logits.shape[1] - win_target.shape[1]
    if pad:
        win_target, score_target, outcome = (F.pad(t, (0, pad)) for t in (win_target, score_target, outcome))
    log_v = torch.log_softmax(win_logits.float(), dim=1)
    ce = -(win_target * log_v).sum(dim=1).mean()
    mse = F.mse_loss(score_pred.float(), score_target)
    outcome_ce = -(outcome * log_v).sum(dim=1).mean()
    hit = outcome.gather(1, log_v.argmax(dim=1, keepdim=True)).mean()
    return {"ce": ce, "mse": mse, "outcome_ce": outcome_ce.detach(), "winner_hit": hit.detach()}


def aux_losses(model, seat_input, aux_seat, aux_global) -> dict:
    """MSE of the per-seat auxiliary head to ``aux_seat`` (B, players, K) and the
    floated head's BCE to ``aux_global``, from the value heads' input ``seat_input``
    (the value rows, (B, P, F))."""
    import torch.nn.functional as F

    seat_pred, global_logits = model.aux_outputs(seat_input)
    players = aux_seat.shape[1]
    mse = F.mse_loss(seat_pred.float()[:, :players], aux_seat)
    bce = F.binary_cross_entropy_with_logits(global_logits.float(), aux_global)
    return {"total": mse + bce, "mse": mse.detach(), "bce": bce.detach()}


def search_losses(
    policy_logits,
    price_logits,
    anchor_logits,
    anchor_price_logits,
    batch: dict,
    temperature: float,
    kl_coef: float,
    price_weight: float = 1.0,
    surprise_weight: float = 0.0,
    surprise_cap: float = 8.0,
) -> dict:
    """Per-row sums of the search-target losses for a batch.

    ``ce``: -sum over the legal moves of target * log_softmax(logits / T); its
    minimum, at policy = target, is the target's entropy (``search_kl`` = ce -
    entropy is KL(target || policy), the policy's distance from the search). ``price_ce``: the same
    against the price cells at searched price decisions (the price head is not
    tempered, as in ``policy_gradient``). ``kl``: KL(learner || anchor) at T over
    the legal moves, plus the price cells at the price rows. ``total`` = sum of
    ``ce`` + ``price_weight`` * ``price_ce`` + ``kl_coef`` * ``kl`` over the rows
    (divide by the step's rows: microbatches then add up to the step's mean).

    With ``surprise_weight`` a > 0 (KataGo's policy surprise weighting) each row's
    ``ce`` in ``total`` weighs w = (1 - a) + a * s / mean(s), s its KL(target ||
    policy) (no gradient through w), capped at ``surprise_cap`` and rescaled to mean
    1 over the batch; ``surprise_*`` sums describe the weights."""
    import torch

    from rl18xx.agent.alphazero.policy_gradient import _masked_log_softmax

    legal = batch["legal"]
    n = legal.shape[0]
    log_p = _masked_log_softmax(policy_logits / temperature, legal)
    anchor_log_p = _masked_log_softmax(anchor_logits / temperature, legal)
    rows, cols, vals = batch["target_rows"], batch["target_cols"], batch["target_vals"]
    ce = torch.zeros(n, device=log_p.device).index_add(0, rows, -vals * log_p[rows, cols])
    p = log_p.exp() * legal
    kl = (p * (log_p - anchor_log_p)).sum(dim=1)
    entropy = -(p * log_p).sum(dim=1)
    top1 = (log_p.masked_fill(~legal, float("-inf")).argmax(dim=1) == batch["target_best"]).float()

    price_ce = torch.zeros((), device=log_p.device)
    price_kl = torch.zeros((), device=log_p.device)
    if price_logits is not None and batch["price_rows"].numel():
        prow, slots, masks = batch["price_rows"], batch["price_slots"], batch["price_masks"]
        cell_log_p = _masked_log_softmax(price_logits[prow, slots], masks)
        anchor_cell_log_p = _masked_log_softmax(anchor_price_logits[prow, slots], masks)
        price_ce = -(batch["price_targets"] * cell_log_p).sum()
        cell_kl = (cell_log_p.exp() * masks * (cell_log_p - anchor_cell_log_p)).sum(dim=1)
        kl = kl.index_add(0, prow, cell_kl)
        price_kl = cell_kl.sum()
    surprise = (ce - batch["target_entropy"]).detach()
    out = {}
    if surprise_weight > 0:
        weights, capped = surprise_weights(surprise, surprise_weight, surprise_cap)
        policy_term = (weights * ce).sum()
        disagree = top1 == 0
        top = max(1, int(round(0.1 * n)))
        out.update(
            surprise_weight_sq=(weights**2).sum(),
            surprise_weight_max=weights.max(),
            surprise_capped=capped,
            surprise_top10_weight=weights.topk(top).values.sum(),
            surprise_disagree_weight=weights[disagree].sum(),
            surprise_disagree_rows=disagree.float().sum(),
        )
    else:
        policy_term = ce.sum()
    total = policy_term + price_weight * price_ce + kl_coef * kl.sum()
    out.update(
        {
            "total": total,
            "ce": ce.sum().detach(),
            "search_kl": surprise.sum(),
            "target_entropy": batch["target_entropy"].sum().detach(),
            "kl_anchor": kl.sum().detach(),
            "entropy": entropy.sum().detach(),
            "top1_agreement": top1.sum().detach(),
            "price_ce": price_ce.detach(),
            "price_kl_anchor": price_kl.detach(),
        }
    )
    return {k: (v if k == "total" else v.detach()) for k, v in out.items()}


def surprise_weights(surprise, a: float, cap: float) -> tuple:
    """``(weights, rows capped)``: KataGo's policy surprise weights (1 - a) + a * s /
    mean(s) for the rows' surprises ``s`` = KL(target || policy) (clamped at 0),
    capped at ``cap``, then rescaled to mean 1 (the step's size is unchanged)."""
    import torch

    s = surprise.detach().float().clamp_min(0.0)
    mean = s.mean()
    if float(mean) <= 1e-12:
        return torch.ones_like(s), s.new_zeros(())
    raw = (1.0 - a) + a * s / mean
    capped = (raw > cap).float().sum()
    w = raw.clamp(max=cap)
    return w * (len(w) / w.sum()), capped


# ---------------------------------------------------------------------------
# Worker side: generation
# ---------------------------------------------------------------------------


class _TreeHandle:
    """What ``lockstep.TreeSearch`` takes from a player: its config and its tree."""

    def __init__(self, config, tree):
        self.config = config
        self._rust_player = tree


@dataclass
class _Game:
    game: Any  # RustGameAdapter
    uid: str
    start: str
    seat_servers: list
    decisions: int = 0
    searched: int = 0
    termination: Optional[str] = None
    price_row: Optional[np.ndarray] = None  # policy_selfplay._advance reads it
    rows: list = field(default_factory=list)
    search_values: list = field(default_factory=list)  # every search's root value, in order
    need: Any = None
    pending: Any = None
    # ``reuse_tree``: the game's search tree, kept across decisions (advanced
    # along every move played), the network version it was searched with, and
    # the root-log length up to which it is known to match the game.
    tree: Any = None
    tree_version: Optional[int] = None
    tree_synced: int = 0


def search_config(settings: dict, client=None):
    """The ``SelfPlayConfig`` of a generator's trees (the inference goes
    through the lockstep searcher; ``client`` only satisfies the player)."""
    from rl18xx.agent.alphazero.config import SelfPlayConfig

    gumbel = settings["target"] == "gumbel"
    return SelfPlayConfig(
        network=None,
        use_inference_server=client is not None,
        inference_client=client,
        num_readouts=settings["n_full"],
        min_readouts=settings["n_full"],
        parallel_readouts=settings["parallel_readouts"],
        c_puct_init=settings["c_puct_init"],
        c_puct_by_round={},
        mcts_mean_q=settings["mean_q"],
        fpu_reduction=settings["fpu_reduction"],
        dedupe_leaves=settings["dedupe_leaves"],
        prior_temperature=settings["prior_temperature"],
        gumbel_k=settings["gumbel_k"] if gumbel else 0,
        gumbel_c_visit=settings["gumbel_c_visit"],
        gumbel_c_scale=settings["gumbel_c_scale"],
        gumbel_noise_scale=settings["gumbel_noise_scale"],
        dirichlet_noise_weight=0.0,
        enable_resign=False,
        noresign_holdout_rate=0.0,
        softpick_move_cutoff=0,
        max_game_length=settings["max_decisions"],
        player_count_distribution={NUM_PLAYERS: 1.0},
    )


def deal_seats(settings: dict, rng: random.Random) -> list:
    """The server each seat's moves come from. All the learner's, unless
    ``settings["opponent_servers"]`` names servers (the league hook): then the
    learner holds ``learner_seats`` random seats (default 2) and each other seat
    plays one of those servers' policy (no rows)."""
    opponents = settings.get("opponent_servers") or []
    if not opponents:
        return ["learner"] * NUM_PLAYERS
    learner = set(rng.sample(range(NUM_PLAYERS), int(settings.get("learner_seats", 2))))
    return ["learner" if seat in learner else rng.choice(opponents) for seat in range(NUM_PLAYERS)]


class Generator:
    """One worker's games: ``games_per_worker`` at once, searched in lockstep;
    :meth:`run` plays for a while and returns the games that ended (with their
    rows). Games in progress carry over to the next call."""

    def __init__(self, settings: dict, evaluators: dict):
        from rl18xx.agent.alphazero.lockstep import LockstepSearcher

        self.settings = settings
        self.searcher = LockstepSearcher(evaluators, dedupe_maps=bool(settings.get("dedupe_maps", False)))
        self.games: dict = {}
        self.config = search_config(settings)
        self.rng = np.random.default_rng()
        self.py_rng = random.Random()
        self.next_slot = 0
        self.counts = Counter()

    # -- games -----------------------------------------------------------------

    def _new_game(self) -> _Game:
        from rl18xx.agent.alphazero.start_positions import AUCTION_START, apply_actions, new_game, sample_start_position

        s = self.settings
        for _ in range(100):
            start = sample_start_position(
                NUM_PLAYERS,
                s["start_positions"],
                s["random_start_fraction"],
                rng=self.py_rng,
                midgame_path=s.get("midgame_positions"),
                midgame_fraction=s.get("midgame_fraction", 0.0),
                auction_fraction=s.get("auction_fraction", 0.0),
            )
            try:
                game = apply_actions(new_game(NUM_PLAYERS), start.actions)
            except RuntimeError:
                self.counts["skipped_starts"] += 1  # recorded before an engine rules change
                continue
            if start.label == AUCTION_START:
                game._game.set_auction_unlock(True)
            return _Game(game=game, uid=uuid.uuid4().hex, start=start.label, seat_servers=deal_seats(s, self.py_rng))
        raise RuntimeError("no start position replays")

    def _tree(self, game):
        """A search tree at ``game`` (an ``engine_rs.BaseGame``, cloned), set up as
        ``RustMCTSPlayer.initialize_game`` sets its tree up, wrapped for
        ``lockstep.TreeSearch`` (which reads only the config and the tree)."""
        import engine_rs

        c = self.config
        tree = engine_rs.RustMCTSPlayer(game, c.pw_c, c.pw_alpha, c.min_price_children, c.price_explore_eps)
        tree.set_search_config(c.c_puct_init, c.c_puct_base, {}, c.max_engine_actions)
        if c.mcts_mean_q:
            tree.set_q_config(True, c.fpu_reduction)
        if c.gumbel_k > 0:
            tree.set_gumbel_config(c.gumbel_k, c.gumbel_c_visit, c.gumbel_c_scale, c.gumbel_noise_scale)
        return _TreeHandle(c, tree)

    def _next_need(self, slot: int, g: _Game):
        """Play forced moves; return the next decision's need (a TreeSearch or a
        PolicyRequest), or None when the game is over."""
        from rl18xx.agent.alphazero.lockstep import PolicyRequest, TreeSearch
        from rl18xx.agent.alphazero.policy_selfplay import _advance

        s = self.settings
        legal = _advance(g, s, self.rng)
        if legal is None:
            return None
        rs = g.game._game
        if len(legal) == 1:
            self.counts["single_move_decisions"] += 1  # an open price: the price head plays it
        mover = rs.encode_canonical_py()  # (state, nodes, round type, rotation, players)
        server = g.seat_servers[int(mover[3])]
        if server == "learner" and len(legal) > 1 and self.rng.random() < s["p_full"]:
            g.pending = mover
            return self._search(g, rs, s["n_full"], server)
        if server == "learner" and len(legal) > 1 and s.get("n_fast", 0) > 0:
            g.pending = None  # a fast search: played, no row
            return self._search(g, rs, s["n_fast"], server)
        g.pending = s["learner_temperature"] if server == "learner" else s.get("opponent_temperature", 1.0)
        return PolicyRequest(rs, legal, key=server)

    def _search(self, g: _Game, rs, readouts: int, key):
        """The decision's search: on the game's kept tree with ``reuse_tree``
        (when it still matches the position and was searched by the current
        network version), else on a fresh tree."""
        from rl18xx.agent.alphazero.lockstep import TreeSearch

        s = self.settings
        reuse = s.get("reuse_tree", "off")
        if reuse == "off":
            return TreeSearch(self._tree(rs), readouts, key=key)
        handle = g.tree if self._tree_in_sync(g, rs) else None
        if handle is not None:
            tree = handle._rust_player
            if g.tree_version != s.get("version", 0):
                self.counts["reuse_dropped_version"] += 1  # searched by an older network
                handle = None
            elif reuse == "additional" and tree.n_at_root() > s.get("reuse_max_root_visits", 2 * s["n_full"]):
                self.counts["reuse_dropped_size"] += 1  # an "additional" tree keeps growing; start over
                handle = None
        if handle is None:
            handle = self._tree(rs)
            g.tree_version = s.get("version", 0)
            g.tree_synced = int(handle._rust_player.root_move_number())
        else:
            self.counts["reused_searches"] += 1
        g.tree = handle
        search = TreeSearch(handle, readouts, key=key, budget="total" if reuse == "total" else "additional")
        self.counts["searches"] += 1
        self.counts["inherited_visits"] += search.inherited
        return search

    def _tree_in_sync(self, g: _Game, rs) -> bool:
        """Whether the game's kept tree is at the game's position (else it is
        dropped). The tree's forced chains apply every single-legal-move
        position, a price drawn from the price head where one is open; the game
        plays an open-price move as a decision, so there they part."""
        if g.tree is None:
            return False
        if not g.tree._rust_player.root_log_matches(rs, g.tree_synced):
            self.counts["reuse_dropped_mismatch"] += 1
            g.tree = None
            return False
        g.tree_synced = int(g.tree._rust_player.root_move_number())
        return True

    def _advance_tree(self, g: _Game, move: int, price) -> None:
        """Move the game's kept tree (at the position the move was played from)
        along it: its child for (move, price) becomes the root, matched to the
        game's position when next used (``_tree_in_sync``)."""
        if g.tree is None:
            return
        tree = g.tree._rust_player
        try:
            tree.advance_root_to(int(move), None if price is None else int(price))
        except Exception:  # not a legal move at the tree's root: forget the tree
            self.counts["reuse_dropped_error"] += 1
            g.tree = None

    def _answer(self, g: _Game) -> None:
        """Play the move the finished need gives (and keep a searched one's row)."""
        from rl18xx.agent.alphazero.lockstep import PolicyRequest
        from rl18xx.agent.alphazero.policy_gradient import choose_price

        s, need = self.settings, g.need
        rs = g.game._game
        if isinstance(need, PolicyRequest):
            p = np.asarray(need.priors, dtype=np.float64)
            temperature = float(g.pending)
            if temperature != 1.0:
                p = np.power(np.clip(p, 1e-12, None), 1.0 / temperature)
            p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
            move = int(need.legal[int(self.rng.choice(len(p), p=p))])
            g.price_row = need.price_row
            price, _, _ = choose_price(g.game, move, need.price_row, self.rng, 0.0)
            in_sync = self._tree_in_sync(g, rs)
            rs.apply_action_index(move, price)
            if in_sync:
                self._advance_tree(g, move, price)
            self.counts["fast_decisions"] += 1
        else:
            tree = need.tree
            self.counts["final_visits"] += float(tree.n_at_root())
            legal = np.asarray(tree.legal_action_indices_at_root(), dtype=np.int64)
            if s["target"] == "gumbel":
                target = np.asarray(tree.gumbel_policy_target(), dtype=np.float64)[legal]
                move = int(tree.gumbel_action())
            else:
                visits = np.asarray(tree.child_n_at_root(), dtype=np.float64)[legal]
                move = self._visit_move(tree, legal, visits, g.decisions)
                target = visits
                if s.get("visit_target", "pruned") == "pruned":
                    target = np.asarray(tree.pruned_child_n_at_root(), dtype=np.float64)[legal]
            price, price_target = self._searched_price(tree, rs, move)
            g.search_values.append(root_value(tree))  # the n-step value targets' bootstrap (before the tree moves on)
            # The move goes straight onto the game (the most-visited price, as
            # advance_root would commit); a kept tree follows it.
            rs.apply_action_index(move, price)
            self._advance_tree(g, move, price)
            if g.pending is None:  # a fast search
                self.counts["fast_search_decisions"] += 1
                g.decisions += 1
                return
            target = sharpen_target(target, legal, move, float(s.get("target_temperature", 1.0)))
            state, nodes, _, rotation, _ = g.pending
            g.rows.append(
                SearchRow(
                    game_state=state,
                    node_features=nodes,
                    rotation=int(rotation),
                    legal=legal.astype(np.int32),
                    target=target.astype(np.float32),
                    price=price_target,
                    version=int(s.get("version", 0)),
                    root_value=g.search_values[-1],
                    search_index=len(g.search_values) - 1,
                )
            )
            g.searched += 1
            self.counts["searched_decisions"] += 1
        g.decisions += 1

    def _searched_price(self, tree, rs, move: int) -> tuple:
        """``(price to play, price target or None)`` for a searched move: a
        price-head slot with an open range plays its most-visited price
        grandchild and is trained toward their visits by cell; a fixed price is
        the engine's; a categorical move has none."""
        from rl18xx.agent.alphazero.policy_gradient import choose_price

        price_range = rs.price_range_for_index(move)
        if price_range is None:
            return None, None
        lo, hi = price_range
        slot = rs.price_head_slot_for_index(move)
        if lo == hi or slot is None:
            return int(lo), None
        visits = {p: n for p, n in tree.price_grandchildren_at_root().get(move, {}).items() if n > 0}
        if not visits:
            # Never searched below (rare): a price from the cells, uniformly.
            self.counts["unsearched_prices"] += 1
            return choose_price(self.game_for(rs), move, None, self.rng, 0.0)[0], None
        total = float(sum(visits.values()))
        slot_index, plo, phi = slot
        targets = [(slot_index, float(p), n / total, float(plo), float(phi)) for p, n in visits.items()]
        return int(tree.most_visited_price_for_slot(move)), price_cell_target(targets)

    @staticmethod
    def game_for(rs):
        from rl18xx.rust_adapter import RustGameAdapter

        return RustGameAdapter(rs)

    def _visit_move(self, tree, legal: np.ndarray, visits: np.ndarray, decisions: int = 0) -> int:
        """The searched move to play: the most visited, or -- with ``full_move_temperature`` t > 0,
        while the game has made fewer than ``explore_decisions`` decisions (0: always) -- one drawn
        with weight N ** (1 / t)."""
        t = float(self.settings.get("full_move_temperature", 0.0))
        k = int(self.settings.get("explore_decisions", 0) or 0)
        if t <= 0 or visits.sum() <= 0 or (k > 0 and decisions >= k):
            return int(tree.pick_best_action())
        self.counts["explored_moves"] += 1
        w = np.power(visits, 1.0 / t)
        return int(legal[int(self.rng.choice(len(legal), p=w / w.sum()))])

    def _finish(self, g: _Game) -> dict:
        """The game's record: its rows with the outcome filled in."""
        from rl18xx.agent.alphazero.policy_gradient import outcome, parse_reward, placement_points
        from rl18xx.agent.alphazero.self_play import _compute_net_worth

        s = self.settings
        win_share, fractions = outcome(g.game)
        net_worth = _compute_net_worth(g.game)
        worth = np.array([float(net_worth[pid]) for pid in sorted(net_worth)])
        kind, points = parse_reward(s.get("reward", "placement"))
        score = None
        if kind == "placement":
            score = placement_points(worth, points or (1.0, 0.5, 0.2, 0.0)).astype(np.float32)
        k = int(s.get("critic_positions_per_game", 0))
        chosen = range(len(g.rows)) if k <= 0 or k >= len(g.rows) else self.rng.choice(len(g.rows), k, replace=False)
        for i in chosen:
            g.rows[int(i)].critic = True
        aux_seat, aux_global = aux_outcomes(g.game, fractions)
        nstep, weight = int(s.get("value_nstep", 0)), float(s.get("value_bootstrap_weight", 0.5))
        lam = s.get("value_lambda")
        returns = lambda_returns(share_of_winners(fractions), g.search_values, float(lam)) if lam is not None else None
        for row in g.rows:
            row.fractions = fractions
            row.score_target = score
            if returns is not None:
                row.win_target = lambda_win_target(share_of_winners(fractions), returns, row.search_index, weight)
            else:
                row.win_target = nstep_win_target(
                    share_of_winners(fractions), g.search_values, row.search_index, nstep, weight
                )
            row.aux_seat, row.aux_global = aux_seat, aux_global
        self.counts["games"] += 1
        return {
            "uid": g.uid,
            "start_kind": g.start.split(":")[0] if g.start else None,
            "termination": g.termination,
            "decisions": g.decisions,
            "searched": g.searched,
            "seat_servers": list(g.seat_servers),
            "win_share": win_share.tolist(),
            "rows": g.rows,
            # How well the search's root values called this game (win CE), on its rows.
            "root_value_ce": root_value_ce(g.rows, share_of_winners(fractions)),
        }

    # -- the loop ----------------------------------------------------------------

    def _step_game(self, slot: int, g: _Game) -> bool:
        """Advance ``g`` to its next need; False if it ended or failed (dropped)."""
        try:
            g.need = self._next_need(slot, g)
        except Exception as e:  # an engine error in one game must not end the worker
            LOGGER.warning(f"search self-play: game {g.uid} dropped: {e!r}")
            self.counts["dropped_games"] += 1
            g.need, g.termination, g.rows = None, "error", []
            return False
        return g.need is not None

    def run(self, seconds: float, settings: Optional[dict] = None) -> dict:
        from rl18xx.agent.alphazero.lockstep import PolicyRequest

        if settings is not None:
            self.settings = settings
            self.searcher.dedupe_maps = bool(settings.get("dedupe_maps", False))
        started = time.time()
        stats_before = dict(vars(self.searcher.stats))
        finished = []
        concurrency = int(self.settings["games_per_worker"])
        while time.time() - started < seconds:
            while len(self.games) < concurrency:
                slot, self.next_slot = self.next_slot, self.next_slot + 1
                g = self._new_game()
                self.games[slot] = g
                self._step_game(slot, g)
            for slot in [k for k, g in self.games.items() if g.need is None]:
                g = self.games.pop(slot)
                if g.termination != "error":
                    finished.append(self._finish(g))
            if not self.games:
                continue
            needs = [g.need for g in self.games.values()]
            self.searcher.round(
                [n for n in needs if not isinstance(n, PolicyRequest)],
                [n for n in needs if isinstance(n, PolicyRequest)],
            )
            for slot, g in self.games.items():
                if g.need.done:
                    try:
                        self._answer(g)
                    except Exception as e:
                        LOGGER.warning(f"search self-play: game {g.uid} dropped: {e!r}")
                        self.counts["dropped_games"] += 1
                        g.need, g.termination, g.rows = None, "error", []
                        continue
                    self._step_game(slot, g)
        after = vars(self.searcher.stats)
        stats = {
            k: after[k] - stats_before.get(k, 0) for k in ("rounds", "requests", "leaves", "policy_states", "maps")
        }
        stats.update(self.counts)
        stats["seconds"] = time.time() - started
        stats["games_in_progress"] = len(self.games)
        self.counts = Counter()
        return {"games": finished, "stats": stats}


_CLIENTS: dict = {}
_GENERATOR: Optional[Generator] = None


def worker_init(server_queues: dict):
    """Pool initializer: one slot on every inference server."""
    import os

    import torch

    from rl18xx.agent.alphazero.inference_server import InferenceClient

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    for name, (request_q, reply_qs, ticket_q) in server_queues.items():
        worker_id = ticket_q.get()
        _CLIENTS[name] = InferenceClient(request_q=request_q, reply_q=reply_qs[worker_id], worker_id=worker_id)
    logging.getLogger().setLevel(logging.WARNING)


def generate(settings: dict, seconds: float) -> dict:
    """Pool task: play this worker's games for ``seconds`` (they continue in the
    next task) and return the ones that ended."""
    global _GENERATOR
    from rl18xx.agent.alphazero.lockstep import ClientEvaluator

    if _GENERATOR is None:
        _GENERATOR = Generator(settings, {name: ClientEvaluator(c) for name, c in _CLIENTS.items()})
    return _GENERATOR.run(seconds, settings)


# ---------------------------------------------------------------------------
# Learner side
# ---------------------------------------------------------------------------


class ReplayWindow:
    """The latest ``capacity`` rows (a ring buffer, uniform sampling)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.rows: list = []
        self.next = 0

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, row) -> None:
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.next] = row
            self.next = (self.next + 1) % self.capacity

    def sample(self, n: int, rng: np.random.Generator) -> list:
        idx = rng.choice(len(self.rows), size=min(n, len(self.rows)), replace=False)
        return [self.rows[i] for i in idx]


def _value_terms(net, seat_input, win_logits, score_pred, batch, cfg) -> tuple:
    """``(loss, stats)`` of the value network on a microbatch's value rows: the win CE
    (``value_loss_weight`` with a shared network) + ``score_loss_weight`` * score MSE +
    ``aux_loss_weight`` * the auxiliary heads' losses (when the batch has their targets)."""
    score_target = batch["score_target"] if batch["score_target"] is not None else batch["fractions"]
    v = value_losses(win_logits, score_pred, batch["win_target"], score_target, batch["outcome_win"])
    win_weight = cfg.value_loss_weight if cfg.shared else 1.0
    loss = win_weight * v["ce"] + cfg.score_loss_weight * v["mse"]
    stats = {
        "critic_ce": v["ce"],
        "critic_outcome_ce": v["outcome_ce"],
        "critic_mse": v["mse"],
        "critic_winner_hit": v["winner_hit"],
    }
    if cfg.aux_loss_weight > 0 and batch["aux_seat"] is not None and seat_input is not None:
        aux = aux_losses(net, seat_input, batch["aux_seat"], batch["aux_global"])
        loss = loss + cfg.aux_loss_weight * aux["total"]
        stats.update(aux_mse=aux["mse"], aux_bce=aux["bce"])
    return loss, stats


def _update(learner, critic, anchor, learner_opt, critic_opt, window: ReplayWindow, cfg, kl_coef, device, rng) -> dict:
    """``cfg.steps_per_update`` optimizer steps on minibatches from the window:
    the policy toward the search targets (KL to the anchor), the value network on
    the sampled rows still flagged for it (each once) -- the critic, or with
    ``critic`` None (``cfg.shared``) the learner's own value heads from the same
    forward as its policy. Eval mode, as in ``policy_gradient._update`` (dropout
    would make the trained policy a different one from the playing one)."""
    import torch

    shared = critic is None
    value_net = learner if shared else critic
    aux = cfg.aux_loss_weight > 0
    learner.eval()
    if not shared:
        critic.eval()
    learner.value_stop_grad = bool(shared and cfg.value_stop_grad)
    temperature = float(cfg.learner_temperature)
    sums, rows_seen, price_rows, critic_rows, steps, ages = Counter(), 0, 0, 0, 0, []
    micro = min(cfg.microbatch, cfg.minibatch)
    for _ in range(cfg.steps_per_update):
        rows = window.sample(cfg.minibatch, rng)
        if not rows:
            break
        learner_opt.zero_grad(set_to_none=True)
        if critic_opt is not None:
            critic_opt.zero_grad(set_to_none=True)
        step_critic = [r for r in rows if r.critic and r.fractions is not None]
        n_step = len(rows)
        n_price = sum(1 for r in rows if r.price is not None)
        for start in range(0, n_step, micro):
            chunk = rows[start : start + micro]
            batch = search_batch(chunk, device)
            tiles = batch["lay_tile_rows"]
            idx = batch["critic_rows"]
            with torch.no_grad():
                anchor_logits, _, _, _ = anchor(batch["game_state"], batch["nodes"], lay_tile_rows=tiles)
                anchor_price = anchor.last_price_components["price_logits"]
            store, hook = learner.capture_value_input() if shared and aux else ({}, None)
            try:
                logits, win_logits, score_pred, _ = learner(batch["game_state"], batch["nodes"], lay_tile_rows=tiles)
            finally:
                if hook is not None:
                    hook.remove()
            price = learner.last_price_components["price_logits"]
            losses = search_losses(
                logits,
                price,
                anchor_logits,
                anchor_price,
                batch,
                temperature,
                kl_coef,
                cfg.price_loss_weight,
                cfg.surprise_weight,
                cfg.surprise_cap,
            )
            if not torch.isfinite(losses["total"]):
                values = {k: float(v) for k, v in losses.items()}
                raise RuntimeError(f"search learning diverged: non-finite loss ({values})")
            loss = losses["total"] / n_step
            for key, value in losses.items():
                if key == "surprise_weight_max":
                    sums[key] = max(sums[key], float(value))
                elif key != "total":
                    sums[key] += float(value)
            if idx.numel():
                share = idx.numel() / len(step_critic)
                if shared:
                    seat_input = store["seat_input"][idx] if aux else None
                    value_loss, vstats = _value_terms(learner, seat_input, win_logits[idx], score_pred[idx], batch, cfg)
                    loss = loss + value_loss * share
                else:
                    store, hook = critic.capture_value_input() if aux else ({}, None)
                    try:
                        _, c_win, c_score, _ = critic(batch["game_state"][idx], batch["nodes"][idx], value_only=True)
                    finally:
                        if hook is not None:
                            hook.remove()
                    value_loss, vstats = _value_terms(critic, store.get("seat_input"), c_win, c_score, batch, cfg)
                    value_loss = value_loss * share
                    if not torch.isfinite(value_loss):
                        raise RuntimeError("search learning: the critic diverged (non-finite loss)")
                    value_loss.backward()
                if not torch.isfinite(value_loss):
                    raise RuntimeError("search learning: the value loss is not finite")
                k = idx.numel()
                for key, value in vstats.items():
                    sums[key] += float(value.detach()) * k
            loss.backward()
        # Each optimizer group is clipped on its own: with a shared network the value
        # heads' gradient (at critic_lr) must not shrink the trunk and policy's step.
        norms = [torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0) for group in learner_opt.param_groups]
        norm = norms[0]
        if not all(torch.isfinite(n) for n in norms):
            raise RuntimeError("search learning diverged: non-finite gradient norm")
        learner_opt.step()
        if len(norms) > 1:
            sums["value_grad_norm"] += float(norms[1])
        if step_critic:
            if not shared:
                cnorm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                if not torch.isfinite(cnorm):
                    raise RuntimeError("search learning: the critic diverged (non-finite gradient norm)")
                critic_opt.step()
            for r in step_critic:
                r.critic = False  # each position trains the value once
        sums["grad_norm"] += float(norm)
        rows_seen += n_step
        price_rows += n_price
        critic_rows += len(step_critic)
        ages.extend(r.version for r in rows)
        steps += 1
    learner.value_stop_grad = False
    per_row = ("ce", "search_kl", "target_entropy", "kl_anchor", "entropy", "top1_agreement")
    stats = {key: sums[key] / max(rows_seen, 1) for key in per_row}
    stats.update({key: sums[key] / max(price_rows, 1) for key in ("price_ce", "price_kl_anchor")})
    value_keys = ["critic_ce", "critic_outcome_ce", "critic_mse", "critic_winner_hit"]
    value_keys += ["aux_mse", "aux_bce"] if aux else []
    stats.update({key: sums[key] / max(critic_rows, 1) for key in value_keys})
    if cfg.surprise_weight > 0:
        n, w_sq = max(rows_seen, 1), sums["surprise_weight_sq"]
        disagree = sums["surprise_disagree_rows"]
        stats.update(
            # (sum w)^2 / (n sum w^2) with mean-1 weights: the share of the rows' worth of signal.
            surprise_ess=n / w_sq if w_sq > 0 else float("nan"),
            surprise_weight_max=sums["surprise_weight_max"],
            surprise_capped_share=sums["surprise_capped"] / n,
            surprise_top10_weight_share=sums["surprise_top10_weight"] / n,
            surprise_disagree_row_share=disagree / n,
            surprise_disagree_weight_share=sums["surprise_disagree_weight"] / n,
        )
    stats.update(
        grad_norm=sums["grad_norm"] / max(steps, 1),
        **({"value_grad_norm": sums["value_grad_norm"] / max(steps, 1)} if shared else {}),
        optimizer_steps=steps,
        rows_trained=rows_seen,
        price_rows_trained=price_rows,
        critic_rows_trained=critic_rows,
        row_version_mean=float(np.mean(ages)) if ages else float("nan"),
    )
    return stats


def worker_settings(cfg: SearchLearningConfig, version: int) -> dict:
    return {
        "version": version,
        "target": cfg.target,
        "p_full": cfg.p_full,
        "n_full": cfg.n_full,
        "n_fast": cfg.n_fast,
        "learner_temperature": cfg.learner_temperature,
        "prior_temperature": cfg.prior_temperature,
        "c_puct_init": cfg.c_puct_init,
        "mean_q": cfg.mean_q,
        "fpu_reduction": cfg.fpu_reduction,
        "dedupe_leaves": cfg.dedupe_leaves,
        "parallel_readouts": cfg.parallel_readouts,
        "gumbel_k": cfg.gumbel_k,
        "gumbel_c_visit": cfg.gumbel_c_visit,
        "gumbel_c_scale": cfg.gumbel_c_scale,
        "gumbel_noise_scale": cfg.gumbel_noise_scale,
        "full_move_temperature": cfg.full_move_temperature,
        "explore_decisions": cfg.explore_decisions,
        "visit_target": cfg.visit_target,
        "target_temperature": cfg.target_temperature,
        "games_per_worker": cfg.games_per_worker,
        "max_decisions": cfg.max_decisions,
        "price_eps": 0.0,
        "start_positions": cfg.start_positions,
        "random_start_fraction": cfg.random_start_fraction,
        "midgame_positions": cfg.midgame_positions if cfg.midgame_fraction > 0 else None,
        "midgame_fraction": cfg.midgame_fraction,
        "auction_fraction": cfg.auction_fraction,
        "critic_positions_per_game": cfg.critic_positions_per_game,
        "reward": cfg.reward,
        "opponent_servers": [],
        "value_nstep": cfg.value_nstep,
        "value_bootstrap_weight": cfg.value_bootstrap_weight,
        "value_lambda": cfg.value_lambda,
        "reuse_tree": cfg.reuse_tree,
        "reuse_max_root_visits": cfg.reuse_max_root_visits or 2 * cfg.n_full,
        "dedupe_maps": cfg.dedupe_maps,
    }


def _validate(cfg: SearchLearningConfig) -> None:
    from rl18xx.agent.alphazero.composite_model import parse_value_head
    from rl18xx.agent.alphazero.policy_gradient import parse_reward

    if cfg.target not in TARGETS:
        raise SystemExit(f"--target must be one of {TARGETS}, got {cfg.target!r}")
    if cfg.visit_target not in ("pruned", "raw"):
        raise SystemExit(f"--visit-target must be pruned or raw, got {cfg.visit_target!r}")
    if not 0.0 <= cfg.p_full <= 1.0:
        raise SystemExit("--p-full must be in [0, 1]")
    try:
        parse_value_head(cfg.value_head)
    except ValueError as e:
        raise SystemExit(f"--value-head: {e}") from None
    if cfg.value_lambda is not None and (cfg.value_nstep > 0 or not 0.0 <= cfg.value_lambda <= 1.0):
        raise SystemExit("--value-lambda must be in [0, 1] and excludes --value-nstep")
    if cfg.value_nstep < 0 or not 0.0 <= cfg.value_bootstrap_weight <= 1.0:
        raise SystemExit("--value-nstep must be >= 0 and --value-bootstrap-weight in [0, 1]")
    if not 0.0 <= cfg.surprise_weight <= 1.0 or cfg.surprise_cap < 1.0:
        raise SystemExit("--surprise-weight must be in [0, 1] and --surprise-cap >= 1")
    if cfg.value_stop_grad and not cfg.shared:
        raise SystemExit("--value-stop-grad needs --shared (the critic has no policy to protect)")
    if cfg.reuse_tree not in ("off", "total", "additional"):
        raise SystemExit(f"--reuse-tree must be off, total or additional, got {cfg.reuse_tree!r}")
    kind, _ = parse_reward(cfg.reward)
    if kind not in ("win", "placement"):
        raise SystemExit("--reward must be win or placement (the critic's score-head target)")
    if cfg.anchor_every and cfg.anchor_every % cfg.snapshot_every:
        raise SystemExit("--anchor-every must be a multiple of --snapshot-every (the anchor's checkpoint is kept)")


def server_spec(cfg: SearchLearningConfig, learner_path, critic_path=None) -> str:
    """The generation server's checkpoint spec: ``<learner>+<critic>`` (the critic's
    ``value_head``), or with ``cfg.shared`` the learner alone (its own heads)."""
    suffix = "" if cfg.value_head == "win" else f"#{cfg.value_head}"
    if cfg.shared:
        return f"{learner_path}{suffix}"
    return f"{learner_path}+{critic_path}{suffix}"


def learner_param_groups(learner, cfg: SearchLearningConfig) -> list:
    """The learner's optimizer groups: with ``cfg.shared`` its value heads (and
    auxiliary heads) at ``critic_lr``, the rest at ``lr``; otherwise all at ``lr``."""
    if not cfg.shared:
        return [{"params": list(learner.parameters()), "lr": cfg.lr}]
    heads = [learner.win_loss_head, learner.score_head]
    heads += [
        h for h in (getattr(learner, "aux_seat_head", None), getattr(learner, "aux_global_head", None)) if h is not None
    ]
    value_ids = {id(p) for h in heads for p in h.parameters()}
    value = [p for p in learner.parameters() if id(p) in value_ids]
    rest = [p for p in learner.parameters() if id(p) not in value_ids]
    return [
        {"params": rest, "lr": cfg.lr, "name": "trunk_policy"},
        {"params": value, "lr": cfg.critic_lr, "name": "value_heads"},
    ]


def run(cfg: SearchLearningConfig, resume: Optional[str] = None) -> Path:
    """Run search learning; returns the run directory."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from rl18xx.agent.alphazero.inference_server import start_inference_server, wait_checking_servers
    from rl18xx.agent.alphazero.policy_gradient import _load, _save, load_server_model, use_tf32

    use_tf32()
    totals = Counter()
    if resume:
        run_dir = Path(resume)
        state = json.loads((run_dir / "state.json").read_text())
        # The run keeps its settings; its length, its generation layout, the
        # microbatch (the same step, other memory), the inference speed-ups
        # (the same outputs) and the exploration (full_move_temperature /
        # explore_decisions: a resume without them turns it off) can change.
        overrides = {
            k: getattr(cfg, k)
            for k in (
                "full_move_temperature",
                "explore_decisions",
                "max_updates",
                "microbatch",
                "workers",
                "games_per_worker",
                "task_seconds",
                "midgame_positions",
                "midgame_fraction",
                "auction_fraction",
                "dedupe_maps",
                "server_pipeline",
                "server_replicas",
            )
        }
        cfg = SearchLearningConfig(**{**state["config"], **overrides})
        update = int(state["snapshot_update"])
        kl_coef = float(state["kl_coef"])
        anchor_update = int(state.get("anchor_update", 0))
        totals.update(state.get("totals", {}))
        learner_path = run_dir / "learner" / f"{update}.pth"
        critic_path = None if cfg.shared else run_dir / "critic" / f"{update}.pth"
    else:
        if cfg.learner_temperature is None:
            cfg.learner_temperature = run_temperature(cfg.policy_checkpoint)
        if cfg.prior_temperature is None:
            cfg.prior_temperature = cfg.learner_temperature
        name = cfg.run_name or datetime.now().strftime("sl_%Y%m%d_%H%M%S")
        cfg.run_name = name
        run_dir = Path(cfg.out_dir) / name
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; pass --resume to continue it")
        run_dir.mkdir(parents=True)
        update, kl_coef, anchor_update = 0, cfg.kl_coef, 0
        learner_path = Path(cfg.policy_checkpoint)
        critic_path = None if cfg.shared else Path(cfg.value_checkpoint)
    _validate(cfg)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    LOGGER.info(f"Search-learning run {run_dir} from update {update}: {asdict(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_path = run_dir / "learner" / f"{anchor_update}.pth" if anchor_update else Path(cfg.policy_checkpoint)
    learner, anchor = _load(learner_path), _load(anchor_path)
    critic = _load(critic_path) if critic_path is not None else None
    anchor.eval()
    for param in anchor.parameters():
        param.requires_grad_(False)
    if cfg.aux_loss_weight > 0:  # training-only heads on the network that learns the value
        (critic if critic is not None else learner).add_aux_heads(AUX_SEAT_TARGETS, AUX_GLOBAL_TARGETS)
    learner_opt = torch.optim.AdamW(learner_param_groups(learner, cfg), weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0) if critic else None
    if resume and (run_dir / "optimizer.pth").exists():
        opt_state = torch.load(run_dir / "optimizer.pth", map_location="cpu")
        learner_opt.load_state_dict(opt_state["learner"])
        if critic_opt is not None:
            critic_opt.load_state_dict(opt_state["critic"])

    current_learner = run_dir / "learner" / f"{update}.pth"
    current_critic = None if cfg.shared else run_dir / "critic" / f"{update}.pth"
    if not current_learner.exists():
        _save(learner, current_learner)
        if critic is not None:
            _save(critic, current_critic)
    writer = SummaryWriter(str(Path("runs") / "alphazero_runs" / run_dir.name))
    history = (run_dir / "history.jsonl").open("a")

    def spec(policy, value) -> str:
        return server_spec(cfg, policy, value)

    handles = {
        "learner": start_inference_server(
            num_workers=cfg.workers,
            model_factory=load_server_model,
            checkpoint_path=spec(current_learner, current_critic),
            batch_size=cfg.inference_batch_size,
            autocast_device=None,
            pipeline=cfg.server_pipeline,
            replicas=cfg.server_replicas,
        )
    }
    queues = {name: (h.request_q, h.reply_qs, h.ticket_q) for name, h in handles.items()}
    stop_file = run_dir / "STOP"
    window = ReplayWindow(cfg.window_rows)
    rng = np.random.default_rng()
    gen, terminations, starts = Counter(), Counter(), Counter()
    new_rows, gen_started, run_started = 0, time.time(), time.time()
    try:
        # Spawned: the learner holds the networks (on the GPU), which a forked worker must not inherit.
        executor = ProcessPoolExecutor(
            max_workers=cfg.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init,
            initargs=(queues,),
        )
        try:
            futures = {
                executor.submit(generate, worker_settings(cfg, update), cfg.task_seconds) for _ in range(cfg.workers)
            }
            while update < cfg.max_updates and not stop_file.exists():
                done, futures = wait_checking_servers(futures, handles, executor, timeout=60)
                for future in done:
                    result = future.result()
                    for key, value in result["stats"].items():
                        if key not in ("games_in_progress",):
                            gen[key] += value
                    for record in result["games"]:
                        terminations[record["termination"]] += 1
                        starts[record["start_kind"]] += 1
                        gen["game_decisions"] += record["decisions"]
                        for row in record["rows"]:
                            window.add(row)
                        new_rows += len(record["rows"])
                        gen["rows"] += len(record["rows"])
                        ce_sum, ce_rows = record.get("root_value_ce", (0.0, 0))
                        gen["root_value_ce_sum"] += ce_sum
                        gen["root_value_rows"] += ce_rows
                    futures.add(executor.submit(generate, worker_settings(cfg, update), cfg.task_seconds))
                if new_rows < cfg.rows_per_update or len(window) < cfg.min_window_rows:
                    continue

                t0 = time.time()
                stats = _update(learner, critic, anchor, learner_opt, critic_opt, window, cfg, kl_coef, device, rng)
                train_seconds = time.time() - t0
                update += 1
                totals["rows_generated"] += new_rows
                totals["rows_trained"] += stats["rows_trained"]
                totals["games"] += gen["games"]
                stats["new_rows"] = new_rows
                stats["reuse_update"] = stats["rows_trained"] / max(new_rows, 1)
                stats["reuse_run"] = totals["rows_trained"] / max(totals["rows_generated"], 1)
                stats["row_age_mean"] = (update - 1) - stats.pop("row_version_mean")
                new_rows = 0

                previous = (current_learner,) if critic is None else (current_learner, current_critic)
                current_learner = run_dir / "learner" / f"{update}.pth"
                _save(learner, current_learner)
                if critic is not None:
                    current_critic = run_dir / "critic" / f"{update}.pth"
                    _save(critic, current_critic)
                reply = handles["learner"].reload(spec(current_learner, current_critic))
                if not (isinstance(reply, dict) and reply.get("ok")):
                    raise RuntimeError(f"the inference server could not load update {update}: {reply}")
                if cfg.anchor_every and update % cfg.anchor_every == 0:
                    anchor.load_state_dict(learner.state_dict(), strict=False)  # not the aux heads
                    anchor_update = update
                for path in previous:
                    if int(path.stem) % cfg.snapshot_every != 0 and path.parent.parent == run_dir and path.exists():
                        path.unlink()
                if cfg.kl_target is not None:
                    if stats["kl_anchor"] > 1.5 * cfg.kl_target:
                        kl_coef *= 1.5
                    elif stats["kl_anchor"] < cfg.kl_target / 1.5:
                        kl_coef = max(kl_coef / 1.5, cfg.kl_coef_min)
                if update % cfg.snapshot_every == 0:
                    torch.save(
                        {
                            "learner": learner_opt.state_dict(),
                            "critic": critic_opt.state_dict() if critic_opt else None,
                        },
                        run_dir / "optimizer.pth",
                    )
                    (run_dir / "state.json").write_text(
                        json.dumps(
                            {
                                "config": asdict(cfg),
                                "snapshot_update": update,
                                "kl_coef": kl_coef,
                                "anchor_update": anchor_update,
                                "totals": dict(totals),
                            },
                            indent=2,
                        )
                    )

                elapsed = time.time() - gen_started
                record = {
                    "update": update,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    **stats,
                    "kl_coef": kl_coef,
                    "anchor_update": anchor_update,
                    "window_rows": len(window),
                    "games": gen["games"],
                    "games_total": totals["games"],
                    "games_per_hour": gen["games"] / elapsed * 3600,
                    "searched_rows_per_hour": gen["rows"] / elapsed * 3600,
                    "decisions_per_second": (
                        gen["searched_decisions"] + gen["fast_decisions"] + gen["fast_search_decisions"]
                    )
                    / elapsed,
                    "evaluations_per_second": (gen["leaves"] + gen["policy_states"]) / elapsed,
                    "searched_share": (
                        gen["searched_decisions"]
                        / max(gen["searched_decisions"] + gen["fast_decisions"] + gen["fast_search_decisions"], 1)
                    ),
                    "explored_share": gen["explored_moves"]
                    / max(gen["searched_decisions"] + gen["fast_decisions"] + gen["fast_search_decisions"], 1),
                    "fast_search_share": gen["fast_search_decisions"]
                    / max(gen["searched_decisions"] + gen["fast_decisions"] + gen["fast_search_decisions"], 1),
                    "mean_decisions": gen["game_decisions"] / max(gen["games"], 1),
                    # The search's root values on the new games' rows vs their outcomes (win CE).
                    "search_value_ce": gen["root_value_ce_sum"] / max(gen["root_value_rows"], 1),
                    "evaluations_per_decision": (gen["leaves"] + gen["policy_states"])
                    / max(gen["searched_decisions"] + gen["fast_decisions"] + gen["fast_search_decisions"], 1),
                    # Tree reuse: searches that started on a kept subtree, and the share of
                    # the searches' final root visits they inherited.
                    "reused_search_share": gen["reused_searches"] / max(gen["searches"], 1),
                    "inherited_visit_share": gen["inherited_visits"] / max(gen["final_visits"], 1),
                    "reuse_dropped": {
                        k[len("reuse_dropped_") :]: gen[k] for k in list(gen) if k.startswith("reuse_dropped_")
                    },
                    # Map dedupe: node-feature rows sent per state.
                    "map_rows_per_state": gen["maps"] / max(gen["leaves"] + gen["policy_states"], 1),
                    "skipped_starts": gen["skipped_starts"],
                    "dropped_games": gen["dropped_games"],
                    "terminations": dict(terminations),
                    "starts": dict(starts),
                    "train_seconds": train_seconds,
                }
                gen, terminations, starts = Counter(), Counter(), Counter()
                gen_started = time.time()
                history.write(json.dumps(record) + "\n")
                history.flush()
                for key, value in record.items():
                    if isinstance(value, (int, float)) and key != "update":
                        writer.add_scalar(f"SL/{key}", value, update)
                LOGGER.info(
                    f"update {update}: ce {stats['ce']:.3f} (target entropy {stats['target_entropy']:.3f}, "
                    f"top-1 {stats['top1_agreement']:.3f}), price ce {stats['price_ce']:.3f}, "
                    f"kl_anchor {stats['kl_anchor']:.4f}, critic CE {stats['critic_ce']:.4f}, "
                    f"{record['games_per_hour']:.0f} games/h, {record['searched_rows_per_hour']:.0f} rows/h, "
                    f"reuse {stats['reuse_update']:.2f}, train {train_seconds:.0f}s"
                )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        history.close()
        writer.close()
        for handle in handles.values():
            handle.shutdown()
    return run_dir
