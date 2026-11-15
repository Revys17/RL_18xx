"""Policy-only policy gradient for the title-generic model (RL bridge milestone C2; ``docs/rl_model.md``
"Policy gradient for the generic model").

The legacy 1830 stage (``policy_gradient.py``, the recipe behind gen4 -> gen6) for any title with an RL layout,
on ``model_generic`` networks. What is the legacy code, imported (not copied): the league and PFSP per player
count (``CountLeague``, ``League``, ``league_for_run``, ``league_slots``, ``min_share`` / ``min_share_players``
pins, own snapshots joining the league), seat dealing (``_deal_seats``), the player-count schedule of
``--player-mix`` (``CountSchedule``, ``player_counts``, ``count_path``, per-count decision caps
``count_limits``), placement points (``placement_points``, ``default_placement_points``), reward specs,
the rolling scores (``_Scores``) and resume semantics (``resume_overrides``). What is generic:

- **Games** (:func:`play_generic_pg_games`, pool workers): starts from ``generic_play.replayed_start`` (1830:
  the legacy starts exactly -- human SR1, random auction endings, mid-game cuts, real-rules auctions); each
  pending position's legal rows come from ``generic_play.TitlePlay`` (the title's layout); positions are
  encoded together (``encode_batch``), split by the seat's server (``slice_positions`` +
  ``add_legal_inputs``) and sent to generic inference servers (``generic_inference``), which reply with the
  legal rows' log-probabilities, a price cell sampled per priced row and per-seat values. The worker samples
  the move at the seat's temperature (in log space), draws the price inside the cell for the move it plays,
  and keeps a reservoir of ``positions_per_seat`` learner decisions per seat: the position's token arrays
  (LZ4), its legal rows and ranges, the move, the probability it was sampled with (move at the learner's
  temperature + price cell), its seat and the player count.
- **Update** (:func:`_update`): PPO-clipped policy gradient on the legal rows only, the learner's policy being
  softmax(logits / ``learner_temperature``) (the map families' factorization included: the tempered row
  log-probabilities are ``segment_log_softmax(logits / T)``), a price-bearing move's probability including
  its price cell's; ``kl_coef`` * KL(learner || anchor) over the legal rows and the chosen row's price cells
  (adaptive with ``kl_target``; the anchor is the starting policy or, with ``anchor_every``, the learner every
  N updates); advantages ``reward - V(s)`` from the critic in one batched pass before the update
  (``gae_lambda`` 1, the legacy default); the critic (the same architecture used value-only, initialized from
  ``value_checkpoint`` -- by default the starting checkpoint's own value heads) trains on the same rows
  (win CE + score MSE on placement points, ``model_generic.value_losses``).
- **Rewards**: ``win`` (win share, ties split) or ``placement`` (placement points per player count, the
  1830 recipe's; ``placement:<1st>,<2nd>,...`` custom). The baseline is the critic's win head (win) or its
  score head (placement). The 1830 exploiter goals (``president:<CORP>``, ``rusher``, ``rust``, ``bankrupt``)
  are not ported.
- **Checkpoints**: ``model_generic.save_checkpoint`` files (title, layout fingerprint, encoder version) in
  ``<out_dir>/<run>/{learner,critic}/<update>.pth``; every ``snapshot_every``-th kept, with ``state.json`` and
  ``optimizer.pth`` (``--resume``); league members and fixed opponents must be generic checkpoints of the
  run's title (legacy 1830 checkpoints are refused: they need the legacy encoder and server).
- **Rules mix** (``rules_mix``; 1822's optional rules, ``titles.TitleSpec.rules_mix``): each game draws its rules
  combination by weight (``titles.draw_rules``; default the title's mix -- 1822: the human games' combinations --,
  base rules for every other title) and starts under it (``generic_play.replayed_start(optional_rules=)``: human /
  mid-game starts of that combination, fresh and random openings under it); the network reads the rules from the
  encoder's rule flags (encoder v3). PFSP and the league stay per player count; the history counts games,
  decisions and terminations per combination (``rules_<name>_*``).
- **All-pass openings** (the 1822 family; ``redraw_all_pass``, on by default): a game whose opening Stock Round
  ends with nothing bought, sold or bid ends by Ruby's rule with every seat tied (``generic_play.all_pass_opening``)
  and teaches nothing: it is dropped (no rows, no league credit, no score) and its slot starts a new game from a
  fresh draw (rules, start, seats; at most ``generic_play.MAX_ALL_PASS_REDRAWS`` in a row, then it is kept with
  termination ``all_pass``). The history counts them (``all_pass_redrawn``, ``p<n>_all_pass_redrawn``). The rule
  itself is unchanged (the user, 2026-11-14).
- **Saved games** (``save_game_every``): ``<run>/games/u<update>_<id>.json`` with the title, players, seed,
  optional rules (``settings.optional_rules`` and the combination's name), rule variants, layout version /
  fingerprint and encoder version under ``"rl18xx"``.

Progress is logged per update to ``<run>/history.jsonl`` (and TensorBoard ``runs/generic_pg/<run>``, ``GPG/*``):
the learner's score against the league as a multiple of a fair share (or vs the fixed opponent / pool), the
PPO and KL statistics, the critic's losses, games and decisions per second, starts and terminations, per-count
stats with a player mix. Touch ``<run>/STOP`` to end after the current update.

**Robustness mechanisms** (``docs/robustness_test_plan.md`` arms A1-A3; ``docs/rl_model.md`` "Robustness
mechanisms"; every one off by default, and off the run is exactly the C2 stage):

- **Tempered league members** (A1): a population entry's ``temperature`` makes the member play at its own
  temperature (its key in the league is ``<path>@<T>``, so one checkpoint can sit in the league at two
  temperatures); ``min_share_by_players`` pins a member's share per player count (``min_share`` /
  ``min_share_players`` as the legacy).
- **Reference KL** (A1, piKL-style): ``ref_kl_coef`` * KL(learner || reference) on top of the moving anchor's
  KL, the reference a fixed checkpoint (default the starting policy) at ``ref_temperature``.
- **Prioritized starts** (A2, ``start_priority``): prioritized level replay over the human SR1 and mid-game
  starts (``generic_play.StartPriorities``), scored from the learner's own rows; scores persist with the run.
- **Main-exploiter mode** (A3, ``exploiter_of``): every seat but the learner's is a frozen checkpoint (one
  league member, no own snapshots joining); with ``population_reload_every`` a main run picks up members added
  to (or dropped from) its population file while it runs (``scripts/launch/2026-11/bridge/exploiter_loop.py``).
- ``seed``: seeds the trainer, the servers and each worker task (task ``i`` of two runs with one seed draws the
  same starts and seats until their policies differ); runs stay asynchronous, not bit-reproducible.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import pickle
import random
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import lz4.frame
import numpy as np
import torch

from rl18xx.agent.alphazero import policy_gradient as pg

LOGGER = logging.getLogger(__name__)

NUM_PLAYERS = 4
REWARDS = ("win", "placement")


@dataclass
class GenericPGConfig:
    """Settings of a generic policy-gradient run (names and defaults as ``policy_gradient.PGConfig``'s where
    the setting exists there)."""

    #: The starting policy (a generic checkpoint, e.g. ``pretrain-generic``'s ``final.pth``).
    policy_checkpoint: str
    #: The critic's starting point (a generic checkpoint; None: the policy checkpoint, i.e. its value heads).
    value_checkpoint: Optional[str] = None
    title: str = "1830"
    out_dir: str = "model_checkpoints_generic_pg"
    run_name: Optional[str] = None
    num_players: int = NUM_PLAYERS
    player_mix: Optional[dict] = None
    #: ``{rules combination name: weight}`` (``titles.TitleSpec.parse_rules_mix``); None: the title's default mix
    #: (1822: the human games' combinations; others: base rules). A new run writes the resolved mix to its config.
    rules_mix: Optional[dict] = None
    max_decisions_by_players: Optional[dict] = None
    auction_stall_by_players: Optional[dict] = None
    # Generation
    workers: int = 48
    games_per_task: int = 32
    tasks_in_flight_per_worker: int = 2
    learner_seats: int = 2
    positions_per_seat: int = 32
    sl_opponent_fraction: float = 0.5
    opponent_checkpoint: Optional[str] = None
    seed_pool: bool = True
    anchor_every: int = 0
    population: Optional[str] = None
    opponent_slots: int = 3
    learner_seat_weights: Optional[dict] = None
    pfsp_power: float = 2.0
    slot_refresh_every: int = 2
    league_add_snapshots: bool = True
    league_snapshot_every: int = 50
    league_max_snapshots: int = 4
    learner_temperature: float = 1.0
    opponent_temperature: float = 1.0
    opponent_price_eps: float = 0.0
    max_decisions: int = 1000
    #: Start files (``{n}``: the player count); None: the title's (``titles.TitleSpec``) templates.
    start_positions: Optional[str] = None
    random_start_fraction: float = 0.2
    midgame_positions: Optional[str] = None
    midgame_fraction: float = 0.0
    auction_fraction: float = 0.0
    auction_unlock: bool = True
    auction_stall_decisions: int = 0
    #: The 1822 family: drop a game whose opening Stock Round ends with nothing bought, sold or bid (Ruby ends it,
    #: every seat tied) and play a fresh draw in its place (``generic_play.all_pass_opening``).
    redraw_all_pass: bool = True
    # Inference servers
    inference_batch_size: int = 512
    server_replicas: int = 1
    server_pipeline: bool = False
    #: Torch threads per server process (None: the process's default / OMP_NUM_THREADS).
    server_threads: Optional[int] = None
    #: Inference servers on the CPU even with a GPU (tests, CPU smoke runs).
    server_cpu: bool = False
    compress_requests: bool = True
    # Updates
    rows_per_update: int = 65536
    minibatch: int = 256
    microbatch: int = 256
    ppo_epochs: int = 1
    clip: float = 0.2
    lr: float = 1e-5
    critic_lr: float = 3e-5
    kl_coef: float = 0.02
    kl_target: Optional[float] = None
    kl_coef_min: float = 0.005
    entropy_coef: float = 0.0
    normalize_advantages: bool = False
    score_loss_weight: float = 1.0
    max_updates: int = 1000
    snapshot_every: int = 10
    pool_refresh_every: int = 5
    score_window: int = 2000
    save_game_every: int = 0
    reward: str = "win"
    #: Trainer device (None: cuda when available) and CPU threads.
    device: Optional[str] = None
    threads: Optional[int] = None
    # --- Robustness mechanisms (docs/rl_model.md "Robustness mechanisms"; all off by default) ---
    #: A1: weight of KL(learner || reference) over the legal rows and the played row's price cells (piKL-style,
    #: fixed, on top of the moving anchor's adaptive KL). 0: off (no reference network is loaded).
    ref_kl_coef: float = 0.0
    #: The reference policy (a generic checkpoint; None: ``policy_checkpoint``, i.e. the supervised start).
    ref_checkpoint: Optional[str] = None
    #: The reference's temperature in that KL (None: ``learner_temperature``, so the KL is 0 at the start; 1.0
    #: compares the learner as it plays with the human-imitation policy's own spread, piKL's form).
    ref_temperature: Optional[float] = None
    #: A2: prioritized level replay over the human SR1 and mid-game starts (``generic_play.StartPriorities``).
    start_priority: bool = False
    #: Per-row score (``generic_play.START_SCORES``): abs_adv = |reward - V(s)|, pos_adv = max(reward - V(s), 0).
    start_priority_score: str = "abs_adv"
    #: Share of prioritized-kind draws left to the uniform choice (unscored starts enter this way).
    start_priority_uniform: float = 0.3
    #: Weight of PLR's staleness term in the replay distribution.
    start_priority_staleness: float = 0.3
    #: Rank temperature: P_S proportional to (1 / rank) ** (1 / T).
    start_priority_temperature: float = 1.0
    #: Weight of an update's mean in a start's running score (1.0: replace, as PLR).
    start_priority_ema: float = 0.5
    #: A3: main-exploiter mode -- every non-learner seat is this frozen checkpoint (a one-member league, no own
    #: snapshots joining, one opponent server for every player count).
    exploiter_of: Optional[str] = None
    #: Every N updates re-read the population file: members added join (even rate), members dropped from it
    #: leave, pins follow the file; on a resume, members neither in the file nor own snapshots leave. 0: off (the
    #: legacy semantics: the file is read at the start and its new members join on a resume).
    population_reload_every: int = 0
    #: Seeds the trainer, the servers and each worker task (None: unseeded).
    seed: Optional[int] = None

    def __post_init__(self):
        for name in ("player_mix", "max_decisions_by_players", "auction_stall_by_players"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, {int(k): v for k, v in value.items()})
        if self.learner_seat_weights is not None:
            self.learner_seat_weights = {int(k): float(v) for k, v in self.learner_seat_weights.items()}


# What a resume may change (``policy_gradient.RESUME_SETTINGS`` plus the generic run's server settings).
RESUME_SETTINGS = tuple(s for s in pg.RESUME_SETTINGS if s != "value_follow") + (
    "rules_mix",
    "workers",
    "inference_batch_size",
    "server_replicas",
    "server_pipeline",
    "server_threads",
    "threads",
    "ref_kl_coef",
    "start_priority",
    "start_priority_uniform",
    "start_priority_staleness",
    "start_priority_temperature",
    "start_priority_ema",
    "population_reload_every",
)


def parse_reward(spec: Optional[str]) -> tuple:
    """``policy_gradient.parse_reward`` limited to the generic rewards (``win``, ``placement[:points]``)."""
    kind, arg = pg.parse_reward(spec)
    if kind not in REWARDS:
        raise ValueError(f"reward {spec!r} is 1830-only (the generic stage has win and placement)")
    return kind, arg


def default_start_templates(cfg: GenericPGConfig) -> None:
    """Fill ``start_positions`` / ``midgame_positions`` from the title's spec templates where unset."""
    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(cfg.title)
    if cfg.start_positions is None:
        cfg.start_positions = spec.start_positions_template
    if cfg.midgame_positions is None and cfg.midgame_fraction > 0:
        cfg.midgame_positions = spec.midgame_positions_template


def check_config(cfg: GenericPGConfig) -> None:
    """Fail fast: the title, its player range, the legacy seat / start-file checks
    (``policy_gradient.check_players``), the reward, and the generic checkpoints."""
    from rl18xx.agent.alphazero.titles import title_spec

    spec = title_spec(cfg.title)
    for n in pg.player_counts(cfg):
        try:
            spec.check_players(n)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    pg.check_players(cfg)
    try:
        cfg.rules_mix = spec.parse_rules_mix(cfg.rules_mix)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    parse_reward(cfg.reward)
    if cfg.anchor_every and cfg.anchor_every % cfg.snapshot_every:
        raise SystemExit("--anchor-every must be a multiple of --snapshot-every (the anchor's checkpoint is kept)")
    for path in [cfg.policy_checkpoint, cfg.value_checkpoint, cfg.opponent_checkpoint, cfg.exploiter_of]:
        if path:
            check_generic_checkpoint(path, cfg.title)
    if cfg.ref_kl_coef < 0:
        raise SystemExit("--ref-kl-coef must be >= 0")
    if cfg.ref_kl_coef > 0 and cfg.ref_checkpoint:
        check_generic_checkpoint(cfg.ref_checkpoint, cfg.title)
    if cfg.ref_temperature is not None and cfg.ref_temperature <= 0:
        raise SystemExit("--ref-temperature must be > 0")
    if cfg.start_priority:
        from rl18xx.agent.alphazero.generic_play import START_SCORES

        if cfg.start_priority_score not in START_SCORES:
            raise SystemExit(f"--start-priority-score: one of {START_SCORES}")
        if not (
            0.0 <= cfg.start_priority_uniform <= 1.0
            and 0.0 <= cfg.start_priority_staleness <= 1.0
            and cfg.start_priority_temperature > 0
            and 0.0 < cfg.start_priority_ema <= 1.0
        ):
            raise SystemExit("--start-priority-*: uniform and staleness in [0, 1], temperature > 0, ema in (0, 1]")
    if cfg.exploiter_of and not cfg.population:
        raise SystemExit("exploiter mode: the run's population is its frozen main (set up by generic_pg.run)")
    if cfg.population and not (cfg.exploiter_of and not Path(cfg.population).exists()):
        # (a new exploiter run's population file is written after these checks: its member is exploiter_of)
        members = load_members(cfg.population)
        legacy = [m for m in members if not is_generic_checkpoint(member_path(m))]
        if legacy:
            raise SystemExit(
                f"population {cfg.population}: {len(legacy)} member(s) are not generic checkpoints "
                f"(e.g. {legacy[0]}); a generic run's league takes generic checkpoints of {cfg.title} only"
            )
        for member in members:
            check_generic_checkpoint(member_path(member), cfg.title)


# ---------------------------------------------------------------------------------------------------
# League members (A1: tempered members, per-count pins; A3: population reload)
# ---------------------------------------------------------------------------------------------------


def member_key(path: str, temperature: Optional[float] = None) -> str:
    """A league member's key: its checkpoint path, or ``<path>@<T>`` for a member with its own temperature."""
    return str(path) if temperature is None else f"{path}@{float(temperature):g}"


def member_path(member: str) -> str:
    """The checkpoint of a member key (``<path>@<T>`` -> ``<path>``)."""
    path, sep, tail = str(member).rpartition("@")
    if sep and path:
        try:
            float(tail)
            return path
        except ValueError:
            pass
    return str(member)


def member_temperature(member: str, default: float) -> float:
    """The play temperature of a member key: its ``@<T>``, else ``default`` (the run's opponent temperature)."""
    path = member_path(member)
    return float(str(member)[len(path) + 1 :]) if path != str(member) else float(default)


def _population_entries(path) -> list:
    data = json.loads(Path(path).read_text())
    return data["members"] if isinstance(data, dict) else data


def _entry_key(entry) -> str:
    if isinstance(entry, dict):
        return member_key(entry["path"], entry.get("temperature"))
    return str(entry)


def load_members(path) -> list:
    """Member keys of a population file (``policy_gradient.load_population``'s formats; an entry's
    ``temperature`` gives the key ``<path>@<T>``, so one checkpoint can be two members), duplicates dropped."""
    return list(dict.fromkeys(_entry_key(e) for e in _population_entries(path)))


def load_member_min_shares(path, num_players: Optional[int] = None) -> dict:
    """Key -> pinned share at ``num_players`` (``policy_gradient.load_min_shares`` with member keys):
    ``min_share_by_players`` (``{"<n>": share}``: the pin at each listed count, none elsewhere) when an entry has
    it, else ``min_share`` limited by ``min_share_players``; ``num_players`` None: every pin at any count."""
    shares = {}
    for entry in _population_entries(path):
        if not isinstance(entry, dict):
            continue
        by_players = entry.get("min_share_by_players")
        if by_players:
            share = (
                max(float(v) for v in by_players.values())
                if num_players is None
                else float({int(k): v for k, v in by_players.items()}.get(int(num_players), 0.0))
            )
        else:
            counts = entry.get("min_share_players")
            applies = num_players is None or not counts or int(num_players) in {int(n) for n in counts}
            share = float(entry.get("min_share") or 0.0) if applies else 0.0
        if share > 0:
            shares[_entry_key(entry)] = share
    return shares


def _own_member(member: str, run_dir: Optional[Path]) -> bool:
    """A member that is one of the run's own snapshots (``<run_dir>/learner/<u>.pth``)."""
    if run_dir is None:
        return False
    path = Path(member_path(member))
    return path.parent.name == "learner" and path.parent.parent.resolve() == Path(run_dir).resolve()


def league_for_run(cfg: GenericPGConfig, league_state: Optional[dict], run_dir: Optional[Path] = None):
    """``policy_gradient.league_for_run`` with member keys (tempered members) and per-count pins. With
    ``population_reload_every`` the file is authoritative on a resume too: members neither in it nor the run's
    own snapshots leave the restored league."""
    if not cfg.population:
        return None
    kw = {"power": cfg.pfsp_power}
    counts = pg.player_counts(cfg)
    members = load_members(cfg.population)
    if league_state:
        league = pg.CountLeague.from_state(league_state, counts, **kw)
        for member in members:
            league.add(member)
        if cfg.population_reload_every:
            for member in league.members:
                if member not in members and not _own_member(member, run_dir):
                    league.remove(member)
    else:
        league = pg.CountLeague(counts, members, **kw)
    for n in league.counts:  # the file's pins at each count, also on a resume
        league.league(n).min_shares = load_member_min_shares(cfg.population, n)
    if not league.members:
        raise SystemExit(f"population {cfg.population} has no members")
    return league


def reload_population(league, cfg: GenericPGConfig, run_dir: Optional[Path], known: set) -> dict:
    """Bring a running league in line with its population file: members new to the file join (at an even rate),
    members that were in the file (``known``: the file's members at the previous read) and are no longer leave,
    every count's pins follow the file. Own snapshots are never touched. Updates ``known``; returns
    ``{"added": [...], "removed": [...]}``. A file that can't be read or checked is ignored (logged)."""
    try:
        members = load_members(cfg.population)
        for member in members:
            if member not in league.members:
                check_generic_checkpoint(member_path(member), cfg.title)
    except (OSError, ValueError, KeyError, SystemExit) as e:
        LOGGER.warning(f"population {cfg.population}: not reloaded ({e})")
        return {"added": [], "removed": []}
    added = [m for m in members if m not in league.members]
    removed = [m for m in league.members if m in known and m not in members and not _own_member(m, run_dir)]
    for member in added:
        league.add(member)
    for member in removed:
        league.remove(member)
    for n in league.counts:
        league.league(n).min_shares = load_member_min_shares(cfg.population, n)
    known.clear()
    known.update(members)
    return {"added": added, "removed": removed}


def is_generic_checkpoint(path) -> bool:
    from rl18xx.agent.alphazero import model_generic as mg

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get(mg.ARCHITECTURE_KEY) == mg.ARCHITECTURE


def check_generic_checkpoint(path, title: str) -> None:
    """SystemExit unless ``path`` is a generic checkpoint that fits the running engine and covers ``title``."""
    from rl18xx.agent.alphazero import model_generic as mg

    if not Path(path).is_file():
        raise SystemExit(f"checkpoint {path} not found")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get(mg.ARCHITECTURE_KEY) != mg.ARCHITECTURE:
        raise SystemExit(f"{path} is not a generic ({mg.ARCHITECTURE}) checkpoint")
    meta = payload[mg.META_KEY]
    try:
        mg.check_checkpoint_meta(meta)
    except mg.CheckpointMismatch as e:
        raise SystemExit(f"{path}: {e}") from None
    if str(title) not in [str(t) for t in meta.get("titles", [])]:
        raise SystemExit(f"{path} was trained on {meta.get('titles')}, not {title}")


# ---------------------------------------------------------------------------------------------------
# Worker side: games
# ---------------------------------------------------------------------------------------------------

_CLIENTS: dict = {}
_SLOT_VERSIONS: dict = {}
_SLOT_TEMPERATURES: dict = {}


def slot_version(slot: str) -> Optional[int]:
    value = _SLOT_VERSIONS.get(slot)
    return None if value is None else int(value.value)


def seat_temperature(name: str, settings: dict) -> float:
    """The sampling temperature of a seat played by server ``name``: the learner's, a league slot's member's own
    (the trainer's shared value, updated as the slot changes members; else the task's ``slot_temperatures``), or
    the run's opponent temperature."""
    if name == "learner":
        return float(settings["learner_temperature"])
    shared = _SLOT_TEMPERATURES.get(name)
    if shared is not None:
        return float(shared.value)
    league = settings.get("league") or {}
    return float((league.get("slot_temperatures") or {}).get(name, settings["opponent_temperature"]))


def worker_init(server_queues: dict, slot_versions: Optional[dict] = None, slot_temperatures: Optional[dict] = None):
    """Pool initializer: one slot on every generic server ("learner", "sl", "pool", league "opp..."), and the
    league slots' shared member-change counters (``policy_gradient.worker_init`` for generic clients) and play
    temperatures (tempered members)."""
    from rl18xx.agent.alphazero.generic_inference import GenericInferenceClient

    torch.set_num_threads(1)
    _SLOT_VERSIONS.clear()
    _SLOT_VERSIONS.update(slot_versions or {})
    _SLOT_TEMPERATURES.clear()
    _SLOT_TEMPERATURES.update(slot_temperatures or {})
    for name, (request_q, reply_qs, ticket_q) in server_queues.items():
        worker_id = ticket_q.get()
        _CLIENTS[name] = GenericInferenceClient(request_q=request_q, reply_q=reply_qs[worker_id], worker_id=worker_id)
    logging.getLogger().setLevel(logging.WARNING)


@dataclass
class PGRow:
    """One learner decision kept for training."""

    tokens: bytes  # lz4(pickle(single-position token arrays)), exactly as the server saw them
    rows: np.ndarray  # (n, 14) int32 descriptor rows of the legal list
    price_range: np.ndarray  # (n, 3) int32
    choice: int  # the played row's index in the legal list
    old_logp: float  # log-probability it was sampled with: move at the learner's temperature + price cell
    price_cell: int  # the played row's price cell (-1: no open price)
    seat: int  # the mover's seat (seating order; the encoder's rotation unless the title reorders its players)
    num_players: int
    advantage: float = 0.0  # the seat's reward, then reward - V(s) (subtract_baselines)
    win: Optional[np.ndarray] = None  # (6,) win share, seat frame
    score: Optional[np.ndarray] = None  # (6,) score-head target (placement points / reward), seat frame
    price_grid: bool = False  # the legal list's price grid (1867: cells in step units, as the server's mask)
    #: (n,) each encoder seat's seating index (``generic_data.seat_order``; None: ``(arange(n) + seat) mod n``). The
    #: 1822 family reorders its players every Stock Round, so the seat frame is not a rotation of the seating order.
    order: Optional[np.ndarray] = None
    #: The game's start label (``human:<id>``, ``midgame:<id>:sr<k>``, ``auction``, ``random``): prioritized starts
    #: score each start from its rows.
    start: Optional[str] = None


@dataclass
class _Game:
    raw: object
    uid: str
    opponent: str
    seat_models: list
    start: str
    seed: Optional[int] = None
    #: Player ids in seating order (the game's start): seats, rewards and saved games use it, not the
    #: engine's current ``players`` (which the 1822 family reorders every Stock Round).
    seating: list = field(default_factory=list)
    decisions: int = 0
    auction_decisions: int = 0  # decisions taken in an Auction round (1830 waterfall, 1867 single-item auction)
    termination: Optional[str] = None
    learner: set = field(default_factory=set)
    seen: dict = field(default_factory=dict)  # learner seat -> decisions so far
    reservoir: dict = field(default_factory=dict)  # learner seat -> [PGRow]
    seat_members: dict = field(default_factory=dict)
    slot_versions: dict = field(default_factory=dict)
    league_mixed: bool = False
    #: The game's rules combination (``titles.rules_name``: "base", 1822's "plus+nr+ss+th", ...).
    rules: str = "base"
    #: All-pass openings this game slot redrew before this game (``redraw_all_pass``).
    redraws: int = 0
    #: A generated move the engine refused (``apply_move``): the game ends there as it stands (``rejected_move``).
    rejected: Optional[dict] = None


def tempered_logp(logp: np.ndarray, temperature: float) -> np.ndarray:
    """log softmax(logp / T) in float64 (``policy_selfplay.tempered`` in log space: a sharp policy's tiny
    probabilities are never clipped)."""
    x = np.asarray(logp, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.where(np.isfinite(x), x, -1e4)
    x = x / float(temperature)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def position_tokens(arrays: dict, b: int) -> dict:
    """Position ``b``'s token arrays (no map de-duplication), relations trimmed to its own."""
    from rl18xx.agent.alphazero.generic_inference import RELATION_KEYS, TOKEN_KEYS

    out = {key: arrays[key][b : b + 1].copy() for key in TOKEN_KEYS}
    width = int(out["rel_mask"].sum())
    for key in RELATION_KEYS:
        out[key] = out[key][:, :width]
    return out


def _compress(obj) -> bytes:
    return lz4.frame.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def _decompress(blob: bytes):
    return pickle.loads(lz4.frame.decompress(blob))


def apply_move(g: _Game, play, legal, k: int, price) -> bool:
    """Apply legal row ``k`` at ``price``; a move the title's generator offered but the engine refuses (a generator
    bug: 2026-11-14, an 1822CA tile lay "216-1 is not legally rotated for 512") ends the game as it stands
    (``termination`` "rejected_move", the move and the error in ``g.rejected``) instead of the whole pool task."""
    try:
        play.apply(g.raw, legal, k, price)
        return True
    except RuntimeError as e:
        wires = getattr(legal, "wires", None)
        g.rejected = {
            "index": int(legal.indices[k]),
            "wire": dict(wires[k]) if wires is not None else None,
            "price": None if price is None else int(price),
            "error": str(e)[:300],
            "action_count": len(g.raw.raw_actions),
        }
        g.termination = "rejected_move"
        LOGGER.warning("Game %s: the engine refused a generated move (%s); the game ends here", g.uid, g.rejected)
        return False


def _advance(g: _Game, play, settings: dict):
    """Apply forced moves until a decision (its ``LegalActions``) or the end (None): a finished game, the
    decision cap (scored on net worth as it stands), an auction stall, a Ruby dead end or a refused move."""
    raw = g.raw
    while True:
        if g.termination == "rejected_move":
            return None
        if raw.finished:
            g.termination = g.termination or "finished"
            return None
        if g.decisions >= settings["max_decisions"]:
            g.termination = "max_length"
            return None
        stall = settings.get("auction_stall_decisions", 0)
        if stall and g.decisions >= stall and raw.round.round_type == "Auction":
            g.termination = "auction_stall"
            return None
        legal = play.legal(raw)
        if len(legal) == 0:
            if raw.g1822_ruby_dead_end():
                # The 1822 family: a state Ruby cannot continue either (it raises on the only way on, e.g. 1822MX's
                # NdeM holding cash at its close; g1822_remaining_work.md "Known issues"): scored as it stands. (The
                # NdeM's eleventh station, the commonest before 2026-11-14, now follows the printed rules.)
                g.termination = "dead_end"
                return None
            raise RuntimeError(f"game {g.uid}: no legal action in an unfinished game")
        if play.is_decision(legal):
            return legal
        apply_move(g, play, legal, 0, play.fixed_price(raw, legal, 0))


def game_outcome(raw, seating: Optional[Sequence[int]] = None) -> tuple:
    """``(net worth, win share, net-worth fractions)`` per seat in seating order (``raw.result()``); ``seating`` = the
    player ids in seating order (default: ``raw.players`` as they stand, which the 1822 family reorders)."""
    result = raw.result()
    ids = list(seating) if seating else [p.id for p in raw.players]
    worth = np.array([float(result.get(pid, 0)) for pid in ids], dtype=np.float64)
    winners = worth >= worth.max() - 1e-6
    win_share = (winners / winners.sum()).astype(np.float32)
    clipped = worth.clip(min=0)
    fractions = clipped / clipped.sum() if clipped.sum() > 0 else np.full(len(worth), 1.0 / len(worth))
    return worth, win_share, fractions.astype(np.float32)


def seat_rewards(spec: str, worth: np.ndarray, win_share: np.ndarray) -> tuple:
    """``(reward, score target)`` per seat (seating order): the reward is the win share or placement
    points; the critic's score head learns placement points (the reward's own points with ``placement:``)."""
    kind, points = parse_reward(spec)
    placement = pg.placement_points(worth, points if kind == "placement" else None).astype(np.float32)
    reward = placement if kind == "placement" else np.asarray(win_share, dtype=np.float32)
    return reward, placement


def _ahead(mine: float, theirs: float) -> float:
    return 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0


def _seat_frame(values, rotation: int, n: int, order=None) -> np.ndarray:
    from rl18xx.agent.alphazero.generic_data import seat_frame

    return seat_frame(values, rotation, n, order)


def mover_seat(raw, seating: Sequence[int], rotation: int) -> int:
    """The seating index of the player the encoder put at seat 0 (``meta[:, 5]`` is the mover's index in the
    engine's current ``players``; equal to the seating index unless the title reorders its players)."""
    return list(seating).index(int(raw.players[int(rotation)].id))


def seating_players(raw, seating: Sequence[int]) -> list:
    """``[[id, name]]`` in seating order (saved games replay from it)."""
    names = {int(p.id): str(p.name) for p in raw.players}
    return [[int(pid), names[int(pid)]] for pid in seating]


def _auction_prices(title: str, start: str, termination: Optional[str], raw) -> Optional[dict]:
    """1830 auction starts: each private's price (``policy_gradient.auction_prices``); None elsewhere."""
    if str(title) != "1830" or start != "auction" or termination == "auction_stall":
        return None
    from rl18xx.agent.alphazero.search_learning import auction_sales

    return {company: int(price) for company, (_, price) in auction_sales(raw.raw_actions).items()}


def play_generic_pg_games(num_games: int, settings: dict) -> dict:
    """Pool task: play ``num_games`` concurrent games of one title and player count on the generic servers
    and return the learner's kept decisions (rewards filled in) and each game's result."""
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_data import seat_order
    from rl18xx.agent.alphazero.generic_inference import slice_positions
    from rl18xx.agent.alphazero.generic_play import (
        ALL_PASS,
        MAX_ALL_PASS_REDRAWS,
        StartPicker,
        all_pass_opening,
        replayed_start,
        title_play,
    )
    from rl18xx.agent.alphazero.generic_tokens import encode_batch
    from rl18xx.agent.alphazero.titles import draw_rules, rules_name

    seed = settings.get("seed")  # [run seed, task index] (GenericPGConfig.seed), else unseeded
    rng = np.random.default_rng(None if seed is None else [int(x) for x in seed])
    py_rng = random.Random() if seed is None else random.Random(":".join(str(int(x)) for x in seed))
    picker = StartPicker(settings["start_picks"]) if settings.get("start_picks") is not None else None
    title = settings["title"]
    rules_mix = settings.get("rules_mix") or {"base": 1.0}
    play = title_play(title)
    k_keep = settings["positions_per_seat"]
    n_players = int(settings["num_players"])
    compress = settings.get("compress_requests", True)
    start_time = time.time()
    stats = Counter()

    games, skipped_starts = [], 0
    redraw_all_pass = settings.get("redraw_all_pass", True)

    def new_game(redraws: int = 0) -> _Game:
        nonlocal skipped_starts
        rules = draw_rules(rules_mix, py_rng, title)
        start, raw, skipped = replayed_start(
            title,
            n_players,
            settings["start_positions"],
            settings["random_start_fraction"],
            rng=py_rng,
            midgame_path=settings.get("midgame_positions"),
            midgame_fraction=settings.get("midgame_fraction", 0.0),
            auction_fraction=settings.get("auction_fraction", 0.0),
            auction_unlock=settings.get("auction_unlock", True),
            optional_rules=rules,
            picker=picker,
        )
        skipped_starts += skipped
        learner, seat_models, seat_members = pg._deal_seats(settings, py_rng)
        opponent = "league" if settings.get("league") else seat_models[min(set(range(n_players)) - learner)]
        g = _Game(raw=raw, uid=uuid.uuid4().hex, opponent=opponent, seat_models=seat_models, start=start.label)
        g.seed = start.seed
        g.rules = rules_name(title, start.optional_rules)
        # the seating order is the new game's (``titles.new_game`` numbers the seats in order), not raw.players after
        # the start's actions: the 1822 family reorders its players every Stock Round, and saved games replay from
        # the fresh game in this order
        g.seating = sorted(int(p.id) for p in raw.players)
        g.learner = set(learner)
        g.seat_members = seat_members
        if seat_members:
            dealt = settings["league"].get("slot_versions") or {}
            g.slot_versions = {seat_models[seat]: slot_version(seat_models[seat]) for seat in seat_members}
            g.league_mixed = any(v is not None and v != dealt.get(slot, v) for slot, v in g.slot_versions.items())
        for seat in learner:
            g.seen[seat], g.reservoir[seat] = 0, []
        g.redraws = redraws
        return g

    games = [new_game() for _ in range(num_games)]
    finished, active, decisions_total, all_pass_redrawn = [], list(games), 0, 0
    t_encode = t_wait = t_play = 0.0
    while active:
        pending = []
        queue = list(active)
        while queue:
            g = queue.pop(0)
            legal = _advance(g, play, settings)
            if legal is not None:
                pending.append((g, legal))
            elif redraw_all_pass and all_pass_opening(title, g.raw):
                if g.redraws < MAX_ALL_PASS_REDRAWS:
                    # nothing bought, sold or bid in the opening Stock Round: Ruby ends the game, every seat tied --
                    # dropped (no rows, no credit) and replaced by a fresh draw
                    all_pass_redrawn += 1
                    queue.append(new_game(g.redraws + 1))
                else:
                    g.termination = ALL_PASS
                    finished.append(g)
            else:
                finished.append(g)
        active = [g for g, _ in pending]
        if not pending:
            break
        t0 = time.monotonic()
        enc = encode_batch([g.raw for g, _ in pending])
        rotation = enc["meta"][:, 5]
        movers = [mover_seat(g.raw, g.seating, int(rotation[i])) for i, (g, _) in enumerate(pending)]
        by_model = defaultdict(list)
        for i, (g, _) in enumerate(pending):
            by_model[g.seat_models[movers[i]]].append(i)
        sent = {}
        for name, rows in by_model.items():
            arrays = mg.add_legal_inputs(slice_positions(enc, rows), [pending[i][1] for i in rows])
            eps = 0.0 if name == "learner" else settings["opponent_price_eps"]
            sent[name] = (_CLIENTS[name].send_inputs(arrays, price_eps=eps, compress=compress), rows, arrays)
        t1 = time.monotonic()
        t_encode += t1 - t0
        for name, (request, rows, arrays) in sent.items():
            tw = time.monotonic()
            reply = _CLIENTS[name].receive_outputs(request)
            tp = time.monotonic()
            t_wait += tp - tw
            is_learner = name == "learner"
            temperature = seat_temperature(name, settings)
            offsets = arrays["legal_offsets"]
            price_of_row = {int(r): j for j, r in enumerate(arrays["price_rows"])}
            for j, i in enumerate(rows):
                g, legal = pending[i]
                lo, hi = int(offsets[j]), int(offsets[j + 1])
                tl = tempered_logp(reply.logp[lo:hi], temperature)
                p = np.exp(tl)
                k = int(rng.choice(len(p), p=p / p.sum()))
                cell, cell_logp, price = -1, 0.0, None
                if play.open_price(legal, k):
                    q = price_of_row[lo + k]
                    cell, cell_logp = int(reply.price_cell[q]), float(reply.price_cell_logp[q])
                    # the cells of the server's mask: on the layout's grid (1867: the move's step)
                    price = int(legal.price_cells(k).sample_price(cell, rng))
                else:
                    price = play.fixed_price(g.raw, legal, k)
                if is_learner:
                    seat = movers[i]
                    g.seen[seat] += 1
                    keep = len(g.reservoir[seat]) < k_keep
                    slot = len(g.reservoir[seat])
                    if not keep:
                        r = int(rng.integers(0, g.seen[seat]))
                        keep, slot = r < k_keep, r
                    if keep:
                        row = PGRow(
                            tokens=_compress(position_tokens(arrays, j)),
                            rows=np.asarray(legal.rows, dtype=np.int32),
                            price_range=np.asarray(legal.price_range, dtype=np.int32),
                            choice=k,
                            old_logp=float(tl[k] + cell_logp),
                            price_cell=cell,
                            seat=seat,
                            num_players=n_players,
                            price_grid=bool(legal.price_grid),
                            order=seat_order(g.raw, g.seating, int(rotation[i])),
                        )
                        if slot < len(g.reservoir[seat]):
                            g.reservoir[seat][slot] = row
                        else:
                            g.reservoir[seat].append(row)
                    stats["learner_decisions"] += 1
                    stats["learner_price_decisions"] += cell >= 0
                g.auction_decisions += g.raw.round.round_type == "Auction"
                if apply_move(g, play, legal, k, price):
                    g.decisions += 1
                    decisions_total += 1
            t_play += time.monotonic() - tp

    records = []
    save_every = settings.get("save_game_every", 0)
    for g in finished:
        worth, win_share, fractions = game_outcome(g.raw, g.seating)
        reward, placement = seat_rewards(settings["reward"], worth, win_share)
        rows = []
        for seat, kept in g.reservoir.items():
            for row in kept:
                row.advantage = float(reward[seat])
                row.win = _seat_frame(win_share, row.seat, n_players, row.order)
                row.score = _seat_frame(placement, row.seat, n_players, row.order)
                row.start = g.start
                rows.append(row)
        record = {
            "uid": g.uid,
            "title": title,
            "num_players": n_players,
            "opponent": g.opponent,
            "pool_label": settings.get("pool_label"),
            "learner_seats": sorted(g.learner),
            "win_share": win_share.tolist(),
            "worth_share": fractions.tolist(),
            "reward": [float(x) for x in reward],
            "decisions": g.decisions,
            "auction_decisions": g.auction_decisions,
            "learner_decisions": sum(g.seen.values()),
            "termination": g.termination,
            "rules": g.rules,
            "start_kind": g.start.split(":")[0] if g.start else None,
            "start_label": g.start,
            "auction_prices": _auction_prices(title, g.start, g.termination, g.raw),
            "first_values": {},
            "rows": rows,
        }
        if g.seat_members:
            mine = [float(fractions[seat]) for seat in g.learner]
            record["seat_members"] = g.seat_members
            record["league_mixed"] = g.league_mixed or any(
                v is not None and slot_version(slot) != v for slot, v in g.slot_versions.items()
            )
            # Per opponent seat, the share of the learner's seats that finished ahead of it (ties half).
            record["pairwise"] = (
                []
                if record["league_mixed"]
                else [
                    (member, float(np.mean([_ahead(m, float(fractions[seat])) for m in mine])))
                    for seat, member in g.seat_members.items()
                ]
            )
        if g.rejected:
            record["rejected_move"] = g.rejected
        if (save_every and py_rng.random() * save_every < 1.0) or g.rejected:  # (a refused move: always kept)
            record["saved_game"] = {
                "start": g.start,
                "seed": g.seed,
                "net_worth": worth.tolist(),
                "worth_share": fractions.tolist(),
                "raw_actions": [dict(a) for a in g.raw.raw_actions],
                "players": seating_players(g.raw, g.seating),
                "optional_rules": list(getattr(g.raw, "optional_rules", None) or []),
                "auction_unlock": bool(g.raw.auction_unlock) if hasattr(g.raw, "auction_unlock") else False,
                "no_buyback_rule": bool(g.raw.no_buyback_rule) if hasattr(g.raw, "no_buyback_rule") else False,
            }
        records.append(record)
    if picker is not None:
        stats.update({f"start_{key}": value for key, value in picker.counts.items()})
    return {
        "games": records,
        "decisions": decisions_total,
        "skipped_starts": skipped_starts,
        "all_pass_redrawn": all_pass_redrawn,
        "num_players": n_players,
        "seconds": time.time() - start_time,
        "worker_seconds": {"encode_send": t_encode, "wait": t_wait, "play": t_play},
        "stats": dict(stats),
    }


def save_game(run_dir: Path, record: dict, update: int, learner_checkpoint: str, opponent_checkpoint: str, cfg) -> str:
    """Keep a ``save_game_every`` game as ``<run_dir>/games/u<update>_<id>.json`` (an 18xx.games-style export
    with the title, players and rule variants, and this run's metadata under ``"rl18xx"``) with a line in
    ``<run_dir>/games.jsonl``; returns the file's name."""
    from rl18xx.agent.alphazero.action_descriptors import action_descriptors
    from rl18xx.agent.alphazero.game_records import GAME_RECORD_VERSION, append_index, save_game_record
    from rl18xx.agent.alphazero.generic_tokens import schema

    saved = record["saved_game"]
    n_players = len(record["win_share"])
    learner = set(int(s) for s in record["learner_seats"])
    name = f"u{update}_{record['uid'][:8]}"
    members = {int(k): v for k, v in (record.get("seat_members") or {}).items()}
    opponent = {"pool": "pool snapshot", "league": "league member"}.get(record["opponent"], "fixed opponent")
    d = action_descriptors(cfg.title)
    entry = {
        "name": name,
        "kind": "generic_policy_gradient",
        "title": cfg.title,
        "run": run_dir.name,
        "update": update,
        "learner_checkpoint": learner_checkpoint,
        "learner_seats": sorted(learner),
        "opponent": record["opponent"],
        "opponent_checkpoint": opponent_checkpoint,
        "sides": ["A" if s in learner else "B" for s in range(n_players)],
        "seat_labels": [
            f"learner (update {update})" if s in learner else members.get(s, f"{opponent} {opponent_checkpoint}")
            for s in range(n_players)
        ],
        "learner_temperature": cfg.learner_temperature,
        "opponent_temperature": cfg.opponent_temperature,
        "start": saved["start"],
        "rules": record.get("rules", "base"),
        "termination": record["termination"],
        **({"rejected_move": record["rejected_move"]} if record.get("rejected_move") else {}),
        "decisions": record["decisions"],
        "win_share": record["win_share"],
        "net_worth": saved["net_worth"],
        "worth_share": saved["worth_share"],
    }
    game = {
        "id": name,
        "title": cfg.title,
        "description": f"generic policy-gradient {run_dir.name} update {update}: learner vs {opponent}",
        "players": [{"id": pid, "name": pname} for pid, pname in saved["players"]],
        "min_players": n_players,
        "max_players": n_players,
        "settings": {"optional_rules": saved["optional_rules"], "seed": saved.get("seed") or 0},
        "actions": saved["raw_actions"],
        "rl18xx": {
            "version": GAME_RECORD_VERSION,
            "auction_unlock": saved["auction_unlock"],
            "no_buyback_rule": saved["no_buyback_rule"],
            "seed": saved.get("seed"),
            "layout_version": d.version,
            "layout_fingerprint": d.fingerprint,
            "encoder_version": int(schema()["version"]),
            **entry,
        },
    }
    entry["game_file"] = save_game_record(run_dir, name, game)
    append_index(run_dir, entry)
    return name


# ---------------------------------------------------------------------------------------------------
# Learner side: batches and losses
# ---------------------------------------------------------------------------------------------------


def collate_rows(rows: Sequence[PGRow], config=None) -> dict:
    """PG rows -> one batch: ``inputs`` (``build_inputs``' arrays: tokens padded and stacked as
    ``encode_batch`` pads them, the flat legal list) and per position ``choice``, ``old_logp``,
    ``advantage``, ``win`` / ``score`` (B, 6) targets, and for the positions whose move had an open price
    ``price_pos`` (Q,), ``price_sel`` (Q,) (the played row's index into ``inputs["price_rows"]``) and
    ``price_cell`` (Q,)."""
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.generic_data import concat_encodings

    arrays = concat_encodings([_decompress(r.tokens) for r in rows])
    legal = [mg.LegalActions(rows=r.rows, price_range=r.price_range, price_grid=r.price_grid) for r in rows]
    mg.add_legal_inputs(arrays, legal, config=config)
    choice = np.array([r.choice for r in rows], dtype=np.int64)
    offsets = arrays["legal_offsets"]
    where = {int(row): j for j, row in enumerate(arrays["price_rows"])}
    price_pos, price_sel, price_cell = [], [], []
    for b, r in enumerate(rows):
        if r.price_cell >= 0:
            price_pos.append(b)
            price_sel.append(where[int(offsets[b]) + int(r.choice)])
            price_cell.append(int(r.price_cell))
    return {
        "inputs": arrays,
        "choice": choice,
        "old_logp": np.array([r.old_logp for r in rows], dtype=np.float32),
        "advantage": np.array([r.advantage for r in rows], dtype=np.float32),
        "win": np.stack([r.win for r in rows]).astype(np.float32),
        "score": np.stack([r.score for r in rows]).astype(np.float32),
        "price_pos": np.array(price_pos, dtype=np.int64),
        "price_sel": np.array(price_sel, dtype=np.int64),
        "price_cell": np.array(price_cell, dtype=np.int64),
    }


def batch_to_device(batch: dict, device) -> dict:
    from rl18xx.agent.alphazero import model_generic as mg

    out = {"inputs": mg.to_device(batch["inputs"], device)}
    out.update(mg.to_device({k: v for k, v in batch.items() if k != "inputs"}, device))
    return out


def _policy_kl(log_p, other_out: dict, batch: dict, temperature: float, cell_lp=None) -> tuple:
    """``(per-position KL(learner || other), mean price-cell KL)``: over the legal rows with ``other`` at
    ``temperature`` (``log_p``: the learner's tempered row log-probabilities), plus the played row's price cells
    (``cell_lp``: the learner's masked cell log-probabilities) for positions with an open price."""
    from rl18xx.agent.alphazero.model_generic import segment_log_softmax

    inputs = batch["inputs"]
    pos, offsets = inputs["legal_pos"], inputs["legal_offsets"]
    b = offsets.shape[0] - 1
    other_log_p = segment_log_softmax(other_out["logits"].float() / temperature, pos, b)
    kl = log_p.new_zeros(b).index_add(0, pos, log_p.exp() * (log_p - other_log_p))
    price_kl = log_p.new_zeros(())
    if cell_lp is not None:
        sel, ppos = batch["price_sel"], batch["price_pos"]
        mask = inputs["price_cell_mask"][sel]
        other_cell_lp = other_out["price_logp"][sel].float().masked_fill(~mask, 0.0)
        cell_kl = (cell_lp.exp() * mask * (cell_lp - other_cell_lp)).sum(dim=1)
        kl = kl.index_add(0, ppos, cell_kl)
        price_kl = cell_kl.mean()
    return kl, price_kl


def generic_pg_losses(
    out: dict,
    anchor_out: dict,
    batch: dict,
    clip: float,
    kl_coef: float,
    entropy_coef: float,
    temperature: float = 1.0,
    ref_out: Optional[dict] = None,
    ref_kl_coef: float = 0.0,
    ref_temperature: Optional[float] = None,
) -> dict:
    """PPO-clipped policy gradient plus ``kl_coef`` * KL(learner || anchor), over each position's legal rows
    (``policy_gradient.pg_losses`` for the generic model's flat legal lists).

    A move's log-probability is its row's under softmax(logits / ``temperature``) over the position's legal
    rows (the map factorization included in the logits) plus, for a move with an open price, its price cell's
    (log-softmax over the range's non-empty cells) -- the probabilities it was sampled with. The KL compares
    both policies at ``temperature`` over the legal rows and the played row's price cells.

    ``ref_out`` (A1's reference policy, piKL-style): adds ``ref_kl_coef`` * KL(learner at ``temperature`` ||
    reference at ``ref_temperature``, default ``temperature``) over the same rows and cells (``kl_ref``)."""
    from rl18xx.agent.alphazero.model_generic import segment_log_softmax

    inputs = batch["inputs"]
    pos, offsets = inputs["legal_pos"], inputs["legal_offsets"]
    b = offsets.shape[0] - 1
    log_p = segment_log_softmax(out["logits"].float() / temperature, pos, b)
    p = log_p.exp()
    entropy = log_p.new_zeros(b).index_add(0, pos, -(p * log_p))
    new_logp = log_p[offsets[:-1] + batch["choice"]]

    cell_lp = None
    if batch["price_sel"].numel():
        sel, ppos = batch["price_sel"], batch["price_pos"]
        mask = inputs["price_cell_mask"][sel]
        cell_lp = out["price_logp"][sel].float().masked_fill(~mask, 0.0)
        new_logp = new_logp.index_add(0, ppos, cell_lp.gather(1, batch["price_cell"][:, None]).squeeze(1))
    kl, price_kl = _policy_kl(log_p, anchor_out, batch, temperature, cell_lp)

    advantage = batch["advantage"]
    log_ratio = new_logp - batch["old_logp"]
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage)
    pg_loss = -surrogate.mean()
    total = pg_loss + kl_coef * kl.mean() - entropy_coef * entropy.mean()
    extra = {}
    if ref_out is not None:
        ref_kl, ref_price_kl = _policy_kl(
            log_p, ref_out, batch, temperature if ref_temperature is None else ref_temperature, cell_lp
        )
        total = total + ref_kl_coef * ref_kl.mean()
        extra = {"kl_ref": ref_kl.mean().detach(), "price_kl_ref": ref_price_kl.detach()}
    return {
        "total": total,
        "pg_loss": pg_loss.detach(),
        "kl_anchor": kl.mean().detach(),
        "price_kl_anchor": price_kl.detach(),
        "entropy": entropy.mean().detach(),
        "clip_frac": ((ratio - 1.0).abs() > clip).float().mean().detach(),
        "approx_kl": (-log_ratio).mean().detach(),
        "ratio_max": ratio.max().detach(),
        "log_ratio_abs_mean": log_ratio.abs().mean().detach(),
        **extra,
    }


def _baseline(out: dict, shaped: bool) -> torch.Tensor:
    """The critic's value of each position for its mover (seat 0): the score head with a placement reward,
    else the win head's share."""
    if shaped:
        return out["score"][:, 0].float()
    return torch.softmax(out["win_logits"].float(), dim=-1)[:, 0]


def _prefetched(parts: list, collate):
    """``collate(part)`` for each part in order, the next one built on a helper thread while the caller runs
    the current one on the device (the collate is CPU work: decompression, padding, legal inputs)."""
    from concurrent.futures import ThreadPoolExecutor

    if not parts:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        upcoming = pool.submit(collate, parts[0])
        for i in range(len(parts)):
            current = upcoming.result()
            if i + 1 < len(parts):
                upcoming = pool.submit(collate, parts[i + 1])
            yield current


def subtract_baselines(critic, rows: List[PGRow], cfg: GenericPGConfig, device, chunk: int = 512) -> dict:
    """Turn each row's advantage (its seat's reward) into ``reward - V(s)`` with the critic's value in one
    batched value-only pass over the sampled rows (``policy_gradient.subtract_baselines``)."""
    critic.eval()
    shaped = parse_reward(cfg.reward)[0] != "win"
    values = []
    chunks = [rows[start : start + chunk] for start in range(0, len(rows), chunk)]
    with torch.no_grad():
        for batch in _prefetched(chunks, collate_rows):
            out = critic(batch_to_device(batch, device)["inputs"], value_only=True)
            values.append(_baseline(out, shaped).cpu().numpy())
    values = np.concatenate(values) if values else np.zeros(0, dtype=np.float32)
    rewards = np.array([r.advantage for r in rows], dtype=np.float64)
    for r, v in zip(rows, values):
        r.advantage = float(r.advantage) - float(v)
    return {"critic_row_brier": float(np.mean((values - rewards) ** 2)) if len(rows) else float("nan")}


def _update(
    learner,
    critic,
    anchor,
    learner_opt,
    critic_opt,
    rows: List[PGRow],
    cfg: GenericPGConfig,
    kl_coef,
    device,
    rng,
    reference=None,
) -> dict:
    """One PPO pass over ``rows`` (``ppo_epochs`` times) and one critic pass (``policy_gradient._update``);
    ``reference``: A1's fixed reference policy (``cfg.ref_kl_coef`` > 0)."""
    from rl18xx.agent.alphazero import model_generic as mg

    learner.eval()
    critic.eval()
    anchor.eval()
    if reference is not None:
        reference.eval()
    advantages = np.array([r.advantage for r in rows], dtype=np.float32)
    scale = (float(advantages.std()) or 1.0) if cfg.normalize_advantages else 1.0
    sums, count, first, steps = Counter(), 0, {}, 0
    critic_sums, critic_count = Counter(), 0
    micro = min(cfg.microbatch, cfg.minibatch)
    t_collate = 0.0

    def collate(part):
        return collate_rows(part, learner.config)

    for epoch in range(cfg.ppo_epochs):
        order = rng.permutation(len(rows))
        # One optimizer step per minibatch, its gradient accumulated over microbatches, collated one ahead.
        minibatches = [order[start : start + cfg.minibatch] for start in range(0, len(rows), cfg.minibatch)]
        micros = [(s, mb[m : m + micro]) for s, mb in enumerate(minibatches) for m in range(0, len(mb), micro)]
        batches = _prefetched([[rows[i] for i in idx] for _, idx in micros], collate)
        for k, (s, idx) in enumerate(micros):
            if k == 0 or s != micros[k - 1][0]:
                learner_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
            tc = time.monotonic()
            batch = batch_to_device(next(batches), device)
            if scale != 1.0:
                batch["advantage"] = batch["advantage"] / scale
            t_collate += time.monotonic() - tc
            n = len(idx)
            share = n / len(minibatches[s])
            with torch.no_grad():
                anchor_out = anchor(batch["inputs"])
                ref_out = reference(batch["inputs"]) if reference is not None else None
            out = learner(batch["inputs"])
            losses = generic_pg_losses(
                out,
                anchor_out,
                batch,
                cfg.clip,
                kl_coef,
                cfg.entropy_coef,
                temperature=cfg.learner_temperature,
                ref_out=ref_out,
                ref_kl_coef=cfg.ref_kl_coef,
                ref_temperature=cfg.ref_temperature,
            )
            total = losses["total"]
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Policy gradient diverged: non-finite loss ({ {k: float(v) for k, v in losses.items()} })"
                )
            (total * share).backward()
            if count == 0:
                first = {
                    "clip_frac_first": float(losses["clip_frac"]),
                    "approx_kl_first": float(losses["approx_kl"]),
                }
            count += n
            for key, value in losses.items():
                if key != "total":
                    sums[key] += float(value) * n
            if epoch == 0:
                c_out = critic(batch["inputs"], value_only=True)
                c = mg.value_losses(c_out, batch["win"], batch["score"])
                loss = c["ce"] + cfg.score_loss_weight * c["mse"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Critic diverged: non-finite loss")
                (loss * share).backward()
                critic_count += n
                critic_sums["ce"] += float(c["ce"].detach()) * n
                critic_sums["mse"] += float(c["mse"].detach()) * n
                critic_sums["winner_hit"] += float(c["winner_hit"]) * n
            if k + 1 < len(micros) and micros[k + 1][0] == s:
                continue  # more microbatches of this minibatch
            norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Policy gradient diverged: non-finite gradient norm")
            learner_opt.step()
            sums["grad_norm"] += float(norm) * len(minibatches[s])
            steps += 1
            if epoch == 0:
                cnorm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                if not torch.isfinite(cnorm):
                    raise RuntimeError("Critic diverged: non-finite gradient norm")
                critic_opt.step()
    stats = {key: value / max(count, 1) for key, value in sums.items()}
    stats.update(first)
    stats.update({f"critic_{key}": value / max(critic_count, 1) for key, value in critic_sums.items()})
    stats.update(
        rows=len(rows),
        optimizer_steps=steps,
        adv_mean=float(advantages.mean()) if len(advantages) else 0.0,
        adv_std=float(advantages.std()) if len(advantages) else 0.0,
        adv_abs_mean=float(np.abs(advantages).mean()) if len(advantages) else 0.0,
        price_rows=sum(1 for r in rows if r.price_cell >= 0),
        collate_seconds=t_collate,
    )
    return stats


# ---------------------------------------------------------------------------------------------------
# Learner side: run
# ---------------------------------------------------------------------------------------------------


#: Main-exploiter mode (A3): the run's own one-member population file, and the opponent server's name.
EXPLOITER_POPULATION = "exploiter_population.json"
EXPLOITER_SLOT = "main"
#: Prioritized starts (A2): the scores, written with each snapshot's state.json.
START_PRIORITIES_FILE = "start_priorities.json"


def _read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def _write_json(path: Path, data) -> None:
    """Write ``data`` atomically (a temporary file renamed over ``path``)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _mtime(path) -> Optional[float]:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return None


def _set_slot(handles, slot, member, slot_labels, slot_versions, slot_temperatures, cfg) -> None:
    """Load league ``member`` (a key: ``<path>`` or ``<path>@<T>``) into opponent server ``slot``: its member-change
    counter first (games that straddle the change are not credited), then the checkpoint, then the play
    temperature the workers read (moves answered between the reload and the temperature change -- a game that is
    not credited anyway -- may be sampled at the previous member's temperature)."""
    slot_versions[slot].value += 1
    handles[slot].reload(member_path(member))
    slot_temperatures[slot].value = member_temperature(member, cfg.opponent_temperature)
    slot_labels[slot] = member


def _save(model, path: Path, cfg: GenericPGConfig, role: str, update: int) -> None:
    from rl18xx.agent.alphazero import model_generic as mg

    mg.save_checkpoint(
        model,
        path,
        titles=[cfg.title],
        extra={"stage": "generic_policy_gradient", "role": role, "run": cfg.run_name, "update": int(update)},
    )


def _load(path, device):
    from rl18xx.agent.alphazero import model_generic as mg

    model, _ = mg.load_checkpoint(str(path), device)
    return model


def _server_factory(cfg: GenericPGConfig):
    from rl18xx.agent.alphazero import generic_inference as gi

    return gi.load_generic_server_model_cpu if cfg.server_cpu else gi.load_generic_server_model


def run(cfg: GenericPGConfig, resume: Optional[str] = None, explicit: Optional[set] = None) -> Path:
    """Run the generic policy-gradient stage; returns the run directory. ``resume``: a run directory to
    continue from its last snapshot (``explicit``: the options given on the command line, as the legacy
    stage's ``resume_overrides``)."""
    from rl18xx.agent.alphazero.generic_inference import start_generic_inference_server, use_tf32
    from rl18xx.agent.alphazero.inference_server import wait_checking_servers

    use_tf32()
    if resume:
        run_dir = Path(resume)
        state = json.loads((run_dir / "state.json").read_text())
        # a run from before the rules mix played base rules: its resume does too (unless --rules-mix is given)
        state["config"].setdefault("rules_mix", {"base": 1.0})
        overrides = pg.resume_overrides(cfg, state["config"], RESUME_SETTINGS, explicit)
        if overrides.get("rules_mix", "") is None:  # (a caller without a command line: only a mix it sets)
            overrides.pop("rules_mix")
        cfg = GenericPGConfig(**{**state["config"], **overrides})
        check_config(cfg)
        update = int(state["snapshot_update"])
        pool = list(state["pool"])
        kl_coef = float(state["kl_coef"])
        anchor_update = int(state.get("anchor_update", 0))
        league_state = state.get("league")
        priorities_state = _read_json(run_dir / state["start_priorities"]) if state.get("start_priorities") else None
        learner_path = run_dir / "learner" / f"{update}.pth"
        critic_path = run_dir / "critic" / f"{update}.pth"
    else:
        default_start_templates(cfg)
        name = cfg.run_name or datetime.now().strftime("gpg_%Y%m%d_%H%M%S")
        cfg.run_name = name
        run_dir = Path(cfg.out_dir) / name
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; pass --resume to continue it")
        if cfg.exploiter_of:
            if cfg.population:
                raise SystemExit("--exploiter-of plays the frozen main in every other seat: no --population")
            check_generic_checkpoint(cfg.exploiter_of, cfg.title)
            cfg.population = str(run_dir / EXPLOITER_POPULATION)
            cfg.league_add_snapshots = False
        check_config(cfg)
        run_dir.mkdir(parents=True)
        if cfg.exploiter_of:
            _write_json(
                run_dir / EXPLOITER_POPULATION,
                {"note": "main-exploiter mode: the frozen main", "members": [{"path": str(cfg.exploiter_of)}]},
            )
        update, kl_coef, anchor_update = 0, cfg.kl_coef, 0
        pool = [str(cfg.policy_checkpoint)] if cfg.seed_pool else []
        league_state = None
        priorities_state = None
        learner_path = Path(cfg.policy_checkpoint)
        critic_path = Path(cfg.value_checkpoint or cfg.policy_checkpoint)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    LOGGER.info(f"Generic policy-gradient run {run_dir} ({cfg.title}) from update {update}: {asdict(cfg)}")

    if cfg.threads:
        torch.set_num_threads(int(cfg.threads))
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    anchor_path = run_dir / "learner" / f"{anchor_update}.pth" if anchor_update else Path(cfg.policy_checkpoint)
    learner, anchor, critic = _load(learner_path, device), _load(anchor_path, device), _load(critic_path, device)
    for param in anchor.parameters():
        param.requires_grad_(False)
    reference = None
    if cfg.ref_kl_coef > 0:  # A1: the fixed reference policy of the piKL-style pull
        reference = _load(cfg.ref_checkpoint or cfg.policy_checkpoint, device)
        for param in reference.parameters():
            param.requires_grad_(False)
    learner_opt = torch.optim.AdamW(learner.parameters(), lr=cfg.lr, weight_decay=0.0)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, weight_decay=0.0)
    if resume and (run_dir / "optimizer.pth").exists():
        opt_state = torch.load(run_dir / "optimizer.pth", map_location="cpu")
        learner_opt.load_state_dict(opt_state["learner"])
        critic_opt.load_state_dict(opt_state["critic"])

    current_learner = run_dir / "learner" / f"{update}.pth"
    current_critic = run_dir / "critic" / f"{update}.pth"
    if not current_learner.exists():
        _save(learner, current_learner, cfg, "learner", update)
        _save(critic, current_critic, cfg, "critic", update)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(Path("runs") / "generic_pg" / run_dir.name))
    except Exception as e:  # TensorBoard is optional here
        LOGGER.warning(f"no TensorBoard writer: {e}")
    history = (run_dir / "history.jsonl").open("a")

    rng = np.random.default_rng(cfg.seed)
    league = league_for_run(cfg, league_state, run_dir)
    known_members = set(load_members(cfg.population)) if league else set()
    population_mtime = _mtime(cfg.population) if league else None
    if league and cfg.exploiter_of:
        # main-exploiter mode: one server holds the frozen main for every player count; it never changes
        slot_names = {n: [EXPLOITER_SLOT] for n in league.counts}
    else:
        slot_names = pg.league_slots(cfg) if league else {}
    if league:
        slot_labels = {s: league.sample(n, rng) for n, slots in slot_names.items() for s in slots}
        servers = {"learner": str(current_learner), **{s: member_path(m) for s, m in slot_labels.items()}}
    else:
        slot_labels = {}
        servers = {
            "learner": str(current_learner),
            "sl": str(cfg.opponent_checkpoint or cfg.policy_checkpoint),
            "pool": pool[-1] if pool else str(cfg.policy_checkpoint),
        }
    slot_versions = {slot: multiprocessing.Value("q", 0, lock=False) for slot in slot_labels}
    slot_temperatures = {
        slot: multiprocessing.Value("d", member_temperature(member, cfg.opponent_temperature), lock=False)
        for slot, member in slot_labels.items()
    }
    factory = _server_factory(cfg)
    handles = {
        name: start_generic_inference_server(
            num_workers=cfg.workers,
            checkpoint_path=path,
            model_factory=factory,
            batch_size=cfg.inference_batch_size,
            pipeline=cfg.server_pipeline,
            replicas=cfg.server_replicas if name == "learner" else 1,
            threads=cfg.server_threads,
            seed=None if cfg.seed is None else int(cfg.seed) * 1000 + 100 * i,
        )
        for i, (name, path) in enumerate(servers.items())
    }
    priorities = None
    if cfg.start_priority:  # A2: prioritized level replay over the human and mid-game starts
        from rl18xx.agent.alphazero.generic_play import StartPriorities

        priorities = StartPriorities.for_run(
            pg.player_counts(cfg),
            cfg.start_positions,
            cfg.midgame_positions if cfg.midgame_fraction > 0 else None,
            uniform=cfg.start_priority_uniform,
            staleness=cfg.start_priority_staleness,
            temperature=cfg.start_priority_temperature,
            ema=cfg.start_priority_ema,
            score=cfg.start_priority_score,
        )
        if priorities_state:
            priorities.load_state(priorities_state)
        LOGGER.info(
            "Prioritized starts: library %s",
            {n: {k: len(v) for k, v in lib.items()} for n, lib in sorted(priorities.libraries.items())},
        )
    task_counter = [0]
    pool_label = servers.get("pool")
    queues = {name: (h.request_q, h.reply_qs, h.ticket_q) for name, h in handles.items()}
    scores = pg._Scores(cfg.score_window)
    stop_file = run_dir / "STOP"
    schedule = pg.CountSchedule(pg.player_counts(cfg))
    mixed = bool(cfg.player_mix)

    def settings() -> dict:
        n = schedule.next()
        max_decisions, stall = pg.count_limits(cfg, n)
        task_counter[0] += 1
        return {
            "seed": None if cfg.seed is None else [int(cfg.seed), task_counter[0]],
            "start_picks": priorities.picks(n, cfg.games_per_task + 8, rng) if priorities else None,
            "title": cfg.title,
            "num_players": n,
            "positions_per_seat": cfg.positions_per_seat,
            "learner_seats": pg.learner_seats_for(cfg, n),
            "sl_opponent_fraction": cfg.sl_opponent_fraction,
            "pool_ready": bool(pool),
            "pool_label": pool_label,
            "learner_temperature": cfg.learner_temperature,
            "opponent_temperature": cfg.opponent_temperature,
            "opponent_price_eps": cfg.opponent_price_eps,
            "max_decisions": max_decisions,
            "start_positions": pg.count_path(cfg.start_positions, n),
            "rules_mix": dict(cfg.rules_mix),
            "random_start_fraction": cfg.random_start_fraction,
            "midgame_positions": pg.count_path(cfg.midgame_positions, n),
            "midgame_fraction": cfg.midgame_fraction,
            "auction_fraction": cfg.auction_fraction,
            "auction_unlock": cfg.auction_unlock,
            "auction_stall_decisions": stall,
            "redraw_all_pass": cfg.redraw_all_pass,
            "save_game_every": cfg.save_game_every,
            "reward": cfg.reward,
            "compress_requests": cfg.compress_requests,
            "league": (
                {
                    "slots": list(slot_names[n]),
                    "slot_labels": {slot: slot_labels[slot] for slot in slot_names[n]},
                    "slot_versions": {slot: int(slot_versions[slot].value) for slot in slot_names[n]},
                    "slot_temperatures": {slot: float(slot_temperatures[slot].value) for slot in slot_names[n]},
                    "learner_seat_weights": cfg.learner_seat_weights or pg.default_learner_seat_weights(n),
                }
                if league
                else None
            ),
        }

    buffer: List[PGRow] = []
    gen, total_games = Counter(), 0
    gen_started = run_started = time.time()
    terminations, lengths, starts, auction_lengths = Counter(), [], Counter(), []
    worker_seconds = Counter()
    prices = defaultdict(list)
    by_count = defaultdict(lambda: defaultdict(list))
    by_rules = defaultdict(lambda: defaultdict(list))
    rules_mixed = len(cfg.rules_mix) > 1
    all_pass_by_count = Counter()
    try:
        executor = ProcessPoolExecutor(
            max_workers=cfg.workers, initializer=worker_init, initargs=(queues, slot_versions, slot_temperatures)
        )
        try:
            futures = {
                executor.submit(play_generic_pg_games, cfg.games_per_task, settings())
                for _ in range(cfg.workers * cfg.tasks_in_flight_per_worker)
            }
            while update < cfg.max_updates and not stop_file.exists():
                done, futures = wait_checking_servers(futures, handles, executor, timeout=60)
                for future in done:
                    result = future.result()
                    gen["decisions"] += result["decisions"]
                    gen["skipped_starts"] += result.get("skipped_starts", 0)
                    gen["all_pass_redrawn"] += result.get("all_pass_redrawn", 0)
                    all_pass_by_count[int(result.get("num_players", 0))] += result.get("all_pass_redrawn", 0)
                    gen.update({f"worker_{k}": v for k, v in (result.get("stats") or {}).items()})
                    worker_seconds.update(result.get("worker_seconds") or {})
                    for record in result["games"]:
                        gen["games"] += 1
                        scores.add(record)
                        players = int(record["num_players"])
                        if league:
                            gen["league_mixed"] += bool(record.get("league_mixed"))
                            for member, pairwise in record.get("pairwise", []):
                                league.record(players, member, pairwise)
                        terminations[record["termination"]] += 1
                        starts[record.get("start_kind")] += 1
                        if priorities is not None:
                            priorities.observe_game(players, record.get("start_label"))
                        for company, price in (record.get("auction_prices") or {}).items():
                            prices[company].append(price)
                        lengths.append(record["decisions"])
                        auction_lengths.append(record.get("auction_decisions", 0))
                        if rules_mixed:
                            r = by_rules[record.get("rules", "base")]
                            r["decisions"].append(record["decisions"])
                            r["finished"].append(record["termination"] == "finished")
                            r["max_length"].append(record["termination"] == "max_length")
                        if mixed:
                            c = by_count[players]
                            c["decisions"].append(record["decisions"])
                            c["auction_decisions"].append(record.get("auction_decisions", 0))
                            c["max_length"].append(record["termination"] == "max_length")
                            c["auction_stall"].append(record["termination"] == "auction_stall")
                            if record["opponent"] == "league":
                                seats = record["learner_seats"]
                                won = sum(record["win_share"][x] for x in seats)
                                c["score"].append(won / (len(seats) / len(record["win_share"])))
                        buffer.extend(record["rows"])
                        if record.get("saved_game"):
                            opponent_checkpoint = (
                                servers.get("sl")
                                if record["opponent"] == "sl"
                                else record["pool_label"] if record["opponent"] == "pool" else "league"
                            )
                            try:
                                save_game(run_dir, record, update, str(current_learner), str(opponent_checkpoint), cfg)
                            except OSError as e:
                                LOGGER.warning(f"Could not save game {record['uid']}: {e}")
                    futures.add(executor.submit(play_generic_pg_games, cfg.games_per_task, settings()))
                if len(buffer) < cfg.rows_per_update:
                    continue

                t0 = time.time()
                baseline_stats = subtract_baselines(critic, buffer, cfg, device)
                baseline_seconds = time.time() - t0
                if priorities is not None:  # each start's score from this update's rows (reward - V(s))
                    baseline_stats["starts_scored_update"] = priorities.observe_rows(buffer)
                stats = _update(
                    learner, critic, anchor, learner_opt, critic_opt, buffer, cfg, kl_coef, device, rng, reference
                )
                stats.update(baseline_stats)
                stats["baseline_seconds"] = baseline_seconds
                train_seconds = time.time() - t0
                buffer = []
                update += 1

                previous = (current_learner, current_critic)
                current_learner = run_dir / "learner" / f"{update}.pth"
                current_critic = run_dir / "critic" / f"{update}.pth"
                _save(learner, current_learner, cfg, "learner", update)
                _save(critic, current_critic, cfg, "critic", update)
                handles["learner"].reload(str(current_learner))
                if cfg.anchor_every and update % cfg.anchor_every == 0:
                    anchor.load_state_dict(learner.state_dict())
                    anchor_update = update
                for path in previous:
                    if int(path.stem) % cfg.snapshot_every != 0 and path.parent.parent == run_dir and path.exists():
                        path.unlink()
                if update % cfg.snapshot_every == 0:
                    pool.append(str(current_learner))
                    if league and cfg.league_add_snapshots and update % cfg.league_snapshot_every == 0:
                        league.add(str(current_learner))
                        own = [m for m in league.members if Path(m).parent.parent == run_dir]
                        for member in own[: max(0, len(own) - cfg.league_max_snapshots)]:
                            league.remove(member)
                    torch.save(
                        {"learner": learner_opt.state_dict(), "critic": critic_opt.state_dict()},
                        run_dir / "optimizer.pth",
                    )
                    extra_state = {}
                    if priorities is not None:
                        _write_json(run_dir / START_PRIORITIES_FILE, priorities.state())
                        extra_state["start_priorities"] = START_PRIORITIES_FILE
                    (run_dir / "state.json").write_text(
                        json.dumps(
                            {
                                "config": asdict(cfg),
                                "snapshot_update": update,
                                "pool": pool,
                                "kl_coef": kl_coef,
                                "anchor_update": anchor_update,
                                "league": league.state() if league else None,
                                **extra_state,
                            },
                            indent=2,
                        )
                    )
                if not league and pool and update % cfg.pool_refresh_every == 0:
                    pool_label = pool[int(rng.integers(0, len(pool)))]
                    handles["pool"].reload(pool_label)
                reloaded = None
                if league and cfg.population_reload_every and update % cfg.population_reload_every == 0:
                    mtime = _mtime(cfg.population)
                    if mtime != population_mtime:
                        population_mtime = mtime
                        reloaded = reload_population(league, cfg, run_dir, known_members)
                        if reloaded["added"] or reloaded["removed"]:
                            LOGGER.info(f"population {cfg.population} reloaded: {reloaded}")
                        # a slot holding a member that left takes a new one now
                        for n, slots in slot_names.items():
                            for slot in slots:
                                if slot_labels[slot] in reloaded["removed"]:
                                    _set_slot(handles, slot, league.sample(n, rng), slot_labels, slot_versions,
                                              slot_temperatures, cfg)
                if league and update % cfg.slot_refresh_every == 0 and not cfg.exploiter_of:
                    for n, slots in slot_names.items():
                        slot = slots[(update // cfg.slot_refresh_every) % len(slots)]
                        _set_slot(handles, slot, league.sample(n, rng), slot_labels, slot_versions,
                                  slot_temperatures, cfg)

                if cfg.kl_target is not None:
                    if stats["kl_anchor"] > 1.5 * cfg.kl_target:
                        kl_coef *= 1.5
                    elif stats["kl_anchor"] < cfg.kl_target / 1.5:
                        kl_coef = max(kl_coef / 1.5, cfg.kl_coef_min)

                elapsed = time.time() - gen_started
                total_games += gen["games"]
                record = {
                    "update": update,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    **stats,
                    **scores.summary(),
                    "kl_coef": kl_coef,
                    "anchor_update": anchor_update,
                    "games": gen["games"],
                    "games_total": total_games,
                    "games_per_hour": gen["games"] / elapsed * 3600,
                    "games_per_hour_run": total_games / (time.time() - run_started) * 3600,
                    "decisions_per_second": gen["decisions"] / elapsed,
                    "mean_decisions": float(np.mean(lengths)) if lengths else 0.0,
                    "max_decisions_game": int(max(lengths, default=0)),
                    "max_auction_decisions": int(max(auction_lengths, default=0)),
                    "terminations": dict(terminations),
                    "starts": dict(starts),
                    "auction_games": max((len(v) for v in prices.values()), default=0),
                    **{f"auction_price_{c}": float(np.mean(v)) for c, v in sorted(prices.items())},
                    "auction_stalls": terminations.get("auction_stall", 0),
                    "skipped_starts": gen["skipped_starts"],
                    "all_pass_redrawn": gen["all_pass_redrawn"],
                    "all_pass_rate": gen["all_pass_redrawn"] / max(1, gen["all_pass_redrawn"] + gen["games"]),
                    "learner_decisions": gen["worker_learner_decisions"],
                    "learner_price_decisions": gen["worker_learner_price_decisions"],
                    "worker_seconds": {k: round(v, 1) for k, v in worker_seconds.items()},
                    "train_seconds": train_seconds,
                    "pool_size": len(pool),
                    "pool_opponent": pool_label if pool else None,
                }
                if stats.get("kl_ref") is not None:
                    record["ref_kl_coef"] = cfg.ref_kl_coef
                if reloaded is not None:
                    record["population_reloaded"] = reloaded
                if priorities is not None:
                    record["start_priority"] = priorities.summary()
                    record["start_picks"] = {
                        k[len("worker_start_") :]: v for k, v in gen.items() if k.startswith("worker_start_")
                    }
                if cfg.exploiter_of:
                    main = load_members(cfg.population)[0]
                    record["exploiter_of"] = cfg.exploiter_of
                    for n in league.counts:
                        key = "vs_main_rate" if len(league.counts) == 1 else f"p{n}_vs_main_rate"
                        record[key] = round(league.win_rate(n, main), 4)
                if league:
                    record["league_size"] = len(league.members)
                    for n in league.counts:
                        hardest = sorted(league.members, key=lambda m: league.win_rate(n, m))[:3]
                        key = "league_hardest" if len(league.counts) == 1 else f"p{n}_league_hardest"
                        record[key] = [[m, round(league.win_rate(n, m), 3)] for m in hardest]
                    record["league_slots"] = dict(slot_labels)
                    record["league_mixed_games"] = gen["league_mixed"]
                for n, c in sorted(by_count.items()):
                    record[f"p{n}_games"] = len(c["decisions"])
                    record[f"p{n}_mean_decisions"] = float(np.mean(c["decisions"]))
                    record[f"p{n}_max_decisions_game"] = int(max(c["decisions"]))
                    record[f"p{n}_max_auction_decisions"] = int(max(c["auction_decisions"], default=0))
                    record[f"p{n}_max_length"] = int(sum(c["max_length"]))
                    record[f"p{n}_auction_stalls"] = int(sum(c["auction_stall"]))
                    if c["score"]:
                        record[f"p{n}_score_vs_league_update"] = float(np.mean(c["score"]))
                if mixed:
                    for n in sorted(by_count):
                        record[f"p{n}_all_pass_redrawn"] = int(all_pass_by_count.get(n, 0))
                for name, r in sorted(by_rules.items()):
                    record[f"rules_{name}_games"] = len(r["decisions"])
                    record[f"rules_{name}_mean_decisions"] = float(np.mean(r["decisions"]))
                    record[f"rules_{name}_max_decisions_game"] = int(max(r["decisions"]))
                    record[f"rules_{name}_not_finished"] = int(len(r["finished"]) - sum(r["finished"]))
                    record[f"rules_{name}_max_length"] = int(sum(r["max_length"]))
                gen.clear()
                worker_seconds.clear()
                gen_started = time.time()
                terminations, lengths, starts, auction_lengths = Counter(), [], Counter(), []
                prices = defaultdict(list)
                by_count = defaultdict(lambda: defaultdict(list))
                by_rules = defaultdict(lambda: defaultdict(list))
                all_pass_by_count = Counter()
                history.write(json.dumps(record) + "\n")
                history.flush()
                if writer is not None:
                    for key, value in record.items():
                        if isinstance(value, (int, float)) and key != "update":
                            writer.add_scalar(f"GPG/{key}", value, update)
                kind = "league" if league else "sl"
                LOGGER.info(
                    f"update {update}: vs {kind} {record.get(f'score_vs_{kind}', float('nan')):.3f}"
                    f"±{record.get(f'score_vs_{kind}_se', float('nan')):.3f} "
                    f"(this update {record.get(f'score_vs_{kind}_update', float('nan')):.3f}), "
                    f"kl_anchor {stats['kl_anchor']:.4f}, entropy {stats['entropy']:.3f}, "
                    f"clip {stats['clip_frac']:.3f}, critic CE {stats.get('critic_ce', float('nan')):.4f}, "
                    f"{record['games_per_hour_run']:.0f} games/h, "
                    f"train {train_seconds:.0f}s"
                )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    finally:
        history.close()
        if writer is not None:
            writer.close()
        for handle in handles.values():
            handle.shutdown()
    return run_dir


__all__ = [
    "EXPLOITER_POPULATION",
    "GenericPGConfig",
    "PGRow",
    "RESUME_SETTINGS",
    "check_config",
    "collate_rows",
    "generic_pg_losses",
    "league_for_run",
    "load_member_min_shares",
    "load_members",
    "member_key",
    "member_path",
    "member_temperature",
    "play_generic_pg_games",
    "reload_population",
    "run",
    "subtract_baselines",
]
