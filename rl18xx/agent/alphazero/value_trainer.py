"""The value trainer (``main.py value-train``): a separate value network trained offline, in
cycles, over every game search learning saved (``--save-data``), and promoted only through a
gate. The generator follows the newest promoted value (``search-learning --value-follow``);
the daily report and the eval ladder read the run directory. docs/long_run_recipe.md
("2. Value", "Value rules", P3/P4, M8); the experiment it rests on is docs/value_gate.md, and
scripts/value_gate.py ``train`` is its prototype (same losses, optimizer and early stopping).

A cycle
-------
- **Trigger**: ``cycle_games`` trainable games arrived since the previous cycle's data cut, or
  ``cycle_hours`` passed since it started and ``min_new_games`` arrived; the first cycle as soon
  as ``min_new_games`` exist. Between checks the trainer polls the data directories every
  ``poll_seconds``.
- **Games**: the shards (``search_data``) of every ``--data`` directory, in the order they were
  begun (shard names start with the writer's start time) -- a game's age is its shard's place
  in that order. A per-shard index (``index/``: uids, action counts, players, when first seen)
  is built once from each shard's header, so choosing games never decompresses whole shards.
  Held out (never trained on): ``crc32(uid) % 1000 < held_out_per_mille`` (30, ~3%). Only
  4-player games are used.
- **Sample**: every trainable game among the newest ``recent_games``, plus as many drawn
  uniformly from the older trainable games (at most all of them); both scaled by the sample
  scale.
- **Positions** (:func:`game_positions`): a game's action log replayed in the Rust engine,
  ``positions_per_game`` decision positions past the auction (more than one legal move) drawn
  uniformly over the game, seeded by ``crc32(uid)`` and the pass (a new draw each pass and
  cycle, reproducible), encoded canonically (the mover first). Every seat's positions, whoever
  played it (league members too).
- **Targets** (outcome only, as scripts/value_gate.py): win CE on the canonical win share +
  ``score_weight`` x MSE of the score head against placement points (1 / 0.5 / 0.2 / 0).
- **Training**: warm start from the current promoted value; AdamW at ``lr`` on the whole
  network (weight decay 0), batch ``batch_size``, bf16 autocast on CUDA only, grad-norm clip
  1.0 per group, ``model.eval()`` throughout (no dropout: the network trained is the one
  searched with). Streamed: a spawn pool re-encodes the chosen games in tasks of
  ~``task_games`` (grouped by shard) into a bounded shuffle buffer (``buffer_positions``);
  ``epochs`` passes over the sample, each in a new order. Early stopping on the rolling
  held-out set every ``eval_every`` steps (the best state is kept, the start included;
  ``patience`` evaluations without a gain > ``min_improvement`` end it).
- **Held-out sets**: *fixed* -- the value gate's test games (``fixed_data``: 3,100 games,
  49,600 positions; comparable for the whole run); *rolling* -- the held-out games among the
  newest ``rolling_games``, ``rolling_positions_per_game`` positions each (the same positions
  every cycle), at most ``rolling_max_positions``, encoded once per cycle. ``select_share`` (0.5)
  keeps that share of the rolling games (by hash) for early stopping only, out of the gate (the
  gate otherwise judges the candidate on the positions that chose it: a small optimistic bias).
- **Gate** (:func:`gate_decision`; candidate and current on the same positions, differences
  paired, SE clustered by game): promote iff the rolling difference <= -``promote_z`` SE and
  the fixed one <= +``promote_z`` SE. *Diverged* (the hard stop the report alerts on): the
  candidate's rolling CE above the current's by > ``diverge_ce``, or training ending above its
  start by that much; never promoted. ``escalate_after`` (3) failures in a row double the cycle
  interval (hours and games) and the sample; twice that many add "review the data mix" to the
  reason. A promotion resets both and the count.

Run directory (other components read these: keep the names)
-----------------------------------------------------------
- ``config.json``: the trainer's settings (:class:`ValueTrainerConfig`).
- ``promoted.json``: ``{"current": v, "versions": [{"version", "path": "promoted/<v>.pth",
  "cycle", "time", "games_total", "fixed_ce", "rolling_ce", "candidate"}, ...]}``, paths
  relative to the run directory, rewritten atomically (temporary file, fsync, rename).
  Version 0 is ``--init`` (cycle 0, its fixed CE measured at creation, rolling CE null).
- ``promoted/<v>.pth``: the promoted values; ``candidates/<cycle>.pth``: every cycle's candidate
  (a promoted one is hard-linked into ``promoted/``; the retention job may delete unpromoted
  candidates after 7 days).
- ``cycles.jsonl``: one line per finished cycle (:data:`CYCLE_FIELDS`, plus diagnostics);
  ``consecutive_failures``, ``cycle_hours`` and ``sample_scale`` are the values after the
  cycle (governing the next; ``sample_scale_used`` is the cycle's own).
- ``state.json``: what a restart resumes from (written after the cycle's line; a restart also
  reconciles it with the last line). A cycle interrupted before its line is redone (after a
  crash between a promotion and the line, the redone cycle starts from that promotion).
- ``status.json``: a heartbeat (stage, cycle, step, time) at every poll and evaluation.
- ``index/``: the per-shard indexes. ``STOP``: touch it to end the trainer cleanly at its next
  poll (a running cycle finishes first); SIGTERM abandons a running cycle at once (exit 0).
"""

from __future__ import annotations

import json
import logging
import lzma
import os
import shutil
import signal
import struct
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

NUM_PLAYERS = 4
DEFAULT_INIT = "logs/value_gate/runs/base_36989/learner/0.pth"
DEFAULT_FIXED = "logs/value_gate/data"
ROLLING_PASS = 2**31 - 1  # the rolling set's "pass": a held-out game gives the same positions every cycle
DRAWS_PER_POSITION = 3  # candidate log positions per wanted one, filtered to decisions (value_gate.replay_game)
# A pool task (~48 games, ~1 s) not done after this long was lost with its worker (an OOM kill: Pool never
# returns it): the cycle fails and a restart redoes it, instead of waiting forever.
TASK_TIMEOUT = 900.0
POSITION_KEYS = ("game_states", "nodes", "win", "placement")
CYCLE_FIELDS = (
    "cycle", "started", "time", "seconds", "games_total", "games_trained", "positions_trained", "steps", "epochs",
    "start_version", "candidate", "fixed_n", "fixed_games", "fixed_ce", "fixed_ce_current", "fixed_diff",
    "fixed_diff_se", "rolling_n", "rolling_games", "rolling_ce", "rolling_ce_current", "rolling_diff",
    "rolling_diff_se", "promoted", "version", "diverged", "reason", "consecutive_failures", "cycle_hours",
    "sample_scale", "encode_seconds", "train_seconds",
)  # fmt: skip


@dataclass
class ValueTrainerConfig:
    run_dir: str
    data: list = field(default_factory=list)  # search_data directories
    init: str = DEFAULT_INIT  # the start value (promoted version 0; first run only)
    fixed_data: Optional[str] = DEFAULT_FIXED  # the value gate's position shards (test split); None: no fixed set
    positions_per_game: int = 16
    held_out_per_mille: int = 30
    rolling_games: int = 200_000
    rolling_positions_per_game: int = 8
    rolling_max_positions: int = 80_000
    # Half the rolling games choose the early-stopped state, the other half judge it: the gate
    # stays an unbiased held-out test (~6k held-out games among the newest 200k).
    select_share: float = 0.5
    cycle_games: int = 40_000
    cycle_hours: float = 12.0
    min_new_games: int = 2_000
    poll_seconds: float = 60.0
    recent_games: int = 200_000
    sample_scale: float = 1.0
    escalate_after: int = 3
    lr: float = 3e-5
    batch_size: int = 256
    score_weight: float = 1.0
    epochs: float = 1.0
    eval_every: int = 2_000
    patience: int = 4
    min_improvement: float = 1e-4
    max_steps: int = 0  # 0: no cap
    promote_z: float = 2.0
    diverge_ce: float = 0.01
    workers: int = 8  # 0: encode in the trainer's process
    task_games: int = 48
    buffer_positions: int = 200_000
    eval_batch_size: int = 1024
    device: Optional[str] = None  # default: cuda if available
    seed: int = 0


class Abandoned(Exception):
    """SIGTERM during a cycle: it is dropped; a restart redoes it."""


# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path, obj) -> None:
    """``obj`` as JSON at ``path``: a temporary file, synced, renamed (a reader sees the old file or the new)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _sync_dir(path.parent)


def append_jsonl(path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path) -> list:
    """A JSONL file's records (a torn last line, from a crash mid-append, is skipped)."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def repair_jsonl(path) -> None:
    """Cut a torn last line (no newline: a crash mid-append) so the next append starts a clean line."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    keep = data.rfind(b"\n") + 1
    LOGGER.warning(f"{path.name}: dropping a torn last line ({len(data) - keep} bytes)")
    with open(path, "r+b") as f:
        f.truncate(keep)
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Shards: headers and action logs without the search columns
# ---------------------------------------------------------------------------


def _take(decompressor, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = decompressor.decompress(b"", max_length=n - len(out))
        if not chunk:
            raise ValueError("search-data shard ends early")
        out += chunk
    return bytes(out)


def read_shard_actions(path, actions: bool = True) -> tuple:
    """``(header, logs)`` of a ``search_data`` shard: its header (the games' metadata with their
    ``counts``) and, with ``actions``, each game's action log (JSON bytes), decompressing only
    the stream's head -- the header and the logs come first, the search columns (~30% of the
    bytes, most of the decoding) are never read. Same bytes as ``search_data.read_shard``."""
    from rl18xx.agent.alphazero import search_data as sd

    d = lzma.LZMADecompressor()
    head = d.decompress(Path(path).read_bytes(), max_length=len(sd.MAGIC) + 8)
    if head[: len(sd.MAGIC)] != sd.MAGIC:
        raise ValueError(f"{path}: not a search-data shard")
    (n,) = struct.unpack_from("<Q", head, len(sd.MAGIC))
    header = json.loads(_take(d, n))
    if header.get("format") != sd.FORMAT:
        raise ValueError(f"{path}: search-data format {header.get('format')}, this reader reads {sd.FORMAT}")
    if not actions:
        return header, None
    layout = header["columns"]["actions"]
    blob = _take(d, layout["offset"] + layout["nbytes"])[layout["offset"] :]
    logs, at = [], 0
    for meta in header["games"]:
        size = meta["counts"]["action_bytes"]
        logs.append(blob[at : at + size])
        at += size
    return header, logs


@dataclass
class ShardIndex:
    """One shard's games: crc32 of the uid, action count, players (the index file keeps the uids)."""

    key: tuple  # (shard name, directory): the order shards were begun
    path: Path
    first_seen: float
    crc: np.ndarray
    actions: np.ndarray
    players: np.ndarray

    def trainable(self, per_mille: int) -> int:
        return int(np.count_nonzero((self.crc % 1000 >= per_mille) & (self.players == NUM_PLAYERS)))


def _index_dir(run_dir: Path, data_dir: Path) -> Path:
    resolved = str(Path(data_dir).resolve())
    return run_dir / "index" / f"{Path(resolved).name}-{zlib.crc32(resolved.encode()):08x}"


def index_shard(path: Path, index_dir: Path) -> ShardIndex:
    """The shard's index, from ``index_dir`` or built from its header (and saved there)."""
    file = index_dir / f"{path.name}.json"
    size = path.stat().st_size
    record = None
    if file.exists():
        try:
            record = json.loads(file.read_text())
            if record.get("bytes") != size:
                record = None
        except ValueError:
            record = None
    if record is None:
        header, _ = read_shard_actions(path, actions=False)
        games = header["games"]
        record = {
            "shard": path.name, "directory": str(path.parent), "bytes": size, "first_seen": time.time(),
            "uids": [g["uid"] for g in games], "actions": [int(g["actions"]) for g in games],
            "players": [int(g["num_players"]) for g in games],
        }  # fmt: skip
        index_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(file, record)
    return ShardIndex(
        key=(path.name, str(path.parent)),
        path=path,
        first_seen=float(record["first_seen"]),
        crc=np.array([zlib.crc32(u.encode()) for u in record["uids"]], dtype=np.uint32),
        actions=np.array(record["actions"], dtype=np.int32),
        players=np.array(record["players"], dtype=np.int8),
    )


@dataclass
class GameIndex:
    """Every game of a cycle's data cut, oldest first: its shard (into ``paths``) and slot there."""

    paths: list
    shard: np.ndarray
    slot: np.ndarray
    crc: np.ndarray
    players: np.ndarray

    @classmethod
    def from_shards(cls, shards: list) -> "GameIndex":
        shards = sorted(shards, key=lambda s: s.key)
        sizes = [len(s.crc) for s in shards]
        empty = np.zeros(0, dtype=np.int32)
        return cls(
            paths=[s.path for s in shards],
            shard=np.repeat(np.arange(len(shards), dtype=np.int32), sizes),
            slot=np.concatenate([np.arange(n, dtype=np.int32) for n in sizes]) if shards else empty,
            crc=np.concatenate([s.crc for s in shards]) if shards else empty.astype(np.uint32),
            players=np.concatenate([s.players for s in shards]) if shards else empty.astype(np.int8),
        )

    def __len__(self) -> int:
        return len(self.crc)

    def held_out(self, per_mille: int) -> np.ndarray:
        return self.crc % 1000 < per_mille

    def trainable(self, per_mille: int) -> np.ndarray:
        """Games that may be trained on, oldest first."""
        return np.flatnonzero(~self.held_out(per_mille) & (self.players == NUM_PLAYERS))


def choose_sample(index: GameIndex, per_mille: int, recent_games: int, scale: float, rng) -> tuple:
    """``(recent, old)``: every trainable game among the newest ``recent_games * scale``, and as
    many (at most all) drawn uniformly from the older trainable games, each oldest first."""
    trainable = index.trainable(per_mille)
    r = max(0, int(round(recent_games * scale)))
    cut = max(0, len(trainable) - r)
    recent, older = trainable[cut:], trainable[:cut]
    old = np.sort(rng.choice(older, size=min(r, len(older)), replace=False)) if len(older) else older
    return recent, old


def rolling_games(index: GameIndex, cfg: ValueTrainerConfig) -> np.ndarray:
    """The rolling held-out games: held out, among the newest ``rolling_games``, at most
    ``rolling_max_positions / rolling_positions_per_game`` of them (the newest), oldest first."""
    newest = np.arange(max(0, len(index) - cfg.rolling_games), len(index))
    held = newest[index.held_out(cfg.held_out_per_mille)[newest] & (index.players[newest] == NUM_PLAYERS)]
    cap = cfg.rolling_max_positions // max(1, cfg.rolling_positions_per_game)
    return held[max(0, len(held) - cap) :]


def selection_games(crc: np.ndarray, share: float) -> np.ndarray:
    """Rolling games kept for early stopping only (``select_share``), by a hash independent of the split."""
    return (crc // 1000) % 1000 < int(round(share * 1000))


def make_tasks(index: GameIndex, games, k: int, pass_id: int, task_games: int) -> list:
    """Encoding tasks for ``games`` (GameIndex positions, in the wanted order): grouped by shard
    (a task decompresses each of its shards once), shards in the order of their first game,
    ~``task_games`` games a task."""
    games = np.asarray(games, dtype=np.int64)
    if not len(games):
        return []
    shards = index.shard[games]
    _, first = np.unique(shards, return_index=True)
    groups: dict = {int(s): [] for s in shards[np.sort(first)]}
    for g, s in zip(games.tolist(), shards.tolist()):
        groups[s].append(g)
    tasks, pieces, count = [], [], 0
    for s, ids in groups.items():
        while ids:
            take, ids = ids[: task_games - count], ids[task_games - count :]
            pieces.append((str(index.paths[s]), [int(index.slot[g]) for g in take], take))
            count += len(take)
            if count >= task_games:
                tasks.append({"k": k, "pass_id": pass_id, "pieces": pieces})
                pieces, count = [], 0
    if pieces:
        tasks.append({"k": k, "pass_id": pass_id, "pieces": pieces})
    return tasks


# ---------------------------------------------------------------------------
# Positions (pool workers: no torch here)
# ---------------------------------------------------------------------------


def game_positions(meta: dict, actions: list, k: int, pass_id: int) -> dict:
    """``k`` decision positions of a saved game past the auction (more than one legal move),
    uniform over the game: ``index`` (the log position: the state before that action),
    ``game_states``, ``nodes`` (float16) and ``rotation`` of ``encode_canonical_py``. Drawn by
    ``(crc32(uid), pass_id)``: ``DRAWS_PER_POSITION * k`` log positions, the decisions among
    them, ``k`` of those (fewer if the game has fewer)."""
    from rl18xx.agent.alphazero.start_positions import new_game

    rng = np.random.default_rng([zlib.crc32(meta["uid"].encode()), int(pass_id)])
    n = len(actions)
    drawn = np.sort(rng.choice(n, size=min(n, DRAWS_PER_POSITION * k), replace=False)) if n and k > 0 else []
    wanted, last = set(int(i) for i in drawn), int(drawn[-1]) if len(drawn) else -1
    rs = new_game(int(meta["num_players"]), bool(meta["auction_unlock"]))._game
    rows = []
    for i in range(last + 1):
        if i in wanted and rs.round.round_type != "Auction" and len(rs.factored_legal_indices()) > 1:
            state, nodes, _, rotation, _ = rs.encode_canonical_py()
            rows.append((i, np.asarray(state, np.float32), np.asarray(nodes).astype(np.float16), int(rotation)))
        rs.process_action(actions[i])
    if len(rows) > k:
        rows = [rows[int(j)] for j in np.sort(rng.choice(len(rows), size=k, replace=False))]
    return {
        "index": np.array([r[0] for r in rows], dtype=np.int64),
        "game_states": np.stack([r[1] for r in rows]) if rows else None,
        "nodes": np.stack([r[2] for r in rows]) if rows else None,
        "rotation": np.array([r[3] for r in rows], dtype=np.int8),
    }


def _worker_init() -> None:
    import sys

    os.environ["OMP_NUM_THREADS"] = "1"
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)


def encode_task(task: dict) -> dict:
    """Pool task: its games' positions, stacked, with each position's game (GameIndex position)
    and each game's outcome (win share, net worth) -- targets are made by the trainer."""
    started = time.time()
    states, nodes, rotation, local, games, win_share, net_worth = [], [], [], [], [], [], []
    skipped = 0
    for path, slots, ids in task["pieces"]:
        try:
            header, logs = read_shard_actions(path)
        except Exception as e:  # a shard that can't be read (moved, damaged): its games are skipped
            LOGGER.warning(f"{path}: not read ({e!r})")
            skipped += len(slots)
            continue
        for slot, gid in zip(slots, ids):
            meta = header["games"][slot]
            try:
                pos = game_positions(meta, json.loads(logs[slot]), task["k"], task["pass_id"])
            except Exception as e:  # a log the engine no longer replays
                LOGGER.debug(f"{meta.get('uid')}: skipped ({e!r})")
                skipped += 1
                continue
            if pos["game_states"] is None:
                continue
            local.append(np.full(len(pos["index"]), len(games), dtype=np.int32))
            games.append(gid)
            win_share.append(meta["win_share"])
            net_worth.append(meta["net_worth"])
            states.append(pos["game_states"])
            nodes.append(pos["nodes"])
            rotation.append(pos["rotation"])
    out = {"games": np.array(games, dtype=np.int64), "skipped": skipped, "seconds": time.time() - started,
           "attempted": sum(len(slots) for _, slots, _ in task["pieces"])}
    if games:
        out.update(
            game_states=np.concatenate(states), nodes=np.concatenate(nodes), rotation=np.concatenate(rotation),
            local=np.concatenate(local), win_share=np.array(win_share, dtype=np.float32),
            net_worth=np.array(net_worth, dtype=np.float64),
        )  # fmt: skip
    return out


def canonical_targets(win_share: np.ndarray, net_worth: np.ndarray, local: np.ndarray, rotation: np.ndarray):
    """Per position ``(win, placement)`` in its canonical frame (``np.roll(x, -rotation)``) from
    its game's (``local``) win share and net worth (placement points by net worth)."""
    from rl18xx.agent.alphazero.policy_gradient import placement_points

    placement = np.stack([placement_points(w) for w in net_worth]).astype(np.float32)
    seats = (np.arange(win_share.shape[1])[None, :] + rotation.astype(np.int64)[:, None]) % win_share.shape[1]
    win = np.take_along_axis(win_share[local], seats, axis=1)
    return win.astype(np.float32), np.take_along_axis(placement[local], seats, axis=1)


def result_positions(result: dict) -> Optional[dict]:
    """An :func:`encode_task` result as training positions (``POSITION_KEYS`` + ``game``)."""
    if not len(result["games"]):
        return None
    win, placement = canonical_targets(result["win_share"], result["net_worth"], result["local"], result["rotation"])
    return {
        "game_states": result["game_states"], "nodes": result["nodes"], "win": win, "placement": placement,
        "game": result["games"][result["local"]],
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Encoding pool and the shuffle buffer
# ---------------------------------------------------------------------------


class _Done:
    def __init__(self, value):
        self.value = value

    def ready(self) -> bool:
        return True

    def wait(self, timeout=None) -> None:
        pass

    def get(self):
        return self.value


class InlinePool:
    """``workers`` 0: tasks run in the trainer's process when submitted (tests, debugging)."""

    def apply_async(self, fn, args):
        return _Done(fn(*args))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_pool(workers: int):
    """A spawn pool of ``workers`` single-threaded encoders (or :class:`InlinePool`)."""
    if workers <= 0:
        return InlinePool()
    from multiprocessing import get_context

    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})  # the workers inherit it at spawn
    try:
        return get_context("spawn").Pool(workers, initializer=_worker_init)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class Stream:
    """Encoded positions from ``tasks``, at most ``max_inflight`` tasks submitted or waiting to
    be consumed (``staged``), so the pool never runs ahead of training by more than that."""

    def __init__(self, pool, tasks: Iterator, max_inflight: int, check=lambda: None, timeout: float = TASK_TIMEOUT):
        self.pool, self.tasks, self.max_inflight, self.check = pool, iter(tasks), max(1, max_inflight), check
        self.timeout = timeout
        self.inflight: list = []  # (async result, submitted)
        self.staged: list = []  # [positions, offset] not yet in the buffer
        self.more = True
        self.games: list = []  # each encoded game (with repeats across passes)
        self.attempted = 0  # games of the finished tasks (with the skipped and those without a position)
        self.positions = 0  # encoded
        self.skipped = 0
        self.worker_seconds = 0.0

    @property
    def exhausted(self) -> bool:
        return not (self.more or self.inflight or self.staged)

    def _submit(self) -> None:
        while self.more and len(self.inflight) + len(self.staged) < self.max_inflight:
            task = next(self.tasks, None)
            if task is None:
                self.more = False
                break
            self.inflight.append((self.pool.apply_async(encode_task, (task,)), time.time()))

    def collect(self, block: bool = False) -> None:
        """Move finished tasks into ``staged`` (``block``: wait for at least one if any is running)."""
        self._submit()
        while True:
            done = [x for x in self.inflight if x[0].ready()]
            for x in done:
                self.inflight.remove(x)
                result = x[0].get()
                self.skipped += result["skipped"]
                self.worker_seconds += result["seconds"]
                self.attempted += result["attempted"]
                self.games.append(result["games"])
                positions = result_positions(result)
                if positions is not None:
                    self.positions += len(positions["win"])
                    self.staged.append([positions, 0])
            self._submit()
            if done or not block or not self.inflight:
                return
            self.inflight[0][0].wait(0.5)
            self.check()
            if time.time() - self.inflight[0][1] > self.timeout:
                raise RuntimeError(f"value trainer: an encoding task has run {self.timeout:.0f}s (a worker lost?)")

    def drain(self) -> Optional[dict]:
        """Every remaining task's positions, concatenated (the rolling set)."""
        parts = []
        while True:
            self.collect(block=True)
            parts.extend(p for p, _ in self.staged)
            self.staged = []  # unbounded here: the next tasks go in
            if not (self.more or self.inflight):
                break
        if not parts:
            return None
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


class ShuffleBuffer:
    """A fixed-size pool of positions: batches are drawn uniformly from it and new positions
    take the drawn ones' places (each position is drawn once)."""

    def __init__(self, capacity: int, example: dict):
        self.capacity, self.n = int(capacity), 0
        self.arrays = {k: np.empty((self.capacity,) + example[k].shape[1:], example[k].dtype) for k in POSITION_KEYS}

    def add(self, positions: dict, start: int = 0) -> int:
        """Copy ``positions[start:]`` in while there is room; returns how many."""
        k = min(self.capacity - self.n, len(positions["win"]) - start)
        for key, array in self.arrays.items():
            array[self.n : self.n + k] = positions[key][start : start + k]
        self.n += k
        return k

    def take(self, k: int, rng) -> dict:
        idx = rng.choice(self.n, size=k, replace=False)
        out = {key: array[idx] for key, array in self.arrays.items()}
        rest = self.n - k
        tail = np.arange(rest, self.n)
        holes, movers = idx[idx < rest], tail[~np.isin(tail, idx)]
        for array in self.arrays.values():
            array[holes] = array[movers]
        self.n = rest
        return out


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def per_position_ce(model, arrays: dict, device, batch_size: int = 1024) -> np.ndarray:
    """Win CE of ``model`` (fp32, no autocast) at each position (canonical frame, first 4 seats)."""
    import torch

    out = []
    with torch.no_grad():
        for b in range(0, len(arrays["win"]), batch_size):
            gs = torch.from_numpy(np.ascontiguousarray(arrays["game_states"][b : b + batch_size])).to(device).float()
            nodes = torch.from_numpy(np.ascontiguousarray(arrays["nodes"][b : b + batch_size])).to(device).float()
            win = torch.from_numpy(np.ascontiguousarray(arrays["win"][b : b + batch_size])).to(device).float()
            _, logits, _, _ = model(gs, nodes, value_only=True)
            logp = torch.log_softmax(logits.float()[:, :NUM_PLAYERS], dim=1)
            out.append((-(win * logp).sum(1)).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_step(model, opt, batch: dict, device, score_weight: float) -> tuple:
    """One AdamW step on win CE + ``score_weight`` x placement MSE (scripts/value_gate.py ``train``)."""
    import torch
    import torch.nn.functional as F

    def t(key):
        return torch.from_numpy(batch[key]).to(device).float()

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        _, win_logits, score, _ = model(t("game_states"), t("nodes"), value_only=True)
    logp = torch.log_softmax(win_logits.float()[:, :NUM_PLAYERS], dim=1)
    ce = -(t("win") * logp).sum(1).mean()
    mse = F.mse_loss(score.float()[:, :NUM_PLAYERS], t("placement"))
    loss = ce + score_weight * mse
    if not torch.isfinite(loss):
        raise RuntimeError("value trainer: non-finite loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for group in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
    opt.step()
    return float(ce.detach()), float(mse.detach())


def load_fixed(data_dir, split: str = "test") -> Optional[dict]:
    """The value gate's positions of the ``split`` games (scripts/value_gate.py ``build``):
    ``POSITION_KEYS`` (already canonical; nodes float16) and ``game``; only those arrays read."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    wanted = np.array([g["split"] == split for g in manifest["games"]], dtype=bool)
    parts: dict = {k: [] for k in (*POSITION_KEYS, "game")}
    for name in manifest["shards"]:
        with np.load(data_dir / "shards" / name) as z:
            game = z["game"]
            rows = np.flatnonzero(wanted[game])
            if not len(rows):
                continue
            for key in POSITION_KEYS:
                parts[key].append(z[key][rows])
            parts["game"].append(game[rows])
    if not parts["game"]:
        return None
    return {k: np.concatenate(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def paired_stats(ce_new, ce_old, games) -> dict:
    """Candidate minus current on the same positions: ``n``, ``games``, the two mean CEs, the
    mean paired difference and its SE clustered by game (a game's positions share an outcome):
    sqrt(G / (G - 1) * sum_g (S_g - n_g * diff)^2) / N, S_g the game's summed differences --
    with equal positions per game, the std of the per-game mean differences / sqrt(G)."""
    ce_new, ce_old = np.asarray(ce_new, np.float64), np.asarray(ce_old, np.float64)
    n = len(ce_new)
    if n == 0:
        return {"n": 0, "games": 0, "ce": None, "ce_current": None, "diff": None, "se": None}
    d = ce_new - ce_old
    _, inverse, counts = np.unique(np.asarray(games), return_inverse=True, return_counts=True)
    g = len(counts)
    diff = float(d.mean())
    resid = np.bincount(inverse, weights=d, minlength=g) - counts * diff
    se = float(np.sqrt(g / (g - 1) * np.sum(resid**2)) / n) if g > 1 else None
    return {"n": int(n), "games": int(g), "ce": float(ce_new.mean()), "ce_current": float(ce_old.mean()),
            "diff": diff, "se": se}  # fmt: skip


def gate_decision(rolling: Optional[dict], fixed: Optional[dict], z: float = 2.0, diverge: float = 0.01,
                  start_ce: Optional[float] = None, final_ce: Optional[float] = None) -> tuple:
    """``(promoted, diverged, reason)`` from :func:`paired_stats` of the rolling and fixed sets
    (``fixed`` None: no fixed set). ``start_ce`` / ``final_ce``: the early-stopping set's CE at
    the start and at the end of training (the last state, not the best)."""
    if rolling is None or not rolling["n"]:
        return False, False, "no rolling held-out positions"
    if rolling["ce"] > rolling["ce_current"] + diverge:
        return False, True, f"diverged: rolling CE {rolling['ce']:.4f}, the current value's {rolling['ce_current']:.4f}"
    if start_ce is not None and final_ce is not None and final_ce > start_ce + diverge:
        return False, True, f"diverged: training ended at rolling CE {final_ce:.4f}, started at {start_ce:.4f}"
    se = rolling["se"]
    if se is None or not (rolling["diff"] < 0 and rolling["diff"] <= -z * se):
        se_text = "n/a" if se is None else f"{se:.4f}"
        return False, False, f"rolling gain too small: {rolling['diff']:+.4f} (SE {se_text}, needs <= -{z:g} SE)"
    if fixed is not None and fixed["n"]:
        limit = 0.0 if fixed["se"] is None else z * fixed["se"]
        if fixed["diff"] > limit:
            return False, False, f"fixed set worse: {fixed['diff']:+.4f} (SE {fixed['se'] or 0:.4f}, limit +{z:g} SE)"
    return True, False, "promoted"


def escalation(cfg: ValueTrainerConfig, failures: int) -> dict:
    """The schedule after ``failures`` consecutive unpromoted cycles: from ``escalate_after`` on,
    twice the interval (hours and games) and twice the sample; from twice that, a review."""
    factor = 2 if failures >= cfg.escalate_after else 1
    return {
        "cycle_hours": cfg.cycle_hours * factor,
        "cycle_games": cfg.cycle_games * factor,
        "sample_scale": cfg.sample_scale * factor,
        "review": failures >= 2 * cfg.escalate_after,
    }


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


class ValueTrainer:
    """One run directory (module docstring). :meth:`serve` runs it until STOP / SIGTERM."""

    def __init__(self, cfg: ValueTrainerConfig):
        if not cfg.data:
            raise ValueError("value trainer: no --data directory")
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.terminating = False
        self.shards: dict = {}  # key -> ShardIndex
        self.last_cycle: dict = {}  # the last cycle's games (tests, diagnostics)
        self._fixed_cache: dict = {}  # version -> per-position fixed CE
        self._device = None
        for sub in ("promoted", "candidates", "index"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        repair_jsonl(self.run_dir / "cycles.jsonl")
        self._write_config()
        self.state = self._load_state()
        self.fixed = load_fixed(cfg.fixed_data) if cfg.fixed_data else None
        if self.fixed is not None:
            LOGGER.info(f"fixed held-out set: {len(self.fixed['win'])} positions, "
                        f"{len(np.unique(self.fixed['game']))} games ({cfg.fixed_data})")
        self.scan()
        if not (self.run_dir / "promoted.json").exists():
            self._create_version_zero()

    # -- run directory ---------------------------------------------------------------

    def _write_config(self) -> None:
        path = self.run_dir / "config.json"
        new = asdict(self.cfg)
        if path.exists():
            try:
                old = json.loads(path.read_text())
            except ValueError:
                old = {}
            changed = sorted(k for k in new if old.get(k) != new[k])
            if changed:
                LOGGER.warning(f"resuming {self.run_dir} with changed settings: {changed}")
            else:
                LOGGER.info(f"resuming {self.run_dir}")
        write_json_atomic(path, new)

    def _load_state(self) -> dict:
        state = {"cycle": 0, "data_cut": None, "last_cycle_started": None, "consecutive_failures": 0}
        path = self.run_dir / "state.json"
        if path.exists():
            state.update(json.loads(path.read_text()))
        cycles = read_jsonl(self.run_dir / "cycles.jsonl")
        if cycles and cycles[-1]["cycle"] > state["cycle"]:  # the cycle's line was written, its state not
            last = cycles[-1]
            state.update(cycle=last["cycle"], data_cut=last.get("data_cut"), consecutive_failures=last[
                "consecutive_failures"], last_cycle_started=last.get("started_ts"))
            self._write_state(state)
        return state

    def _write_state(self, state: dict) -> None:
        state = dict(state)
        state.update({k: v for k, v in escalation(self.cfg, state["consecutive_failures"]).items()})
        write_json_atomic(self.run_dir / "state.json", state)

    def read_promoted(self) -> dict:
        return json.loads((self.run_dir / "promoted.json").read_text())

    def _create_version_zero(self) -> None:
        init = Path(self.cfg.init)
        target = self.run_dir / "promoted" / "0.pth"
        tmp = target.with_name(".0.pth.tmp")
        shutil.copyfile(init, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, target)
        fixed_ce = None
        if self.fixed is not None:
            fixed_ce = float(self._fixed_ce(self._load_model(target), 0).mean())
        entry = {"version": 0, "path": "promoted/0.pth", "cycle": 0, "time": _now(),
                 "games_total": self.games_total(), "fixed_ce": fixed_ce, "rolling_ce": None, "source": str(init)}
        write_json_atomic(self.run_dir / "promoted.json", {"current": 0, "versions": [entry]})
        LOGGER.info(f"promoted version 0 = {init} (fixed CE {fixed_ce})")

    def write_status(self, stage: str, **extra) -> None:
        status = {"time": _now(), "ts": time.time(), "pid": os.getpid(), "stage": stage,
                  "cycle": self.state["cycle"] + (1 if stage not in ("waiting", "stopped") else 0), **extra}
        try:
            write_json_atomic(self.run_dir / "status.json", status)
        except OSError as e:  # a heartbeat must not stop training
            LOGGER.warning(f"status.json not written: {e!r}")

    # -- data ------------------------------------------------------------------------------

    def scan(self) -> None:
        """Index new shards in every data directory; forget the ones that are gone."""
        from rl18xx.agent.alphazero import search_data as sd

        seen = set()
        for directory in self.cfg.data:
            index_dir = _index_dir(self.run_dir, Path(directory))
            for path in sd.shard_paths(directory):
                key = (path.name, str(path.parent))
                seen.add(key)
                if key in self.shards:
                    continue
                try:
                    self.shards[key] = index_shard(path, index_dir)
                except (OSError, ValueError, lzma.LZMAError) as e:
                    LOGGER.warning(f"{path}: not indexed ({e!r})")
        for key in set(self.shards) - seen:
            del self.shards[key]

    def games_total(self) -> int:
        return int(sum(len(s.crc) for s in self.shards.values()))

    def new_games(self) -> int:
        cut = self.state["data_cut"]
        per_mille = self.cfg.held_out_per_mille
        return sum(s.trainable(per_mille) for s in self.shards.values() if cut is None or s.first_seen > cut)

    def due(self, now: Optional[float] = None) -> tuple:
        """``(due, why)``: the cycle trigger (module docstring)."""
        now = time.time() if now is None else now
        new = self.new_games()
        if self.state["data_cut"] is None:
            return new >= self.cfg.min_new_games, f"first cycle: {new} trainable games (needs {self.cfg.min_new_games})"
        sched = escalation(self.cfg, self.state["consecutive_failures"])
        hours = (now - (self.state["last_cycle_started"] or 0.0)) / 3600.0
        if new >= sched["cycle_games"]:
            return True, f"{new} new trainable games (>= {sched['cycle_games']})"
        if hours >= sched["cycle_hours"] and new >= self.cfg.min_new_games:
            return True, f"{hours:.1f} h since the last cycle and {new} new trainable games"
        return False, f"{new} new trainable games, {hours:.1f} h since the last cycle"

    # -- loop --------------------------------------------------------------------------------

    def _on_sigterm(self, signum, frame) -> None:
        LOGGER.warning("SIGTERM: stopping (a running cycle is abandoned and redone after a restart)")
        self.terminating = True

    def check(self) -> None:
        if self.terminating:
            raise Abandoned()

    def serve(self, max_cycles: int = 0) -> int:
        """Poll, run cycles when due, until a STOP file, SIGTERM or ``max_cycles`` cycles. Returns
        the cycles run."""
        import threading

        previous = None
        if threading.current_thread() is threading.main_thread():
            previous = signal.signal(signal.SIGTERM, self._on_sigterm)
        try:
            return self._serve(max_cycles)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)

    def _serve(self, max_cycles: int) -> int:
        done = 0
        stop = self.run_dir / "STOP"
        while not self.terminating:
            if stop.exists():
                LOGGER.info(f"{stop} exists: stopping")
                break
            self.scan()
            due, why = self.due()
            if due:
                LOGGER.info(f"cycle {self.state['cycle'] + 1}: starting ({why})")
                try:
                    self.run_cycle()
                except Abandoned:
                    LOGGER.warning(f"cycle {self.state['cycle'] + 1} abandoned")
                    break
                done += 1
                if max_cycles and done >= max_cycles:
                    break
                continue
            self.write_status("waiting", why=why, games_total=self.games_total())
            deadline = time.time() + self.cfg.poll_seconds
            while time.time() < deadline and not self.terminating and not stop.exists():
                time.sleep(min(1.0, max(0.0, deadline - time.time())))
        self.write_status("stopped")
        return done

    # -- a cycle -----------------------------------------------------------------------------

    def device(self):
        import torch

        if self._device is None:
            self._device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return self._device

    def _load_model(self, path):
        from rl18xx.agent.alphazero.policy_gradient import _load

        model = _load(path).to(self.device())
        model.eval()  # no dropout: the network trained is the one searched with
        return model

    def _fixed_ce(self, model, version: Optional[int] = None) -> np.ndarray:
        if version is not None and version in self._fixed_cache:
            return self._fixed_cache[version]
        ce = per_position_ce(model, self.fixed, self.device(), self.cfg.eval_batch_size)
        if version is not None:
            self._fixed_cache = {version: ce}
        return ce

    def _pass_tasks(self, index: GameIndex, sample: np.ndarray, cycle: int, rng) -> Iterator:
        """Every pass's tasks: the sample in a new order each pass (a fractional last pass: its share)."""
        full, frac = divmod(float(self.cfg.epochs), 1.0)
        passes = int(full) + (1 if frac > 1e-9 else 0)
        for p in range(passes):
            order = rng.permutation(sample)
            if p == int(full):
                order = order[: int(round(frac * len(order)))]
            yield from make_tasks(index, order, self.cfg.positions_per_game, cycle * 1000 + p, self.cfg.task_games)

    def run_cycle(self) -> dict:
        cfg = self.cfg
        cycle = self.state["cycle"] + 1
        started = time.time()
        index = GameIndex.from_shards(list(self.shards.values()))
        data_cut = time.time()
        failures = self.state["consecutive_failures"]
        scale = escalation(cfg, failures)["sample_scale"]
        rng = np.random.default_rng([cfg.seed, cycle])
        recent, old = choose_sample(index, cfg.held_out_per_mille, cfg.recent_games, scale, rng)
        sample = np.concatenate([recent, old])
        rolling = rolling_games(index, cfg)
        new = self.new_games()
        start_version = self.read_promoted()["current"]
        LOGGER.info(f"cycle {cycle}: {len(index)} games in {len(index.paths)} shards ({new} new trainable); sample "
                    f"{len(recent)} recent + {len(old)} older (scale {scale:g}); {len(rolling)} rolling held-out "
                    f"games; warm start from version {start_version}")
        self.write_status("encoding rolling set", games=len(rolling))
        timers = {"encode": 0.0, "train": 0.0}
        import torch

        torch.manual_seed(cfg.seed + cycle)
        model = self._load_model(self.run_dir / "promoted" / f"{start_version}.pth")
        fixed_current = self._fixed_ce(model, start_version) if self.fixed is not None else None
        with make_pool(cfg.workers) as pool:
            t = time.time()
            tasks = make_tasks(index, rolling, cfg.rolling_positions_per_game, ROLLING_PASS, cfg.task_games)
            roll_stream = Stream(pool, tasks, 2 * max(1, cfg.workers), self.check)
            roll = roll_stream.drain()
            timers["encode"] += time.time() - t
            LOGGER.info(f"cycle {cycle}: rolling set {0 if roll is None else len(roll['win'])} positions "
                        f"({time.time() - t:.0f}s)")
            trained = self._train(model, pool, index, sample, cycle, roll, rng, timers)
        model.load_state_dict(trained["best_state"])
        candidate = Path("candidates") / f"{cycle}.pth"
        self.write_status("gate", steps=trained["steps"])
        fixed_candidate = self._fixed_ce(model) if self.fixed is not None else None
        self._save(model, self.run_dir / candidate, cycle, trained)

        gate_mask = None
        if roll is not None:
            gate_mask = ~selection_games(index.crc[roll["game"]], cfg.select_share) if cfg.select_share > 0 else None
        rolling_stats = None
        if roll is not None:
            m = slice(None) if gate_mask is None else gate_mask
            rolling_stats = paired_stats(trained["best_ce"][m], trained["start_ce"][m], roll["game"][m])
        fixed_stats = None
        if self.fixed is not None:
            fixed_stats = paired_stats(fixed_candidate, fixed_current, self.fixed["game"])
        promoted, diverged, reason = gate_decision(
            rolling_stats, fixed_stats, cfg.promote_z, cfg.diverge_ce, trained["start_select"], trained["final_select"]
        )
        if not diverged and trained["best_step"] == 0:
            promoted, reason = False, f"no improvement on the rolling set in training ({reason})"
        failures = 0 if promoted else failures + 1
        sched = escalation(cfg, failures)
        if sched["review"] and not promoted:
            reason += "; review the data mix"
        version = start_version
        if promoted:
            version = self._promote(candidate, cycle, start_version, index, fixed_stats, rolling_stats)
            if fixed_candidate is not None:
                self._fixed_cache = {version: fixed_candidate}
        empty = {"n": 0, "games": 0, "ce": None, "ce_current": None, "diff": None, "se": None}
        fs, rs = fixed_stats or empty, rolling_stats or empty
        games_encoded = np.unique(np.concatenate(trained["games"])) if trained["games"] else np.zeros(0, np.int64)
        # Passes done: the game encodings asked for per sampled game, times the share of their positions trained.
        epochs = trained["attempted"] / max(1, len(sample)) * trained["positions"]
        epochs /= max(1, trained["positions_encoded"])
        record = {
            "cycle": cycle, "started": datetime.fromtimestamp(started).isoformat(timespec="seconds"), "time": _now(),
            "seconds": round(time.time() - started, 1), "games_total": len(index), "games_trained": int(len(sample)),
            "positions_trained": trained["positions"], "steps": trained["steps"],
            "epochs": round(epochs, 4), "start_version": start_version,
            "candidate": str(candidate), "fixed_n": fs["n"], "fixed_games": fs["games"], "fixed_ce": fs["ce"],
            "fixed_ce_current": fs["ce_current"], "fixed_diff": fs["diff"], "fixed_diff_se": fs["se"],
            "rolling_n": rs["n"], "rolling_games": rs["games"], "rolling_ce": rs["ce"],
            "rolling_ce_current": rs["ce_current"], "rolling_diff": rs["diff"], "rolling_diff_se": rs["se"],
            "promoted": bool(promoted), "version": version, "diverged": bool(diverged), "reason": reason,
            "consecutive_failures": failures, "cycle_hours": sched["cycle_hours"],
            "sample_scale": sched["sample_scale"], "encode_seconds": round(timers["encode"], 1),
            "train_seconds": round(timers["train"], 1),
            # diagnostics
            "started_ts": started, "data_cut": data_cut, "new_games": new, "games_recent": int(len(recent)),
            "games_old": int(len(old)), "games_encoded": int(len(games_encoded)), "games_skipped": trained["skipped"],
            "sample_scale_used": scale, "cycle_games": sched["cycle_games"], "review": bool(sched["review"]),
            "best_step": trained["best_step"], "select_n": trained["select_n"],
            "select_ce_start": trained["start_select"], "select_ce_best": trained["best_select"],
            "select_ce_final": trained["final_select"], "worker_encode_seconds": round(trained["worker_seconds"], 1),
            "history": trained["history"], "device": str(self.device()),
        }  # fmt: skip
        append_jsonl(self.run_dir / "cycles.jsonl", record)
        self.state.update(cycle=cycle, data_cut=data_cut, last_cycle_started=started, consecutive_failures=failures)
        self._write_state(self.state)
        self.last_cycle = {"trained_crc": index.crc[games_encoded], "rolling_crc": index.crc[rolling],
                           "record": record}  # fmt: skip
        LOGGER.info(f"cycle {cycle}: {reason}; rolling {rs['diff']} (SE {rs['se']}), fixed {fs['diff']} "
                    f"(SE {fs['se']}); version {version}; {record['seconds']:.0f}s")
        return record

    def _train(self, model, pool, index: GameIndex, sample, cycle: int, roll: Optional[dict], rng, timers) -> dict:
        """Stream the sample's positions through the shuffle buffer into AdamW steps, early-stopped
        on the rolling set; returns the best state with its per-position rolling CE."""
        import torch

        cfg = self.cfg
        device = self.device()
        opt = torch.optim.AdamW([{"params": list(model.parameters()), "lr": cfg.lr}], weight_decay=0.0)
        has_roll = roll is not None and len(roll["win"]) > 0
        select = None
        if has_roll:
            select = selection_games(index.crc[roll["game"]], cfg.select_share) if cfg.select_share > 0 else None
            if select is not None and not select.any():
                select = None

        def evaluate() -> tuple:
            t = time.time()
            ce = per_position_ce(model, roll, device, cfg.eval_batch_size)
            timers["train"] += time.time() - t
            return ce, float(ce.mean() if select is None else ce[select].mean())

        def snapshot() -> dict:
            return {k: v.detach().clone() for k, v in model.state_dict().items()}

        start_ce, start_select = evaluate() if has_roll else (None, None)
        best = {"step": 0, "select": start_select, "ce": start_ce, "state": snapshot()}
        history = [[0, start_select]]
        final_select, bad, step, positions = start_select, 0, 0, 0
        stream = Stream(pool, self._pass_tasks(index, sample, cycle, rng), 2 * max(1, cfg.workers), self.check)
        buffer: Optional[ShuffleBuffer] = None
        warm = True
        last_log = time.time()
        while True:
            self.check()
            stream.collect(block=False)
            if buffer is None and stream.staged:
                buffer = ShuffleBuffer(cfg.buffer_positions, stream.staged[0][0])
            while buffer is not None and stream.staged and buffer.n < buffer.capacity:
                entry = stream.staged[0]
                entry[1] += buffer.add(entry[0], entry[1])
                if entry[1] >= len(entry[0]["win"]):
                    stream.staged.pop(0)
            n = 0 if buffer is None else buffer.n
            low = cfg.buffer_positions if warm else max(cfg.batch_size, cfg.buffer_positions // 2)
            if n < low and not stream.exhausted and (stream.inflight or stream.more):
                t = time.time()
                stream.collect(block=True)
                timers["encode"] += time.time() - t
                continue
            warm = False
            if n == 0:
                break
            t = time.time()
            batch = buffer.take(min(cfg.batch_size, n), rng)
            ce, mse = train_step(model, opt, batch, device, cfg.score_weight)
            timers["train"] += time.time() - t
            step += 1
            positions += len(batch["win"])
            if has_roll and step % cfg.eval_every == 0:
                ce_all, sel = evaluate()
                history.append([step, sel])
                final_select = sel
                if sel < best["select"] - cfg.min_improvement:
                    best, bad = {"step": step, "select": sel, "ce": ce_all, "state": snapshot()}, 0
                else:
                    bad += 1
                LOGGER.info(f"cycle {cycle}: step {step} rolling CE {sel:.4f} (start {start_select:.4f}, best "
                            f"{best['select']:.4f} at {best['step']}) train CE {ce:.4f} MSE {mse:.4f}")
                self.write_status("training", step=step, rolling_ce=sel, best_step=best["step"])
            elif time.time() - last_log > 300:
                self.write_status("training", step=step)
                last_log = time.time()
            if (cfg.max_steps and step >= cfg.max_steps) or bad >= cfg.patience:
                break
        if has_roll and history[-1][0] != step:
            ce_all, sel = evaluate()
            history.append([step, sel])
            final_select = sel
            if sel < best["select"] - cfg.min_improvement:
                best = {"step": step, "select": sel, "ce": ce_all, "state": snapshot()}
        if not has_roll:  # nothing to choose by: the trained state
            best = {"step": step, "select": None, "ce": None, "state": snapshot()}
        LOGGER.info(f"cycle {cycle}: trained {step} steps, {positions} positions; best at step {best['step']}")
        return {
            "best_state": best["state"], "best_step": best["step"], "best_ce": best["ce"],
            "best_select": best["select"], "start_ce": start_ce, "start_select": start_select,
            "final_select": final_select, "steps": step, "positions": positions, "history": history,
            "games": stream.games, "skipped": stream.skipped, "worker_seconds": stream.worker_seconds,
            "attempted": stream.attempted, "positions_encoded": stream.positions,
            "select_n": int(len(roll["win"]) if select is None else select.sum()) if has_roll else 0,
        }  # fmt: skip

    def _save(self, model, path: Path, cycle: int, trained: dict) -> None:
        from rl18xx.agent.alphazero.value_refit import write_model_checkpoint

        provenance = {"value_trainer": {"run_dir": str(self.run_dir), "cycle": cycle, "best_step": trained["best_step"],
                                        "steps": trained["steps"], "positions": trained["positions"]}}
        write_model_checkpoint(model, self.cfg.init, path, provenance)

    def _promote(self, candidate: Path, cycle: int, start_version: int, index: GameIndex, fixed, rolling) -> int:
        """``candidate`` as the next version: hard-linked (or copied) into ``promoted/``, then ``promoted.json``."""
        promoted = self.read_promoted()
        version = int(promoted["current"]) + 1
        if version != start_version + 1:
            LOGGER.warning(f"promoted.json moved from {start_version} to {promoted['current']} during the cycle")
        target = self.run_dir / "promoted" / f"{version}.pth"
        tmp = target.with_name(f".{version}.pth.tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(self.run_dir / candidate, tmp)
        except OSError:
            shutil.copyfile(self.run_dir / candidate, tmp)
        os.replace(tmp, target)
        entry = {"version": version, "path": f"promoted/{version}.pth", "cycle": cycle, "time": _now(),
                 "games_total": len(index), "fixed_ce": None if fixed is None else fixed["ce"],
                 "rolling_ce": None if rolling is None else rolling["ce"], "candidate": str(candidate)}
        versions = [v for v in promoted["versions"] if v["version"] < version] + [entry]
        write_json_atomic(self.run_dir / "promoted.json", {"current": version, "versions": versions})
        LOGGER.info(f"cycle {cycle}: promoted version {version}")
        return version


def serve(cfg: ValueTrainerConfig, max_cycles: int = 0) -> int:
    """``main.py value-train``: the trainer on ``cfg.run_dir`` (resumed if it exists) until STOP,
    SIGTERM or ``max_cycles`` cycles (0: forever)."""
    return ValueTrainer(cfg).serve(max_cycles)
