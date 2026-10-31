"""Search learning's saved games (``main.py search-learning --save-data <dir>``): every
finished self-play game with its searches, kept on disk so the value can be trained over
many passes, positions reanalyzed and a run restarted from data after a recipe change.

Kept per game (:func:`pack_game`, in the worker):

- the action log -- the engine's ``raw_actions``, the actions of a
  ``game_records.make_game_record`` (:meth:`SavedGame.game_record`) -- and the start: its
  label, kind, number of actions (the log's first) and ``auction_unlock``;
- the outcome: win share, net-worth fractions, net worths, the score target and the
  auxiliary targets (final net-worth / cash shares, corporations floated);
- per search, full and fast, in the game's order: the decision index, the action-log
  position (the position is the log's first that many actions), flags (fast, has a row,
  explored: the move was drawn from the visits), the network version and the root value
  per seat (absolute order) -- the n-step / lambda-return inputs;
- per row: the mover's seat, the legal policy indices, the policy target over them, the
  price target (slot, range, cell distribution), the critic flag and the loss weight.

No encodings (a ``SearchRow`` pickles to ~21 KB): :func:`rebuild_rows` replays the log in
the Rust engine and encodes each row's position (``encode_canonical_py``, deterministic),
giving the generator's rows bit for bit, with the value targets of any ``value_lambda`` /
``value_nstep`` / ``value_bootstrap_weight`` / ``reward``.

On disk (:class:`SearchDataWriter`, in the trainer): shards of ``shard_games`` games,
``<writer>-<seq>.sld``, each one xz stream of a header (JSON: the run's config, the code's
commit, the games' metadata, the columns' layout) and the games' columns concatenated
(multi-byte columns byte-shuffled, root values XOR-ed with the previous search's). A shard is
compressed in a background thread,
written to a temporary file, synced and renamed: a complete shard is never written again.
Until then its games are appended to the shard's journal (``.journal``: one length- and
CRC-prefixed pickle per game, locked by its writer); a writer starting in the directory
turns journals left by a crash into shards (a torn last record is dropped), so a crash
loses at most the game being appended. Readers see complete shards only.
"""

from __future__ import annotations

import fcntl
import json
import logging
import lzma
import os
import pickle
import struct
import threading
import zlib
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

FORMAT = 1
MAGIC = b"SLDATA\x00\x01"
SHARD_SUFFIX = ".sld"
JOURNAL_SUFFIX = ".journal"
FAST, ROW, EXPLORED = 1, 2, 4  # search flags
PRESET = 6 | lzma.PRESET_EXTREME  # xz: 2.4% smaller than 6 on azl's games, 36 ms a game in the writer's thread
CRITIC = 1  # row flags
# What rebuild_rows may set differently from the generating run.
VALUE_SETTINGS = ("value_nstep", "value_bootstrap_weight", "value_lambda", "reward")
# Each column: its dtype and the per-game count its length is.
COLUMNS = {
    "search_decision": (np.int32, "searches"),
    "search_action": (np.int32, "searches"),
    "search_flags": (np.uint8, "searches"),
    "search_version": (np.int32, "searches"),
    "search_value": (np.float32, "searches"),  # (searches, players)
    "row_rotation": (np.uint8, "rows"),
    "row_legal_count": (np.int32, "rows"),
    "row_flags": (np.uint8, "rows"),
    "row_weight": (np.float64, "rows"),
    "row_price_slot": (np.int16, "rows"),  # -1: no price target
    "row_price_lo": (np.int32, "rows"),
    "row_price_hi": (np.int32, "rows"),
    "legal": (np.uint16, "legal"),  # a game's legal indices, row after row
    "target": (np.float32, "legal"),
    "price_cells": (np.float32, "price"),  # (price rows, cells)
}


# ---------------------------------------------------------------------------
# Packing (worker side)
# ---------------------------------------------------------------------------


def pack_game(g, worth, win_share, fractions, score, aux_seat, aux_global, value_version=None) -> dict:
    """A finished generator game (``search_learning._Game``, its rows' outcome filled in)
    as a shard keeps it: ``{"meta", "actions" (JSON bytes), "columns"}``; ``value_version``: the
    followed value run's promoted version the game was searched with (``value_follow``)."""
    from rl18xx.agent.alphazero import price_pmf
    from rl18xx.agent.alphazero.start_positions import AUCTION_START

    actions = list(g.game._game.raw_actions)
    rows, searches = g.rows, g.searches
    if len(searches) != len(g.search_values):
        raise ValueError(f"game {g.uid}: {len(searches)} searches recorded, {len(g.search_values)} values")
    row_searches = [i for i, s in enumerate(searches) if s[3]]
    if row_searches != [r.search_index for r in rows]:
        raise ValueError(f"game {g.uid}: rows don't match the searches that made them")
    players = len(fractions)
    legal = [np.asarray(r.legal, dtype=np.int64) for r in rows]
    flat = np.concatenate(legal) if legal else np.zeros(0, dtype=np.int64)
    if flat.size and (flat.min() < 0 or flat.max() >= 1 << 16):
        raise ValueError(f"game {g.uid}: a legal index outside uint16")
    priced = [r.price for r in rows if r.price is not None]
    columns = {
        "search_decision": np.array([s[0] for s in searches], dtype=np.int32),
        "search_action": np.array([s[1] for s in searches], dtype=np.int32),
        "search_flags": np.array(
            [FAST * bool(s[2]) | ROW * bool(s[3]) | EXPLORED * bool(s[4]) for s in searches], dtype=np.uint8
        ),
        "search_version": np.array([s[5] for s in searches], dtype=np.int32),
        "search_value": (
            np.stack([np.asarray(v, dtype=np.float32) for v in g.search_values])
            if searches
            else np.zeros((0, players), dtype=np.float32)
        ),
        "row_rotation": np.array([r.rotation for r in rows], dtype=np.uint8),
        "row_legal_count": np.array([len(x) for x in legal], dtype=np.int32),
        "row_flags": np.array([CRITIC * bool(r.critic) for r in rows], dtype=np.uint8),
        "row_weight": np.array([r.weight for r in rows], dtype=np.float64),
        "row_price_slot": np.array([-1 if r.price is None else r.price[0] for r in rows], dtype=np.int16),
        "row_price_lo": np.array([0 if r.price is None else r.price[1] for r in rows], dtype=np.int32),
        "row_price_hi": np.array([0 if r.price is None else r.price[2] for r in rows], dtype=np.int32),
        "legal": flat.astype(np.uint16),
        "target": (
            np.concatenate([np.asarray(r.target, dtype=np.float32) for r in rows])
            if rows
            else np.zeros(0, dtype=np.float32)
        ),
        "price_cells": (
            np.stack([np.asarray(p[3], dtype=np.float32) for p in priced])
            if priced
            else np.zeros((0, price_pmf.NUM_CELLS), dtype=np.float32)
        ),
    }
    meta = {
        "uid": g.uid,
        "start": g.start,
        "start_kind": g.start.split(":")[0] if g.start else None,
        "start_actions": int(g.start_actions),
        "auction_unlock": g.start == AUCTION_START,
        "no_buyback_rule": bool(g.game._game.no_buyback_rule),
        "num_players": players,
        "termination": g.termination,
        "decisions": int(g.decisions),
        "searched": int(g.searched),
        "seat_servers": list(g.seat_servers),
        "seat_members": list(getattr(g, "seat_members", None) or ["main"] * players),  # MAIN or a league member
        "actions": len(actions),
        "win_share": [float(x) for x in win_share],
        "fractions": [float(x) for x in fractions],
        "net_worth": [float(x) for x in worth],
        "score_target": None if score is None else [float(x) for x in score],
        "aux_seat": np.asarray(aux_seat).tolist(),
        "aux_global": np.asarray(aux_global).tolist(),
        "value_version": None if value_version is None else int(value_version),
    }
    return {"meta": meta, "actions": json.dumps(actions, separators=(",", ":")).encode(), "columns": columns}


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def _counts(game: dict) -> dict:
    c = game["columns"]
    return {
        "searches": len(c["search_decision"]),
        "rows": len(c["row_legal_count"]),
        "legal": len(c["legal"]),
        "price": len(c["price_cells"]),
        "action_bytes": len(game["actions"]),
    }


def source_commit() -> Optional[str]:
    """The git commit of this code (the engine that played the games), if it is a checkout."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _xor_previous(values: np.ndarray, decode: bool = False) -> np.ndarray:
    """A game's float32 root values (searches, seats), each search's bits XOR-ed with the previous
    search's (the first kept), and back (``decode``): a game's values move slowly, so their high bytes
    then compress to almost nothing (8.0 -> 7.5 KB a game on azl's)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    if decode:
        return np.bitwise_xor.accumulate(bits, axis=0).view(np.float32)
    out = bits.copy()
    out[1:] ^= bits[:-1]
    return out.view(np.float32)


def column_bytes(name: str, games: list) -> tuple:
    """``(shape, bytes)`` of column ``name`` of packed ``games`` as a shard stores it: the games'
    arrays concatenated (root values XOR-coded per game), in byte planes when an item has several
    bytes (as blosc's shuffle: a float's sign / exponent bytes compress apart from its mantissa)."""
    dtype = COLUMNS[name][0]
    parts = [np.asarray(game["columns"][name], dtype=dtype) for game in games]
    if name == "search_value":
        parts = [_xor_previous(part) for part in parts]
    array = np.ascontiguousarray(np.concatenate(parts) if parts else np.zeros(0, dtype=dtype))
    itemsize = array.dtype.itemsize
    data = array.view(np.uint8).reshape(-1, itemsize).T.tobytes() if itemsize > 1 else array.tobytes()
    return list(array.shape), data


def encode_shard(games: list, config: dict, commit: Optional[str] = None) -> bytes:
    """The uncompressed bytes of a shard of packed games (:func:`pack_game`)."""
    metas = [{**game["meta"], "counts": _counts(game)} for game in games]
    actions = b"".join(game["actions"] for game in games)
    chunks = [actions]
    layout = {"actions": {"offset": 0, "nbytes": len(actions)}}
    offset = len(actions)
    for name, (dtype, _) in COLUMNS.items():
        shape, data = column_bytes(name, games)
        layout[name] = {"dtype": np.dtype(dtype).str, "shape": shape, "offset": offset, "nbytes": len(data)}
        chunks.append(data)
        offset += len(data)
    header = json.dumps(
        {
            "format": FORMAT,
            "written": datetime.now().isoformat(timespec="seconds"),
            "commit": commit,
            "config": config,
            "games": metas,
            "columns": layout,
        }
    ).encode()
    return b"".join([MAGIC, struct.pack("<Q", len(header)), header, *chunks])


def write_shard(path, games: list, config: dict, preset: int = PRESET, commit: Optional[str] = None) -> int:
    """Compress ``games`` into the shard ``path``: a temporary file, synced, then renamed (a
    reader sees the whole shard or none of it). Returns its size."""
    path = Path(path)
    data = lzma.compress(encode_shard(games, config, commit), preset=preset)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _sync_dir(path.parent)
    return len(data)


def _sync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass
class SavedGame:
    """One saved game: ``meta`` (start, outcome, counts), the generating run's ``config``,
    its action log, its columns (``COLUMNS``; ``legal`` decoded to int32) and the writer's commit."""

    meta: dict
    config: dict
    actions_json: bytes
    columns: dict
    commit: Optional[str] = None  # the code that played it (git)

    @cached_property
    def actions(self) -> list:
        return json.loads(self.actions_json)

    @property
    def uid(self) -> str:
        return self.meta["uid"]

    @property
    def row_searches(self) -> np.ndarray:
        """Each row's search (its ``search_index``)."""
        return np.flatnonzero(self.columns["search_flags"] & ROW)

    def row_slices(self) -> list:
        """Each row's slice of ``legal`` / ``target``."""
        ends = np.cumsum(self.columns["row_legal_count"])
        return [slice(int(e - n), int(e)) for e, n in zip(ends, self.columns["row_legal_count"])]

    def detached(self) -> "SavedGame":
        """A copy whose columns own their memory. A shard's games are views of its arrays, so keeping one game
        keeps the whole shard (~90 MB) alive: keep detached copies when sampling a few games from many shards."""
        columns = {name: np.array(c) for name, c in self.columns.items()}
        return SavedGame(self.meta, self.config, self.actions_json, columns, self.commit)

    def packed(self) -> dict:
        """The game as :func:`pack_game` packed it (to write it into another shard)."""
        return {"meta": dict(self.meta), "actions": self.actions_json, "columns": dict(self.columns)}

    def game_record(self) -> dict:
        """The game as a ``game_records.make_game_record`` (ids added as ``RustGameAdapter.raw_actions`` does)."""
        from rl18xx.agent.alphazero.game_records import make_game_record

        actions = [dict(a) for a in self.actions]
        next_id = max([a["id"] + 1 for a in actions if a.get("id") is not None] + [1])
        for a in actions:
            if a.get("id") is None:
                a["id"], next_id = next_id, next_id + 1
        keys = (
            "start", "termination", "decisions", "searched", "seat_servers", "seat_members", "win_share", "net_worth",
        )  # fmt: skip
        meta = {"name": self.uid, **{k: self.meta.get(k) for k in keys}}
        meta["worth_share"] = self.meta.get("fractions")
        meta["description"] = f"search-learning {self.config.get('run_name') or ''} game {self.uid}".strip()
        return make_game_record(
            actions, self.meta["num_players"], auction_unlock=self.meta["auction_unlock"], meta=meta, game_id=self.uid
        )


def decode_shard(raw: bytes) -> list:
    """The :class:`SavedGame` s of a shard's uncompressed bytes."""
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError("not a search-data shard")
    (n,) = struct.unpack_from("<Q", raw, len(MAGIC))
    start = len(MAGIC) + 8
    header = json.loads(raw[start : start + n])
    if header.get("format") != FORMAT:
        raise ValueError(f"search-data format {header.get('format')}, this reader reads {FORMAT}")
    base = start + n
    view = memoryview(raw)
    layout = header["columns"]
    arrays = {}
    for name, c in layout.items():
        if name == "actions":
            continue
        dtype = np.dtype(c["dtype"])
        data = np.frombuffer(view[base + c["offset"] : base + c["offset"] + c["nbytes"]], dtype=np.uint8)
        if dtype.itemsize > 1:
            data = data.reshape(dtype.itemsize, -1).T.copy()
        arrays[name] = data.view(dtype).reshape(c["shape"])
    start = base + layout["actions"]["offset"]
    actions = bytes(view[start : start + layout["actions"]["nbytes"]])
    games, at, action_at = [], {key: 0 for key in ("searches", "rows", "legal", "price")}, 0
    for meta in header["games"]:
        counts = meta.pop("counts")
        columns = {}
        for name, (_, key) in COLUMNS.items():
            columns[name] = arrays[name][at[key] : at[key] + counts[key]]
        columns["legal"] = columns["legal"].astype(np.int32)
        columns["search_value"] = _xor_previous(columns["search_value"], decode=True)
        for key in at:
            at[key] += counts[key]
        log = actions[action_at : action_at + counts["action_bytes"]]
        games.append(SavedGame(meta, header["config"], log, columns, header.get("commit")))
        action_at += counts["action_bytes"]
    return games


def read_shard(path) -> list:
    return decode_shard(lzma.decompress(Path(path).read_bytes()))


def shard_paths(directory) -> list:
    """The complete shards under ``directory`` in the order they were begun."""
    return sorted(Path(directory).glob(f"*{SHARD_SUFFIX}"))


def iter_games(directory) -> Iterator[SavedGame]:
    for path in shard_paths(directory):
        yield from read_shard(path)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def replay(game: SavedGame, positions) -> Iterator:
    """``(i, rs)`` for each action-log position ``positions[i]`` (non-decreasing): the
    game's Rust engine (``engine_rs.BaseGame``, live -- encode or clone it before the next
    step) after the log's first ``positions[i]`` actions, replayed from a fresh game."""
    from rl18xx.agent.alphazero.start_positions import new_game

    # Games saved before the rule against train shuffling were generated without it.
    rule = bool(game.meta.get("no_buyback_rule", False))
    rs = new_game(int(game.meta["num_players"]), bool(game.meta["auction_unlock"]), no_buyback_rule=rule)._game
    actions = game.actions
    done = 0
    for i, position in enumerate(positions):
        position = int(position)
        if position < done or position > len(actions):
            raise ValueError(f"game {game.uid}: position {position} after {done} of {len(actions)} actions")
        while done < position:
            rs.process_action(actions[done])
            done += 1
        yield i, rs


def rebuild_rows(game: SavedGame, encode: bool = True, check: bool = True, **settings) -> list:
    """The game's ``SearchRow`` s as its generator made them: positions encoded by replaying
    the action log (``encode`` False: ``game_state`` / ``node_features`` None, no replay),
    value targets from the saved search values under the generating run's settings unless
    ``settings`` (``VALUE_SETTINGS``) say otherwise, e.g. ``value_lambda=0.9`` or
    ``value_nstep=8, value_lambda=None``. ``check``: each replayed position must have the
    row's mover and legal moves (the log replays as it was played)."""
    from rl18xx.agent.alphazero.search_learning import (
        SearchLearningConfig,
        SearchRow,
        score_target,
        share_of_winners,
        win_targets,
    )

    unknown = set(settings) - set(VALUE_SETTINGS)
    if unknown:
        raise TypeError(f"rebuild_rows: unknown settings {sorted(unknown)} (of {VALUE_SETTINGS})")
    defaults = SearchLearningConfig.__dataclass_fields__
    s = {**{k: game.config.get(k, defaults[k].default) for k in VALUE_SETTINGS}, **settings}
    meta, c = game.meta, game.columns
    fractions = np.asarray(meta["fractions"], dtype=np.float32)
    values = list(c["search_value"])
    win_target = win_targets(share_of_winners(fractions), values, s)
    score = score_target(np.asarray(meta["net_worth"], dtype=np.float64), s["reward"])
    aux_seat = np.asarray(meta["aux_seat"], dtype=np.float32)
    aux_global = np.asarray(meta["aux_global"], dtype=np.float32)
    searches = game.row_searches
    slices = game.row_slices()
    encodings = [None] * len(searches)
    if encode:
        for k, rs in replay(game, c["search_action"][searches]):
            state, nodes, _, rotation, _ = rs.encode_canonical_py()
            if check:
                legal = np.sort(np.asarray(rs.factored_legal_indices(), dtype=np.int64))
                if int(rotation) != int(c["row_rotation"][k]) or not np.array_equal(legal, c["legal"][slices[k]]):
                    raise ValueError(f"game {game.uid}: row {k}'s position doesn't replay as it was played")
            encodings[k] = (state, nodes)
    rows, price_at = [], 0
    for k, si in enumerate(searches):
        si = int(si)
        price = None
        if c["row_price_slot"][k] >= 0:
            cells = c["price_cells"][price_at].copy()
            price = (int(c["row_price_slot"][k]), int(c["row_price_lo"][k]), int(c["row_price_hi"][k]), cells)
            price_at += 1
        state, nodes = encodings[k] or (None, None)
        rows.append(
            SearchRow(
                game_state=state,
                node_features=nodes,
                rotation=int(c["row_rotation"][k]),
                legal=c["legal"][slices[k]].copy(),
                target=c["target"][slices[k]].copy(),
                price=price,
                version=int(c["search_version"][si]),
                fractions=fractions,
                score_target=score,
                critic=bool(c["row_flags"][k] & CRITIC),
                root_value=values[si].copy(),
                search_index=si,
                win_target=win_target(si),
                aux_seat=aux_seat,
                aux_global=aux_global,
                fast=bool(c["search_flags"][si] & FAST),
                weight=float(c["row_weight"][k]),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Writer (trainer side)
# ---------------------------------------------------------------------------


def _append_record(f, obj) -> None:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack("<II", len(blob), zlib.crc32(blob)) + blob)
    f.flush()


def _read_records(data: bytes) -> list:
    """A journal's complete records (a torn or corrupt tail is dropped)."""
    records, at = [], 0
    while at + 8 <= len(data):
        n, crc = struct.unpack_from("<II", data, at)
        blob = data[at + 8 : at + 8 + n]
        if len(blob) < n or zlib.crc32(blob) != crc:
            break
        records.append(pickle.loads(blob))
        at += 8 + n
    return records


def recover_journals(directory, preset: int = PRESET) -> int:
    """Turn the journals under ``directory`` whose writers are gone (unlocked) into shards;
    returns the games recovered. A journal whose shard exists is a finished one's leftover."""
    recovered = 0
    for path in sorted(Path(directory).glob(f"*{JOURNAL_SUFFIX}")):
        try:
            f = open(path, "rb")
        except FileNotFoundError:
            continue  # its writer just finished it
        with f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # a live writer's
            target = path.with_suffix(SHARD_SUFFIX)
            if not target.exists():
                records = _read_records(f.read())
                if len(records) > 1:
                    write_shard(target, records[1:], records[0]["config"], preset, records[0].get("commit"))
                    recovered += len(records) - 1
                    LOGGER.warning(f"search data: recovered {len(records) - 1} games from {path.name}")
            path.unlink(missing_ok=True)
    return recovered


class SearchDataWriter:
    """Appends packed games (:func:`pack_game`) to shards under ``directory`` (module
    docstring). ``config``: the run's settings, kept in every shard."""

    def __init__(self, directory, config: dict, shard_games: int = 256, preset: int = PRESET):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = json.loads(json.dumps(config, default=str))
        self.shard_games = max(1, int(shard_games))
        self.preset = preset
        # Shards sort in the order their writers started (to the microsecond: a resume in the
        # same second must not reuse a name).
        self.name = f"{datetime.now():%Y%m%dT%H%M%S%f}-{os.getpid()}"
        self.seq = 0
        self.games = 0  # added (in a shard, or still in the journal)
        self.shards = 0  # written
        self.bytes = 0  # of the shards written
        self.errors = 0
        self._pending: list = []
        self._journal = None  # (path, open file) of the current shard
        self._thread: Optional[threading.Thread] = None
        self.commit = source_commit()
        recover_journals(self.directory, preset)

    def _shard_path(self, suffix: str) -> Path:
        return self.directory / f"{self.name}-{self.seq:05d}{suffix}"

    def add(self, game: dict) -> None:
        try:
            if self._journal is None:
                path = self._shard_path(JOURNAL_SUFFIX)
                f = open(path, "ab")
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._journal = (path, f)
                _append_record(f, {"config": self.config, "commit": self.commit})
            _append_record(self._journal[1], game)
        except OSError as e:  # the game still goes into the shard
            self.errors += 1
            LOGGER.error(f"search data: journal write failed: {e!r}")
        self._pending.append(game)
        self.games += 1
        if len(self._pending) >= self.shard_games:
            self._rotate()

    def _rotate(self) -> None:
        games, journal, target = self._pending, self._journal, self._shard_path(SHARD_SUFFIX)
        self._pending, self._journal = [], None
        self.seq += 1
        if self._thread is not None:
            self._thread.join()  # one shard at a time
        self._thread = threading.Thread(target=self._write, args=(games, journal, target), name="search-data-writer")
        self._thread.start()

    def _write(self, games: list, journal, target: Path) -> None:
        try:
            size = write_shard(target, games, self.config, self.preset, self.commit)
        except Exception as e:  # the journal stays: the next writer here recovers it
            self.errors += 1
            LOGGER.error(f"search data: shard {target.name} not written ({e!r}); its journal is kept")
            if journal is not None:
                journal[1].close()
            return
        if journal is not None:
            journal[0].unlink(missing_ok=True)
            journal[1].close()
        self.shards += 1
        self.bytes += size

    def close(self) -> None:
        """Write the games still pending (a smaller shard) and wait for the writes."""
        if self._pending:
            self._rotate()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Command line: python -m rl18xx.agent.alphazero.search_data stats|export ...
# ---------------------------------------------------------------------------


def stats(directory) -> dict:
    """Counts and sizes of the saved games under ``directory``."""
    out = {"shards": 0, "games": 0, "searches": 0, "fast_searches": 0, "rows": 0, "actions": 0, "bytes": 0}
    for path in shard_paths(directory):
        out["shards"] += 1
        out["bytes"] += path.stat().st_size
        for game in read_shard(path):
            flags = game.columns["search_flags"]
            out["games"] += 1
            out["searches"] += len(flags)
            out["fast_searches"] += int(np.count_nonzero(flags & FAST))
            out["rows"] += int(np.count_nonzero(flags & ROW))
            out["actions"] += int(game.meta["actions"])
    out["bytes_per_game"] = out["bytes"] / max(out["games"], 1)
    return out


def export_collection(directory, collection, limit: Optional[int] = None) -> int:
    """Write saved games as a game-viewer collection (``game_records``) under ``collection``."""
    from rl18xx.agent.alphazero.game_records import append_index, save_game_record

    written = 0
    for game in iter_games(directory):
        if limit is not None and written >= limit:
            break
        record = game.game_record()
        entry = {k: v for k, v in record["rl18xx"].items() if k not in ("version", "auction_unlock", "description")}
        entry["game_file"] = save_game_record(collection, game.uid, record)
        entry["auction_unlock"] = game.meta["auction_unlock"]
        append_index(collection, entry)
        written += 1
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search learning's saved games (--save-data)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stats", help="Counts and bytes per game")
    p.add_argument("directory")
    p = sub.add_parser("export", help="Write games as a game-viewer collection (dashboard /games)")
    p.add_argument("directory")
    p.add_argument("collection")
    p.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "stats":
        print(json.dumps(stats(args.directory), indent=1))
    else:
        print(f"{export_collection(args.directory, args.collection, args.limit)} games written to {args.collection}")
