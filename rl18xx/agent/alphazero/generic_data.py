"""Title-generic supervised data for the generic model (RL bridge milestone C1; ``docs/rl_model.md``
"Supervised data and pretraining").

A dataset is a directory of **compact game records**, not encoded positions: per human game its action
log (as the engine replays it) and, per decision, the legal layout indices (sparse), the chosen index
(exactly one match, or the position is skipped and counted -- never guessed), the price target with the
chosen move's legal range (lo, hi, step), the mover's seat rotation, and per game the outcome in
**seating order** (net worth, win share with ties split, placement points as
``policy_gradient.default_placement_points``). Training replays a game in a data-loader worker and
encodes its positions on the fly (``generic_tokens.encode_batch``, ~25k positions/s; replay ~15 us per
action), so a dataset does not depend on the encoder version and stays small (1830's corpus: ~0.3 GB
against ~12 GB of LZ4-compressed token arrays, measured 2026-11-13).

Layout::

    <dir>/data.mdb, lock.mdb   LMDB: b"g:<game id>" -> lz4(pickle(record)), b"__meta__" -> meta JSON
    <dir>/meta.json            title(s), layout (version, fingerprint), encoder version at conversion,
                               sources, split, counts, match / skip statistics
    <dir>/games.jsonl          one line per game: id, split, players, decisions, actions

Records are made by a per-title :class:`TitleImporter`. 1830's is the legacy converter's own matching
(``pretraining._action_dict_to_factored_index`` on the replayed ``RustGameAdapter``), so its labels are
the legacy converter's, index for index; titles without an importer (1867 until milestone A2's
``match_human_action``) raise :class:`UnsupportedTitle`.

Reading: :class:`GenericDataset` (meta, index, records); :func:`replay` (a record's decision positions);
:func:`position_item` (one decision's encoding, legal rows and targets in the encoder's seat frame);
:func:`collate` (items -> one batch in ``model_generic.build_inputs``' flat legal layout, plus targets);
:class:`PositionBatches` (a ``torch`` ``IterableDataset`` of collated batches: games sharded over loader
workers, positions mixed through an LZ4-compressed shuffle buffer).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import lz4.frame
import numpy as np

LOGGER = logging.getLogger(__name__)

#: Version of the record / directory format.
DATA_FORMAT = 1
#: Per-game probability of the validation split (``TrainingConfig.pretrain_validation_percentage``, the
#: legacy converter's): the split is ``pretraining.in_validation_split(game id, 0.05)``, so a game is held
#: out here exactly when it was held out of ``lmdb_v6k`` / SL checkpoint 10.
DEFAULT_VALIDATION_PERCENTAGE = 0.05
SPLITS = ("train", "validation")
MAX_SEATS = 6
_META_KEY = b"__meta__"
_LMDB_MAP_SIZE = 1 << 36  # 64 GiB of address space; the file grows as written

# Token arrays by leading dimension, and the value padding takes (encode_batch's own padding).
_HEX_KEYS = ("hex_feat", "hex_ptr", "hex_res", "hex_slot_city", "hex_ids", "hex_mask")
_ENTITY_KEYS = ("ent_type", "ent_feat", "ent_ptr", "ent_hex", "ent_mask")
_RELATION_KEYS = ("rel", "rel_val", "rel_mask")
_PAD_MINUS_ONE = {"hex_ptr", "hex_res", "hex_slot_city", "ent_ptr", "ent_hex"}


class UnsupportedTitle(NotImplementedError):
    """No human-game importer for this title yet."""


class DatasetMismatch(ValueError):
    """A dataset whose layout or format does not fit the running engine."""


# ---------------------------------------------------------------------------------------------------
# Importers: from a cleaned human game to decisions with exactly-one-index labels
# ---------------------------------------------------------------------------------------------------


class TitleImporter:
    """How one title's cleaned human games become records. Subclasses fill these in; the conversion
    loop (:func:`convert_game`) is title-generic."""

    title: str = ""

    def load(self, game: dict) -> Tuple[Optional[list], Optional[str]]:
        """``(action log, None)`` -- the game's actions as the engine replays them -- or
        ``(None, reason)`` for a game that cannot be used."""
        raise NotImplementedError

    def players(self, game: dict) -> Dict[int, str]:
        """``{player id: name}`` in seating order."""
        out = {}
        for i, p in enumerate(game.get("players") or []):
            if isinstance(p, dict):
                pid = int(p.get("id", i + 1))
                out[pid] = str(p.get("name", f"Player {pid}"))
            else:
                out[i + 1] = str(p)
        return out

    def optional_rules(self, game: dict) -> List[str]:
        return []

    def seed(self, game: dict) -> Optional[int]:
        return None

    def new_state(self, players: Dict[int, str], optional_rules: Sequence[str], seed: Optional[int]):
        """The state labels are matched on (an adapter for 1830); :meth:`raw` gives its engine game."""
        return new_engine_game(self.title, players, optional_rules, seed)

    def raw(self, state):
        return getattr(state, "_game", state)

    def match(self, state, action: dict, legal_indices: np.ndarray) -> Tuple[Optional[int], str]:
        """``(layout index, "ok")`` for the one legal index the human action names, else
        ``(None, "unmatched")`` or ``(None, "ambiguous")`` (several indices: never guess)."""
        raise NotImplementedError

    def apply(self, state, action: dict) -> None:
        state.process_action(action)

    def price(self, action: dict) -> Optional[int]:
        """The price the human chose (bids, company and train purchases), or None."""
        price = action.get("price")
        return None if price is None else int(price)


class Importer1830(TitleImporter):
    """1830: the legacy converter's matching, unchanged (``pretraining._action_dict_to_factored_index``:
    the human dict against the factored choices of the replayed ``RustGameAdapter``, an index through
    ``ActionMapper.index_for_factored``; more than one index is ambiguous). Cleaned games
    (``main.py clean``) replay as ``pretraining._load_cleaned_game_via_rust`` replays them."""

    title = "1830"

    def __init__(self):
        from rl18xx.agent.alphazero.action_mapper import ActionMapper

        self.mapper = ActionMapper()

    def optional_rules(self, game: dict) -> List[str]:
        from rl18xx.agent.alphazero.pretraining import engine_optional_rules

        return engine_optional_rules(game)

    def load(self, game: dict) -> Tuple[Optional[list], Optional[str]]:
        # pretraining._load_cleaned_game_via_rust, keeping the engine's error as the reason.
        if game.get("status") == "error" or "title" not in game:
            return None, f"not_cleaned: {game.get('reason', 'error record')}"
        if game.get("title") != "1830":
            return None, f"title {game.get('title')}"
        players = self.players(game)
        if not players:
            return None, "no_players"
        state = self.new_state(players, self.optional_rules(game), None)
        try:
            for action in game.get("actions", []):
                state.process_action(action)
        except Exception as e:  # a rule fixed since the game was cleaned (Ruby HEAD)
            return None, f"replay_failed: {type(e).__name__}: {str(e)[:120]}"
        return state.raw_actions, None

    def new_state(self, players, optional_rules, seed):
        from rl18xx.rust_adapter import RustGameAdapter

        return RustGameAdapter(new_engine_game("1830", players, optional_rules, seed))

    def match(self, state, action: dict, legal_indices: np.ndarray) -> Tuple[Optional[int], str]:
        from rl18xx.agent.alphazero.pretraining import AmbiguousActionMatch, _action_dict_to_factored_index

        factored = state.get_factored_choices()
        try:
            index = _action_dict_to_factored_index(action, state, factored, self.mapper)
        except AmbiguousActionMatch:
            return None, "ambiguous"
        if index is None:
            return None, "unmatched"
        return int(index), "ok"


_IMPORTERS = {"1830": Importer1830}


def register_importer(title: str, factory) -> None:
    """Register a title's importer (a :class:`TitleImporter` subclass or factory). Milestone A2 adds
    1867's (labels from the engine's ``match_human_action``, design §5.2)."""
    _IMPORTERS[str(title)] = factory


def importer_for(title: str) -> TitleImporter:
    try:
        factory = _IMPORTERS[str(title)]
    except KeyError:
        raise UnsupportedTitle(
            f"no human-game importer for {title!r} yet (have: {', '.join(_IMPORTERS)}); "
            "TODO(A2): 1867's labels come from its layout's match_human_action"
        ) from None
    return factory()


def new_engine_game(title: str, players: Dict[int, str], optional_rules: Sequence[str] = (), seed=None):
    """A fresh ``engine_rs.BaseGame`` for replaying a record (rule variants off, as human import)."""
    import engine_rs

    players = {int(k): str(v) for k, v in players.items()}
    if title == "1830":  # as pretraining / titles.new_game build 1830 games
        return engine_rs.BaseGame(players, optional_rules=list(optional_rules))
    return engine_rs.BaseGame.new_titled(title, players, list(optional_rules) or None, seed)


# ---------------------------------------------------------------------------------------------------
# Outcome targets
# ---------------------------------------------------------------------------------------------------


def outcome_targets(worth: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """``(win share, placement points)`` per seat from net worth in seating order: winners share 1
    (ties split, as ``policy_gradient.outcome``), places as ``policy_gradient.placement_points`` with
    the player count's default points (ties share their places' points)."""
    from rl18xx.agent.alphazero.policy_gradient import placement_points

    worth = np.asarray(worth, dtype=np.float64)
    winners = worth >= worth.max() - 1e-6
    return (winners / winners.sum()).astype(np.float32), placement_points(worth).astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------------


def in_validation(game_id: str, validation_percentage: float = DEFAULT_VALIDATION_PERCENTAGE) -> bool:
    from rl18xx.agent.alphazero.pretraining import in_validation_split

    return in_validation_split(game_id, validation_percentage)


def convert_game(
    game: dict,
    importer: TitleImporter,
    validation_percentage: float = DEFAULT_VALIDATION_PERCENTAGE,
    skip_forced: bool = False,
) -> Tuple[Optional[dict], dict]:
    """One cleaned human game -> ``(record or None, stats)``.

    Every action of the replayed log is a candidate decision; its legal list is
    ``model_generic.legal_actions`` (layout indices, descriptor rows, price ranges). A position whose
    action the importer cannot match to exactly one legal index is skipped and counted by the action's
    type (``unmatched`` / ``ambiguous`` / ``not_legal``) and the action is still applied. Forced
    positions (one legal index) are kept unless ``skip_forced`` (as the legacy converter's default)."""
    from rl18xx.agent.alphazero import model_generic as mg
    from rl18xx.agent.alphazero.action_descriptors import action_descriptors
    from rl18xx.agent.alphazero.generic_tokens import encode_batch

    gid = str(game["id"])
    stats: dict = {"games": 1}
    actions, reason = importer.load(game)
    if actions is None:
        stats["dropped"] = {reason.split(":")[0]: 1}
        stats["dropped_detail"] = {gid: reason}
        return None, stats
    title = importer.title
    d = action_descriptors(title)
    players = importer.players(game)
    rules = importer.optional_rules(game)
    seed = importer.seed(game)
    state = importer.new_state(players, rules, seed)
    raw = importer.raw(state)
    n = len(players)

    action_pos, chosen, target, family, price, meta = [], [], [], [], [], []
    legal_lengths, legal_flat, legal_ranges = [], [], []
    matched_by_type: Counter = Counter()
    matched_by_family: Counter = Counter()
    skipped: Dict[str, Counter] = defaultdict(Counter)
    forced = forced_skipped = price_targets = price_outside = 0
    families = d.families
    for pos, action in enumerate(actions):
        atype = str(action.get("type", "?"))
        legal = mg.legal_actions(raw)
        index, status = importer.match(state, action, legal.indices)
        if index is not None:
            where = np.flatnonzero(legal.indices == index)
            if len(where) != 1:
                index, status = None, "not_legal"
        if index is None:
            skipped[status][atype] += 1
            importer.apply(state, action)
            continue
        if len(legal) == 1:
            if skip_forced:
                forced_skipped += 1
                importer.apply(state, action)
                continue
            forced += 1
        row = int(where[0])
        p = -1
        if legal.rows[row, mg.C_PRICE_KIND] >= 0:
            human = importer.price(action)
            lo, hi, _ = (int(x) for x in legal.price_range[row])
            if human is not None:
                if lo <= human <= hi:
                    p = human
                    price_targets += hi > lo
                else:
                    price_outside += 1
        enc_meta = encode_batch([raw])["meta"][0]
        action_pos.append(pos)
        chosen.append(index)
        target.append(row)
        fam = int(legal.rows[row, mg.C_FAMILY])
        family.append(fam)
        price.append(p)
        meta.append(enc_meta)
        legal_lengths.append(len(legal))
        legal_flat.append(legal.indices.astype(np.int32))
        legal_ranges.append(legal.price_range.astype(np.int32))
        matched_by_type[atype] += 1
        matched_by_family[families[fam]] += 1
        importer.apply(state, action)

    result = raw.result()
    ids = list(players)
    worth = np.array([float(result.get(pid, 0)) for pid in ids], dtype=np.float64)
    win_share, placement = outcome_targets(worth)
    split = "validation" if in_validation(gid, validation_percentage) else "train"
    offsets = np.concatenate([[0], np.cumsum(legal_lengths, dtype=np.int64)]).astype(np.int64)
    record = {
        "format": DATA_FORMAT,
        "id": gid,
        "title": title,
        "layout_version": d.version,
        "layout_fingerprint": d.fingerprint,
        "split": split,
        "num_players": n,
        "players": [[int(pid), name] for pid, name in players.items()],
        "optional_rules": list(rules),
        "seed": seed,
        "auction_unlock": False,
        "no_buyback_rule": False,
        "finished": bool(raw.finished),
        "actions": json.dumps(actions, separators=(",", ":")),
        "num_actions": len(actions),
        "action_pos": np.asarray(action_pos, dtype=np.int32),
        "chosen_index": np.asarray(chosen, dtype=np.int32),
        "target": np.asarray(target, dtype=np.int32),
        "family": np.asarray(family, dtype=np.int16),
        "price": np.asarray(price, dtype=np.int32),
        "meta": np.asarray(meta, dtype=np.int32).reshape(-1, 8),
        "legal_offsets": offsets,
        "legal_indices": np.concatenate(legal_flat) if legal_flat else np.zeros(0, np.int32),
        "legal_price_range": np.concatenate(legal_ranges) if legal_ranges else np.zeros((0, 3), np.int32),
        "worth": worth.astype(np.float32),
        "win_share": win_share,
        "placement": placement,
    }
    stats.update(
        converted=1,
        positions=len(action_pos),
        actions=len(actions),
        forced=forced,
        forced_skipped=forced_skipped,
        price_targets=price_targets,
        price_outside_range=price_outside,
        unfinished=int(not raw.finished),
        matched_by_type=dict(matched_by_type),
        matched_by_family=dict(matched_by_family),
        **{status: dict(c) for status, c in skipped.items()},
    )
    stats[f"positions_{split}"] = len(action_pos)
    stats[f"games_{split}"] = 1
    stats[f"positions_{n}p"] = len(action_pos)
    return record, stats


def _merge_stats(total: dict, part: dict) -> None:
    for key, value in part.items():
        if isinstance(value, dict):
            sub = total.setdefault(key, {})
            for k, v in value.items():
                sub[k] = sub.get(k, 0) + v if isinstance(v, (int, float)) else v
        else:
            total[key] = total.get(key, 0) + value


def encode_record(record: dict) -> bytes:
    return lz4.frame.compress(pickle.dumps(record, protocol=5))


def decode_record(blob: bytes) -> dict:
    return pickle.loads(lz4.frame.decompress(blob))


def list_game_files(sources: Sequence[str]) -> List[Path]:
    """Cleaned-game JSON files of ``sources`` (directories or files), one per game id (first wins),
    sorted by id."""
    seen: Dict[str, Path] = {}
    for src in sources:
        p = Path(src)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            seen.setdefault(f.stem, f)
    return [seen[k] for k in sorted(seen, key=lambda s: (len(s), s))]


_WORKER_IMPORTER: Optional[TitleImporter] = None


def _convert_file(job: tuple) -> tuple:
    path, title, validation_percentage, skip_forced = job
    global _WORKER_IMPORTER
    if _WORKER_IMPORTER is None or _WORKER_IMPORTER.title != title:
        logging.getLogger("rl18xx").setLevel(logging.ERROR)
        _WORKER_IMPORTER = importer_for(title)
    gid = Path(path).stem
    try:
        with open(path) as f:
            game = json.load(f)
        game["id"] = gid
        record, stats = convert_game(game, _WORKER_IMPORTER, validation_percentage, skip_forced)
    except Exception as e:  # never lose the whole conversion to one game
        reason = f"convert_error: {type(e).__name__}: {str(e)[:120]}"
        return gid, None, {"games": 1, "dropped": {"convert_error": 1}, "dropped_detail": {gid: reason}}, None
    if record is None:
        return gid, None, stats, None
    entry = {
        "id": record["id"],
        "split": record["split"],
        "players": record["num_players"],
        "decisions": record_decisions(record),
        "actions": record["num_actions"],
        "finished": record["finished"],
    }
    return gid, encode_record(record), stats, entry


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or None
    except Exception:
        return None


def convert_dataset(
    sources: Sequence[str],
    out: str,
    title: str = "1830",
    workers: int = 4,
    limit: Optional[int] = None,
    validation_percentage: float = DEFAULT_VALIDATION_PERCENTAGE,
    skip_forced: bool = False,
    progress_every: int = 200,
) -> dict:
    """Convert cleaned human games (JSON files under ``sources``) of ``title`` into a dataset directory
    ``out`` (must not exist yet). ``limit`` takes the first N game files by id. Returns the meta dict
    (also written to ``meta.json``)."""
    import lmdb

    from rl18xx.agent.alphazero.action_descriptors import action_descriptors
    from rl18xx.agent.alphazero.generic_tokens import schema

    importer_for(title)  # fail early for a title without an importer
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; convert into a new directory")
    out.mkdir(parents=True, exist_ok=True)
    files = list_game_files(sources)
    if limit:
        files = files[:limit]
    jobs = [(str(f), title, validation_percentage, skip_forced) for f in files]
    LOGGER.info("Converting %d %s games from %s into %s (%d workers)", len(jobs), title, list(sources), out, workers)
    env = lmdb.open(str(out), map_size=_LMDB_MAP_SIZE, subdir=True, max_dbs=0)
    stats: dict = {}
    index: List[dict] = []
    started = time.time()
    txn = env.begin(write=True)
    pending = 0

    def handle(result):
        nonlocal txn, pending
        gid, blob, part, entry = result
        _merge_stats(stats, part)
        if blob is not None:
            txn.put(f"g:{gid}".encode(), blob)
            index.append(entry)
            pending += 1
            if pending >= 64:
                txn.commit()
                txn = env.begin(write=True)
                pending = 0

    try:
        if workers > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_quiet_worker) as pool:
                for k, result in enumerate(pool.imap_unordered(_convert_file, jobs, chunksize=4)):
                    handle(result)
                    if progress_every and (k + 1) % progress_every == 0:
                        LOGGER.info("%d/%d games (%.0f s)", k + 1, len(jobs), time.time() - started)
        else:
            for k, job in enumerate(jobs):
                handle(_convert_file(job))
                if progress_every and (k + 1) % progress_every == 0:
                    LOGGER.info("%d/%d games (%.0f s)", k + 1, len(jobs), time.time() - started)
        index.sort(key=lambda e: (len(e["id"]), e["id"]))
        d = action_descriptors(title)
        meta = {
            "format": DATA_FORMAT,
            "kind": "generic_supervised",
            "titles": [title],
            "layouts": {title: {"version": d.version, "fingerprint": d.fingerprint, "total": d.total}},
            "encoder_version_at_conversion": int(schema()["version"]),
            "value_frame": "seating order (record['players']); collate rotates into the encoder's seats",
            "value_targets": "win_share (ties split), placement (policy_gradient.default_placement_points), worth ($)",
            "validation_percentage": validation_percentage,
            "split": "pretraining.in_validation_split(game id, validation_percentage)",
            "skip_forced": skip_forced,
            "sources": [str(s) for s in sources],
            "limit": limit,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _git_commit(),
            "seconds": round(time.time() - started, 1),
            "counts": {
                "games": {s: sum(1 for e in index if e["split"] == s) for s in SPLITS},
                "positions": {s: sum(e["decisions"] for e in index if e["split"] == s) for s in SPLITS},
                "positions_by_players": {
                    str(n): sum(e["decisions"] for e in index if e["players"] == n)
                    for n in sorted({e["players"] for e in index})
                },
            },
            "stats": stats,
        }
        txn.put(_META_KEY, json.dumps(meta).encode())
        txn.commit()
        txn = None
    finally:
        if txn is not None:
            txn.abort()
        env.sync()
        env.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    with open(out / "games.jsonl", "w") as f:
        for e in index:
            f.write(json.dumps(e) + "\n")
    LOGGER.info(
        "Converted %d games (%d train / %d validation positions) in %.0f s; dropped %s",
        len(index), meta["counts"]["positions"]["train"], meta["counts"]["positions"]["validation"],
        meta["seconds"], stats.get("dropped", {}),
    )
    return meta


def _quiet_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    logging.getLogger().setLevel(logging.ERROR)


# ---------------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------------


class GenericDataset:
    """A converted dataset: ``meta``, the per-game ``index`` and the records (LMDB opened lazily, so
    the object pickles into loader workers before it is used)."""

    def __init__(self, path, check_layouts: bool = True):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.meta.get("format") != DATA_FORMAT:
            raise DatasetMismatch(f"{path}: data format {self.meta.get('format')} != {DATA_FORMAT}")
        with open(self.path / "games.jsonl") as f:
            self.index = [json.loads(line) for line in f if line.strip()]
        self._env = None
        if check_layouts:
            self.check_layouts()

    def check_layouts(self) -> None:
        """Refuse a dataset whose layout indices mean something else in the running engine."""
        from rl18xx.agent.alphazero.action_descriptors import action_descriptors

        for title, saved in self.meta["layouts"].items():
            d = action_descriptors(title)
            if (d.version, d.fingerprint) != (saved["version"], saved["fingerprint"]):
                raise DatasetMismatch(
                    f"{self.path}: {title} layout {saved['version']} {saved['fingerprint']} != engine "
                    f"{d.version} {d.fingerprint}; re-convert the games"
                )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    def _open(self):
        if self._env is None:
            import lmdb

            self._env = lmdb.open(
                str(self.path), readonly=True, lock=False, readahead=False, subdir=True, max_readers=512
            )
        return self._env

    @property
    def titles(self) -> List[str]:
        return list(self.meta["titles"])

    def game_ids(self, split: Optional[str] = None) -> List[str]:
        return [e["id"] for e in self.index if split is None or e["split"] == split]

    def num_positions(self, split: Optional[str] = None) -> int:
        return sum(e["decisions"] for e in self.index if split is None or e["split"] == split)

    def record(self, game_id: str) -> dict:
        with self._open().begin() as txn:
            blob = txn.get(f"g:{game_id}".encode())
        if blob is None:
            raise KeyError(f"{self.path}: no game {game_id}")
        return decode_record(blob)


def record_decisions(record: dict) -> int:
    return int(len(record["action_pos"]))


def replay(record: dict, decisions: Optional[Iterable[int]] = None) -> Iterator[Tuple[int, object]]:
    """Replay a record and yield ``(decision, engine game)`` at each wanted decision (all by default),
    the game positioned **before** the decision's action. The game object is live: encode or clone it
    before advancing the iterator."""
    count = record_decisions(record)
    wanted = sorted(set(range(count) if decisions is None else decisions))
    if not wanted:
        return
    if wanted[0] < 0 or wanted[-1] >= count:
        raise IndexError(f"game {record['id']} has decisions 0..{count - 1}, asked for {wanted[0]}..{wanted[-1]}")
    actions = json.loads(record["actions"])
    players = {int(pid): name for pid, name in record["players"]}
    game = new_engine_game(record["title"], players, record["optional_rules"], record.get("seed"))
    if record.get("auction_unlock"):
        game.set_auction_unlock(True)
    if record.get("no_buyback_rule"):
        game.set_no_buyback_rule(True)
    action_pos = record["action_pos"]
    k = 0
    for pos, action in enumerate(actions):
        while k < len(wanted) and int(action_pos[wanted[k]]) == pos:
            yield wanted[k], game
            k += 1
        if k >= len(wanted):
            return
        _apply_logged(game, action)


def _apply_logged(game, action: dict) -> None:
    # A logged action replays exactly; the adapter's no-ops (message / undo / redo / end_game of a
    # finished game) never reach a cleaned log.
    if action.get("type") in ("message", "undo", "redo"):
        return
    game.process_action(action)


def seat_frame(values: np.ndarray, rotation: int, num_players: int) -> np.ndarray:
    """Seating-order per-player ``values`` (n,) in the encoder's seat frame, padded to ``MAX_SEATS``:
    seat k is player ``(k + rotation) mod n`` (``docs/rl_encoder.md`` "Seats")."""
    out = np.zeros(MAX_SEATS, dtype=np.float32)
    n = int(num_players)
    out[:n] = np.asarray(values, dtype=np.float32)[(np.arange(n) + int(rotation)) % n]
    return out


def position_item(record: dict, decision: int, game, descriptors=None) -> dict:
    """One decision as a training item: the position's encoding (``encode_batch([game])``), its legal
    descriptor rows and ranges, the target row, the price target and the outcome in the encoder's seat
    frame. Raises if the replayed position's mover differs from the converted one's."""
    from rl18xx.agent.alphazero.action_descriptors import action_descriptors
    from rl18xx.agent.alphazero.generic_tokens import encode_batch

    enc = encode_batch([game])
    if enc.get("map_index") is None:
        enc.pop("map_index", None)
    m = enc["meta"][0]
    saved = record["meta"][decision]
    if int(m[5]) != int(saved[5]) or int(m[0]) != int(saved[0]):
        raise RuntimeError(
            f"game {record['id']} decision {decision}: replay diverged (encoder meta {m.tolist()} vs "
            f"converted {saved.tolist()})"
        )
    d = descriptors or action_descriptors(record["title"])
    lo, hi = int(record["legal_offsets"][decision]), int(record["legal_offsets"][decision + 1])
    indices = record["legal_indices"][lo:hi]
    n = int(record["num_players"])
    rot = int(m[5])
    return {
        "enc": enc,
        "rows": d.gather(indices),
        "price_range": record["legal_price_range"][lo:hi],
        "indices": indices,
        "target": int(record["target"][decision]),
        "price": int(record["price"][decision]),
        "family": int(record["family"][decision]),
        "num_players": n,
        "win": seat_frame(record["win_share"], rot, n),
        "score": seat_frame(record["placement"], rot, n),
        "game": record["id"],
        "decision": int(decision),
    }


def iter_items(record: dict, decisions: Optional[Iterable[int]] = None) -> Iterator[dict]:
    from rl18xx.agent.alphazero.action_descriptors import action_descriptors

    d = action_descriptors(record["title"])
    for decision, game in replay(record, decisions):
        yield position_item(record, decision, game, d)


# ---------------------------------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------------------------------


def _pad_value(key: str):
    return -1 if key in _PAD_MINUS_ONE else 0


def concat_encodings(encs: Sequence[dict]) -> dict:
    """Stack single-position (or batch) encodings into one batch, padding hexes / entities / relations
    to the batch maxima with ``encode_batch``'s own padding values (bit-identical to encoding the
    positions together without map de-duplication)."""
    out = {}
    for keys in (_HEX_KEYS, _ENTITY_KEYS, _RELATION_KEYS):
        width = max(int(e[keys[0]].shape[1]) for e in encs)
        for key in keys:
            parts = []
            for e in encs:
                a = e[key]
                if a.shape[1] < width:
                    pad = np.full((a.shape[0], width - a.shape[1]) + a.shape[2:], _pad_value(key), dtype=a.dtype)
                    a = np.concatenate([a, pad], axis=1)
                parts.append(a)
            out[key] = np.concatenate(parts, axis=0)
    out["meta"] = np.concatenate([e["meta"] for e in encs], axis=0)
    out["map_index"] = None
    return out


def collate(items: Sequence[dict], config=None) -> dict:
    """Items (:func:`position_item`) -> one batch: ``inputs`` (``model_generic.build_inputs``' arrays:
    tokens + flat legal list) and the targets -- ``target_index`` (B,) into each position's legal list,
    ``price_cell`` / ``price_log_count`` / ``price_valid`` (P,) for ``inputs["price_rows"]`` (the chosen
    row's observed price; other priced rows none), ``win`` / ``score`` (B, 6) in the seat frame,
    ``family`` (B,) of the target row, ``num_players`` (B,)."""
    from rl18xx.agent.alphazero import model_generic as mg

    arrays = concat_encodings([it["enc"] for it in items])
    legal = [mg.LegalActions(rows=it["rows"], price_range=it["price_range"]) for it in items]
    mg.add_legal_inputs(arrays, legal, config=config)
    target = np.array([it["target"] for it in items], dtype=np.int64)
    offsets = arrays["legal_offsets"]
    prices = np.full(len(arrays["price_rows"]), -1, dtype=np.int64)
    if len(prices):
        pos = arrays["legal_pos"][arrays["price_rows"]]
        chosen = arrays["price_rows"] == offsets[pos] + target[pos]
        item_price = np.array([it["price"] for it in items], dtype=np.int64)
        prices = np.where(chosen, item_price[pos], -1)
    out = {
        "inputs": arrays,
        "target_index": target,
        "win": np.stack([it["win"] for it in items]).astype(np.float32),
        "score": np.stack([it["score"] for it in items]).astype(np.float32),
        "family": np.array([it["family"] for it in items], dtype=np.int64),
        "num_players": np.array([it["num_players"] for it in items], dtype=np.int64),
    }
    out.update(mg.price_targets(arrays, prices))
    return out


def batch_to_device(batch: dict, device) -> dict:
    """A collated batch as tensors on ``device`` (``model_generic.to_device`` for every array)."""
    from rl18xx.agent.alphazero import model_generic as mg

    out = {"inputs": mg.to_device(batch["inputs"], device)}
    out.update(mg.to_device({k: v for k, v in batch.items() if k != "inputs"}, device))
    return out


# ---------------------------------------------------------------------------------------------------
# Streaming batches
# ---------------------------------------------------------------------------------------------------

try:  # torch is a dependency of the training side only
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:  # pragma: no cover
    _IterableDataset = object


class PositionBatches(_IterableDataset):
    """Collated batches of one split. Loader workers (``torch.utils.data.DataLoader(..., batch_size=None,
    num_workers=W)``) each take every W-th game of the epoch's game order, replay it and encode its
    decisions. With ``shuffle`` the game order is permuted per (seed, epoch) and positions pass
    through a shuffle buffer of ``buffer_positions`` items (LZ4-compressed, ~5 KB each for 1830), so a
    batch mixes ~buffer / (positions per game) games; a worker's trailing partial batch is dropped
    (``drop_last``, default with shuffle). Without ``shuffle`` the order is fixed (validation).
    ``max_games`` truncates the game list (smoke runs, partial evals)."""

    def __init__(
        self,
        path,
        split: str,
        batch_size: int,
        shuffle: bool,
        buffer_positions: int = 32768,
        seed: int = 0,
        epoch: int = 0,
        max_games: Optional[int] = None,
        drop_last: Optional[bool] = None,
        config=None,
    ):
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.path = str(path)
        self.split = split
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.buffer_positions = max(int(buffer_positions), self.batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.max_games = max_games
        self.drop_last = self.shuffle if drop_last is None else bool(drop_last)
        self.config = config

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def game_order(self, dataset: GenericDataset) -> List[str]:
        ids = dataset.game_ids(self.split)
        if self.max_games:
            ids = ids[: self.max_games]
        if self.shuffle:
            rng = random.Random(f"{self.seed}:{self.epoch}")
            rng.shuffle(ids)
        return ids

    def __iter__(self) -> Iterator[dict]:
        import torch

        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        dataset = GenericDataset(self.path, check_layouts=False)
        ids = self.game_order(dataset)[wid::nw]
        rng = random.Random(f"{self.seed}:{self.epoch}:{wid}")
        if not self.shuffle:
            batch: List[dict] = []
            for gid in ids:
                for item in iter_items(dataset.record(gid)):
                    batch.append(item)
                    if len(batch) == self.batch_size:
                        yield collate(batch, self.config)
                        batch = []
            if batch and not self.drop_last:
                yield collate(batch, self.config)
            return
        buffer: List[bytes] = []
        for gid in ids:
            for item in iter_items(dataset.record(gid)):
                buffer.append(lz4.frame.compress(pickle.dumps(item, protocol=5)))
                if len(buffer) >= self.buffer_positions:
                    yield collate([_draw(buffer, rng) for _ in range(self.batch_size)], self.config)
        rng.shuffle(buffer)
        while len(buffer) >= self.batch_size or (buffer and not self.drop_last):
            take = min(self.batch_size, len(buffer))
            yield collate([pickle.loads(lz4.frame.decompress(buffer.pop())) for _ in range(take)], self.config)


def _draw(buffer: List[bytes], rng: random.Random) -> dict:
    """Remove and return a random buffer entry (swap with the last: O(1))."""
    i = rng.randrange(len(buffer))
    buffer[i], buffer[-1] = buffer[-1], buffer[i]
    return pickle.loads(lz4.frame.decompress(buffer.pop()))


def loader_worker_init(_worker_id: int) -> None:
    """``DataLoader`` worker init: one torch / BLAS thread per loader worker."""
    import torch

    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


__all__ = [
    "DATA_FORMAT",
    "DEFAULT_VALIDATION_PERCENTAGE",
    "DatasetMismatch",
    "GenericDataset",
    "Importer1830",
    "PositionBatches",
    "TitleImporter",
    "UnsupportedTitle",
    "batch_to_device",
    "collate",
    "concat_encodings",
    "convert_dataset",
    "convert_game",
    "importer_for",
    "iter_items",
    "new_engine_game",
    "outcome_targets",
    "position_item",
    "register_importer",
    "replay",
    "seat_frame",
]
