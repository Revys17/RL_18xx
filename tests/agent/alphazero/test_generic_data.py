"""Title-generic supervised data (``generic_data``, RL bridge milestone C1) on 1830 fixture games.

Checked: the labels, legal lists and price targets equal the legacy converter's
(``pretraining.convert_game_to_training_data``) position for position; outcome targets in seating
order and their rotation into the encoder's seats; forced positions kept (``skip_forced`` drops and
counts them); a match that is not exactly one legal index is skipped and counted, never guessed; the
split is ``in_validation_split``'s and stable; a dataset round trip (records, every position once through
the streaming loader, the layout check); the collate equals ``build_inputs`` bit for bit, padding included
(1830 + 1867 encodings in one batch).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

engine_rs = pytest.importorskip("engine_rs")

from rl18xx.agent.alphazero import generic_data as gd  # noqa: E402
from rl18xx.agent.alphazero import model_generic as mg  # noqa: E402
from rl18xx.agent.alphazero.generic_tokens import encode_batch, raw_game  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "1830"
# Cleaned-format 1830 games: 2 players to the end (D trains), 4 players to the end, 4 players cut mid-game.
GAMES = ("router_d_trains.json", "train_rotation_game.json", "buy_train_at_limit.json")


def load_fixture(name: str) -> dict:
    game = json.loads((FIXTURES / name).read_text())
    game["id"] = Path(name).stem
    game.setdefault("title", "1830")
    return game


@pytest.fixture(scope="module")
def importer():
    return gd.importer_for("1830")


@pytest.fixture(scope="module")
def records(importer):
    out = {}
    for name in GAMES:
        record, stats = gd.convert_game(load_fixture(name), importer)
        assert record is not None, stats
        out[name] = (record, stats)
    return out


# ---------------------------------------------------------------------------------------------------
# Labels: the legacy converter's, index for index
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", GAMES)
def test_matches_legacy_converter(name, records):
    from rl18xx.agent.alphazero import pretraining as pt
    from rl18xx.agent.alphazero.config import TrainingConfig

    record, stats = records[name]
    game = load_fixture(name)
    loaded = pt._load_cleaned_game_via_rust(game)
    assert loaded is not None
    stub_encoder = SimpleNamespace(encode=lambda state: None)
    legacy_stats: dict = {}
    train, val = pt.convert_game_to_training_data(
        loaded, stub_encoder, config=TrainingConfig(), stats=legacy_stats, game_id=game["id"]
    )
    legacy = train + val
    assert len(legacy) == gd.record_decisions(record) == stats["positions"]
    assert legacy_stats.get("unmatched", {}) == stats.get("unmatched", {})
    assert legacy_stats.get("ambiguous", {}) == stats.get("ambiguous", {})
    assert (record["split"] == "validation") == bool(val) or not legacy
    offsets = record["legal_offsets"]
    for k, (_, legal, pi, _, price_targets) in enumerate(legacy):
        ours = record["legal_indices"][offsets[k] : offsets[k + 1]].tolist()
        assert ours == list(legal), f"decision {k}: legal lists differ"
        assert int(record["chosen_index"][k]) == int(pi.argmax())
        assert ours[int(record["target"][k])] == int(record["chosen_index"][k])
        if price_targets:  # legacy: (slot, price, weight, lo, hi) for an open price
            (_, price, _, lo, hi) = price_targets[0]
            row = offsets[k] + int(record["target"][k])
            assert int(record["price"][k]) == int(price)
            assert tuple(record["legal_price_range"][row][:2]) == (int(lo), int(hi))
        else:
            row = offsets[k] + int(record["target"][k])
            lo, hi, _ = record["legal_price_range"][row]
            assert int(record["price"][k]) < 0 or lo == hi


def test_legal_rows_and_price_ranges_are_model_generic(records):
    """The stored legal lists and ranges are ``model_generic.legal_actions``' (one place for the step)."""
    record, _ = records["router_d_trains.json"]
    d = mg.vocab()
    picks = list(range(0, gd.record_decisions(record), 37))
    for k, game in gd.replay(record, picks):
        la = mg.legal_actions(game)
        lo, hi = record["legal_offsets"][k], record["legal_offsets"][k + 1]
        assert la.indices.tolist() == record["legal_indices"][lo:hi].tolist()
        assert np.array_equal(la.price_range, record["legal_price_range"][lo:hi])
    assert d["families"][int(record["family"][0])] in d["families"]


# ---------------------------------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------------------------------


def test_outcome_targets_in_seating_order(records):
    from rl18xx.agent.alphazero.policy_gradient import placement_points

    for name in GAMES:
        record, _ = records[name]
        # the final state's result, seat by seat in the record's seating order
        game = gd.new_engine_game("1830", {pid: n for pid, n in record["players"]}, record["optional_rules"])
        for action in json.loads(record["actions"]):
            game.process_action(action)
        result = game.result()
        worth = [result[pid] for pid, _ in record["players"]]
        assert record["worth"].tolist() == pytest.approx(worth)
        assert record["win_share"].sum() == pytest.approx(1.0)
        assert record["win_share"][int(np.argmax(worth))] > 0
        assert record["placement"].tolist() == pytest.approx(placement_points(worth).tolist())
        assert [pid for pid, _ in record["players"]] == [p.id for p in game.players]


def test_outcome_ties_split():
    win, place = gd.outcome_targets([100, 300, 300, 50])
    assert win.tolist() == pytest.approx([0, 0.5, 0.5, 0])
    assert place.tolist() == pytest.approx([0.2, 0.75, 0.75, 0.0])


def test_seat_frame_rotation():
    values = np.array([10, 20, 30, 40], dtype=np.float32)
    assert gd.seat_frame(values, 0, 4).tolist() == [10, 20, 30, 40, 0, 0]
    assert gd.seat_frame(values, 2, 4).tolist() == [30, 40, 10, 20, 0, 0]
    assert gd.seat_frame(values[:3], 1, 3).tolist() == [20, 30, 10, 0, 0, 0]


def test_items_rotate_targets_to_the_mover(records):
    """Seat 0 of an item's targets is the player deciding (the legacy encoder's rotation agrees)."""
    from rl18xx.agent.alphazero.mcts import _rust_encode
    from rl18xx.rust_adapter import RustGameAdapter

    record, _ = records["train_rotation_game.json"]
    n = record["num_players"]
    rotations = set()
    picks = list(range(0, gd.record_decisions(record), 7))
    for k, game in gd.replay(record, picks):
        item = gd.position_item(record, k, game)
        rot = int(item["enc"]["meta"][0, 5])
        rotations.add(rot)
        assert rot == int(_rust_encode(RustGameAdapter(game.pickle_clone()))[6])
        assert item["win"][0] == pytest.approx(record["win_share"][rot])
        assert item["score"][: n].tolist() == pytest.approx(np.roll(record["placement"], -rot).tolist())
        assert item["win"][n:].tolist() == [0.0] * (gd.MAX_SEATS - n)
    assert len(rotations) > 1


# ---------------------------------------------------------------------------------------------------
# Forced positions, exactly-one matching, split
# ---------------------------------------------------------------------------------------------------


def test_forced_positions_kept_and_skippable(importer, records):
    record, stats = records["router_d_trains.json"]
    lengths = np.diff(record["legal_offsets"])
    assert stats["forced"] == int((lengths == 1).sum()) > 0
    skipped, s2 = gd.convert_game(load_fixture("router_d_trains.json"), importer, skip_forced=True)
    assert s2["forced_skipped"] == stats["forced"]
    assert gd.record_decisions(skipped) == gd.record_decisions(record) - stats["forced"]
    assert (np.diff(skipped["legal_offsets"]) > 1).all()


class _Flaky(gd.Importer1830):
    """Every 5th action ambiguous, every 7th matched to an index outside the legal list."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def match(self, state, action, legal_indices):
        self.calls += 1
        if self.calls % 5 == 0:
            return None, "ambiguous"
        if self.calls % 7 == 0:
            return 10**6, "ok"
        return super().match(state, action, legal_indices)


def test_not_exactly_one_match_is_skipped_and_counted(records):
    record, stats = records["train_rotation_game.json"]
    flaky, s2 = gd.convert_game(load_fixture("train_rotation_game.json"), _Flaky())
    ambiguous = sum(s2.get("ambiguous", {}).values())
    not_legal = sum(s2.get("not_legal", {}).values())
    assert ambiguous > 0 and not_legal > 0
    assert gd.record_decisions(flaky) == gd.record_decisions(record) - ambiguous - not_legal
    # the replay went on through the skipped actions: every kept label is the unflaky one
    full = dict(zip(record["action_pos"].tolist(), record["chosen_index"].tolist()))
    for pos, idx in zip(flaky["action_pos"].tolist(), flaky["chosen_index"].tolist()):
        assert full[pos] == idx
    assert flaky["worth"].tolist() == record["worth"].tolist()


def test_split_is_the_legacy_hash_and_stable():
    from rl18xx.agent.alphazero.pretraining import in_validation_split

    ids = [str(i) for i in range(20000, 22000)]
    split = [gd.in_validation(i) for i in ids]
    assert split == [in_validation_split(i, 0.05) for i in ids]
    assert split == [gd.in_validation(i) for i in ids]
    assert 0.02 < np.mean(split) < 0.08
    wider = [gd.in_validation(i, 0.10) for i in ids]
    assert all(w for s, w in zip(split, wider) if s)  # raising the share only adds games


def test_unsupported_title():
    with pytest.raises(gd.UnsupportedTitle):
        gd.importer_for("1867")


# ---------------------------------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------------------------------


def test_collate_equals_build_inputs(records):
    record, _ = records["router_d_trains.json"]
    picks = sorted(random.Random(3).sample(range(gd.record_decisions(record)), 48))
    items, games = [], []
    for k, game in gd.replay(record, picks):
        items.append(gd.position_item(record, k, game))
        games.append(game.pickle_clone())
    batch = gd.collate(items)
    ref = mg.build_inputs(games, [mg.legal_actions(g) for g in games])
    assert set(ref) == set(batch["inputs"])
    for key, value in ref.items():
        if value is None:
            assert batch["inputs"][key] is None
        else:
            assert value.dtype == batch["inputs"][key].dtype, key
            assert np.array_equal(value, batch["inputs"][key]), key
    # targets: each position's target row is its chosen index; prices only on chosen priced rows
    offs = batch["inputs"]["legal_offsets"]
    for b, k in enumerate(picks):
        assert ref["legal_rows"][offs[b] + batch["target_index"][b]].tolist() == mg.action_descriptors(
            "1830"
        ).gather([record["chosen_index"][k]])[0].tolist()
    assert batch["price_valid"].sum() <= len(picks)


def test_concat_encodings_pads_like_encode_batch():
    from scripts.g1867_fuzz import pick as pick_1867

    from rl18xx.agent.alphazero.titles import new_game

    rng = random.Random(5)
    positions = []
    g30 = raw_game(new_game("1830", 3))
    g67 = new_game("1867", 5, seed=2)
    for step in range(120):
        if step % 40 == 0:
            positions += [g30.pickle_clone(), g67.pickle_clone()]
        idx = rng.choice([int(i) for i in g30.factored_legal_indices()])
        pr = g30.price_range_for_index(idx)
        g30.apply_action_index(idx, rng.randint(*pr) if pr else None)
        g67.process_action(pick_1867(json.loads(g67.g1867_legal_moves_json(None)), rng))
    ref = encode_batch(positions)
    cat = gd.concat_encodings([encode_batch([g]) for g in positions])
    for key, value in ref.items():
        if value is None:
            assert cat[key] is None
        else:
            assert np.array_equal(value, cat[key]), key


# ---------------------------------------------------------------------------------------------------
# Dataset round trip and the streaming loader
# ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("generic_data")
    out = root / "ds"
    gd.convert_dataset([str(p) for p in write_cleaned(root / "games")], str(out), workers=1, validation_percentage=0.5)
    return out


def write_cleaned(directory: Path) -> list:
    """The fixture games as cleaned-game files (with their title, as ``main.py clean`` writes them)."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in GAMES:
        path = directory / name
        path.write_text(json.dumps(load_fixture(name)))
        paths.append(path)
    return paths


def test_dataset_round_trip(dataset_dir, records):
    ds = gd.GenericDataset(dataset_dir)
    assert ds.meta["layouts"]["1830"]["fingerprint"] == mg.action_descriptors("1830").fingerprint
    assert sorted(ds.game_ids()) == sorted(Path(g).stem for g in GAMES)
    for name in GAMES:
        gid = Path(name).stem
        rec = ds.record(gid)
        ref, _ = records[name]
        for key in ("chosen_index", "target", "legal_indices", "legal_price_range", "price", "meta", "worth"):
            assert np.array_equal(rec[key], ref[key]), key
        assert rec["split"] == ("validation" if gd.in_validation(gid, 0.5) else "train")
    total = ds.num_positions()
    assert total == sum(gd.record_decisions(records[n][0]) for n in GAMES)


@pytest.mark.parametrize("shuffle", [False, True])
def test_position_batches_cover_every_position_once(dataset_dir, shuffle):
    ds = gd.GenericDataset(dataset_dir)
    seen = []
    for split in gd.SPLITS:
        if not ds.game_ids(split):
            continue
        stream = gd.PositionBatches(
            dataset_dir, split, batch_size=64, shuffle=shuffle, buffer_positions=200, seed=1, drop_last=False
        )
        for batch in stream:
            n = len(batch["target_index"])
            assert n <= 64
            assert len(batch["inputs"]["legal_offsets"]) == n + 1
            seen.append(n)
    assert sum(seen) == ds.num_positions()


def test_dataset_refuses_another_layout(dataset_dir, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dataset_dir, copy)
    meta = json.loads((copy / "meta.json").read_text())
    meta["layouts"]["1830"]["fingerprint"] = "0000000000000000"
    (copy / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(gd.DatasetMismatch):
        gd.GenericDataset(copy)
