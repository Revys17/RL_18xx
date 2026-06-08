"""Regression tests for two Rust action-mapper index divergences from Python.

Both were *enumeration*-tag bugs masked by the categorical ``_key`` (which reads
only params, not the entity), so they passed enumeration parity yet produced a
different flat policy index than Python's ``ActionMapper`` — i.e. the Rust MCTS
would search a slot the Python training target never points at.

1. BuyTrain discarded depot trains.
   Python (`action_mapper.py::_index_for_factored_buy_train`) routes a depot
   train whose *name* is in ``depot.discarded`` to its per-train-type discard
   slot (``BuyTrain + 1 + type``). Rust's pure index function previously sent
   every ``source="depot"`` train to the single cheapest-depot slot, collapsing
   distinct discarded trains (e.g. a discarded 3 and a discarded 4) onto one
   index. The enumeration now stamps a ``_depot_discarded`` hint so the index
   resolves to the correct per-type slot.

2. DH F16 teleport token (CompanyPlaceToken).
   Python's ``_company_place_token_choices`` tags the F16 teleport ``PlaceToken``
   ``entity={private:"DH"}`` -> the ``CompanyPlaceToken`` slot. Rust's
   enumeration previously tagged it with the corporation descriptor -> the
   regular per-city ``PlaceToken`` block.

These exercise the Rust index function directly via ``legal_action_to_index_py``
(the layout was the source of the bug; the enumeration fixes feed it the right
descriptors). The full state-level parity is covered by the strict lockstep in
``tests/cleaning_diff.py`` / ``tests/index_parity_corpus.py``.
"""

import pytest

engine_rs = pytest.importorskip("engine_rs")
from engine_rs import legal_action_to_index_py as _idx, action_offsets_py


@pytest.fixture(scope="module")
def off():
    return action_offsets_py()


def _bt(train, discarded=False, source="depot"):
    entity = {"source": source, "train": train}
    if discarded:
        entity["_depot_discarded"] = True
    return _idx("BuyTrain", entity, {})


# --- Bug 1: BuyTrain discarded-train per-type differentiation -----------------


def test_fresh_depot_trains_share_the_single_cheapest_slot(off):
    """A fresh (non-discarded) depot train of any non-D type maps to the one
    cheapest-depot slot — the layout assumes the depot offers a single train."""
    bt = off["BuyTrain"]
    for name in ("2", "3", "4", "5", "6"):
        assert _bt(name, discarded=False) == bt


def test_discarded_depot_trains_get_distinct_per_type_slots(off):
    """The fix: discarded depot trains of *different* types must NOT collapse.

    This is the exact scenario that previously diverged from Python (a discarded
    3 and a discarded 4 both landing on the cheapest-depot slot)."""
    bt = off["BuyTrain"]
    type_offsets = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "D": 5}
    seen = {}
    for name, t in type_offsets.items():
        idx = _bt(name, discarded=True)
        assert idx == bt + 1 + t, f"discarded {name} -> {idx}, expected {bt + 1 + t}"
        assert idx not in seen, f"discarded {name} collapses onto {seen.get(idx)}"
        seen[idx] = name
    # And they are all distinct from the fresh cheapest-depot slot.
    assert bt not in seen


def test_discarded_3_and_4_are_differentiated(off):
    """Minimal restatement of the original NYNH 3/4 divergence."""
    assert _bt("3", discarded=True) != _bt("4", discarded=True)
    # Without the discard hint both would (wrongly) be the same slot.
    assert _bt("3", discarded=False) == _bt("4", discarded=False)


# --- Bug 2: DH F16 teleport token -> CompanyPlaceToken ------------------------


def test_dh_teleport_token_maps_to_company_place_token(off):
    params = {"hex": "F16", "city": 0, "slot": 0}
    assert _idx("PlaceToken", {"private": "DH"}, params) == off["CompanyPlaceToken"]


def test_regular_token_stays_in_place_token_block(off):
    params = {"hex": "F16", "city": 0, "slot": 0}
    idx = _idx("PlaceToken", {"corp": "PRR"}, params)
    assert off["PlaceToken"] <= idx < off["CompanyPlaceToken"]
    assert idx != off["CompanyPlaceToken"]
