"""Fast pytest gates over the manual full-corpus parity harnesses.

The full gates (``tests/parity_runner.py``, ``tests/index_parity_corpus.py``,
``tests/cleaning_diff.py``, ``tests/decode_parity_check.py``) take hours over
the whole corpus and are run manually before/after engine refactors (see
``docs/verification_rituals.md``). This module keeps a small, deterministic
slice of each in the default ``pytest tests/`` run so parity regressions fail
loud in CI instead of waiting for the next manual sweep.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.disable(logging.CRITICAL)

from tests import cleaning_diff  # noqa: E402
from tests.parity_runner import run_random_seed  # noqa: E402

HUMAN_GAMES_DIR = Path(__file__).parent.parent / "human_games" / "1830"

# Statuses from cleaning_diff that indicate the engines agreed: either both
# replayed the game in lockstep ("parity") or both dropped it consistently
# ("dropped"). Anything else (rust_error, state_divergence, enum_divergence,
# index_divergence, decode_divergence, pass_acceptance_divergence,
# python_error, ...) is a failure of this gate.
GOOD_STATUSES = {"parity", "dropped"}

# Known standing non-failure: 87895 yields python_error (a Python-oracle
# WaterfallAuction limitation, not a Rust divergence) — never select it here.
EXCLUDED_GAME_IDS = {"87895"}


def _smallest_games():
    """Deterministic small-game selection: the 3 smallest .json by byte size
    plus the 2 smallest at >= 10 KB (the sub-KB files are stub games both
    engines drop before any lockstep, so the >=10KB pair makes sure the
    strict per-step lockstep is actually exercised)."""
    candidates = sorted(
        (p for p in HUMAN_GAMES_DIR.glob("*.json") if p.stem not in EXCLUDED_GAME_IDS),
        key=lambda p: (p.stat().st_size, p.name),
    )
    if not candidates:
        return []  # empty parametrization -> pytest reports the tests as skipped
    small = candidates[:3]
    real = [p for p in candidates if p.stat().st_size >= 10_000][:2]
    return small + real


@pytest.mark.parametrize("seed", [42, 43, 44, 45, 46])
def test_parity_runner_random_fast(seed):
    """parity_runner random mode (state + factored enumeration with exact
    prices at every step), bounded to 150 steps per seed."""
    failure = run_random_seed(seed, max_steps=150)
    assert failure is None, f"parity_runner seed={seed} failed: {failure}"


@pytest.mark.parametrize(
    "path", _smallest_games(), ids=lambda p: f"{p.stem}-{p.stat().st_size}B"
)
def test_cleaning_diff_strict_small_games(path):
    """Strict cleaning_diff lockstep (state + enumeration + policy-index +
    native-decode + acceptance at every step of the real human import) over a
    deterministic set of small human games."""
    cleaning_diff.set_check_decode(True)  # all four parity axes, incl. decode
    try:
        res = cleaning_diff.run(str(path), strict=True)
    finally:
        cleaning_diff.set_check_decode(False)
    assert res["status"] in GOOD_STATUSES, (
        f"game {path.stem}: status={res['status']!r} (expected parity/dropped): "
        f"{ {k: v for k, v in res.items() if k not in ('trace',)} }"
    )
