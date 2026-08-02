# Verification rituals (Python ↔ Rust engine parity)

Two tiers: a **fast gate** that must always be green (enforced by the pre-push
hook, `scripts/pre-push.hook`), and **full corpus gates** run manually before
and after any engine/rules/decode/encoder refactor.

## Fast gate (seconds–minutes, runs in CI / pre-push)

```bash
uv run pytest tests/
# Rust in-crate tests, incl. the frozen-1830-action-layout pin (pytest never
# runs these; PyO3 needs a linkable libpython for the test binary):
PY="$(pwd)/.venv/bin/python"
LIBDIR="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
(cd engine-rs && PYO3_PYTHON="$PY" LD_LIBRARY_PATH="$LIBDIR" cargo test --release --no-default-features -q)
```

Includes bounded slices of every parity axis: random-walk lockstep
(`test_parity_gates_fast.py`, `test_rust_*_game.py`), factored-helper parity,
strict cleaning lockstep on small human games, encoder parity, adapter compat.

## Full corpus gates (hours, manual)

```bash
# Strict 4-axis lockstep over ALL human games (state, enumeration,
# policy-index, native decode). Exits 1 on any BAD status.
# NOTE: a clean full-corpus run now exits 0 with ZERO bad entries. (The
# historical 87895 python_error baseline is gone: that game's record carried a
# single dangling auction bid — a stale-client submit on an already-won lot,
# inert under Ruby, rejected by both our ports — and the offending action was
# excised from the local corpus copy on 2026-08-02 after verifying the rest of
# the game reaches full parity without it.)
uv run python tests/index_parity_corpus.py --json /tmp/index_parity.json

# Strict runner: human-game imports and/or random walks. Exits 1 on failures.
uv run python tests/parity_runner.py --human 'human_games/1830/*.json' --out /tmp/parity_human.json
uv run python tests/parity_runner.py --random 0:500 --out /tmp/parity_random.json

# Behavioral decode parity: native Rust apply_action_index vs Python
# ActionMapper decode, for every legal index at every state.
uv run python tests/decode_parity_check.py

# Single-game drill-down when any of the above flags a game:
uv run python tests/cleaning_diff.py <game.json> --strict
```
