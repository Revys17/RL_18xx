# Pretraining → Rust migration: status and blockers

The plan called for pretraining to run on the Rust engine ("Once landed, both
pretraining and self-play run entirely on the Rust engine"). Self-play and
MCTS already use Rust exclusively. Pretraining still touches the Python
engine in two places:

```
rl18xx/agent/alphazero/pretraining.py:517  game_state = game_class(players)  # Python BaseGame
rl18xx/agent/alphazero/pretraining.py:686  fresh_game_state = game_class(players)
```

## Why this isn't a one-line replacement

`get_game_object_for_game` (pretraining.py:509) is the data-cleaning pass
that runs once per game JSON. It uses Python BaseGame entity-resolution APIs
that `RustGameAdapter` doesn't expose:

- `game_state.get(entity_type, entity_id)` — used by `BaseAction.action_from_dict` (actions.py:95)
- `game_state.company_by_id(name)` — used for cross-player buy_company filter (line 548)
- `game_state.train_by_id(id).owner` — cross-president buy_train filter (line 558)
- `game_state.company_by_id("MH").player()` — MH out-of-turn filter (line 569)
- `game_state.share_by_id(name)` — illegal_share_buy filter (line 593)
- `game_state.current_entity.player()` — used by filter heuristics
- `game_state.round.active_step()` — used to dispatch on round state

`convert_game_to_training_data` (pretraining.py:668) — the per-iteration
training-data conversion — also calls `BaseAction.action_from_dict` to get
an action object for the action-mapper helpers.

## What needs to land before migration is feasible

1. **RustGameAdapter API parity for entity resolution**: implement `.get()`,
   `.company_by_id()`, `.train_by_id()`, `.share_by_id()`, `.current_entity`,
   `.round.active_step()`. Several already exist; the missing ones need
   PyO3 wrappers around the Rust engine's existing accessors.
2. **Action-dict-direct paths in action_mapper**: alternatively, add a
   `canonical_index_for_action_dict(action_dict, state)` variant that
   doesn't need a constructed `BaseAction` object — letting pretraining
   skip `BaseAction.action_from_dict` entirely on the Rust path.

## Recommended approach (when picked up)

- Step 1 (`get_game_object_for_game`) stays Python. It's a one-time prep
  pass that produces `human_games/1830_clean/*.json`; the cleaned JSONs
  are then read by step 2. Migrating step 1 buys little since it doesn't
  run during the training loop.
- Step 2 (`convert_game_to_training_data`) migrates to Rust. This is the
  hot path during pretraining. It needs:
  - Replace `fresh_game_state = game_class(players)` with
    `RustGameAdapter(RustGame(players))`.
  - Replace `BaseAction.action_from_dict(action, fresh_game_state)` with
    the dict-direct action_mapper variant (point 2 above) so the action
    object doesn't need entity resolution.
  - Keep the input `game: BaseGame` for read-only access to `players`,
    `result()`, `raw_actions` (these the Python-loaded JSON already
    provides).

## Engine-parity status that gates this

At the time of writing, `docs/rust_engine_corpus_audit.md` reports
**99.73% of replay-eligible human games match exactly** between Python
and Rust (1113 / 1116). The remaining 3 games are all MH any-time exchange
edge cases or a known action.entity-mismatch case in game 54156 — neither
should block training data generation on the cleaned corpus.

So migrating step 2 to Rust is engine-blocker-free; the work is purely on
the API surface side.
