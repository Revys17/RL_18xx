# Rust engine corpus audit

Audited **1261** games from `human_games/1830_clean/` against the Python and Rust engines side-by-side.

## Headline stats

- Perfect (zero state divergence): **1105 / 1261** (87.63%)
- Dropped by pretraining filters: **145 / 1261** (11.50%)
- Divergence in state compare: **0 / 1261** (0.00%)
- Rust engine rejected an action: **1 / 1261** (0.08%)
- Python engine raised during replay: **10 / 1261** (0.79%)
- Audit harness error: **0 / 1261** (0.00%)

## Top divergence categories (across divergence + rust_error + python_error)

- `python_error`: **10**
- `rust_rejected_dividend`: **1**

## Top first-divergence action types

- `run_routes`: **10**
- `dividend`: **1**

## Drop reasons (for status=dropped)

- `illegal_share_buy`: **109**
- `cross_president_buy_train`: **14**
- `mh_out_of_turn`: **10**
- `cross_player_buy_company`: **10**
- `optional_rules`: **2**

## Replay-survivors-only view

- Replay-eligible games (status != dropped): **1116**
- Perfect among survivors: **1105 / 1116** (99.01%)
- Divergence among survivors: **11 / 1116** (0.99%)

Per-game rows: `docs/rust_engine_corpus_audit.csv`
