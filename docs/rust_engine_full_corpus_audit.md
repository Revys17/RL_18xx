# Rust engine full-corpus audit

Audited **1991** raw games from `human_games/1830/` through the full cleaning + engine-parity pipeline.

## Headline stats

- perfect: **1661 / 1991** (83.43%)
- dropped: **316 / 1991** (15.87%)
- cleaning_error: **14 / 1991** (0.70%)

## By player count

| Players | Total | Perfect | Dropped | Diverged | Other |
|---|---|---|---|---|---|
| 2 | 35 | 24 | 11 | 0 | 0 |
| 3 | 238 | 188 | 47 | 0 | 3 |
| 4 | 1261 | 1115 | 146 | 0 | 0 |
| 5 | 334 | 245 | 82 | 0 | 7 |
| 6 | 123 | 89 | 30 | 0 | 4 |

## Drop reasons

- `illegal_share_buy`: **238**
- `cross_president_buy_train`: **29**
- `cross_player_buy_company`: **23**
- `mh_out_of_turn`: **19**
- `other`: **5**
- `mis_attributed_corp_action`: **2**

## Replay-survivors-only view

- Replay-eligible (status != dropped): **1675**
- Perfect among survivors: **1661 / 1675** (99.16%)
- Non-perfect among survivors: **14 / 1675**

Per-game rows: `docs/rust_engine_full_corpus_audit.csv`
