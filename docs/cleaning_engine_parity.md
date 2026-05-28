# Cleaning pipeline: Python vs. Rust engine parity audit

Total games audited: **100**

Discrepancies: **0**

Match rate (cleaning outcome): **100.00%**


## Category counts

| Category | Count |
|----------|-------|
| `both_ok` | 62 |
| `skip_not_finished` | 36 |
| `both_dropped_same_reason` | 2 |

## Drop-reason cross-tab (both engines dropped)

| Python reason | Rust reason | Count |
|---------------|-------------|-------|
| `mh_out_of_turn` | `mh_out_of_turn` | 1 |
| `company_tile_lay_outside_or` | `company_tile_lay_outside_or` | 1 |
