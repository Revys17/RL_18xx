# 1867 corpus action shapes (18xx.games export, 2022 scrape)

Scanned 400 finished games from `human_games/1867/` (893 finished total,
1027 files; 85×2p / 308×3p / 496×4p / 117×5p / 20×6p; only optional rule in
the wild: `grid_market` ×161). This is the RAW action vocabulary the
import/replay path must parse — the corpus-side complement to the Ruby
analysis in `g1867_port_notes.md`.

## New action types vs 1830 (with observed field shapes)

| type | fields | sample |
|------|--------|--------|
| `merge` | entity (corporation), corporation | `{"type":"merge","entity":"NO","corporation":"CNR","entity_type":"corporation"}` — entity = the minor being folded in, corporation = the (forming) major |
| `convert` | entity (corporation) | `{"type":"convert","entity":"NO","entity_type":"corporation"}` — 5-share major → 10-share (and minor→major path) |
| `choose` | entity (corporation), choice | `{"type":"choose","choice":"nationalize","entity":"NYC"}` |
| `remove_token` | entity (corporation), city (tile-coord ref like `"14-0-0"`), slot | merger token reduction |

## 1830-shared types with NEW field shapes

- `par`: `share_price` is a market-cell coordinate string `"200,0,17"`
  (price, row, column) — NOT a bare price like 1830. Majors par via this.
- `bid`: bids on a `company` (privates, e.g. C&SL) **or** on a
  `corporation` (~5.6k observed — the single-item auction sells MINORS;
  minors are corporations with letter syms).
- `buy_train`: carries `variant` (e.g. `"2"`) alongside `train` id
  (`"2-0"`); `exchange` appears (~524) for trade-ins.
- `buy_shares`: `shares: ["CNR_1"]` cert-id lists + `percent`;
  corporations as buyers appear (~337 — share REDEMPTION is
  `buy_shares` by the corp; `sell_shares` by corps = issue).
- `dividend`: same shape as 1830 (`kind` payout/withhold — 1867 also has
  half-pay: check Ruby for the third kind).

## Gotchas

- **Minor syms are letters, not numbers** (e.g. `NO`, `CS`) — and `CS`
  collides with 1830's Champlain & St. Lawrence private sym. Ability maps
  are already title-keyed (Phase 0.5); keep every other sym lookup
  title-scoped too.
- `auto_actions` appear on `buy_shares`/`dividend`/`remove_token` — the
  shared `filter_actions` already flattens them.
- Client-side types to filter (already handled by `filter_actions`):
  `message`, `undo`, `redo`, `program_*`; plus `end_game` (145×) and `log`
  (the consent audit line; 13 scrape_2026_10 games), which is stripped only
  after the undo/redo pass because Ruby lets a bare `undo` target it.
