# The site on engine-rs, step 3a: Rust decides 1830 and 1867 moves

Status: **design 2026-11-27; package A built the same day** (§15: 00e8bf6, 3bb8916, e66ba2c and after) (branch `site/engine-rs-step3`, RL_18xx worktree `.claude/worktrees/site-engine-step3`,
on `site/engine-rs-1867` b4b2391; first commit **9fc1b38**, the 1822 neutrality pin, §9.1). Approved by Caleb ("Just
move to the Rust engine", "yes yes yes" to step 3): his site (18xx.revys.net, `rl18xx-next`) moves from the Ruby/Opal
engine to engine-rs, keeping only the titles engine-rs implements (1830, 1867). Steps 1 and 2 (docs/SITE_ENGINE_RS.md,
docs/SITE_ENGINE_RS_1867.md) put engine-rs in the browser (opt-in) and beside Ruby on the server (the shadow, dev and
production), Ruby authoritative.

**Step 3a** (this doc): engine-rs becomes **authoritative** for 1830 and 1867 on the server, Ruby kept running beside it
as the **check** (the shadow reversed) and as the **rollback**; Rust the default engine in the browser, Opal the
fallback; on the **dev site** (:9301), ready for a production switch the parent makes. **Step 3b** (later, not here):
removing Ruby/Opal and the other titles.

Binding user decisions this step follows: 1867's phase-8 train limit is the rulebook's (§8.11.2: a company at its limit
may not buy; at the limit only trade-ins); voluntary bankruptcy stays for human players (X3, the site menu); Ruby's 1867
defects (the stale operator after a MajorTrainless nationalization with majors queued; the stale graph cache that skips
a legal station step) are **not** fixed in Ruby: Rust in charge fixes them, and the Ruby check classifies the
disagreements (§4.4); own companies only in every menu (1867 majors merge only their president's minors).

Read with: SITE_ENGINE_RS.md (the site layer, WasmGame API 2, `site_rules` S0-S6 / H1-H5 / U1, the shadow key, the
`site_engine` CLI, contracts K1-K11), SITE_ENGINE_RS_1867.md (1867's site layer, key v2, §5.3's edits, §13 the record),
rl18xx-next `docs/rl18xx-next/ENGINE_RS.md` (B's notes, "Open points, step 3"), living-board `docs/RUST_ENGINE.md`
(§6, §9.4: the client's gaps). Section numbers here are cited from code (`engine-rs/src/neutrality.rs` cites §9.1).

---

## 0. In one page

| Question | Decision |
|---|---|
| (1) How the server runs engine-rs as the authority | **The `site_engine` CLI, one process per move** (a new command `move`: the stored game + the posted move in, the verdict, the action as stored, the new state and the key out). Measured on the most loaded cores (§2.1): p50 4-45 ms per move, p95 12-100 ms, the longest real game 99 ms, against **143-304 ms** that Ruby's whole-game load costs every move today. Crash-isolated, already in both images, no service to run. A resident `site_engine serve` (same handlers, an LRU of loaded games: 0.03 ms 1830 / 1-2 ms 1867 per move, measured) is specified but built only if production's p95 goes over 100 ms (§3.6). Switch per title (`ENGINE_AUTHORITY_TITLES`, plus an admin runtime switch and a per-game override); rollback = the title back to Ruby at once (§3.5). |
| (2) What Ruby does in 3a | **The check**: the queue replays every Rust-stored move in Ruby (as `record_action` did: load, then the move, Ruby's own auto actions) and compares Ruby's verdict, auto actions and key v2 with Rust's; disagreements are rows in `engine_shadow` (direction `ruby_checks_rust`), classified by signature into the expected classes (R1-R9, §4.4) or `unclassified`. Ruby also still serves `POST /api/bot/engine` (18xx.games games), every other title, and the rollback. |
| (3) Programs | **Ruby's Programmer ported to the site layer** (§5.3, `src/site/programs.rs`: `program_share_pass`, `program_buy_shares`, `program_merger_pass`, `program_disable` with Ruby's reasons and shenanigan checks, 1867's step passes as today), run in Ruby's loop (`round.auto_actions` until none) after every move, chat included. Posted auto actions are ignored for authority titles: the server stores Rust's and returns them. |
| (4) Run pricing | **engine-rs checks and prices every posted run** (a new route check, `src/site/route_check.rs`, Ruby's `Route#revenue` checks; the replay pricers that already price every stored run as stored). Any valid run is accepted, not only the auto-router's: **20% of stored runs are not the router's run** (2,867 of 14,410: 96% equal revenue, 4% lower, none higher; §2.3). So runs drawn by hand run on Rust too. |
| (5) Stored games Ruby accepted that Rust refuses | Counted on every 1830 / 1867 game of :9301 (1,097) and production (31): **every game's current action list replays to its end in Rust**; Rust stops at **5 positions on :9301, all in undone branches** (111@9, 220@70, 415@14, 643@17, 1120@4), **none in production**; every stored move re-checked live at its position: 0 refused (runs aside). Decision: **a replay-only allowance, no repair** — the replay trusts the record exactly as Ruby's `load` does (player-time checks only for new moves); the 5 undone positions only matter to the history view, which says so (§2.5). |
| (6) Games engine-rs can't run | 1867 for 2 players and `grid_market` (none exist on either site), any 1830 optional rule engine-rs lacks (Ruby's 1830 has only the two engine-rs has). New tables of those are refused by the lobby with a sentence; an existing one would stay Ruby-authoritative per game in 3a. Runs drawn by hand (`auto_routing: false`: 7 :9301 games, none in production) are **supported** by (4). |
| (7) The server's other uses of Ruby | Each listed with its 3a replacement (§8): game start, acting players, notifications, results and the end, undo/redo and master mode, run pricing, the bot API's views (from Rust, **byte-compatible**, checked position by position), `POST /api/bot/engine` (**unchanged, Ruby**, in 3a: it models 18xx.games' engine, not this site's; its 3b fate is a decision for the training session, §8.2). |
| (8) The client | Rust the default for 1830 and 1867 (`RUST_ENGINE_DEFAULT=rust`), Opal the fallback for a Rust **failure** only; a refused move re-reads the game on Rust; Opal pages take the stored action (auto actions) the server returns. The listed gaps closed (§10): liquidity, last-run panel, merger candidates' order, an unlimited tile's count, wasm memory (checkpoints), dense replays, programs and hand-drawn runs on Rust pages. |
| (9) The corporation-graph speed-up | **The same walk on flat arrays** (exact by construction: same control flow, same `tile.paths` order, every membership test the same answer; integer indices and bitsets instead of `String`-keyed hash sets allocated on every step). Two semantic shortcuts were probed and **rejected** because they are not exact (§11.2). Functions that change: `src/map.rs` `compute_corp_graph`, `walk_from_node`, `walk_into_hex`, `is_converging_exit`, `is_city_blocked`, nothing else (§11.3). Pinned first: the 1822 neutrality walks (**9fc1b38**); all three pins unchanged at every commit. |
| (10) Verification, production | §12: the stored-move authority check over every corpus, generated games replayed by Ruby, mutated runs judged by Ruby, the bot views byte for byte, a scratch stack, then :9301 with the Ruby check for 7 days, every non-equal row classified. Production runbook one setting at a time, each with its rollback (§12.4). |
| (11) Packages | **A** engine (this worktree), **B** server (`~/workspace/18xx-ws/rust-authority`, `next/rust-authority`), **C** client (living-board worktree, branch `rust-authority`); contracts K1'-K15 (§13). |

---

## 1. Step 3a at a glance

```
 browser (living board)                         server (rl18xx-next, rack)                    queue process
 ──────────────────────                         ──────────────────────────                     ─────────────
 Rust (default for 1830, 1867) ── act/check ┐   POST /api/game/<id>/action
   programs and hand-drawn runs in wasm     │     authority title? (title switch, per-game override)
 Opal (fallback on a Rust failure only)     │       yes: site_engine move  ── verdict, stored action
   its auto actions ignored by the server   │            (check, route check + price, Programmer)
                                            │            store; acting/round/result from Rust
                                            │            Rust's key → engine_shadow row (pending) ──►  RubyCheck.tick
                                            │       no:  Ruby record_action (as today) + the old shadow  Engine::Game.load
                                            │     response: game + stored action (+ repriced)              process the move
 GET /api/bot/game/<id> view ◄── site_engine view (byte-compatible with Ruby's bot_view)                  compare verdict,
 POST /api/bot/engine ◄── Ruby (unchanged: 18xx.games games)                                               autos, key v2
                                                                                                           classify R1-R9
```

What does **not** change in 3a: the database schema of games and actions (an action stored by Rust looks exactly as
Ruby would have stored it: ids, `user`, `created_at`, `auto_actions`, run revenues and strings); undo/redo resolution
(Ruby's `filtered_actions`, already in the site layer); every other title (Ruby, Opal); `POST /api/bot/engine`; the bot
API's shapes; the client's hands (both engines word menus from the same facts); production until the parent switches it.

---

## 2. Measurements (2026-11-27; the side-work cores 24-31,56-63, which other runs kept at ~100%: every number is pessimistic)

### 2.1 Per move: Ruby today, Rust cold (the CLI), Rust warm (a held game)

Ruby: what `record_action` spends now on every move (`Engine::Game.load` of the whole game; `ENGINE_LOG_LEVEL=fatal` as
production, a throwaway `18xx_rack:dev` container); min of 3. Rust cold: `site_engine state` wall time including spawn,
JSON parse, replay and key (the b4b2391 binary :9301 runs; min of 3). Rust warm: a scratch harness holding the replayed
game, then `act` of the recorded move on a copy at every position (min of 3 runs).

| Game (raw actions) | Ruby load / move | Rust cold (CLI) | Rust warm `act`: p50 / p99 / max |
|---|---|---|---|
| :9301 1263, 1830 (632) | 178 ms | 6 ms | 0.01 / 0.04 / 0.05 ms |
| :9301 1119, 1830 (3,658) | 286 ms | 15 ms | 0.03 / 0.24 / 3.2 ms |
| production 12, 1830 (1,116) | 267 ms | 21 ms | 0.02 / 0.27 / 0.5 ms |
| production 33, 1867 (1,298) | 143 ms | 51 ms | 1.41 / 8.9 / 16 ms |
| :9301 1269, 1867 (1,083) | 304 ms | 49 ms | 1.08 / 5.6 / 10 ms |
| generated gen-2053, 1867 (967, the dense map) | 282 ms | **9.2 s** | 1.08 / 915 / 2,584 ms |

Rust cold across the corpora (`site_engine state` wall, min of 3; spawn alone 0.4 ms):

| Set | Games | p50 | p95 | max |
|---|---|---|---|---|
| :9301 1830 (a random 80) | 80 | 3.7 ms | 11.8 ms | 67 ms (881 actions) |
| production 1830 | 19 | 10.7 ms | 25.1 ms | 25.5 ms (1,094 actions) |
| :9301 1867 | 14 | 11.8 ms | 96 ms | 99 ms (765 actions) |
| production 1867 | 12 | 45.1 ms | 80.6 ms | 99 ms (778 actions) |

1867's warm cost is the legal-move generator checking the move (§5.2 of the 1867 doc: p50 0.4 ms there). Memory of a
held game: 1.3 MB (the 100 largest 1830 games: 133 MB), 1.2 MB (1867), 6.9 MB (gen-2053). The dense game's seconds are
the corporation graph (§11); Ruby plays it at its usual speed.

### 2.2 The corpora

| Site | 1830 | 1867 | Other titles | Notes |
|---|---|---|---|---|
| :9301 (ids to 1273) | 1,083 games (158,701 actions; 67 two-player; 210 `multiple_brown_from_ipo`; 7 `auto_routing: false`; 41 with programs) | 14 (5,021 actions; all 4 players; 9 with step auto passes) | 5 not new (960 1889, 961 1846, 962 18Chesapeake, 964 18EU active, 1001 1822CA WRS) + 23 new | read with `GET /api/game/<id>` |
| production (ids 5-35) | 19 (12,356 actions; 10 with programs) | 12 (8,126 actions; 10 with step passes, programs in 6) | **none** | the bot API's read-only export, files 600 in the design's scratch, deleted after |

No game on either site is a 2-player 1867, uses `grid_market`, or uses an 1830 optional rule engine-rs lacks.

### 2.3 Stored moves Ruby accepted that Rust refuses (scratch probe `tips`, not committed)

1. **Replay** (as the server loads a game: the record trusted, Ruby's `load`): every "tip" of every game (the position
   just before each undo, and the end) replayed. **All 1,128 games replay to their current end.** Rust stops at 5
   positions, all on :9301 and all in undone branches: 111@9 (`par` of PRR while B&O's par is pending), 220@70
   (`buy_company` at a price of 1.5), 415@14 (NYC's IPO share before NYC was parred) — step 1's three — and 643@17,
   1120@4 (bids on another private during an auction, stored before the site's auction lock; today's site Ruby refuses
   them too). Production: none.
2. **Live** (each kept engine move of each game acted at its position with today's `act`: 1867's generator check,
   1830's processing): **0 refused** (147,004 + 11,509 + 4,974 + 6,526 accepted). The only other outcome: runs that are
   not the auto-router's run, which today's `act` answers `unsupported`:

| Set | Runs | Not the router's run | Same revenue | Lower | Higher |
|---|---|---|---|---|---|
| :9301 1830 | 11,796 | 2,367 | 2,255 | 112 | 0 |
| production 1830 | 1,177 | 239 | 239 | 0 | 0 |
| :9301 1867 | 515 | 45 | 45 | 0 | 0 |
| production 1867 | 922 | 216 | 215 | 1 | 0 |

(39 of them are the router's routes on other trains.) Hence §6's route check: the authority must accept any valid run.
1830's live check in 3a is stricter than today's processing (§5.2); A1 re-runs this probe with it before the switch.

### 2.4 Stored auto actions (what the Programmer must reproduce)

| Set | Actions carrying auto actions | Kinds |
|---|---|---|
| :9301 1830 | 124 in 40 games | `pass` 78, `program_disable` 23, `buy_shares` 7, `buy_shares`+`program_disable` 7, `buy_shares`+`pass` 7, `pass`+`program_disable` 2 |
| production 1830 | 104 in 10 games | `pass` 53, `buy_shares`+`pass` 20, `program_disable` 16, `buy_shares`+`program_disable` 6, `buy_shares` 4, `pass`+`program_disable` 4, one with all four |
| :9301 1867 | 260 in 9 games | `pass` (step passes) |
| production 1867 | 566 in 10 games | `pass` (step passes and `program_share_pass`'s) |

### 2.5 The decision on stored games: a replay-only allowance, no repair

- engine-rs's replay already trusts the record the way Ruby's `load` does (the player-time checks are skipped: steps
  1-2), and 3a's checks (§5, §6, the Programmer) apply only to a move being made. That is the whole allowance: nothing
  is tolerated beyond what Ruby's own `load` tolerates, and no stored action is rewritten (a repair would change games'
  histories, their analyses and their exports).
- The 5 positions Rust refuses are in undone branches: no redo can reach them (later moves cleared the redo), so no
  game's play is affected. Only the history view at exactly those actions (`/game/111?action=9` and the like) replays
  them: the Rust page stops there with the engine's words and shows the position before; while Opal exists (3a) the page
  offers it for that view.
- Re-counted before each switch (A's `site_replay --authority` and the probe of §2.3 on a fresh export); a production
  game Rust can't replay would stay Ruby-authoritative per game (§3.4) until decided.

---

## 3. (1) engine-rs as the authority on the server

### 3.1 The transport: the CLI per move

| Option | Latency per move (measured, §2.1) | Robustness | Work | Verdict |
|---|---|---|---|---|
| Ruby today (the baseline) | 143-304 ms (a whole-game load every move) | — | — | replaced |
| **`site_engine move`, one process per move** | p50 4-45 ms, p95 12-100 ms, max 99 ms on real games | a crash, panic or hang is one failed move (killed at the timeout); stateless: the database is the only state | a command in `src/site/cli.rs`; `EngineRs.run` exists | **chosen** |
| `site_engine serve`, a resident process with an LRU of loaded games | 0.03 ms (1830), 1-2 ms p50 / 9 ms p99 (1867) after the first load | a cache that must never serve a stale game (fingerprints), a process to supervise per rack worker, memory (1.3 MB a game) | ~300 lines Rust + Ruby | specified (§3.6), built only on the trigger |
| magnus / rb-sys in process | µs | an abort or overflow takes the worker and the move; a Rust toolchain in the gems stage | as step 1 found | no |

Every real game's move is within the ~100 ms budget on the CLI, on the busiest cores, and 3-30x faster than Ruby
today; the dense-map tail is the graph's (§11), which a resident process does not cure (gen-2053's warm act is 2.6 s).

### 3.2 `site_engine move` (contract K3')

```
site_engine move {"game": <site game JSON, every raw action, chat included>, "action": <as posted>, "user": <user id>,
                  "opts": {"ignore_posted_autos": true}}
  -> {"verdict": "accepted" | "refused",
      "error": null | "<words>",                     // refused: the engine's words (Ruby's where step 1-2 matched them)
      "action": <the action as stored: id = last + 1, user, auto_actions (Rust's), a run's revenues and strings> | null,
      "posted_autos": [...] | null,                  // what the page posted, when it differs from Rust's (logged)
      "repriced": [{path, posted, stored}],          // a posted revenue Rust priced otherwise (lib/run_revenue.rb's shape)
      "actors": [<player ids who may make this move>],  // Ruby valid_actors: master mode (for_others)
      "state": {"acting": [...], "round": "...", "turn": n, "round_num": n, "finished": bool, "result": {...},
                "manually_ended": bool, "game_end_reason": null | "...", "last_game_action_id": n},
      "key": <shadow key v2 after the move>, "ms": n}
site_engine start {"game": ...}  -> {"state": ..., "key": ...}          (the opening position: start_game!)
site_engine view  {"game": ..., "upto": <id>?}  -> {"view": <the bot view facts, §8.1>, "key": ...}
```

`view` as built: `{finished, error, entity: {id, type, player}, step, actions, acting, trains: [[sym, [{id, name}]],
...], depot: [{id, name}]}` -- bot_view's values but `last_id` and the `status` half of `finished` (the server's);
`trains` as `[sym, list]` pairs in Ruby's `engine.corporations` order (B rebuilds the Ruby hash in that order);
`depot` is Ruby's `depot.depot_trains`. `acting` of a finished game is Ruby's `active_players_id` at its best guess
(a manual end: the players to act when it ended; otherwise every player not bankrupt): B's `bot_view_compare.rb`
says whether it matters.

**As built (A, 00e8bf6)**: the input is `{"game", "action", "user", "now"}` (`now`: the server's `Time.now.to_i`; the
posted `created_at`, if any, wins as in Ruby's `from_h`; posted auto actions are always ignored, `opts` is not read);
`error` carries the engine's words without the `refused: ` prefix; `state` also has `stopped` (null, or the replay's
stop after an undo / redo); `posted_autos` is set only when the posted list differs from Rust's. The stored action is
Ruby's `to_h`: `user` only when the action's entity has a player and the user isn't that player, the type's own
arguments (Ruby's `args_to_h` keys; nils dropped; a bundle's `percent` filled in), Rust's auto actions each with
`created_at` = `now`, no id, no user. A stored game whose replay stops is exit 3 (`unsupported: the replay stopped at
action N: ...`: B keeps such a game on Ruby, §3.4); exit **4** is an `internal:` error the engine caught itself (e.g.
the stored action failing to replay as decided); the server treats 2 / 3 / 4 / 101 / a timeout alike (503).

Exit codes as K3 (0 a verdict, refusals included; 2 bad input; 3 unsupported; 4 internal; 101 a panic). `created_at` stays the
server's (Ruby's `Action.create`). An undo / redo is accepted as Ruby accepts it (no check: Ruby's `process_action`
returns at once for them), the state is the reloaded game's. `end_game` (manual end), `message` and `log` as the site
layer already handles them; `program_*` actions as Ruby's `Step::Program` (offered per `available_programmed_actions`; its one check: a
`program_buy_shares` needs its `until_condition`).

### 3.3 The request (B: `routes/game.rb`, a new `lib/engine_authority.rb`)

`record_action(game, user, users, params, ...)` branches once, on `EngineAuthority.rust?(game)`:

1. Rust (an authority title, a supported game, no per-game `ruby` override): build the game JSON as the shadow worker
   does (`EngineShadow.game_json`: players in order, settings without private notes, every raw action), run
   `site_engine move` under the game's advisory lock (already held), then:
   - `refused` → `halt(500, "Illegal action: <words>")` (today's status and shape: bots and pages read it unchanged);
   - `accepted` → `Action.create` with `action` (as today), `for_others` from `actors`, `set_game_state` from `state`
     (round, turn, acting, result, finished_at, manually_ended, status, finished_action_id: the same columns as
     today), `repriced` from Rust's list (the response's `repriced: true`, a `revenue_mismatch` row as today), a pending
     `engine_shadow` row with Rust's key for the Ruby check (§4), and `announce_action` unchanged;
   - exit 2 / 101 / timeout / no binary → `halt(503, "The rules engine is unavailable; your move was not stored")`, an
     `engine_shadow` row (`source: authority`, `status: error`), a log line. No silent Ruby fallback: one authority per
     game. The client retries a 503 once (as today) and then says so.
2. Ruby (any other title or game): today's code, unchanged, with today's shadow where it applies.

`start_game!` the same with `site_engine start`. Timeout `ENGINE_RS_MOVE_TIMEOUT` (default 5 s, separate from the
shadow's 10 s). Concurrency: the per-game advisory lock serialises a game's moves as now; different games run their
CLI processes in parallel (each one core for its milliseconds); the rack's worker count bounds them.

### 3.4 Which games Rust decides

`EngineAuthority.rust?(game)`, read on every request (no restart needed to change it):
- the title is in `ENGINE_AUTHORITY_TITLES` (env, rack and queue; default empty: Ruby everywhere) and not switched off
  at run time (`POST /api/admin/engine_authority {title, engine: "ruby"|"rust"}`, redis `engine_authority:<title>`;
  redis trouble keeps the env's value);
- the game is one engine-rs runs (`SiteGame::load`'s rules, mirrored in Ruby: 1867 with 3-6 players, optional rules
  within engine-rs's; §7), and is not pinned (`settings.pin`);
- the game has no per-game override `settings.engine_authority = "ruby"` (the rollback's tool, §3.5; `"rust"` forces
  Rust for a game when its title is back on Ruby).

A game's history may mix moves stored under both authorities: both engines load both (steps 1-2's shadow, this
step's check).

### 3.5 Rollback to Ruby-authoritative

- **One title, at once**: the admin switch (`engine: "ruby"`), or drop it from `ENGINE_AUTHORITY_TITLES` and recreate
  rack and queue. The next move of every game of the title is Ruby's.
- **Games Ruby cannot follow** (moves Rust stored that Ruby refuses: the check's `ruby_refused` / `ruby_stuck` rows,
  expected only in R1-R4's 1867 cases, §4.4): listed at any time by `GET /api/admin/engine_shadow?direction=
  ruby_checks_rust&status=ruby_refused,ruby_stuck`; the runbook sets `settings.engine_authority = "rust"` on each, so
  they keep running on Rust while the rest of the title goes back (Ruby would refuse their every move). If Rust itself
  is the problem for such a game, it waits for the fix (it can't be played on either engine: Ruby's own defect).
- **The whole step**: the previous image (production) or the previous `rl18xx-next` head (dev), as DEPLOY.md does.

### 3.6 `site_engine serve` (specified; built only on the trigger)

Trigger: production's `engine_ms` (logged per move, §3.3) has p95 > 100 ms over 7 days, or a title with longer games
(the 1822 family). Then: `site_engine serve` reads JSON lines on stdin and answers on stdout; one child per rack worker
process (spawned lazily after fork, killed and respawned on timeout or a bad answer, the CLI as the fallback); an LRU
(64 games or 128 MB) keyed by game id, each entry carrying the fingerprint it was loaded at (action count, last action
id, FNV of the last raw action's JSON); a request names the database's fingerprint and sends the full game only when
the child asks for it (a miss or a mismatch). The handlers are `move` / `start` / `view` themselves (one code path).

---

## 4. (2) Ruby in 3a: the check (the shadow reversed)

### 4.1 Flow

1. In the request, after Rust's move is stored: a row `engine_shadow {source: action|bot|start, direction:
   "ruby_checks_rust", status: pending, rust: <Rust's key v2>, action_id}` (one INSERT, as today's shadow).
2. The queue's `RubyCheck.tick` (every 3 s, oldest first, a time budget per tick): for each row, the game from the
   database up to the action before it; `Engine::Game.load(...)` (Ruby's `load`: the record trusted), then
   `process_action(<the stored action without its auto_actions>, add_auto_actions: true)` exactly as `record_action`
   ran Ruby before 3a (player-time checks on, Ruby's own Programmer and step auto actions); then
   `EngineShadow.capture` (key v2). Cost: one Ruby load, 150-300 ms on a long game (§2.1); dev's and production's move
   rates are far below one worker's capacity.
3. The verdict, field by field: Ruby's acceptance (`ruby_refused` with Ruby's words if Ruby raises); Ruby's auto actions
   against the stored ones (`autos_differ`, both lists); the keys (`differs`, up to 20 paths, step 1's order of
   importance). Each row is checked from the stored history (Rust's), so a disagreement does not carry into the next
   row unless its cause does (then `known`, step 1's rule).
4. Classification (§4.4): a non-equal row whose signature matches an expected class gets `class: R<n>`; else
   `unclassified` (logged, and on the admin page's top).

The forward shadow (Rust checks Ruby's moves) keeps running for games Ruby still decides.

### 4.2 Statuses (contract K13)

`pending`, `equal`, `ruby_refused` (Ruby raises on a move Rust stored), `ruby_stuck` (Ruby already refused an earlier
move of the game: every later row, reported once), `autos_differ`, `differs`, `known`, `error` (Ruby itself failed:
an exception outside the move, a timeout), `skipped` (a key version or pinned game). Columns as today plus `direction`
(`rust_checks_ruby` for the old rows, the default) and `class`.

### 4.3 What the check proves, and what it can't

It proves, move by move, that Rust stored what Ruby would have, or names the difference; and it keeps the rollback
honest (§3.5: the games Ruby can't follow are known at every moment). It can't see a move Rust **refused** that Ruby
would have taken (nothing is stored): those come from the client's `engine_report` rows (`refused`, Rust's words) and
from A's verification (§12.1), which judges Rust's refusals by Ruby on every corpus and mutation set.

### 4.4 Expected classes (from the binding decisions and the steps 1-2 records)

| Class | What | Signature the classifier matches |
|---|---|---|
| **R1** Ruby's stale operator (1867) | After a MajorTrainless `choose nationalize` with other majors queued, Ruby's round keeps the reset major as operator and refuses every later move | `ruby_refused`, the game's operating round has a `choose` / `nationalize` before it with ≥ 2 majors in MajorTrainless's queue at the time; later rows `ruby_stuck` R1 |
| **R2** Ruby's stale graph cache (1867) | Ruby's `mergeable_candidates` / `can_really_place_token?` read an old graph: Ruby auto-passes a minor (Merge) or skips a station step (Token) the rules allow | `autos_differ` where Ruby's list has an extra `pass` of a Merge or Token step entity; or `ruby_refused` of that entity's merge / station right after such a row |
| **R3** the opening auction nobody can bid (1867) | Ruby raises (`undefined method 'cash' for nil`); engine-rs gives the lot to the first player at $0 | `ruby_refused` / `error` in `SingleItemAuction` with that message |
| **R4** MajorTrainless' phantom (1867) | after `event_minors_nationalized!` resets a major, Ruby's MajorTrainless keeps it | `differs` on `trainless`, then `ruby_refused` |
| **R5** orphan trains (1867) | a major nationalized with trains: Ruby keeps them on the old object (`Depot#other_trains`), engine-rs drops them | `differs` on `depot` / a reset major's `trains` |
| **R6** 5+5E on Toronto / Montreal (1867) | the replay pricer counts a stop as tokened per hex where Ruby counts per city | `differs` on `last_action.revenues` only, a 5+5E route through Toronto (F16) or Montreal (L12) |
| **R7** the train limit (1867, §8.11.2) | Rust refuses a depot train for a company at its limit; Ruby took it | never a check row (nothing stored): an Opal page's `engine_report refused` whose words name the limit |
| **R8** program reasons' words | Ruby's `program_disable` reason names a player differently | `autos_differ` where only `program_disable.reason` differs |
| **R9** Ruby-accepted probes Rust refuses (1830's 111@9 kind) | Rust stricter than Ruby on malformed moves | `engine_report refused` only (nothing stored) |

Everything else is `unclassified` and is a bug of Rust's until shown otherwise.

---

## 5. Rust's live checks (what a posted move must pass)

### 5.1 1867 (unchanged from step 2)

The legal-move generator under Ruby's set (`RULES`: the engine accepts what Ruby accepts; only the menus are own-only),
the move matched on the importer's outcome key, a ranged price on its range and step (SITE_ENGINE_RS_1867.md §3.2);
runs by the route check (§6). The train limit is the rulebook's (the binding decision): no depot train at the limit,
trade-ins only (already engine-rs's and the Rust menu's since 3fdd015).

### 5.2 1830 (new: today's `act` only processes)

engine-rs's 1830 processing trusts what it is given the way Ruby's `load` does: a lay's upgrade path and rotation, a
station's connection, a blocked hex are never checked (Ruby checks them only while not loading: `Tracker#lay_tile`'s
`legal_tile_rotation?`, `check_track_restrictions!`, `Token#process_place_token`'s `available_hex`,
`Tokener#check_connected`, an ability's blocking hex). As the authority, a posted 1830 move must match a move of **the
site menu of any acting seat** (the RawMenu's complete actions, a bid's or a priced option's range, the tile options of
every hex the menu lists, `tileOptions(hex)`'s tiles and rotations, the station options) **or of Ruby's cross-player set**
(another president's company's train at $1..cash: `cross_player_train_buys`; a private from another player in a stock
round while `site_interplayer_purchase_open`: S1-S6), matched on an outcome key like 1867's (type, entity, company /
corporation, the bundle's corporation, holder, percent and president's certificate, hex, tile, rotation, city, train by
id, price on its range). The menu is the right model: steps 1-2 checked it against Opal's hand on 322,145 / 322,150
menus and every option it offers against Ruby (7,619 cases equal). The words of a refusal name the rule
("Not a legal lay: ...", as 1867's). Ruby's cross-player deals stay accepted although no menu offers them: 3a does not
change what the site accepts beyond the binding decisions.

A1's first measurement: §2.3's live probe re-run with this check over every corpus; the target is 0 refusals of stored
moves other than R9's kind, each one listed.

**As built (`src/site/check1830.rs`)**: the first run refused 110 of :9301's 149,335 stored moves, every one a move
Ruby takes that no menu offers: a sale whose bundle isn't the menu's (the president's certificate alone, or a partial
president's bundle: Ruby's `SellShares.from_h` builds `ShareBundle.new(shares, percent)` from whatever was posted and
asks `can_sell?` of it), a company buying another player's private in an operating round (`purchasable_companies` is
any player's), another president's company's train, M&H's exchange on another player's turn (a company's step never
blocks: `Round#active_step(company)`), a crowded company's discard while another is first. The check now accepts those
as Ruby does: a sale is any distinct set of the seller's shares of one corporation whose percent is their sum (or less
by whole shares with the president's certificate among them) for a corporation and percent the menu sells, with
`can_dump?` when the certificate is in it; the cross-player set has the four others. Then: **0 refused** of 149,335
(:9301 1830) and 11,747 (production 1830); 1867 (generator, unchanged) 5,019 and 6,742.

### 5.3 (3) Programs and auto actions (`src/site/programs.rs`)

After every move (chat included, as Ruby's `process_action`), Ruby's loop: `until (actions = round.auto_actions).empty?`
process them and append them to the move's `auto_actions`. `round.auto_actions` is the active step's
`auto_actions(current_entity)`; the site layer ports each that 1830 and 1867 have:

| Ruby | What it does | Rust (site layer) |
|---|---|---|
| `Step::Programmer#programmed_auto_actions` | each of the player's programs' `activate_<type>` (one program a player: `ALLOW_MULTIPLE_PROGRAMS` false) | the programs the site layer already records (`note_program`, dropped per round by `check_programs`: step 1), each with the id of the action that enabled it |
| `BuySellParShares#activate_program_share_pass` | `pass` when `pass` is offered and the pass is normal (`normal_pass?`: not inside 1867's minor auction), unless `should_stop_applying_program` (unconditional programs skip it): then `program_disable` with the reason | the stock step's offers (the menu's facts); the reasons below |
| `BuySellParShares#activate_program_buy_shares` | stop reasons (floated / N shares: `program_disable`, then `program_share_pass` if `auto_pass_after`); `buy_shares` of one share from the IPO or the market (`from_market`) the player `can_buy?`, refusing mixed share sizes; after a buy, `pass` unless the company just floated with `until_condition: float` | the menu's buy facts (the menus equal Opal's hand on 322,145 of 322,150 in step 1) |
| `should_stop_applying_program` | `must_sell?` (over the certificate limit or a 60% holding), then the round's other players' actions since the program was enabled, each through `action_is_shenanigan?` (a par, a buy that threatens a presidency, a sale, a redemption, ...) | a per-round log of each player's stock actions kept by the site layer (Ruby's `@round.players_history`), Ruby's reason strings word for word (they are stored) |
| `ProgrammerMergerPass#activate_program_merger_pass` (1867) | with `disable_others` and another player having acted (`others_acted?`: a conversion this round): `program_disable`; else `pass` for `merger_auto_pass_entity` when the program names it for this round (`corporations_by_round`) | the merger state the site layer keeps (step 2's `MergerRecord`) |
| 1867's step auto actions (Token, Merge, BuyCompanyPreloan passes) | no program needed | step 2's `autos::predict`, unchanged |

1830's `WaterfallAuction` includes `ProgrammerAuctionBid`, but 1830 offers only `program_buy_shares` and
`program_share_pass` (`available_programmed_actions`), 1867 those and `program_merger_pass`: no other program is
ported. The wasm runs the same code, so Rust pages in games with programs act at once and hotseat games keep their
programs on Rust (§10). Judged on every stored action carrying auto actions (§2.4: 228 in 1830, 826 in 1867), on
generated games with programs replayed by Ruby, and live by the Ruby check (R8 is the only expected class).

**As built (`src/site/programs.rs`)**: `SiteGame` keeps Ruby's `@round.players_history` (`stock_history`: a player's
key when their turn starts or an action of theirs is tracked, cleared at each of their turns, every entity the round
visits; 1867's internal `Par` of a minor won at auction tracked with no id and the move's time) and each program's
`(id, created_at)` (`program_meta`: Ruby's `action < program` compares ids, else times). Every stored action with auto
actions in the corpora is reproduced field for field (created_at aside): :9301 1830 123 / 123, production 1830 101 /
101, :9301 1867 260 / 260, production 1867 469 / 469 (the 1867 counts are the kept actions; the rest of §2.4's are in
undone branches).

---

## 6. (4) Runs: checked and priced by engine-rs

`src/site/route_check.rs` (both titles): a posted run's routes are built from their `connections` and `nodes` on the
position (as Ruby's `Route` builds `connection_data` from the hex chains), then Ruby's `Route#revenue` checks in Ruby's
order: at least 2 stops (unless a local train), a token of the company on the route (`check_route_token`), no group
twice, `check_terminals!`, `check_other!` (1867: none beyond the base), `check_cycles!`, `check_double_towns!`,
`check_distance!` (1830's n-trains and D; 1867's +, 2+2's x2, 5+5E's express: G1867 `compute_stops`), `check_overlap!`
(no track twice across the run), `check_connected!`; then priced by the replay pricers that already price every stored
run as stored (`shadow::price_routes` 1830, `g1867::price_routes`: 11,643 + 1,151 + 1,273 runs, steps 1-2). Accepted:
the run with Rust's revenues and Ruby's run strings (`revenue_str`: Ruby's format, c57803b); a posted revenue that
differs is `repriced`, as lib/run_revenue.rb does today. Refused: Ruby's words for the failed check.

The router's own run (`autoRoute`) passes the same check (a test). The wasm's `act` uses the same check, so a Rust page
may post a run drawn by hand: games with `auto_routing: false` open on Rust (C, §10).

`lib/run_revenue.rb` stays for the titles Ruby decides; for authority titles the server uses Rust's pricing
(`repriced`, the `revenue_mismatch` row) and never builds a Ruby engine.

**As built (`src/site/route_check.rs`)**: the routes are rebuilt from their `connections` with Ruby's own walk
(`Path#walk` pruned to the posted hex chain; `find_matching_chains` over the first and last hexes' nodes in
`tile.nodes` order; one chain: the first with a station of the company sharing no track with the other routes;
several: `find_pairwise_chain`, the posted `nodes` breaking ties) and, as `compute_other_paths` does while the run's
routes are being built, **last first**, each excluding the track of those after it. Every stored run of the corpora
checks and prices as stored: 1867 1,315 / 1,315 field for field (train, revenue, `revenue_str`, `hexes`, `nodes`);
1830 every run stored since the server prices runs (310 on :9301) field for field, and the older records differ only
where Ruby's own format changed since (routes stored before c57803b: old run strings, stale `hexes`, no `nodes`: 1,426;
between c57803b and lib/run_revenue.rb: the page's claimed `revenue_str`: 568): the revenue equal on every one of the
12,877 stored 1830 runs.
Not modelled (none in 1830's or 1867's tiles): lanes, local trains, intra-tile ambiguous connections (refused as
`unsupported`), double towns, the off-centre town rule of `check_overlap`.

---

## 7. (6) Games engine-rs can't run

| Kind | On the sites | 3a |
|---|---|---|
| 1867 for 2 players (Ruby's `setup_for_2_players`) | none | the lobby refuses a new 2-player 1867 table while 1867 is an authority title ("1867 is played by 3 to 6 players here"); an existing one would stay Ruby-authoritative (§3.4) |
| 1867 `grid_market` | none | the lobby refuses the rule ("1867's grid market isn't played here"); existing: as above |
| 1830 optional rules engine-rs lacks | none: Ruby's 1830 has `multiple_brown_from_ipo` and `optional_6_train`, both in engine-rs | — |
| Runs drawn by hand (`auto_routing: false`) | 7 :9301 1830 games | supported (§6) |
| Other titles | :9301 5 games (1889, 1846, 18Chesapeake, 18EU, 1822CA WRS), production none | Ruby, unchanged (3b decides) |

`Titles.table_problem` (B) mirrors `SiteGame::load`'s rules for authority titles, and `site_engine` answers exit 3 with
the reason if anything slips through.

---

## 8. (7) The server's uses of Ruby and their 3a replacement

| Use | Where | 3a (authority titles) |
|---|---|---|
| Deciding a move, storing it, its auto actions | `record_action` | `site_engine move` (§3) |
| The run's price | `RunRevenue.prepare` / `mismatches` | Rust's (§6); `repriced` unchanged |
| Master mode (`for_others`) | `moves_for_others?` (`valid_actors`) | `actors` (§3.2) |
| Who acts, round, turn; turn mail and notifications | `set_game_state`, `acting_users`, `publish_turn` | `state.acting` / `round` / `turn` (key v1's `acting`, equal to Ruby's at every corpus position) |
| Result, the end, `finished_action_id`, reopening by undo | `set_game_state` | `state.finished` / `result` / `manually_ended` (results equal on every finished corpus game: 144 + 14 + 13) |
| Undo / redo | `process_action` (no check), `filtered_actions` | the site layer's `filtered_actions` (step 1) |
| Game start (seats, the opening position, the first acting player) | `start_game!` | `site_engine start`; seats stay `lib/lobby.rb`'s |
| Player counts, optional rules, the title list | `Lobby.player_range`, `Titles` (Ruby's `Meta`) | unchanged (metadata, not the engine); `table_problem` adds §7's refusals |
| The bot API's `view` | `GET /api/bot/game/<id>` (`bot_view`) | built by the route from `site_engine view`'s facts, **byte-compatible** (§8.1) |
| The bot's move | `POST /api/bot/game/<id>/action` | `record_action` → Rust (with `add_auto_actions`: Rust's Programmer) |
| 18xx.games games for the bots | `POST /api/bot/engine` | **Ruby, unchanged** (§8.2) |
| The analysis worker's input | `AnalysisWorker` (the game from the database) | unchanged (no engine) |
| The shadow | `EngineShadow` | reversed for authority titles (§4) |
| Pinned games | `settings.pin` | unchanged (no engine) |
| `spec/lib/engine` | specs | unchanged in 3a |

### 8.1 The bot view, byte for byte (contract K12)

`bot_view` keeps its Ruby code path and key order; for authority titles its values come from Rust: `last_id`,
`finished`, `error` (a replay that stopped: its words), `entity` `{id, type, player}`, `step` (Ruby's class name),
`actions` (Ruby's `step.actions(entity)`), `acting`, `trains` (corporations holding trains, in Ruby's
`engine.corporations` order, `[{id, name}]`), `depot` (Ruby's `depot.depot_trains`: the available upcoming trains and
the discarded ones, `[{id, name}]`). Most are key v2 fields already equal to Ruby's at every corpus position; `trains`
and `depot` are new facts in `site_engine view`. Check (B): `scripts/engine_rs/bot_view_compare.rb`, Ruby's view and
Rust's at every position of every corpus game, byte-equal JSON; then live on the scratch stack with the dev bots.
The bots' backends (:5004 prod, :5014 dev) need no change. **No contract change is proposed.**

A's check (read-only, production's bot API, the files deleted after): `site_engine view` of each of production's 31
games (8 active, 23 finished) equals the `view` the bot API serves for it, field for field (entity, step, actions,
acting, the trains in Ruby's order, the depot; a finished game's `acting`: a bankruptcy's is the bankrupt president,
a round-end's every player not bankrupt). B's `bot_view_compare.rb` still checks every position.

### 8.2 `POST /api/bot/engine`

It answers for games of another site (18xx.games, 1830 only): who that site's engine waits for and how it stores a
move. 18xx.games runs Ruby HEAD, which already differs from this site's Ruby (the auction lock 1b5c534, the run strings
c57803b); engine-rs would differ more (e.g. its stricter checks). 3a leaves it on Ruby, byte for byte as today (B
re-runs step 1's 250-request comparison). **For the parent to agree with the training session before 3b**: keep a
minimal Ruby for this endpoint after 3b, or move it to engine-rs with its differences listed and accepted.

---

## 9. Training neutrality and the edits outside `src/site`

### 9.1 The 1822 pin (done first: 9fc1b38)

`engine-rs/src/neutrality.rs`, its third section: 16 fixed-seed 1822 walks through the generator layout as the PG
workers drive them (`rl_g1822_indexed_moves` / `rl_g1822_apply_index`), 3-6 players × {base, `plus_expansion`,
`plus_expansion` + `plus_expansion_no_removals` + `plus_expansion_single_stack` + `tax_haven_multiple`,
`tax_haven_multiple`}, each combination under RESTRICTED (the title's, self-play's) and RULES, busy and uniform picks,
seeded; hashing every position's legal moves (index, type, price range, pass), the index and price chosen, the generic
token encoder every 5th position, the action log and the result. 18,523 moves, 15 of 16 walks finished, ~40 s release.
Computed on b4b2391 before any change, and equal on a `git archive` of RL_18xx origin/master **995ccef**'s engine-rs
(with the 1830 and 1867 pins). Gate at 9fc1b38: `--features python` 272 passed / 0 failed / 11 ignored, `--features
site` 298 / 0 / 10, the wasm checks (`wasm,site-titles` and `wasm`) and the default `cargo check` clean. (Player ids
1001.. as the 1822 tools use: with ids 1..n the 1822 generator offers no move at the start; noted, not investigated.)

All three pins (1830, 1867, 1822) must pass unchanged at every commit of A.

### 9.2 The list (for the training session; the parent sends it before any of it lands)

| # | File, item | Edit | Status |
|---|---|---|---|
| N3 | `src/neutrality.rs` | the 1822 pin (tests only) | **9fc1b38** |
| G1 | `src/map.rs` (as built, function by function below) | the corporation-graph speed-up, speed only (§11.3) | its own commit (§15) |
| V3 | none identified | any `fn` → `pub(crate) fn` the Programmer, the route check or the 1830 move check turns out to need is listed here, function by function, before its commit | — |

G1 as built, every item of `src/map.rs` it touches:

| Item | Change |
|---|---|
| `compute_corp_graph` | body only (signature unchanged): calls `compute_corp_graph_flat`; under `cfg(test)` also the reference, asserting the two graphs equal (`graph_cross_check`, on unless `GRAPH_CROSS_CHECK=0`) |
| new `compute_corp_graph_flat` | the token walks on the flat state, `route_available` / `route_train_purchase` as before, the result converted back to today's `CorpGraph` |
| new `FlatCtx` (`new`, `n_nodes`, `node`, `node_id`), `FlatWalk` (`new`, `path_visited`, `set_path`, `clear_path`), `bit_get` / `bit_set` / `bit_clear`, `all_present`, `NO_NBR`, `HEX_PRESENT` | the flat context (neighbour per edge, converging-exit mask, node numbering, `is_city_blocked` per city node) and the walk's integer state |
| `walk_from_node`, `walk_into_hex` | index-based: the same control flow on the flat state (replace the String-keyed ones) |
| new `tokenable_cities_of` | the tokenable-city rule moved out of `compute_corp_graph` verbatim (same conditions, same sort) |
| new `compute_corp_graph_reference`, `walk_from_node_reference`, `walk_into_hex_reference` | the walk as it was, verbatim (renamed only): the fallback for a token on a hex the board doesn't index (never on a real board) and what the cross-check compares with. Not test-only (a deviation from §11.3's plan: the fallback needs it) |
| `is_converging_exit`, `is_city_blocked` | unchanged (moved above the flat walk, which precomputes them) |
| tests: `graph_cross_check`, `GRAPH_CROSS_CHECKS`, `assert_same_graph` (`cfg(test)`); `the_flat_walk_equals_the_reference_on_generated_1867_games`; `the_flat_walk_equals_the_reference_on_a_corpus` (ignored, `GRAPH_CORPUS`) | the differential checks |

Unchanged: `GraphCache`, `get_or_compute`, `CorpGraph`, `TokenableCity`, `NodeId`, `NodeType`, `corp_has_token_in_city`,
`TILE_RESERVATION*`, every caller, `g1867/graph.rs`, `g1867/fastwalk.rs`, `g1822/*`. No V3 (no visibility change): the
Programmer, the route check and the 1830 check use only `pub(crate)` items that already were.

Not edited: `action_index.rs`, the encoder, `decode`, `POLICY_SIZE` 26537, `src/rl`, the g1867 / g1822 layouts and
generators, `mcts.rs`, `mcts_lockstep.rs` (the training session's subagent is editing those). No `site_rules` branch:
the Programmer, the route check, the 1830 move check, `move` / `start` / `view` and the client gaps are all in
`src/site` (site tooling outside it: `src/bin/site_engine.rs`'s dispatch, `src/bin/site_replay.rs`'s new modes,
`scripts/site_engine/*`).

---

## 10. (8) The client (package C)

| Item | 3a |
|---|---|
| Default | `RUST_ENGINE_DEFAULT=rust` (dev first); the boot block's `engine.rust` gains `authority: [titles]` (K7'); Settings' "The site's engine" becomes "Opal (fallback)" for authority titles |
| Fallback | Opal on a Rust **failure** (wasm load, `internal:`, `unsupported:`, a replay that stops on a stored move): as today, reported. A move the server **refuses** no longer re-opens on Opal for authority titles (the site's engine is Rust): the page reads the game again and stays on Rust |
| Opal pages | the server ignores their posted auto actions (K8'); the page compares the stored action (`data.action`) with what it applied and reads the game again when they differ (one condition in `session.ts`, today only for Rust pages); Opal's 1867 hand drops depot trains at the train limit (the binding decision; as the Rust menu since 3fdd015) |
| Programs on Rust pages | the wasm's Programmer (A, §0 (3)) makes the auto actions in `act`: no more check → post → append for games with programs, and hotseat games with programs stay on Rust |
| Runs drawn by hand on Rust pages | the route editor (`manual.ts`'s hand) on Rust: `act` checks the run with the route check (§6); `auto_routing: false` games open on Rust |
| Liquidity (Spreadsheet) | A: `details` gains Ruby's `Game#liquidity` per player; C shows it (both titles) |
| The last-run panel (95.4%) | A: `lastRun`'s route hexes' order, tracks and laid tiles as Opal's (the step-2 list); target 100% of `ports:last_run` positions |
| Merger candidates' order | A: `merger.partners` in Ruby's `mergeable_candidates` order (its graph walk's order) |
| An unlimited tile's count after an upgrade | A: the Tiles drawer's count from Ruby's supply (`add_extra_tile` / `update_tile_lists`: an upgraded-over 7/8/9 returns) |
| ~100 MB of wasm memory in long games | A: `at`'s checkpoints at most 8 (spaced by the game's length), the rest replayed from the nearest (1830 ~10 µs, 1867 ~40 µs an action); target < 30 MB after game 1119 |
| Slow replays on dense maps | A: §11 |
| API | WasmGame API 3 = API 2 + `version().features` `["programs", "route_check"]`; `act` accepts any valid run and makes the programs' auto actions; the worker accepts api 2 or 3 (a page on an api-2 package keeps step 2's behaviour) |

---

## 11. (9) The corporation-graph speed-up

### 11.1 Where the time goes

`map.rs` `compute_corp_graph` walks from each of the company's stations (`walk_from_node` / `walk_into_hex`). Where
track converges (`is_converging_exit`), the walk un-visits paths and nodes after each branch, so it enumerates every
path through the converging region. The callers that run it after every lay (the cache is cleared on any tile or token
change) are `steps.rs` `track_has_layable_tile` (via `factored.rs` `lay_tile_candidate_hexes`, the Track step's skip
check) and `game.rs` `can_run_route` (the Route skip); also `must_buy_train`, `tokenable_cities_for`,
`connected_hexes`. Instrumented on gen-2053 (scratch): the slow computations are NTR's and NYC's with 3 stations,
**~1.0-1.1 million walk calls each** (190-230k node walks, 680-870k hex walks, all but a few thousand in converging
mode), 0.6-2.3 s each; 304 computations make the game's 6.4-9.7 s. Each call allocates and hashes `String` keys
(`(hex.to_string(), path)` for visited paths, `(hex.to_string(), edge)` for the edge counter, `NodeId`'s `String`,
`visited_hexes.entry(hex.to_string())`): ~1-1.5 µs a call.

### 11.2 Shortcuts probed and rejected (they are not exact)

- **Prune a converging sub-walk whose over-approximated reach is already recorded** (a converging call restores every
  constraint it sets, so its only lasting effect is what it records; skipping it when everything it could record is
  recorded is exact). Implemented in scratch with a differential check (old and new graph compared at every
  computation: equal on gen-2053), but **ineffective**: only 17-18% of the checks found the reach covered (the walk's
  constraints keep it from recording much of its over-approximated reach), and the game got slower (8.1 s vs 6.4 s).
- **Replace the walk by plain reachability** (a BFS that ignores the walk's stack constraints): the walk's result
  equals it in only 73% (gen-2053, 1269) and 88% (40 large 1830 games) of token walks; elsewhere reachability finds
  more hexes, edges and cities (e.g. C&O at H10: H14's edges, I15's city). A BFS would change which tiles and stations
  are legal: ruled out by the training session's condition.

### 11.3 The approach: the same walk on flat arrays (exact by construction)

`compute_corp_graph` builds, once per call, a flat context over the game's hexes (index-based: per hex its tile's
paths as `[exit edges, nodes, junction, terminal]` in `tile.paths` order, its converging-exit mask, each city's
`blocked` flag for this company, the neighbour index across each edge), then runs **the same recursion** with integer
state: visited paths as a `u16` mask per hex, the edge counter as `[u8; 6]` per hex, visited / result nodes as bitsets
over a node numbering, visited hexes as a 7-bit mask per hex. Every decision the walk makes reads the same facts in the
same order (`tile.paths` order; `is_converging_exit`'s and `is_city_blocked`'s answers, precomputed per call, are
pure functions of the tile and the company); only the containers change. The result is converted back to today's
`CorpGraph` (`String` keys, `HashSet<NodeId>`, `tokenable_cities` sorted as now). Expected: 30-80 ns a call instead of
1-1.5 µs, the worst computation ~50-100 ms, gen-2053 ~0.5-1 s.

**Functions that change** (all private to `src/map.rs`): `compute_corp_graph` (body: the flat context, the token
walks, the conversion; signature unchanged), `walk_from_node` and `walk_into_hex` (index-based, same control flow),
`is_converging_exit` and `is_city_blocked` (computed into the context; the functions stay for the reference copy).
**Unchanged**: `GraphCache` and `get_or_compute` (and its `g1867::legal::graph_memo` memo), `CorpGraph`,
`TokenableCity`, `NodeId`, `NodeType`, `corp_has_token_in_city`, `TILE_RESERVATION*`, every caller (`game.rs`,
`steps.rs`, `factored.rs`, `rounds/merger.rs`), and 1867's and 1822's own graph walks (`g1867/graph.rs`,
`g1867/fastwalk.rs`, `g1822/*`).

**As built (G1, §9.2's list)**: as planned, but the reference walk lives beside the flat one (not only in tests): it
is the fallback for a token on a hex the board doesn't index, the one input the flat context doesn't number. In
tests every `compute_corp_graph` computes both and asserts them equal (`graph_cross_check`), so the three neutrality
pins and every other test are differential tests; `the_flat_walk_equals_the_reference_on_generated_1867_games` adds
1867 generator walks (thousands of graphs); the ignored `..._on_a_corpus` (scratch, `GRAPH_CORPUS`) compared every
company's graph at every position of :9301's 1867 games, production's 1830 and 1867 games, 150 :9301 1830 games,
1119 / 1263 / 1269 / gen-2053 and the site fixtures: **64,788 positions, 697,280 graphs equal**. Every position's
shadow key of those four games and :9301's 1867 games is the same under both builds (18 / 18). Timings in §15.

**Verification**: the current code kept verbatim in the test module as `compute_corp_graph_reference`; a differential
test comparing the two `CorpGraph`s (every field, sorted) for every company at every position of the three pins'
walks, the site fixtures and generated 1867 games including gen-2053; a scratch run of the same comparison over every
corpus game (:9301, production, the RL fixtures); the three neutrality pins unchanged under `--features python` and
`--features site`; the timing table of §2.1 re-measured. A second stage (an exact memo of converging sub-walks keyed by
their region's constraints) only if the first falls short, with its own proof and the same checks.

---

## 12. (10) Verification and the production switch

### 12.1 Package A (engine)

- **The stored-move authority check** (`site_replay --authority`, A1): every stored engine move of every corpus game
  (:9301, production read-only, the fixtures) acted at its position by `move`: accepted, with the stored action equal
  to the recorded one (auto actions included where the record has them, revenues and strings), or listed. Targets: 0
  refused but R9's kind; programs' auto actions equal on the 228 1830 actions and every 1867 step pass of §2.4.
- **Runs** (A2): every stored run passes the route check and prices as stored; every router run passes; mutations
  (a connection dropped, a route one stop too long, two routes sharing track, no station, a blocked city crossed, a
  route visiting a hex twice) judged by Ruby (K11's `ruby_verdicts.rb` with run cases): no disagreement.
- **The 1830 move check** (A1): step 1's act cases re-run (76,181 options accepted / refused as before) plus every
  stored move; Opal's options acted in Rust (C's `acts.cjs`): no option Ruby accepts and Rust refuses but R7 / R9.
- **Generated games**: 1830 and 1867 walks under `site_rules` (own-only menus, programs on in some, runs drawn off the
  router's in some) converted to site JSON and replayed by the site's Ruby move by move as `record_action` (B's
  `ruby_verdicts.rb`): every Ruby refusal is a Rust bug or an R-class, listed.
- **The graph** (A4): §11.3's differential test and pins.
- **Native = wasm** for every new output (`move`'s, `view`'s, the Programmer's), the corpus digest as before.
- **Gate at every commit**: `cargo test --release --no-default-features --features python` and `--features site`
  (all three pins), `cargo check --target wasm32-unknown-unknown --features wasm,site-titles` and `--features wasm`,
  the default `cargo check`; before proposing the branch, `uv sync` and the full `pytest tests/` in this worktree's
  venv (run in the foreground, 20-40 min). All under `taskset -c 24-31,56-63 nice -n 19`, `CUDA_VISIBLE_DEVICES=`.

### 12.2 Package B (server)

- `bot_view_compare.rb` byte-equal at every corpus position (§8.1); `POST /api/bot/engine` byte-equal on 250 requests;
  specs (`spec/requests spec/routes spec/models spec/lib/*_spec.rb`) green with authority on and off.
- A scratch stack (`scripts/next-stack.sh up <free 93xx>` from B's worktree; `down -v` at the end): the test users
  `lbfb_p1`..`p4` replay corpus games move by move through `POST /api/game/<id>/action` (Rust pages and Opal pages,
  undo / redo every ~90 moves, mispriced runs, runs drawn off the router, programs on in some), the dev bots playing
  1830 and 1867 tables: every move stored by Rust, every Ruby check row `equal` or classified; the request's
  `engine_ms` p50 / p95 recorded; the switch flipped both ways mid-game; a refused move, a 503 (binary removed), a
  timeout; the rollback runbook rehearsed (a game pinned `rust`).
- :9301: `ENGINE_AUTHORITY_TITLES=1830,1867` with the Ruby check for **7 days**, every non-equal row classified;
  then `RUST_ENGINE_DEFAULT=rust`. Games 47, 1218, 1219 and real users' games are only read.

### 12.3 Package C (client)

Every corpus game opens on Rust (default) equal to Opal's at every step (parity as before); programs and hand-drawn
runs on Rust pages against the scratch stack; the Opal page's stored-action resync; the fallback paths; the gaps'
numbers (liquidity on every Spreadsheet position, last run 100%, merger order, tile counts, memory < 30 MB after
1119); a hotseat game with programs on Rust from the auction to the end.

### 12.4 The production switch (the parent's, each step with Caleb's OK in the session that makes it)

Preconditions: dev accepted (§12.2's 7 days); the training session reviewed §9.2's list and merged or acknowledged the
branch; the image built from `rl18xx-next` with the pin; production's games re-checked with `site_replay --authority`
on a fresh read-only export (deleted after).

| Step | Change | Watch | Roll back |
|---|---|---|---|
| 0 | deploy the 3a image, `ENGINE_AUTHORITY_TITLES` unset | nothing changes: Ruby decides, the shadow as today | the previous image |
| 1 | `ENGINE_AUTHORITY_TITLES=1830` | `GET /api/admin/engine_shadow?direction=ruby_checks_rust`: every row `equal` or classified; `engine_ms` p95 < 100 ms; no 503 | admin switch `{title: "1830", engine: "ruby"}` (at once), then unset the env |
| 2 | `ENGINE_AUTHORITY_TITLES=1830,1867` | the same; R1-R6 rows expected only in their situations | admin switch for 1867; games listed `ruby_refused` / `ruby_stuck` pinned `rust` first (§3.5) |
| 3 | `RUST_ENGINE_TITLES=1830,1867`, `RUST_ENGINE_DEFAULT=rust` | client `engine_report` rows (`fallback` should be rare) | `RUST_ENGINE_DEFAULT=opal` (pages), or unset `RUST_ENGINE_TITLES` (the kill switch) |

A bad engine build: re-pin `engine-rs/PINNED` to the previous commit and rebuild (as steps 1-2); the authority switch
covers the time it takes.

---

## 13. (11) Packages, files and contracts

### 13.1 Shared ground

- Scratch: `RUST3=/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust3`:
  `pkg/<sha>/` and `pkg/current` (A: K2's files with `move` / `start` / `view`), `corpus/` (fetched by
  `scripts/site_engine/fetch_corpus.py --root`; production's files 600, deleted when the checks are done), `acts/`,
  `reports/`, `A/`, `B/`, `C/`; this design's probes in `design/` (its production files deleted with it).
- The rules of steps 1-2 (§9.1 there), and: all three neutrality pins at every commit of A; nothing in `mcts.rs` /
  `mcts_lockstep.rs`; §9.2's list relayed to the training session before G1 or any V3 lands.
- Ports: B's scratch stack **:9393**, C's `server.mjs` **:9494**, A's browser smoke **:9495** (check they are free).

### 13.2 Contracts (a change is a commit to this doc first; the parent relays it)

| Contract | From → to | Step 3a |
|---|---|---|
| K1' WasmGame API 3 | A → C | API 2 + `features: ["programs", "route_check", "authority"]`; `version().api` is **3** (C's worker must accept it before B serves the package to pages: today's worker falls back to Opal unless api is 2); `act` is the server's decision (the 1830 menu check, the route check, the Programmer) and returns the action as the server stores it (id, `created_at` from the page's clock, Rust's `auto_actions`); `details.sheet.liquidity` `{player id: n}`; `tiles.left` of an unlimited tile from Ruby's supply; `lastRun` as Opal's; 1867's merger `partners` in Ruby's order; at most 8 checkpoints |
| K2 package files | A → B, C | unchanged names |
| K3' `site_engine` | A → B | + `move`, `start`, `view` (§3.2, as built there); exit 4 added (an internal error the engine caught); `act` now makes the auto actions (its `add_auto_actions` ignored) |
| K4 key v2 | A ↔ B | unchanged; now also Rust's key stored with each Rust-decided move |
| K5, K6 build, vendoring | A → B | unchanged (`site/engine-rs-step3`, `PINNED`) |
| K7' boot / titles API | B → C | `engine.rust.authority: [titles]` |
| K8' posting | B → C | for authority titles every post is treated as `X-Engine: rust` (posted auto actions ignored); the response unchanged in shape (`action`, `repriced?`); a refusal `500 Illegal action: ...` as today; `503` when the engine is unavailable |
| K9 engine_report | B → C | unchanged (`refused` now carries Rust's refusals on Opal pages: R7 / R9) |
| K10, K11 title packs, act cases | | unchanged; K11 gains run cases (§12.1) |
| **K12** bot view | B → the bots | byte-compatible (§8.1): no change |
| **K13** the Ruby check rows | B | `direction`, `class`, the statuses of §4.2; `GET /api/admin/engine_shadow?direction=&class=` |
| **K14** the authority switch | B | `ENGINE_AUTHORITY_TITLES`, `POST /api/admin/engine_authority`, `settings.engine_authority` |
| **K15** `POST /api/bot/engine` | B → the bots | unchanged in 3a (§8.2) |

### 13.3 The packages

**A — engine** (this worktree, `site/engine-rs-step3`). Files: `engine-rs/src/site/**`, `engine-rs/src/bin/*`,
`engine-rs/site/*`, §9.2's list, `scripts/site_engine/**`, this doc. Never Python, the site's or the client's repos.

| Step | Work |
|---|---|
| A0 | (done) the 1822 pin (9fc1b38); this doc |
| A1 | `move` / `start` / `view` (K3'); the 1830 move check (§5.2); `actors`, undo / redo, `state`; `site_replay --authority` over every corpus (§12.1); publish a package |
| A2 | the route check and pricing for both titles (§6), in `move` and the wasm's `act`; run mutation cases (K11) |
| A3 | the Programmer (§0 (3)), in `move` and the wasm's `act`; the stored auto actions of §2.4 reproduced |
| A4 | G1, the graph (§11), after the parent relayed §9.2 to the training session; its differential test and the pins |
| A5 | the client gaps' engine parts (§10): liquidity, last run, merger order, tile counts, checkpoints; API 3 |
| A6 | generated games through Ruby, native = wasm, the full gate, the record here, the branch proposed |

**B — server** (worktree `~/workspace/18xx-ws/rust-authority`, branch `next/rust-authority` from `rl18xx-next`).
Files: `routes/game.rb`, `routes/bot.rb` (the view's values), new `lib/engine_authority.rb`, `lib/engine_rs.rb`
(`move` / `start` / `view`, the move timeout), `lib/engine_shadow.rb` (the reversed check, classes), `queue.rb`,
`routes/engine_rs.rb` + `routes/admin.rb` (the switch, the filters), `lib/titles.rb` (`table_problem`), a migration
(`engine_shadow.direction`, `class`), `scripts/engine_rs/bot_view_compare.rb`, `scripts/engine-rs-pull.sh` (the
step-3 repo), `engine-rs/` (pull script only), specs, `docs/rl18xx-next/ENGINE_RS.md`, `DEPLOY.md`. Never
`lib/engine/**`, `client/**` (subtree pulls only).

| Step | Work |
|---|---|
| B0 | vendor A1's package; `EngineRs.move` / `start` / `view` |
| B1 | `EngineAuthority` (K14) and `record_action` / `start_game!` / `set_game_state` on Rust (§3.3); the lobby's refusals (§7) |
| B2 | the Ruby check (§4): rows, worker, comparison, classifier, admin API (K13) |
| B3 | the bot view from Rust + `bot_view_compare.rb` (K12); `POST /api/bot/engine` re-compared (K15) |
| B4 | the scratch stack (§12.2), then :9301; DEPLOY.md's runbook (§12.4) |

**C — client** (new living-board worktree `~/workspace/18xx-concepts/lb-rust-authority`, branch `rust-authority` from
master). Files: `src/engine/session.ts` (authority-aware fallback, the Opal pages' stored-action resync),
`src/engine/moves.ts` (Opal's 1867 train-limit filter), `src/engine-rs/**` (API 3, programs, runs drawn by hand),
`src/engine/manual.ts` (the route editor on Rust), `src/ui/settings.ts`, `src/engine-rs/EngineCard.tsx`, the details
drawer (liquidity), `test/rust/**`, `docs/RUST_ENGINE.md`.

| Step | Work |
|---|---|
| C0 | K7' / K8' handling: the authority-aware fallback and the Opal pages' resync |
| C1 | API 3 in the worker: programs and hand-drawn runs on Rust pages (local games too) |
| C2 | the gaps' client parts (§10) |
| C3 | the checks (§12.3); merge master; suites green |

### 13.4 Order

A0 (done) → the parent relays §9.2 to the training session → A1 publishes; B0-B1 and C0 start on it → A2, A3 (B2, B3,
C1 in parallel) → A4 once the training session has the list → A5, C2 → B4 (scratch stack, then :9301 for 7 days) →
the production steps (§12.4), the parent's.

---

## 14. Risks and open points

- **1830's move check is new** (§5.2): its refusals of stored moves are measured before the switch (A1); a live
  disagreement shows as a client `refused` report and a re-read, never a stored wrong move.
- **The route check** is new code that decides money: it is judged by Ruby on every stored run and on mutations (A2),
  and the Ruby check prices every Rust-stored run again (`last_action`).
- **The Programmer**'s Ruby subtleties (shenanigans since the program, `normal_pass?`, `from_market`, `auto_pass_after`,
  1867's merger pass `disable_others`): judged on the 228 stored 1830 program actions and generated games with programs.
- **Load**: the Ruby check loads a whole game per Rust move (150-300 ms of a queue core); a burst (bots) queues rows,
  never moves.
- **The dense-map tail** until G1 lands: a move up to ~2.5 s natively on a gen-2053-like map (none in real games: their
  worst move is 0.5 ms warm, 99 ms cold).
- **POST /api/bot/engine** after 3b (§8.2) and **engine ownership** (the site pins this branch until the training
  session merges it with Caleb's OK) are the parent's to settle.
- **Known gaps after A (2026-11-27)**: generated games are not yet replayed through the site's Ruby by A (B's
  `ruby_verdicts.rb`, the Ruby check); the route check models neither lanes, local trains, intra-tile ambiguous
  connections (refused `unsupported`), double towns nor `check_overlap`'s off-centre town rule (none in 1830's or
  1867's tiles); 1867's 5+5E through Toronto / Montreal is priced by the replay pricer (R6); a stored game whose replay
  stops is exit 3 (none today: §2.5); the bot view is matched on production's 31 games (23 finished), B's
  comparison covers every position; `site_engine serve` not built (§3.6); a 1830 move the menus don't
  offer and that isn't in §5.2's cross-player set is refused even where Ruby would take it (none in the 172,843 stored
  moves; the client's `engine_report` rows show any live one).

## 15. The record

- **2026-11-27, A0**: worktree and branch created from `site/engine-rs-1867` b4b2391; **9fc1b38** pins the 1822
  neutrality walks (§9.1), its gate green; this doc. The measurements of §2 and §11 were made with scratch tools on a
  copy of b4b2391's engine-rs (`$RUST3/design`, not committed); production's exported games were read through the bot
  API only and deleted with the scratch.
- **2026-11-27, A1-A3 (00e8bf6)**: `site_engine move` / `start` / `view` (K3' as built: §3.2), the 1830 move check
  (§5.2), the route check (§6), the Programmer (§5.3), `act` = the authority's decision (WasmGame API 3, features
  `programs`, `route_check`, `authority`), `site_replay --authority`. Corpus: every kept move of every game decided at
  its position as `move` decides it -- :9301 1830 149,335 / 149,335, 1867 5,019 / 5,019; production (read-only export,
  deleted after) 1830 11,747 / 11,747, 1867 6,742 / 6,742; auto actions and runs as §5.3 and §6 say. No edit outside
  src/site. Package published: `$RUST3/pkg/00e8bf63f6fa`.
- **2026-11-27, A5 (3bb8916)**: the client gaps (§10): `details.sheet.liquidity` (Ruby's `Game#liquidity`), an
  unlimited tile's count from Ruby's supply, ports:last_run rebuilt as Opal's map does (the train's Ruby owner: a
  rusted train's is nil, a discarded one's the depot; lays recorded at the turn they were made), 1867's merger
  partners in Ruby's `mergeable_candidates` order (`g1867/order.rs`), at most 8 `at` checkpoints without their raw
  actions. Against the Opal goldens (315 1830 games, 15 1867): last run 47,458 / 47,458 and 2,065 / 2,065 (were 93%,
  95.4%), liquidity 32,805 / 32,805 and 1,084 / 1,084, tile counts 384,422 / 384,422 and 18,428 / 18,428, 1867 menus
  10,578 / 10,704 (was 10,557; the rest: words the client fills in, R2's stale-graph minor). wasm memory after opening
  1119 in the client's batches of 80: 98.8 -> 20.0 MB (1269: 29.1 -> 22.6 MB); `at` 5-31 ms.
- **2026-11-27, G1 (e66ba2c)**: the corporation graph's walk on flat arrays, its own commit, §9.2's list as built
  (§11.3 "As built"). Every graph computed in the tests is checked against the walk as it was (the three neutrality
  pins unchanged); the scratch corpus run: 697,280 graphs at 64,788 positions equal. Timings (cores 24-31,56-63,
  load ~35, min of 3; before -> after): `site_engine state` gen-2053 6,207 -> 193 ms, 1269 51 -> 50 ms, 1119 21 ->
  20 ms, 1263 8 -> 9 ms (real games' graphs were never the cost); every move of gen-2053 decided as `move` decides
  it (`site_replay --authority`) 27.1 -> 2.9 s (28 -> 3 ms a move). Package published: `$RUST3/pkg/e66ba2c3c233`.
- **2026-11-27, A6**: generated 1867 games (`site_replay --generate 12 --seed 3001`, 3-6 players, all finished)
  decided move by move: 13,380 / 13,380 accepted, their 407 step passes equal, 1,003 router runs pass the route check
  at the router's price. Native = wasm: the last 3 engine moves of 62 games (the fixtures, :9301's 1867 games, 40
  1830 games) decided by the wasm's `act` and by `site_engine move` on the game cut before them: 178 / 178 equal
  (stored action with its auto actions, and the key after). The bot views (§8.1): production's 31 equal. The Opal
  goldens re-run at e66ba2c: 1830 snapshots 82,083 / 82,083, menus 322,145 / 322,150, tile options 42,082 / 42,127,
  restartInfo 82,056 / 82,083, last run, liquidity and tile counts 100%, auto routes 7,562 / 7,562, log lines
  171,590 / 171,590; 1867 as A5. Not done here: generated games replayed through the site's Ruby (no Ruby run in A:
  B's `ruby_verdicts.rb` and the Ruby check of §4 judge Rust's moves by Ruby).
- **2026-11-27, 1088b50**: the bot view's `acting` of a game a bankruptcy ended (§8.1): production's 31 views equal.
  `site_engine move` on each game's last engine move (wall, spawn included, min of 3, load ~39): 1119 16 ms, 1263
  8 ms, 1269 49 ms, :9301 1218 (1867) 33 ms, gen-2053 190 ms. **Final package: `$RUST3/pkg/1088b500d2f8`**
  (`pkg/current`; `site_engine version`: api 3, key 2, features site, programs, route_check, authority).
- **2026-11-27, the gate at 1088b50**: `cargo test --release --no-default-features --features python` 273 passed / 0
  failed / 11 ignored and `--features site` 305 / 0 / 11 (the 1830, 1867 and 1822 neutrality pins unchanged, every
  graph cross-checked), `cargo check --target wasm32-unknown-unknown` with `wasm,site-titles` and with `wasm`, the
  default `cargo check`: clean; `uv sync` and the full `pytest tests/` in this worktree's venv (the wheel of e66ba2c:
  1088b50 changed only site code): **1,822 passed, 7 skipped, 0 failed** (25:51). Edits outside src/site in A: N3
  (9fc1b38) and G1 (e66ba2c), as §9.2 lists them; nothing in mcts.rs, mcts_lockstep.rs, the policy layouts or
  `site_rules`.

