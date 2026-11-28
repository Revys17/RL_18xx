# The site on engine-rs, step 4: 1822 and 1822CA (plan)

Status: **plan, 2026-11-27; package A (engine) built 2026-11-28** (§8: branch `site/engine-rs-1822`, RL_18xx worktree
`.claude/worktrees/site-engine-1822`, on step 3a's rebased head fa11715; the verification's fixes in §8.10, code head
1069a7e). The plan below is unchanged; §8 is the record of what was built, the edits outside `src/site` function by
function, the Ruby diff, the RESTRICTED-vs-menu comparison and the gaps. Original status: plan, 2026-11-27 (branch `site/engine-rs-1822-plan`, RL_18xx worktree `.claude/worktrees/site-1822-plan`, on
`site/engine-rs-1867` b4b2391). Approved by Caleb: after the site's move to engine-rs (steps 1-3), 1822 and 1822CA
become playable on 18xx.revys.net (`rl18xx-next`), with bots once the RL session's 1822 model is ready (run
`gpg1822_a4_20261127`, started 2026-11-27 13:12). This document is read-only research plus a plan: nothing is built
here. **Building starts after step 3a lands** (engine-rs authoritative on the server, branch `site/engine-rs-step3`,
design `docs/SITE_ENGINE_RS_STEP3.md` when written), because it touches the same code (the site layer's dispatch,
`site_engine`, the server's authority path, the Rust Programmer, the client's worker).

Read with: `docs/SITE_ENGINE_RS.md` (step 1: the site layer, WasmGame API 2, RawMenu, `site_rules`, the shadow key,
`site_engine`, contracts K1-K11), `docs/SITE_ENGINE_RS_1867.md` (step 2: how a title is added), the step-3a design,
`docs/g1822_remaining_work.md` (engine-rs's 1822 family: rules, generator, router, validation), `docs/rl_model.md`
"1822 family" and "1822+: the optional rules" (the RL side), living-board `docs/TITLES.md` §10 (the Opal hands today).
Scratch of this probe: `$RUST4=/tmp/claude-1000/-home-revys-workspace/252285c2-5545-52b0-bcda-165a3e4e9ceb/scratchpad/rust4`.

---

## 0. In one page

| Question | Answer |
|---|---|
| What engine-rs has | The whole 1822 family's rules, generator and router: **1822** (2-7 players, all four optional rules), **1822CA** (2-7, no optional rules), 1822PNW, 1822MX; each validated against Ruby HEAD (fixtures, scrape corpus with zero per-action divergence, router check, generator completeness >= 99.985%, fuzz games replayed by Ruby). Not implemented: **1822CA ERS / WRS**, 1822NRS / MRS, 1822Africa, the PNW scenarios. (§1) |
| What "1822CA" is on the site | The site's `1822CA` is the **base game** (Ruby `G1822CA`, beta, 2-7 players, no optional rules). ERS and WRS are **separate titles** (`1822CA ERS`, `1822CA WRS`: variants on half maps, 2-5 players, their own trains, market and companies). engine-rs plays base 1822CA only. (§1.2) |
| The site's own 1822 games | **None real.** Production: 0 of its 31 games (19 × 1830, 12 × 1867; the read-only export, nothing kept). Dev :9301: one, game 1001 (`1822CA WRS`), an import of the WRS fixture by the titles test. So the probe's corpus is the site's fixtures. (§2.1) |
| Parity probe | 1822: **4 / 4** fixtures replay to the end, results equal; recorded decisions offered by the generator **2,880 / 2,880**; router vs stored runs 286 equal / 59 higher / **3 lower**. 1822CA: **7 / 7**, results equal; **5,551 / 5,553** (the 2: out-of-order discards, replay-only); router 597 / 58 / **1 lower**. Every "lower" is an old record the site's Ruby itself refuses or prices lower (checked in a throwaway container). ERS / WRS (2 + 2 fixtures, dev game 1001): **0 / 5**, unknown title. Step auto actions predicted = Ruby's at every 1822CA move and every 1822 move but 55 old-record positions. (§2) |
| Human decisions | ~30 kinds (§3). The Opal hand today plays the stock round's buys and sells, track, stations, runs, dividends and depot trains; **everything particular to 1822 goes to the plain list** or isn't offered: bid boxes (994 list-only decisions in the CA fixtures, 272 in 1822's), private acquisitions, minor acquisitions, every `choose` (pullman, extra-train dividend, P9, P8, acquisition pay / token, sawmill), tax haven, debt payoff; concession conversions (`par`) and company stations with a `tokener` aren't offered at all. RawMenu covers the shared half; **~12 new facts** are needed (§3.3). |
| Title pack, shadow, site_rules | Packs `title-1822` and `title-1822CA` (§4.1). Shadow key: v2 plus 1822-only fields (bid boxes, stacks, debt, tax haven, credits, exchange tokens, reservations, CA's sawmill / blockers) (§4.3). **No `site_rules` branch** (§4.4). Edits outside `src/site`: the registry arm (P4) and the 1822 timing calls (P5: the generator and router call `std::time::Instant::now`, which panics in wasm), plus visibility only (V3). The training session's note on `site_interplayer_purchase_open` is already met by step 2's phase-status table; for 1822 a test pins it (§4.5). |
| Site-layer specifics | **Seat ids**: site user ids collide with 1822's numeric minors ("1".."30"; the bots are users 2, 3, 4, 11, 12 on :9301, whose DB was seeded from the live one) and engine-rs refuses colliding ids, so the 1822 site layer maps every site id to a seat id (id + 1000) at the boundary. **Records**: the 1822 step machine is its own (`crate::g1822`), so step 1's record hooks H1-H5 never fire: the site layer records operating history and round values itself. **Auto actions**: Ruby's DestinationToken, MinorAcquisition and P9 Choose auto actions are predicted in Rust (as 1867's three). (§4.2) |
| Bots | Site: `BOT_TITLES` += 1822; `OPTIONAL_RULES['1822']` = the four rules; `PLAYER_RANGE['1822'] = [3, 6]`; analysis stays 1830; 1822CA stays bot-free (no model). The RL driver (`titled.py`) needs an 1822 arm; questions for the training session in §5.2. |
| Effort | ~30-45 agent-hours: A engine site layer 12-16 h, B server 3-5 h, C client 10-16 h (new hands), verification 4-6 h, fixes 2-4 h; about 1.5-2 days wall-clock as a workflow (A first, then B and C in parallel). ERS / WRS would add 6-10 h of RL-side engine work, if wanted. (§6) |
| Main risks | Size of the decision surface; 1822CA's router and generator tails on dense late maps (natively up to ~5-9 s in the RL corpus; 2-3× in wasm); the step-1 rule "act accepts only the router's run" meeting a router shortfall (none on the site's corpus; 1 known in the RL scrape); MRI can't price 1822CA runs (arity bug) for Ruby's reversed check; wasm +~100 KB gzip (measured) plus the site layer. (§6.3) |

---

## 1. Variants: engine-rs against the site's Ruby

### 1.1 The family

| Site title (Ruby) | Site's Ruby (`rl18xx-next`, tobymao 24e59e1 2026-10-30) | engine-rs (`title/*.rs`, `g1822/*`) | RL training |
|---|---|---|---|
| `1822` (`G1822`, production) | 2-7 players; optional rules `plus_expansion` (1822+), `plus_expansion_no_removals`, `plus_expansion_single_stack`, `tax_haven_multiple`; variants 1822NRS / 1822MRS (other titles) | **Done**: 2-7 construct (the 2p / 7p rules are data; validated by replay only: "no effort goes into 2p/7p-only rules"); all four optional rules (anything else refused at construction). Fixtures 4 / 4 exact scores; 2026-10 scrape 743 / 743 outcome match; zero per-action divergence from Ruby HEAD; completeness 99.986% (rules set), 99.9874% by index over all ten rule combinations | **Training now**: generic PG `gpg1822_a4_20261127` from the 1822+ SL (top-1 .715), 3-6 players (mix 3:26.5 / 4:45.5 / 5:18 / 6:10), the ten canonical rule combinations weighted by the human games (`plus+nr+ss+th` 23.7%, `base` 18.3%, ...), T 0.5, plateau rule (backstop 1,000 updates) |
| `1822CA` (`G1822CA`, beta) | 2-7 players; **no optional rules**; variants ERS / WRS (other titles) | **Done**: base 1822CA, 2-7 (3-6 validated); fixtures 7 / 7; scrape 178 / 187 + 9 Ruby-rejects-same; completeness 99.985%; router 1 known shortfall (257158 @1556) | SL only (top-1 .680); **no PG run, no bot** |
| `1822CA ERS` / `1822CA WRS` (variants of 1822CA) | 2-5 players; half maps (`EASTERN_HEXES` / `WESTERN_HEXES` + two offboards), `G1822CA::Scenario`: one-row 33-cell market, own train roster (WRS adds two G trains), cert limit / cash for 2-5, own starting companies (WRS: M16-M30, C1-C3 C5 C8 C9, P1-P8 P11 P18-P20 P22 P24 P26-P29), no Detroit-Duluth blocker, no extra OR at the bank break, UPGRADE_COST_L_TO_2_PHASE_2 70, WRS: GNWR's pending home in Thunder Bay with a cheater token | **Not implemented** ("would go the same way": `scripts/g1822_title_data.{rb,py}`, a `Family` row, a `Variant` arm, the same validation) | none |
| 1822PNW, 1822MX | production | done | SL only |
| 1822NRS, 1822MRS, 1822Africa, 1822PNW Short / SRS | various | not implemented | none |

Two Ruby references differ: engine-rs's 1822 data and checks came from a tobymao clone at **46a592e** (~2026-10-07),
the site's Ruby is tobymao **24e59e1** (2026-10-30, the fork's base). Every site fixture replays to the same result
in both (§2), but a `git diff 46a592e 24e59e1 -- lib/engine/game/g_1822 lib/engine/game/g_1822_ca` is step A0's first
check (the training session's rig clone has 46a592e; the site repo starts at 24e59e1).

### 1.2 Which "1822CA"

The user's "1822CA" can only mean the base game on engine-rs today. On the site the three are distinct titles in the
catalog (`GET /api/titles`: `1822CA` with `variants: [ERS, WRS]`, each variant `variant_of: "1822CA"`, `depends_on`
1822CA, `engine: [g_1822, g_1822_ca, g_1822_ca_<ers|wrs>]`). The site's only 1822-family game (dev 1001) is a WRS, and
living-board's titles check names WRS as its 1822CA sample (TITLES.md §10: 450 hand / 168 list / 1 not offered), which
is probably where "1822CA" and "WRS" got mixed. Recommendation: **step 4 = 1822 + base 1822CA**; ERS / WRS only if
Caleb wants them (§6.4 D1), as RL-side engine work first.

### 1.3 What the site layer supports

| | 1822 | 1822CA |
|---|---|---|
| Players | 2-7 (Ruby's range). 3-6 fully validated; 2 and 7 by replay (a 7-player fixture each passes) and by generated games before the default switch | 2-7, same |
| Optional rules | the four, any combination (No Removals / Single Stack without 1822+ are no-ops in Ruby and engine-rs alike) | none (`unsupported:` for any) |
| Seed | `settings.seed` (else the id's first digit run, Ruby `initialize_seed`): the company stacks are dealt by Ruby's LCG, which engine-rs reproduces | same |
| Not supported | runs drawn by hand (`auto_routing: false`, as 1830/1867), ERS / WRS (another title) | same |

---

## 2. Parity probe

### 2.1 Corpus

| Set | How | 1822 | 1822CA | ERS / WRS |
|---|---|---|---|---|
| Production (:9292) | the bot API's read-only export, ids 1-35 (stop after 80 misses), keeping the 1822 family, files 600 | 0 | 0 | 0 |
| Dev (:9301) | DB (read-only query) + `GET /api/game/1001` | 0 | 0 | 1 (game 1001 "titles test: 1822CA WRS full", an import of `1822CAWRS_game_end_stock_market`) |
| Fixtures | rl18xx-next `public/fixtures/` (= `spec/fixtures/`) | 4: `1822_game_end_stock_market` (4p, plus+nr+ss), `33845` (4p, plus), `discard_trains_after_sr_export` (3p), `hs_lxemeslq_30797` (7p) | 7: `1`, `2` (7p), `3`, `4`, `1822CA_game_end_bank`, `1822CA_game_end_stock_market`, `loan_payments_preserve_bank` (3-7p) | 2 + 2 |
| Raw / processed actions | | 3,971 / 3,345 | 7,549 / 6,819 | |

The RL human corpus (`human_games/scrape_2026_10/1822` 743 games, `.../1822CA` 187) lives only in the RL_18xx main
checkout, which this work doesn't touch; its numbers are the RL session's (§1.1). As in step 2, the verification stage
asks the parent once for a read-only copy (real users' names: files 600, deleted after).

### 2.2 Results

engine-rs as built at b4b2391 (`--features python` wheel of the step-2 worktree's venv: its `g1822` is master's;
runs in `$RUST4/probe`, through RL_18xx's own audit scripts: `tests/corpus_audit.py`, `scripts/g1822_completeness.py`,
`scripts/g1822_router_check.py`, plus three scratch probes `autos.py`, `timing.py`, `census.py`):

| | 1822 (4 games) | 1822CA (7 games) |
|---|---|---|
| Strict replay to the last action | 4 / 4 | 7 / 7 |
| Result = the stored result | 4 / 4 (3p, 4p, 4p, 7p) | 7 / 7 (3p, 3p, 4p, 4p, 5p, 6p, 7p) |
| Recorded decisions (`run_routes` aside) matching exactly one move of Ruby's set (`RULES`) | **2,880 / 2,880** (0 ambiguous) | **5,551 / 5,553** (0 ambiguous) |
| ... of which the own-only set (`RESTRICTED`) leaves out | 0 | 13 (7 private acquisitions from another player, 5 trains from another player's company, 1 minor of another player: the site's rule, §3.4) |
| Stored runs vs the router: equal / router higher / router lower | 286 / 59 / **3** | 597 / 58 / **1** |
| Ruby's step auto actions after each recorded move = Rust's prediction (§4.2) | 3,106 equal + 67 programs' (players' passes / buys) + **55 old records** of 3,228 | 5,957 equal + 252 programs' of 6,209; **0 differ** |
| ERS / WRS (2 + 2 fixtures, dev 1001) | — | 0 / 5: `unknown game title: 1822CA WRS` |

Decisions by family (Ruby's set, every one matched): 1822: pass 1,281, lay_tile 405, bid 371, dividend 202,
buy_shares 154, buy_train 149, choose 131, acquire_company 51, place_token 38, merge 32, par 24, hex_token 13,
discard_train 11, sell_shares 11, choose_ability 7. 1822CA: pass 2,458, bid 870, lay_tile 831, dividend 298,
buy_train 269, choose 244, buy_shares 218, acquire_company 110, par 64, merge 54, place_token 53, sell_shares 39,
discard_train 16 (+2 missing), payoff_player_debt 6, choose_ability 6, assign 2.

### 2.3 The gaps, each with an example

| # | Gap | Example | Verdict |
|---|---|---|---|
| G1 | Router below a stored run (1822, 3) | `1822_game_end_stock_market` @1319 CR: stored 1,040, router 1,010; @1499 LYR: 1,560 vs 1,490 | The site's Ruby (MRI, `18xx-next-prod:128467c8ff2b` in a throwaway `--network none` container, revenue stripped) **refuses the stored run**: "E-train route must start and end at tokened cities". Old record. |
| | | same game @1404 CR: stored 1,280, router 1,250 | Ruby prices the stored routes at **1,250** (780 + 310 + 160): the record counted a destination bonus on two routes. Old record. |
| G2 | Router below a stored run (1822CA, 1) | `3` @942 GTP: stored 540 (5P 310 + 3 140 + 2P 90), router 530 | Ruby refuses the 5P route: "Route must contain token". Old record. Note: engine-rs's `g1822_price_routes_json` priced that route at 310, so **pricing is not validation**: it skips Ruby's token check. Harmless for the site as long as `act` accepts only the router's run; a hand-drawn-run feature for 1822 would need a validator (§6.3 R4). |
| G3 | Recorded moves no generated move matches (1822CA, 2) | `1822CA_game_end_stock_market` @737, `loan_payments_preserve_bank` @1086: `discard_train` | Out-of-order discards (Ruby's view lets any crowded company discard first; the generator discards in Ruby's crowded order, same end states). Replay-only; `append` takes them; `act` would refuse such a post. |
| G4 | Ruby adds auto actions the old records lack (1822, 55) | `33845` after file action 316 (lay_tile by LBSCR): Ruby now auto-passes LBSCR's DestinationToken step, the record has the player's own pass next; `discard_trains_after_sr_export` after 107: Ruby auto-passes 1002's P9 choice (doubling wouldn't change the order), the record has a `choose` | Records made before Ruby's step auto actions existed (DestinationToken 38, MinorAcquisition 13, Choose 4; the newest 1822 fixture has only the P9 one). The same class as step 2's 104 old 1867 fixture records. Also why living-board's titles check stops replaying these fixtures early (§3.1). Not a Rust gap. |
| G5 | ERS / WRS | dev 1001, 4 fixtures | Not implemented in engine-rs (§1.1). |
| G6 | Player ids | `BaseGame.new_titled("1822", {2:..,3:..,4:..})` → "player id 2 collides with a corporation sym (use ids above 100)" | The probe seated players at 1001+ as RL's harness does; the site layer must do the same (§4.2). |
| G7 | wasm panics | `g1822/legal.rs` (3), `router.rs` (3), `prof.rs` (3) call `std::time::Instant::now` unconditionally | As 1867's before P3: the first 1822 menu in the browser would abort. Edit P5 (§4.4). |

Not gaps but worth knowing: the 1822 fixture with the most auto actions (`1822_game_end_stock_market`, the newest)
predicts at all but one position; 1822CA's P10 (Winnipeg token) is a parallel company ability the generator lists
beside the blocking step's pass, and Ruby still auto-passes the step (the prediction reads the blocking step's own
moves only: with that, 0 differences).

### 2.4 Speed (native, this machine, the RL training on all 64 cores, probes niced on 16)

| | 1822 | 1822CA |
|---|---|---|
| Plain replay of a whole game | 113-170 ms (~0.14 ms an action; 1867: 25-50 ms a game) | 55-192 ms |
| Generator per decision (Ruby's set) p50 / p90 / p99 / max | 0.55 / 3.3 / 18 / 53 ms | 0.61 / 2.4 / 7.1 / 48 ms |
| Router per run p50 / p99 / max | 0.2 / 28 / 85 ms | 0.13 / 8 / 48 ms |
| RL corpus (g1822_remaining_work.md, after the 2026-11-14 speed-up) | generator p99 51 ms, max 372 ms; router 1.3 ms a call, slowest 0.59 s | generator p99 40 ms, **max 5.5 s** (one position); router ~30 ms a call, **5-9 s** at the densest late positions |

The browser's wasm is 2-3× native.

### 2.5 wasm size (measured on scratch copies of b4b2391, `wasm-release` + wasm-bindgen 0.2.123 + `wasm-opt -Oz`)

| Build | raw | gzip -9 |
|---|---|---|
| Today's `site-titles` (1830 + 1867) | 1,907,840 | 694,964 |
| + 1822 and 1822CA registered (the g1822 engine was already linked through the shared dispatch) | 2,011,616 (+104 KB) | 725,276 (+30 KB) |
| + the 1822 generator, router and move wires linked (a scratch export calling them, as the site layer will) | 2,200,198 (+292 KB) | 794,447 (+99 KB, +14%) |

The 1822 site layer itself comes on top (1867's added several hundred KB raw); estimate ~2.4 MB raw / ~850-900 KB
gzip for the four-title package. The Opal bundles it replaces for these titles are `g_1822.js` 410 KB +
`g_1822_ca.js` 207 KB (uncompressed, besides `engine.js`).

---

## 3. Human decisions

### 3.1 How the Opal hand does today

living-board's `test/titles_engine.cjs --base :9301` (LIST_ONLY=1, every site fixture; read-only: the bundles are
GETs, cached under `$RUST4/tmp`):

| | 1822 (4 fixtures) | 1822CA (7 fixtures) |
|---|---|---|
| Replayed | 827 of 3,971 actions: every fixture stops early on Ruby's newer auto actions (G4) or old-rule moves | 5,485 of 7,549 (4 whole games) |
| Decisions: hand / plain list / as JSON / not offered | 517 / 274 / 0 / 2 | 3,525 / 994 / 2 / 30 |
| List only, by type | bid 272, acquire_company 1, par 1 | bid 722, choose 152, acquire_company 74, merge 33, sell_shares 7 (IssueShares), payoff_player_debt 3, par 2, assign 1 |
| As JSON only | — | choose_ability 2 |
| Not offered | par 2 (concession conversions) | par 8 (concession conversions), place_token 7 (with `tokener`: P10's Winnipeg token, a destination token, old records' station format), buy_train 3, sell_shares 1, chat `log` 11 (fixture noise) |

So the hand covers the shared 1830-shaped decisions; the 1822-specific ones are the list's (one plain picker per type,
asked twice) or missing.

### 3.2 Every decision

Columns: the Ruby step and wire (what the generator produces, `g1822_legal_moves_json`); **Opal today** (hand / list /
JSON / none); **RawMenu** (existing facts reused, or the new fact of §3.3); the **own-only** rule where it applies.
"Auto" = Ruby makes it without the player (the Rust page predicts it, §4.2).

**Stock round (`G1822::Step::BuySellParShares`)**

| Decision | Wire | Opal today | RawMenu |
|---|---|---|---|
| Bid on a bid-box lot: a minor (M1-M30 → minor `1`-`30`), a concession (C1-C10 → its major), a private (P1-P21; CA P1-P30). Up to 3 bids a turn (`BIDDING_TOKENS_PER_ACTION`), one bidding token per lot bid on (`BIDDING_TOKENS` 8/6/5/4/3/3 for 2-7 players), committed cash, the certificate limit counts bids, no bid where one already holds the high bid, a raise of one's own bid allowed; lots resolve at the end of the round | `bid {company, price}`, range [min_bid, available cash] step £5 (`MIN_BID_INCREMENT`) | list | **new `bidBoxes`** |
| Convert a concession (phase 2+, `can_convert_concessions`): par its major, pay `price × 2 − the concession's value` (LNWR's C1: its share price) | `par {corporation, share_price}` (1822: "60,8,4" on the 2-D market; CA: "60,0,10" on the one-row market) | **none** (the hand's 1830 par arithmetic misses the exchange) | `par` with `via` (the concession) + **`exchangeValue`** |
| Par a major from the market (phase 5+ `can_par`, CA) | `par` | hand where 1830's arithmetic fits | `par` |
| Buy a share (majors only; IPO is "Treasury"; not after bidding this turn; not with debt) | `buy_shares {shares, percent}` | hand | `buy`, `turn`, `blocks` + **block codes `bids`, `debt`** |
| Sell shares (president's sell rules; CA `MUST_SELL_IN_BLOCKS`) | `sell_shares` | hand | `sell` |
| Must sell (certificates with bids over the limit) | `sell_shares` only | hand | `turn.mustSell` (+ "bids count") |
| Tax haven (P16; CA P11): buy a share into the haven, from IPO or market, paid by the owner (base: one share in all; `tax_haven_multiple`: with the haven's own cash, one a round, at most one per company) — the company acts, in its owner's turn, not after a bid or with debt | `choose_ability {entity: P16, choice: "<corp>_ipo" / "<corp>_market"}` | list / JSON | **new `taxHaven`** |
| Pay off player debt (whole / part) | `payoff_player_debt`; `payoff_player_debt_partial {amount}` 1..min(cash, debt) | list | **new `debt`** |
| Pass ("Pass (Bids)" after a bid) | `pass` | hand | `pass` (+ the description) |
| Programs (`program_buy_shares`, `program_share_pass`, Ruby's base `available_programmed_actions`) | programs | Auto tab | `programs` (step 3a's Rust Programmer, §3.5) |

**Choices round (Ruby `G1822::Round::Choices`, named "Choose")**

| Decision | Wire | Opal today | RawMenu |
|---|---|---|---|
| P9 (CA P7) held by a player: double cash for the turn order | `choose {choice: "double"}` / `pass`; **auto pass** when doubling can't change the order | list | **new `choices`** (context `double_cash`) |

**Operating round** (Ruby's step order: DiscardTrain, PendingToken, FirstTurnHousekeeping, AcquireCompany,
SpecialChoose, SpecialTrack, SpecialToken, Track, DestinationToken, Token, Route, Dividend, BuyTrain,
MinorAcquisition, PendingToken, IssueShares; 1822CA adds AssignSawmill and AcquisitionTrack)

| Decision | Wire | Opal today | RawMenu | Own-only |
|---|---|---|---|---|
| A minor's first turn: buy its L train from the depot (or pass) | `buy_train {train, price, variant: "L"}` / `pass` | hand | `trains` | |
| Acquire a private (start of the turn; free; majors, or minors for MAJOR/MINOR privates; P1 brings a 5-train, P3/P4/P19 permanent trains, P13/P14 a pullman, P15/P20 their credits, P8/P10 discounts) | `acquire_company {company}` / `pass` | list | **new `acquire`** (+ `othersLeftOut`) | the president's own privates only |
| P8 E&GR after acquisition: a free rough/hill/mountain lay, or £20 off every hill/mountain | `choose {choice: "token" / "discount"}` (SpecialChoose) | list | `choices` (`egr`) | |
| Company lays: P2 MtonR (remove a town), P11 B&ER (next colour early), P12 L&SR (extra lays), P21 HSBC (Grimsby/Hull); combos (`combo_entities`, e.g. P2 with P11); CA: P12, P14-P18 (free big-city upgrades, once an OR), P19 / P20 (pass tiles, whole step), P21 (three free yellows, any order), P29 / P30 (remove a town, with P12) | `lay_tile {entity: Pn, hex, tile, rotation[, combo_entities]}` | hand (`abilities`) except combos | `abilities` + **`combos`** | own company's privates |
| Track (minors one lay; majors two yellow or one upgrade; terrain, upgrade fees, water borders, discount tokens, London / multi-city tiles, max exits on city hexes) | `lay_tile` / `pass` | hand | `tileHexes`, `tileDear`, `tilePower`, tile options (cost with the discounts) | |
| Move a card (P17 LUR: a stack's minor / concession / private to the top or bottom; Track, Token or IssueShares time) | `choose_ability {entity: P17, choice: "M26_top"}` | JSON | **new `abilityChoices`** | |
| Swap a station (P18 C&HPR; CA P28): exchange ↔ available | `choose_ability {choice: "exchange" / "available"}` | JSON | `abilityChoices` | |
| P5 LC&DR: move an exchange token to available; its English Channel station | `choose_ability {choice: "exchange"}`; `place_token {entity: P5, city, slot, tokener}` | JSON; hand (abilities) | `abilityChoices`; `tokens` + **`via`, `tokener`** | |
| Destination token (a major's, once connected) | `hex_token {hex, token_type: "destination"}` / `pass`: **auto** (Ruby places it when connected, else passes) | auto | none (CA's exception below) | |
| CA: ICR's destination when it reaches several Québec cities; QMOO's home in Québec; minor 14's London token (1822) | `place_token {city, slot[, token_type]}` (DestinationToken / PendingToken) | none / hand | `tokens` + **`tokenType`** | |
| Station (price £100 / CA's; exchange tokens; London) | `place_token {city, slot}` / `pass` | hand | `tokens` | |
| CA P10: the Winnipeg exchange station (extra slot, any time in the owner's turn) | `place_token {entity: P10, city, slot, tokener}` | none | `tokens` + `via`, `tokener` (beside the step) | |
| Attach a pullman (P13/P14; CA P5/P6) to a train before running | `choose {choice: "<train index>"}` (CA: or "None") | list | **new `routeChoices`** (`pullman`) | |
| CA: attach a grain train (P26/P27) | `choose` (Route) | list | `routeChoices` (`grain`) | |
| Run (L local runs, numbered trains, E-trains at ×2 between tokened cities, the English Channel and France, destination bonuses, mail contracts P6/P7 / CA P22-P25, CA's sawmill and grain / port bonuses) | `run_routes` (the router's, `subsidy`, `extra_revenue`) | hand (auto-route) | `routes {pending}` + `autoRoute` | |
| Dividend (majors): payout / half / withhold; minors pay automatically | `dividend {kind}` | hand | `dividend` + previews (half pay, the 2-D market moves) | |
| The extra train's own dividend (a permanent 2P / LP train's run paid separately) | `choose {choice: payout/half/withhold}` (Dividend) | list | **new `extraDividend`** | |
| Buy trains: depot (L or its 2 variant, 3-7, the 7's E variant), the L → 2 upgrade (£80, `UPGRADE_COST_L_TO_2`; ERS / WRS: 70 from phase 2), another company's train £1..cash; emergency: the president sells, then takes **player debt** (50% interest; `EBUY_CAN_TAKE_PLAYER_LOAN :after_sell`; never from other companies, `EBUY_FROM_OTHERS :never`) | `buy_train {train, price, variant[, exchange]}`, `sell_shares` (player) | hand (3 CA buys not offered) | `trains`, `trainBuy` (`exchanges` for the upgrade, `others`) + **`debt` in `trainBuy`** | other companies of the same president only |
| Discard (crowded companies) | `discard_train` | hand | `discard` | |
| Acquire a minor (majors that have operated twice; own connected minors, or a bank bid-box minor for £200 from phase 5 `can_acquire_minor_bidbox`): pick it, choose the pay (money / one share / two shares, with the difference in cash either way), choose the token (`replace` the minor's token with an exchange token / `exchange` to the charter, or `remove` when no exchange token is left; minor 14: London's pending token) | `merge {corporation}`, then `choose {choice: "money"/"one_share"/"two_shares"}`, then `choose` (token); **auto pass** when nothing connected | list | **new `acquireMinor`** (+ `othersLeftOut`) | the president's own minors (and the bank's) |
| CA: the bonus lay after an acquisition | `lay_tile` / `pass` (AcquisitionTrack) | hand | `tileHexes` (labelled) | |
| CA P13 sawmill: place it, then open (+$20 all) or closed (+$10 owner) | `assign {target: hex}`, `choose {choice: "open"/"closed"}` | list | **new `sawmill`** | |
| Issue (sell treasury shares) or redeem shares (majors, after two runs) | `sell_shares {share_price}` / `buy_shares` by the company | list / hand (`redeem`) | `redeem` + **new `issue`** | |

**Player debt** is the only loan-like thing: taken inside an emergency train buy after the president has sold what
he may, paid off (whole or part) in a stock round, 50% interest added at each stock round's end
(`G1822::Round::Stock`: `add_interest_player_loans!`); companies have no loans.

### 3.3 New RawMenu facts (A builds them from the generator and the 1822 state; C words them)

| Fact | Shape (sketch) |
|---|---|
| `bidBoxes` | `{boxes: [{kind: "minor"/"concession"/"private", category?: "train"/"track"/"other" (1822+), lots: [{sym, name, value, minBid, maxBid, high: {player, price} \| null, mine: price \| null, canBid, why?: "tokens"/"certs"/"high"/"cash"/"debt", action}]}], tokens: {left, total}, committed, available, bidsThisTurn, perTurn: 3}` |
| `exchangeValue` (in `par[]`) | the concession's value toward the president's certificate (C1 LNWR: the share price) |
| `taxHaven` | `{company, spender: "owner"/"haven", cash, options: [{corp, from: "IPO"/"market", price, action}]}` |
| `debt` | `{amount, interest: 50, payoff: action, partial: {min: 1, max, step: 1}}`; in `trainBuy`: `{afterSell: true, wouldBorrow}` |
| `choices` | `{context: "double_cash"/"egr"/"acq_pay"/"acq_token", company?, options: [{choice, numbers..., action}], autoPass?}` (words in `words.ts`) |
| `abilityChoices` | `[{company, kind: "lur"/"chpr"/"lcdr", options: [{choice, card?, where?: "top"/"bottom", action}]}]` |
| `combos` | per company lay: the privates it may combine with (`combo_entities`) |
| `tokens[].via`, `tokener`, `tokenType` | a company's station (P5, P10), a destination token (ICR) |
| `acquire` | `[{company, name, value, gives: "5-train"/"2P"/"LP"/"pullman"/"credits"/"discount"..., action}]`, `othersLeftOut` |
| `routeChoices` | `{kind: "pullman"/"grain", options: [{train, action}], none?: action}` |
| `extraDividend` | `{train, revenue, options: [{kind, perShare, action}]}` |
| `acquireMinor` | `{phase: "select"/"pay"/"token", candidates: [{minor, owner, price, action}], othersLeftOut, pay: [{choice, cash, shares, action}], token: [{choice, hex?, action}]}` |
| `sawmill` | `{hexes: [...], then: [{choice: "open"/"closed", action}]}` |
| `issue` | `[{corp, percent, price, priceAfter, action}]` |

Already covered by RawMenu as is: `buy`, `sell`, `turn`, `blocks` (two new codes), `par` (with `via`), `tileHexes`,
`tileDear`, `tilePower`, tile options, `tokens`, `routes`, `dividend`, `trains`, `trainBuy`, `discard`, `abilities`
(company lays), `redeem`, `pass`, `programs`, `unsupported`.

### 3.4 Own companies only, confirm-first

Menus come from the generator under the title's `CrossPlayerPolicy::RESTRICTED` (= engine-rs's 1822 rule since
2026-10-17, the same as the site's): no train from another player's company, no private acquired from another player,
no minor of another player's; bank-owned lots (bid boxes, phase-5 bid-box minors) stay. The generator falls back to
Ruby's set when its own would be empty (`CROSS_FALLBACK`, never hit in fuzz): **the site layer must not** — it filters
with the same predicate itself and says why when nothing is left. `act` / `check` accept Ruby's whole set (an Opal page
or a bot may post a cross-player move; the engine processes it as Ruby). Every new hand asks before it posts
(confirm-first), as today's.

### 3.5 Automatic actions and programs

- **Step auto actions** (no program): `G1822::Step::DestinationToken#auto_actions` (the token if connected, else pass;
  CA: none when ICR reaches several Québec cities), `G1822::Step::MinorAcquisition#auto_actions` (pass when nothing is
  mergeable), `G1822::Step::Choose#auto_actions` (P9 pass when doubling can't change the order; Ruby's
  `players.map(&:cash).sort[1]`, an ascending sort). The probe's predictor (the blocking step's own moves from the
  generator: DestinationToken with one move → that move; MinorAcquisition with only `pass` → pass; Choose → Ruby's
  cash rule) equals Ruby's at every position of every fixture but the 55 old records (§2.2). A predicts them in
  `src/site/g1822/autos.rs` and posts them as 1867's.
- **Programs**: step 3a's Rust Programmer extended to G1822's stock round (programmed buy shares / share pass with the
  bid boxes: Ruby's activation and `program_disable` rules). The fixtures hold 319 recorded program auto actions as the
  test set.

---

## 4. Title pack, shadow key, site_rules

### 4.1 Title packs (C, contract K10)

`title-1822.<sha12>.json` and `title-1822CA.<sha12>.json` (`scripts/rust/titlepack.cjs --title`), from the site's Opal
bundles: StaticInfo by player count (1822: cert limit 33/26/20/16/13/11, cash 900/700/525/420/350/300 for 2-7; CA
40/26/20/16/13/11, 1000/700/...) and by rules (1822+: minors 25-30, P19-P21, the three category boxes, No Removals,
Single Stack); the corporations (30 minors `kind: "minor"`, 10 majors with destinations and exchange-token counts; CA's
majors incl. QMOO, ICR, GT, CPR, CNoR, GNWR, GTP, NTR, PGE, GWR, the DD blocker); the companies (minors' and concessions'
certificates, privates with category, value, revenue, ability text); 1822's sparse 2-D market (`par_1` / `par` cells,
`MARKET_TEXT`), CA's one-row market; the maps (1822: 211 hexes with London, the English Channel / France, estuary
borders and their costs, D35's future S; CA: 279 hexes, coordinates AA..AP and a6/a8, the five big cities, mountain
passes, grain elevators and ports, the sawmill's icons); every tile drawing in Ruby's supply order (1822's 62 incl.
London's and the BC / X tiles; CA's big-city tiles); trains (L / 2, 3-7, E, 2P, LP, P+, 5P, P, G) and phases with
their status words (`minor_float_phase*`, `can_convert_concessions`, `can_buy_trains`, `can_acquire_minor_bidbox`,
`can_par`, `full_capitalisation`, `minors_green_upgrade`); events (`close_concessions`, `full_capitalisation`,
`phase_revenue`, CA `open_detroit_duluth`). Size by extrapolation from 1830's 143 KB and 1867's 209 KB: ~350-500 KB raw
each. A's test checks the facts both carry (tiles, phases, trains, market, entities) against engine-rs's title data.

### 4.2 The 1822 site layer (A, `src/site/g1822/*`)

- **Load** (`SiteGame::load`): titles `1822` (2-7, the four rules) and `1822CA` (2-7, none), `BaseGame::new_titled`
  with Ruby's seed, `site_rules = true`, records on. ERS / WRS: `unsupported:`.
- **Seat ids**: every site player id `u` becomes seat `u + 1000` (engine-rs refuses ids equal to a corporation sym:
  minors "1".."30"; :9301's bots are users 2, 3, 4, 11, 12) and back at every boundary: actions' `entity` /
  `user` for players, RawMenu actors and actions, snapshots, `acting`, the key's `players[].id`, company owners
  `"p<id>"`, `result`'s keys, the posted action. 1830 and 1867 keep the identity map (their goldens must stay
  byte-identical).
- **Actions** (`json.rs`): `bid` (with `company`), `choose`, `choose_ability`, `acquire_company`, `merge`, `hex_token`,
  `payoff_player_debt(_partial)`, `assign`, `lay_tile` with `combo_entities`, `buy_shares` / `sell_shares` with
  `share_price` (the 1822 `pending_combo` / `pending_share_price` the Python path sets today), `place_token` with
  `tokener` / `token_type`. City refs are Ruby's (`<tile id>-<city>`; preprinted `<hex>-<index>`): no swap.
- **Replay** as Ruby's `load` (the 1822 step machine already follows `@game.loading`); recorded auto actions after
  their action; chat's auto actions too (step 2's fix).
- **Act / check**: the move must match a move of `g1822_legal_moves_with(RULES)` on the importer's key (RL
  `generic_import_1822`'s matching: type, entity, company / corporation, bundle, hex / tile / rotation, city, train
  type and seller, choice, kind; ranged moves on their range and step); then processed with the predicted auto
  actions. `run_routes`: the router's run only (step 1's rule).
- **Records**: 1822's own step machine never reaches H1-H5. The site layer records them by watching each processed
  action: an operating-history entry when `G1822State::operating_history[corp]` grows (revenue from the run, kind from
  the dividend or the minors' automatic one), player values when the round changes (`g1822_player_values`), the end.
  Fallback only if a case can't be seen: one H-hook in `g1822/operating.rs` (listed first, its own commit).
- **View / StaticInfo**: Ruby's corporation list and order (minors appear when floated, majors when parred; closed
  ones leave), bid boxes and stacks, debt, the tax haven, the privates' credits, exchange tokens, destinations,
  reservations, CA's sawmill and DD blocker; the round names ("Stock", "Choose", "Operating"), step labels (Ruby's
  descriptions: "Bid, convert concession or buy/sell shares", "Acquire a minor", "Place the Destination Token", ...).
- **Menus** (§3.3), **tile options** (the generator's lays per hex, Ruby's tile instance — `tile_supply_tail` keeps
  Ruby's supply order), **router** (`g1822_best_routes`: routes as site JSON with Ruby's `hexes`, `connections`,
  `nodes`, the subsidy), **log lines** (bids, the end-of-round resolution, floats, conversions, acquisitions, debt and
  interest, events, mail, destinations; target as step 2: every step's beats, >= 99.5% of Opal's lines), **details**.

### 4.3 The shadow key: fields 1822 adds

v2's rule applies (fields only where the title has them; `v` unchanged unless step 3a moved it; A and B deploy
together). Ruby side in `lib/engine_shadow.rb` (B), Rust in `src/site/g1822/key.rs` (A); every field with its Ruby
method:

| Field | Ruby (B) | Rust (A) |
|---|---|---|
| `bidboxes` `{minors, concessions, privates}` | `engine.bidbox_minors` / `bidbox_concessions` / `bidbox_privates` ids | `bidbox_minors()` / `_concessions()` / `_privates()` |
| `stacks` | `engine.companies.map(&:id)` (the stacks' order; P17 moves cards) | `G1822State::company_order` |
| `auction` (v2's field) | `{lot: null, bids: [[company, player, price], ...]}` from the stock round's `@bids` (kept on `round.bids` across turns), sorted | the round's bids |
| `debt` | `{player id => player.debt}` (non-zero) | `player_debt` |
| `tax_haven` | `engine.tax_haven` (a dummy `Player(-1, 'P16')`): `{cash, shares: {corp => percent}}` | `tax_haven_cash`, the haven's shares |
| `credits` | P15 / P20 (CA P8 / P9) charters' credits (Ruby's dummy players' cash) | `phase_revenue` |
| `exchange_tokens` | `{major => engine.exchange_tokens(major)}` | `exchange_tokens` |
| `reservations` | `{"hex-city" => reserved ids}` for cities with a reservation, plus London's tile reservation | `city_reservations`, `tile_reservations` |
| `companies` (v1's field) | also `"bank"` for bid-box lots (`owner == @bank`) | `bank_owned` |
| `corps[sym].type` | not needed: minors are numeric syms | |
| CA `sawmill` | `{hex, kind}` of the assigned sawmill | the CA state |
| CA `blockers` | whether Detroit-Duluth is still blocked (DD's tokens) | the CA state |
| `step` / `round` | `step.class.name.split('::').last` (BuySellParShares, Choose, DiscardTrain, PendingToken, FirstTurnHousekeeping, AcquireCompany, SpecialChoose, SpecialTrack, SpecialToken, Track, DestinationToken, Token, Route, Dividend, BuyTrain, MinorAcquisition, IssueShares, CA AssignSawmill, AcquisitionTrack); `round.name` ("Stock", "Choose", "Operating") | `g1822::Step` names are Ruby's; `RoundKind::Choices` → "Choose" |
| `last_action` for a run | v1's (revenues, `extra_revenue`, `subsidy`: the mail contracts) | the stored routes priced by engine-rs |

MRI and 1822CA: MRI raises on every CA route it validates (`G1822CA#check_distance(route, visits, train = nil)` calls
a two-argument `super`; Opal ignores arity), so 1822CA is not in `RunRevenue::VERIFIED` and Ruby's reversed check
can't price a CA run. Either the check loads RL's MRI shim (`scripts/ruby_shims.rb`: the same call with two arguments,
in the checker, not in `lib/engine`) or it compares CA runs by their posted numbers (§6.4 D3). 1822 (base) is not
`VERIFIED` either: its fixtures' few refused runs are the old records of §2.3.

### 4.4 site_rules and the edits outside `src/site`

**No `site_rules` branch.** S1-S6 and H1-H5 are on the shared 1830/1867 paths, which the 1822 family never reaches
(`BaseGame::g1822` routes every action to `crate::g1822`); 1822's processing already equals Ruby's (cross-player
purchases included: the generator's policy is a move filter only); the house rule is the menus' (§3.4); Ruby's
emergency rules (`EBUY_FROM_OTHERS :never`, debt after selling) are the engine's own. Nothing Ruby allows that the
site wants is missing from engine-rs's processing. Edits outside `src/site` (each its own commit, listed to the
training session before it lands; the 1830, 1867 and 1822 neutrality pins unchanged at every commit):

| # | File, item | Edit |
|---|---|---|
| P4 | `title/mod.rs` `all_titles`, the `site-titles` arm | `[G1830, G1867, G1822, G1822Ca]` (cfg only; the default and test registries unchanged) |
| P5 | `g1822/legal.rs` (`g1822_validate`, `gen_corp_lays`, the restriction walk), `g1822/router.rs` (the walk and combination timings), `g1822/prof.rs` (`start` / `stop`) | `std::time::Instant` → `crate::clock::Instant` (9 sites): the same type on native (a re-export), a zero reading on wasm32. The readings feed only the stats counters and `G1822_ROUTER_DEBUG`. |
| V3 | `g1822/*` readers the site layer needs (`g1822_active_step`, `g1822_step_current_entity`, `g1822_action_types`, the bid-box and bid readers, `g1822_acting_player`, the router's route data) | `fn` → `pub(crate) fn` where still private, bodies unchanged (listed exactly at build time) |
| (H6) | `g1822/operating.rs` | only if §4.2's records can't be observed: `if self.site_rules { self.site_note_operated(..) }` |

**Training neutrality**: step 3a pins an 1822 neutrality hash before any change outside `src/site` (its workflow's
instruction, because the 1822 PG run is training on engine-rs). Step 4 runs that pin at every commit; if 3a's pin
covers only 1822, A0 adds 1822CA walks (both policies, 3-6 players) as its own first commit, computed before any
change. P5 changes no reading the generator or router decides on, so the pins must hold.

### 4.5 The training session's note: `site_interplayer_purchase_open` reads the phase data

Done in step 2 (`src/site/rules.rs`: `ruby_phase_status()` = the title's `PhaseDef.status` ∪ the
`RUBY_SITE_PHASE_STATUS` table; 1830's rows only). For 1822: engine-rs's 1822 and 1822CA phase statuses are Ruby's
verbatim (checked: `minor_float_phase1`, `can_convert_concessions`, `can_buy_trains`, `can_acquire_minor_bidbox`,
`can_par`, `full_capitalisation`, `minors_green_upgrade`, ...) and none has `can_buy_companies_from_other_players`, so
**no table rows**; and the S3/S4 call sites never run for the family anyway. A adds a test: for every phase of 1822 and
1822CA, `ruby_phase_status()` equals the pack's (Ruby's) statuses and `site_interplayer_purchase_open` is false.

---

## 5. Bots

### 5.1 The site side (B, when the RL session says the driver is ready; dev first, production on Caleb's OK)

| Where | Change |
|---|---|
| `lib/bots.rb` | `OPTIONAL_RULES['1822'] = %i[plus_expansion plus_expansion_no_removals plus_expansion_single_stack tax_haven_multiple]`; `PLAYER_RANGE['1822'] = [3, 6]`; `ANALYSIS_TITLES` unchanged (1830) |
| `.env` / `next-prod.env` | `BOT_TITLES=1867,1822` (dev first) |
| Client (C) | `isBotTitle` today mixes three meanings (1830's beats planner, the Analysis link, the par fallback): split them; bot titles and bot rules from `GET /api/titles` (`bot_titles`, each rule's `bots`) instead of the static `BOT_TITLES` / `BOT_RULES`; Analysis links only for `ANALYSIS_TITLES` |
| 1822CA | no bots: a human-only title (no model) |
| Bot API | Under step 3a the views come from engine-rs; for 1822 the contract adds nothing new: `view.step` = Ruby's class names (§4.3), `view.entity.type` `corporation` for minors (Ruby's minors are corporations), `view.acting` in site ids (the seat map), `view.trains` with Ruby's train ids. `POST /api/bot/engine` (18xx.games) stays 1830. |

### 5.2 What the RL bot driver needs (questions for the training session, relayed by the parent)

1. **Scope of the model**: 1822 only (3-6 players, the ten rule combinations) — confirm; 1822CA not planned? The
   checkpoint and temperature for `--title-policy 1822=<ckpt>@T`, policy-only or searched, and when (the run's plateau).
2. **`titled.py` for 1822**: `TITLES["1822"] = (3, 6)`; following a site game in engine-rs (`ruby_filtered_actions`,
   `settings.seed`, `settings.optional_rules` incl. the no-op combinations, the encoder v3 rule flags); seat mapping
   (site ids collide with minors 1-30: the importer's 1001+ seating, mapped back in `to_site_action`); who acts from
   `g1822_acting_player` / the step names (step 3a's `bot_view` will be engine-rs's: same names).
3. **Auto actions**: the bot posts only its move; the server adds DestinationToken / MinorAcquisition / P9 autos (and
   programs). The driver must expect them after its move (no separate pass from the bot).
4. **Ruby dead ends** (the driver's own avoidance, like 1867's): the tax-haven presidency-distance crash; a broke
   major owing minor 14's London token; anything else their fuzz met. Under Rust authority these are engine-rs's
   "(Ruby raises)" refusals: should the site keep them (stuck game) or follow printed rules (as MX's NdeM)? Caleb's.
5. **Replay-only moves humans may make** that the generator doesn't offer (an out-of-turn P16 choice, out-of-order
   discards, a raise of one's own bid): the follower must accept them from records (it replays, so yes); the bots
   never make them.
6. **Prices**: bids on the £5 grid within [min, available]; trains £1..cash; partial debt £1..; their sampling cells.
7. **The engine pin**: the site pins its branch; the driver runs RL master. Any g1822 rules commit on master must reach
   the site's pin before 1822 bot games run (or both diverge on who acts).
8. **Speed**: the generator's tails (1822 max ~0.4 s natively) in a live bot's per-move budget; CA not needed.
9. **Cross-player policy**: the bots under `RESTRICTED` (own-only) — the site's rule; confirm nothing in the 1822
   driver falls back to `RULES`.

---

## 6. Plan

### 6.1 Packages (after step 3a lands; contracts K1-K11 as in steps 1-2, deltas below)

**A — engine** (new RL worktree, branch `site/engine-rs-1822` from step 3a's head). 12-16 h.

| Step | Work |
|---|---|
| A0 | Rebase on 3a; the 1822 / 1822CA neutrality pins (3a's, plus CA if missing); the Ruby diff 46a592e → 24e59e1 for g_1822 / g_1822_ca; P4, P5 (own commits) |
| A1 | Load / seat map / replay / records for both titles; `site_engine state / keys` with the 1822 key fields; publish a package (`$RUST4/pkg`) |
| A2 | Act / check on the generator; the step auto actions (§3.5); programs on 3a's Programmer for G1822's stock round; generated games (engine-rs walks as site JSON, both policies, 2-7 players, rule combinations) |
| A3 | View, StaticInfo's engine part, RawMenu (shared facts + §3.3's new ones), tile options, router, details |
| A4 | Log lines |
| A5 | Against C's Opal goldens; act cases (K11) judged by B's Ruby; native = wasm; the §4.5 test; the gate |

**B — server** (rl18xx-next worktree, branch `next/rust-1822`). 3-5 h. Key fields in `lib/engine_shadow.rb` + specs on
the 1822 fixtures; `ruby_keys.rb` / `ruby_verdicts.rb` for 1822 (the MRI shim decision, D3); the title switches
(`RUST_ENGINE_TITLES`, the authority switch of 3a, `ENGINE_SHADOW_TITLES`) for 1822 and 1822CA on dev; `lib/bots.rb`
(§5.1, inert until `BOT_TITLES`); vendoring the pin; DEPLOY.md / ENGINE_RS.md.

**C — client** (living-board worktree, branch `rust-1822`). 10-16 h. Packs (§4.1); `rawmenu.ts` / `words.ts` for the new
facts; **new hands** built on Rust facts: the bid boxes panel (three boxes, each lot's high bid and yours, a stepper on
the £5 grid, tokens left, "Pass (Bids)"), concession conversion in the par question, acquisitions (privates, then the
minor flow: pick, pay, token), ability choices (P17's card picker, P18 / P5 / P28 swaps), pullman / grain attach before
the run, the extra train's dividend, the tax haven, debt, CA's sawmill, company stations (`tokener`), combos; the
worker (pack by title, seat ids are the server's); the bot-title split (§5.1); `test/rust/*` for 1822.
Recommendation (D2): since Opal goes in step 3b, the new 1822 hands read Rust facts only, and the Opal path keeps its
plain list; that saves a third of C.

**V — verification** (4-6 h): the site's fixtures + generated games (>= 300, 2-7 players, 1822's rule combinations,
both policies) judged move by move by the site's Ruby (throwaway container, the CA shim); the RL scrape copy if granted;
key = Ruby's at every position; every offered option acted in Rust and judged by Ruby (K11), none accepted by one and
refused by the other except the replay-only classes; Rust vs Opal on the shared facts; wasm memory and timings on the
densest CA positions; live on a scratch stack as test users (all decisions of §3.2 at least once each, both titles),
then :9301 with bots once the driver is ready. **Fix**: 2-4 h.

Total ~30-45 agent-hours; ~1.5-2 days wall-clock (A0-A1 first, then A2-A5 / B / C in parallel, then V and the fix).

### 6.2 The ladder

No stored 1822 game exists on either site, so the stored-game risk of steps 1-3 is absent: on dev, 1822 can start
directly under 3a's authority path (Rust decides, Ruby as the reversed check), Rust the browser default, Opal the
fallback (with its plain list). Production: the same switches after V and Caleb's OK; bots after that, on Caleb's OK
and the RL session's confirmation that the deployed driver plays 1822.

### 6.3 Risks

- **R1 Size.** ~30 decision kinds and ~50 private abilities across two titles; a dozen new facts and hands. The
  generator already decides legality (completeness >= 99.985%), so the risk is in words and UI, not rules.
- **R2 Dense maps.** Natively the RL corpus has 1822CA generator positions up to 5.5 s and router calls 5-9 s; 2-3× in
  wasm, and the shadow / authority path's latency budget (~100 ms typical) is exceeded there. Mitigation: the router
  and menus in the worker with the "Reading the rules…" state; the server caches the game between moves (3a's design);
  a hard budget with an honest message; the corp-graph work of 3a doesn't cover g1822 (its own graph is already memoized).
- **R3 Replay cost.** A whole 1822 game replays in ~0.15 s natively (5× 1867's); a per-move full replay on the server is
  avoidable with 3a's cached authority; wasm checkpoints (every 100 actions) hold larger states: measure memory (step
  1's ~100 MB in long 1830 games is the reference).
- **R4 The router as the only accepted run.** None of the site's stored runs beats the router; the RL scrape has one
  real CA shortfall (257158 @1556, $10). If a player ever finds a better run, the site can't take it. A hand-drawn-run
  path would need a Ruby-equivalent validator (engine-rs's pricing helper skips the token check, §2.3 G2).
- **R5 Ruby's dead ends** (tax haven, minor 14's London token): both engines raise; a live game would stop there.
- **R6 MRI.** 1822CA's arity bug: the reversed check can't price CA runs without the shim (D3).
- **R7 wasm size.** +99 KB gzip measured for the engine parts, more with the site layer (~850-900 KB gzip total).
  Acceptable; per-title packages are possible later if it matters.
- **R8 Reference drift.** The site's Ruby is 3 weeks newer than engine-rs's reference (A0's diff); upstream changes to
  g_1822 after the fork aren't merged (the site is not tracking upstream).
- **R9 2 and 7 players** are validated only by replay of two 7p fixtures; generated games at 2 and 7 before the switch.

### 6.4 Decisions for Caleb

- **D1** ERS / WRS: wanted? (Not in engine-rs; 6-10 h of RL-side work first.)
- **D2** The new 1822 hands on the Rust path only (Opal keeps the plain list until 3b)?
- **D3** Ruby's reversed check for 1822CA runs: load the MRI arity shim in the checker, or compare posted numbers?
- **D4** 1822 at 2 and 7 players for humans (Ruby's range) — yes once generated games pass?
- **D5** 1822CA without bots: offered as a human-only title (no "experimental" setting needed once on Rust)?
- **D6** The Ruby dead ends (R5): keep Ruby's behaviour or follow the printed rules (as the NdeM)?

---

## 7. How this probe was made (scratch `$RUST4`, nothing committed but this doc)

- Production: `fetch_prod.py` (the bot API's read-only export, ids 1-35, 80 misses to stop, 0.25 s apart): 31 games
  read, titles only counted, none of the 1822 family, so nothing kept; the empty directory deleted at the end.
- Dev: one read-only `psql` query (`select title, status, count(*) ... like '1822%'`), `GET /api/game/1001`.
- Fixtures: copied from rl18xx-next `public/fixtures/{1822,1822CA,1822CAERS,1822CAWRS}`.
- engine-rs: the step-2 worktree's venv (`site-engine-1867/.venv`, built 2026-11-27 from the same engine code as this
  branch; `PYTHONDONTWRITEBYTECODE=1`), scripts run from this worktree under `taskset -c 24-31,56-63 nice -n 19`,
  `CUDA_VISIBLE_DEVICES=`: `tests/corpus_audit.py`, `scripts/g1822_completeness.py`, `scripts/g1822_router_check.py`;
  `probe/autos2.py` (step auto actions), `probe/timing.py`, `probe/census.py` (generated move shapes per step),
  `probe/below.py` (engine pricing of the human runs).
- The site's Ruby: `ruby/price.rb` in `docker run --rm --network none` of the production image's Ruby (`18xx-next-prod:
  128467c8ff2b`, its own `/18xx/lib`), fixtures mounted read-only, with the RL rig's CA arity shim.
- The Opal hand: living-board `test/titles_engine.cjs --base http://127.0.0.1:9301` (LIST_ONLY=1, `TMPDIR=$RUST4/tmp`).
- wasm: `git archive` of b4b2391's `engine-rs` twice into `$RUST4/wasm`, one patched (P4 + P5 + a scratch export
  linking the generator and router), both built as `site/build-site.sh` does.

---

## 8. Package A as built (2026-11-28)

Branch `site/engine-rs-1822` (RL_18xx worktree `.claude/worktrees/site-engine-1822`), on step 3a's rebased head
fa11715. Scratch: `$RUST4/A` (builds, generated games, reports), packages `$RUST4/pkg/<commit>` (`current` the last).
Every build, test and sweep under `taskset -c 24-31,56-63 nice -n 19`, no GPU.

### 8.1 Commits

| Commit | What |
|---|---|
| fbad02a | **P5** (outside `src/site`): the 1822 generator's and router's timer readings through `crate::clock` |
| f1ad358 | **P4** (outside `src/site`): the `site-titles` registry adds 1822 and 1822CA |
| ab767fa | **V3** (outside `src/site`): `g1822_routes_subsidy` visible to the crate |
| 2246712 | the 1822 site layer, first cut (`src/site/g1822/*`, the shared entry points' dispatch, the site fixtures, tests) |
| e875f02 | the log lines; generated games (`site_replay --generate-1822`) |
| ea434f2 | key and static fixes (par of always_market_price companies, laid tiles only, 1822CA's DD blocker, the destination token in tokenCount, a minor's per-share payout, the dead end's words) |
| b455965 | the RESTRICTED-vs-menu check as a module (`g1822/check.rs`) and `site_replay --restricted-check`; the liquidity column by seat |
| e61aeb0 | the menus say why when only cross-player deals are left; the check lists the generator's fallbacks |
| 11e3ca9 | a manual end's player values in the site's ids |
| (this) | this record (§8) |

### 8.2 Every edit outside `src/site`, function by function (for the training session)

| # | Commit | File | Function(s) | Edit |
|---|---|---|---|---|
| P5 | fbad02a | `engine-rs/src/g1822/legal.rs` | `g1822_validate`, `gen_corp_lays`, `lay_passes_restrictions` (the restriction walk's wrapper) | `std::time::Instant::now()` → `crate::clock::Instant::now()` (3 sites). The readings feed `stats::VALIDATE_NS`, `LAYS_NS`, `RESTRICT_NS` only |
| P5 | fbad02a | `engine-rs/src/g1822/router.rs` | `g1822_best_routes_v2` (the `walk` closure's `t_walk`; the scout and full combines' `t0`) | the same (3 sites); the readings feed `G1822_ROUTER_DEBUG`'s eprintln only |
| P5 | fbad02a | `engine-rs/src/g1822/prof.rs` | `start`, `stop` | the type `Option<std::time::Instant>` → `Option<crate::clock::Instant>` (3 sites); profile counters only |
| P4 | f1ad358 | `engine-rs/src/title/mod.rs` | `all_titles` | the `#[cfg(all(not(test), feature = "site-titles"))]` arm: `[G1830, G1867]` → `[G1830, G1867, G1822, G1822Ca]` (the default and test registries unchanged) |
| V3 | ab767fa | `engine-rs/src/g1822/operating.rs` | `g1822_routes_subsidy` | `fn` → `pub(crate) fn`, body unchanged (the stored run's mail-contract `subsidy`, as Ruby's RunRoutes stores it) |

No H6 record hook was needed (§4.4): the operating history and the players' values are observed by the site layer.
Site tooling only, not engine code: `engine-rs/src/bin/site_replay.rs` (the `site` feature's own binary) gains
`--generate-1822` and `--restricted-check`. Nothing touches policy layouts, `mcts*.rs`, the encoder, `action_index.rs`,
`src/rl`, Python or Ruby. `crate::clock` on native is a re-export of `std::time::Instant`, so P5 is the same code there.

**Neutrality**: the three pins of `engine-rs/src/neutrality.rs` (1830, 1867, 1822) pass unchanged after P5 (site build)
and at the head in both builds (§8.6). The 1822CA generator and router, which P5 also touches and no pin covered, have
their own pin inside `src/site` (`g1822::tests::ca_walks_are_pinned`: four fixed-seed walks, 3-6 players, both
policies, 700 moves each): its digests were computed on the three g1822 files of fbad02a's parent (before P5) and are
equal after it (0xf3992ff745f9075c, 0xc645efc85ef47ac4, 0x160839b73374634a, 0x486dd74f0ab54232).

### 8.3 The Ruby diff (engine-rs's reference 46a592e vs the site's 24e59e1 and the fork)

`diff -r` of the clone at 46a592e (the training session's rig) against rl18xx-next's `lib/engine` (24e59e1 plus the
fork's commits): **no difference in `lib/engine/game/g_1822/` or `g_1822_ca/`**. The shared engine files that differ, and
what each means for the family (saved as `$RUST4/A/ruby_diff_shared.txt`):

| File | Change | 1822 / 1822CA |
|---|---|---|
| `game/base.rb` `revenue_str` | `route.hexes` → `route.visited_hexes` (the fork: a stop visited twice shows twice) | **display**: G1822 / G1822CA call `super`, so a run's string follows the connections' nodes; the site layer writes it that way (`act.rs visited`). Old records keep their strings (50 of the fixtures' accepted runs: order only) |
| `action/run_routes.rb` | a stored route with fewer hexes than nodes takes its hexes from the track (the fork) | replay of stored records only; the site layer uses the router's hexes for such a route |
| `route.rb` | `check_double_towns!`, `visited_hexes`, a nil guard in `touch_node` | no double towns on either map: no effect |
| `game/base.rb` `upgrades_to?` | double towns' count must match | none on either map: no effect |
| `game/base.rb` neighbours | a gray/red/blue neighbour with a stub back to the hex stays a neighbour | 1822's stubs are on white hexes (L37, L39, M36, M40, N37, N39), 1822CA has none: no effect |
| `action/choose.rb` | optional `target` / `target_type` | the family's choices name no target: no effect |
| `hex.rb`, `tile.rb`, `part/*.rb` | patronage, `double_town`, label `loc`, path `color` | features no 1822 map uses: no effect |
| `step/waterfall_auction.rb`, `auto_router.rb`, `logger.rb` | the auction lock, MRI's JS guard, log levels | not reached by the family |

**No rules difference**: nothing to report beyond the revenue string, which the site layer follows.

### 8.4 What the site layer does (`src/site/g1822/*`)

| Module | |
|---|---|
| `mod.rs` | titles `1822` (2-7, its four rules, any combination) and `1822CA` (2-7, none); Ruby's seed; **seat ids** (`SiteGame::seat_offset` 1000: every player id in, every id out, at every boundary; 1830 / 1867 keep 0); the active step and entity, Ruby's step class names, `active_players` (DiscardTrain: every crowded company's president), `ruby_actions`, Ruby's corporation list (1822CA's DD blocker left out), the bot view |
| `apply.rs` | one engine action: `combo_entities` and a bundle's `share_price` handed to the engine as the Python bridge does; the records (operating history with the run's revenue and Ruby's kind, a minor's automatic `half` / `withhold`; players' values at each round's end and at the end); the turn in progress; the Programmer's `players_history` (bids included) and G1822's stored winning bids; programs disabled at a round change (`check_programmed_actions`); companies' order, last runs and lays; the log |
| `act.rs` | the authority's check: the posted move must match a move of the generator's **Ruby set** (`RULES`; an Opal page or a bot may post a cross-player deal) on the importer's key, a ranged move by range and step; the stored action is the generator's wire (site ids) with the posted price. **A run must be the router's** (the same trains over the same stops, in any order and direction), stored as posted with Ruby's fields (each route's revenue by the engine's Ruby pricer in the posted order, `revenue_str` with the destination bonus, 1822CA's sawmill and grain / port parts, `extra_revenue` 0, the mail contracts' `subsidy`). The dead ends' words (`dead_end`) |
| `autos.rs` | Ruby's step auto actions (DestinationToken, MinorAcquisition, P9 / CA P7) and the Programmer on G1822's stock round (`program_share_pass`, `program_buy_shares`, the shenanigan reasons incl. "No longer winning bid on X") |
| `menu.rs` | RawMenu (§8.5) |
| `view.rs` | the Snapshot (shared fields, players in Ruby's `@players` order) + `g1822`; StaticInfo's engine part; round info, Ruby's step descriptions and pass descriptions |
| `key.rs` | the shadow key: v2 + the family's fields (§4.3; the Ruby expression of each in the module doc) |
| `log.rs` | Ruby's templates for the main beats (§8.7 gaps) |
| `generate.rs`, `check.rs` | generated games; the RESTRICTED-vs-menu check |

`site_rules`: no branch (§4.4). Shared entry points dispatch by `g1822.is_some()`: `SiteGame::load / apply / apply_engine /
result / end_reason`, `view::snapshot / static_info / step_json`, `menu::menu / tile_options / auto_route / legal_actions`,
`shadow::key`, `authority::check / entity_player / ruby_active_players / bot_view / finished_acting`, `details::meta /
details / program_info / liquidity`, `programs::auto_actions / stock_turn / owner / sale_timing / shenanigan / observe`,
`json::action_from_json` (+ `choose_ability`, `acquire_company`, `hex_token`, `payoff_player_debt(_partial)`, `assign`).
1830 and 1867 keep their paths (every step-1/2/3a suite passes unchanged).

### 8.5 API additions (contracts K1'/K3'/K4/K10/K12 for B and C)

- **Version**: `titles` = `["1830","1867","1822","1822CA"]`; api 3 and key 2 unchanged; WasmGame's methods unchanged.
- **Snapshot**: `players[].debt`; `corps[].type` (`minor` / `major`); `companies[]` in Ruby's `@companies` order with
  `bank` (a bid-box lot); `hexes[h].tileRes` (London's tile reservation); `round.name` "Stock Round" / "Choose" /
  "Operating Round"; and **`g1822`**: `bidBoxes {minors, concessions, privates}`, `bids [{company, player, price}]`,
  `stacks`, `bankOwned`, `biddingTokens {player: left}`, `biddingTokensTotal`, `bidsPerTurn`, `bidStep`, `committed
  {player: cash}`, `debt {player: owed}`, `taxHaven {company, owner, cash, shares}`, `credits {company: cash}`,
  `exchangeTokens`, `destinations {major: {hex, placed}}`, `plus`, `currency`; 1822CA `sawmill {hex, owner, bonus}`.
- **StaticInfo**: corporations' `type`, `destination`, `exchangeTokens`, `tokenCount` incl. the destination token;
  companies' `kind` (minor / concession / private) and 1822+ `category` (train / track / other); trains' `multiplier`,
  `events`, `unlimited`; phases' `minorTrainLimit`, `operatingRounds`, `status`, `on`; `layout` flat; `currency`
  £%s / $%s; `optionalRules`; `biddingTokens`.
- **RawMenu** (every move also in `moves: [{type, action, range: {min, max, step} | null}]`; `step` the step's class):
  `bidBoxes {boxes: [{kind, category?, lots: [{sym, kind, name, value, minBid, maxBid, high {player, name, price} | null,
  mine, canBid, why (high / debt / turn / tokens / cash / certs / other), action, range}]}], tokens {left, total},
  committed, cash, available, bidsThisTurn, perTurn, step}`; `par [{corp, via (the concession), exchangeValue, cash,
  floatPercent, presPercent, sharePercent, prices [{price, action}]}]`; `buy`, `sell` (with the sale's preview), `turn`
  (+ `certsWithBids`, `bids`); `taxHaven {company, name, spender owner / haven, cash, multiple, options [{corp, from,
  price, choice, action}]}`; `debt {amount, interest, payoff {action}, partial {min, max, step, action}}`; `choices
  {context double_cash / special, company, companyName, options [{choice, action}]}`; `abilityChoices [{company, name,
  kind lur / chpr / lcdr / other, options [{choice, card?, where?, action}]}]`; `acquire [{company, name, value, owner,
  action}]`; `othersLeftOut [{kind train / private / minor, sym, owner, ownerName}]`; `tileHexes`, `tilePower`,
  `abilities [{company, name, owner, hexes, combos, ...}]`; `tokens [{hex, city, cities, price, action, via?, tokener?,
  tokenType?}]`; `hexToken [{hex, tokenType, action}]`; `routes {pending}`; `routeChoices {kind pullman / grain,
  options [{choice, train {id, name}, action}]}`; `dividend [{kind, action, preview}]`; `extraDividend {train, revenue,
  options}`; `trains`, `trainBuy` (+ `exchanges` the L→2 upgrades, `others` with `range`, `debt {afterSell, owed,
  interest}`); `discard`; `acquireMinor {phase select / pay / token, selected, candidates [{minor, name, owner, bank,
  action}], pay, token}`; `sawmill {hexes [{hex, company, action}], then [{choice, action}]}`; `issue [{corp, percent,
  price, priceAfter, action}]`; `redeem`; `deadEnd` (no move left: the engine's "(Ruby raises)" words).
- **tileOptions(hex, via)**: the generator's lays on the hex (Ruby's tile instance), each `{tile, name, rotation, cost,
  combo, action}`. **autoRoute()**: `{action (the run as `act` stores it), revenue, subsidy, routes (+track)}`.
- **Shadow key**: §4.3's fields: `bidboxes`, `stacks`, `auction` (stock round bids), `debt`, `tax_haven`, `credits`,
  `exchange_tokens`, `reservations` ("hex-city"), `tile_reservations`; 1822CA `sawmill {hex, owner}`, `destinated`.
- **site_engine**: the same commands (`state`, `keys`, `act`, `move`, `start`, `view`) for both titles.
- **site_replay**: `--generate-1822 N --title 1822|1822CA [--rules-policy]`, `--restricted-check <files>`.
- **Title packs** (C's to commit): `$RUST4/A/pack/title-1822.json` (1.8 MB raw, 15 KB brotli: 583 tile specs, 88
  variants) and `title-1822CA.json` (366 KB, 856 specs, 5 variants), made from :9301's Opal bundles by a scratch copy of
  `client/scripts/rust/titlepack.cjs` that loads every bundle the title needs (`$RUST4/A/pack/titlepack1822.cjs`,
  `opal1822.cjs`; 1822CA needs g_1822 before g_1822_ca; the depot's private trains matched by name).

### 8.6 Verification

| Check | Result |
|---|---|
| The 11 site fixtures (rl18xx-next `public/fixtures/{1822,1822CA}`, vendored in `engine-rs/site/fixtures/g1822/`) replayed through the site layer | 11 / 11 to the end; results 11 / 11 equal (4 bank, 2 stock market, 5 manual ends) |
| **Menus vs engine-rs's RESTRICTED set** at every position (`g1822::check`, the binding check; both directions; the generator's fallback to Ruby's set counted apart) | fixtures **9,709 / 9,709 equal**; generated games: 1822 restricted 39,676 / 39,676, 1822 rules policy 40,740 / 40,740, 1822CA restricted 20,891 / 20,891, 1822CA rules policy 22,704 / 22,707; **0 menu-only, 0 engine-only moves outside the generator's fallbacks**; every fact's action is a move, every move has a fact. **The 3 differences** (both in 1822CA rules-policy games, after cross-player trades): a minor that must buy a train where only other players' trains are within reach (gen-1822CA-5004 @964 and @1017: minor 3, trains of other players' companies at £1+; gen-1822CA-5016 @1431: minor 16) -- engine-rs's RESTRICTED set falls back to Ruby's set there (those cross-player buys), the site's menu offers nothing and says `deadEnd` "The only moves left are deals with other players' companies, which this site doesn't offer" with `othersLeftOut` (since ef124bf; before it, a raising candidate's "(Ruby raises)" words: §8.10 4). In own-only games (fixtures and restricted walks): 0 fallbacks in 70,276 positions |
| Recorded moves through the authority (`corpus_check`: each kept move decided at its position as `site_engine move` decides it) | 9,691 moves: 9,190 accepted; refused: 377 runs that aren't the router's (§8.7 1; `corpus_check` lists 60 refusals per game: 499 of the 1,004 runs) and the 2 out-of-order discards (§2.3 G3); of the 505 accepted runs 455 store exactly the record's fields (50 differ in the revenue string's stop order only: old records, §8.3). **At 1069a7e: 9,377 accepted, 312 runs refused** (§8.10) |
| Ruby's auto actions after each recorded move = Rust's | 680 / 741 groups; the 61: old records made before Ruby's step auto actions (G4: DestinationToken passes and tokens, MinorAcquisition passes, one P9 choice) and 3 program disables whose reasons name players by their pre-anonymisation names |
| Generated games through the site layer (`site_replay --generate-1822`: 2-7 players in turn, 1822's six rule combinations, moves from the menus or Ruby's whole set, posted by `act`, the written JSON replayed) | **108 games, 124,013 moves**: 1822 restricted 36 / 36 finished, rules policy 36 / 36 (3,749 cross-player moves accepted), 1822CA restricted 18 / 18, rules policy 17 / 18 + 1 stopped at the known Ruby dead end (§8.7 4); **0 generated moves refused by `act`, 108 / 108 replay to the walk's key** |
| Title packs vs engine-rs's StaticInfo (cert limit, bank, currency, layout, corporations' home / tokens / float / share percent, companies' value / revenue, market prices, trains, phases' tiles, hexes) | 1822: 960 / 960 (every player count 2-7 × every subset of the four rules); 1822CA: 60 / 60 |
| The wasm package vs native (`scripts/site_engine/pkg_smoke.cjs --native`: replay, view, the key equal to `site_engine state`'s, menus) | 11 / 11 fixtures; `act` / `check` / `menu` / `autoRoute` driven from the wasm at site ids 2, 3, 11 (minors' syms): 400 moves each title |
| The seat map (site ids 2, 3, 4) | `g1822::tests::site_ids_colliding_with_minors_are_seated` |
| §4.5: phase statuses | every phase of both titles: `ruby_phase_status()` = the title's, `site_interplayer_purchase_open` false |
| Per move, `site_engine move` (load + replay + check + autos + key; the loaded cores) | fixtures p50 61 / p95 169 / max 222 ms (412 moves); the longest generated games (~2,000 actions) p50 158 / p95 350 / max 517 ms |
| Gate (at 11e3ca9, the code head) | `cargo test --release --no-default-features --features python` 277 passed, 0 failed, 11 ignored (= 3a's); `--features site` 319 passed, 0 failed, 13 ignored (3a: 312 / 0 / 11; + the seven 1822 tests, + two ignored debugging aids); the 1830, 1867 and 1822 neutrality pins pass in both builds, unchanged; the wasm32 checks (`wasm,site-titles`, `wasm`) and the default `cargo check` clean; `uv sync` and `pytest tests/` in the worktree's own venv: 1,838 passed, 7 skipped, 2 deselected, 0 failed (30:34; = 3a's) |

### 8.7 Known gaps

1. **Runs: only the router's run** (the same trains over the same stops; any order and direction). A run drawn by hand
   or another run of the same revenue is refused with "Only the best run can be posted for X in this title on this site
   (the auto-router's, £N)" (§6.3 R4: engine-rs's pricer is not a validator). At 1069a7e 312 of the fixtures' 1,004
   recorded runs are not the router's (191 equal revenue, 117 lower, 4 where the router is lower: §2.3 G1 / G2's old
   records); a route posted without `nodes` is matched on the hexes its connections end on, and every accepted run is
   stored in a form Ruby rebuilds as the router's run (§8.10 1, 3). An Opal page (the fallback) posts Opal's own
   auto-route, which may be another equal run: refused; the router's stops on other track: stored in the router's form.
   **The bots must post the generator's RunRoutes move from the site's engine commit** (the router must be the pin's).
2. **Log lines**: Ruby's templates for the main beats only, not yet compared with Opal's (C's goldens). Missing:
   Ruby's automatic "X skips <step>" lines, home and pending token placements' own words, minor acquisitions' "X
   acquired Y receiving ...", the Choices round's and the privates' ability lines (closes, gains a train, moves a
   card), phase-revenue credits, debt interest at the stock round's end, the tax haven's buys, the end-of-round train
   export, the events' texts (concessions close, full capitalisation), "has priority deal", floats' "receives".
3. **RawMenu**: `buy[].floats` is always false; no `blocks` (each bid lot carries its `why`), no `tileDear` (the
   generator offers only affordable lays), `trainBuy.cheapest` null; a concession's `exchangeValue` is its face value
   (Ruby's exchange price is `price × multiplier − value`).
4. **Ruby's two dead ends** (the tax haven's presidency distance; a broke major owing minor 14's London token) raise
   "(Ruby raises)" in engine-rs: the move is refused with those words and the menu says `deadEnd`; a game stops there
   until the RL session implements the printed rules (Caleb's decision). 1 of the 108 generated games met one
   (1822CA, rules policy: "CPR: the presidency would be ranked against the tax haven (Ruby raises)").
5. **Server cost**: `site_engine move` replays the whole game per move (§8.6). B's engine timeout for the family must
   allow the generator's tails (RL corpus: up to 5-9 s natively on the densest 1822CA positions; 2-3× in wasm):
   ≥ 15 s suggested; a resident `site_engine serve` (STEP3 §3.6) would take a move to milliseconds.
6. **Out-of-order discards** (Ruby's view lets any crowded company discard first) are refused for new moves (§2.3 G3);
   the menu answers null for the president of a crowded company that isn't first. A discard naming another instance of
   an offered train (L-2 for the generator's L-19) is accepted and stored with the generator's instance (the page
   re-reads); on the fixtures only §2.3 G3's 2 are refused (§8.10 6).
7. **B's Ruby side of the key's family fields** does not exist yet: the field list and the Ruby expression of each are
   in `key.rs`'s module doc and §4.3. Until B has them, the keys differ in those fields only.
8. **Snapshot**: `ending`'s words are base.rb's shape for the family's triggers; `opOrder` / `waiting` empty outside
   operating rounds.
9. Not supported (`unsupported:`): 1822CA ERS / WRS (other titles), any 1822CA optional rule, more than 7 or fewer than
   2 players.
10. **A position where only cross-player deals are left** (§8.6's 3, only after cross-player trades, which the site's
    menus never make but an Opal page or a bot may): the site's menu offers nothing (`deadEnd` "The only moves left are
    deals with other players' companies, which this site doesn't offer", `othersLeftOut`; since ef124bf, §8.10 4)
    while engine-rs's RESTRICTED generator falls back to Ruby's set. A bot driven by RESTRICTED would buy another player's
    train there (the authority accepts Ruby's set); a human on a Rust page can't move until such a deal is offered
    (Caleb's own-only rule as binding; whether to offer the deal there is a decision for him).

### 8.8 For B, C and the training session

- **B**: the key's family fields in `lib/engine_shadow.rb` (§4.3 / `key.rs`); `round.name` is "Stock Round" / "Choose" /
  "Operating Round"; 1822CA runs compared without Ruby pricing (D3); the family's engine timeout (§8.7 5);
  `RUST_ENGINE_TITLES` / `ENGINE_AUTHORITY_TITLES` += 1822, 1822CA on dev; the title packs from `$RUST4/A/pack`.
- **C**: the new hands on `bidBoxes`, `par[].via`, `taxHaven`, `debt`, `choices`, `abilityChoices`, `acquire`,
  `routeChoices`, `extraDividend`, `acquireMinor`, `sawmill`, `issue`, `hexToken`, `deadEnd`, `othersLeftOut`; the plain
  list on `moves` for anything else; `Snapshot.g1822`; the packs.
- **Training session**: the edits outside `src/site` are §8.2's five (P5 ×3 files, P4, V3); the bots' driver must post
  the generator's run (§8.7 1) and expects Ruby's step autos after its move; a bid is £5 steps in [min, max], a
  corporation's train £1 steps, a partial payoff £1 steps, depot trains fixed (the generator's ranges, unchanged).

### 8.9 After package B (2026-11-28): two `src/site` commits B's Ruby side found

rl18xx-next's package B (branch `next/rust-1822`) wrote Ruby's side of the family's key fields (`lib/engine_shadow.rb`
`family_1822_fields`, §4.3) and compared it with this branch's key at every position of the 11 fixtures, then re-posted
the fixtures through the server's authoritative path with Ruby's check on. Everything it found was in `src/site/g1822`
(no edit outside `src/site`; §8.2's list is unchanged):

| Commit | What |
|---|---|
| 2030139 | the key and turn as Ruby's: an unsold minor's (unissued) shares are its IPO's (`corps.*.ipo` 100, Ruby `ipo_shares`; the engine keeps them unowned: was 0 at every position); 1822CA's P10 Winnipeg token found among the city's extra tokens (`"N16-1"`, was `"N16-?"`); DiscardTrain's acting player is the first crowded company's president only (Engine::Step::DiscardTrain#active_entities = `crowded_corps.take(1)`; was every crowded company's president) |
| c31a88c | a run with fewer hexes than stops (both cities of an OO hex: D35-D35) stored with Ruby's `Route#hexes` from the track (one per stop, as RunRoutes#to_h writes it; was the router's hex list, so Ruby's `to_h` of the stored run differed); 1822CA's predicted ICR DestinationToken PlaceToken leaves out its `slot` where Ruby's `city.get_slot(entity)` is nil (no free or reserved slot: an extra token) |

Each has a test in `g1822/tests.rs`. After both, Ruby's key equals this branch's at 9,551 of the fixtures' 9,552
positions (the one: `1822_game_end_stock_market` @1404, an old record whose stored run counted the destination bonus
twice, §2.3 G1).

**Gate at c31a88c**: `--features python` 277 passed, 0 failed, 11 ignored; `--features site` 322 passed, 0 failed,
13 ignored (+3 tests); the 1830, 1867 and 1822 neutrality pins and 1822CA's walk pin pass in both builds; the wasm32
checks (`wasm,site-titles`, `wasm`) and the default `cargo check` clean. `src/site/g1822` is behind the `site` feature,
which the wheel doesn't build, so `pytest tests/` can't differ from e6599f0's; run on 2030139 in this worktree's venv
(`uv sync --reinstall-package rl18xx`, then `pytest tests/`): **1,838 passed, 7 skipped, 2 deselected, 0 failed**
(69:36 on cores shared with B's checks; = e6599f0's).

### 8.10 After the verification (2026-11-28): what each finding became

The step-4 verification (a scratch stack of rl18xx-next `next/rust-1822` on c31a88c, the fixtures, generated games and
the Opal fallback probed) reported six findings. Fixed on this branch (three `src/site` commits: e3ec915, ef124bf, and
1069a7e, which the first finding's fix needed once rl18xx-next's scratch stack posted hand-made forms of runs),
rl18xx-next `next/rust-1822` and living-board `rust-1822`; **no edit outside `src/site`** (§8.2's list is unchanged: P5
×3 files, P4, V3). The one new file outside `src/site` is test data:
`engine-rs/site/fixtures/g1822_gen/gen-1822CA-5004-964.json` (a generated 1822CA game cut at its dead end, read only by
`g1822::tests`; beside `site/fixtures/g1822`, so the 11-fixture sweeps don't pick it up).

| # | Finding | What it became |
|---|---|---|
| 1 (major) | A run posted in a malformed or non-canonical form was accepted and **stored as posted**: Ruby can't load some (a local run's `[["I30", "local"]]`: "undefined method 'tile' for nil"), others it rebuilds differently ("differs: action.routes"); the admin's rollback to Ruby and an Opal page could no longer load such a game | **e3ec915**: `act::checked_run` stores each route in a form Ruby rebuilds as the router's route. `ruby_form`: the posted connections are kept only where Ruby's `Route#connection_data` reads them as the router's route -- the router's connections in turn or in reverse, each one's hexes either way round (a connection is found from either end, `get_node_chains`) -- and then the route is stored with the `nodes` Ruby writes for them (each connection's two stops in the order of its hexes, in turn, repeats dropped: `Route#node_signatures`, which Ruby recomputes after rebuilding) and Ruby's `hexes` (the stops' hexes in turn; a one-connection route's in either order, as Ruby keeps them). A local run, and a route with a connection inside one hex (both cities of an OO hex, which Ruby's chain search finds both ways round), are kept only exactly as the router writes them. **Any other form over the router's stops is stored in the router's form** (autoRoute's): other track between the same stops, the stops in another order, a local connection the wrong way round, connections reversed against their nodes, junk. The rule "only the router's run" is unchanged (the same trains over the same stops, matched on the nodes); only what is stored changed. **1069a7e**: such a form is priced over its stops in the order the train reaches them (the router's, or their reverse), since the nodes Ruby writes for connections in reverse turn aren't that order and the engine's pricer reads its route's nodes in order (on the scratch stack a 7-train's form was refused "illegal route", `1822CA_game_end_bank` @1259; an order the pricer took could have priced a form otherwise, which Ruby's check can't see in 1822CA, whose runs MRI doesn't price); the stored nodes stay Ruby's. Tests: `posted_runs_are_stored_in_a_form_ruby_rebuilds` (every form at two positions, each one's revenue the router's, @1259 too) |
| 2 (minor) | The Opal fallback (living-board `src/titles/worker.ts`) offered other players' minors (MinorAcquisition: Ruby's `mergeable` unfiltered) and every open company (AcquireCompany: G1822's step has no `acquire_companies`, so `game.companies`) | living-board `rust-1822` 7aa5388: in the 1822 family (a game whose class is or descends from `G1822::Game`) a merge's candidates (the step's `mergeable`, nothing where it is empty: the list used to fall back to every corporation) and an acquisition's companies (Ruby's `purchasable_companies(entity)`) are kept to the acting company's president's own and the bank's; both types are offered from the list only, never as JSON (as `buy_train` / `buy_company` everywhere). With the site's Opal bundles: at the 1,339 fixture positions where Ruby's set has a minor or private acquisition (1,167 with other players' too) the list = engine-rs's RESTRICTED set at every one (before: 113); C's `test/rust/fallback1822.cjs` at every acquisition step of the fixtures, 540 / 540 and 986 / 986 (C's §11.5). The "$200" in a £ title was the probe's: the worker sets the title's money when it opens a game |
| 3 (minor) | A run posted without `nodes` (optional in Ruby's RunRoutes; the older records) was refused although it was the router's run, with "Only the best run…" | **e3ec915**: a route without nodes is the router's when its connections end on the router's stops' hexes (Ruby takes the stops from the connections); it is stored with the posted connections and Ruby's nodes where those are a form Ruby rebuilds (above), else in the router's form. The verification's "242 refused only because nodes are missing" counted runs whose stops differ too (it added the router's nodes, which the old check compared alone): **187** of the 272 recorded runs without nodes are the router's, **85** are other runs (refused, rightly) |
| 4 (minor) | At the 3 RESTRICTED fallback positions the menu's `deadEnd` was a raising candidate's "(Ruby raises)" words (act::dead_end first), so the page added "the printed rules … come with an engine update" while the block is the own-only rule | **ef124bf**: where deals with other players' companies are left out (`othersLeftOut`, legal moves), the menu says "The only moves left are deals with other players' companies, which this site doesn't offer" whatever a candidate of the player's own raises; a Ruby dead end's words only where Ruby's set has nothing either. The page adds the engine-update words only to "(Ruby raises)" (`G1822Hand.tsx`), so it needs no change. Test `only_cross_player_deals_left_says_so` (gen-1822CA-5004 @964; it fails on the parent with the CPR tax-haven words) |
| 5 (minor) | Three specs hard-coded the bots' titles (`['1830']`, the boot's `deps engine g_1830`) and failed on a stack with `BOT_TITLES=1867` | rl18xx-next `next/rust-1822` 535645b: those examples stub `Lib::Bots::TITLES` to a site without BOT_TITLES |
| 6 (minor) | Most recorded "missing" discards in the reports are not out-of-order: they name an interchangeable instance of the offered train (the authority accepts and stores the generator's instance) | Reports corrected: of the fixtures' recorded discards the menus don't offer as recorded, those naming another instance of an offered train are offered (`check` takes them and stores the menu's instance, so the page re-reads); the **2 true out-of-order discards** are §2.3 G3's (`1822CA_game_end_stock_market` @737 GWR, `1822CA_loan_payments_preserve_bank` @1086 PGE), refused "may not act now". C's `hand1822.cjs` counts the instance discards apart and gives each still-missing record the engine's own verdict (C's §11.5) |

**The fixtures' recorded runs** (1,004 kept `run_routes` before a game's end; `print_recorded_runs`, an ignored
debugging aid printing each run's verdict; §8.6's "377" was `corpus_check`'s listing, which keeps 60 refusals per game:
the count at c31a88c was 505 accepted, 499 refused):

| | c31a88c | 1069a7e |
|---|---|---|
| Accepted | 505 (all stored as posted) | **692**, every one at the router's revenue: with nodes 505 -- 465 stored exactly as recorded, 1 with Ruby's own nodes (a one-hex connection, `1822CA_game_end_stock_market` @63: Ruby recomputes `["AG13-0", "AG13-1"]` from the record's `["AG13-1", "AG13-0"]`; its re-post's Ruby check is equal), 39 in the router's form (other track between the same stops, or the stops in another order); without nodes 187 -- 176 with the recorded connections and Ruby's nodes added, 11 in the router's form (ef124bf: 185, two 7-train runs of `discard_trains_after_sr_export` @870 / @903 refused "illegal route" until 1069a7e) |
| Refused ("Only the best run…") | 499 (242 without nodes) | **312**, every one another run than the router's (222 with nodes, 85 without, 5 with a local route recorded without nodes beside others): 191 of equal revenue, 117 lower, 4 higher (§2.3 G1 / G2's old records) |

`corpus_check` on the fixtures: 9,691 moves, **9,377 accepted**, 312 runs and the 2 discards refused (was 9,190).

**The binding check** (menus vs engine-rs's RESTRICTED set, both directions) at every fixture position after both
commits: **9,709 / 9,709 equal, 0 menu-only, 0 engine-only, 0 fallbacks** (`menus_equal_the_restricted_set_at_every_
fixture_position`); the generated games' 3 fallback positions (§8.6) are unchanged as moves and now carry the own-only
words. No mismatch to report.

**Ruby's side** (rl18xx-next `next/rust-1822`, its ENGINE_RS.md "The verification's findings, step 4"; a scratch stack
with `BOT_TITLES=1867`): specs 508 / 0 in both modes; every recorded action of the fixtures through the authority
12,094: 11,749 accepted, 343 refused (c31a88c: 11,542 / 550; 207 more runs taken; the 33 runs whose fields differ in
revenue differ only by a mail-contract subsidy the old record lacks); the fixtures re-posted live (8,182 moves) with
**979 hand-made forms of the auto-router's run at ef124bf and 958 at 1069a7e** (all three fields reversed, the
connections reversed alone, their order reversed alone, a local connection the wrong way round, no nodes, junk
connections), each undone after: all accepted but ef124bf's 7-train (1069a7e's fix), at 1069a7e every one at the
router's revenue; **Ruby's check: every stored run equal** (1,718 and 1,330 rows, the hand-made forms among them), no
`ruby_refused`, no `differs`; R10 refusals, and two unclassified: the 7-train (fixed) and a recorded merge after a
re-post diverged from its record, which Ruby's own `mergeable` leaves out and its `process_merge` takes (rl18xx-next's
known leniency, not this change).

**Gate at 1069a7e** (the code head; the same at ef124bf): `--features python` 277 passed, 0 failed, 11 ignored;
`--features site` **324 passed, 0 failed, 14 ignored** (+2 tests, +1 ignored debugging aid); the 1830, 1867 and 1822
neutrality pins and 1822CA's walk pin pass in both builds (`neutrality.rs`, `map.rs` and `mcts*.rs` unchanged since
fa11715); the wasm32 checks (`wasm,site-titles`, `wasm`) and the default `cargo check` clean; the wasm package built
from the pin in rl18xx-next's docker build (`site,wasm,site-titles`). `uv sync --reinstall-package rl18xx`, then `pytest
tests/` in this worktree's venv, on ef124bf (`src/site` is behind the `site` feature, which the wheel doesn't build, so
1069a7e's wheel is the same): **1,838 passed, 7 skipped, 2 deselected, 0 failed** (56:45 on cores shared with the
scratch stack's checks; = §8.9's).
