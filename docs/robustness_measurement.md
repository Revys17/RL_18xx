# Robustness measurement: M1-M3 and Step 0 (2026-11-15)

The instruments of `docs/robustness_test_plan.md`, built on CPU before any arm runs. They measure the five weaknesses
of `docs/bot_weaknesses_2026_11.md` (Caleb's games against the bots): **1** auction (uncontested CA / MH, DH / CS wars),
**2** loot-and-dump buy-ins, **3** sell-outs handing over a treasury, **4** doomed buys, **5** suitcases.

| | Tool | Runs on | Cost |
|---|---|---|---|
| M1 | `scripts/robustness/regression_set.py score` | CPU (GPU optional) | ~5 s a position at 400 readouts (4 threads), 59 positions ~5 min a system; policy + value only ~0.3 s |
| M2 | `scripts/robustness/behaviour_stats.py run` | CPU | ~0.3-0.5 s a game per worker (Python engine replay): 1,952 games in 2.5 min on 6 workers |
| M3 | `scripts/robustness/exploiter_panel.sh` | GPU (CPU works) | 5 members x 800 policy-only games: ~1.5 min a member on the GPU (gen6's out-of-family panel), + M2 on 120 saved games each |
| Step 0 | `scripts/robustness/search_settings.sh` | M1 on CPU, matches on the GPU | M1 ~5 min a variant (5 variants); ~1.5-2 GPU h per 480-game 400-readout match, 3 main matches |
| M4 | strength guard: gen6's `scripts/launch/2026-11/gen6_20261110/outfamily.sh` panel (not rebuilt here) | GPU | ~10 min GPU a checkpoint (4p) |

Everything runs with `taskset -c 24-31,56-63 nice -19` (the scripts do it themselves) and `python -P`. Tests:
`tests/robustness/` (16 tests, ~15 s; the generic M1 search and `round_summary.py` included).

## M1: the regression set

### What it is

`scripts/robustness/data/regression_set.json`: **59 positions** from 12 of Caleb's games on the local site, each just
before a bot mistake, one entry per distinct position (Caleb's CA $165 opening recurred in games 5, 10, 11, 14 and 15:
kept once; `build` refuses repeats).

| Weakness | Kind | n | Bad moves | Preferred | Pass when the search... |
|---|---|---|---|---|---|
| auction | contest | 13 | the move played (pass, buy SV, bid DH / CS) | the mover's bids on CA or MH (Caleb's uncontested openers), any price | chooses a preferred move |
| auction | war | 8 | the raise in a DH / CS bid war ($100-$215; face 40 / 70) | pass | avoids the raise |
| loot_dump | buy | 11 | buying any share of the company Caleb then empties and dumps on the buyer | pass | avoids every such buy |
| sellout | sellout | 9 | every sale of the company that hands its presidency over ($340-$900 treasuries; computed by playing each sale) | pass | avoids those sales |
| doomed | buy | 10 | buying a share of a floated, trainless company that can't afford the next train (bot-run, with a route) | pass | avoids the buy |
| suitcase | buy | 8 | buying into / taking over Caleb's trainless CPR that has no route (no train obligation) | pass | avoids the buy |

Era of the bots that made the mistake: 15 positions under gen6 + its critic, 28 under gen6 + lr1 v4, 16 under gen4;
one 6-player game (game 12, 3 positions). The list is `CURATED` in `regression_set.py` (game, site action id,
weakness, kind, note); every entry's site id is the one `docs/bot_weaknesses_2026_11.md` and the analysis' dossiers
use, so `?action=<id>` on the site shows it.

Data: `data/caleb_games/<id>.json.gz` (13 games, 204 KB) are the site games as the advisor's importer played them in
the Rust engine (`pretraining.filter_actions` + `replay_human_action`, G19's cities in the engines' numbering) with
each engine action's site action id (`rl18xx.site_action_ids`) and the players' names; identical, action for action, to
the analysis' timelines. Positions name a game file and `action_index` (engine actions to replay); `played`,
`bad_moves`, `preferred_moves` are engine action dicts. A position is replayed in the Rust engine with the real auction
rules and the rule against train shuffling on (the bots' `autoplay.bot_position`).

Move matching: a legal move matches a fixture move when playing it gives the same position (equal canonical
encodings), so share-certificate ids don't matter; an open-price move (a bid) is played at the fixture move's price,
so "bid on CA" matches at any price.

### Running it

```bash
# legacy systems: the composite_model.load_policy_value / eval_head_to_head spec; search settings in eval_head_to_head's
# syntax (default: the live bots' @400/0.05:mean,dedupe,par=8,ptemp=0.5, a fresh tree per position)
uv run python -P scripts/robustness/regression_set.py score \
    --model <gen6 learner/2400.pth>+<gen6 critic/2400.pth>#score --out logs/robustness/m1_gen6_score.jsonl
uv run python -P scripts/robustness/regression_set.py score --model <policy.pth> --no-search   # policy + value only
# generic: eval_head_to_head's generic spec, searched in-process at the same settings (--search)
uv run python -P scripts/robustness/regression_set.py score \
    --model generic:<run>/learner/<u>.pth+generic:<run>/critic/<u>.pth#score --out logs/robustness/m1_<run>_u<u>.jsonl
uv run python -P scripts/robustness/regression_set.py score --model generic:<ckpt.pth>@0.5 --no-search  # T 0.5 priors
# mixed: a legacy policy searched with a generic value (game_network.GameNetwork, as the advisor's --play-* mixes)
uv run python -P scripts/robustness/regression_set.py score --model <gen6 learner/2400.pth>+generic:<critic.pth>#score
uv run python -P scripts/robustness/regression_set.py summary <results>.jsonl[.gz]... [--positions]
uv run python -P scripts/robustness/regression_set.py list
# rebuild: site exports (18xx.games-style; from the site DB with read-only SELECTs) -> data/caleb_games, then the fixture
uv run python -P scripts/robustness/regression_set.py convert <export.json | dir>...
uv run python -P scripts/robustness/regression_set.py build
```

`--weakness`, `--ids`, `--limit` select positions; `--readouts` overrides the budget; `--threads` sets torch's threads.
`summary` prints one table per `--label`; `--positions` adds a per-position pass/fail matrix across labels (which
positions flipped between two checkpoints).

**Generic checkpoints** (`model_generic`): `generic:<policy.pth>[+generic:<value.pth>][#win|#score|#blend=<w>]`, the
syntax of `eval_head_to_head`'s generic players (no `+value`: the policy network's own heads; the spec may carry its
own `@<n>/<c>:<options>`, else `--search` applies; the older `generic:<ckpt>@0.5`, a decimal after `@`, still means
the prior temperature). `GenericSystem` (2026-11-15) gives the policy's probabilities over the legal moves (1830's
legal indices are the factored layout's, the legacy model's), the value network's win / score heads for the mover and
the search's head, and **searches** at the same settings as the legacy systems: `game_network.GameNetwork` in-process
(priors at T 1 tempered by the tree's `ptemp`, as eval_head_to_head's generic players; price cells as the legacy price
rows; values in absolute seat order) through `RustMCTSPlayer.tree_search`'s `evaluates_games` hook, the root evaluated
by a one-leaf batch as the advisor's generic searches; `#score` backs up placement points at finished games when the
critic's run recorded a placement reward. Cost: default-size learner + critic ~12-17 ms of CPU a leaf (docs/rl_model.md
"Head-to-head evals of generic searches"), ~5-7 s a position at 400 readouts on 4 threads, ~6 min for the 59 positions;
`:sample=<t>` specs are policy-only. Tests: `tests/robustness/test_regression_set.py` (tiny generic learner + critic
searched on the score head at 16 readouts, a 6-player position included; the CLI end to end).

### What the numbers mean

Per position (`--out` JSONL): `policy.p_bad` / `p_preferred` (softmax at T 1 over the legal moves, summed over the
set), `prior_bad` (at the search's prior temperature), `argmax_bad`; `search.chosen` (most visited), `share_bad` /
`share_preferred` (root visit shares), `q_bad` / `q_preferred`; `value.before`, `after_bad`, `after_preferred` for the
mover (the search's head, and the value network's `win` and `score` heads), `pref_minus_bad` (1-ply: > 0 = the value
prefers the preferred move; this is "does the value see the danger"); `pass`.

Per weakness (`summary`): `pass` (share of positions passed: the search's choice, else the policy's argmax), `search
chose bad` / `chose pref`, `policy argmax bad`, mean `p(bad)`, `p(pref)`, `prior(bad)`, mean 1-ply value gap.

Pass rule from the plan: an arm improves M1 when `pass` rises and `p(bad)` falls. With 8-21 positions a weakness, one
position is 5-12 points of `pass`: read the weakness rows together with the continuous `p(bad)` and `share_bad`, and
use `--positions` to see which positions moved. Positions whose search splits ~50/50 (e.g. g12-a908, g18-a512) flip
with small changes. The search has no Dirichlet noise but is not bit-for-bit repeatable (open-price moves sample
their price children): two runs of the win head chose the same move at all 59 positions, with the bad moves' visit
share moving by 0.02 on average (at most 0.25, at auction positions).

Differences from the live bots: a fresh tree (the bots kept trees between their moves, so their visit counts included
inherited ones), and a search on the score / blend head backs up placement points at finished games
(`eval_head_to_head`'s terminal utility; the advisor backs up the win share there; this only matters where the tree
reaches the end of the game). M1 says what a system would do at these positions, not what the bot did then: gen6 + critic today
avoids some of the moves the gen6 + lr1 v4 bots made (e.g. g15-a522, p(bad) 0.000).

### Baseline (2026-11-15, CPU)

gen6 u2400 + critic 2400 at the live settings (400 readouts, mean-Q, c_puct 0.05, 8 leaves a batch, priors at T 0.5),
win head vs placement-points head; and the two policies alone (argmax). Results:
`scripts/robustness/data/m1_baselines/*.jsonl.gz` (`summary` reads them).

| Weakness (n) | gen6 + critic **#win** @400: pass / search chose bad / p(bad) | gen6 + critic **#score** @400 | gen6 policy alone (argmax) | SL ckpt 10 alone (argmax) |
|---|---|---|---|---|
| auction (21) | 0.05 / 0.67 / 0.40 | 0.29 / 0.52 / 0.40 | 0.24 / - / 0.40 | 0.62 / - / 0.33 |
| ... contest CA/MH (13) | 0.00 (chose CA/MH 0.00) | 0.00 (0.00) | 0.38 | 1.00 |
| ... bid wars (8) | 0.12 | 0.75 | 0.00 | 0.00 |
| loot_dump (11) | 0.18 / 0.82 / 0.78 | 0.36 / 0.64 / 0.78 | 0.27 / - / 0.78 | 0.64 / - / 0.23 |
| sellout (9) | 0.44 / 0.56 / 0.44 | 0.56 / 0.44 / 0.44 | 0.44 / - / 0.44 | 0.89 / - / 0.09 |
| doomed (10) | 0.10 / 0.90 / 0.83 | 0.50 / 0.50 / 0.83 | 0.10 / - / 0.83 | 0.60 / - / 0.26 |
| suitcase (8) | 0.38 / 0.62 / 0.53 | 1.00 / 0.00 / 0.53 | 0.38 / - / 0.53 | 1.00 / - / 0.17 |
| **all (59)** | **0.19** / 0.71 / 0.56 | **0.47** / 0.46 / 0.56 | 0.27 / - / 0.56 | 0.71 / - / 0.24 |

Mean 1-ply value gap (preferred - bad, critic 2400): auction -0.085 (win) / -0.071 (score), loot_dump -0.041 / -0.063,
sellout -0.070 / -0.084, doomed -0.004 / -0.003, suitcase +0.019 / +0.021: one ply after the move, the critic prefers
the bad move everywhere but in the suitcases -- it doesn't see the dump coming. 5.5 s a position (two runs sharing
the 16 pinned threads with other jobs).

Readings:
- **The score head fixes most of the search-side failures and none of the value's blind spots.** It stops 6 of 8
  bid-war raises, half the doomed buys and every suitcase buy (pass 0.19 -> 0.47), consistent with the bot-weakness
  doc's guess (a losing bot searching the win head is indifferent to bankruptcy). It changes nothing in the CA / MH
  openings: with either head the search chooses CA / MH in 0 of 13 positions, although gen6's own policy contests 5
  of them (p(pref) 0.47) -- the critic writes those bids off (the doc's probes: CA $165 0.05 to win).
- **Loot-and-dump buy-ins survive both heads** (pass 0.18 / 0.36; the policy puts 0.78 on the buys and the value
  gap favours buying): the value and policy problem the plan's arms target.
- **The human-imitation policy (SL ckpt 10) avoids most of these mistakes by itself** (pass 0.71, p(bad) 0.24 vs
  gen6's 0.56; it contests CA / MH everywhere and buys into a looted company with p 0.23 instead of 0.78). Evidence
  for arm A1 (human-imitation seats / a pull toward SL). SL raises in all 8 bid wars too (p 0.72): wars this high
  are rare among humans (DH mean $86), so the SL prior has little to say there.
- The live bots search the score head since 2026-11-14 18:32 (`--play-value-head score`): the second column is what
  they would do today, from a fresh tree.

## M2: behaviour statistics

`scripts/robustness/behaviour_stats.py` replays every game in the Python reference engine (`behaviour_events.py`: a
port of the analysis' `timeline.py` / `events.py` / `corpus.py`) and counts, from the first move the recorded players
chose themselves (a policy-gradient game's human start prefix is skipped; auction statistics only from games that
start at the auction):

| Weakness | Metric | Attributed to |
|---|---|---|
| 1 | per private (CA, MH, DH, CS): price, contested (2+ bidders), over face, buyer won | the buyer |
| 2 | `broke_dump` (a president sells out of a floated, trainless company short of the next train's price), `broke_dump_p4` (phase 4+), `looted_broke_dump` (the dumper sold one of its trains away for <= $10, or had it buy one for >= 1.5x list, in the 150 actions before), `broke_dump_victim_bankrupt` | the victim |
| 3 | `pres_sells_lost` (stock-round sale that hands the presidency over), `handover_cash300` (... of a floated company with >= $300), `takeover_bought`, `dump` | seller / seller / buyer / victim |
| 4 | `buy_into_trainless_broke` (a share of such a company someone else runs), `bought_then_dumped_same_sr`, `took_trainless_broke` (its presidency by buying) | buyer / victim / buyer |
| 5 | `trainless_turn` (a company ends an operating turn without a train), `suitcase2` / `suitcase3` (2+ / 3+ ORs in a row under one president) | the president |
| | `bankrupt`, `contrib` / `contrib$` (forced train money), `shuffle_$1`, `overpay_own_train`, `game_with_bankruptcy` | player |

Rates: per game for the whole table (`all`), per seat-game for a group. Groups come from the saved game
(`robust_games.py`): eval sides `A` / `B` (`eval_policy_only`, `eval_head_to_head`), policy-gradient `sides` (A = the
learner), `bot` / `human` for site games (names), `--default-group` otherwise (e.g. `human` for the corpus).

Inputs: collection directories (`games.jsonl` + `games/`: `eval_policy_only --save-games`, policy-gradient
`--save-game-every`, `eval_head_to_head` bare action lists with the rule variant from `settings.json`), directories
or single game files (`.json` / `.json.gz`: `make_game_record` exports, the human corpus, `data/caleb_games`), and raw
18xx.games site exports (players = the site's user ids: converted with the advisor's importer). Generic runs save the
same `make_game_record` files (`generic_pg` / `eval_generic.py`), so their collections read the same way; 1830 only.

```bash
uv run python -P scripts/robustness/behaviour_stats.py run <collection or dir>... [--num-players 4] [--finished-only] \
    [--min-update 2000] [--max-update 2100] [--sample N] [--workers 6] --label <name> --out logs/robustness/m2_<name>
uv run python -P scripts/robustness/behaviour_stats.py compare logs/robustness/m2_*/summary.json
```

Each run prints its table next to the cached baselines (`scripts/robustness/data/behaviour_baselines.json`;
`--save-baseline <name>` adds or replaces one). `--out` keeps per-game counts (`games.jsonl`; `--keep-events` adds the
events) and `summary.json`. `--num-players` also drops, before `--sample` draws, the index records that name another
count (a mixed-count run's `games.jsonl`). Games started at a practice position (`practice:<id>:sr<k>`) are counted
from that Stock Round, as mid-game starts. `behaviour_stats.broke_dumps(events)` lists a game's broke dumps (looted or
not, victim bankrupt or not): the practice-position builder reads it.

### Baselines (per game, 4 players)

Human corpus: the 1,952 finished 4-player games of `human_games/1830_clean_20261017`. gen6: its 4-player saved
policy-gradient games from update 2000 on (`gen6_20261110/games`, learner vs league at T 0.5; auction numbers from the
games that start at the auction). Caleb's games: the 13 site games in `data/caleb_games` (11 finished), per game; the
bot / human split is per seat-game. The counts equal the analysis' (all 13 games, all metrics, checked against its
own code), and so do the corpus rates.

| Per game | Human corpus (1,952) | gen6 self-play (10,788) | Caleb's games (13) | ... per seat-game: bots / Caleb |
|---|---|---|---|---|
| `broke_dump` (all phase 4+ here) | 0.19 | 0.17 | 0.62 | 0.20 / - |
| `looted_broke_dump` | 0.16 | 0.05 | 0.38 | 0.12 / - |
| `broke_dump_victim_bankrupt` | 0.03 | 0.03 | 0.31 | 0.10 / - |
| `pres_sells_lost` | 0.85 | 2.53 | 4.08 | 0.90 / 1.23 |
| `handover_cash300` | 0.19 | 1.02 | 1.92 | 0.44 / 0.54 |
| `takeover_bought` | 0.76 | 2.09 | 3.62 | 0.78 / 1.15 |
| `buy_into_trainless_broke` | 0.89 | 1.20 | 2.62 | 0.80 / 0.08 |
| `bought_then_dumped_same_sr` | 0.30 | 0.54 | 1.46 | 0.37 / 0.31 |
| `took_trainless_broke` | 0.01 | 0.08 | 0.31 | 0.10 / - |
| `trainless_turn` | 1.08 | 0.03 | 2.00 | 0.15 / 1.54 |
| `suitcase2` / `suitcase3` | 0.20 / 0.13 | 0.00 / 0.00 | 0.46 / 0.38 | 0.05 / 0.31, 0.05 / 0.23 |
| `bankrupt` (games ending in one) | 0.38 (0.378) | 0.28 (0.280) | 0.31 (0.308) | 0.10 / - |
| `contrib$` | 1,171 | 1,445 | 1,420 | 447 / 9 |
| CA: price, contested, buyer wins | $208, 86%, 0.34 | $271, 96%, 0.18 | $164, 0%, 0.64 | (Caleb took it 12 times of 13) |
| MH | $134, 62%, 0.26 | $154, 58%, 0.27 | $118, 8%, 0.64 | |
| DH | $86, 52%, 0.26 | $116, 88%, 0.27 | $133, 54%, 0.09 | bots $144 |
| CS | $47, 27%, 0.24 | $59, 37%, 0.27 | $82, 46%, 0.00 | bots $85 |

All three are cached in `data/behaviour_baselines.json` (`human`, `gen6`, `caleb`; gen6's groups are its saved
games' sides, A = the learner). Time: the human corpus 2.5 min (6 workers, 4,403 files read, 1,952 kept), gen6 15 min
(6 workers, 30,840 files from update 2000 read, 10,788 four-player games kept); a game costs ~0.3-0.5 s a worker.

How to read an arm: its eval games (M3's saved games, or `eval_policy_only --save-games` 2v2 against gen6) should move
toward the human column on the weakness rows -- more `looted_broke_dump` by the arm (it learns the tactic), fewer
`handover_cash300` / `buy_into_trainless_broke` / `took_trainless_broke` by it (it stops being the victim), `trainless_turn`
above ~0 (it has met suitcases), CA / MH contested near the human rates at human prices. Per side (A = the arm) matters
more than the totals: a total can move because the opponents changed.

## M3: the exploiter panel

`scripts/robustness/exploiter_panel.sh`: the bot alone against three copies of each member, 4 players, policy-only
(`scripts/eval_policy_only.py --a-seats 1 --auction-start`: the real auction rules, every seating), 800 games each
(+-0.015 around the fair 0.25), max 1000 decisions, 120 games saved per member and their M2 table by side (A = the
bot, B = the member). Members: SL ckpt 10 at T 1.0 (`sl10_t1`, the human spread) and T 0.5 (`sl10_t05`),
`gx2_bankrupt` and `gx2_rust` (2026-10-13 goal exploiters; they never trained the auction, `auction_fraction` 0) at
0.5, gen6 u2400 at 0.5 (the mirror). `EXTRA="name=<spec> ..."` adds members (the arm A3 exploiters); `BOT=` takes a
legacy `<ckpt>@T` or a `generic:<ckpt>@T`; `ONLY=` picks members; finished members are kept.

```bash
bash scripts/robustness/exploiter_panel.sh                                  # gen6 u2400 @0.5 vs the panel
BOT=generic:<arm>/final.pth@0.5 OUT=logs/robustness/m3_<arm> EXTRA="mx1=<exploiter>@0.5" \
    bash scripts/robustness/exploiter_panel.sh
DRY=1 DRY_ROOT=<scratch> bash scripts/robustness/exploiter_panel.sh         # CPU, 4 games a member
```

Reading: robustness = the bot's share against each member (`results.txt`); compare arms member by member (an arm
that gains against gen6 but loses against SL T 1.0 has learned the league, not humans). Existing numbers for gen6
u2400 from its out-of-family panel (same eval, 800 games): 0.705 vs SL ckpt 10 @0.5, 0.609 vs gx2_rust, 0.489 vs
gx2_bankrupt (the weakest), 0.654 vs gen4.

Dry run (2026-11-15, CPU, 4 games of max 300 decisions a member, gen6 u2400 as the bot): every member ran, wrote
`summary.json`, 4 saved games and their M2 table (~20 s a member). Scores are noise at 4 capped games.
On the GPU, ~1.5 min a member (gen6's panel: 72-100 s for 800 games, 16 workers); on 16 pinned CPU threads roughly
15-30 min a member (estimate from the dry run's ~80 decisions/s on 2 workers).

## Step 0: search settings

`scripts/robustness/search_settings.sh`: gen6 u2400 + critic 2400 at 400 readouts with the live PUCT (mean Q, c_puct
0.05, 8 leaves, dedupe, kept trees with a total budget); variants `win05`, `score05` (the live bots), `blend05`
(`#blend=0.5`), `win10` and `score10` (priors at T 1.0). Step `m1` scores every variant on the regression set (CPU is
fine); the matches are 4-player 2v2 head-to-heads against the live setting, 480 games each (+-0.023), from the
auction under the real rules (`STARTS=sr1`: human first-Stock-Round starts as `search_value_gen6.sh`), argmax play:
`win_vs_score`, `blend_vs_score`, `ptemp1_vs_ptemp05` (and `win_ptemp1`). Same spec syntax as
`scripts/launch/2026-11/bridge/search_value_gen6.sh` (left as is: its matches answer a different question, which
value to search with; run both on the same GPU day).

```bash
bash scripts/robustness/search_settings.sh                    # m1 + the three main matches
bash scripts/robustness/search_settings.sh m1                 # CPU only
DRY=1 DRY_ROOT=<scratch> bash scripts/robustness/search_settings.sh
```

### Step 0's M1, run in full on CPU (2026-11-15)

`search_settings.sh m1` for all five variants (59 positions at 400 readouts, fresh trees; ~9 min a variant on the
loaded box; results `data/m1_baselines/step0_*.jsonl.gz`):

| Variant | auction | loot_dump | sellout | doomed | suitcase | **all: pass** | search chose bad |
|---|---|---|---|---|---|---|---|
| win05 (the bots until 11-14) | 0.05 | 0.18 | 0.44 | 0.10 | 0.38 | **0.19** | 0.71 |
| win10 | 0.05 | 0.27 | 0.56 | 0.10 | 0.38 | **0.22** | 0.68 |
| blend05 | 0.24 | 0.27 | 0.56 | 0.20 | 0.62 | **0.34** | 0.58 |
| score05 (the bots now) | 0.24 | 0.36 | 0.56 | 0.50 | 1.00 | **0.46** | 0.47 |
| score10 | 0.29 | 0.27 | 0.56 | 0.50 | 1.00 | **0.46** | 0.47 |

The value head is what matters on M1 (win 0.19 -> blend 0.34 -> score 0.46); the prior temperature doesn't (+0.03
for the win head, 0.00 for the score head): at these positions the critic, not the prior, decides what search does,
and loosening the prior buys nothing. score05 run twice gave 0.47 and 0.46: 4 of 59 positions changed the chosen move
between runs (all auction positions with near-even visits), one of them (g15-a14) pass -> fail. Differences under
~0.05 overall are noise. The head-to-heads decide whether the score head costs strength.

Dry run of the whole script (CPU, `DRY=1`: M1 on 3 positions at 16 readouts for each variant, the three matches at 8
readouts, 6 games of max 300 decisions): every step ran and wrote its summary; a 6-game match took ~1.2 min.

### Step 0's matches (2026-11-18/19, GPU)

Step 2 of the GPU runbook (`/home/revys/longrun/experiments/step2_search_20261118/`; results
`logs/robustness/step0_search_settings/` in the main checkout). 4-player 2v2, 480 games a match over all six seat
arrangements, from the private auction under the real rules, argmax play; score = A's share of the wins (fair .50);
~2 h a match with the GPU shared.

| Match (A vs B) | **A's win share** | A's worth share | by arrangement | games at the length cap |
|---|---|---|---|---|
| win05 vs score05 (the head the bots searched until 11-14) | .481 +- .023 | .454 | | 42 |
| blend05 vs score05 | **.562** +- .023 | .501 | | 48 |
| score10 vs score05 (prior T 1.0 vs gen6's 0.5, the live head) | **.609** +- .022 | .524 | .544-.725, all six > .5 | 55 |
| blend10 vs score10 (added 2026-11-19, `blend10_vs_score10`) | **.548** +- .023 | .494 | | 57 |

- **Loosening the prior is the biggest gain**: T 1.0 beats the bots' T 0.5 by ~5 standard errors with the same head and
  budget. M1 doesn't see it (score10 = score05 = 0.46 pass): at the regression positions the critic decides what
  search does, but over whole games a search that can leave gen6's sharp policy plays better.
- The blend (0.5 win + 0.5 score) beats the score head by ~2.7 standard errors but passes fewer M1 positions (0.34 vs
  0.46). The score head is no weaker than the win head (.481, within noise) and passes far more M1 positions (0.46 vs
  0.19), so the 11-14 switch to it cost no strength.
- M1 of `blend10` (the blend at T 1.0): 0.41 pass (auction .24, loot_dump .27, sellout .56, doomed .30, suitcase 1.00).
- At T 1.0 the blend still wins more games than the score head (.548, ~2 standard errors) but ends with no more of the
  net worth (.494, against .524 for the temperature change) and passes fewer M1 positions (0.41 vs 0.46): it trades
  some of the score head's weakness gains for first places.
- `main.py advisor --play-prior-temperature T` (2026-11-19) sets the play searches' prior temperature (default = the
  checkpoint's training temperature, gen6 0.5; the advice search keeps it).
- **The live bots search at prior T 1.0 since 2026-11-19 11:10, score head kept** (the user's decision; the blend
  not taken because the weaknesses are the priority): all three services (5002, 5004, 5014) moved to deployment
  advisor-20261119 with `--play-prior-temperature 1.0`
  (`/home/revys/longrun/experiments/step2b_blend10_20261119/swap_prior.sh`; old command lines in `swap_prior.log`
  there, for a rollback).

### gen6's search value (`search_value_gen6.sh`, 2026-11-19, GPU)

`scripts/launch/2026-11/bridge/search_value_gen6.sh`: gen6 u2400's policy searched at 200 readouts (`@200/0.05:mean,
dedupe,par=8,ptemp=0.5,budget=total`, win head) with its own critic or with lr1's v4, against gen6's policy alone at
T 0.5; 4p 2v2, 480 games a match from human first-Stock-Round starts (+ 20% random auction endings);
`/home/revys/longrun/experiments/step2_search_20261118/`.

| Match (A vs B) | A's win share | A's worth share |
|---|---|---|
| gen6 + critic 2400, searched, vs gen6 alone | **.840** +- .017 | .557 |
| gen6 + lr1 v4, searched, vs gen6 alone | **.842** +- .017 | .556 |
| gen6 + critic vs gen6 + v4 (both searched) | .481 +- .023 | .488 |
| control: gen6 alone at T 0.05 vs at T 0.5 | .485 +- .023 | .500 |

Search is worth a great deal on top of gen6 (~.84, where greedier play alone gains nothing), and the two values are
equal: the bots' critic (since 11-13) costs nothing against v4, which they searched with before.

### The auction check (2026-11-19, CPU)

`scripts/robustness/auction_check.py`: a candidate auction system holds 1 seat of a real 4-player waterfall auction,
the bots' auction system (gen6 u2400 + its critic, score head, 400 readouts, priors at T 0.5: the bots before the
switch above) the other 3. 400 auctions a run, each opened with 0-6 actions of a random human auction; the same
openings and candidate seats in every run, so runs pair up against the control (all four seats the bots' system).
Every decision searched with a fresh tree. Judged at the first Stock Round: 8 play-outs of the rest by gen6's policy
(T 0.5) per auction, and value networks' estimates. A side's own value favours it: the play-outs, v4 and value_v2 are
the neutral judges, and the bots' critic is the control's own value. Results `logs/robustness/auction_check/` (main
checkout; `report.txt`), `/home/revys/longrun/experiments/auction_check_20261119/`.

Paired difference from the control, the candidate seat's win share (fair .25):

| Candidate (1 seat) | play-outs | v4 | value_v2 | bots' critic | human critic (its own) |
|---|---|---|---|---|---|
| gen6 + the human-game critic (critic1ep), score head, T 0.5 | +.017 +- .012 | +.006 +- .002 | -.013 +- .002 | +.011 +- .004 | +.013 +- .003 |
| generic SL + critic1ep, score head, T 1.0 | +.019 +- .013 | +.017 +- .002 | -.015 +- .002 | **+.031** +- .005 | +.025 +- .003 |

Mean price paid by the candidate seat vs the bots' seats (human median):

| Private | control | gen6 + critic1ep | generic SL + critic1ep | human median |
|---|---|---|---|---|
| CS | 52 | 49 | **44** | 45 |
| DH | 100 | 94 | **77** | 80 |
| MH | 120 | 125 | 119 | 135 |
| CA | 189 | 197 | 194 | 210 |

- **Swapping only the value** (gen6's policy with the human-game critic) buys more of the big privates (MH, CA, B&O:
  454 purchases by the candidate seat in 400 auctions vs the control's 293) at much the same prices, and the judges disagree on
  it: not worth switching.
- **The generic SL policy with the human-game critic bids like humans** (DH $77 where the bots pay $98-100, CS $44 vs
  $52), leaves the small privates to others (SV / CS / DH 103 purchases vs the control's 274) and takes the big ones
  (541 vs 293); every judge but value_v2 scores its auctions higher, the bots' own critic by +.031.
  The play-outs, the most direct judge, lean the same way but within noise. Auction-only scoring can flatter a
  system (it judges the position, not the rest of the game), so the next test is full games with only the auction
  played by this system. A lead for weakness 1, not a switch yet.

### The generic SL start and a human-game value (2026-11-18, GPU)

The arms' start (`model_checkpoints_generic/bridge_1830_check_20261113/final.pth`, the generic SL) and its critic start
(`bridge_1830_critic1ep_20261118/final.pth`: the same recipe for 1 epoch, value heads refit; win CE .952 on the human
validation games), and gen6's policy searched with that human-game critic (`<legacy>+generic:<value>`); 400 readouts,
the live bots' PUCT; results `logs/robustness/generic_sl_baseline/` (main checkout):

| System | auction | loot_dump | sellout | doomed | suitcase | **all: pass** | value pref-bad |
|---|---|---|---|---|---|---|---|
| generic SL, policy alone (argmax) | 0.57 | 0.27 | 1.00 | 0.90 | 0.88 | **0.68** | +0.017 |
| generic SL + critic1ep #score, searched (ptemp 1.0) | **0.95** | 0.45 | 1.00 | 0.80 | 0.88 | **0.83** | +0.017 |
| gen6 u2400 + critic1ep #score | 0.52 | 0.18 | 0.67 | 0.30 | 0.62 | **0.46** | +0.017 |
| gen6 u2400 + critic1ep #win | 0.57 | 0.18 | 0.67 | 0.10 | 0.50 | **0.42** | +0.012 |
| (gen6 u2400 + its critic #score, the bots now) | 0.24 | 0.36 | 0.56 | 0.50 | 1.00 | 0.46 | -0.047 |

- The arms start from a system that already avoids most of the tactics (0.68 alone, like SL ckpt 10's 0.71); the
  question for step 3 is which arm keeps that while gaining strength (gen6's PG lost it: 0.39 argmax-bad -> 0.61).
- The human-game value ranks the auction, sell-out and suitcase moves the right way round (pref - bad +0.04 / +0.06
  on auction / sellout where gen6's critic is -0.085 / -0.084): searched with it, gen6's auction pass doubles (0.24 ->
  0.52) while doomed / loot-and-dump get worse. Loot-and-dump is the one weakness no value here sees (pref - bad < 0
  for every value).

## In the arms' evals (2026-11-15)

`scripts/launch/2026-11/bridge/robust_arm.sh <arm> evals` / `evalloop` runs the instruments on every eval round (each
`EVAL_EVERY` 100 updates) after `pg_generic_1830.sh`'s matches (`MEASURE=1`, the default; `MEASURE=0`: the matches only,
as before). Files under the arm's eval directory (`logs/eval/<run>/`):

| Step | What | Output |
|---|---|---|
| M1 (CPU, in the background while M3 plays) | the round's learner alone (`generic:<run>/learner/<u>.pth@0.5 --no-search`: argmax, priors at T 0.5) and searched with its critic on the score head at the bots' settings (`generic:<learner>+generic:<run>/critic/<u>.pth#score`, `@400/0.05:mean,dedupe,par=8,ptemp=0.5`) | `u<u>_m1_policy.jsonl`, `u<u>_m1_search_score.jsonl` (+ `.log`) |
| M3 | `exploiter_panel.sh` with `BOT=generic:<learner>@0.5`: SL ckpt 10 @1.0 / @0.5, gx2_bankrupt, gx2_rust, gen6 u2400, + every loop exploiter of the run (A3, A4 with `WITH_A3`) as `mx<u>`; 800 games a member, the generic bot on a generic inference server (`GENERIC_SERVER=1`, new: `--generic-server` instead of `--generic-device`); each member's 120 saved games through M2 by side | `u<u>_m3/` (`results.txt`, `<member>/`, `<member>.m2/`) |
| M2 | (a) the round's saved 4p eval games: the matches' (`--save-games 6` each) and M3's (A = the arm); (b) the run's saved 4p PG games since the previous round (`--min-update` / new `--max-update`, a sample of 1,000 of them, A = the learner) | `u<u>_m2_eval/`, `u<u>_m2_pg/` |
| summary | `scripts/robustness/round_summary.py`: the round's numbers (`round_u<u>.json`), one line per round (`rounds.txt`, like gen6's `outfamily.txt`: the matches, M1 pass / p(bad) / chose bad per weakness, M2 per-seat-game rates of the arm's side and CA / MH prices, M3 per member), and the rounds side by side with references (`robust_status.txt`: M1 of gen6 / SL ckpt 10, M2 of the human corpus / gen6) | `rounds.txt`, `robust_status.txt` |

With exploiters, M3's 800-game matches replace the 400-game 1v3 per loop exploiter of the round. M1's searched variant
uses the run's critic (`critic/<u>.pth`, saved with every learner snapshot); the M2 PG sample counts only 4-player games
(the index names the seats). A failed step leaves a dash in the summary and a line in `results.txt`; the round goes on.
`exploiter_panel.sh` and the M1 / M2 outputs are kept: re-running `evals` for a round redoes only what is missing.

Per round, estimated for the GPU box (nothing measured on the GPU): the 11 matches ~10 min (400 games each, mostly
pool-worker CPU), M3 ~8-12 min (5 members + exploiters, 800 games each: gen6's panel took 72-100 s a member) + ~15 s of
M2 per member, M1 ~5 min CPU beside it (59 positions x ~4 s searched on 4 threads at load ~25, measured with
default-size untrained networks: 3.0-6.8 s a position; ~20 s policy-only), M2 ~2-3 min (6 workers, ~650 eval / M3 games
and 1,000 sampled 4p PG games): **~25-30 min a round**, ~20 of them holding the GPU servers; three rounds in
a 300-update arm add ~1.5 h of wall time but little GPU time next to the ~7 h run (the PG run keeps training).

CPU dry run (2026-11-15, `DRY=1`, cores 24-31,56-63 at load ~25; tiny generic SL and tiny legacy stand-ins for SL ckpt
10, gen6, gx2_bankrupt and gx2_rust; M1 on one position per weakness at 16 readouts, M3 4 games of max 300 decisions a
member, M2 on 20 sampled PG files): every step of every round ran and wrote its outputs. A0: two rounds (u1, u2: 283 s
and 276 s each with the 11 matches; M3 ~12 s a member, M2 of 48 eval + M3 games), `rounds.txt` with both lines,
`robust_status.txt` with both columns, the u2 PG window u2-u2 (after u1's). A5p: one round (273 s). A3: the loop
exploiter (`mx2`) joined M3 as an extra member, M3 run with `M3_GENERIC_SERVER=1` (the generic bot on a CPU generic
inference server against the legacy stand-ins on legacy servers: the GPU path's plumbing).

## Step 3's first pass (2026-11-19..22, GPU)

Every arm 300 updates from the generic SL with the 1-epoch critic, one after another on all 64 CPUs
(`/home/revys/longrun/experiments/robust_arms_20261118/`: `queue.sh`, then `a4.sh` and `second_pass.sh`); A2 then
resumed to u600. Policy alone at T 0.5, 400 games a match (+- ~.024); the h2h columns are the arm's share against
gen6 u2400 at its own fair value; M3 is 1v3 (fair .25), 800 games a member.

| arm @ update | 2p (.50) | 3p 1v2 (.33) | 4p 1v3 (.25) | 5p 2v3 (.40) | 6p 3v3 (.50) | 1v3 vs SL | M1 policy (auction / loot) | M1 searched | M3 gen6 / gx2b / gx2r |
|---|---|---|---|---|---|---|---|---|---|
| A0 baseline @300 | .560 | .332 | .371 | .470 | .598 | .623 | .69 (.67 / .18) | .71 | .341 / .484 / .536 |
| A1 human-imitation seats @300 | .631 | .356 | .299 | .466 | .505 | .685 | .76 (.86 / .09) | .68 | .345 / .559 / .565 |
| A1 + REF_KL .02 @300 | .566 | .363 | .275 | .300 | .412 | .583 | .76 (.86 / .18) | .75 | .241 / .627 / .654 |
| A2 prioritized starts @300 | .755 | .381 | .370 | .487 | .580 | .625 | .68 (.67 / .09) | .63 | .323 / .532 / .565 |
| A2 @600 | .715 | .445 | .395 | .534 | .675 | .632 | .68 (.67 / .09) | .71 | .409 / .479 / .569 |
| A3 main exploiters @300 | .650 | .425 | .352 | .453 | .593 | .593 | .68 (.67 / .09) | .68 | .329 / .500 / .584 |
| A4 = A1 + A2 @300 | .637 | .384 | .328 | .477 | .560 | .672 | .76 (.86 / .18) | .73 | .379 / .550 / .614 |
| A5p practice positions @300 | .620 | .428 | .378 | .477 | .524 | .671 | .69 (.67 / .18) | .69 | .331 / .495 / .531 |

- Every arm beats gen6's policy at every count by u300 and keeps most of the SL's behaviour (M1 .68-.76 vs gen6's
  .27; M2 in their PG games: the CA at about $190-230 against humans' $208 and gen6's $271, >= $300 treasury handovers
  .02-.03, buying into a trainless broke company .06-.12 against humans' .22). Most strength gaps between arms are
  within two standard errors; A2 is the strongest (2p .755) and still gaining at u600 at 3-6 players.
- **The A1 mechanism (the SL at T 1.0 and 0.5 pinned in the league) is what holds the auction positions**: A1, A1 +
  REF_KL and A4 stay at .86 while every other arm falls from ~.81-.86 at u200 to .67 at u300 (A2 still .67 at u600),
  and it raises M3 against the old exploiters (gx2b / gx2r). The reference KL holds the policy closer to the SL and
  costs strength against gen6 (6p .412 vs A1's .505), for the best exploiter shares.
- **Loot-and-dump is not fixed by any arm**: policy-only 1-2 of M1's 11 positions; searched, A5p @300 and A2 @600 pass
  4 of 11 (gen6 searched: .36 too). A3's exploiters didn't make the main harder to exploit (each refresh beat its main
  in 55-60% of its training games, the u300 one about as much as the u100 one); A5p's practice positions move only the
  searched pass.
- **Adopted for the plateau run (2026-11-22): A4**, resumed from u300 (`plateau.sh`, unit rl18xx-plateau-1830,
  backstop u2000): at u300 about A2's strength (within noise at every count but 2p), with A1's auction passes and M3
  against the old exploiters -- the test plan's rule (M1 and M3 up, the out-of-family panel kept) picks it over A2.

## Step 4: A4 to plateau (2026-11-22..24, GPU)

A4 resumed from u300 (`plateau.sh`, all 64 CPUs, ~1.7 min an update); eval rounds every 100 updates
(`logs/eval/robust_A4_20261118/`, summaries in `A4_plateau_evals.log`). Stopped by hand at u1528 (2026-11-24 09:12,
`STOP`; backstop u2000) because its strength vs gen6 had plateaued since u900 -- and, at the time, for its rising
auction prices, a misreading of weakness 1 (corrected below); 1867's PG started behind it. Columns as step 3's table (policy-only vs
gen6 u2400, fair .50 / .33 / .25 / .40 / .50, +- .022-.025); M1 policy and searched as pass (auction / loot); the last
two columns are M2 on the run's own PG games (1,000 a round): the C&A's and the M&H's mean price and the share of
auctions with 2+ bidders (humans' C&A: $208; gen6's: $271).

| update | 2p | 3p | 4p 1v3 | 5p | 6p | M1 policy | M1 searched | M3 gen6 / gx2b / gx2r | C&A (PG games) | M&H |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | .637 | .384 | .328 | .477 | .560 | .76 (.86 / .18) | .73 | .379 / .550 / .614 | - | - |
| 400 | .695 | .408 | .395 | .492 | .613 | .71 (.71 / .18) | .64 (.52 / .18) | .441 / .557 / .634 | $201, 61% | $123, 28% |
| 500 | .740 | .445 | .415 | .588 | .605 | .80 (.90 / .18) | .73 (.67 / .27) | .409 / .536 / .605 | $199, 55% | $122, 25% |
| 600 | .698 | .490 | .458 | .557 | .662 | .78 (.86 / .18) | .71 (.62 / .27) | .422 / .587 / .676 | $195, 54% | $120, 18% |
| 700 | .698 | .520 | .455 | .630 | .662 | .76 (.81 / .18) | .73 (.62 / .36) | .419 / .596 / .646 | $201, 61% | $121, 20% |
| 800 | .790 | .560 | .460 | .560 | .675 | .69 (.62 / .18) | .71 (.67 / .18) | .429 / .605 / .669 | $195, 56% | $118, 12% |
| **900** | **.823** | **.575** | **.497** | **.623** | **.725** | .71 (.67 / .18) | .68 (.57 / .18) | .472 / .640 / .696 | $206, 71% | $121, 18% |
| 1000 | .825 | .570 | .472 | .665 | .699 | .71 (.67 / .18) | .75 (.76 / .18) | .476 / .611 / .713 | $225, 85% | $123, 25% |
| 1100 | .790 | .613 | .463 | .632 | .698 | .71 (.67 / .18) | .71 (.71 / .18) | .487 / .616 / .688 | $235, 95% | $131, 50% |
| 1200 | .762 | .540 | .472 | .588 | .662 | .71 (.67 / .18) | .69 (.67 / .18) | .430 / .623 / .754 | $244, 96% | $139, 73% |
| 1300 | .757 | .547 | .490 | .593 | .735 | .73 (.71 / .18) | .68 (.57 / .18) | .434 / .654 / .738 | $254, 99% | $144, 91% |
| 1400 | .782 | .540 | .482 | .632 | .704 | .71 (.71 / .09) | .61 (.43 / .09) | .470 / .660 / .713 | $259, 99% | $145, 93% |

- **Strength vs gen6 plateaued at about u900**: mean margin over fair .252 at u900, .250 / .243 / .208 / .228 / .231
  after (the plateau rule of `panel_status.py`: the last two rounds fail to beat the best). Against gen6 alone at 4
  players it wins ~.48 where fair is .25.
- **Weakness 1 is inconsistency, not overpaying** (the user, 2026-11-24): the bots bid some privates far over face
  while letting others go at face -- if a private is worth what they pay, letting another player have one at face is
  the mistake. Measured as each private's share of 4-player auctions with 2+ bidders and its buyer's win rate when
  uncontested vs contested (PG games, `u<n>_m2_pg/games.jsonl`). By that measure the run got **more consistent after
  u900, not worse**:
  - u500: the C&A went uncontested in 45% of games ($171) and those buyers won .31 against .19 when it was bid up
    ($222); the M&H uncontested 75% ($115), buyers .30 vs .20 -- bargains left on the table.
  - u900: the C&A uncontested in 29% (buyers .25 vs .22); the CS / DH / MH uncontested in 79-87%.
  - u1400: every private contested in 87-99% of games (CS $61, DH $103, MH $147, CA $259) and buyers' win rates level
    across privates (.24-.32): no private a bargain. Prices rose toward gen6's ($271 C&A), which is not the fault.
- **Against the user's own openings it still lets the C&A / M&H go.** M1's auction positions are two kinds: 13
  "contest" (after the user's CA $165 or MH $115 opener, bid on it: the face-value half of the inconsistency) and 8
  "war" (don't raise a DH / CS war far over face). Contest passes: the SL policy 12/13, A4 policy 12/13 at u300 and
  u500, 8/13 from u900 to u1400; searched u900 4/13, u1400 2/13; gen6 searched (the bots) 0/13 -- search moves away
  from contesting (the critic writes those bids off). War passes: 6-8/8 throughout (gen6 searched 5-6/8). One-move
  positions are a proxy -- the model could bid elsewhere first and contest the C&A later -- so the auctions were then
  played out from the user's openings ("Caleb's openers, played out" below): u1528 contests them most.
- M3 kept rising slowly (gx2b .55 -> .66, gx2r .61 -> .71-.75); loot-and-dump unchanged (2 of 11 policy-only).
- **Candidates: u900** (the best panel margin) **and u1528** (the last; within about 1-2 SE of u900 at each count, and
  consistent in its own auctions). Searched matches of both against the live bots -- the bots' settings for both
  sides (score head, 400 readouts, PUCT 0.05 mean-Q, dedupe, 8 a batch, kept trees, priors at T 1.0), 480 4-player
  2v2 games from the private auction: `/home/revys/longrun/experiments/search_u900_20261124/` (units
  rl18xx-search-u900, then rl18xx-search-u1528). The bots stay on gen6 until a candidate beats it there and the user
  OKs the swap. On weakness 1 u1528 is the better of the two (below), so u900's match was stopped at 120 of 480
  games (0.475 +- 0.046; `u900_vs_bots_partial120/`) and u1528's started first (2026-11-24 13:19, the user), with a
  full u900 match queued behind it (unit rl18xx-search-u900b; `touch STOP_900` skips it). Each takes ~10 h
  sharing the GPU with 1867's PG.
- **u1528 searched beats the live bots: 0.625 +- 0.022** (fair 0.50; 480 games, 2026-11-24 13:19 - 23:06;
  `u1528_vs_bots/`): worth share 0.537; by arrangement AABB .863, ABBA .750, BBAA .650, ABAB .588, BABA .512,
  BAAB .388 (u1528 in the middle seats its weakest); 28 games (5.8%) at the length cap. It passes the swap gate's match;
  the swap waits for the user's OK. u900's full match started 23:07 (`u900_vs_bots/`).
- **u900 searched is level with the live bots: 0.494 +- 0.023** (fair 0.50; 480 games, 2026-11-24 23:07 - 11-25
  09:48; `u900_vs_bots/`): worth share 0.485; by arrangement BBAA .650, BAAB .637, ABAB .613, ABBA .525, AABB .325,
  BABA .212; 66 games (13.8%) at the length cap, against u1528's 5.8% -- u900's tables stall more than twice as often.
  Its stopped first run (.475 +- .046 over 120) agreed. **u1528 is the stronger candidate** (.625 vs .494, a gap of
  about 4 SE), and the better one on weakness 1 (below); u900 is out. Neither is swapped in: the bots stay on gen6
  until the user is happy with the 1830 model as a whole (2026-11-25).

### Caleb's openers, played out (2026-11-24, CPU)

`scripts/robustness/caleb_openers.py` plays the rest of the auction from the user's openings, where M1's "contest"
positions ask for one move.

- **Openings**: the 11 four-player games in which Caleb opened with bids on the C&A / M&H (game 13 also on the D&H at
  $75). His seat places exactly those bids on his first turns. A bid is dropped when a bot bid on that private first;
  those cases are counted apart and are not in the n's below.
- **Players**: after the script, the generic SL at T 1.0 plays Caleb's seat (a human imitation, not Caleb). The bots
  hold the other three seats.
- **Searched runs**: the live bots' settings (score head, 400 readouts, priors at T 1.0), with a fresh tree for every
  decision where the bots keep theirs. 110 auctions, 10 per opening. A searched bot is deterministic, so its repeats
  differ only through Caleb's seat.
- **Policy-only runs**: T 0.5 (the SL bots at T 1.0), 440 auctions.
- **Contested**: a bot bid on the private after Caleb's opening bid.
- **Inconsistent**: Caleb keeps an opened private at his price while a bot pays $30+ over face for another.
- **Output**: `logs/robustness/caleb_openers/` (`report.txt`, a `.summary.json` per run). Launcher:
  `/home/revys/longrun/experiments/caleb_openers_20261124/run.sh`.
- **Cost**: the searched runs took ~25 min each on 24 cores. Meanwhile 1867's PG slowed from ~2.3 to ~4.6 min an update.

| bots | C&A contested (n) | M&H contested (n) | C&A / M&H price | bots over face: CS / DH | inconsistent | cash + face: Caleb / a bot |
|---|---|---|---|---|---|---|
| **gen6 searched (the live bots)** | 0% (81) | 0% (70) | $165 / $115 | $37 / $25 | 83% +- 4% | $592 / $580 |
| u900 searched | 97% (60) | 0% (79) | $211 / $115 | $22 / $43 | 70% +- 4% | $586 / $562 |
| u1528 searched | 91% (69) | 49% (76) | $221 / $129 | $13 / $43 | 40% +- 5% | $579 / $562 |
| gen6 policy | 99% (394) | 97% (261) | $233 / $158 | $22 / $42 | 3% | $589 / $544 |
| u900 policy | 94% (202) | 10% (112) | $219 / $118 | $9 / $14 | 18% | $577 / $577 |
| u1528 policy | 98% (207) | 66% (110) | $233 / $136 | $19 / $30 | 7% | $579 / $553 |
| SL policy (human imitation) | 93% (229) | 66% (114) | $226 / $136 | $8 / $18 | 7% | $573 / $574 |

- **The test reproduces the site.** With the live bots' settings no bot ever bid on Caleb's C&A or M&H. He took both
  at his price, the bots paid $37 over face for the CS, and he left the auction with $592 (his real games:
  $590-595).
- **The policies contest; the search stops them.** gen6's policy alone contests 97-99% of the time, paying a lot (C&A
  $233). Searched, it contests 0%: the critic overrules the policy, rating any C&A / M&H bid a loser
  (`docs/bot_weaknesses_2026_11.md` "1.").
- **u900 fixes the C&A under search but never contests the M&H** (0 of 79; its policy 10%): 70% of auctions are
  inconsistent.
- **u1528 is the most consistent searched system**: C&A 91%, M&H 49%, inconsistent in 40% of auctions. The SL
  contests the M&H in 66%; the human corpus' M&H has 2+ bidders in 62% of auctions. Per opening, u1528 searched contests
  the M&H 20% of the time after game 21's opening (MH $115 then CA $165) and 70% after game 13's.
- **The bots often bid first.** In 30-50% of the C&A scripts the generic bots and the SL bid on the C&A before Caleb's
  seat could (gen6 searched 19%, gen6 policy 2%).
- **Caveats**:
  - Only 11 openings.
  - After his script Caleb's seat is the SL; Caleb may defend his openers harder or less.
  - The searched bots use fresh trees.
  - "Contested" says nothing about whether the price was right.

On weakness 1: u1528 > u900 > gen6. What is left is the M&H, and the war over the D&H (u1528 searched pays $43 over
face for it).

### Auctions in u1528's match against the live bots (2026-11-25)

The 480 games of u1528 searched vs gen6 searched (0.625 above), per private (`scripts/robustness/auction_probe/
match_auctions.py`): contested = 2+ bidders; win rates are each side's buyers' (side averages: u1528 .312, gen6 .188).

| Private | Contested | Human corpus | gen6 opened -> u1528 bid too | u1528 opened -> gen6 bid too | Buyer wins, uncontested |
|---|---|---|---|---|---|
| C&A | 22% | 86% | 15 of 105 | 0 of 375 | gen6 .48 (90), u1528 .33 (283) |
| M&H | 4% | 62% | 0 of 426 | 0 of 54 | gen6 .09 (408), u1528 .48 (54) |
| D&H | 62% | 52% | 26 of 48 | 171 of 432 | gen6 .27, u1528 .35 (contested: .26 / .37) |
| C&St.L | 21% | 27% | 0 of 317 | 0 of 163 | |

- Neither side contests the other's C&A or M&H. Every contested C&A was u1528 against u1528 (the other seat).
- u1528 lets gen6 keep its C&A opener at $160-165 in 90 of 105 games, and gen6 wins .48 of those (its average .19):
  the face-value bargain of weakness 1, here left by u1528. The M&H at face is no bargain in this company (gen6's
  uncontested M&H buyers win .09). So u1528 contests the user's openers more than the live bots do (played out
  above), but in this match it is still inconsistent.

### Why search stops the bots contesting (2026-11-25)

The user asked whether a search bug makes searched bots bid less than their policies (gen6's policy contests the
user's openers 97-99%, searched 0%). Probes: `scripts/robustness/auction_probe/` (CPU).

- **The tree is not at fault.** After a C&A $165 opener, gen6's live search (400 readouts, PUCT 0.05 mean-Q) gives
  bid C&St.L $45 183 visits (Q .489), D&H $75 141 (.465), pass 33 (.461), M&H $115 3 (.328), **C&A $170 1 (.194)**.
  The C&A bid's Q is exactly the critic's own value of that move for the mover (placement score), so the tree backs up
  the right player's value.
- **The critic prices a standing bid as a war.** Along a line where the bidder holds C&A $170 and nobody contests,
  gen6's critic gives the bidder .10-.26 while the bid stands and .46-.49 once the C&A is theirs at $170 (same mover
  status). The critic is the value of gen6's own play, where a C&A bid starts a war to ~$271 and the bidder keeps
  raising; it never values "bid, then drop out if raised". u1528's critic marks the bid down much less (.51 to move
  -> .36 after bidding, .57 once it owns the C&A), consistent with u1528 contesting more.
- **More search doesn't reach the resolution.** PUCT 1.0 / 3.0 at 1,600 readouts gives the C&A bid 9-18 visits and Q
  stays .22-.23: the C&A is sold 5-20 moves later, past what the search explores, so the critic's mid-auction number
  decides.
- **A real bug found on the way: the critics overrate the player to move.** On a pass that changes only whose turn
  it is, both critics move the passer down and the next player up: +.15-.25 in the auction (scripted lines), +.06 in
  stock rounds (466 passes, +-.003; `pass_swing.py`). Against the match's actual placement points the mover is
  overrated by ~.05 relative to the other seats overall (920 positions; gen6 .047, u1528 .052) and ~.25 in the auction
  (19 positions; `mover_calibration.py`). Cause: policy-gradient critics train only on the learner's own decisions
  (`policy_gradient.py`'s critic step on the PPO rows; `generic_pg.py` the same), so the player to move is always
  the learner, usually the strongest at its table. Harmless in PG (the critic is a baseline for the learner's moves);
  in a search any seat moves, which should favour moves that keep the turn (a sale before a purchase in a stock
  round) over moves that hand it on -- possibly part of weakness 3's search-chosen sell-outs (untested). It does not
  explain the auction, where every move hands the turn on.
- **Fixed in training (2026-11-25, the user's go-ahead)**: `policy-gradient` and `policy-gradient-generic` train the
  critic on a uniform sample of every seat's decisions, as many per game as the learner's rows (`--critic-seats all`,
  the default for new runs; a resumed run keeps its own setting, `learner` for runs before the option, unless the
  flag is passed). The PPO rows are unchanged; the baseline now predicts an average seat, not the learner -- still a
  function of the position alone, so the gradient stays unbiased (a little more variance: the advantages' mean
  rises by about the learner's edge). Existing
  critics (gen6 2400, A4's) keep the bias until retrained; 1867's running PG (`gpg1867_a4_20261122`) too.

### u1528's critic retrained on every seat (2026-11-25)

The user's go-ahead ("give u1528 a critic trained on every seat, then redo the auction checks").

- **Run**: `model_checkpoints_generic_pg/u1528_critic_all_20261125`, unit rl18xx-critic-all, beside 1867's PG.
  - Launcher: `scripts/launch/2026-11/critic_all_u1528/` (`prepare.py`, `run.sh`), running from deployment
    critic-all-d8d95ac.
  - Setup: A4's resume state with u1528's learner and critic. The policy is frozen (lr 0 in the config and in the
    copied optimizer state; the learner's weights are bit-identical at every snapshot). `--critic-seats all`.
  - Kept from A4 at u1520: the league, the opponent pool, the A2 start priorities and every game setting. The league
    and pool stop growing.
  - About 3.7-4.3 min an update while sharing the machine (1867's PG slows from ~2 to ~3.8 min).
- **Probes** (`scripts/robustness/auction_probe/`; `ca<N>` in `_specs.py` = u1528's policy with this run's critic N).
  Outputs are in `/home/revys/longrun/experiments/critic_all_20261125/probes/`.
- **The mover bias is gone by u1560** (32 updates). Placement points actual minus predicted, mover vs other seats,
  on u1528's match games (`mover_calibration.py`; `FIRST=30` samples the auction and SR1):

| positions | gen6 2400 | u1528 (old critic) | ca1560 |
|---|---|---|---|
| auction (1,053) | .129 +- .014 | .083 +- .014 | .005 +- .014 |
| whole games (1,808) | -- | .057 +- .008 | .015 +- .008 |
| stock rounds (913) | -- | .038 | -.004 |
| operating rounds (856) | -- | .074 | .035 +- .012 |

  The earlier ~.25 auction figure came from 19 positions.
- **After the C&A $165 opener** (`root_moves.py`): the next bot still bids the D&H at $75 with either critic. The C&A
  bid's Q gap to that move narrows from .105 to .064.
- **Caleb's openers, played out** ("Caleb's openers, played out" above; u1528 searched, 110 auctions each):

| critic | C&A contested (n) | M&H contested (n) | bots bid first on the M&H | C&A / M&H price | inconsistent | cash + face: Caleb / a bot |
|---|---|---|---|---|---|---|
| u1528's own | 91% (69) | 49% (76) | 3 / 79 | $221 / $129 | 40% +- 5% | $579 / $562 |
| ca1560 | 73% (60) | 83% (30) | 50 / 80 | $225 / $141 | **19% +- 4%** | $573 / $555 |

  The inconsistent auctions halve, mostly because the bots now go after the M&H: they bid on it before Caleb's script
  in 62% of auctions (the human corpus: 62% of M&H auctions have 2+ bidders). The bots also pay more for the C&A,
  $84 over face against $60.
- **M1** (59 positions, the live search with priors at T 1.0):

| weakness (n) | u1528's critic | ca1560 |
|---|---|---|
| auction (21) | .38 | .48 |
| loot_dump (11) | .18 | .09 |
| sellout (9) | 1.00 | 1.00 |
| doomed (10) | .90 | .90 |
| suitcase (8) | 1.00 | 1.00 |
| all | .61 | .63 |

  Loot-and-dump is not fixed. The policy puts .82 on the bad buy and the search follows it. The new critic narrows
  the preferred move's margin from -.056 to -.006, but doesn't flip it. One loot-and-dump position moves the pass
  rate 9 points.
- **Still to measure**: a searched match against the live bots (480 games) once the critic settles.

## Time and GPU estimates

- M1: CPU, ~5 min a legacy system at 400 readouts (59 positions x ~5 s, 4 threads); policy + value only, 20 s.
  Run it for every checkpoint an arm produces.
- M2: CPU, ~0.4 s a game a worker: an arm's 600-game eval sample ~1 min on 6 workers; the gen6 baseline (10.8k
  games, 30.8k files read) 15 min on 6 workers.
- M3: GPU ~8 min for the 5-member panel (800 games each), ~10 min with M2; CPU-only ~1.5-2.5 h.
- Step 0: M1 25-45 min CPU for the five variants; matches ~1.5-2 GPU h each at 400 readouts (3 main ones ~5-6 h, more
  than the plan's 3-4 h; `GAMES=240` halves it at +-0.032).
- M4: unchanged (gen6's `outfamily.sh`: ~10 min a checkpoint for the 4p panel, ~30 min with 2/3/5/6p).

## Open questions

- M1 measures one choice per position; a position where the bot's mistake was the *sequence* (three buys in one SR)
  counts each buy. Weighting by game would change the per-weakness rates a little.
- The "contest" positions assume contesting CA / MH is right (human buyers of CA at <= $195 win 42%). A model that
  bids CA at $300 passes them: read `search.chosen` / the price for those positions (the scorer reports the index,
  not the price the search would pay).
- Route laying was looked for (a bot tile lay that gives Caleb's suitcase a route): none in Caleb's games -- Caleb laid
  the routes himself onto the bots' taken-over CPRs (games 12 [996], 14 [697]). The suitcase positions are therefore
  buy-ins and take-overs only.
- Generic checkpoints are searched in-process (CPU, ~6 min a system); a generic M1 on the GPU would need
  `GenericNetwork(device="cuda")` (`--device cuda`: untested here, no GPU) or the lockstep `GenericEvaluator` on servers.
- The analysis' probe output (`probes_gen6.txt` of session 48dd27c6) printed, for auction positions, policy
  probabilities that are SL ckpt 10's, not gen6's (g21 [2]: CA $165 0.55 = SL; gen6 gives DH $75 0.68, CA 0.02;
  g21 [33]: raise 0.718 = SL, gen6 0.941); its search priors in the same output are gen6's (`probes_gen6lr1.txt`'s
  auction numbers are gen6's: g14 [3] prior 0.68 on CA). The weakness doc quotes the search priors and gen6lr1 probes
  only, so it stands; M1 always reports the scored system's own policy.
