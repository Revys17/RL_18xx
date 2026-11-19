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
| blend10 vs score10 (added 2026-11-19, `blend10_vs_score10`) | running | | | |

- **Loosening the prior is the biggest gain**: T 1.0 beats the bots' T 0.5 by ~5 standard errors with the same head and
  budget. M1 doesn't see it (score10 = score05 = 0.46 pass): at the regression positions the critic decides what
  search does, but over whole games a search that can leave gen6's sharp policy plays better.
- The blend (0.5 win + 0.5 score) beats the score head by ~2.7 standard errors but passes fewer M1 positions (0.34 vs
  0.46). The score head is no weaker than the win head (.481, within noise) and passes far more M1 positions (0.46 vs
  0.19), so the 11-14 switch to it cost no strength.
- M1 of `blend10` (the blend at T 1.0): 0.41 pass (auction .24, loot_dump .27, sellout .56, doomed .30, suitcase 1.00).
- The bots can take a prior temperature without a code change to their checkpoints: `main.py advisor
  --play-prior-temperature T` (2026-11-19; default = the checkpoint's training temperature, gen6 0.5). Switching the
  live bots is the user's call.

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
