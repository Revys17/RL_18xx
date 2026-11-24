# The bots against Caleb on the local site: five weaknesses (analysis 2026-11-14)

Read-only analysis of the 1830 games the user (site user `a`, id 9) played against the bots on the local site
(docker project `18xx-next-prod`, database `18xx_production`; read with SELECT queries). Games replayed through
the advisor's importer in the Rust engine and the Python engine (all 12 finished games end at the site's stored
net worths). Bracketed numbers like [436] are the site's action ids (`?action=N`). Scripts, per-game dossiers,
network probes and corpus comparisons: `scratchpad/botgames/` of session 48dd27c6 (convert.py, timeline.py,
events.py, corpus.py, compare.py, probe.py, policy_agree.py, trajectory.py).

Comparison corpora: gen6 self-play (10,788 four-player games from update 2000 on, `gen6_20261110/games/`,
policy-only at T 0.5 against its league; events counted after any human-game prefix; auction statistics from the
5,360 games that start at the auction) and the human corpus (1,952 finished four-player games,
`human_games/1830_clean_20261017`).

Bot settings in every era: one policy for play and the auction, searched with 400 readouts, trees kept between
moves, mean-Q PUCT (`c_puct_init` 0.05, 8 parallel readouts), priors at the learner temperature 0.5, the train-
shuffle rule, no purchases from other players' companies.

| Era (UTC) | Policy | Play value | Auction value | Games |
|---|---|---|---|---|
| until 11-12 02:54 | gen4 `rab_place/1800` | lr1 v4 | gen4 critic 1800 | 5, 7, 8, 9 (bots only), 10, 11, 12 (6p), 13 |
| 11-12 02:54 - 11-13 09:03 | gen5 u2800 | lr1 v4 | gen5 critic | none (33 moves of game 6) |
| 11-13 09:03 - 11-14 09:36 | gen6 u2400 | lr1 v4 | gen6 critic 2400 | 14, 15, 18, 19 |
| after 11-14 09:36 | gen6 u2400 | gen6 critic 2400 | gen6 critic 2400 | 20 (in progress), 21 |

## Results

Of 11 finished games with Caleb, Caleb won 7, the bots 4 (games 5, 8, 11 under gen4; 14 under gen6); against
gen6 he is 4-1. Four of his wins ended with a bot bankrupt (games 10, 15, 19, 21), three after Caleb dumped a
looted company on that bot. Caleb never went bankrupt; his forced train contributions total $117 over 11 games,
the bots' $18,340.

## Summary of causes

- **The values were trained only on bot-vs-bot games and misjudge a player like Caleb.** In the auction, the PG
  critics rate any CA or MH bid as a loser even at face value (in self-play CA is contested 96%, sells for $271
  on average, and its buyer wins 18%; among humans CA's buyer wins 34%, 42% at <= $195). So the bots never contest
  Caleb's MH $115 + CA $165 opening and bid DH / CS up against each other to $110-215. During play the values
  never see a loot-then-dump coming (0.05 per game in self-play vs 0.16 humans, 0.36 Caleb), and are happy to
  hand a cash-rich company to another player by selling out of it (harmless in self-play).
- **A losing bot has no reason to protect itself.** The bots' search maximises the critic's WIN head. Near 0%
  to win, all moves score about the same, so the search follows the policy, which keeps buying into Caleb's
  companies; nothing distinguishes finishing 3rd from going bankrupt (gen4-gen6 were trained on placement points,
  but the bots search the win head).
- **Search rarely overrules the policy**: 88-91% of the bots' moves were the policy's top move (priors sharpened
  to T 0.5 plus 400 readouts incl. inherited visits). Stock-round sales are the exception; in the gen6-critic era
  some search-chosen sales are the sell-outs that hand companies to Caleb.

## 1. Overpaying contested privates, uncontested ones at face: real

The fault is the inconsistency, not the prices (the user, 2026-11-24): if a private is worth what the bots pay for it,
letting another player have one at face is the mistake. Judge an auction by whether every private is contested and
no buyer does better for having paid face (`docs/robustness_measurement.md` "Step 4"), not by how high prices go.

Caleb took CA in 12 of 13 games ($165, once $160) and MH in 9 ($110-115), never with a second bidder. Every
contested private was a bot-vs-bot war (11-29 bids): DH $170-215 (face 70), CS $100-180 (face 40). Contested
privates went $99 over face on average, uncontested $1 over. Caleb leaves the auction with $590-595 of cash +
private face value; a war winner with $455-545. Over the game Caleb's CA returned +$264, MH +$223; the bots'
war prizes +$42 (CS), +$48 (DH).

| Private | Caleb's games | gen6 self-play | Human corpus |
|---|---|---|---|
| CA | $165, never contested | mean $271, contested 96%, buyer wins 18% | mean $208, contested 86%, buyer wins 34% |
| MH | $110-115, never contested | mean $154, contested 58% | mean $134, contested 62% |
| DH | bots: mean $144, median $170 | mean $116, contested 88% | mean $86 |
| CS | bots: mean $85 | mean $59 | mean $47 |

Probes (critic 1-ply win for the bidder): game 21 [2] CA $165-170 0.05, MH 0.11, DH $75 0.25, CS 0.30 (SV 0.34);
the gen4 critic agrees; lr1 v4 says the opposite (CA 0.32-0.35). At game 14 [3] the policy's prior on CA $170 was
0.68 but the search (critic) bid CS instead. The critic writes Caleb off during the auction (0.03-0.14 to win)
and gives him 0.47 right after it. The DH / CS wars come from the policy (85-97% of bids are its top move; e.g.
game 21 [33] prior 0.996 on raising beat a critic preference for passing).

## 2. Not afraid of emptied companies being dumped on them: real

Caleb dumped 15 presidencies on bots; five were floated, trainless, short of the next train's cost, phase 4+,
four of them emptied first (games 10, 15, 19, 21, 14 — dossiers in `dumps_user.txt`). In 10, 15 and 19 the bot
had no move left in the dump SR: the mistake was buying in earlier (prior 0.71-1.0 on those buys; the values saw
no danger: e.g. game 15 [402] lr1 v4 buy 0.11 vs pass 0.12). The values only update after the dump (game 19:
Caleb 0.07 at OR 6.1 -> 0.94 after his OR 6 train trades).

| Per game | gen6 self-play | Humans | Caleb's games |
|---|---|---|---|
| Dumps of a trainless, broke company, phase 4+ | 0.17 | 0.19 | 0.45 by Caleb + 0.27 bot-on-bot |
| ... looted first | 0.049 | 0.156 | 0.36 |
| Victim bankrupt afterwards | 18% | 14% | 3 of 5 (Caleb's) |

No league member dumps more than 0.06 per seat-game; the bankruptcy hunter gx2_bankrupt dumps the least (0.017).

## 3. Floating, then selling, enabling takeovers: real

Bots' own sales gave Caleb a presidency 14 times; at least 7 of those companies held $300+ (e.g. C&O $590 in
games 10, 11, 15, 18; NYC $900 in game 21 sold the same SR it was parred). Caleb took 15 more by outbuying a bot,
6 of them PRR in SR 2-4 (a bot pars PRR at $67 next to Caleb's free CA share and never defends it): the CA holder
takes over a PRR someone else parred 50% of the time in Caleb's games, 18.5% in self-play, 5.8% among humans.

| Per game | gen6 self-play | Humans |
|---|---|---|
| Presidencies given up by selling | 2.53 | 0.85 |
| ... with >= $300 in the treasury | 1.02 | 0.19 |
| Takeovers by buying | 2.09 | 0.76 |

Learned in self-play, where it's harmless. In the gen6-critic era the search chose sell-outs against a policy
that said Pass (game 21 [408], handing Caleb NYC's $900: prior 0.000, critic sell 0.17 vs pass 0.13).

## 4. Buying doomed companies: real (overlaps with 2)

| Pattern | Caleb's bots | gen6 self-play | Humans |
|---|---|---|---|
| Buying shares of a trainless, broke company someone else runs (per player-game) | 1.0 | 0.30 | 0.22 |
| Buying, then dumped on in the same SR (per player-game) | 0.39 | 0.13 | 0.08 |
| Taking over a trainless, broke company by buying (per game) | 0.36 | 0.08 | 0.01 |

Mostly bots already losing (win estimate 0.00-0.13): every move scores within ~0.01, the search follows the
policy (prior 0.6-1.0 on the buy). Likely cause (inferred): searching the win head only.

## 5. Not understanding suitcasing: real

Caleb's companies ended an operating turn trainless 20 times in 11 games (game 18's CPR for 9 ORs): float (often
CPR, whose Montreal home has no route until track is laid), spend its $1000 on his main companies, never build a
route, never be forced to buy a train. No bot ever suitcased. In gen6 self-play a company ends its turn trainless
0.03 times per game (2+ ORs 0.004); humans 1.08 (0.20). The bots buy into Caleb's suitcases, take them over, lay
the track that gives them a route (creating a train obligation), and never plan around Caleb's immunity to
forced purchases.

## Other observations

- Bots fight each other: bot-on-bot dumps 2.36 per game (0.27 deadly), bot-on-bot takeovers 2.27 per game.
- Rules decided nothing (train-shuffle rule, no cross-player purchases); no refused moves in the service log.

## Likely causes -> what might help (untested at the time of writing)

| Cause | Weaknesses |
|---|---|
| The policy never saw it in training (loot-then-dump at 1/3 the human rate, suitcasing almost absent, a receiver who loots handed-over treasuries) | 2, 5, 3 |
| Values misjudge positions they weren't trained on (CA/MH and Caleb in the auction; being next in line for a looted company; handing over full treasuries) | 1, 2, 3 |
| Search settings (guess): win-only leaf value -> a losing bot is indifferent to bankruptcy; sharp priors -> policy habits pass through | 4; 1, 2 |

Ideas: train the auction against openers like Caleb (or with a value that knows human prices); loot-and-dump and
suitcase exploiters in the league; search with the critic's placement (`#score`) head or a blend; train critics
on positions where the receiver empties a handed-over company.
