# gen6: gen5's recipe at every player count (2-6)

Status: PREPARED, NOT LAUNCHED (the user, 2026-11-10: "start building it now and we can switch it over once
we're happy with gen5"). Launch = gen5 has plateaued (or the user calls it) -> deploy, population, run.sh, evals.sh.

## Why
The user's goal is one model for the whole game at any table size. Evidence (2026-11-10):
- pg2p u2000 (2-player PG only) in 4-player games: vs ckpt 10 2v2 .826 / 1v3 .417; vs gen4 2v2 .241 / 1v3 .060
  (ckpt 10 vs gen4 .077): 2-player training gave about half of gen4's 4-player gain.
- gen5 u600 (4-player PG only) vs gen4 at other counts (transfer/, auction starts, T 0.5, 400 games):
  2p 1v1 .532 (fair .50), 3p 1v2 .553 (fair .33), 4p 2v2 .754 (.50), 5p 2v3 .642 (.40), 6p 3v3 .720 (.50).
  4-player training carries to 3, 5 and 6 players, hardly to 2 (as gen4: ckpt 10 vs gen4 in 2p .307).
So counts help each other, and 2 players needs its own share of the games.

## Code (branch gen6-mixed-count, 5d0ac8e on az0-2p-20261108 = gen5's code + the mix)
`--player-mix 2:w,3:w,...`: each worker task plays one count (a task's inference requests share a layout), counts
handed out by a deficit schedule (exact shares, never more than one task off). Per count: start / mid-game files
(`{n}` templates; human_games/{start,midgame}_positions_1830_{3,5,6}p_v2.jsonl built 2026-11-10: 392/391, 445/439,
197/194 games), league learner seats and placement points (tables below), `--max-decisions-by-players`,
`--auction-stall-by-players`. Training batches that mix counts pad states to the 6-player layout and value targets to
the 6 value seats and pass the counts to the network (tested: same outputs as one-count batches). History: p<N>_*.

## Planned settings (departures from gen5 named)
- Start: gen5's final learner + critic (the checkpoint gen5's plateau picks).
- Mix: 4p 35%, 3p 20%, 5p 20%, 2p 15%, 6p 10% (Claude's proposal; the user approved it 2026-11-10). Human games by count:
  2p 366, 3p 591, 4p 2326, 5p 599, 6p 254 (4p would be 56% by frequency); 2p gets more than its frequency because
  4-player training doesn't carry to it.
- Placement points (new for 3/5/6; 2 and 4 unchanged): 2p 1/0, 3p 1/.4/0, 4p 1/.5/.2/0, 5p 1/.5/.25/.1/0,
  6p 1/.5/.3/.15/.05/0.
- League learner seats (new for 3/5/6): 2p {1:1}, 3p {1:.6, 2:.4}, 4p {1:.4, 2:.4, 3:.2}, 5p {1:.3, 2:.4, 3:.3},
  6p {1:.25, 2:.3, 3:.3, 4:.15}.
- Decision caps by count: 2p 1000, 3p 1100, 4p 1200 (DEPARTURE: gen5 1000, which cut 1-2% of 4p games), 5p 1300,
  6p 1500. Measured (gen5 u600 vs gen4, cap 2000): mean / p99 / max decisions 2p 648/798/896, 3p 717/877/948,
  4p 800/1015/1053, 5p 864/1088/1122, 6p 933/1210/1343.
- Auction stall 150 at every count (auction bid+pass actions max 61-90 at any count).
- League: league_v7_gen4 + gen5's final checkpoint + pg2p u2000 (the strongest 2-player player) -> league_v8_gen5.json
  (to create at launch). Everything else as gen5 (PG/PPO settings, auction 0.5 real rules, mid-game 0.3, random 0.2,
  placement reward, T 0.5, KL target .08, anchor every 50).
- Length: until it plateaus (as gen5), backstop 4000.

## Evals (evals.sh): every 100 updates, policy-only T 0.5, auction starts, vs gen5's final checkpoint
2p 1v1 (and vs pg2p u2000), 3p 1v2, 4p 2v2 + 1v3, 5p 2v3, 6p 3v3; 400 games each.

## 2026-11-10 23:26: queued to launch when gen5 plateaus (the user)

The user (22:55): "Yeah if gen5 stalls just go ahead with gen6". Deployment gen6-5d0ac8e (local tag on branch
gen6-mixed-count 5d0ac8e, not pushed) at /home/revys/longrun/deploy/gen6-5d0ac8e (verified). queue.sh (unit
rl18xx-gen6-queue) waits for gen5's STOP file (gen5's plateau rule) and rl18xx-gen5 gone, then G5 = gen5's last
snapshot, writes league_v8_gen5.json (league_v7_gen4's 31 + gen5/G5 + pg2p/2000 = 33), and starts rl18xx-gen6
(run.sh) and rl18xx-gen6-evals (evals.sh), both with the MPS env. gen5 ending without a STOP file launches nothing.
Checked beforehand: check_players on the deployment with the real start files and league (all five counts), the
league builder on a scratch copy. Departures from gen5, as named above: the mix itself, 4p cap 1200 (gen5 1000),
per-count caps, the 3/5/6 placement tables and learner seats, league_v8 (+gen5 final, +pg2p).
Known limit: league PFSP stats are per member over all counts, so pg2p (strong at 2p, weak at 4p+) is weighted by
its average.

## 2026-11-11 ~18:55: launched from gen5 u2800 (the user's choice)
gen5 was stopped by hand (its u-200 plateau rule kept firing "gain" on within-family progress; outside styles flat
since u1600: gen5_20261107/DESIGN.md). queue.sh got a G5_PIN override and was restarted with G5_PIN=2800.
Don't judge gen6's plateau by u-200 matches alone: add an out-of-family panel (one seat vs three outside opponents).
Out-of-family panel: outfamily.sh (unit rl18xx-gen6-outfamily, every 200 updates; 4p 1v3 vs five outside opponents =
gen5's panel, baseline gen5 u2800 mean .593; 2/3/5/6p one seat vs gen4, baseline u0 = gen5 u2800) -> outfamily.txt.

## 2026-11-12 ~00:20: PFSP per player count (the user: "pick opponents based on their strength at each level")
Why: PFSP pooled each member's rate over all counts. pg2p (2-player specialist) beat gen6 98% in 2p evals but
gen6 beat it at 3-6p, pooled rate .59 -> ~3% of slot draws, ~2.5% of 2p training games (gen6 vs pg2p 2p eval
.028 at u0, .018 at u400, worth share flat .477 -> .473). Code 76e51c4 (CountLeague): per count its own
--opponent-slots (3) servers (15 opponent servers, ~+20 GB host / GPU) and its own rates, credited by that
count's games; each count refreshes one of its slots every 2 updates. Deployment gen6-76e51c4.
Restart (restart_countleague.sh, unit rl18xx-gen6-restart) at the snapshot after u480: per-count rates seeded
by seed_count_league.py from the saved training games since u270 on top of the pooled rate as 4 pseudo-games
(pooled state kept as state.pooled-*.json). Seeded 2p: pg2p rate .20 -> 9.6% of 2p slot draws (was 3.3%).
Not fixed by this: (a) the five min_share pins (gx2 x3 at .1, gx x2 at .05) take 40% of draws at EVERY count
(no pins at 2p would give pg2p 16%, plus PFSP power 3: 28%); (b) the 4p 1v3 shortfall vs gen5: in mixed 4p
training tables gen6 finishes ahead of gen5 60% of the time, so no per-member rate sees the all-gen5 table.
Both put to the user.

## 2026-11-12 02:30: the 4p outside panel drifts down slowly (not the per-count restart)
4p 1v3 out-of-family panel mean (5 opponents x 800 games, SE ~.008; extra points from panel4_extra.sh):
u200 .625, u300 .617, u400 .619, u450 .610, u480 .615, u500 .599, u550 .612, u600 .593 (gen5 u2800 .593).
Least-squares slope -.0065 +- .002 per 100 updates; the drop began before the u490 per-count restart.
Over the same span 2p vs gen4 .618 -> .792 and 3p .675 -> .728; 5p/6p vs gen4 flat (.53 / .52).
Reading: the shared policy is trading a little 4p strength against outside styles for 2p/3p gains. Watch it;
if the panel is below gen5's .593 by u800, raise the 4p share of the mix or stop and pick a checkpoint.

## 2026-11-12 11:30: drift reading withdrawn; pins stay
4p panel u800 .605, u1000 .613, u1200 .618, u1400 .613: the u500-u600 dip was noise, the panel sits ~.61.
pg2p (2p 1v1) .025 (u600) -> .124 (u1000) -> .245 (u1300) / .230 (u1400) after per-count PFSP; training rate .08 -> .33.
Exploiter pins (5 pinned of 15 exploiters, ~53% of draws at every count, learner ahead 70-89%): limiting them to 4p
would move ~30% of 2/3/5/6p draws to harder members. The user agreed (11-12) to leave them as they are; revisit if
the pg2p eval stalls around .25.

## 2026-11-12 15:05: exploiter pins limited to 4p (the user: "Yes")
pg2p eval flat u1300-u1700 (.245/.230/.215/.273/.237). Commit 96ba807 (population member field
"min_share_players": [4]; load_min_shares(path, n)); deployment gen6-96ba807; restart_pins4p.sh (unit
rl18xx-gen6-restart2) restarts at the snapshot after u1750 and adds the field to league_v8_gen5.json's 5 pinned
members (backup league_v8_gen5.pre-pins4p-<time>.json). Dry run on the u1750 state: exploiters' share of draws
2p .20 / 3p .20 / 5p .24 / 6p .26 (were ~.53), 4p unchanged .54; pg2p's 2p share .196. Watch: pg2p eval, the 4p
outside panel (shared network), and exploiter rates at 2/3/5/6p (PFSP raises them again if gen6 starts losing).

## 2026-11-12 15:20: plan agreed with the user
(1) pins at 4p only from u1760, ~200 updates; (2) then, if the 4p outside panel holds, drop the 4p pins (exploiters
plain PFSP members; maybe retire the 8 round-2 president:<corp> exploiters); (3) if the outside panel then declines,
exploiters trained from gen6 against frozen gen6 (3p first); (4) the user is happy to stop and move to another title
once the panel stops improving.

## 2026-11-12 18:46: step 2 (the user: "Yes")
u2000: 4p outside panel .632 (best, = u1600 .631), pg2p .305 (best; training rate .41), vs gen5 best at every count.
restart_unpin.sh (unit rl18xx-gen6-restart3) at the snapshot after u2070: no pins anywhere; the 8 round-2
president:<corp> exploiters retired from league_v8_gen5.json AND from state.json's per-count stats (a resume keeps
members the file dropped); backups *.pre-unpin-<time>.json. Dry run: exploiters 11-15% of draws at every count (4p .53 -> .13).
Plateau window restarts here: stop if two panels (u2200 / u2400 ...) don't beat .632.

## 2026-11-12 22:50: gen6 stopped (plateau rule); choosing the final checkpoint
4p outside panel u2200 .621, u2400 .618 vs .632 (u2000): two panels without a gain -> stopped at u2431 (state.json
snapshot 2430, resumable). Note u2400 vs gx2_bankrupt .489, the run's lowest (u2200 .542): possibly the cost of
unpinning it. Candidates u2000 / u2200 / u2400 re-measured on fresh games (select_final.sh, seed 1, 4p panel 1600
games each, other counts vs gen4 + 2p vs pg2p 800), to pool with the seed-0 runs. Bots stay on gen5 u2800 until the
user picks.

## 2026-11-12 23:10: final-checkpoint selection (seed 0 + seed 1 pooled)
4p outside panel mean: u2000 .636, u2200 .615, u2400 .623 (gen5 u2800 .593); u2000 ahead on SL ck10, gx2_rust,
gx2_bankrupt (.551 vs .513), u2400 on gen4. vs gen4 alone: 2p u2000 .951 / u2400 .890; 3p .779 / .719; 5p .531 / .555;
6p .532 / .574. 2p vs pg2p: .308 / .411. Recommendation: u2000 (4p + 3p = 55% of the mix, no unpinning cost);
u2400 is the 5-6p / pg2p pick. Awaiting the user before swapping the bots.

## 2026-11-13 01:03: bots on gen6 u2400 (the user: "just do u2400")
swap_bots.sh 2400: 5004 / 5002 / 5014 in place, gen5 u2800 learner+critic -> gen6 u2400 (value_v2/26, auction ck10,
lr1 promoted/4 unchanged); all three healthy. Old command lines in swap_bots.log.
