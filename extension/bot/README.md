# The live bot

Plays one seat of live 1830 games on 18xx.games as a bot account, using the advisor
extension's autoplay in a headless Chromium on this machine. Your friends see an ordinary
player making moves; the bot thinks about 2 s per decision (200 readouts on CPU).

**Before you invite anyone**: tell the other players that seat is a bot, and keep the game
unlisted. The model has never played people other than you.

## Setup (once)

1. For each bot account, log in to 18xx.games in a private window, copy the value of its
   `auth_token` cookie (developer tools → Application/Storage → Cookies), and close the window
   (don't log out: that ends the session). Save it, alone, in a file named exactly after the
   account's name on the site:
   ```bash
   mkdir -p ~/.config/18xx/bots && chmod 700 ~/.config/18xx/bots
   install -m 600 /dev/null ~/.config/18xx/bots/<account name>.token && nano ~/.config/18xx/bots/<account name>.token
   ```
   An account is only an API session (it joins games and posts its seats' moves); the runner
   never prints or logs the cookies. (One account only: `--player "<name>"` with `~/.config/18xx/bot_auth_token`.)
2. Node.js (nvm is fine). The launcher installs Playwright here on first use (and its
   Chromium goes to `~/.cache/ms-playwright`).

## Play

18xx.games has no "add a player": each account joins a game itself before it starts. So the
bots join your game, then you start it. Run them in watch mode (in tmux/zellij, it keeps
running):

```bash
cd /home/revys/workspace/RL_18xx/.claude/worktrees/autoplay
scripts/launch/2026-10/live_bot.sh --watch
```

To add the bots to a game: create the 1830 game from your own account (unlisted is fine),
take its id from the URL (`18xx.games/game/<id>`), and

```bash
echo <id> >> ~/.config/18xx/bot_join
```

Within 2 minutes every bot account joins it. Start the game when they're in; each bot plays
its seat from then on. In watch mode they also play every 1830 game they are already in.

- `--dry-run`: moves are searched and checked with the site's engine, then only logged
  ("Dry run — would play …"), never posted, and nothing is joined. Do this first.
- `--join <id>` (at start), `--game <id>` (only these games, no watching), `--readouts N`
  (default 200), `--headed` (show the browser), `--cookie-file`, `--join-file`, `--backend`.

The launcher starts the bot's own advisor backend on `127.0.0.1:5004` (this machine only)
unless one is running, builds the extension, and runs `run.js`. Stop with Ctrl-C (the
backend keeps running; the next start reuses it). Run one bot at a time (one browser profile).

What the runner logs: the seats it plays, every new move in the game (once a minute), and
autoplay's state when it changes. If the bot's turn waits more than 5 minutes it saves a
screenshot under `~/.cache/rl18xx-bot/logs/` and reloads the page. If autoplay stops on an
error (e.g. the engines disagree), the reason is in the log; fix or undo, then restart the runner.
A move that gets no answer (a dropped connection) is not an error: the move may have landed, so the
bridge reads the game afresh and plays on from there (the server would take a resent move as a
second one); only 3 such moves in a row stop autoplay.

## How a move is made

One headless Chromium, logged out, opens each started game once, with the extension's autoplay on
for every bot seat in it. When a bot seat is to move, the extension's page bridge asks the site's
own engine (in the page) who must act, the backend searches and returns the move, the bridge checks
it with the site's engine, and the runner posts it as the account of the seat that makes it
(`POST /api/game/<id>/action`, as the site's page does); the server checks it again and publishes
it to every player's page.

## Traffic

18xx.games allows each IP 2 requests a second (a burst of 5 beyond that), and a burst over it got
this machine's IP blocked once (2026-10-30: three accounts joining in one go). So every request the
runner makes -- game reads, moves, joins, game lists, page loads -- waits its turn in one queue, at
least 1.5 s apart (`--min-request-interval-ms`), with 8 s after a page load. One read of a game
serves every bot seat in it; a move the server accepted updates that copy instead of a re-read.
While other players think, the game is read every 30 s, and once more when the page shows a new
move. The page's own live-update connection is the only other traffic. Your own browsing of the
site shares the same per-IP allowance.

## Rules our engine handles differently

It never buys trains from other players' corporations or privates from other players. If a human
player sells a bot a train or private that way, the backend's engine can't follow the game:
autoplay stops and says so.
