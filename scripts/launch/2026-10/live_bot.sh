#!/bin/bash
# The live bot (extension/bot/README.md): plays one seat of live 18xx.games 1830 games as the bot's
# account. Starts the bot's own advisor backend on 127.0.0.1:$BOT_PORT (default 5004; this machine
# only) unless one answers there, builds the extension, and runs extension/bot/run.js with the
# arguments given, e.g.
#   scripts/launch/2026-10/live_bot.sh --game 274000 --player "<bot account name>" --dry-run
# The account's auth_token cookie value goes in ~/.config/18xx/bot_auth_token (or --cookie-file).
# AUCTION_POLICY / AUCTION_VALUE pass through to advisor_autoplay.sh (another auction system).
#   SITE=http://localhost:9292 scripts/launch/2026-10/live_bot.sh --watch     # a local copy of the site
cd "$(dirname "$0")/../../.."  # the repository root
PORT=${BOT_PORT:-5004}
health() { curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Origin: chrome-extension://bot' "http://127.0.0.1:$PORT/api/health"; }
if [ "$(health)" != 200 ]; then
  HOST=127.0.0.1 scripts/launch/2026-10/advisor_autoplay.sh "$PORT"
  for _ in $(seq 60); do [ "$(health)" = 200 ] && break; sleep 2; done
  [ "$(health)" = 200 ] || { echo "the bot's backend did not start: see logs/advisor_autoplay_$PORT.log" >&2; exit 1; }
fi
# SITE: another copy of 18xx.games (e.g. a local one, http://localhost:9292): the extension needs it as
# an extra origin, the runner as --site; its accounts go in their own directory (ACCOUNTS_DIR).
SITE_ARGS=()
if [ -n "$SITE" ]; then
  python3 extension/build.py --extra-origin "$SITE" > /dev/null || exit 1
  SITE_ARGS=(--site "$SITE" --accounts-dir "${ACCOUNTS_DIR:-$HOME/.config/18xx/local_bots}"
             --join-file "${JOIN_FILE:-$HOME/.config/18xx/local_bot_join}" --profile "$HOME/.cache/rl18xx-bot/viewer-local")
else
  python3 extension/build.py > /dev/null || exit 1
fi
NODE=$(command -v node || ls -d "$HOME"/.nvm/versions/node/*/bin/node 2>/dev/null | tail -1)
[ -x "$NODE" ] || { echo "node not found (install it, or nvm)" >&2; exit 1; }
[ -d extension/bot/node_modules/playwright ] || (cd extension/bot && "$(dirname "$NODE")/npm" install --no-audit --no-fund)
exec "$NODE" extension/bot/run.js --backend "http://127.0.0.1:$PORT" "${SITE_ARGS[@]}" "$@"
