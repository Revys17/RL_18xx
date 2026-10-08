# RL18xx 1830 advisor (browser extension)

A panel on 18xx.games game pages with the model's view of the game: each
player's chance of winning, the top five moves for whoever is to move (with
prices for bids and other open-price moves, and the hex outlined on the map),
an optional deeper search ("Think harder"), and how likely the model found the
last few moves actually played. You still make every move on the site
yourself; the extension never acts in a game.

It has two parts:

- **The backend** (`python main.py advisor`): loads the checkpoints once and
  answers on `http://127.0.0.1:5002`. It never contacts 18xx.games or any
  other host, never sees your login or cookies, and keeps games in memory
  only. It answers only browser extensions (requests whose `Origin` is
  `chrome-extension://…` or `moz-extension://…`).
- **The extension** (this directory): on `https://18xx.games/game/<id>` it
  reads the game from the site the way the site's own page does
  (`GET /api/game/<id>`, same origin, without cookies), at most once every 10 s:
  when the game opens, when the game log shows a new line, and when you press
  ↻. Chat lines are dropped before the game leaves the page. Its background
  worker posts the game to the backend and the content script draws the answer.

## Run the backend

```bash
uv run python main.py advisor                      # GPU if available, else CPU
uv run python main.py advisor --device cpu         # keep it off a busy GPU
uv run python main.py advisor --policy model_checkpoints_pg/<run>/learner/<n>.pth   # another policy
uv run python main.py advisor --debug-page         # + http://127.0.0.1:5002/debug
```

Defaults: policy `model_checkpoints_pg/pg4_20261007/learner/600.pth`, auction
policy `model_checkpoints/AlphaZeroTransformer/20261004_134558_804974562/10.pth`
(the policy-gradient runs never trained on the waterfall auction), value
`model_checkpoints_value/AlphaZeroTransformer/20261004_134558_804974562/26.pth`.
`--port` changes the port (then set the same URL in the extension's options).
The debug page takes a pasted or uploaded game JSON (as `/api/game/<id>`
returns it) and shows the same panel, without the extension.

## Build and install the extension

```bash
python extension/build.py        # writes extension/dist/chrome and extension/dist/firefox
python extension/build.py --extra-origin http://localhost:9292   # also run on a self-hosted 18xx server
```

**Chrome / Chromium / Edge**: open `chrome://extensions`, switch on *Developer
mode*, click *Load unpacked* and choose `extension/dist/chrome`. After a
rebuild, click the extension's reload button.

**Firefox** (128 or later): open `about:debugging#/runtime/this-firefox`, click
*Load Temporary Add-on…* and choose `extension/dist/firefox/manifest.json`.
Firefox asks for site access separately for Manifest V3 extensions: if the
panel doesn't appear, open the extensions menu (puzzle icon) and allow the
advisor on 18xx.games, and allow access to 127.0.0.1 / localhost. A temporary
add-on is removed when Firefox closes. For a permanent install, Firefox
release builds only take signed add-ons: package `dist/firefox` (e.g.
`npx web-ext build`), and sign it as an unlisted add-on on
addons.mozilla.org (`web-ext sign --channel unlisted`, needs an AMO API key);
Developer Edition / Nightly can instead set `xpinstall.signatures.required` to
false in `about:config` and install the unsigned zip.

**Options** (the extension's details page → *Extension options*): the backend
URL (only `http://127.0.0.1:<port>` or `http://localhost:<port>`), on/off, the
readouts for "Think harder", and the panel's corner. Drag the panel by its
title bar; ↻ refetches the game (at most once every 10 s), – collapses it to
one line with the top move.

## What the panel shows

- Status lines: backend unreachable, unsupported game (another title, or an
  optional rule other than `optional_6_train`), untested player count (the
  model was trained on 4-player games), the engine not following a move.
- Win chances: the value network's estimate per player, normalised to 100%.
- Who is to move (a player, or a corporation and its president).
- Model's moves: the policy's top five at temperature 1 (during the private
  auction, the auction policy's), with the recommended price and the next
  likeliest prices for open-price moves; hovering a tile lay or token outlines
  its hex on the map (when the site's map is on screen).
- Think harder: an MCTS search from the position (one at a time), with visit
  shares and the search's win estimate.
- Last moves: the probability the model gave each of the last five decisions,
  its rank, and expected / plausible / surprising.

## Permissions

`storage`; host access to `https://18xx.games/*` (and any `--extra-origin`) for
the content script and the game fetch; `http://127.0.0.1/*` and
`http://localhost/*` for the backend (match patterns can't name a port). No
remote code, no other sites.

## Tests

```bash
node --test "extension/test/*.test.js"      # unit tests (also run by pytest tests/extension)
```

The unit tests drive the content script's logic and the background worker with
stand-ins for the browser; they don't load the extension in a real browser.
