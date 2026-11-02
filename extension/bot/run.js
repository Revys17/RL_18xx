#!/usr/bin/env node
// The live bot: plays its accounts' seats in live 18xx.games 1830 games with the advisor
// extension's autoplay (README.md).
//
//   node run.js --watch                      # every account in ~/.config/18xx/bots/<name>.token
//   node run.js --player "<the bot account's name>" --watch [--join 274001] [--join ...]
//   node run.js --player "<name>" --game 274000 [--game ...]
//     [--cookie-file ~/.config/18xx/bot_auth_token] [--accounts-dir ~/.config/18xx/bots]
//     [--backend http://127.0.0.1:5004] [--readouts 200] [--min-request-interval-ms 1500]
//     [--site https://18xx.games] [--dry-run] [--headed]
//
// --site: another copy of the site, e.g. a local one (http://localhost:9292; build the extension with
// --extra-origin for it -- scripts/launch/2026-10/live_bot.sh does, given SITE).
//
// Accounts: with --player, that one account (its ``auth_token`` cookie value in --cookie-file);
// otherwise every ``<account name>.token`` file in --accounts-dir, each holding that account's
// ``auth_token`` value (only that; never printed). An account is only an API session: it lists its
// games, joins games and posts its seats' moves.
//
// --join: the accounts join these games (18xx.games has no "add a player": an account joins a game
// itself, before it starts; the game's owner then starts it). --watch: every 2 minutes the runner
// joins the game ids appended to the join file (--join-file, default ~/.config/18xx/bot_join: ``echo
// 274001 >> ~/.config/18xx/bot_join``; every account joins), lists the accounts' own games (new and
// active, unlisted ones included) and plays every 1830 game they are seated in, until stopped.
// --game: these games only.
//
// Playing: one headless Chromium (logged out) opens each started game once, with the extension's
// autoplay on for every bot seat in it. The page bridge reads the game and posts moves through the
// runner (``rl18xxReadGame`` / ``rl18xxPostAction``, below): one read serves every seat, a posted move
// updates the shared copy instead of a re-read, and each move goes out as the account of the seat
// that makes it. Every request the runner makes -- reads, joins, moves, game lists, page loads -- waits
// its turn in one queue, at least --min-request-interval-ms apart (18xx.games allows 2 requests a
// second per IP, a short burst of 5 beyond that; this machine's other traffic shares it). The page's
// own connection (its live updates) is the only other traffic.
//
// --dry-run: moves are checked and shown in the log, never posted, and nothing is joined.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
// (Required lazily: the unit tests load this file without Playwright installed.)
const playwright = () => require('playwright');

let SITE = 'https://18xx.games'; // --site
const HERE = __dirname;
const EXTENSION = path.resolve(HERE, '..', 'dist', 'chrome');
const MONITOR_MS = 60_000;
const DISCOVER_MS = 2 * 60_000;
const STUCK_MS = 5 * 60_000;
const MONITOR_MAX_AGE_MS = 30_000; // the runner's own once-a-minute look may reuse a read this recent
const PAGE_SPACING_MS = 8_000; // a page load is a handful of requests: keep the queue clear after one
const MAX_AUTOPLAY_RESTARTS = 3; // per stuck turn: a stop that keeps coming back needs a person

/** Whether the panel's autoplay status (``data-autoplay``: "<phase>: <text>", absent while it is
 * off) says autoplay has stopped: off, or stopped on an error. */
function autoplayStopped(status) {
  return !status || /^(error|off)\b/.test(status);
}

function parseArgs(argv) {
  const args = {
    games: [],
    joins: [],
    watch: false,
    player: null,
    cookieFile: path.join(os.homedir(), '.config', '18xx', 'bot_auth_token'),
    accountsDir: path.join(os.homedir(), '.config', '18xx', 'bots'),
    joinFile: path.join(os.homedir(), '.config', '18xx', 'bot_join'),
    backend: 'http://127.0.0.1:5004',
    site: SITE,
    readouts: 200,
    minInterval: 1500,
    dryRun: false,
    headed: false,
    profile: path.join(os.homedir(), '.cache', 'rl18xx-bot', 'viewer'),
    logDir: path.join(os.homedir(), '.cache', 'rl18xx-bot', 'logs'),
  };
  const id = (value) => {
    if (!/^\d+$/.test(String(value))) throw new Error(`not a game id: ${value}`);
    return String(Number(value));
  };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      i += 1;
      return argv[i];
    };
    if (a === '--game') args.games.push(id(next()));
    else if (a === '--join') args.joins.push(id(next()));
    else if (a === '--watch') args.watch = true;
    else if (a === '--player') args.player = next();
    else if (a === '--cookie-file') args.cookieFile = next();
    else if (a === '--no-cookie') args.cookieFile = null;
    else if (a === '--accounts-dir') args.accountsDir = next();
    else if (a === '--join-file') args.joinFile = next();
    else if (a === '--backend') args.backend = next();
    else if (a === '--site') args.site = next().replace(/\/+$/, '');
    else if (a === '--readouts') args.readouts = Number(next());
    else if (a === '--min-request-interval-ms') args.minInterval = Math.max(1000, Number(next()));
    else if (a === '--dry-run') args.dryRun = true;
    else if (a === '--headed') args.headed = true;
    else if (a === '--profile') args.profile = next();
    else if (a === '--log-dir') args.logDir = next();
    else if (a === '-h' || a === '--help') {
      console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(1, 36).join('\n'));
      process.exit(0);
    } else throw new Error(`unknown option ${a}`);
  }
  if (!args.games.length && !args.joins.length && !args.watch) throw new Error('give --watch, --join <id> or --game <id>');
  if (!args.cookieFile && !args.dryRun) throw new Error('--no-cookie only with --dry-run (posting needs the login)');
  return args;
}

function log(...parts) {
  console.log(new Date().toISOString(), ...parts);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readToken(file) {
  const token = fs.readFileSync(file, 'utf8').trim();
  if (!token || /\s/.test(token)) throw new Error(`${file} must hold the auth_token cookie's value, alone`);
  return token;
}

/** ``[{name, cookieFile}]``: --player's one account, else every <name>.token in --accounts-dir. */
function accountsOf(args) {
  if (args.player) return [{ name: args.player, cookieFile: args.cookieFile }];
  let files = [];
  try {
    files = fs.readdirSync(args.accountsDir).filter((f) => f.endsWith('.token')).sort();
  } catch (e) {
    files = [];
  }
  if (!files.length) throw new Error(`no accounts: give --player, or put <account name>.token files in ${args.accountsDir}`);
  return files.map((f) => ({ name: f.slice(0, -'.token'.length), cookieFile: path.join(args.accountsDir, f) }));
}

/** The game ids appended to the join file since the last look (the file is emptied). */
function takeJoinRequests(file) {
  let text = '';
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch (e) {
    return [];
  }
  if (!text.trim()) return [];
  fs.writeFileSync(file, '');
  return text.split(/\s+/).filter((t) => /^\d+$/.test(t)).map((t) => String(Number(t)));
}

/** One queue for every request this process makes: each waits until ``interval`` ms after the last. */
class Pacer {
  constructor(interval) {
    this.interval = interval;
    this.next = 0;
    this.chain = Promise.resolve();
  }

  /** Resolves when it is this request's turn; ``after``: hold the queue that long after it (a page load). */
  turn(after = this.interval) {
    const mine = this.chain.then(async () => {
      const wait = this.next - Date.now();
      if (wait > 0) await sleep(wait);
      this.next = Date.now() + Math.max(after, this.interval);
    });
    this.chain = mine.catch(() => {});
    return mine;
  }
}

/** Every game read, shared: a caller says how old a copy it accepts; a read in flight is joined. */
class Games {
  constructor(pacer, api) {
    this.pacer = pacer;
    this.api = api; // logged out: a game is readable by its id
    this.entries = new Map(); // id -> {at (when its read began), data}
    this.inflight = new Map(); // id -> {at (null until it begins), promise}
  }

  async get(id, maxAgeMs) {
    const since = Date.now() - maxAgeMs;
    const entry = this.entries.get(id);
    if (entry && entry.at >= since) return entry.data;
    const pending = this.inflight.get(id);
    if (pending && (pending.at === null || pending.at >= since)) return pending.promise;
    const read = { at: null, promise: null };
    read.promise = (async () => {
      await this.pacer.turn();
      read.at = Date.now();
      const response = await this.api.get(`${SITE}/api/game/${id}`, { headers: { Accept: 'application/json' } });
      if (!response.ok()) throw new Error(`game ${id}: the site answered HTTP ${response.status()}`);
      const data = await response.json();
      const current = this.entries.get(id);
      if (read.at !== -Infinity && (!current || current.at <= read.at)) this.entries.set(id, { at: read.at, data });
      return data;
    })().finally(() => {
      if (this.inflight.get(id) === read) this.inflight.delete(id);
    });
    this.inflight.set(id, read);
    return read.promise;
  }

  /** A move the server accepted (``stored``: as the site's engine records it), or another change
   * (null): the shared copy takes the move, or is dropped so the next look reads afresh. */
  posted(id, stored) {
    const pending = this.inflight.get(id);
    if (pending) pending.at = -Infinity; // a read that may predate the change: nobody joins it now
    const entry = this.entries.get(id);
    if (!entry || !stored) return this.entries.delete(id);
    const actions = entry.data.actions || [];
    const last = actions.length ? actions[actions.length - 1].id : 0;
    if (stored.id !== last + 1) return this.entries.delete(id); // something else happened too: re-read
    entry.data = Object.assign({}, entry.data, { actions: actions.concat([stored]) });
    entry.at = Date.now();
    return true;
  }
}

function describe(action, names) {
  const who = action.entity_type === 'player' ? names[action.entity] || action.entity : action.entity;
  const extra = ['company', 'corporation', 'hex', 'tile', 'train', 'price', 'kind', 'shares']
    .filter((k) => action[k] !== undefined)
    .map((k) => `${k}=${JSON.stringify(action[k])}`)
    .join(' ');
  return `#${action.id} ${who} ${action.type}${extra ? ` ${extra}` : ''}`;
}

/** One game the bots play: its bot seats, the viewer page that plays them, the log. */
class GameWatch {
  constructor(bot, id) {
    this.bot = bot;
    this.id = String(id);
    this.state = 'new'; // new (waiting for its start) / playing / done
    this.seats = [];
    this.byPlayerId = new Map(); // site player id -> account
    this.names = {};
    this.lastSeen = 0;
    this.turnSince = null;
    this.turnLength = null;
    this.page = null;
    this.lastStatus = null;
    this.waitLogged = false;
    this.restarts = 0; // autoplay switched back on since the game last moved
  }

  async open() {
    await this.bot.pacer.turn(PAGE_SPACING_MS);
    this.page = await this.bot.viewer.newPage();
    this.page.on('pageerror', (e) => log(`game ${this.id}: page error: ${String(e).slice(0, 200)}`));
    this.page.on('crash', () => {
      log(`game ${this.id}: the page crashed; reopening`);
      this.page = null;
    });
    await this.page.goto(`${SITE}/game/${this.id}`, { waitUntil: 'domcontentloaded', timeout: 90_000 });
  }

  async start(game) {
    const players = game.players || [];
    this.names = Object.fromEntries(players.map((p) => [p.id, p.name]));
    for (const [i, p] of players.entries()) {
      const account = this.bot.accounts.find((a) => a.name === String(p.name));
      if (!account) continue;
      this.seats.push(i + 1);
      this.byPlayerId.set(String(p.id), account);
    }
    if (!this.seats.length) {
      log(`game ${this.id}: none of the bots is a player (${players.map((p) => p.name).join(', ')}); skipped`);
      this.state = 'done';
      return;
    }
    this.lastSeen = (game.actions || []).length;
    await this.bot.setOptions({ autoplay: { [this.id]: { enabled: true, seats: this.seats } } });
    await this.open();
    this.state = 'playing';
    const who = this.seats.map((s) => `${s} (${players[s - 1].name})`).join(', ');
    log(`game ${this.id}: playing seat${this.seats.length > 1 ? 's' : ''} ${who} of ${players.length}` +
        `${this.bot.args.dryRun ? ', DRY RUN (nothing posted)' : ''}`);  // prettier-ignore
  }

  async status() {
    if (!this.page) return null;
    try {
      return await this.page.evaluate(() => {
        const host = document.getElementById('rl18xx-advisor');
        return host ? host.getAttribute('data-autoplay') : null;
      });
    } catch (e) {
      return null;
    }
  }

  async check() {
    if (this.state === 'done') return;
    const game = await this.bot.games.get(this.id, MONITOR_MAX_AGE_MS);
    if (String(game.title) !== '1830') {
      log(`game ${this.id} is ${game.title}: the model plays only 1830; skipped`);
      this.state = 'done';
      return;
    }
    if (this.state === 'new') {
      if (game.status === 'new') {
        if (!this.waitLogged) log(`game ${this.id}: waiting for its owner to start it`);
        this.waitLogged = true;
        return;
      }
      if (game.status !== 'active') {
        log(`game ${this.id} is ${game.status}; skipped`);
        this.state = 'done';
        return;
      }
      await this.start(game);
      if (this.state !== 'playing') return;
    }
    if (!this.page) await this.open();
    const actions = game.actions || [];
    for (const action of actions.slice(this.lastSeen)) {
      if (action.type !== 'message') log(`game ${this.id}: ${describe(action, this.names)}`);
    }
    this.lastSeen = Math.max(this.lastSeen, actions.length);
    const status = await this.status();
    if (status && status !== this.lastStatus) log(`game ${this.id}: autoplay ${status}`);
    this.lastStatus = status;
    if (game.status === 'finished' || game.status === 'archived') {
      log(`game ${this.id}: ${game.status} (${JSON.stringify(game.result || {})})`);
      this.state = 'done';
      await this.page.close().catch(() => {});
      this.page = null;
      return;
    }
    const botTurn = (game.acting || []).some((pid) => this.byPlayerId.has(String(pid)));
    const moved = actions.length !== this.turnLength;
    this.turnLength = actions.length;
    if (!botTurn || moved) this.turnSince = botTurn ? Date.now() : null;
    if (moved) this.restarts = 0;
    if (botTurn && !this.bot.args.dryRun && this.turnSince && Date.now() - this.turnSince > STUCK_MS) {
      const shot = path.join(this.bot.args.logDir, `stuck_${this.id}_${Date.now()}.png`);
      await this.page.screenshot({ path: shot }).catch(() => {});
      log(`game ${this.id}: a bot's turn has waited ${Math.round((Date.now() - this.turnSince) / 60000)} min ` +
          `(autoplay: ${status || 'no status'}); screenshot ${shot}. Reloading the page.`);  // prettier-ignore
      // Autoplay switches itself off on an error (the extension saves it off for this game), and
      // a reload doesn't switch it back on: do, a few times while the game doesn't move.
      if (autoplayStopped(status)) {
        if (this.restarts < MAX_AUTOPLAY_RESTARTS) {
          this.restarts += 1;
          log(`game ${this.id}: autoplay is off; switching it back on (${this.restarts} of ${MAX_AUTOPLAY_RESTARTS})`);
          await this.bot.setOptions({ autoplay: { [this.id]: { enabled: true, seats: this.seats } } });
        } else if (this.restarts === MAX_AUTOPLAY_RESTARTS) {
          this.restarts += 1;
          log(`game ${this.id}: autoplay keeps stopping without a move; left off (fix or undo, then restart the runner)`);
        }
      }
      this.turnSince = Date.now();
      await this.bot.pacer.turn(PAGE_SPACING_MS);
      await this.page.reload({ waitUntil: 'domcontentloaded' }).catch(() => {});
    }
  }
}

class Bot {
  constructor(args) {
    this.args = args;
    this.pacer = new Pacer(args.minInterval);
    this.accounts = []; // {name, api}
    this.watches = new Map();
    this.viewer = null;
    this.games = null;
    this.extensionId = null;
  }

  async start() {
    for (const account of accountsOf(this.args)) {
      const storageState = account.cookieFile
        ? {
            cookies: [{ name: 'auth_token', value: readToken(account.cookieFile), domain: new URL(SITE).hostname,
                        path: '/', expires: -1, httpOnly: true, secure: SITE.startsWith('https:'), sameSite: 'Lax' }],
            origins: [],
          }
        : undefined;  // prettier-ignore
      this.accounts.push({ name: account.name, api: await playwright().request.newContext({ storageState }) });
    }
    this.games = new Games(this.pacer, await playwright().request.newContext());
    fs.mkdirSync(this.args.profile, { recursive: true });
    this.viewer = await playwright().chromium.launchPersistentContext(this.args.profile, {
      channel: 'chromium',
      headless: !this.args.headed,
      viewport: { width: 1400, height: 900 },
      args: [`--disable-extensions-except=${EXTENSION}`, `--load-extension=${EXTENSION}`],
    });
    await this.viewer.clearCookies(); // logged out: the accounts post, the viewer only shows the game
    await this.viewer.exposeFunction('rl18xxReadGame', async (id, maxAgeMs) =>
      JSON.stringify(await this.games.get(String(id), Number(maxAgeMs) || 0)),
    );
    await this.viewer.exposeFunction('rl18xxPostAction', (id, move, playerId, stored) =>
      this.post(String(id), JSON.parse(move), playerId, stored ? JSON.parse(stored) : null),
    );
    let [worker] = this.viewer.serviceWorkers();
    if (!worker) worker = await this.viewer.waitForEvent('serviceworker');
    this.extensionId = new URL(worker.url()).host;
    await this.setOptions({
      options: {
        backendUrl: this.args.backend, autoplayLive: true, autoplayDryRun: this.args.dryRun,
        autoplayReadouts: this.args.readouts, autoplayHints: false,
      },
    });  // prettier-ignore
  }

  /** The extension's options, changed from one of its own pages (its service worker may be asleep). */
  async setOptions(change) {
    const page = await this.viewer.newPage();
    try {
      await page.goto(`chrome-extension://${this.extensionId}/options.html`);
      await page.evaluate(async (c) => {
        const stored = await chrome.storage.local.get(null);
        const autoplay = Object.assign({}, stored.autoplay || {}, c.autoplay || {});
        await chrome.storage.local.set(Object.assign({}, c.options || {}, { autoplay }));
      }, change);
    } finally {
      await page.close();
    }
  }

  /** The bridge's move, posted as the account playing ``playerId``: JSON ``{ok, status, error}``. */
  async post(id, move, playerId, stored) {
    const watch = this.watches.get(id);
    const account = watch && watch.byPlayerId.get(String(playerId));
    if (!account) return JSON.stringify({ ok: false, error: `no bot plays player ${playerId} in game ${id}` });
    await this.pacer.turn();
    const response = await account.api.post(`${SITE}/api/game/${id}/action`, {
      headers: { 'Content-Type': 'application/json' },
      data: move,
    });
    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch (e) {
      body = null;
    }
    const ok = response.ok() && !(body && body.error);
    this.games.posted(id, ok ? stored : null); // refused: the next look reads afresh
    const error = ok ? null : (body && body.error) || text.slice(0, 300) || `HTTP ${response.status()}`;
    return JSON.stringify({ ok, status: response.status(), error });
  }

  track(id) {
    if (!this.watches.has(String(id))) this.watches.set(String(id), new GameWatch(this, id));
  }

  /** An account joins a new game (it may already be in it). */
  async join(account, id) {
    const game = await this.games.get(id, 10_000);
    if ((game.players || []).some((p) => String(p.name) === account.name)) {
      log(`[${account.name}] game ${id}: already a player`);
    } else if (game.status !== 'new') {
      log(`[${account.name}] game ${id} has started (${game.status}): it can't be joined any more`);
      return;
    } else if (this.args.dryRun) {
      log(`[${account.name}] game ${id}: DRY RUN, would join (${(game.players || []).length}/${game.max_players} players)`);
      return;
    } else {
      await this.pacer.turn();
      const response = await account.api.post(`${SITE}/api/game/${id}/join`, {
        headers: { 'Content-Type': 'application/json' },
        data: {},
      });
      const body = await response.json().catch(() => ({}));
      this.games.posted(id, null); // the roster changed
      if (!response.ok() || body.error) {
        log(`[${account.name}] game ${id}: could not join: ${body.error || `HTTP ${response.status()}`}`);
        return;
      }
      log(`[${account.name}] game ${id}: joined (${(body.players || []).length}/${body.max_players} players)`);
    }
    this.track(id);
  }

  /** --watch: join the requested games, then track every new or active 1830 game an account is in. */
  async discover() {
    for (const id of takeJoinRequests(this.args.joinFile)) {
      for (const account of this.accounts) await this.join(account, id).catch((e) => log(`game ${id}: ${e.message || e}`));
    }
    for (const account of this.accounts) {
      try {
        await this.pacer.turn();
        const response = await account.api.get(`${SITE}/api/game?title=1830`, { headers: { Accept: 'application/json' } });
        if (!response.ok()) throw new Error(`HTTP ${response.status()}`);
        const { games } = await response.json();
        for (const game of games || []) {
          if (!['new', 'active'].includes(game.status)) continue;
          if (!(game.players || []).some((p) => String(p.name) === account.name)) continue;
          if (!this.watches.has(String(game.id))) log(`game ${game.id}: found (${game.status}, ${account.name} plays)`);
          this.track(game.id);
        }
      } catch (e) {
        log(`[${account.name}] listing its games: ${e.message || e}`);
      }
    }
  }

  async run() {
    for (const id of this.args.joins) {
      for (const account of this.accounts) await this.join(account, id).catch((e) => log(`game ${id}: ${e.message || e}`));
    }
    for (const id of this.args.games) this.track(id);
    let stopping = false;
    let lastDiscovery = 0;
    process.on('SIGINT', () => (stopping = true));
    process.on('SIGTERM', () => (stopping = true));
    while (!stopping) {
      if (this.args.watch && Date.now() - lastDiscovery >= DISCOVER_MS) {
        lastDiscovery = Date.now();
        await this.discover();
      }
      for (const watch of this.watches.values()) {
        try {
          await watch.check();
        } catch (e) {
          log(`game ${watch.id}: ${String((e && e.message) || e).slice(0, 300)}`);
        }
      }
      if (!this.args.watch && [...this.watches.values()].every((w) => w.state === 'done')) {
        log('every game is over');
        return;
      }
      for (let waited = 0; waited < MONITOR_MS && !stopping; waited += 1000) await sleep(1000);
    }
    log('stopping');
  }

  async close() {
    if (this.viewer) await this.viewer.close().catch(() => {});
    for (const account of this.accounts) await account.api.dispose().catch(() => {});
    if (this.games) await this.games.api.dispose().catch(() => {});
  }
}

async function main() {
  const args = parseArgs(process.argv);
  SITE = args.site;
  if (!fs.existsSync(path.join(EXTENSION, 'manifest.json'))) {
    throw new Error(`no built extension at ${EXTENSION}: run python extension/build.py first`);
  }
  fs.mkdirSync(args.logDir, { recursive: true });
  const bot = new Bot(args);
  try {
    await bot.start();
    log(`bots ${bot.accounts.map((a) => JSON.stringify(a.name)).join(', ')} on ${SITE}: backend ${args.backend}, ` +
        `${args.readouts} readouts, at most one request every ${args.minInterval} ms` +
        `${args.watch ? `, watching their games (join file ${args.joinFile})` : ''}${args.dryRun ? ', DRY RUN' : ''}`);  // prettier-ignore
    await bot.run();
  } finally {
    await bot.close();
  }
}

if (require.main === module) {
  main().catch((e) => {
    console.error(new Date().toISOString(), String((e && e.message) || e));
    process.exit(1);
  });
} else {
  module.exports = { Pacer, Games, takeJoinRequests, accountsOf, autoplayStopped };
}
