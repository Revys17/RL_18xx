// The page bridge (autoplay): runs in the PAGE's JavaScript world (manifest "world": "MAIN"),
// where the site's own Ruby engine (compiled by Opal) lives. The content script (isolated world)
// asks it with window.postMessage:
//
//   {source: 'rl18xx-advisor', id, op: 'inspect', game}
//       -> who the site's engine waits for in game ``game`` (the id of the game open in this page):
//          ``last_id``, ``finished``, ``entity`` {id, type, player}, ``step``, the ``actions`` it
//          offers, ``players``, and the game itself (``data``). A hotseat game is read from
//          localStorage, where the site keeps it; a live game from the site (GET /api/game/<id>,
//          what the page itself loads).
//   {source: 'rl18xx-advisor', id, op: 'apply', game, action, last_id, dry_run}
//       -> plays ``action`` (18xx.games format, unnumbered), refused when Ruby refuses it or the game
//          no longer ends at action ``last_id``. Hotseat: as the site's own UI does
//          (Engine::Game#process_action), saved to localStorage with the site's bookkeeping and shown
//          through the site's back/forward handler (no reload). Live: checked with the page's engine,
//          then posted as the site's page posts a move (POST /api/game/<id>/action, this browser's
//          login) -- the server checks it again with its own engine and publishes it, which updates
//          this page. With ``dry_run`` a live move is only checked, never posted.
//
// and answers {source: 'rl18xx-bridge', id, ok, ...}. It only touches the game open in this page.
//
// Under the live bot (extension/bot/run.js) a live game is read and its moves posted through the
// runner instead (``window.rl18xxReadGame(id, maxAgeMs)`` / ``window.rl18xxPostAction(id, move,
// playerId, stored)``, functions it gives the page): one paced request queue for every bot account,
// one read shared by every bot seat, and each move posted by the account of the seat that makes it.
(function (root) {
  'use strict';

  const REQUEST = 'rl18xx-advisor';
  const REPLY = 'rl18xx-bridge';
  let cache = null; // {id, keys: [action keys], game: Opal game} -- the last game read

  function engine() {
    const Opal = root.Opal;
    if (!Opal || !Opal.Engine || !Opal.Engine.Game || !Opal.Hash) {
      throw new Error("the site's game engine isn't loaded yet");
    }
    return Opal;
  }

  function isNil(value) {
    return value === null || value === undefined || (root.Opal && value === root.Opal.nil);
  }

  /** The game open in this page: ``{id, live}`` (a hotseat ``hs_...`` id or a live game's number), or null. */
  function pageGame() {
    const path = root.location.pathname || '';
    const hotseat = /^\/hotseat\/(hs_[A-Za-z0-9_]+)(?:[/?#]|$)/.exec(path);
    if (hotseat) return { id: hotseat[1], live: false };
    const live = /^\/game\/(\d+)(?:[/?#]|$)/.exec(path);
    return live ? { id: live[1], live: true } : null;
  }

  async function readData(game, maxAgeMs) {
    if (!game.live) {
      const stored = root.localStorage.getItem(game.id);
      if (!stored) throw new Error(`hotseat game ${game.id} isn't in this browser's storage`);
      const data = JSON.parse(stored);
      if (!data || !Array.isArray(data.actions)) throw new Error(`hotseat game ${game.id} has no action list`);
      return data;
    }
    if (typeof root.rl18xxReadGame === 'function') {
      const data = JSON.parse(await root.rl18xxReadGame(String(game.id), Number(maxAgeMs) || 0));
      if (!data || !Array.isArray(data.actions)) throw new Error(`game ${game.id} came without its actions`);
      return data;
    }
    const response = await root.fetch(`/api/game/${encodeURIComponent(game.id)}`, {
      credentials: 'same-origin',
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) throw new Error(`the site answered HTTP ${response.status}`);
    const data = await response.json();
    if (!data || !Array.isArray(data.actions)) throw new Error(`game ${game.id} came without its actions`);
    return data;
  }

  function lastId(actions) {
    for (let i = actions.length - 1; i >= 0; i -= 1) {
      if (actions[i] && typeof actions[i].id === 'number') return actions[i].id;
    }
    return 0;
  }

  /** An action's identity: ids only grow (an undo is itself an action), so the list a cached game
   * has seen is a prefix of the next list read exactly when these keys agree. */
  function actionKey(action) {
    return `${action && action.id}|${action && action.type}|${action && action.entity}`;
  }

  function broken(game) {
    return !isNil(game.$exception());
  }

  /** The game's data and an Opal game at its last action: the cached one moved on by the new actions
   * when the list extends what it saw, else loaded afresh. */
  async function sync(game, maxAgeMs = 0) {
    const Opal = engine();
    const data = await readData(game, maxAgeMs);
    const keys = data.actions.map(actionKey);
    let opal = null;
    if (cache && cache.id === game.id && cache.keys.length <= keys.length && cache.keys.every((k, i) => k === keys[i])) {
      opal = cache.game;
      for (let i = cache.keys.length; i < keys.length && !broken(opal); i += 1) {
        opal = opal.$process_action(Opal.Hash.$new(data.actions[i]));
      }
    } else {
      opal = Opal.Engine.Game.$load(Opal.Hash.$new(data));
    }
    if (broken(opal)) {
      cache = null;
      throw new Error(`the site's engine can't load this game: ${String(opal.$exception())}`);
    }
    cache = { id: game.id, keys, game: opal };
    return { data, opal };
  }

  function entityType(entity) {
    if (entity['$player?']()) return 'player';
    if (entity['$corporation?']()) return 'corporation';
    if (entity['$company?']()) return 'company';
    return 'other';
  }

  /** Who the game waits for, as the backend's ``autoplay.SiteView`` reads it, and the game itself. */
  function view(opal, data) {
    const out = {
      ok: true,
      game: data.id,
      data,
      last_id: lastId(data.actions),
      finished: !!opal.$finished() || data.status === 'finished',
      players: (data.players || []).map((p) => ({ id: p.id, name: p.name })),
      entity: null,
      step: null,
      actions: [],
      acting: Array.from(opal.$active_players_id() || []),
      round: String(opal.$round().$name()),
      turn: opal.$turn(),
    };
    if (out.finished) return out;
    const step = opal.$round().$active_step();
    const active = isNil(step) ? [] : step.$active_entities() || [];
    const entity = active.length && !isNil(active[0]) ? active[0] : opal.$current_entity();
    if (isNil(entity)) return out;
    const player = entity.$player();
    out.entity = { id: entity.$id(), type: entityType(entity), player: isNil(player) ? null : player.$id() };
    if (!isNil(step)) {
      out.step = String(step.$class().$name()).split('::').pop();
      out.actions = Array.from(step.$actions(entity) || []).map(String);
    }
    // Which train is whose, as the site's engine has it: the backend matches the train ids of its
    // moves to these (rl18xx/agent/advisor/autoplay.py, match_train_ids).
    out.trains = {};
    for (const corp of Array.from(opal.$corporations() || [])) {
      const trains = Array.from(corp.$trains() || []);
      if (trains.length) out.trains[String(corp.$id())] = trains.map((t) => ({ id: String(t.$id()), name: String(t.$name()) }));
    }
    out.depot = Array.from(opal.$depot().$depot_trains() || []).map((t) => ({ id: String(t.$id()), name: String(t.$name()) }));
    return out;
  }

  async function inspect(game, maxAgeMs) {
    const { data, opal } = await sync(game, maxAgeMs);
    return view(opal, data);
  }

  /** The site's own bookkeeping of a hotseat game after an action (assets/app/view/game/actionable.rb). */
  function record(data, opal, stored) {
    data.actions.push(stored);
    if (opal.$finished()) {
      data.result = opal.$result().$to_n();
      data.manually_ended = isNil(opal.$manually_ended()) ? null : opal.$manually_ended();
      data.status = 'finished';
      data.game_end_reason = isNil(opal.$game_end_reason()) ? null : String(opal.$game_end_reason());
    } else {
      data.result = {};
      data.status = 'active';
      data.manually_ended = null;
      data.game_end_reason = null;
    }
    data.turn = opal.$turn();
    data.round = String(opal.$round().$name());
    data.acting = Array.from(opal.$active_players_id() || []);
    data.updated_at = Math.floor(Date.now() / 1000);
  }

  /** Show the page a hotseat game without reloading it: the site's popstate handler (App#on_hash_change)
   * takes a history entry's state as the game, and its game page reloads the engine when the last
   * action changed. */
  function showPage(data) {
    const url = root.location.pathname + root.location.search + root.location.hash;
    const state = Object.assign({}, data, { loaded: true });
    try {
      root.history.replaceState(state, '', url);
      root.history.pushState(state, '', url);
      root.history.back();
    } catch (e) {
      root.location.reload();
    }
  }

  /** POST a live game's move as the site's page does (or through the live bot's runner, as the
   * account playing ``playerId``); the server's error, if any. */
  async function post(game, move, playerId, stored) {
    if (typeof root.rl18xxPostAction === 'function') {
      const reply = JSON.parse(
        await root.rl18xxPostAction(String(game.id), JSON.stringify(move), playerId, JSON.stringify(stored)),
      );
      return reply.ok ? null : reply.error || `HTTP ${reply.status}`;
    }
    const response = await root.fetch(`/api/game/${encodeURIComponent(game.id)}/action`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(move),
    });
    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch (e) {
      body = null;
    }
    if (!response.ok || (body && body.error)) {
      return (body && body.error) || text.slice(0, 300) || `HTTP ${response.status}`;
    }
    return null;
  }

  async function apply(game, action, expectedLastId, dryRun) {
    const Opal = engine();
    if (!action || typeof action !== 'object' || typeof action.type !== 'string') {
      return { ok: false, error: 'not an action' };
    }
    const { data, opal } = await sync(game, 1000);
    if (lastId(data.actions) !== expectedLastId) return { ok: false, stale: true, view: view(opal, data) };
    const acting = view(opal, data).entity;
    const move = Object.assign({}, action);
    delete move.created_at;
    move.id = expectedLastId + 1;
    const next = opal.$process_action(Opal.Hash.$new(move));
    cache = null; // whatever happens next, the stored or posted game is read afresh
    if (broken(next)) return { ok: false, invalid: true, error: String(next.$exception()) };
    const stored = next.$raw_actions().$last().$to_n();
    if (game.live) {
      if (dryRun) return { ok: false, dry_run: true, action: stored };
      const refused = await post(game, move, acting && acting.player, stored);
      if (refused) return { ok: false, invalid: true, error: `the server refused it: ${refused}` };
      return { ok: true, action: stored, live: true };
    }
    record(data, next, stored);
    root.localStorage.setItem(game.id, JSON.stringify(data));
    cache = { id: game.id, keys: data.actions.map(actionKey), game: next };
    showPage(data);
    return { ok: true, action: stored };
  }

  async function handle(message) {
    const game = pageGame();
    if (!game || String(message.game) !== game.id) return { ok: false, error: 'not the game open in this page' };
    if (message.op === 'inspect') return inspect(game, message.max_age_ms);
    if (message.op === 'apply') return apply(game, message.action, message.last_id, !!message.dry_run);
    return { ok: false, error: `unknown request ${message.op}` };
  }

  root.addEventListener('message', async (event) => {
    const message = event.data;
    if (event.source !== root || !message || message.source !== REQUEST) return;
    let reply;
    try {
      reply = await handle(message);
    } catch (e) {
      cache = null;
      reply = { ok: false, error: String((e && e.message) || e) };
    }
    root.postMessage(Object.assign({ source: REPLY, id: message.id }, reply), root.location.origin);
  });
})(window);
