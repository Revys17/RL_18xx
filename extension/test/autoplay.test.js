// Unit tests for autoplay (src/autoplay.js createAutoplay): when it asks the backend
// for a move, what it hands the page bridge, and when it stops.
const test = require('node:test');
const assert = require('node:assert/strict');
const { createAutoplay } = require('../src/autoplay.js');
const { renderBody, summary } = require('../src/panel.js');
const { fakeClock, settle } = require('./fakes.js');
const { sampleAdvice } = require('./samples.js');

const GAME_ID = 'hs_abcdefgh_1';
const PLAYERS = [
  { id: 0, name: 'Me' },
  { id: 1, name: 'Bot A' },
  { id: 2, name: 'Bot B' },
  { id: 3, name: 'Bot C' },
];

/** A page whose game waits for ``entity``; ``apply`` answers the bridge's apply. */
function setup({ entity = { id: 'PRR', type: 'corporation', player: 1 }, step = 'Track', actions = ['lay_tile', 'pass'], apply, move, finished = false, gameId = GAME_ID } = {}) {
  const clock = fakeClock();
  const page = { lastId: 10, entity, step, actions, finished, applied: [] };
  const sent = [];
  const env = {
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    bridge: async (op, args) => {
      if (op === 'inspect') {
        page.inspects = (page.inspects || 0) + 1;
        page.maxAges = (page.maxAges || []).concat([args.max_age_ms]);
        const data = { id: gameId, players: PLAYERS, actions: Array.from({ length: page.lastId }, (_, i) => ({ id: i + 1, type: 'pass' })) };
        return { ok: true, data, last_id: page.lastId, finished: page.finished, players: PLAYERS, entity: page.entity, step: page.step, actions: page.actions };
      }
      if (op === 'apply') {
        page.applied.push(args);
        const answer = apply ? apply(args, page) : { ok: true };
        if (answer.ok) {
          page.lastId += 1;
          page.entity = { id: 0, type: 'player', player: 0 }; // the human's turn next
          answer.action = Object.assign({ id: page.lastId }, args.action);
        }
        return answer;
      }
      return { ok: false, error: 'unexpected' };
    },
    send: async (message) => {
      sent.push(message);
      if (message.type === 'move') {
        if (move) return move(message);
        return { ok: true, data: { job: 'j1', status: 'running', progress: 0.2 } };
      }
      if (message.type === 'move_status') {
        return {
          ok: true,
          data: {
            job: 'j1',
            status: 'done',
            result: {
              kind: 'search',
              action: { type: 'lay_tile', entity: 'PRR', entity_type: 'corporation', hex: 'F20', tile: '57-0', rotation: 1 },
              actor: 'PRR (Bot A)',
              description: 'Lay tile #57 on F20',
              visits: 201,
              share: 0.6,
            },
          },
        };
      }
      return { ok: false, error: 'unexpected' };
    },
    onChange: () => {},
  };
  const autoplay = createAutoplay(env);
  return { autoplay, page, sent, clock };
}

test('on a model seat it searches, then hands the move to the bridge with the last action id', async () => {
  const { autoplay, page, sent, clock } = setup();
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2, 3, 4], readouts: 400 });
  await clock.advance(2000);
  const move = sent.find((m) => m.type === 'move');
  assert.deepEqual(move.seats, [2, 3, 4]);
  assert.equal(move.readouts, 400);
  assert.equal(move.site.last_id, 10);
  assert.equal(move.site.entity.id, 'PRR');
  assert.equal(move.game.actions.length, 10);
  assert.equal(page.applied.length, 1);
  assert.equal(page.applied[0].last_id, 10);
  assert.equal(page.applied[0].action.hex, 'F20');
  const view = autoplay.view();
  assert.equal(view.moves[0].description, 'Lay tile #57 on F20');
  assert.equal(view.moves[0].id, 11);
  // Then it's the human's turn: it waits, asking only the page.
  assert.equal(view.phase, 'waiting');
  assert.match(view.text, /Me to move/);
  const before = sent.length;
  await clock.advance(10000);
  assert.equal(sent.length, before);
});

test("on a human seat's turn it never asks the backend", async () => {
  const { autoplay, sent, clock } = setup({ entity: { id: 0, type: 'player', player: 0 } });
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2, 3, 4] });
  await clock.advance(10000);
  assert.equal(sent.length, 0);
  assert.equal(autoplay.view().phase, 'waiting');
});

test('switched off or with the history view open it does nothing', async () => {
  const { autoplay, sent, page, clock } = setup();
  autoplay.setGame(GAME_ID, { enabled: false, seats: [2, 3, 4] });
  await clock.advance(5000);
  assert.equal(sent.length, 0);
  autoplay.setPaused(true);
  autoplay.configure({ enabled: true });
  await clock.advance(5000);
  assert.equal(sent.length, 0);
  autoplay.setPaused(false);
  await clock.advance(3000);
  assert.equal(page.applied.length, 1);
});

test('a move the game outran is not played; the next look decides again', async () => {
  let applies = 0;
  const { autoplay, page, clock } = setup({
    apply: () => {
      applies += 1;
      return applies === 1 ? { ok: false, stale: true } : { ok: true };
    },
  });
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2] });
  await clock.advance(3000);
  assert.equal(applies, 2);
  assert.equal(page.lastId, 11);
  assert.equal(autoplay.view().phase, 'waiting');
});

test('a move Ruby refuses in a purchase step becomes the pass Ruby waits for', async () => {
  const { autoplay, page, clock } = setup({
    step: 'BuyTrain',
    actions: ['buy_train', 'pass'],
    apply: (args) => (args.action.type === 'pass' ? { ok: true } : { ok: false, invalid: true, error: 'GameError: Not your turn' }),
  });
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2] });
  await clock.advance(3000);
  assert.deepEqual(
    page.applied.map((a) => a.action.type),
    ['lay_tile', 'pass'],
  );
  assert.deepEqual(page.applied[1].action, { type: 'pass', entity: 'PRR', entity_type: 'corporation' });
  assert.equal(autoplay.view().moves[0].kind, 'implicit_pass');
});

test('any other refusal stops autoplay and says why', async () => {
  const { autoplay, page, clock } = setup({
    apply: () => ({ ok: false, invalid: true, error: 'GameError: Cannot lay tile' }),
  });
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2] });
  await clock.advance(3000);
  assert.equal(page.applied.length, 1);
  const view = autoplay.view();
  assert.equal(view.phase, 'error');
  assert.equal(view.enabled, false);
  assert.match(view.text, /refused.*Cannot lay tile/);
  await clock.advance(10000);
  assert.equal(page.applied.length, 1);
});

test('a busy backend is asked again; an autoplay error stops', async () => {
  let calls = 0;
  const { autoplay, page, clock } = setup({
    move: () => {
      calls += 1;
      if (calls === 1) return { ok: false, status: 409, error: 'A search is already running', data: { busy: true } };
      return { ok: true, data: { job: 'j1', status: 'running', progress: 0 } };
    },
  });
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2] });
  await clock.advance(5000);
  assert.equal(calls, 2);
  assert.equal(page.applied.length, 1);

  const other = setup({ move: () => ({ ok: false, status: 409, error: 'The engines disagree', data: { autoplay: true } }) });
  other.autoplay.setGame(GAME_ID, { enabled: true, seats: [2] });
  await other.clock.advance(5000);
  assert.equal(other.autoplay.view().phase, 'error');
  assert.match(other.autoplay.view().text, /disagree/);
});

test('a finished game ends autoplay', async () => {
  const { autoplay, sent, clock } = setup({ finished: true });
  autoplay.setGame(GAME_ID, { enabled: true, seats: [2, 3, 4] });
  await clock.advance(5000);
  assert.equal(sent.length, 0);
  assert.equal(autoplay.view().phase, 'over');
});

test('the panel: seats, start, and hints hidden while the model plays', async () => {
  const advice = sampleAdvice();
  const players = PLAYERS.map((p, i) => ({ seat: i + 1, name: p.name }));
  const off = { enabled: false, seats: [2, 3], readouts: 200, phase: 'off', text: '', progress: null, moves: [], players, hints: false };
  let html = renderBody({ statuses: [], advice, think: { state: 'idle' }, readouts: 200, autoplay: off });
  assert.match(html, /data-act="seat" data-seat="2" checked/);
  assert.match(html, /data-act="seat" data-seat="1" (?!checked)/);
  assert.match(html, />Start</);
  assert.match(html, /Model's moves/);
  const on = Object.assign({}, off, {
    enabled: true,
    phase: 'thinking',
    text: 'PRR (Bot A) is thinking…',
    progress: 0.5,
    moves: [{ id: 11, kind: 'search', actor: 'PRR (Bot A)', description: 'Lay tile #57 on F20', visits: 201, share: 0.6 }],
  });
  html = renderBody({ statuses: [], advice, think: { state: 'idle' }, readouts: 200, autoplay: on });
  assert.match(html, />Pause</);
  assert.match(html, /disabled/);
  assert.match(html, /201 visits/);
  assert.doesNotMatch(html, /Model's moves/);
  assert.doesNotMatch(html, /Think harder/);
  assert.match(html, /Win chances/);
  assert.equal(summary({ advice, autoplay: on }), 'PRR (Bot A) is thinking…');
  html = renderBody({ statuses: [], advice, think: { state: 'idle' }, readouts: 200, autoplay: Object.assign({}, on, { hints: true }) });
  assert.match(html, /Model's moves/);
  html = renderBody({ statuses: [], advice, think: { state: 'idle' }, readouts: 200, autoplay: Object.assign({}, off, { players: [{ seat: 1, name: '<b>x</b>' }] }) });
  assert.doesNotMatch(html, /<b>x<\/b>/);
  await settle();
});

test('a live game in dry run checks the move, shows it, and waits for the game to change', async () => {
  const t = setup({
    gameId: 123456,
    apply: (args) => (args.dry_run ? { ok: false, dry_run: true, action: args.action } : { ok: true }),
  });
  t.autoplay.setGame(123456, { enabled: true, seats: [2], dryRun: true });
  await t.clock.advance(3000);
  assert.equal(t.page.applied.length, 1);
  assert.equal(t.page.applied[0].dry_run, true);
  const view = t.autoplay.view();
  assert.equal(view.live, true);
  assert.match(view.text, /Dry run — would play PRR \(Bot A\): Lay tile #57 on F20/);
  const moves = t.sent.filter((m) => m.type === 'move').length;
  await t.clock.advance(60000); // the same position: shown once, not searched again
  assert.equal(t.sent.filter((m) => m.type === 'move').length, moves);
  assert.equal(t.page.applied.length, 1);
});

test("a live game is read every 30 s while it is someone else's turn (a hotseat game every 1.5 s)", async () => {
  const live = setup({ gameId: 123456, entity: { id: 0, type: 'player', player: 0 } });
  live.autoplay.setGame(123456, { enabled: true, seats: [2] });
  await live.clock.advance(61000);
  assert.equal(live.page.inspects, 3); // at 0, 30 and 60 s
  // The first look follows a change (fresh); the timed ones may share a recent read.
  assert.deepEqual(live.page.maxAges, [1000, 15000, 15000]);
  live.autoplay.poke();
  await live.clock.advance(100);
  assert.equal(live.page.maxAges[3], 1000);
  assert.equal(live.sent.length, 0);
  assert.match(live.autoplay.view().text, /^Me to move\.$/);
  const hotseat = setup({ entity: { id: 0, type: 'player', player: 0 } });
  hotseat.autoplay.setGame(GAME_ID, { enabled: true, seats: [2] });
  await hotseat.clock.advance(61000);
  assert.ok(hotseat.page.inspects > 30);
  assert.match(hotseat.autoplay.view().text, /your turn/);
});
