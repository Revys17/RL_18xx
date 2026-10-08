// Unit tests for the content script's logic (src/content.js createController):
// when it fetches the game, what it sends, what the panel shows.
const test = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/common.js');
const { createController, mergeOptions } = require('../src/content.js');
const { fakeDocument, fakeClock, settle } = require('./fakes.js');
const { sampleAdvice } = require('./samples.js');

const GAME = {
  id: 123,
  title: '1830',
  players: [{ id: 101, name: 'Player 1' }],
  actions: [{ type: 'bid', id: 1 }, { type: 'message', message: 'gl hf', id: 2 }],
};

function setup({ pathname = '/game/123', advise = { ok: true, data: sampleAdvice() }, thinkStatus = [] } = {}) {
  const clock = fakeClock();
  const document = fakeDocument({ log: ['-- Operating Round 1.1 --'], hexes: [[186.6, 250], [1918.65, 850]] });
  const location = { pathname };
  const fetches = [];
  const sent = [];
  const panels = [];
  const statuses = [...thinkStatus];
  const env = {
    location,
    document,
    now: clock.now,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    options: mergeOptions({}),
    saveOptions: () => {},
    fetchGame: async (id) => {
      fetches.push({ id, t: clock.t });
      return JSON.parse(JSON.stringify(GAME));
    },
    send: async (message) => {
      sent.push(message);
      if (message.type === 'advise') return typeof advise === 'function' ? advise(message) : advise;
      if (message.type === 'think') return { ok: true, data: { job: 'j1', status: 'running', progress: 0 } };
      if (message.type === 'think_status') return statuses.shift();
      return { ok: false, error: 'unexpected' };
    },
    makePanel: (callbacks, placement) => {
      const panel = {
        callbacks,
        placement,
        views: [],
        destroyed: false,
        update(view) {
          this.views.push(JSON.parse(JSON.stringify(view)));
        },
        setPlacement() {},
        destroy() {
          this.destroyed = true;
        },
      };
      panels.push(panel);
      return panel;
    },
  };
  const controller = createController(env);
  const lastView = () => panels[panels.length - 1].views.slice(-1)[0];
  return { clock, document, location, fetches, sent, panels, controller, lastView };
}

test('opening a game fetches it once, sends it without chat, and shows the advice', async () => {
  const t = setup();
  t.controller.checkLocation();
  await settle();
  assert.equal(t.fetches.length, 1);
  assert.equal(t.fetches[0].id, 123);
  const advise = t.sent.filter((m) => m.type === 'advise');
  assert.equal(advise.length, 1);
  assert.deepEqual(advise[0].game.actions.map((a) => a.type), ['bid'], 'the chat line stays in the page');
  assert.equal(t.panels.length, 1);
  assert.equal(t.lastView().advice.moves[0].hex, 'F22');
  assert.deepEqual(t.lastView().statuses, []);
  t.controller.checkLocation(); // same game: nothing new
  await settle();
  assert.equal(t.fetches.length, 1);
});

test('the top move\'s hex is outlined on the map, and hovering another move moves the outline', async () => {
  const t = setup();
  t.controller.checkLocation();
  await settle();
  const outlined = t.document.hexGroups[1].children;
  assert.equal(outlined.length, 1);
  assert.equal(outlined[0].attributes.points, R.HEX_POINTS);
  t.panels[0].callbacks.onHoverMove({ map: [186.6, 250] });
  assert.equal(t.document.hexGroups[1].children.length, 0);
  assert.equal(t.document.hexGroups[0].children.length, 1);
  t.panels[0].callbacks.onHoverMove(null); // back to the top move
  assert.equal(t.document.hexGroups[1].children.length, 1);
});

test('a new log line triggers a fetch, but never sooner than 10 s after the last one', async () => {
  const t = setup();
  t.controller.checkLocation();
  await settle();
  t.controller.onMutations(); // DOM noise, the log unchanged
  await t.clock.advance(2_000);
  assert.equal(t.fetches.length, 1);

  t.document.log.push('PRR lays tile #57 with rotation 1 on F22');
  t.controller.onMutations();
  await t.clock.advance(2_000); // debounced, then held back by the 10 s spacing
  assert.equal(t.fetches.length, 1);
  await t.clock.advance(6_000);
  assert.equal(t.fetches.length, 2);
  assert.ok(t.fetches[1].t - t.fetches[0].t >= R.MIN_FETCH_INTERVAL_MS);
});

test('a hotseat game is read again on every new log line, with no 10 s spacing', async () => {
  const t = setup({ pathname: '/hotseat/hs_abcdefgh_1' });
  t.controller.checkLocation();
  await settle();
  assert.deepEqual(t.fetches.map((f) => f.id), ['hs_abcdefgh_1']);
  t.document.log.push('Player 1 bids $45 on C&A');
  t.controller.onMutations();
  await t.clock.advance(2_000); // just the debounce
  assert.equal(t.fetches.length, 2);
  t.controller.refresh();
  await settle();
  assert.equal(t.fetches.length, 3);
});

test('a burst of triggers still means at most one fetch per 10 s', async () => {
  const t = setup();
  t.controller.checkLocation();
  await settle();
  for (let i = 0; i < 60; i += 1) {
    t.document.log.push(`line ${i}`);
    t.controller.onMutations();
    t.controller.refresh();
    await t.clock.advance(500);
  }
  await t.clock.advance(20_000);
  const times = t.fetches.map((f) => f.t);
  for (let i = 1; i < times.length; i += 1) assert.ok(times[i] - times[i - 1] >= R.MIN_FETCH_INTERVAL_MS);
  assert.ok(times.length >= 3 && times.length <= 6, `fetched ${times.length} times in 50 s`);
});

test('the refresh button waits out the 10 s and says so', async () => {
  const t = setup();
  t.controller.checkLocation();
  await settle();
  await t.clock.advance(3_000);
  t.controller.refresh();
  assert.match(t.lastView().statuses[0].text, /Refreshing in 7 s/);
  assert.equal(t.fetches.length, 1);
  await t.clock.advance(7_000);
  assert.equal(t.fetches.length, 2);
});

test('a hidden tab waits until it is shown', async () => {
  const t = setup();
  t.document.hidden = true;
  t.controller.checkLocation();
  await settle();
  assert.equal(t.fetches.length, 0);
  t.document.hidden = false;
  t.controller.onVisibility();
  await settle();
  assert.equal(t.fetches.length, 1);
});

test('leaving the game removes the panel; another game is fetched afresh', async () => {
  const t = setup();
  t.controller.checkLocation();
  await settle();
  t.location.pathname = '/';
  t.controller.checkLocation();
  assert.equal(t.panels[0].destroyed, true);
  t.location.pathname = '/game/124';
  t.controller.checkLocation();
  await t.clock.advance(10_000);
  assert.deepEqual(t.fetches.map((f) => f.id), [123, 124]);
});

test('switching games while a fetch is under way still fetches the new game', async () => {
  let release;
  const t = setup({
    advise: () =>
      new Promise((resolve) => {
        release = () => resolve({ ok: true, data: sampleAdvice() });
      }),
  });
  t.controller.checkLocation();
  await settle(); // game 123's advice is pending
  t.location.pathname = '/game/124';
  t.controller.checkLocation();
  release();
  await t.clock.advance(10_000);
  assert.deepEqual(t.fetches.map((f) => f.id), [123, 124]);
  assert.equal(t.panels[0].destroyed, true);
  assert.equal(t.panels[1].views.slice(-1)[0].advice, null, "game 123's late answer isn't shown for 124");
  release(); // game 124's answer
  await settle();
  assert.ok(t.panels[1].views.slice(-1)[0].advice);
});

test('status lines for an unreachable backend and an unsupported game', async () => {
  const down = setup({ advise: { ok: false, unreachable: true, error: 'can\'t reach the advisor backend' } });
  down.controller.checkLocation();
  await settle();
  assert.equal(down.lastView().statuses[0].kind, 'error');
  assert.match(down.lastView().statuses[0].text, /^Backend unreachable/);

  const other = setup({ advise: { ok: true, data: { supported: false, reason: '1867 isn\'t supported' } } });
  other.controller.checkLocation();
  await settle();
  assert.match(other.lastView().statuses[0].text, /Unsupported game: 1867/);
});

test('think harder: starts a search for the advised position and follows it to the end', async () => {
  const result = { visits: 64, moves: [], win: [] };
  const t = setup({
    thinkStatus: [
      { ok: true, data: { status: 'running', progress: 0.5, position: 'abc' } },
      { ok: true, data: { status: 'done', progress: 1, result, position: 'abc' } },
    ],
  });
  t.controller.checkLocation();
  await settle();
  await t.controller.think();
  const start = t.sent.find((m) => m.type === 'think');
  assert.deepEqual(start, { type: 'think', game_id: 123, position: 'abc', readouts: 200 });
  await t.clock.advance(1_000);
  assert.equal(t.lastView().think.state, 'running');
  assert.equal(t.lastView().think.progress, 0.5);
  await t.clock.advance(1_000);
  assert.equal(t.lastView().think.state, 'done');
  assert.deepEqual(t.lastView().think.result, result);
});

test('switched off, it does nothing', async () => {
  const t = setup();
  t.controller.setOptions(mergeOptions({ enabled: false }));
  t.controller.checkLocation();
  await settle();
  assert.equal(t.fetches.length, 0);
  assert.equal(t.panels.length, 0);
});
