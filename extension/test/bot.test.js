// Unit tests for the live bot's traffic control (bot/run.js): one paced queue for every request,
// and one shared copy of each game.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Pacer, Games, takeJoinRequests, accountsOf } = require('../bot/run.js');

test('the pacer spaces requests at least its interval apart, in order', async () => {
  const pacer = new Pacer(1000);
  const times = [];
  const t0 = Date.now();
  // Real time but short: shrink the interval for the test.
  pacer.interval = 60;
  await Promise.all([0, 1, 2, 3].map(() => pacer.turn().then(() => times.push(Date.now() - t0))));
  for (let i = 1; i < times.length; i += 1) assert.ok(times[i] - times[i - 1] >= 55, `gap ${times[i] - times[i - 1]} ms`);
  const before = Date.now();
  await pacer.turn(150); // a page load holds the queue longer
  await pacer.turn();
  assert.ok(Date.now() - before >= 145);
});

function fakeSite(game) {
  const site = { reads: 0, game };
  site.api = {
    get: async () => {
      site.reads += 1;
      const data = JSON.parse(JSON.stringify(site.game));
      return { ok: () => true, status: () => 200, json: async () => data };
    },
  };
  return site;
}

test('one read serves every seat; a fresh look after a change reads again', async () => {
  const site = fakeSite({ id: 1, actions: [{ id: 1, type: 'pass', entity: 1 }] });
  const pacer = new Pacer(1000);
  pacer.interval = 5;
  const games = new Games(pacer, site.api);
  const [a, b, c] = await Promise.all([games.get('1', 1000), games.get('1', 1000), games.get('1', 15000)]);
  assert.equal(site.reads, 1); // three seats' looks, one request
  assert.deepEqual(a, b);
  assert.deepEqual(b, c);
  await games.get('1', 15000);
  assert.equal(site.reads, 1); // recent enough
  await new Promise((r) => setTimeout(r, 30));
  await games.get('1', 10);
  assert.equal(site.reads, 2); // too old for a fresh look
});

test("a posted move updates the shared copy; anything else makes the next look read", async () => {
  const site = fakeSite({ id: 1, actions: [{ id: 1, type: 'pass', entity: 1 }] });
  const pacer = new Pacer(1000);
  pacer.interval = 5;
  const games = new Games(pacer, site.api);
  await games.get('1', 1000);
  games.posted('1', { id: 2, type: 'bid', entity: 2 });
  const after = await games.get('1', 1000);
  assert.equal(site.reads, 1); // no re-read for our own move
  assert.deepEqual(after.actions.map((x) => x.id), [1, 2]);
  games.posted('1', { id: 5, type: 'pass', entity: 3 }); // not the next action: something else happened
  await games.get('1', 60000);
  assert.equal(site.reads, 2);
  games.posted('1', null); // a refused move or a join
  await games.get('1', 60000);
  assert.equal(site.reads, 3);
});

test('the join file is read once and emptied; accounts come from <name>.token files', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bot-'));
  const file = path.join(dir, 'join');
  fs.writeFileSync(file, '274001\n 274002 junk\n');
  assert.deepEqual(takeJoinRequests(file), ['274001', '274002']);
  assert.deepEqual(takeJoinRequests(file), []);
  fs.writeFileSync(path.join(dir, 'Bot One.token'), 'x');
  fs.writeFileSync(path.join(dir, 'Bot2.token'), 'y');
  const accounts = accountsOf({ player: null, accountsDir: dir });
  assert.deepEqual(accounts.map((a) => a.name), ['Bot One', 'Bot2']);
  assert.deepEqual(accountsOf({ player: 'Solo', cookieFile: '/c' }), [{ name: 'Solo', cookieFile: '/c' }]);
});
