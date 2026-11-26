// Smoke test of the browser package (docs/SITE_ENGINE_RS.md §3.2, K1/K2): loads the no-modules glue the way a
// worker's importScripts does (a vm context; `wasm_bindgen.initSync` on the bytes), then for each game: version,
// load, replay to the end, view, shadowKey (must equal `site_engine state`'s), menu of every seated player,
// programInfo, and the native-vs-wasm digest of every step's JSON when SITE_ENGINE is given.
//
//   node scripts/site_engine/pkg_smoke.cjs <pkg dir> <game.json | dir>... [--native <site_engine>]
//   node scripts/site_engine/pkg_smoke.cjs <pkg dir> <game.json | dir>... --digest   (= site_replay --digest)
const fs = require('fs'), vm = require('vm'), path = require('path'), crypto = require('crypto');
const { execFileSync } = require('child_process');
const args = process.argv.slice(2);
const pkg = args.shift();
let native = null, digestOnly = false;
const games = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--native') native = args[++i];
  else if (args[i] === '--digest') digestOnly = true;
  else if (fs.statSync(args[i]).isDirectory()) {
    for (const f of fs.readdirSync(args[i]).filter((x) => x.endsWith('.json')).sort()) games.push(path.join(args[i], f));
  } else games.push(args[i]);
}
// a worker-like global: the glue runs in its own realm (as importScripts would), so the bytes are copied into it
const ctx = { console, TextDecoder, TextEncoder };
ctx.self = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(pkg, 'engine_rs.js'), 'utf8') + '\nself.wasm_bindgen = wasm_bindgen;', ctx);
ctx.__bytes = fs.readFileSync(path.join(pkg, 'engine_rs_bg.wasm'));
vm.runInContext('wasm_bindgen.initSync({ module: new Uint8Array(__bytes) })', ctx);
const wb = ctx.wasm_bindgen;
const version = JSON.parse(wb.WasmGame.version());
if (version.api !== 2) throw new Error(`api ${version.api}`);
console.log('version', JSON.stringify(version));
let failures = 0;
if (digestOnly) {
  // `<file> <digest> <steps>` per game, as `site_replay --digest` prints (the native build)
  for (const file of games) {
    try {
      const d = JSON.parse(wb.WasmGame.debugDigest(fs.readFileSync(file, 'utf8')));
      console.log(`${path.basename(file)} ${d.digest} ${d.steps}`);
    } catch (e) {
      console.log(`${path.basename(file)} error ${String(e.message).split(':')[0]}`);
    }
  }
  process.exit(0);
}
for (const file of games) {
  const text = fs.readFileSync(file, 'utf8');
  const data = JSON.parse(text);
  const t0 = Date.now();
  const g = wb.WasmGame.load(text);
  let steps = 0, out;
  do {
    out = JSON.parse(g.replay(500));
    steps += out.steps.length;
  } while (!out.done);
  const ms = Date.now() - t0;
  const snap = JSON.parse(g.view());
  const key = JSON.parse(g.shadowKey());
  const menus = data.players.map((p) => g.menu(p.id)).filter((m) => m !== 'null').length;
  const pi = JSON.parse(g.programInfo(data.players[0].id));
  let nativeEq = '';
  if (native) {
    const st = JSON.parse(execFileSync(native, ['state'], { input: JSON.stringify({ game: data, upto: null }) }).toString());
    const eq = JSON.stringify(st.state) === JSON.stringify(key);
    nativeEq = eq ? ' key=native' : ' KEY DIFFERS FROM NATIVE';
    if (!eq) failures++;
  }
  console.log(`${path.basename(file)}: ${steps} steps in ${ms} ms, stopped=${JSON.stringify(out.stopped || null)}, round=${snap.round.label}, ` +
    `acting=${JSON.stringify(key.acting)} step=${key.step} menus=${menus} programs=${pi ? pi.available.length : null}${nativeEq}`);
  if (out.stopped) failures++;
  g.free();
}
process.exit(failures ? 1 : 0);
