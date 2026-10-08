// Small stand-ins for the browser, for the extension's node unit tests.

/** A document with the bits the extension reads: the site's game log
 * (``#chatlog`` lines, or none with ``log: null``) and map (``#map-hexes``
 * hex groups at the given translate positions, or none with ``hexes: null``). */
function fakeDocument({ log = [], hexes = [], hidden = false } = {}) {
  const doc = { hidden, log, hexGroups: [], created: [] };
  doc.hexGroups = (hexes || []).map(([x, y]) => ({
    tagName: 'g',
    children: [],
    getAttribute: (name) => (name === 'transform' ? `translate(${x}, ${y}) rotate(30)` : null),
    appendChild(child) {
      child.isConnected = true;
      child.parent = this;
      this.children.push(child);
    },
  }));
  doc.querySelector = (selector) => {
    if (selector === '#chatlog') {
      if (!doc.log) return null;
      return { querySelectorAll: () => doc.log.map((text) => ({ textContent: text })) };
    }
    if (selector === '#map-hexes') return hexes ? { children: doc.hexGroups } : null;
    return null;
  };
  doc.createElementNS = (ns, tag) => {
    const element = {
      ns,
      tag,
      attributes: {},
      isConnected: false,
      setAttribute(name, value) {
        this.attributes[name] = value;
      },
      remove() {
        this.isConnected = false;
        if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
      },
    };
    doc.created.push(element);
    return element;
  };
  return doc;
}

/** Timers on a manual clock: ``advance(ms)`` runs what falls due, letting
 * promises settle in between. */
function fakeClock(start = 1_000_000) {
  const clock = { t: start, timers: [] };
  clock.now = () => clock.t;
  clock.setTimeout = (fn, ms) => {
    const timer = { fn, at: clock.t + ms, live: true };
    clock.timers.push(timer);
    return timer;
  };
  clock.clearTimeout = (timer) => {
    if (timer) timer.live = false;
  };
  clock.advance = async (ms) => {
    const end = clock.t + ms;
    for (;;) {
      await settle();
      const due = clock.timers.filter((t) => t.live && t.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      clock.t = due.at;
      due.live = false;
      due.fn();
    }
    clock.t = end;
    await settle();
  };
  return clock;
}

async function settle() {
  for (let i = 0; i < 5; i += 1) await new Promise((resolve) => setImmediate(resolve));
}

module.exports = { fakeDocument, fakeClock, settle };
