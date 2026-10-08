// The advisor panel: a floating, draggable, collapsible box in a closed Shadow
// DOM (the site's CSS can't reach in, ours can't leak out). It only draws what
// it is given (``update(view)``) and reports clicks through callbacks; the
// content script (content.js) and the backend's debug page drive it.
(function (root) {
  'use strict';

  const R = root.RL18XX || (typeof require === 'function' ? require('./common.js') : null);
  const esc = R.escapeHtml;
  const pct = R.percent;

  const STYLE = `
    :host { all: initial; }
    .panel { font: 12px/1.4 -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #e5e7eb;
      background: rgba(17, 20, 28, 0.96); border: 1px solid #374151; border-radius: 8px; width: 340px;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35); overflow: hidden; }
    .hdr { display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: #1f2937; cursor: move;
      user-select: none; }
    .title { font-weight: 700; }
    .summary { color: #9ca3af; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
    button { font: inherit; color: #e5e7eb; background: #374151; border: 1px solid #4b5563; border-radius: 4px;
      padding: 1px 7px; cursor: pointer; }
    button:hover { border-color: #93c5fd; }
    button:disabled { opacity: 0.5; cursor: default; }
    .body { padding: 6px 8px 8px; max-height: 70vh; overflow-y: auto; }
    .sec { margin-top: 8px; }
    .h { font-weight: 700; color: #9ca3af; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px;
      margin-bottom: 3px; }
    .note { color: #9ca3af; font-weight: 400; text-transform: none; letter-spacing: 0; }
    .status { padding: 4px 6px; border-radius: 4px; margin-top: 4px; }
    .status.error { background: #7f1d1d; } .status.warn { background: #78350f; } .status.info { background: #1e3a8a; }
    .row { display: grid; grid-template-columns: 90px 1fr 42px; gap: 6px; align-items: center; margin: 2px 0; }
    .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .name.acting { font-weight: 700; color: #fde68a; }
    .bar { height: 9px; background: #374151; border-radius: 3px; overflow: hidden; }
    .bar > div { height: 100%; background: #60a5fa; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    ol { margin: 0; padding-left: 0; list-style: none; }
    li.move { display: grid; grid-template-columns: 42px 1fr; gap: 4px; padding: 2px 3px; border-radius: 4px; }
    li.move:hover { background: #1f2937; }
    li.move .p { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
    .price { color: #9ca3af; grid-column: 2; }
    .badge { font-size: 10px; padding: 0 4px; border-radius: 3px; margin-left: 4px; }
    .badge.expected { background: #14532d; } .badge.plausible { background: #374151; }
    .badge.surprising { background: #7f1d1d; } .badge.unknown { background: #374151; }
    .recent div { margin: 2px 0; }
    .progress { height: 6px; background: #374151; border-radius: 3px; overflow: hidden; margin-top: 4px; }
    .progress > div { height: 100%; background: #34d399; }
  `;

  function priceText(price) {
    if (!price || price.fixed || !price.options || !price.options.length) return '';
    const [best, ...rest] = price.options;
    const cell = (o) => (o.low === o.high ? `$${o.low}` : `$${o.low}–${o.high}`);
    const alternatives = rest.map((o) => `${cell(o)} (${pct(o.probability)})`).join(', ');
    return `price ${cell(best)} (${pct(best.probability)})${alternatives ? `; also ${alternatives}` : ''}`;
  }

  function actingText(advice) {
    const acting = advice.acting;
    if (!acting) return '';
    const who = acting.kind === 'player' ? esc(acting.name) : `${esc(acting.name)} <span class="note">(${esc(acting.player || '?')})</span>`;
    const where = advice.round ? ` · ${esc(advice.round.type || '')}${advice.round.step ? ` · ${esc(advice.round.step)}` : ''}` : '';
    return `<b>${who}</b> to move${where}`;
  }

  function winRows(advice) {
    const win = advice.win;
    if (!win || !win.players) return '';
    const actingSeat = advice.acting ? advice.acting.player_id : null;
    const rows = win.players
      .map((p) => {
        const width = Math.max(0, Math.min(100, 100 * (p.probability || 0)));
        const cls = p.seat === actingSeat ? 'name acting' : 'name';
        return `<div class="row"><span class="${cls}" title="${esc(p.name)}">${esc(p.name)}</span><div class="bar"><div style="width:${width.toFixed(1)}%"></div></div><span class="num">${pct(p.probability)}</span></div>`;
      })
      .join('');
    const note = win.final ? 'final: share of the win' : 'value net, normalised to 100%';
    return `<div class="sec"><div class="h">Win chances <span class="note">${note}</span></div>${rows}</div>`;
  }

  function moveList(moves, kind) {
    return moves
      .map((m, i) => {
        const share = kind === 'search' ? m.share : m.probability;
        // A search move's description already names its most visited price.
        const price = kind === 'search' ? '' : priceText(m.price);
        const visits = kind === 'search' ? ` <span class="note">${m.visits} visits</span>` : '';
        const actor = m.actor ? `<span class="note">${esc(m.actor)}:</span> ` : '';
        return `<li class="move" data-kind="${kind}" data-i="${i}"><span class="p">${pct(share)}</span><span>${actor}${esc(m.description)}${visits}</span>${price ? `<span class="price">${esc(price)}</span>` : ''}</li>`;
      })
      .join('');
  }

  function movesSection(advice) {
    if (!advice.moves || !advice.moves.length) return '';
    const policy = advice.policy || {};
    const note = `${advice.forced ? 'forced move · ' : ''}${advice.num_legal} legal · ${policy.role === 'auction_policy' ? 'auction policy' : 'policy'} at T=1`;
    return `<div class="sec"><div class="h">Model's moves <span class="note">${esc(note)}</span></div><ol class="moves">${moveList(advice.moves, 'policy')}</ol></div>`;
  }

  function thinkSection(view, advice) {
    if (!advice.moves || !advice.moves.length || advice.finished) return '';
    const think = view.think || { state: 'idle' };
    const running = think.state === 'running';
    const label = running ? `Thinking… ${pct(think.progress || 0)}` : `Think harder (${view.readouts || 200} readouts)`;
    let out = `<div class="sec"><button data-act="think" ${running || view.busy ? 'disabled' : ''}>${esc(label)}</button>`;
    if (running) out += `<div class="progress"><div style="width:${(100 * (think.progress || 0)).toFixed(1)}%"></div></div>`;
    if (think.state === 'error') out += `<div class="status error">${esc(think.error)}</div>`;
    if (think.state === 'done' && think.result) {
      const r = think.result;
      const stale = think.position && advice.position && think.position !== advice.position ? ' (an earlier position)' : '';
      out += `<div class="h" style="margin-top:6px">Search: ${r.visits} visits${esc(stale)}</div><ol class="moves">${moveList(r.moves.slice(0, 5), 'search')}</ol>`;
      if (r.win) {
        out += `<div class="note">Search win estimate: ${r.win.map((p) => `${esc(p.name)} ${pct(p.probability)}`).join(' · ')}</div>`;
      }
    }
    return `${out}</div>`;
  }

  function recentSection(advice) {
    if (!advice.recent || !advice.recent.length) return '';
    const rows = advice.recent
      .map((r) => {
        const rank = r.rank ? ` #${r.rank}/${r.num_legal}` : '';
        return `<div><span class="note">${r.action}.</span> ${esc(r.actor || '')}: ${esc(r.description || r.type)} — <b>${pct(r.probability)}</b>${esc(rank)}<span class="badge ${esc(r.label || 'unknown')}">${esc(r.label || '?')}</span></div>`;
      })
      .join('');
    return `<div class="sec recent"><div class="h">Last moves <span class="note">the model's probability for each</span></div>${rows}</div>`;
  }

  /** The panel's body HTML for a view: ``{statuses: [{kind, text}], advice, think,
   * readouts, busy}``. Pure: the unit tests check it directly. */
  function renderBody(view) {
    const parts = (view.statuses || []).map((s) => `<div class="status ${esc(s.kind || 'info')}">${esc(s.text)}</div>`);
    const advice = view.advice;
    if (advice && advice.supported && advice.started !== false) {
      for (const warning of advice.warnings || []) parts.push(`<div class="status warn">${esc(warning)}</div>`);
      if (advice.finished) parts.push('<div class="sec">The game is over.</div>');
      else if (advice.acting) parts.push(`<div class="sec">${actingText(advice)}</div>`);
      parts.push(winRows(advice), movesSection(advice), thinkSection(view, advice), recentSection(advice));
    }
    return parts.join('');
  }

  /** A one-line summary for the collapsed panel. */
  function summary(view) {
    const advice = view.advice;
    if (view.statuses && view.statuses.length && (!advice || !advice.supported)) return view.statuses[0].text;
    if (!advice || !advice.moves || !advice.moves.length) return '';
    const top = advice.moves[0];
    return `${pct(top.probability)} ${top.description}`;
  }

  class Panel {
    /** ``callbacks``: ``onRefresh()``, ``onThink()``, ``onHoverMove(move)`` (null when
     * the pointer leaves), ``onPlace({left, top})`` after a drag, ``onCollapse(bool)``. */
    constructor(doc, callbacks = {}, placement = {}) {
      this.doc = doc;
      this.callbacks = callbacks;
      this.placement = Object.assign({ corner: 'top-right', left: null, top: null, collapsed: false }, placement);
      this.view = { statuses: [] };
      this.host = doc.createElement('div');
      this.host.id = 'rl18xx-advisor';
      this.shadow = this.host.attachShadow({ mode: 'closed' });
      this.shadow.innerHTML = `<style>${STYLE}</style><div class="panel"><div class="hdr"><span class="title">1830 advisor</span><span class="summary"></span><button data-act="refresh" title="Fetch the game again (at most every 10 s)">↻</button><button data-act="collapse" title="Collapse / expand">–</button></div><div class="body"></div></div>`;
      this.body = this.shadow.querySelector('.body');
      this.summaryEl = this.shadow.querySelector('.summary');
      this.shadow.addEventListener('click', (event) => this._click(event));
      this.shadow.addEventListener('mouseover', (event) => this._hover(event));
      this.shadow.querySelector('.panel').addEventListener('mouseleave', () => this._emit('onHoverMove', null));
      this.shadow.querySelector('.hdr').addEventListener('mousedown', (event) => this._dragStart(event));
      this._place();
      doc.body.appendChild(this.host);
      this.update(this.view);
    }

    _emit(name, ...args) {
      if (typeof this.callbacks[name] === 'function') this.callbacks[name](...args);
    }

    _place() {
      const style = this.host.style;
      style.position = 'fixed';
      style.zIndex = '2147483000';
      for (const side of ['left', 'top', 'right', 'bottom']) style[side] = '';
      const { left, top, corner } = this.placement;
      if (Number.isFinite(left) && Number.isFinite(top)) {
        style.left = `${left}px`;
        style.top = `${top}px`;
        return;
      }
      const [vertical, horizontal] = String(corner || 'top-right').split('-');
      style[vertical === 'bottom' ? 'bottom' : 'top'] = vertical === 'bottom' ? '16px' : '64px';
      style[horizontal === 'left' ? 'left' : 'right'] = '16px';
    }

    setPlacement(placement) {
      this.placement = Object.assign(this.placement, placement);
      this._place();
      this.update(this.view);
    }

    _click(event) {
      const button = event.target.closest ? event.target.closest('[data-act]') : null;
      if (!button || button.disabled) return;
      const act = button.getAttribute('data-act');
      if (act === 'refresh') this._emit('onRefresh');
      else if (act === 'think') this._emit('onThink');
      else if (act === 'collapse') {
        this.placement.collapsed = !this.placement.collapsed;
        this.update(this.view);
        this._emit('onCollapse', this.placement.collapsed);
      }
    }

    _hover(event) {
      const item = event.target.closest ? event.target.closest('li.move') : null;
      if (!item) return;
      const advice = this.view.advice || {};
      const i = Number(item.getAttribute('data-i'));
      const list = item.getAttribute('data-kind') === 'search' ? ((this.view.think || {}).result || {}).moves : advice.moves;
      this._emit('onHoverMove', list ? list[i] || null : null);
    }

    _dragStart(event) {
      if (event.button !== 0 || (event.target.closest && event.target.closest('button'))) return;
      const rect = this.host.getBoundingClientRect();
      const dx = event.clientX - rect.left;
      const dy = event.clientY - rect.top;
      const win = this.doc.defaultView;
      const move = (e) => {
        const left = Math.max(0, Math.min(win.innerWidth - 60, e.clientX - dx));
        const top = Math.max(0, Math.min(win.innerHeight - 30, e.clientY - dy));
        this.placement.left = left;
        this.placement.top = top;
        this._place();
      };
      const up = () => {
        this.doc.removeEventListener('mousemove', move, true);
        this.doc.removeEventListener('mouseup', up, true);
        this._emit('onPlace', { left: this.placement.left, top: this.placement.top });
      };
      this.doc.addEventListener('mousemove', move, true);
      this.doc.addEventListener('mouseup', up, true);
      event.preventDefault();
    }

    update(view) {
      this.view = view;
      this.summaryEl.textContent = this.placement.collapsed ? summary(view) : '';
      this.body.style.display = this.placement.collapsed ? 'none' : '';
      this.shadow.querySelector('[data-act="collapse"]').textContent = this.placement.collapsed ? '+' : '–';
      if (!this.placement.collapsed) this.body.innerHTML = renderBody(view);
    }

    destroy() {
      this.host.remove();
    }
  }

  const exported = { Panel, renderBody, summary, priceText };
  root.RL18XX = Object.assign(root.RL18XX || {}, exported);
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
})(typeof globalThis !== 'undefined' ? globalThis : this);
