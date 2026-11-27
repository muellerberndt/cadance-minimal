// Cadance: the waveform below the devices. It keeps what the player played (35 minutes): per bar the composition
// the server sent (tokens, mods, patterns, patches), the checkpoint that wrote it, the mood asked for and where the
// bar starts in the audio; per 10 ms the min and max of the output (taken in the worklet, site/worklet.js).
// Whole bars are selected and marked liked or disliked; the server keeps every mark for training rounds
// (site/feedback_marks.py names the file format). The list of marks replays a mark's stored bars through the engine.
import { loadKit } from './kit.js';

const COLS_PER_S = 100;                  // waveform columns per second (min and max per 10 ms)
const KEEP_S = 35 * 60;                  // the buffer: 35 minutes of played audio
const CAP = COLS_PER_S * KEEP_S;         // 210,000 columns, 16-bit min and max: 840 kB
const G = 16;                            // the second level: min and max per 16 columns, for wide views
const MAX_BARS = 4000;
const FPS = 30;
const RULER = 15, STRIP = 13, GAP = 3;   // css px: bar numbers on top, the whole buffer in a strip at the bottom
const MIN_BARS = 4;                      // the closest zoom

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const mss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
async function getJSON(path, options) {
  const r = await fetch(path, options);
  const j = await r.json().catch(() => ({}));
  if (!r.ok || j.error) throw Error(j.error || `request failed (${r.status})`);
  return j;
}

export class Wave {
  // opts: tempo() the bpm in force, balance() the loudness balance sent to the engine, onReplay() before a replay
  constructor(root, opts) {
    this.root = root; this.opts = opts;
    this.session = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
    this.ring = new Int16Array(CAP * 2); this.ring2 = new Int16Array((CAP / G) * 2);
    this.head = null; this.tail = null; this.offset = null; this.peak = 0.05; this.lastCols = 0;
    this.ctx = null; this.sr = 44100; this.per = 441;
    this.bars = []; this.base = 0; this.nextI = 0;            // bars[k] has the ring index base + k
    this.sent = new Map(); this.seq = 0; this.life = 0; this.lastRec = null;
    this.span = 0; this.end = 0; this.follow = true;          // the view: columns shown, the column at the right edge
    this.sel = null; this.anchor = null; this.drag = null; this.hover = null;
    this.marks = []; this.listOpen = false; this.busy = false;
    this.replay = new Replayer((what) => (what === 'bar' ? this.paintProgress() : this.paintList()));
    this.dirty = true; this.visible = true; this.lastDraw = 0; this.lastHead = 0;
    this.build();
    this.loadMarks();
    requestAnimationFrame((t) => this.tick(t));
  }

  // ---------- the page's side: audio node, bars sent, bars started, columns ----------
  attach(ctx, node) {
    this.ctx = ctx; this.sr = ctx.sampleRate; this.per = Math.round(this.sr / COLS_PER_S);
    node.port.postMessage({ type: 'scope', on: true, per: this.per });
  }
  // a bar the server answered, about to be queued in the engine: its id for the worklet
  queued(j, mood) {
    const s = ++this.seq;
    this.sent.set(s, { bar: j.bar ?? null, checkpoint: j.checkpoint ?? null, mode: j.mode === 'brain' ? 'brain' : 'demo',
      tokens: j.tokens, mods: j.mods, patterns: j.patterns ?? null, patches: j.patches ?? null, mix: j.mix ?? null,
      mood: mood ? mood.slice() : null, life: 0 });
    if (this.sent.size > 64) this.sent.delete(this.sent.keys().next().value);
    return s;
  }
  // the engine started a bar (worklet 'bar': col, its place in the scope's columns; seq, the queued bar, or null for
  // the bar before again; held, the bar that sounds). A bar while nothing from the page sounds takes no place.
  bar(m) {
    if (!Number.isFinite(m.col) || !(m.held >= 0)) return;
    let rec = null, repeat = false;
    if (m.seq != null) {
      rec = m.seq >= 0 ? this.sent.get(m.seq) || null : null;
      for (const k of this.sent.keys()) if (k <= m.seq) this.sent.delete(k);
    } else if (m.held >= 0 && this.lastRec) { rec = this.lastRec; repeat = true; }
    if (rec && !repeat) {                                       // a new composition or another brain: a new life
      const p = this.lastRec;
      if (!p || rec.bar !== p.bar + 1 || rec.checkpoint !== p.checkpoint) this.life++;
      rec.life = this.life;
    }
    if (!repeat) this.lastRec = rec;
    this.bars.push({ i: this.nextI++, col: m.col, rec, repeat, bpm: this.opts.tempo(), t: Date.now() });
    this.prune();
    this.dirty = true;
  }
  columns(m) {
    const d = m.data, n = d.length >> 1;
    this.offset = m.offset; this.lastCols = performance.now();
    let c = m.col;
    if (this.head == null) { this.head = this.tail = c; }
    if (c > this.head) {                                        // a gap (no columns came): silence
      for (let k = Math.max(this.head, c - CAP); k < c; k++) this.write(k, 0, 0);
      this.head = c;
    }
    let top = 0;
    for (let k = 0; k < n; k++, c++) {
      if (c < this.head) continue;
      const lo = d[2 * k], hi = d[2 * k + 1];
      this.write(c, lo, hi);
      top = Math.max(top, -lo, hi);
      this.head = c + 1;
    }
    this.peak = Math.max(this.peak * 0.998, top, 0.02);       // the display's scale: halves in about 35 s
    this.tail = Math.max(this.tail, this.head - CAP);
    this.prune();
    this.dirty = true;
  }
  write(c, lo, hi) {
    const q = (c % CAP) * 2, a = clamp(Math.round(lo * 32767), -32767, 32767), b = clamp(Math.round(hi * 32767), -32767, 32767);
    this.ring[q] = a; this.ring[q + 1] = b;
    const g = (Math.floor(c / G) % (CAP / G)) * 2;
    if (c % G === 0 || c === this.tail) { this.ring2[g] = a; this.ring2[g + 1] = b; }
    else { if (a < this.ring2[g]) this.ring2[g] = a; if (b > this.ring2[g + 1]) this.ring2[g + 1] = b; }
  }
  prune() {
    while (this.bars.length > 1 && (this.bars.length > MAX_BARS || (this.tail != null && this.bars[1].col <= this.tail))) { this.bars.shift(); this.base++; }
    if (this.sel && this.sel.b < this.base) this.sel = null;
    else if (this.sel && this.sel.a < this.base) this.sel.a = this.base;
  }
  stopReplay() { this.replay.stop(); }
  // a new composition after a stop: what played before leaves the display (the column clock runs on)
  clear() {
    if (this.head != null) this.tail = this.head;
    this.base = this.nextI; this.bars = [];
    this.sel = null; this.anchor = null; this.hover = null; this.lastRec = null; this.sent.clear();
    this.peak = 0.05; this.setFollow(true);
    this.dirty = true; this.paintHead();
  }

  // ---------- where things are ----------
  barCols() {                                                   // a bar's length in columns, from the last bars played
    const b = this.bars, n = b.length;
    if (n >= 3) return clamp(b[n - 1].col - b[n - 2].col, 40, 2000);
    return (240 / (this.opts.tempo() || 126)) * COLS_PER_S;
  }
  playCol() {                                                   // the column heard now
    if (this.head == null) return 0;
    const ctx = this.ctx;
    if (!ctx || ctx.state !== 'running' || this.offset == null) return this.head;
    const ts = ctx.getOutputTimestamp ? ctx.getOutputTimestamp() : null;
    const t = ts && ts.contextTime > 0 ? ts.contextTime : ctx.currentTime - (ctx.outputLatency || 0) - (ctx.baseLatency || 0);
    return clamp((t * this.sr - this.offset) / this.per, Math.max(this.tail, this.head - COLS_PER_S), this.head);
  }
  limits() {
    const min = MIN_BARS * this.barCols();
    return [min, Math.max(min, (this.head ?? 0) - (this.tail ?? 0))];
  }
  clampView() {
    const [min, max] = this.limits();
    if (!this.zoomed) this.span = 32 * this.barCols();         // 32 bars until the listener zooms
    this.span = clamp(this.span, min, max * 1.02);
    const hi = (this.head ?? 0) + this.span * 0.02, lo = (this.tail ?? 0) + this.span * 0.98;
    this.end = lo > hi ? hi : clamp(this.end, lo, hi);
  }
  barIndexAt(col) {                                             // the ring index of the bar sounding at a column
    const b = this.bars;
    if (!b.length || col < b[0].col) return b.length ? this.base : null;
    let lo = 0, hi = b.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (b[m].col <= col) lo = m; else hi = m - 1; }
    return this.base + lo;
  }
  barEnd(k) { return k + 1 < this.bars.length ? this.bars[k + 1].col : this.head ?? this.bars[k].col; }
  colAtX(x) { return this.end - this.span + (x / this.cv.clientWidth) * this.span; }
  ranges() {                                                    // marks made on this page, as ring indices
    const out = [];
    for (const m of this.marks) if (m.session === this.session && Array.isArray(m.ring)) out.push({ a: m.ring[0], b: m.ring[1], kind: m.mark });
    return out;
  }
  colsOf(a, b) {                                                // ring indices a..b as a column range
    const ka = Math.max(0, a - this.base), kb = b - this.base;
    if (kb < 0 || ka >= this.bars.length) return null;
    return [this.bars[ka].col, this.barEnd(Math.min(kb, this.bars.length - 1))];
  }

  // ---------- the panel ----------
  build() {
    const r = this.root;
    r.innerHTML = '';
    const head = el('div', 'wv-head');
    this.$info = el('span', 'wv-info');
    this.$follow = el('button', 'wv-b on', 'follow');
    this.$follow.title = 'keep the newest bars in view (f)';
    this.$follow.setAttribute('aria-pressed', 'true');
    this.$follow.onclick = () => this.setFollow(!this.follow);
    this.$sel = el('span', 'wv-sel');
    this.$clear = el('button', 'wv-x', '✕'); this.$clear.title = 'clear the selection (esc)'; this.$clear.onclick = () => this.select(null);
    this.$note = el('input', 'wv-note'); this.$note.maxLength = 500; this.$note.placeholder = 'note (optional)';
    this.$note.setAttribute('aria-label', 'a note with the mark');
    this.$note.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === 'Escape') { e.preventDefault(); this.cv.focus({ preventScroll: true }); } });
    this.$like = el('button', 'wv-b like', 'like'); this.$like.title = 'mark the selected bars liked (l)'; this.$like.onclick = () => this.mark('liked');
    this.$dislike = el('button', 'wv-b dislike', 'dislike'); this.$dislike.title = 'mark the selected bars disliked (d)'; this.$dislike.onclick = () => this.mark('disliked');
    this.$marksBtn = el('button', 'wv-b', 'marks'); this.$marksBtn.setAttribute('aria-expanded', 'false');
    this.$marksBtn.title = 'the marks, newest first: click one to replay it';
    this.$marksBtn.onclick = () => { this.listOpen = !this.listOpen; this.paintList(); if (this.listOpen) this.loadMarks(); };
    head.append(el('span', 'wv-t', 'played'), this.$info, this.$follow, el('span', 'wv-fill'), this.$sel, this.$clear, this.$note, this.$like, this.$dislike, this.$marksBtn);
    this.cv = el('canvas', 'wv-canvas');
    this.cv.tabIndex = 0;
    this.cv.setAttribute('aria-label', 'what played: drag across bars to select them; l likes, d dislikes the selection');
    this.g = this.cv.getContext('2d');
    this.$say = el('div', 'wv-say'); this.$say.setAttribute('role', 'status');
    this.$list = el('div', 'wv-list'); this.$list.hidden = true;
    r.append(head, this.cv, this.$say, this.$list);
    r.hidden = false;
    r.parentElement?.classList.add('has-wave');
    const css = getComputedStyle(document.documentElement), v = (n, d) => css.getPropertyValue(n).trim() || d;
    this.C = { fg: v('--fg', '#c9ced6'), mid: v('--mid', '#8d959f'), dim: v('--dim', '#4b525b'), line: v('--line', '#16191d'),
      like: v('--calm', '#5fd3c6'), dislike: v('--heart', '#ef6f8a'), start: v('--aroused', '#f2a65a') };
    new ResizeObserver(() => this.resize()).observe(this.cv);
    new IntersectionObserver((es) => { this.visible = es[es.length - 1].isIntersecting; this.dirty = true; }).observe(this.cv);
    this.bindPointer(); this.bindKeys();
    this.resize(); this.paintHead(); this.paintList(); this.sayHover();
  }
  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1), w = Math.round(this.cv.clientWidth * dpr), h = Math.round(this.cv.clientHeight * dpr);
    if (w && h && (w !== this.cv.width || h !== this.cv.height)) { this.cv.width = w; this.cv.height = h; }
    this.dpr = dpr; this.dirty = true;
  }
  setFollow(on) {
    this.follow = on;
    this.$follow.classList.toggle('on', on); this.$follow.setAttribute('aria-pressed', String(on));
    this.dirty = true;
  }
  say(text) { if (this.$say.textContent !== text) this.$say.textContent = text || ''; }
  selectionInfo() {
    if (!this.sel) return null;
    const ka = this.sel.a - this.base, kb = this.sel.b - this.base, n = kb - ka + 1;
    const recs = this.bars.slice(ka, kb + 1).map((b) => b.rec).filter(Boolean);
    const secs = ((this.barEnd(kb) - this.bars[ka].col) / COLS_PER_S);
    return { n, first: recs[0]?.bar, last: recs[recs.length - 1]?.bar, secs, played: recs.length };
  }
  paintHead() {
    const s = this.head == null ? 0 : (this.head - this.tail) / COLS_PER_S;
    const viewBars = this.span ? Math.round(this.span / this.barCols()) : 0;
    const info = this.head == null ? 'nothing played yet' : `${mss(s)} kept · ${this.bars.length} bars · view ${viewBars} bars`;
    if (this.$info.textContent !== info) this.$info.textContent = info;
    const si = this.selectionInfo();
    const sel = si ? `${si.n} bar${si.n === 1 ? '' : 's'} selected${si.first != null ? ` · bars ${si.first} to ${si.last}` : ''} · ${mss(si.secs)}` : 'drag across bars to select';
    if (this.$sel.textContent !== sel) this.$sel.textContent = sel;
    this.$sel.classList.toggle('on', !!si);
    this.$clear.hidden = !si;
    this.$like.disabled = this.$dislike.disabled = !si || this.busy;
    const n = this.marks.length, label = n ? `marks · ${n}` : 'marks';
    if (this.$marksBtn.textContent !== label) this.$marksBtn.textContent = label;
    this.$marksBtn.classList.toggle('on', this.listOpen);
    this.$marksBtn.setAttribute('aria-expanded', String(this.listOpen));
  }
  select(sel) { this.sel = sel; if (!sel) this.anchor = null; this.dirty = true; this.paintHead(); }

  // ---------- pointer and keys ----------
  bindPointer() {
    const cv = this.cv;
    const at = (e) => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height }; };
    const zone = (p) => (p.y < RULER ? 'ruler' : p.y > p.h - STRIP ? 'strip' : 'wave');
    const stripTo = (p) => {                                   // the strip shows the whole buffer: center the view there
      const col = this.tail + (p.x / p.w) * (this.head - this.tail);
      this.end = col + this.span / 2; this.clampView(); this.dirty = true;
    };
    cv.addEventListener('pointerdown', (e) => {
      if (this.head == null || (e.button !== 0 && e.button !== 1)) return;
      e.preventDefault();
      cv.focus({ preventScroll: true });
      const p = at(e), z = zone(p);
      cv.setPointerCapture(e.pointerId);
      if (e.button === 1 || e.altKey || z === 'ruler') {
        this.setFollow(false);
        this.drag = { kind: 'pan', x: p.x, end: this.end };
      } else if (z === 'strip') {
        this.setFollow(false);
        this.drag = { kind: 'strip' }; stripTo(p);
      } else {
        const i = this.barIndexAt(this.colAtX(p.x));
        if (i == null) return;
        if (e.shiftKey && this.sel) {
          const a = this.anchor ?? this.sel.a;
          this.anchor = a; this.select({ a: Math.min(a, i), b: Math.max(a, i) });
          this.drag = { kind: 'select' };
        } else if (this.sel && i >= this.sel.a && i <= this.sel.b) {
          // a click inside the selection clears it; a drag from there starts a new one
          this.drag = { kind: 'select', inside: true, from: i, moved: false };
        } else { this.anchor = i; this.select({ a: i, b: i }); this.drag = { kind: 'select' }; }
      }
      this.dirty = true;
    });
    cv.addEventListener('pointermove', (e) => {
      const p = at(e), d = this.drag;
      if (!d) {
        const z = this.head == null ? 'wave' : zone(p);
        cv.style.cursor = z === 'wave' && !e.altKey ? 'crosshair' : 'grab';
        this.hover = z === 'wave' && this.head != null ? this.barIndexAt(this.colAtX(p.x)) : null;
        this.sayHover();
        return;
      }
      if (d.kind === 'pan') { this.end = d.end - ((p.x - d.x) / p.w) * this.span; this.clampView(); }
      else if (d.kind === 'strip') stripTo(p);
      else if (d.kind === 'select') {
        const i = this.barIndexAt(this.colAtX(clamp(p.x, 0, p.w - 1)));
        if (d.inside && !d.moved) {
          if (i == null || i === d.from) return;
          d.moved = true; this.anchor = d.from;                  // the drag leaves the first bar: a new selection
        }
        if (i != null && this.anchor != null) this.select({ a: Math.min(this.anchor, i), b: Math.max(this.anchor, i) });
      }
      this.dirty = true;
    });
    const end = () => { this.drag = null; this.dirty = true; this.paintHead(); };
    cv.addEventListener('pointerup', () => {
      const d = this.drag;
      if (d && d.kind === 'select' && d.inside && !d.moved) this.select(null);     // one click on the selection clears it
      end();
    });
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', () => { if (!this.drag) { this.hover = null; this.sayHover(); } });
    // wheel or pinch: zoom around the pointer (with follow on, the newest bar stays at the right edge); sideways: scroll
    cv.addEventListener('wheel', (e) => {
      if (this.head == null) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
      const dx = (e.shiftKey ? e.deltaY : e.deltaX) * unit, dy = e.shiftKey ? 0 : e.deltaY * unit;
      if (e.ctrlKey || Math.abs(dy) > Math.abs(dx)) {
        const p = at(e);
        this.zoom(Math.exp(dy * (e.ctrlKey ? 0.012 : 0.0025)), clamp(p.x / p.w, 0, 1));
      } else if (dx) {
        this.setFollow(false);
        this.end += (dx / this.cv.clientWidth) * this.span; this.clampView();
      }
      this.dirty = true;
    }, { passive: false });
  }
  zoom(factor, ax) {
    this.clampView();
    const s0 = this.span, [min, max] = this.limits(), s1 = clamp(s0 * factor, min, max * 1.02);
    if (!this.follow) { const c = this.end - s0 * (1 - ax); this.end = c + s1 * (1 - ax); }
    this.span = s1; this.zoomed = true; this.clampView(); this.dirty = true; this.paintHead();
  }
  bindKeys() {
    // space keeps its meaning (play and pause, main.js); l, d, f, esc, arrows, + and - act here while the waveform has focus
    this.cv.addEventListener('keydown', (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      if (k === 'l' || k === 'L') this.mark('liked');
      else if (k === 'd' || k === 'D') this.mark('disliked');
      else if (k === 'Escape') this.select(null);
      else if (k === 'f' || k === 'F') this.setFollow(!this.follow);
      else if (k === 'ArrowLeft' || k === 'ArrowRight') {
        this.setFollow(false); this.end += (k === 'ArrowLeft' ? -1 : 1) * this.span / 8; this.clampView();
      } else if (k === '+' || k === '=') this.zoom(1 / 1.5, 1);
      else if (k === '-' || k === '_') this.zoom(1.5, 1);
      else return;
      e.preventDefault(); this.dirty = true;
    });
  }
  sayHover() {
    if (this.busy) return;
    const i = this.hover;
    if (i == null || this.drag) { this.say(this.sel ? 'l: like · d: dislike · shift-click: extend · click the selection or esc: clear' : 'wheel or pinch: zoom · drag the bar numbers or the strip below: scroll · drag across bars: select'); return; }
    const b = this.bars[i - this.base];
    if (!b) return;
    const ago = Math.max(0, (this.playCol() - b.col) / COLS_PER_S);
    const who = !b.rec ? 'silence' : b.rec.mode === 'brain' ? (b.rec.checkpoint || 'a brain') : 'instrument check, no brain';
    this.say(`bar ${b.rec?.bar ?? '-'}${b.repeat ? ' (the bar before again: the answer came late)' : ''} · ${who} · ${Math.round(b.bpm)} bpm · ${mss(ago)} ago`);
  }

  // ---------- marks ----------
  async mark(kind) {
    if (!this.sel || this.busy) return;
    const { a, b } = this.sel, rows = [];
    for (let i = a; i <= b; i++) {
      const k = i - this.base, bar = this.bars[k];
      if (!bar || !bar.rec) continue;
      const r = bar.rec;
      rows.push({ bar: r.bar, life: r.life, checkpoint: r.checkpoint, mode: r.mode, tokens: r.tokens, mods: r.mods,
        patterns: r.patterns, patches: r.patches, mix: r.mix, mood: r.mood, repeat: bar.repeat, bpm: bar.bpm,
        seconds: (this.barEnd(k) - bar.col) / COLS_PER_S, started: new Date(bar.t).toISOString() });
    }
    if (!rows.length) { this.say('nothing played in the selection'); return; }
    this.busy = true; this.paintHead();
    const si = this.selectionInfo();
    try {
      const j = await getJSON('/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mark: kind, note: this.$note.value.trim(), page_time: new Date().toISOString(), bpm: this.opts.tempo(),
          balance: this.opts.balance(), session: this.session, ring: [a, b], life_now: this.life, bars: rows }) });
      this.marks.unshift(j);
      this.$note.value = '';
      this.select(null);
      this.say(`${kind}: ${si.n} bar${si.n === 1 ? '' : 's'}${si.first != null ? ` (bars ${si.first} to ${si.last})` : ''}, stored as ${j.id}`);
    } catch (error) {
      this.say(`the mark was not stored: ${error.message}`);
    } finally {
      this.busy = false; this.paintHead(); this.paintList(); this.dirty = true;
    }
  }
  async loadMarks() {
    try { this.marks = (await getJSON('/api/feedback')).marks || []; } catch { return; }
    this.paintHead(); this.paintList(); this.dirty = true;
  }
  async remove(m) {
    try { await getJSON(`/api/feedback?id=${encodeURIComponent(m.id)}`, { method: 'DELETE' }); }
    catch (error) { this.say(`could not remove ${m.id}: ${error.message}`); return; }
    if (this.replay.id === m.id) this.replay.stop();
    this.marks = this.marks.filter((x) => x.id !== m.id);
    this.say(`removed ${m.id}`);
    this.paintHead(); this.paintList(); this.dirty = true;
  }
  paintList() {
    const box = this.$list;
    box.hidden = !this.listOpen;
    this.paintHead();
    if (!this.listOpen) return;
    box.innerHTML = '';
    if (!this.marks.length) { box.appendChild(el('div', 'wv-empty', 'no marks yet: select bars above, then like or dislike them')); return; }
    for (const m of this.marks) {
      const on = this.replay.id === m.id;
      const row = el('div', `wv-row ${m.mark}${on ? ' on' : ''}`);
      row.tabIndex = 0;
      row.title = `${on ? 'stop' : 'replay these bars in a loop'} (stored as ${m.file || m.id})`;
      const play = el('span', 'wv-pl', on ? '■' : '▶');
      const kind = el('span', 'wv-k', m.mark);
      const len = el('span', 'wv-n', `${m.bars_count} bar${m.bars_count === 1 ? '' : 's'}`);
      const what = el('span', 'wv-w', `${m.first_bar != null ? `bars ${m.first_bar} to ${m.last_bar} · ` : ''}${(m.checkpoints || []).map((c) => c || 'no brain').join(', ')}${m.note ? ` · ${m.note}` : ''}`);
      const when = el('span', 'wv-d', on && this.replay.total ? `${this.replay.now + 1} of ${this.replay.total}` : (m.time || '').replace('T', ' ').slice(5, 16));
      const x = el('button', 'wv-rm', '✕'); x.title = 'remove this mark (click twice)'; x.setAttribute('aria-label', `remove the mark ${m.id}`);
      x.onclick = (e) => {
        e.stopPropagation();
        if (!x.classList.contains('confirm')) { x.classList.add('confirm'); x.textContent = 'remove?'; setTimeout(() => { x.classList.remove('confirm'); x.textContent = '✕'; }, 3000); return; }
        this.remove(m);
      };
      const toggle = () => this.toggleReplay(m);
      row.onclick = toggle;
      row.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); toggle(); } };
      row.append(play, kind, len, what, when, x);
      box.appendChild(row);
    }
  }
  paintProgress() {                                             // the replayed bar, in the row that plays
    const cell = this.$list.querySelector('.wv-row.on .wv-d');
    if (cell && this.replay.total) cell.textContent = `${this.replay.now + 1} of ${this.replay.total}`;
  }
  async toggleReplay(m) {
    if (this.replay.id === m.id) { this.replay.stop(); return; }
    this.opts.onReplay?.();
    this.say(`replaying ${m.id}…`);
    try { await this.replay.play(m, this.opts.balance()); this.say(`replaying ${m.bars_count} bars of ${m.id} in a loop: click it again to stop`); }
    catch (error) { this.say(`could not replay ${m.id}: ${error.message}`); this.replay.stop(); }
  }

  // ---------- drawing: at most 30 times a second, only when the canvas is on screen ----------
  tick(t) {
    requestAnimationFrame((t2) => this.tick(t2));
    if (document.hidden || !this.visible || !this.cv.offsetParent) return;
    const moving = this.ctx && this.ctx.state === 'running' && performance.now() - this.lastCols < 400;
    if ((!this.dirty && !moving) || t - this.lastDraw < 1000 / FPS - 3) return;
    this.lastDraw = t; this.dirty = false;
    this.draw();
    if (t - this.lastHead > 250) { this.lastHead = t; this.paintHead(); }
  }
  draw() {
    const g = this.g, W = this.cv.width, H = this.cv.height, dpr = this.dpr || 1, C = this.C;
    g.clearRect(0, 0, W, H);
    const top = Math.round(RULER * dpr), strip = Math.round(STRIP * dpr), wh = H - top - strip - Math.round(GAP * dpr), mid = top + wh / 2;
    g.fillStyle = 'rgba(255,255,255,0.018)'; g.fillRect(0, top, W, wh);
    g.font = `${Math.round(9 * dpr)}px ui-monospace, SFMono-Regular, Menlo, monospace`; g.textBaseline = 'middle';
    if (this.head == null) { g.fillStyle = C.dim; g.fillText('what the player plays appears here', 8 * dpr, mid); return; }
    const play = this.playCol();
    if (this.follow && !this.drag) this.end = play + this.span * 0.01;
    this.clampView();
    const span = this.span, start = this.end - span, cpp = span / W, x = (col) => (col - start) / cpp;

    // marks and the selection: bands behind the waveform
    const bands = [];
    for (const r of this.ranges()) { const c = this.colsOf(r.a, r.b); if (c) bands.push({ c, color: r.kind === 'liked' ? C.like : C.dislike, kind: r.kind }); }
    bands.sort((p, q) => (q.c[1] - q.c[0]) - (p.c[1] - p.c[0]));          // longer first: a shorter mark inside stays visible
    const sc = this.sel ? this.colsOf(this.sel.a, this.sel.b) : null;
    for (const bd of bands) {
      const x0 = x(bd.c[0]), x1 = x(bd.c[1]);
      if (x1 < 0 || x0 > W) continue;
      g.globalAlpha = 0.1; g.fillStyle = bd.color; g.fillRect(x0, top, x1 - x0, wh);
      g.globalAlpha = 0.9; g.fillRect(x0, top + wh - 2 * dpr, x1 - x0, 2 * dpr);
      g.globalAlpha = 1;
    }
    if (sc) {
      const x0 = x(sc[0]), x1 = x(sc[1]);
      g.fillStyle = 'rgba(201,206,214,0.07)'; g.fillRect(x0, top, x1 - x0, wh);
      g.fillStyle = C.fg; g.globalAlpha = 0.6; g.fillRect(Math.round(x0), top, dpr, wh); g.fillRect(Math.round(x1) - dpr, top, dpr, wh); g.globalAlpha = 1;
    }

    // bar lines, 4- and 16-bar lines, the bar numbers, a new composition or brain in the accent colour
    const bars = this.bars, barPx = this.barCols() / cpp;
    let step = 1; for (const s of [1, 4, 16, 64, 256]) { step = s; if (barPx * s >= 26 * dpr) break; }
    let k0 = Math.max(0, this.barIndexAt(start) - this.base);
    let lastLife = k0 > 0 ? bars[k0 - 1].rec?.life : null;
    for (let k = k0; k < bars.length; k++) {
      const b = bars[k], bx = Math.round(x(b.col));
      if (bx > W) break;
      const n = b.rec?.bar, life = b.rec?.life;
      const fresh = b.rec && !b.repeat && life !== lastLife;
      lastLife = life ?? lastLife;
      if (bx < 0) continue;
      const q = n != null ? (n - 1) % 16 === 0 ? 16 : (n - 1) % 4 === 0 ? 4 : 1 : 1;
      if (fresh) { g.fillStyle = C.start; g.globalAlpha = 0.7; g.fillRect(bx, 0, dpr, top + wh); g.globalAlpha = 1; }
      else if (barPx * q >= 3 * dpr) {
        g.fillStyle = q === 16 ? '#30353c' : q === 4 ? '#22262c' : '#16191d';
        g.fillRect(bx, q === 16 ? top - 4 * dpr : top, dpr, wh + (q === 16 ? 4 * dpr : 0));
      }
      if (n != null && (n - 1) % step === 0) {
        g.fillStyle = fresh ? C.start : (n - 1) % 16 === 0 ? C.mid : C.dim;
        g.fillText(String(n), bx + 3 * dpr, top / 2);
      }
    }

    // the waveform up to the playhead: per pixel the min and max of its columns (the second level for wide views)
    const gain = (wh / 2) * 0.92 / (this.peak * 32767), R = this.ring, R2 = this.ring2;
    const lim0 = this.tail, lim1 = Math.min(this.head, play);
    // the selection's waveform in the page's foreground, a mark's in its colour, muted; the shortest range on top
    const tints = bands.map((bd) => ({ c: bd.c, color: bd.kind === 'liked' ? 'rgba(95,211,198,0.62)' : 'rgba(239,111,138,0.62)' }))
      .sort((p, q) => (p.c[1] - p.c[0]) - (q.c[1] - q.c[0]));
    if (sc) tints.unshift({ c: sc, color: 'rgba(201,206,214,0.9)' });
    let color = null;
    for (let px = 0; px < W; px++) {
      const c0 = start + px * cpp, c1 = c0 + cpp;
      if (c1 <= lim0 || c0 >= lim1) continue;
      let lo = 32767, hi = -32767;
      if (cpp <= 1) { const c = Math.floor(clamp(c0, lim0, lim1 - 1)); lo = R[(c % CAP) * 2]; hi = R[(c % CAP) * 2 + 1]; }
      else if (cpp < 4 * G) {
        for (let c = Math.max(lim0, Math.floor(c0)), e = Math.min(lim1, Math.ceil(c1)); c < e; c++) { const q = (c % CAP) * 2; if (R[q] < lo) lo = R[q]; if (R[q + 1] > hi) hi = R[q + 1]; }
      } else {
        for (let gi = Math.floor(Math.max(lim0, c0) / G), ge = Math.floor((Math.min(lim1, c1) - 1) / G); gi <= ge; gi++) { const q = (gi % (CAP / G)) * 2; if (R2[q] < lo) lo = R2[q]; if (R2[q + 1] > hi) hi = R2[q + 1]; }
      }
      if (hi < lo) continue;
      const cm = c0 + cpp / 2;
      let want = '#5a636d';
      for (const tn of tints) if (cm >= tn.c[0] && cm < tn.c[1]) { want = tn.color; break; }
      if (want !== color) { g.fillStyle = color = want; }
      const y0 = mid - hi * gain, y1 = mid - lo * gain;
      g.fillRect(px, y0, 1, Math.max(dpr, y1 - y0));
    }
    // the playhead
    const ph = x(play);
    if (ph >= 0 && ph <= W) { g.fillStyle = C.fg; g.globalAlpha = 0.65; g.fillRect(Math.round(ph) - dpr, top - 4 * dpr, dpr, wh + 4 * dpr); g.globalAlpha = 1; }

    // the strip: the whole buffer, the marks, the part in view
    const sy = H - strip, total = Math.max(1, this.head - this.tail), spx = total / W;
    g.fillStyle = 'rgba(255,255,255,0.02)'; g.fillRect(0, sy, W, strip);
    g.fillStyle = '#3a4048';
    const smid = sy + strip / 2, sgain = (strip / 2 - dpr) / (this.peak * 32767);
    for (let px = 0; px < W; px++) {
      const c0 = this.tail + px * spx, c1 = c0 + spx;
      let lo = 32767, hi = -32767;
      for (let gi = Math.floor(c0 / G), ge = Math.floor((c1 - 1) / G); gi <= ge; gi++) { const q = (gi % (CAP / G)) * 2; if (R2[q] < lo) lo = R2[q]; if (R2[q + 1] > hi) hi = R2[q + 1]; }
      if (hi >= lo) g.fillRect(px, smid - hi * sgain, 1, Math.max(dpr, (hi - lo) * sgain));
    }
    for (const bd of bands) { g.fillStyle = bd.color; g.fillRect((bd.c[0] - this.tail) / spx, H - 2 * dpr, Math.max(dpr, (bd.c[1] - bd.c[0]) / spx), 2 * dpr); }
    if (sc) { g.fillStyle = C.fg; g.fillRect((sc[0] - this.tail) / spx, H - 2 * dpr, Math.max(dpr, (sc[1] - sc[0]) / spx), 2 * dpr); }
    const vx0 = (start - this.tail) / spx, vx1 = (this.end - this.tail) / spx;
    g.fillStyle = 'rgba(255,255,255,0.06)'; g.fillRect(vx0, sy, vx1 - vx0, strip);
    g.strokeStyle = C.dim; g.lineWidth = dpr; g.strokeRect(Math.max(0, vx0) + dpr / 2, sy + dpr / 2, Math.min(W, vx1) - Math.max(0, vx0) - dpr, strip - dpr);
  }
}

// ---------- replay: a mark's stored bars through the engine (its own audio context, as the curator plays them) ----------
class Replayer {
  constructor(onChange) { this.onChange = onChange; this.ctx = null; this.node = null; this.kit = null; this.pools = {}; this.id = null; this.bars = null; this.mirror = []; this.pos = 0; this.now = 0; this.total = 0; this.token = 0; }
  async play(mark, balance) {
    const token = ++this.token;
    this.stop(false);
    const full = await getJSON(`/api/feedback?id=${encodeURIComponent(mark.id)}`);
    const bars = (full.bars || []).filter((b) => Array.isArray(b.tokens));
    if (!bars.length) throw Error('the mark holds no bars');
    const name = bars[0].pool || 'pool_v0.json';
    const pool = this.pools[name] || (this.pools[name] = await getJSON(`/api/feedback/pool?name=${encodeURIComponent(name)}`));
    if (!this.ctx) {
      this.ctx = new AudioContext({ sampleRate: 44100, latencyHint: 'playback' });
      await this.ctx.audioWorklet.addModule('worklet.js');
    }
    await this.ctx.suspend();
    if (token !== this.token) return;
    // a node of its own per replay: its engine starts at bar 0 with the first stored bar
    const node = new AudioWorkletNode(this.ctx, 'minimal', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
    if (this.kit) node.port.postMessage({ type: 'samples', kit: this.kit, sampleRate: this.ctx.sampleRate });
    else this.kit = await loadKit(this.ctx, node);
    if (token !== this.token) { node.disconnect(); return; }
    node.port.postMessage({ type: 'balance', ...(full.balance || balance) });
    node.port.postMessage({ type: 'pool', pool });
    node.port.postMessage({ type: 'tempo', bpm: full.bpm || bars[0].bpm || pool.bpm || 126 });
    node.port.onmessage = (e) => {
      if (e.data.type !== 'bar' || node !== this.node) return;
      while (this.mirror.length > e.data.queued) this.now = this.mirror.shift();
      this.top(); this.onChange('bar');
    };
    node.connect(this.ctx.destination);
    Object.assign(this, { node, bars, id: mark.id, mirror: [], pos: 0, now: 0, total: bars.length });
    this.top();
    await new Promise((r) => setTimeout(r, 40));                // the messages reach the engine before it plays
    if (token === this.token) await this.ctx.resume();
    this.onChange();
  }
  top() {
    while (this.node && this.mirror.length < 3) {
      const k = this.pos, b = this.bars[k];
      this.pos = (this.pos + 1) % this.bars.length;
      this.mirror.push(k);
      this.node.port.postMessage({ type: 'queue', tokens: b.tokens, mods: b.mods, patterns: b.patterns || null, patches: b.patches || null });
    }
  }
  stop(paint = true) {
    const node = this.node, ctx = this.ctx;
    this.node = null; this.id = null; this.bars = null; this.mirror = []; this.total = 0;
    if (node) {
      node.port.postMessage({ type: 'silence' });
      setTimeout(() => { node.disconnect(); if (!this.node && ctx.state === 'running') ctx.suspend(); }, 80);
    }
    if (paint) { this.token++; this.onChange(); }
  }
}
