// Cadance mixer: three decks, each a brain of its own composing in the local server (/api/deck/*), played by three
// instruments on ONE clock (deckworklet.js), so the decks always run in time and a deck that starts comes in on the
// next bar line. Each deck runs through a channel strip: high/mid/low EQ (the left stop kills the band),
// highpass, lowpass, crossfader assign (A / thru / B), mute, fader with meter. The three meet in the master fader
// and a limiter. The player page (/) and this page are independent: each has its own brains in the server.
import { loadKit } from './kit.js';
import { arrived } from './brains.js';
import { FxUnit, FX, FX_LABEL } from './djfx.js';
import { AutoDJ } from './autodj.js';
import { BrainViz, MODE_GROUPS } from './brainviz.js';

let dj = null;                 // the Auto DJ (made below, once the desk exists)

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const DECKS = 3, AHEAD = 2, MOOD = [0.45, 0.4];
const LAYERS = ['beat', 'bass', 'lead', 'air'];
const store = {
  get(k, d) { try { const v = localStorage.getItem('cadance.mixer.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('cadance.mixer.' + k, JSON.stringify(v)); } catch { /* private window */ } },
};
async function post(path, body, ms = 90000) {
  const ctl = new AbortController(); const timer = setTimeout(() => ctl.abort(), ms);
  try { return await (await fetch(path, { method: 'POST', body: JSON.stringify(body), signal: ctl.signal })).json(); } finally { clearTimeout(timer); }
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---------- parameter laws: knob position 0..1 -> value ----------
function hz(v) { return v === 0 ? 'off' : v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)} kHz` : `${Math.round(v)} Hz`; }
const sdb = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)} dB`;
const LAW = {
  eq: { def: 0.5, bi: true, val: (p) => (p <= 0.015 ? -40 : p < 0.5 ? (-26 * (0.5 - p)) / 0.485 : (6 * (p - 0.5)) / 0.5), fmt: (v) => (v <= -40 ? 'kill' : sdb(v)) },
  hp: { def: 0, val: (p) => (p <= 0.005 ? 0 : 20 * Math.pow(125, p)), fmt: hz },                 // 20 Hz .. 2.5 kHz
  lp: { def: 1, from: 1, val: (p) => (p >= 0.995 ? 0 : 180 * Math.pow(20000 / 180, p)), fmt: hz }, // 180 Hz .. 20 kHz
  fader: { def: 0.8, val: (p) => Math.pow(p / 0.8, 2.5), fmt: (g) => (g <= 0.0005 ? '−∞' : sdb(20 * Math.log10(g))) },
  fx: { def: 0, val: (p) => p, fmt: (v) => (v <= 0.005 ? 'off' : `${Math.round(v * 100)}%`) },
};

// ---------- audio ----------
let ctx = null, node = null, master = null, masterMeter = null, bpm = 126, step = -1;
let userBpm = store.get('bpm', null);
let xfPos = store.get('xf', 0.5);
const latency = () => (ctx ? (ctx.outputLatency || 0) + (ctx.baseLatency || 0) : 0);
const later = (t, fn) => { const ms = Math.max(0, (t + latency() - (ctx ? ctx.currentTime : 0)) * 1000); ms > 4 ? setTimeout(fn, ms) : fn(); };
let starting = null;
// the kick boost of the player's balance (balance.drums.kickBoost): its own switch beside the tempo, kept in this browser;
// the space (balance.space: a base reverb and delay send per synth voice) likewise
let deckBalance = null, kickBoost = store.get('kickBoost', true), spaceSend = store.get('space', true), bassLift = store.get('bass', true);
function sendDeckBalance() {
  $('kickBtn')?.classList.toggle('on', kickBoost); $('kickBtn')?.setAttribute('aria-pressed', String(kickBoost));
  $('spaceBtn')?.classList.toggle('on', spaceSend); $('spaceBtn')?.setAttribute('aria-pressed', String(spaceSend));
  $('bassBtn')?.classList.toggle('on', bassLift); $('bassBtn')?.setAttribute('aria-pressed', String(bassLift));
  const kb = deckBalance?.drums?.kickBoost, sp = deckBalance?.space, bs = deckBalance?.bass;
  if (node && deckBalance) node.port.postMessage({ type: 'balance', ...deckBalance, ...(kb ? { drums: { ...deckBalance.drums, kickBoost: { ...kb, on: kickBoost && kb.on !== false } } } : {}),
    ...(sp ? { space: { ...sp, on: spaceSend && sp.on !== false } } : {}), ...(bs ? { bass: { ...bs, on: bassLift && bs.on !== false } } : {}) });
}
if ($('kickBtn')) $('kickBtn').onclick = () => { kickBoost = !kickBoost; store.set('kickBoost', kickBoost); sendDeckBalance(); };
if ($('spaceBtn')) $('spaceBtn').onclick = () => { spaceSend = !spaceSend; store.set('space', spaceSend); sendDeckBalance(); };
if ($('bassBtn')) $('bassBtn').onclick = () => { bassLift = !bassLift; store.set('bass', bassLift); sendDeckBalance(); };
sendDeckBalance();
function ensureAudio() {
  if (ctx) return Promise.resolve();
  if (starting) return starting;
  starting = (async () => {
    const c = new AudioContext({ sampleRate: 44100, latencyHint: 'playback' });
    await c.audioWorklet.addModule('deckworklet.js');
    node = new AudioWorkletNode(c, 'decks', { numberOfInputs: 0, numberOfOutputs: DECKS, outputChannelCount: Array(DECKS).fill(2) });
    await loadKit(c, node);
    master = c.createGain();
    const lim = c.createDynamicsCompressor();
    lim.threshold.value = -1.5; lim.knee.value = 0; lim.ratio.value = 20; lim.attack.value = 0.002; lim.release.value = 0.12;
    masterMeter = c.createAnalyser(); masterMeter.fftSize = 1024;
    master.connect(lim); lim.connect(masterMeter); masterMeter.connect(c.destination);
    ctx = c;
    viz.listen(c, master);                                   // the visuals hear the master
    for (const ch of chans) ch.build();
    applyMaster(true);
    node.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'bar') {
        const ch = chans[m.deck];
        if (m.deck === 0) {                                      // the clock of every deck: this bar line
          const len = 240 / bpm;
          for (const c of chans) if (!c.auto) c.gate(m.t + len, len, c.p.fx);      // a hand-set gate, for the next bar
          // the decks are fed below whatever the autopilot does: an error in it must not stop the music
          if (dj) { try { dj.onBar(m.bar, m.t); } catch (err) { status(`auto dj: ${err.message}`); } }
        }
        const had = ch.mirror.length;                            // after the autopilot's commands: it may have ended or cued this deck
        // a deck cued since its last bar message has silenced its engine and queued the new track's first bars:
        // everything in its mirror is waiting, and m.queued (counted before the cue) does not apply
        const requeued = ch.requeued; ch.requeued = false;
        ch.feed(requeued ? ch.mirror.length : m.queued);         // a deck that plays a track gets its next bars at once
        const queued = requeued ? ch.mirror.length : Math.max(0, m.queued + (ch.mirror.length - had));
        later(ctx.currentTime, () => ch.onBar(queued));
      } else if (m.type === 'step') { if (dj) dj.pump(); later(m.t, () => { step = m.step; stepAt = performance.now(); for (const ch of chans) ch.paintStep(); vizStep(); }); }
      else if (m.type === 'error') status('engine: ' + m.message.split('\n')[0]);
    };
    node.port.postMessage({ type: 'tempo', bpm });
    // the player's loudness balance (site/play-reading.json): every synth voice at a set share of the kit's level
    try { const j = await (await fetch('/play-reading.json')).json(); if (j && j.balance) { deckBalance = { ...j.balance, on: j.balance.on !== false }; sendDeckBalance(); } } catch { /* the decks play as written */ }
    for (const ch of chans) if (ch.pool) node.port.postMessage({ type: 'pool', deck: ch.d, pool: ch.pool });
  })();
  return starting;
}
function xfGain(assign) {
  if (assign === 'A') return xfPos <= 0.5 ? 1 : Math.cos((xfPos - 0.5) * Math.PI);
  if (assign === 'B') return xfPos >= 0.5 ? 1 : Math.cos((0.5 - xfPos) * Math.PI);
  return 1;
}
const masterP = { fader: store.get('master', 0.72) };
function applyMaster(now) {
  if (!ctx) return;
  const g = LAW.fader.val(masterP.fader);
  now ? master.gain.setValueAtTime(g, ctx.currentTime) : master.gain.setTargetAtTime(g, ctx.currentTime, 0.012);
}

// ---------- controls ----------
const TAU = Math.PI * 2, A0 = (-135 * Math.PI) / 180, A1 = (135 * Math.PI) / 180;
const pt = (a, r) => `${(23 + r * Math.sin(a)).toFixed(2)},${(23 - r * Math.cos(a)).toFixed(2)}`;
function arc(a, b, r) {
  if (Math.abs(b - a) < 1e-3) return '';
  const [s, e] = a < b ? [a, b] : [b, a];
  return `M${pt(s, r)} A${r},${r} 0 ${e - s > Math.PI ? 1 : 0} 1 ${pt(e, r)}`;
}
// a knob: drag up/down (shift: fine), scroll, arrows; double-click resets
function knob(parent, { label, law, get, set, title, color }) {
  const k = el('div', 'knob'); k.tabIndex = 0; k.setAttribute('role', 'slider'); k.setAttribute('aria-label', label);
  k.title = `${title}\ndrag up/down (shift: fine), scroll, arrow keys; double-click resets`;
  k.innerHTML = `<svg viewBox="0 0 46 46" aria-hidden="true"><path class="arc0" d="${arc(A0, A1, 18)}"/><path class="arc1" stroke="${color}"/><circle class="cap" cx="23" cy="23" r="12.5"/><line class="ind" x1="23" y1="23" x2="23" y2="13"/></svg><div class="kt">${label}</div><div class="kv"></div>`;
  parent.appendChild(k);
  const val = k.querySelector('.arc1'), ind = k.querySelector('.ind'), kv = k.querySelector('.kv');
  const draw = () => {
    const p = get(), a = A0 + (A1 - A0) * p, base = law.bi ? 0 : law.from === 1 ? A1 : A0;
    val.setAttribute('d', arc(base, a, 18));
    ind.setAttribute('x1', pt(a, 4).split(',')[0]); ind.setAttribute('y1', pt(a, 4).split(',')[1]);
    ind.setAttribute('x2', pt(a, 11).split(',')[0]); ind.setAttribute('y2', pt(a, 11).split(',')[1]);
    kv.textContent = law.fmt(law.val(p));
    k.classList.toggle('changed', Math.abs(p - law.def) > 0.004);
    k.setAttribute('aria-valuetext', kv.textContent);
  };
  const to = (p) => { set(clamp(p, 0, 1)); draw(); };
  let drag = null;
  k.addEventListener('pointerdown', (e) => { k.setPointerCapture(e.pointerId); drag = { y: e.clientY, p: get() }; e.preventDefault(); k.focus(); });
  k.addEventListener('pointermove', (e) => { if (drag) to(drag.p + (drag.y - e.clientY) / (e.shiftKey ? 700 : 170)); });
  k.addEventListener('pointerup', () => { drag = null; });
  k.addEventListener('wheel', (e) => { e.preventDefault(); to(get() + (e.deltaY < 0 ? 1 : -1) * (e.shiftKey ? 0.005 : 0.025)); }, { passive: false });
  k.addEventListener('dblclick', () => to(law.def));
  k.addEventListener('keydown', (e) => {
    const d = { ArrowUp: 0.02, ArrowRight: 0.02, ArrowDown: -0.02, ArrowLeft: -0.02, PageUp: 0.1, PageDown: -0.1 }[e.key];
    if (d != null) { e.preventDefault(); e.stopPropagation(); to(get() + d); } else if (e.key === 'Home') to(0); else if (e.key === 'End') to(1);
  });
  draw();
  return { draw };
}
// a fader (vertical) or the crossfader (horizontal): grab anywhere, scroll, arrows; double-click resets
function slider(elm, { get, set, def, vertical = true, label, fmt }) {
  elm.tabIndex = 0; elm.setAttribute('role', 'slider'); elm.setAttribute('aria-label', label);
  const th = elm.querySelector('.th') || elm.appendChild(el('div', 'th'));
  const draw = () => {
    const p = get();
    if (vertical) th.style.top = `calc(8px + (100% - 16px) * ${(1 - p).toFixed(4)})`; else th.style.left = `calc(8px + (100% - 16px) * ${p.toFixed(4)})`;
    elm.setAttribute('aria-valuetext', fmt ? fmt(p) : String(Math.round(p * 100)));
  };
  const at = (e) => { const r = elm.getBoundingClientRect(); return vertical ? 1 - (e.clientY - r.top - 8) / (r.height - 16) : (e.clientX - r.left - 8) / (r.width - 16); };
  const to = (p) => { set(clamp(p, 0, 1)); draw(); };
  let drag = null;
  elm.addEventListener('pointerdown', (e) => { elm.setPointerCapture(e.pointerId); drag = true; to(at(e)); e.preventDefault(); elm.focus(); });
  elm.addEventListener('pointermove', (e) => { if (drag) to(at(e)); });
  elm.addEventListener('pointerup', () => { drag = null; });
  elm.addEventListener('wheel', (e) => { e.preventDefault(); to(get() + (e.deltaY < 0 ? 1 : -1) * 0.02); }, { passive: false });
  elm.addEventListener('dblclick', () => to(def));
  elm.addEventListener('keydown', (e) => {
    const d = { ArrowUp: 0.02, ArrowRight: 0.02, ArrowDown: -0.02, ArrowLeft: -0.02, PageUp: 0.1, PageDown: -0.1 }[e.key];
    if (d != null) { e.preventDefault(); e.stopPropagation(); to(get() + d); }
  });
  draw();
  return { draw };
}
function meterCanvas() { const c = el('canvas', 'meter'); c.width = 8; c.height = 170; return c; }
function drawMeter(c, an, st) {
  const g = c.getContext('2d'), h = c.clientHeight || c.height;
  if (c.height !== h) c.height = h;
  let peak = 0;
  if (an && ctx && ctx.state === 'running') {
    const buf = st.buf || (st.buf = new Float32Array(an.fftSize));
    an.getFloatTimeDomainData(buf);
    for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
  }
  const db = peak > 0 ? 20 * Math.log10(peak) : -90, p = clamp((db + 48) / 48, 0, 1);
  st.v = Math.max(p, (st.v || 0) - 0.02); st.hold = p >= (st.hold || 0) ? p : Math.max(0, (st.hold || 0) - 0.004);
  g.clearRect(0, 0, c.width, h);
  const y = (v) => h - v * h;
  const grad = g.createLinearGradient(0, h, 0, 0);
  grad.addColorStop(0, '#2f6b4a'); grad.addColorStop(0.7, '#7fe0a8'); grad.addColorStop(0.86, '#f2c14e'); grad.addColorStop(1, '#f2545b');
  g.fillStyle = grad; g.fillRect(0, y(st.v), c.width, h - y(st.v));
  g.fillStyle = st.hold > 0.97 ? '#f2545b' : '#c9ced6'; g.fillRect(0, Math.max(0, y(st.hold) - 1), c.width, 1.5);
}

// ---------- a deck: its brain, its composition, its channel strip ----------
class Channel {
  constructor(d, parent) {
    this.d = d; this.name = store.get('deck' + d, null); this.loaded = null; this.playing = false; this.pending = 0;
    this.mirror = []; this.now = null; this.pool = null; this.poolName = null; this.seq = 0; this.meterState = {};
    this.p = Object.assign({ hi: 0.5, mid: 0.5, lo: 0.5, hp: 0, lp: 1, fader: 0.8, mute: false, xa: 'thru', fx: 0, fxType: 'vanish' }, store.get('ch' + d, {}));
    if (this.p.fxType === 'echo' && !store.get('fxChosen' + d, false)) this.p.fxType = 'vanish';      // the hand-set effect starts as vanish (an older stored echo was the old default)
    this.auto = false; this.track = null; this.trackAt = 0; this.role = ''; this.kept = null;
    this.dom(parent);
  }
  save() { if (!this.auto) store.set('ch' + this.d, this.p); }
  dom(parent) {
    const c = el('section', 'ch'); c.setAttribute('aria-label', `deck ${this.d + 1}`); parent.appendChild(c); this.sec = c;
    const h = el('div', 'chh'); c.appendChild(h);
    this.orb = el('span', 'orb idle'); this.roleEl = el('span', 'role'); h.append(el('span', 'num', String(this.d + 1)), this.orb, this.roleEl);
    this.sel = el('select'); this.sel.title = `the brain of deck ${this.d + 1}`; h.appendChild(this.sel);
    this.what = el('span', 'what'); this.what.hidden = true; h.appendChild(this.what);      // Auto DJ: what the deck plays
    this.sel.addEventListener('change', () => { this.switchTo(this.sel.value); this.sel.blur(); });
    this.playB = el('button', 'sb', '▶'); this.playB.title = `play deck ${this.d + 1} (key ${this.d + 1}): a new composition`;
    this.stopB = el('button', 'sb off', '■'); this.stopB.title = `stop deck ${this.d + 1}: ▶ then starts a new composition`;
    this.playB.onclick = () => (this.playing ? this.stop() : this.play());
    this.stopB.onclick = () => this.stop();
    h.append(this.playB, this.stopB);
    this.lanesEl = el('div', 'lanes'); c.appendChild(this.lanesEl);
    this.lanes = LAYERS.map((n) => {
      const row = el('div', 'lane silent'); row.style.setProperty('--c', `var(--${n})`);
      const ln = el('span', 'ln', n); row.appendChild(ln);
      const cells = Array.from({ length: 16 }, (_, i) => { const x = el('span', 'cell' + (i % 4 === 0 && i ? ' q' : '')); row.appendChild(x); return x; });
      this.lanesEl.appendChild(row);
      return { row, ln, cells, on: new Array(16).fill(false) };
    });
    this.info = el('div', 'info', 'stopped'); c.appendChild(this.info);
    const strip = el('div', 'strip'); c.appendChild(strip);
    const kn = el('div', 'knobs'); strip.appendChild(kn);
    const accent = '#8d959f';
    const mk = (key, label, law, title) => knob(kn, { label, law, title, color: accent, get: () => this.p[key], set: (v) => { this.p[key] = v; this.apply(); this.save(); } });
    this.knobs = [
      mk('hi', 'HI', LAW.eq, 'treble (shelf above 5.5 kHz): −26..+6 dB, kill at the left stop'),
      mk('mid', 'MID', LAW.eq, 'mids (around 1.1 kHz): −26..+6 dB, kill at the left stop'),
      mk('lo', 'LOW', LAW.eq, 'bass (shelf below 150 Hz): −26..+6 dB, kill at the left stop'),
      mk('hp', 'HP', LAW.hp, 'highpass filter: off at the left stop, up to 2.5 kHz'),
      mk('lp', 'LP', LAW.lp, 'lowpass filter: off at the right stop, down to 180 Hz'),
      mk('fx', 'FX', LAW.fx, 'how much of the effect is heard (the button below picks it): echo and reverb ring on beside the track, the flanger sweeps it, the gate chops it in time'),
    ];
    const r2 = el('div', 'row2'); kn.appendChild(r2);
    this.fxB = el('button', 'tg fxb', FX_LABEL[this.p.fxType] || 'ECHO'); this.fxB.title = 'the effect of this deck: echo, reverb, flanger, gate (click for the next)'; r2.appendChild(this.fxB);
    this.fxB.onclick = () => { store.set('fxChosen' + this.d, true); this.setFx(FX[(FX.indexOf(this.p.fxType) + 1) % FX.length]); };
    this.muteB = el('button', 'tg mute', 'M'); this.muteB.title = 'mute this deck'; r2.appendChild(this.muteB);
    this.muteB.onclick = () => { this.p.mute = !this.p.mute; this.apply(); this.save(); this.paintToggles(); };
    const xa = el('div', 'xa'); xa.title = 'crossfader side: A, thru (not on the crossfader), B'; r2.appendChild(xa);
    this.xaB = ['A', 'thru', 'B'].map((v) => { const b = el('button', '', v === 'thru' ? '·' : v); b.setAttribute('aria-label', `crossfader ${v}`); b.onclick = () => { this.p.xa = v; this.apply(); this.save(); this.paintToggles(); }; xa.appendChild(b); return b; });
    const fz = el('div', 'fz'); strip.appendChild(fz);
    const col = el('div'); col.style.display = 'flex'; col.style.flexDirection = 'column'; col.style.gap = '4px';
    const f = el('div', 'fader');
    const tick = el('div', 'tick'); tick.style.top = `calc(8px + (100% - 16px) * 0.2)`; f.appendChild(tick);
    this.fv = el('div', 'fv');
    col.append(f, this.fv); fz.appendChild(col);
    this.meter = meterCanvas(); fz.appendChild(this.meter);
    this.fader = slider(f, { label: `deck ${this.d + 1} fader`, def: LAW.fader.def, get: () => this.p.fader, set: (v) => { this.p.fader = v; this.apply(); this.save(); this.paintFader(); }, fmt: (p) => LAW.fader.fmt(LAW.fader.val(p)) });
    this.paintToggles(); this.paintFader();
  }
  paintFader() { this.fv.textContent = LAW.fader.fmt(LAW.fader.val(this.p.fader)); }
  paintToggles() {
    this.muteB.classList.toggle('on', this.p.mute);
    this.fxB.textContent = FX_LABEL[this.p.fxType] || 'ECHO'; this.fxB.classList.toggle('on', this.p.fx > 0.005);
    this.xaB.forEach((b, i) => b.classList.toggle('on', ['A', 'thru', 'B'][i] === this.p.xa));
  }
  build() {
    const g = () => ctx.createGain();
    const bq = (type, f, q) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; if (q != null) b.Q.value = q; return b; };
    this.n = { lo: bq('lowshelf', 150), mid: bq('peaking', 1100, 0.6), hi: bq('highshelf', 5500), hp: bq('highpass', 10, 0.9),
               lp: bq('lowpass', 22000, 0.9), cut: g(), xf: g(), mute: g(), fader: g(), meter: ctx.createAnalyser() };
    this.n.meter.fftSize = 1024;
    this.fx = new FxUnit(ctx); this.fx.setTempo(bpm); this.fx.setType(this.p.fxType);
    // two decks at full level must not sit in the master limiter: every deck enters its strip 4 dB down
    this.n.pre = g(); this.n.pre.gain.value = 0.63; node.connect(this.n.pre, this.d, 0); this.n.pre.connect(this.n.lo);
    // EQ and filters, then the cut (the autopilot's: the track stops feeding its echo), the effect, and the fader
    const chain = ['lo', 'mid', 'hi', 'hp', 'lp', 'cut'].map((k) => this.n[k]).concat([this.fx.input]);
    for (let i = 0; i < chain.length - 1; i++) chain[i].connect(chain[i + 1]);
    this.fx.output.connect(this.n.xf); this.n.xf.connect(this.n.mute); this.n.mute.connect(this.n.fader);
    this.n.fader.connect(this.n.meter); this.n.fader.connect(master);
    this.apply(true);
  }
  apply(now) {
    if (!this.n) return;
    const t = ctx.currentTime, set = (param, v) => (now ? param.setValueAtTime(v, t) : param.setTargetAtTime(v, t, 0.012));
    set(this.n.mute.gain, this.p.mute ? 0 : 1);
    set(this.n.xf.gain, xfGain(this.p.xa));
    if (this.auto) return;                                  // the autopilot holds the strip's lines
    set(this.n.lo.gain, LAW.eq.val(this.p.lo)); set(this.n.mid.gain, LAW.eq.val(this.p.mid)); set(this.n.hi.gain, LAW.eq.val(this.p.hi));
    set(this.n.hp.frequency, LAW.hp.val(this.p.hp) || 10); set(this.n.lp.frequency, LAW.lp.val(this.p.lp) || 22000);
    set(this.n.fader.gain, LAW.fader.val(this.p.fader));
    this.fx.set(this.p.fx, t);
    this.paintToggles();
  }
  setFx(type, opts) { this.p.fxType = type; if (this.fx) this.fx.setType(type, opts); this.save(); this.paintToggles(); }
  setRole(role) { this.role = role; this.roleEl.textContent = { primary: 'PRIMARY', in: 'IN', out: 'OUT' }[role] || ''; this.roleEl.className = 'role ' + role; this.sec.classList.toggle('primary', role === 'primary'); }
  // ---- the autopilot's side of the strip: lines on the context clock, in knob positions ----
  target(key) {
    const n = this.n;
    return { lo: [n.lo.gain, LAW.eq.val], mid: [n.mid.gain, LAW.eq.val], hi: [n.hi.gain, LAW.eq.val], hp: [n.hp.frequency, (p) => LAW.hp.val(p) || 10],
             lp: [n.lp.frequency, (p) => LAW.lp.val(p) || 22000], fader: [n.fader.gain, LAW.fader.val], cut: [n.cut.gain, (v) => v] }[key];
  }
  autoBegin() {
    if (this.auto) return;
    this.kept = { ...this.p }; this.auto = true;
    Object.assign(this.p, { hi: 0.5, mid: 0.5, lo: 0.5, hp: 0, lp: 1, fader: 0, fx: 0, mute: false, xa: 'thru' });
    const t = ctx.currentTime;
    for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'cut']) { const [param, law] = this.target(key); param.cancelScheduledValues(t); param.setValueAtTime(law(key === 'cut' ? 1 : this.p[key]), t); }
    this.fx.hold(t); this.fx.set(0, t); this.apply(true); this.sec.classList.add('auto'); this.redraw();
  }
  autoEnd() {
    if (!this.auto) return;
    const t = ctx.currentTime;
    for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'cut']) { const [param] = this.target(key); param.cancelScheduledValues(t); }
    this.n.cut.gain.setValueAtTime(1, t); this.fx.hold(t); this.fx.dry.gain.cancelScheduledValues(t); this.fx.dry.gain.setValueAtTime(1, t);
    this.auto = false; Object.assign(this.p, this.kept || {}); this.kept = null;
    this.fx.setType(this.p.fxType); this.apply(true); this.sec.classList.remove('auto'); this.redraw();
  }
  // a deck the autopilot has stopped: its strip rests at neutral with the fader down
  autoRest() {
    if (!this.auto) return;
    const t = ctx.currentTime;
    Object.assign(this.p, { hi: 0.5, mid: 0.5, lo: 0.5, hp: 0, lp: 1, fader: 0, fx: 0 });
    for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'cut']) { const [param, law] = this.target(key); param.cancelScheduledValues(t); param.setValueAtTime(law(key === 'cut' ? 1 : this.p[key]), t); }
    this.fx.hold(t); this.fx.set(0, t); this.fx.dry.gain.cancelScheduledValues(t); this.fx.dry.gain.setValueAtTime(1, t); this.redraw();
  }
  autoSet(key, v, t) { if (key === 'fx') this.fx.step(v, t); else { const [param, law] = this.target(key); param.setValueAtTime(law(v), t); } }
  autoLine(key, v, t) { if (key === 'fx') this.fx.ramp(v, t); else { const [param, law] = this.target(key); param.linearRampToValueAtTime(law(v), t); } }
  autoShow(values) { for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'fx']) this.p[key] = values[key]; this.redraw(); }
  gate(t0, barLen, depth) { if (this.fx) this.fx.scheduleGate(t0, barLen, depth); }
  redraw() { for (const k of this.knobs) k.draw(); this.fader.draw(); this.paintFader(); this.paintToggles(); }
  // ---- a deck that plays a whole track (the Auto DJ): the bars come from the track, not from a live brain ----
  cueTrack(track) {                                           // starts on the next bar line
    this.seq++; this.track = track; this.trackAt = 0; this.playing = true; this.mirror.length = 0; this.now = null;
    node.port.postMessage({ type: 'silence', deck: this.d });
    this.pool = track.pool; this.poolName = track.poolName; this.sendPool();
    this.feed(0); this.requeued = true; this.light('ok', track.checkpoint || 'track'); this.paint();
  }
  feed(queued) {
    const t = this.track;
    if (!t || !node) return;
    while (queued < 3) {
      const n = t.bars.length;
      // past the end of a finished track its last section goes round (the next track is late)
      const k = this.trackAt < n ? this.trackAt : t.done && n >= 32 ? n - 32 + ((this.trackAt - n) % 32) : -1;
      if (k < 0) break;
      const j = t.bars[k];
      this.mirror.push(j); this.trackAt++; queued++;
      node.port.postMessage({ type: 'queue', deck: this.d, tokens: j.tokens, mods: j.mods, patterns: j.patterns || null, patches: j.patches || null });
    }
  }
  endTrack() {
    if (!this.track) return;
    this.track = null; this.seq++; this.playing = false;
    if (node) node.port.postMessage({ type: 'silence', deck: this.d });
    this.mirror.length = 0; this.now = null; this.paint(); this.orb.className = 'orb idle'; this.orb.title = 'stopped';
  }
  light(state, msg) {
    const was = this.loadingBar;
    this.loadingBar = state === 'loading';
    if (this.loadingBar) this.loadMsg = msg || this.loadMsg || 'loading…';
    this.orb.className = 'orb ' + ({ ok: '', loading: 'loading', down: 'down', idle: 'idle' }[state] || '');
    this.orb.style.removeProperty('--load');
    this.orb.title = msg || state;
    if (msg && state !== 'ok') status(`deck ${this.d + 1}: ${msg}`);
    else if (state === 'ok' && $('status').textContent.startsWith(`deck ${this.d + 1}:`)) status('');
    if (this.loadingBar) this.paintLoad();
    else if (was) { this.loadDone = state === 'ok' ? performance.now() : 0; this.paint(); }
    watchLoads();
  }
  // this deck's load as the server reports it (/api/load/progress): a bar with the percent and the stage in words
  loadRecord() {
    const r = loads.rec && loads.rec.decks ? loads.rec.decks[this.d] : null;
    return r && !r.error && (!r.finished || r.ago < 4) ? r : null;
  }
  paintLoad() {
    const r = this.loadRecord(), known = !!r && Number.isFinite(r.percent), pct = known ? Math.floor(r.percent) : null;
    const base = this.loadMsg.replace(/…$/, '');
    if (!this.lb) {                                   // made once: the fill slides from one reading to the next
      this.lb = { bar: el('span', 'lbar'), fill: el('span'), text: el('span', 'lt') };
      this.lb.bar.setAttribute('role', 'progressbar'); this.lb.bar.setAttribute('aria-label', `deck ${this.d + 1} loads a brain`);
      this.lb.bar.appendChild(this.lb.fill);
    }
    const { bar, fill, text } = this.lb;
    bar.classList.remove('done');
    bar.classList.toggle('indet', !known);
    fill.style.width = known ? `${r.percent}%` : '';
    if (known) bar.setAttribute('aria-valuenow', String(pct)); else bar.removeAttribute('aria-valuenow');
    const left = known && !r.finished && r.percent < 99 ? r.left_text || '' : '';
    text.textContent = known ? `${pct}% · ${r.text}${left ? ` · ${left}` : ''}` : this.loadMsg;
    this.info.className = 'info loading';
    if (this.info.firstChild !== bar) this.info.replaceChildren(bar, text);
    this.orb.classList.toggle('pct', known);
    if (known) this.orb.style.setProperty('--load', r.percent);
    this.orb.title = known ? `${base} · ${r.text} · ${pct}%` : this.loadMsg;
    if (known) status(`deck ${this.d + 1}: ${base} · ${r.text} · ${pct}%${left ? ` · ${left}` : ''}`);
  }
  // the server loads this deck's brain; the other decks play on meanwhile
  async load(name) {
    this.light('loading', `loading ${name}…`);
    let j;
    this.loading = true; this.paint();
    try { j = await post('/api/deck/load', { deck: this.d, name }, 300000); } catch { this.light('down', 'no answer from the server'); return false; } finally { this.loading = false; }
    if (name !== this.name) return false;                     // another brain was picked meanwhile
    if (j.error) { this.light('down', j.error); return false; }
    this.loaded = name;
    if (j.pool !== this.poolName) { this.pool = await (await fetch(`/api/deck/pool?deck=${this.d}`)).json(); this.poolName = j.pool; }
    this.light('ok');
    return true;
  }
  sendPool() { if (node && this.pool) node.port.postMessage({ type: 'pool', deck: this.d, pool: this.pool }); }
  async play() {
    if (this.auto) { status('the Auto DJ holds the decks: switch it off to play a deck by hand'); return; }
    if (!this.name) { status(`deck ${this.d + 1}: pick a brain first`); return; }
    await ensureAudio();
    const seq = ++this.seq;
    this.playing = true; this.paint(); paintTransport();
    if (this.loaded !== this.name) { if (!(await this.load(this.name))) { if (seq === this.seq) this.stop(); return; } }
    else { try { await post('/api/deck/new', { deck: this.d }); } catch { this.light('down', 'no answer from the server'); } }   // a new composition
    if (seq !== this.seq) return;
    node.port.postMessage({ type: 'silence', deck: this.d });
    this.sendPool(); this.mirror.length = 0;
    if (userBpm == null && this.pool && this.pool.bpm && !chans.some((c) => c !== this && c.playing && c.mirror.length)) setBpm(this.pool.bpm, false);
    if (ctx.state !== 'running') await ctx.resume();
    for (let i = 0; i < AHEAD; i++) await this.request(seq);
    paintTransport();
  }
  stop() {
    if (this.track) return;                                   // the Auto DJ's deck: it stops its own tracks
    this.seq++; this.playing = false;
    if (node) node.port.postMessage({ type: 'silence', deck: this.d });
    this.mirror.length = 0; this.now = null; this.paint(); paintTransport();
    this.light('idle', 'stopped');
  }
  async switchTo(name) {
    this.name = name; store.set('deck' + this.d, name);
    if (!this.playing) { this.paint(); return; }               // loaded on ▶
    const seq = ++this.seq;
    if (!(await this.load(name)) || seq !== this.seq) return;
    node.port.postMessage({ type: 'flush', deck: this.d });     // the next bar is the new brain's
    this.mirror.length = 0; this.sendPool();
    for (let i = 0; i < AHEAD; i++) await this.request(seq);
  }
  async request(seq = this.seq) {
    if (!this.playing || this.pending || seq !== this.seq) return;
    this.pending++;
    try {
      let j;
      try { j = await post('/api/deck/next', { deck: this.d, mood: MOOD }, 8000); }
      catch { this.light('down', 'no answer from the server: is it running?'); setTimeout(() => this.top(), 2000); return; }
      if (!this.playing || seq !== this.seq) return;
      if (j.loading) { this.light('loading', `loading ${j.loading}…`); setTimeout(() => this.top(), 700); return; }
      if (j.empty) { this.loaded = null; setTimeout(() => this.revive(seq), 0); return; }   // the server let the brain go (idle): load it again
      if (j.error) { this.light('down', j.error); return; }
      this.light('ok', this.name);
      this.mirror.push(j);
      node.port.postMessage({ type: 'queue', deck: this.d, tokens: j.tokens, mods: j.mods, patterns: j.patterns || null, patches: j.patches || null });
    } finally { this.pending--; }
  }
  async revive(seq) { if (seq === this.seq && this.playing && (await this.load(this.name)) && seq === this.seq) { this.sendPool(); this.top(); } }
  top() { if (this.track) return; if (this.playing && this.mirror.length < AHEAD && !this.pending) this.request(); }
  onBar(queued) {
    while (this.mirror.length > Math.max(0, queued | 0)) this.now = this.mirror.shift();
    this.paint(); this.top(); vizBar(this);
  }
  paint() {
    this.playB.textContent = '▶'; this.playB.classList.toggle('on', this.playing);
    this.playB.title = this.playing ? `deck ${this.d + 1} plays; ▶ again stops it` : `play deck ${this.d + 1} (key ${this.d + 1}): a new composition`;
    this.stopB.classList.toggle('off', !this.playing);
    const nb = this.now;
    for (let l = 0; l < 4; l++) {
      const lane = this.lanes[l], tok = nb ? nb.tokens[l] : 0;
      let on = new Array(16).fill(false);
      if (nb && tok > 0) {
        const pat = (nb.patterns && nb.patterns[l]) || (this.pool && this.pool.layers[l] && this.pool.layers[l].tokens[tok] && this.pool.layers[l].tokens[tok].pattern) || null;
        if (pat && pat.hits) for (const a of Object.values(pat.hits)) { if (Array.isArray(a)) a.forEach((v, i) => { if (i < 16 && +v > 0) on[i] = true; }); }
        else if (pat && pat.notes) pat.notes.forEach((v, i) => { if (i < 16 && v != null) on[i] = true; });
      }
      lane.on = on;
      lane.row.classList.toggle('silent', !nb || tok === 0 || !on.some(Boolean));
      lane.ln.textContent = tok > 0 ? `${LAYERS[l]}` : LAYERS[l];
      lane.cells.forEach((c, i) => c.classList.toggle('on', on[i]));
    }
    if (this.loadingBar) this.paintLoad();
    else if (this.loadDone && this.lb && performance.now() - this.loadDone < 900) {     // a brain came in: the bar at 100% for a moment
      const { bar, fill, text } = this.lb;
      bar.classList.remove('indet'); bar.classList.add('done'); fill.style.width = '100%'; text.textContent = '100% · ready';
      this.info.className = 'info loading';
      if (this.info.firstChild !== bar) this.info.replaceChildren(bar, text);
      setTimeout(() => { if (!this.loadingBar) this.paint(); }, 950);
    } else {
      this.info.className = 'info';
      this.info.textContent = this.loading ? 'loading…' : !this.playing ? (this.auto ? '' : this.name ? '' : 'pick a brain') : nb ? `bar ${nb.bar}${this.track ? ` of ${this.track.bars.length}` : ''}` : 'starting…';
    }
    this.sel.hidden = this.auto; this.what.hidden = !this.auto;
    if (this.auto) {
      const t = this.track, brain = t && t.checkpoint ? t.checkpoint.replace(/^slate_/, '') : 'a brain';
      this.what.textContent = !t ? 'free deck' : t.name ? `repertoire piece · ${t.name}` : t.kept ? `kept track · composed by ${brain}` : `new track · composed by ${brain}`;
      this.what.title = !t ? 'the Auto DJ has no track on this deck' : `${this.what.textContent}\n${t.name ? `composed by ${brain}; played from its notes on this deck's instrument (its MP3 is the same piece as a file)` : 'played from its notes on this deck\'s instrument'}`;
      this.what.className = 'what' + (!t ? ' free' : t.name ? ' rep' : '');
    }
    this.paintStep();
  }
  paintStep() {
    for (const lane of this.lanes) lane.cells.forEach((c, i) => c.classList.toggle('now', this.playing && i === step));
  }
}

// ---------- brain loads: the decks' and the Auto DJ's, from the server's /api/load/progress ----------
// The page asks every 500 ms while a deck loads or the Auto DJ is on. A server started before the route answers 404:
// then the bars show a running stripe and the words of the old messages.
const loads = { rec: null, timer: null, supported: null };
async function pollLoads() {
  if (loads.supported === false) return;
  try {
    if (loads.supported === null) {                // a server with the route names each deck's record in /api/decks
      const d = await (await fetch('/api/decks', { cache: 'no-store' })).json();
      loads.supported = !!(d.decks && d.decks[0] && 'progress' in d.decks[0]);
      if (!loads.supported) { watchLoads(); return; }
    }
    const response = await fetch('/api/load/progress', { cache: 'no-store' });
    if (!response.ok) return;
    loads.rec = await response.json();
  } catch { return; }                              // the next tick asks again
  for (const c of chans) if (c.loadingBar) c.paintLoad();
  paintDjSay();
}
function watchLoads() {
  const need = loads.supported !== false && (chans.some((c) => c.loadingBar) || (dj && dj.on));
  if (need && !loads.timer) { loads.timer = setInterval(pollLoads, 500); pollLoads(); }
  else if (!need && loads.timer) { clearInterval(loads.timer); loads.timer = null; loads.rec = null; paintDjSay(); }
}
// the Auto DJ's readout with its factory's load: a bar beside the words while it finds or loads a brain
let djText = '';
function paintDjSay() {
  const r = loads.rec && loads.rec.dj, active = !!(dj && dj.on && r && !r.finished && !r.error);
  const named = /^(finding|loading) /.test(djText), box = $('djLoad');
  let text = djText;
  if (active && !named) text += `${text ? ' · ' : ''}the factory loads ${String(r.name).replace(/^slate_/, '')}: ${r.text} · ${Math.floor(r.percent)}%${r.left_text ? ` · ${r.left_text}` : ''}`;
  $('djSay').textContent = text; $('djSay').title = text;
  box.hidden = !(dj && dj.on && (active || named));
  box.classList.toggle('indet', !active);
  box.firstChild.style.width = active ? `${r.percent}%` : '';
  if (active) box.setAttribute('aria-valuenow', String(Math.floor(r.percent))); else box.removeAttribute('aria-valuenow');
}

// ---------- the visuals: the player's (brainviz.js), fed by the deck in front and the master's sound ----------
const viz = new BrainViz($('viz'));
let stepAt = performance.now(), vizDeck = -1, vizTrack = null;
const vizHistory = [];
function front() {                                              // the primary deck, or the loudest deck that plays
  const playing = chans.filter((c) => c.playing && c.now);
  return playing.find((c) => c.role === 'primary') || playing.sort((x, y) => y.p.fader - x.p.fader)[0] || null;
}
function vizBar(ch) {                                           // a deck began a bar: the visuals follow the deck in front
  const f = front();
  if (!f || f !== ch || !ch.now) return;
  if (vizDeck !== ch.d || vizTrack !== ch.track) { vizDeck = ch.d; vizTrack = ch.track; vizHistory.length = 0; viz.setLayout(ch.track && ch.track.layout ? ch.track.layout : null); }
  const nb = ch.now;
  vizHistory.push({ bar: nb.bar, tokens: nb.tokens.slice(), mods: nb.mods }); if (vizHistory.length > 96) vizHistory.shift();
  viz.setBar({ ...nb, mode: 'brain', checkpoint: ch.track ? ch.track.checkpoint : ch.name }, vizHistory);
}
function vizStep() {
  const f = front(), p = f && f.now && f.now.patterns && f.now.patterns[0];
  viz.step(step, p && p.hits && p.hits.kick ? p.hits.kick[step] > 0 : step % 4 === 0);
}
{
  const sel = $('vizMode');
  for (const [group, modes] of Object.entries(MODE_GROUPS)) { const og = document.createElement('optgroup'); og.label = group === 'demo' ? 'demo scene' : 'the brain'; for (const m of modes) og.appendChild(new Option(m, m)); sel.appendChild(og); }
  const setMode = (m) => { viz.setMode(m); sel.value = viz.modeName; store.set('vizMode', viz.modeName); };
  setMode(store.get('vizMode', 'cadance'));
  sel.addEventListener('change', () => { setMode(sel.value); sel.blur(); });
  const full = () => { const box = $('vizbox'); if (document.fullscreenElement) document.exitFullscreen(); else if (box.requestFullscreen) box.requestFullscreen().catch(() => {}); };
  $('vizFull').onclick = full; $('viz').ondblclick = full;
  document.addEventListener('fullscreenchange', () => { setTimeout(() => viz.resize(), 60); $('vizFull').textContent = document.fullscreenElement ? '⛶ leave' : '⛶ full screen'; });
  addEventListener('keydown', (e) => {
    if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'f') full(); else if (e.key === 'm') setMode(viz.mode + 1);
  });
}

// ---------- the desk ----------
const desk = $('desk');
const chans = Array.from({ length: DECKS }, (_, d) => new Channel(d, desk));
const masterUi = {};
(() => {                                                       // the master section
  const m = el('section', 'ch master'); m.setAttribute('aria-label', 'master'); desk.appendChild(m);
  m.appendChild(el('div', 'mt', 'MASTER'));
  const fz = el('div', 'fz'); m.appendChild(fz);
  const col = el('div'); col.style.display = 'flex'; col.style.flexDirection = 'column'; col.style.gap = '4px';
  const f = el('div', 'fader');
  const tick = el('div', 'tick'); tick.style.top = `calc(8px + (100% - 16px) * 0.2)`; f.appendChild(tick);
  const fv = el('div', 'fv');
  col.append(f, fv); fz.appendChild(col);
  const mc = meterCanvas(); fz.appendChild(mc);
  const paint = () => { fv.textContent = LAW.fader.fmt(LAW.fader.val(masterP.fader)); };
  slider(f, { label: 'master fader', def: 0.72, get: () => masterP.fader, set: (v) => { masterP.fader = v; store.set('master', v); applyMaster(); paint(); }, fmt: (p) => LAW.fader.fmt(LAW.fader.val(p)) });
  paint();
  masterUi.meter = mc;
})();
slider($('xf'), { vertical: false, label: 'crossfader', def: 0.5, get: () => xfPos, set: (v) => { xfPos = v; store.set('xf', v); for (const ch of chans) ch.apply(); } });

// brains: the same list as the player (newest first, then the ones placed by hand)
async function loadBrains() {
  let j;
  try { j = await (await fetch('/api/checkpoints')).json(); } catch { $('orb').className = 'orb down'; status('server not reachable'); return; }
  $('orb').className = 'orb';
  const list = j.checkpoints || [];
  chans.forEach((ch, d) => {
    if (document.activeElement === ch.sel) return;
    ch.sel.innerHTML = '';
    ch.sel.add(new Option('— no brain —', ''));
    for (const c of list) ch.sel.add(new Option(`${arrived(c)}  ${c.label || c.candidate || c.name}`, c.name));
    if (!ch.name || !list.some((c) => c.name === ch.name)) ch.name = list[d] ? list[d].name : null;
    ch.sel.value = ch.name || '';
    ch.paint();
  });
}

// ---------- transport and tempo ----------
function anyPlaying() { return chans.some((c) => c.playing); }
function paintTransport() {
  const running = ctx && ctx.state === 'running' && anyPlaying();
  $('play').textContent = running ? '⏸' : '▶';
  $('play').title = running ? 'pause all decks (space)' : anyPlaying() ? 'go on (space)' : 'play all decks (space): each starts a new composition';
  $('stop').classList.toggle('off', !anyPlaying());
}
$('play').onclick = async () => {
  if (dj.on && !anyPlaying()) return;                          // the Auto DJ is starting
  if (!anyPlaying()) { await ensureAudio(); await Promise.all(chans.filter((c) => c.name).map((c) => c.play())); }
  else if (ctx.state === 'running') await ctx.suspend();
  else await ctx.resume();
  paintTransport();
};
$('stop').onclick = () => { if (dj.on) dj.stop(); for (const c of chans) if (c.playing) c.stop(); paintDj(); };
function setBpm(v, mine = true) {
  bpm = clamp(Math.round(+v || 126), 60, 200);
  if (mine) { userBpm = bpm; store.set('bpm', bpm); }
  if (node) node.port.postMessage({ type: 'tempo', bpm });
  for (const c of chans) if (c.fx) c.fx.setTempo(bpm);
  if (dj) dj.retime();
  if (document.activeElement !== $('bpm')) $('bpm').value = bpm;
}
{
  const f = $('bpm');
  f.addEventListener('change', () => setBpm(f.value));
  f.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') f.blur(); });
  f.addEventListener('wheel', (e) => { e.preventDefault(); setBpm(bpm + (e.deltaY < 0 ? 1 : -1)); f.value = bpm; }, { passive: false });
  $('bpmBox').addEventListener('dblclick', (e) => {
    e.preventDefault(); userBpm = null; store.set('bpm', null);
    const p = chans.find((c) => c.pool && c.pool.bpm); setBpm(p ? p.pool.bpm : 126, false); f.value = bpm; f.blur();
  });
  let drag = null;
  f.addEventListener('pointerdown', (e) => { drag = { y: e.clientY, v: bpm, moved: false }; f.setPointerCapture(e.pointerId); });
  f.addEventListener('pointermove', (e) => { if (!drag) return; const d = Math.round((drag.y - e.clientY) / 4); if (d) { drag.moved = true; setBpm(drag.v + d); f.value = bpm; } });
  f.addEventListener('pointerup', (e) => { if (drag && drag.moved) { e.preventDefault(); f.blur(); } drag = null; });
  setBpm(userBpm ?? 126, false);
}
addEventListener('keydown', (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === 'Space' && !(e.target.getAttribute && e.target.getAttribute('role') === 'slider')) { e.preventDefault(); $('play').click(); }
  else if (e.key >= '1' && e.key <= String(DECKS)) { const c = chans[+e.key - 1]; c.playing ? c.stop() : c.play(); }
});
function status(msg) { $('status').textContent = msg || ''; }

// ---------- the Auto DJ: an autopilot that plays whole tracks of one brain and mixes them (autodj.js) ----------
const query = new URLSearchParams(location.search);
dj = new AutoDJ({ chans, ctx: () => ctx, ensureAudio, bpm: () => bpm, userBpm: () => userBpm, setBpm, status,
                        paint: () => { paintTransport(); paintDj(); }, readout: (text) => { djText = text; paintDjSay(); watchLoads(); },
                        brainName: () => djShown },
                      { section: clamp(+query.get('section') || +store.get('djPace', 16) || 16, 4, 32) });
dj.debug = query.has('djdebug');
if (dj.debug) window.__mixer = { chans, dj, peak: (an) => { if (!an) return 0; const b = new Float32Array(an.fftSize); an.getFloatTimeDomainData(b); let p = 0; for (const v of b) p = Math.max(p, Math.abs(v)); return p; }, master: () => masterMeter };
function paintDj() {
  $('djB').classList.toggle('on', dj.on); $('djB').setAttribute('aria-pressed', String(dj.on));
  $('djB').textContent = dj.on ? 'auto dj · on' : 'auto dj';
  document.body.classList.toggle('djon', dj.on);   // the brain can change while it plays
  if (typeof paintPace === 'function') paintPace();
}
// the set: live tracks of the chosen brain, a repertoire piece as the transition between two of them (autodj.js);
// no other choice (owner, 26 Nov 09:48 and 09:52)
// the leader gate (site/dj-passed.json): while the brain field says "ranking leader" and no run has passed the play
// checks, the Auto DJ has no brain to start with and says so
let djGate = null;
$('djB').onclick = () => {
  if (dj.on) dj.stop();
  else if (djGate) { $('djSay').textContent = `${djGate}: pick a run in the brain field`; status(`Auto DJ: ${djGate}`); }
  else dj.start('approved');
  paintDj();
};
// The brain picker (owner, 26 Nov 08:40: "just tell it to use slate_split_s1_lag4in", "for now the top 10 ranked
// models"): one brain for all tracks, picked from the ten runs with the lowest newest held-out error or typed by
// name. It replaces the approved list; while the Auto DJ plays, the next track comes from the new brain.
let djRuns = [], djKept = [], djShown = '', listenStates = {};       // listenStates: the runs that play as a kept state (listen-states.json)
const shortRun = (c) => c.replace(/^slate_/, '');
async function chooseBrain(text) {
  const t = (text || '').trim();
  if (!t || t === djShown) { $('djModel').value = djShown; return; }
  let id = null;
  if (/^ranking leader/i.test(t)) id = 'leader';
  else {
    const name = t.split(/\s/)[0];
    const hit = djRuns.find((r) => r.candidate === name || shortRun(r.candidate) === name);
    const kept = djKept.find((k) => k === name || shortRun(k) === name);       // a kept state by its own name
    if (hit) id = `candidate:${hit.candidate}`;
    else if (kept) id = `checkpoint:${kept}`;
  }
  if (!id) { status(`no run named ${t}`); $('djModel').value = djShown; return; }
  try {
    let j = await post('/api/autodj/approved', { model: id, on: true });
    for (const other of (j.approved || [])) if (other !== id) j = await post('/api/autodj/approved', { model: other, on: false });
    paintModels(j);
    if (dj.on) dj.brainChanged();                                   // the next track is the new brain's; the decks play on
    status(`Auto DJ brain: ${brainName(id, j.leader)}${dj.on ? ', from the next track' : ''}`);
  } catch { status('the server did not answer'); }
}
$('djModel').addEventListener('input', (e) => {                    // a pick from the list takes effect at once
  const v = $('djModel').value;
  if (!e.inputType || e.inputType === 'insertReplacementText') { if ([...$('djModels').options].some((o) => o.value === v)) { chooseBrain(v); $('djModel').blur(); } }
});
$('djModel').addEventListener('change', () => { chooseBrain($('djModel').value); });
$('djModel').addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('djModel').value = djShown; $('djModel').blur(); } e.stopPropagation(); });
const stateOf = (run) => (listenStates[run] ? listenStates[run].slice(run.length).replace(/^_/, '') : null);   // e001b002
const brainName = (id, leader) => (id === 'leader' ? `ranking leader${leader ? ` (${leader.replace(/^slate_/, '')}${stateOf(leader) ? ` ${stateOf(leader)}` : ''})` : ''}`
  : id.replace(/^(candidate|checkpoint):/, '').replace(/^slate_/, '') + (id.startsWith('candidate:') ? ` (${stateOf(id.slice(10)) || 'newest'})` : ''));
function paintModels(j) {
  if (!j || j.error) return;
  listenStates = j.listen_states || {};
  djKept = (j.models || []).filter((m) => String(m.id).startsWith('checkpoint:')).map((m) => m.id.slice(11));
  const box = $('djBrains'), approved = j.approved || ['leader'];
  box.innerHTML = '';
  if (approved.length > 1) for (const id of approved) {             // an older list of several brains: shown until one is picked
    const chip = el('span', 'chip', brainName(id, j.leader)); chip.title = id === 'leader' ? 'the run that leads the Stats ranking, by its newest checkpoint' : id;
    const x = el('button', '', '✕'); x.title = 'remove from the approved brains'; x.setAttribute('aria-label', `remove ${brainName(id, j.leader)}`);
    x.onclick = async () => { try { paintModels(await post('/api/autodj/approved', { model: id, on: false })); } catch { status('the server did not answer'); } };
    chip.appendChild(x); box.appendChild(chip);
  }
  const one = approved.length === 1 ? approved[0] : null;
  // the ranking leader without a run that passed the play checks: the field stays empty and says why
  djGate = one === 'leader' && !j.leader ? (j.leader_gate || 'no brain has passed the play checks yet') : null;
  djShown = djGate ? '' : one === 'leader' ? 'ranking leader' : one ? shortRun(one.replace(/^(candidate|checkpoint):/, '')) : '';
  $('djModel').placeholder = djGate ? `${djGate}: type or pick a run` : approved.length > 1 ? 'pick one brain for all tracks' : 'type or pick a run';
  if (document.activeElement !== $('djModel')) $('djModel').value = djShown;
  if (!dj.on) { $('djSay').textContent = djGate ? `${djGate}: pick a run in the brain field` : ''; $('djSay').title = $('djSay').textContent; }
  const n = j.repertoire || 0;                                     // the transitions come from the repertoire
  $('djB').title = `Auto DJ: live tracks of the chosen brain, a repertoire piece as the transition between two of them while the brain composes the next one${n ? ` (${n} pieces in the repertoire)` : ' (the repertoire is empty: live tracks only)'}. The autopilot mixes on the three decks in sections of 16 or 32 bars; one deck is the primary, effects are used on the others only.`;
}
async function loadDjModels() {
  if (document.activeElement === $('djModel')) return;
  let j;
  try { j = await (await fetch('/api/autodj/models')).json(); } catch { return; }
  paintModels(j);
}
// the runs as Stats has them: composer runs with a measured reading, the four-passage probes left out (their
// held-out passages are easier, so their numbers do not compare), ordered by the newest held-out error
async function loadDjRuns() {
  let data;
  try { data = await (await fetch('/api/curves', { cache: 'no-store' })).json(); } catch { return; }
  const rows = [];
  for (const b of (data && data.brains) || []) {
    if (b.kind !== 'composer' || !b.candidate || b.lessons?.training_bars === 6144) continue;
    const pts = (b.points || []).filter((p) => Number.isFinite(p.train) && Number.isFinite(p.test));
    if (!pts.length) continue;
    const last = pts.at(-1);
    rows.push({ candidate: b.candidate, test: last.test, train: last.train, minutes: Math.max(...pts.map((p) => +p.minutes || 0)) });
  }
  rows.sort((x, y) => x.test - y.test || x.candidate.localeCompare(y.candidate));
  djRuns = rows;
  const list = $('djModels'); list.innerHTML = '';
  const lead = document.createElement('option'); lead.value = 'ranking leader'; lead.label = 'the run that leads the Stats ranking'; list.append(lead);
  for (const r of rows.slice(0, 10)) {
    const o = document.createElement('option'); o.value = shortRun(r.candidate);
    o.label = `held-out ${(r.test * 100).toFixed(1)}% · training ${(r.train * 100).toFixed(1)}% · ${Math.round(r.minutes)} min`
      + (stateOf(r.candidate) ? ` · plays ${stateOf(r.candidate)}` : '');
    list.append(o);
  }
  for (const k of djKept) {                                         // the kept states (site/pinned), by their own names
    const o = document.createElement('option'); o.value = shortRun(k); o.label = 'kept state'; list.append(o);
  }
}
$('djModel').addEventListener('focus', () => { $('djModel').select(); loadDjRuns(); });
loadDjRuns(); setInterval(loadDjRuns, 120000);
loadDjModels(); setInterval(loadDjModels, 120000); paintDj();

// Cadance Live Radio (/mixer?radio=1, the button on the player page): the Auto DJ starts at once with the chosen
// brain, a repertoire piece as the transition between two of its live tracks. Where the browser holds the audio back
// until a click on this page, one click on the radio button starts it.
if (query.get('radio') === '1') {
  const tuneIn = async () => {
    await ensureAudio();
    if (ctx.state !== 'running') { try { await ctx.resume(); } catch { /* needs a click */ } }
    if (ctx.state !== 'running') return false;
    if (!dj.on) {
      dj.start('approved');
      paintDj();
    }
    $('radioGate').hidden = true;
    return true;
  };
  tuneIn().then((ok) => {
    if (ok) return;
    $('radioGate').hidden = false;
    $('radioGo').onclick = () => { tuneIn(); };
  }).catch(() => { $('radioGate').hidden = false; $('radioGo').onclick = () => { tuneIn(); }; });
}
function paintTempo() { for (const b of document.querySelectorAll('.djt button[data-bpm]')) b.classList.toggle('on', b.dataset.bpm === 'own' ? userBpm == null : +b.dataset.bpm === bpm && userBpm != null); }
// the pace of the set: a section of 16 or 32 bars (taken up when the Auto DJ starts)
function paintPace() { for (const b of document.querySelectorAll('.djt button[data-sec]')) { b.classList.toggle('on', +b.dataset.sec === dj.section); b.disabled = dj.on; } }
for (const b of document.querySelectorAll('.djt button[data-sec]')) b.onclick = () => { if (dj.on) return; dj.section = +b.dataset.sec; store.set('djPace', dj.section); paintPace(); b.blur(); };
paintPace();
for (const b of document.querySelectorAll('.djt button[data-bpm]')) b.onclick = () => {
  if (b.dataset.bpm === 'own') { userBpm = null; store.set('bpm', null); const own = (dj.first && dj.first.bpm) || (chans.find((c) => c.pool && c.pool.bpm) || {}).pool?.bpm; setBpm(own || 126, false); }
  else setBpm(+b.dataset.bpm);
  $('bpm').value = bpm; paintTempo();
};
paintTempo();
setInterval(() => dj.pump(), 250);                             // steps drive the schedule; this covers a quiet engine
addEventListener('pagehide', () => { if (dj.on) navigator.sendBeacon('/api/autodj/stop', JSON.stringify({ session: dj.sid })); });

// meters at the screen's frame rate
function frame() {
  requestAnimationFrame(frame);
  if (!document.hidden) { const ms = 15000 / bpm; viz.frame(anyPlaying() && step >= 0 ? Math.min(0.999, (step + Math.min(1, (performance.now() - stepAt) / ms)) / 16) : 0, ms); }
  if (dj && dj.debug && window.__mixer) for (const c of chans) c.held = Math.max(c.held || 0, window.__mixer.peak(c.n && c.n.meter));   // the loudest moment since the test last asked
  if (document.hidden) return;
  for (const c of chans) drawMeter(c.meter, c.n && c.n.meter, c.meterState);
  drawMeter(masterUi.meter, masterMeter, masterUi.state || (masterUi.state = {}));
}
requestAnimationFrame(frame);
loadBrains();
setInterval(loadBrains, 60000);
status('pick a brain per deck, ▶ plays all decks (or 1, 2, 3 per deck); the decks always run in time');
