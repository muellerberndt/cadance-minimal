// Cadance Radio (radio.html): one screen of visuals; the Auto DJ (autodj.js) mixes the track store's MP3s out of sight
// on three file decks, each through a channel strip (EQ, filters, the DJ effects, a fader) as on the mixer, all on one
// bar clock of the AudioContext: bar k starts at clock.t0 + k * 240 / bpm, and a deck's MP3 starts on a bar line
// (AudioBufferSourceNode.start(when, offset)), so two decks run in time to the sample. The views follow the deck in
// front: the brain (brainviz.js, the 3D brain lit by the recorded view of the bar that sounds, from the track's note
// file, or by the audio), the instruments (the bar's drum steps and lines, lit on the step), and the other modes.
import { AutoDJ } from './autodj.js';
import { BrainViz, MODE_GROUPS } from './brainviz.js';
import { FxUnit } from './djfx.js';

const R = window.__radio;
await R.ready;
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('cadance.radio.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('cadance.radio.' + k, JSON.stringify(v)); } catch { /* private window */ } },
};
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const q = new URLSearchParams(location.search);
const DECKS = 3;
const TAGLINE = 'Build real-time brains that work like human brains.';
const LAW = {        // knob position 0..1 -> value, as on the mixer
  eq: { val: (p) => (p <= 0.015 ? -40 : p < 0.5 ? (-26 * (0.5 - p)) / 0.485 : (6 * (p - 0.5)) / 0.5) },
  hp: { val: (p) => (p <= 0.005 ? 0 : 20 * Math.pow(125, p)) },
  lp: { val: (p) => (p >= 0.995 ? 0 : 180 * Math.pow(20000 / 180, p)) },
  fader: { val: (p) => Math.pow(p / 0.8, 2.5) },
};
const calm = () => !!store.get('calm', false) || matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- audio and the bar clock ----------
let ctx = null, master = null, vol = null, bpm = 122.14, dj = null, note = '', starting = null, trackNo = 0;
const latency = () => (ctx ? (ctx.outputLatency || 0) + (ctx.baseLatency || 0) : 0);
const later = (t, fn) => { const ms = Math.max(0, (t + latency() - (ctx ? ctx.currentTime : 0)) * 1000); ms > 4 ? setTimeout(fn, ms) : fn(); };
const clock = { t0: 0, next: 0, bar: 0, step: -1, stepAt: 0, timer: null };
const barLen = () => 240 / bpm;
function ensureAudio() {
  if (ctx) return Promise.resolve();
  if (starting) return starting;
  starting = (async () => {
    const c = new AudioContext({ sampleRate: 44100, latencyHint: 'playback' });
    vol = c.createGain(); vol.gain.value = store.get('volume', 0.8);
    master = c.createGain();
    const lim = c.createDynamicsCompressor();
    lim.threshold.value = -1.5; lim.knee.value = 0; lim.ratio.value = 20; lim.attack.value = 0.002; lim.release.value = 0.12;
    master.connect(lim); lim.connect(vol); vol.connect(c.destination);
    ctx = c;
    viz.listen(c, master);
    for (const d of decks) d.build();
    clock.t0 = clock.next = c.currentTime + 0.3; clock.bar = 0;
    clock.timer = setInterval(tick, 25);
  })();
  return starting;
}
function tick() {                     // the bar lines within the next 150 ms: the work the engine's bar message does on the mixer
  if (!ctx || ctx.state !== 'running') return;
  while (clock.next <= ctx.currentTime + 0.15) {
    const t = clock.next, G = clock.bar++;
    clock.next = t + barLen();
    try { if (dj) dj.onBar(G, t); } catch (e) { say(`auto dj: ${e.message}`); }
    for (const d of decks) later(t, () => d.onBar(G));
    const len = barLen();
    for (let s = 0; s < 16; s++) later(t + (s * len) / 16, () => { clock.step = s; clock.stepAt = performance.now(); stepViews(s); });
    if (G % ((dj && dj.section) || 32) === 0) later(t, () => vjSection(G));
  }
  if (dj) dj.pump();
}

// ---------- the file decks: the mixer's channel strip, fed by an MP3 started on a bar line ----------
class Deck {
  constructor(d) {
    this.d = d; this.p = { hi: 0.5, mid: 0.5, lo: 0.5, hp: 0, lp: 1, fader: 0.8, fx: 0, fxType: 'vanish', mute: false };
    this.auto = false; this.track = null; this.trackAt = 0; this.role = ''; this.playing = false; this.now = null; this.src = null;
    this.mirror = []; this.startBar = 0;
  }
  build() {
    const g = () => ctx.createGain();
    const bq = (type, f, qq) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; if (qq != null) b.Q.value = qq; return b; };
    this.n = { lo: bq('lowshelf', 150), mid: bq('peaking', 1100, 0.6), hi: bq('highshelf', 5500), hp: bq('highpass', 10, 0.9),
               lp: bq('lowpass', 22000, 0.9), cut: g(), fader: g(), meter: ctx.createAnalyser(), pre: g() };
    this.n.meter.fftSize = 512;
    this.fx = new FxUnit(ctx); this.fx.setTempo(bpm); this.fx.setType(this.p.fxType);
    this.n.pre.gain.value = 0.63;                         // as on the mixer: two decks at full level stay out of the limiter
    this.n.pre.connect(this.n.lo);
    const chain = ['lo', 'mid', 'hi', 'hp', 'lp', 'cut'].map((k) => this.n[k]).concat([this.fx.input]);
    for (let i = 0; i < chain.length - 1; i++) chain[i].connect(chain[i + 1]);
    this.fx.output.connect(this.n.fader); this.n.fader.connect(this.n.meter); this.n.fader.connect(master);
    this.apply(true);
  }
  apply(now) {
    if (!this.n || this.auto) return;
    const t = ctx.currentTime, set = (param, v) => (now ? param.setValueAtTime(v, t) : param.setTargetAtTime(v, t, 0.012));
    set(this.n.lo.gain, LAW.eq.val(this.p.lo)); set(this.n.mid.gain, LAW.eq.val(this.p.mid)); set(this.n.hi.gain, LAW.eq.val(this.p.hi));
    set(this.n.hp.frequency, LAW.hp.val(this.p.hp) || 10); set(this.n.lp.frequency, LAW.lp.val(this.p.lp) || 22000);
    set(this.n.fader.gain, LAW.fader.val(this.p.fader));
    this.fx.set(this.p.fx, t);
  }
  setFx(type, opts) { this.p.fxType = type; if (this.fx) this.fx.setType(type, opts); }
  setRole(role) { this.role = role; }
  target(key) {
    const n = this.n;
    return { lo: [n.lo.gain, LAW.eq.val], mid: [n.mid.gain, LAW.eq.val], hi: [n.hi.gain, LAW.eq.val], hp: [n.hp.frequency, (p) => LAW.hp.val(p) || 10],
             lp: [n.lp.frequency, (p) => LAW.lp.val(p) || 22000], fader: [n.fader.gain, LAW.fader.val], cut: [n.cut.gain, (v) => v] }[key];
  }
  autoBegin() {
    if (this.auto) return;
    this.auto = true;
    Object.assign(this.p, { hi: 0.5, mid: 0.5, lo: 0.5, hp: 0, lp: 1, fader: 0, fx: 0 });
    const t = ctx.currentTime;
    for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'cut']) { const [param, law] = this.target(key); param.cancelScheduledValues(t); param.setValueAtTime(law(key === 'cut' ? 1 : this.p[key]), t); }
    this.fx.hold(t); this.fx.set(0, t);
  }
  autoEnd() {
    if (!this.auto) return;
    const t = ctx.currentTime;
    for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'cut']) { const [param] = this.target(key); param.cancelScheduledValues(t); }
    this.n.cut.gain.setValueAtTime(1, t); this.fx.hold(t); this.fx.dry.gain.cancelScheduledValues(t); this.fx.dry.gain.setValueAtTime(1, t);
    this.auto = false; this.apply(true);
  }
  autoRest() {
    if (!this.auto) return;
    const t = ctx.currentTime;
    Object.assign(this.p, { hi: 0.5, mid: 0.5, lo: 0.5, hp: 0, lp: 1, fader: 0, fx: 0 });
    for (const key of ['lo', 'mid', 'hi', 'hp', 'lp', 'fader', 'cut']) { const [param, law] = this.target(key); param.cancelScheduledValues(t); param.setValueAtTime(law(key === 'cut' ? 1 : this.p[key]), t); }
    this.fx.hold(t); this.fx.set(0, t); this.fx.dry.gain.cancelScheduledValues(t); this.fx.dry.gain.setValueAtTime(1, t);
  }
  autoSet(key, v, t) { if (key === 'fx') this.fx.step(v, t); else { const [param, law] = this.target(key); param.setValueAtTime(law(v), t); } }
  autoLine(key, v, t) { if (key === 'fx') this.fx.ramp(v, t); else { const [param, law] = this.target(key); param.linearRampToValueAtTime(law(v), t); } }
  autoShow(values) { Object.assign(this.p, values); }
  gate(t0, len, depth) { if (this.fx) this.fx.scheduleGate(t0, len, depth); }
  // ---- a track: its MP3 starts on the next bar line (later lines while it decodes: the track then starts a few bars in)
  cueTrack(track) {
    this.endTrack();
    this.track = track; this.trackAt = 0; this.playing = true; this.now = null;
    this.startBar = clock.bar; const line = clock.next;          // the next bar line is bar clock.bar
    trackNo++; track.no = trackNo;
    (R.onDeck ||= new Set()).add(track.id);
    R.decode(track.id).then((buf) => this.play(buf, track, line)).catch((e) => {
      if (this.track !== track) return;
      say(`${e.message}: the next track is tried`);
      this.failed = true;
    });
  }
  play(buf, track, line) {
    if (this.track !== track || !ctx) return;
    const len = barLen(), soon = ctx.currentTime + 0.06;
    let k = 0, when = line;
    if (when < soon) { k = Math.ceil((soon - when) / len); when += k * len; }       // decoded late: in on a later line
    const s = ctx.createBufferSource(), g = ctx.createGain();
    s.buffer = buf;
    const n = track.bars.length;
    if (n >= 32 && buf.duration >= n * len) { s.loop = true; s.loopStart = (n - 32) * len; s.loopEnd = n * len; }  // past the end its last section goes round
    s.connect(g); g.connect(this.n.pre);
    s.start(when, k * len);
    this.src = { s, g };
    if (window.__radioApp) (window.__radioApp.starts ||= []).push({ deck: this.d, id: track.id, when, line, offset: k * len, bar: this.startBar + k });
  }
  onBar(G) {
    if (!this.playing || !this.track) return;
    const i = G - this.startBar;
    if (i < 0) return;
    const n = this.track.bars.length, k = i < n ? i : Math.max(0, n - 32 + ((i - n) % 32));
    this.trackAt = i + 1; this.now = this.track.bars[k] || null; this.barIndex = k;
    frontBar(this);
  }
  endTrack() {
    const tr = this.track;
    if (!tr) return;
    this.track = null; this.playing = false; this.now = null; this.failed = false;
    if (this.src && ctx) {
      const { s, g } = this.src, t = ctx.currentTime;
      g.gain.setTargetAtTime(0, t, 0.01);
      try { s.stop(t + 0.08); } catch { /* not started */ }
      setTimeout(() => { try { s.disconnect(); g.disconnect(); } catch { /* gone */ } }, 300);
    }
    this.src = null;
    if (R.onDeck) R.onDeck.delete(tr.id);
    R.release(tr.id);
  }
  stop() { this.endTrack(); }
  feed() {}
  paint() {}
  light() {}
}
const decks = Array.from({ length: DECKS }, (_, d) => new Deck(d));
function front() {                                              // the primary deck, or the loudest deck that plays
  const playing = decks.filter((c) => c.playing && c.now);
  return playing.find((c) => c.role === 'primary') || playing.sort((x, y) => y.p.fader - x.p.fader)[0] || null;
}

// ---------- the views ----------
const viz = new BrainViz($('viz'), { captions: false });
const inst = $('inst'), ig = inst.getContext('2d');
const VIEW_BRAIN = 'compact', VJ_SCENES = ['compact', 'connectome', 'storm'];
let view = store.get('view', 'brain'), vizTrack = null;
const vizHistory = [];
function frontBar(d) {                                          // a deck began a bar: the views follow the deck in front
  const f = front();
  if (!f || f !== d || !d.now) return;
  if (vizTrack !== d.track) { vizTrack = d.track; vizHistory.length = 0; viz.setLayout(d.track.layout || null); }
  const nb = d.now;
  vizHistory.push({ bar: nb.bar, tokens: (nb.tokens || [1, 1, 1, 1]).slice(), mods: nb.mods }); if (vizHistory.length > 96) vizHistory.shift();
  viz.setBar({ ...nb, mode: 'brain', checkpoint: d.track.checkpoint }, vizHistory);
}
function stepViews(s) {
  const f = front(), p = f && f.now && f.now.patterns && f.now.patterns[0];
  viz.step(s, p && p.hits && p.hits.kick ? p.hits.kick[s] > 0 : s % 4 === 0);
}
function vjSection(G) {                       // VJ: a new brain scene on the Auto DJ's section lines
  if (!store.get('vj', true) || calm() || view !== 'brain' || !G) return;
  const i = VJ_SCENES.indexOf(viz.modeName);
  viz.setMode(VJ_SCENES[(i + 1) % VJ_SCENES.length]);
}
// the instruments: the bar that sounds on the deck in front, step by step
const ROWS = [['kick', ['kick']], ['snare', ['snare', 'clap']], ['hats', ['ch', 'oh', 'cym']]];
const LINES = [['bass', 1], ['lead', 2], ['air', 3]];
function drawInstruments() {
  const D = devicePixelRatio || 1, r = inst.getBoundingClientRect();
  if (inst.width !== Math.round(r.width * D) || inst.height !== Math.round(r.height * D)) { inst.width = Math.round(r.width * D); inst.height = Math.round(r.height * D); }
  const W = inst.width, H = inst.height, g = ig;
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  const f = front(), b = f && f.now, s = clock.step;
  const left = Math.round(W * 0.12), right = Math.round(W * 0.94), top = Math.round(H * 0.16), cw = (right - left) / 16;
  g.font = `${Math.round(12 * D)}px ui-monospace, Menlo, monospace`; g.textBaseline = 'middle';
  if (!b || !b.patterns) {
    g.fillStyle = 'rgba(74,241,242,.62)'; g.textAlign = 'center';
    g.fillText(f ? 'this track has no note file: the instruments show the notes of tracks that carry one' : 'the instruments light up when a track plays', W / 2, H / 2);
    return;
  }
  const pats = b.patterns || [], tok = b.tokens || [], mods = b.mods || [];
  const rowH = Math.round((H * 0.66) / 7), glow = (on, now) => (now ? '#e8ffff' : on ? '#4af1f2' : 'rgba(74,241,242,.10)');
  let y = top;
  g.textAlign = 'right';
  const drumHits = (pats[0] && pats[0].hits) || {};
  for (const [name, keys] of ROWS) {
    const row = keys.map((k) => drumHits[k]).find((a) => Array.isArray(a)) || [];
    g.fillStyle = 'rgba(201,251,251,.85)'; g.fillText(name, left - 12 * D, y + rowH / 2);
    for (let i = 0; i < 16; i++) {
      const v = +row[i] || 0, x = left + i * cw, now = i === s && v > 0;
      g.fillStyle = glow(v > 0, now); g.globalAlpha = v > 0 ? 0.4 + 0.6 * Math.min(1, v) : 1;
      if (now) { g.shadowColor = '#4af1f2'; g.shadowBlur = 24 * D; }
      g.fillRect(x + 2 * D, y + 3 * D, cw - 4 * D, rowH - 6 * D);
      g.shadowBlur = 0; g.globalAlpha = 1;
    }
    y += rowH;
  }
  y += rowH * 0.4;
  for (const [name, l] of LINES) {
    const p = pats[l], notes = (p && p.notes) || [], lens = (p && p.len) || [], on = tok[l] > 0;
    const m = mods[l] || [0, 0];
    g.fillStyle = on ? 'rgba(201,251,251,.85)' : 'rgba(201,251,251,.3)'; g.fillText(name, left - 12 * D, y + rowH / 2);
    g.fillStyle = 'rgba(74,241,242,.5)'; g.textAlign = 'left';
    g.fillText(on ? `sound ${tok[l]} · filter ${fmt(m[0])} · space ${fmt(m[1])}` : 'silent', right + 8 * D - (right - left) - 0, y - rowH * 0.18);
    g.textAlign = 'right';
    const pitched = notes.filter((n) => n != null), lo = pitched.length ? Math.min(...pitched) : 0, hi = pitched.length ? Math.max(...pitched) : 1;
    for (let i = 0; i < 16; i++) {
      const x = left + i * cw;
      g.fillStyle = 'rgba(74,241,242,.07)'; g.fillRect(x + 2 * D, y + 3 * D, cw - 4 * D, rowH - 6 * D);
      const n = notes[i];
      if (on && n != null) {
        const span = Math.max(1, (+lens[i] || 1)), hgt = (rowH - 10 * D) * (hi > lo ? (n - lo) / (hi - lo) : 0.5);
        const now = s >= i && s < i + span;
        if (now) { g.shadowColor = '#4af1f2'; g.shadowBlur = 24 * D; }
        g.fillStyle = now ? '#e8ffff' : '#4af1f2';
        g.fillRect(x + 2 * D, y + rowH - 6 * D - hgt - 4 * D, cw * span - 4 * D, 5 * D);
        g.shadowBlur = 0;
      }
    }
    y += rowH;
  }
  // the step that sounds
  if (s >= 0) { g.strokeStyle = 'rgba(232,255,255,.55)'; g.lineWidth = D; g.strokeRect(left + s * cw + 0.5, top - 6 * D, cw - 1, y - top + 6 * D); }
  g.textAlign = 'left'; g.fillStyle = 'rgba(74,241,242,.62)';
  const blend = decks.filter((c) => c.playing && c.now && c !== f && c.p.fader > 0.05);
  g.fillText(`deck ${f.d + 1} · bar ${f.barIndex + 1} of ${f.track.bars.length}${blend.length ? ` · deck ${blend[0].d + 1} blends in (${Math.round(blend[0].p.fader * 100)}%)` : ''}`, left, top - 22 * D);
}
const fmt = (v) => `${Math.round(((+v || 0) + 1) * 50)}%`;

// the visual mode control: the brain (3D) first, then the instruments, then the other modes
const modeSel = $('modeSel');
{
  const og = (label, items) => { const o = document.createElement('optgroup'); o.label = label; for (const [v, t] of items) o.appendChild(new Option(t, v)); modeSel.appendChild(o); };
  og('brain', [['brain', 'brain (3D)'], ...['connectome', 'storm', 'mri'].map((m) => [`m:${m}`, `brain: ${m}`])]);
  og('instruments', [['instruments', 'instruments']]);
  og('the brain, 2D', MODE_GROUPS.brain.filter((m) => !['connectome', 'storm', 'compact', 'mri'].includes(m)).map((m) => [`m:${m}`, m]));
  og('demo scene', MODE_GROUPS.demo.map((m) => [`m:${m}`, m]));
}
function setView(v) {
  view = v; store.set('view', v);
  if (v === 'brain') viz.setMode(VIEW_BRAIN);
  else if (v.startsWith('m:')) viz.setMode(v.slice(2));
  inst.hidden = v !== 'instruments';
  modeSel.value = v;
}
setView(view);
modeSel.addEventListener('change', () => { setView(modeSel.value); modeSel.blur(); });

function frame() {
  requestAnimationFrame(frame);
  if (document.hidden) return;
  if (view === 'instruments') { drawInstruments(); if (viz.gl3) viz.gl3.hide(); return; }
  const ms = 15000 / bpm, playing = decks.some((d) => d.playing);
  viz.frame(playing && clock.step >= 0 ? Math.min(0.999, (clock.step + Math.min(1, (performance.now() - clock.stepAt) / ms)) / 16) : 0, ms);
}
requestAnimationFrame(frame);

// ---------- the Auto DJ ----------
function say(text) { note = text; paintReadout(); setTimeout(() => { if (note === text) { note = ''; paintReadout(); } }, 12000); }
dj = new AutoDJ({ chans: decks, ctx: () => ctx, ensureAudio, bpm: () => bpm, userBpm: () => null,
  setBpm: (v) => { bpm = +v || bpm; for (const d of decks) if (d.fx) d.fx.setTempo(bpm); if (dj) dj.retime(); },
  status: say, paint: () => paintReadout(), readout: () => {}, brainName: () => 'radio' },
// the radio's blend: the second track starts at bar 32, a long blend takes 64 bars with the bass swap in its middle,
// a track lives 128 to 160 bars, a primary plays alone for one section at most (autodj.js, Planner)
{ section: clamp(+q.get('section') || 32, 4, 32), blend: { swaps: [64], swapAt: 32, solo: [1, 1, 2] } });
dj.debug = q.has('djdebug');
window.__radioApp = { dj, decks, viz, clock, ctx: () => ctx, starts: [], setView };

// ---------- the controls ----------
const PLAIN = { slate_inst_reads_s2_echo2: 'deeper QUAD echo', slate_inst_reads_s2_echor: 'QUAD echo, returns only',
  slate_split_s2_echor: 'DUO echo, returns only', slate_inst_reads_s2_echo: 'QUAD echo', slate_inst_reads_s2_echo2q: 'deeper QUAD echo, second run' };
function plainBrain(ck) {
  const run = String(ck || '').replace(/_(e\d+(b\d+)?|l\d{8}|latest)$/, '');
  return PLAIN[run] || run.replace(/^slate_/, '').replace(/_/g, ' ') || 'a Cadance brain';
}
let playingNow = false;
// what the Auto DJ does, in one short line: a track coming in, one leaving, or the primary alone until the next
function djLine() {
  const P = dj.planner, c = dj.clock;
  if (!dj.on || !P || !c) return '';
  const G = c.G, no = (i) => (decks[i] && decks[i].track ? `track ${decks[i].track.no}` : 'the next track');
  const kind = (t) => t.name.split(/[,(]/)[0].trim();
  const coming = P.active.find((t) => G < t.swap), leaving = P.active.find((t) => G >= t.swap && G < t.end);
  if (coming) return `mixing: ${no(coming.in)} comes in (${kind(coming)}), the bass changes decks in ${coming.swap - G} bars`;
  if (leaving) return `mixing: ${no(leaving.out)} leaves in ${leaving.end - G} bars`;
  if (P.nextStart != null && P.nextStart > G) return `${no(P.primary)} alone, the next track comes in ${P.nextStart - G} bars`;
  return dj.say || '';
}
function paintReadout() {
  const st = $('status');
  st.textContent = R.note + (R.error ? ` (${R.error})` : '');
  st.toggleAttribute('data-live', !!R.live);
  const f = front();
  let what = '';
  if (f && f.track) what = `${plainBrain(f.track.checkpoint)} · track ${f.track.no}`;
  else if (playingNow && !R.tracks) what = 'no tracks yet';
  else if (playingNow) what = 'the first track is loading';
  if (note) what = what ? `${what} · ${note}` : note;
  $('nowPlaying').textContent = what;
  $('djLine').textContent = playingNow ? djLine() : '';
  $('playB').innerHTML = playingNow && ctx && ctx.state === 'running' ? '&#10074;&#10074;' : '&#9654;';
  $('playB').setAttribute('aria-label', playingNow ? 'pause' : 'play');
}
setInterval(paintReadout, 1000);
async function startRadio() {
  if (!R.tracks) { await R.manifest(true); }
  if (!R.tracks) { paintBig(); return; }
  $('bigPlay').hidden = true;
  await ensureAudio();
  if (ctx.state !== 'running') await ctx.resume();
  playingNow = true;
  if (!dj.on) dj.start('approved');
  paintReadout();
}
async function togglePlay() {
  if (!playingNow) return startRadio();
  if (ctx.state === 'running') await ctx.suspend(); else await ctx.resume();
  paintReadout();
}
async function nextTrack() {
  if (!playingNow) return startRadio();
  dj.stop(); for (const d of decks) d.endTrack();
  dj.start('approved');
}
function paintBig() {
  const b = $('bigPlayB');
  const none = R.tracks === 0 && R.listed;
  b.disabled = none;
  b.querySelector('.lab').textContent = none ? 'no tracks yet' : 'play the radio';
}
$('bigPlayB').onclick = startRadio;
$('playB').onclick = togglePlay;
$('nextB').onclick = nextTrack;
const volume = $('volume');
volume.value = store.get('volume', 0.8);
volume.oninput = () => { store.set('volume', +volume.value); if (vol) vol.gain.setTargetAtTime(+volume.value, ctx.currentTime, 0.02); };
const full = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.().catch(() => {}); };
$('fullB').onclick = full;
document.addEventListener('fullscreenchange', () => setTimeout(() => viz.resize(), 60));
addEventListener('keydown', (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'f') full();
  else if (e.key === 'm') { const opts = [...modeSel.options].map((o) => o.value); setView(opts[(opts.indexOf(view) + 1) % opts.length]); }
  else if (e.key === 'n') nextTrack();
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'Escape') { closePanels(); }
});
// the store: an empty list waits; the list is read again every minute until it has tracks
(async () => {
  await R.manifest(); R.listed = true; paintBig(); paintReadout();
  const again = setInterval(async () => { if (R.tracks) return clearInterval(again); await R.manifest(true); paintBig(); paintReadout(); }, 60000);
})();

// ---------- the settings (gear) ----------
const settings = $('settings');
function openSettings() {
  closePanels();
  $('apiUrl').value = R.api || '';
  $('apiState').textContent = R.live ? `Live brain found at ${R.api}. Live playback arrives with the next update.` : R.detail || '';
  settings.hidden = false;
  if (!settings.querySelector('.toggles')) {
    const t = document.createElement('div'); t.className = 'toggles';
    t.innerHTML = '<label><input id="calmT" type="checkbox"> calm: no scene changes, no scanline</label><label><input id="vjT" type="checkbox"> VJ: a new brain scene on every section</label>';
    settings.insertBefore(t, settings.querySelector('.row'));
    $('calmT').checked = !!store.get('calm', false); $('vjT').checked = !!store.get('vj', true);
    $('calmT').onchange = () => { store.set('calm', $('calmT').checked); paintCalm(); };
    $('vjT').onchange = () => store.set('vj', $('vjT').checked);
  }
}
function paintCalm() { document.querySelector('.scan').hidden = calm(); }
paintCalm();
$('gearB').onclick = () => (settings.hidden ? openSettings() : closePanels());
$('apiClose').onclick = closePanels;
$('apiSave').onclick = () => {
  const v = $('apiUrl').value.trim().replace(/\/+$/, '');
  try { if (v) localStorage.setItem('cadance.api', v); else localStorage.removeItem('cadance.api'); } catch { /* kept for this page only */ }
  location.reload();
};
$('apiUrl').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('apiSave').click(); e.stopPropagation(); });

// ---------- the tutorial and "what am I looking at" ----------
const tour = $('tour');
const TOUR = [
  { mark: ['viz'], html: `<p>Cadance Radio plays tracks composed by Cadance brains. The Auto DJ mixes them live in your browser, and the picture shows the brain that wrote the bar you hear.</p><p><a href="https://github.com/muellerberndt/cadence" target="_blank" rel="noopener">${TAGLINE}</a></p>` },
  { mark: ['gearB'], html: '<p>Composing live needs a server that hosts the Cadance neural network. <a href="https://github.com/muellerberndt/cadance-minimal/blob/main/docs/running.md#a-live-composer-for-the-radio-site" target="_blank" rel="noopener">How to run one</a></p><p>The gear is where you set its address.</p>' },
  { mark: ['bigPlayB', 'gearB'], html: '<p>Two options:</p><div class="row"><button id="tourPlay" class="go">Play the radio now</button><button id="tourGear">Configure a neural network server first</button></div>' },
];
const WHAT = '<p>What you are looking at:</p><p>its real wiring: the regions of the brain and the synapses between them;</p><p>it settles until it comes to rest, there is no forward pass;</p><p>a memory it reads, bar by bar;</p><p>one stream that never resets;</p><p>not a feed-forward network, not a transformer.</p>';
let tourAt = 0, rings = [];
function mark(ids) {
  for (const r of rings) r.remove();
  rings = ids.map((id) => {
    const e = $(id); if (!e || !e.getClientRects().length) return null;
    const b = e.getBoundingClientRect(), r = document.createElement('div');
    r.className = 'ring2'; r.dataset.for = id;
    Object.assign(r.style, { left: `${b.left - 6}px`, top: `${b.top - 6}px`, width: `${b.width + 12}px`, height: `${b.height + 12}px` });
    document.body.appendChild(r); return r;
  }).filter(Boolean);
}
function showTour(i) {
  closePanels();
  tourAt = clamp(i, 0, TOUR.length - 1);
  if (tourAt === 2) { $('bigPlay').hidden = playingNow; }
  tour.querySelector('.step').textContent = `${tourAt + 1} of ${TOUR.length}`;
  tour.querySelector('.body').innerHTML = TOUR[tourAt].html;
  $('tourBack').hidden = tourAt === 0; $('tourNext').hidden = tourAt === TOUR.length - 1;
  tour.hidden = false; tour.dataset.step = String(tourAt + 1);
  mark(TOUR[tourAt].mark.filter((id) => id !== 'viz'));
  if ($('tourPlay')) $('tourPlay').onclick = () => { seen(); closePanels(); startRadio(); };
  if ($('tourGear')) $('tourGear').onclick = () => { seen(); openSettings(); };
}
function showWhat() {
  closePanels();
  tour.querySelector('.step').textContent = 'the brain';
  tour.querySelector('.body').innerHTML = WHAT + '<div class="row"><button id="whatTour">the tutorial</button></div>';
  $('tourBack').hidden = true; $('tourNext').hidden = true; tour.hidden = false; tour.dataset.step = 'what';
  $('whatTour').onclick = () => showTour(0);
}
const seen = () => store.set('tour', true);
function closePanels() { tour.hidden = true; settings.hidden = true; for (const r of rings) r.remove(); rings = []; }
$('tourNext').onclick = () => showTour(tourAt + 1);
$('tourBack').onclick = () => showTour(tourAt - 1);
$('tourClose').onclick = () => { seen(); closePanels(); };
$('helpB').onclick = () => (tour.hidden ? showWhat() : closePanels());
addEventListener('resize', () => { if (!tour.hidden && tour.dataset.step && tour.dataset.step !== 'what') mark(TOUR[tourAt].mark.filter((id) => id !== 'viz')); });
if (!store.get('tour', false)) showTour(0);
paintReadout();
