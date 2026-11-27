// Cadance Radio (radio.html): one screen of visuals; the Auto DJ (autodj.js) mixes the track store's MP3s on three
// file decks, each through a channel strip (EQ, filters, the DJ effects, a fader) as on the mixer, all on one bar
// clock of the AudioContext: each bar line comes 240 / bpm after the one before, and a deck's MP3 starts on a bar line
// (AudioBufferSourceNode.start(when, offset)), so two decks run in time to the sample. The views follow the deck in
// front: the brain stage (brainstage.js, from the track's note file: the bars as the brain wrote them, with its
// recorded view), the devices (devices.js), brainviz's MRI and other modes, and the Auto DJ's decks (mixerstage.js). The page writes
// nothing of what plays (owner, 27 Nov): no brain names, no track numbers, no DJ text; the controls are icons, and the
// state (radio or live) is in the settings. For tests, window.__radioApp.state() reads it all.
import { AutoDJ } from './autodj.js';
import { BrainStage } from './brainstage.js';
import { BrainViz, MODE_GROUPS } from './brainviz.js';
import { Devices } from './devices.js';
import { FxUnit } from './djfx.js';
import { MixerStage } from './mixerstage.js';

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
const PRE = 0.63;                     // a deck's input gain, as on the mixer: two decks at full level stay out of the limiter
const TAGLINE = 'Build real-time brains that work like human brains.';
const LAW = {        // knob position 0..1 -> value, as on the mixer
  eq: { val: (p) => (p <= 0.015 ? -40 : p < 0.5 ? (-26 * (0.5 - p)) / 0.485 : (6 * (p - 0.5)) / 0.5) },
  hp: { val: (p) => (p <= 0.005 ? 0 : 20 * Math.pow(125, p)) },
  lp: { val: (p) => (p >= 0.995 ? 0 : 180 * Math.pow(20000 / 180, p)) },
  fader: { val: (p) => Math.pow(p / 0.8, 2.5) },
};
const calm = () => !!store.get('calm', false) || matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- audio and the bar clock ----------
let ctx = null, master = null, vol = null, bpm = R.defaultBpm || 145, dj = null, starting = null, trackNo = 0;
const notes = [];                    // what went wrong, for the console and state(): a failed track is retried silently
const latency = () => (ctx ? (ctx.outputLatency || 0) + (ctx.baseLatency || 0) : 0);
const later = (t, fn) => { const ms = Math.max(0, (t + latency() - (ctx ? ctx.currentTime : 0)) * 1000); ms > 4 ? setTimeout(fn, ms) : fn(); };
const clock = { t0: 0, next: 0, bar: 0, step: -1, stepAt: 0, timer: null };
const barLen = () => 240 / bpm;
function ensureAudio() {
  if (ctx) return Promise.resolve();
  if (starting) return starting;
  starting = (async () => {
    // on an iPhone, sound in spite of the silent switch, as a music app (Safari's audio session; elsewhere absent)
    try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch { /* older Safari */ }
    const c = new AudioContext({ sampleRate: 44100, latencyHint: 'playback' });
    vol = c.createGain(); vol.gain.value = store.get('volume', 0.8);
    master = c.createGain();
    const lim = c.createDynamicsCompressor();
    lim.threshold.value = -1.5; lim.knee.value = 0; lim.ratio.value = 20; lim.attack.value = 0.002; lim.release.value = 0.12;
    master.connect(lim); lim.connect(vol); vol.connect(c.destination);
    ctx = c;
    viz.listen(c, master);
    const an = c.createAnalyser(); an.fftSize = 2048; master.connect(an); hero.setAudio(an);
    Object.assign(window.__radioApp, { master, analyser: an });          // for a view of the mix (the mixer stage)
    mixAudio();
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
    for (const d of decks) later(t, () => d.onBar(G, t));
    if (mixer) mixer.setClock({ ctx, bpm, barZeroTime: t - G * barLen() });
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
    this.n.pre.gain.value = PRE;
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
  setRole(role) { if (role === 'primary' && this.role !== 'primary') pulse('swap', this.d); this.role = role; }
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
  gate(t0, len, depth) { if (this.fx) this.fx.scheduleGate(t0, len, depth); if (depth > 0 && this.p.fxType === 'gater') pulse('gate', this.d); }
  // ---- a track: its MP3 starts on the next bar line (later lines while it decodes: the track then starts a few bars in)
  cueTrack(track) {
    this.endTrack();
    this.track = track; this.trackAt = 0; this.playing = true; this.now = null;
    this.startBar = clock.bar; const line = clock.next;          // the next bar line is bar clock.bar
    trackNo++; track.no = trackNo;
    (R.onDeck ||= new Set()).add(track.id);
    // the file's loudness, trimmed to -16 LUFS at the deck's input (at most 3 dB either way; no measure, no trim)
    const h = R.track(track.id), lufs = h && h.entry ? +h.entry.lufs : NaN;
    this.trim = Number.isFinite(lufs) && h.entry.lufs !== null ? clamp(-16 - lufs, -3, 3) : 0;
    this.n.pre.gain.setValueAtTime(PRE * Math.pow(10, this.trim / 20), ctx.currentTime);
    pulse('cue', this.d);
    if (mixer) { mixer.setTrack(this.d, mixTrack(this)); mixer.setWave(this.d, { bars: track.bars }); }      // the note wave first
    (async () => {
      if (await R.ranged(track.id)) return this.segments(track, line);     // by segments: a store with byte ranges
      this.play(await R.decode(track.id), track, line);                    // the whole track at once
    })().catch((e) => {
      if (this.track !== track) return;
      say(`${e.message}: the next track is tried`);
      this.failed = true;
    });
  }
  // By segments of 16 bars (9 MB of decoded sound at 145 bpm), each read and decoded when less than 20 s of the deck
  // is scheduled, so a deck holds two at most. Segment k covers the samples [S(a), S(b)) of the track (S(bar) =
  // round(bar * 240 / bpm * 44100)) and starts at line + S(a) / 44100: the segments abut to the sample and every bar line
  // falls where the clock has it. A segment that is ready late starts at once with the offset that keeps it on the
  // grid. Past the track's end its last 32 bars go round, as its two last segments read again.
  async segments(track, line) {
    const sr = 44100, len = barLen(), n = track.bars.length, SEG = 16, AHEAD = 20;
    const S = (b) => Math.round(b * len * sr), g = ctx.createGain();
    g.connect(this.n.pre);
    const run = { track, g, list: [], end: 0, busy: false, b: 0, at: line };
    this.run = run; this.src = { g, run }; this.mode = 'segments';
    const next = async () => {
      // the bars [a, e) of the track: in order to its end, then its last 32 bars round and round
      const a = run.b < n ? run.b : Math.max(0, n - 32) + ((run.b - n) % Math.min(32, n)), e = Math.min(n, a + SEG);
      const buf = await R.pcm(track.id, S(a), S(e));
      if (this.run !== run || !ctx) return;
      const at = run.at, t = Math.max(at, ctx.currentTime + 0.03);
      const s = ctx.createBufferSource();
      s.buffer = buf; s.from = a; s.connect(g); s.start(t, t - at);
      run.list.push(s); s.onended = () => { run.list = run.list.filter((x) => x !== s); };
      if (mixer && run.b < n) mixer.addWave(this.d, { fromBar: a, audio: buf });              // the sound's wave, as it comes
      run.at = at + buf.length / sr; run.end = run.at; run.b += e - a;
      if (window.__radioApp) (window.__radioApp.segments ||= []).push({ deck: this.d, id: track.id, from: a, to: e, at, t, offset: t - at, samples: buf.length });
    };
    await next();
    if (window.__radioApp) (window.__radioApp.starts ||= []).push({ deck: this.d, id: track.id, when: line, line, bar: this.startBar, segments: true });
    run.timer = setInterval(async () => {
      if (this.run !== run) { clearInterval(run.timer); return; }
      if (run.busy || !ctx || run.end - ctx.currentTime > AHEAD) return;
      run.busy = true;
      try { await next(); } catch (e) { say(`${e.message}: the next track is tried`); }
      run.busy = false;
    }, 1000);
  }
  play(buf, track, line) {                         // the whole track as one buffer (a store without byte ranges)
    if (this.track !== track || !ctx) return;
    const len = barLen(), soon = ctx.currentTime + 0.06;
    let k = 0, when = line;
    if (when < soon) { k = Math.ceil((soon - when) / len); when += k * len; }       // decoded late: in on a later line
    const s = ctx.createBufferSource(), g = ctx.createGain();
    s.buffer = buf; this.buffer = buf;
    const n = track.bars.length;
    if (n >= 32 && buf.duration >= n * len) { s.loop = true; s.loopStart = (n - 32) * len; s.loopEnd = n * len; }  // past the end its last section goes round
    s.connect(g); g.connect(this.n.pre);
    s.start(when, k * len);
    this.src = { s, g }; this.mode = 'whole';
    if (mixer) mixer.setWave(this.d, buf);
    if (window.__radioApp) (window.__radioApp.starts ||= []).push({ deck: this.d, id: track.id, when, line, offset: k * len, bar: this.startBar + k });
  }
  onBar(G, t) {
    if (!this.playing || !this.track) return;
    const i = G - this.startBar;
    if (i < 0) return;
    const n = this.track.bars.length, k = barOf(n, i);
    this.trackAt = i + 1; this.lineAt = t; this.now = this.track.bars[k] || null; this.barIndex = k;
    frontBar(this);
  }
  endTrack() {
    const tr = this.track;
    if (!tr) return;
    this.track = null; this.playing = false; this.now = null; this.failed = false; this.buffer = null;
    pulse('leave', this.d);
    if (mixer) { mixer.setTrack(this.d, null); mixer.setWave(this.d, null); }
    if (this.src && ctx) {                        // the sounding sources fade out in 10 ms and stop
      const { s, g, run } = this.src, t = ctx.currentTime, all = run ? run.list : [s];
      g.gain.setTargetAtTime(0, t, 0.01);
      for (const x of all) { try { x.stop(t + 0.08); } catch { /* not started */ } }
      setTimeout(() => { try { for (const x of all) x.disconnect(); g.disconnect(); } catch { /* gone */ } }, 300);
      if (run) clearInterval(run.timer);
    }
    this.src = null; this.run = null;
    if (R.onDeck) R.onDeck.delete(tr.id);
    R.release(tr.id);
  }
  stop() { this.endTrack(); }
  feed() {}
  paint() {}
  light() {}
}
const decks = Array.from({ length: DECKS }, (_, d) => new Deck(d));
// the mixer stage (mixerstage.js), built when the DJ view first shows (its shaders take about 130 ms): the decks'
// tracks and waves, their meters, the master, the clock and a reader of the planner's state; without WebGL the DJ
// view is the plain canvas below
let mixer = null, mixerTried = false;
const mixTrack = (d) => ({ id: d.track.id, bars: d.track.bars.length, startBar: d.startBar, loopFrom: Math.max(0, d.track.bars.length - 32) });
function mixerStage() {
  if (mixerTried) return mixer;
  mixerTried = true;
  const m = new MixerStage($('mixv'), { decks: DECKS, calm: calm() });
  if (!m.ok) { say(`mixer stage: ${m.error || 'no WebGL'}`); return null; }
  mixer = m;
  mixer.setReader(() => {
    const P = dj.planner;
    return { bar: dj.clock && ctx ? Math.floor(dj.posOf(ctx.currentTime)) : 0,
      decks: decks.map((d) => ({ playing: !!d.track && d.playing, role: d.role || '', fxType: d.p.fxType,
        knobs: { fader: d.p.fader, lo: d.p.lo, mid: d.p.mid, hi: d.p.hi, hp: d.p.hp, lp: d.p.lp, fx: d.p.fx, cut: d.p.cut ?? 1 } })),
      dj: { on: dj.on, primary: P ? P.primary : -1, section: dj.section, nextStart: P ? P.nextStart : null,
        transitions: P ? P.active.map((t) => ({ in: t.in, out: t.out, start: t.start, swap: t.swap, end: t.end })) : [] } };
  });
  pulseTo.push((kind, deck) => mixer.pulse(kind, deck));
  mixAudio();
  for (const d of decks) {                        // the decks' tracks, and the sound of every segment they hold now
    if (!d.track) continue;
    mixer.setTrack(d.d, mixTrack(d)); mixer.setWave(d.d, d.buffer || { bars: d.track.bars });
    if (d.run) for (const s of d.run.list) if (s.buffer && s.from < d.track.bars.length) mixer.addWave(d.d, { fromBar: s.from, audio: s.buffer });
  }
  return mixer;
}
function mixAudio() {                             // once the sound is up: the master and the decks' meters, the clock
  if (!mixer || !ctx || !window.__radioApp.analyser) return;
  mixer.setAudio(window.__radioApp.analyser);
  for (const d of decks) if (d.n) mixer.setMeter(d.d, d.n.meter);
  if (dj.clock) mixer.setClock({ ctx, bpm, barZeroTime: dj.clock.t - dj.clock.G * barLen() });
}
// the Auto DJ's moments, for a view that marks them: 'cue' (a track goes onto a deck), 'swap' (a deck becomes the
// primary), 'leave' (a deck's track ends), 'gate' (the gater opens on a deck this bar)
const pulseTo = [];
function pulse(kind, deck) { for (const fn of pulseTo) { try { fn(kind, deck); } catch (e) { console.warn(e); } } }
function front() {                                              // the primary deck, or the loudest deck that plays
  const playing = decks.filter((c) => c.playing && c.now);
  return playing.find((c) => c.role === 'primary') || playing.sort((x, y) => y.p.fader - x.p.fader)[0] || null;
}

// ---------- the views ----------
// brain (the default): the brain stage (brainstage.js), the whole brain as it hears, plans and plays with its memory
// above, given the bars as they sound on the deck in front; with VJ on, its scene changes on the Auto DJ's section
// lines; without a canvas context it is brainviz's 3D brain. devices: the drum machine and the three synth voices of
// the track in front (devices.js), with the sounds of the pool the track names; a deck that blends in as a small strip.
// MRI: brainviz's MRI, lit by the recorded view. DJ: the Auto DJ's three decks, read-only, as the autopilot moves them.
// more: brainviz's other modes.
const viz = new BrainViz($('viz'), { captions: false });
const hero = new BrainStage($('hero'), { calm: calm() });
const HERO = hero.ok;
const devv = new Devices($('devCanvas'));
const djc = $('djv'), dg = djc.getContext('2d');
const MAIN = [['brain', 'brain'], ['devices', 'devices'], ['mri', 'MRI'], ['dj', 'DJ']];
const known = (v) => MAIN.some(([x]) => x === v) || String(v).startsWith('m:');
let view = known(store.get('view', 'brain')) ? store.get('view', 'brain') : 'brain';
let vizTrack = null, heroTrack = null, devTrack = null;
const vizHistory = [];
fetch('brain3d/brain.b3d').then((r) => (r.ok ? r.arrayBuffer() : null)).then((b) => { if (b) hero.setBrain(b); }).catch(() => {});
// the bar a deck plays at its i-th bar: past the track's end its last 32 bars go round
const barOf = (n, i) => (i < n ? i : Math.max(0, n - 32 + ((i - n) % 32)));
function frontBar(d) {                                          // a deck began a bar: the views follow the deck in front
  const f = front();
  if (!f || f !== d || !d.now) return;
  const tr = d.track, nb = d.now, i = d.trackAt - 1, n = tr.bars.length;
  if (vizTrack !== tr) { vizTrack = tr; vizHistory.length = 0; viz.setLayout(tr.layout || null); }
  vizHistory.push({ bar: nb.bar, tokens: (nb.tokens || [1, 1, 1, 1]).slice(), mods: nb.mods }); if (vizHistory.length > 96) vizHistory.shift();
  viz.setBar({ ...nb, mode: 'brain', checkpoint: tr.checkpoint }, vizHistory);
  // the stage counts bars from the deck's first, so a track that goes round stays one stream; a new track in front
  // brings the bars it has played so far (the ear shows 1 to 16 bars back, the memory ring up to 128)
  if (heroTrack !== tr) {
    heroTrack = tr; hero.setLayout(tr.layout || null);
    for (let j = Math.max(0, i - 130); j < i; j++) hero.setBar(j, tr.bars[barOf(n, j)]);
  }
  hero.setClock({ ctx, bpm, barZeroTime: d.lineAt - i * barLen() });
  hero.setBar(i, nb);
  devBar(d, tr, nb, i > 0 ? tr.bars[barOf(n, i - 1)] : nb);
}
let devNote = '';
async function devBar(d, tr, nb, prev) {
  devv.setBar(nb, prev);
  if (devTrack === tr) return;
  devTrack = tr;
  const h = R.track(tr.id), pool = await R.pool(h && h.notes && h.notes.pool);
  if (devTrack !== tr) return;
  devv.setPool(pool);
  devNote = pool ? '' : `no sound pool for ${tr.id} in the store`;
  if (!pool) say(devNote);
}
function secondDeck() {                                         // a deck beside the one in front that is heard
  const f = front(), o = decks.find((c) => c !== f && c.playing && c.now && c.p.fader > 0.05);
  return o ? { bar: o.now, level: o.p.fader } : null;
}
function stepViews(s) {
  const f = front(), p = f && f.now && f.now.patterns && f.now.patterns[0];
  viz.step(s, p && p.hits && p.hits.kick ? p.hits.kick[s] > 0 : s % 4 === 0);
  devv.setStep(s);
}
const VJ_SCENES = ['compact', 'connectome', 'storm'];         // brainviz's, when the stage cannot draw
function vjSection(G) {                       // VJ: a new brain scene on the Auto DJ's section lines
  if (!store.get('vj', true) || calm() || view !== 'brain' || !G) return;
  if (HERO) { const s = hero.scenes(); hero.setScene(s[(s.indexOf(hero.scene) + 1) % s.length]); return; }
  const i = VJ_SCENES.indexOf(viz.modeName);
  viz.setMode(VJ_SCENES[(i + 1) % VJ_SCENES.length]);
}

// the DJ: three channel strips without words (EQ, filters, the effect, the fader and its level; the primary's strip
// lit, a cut in red) and the plan ahead: each deck's fader over the next bars, with a mark where it holds the bass
const DECK_COL = ['#4af1f2', '#ff2a6d', '#2a6dff'];
const levels = decks.map(() => ({ buf: new Float32Array(512), v: 0 }));
function drawDJ() {
  const D = Math.min(2, devicePixelRatio || 1), rc = djc.getBoundingClientRect();
  if (djc.width !== Math.round(rc.width * D) || djc.height !== Math.round(rc.height * D)) { djc.width = Math.round(rc.width * D); djc.height = Math.round(rc.height * D); }
  const W = djc.width, H = djc.height, g = dg, P = dj.planner;
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  const head = (document.querySelector('.brand').getBoundingClientRect().bottom + 16) * D;      // below the wordmark and the claim
  const m = 16 * D, avail = H - $('bar').offsetHeight * D - 24 * D - head, planH = Math.min(120 * D, avail * 0.3);
  const gap = 12 * D, sw = Math.min(250 * D, (W - 2 * m - 2 * gap) / 3), x0 = (W - 3 * sw - 2 * gap) / 2;
  const sh = Math.min((W / D < 640 ? 240 : 300) * D, avail - planH - 36 * D), u = Math.min(1, sw / (230 * D));
  const top = head + Math.max(0, (avail - sh - 36 * D - planH) / 2), bottom = top + sh + 36 * D + planH;
  g.textBaseline = 'middle';
  decks.forEach((d, k) => {
    const x = x0 + k * (sw + gap), y = top, col = DECK_COL[k], p = d.p, on = !!d.track;
    g.globalAlpha = on ? 1 : 0.35;
    g.strokeStyle = on ? col : 'rgba(74,241,242,.28)'; g.lineWidth = d.role === 'primary' ? 2 * D : D; g.strokeRect(x + 0.5, y + 0.5, sw - 1, sh - 1);
    if (d.role === 'primary') { g.shadowColor = col; g.shadowBlur = 18 * D; g.strokeRect(x + 0.5, y + 0.5, sw - 1, sh - 1); g.shadowBlur = 0; }
    g.fillStyle = col; g.fillRect(x, y, sw * (d.role === 'in' ? 0.5 : d.role === 'out' ? 0.25 : on ? 1 : 0), 3 * D);
    // the knobs: EQ in the first row, the filters and the effect in the second; the fader at the right
    const kr = 15 * D * u, colW = (sw - 54 * D) / 3, row1 = y + 48 * D * Math.max(0.8, u), row2 = row1 + 56 * D * Math.max(0.8, u);
    const knobs = [[[p.hi, true], [p.mid, true], [p.lo, true]], [[p.hp, false], [p.lp, false], [p.fx, false]]];
    knobs.forEach((rowK, r) => rowK.forEach(([v, bip], c) => knob(g, x + 10 * D + colW * (c + 0.5), r ? row2 : row1, kr, +v || 0, col, bip, D)));
    if ((p.cut ?? 1) < 0.5) { g.fillStyle = '#ff2a6d'; g.fillRect(x + 10 * D, row2 + 34 * D * u, sw - 64 * D, 3 * D); }
    // the fader and the level after it
    const fx = x + sw - 30 * D, fy0 = y + 52 * D, fy1 = y + sh - 16 * D, fv = clamp(+p.fader || 0, 0, 1);
    g.fillStyle = 'rgba(74,241,242,.12)'; g.fillRect(fx - 2 * D, fy0, 4 * D, fy1 - fy0);
    const cy = fy1 - fv * (fy1 - fy0);
    g.fillStyle = col; g.shadowColor = col; g.shadowBlur = on ? 10 * D : 0; g.fillRect(fx - 9 * D, cy - 3 * D, 18 * D, 6 * D); g.shadowBlur = 0;
    const L = levels[k];
    if (d.n && d.n.meter && on) {
      d.n.meter.getFloatTimeDomainData(L.buf);
      let s = 0; for (let i = 0; i < L.buf.length; i++) s += L.buf[i] * L.buf[i];
      const db = 20 * Math.log10(Math.sqrt(s / L.buf.length) + 1e-6), v = clamp((db + 48) / 48, 0, 1);
      L.v = Math.max(v, L.v - 0.04);
    } else L.v = 0;
    g.fillStyle = 'rgba(74,241,242,.08)'; g.fillRect(fx + 14 * D, fy0, 4 * D, fy1 - fy0);
    g.fillStyle = col; g.fillRect(fx + 14 * D, fy1 - L.v * (fy1 - fy0), 4 * D, L.v * (fy1 - fy0));
    g.globalAlpha = 1;
  });
  // the plan: the faders from 8 bars back to 56 ahead, the bass as a thick mark, section lines every 32 bars
  const py = bottom - planH, px0 = x0, px1 = x0 + 3 * sw + 2 * gap;
  g.strokeStyle = 'rgba(74,241,242,.28)'; g.lineWidth = D; g.strokeRect(px0 + 0.5, py + 0.5, px1 - px0 - 1, planH - 1);
  if (!P || !dj.clock || !ctx) return;
  const now = dj.posOf(ctx.currentTime), a = now - 8, b = now + 56, X = (pos) => px0 + ((pos - a) / (b - a)) * (px1 - px0);
  g.fillStyle = 'rgba(74,241,242,.45)';
  const sec = dj.section || 32;
  for (let s = Math.ceil((a - P.origin) / sec) * sec + P.origin; s <= b; s += sec) {
    g.fillRect(X(s), py, D, planH);
  }
  decks.forEach((d, k) => {
    if (P.role[k] === 'idle' && !d.track) return;
    g.strokeStyle = DECK_COL[k]; g.lineWidth = 2 * D; g.beginPath();
    for (let i = 0; i <= 128; i++) {
      const pos = a + ((b - a) * i) / 128, v = P.role[k] === 'idle' ? 0 : P.value(k, 'fader', pos), yy = py + planH - 8 * D - v * (planH - 26 * D);
      if (i) g.lineTo(X(pos), yy); else g.moveTo(X(pos), yy);
    }
    g.stroke();
    g.fillStyle = DECK_COL[k];
    for (let i = 0; i < 128; i++) {
      const pos = a + ((b - a) * (i + 0.5)) / 128;
      if (P.role[k] !== 'idle' && P.value(k, 'lo', pos) >= 0.3 && P.value(k, 'fader', pos) > 0.3) g.fillRect(X(a + ((b - a) * i) / 128), py + 4 * D + k * 5 * D, (px1 - px0) / 128 + 0.5, 3 * D);
    }
  });
  g.fillStyle = '#e8ffff'; g.fillRect(X(now), py - 2 * D, D, planH + 4 * D);
}
function knob(g, x, y, r, v, col, bip, D) {
  const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25, a = a0 + (a1 - a0) * clamp(v, 0, 1), mid = (a0 + a1) / 2;
  g.lineWidth = 3 * D; g.lineCap = 'round';
  g.strokeStyle = 'rgba(74,241,242,.14)'; g.beginPath(); g.arc(x, y, r, a0, a1); g.stroke();
  g.strokeStyle = col; g.beginPath(); if (bip) g.arc(x, y, r, Math.min(mid, a), Math.max(mid, a)); else g.arc(x, y, r, a0, a); g.stroke();
  g.lineWidth = 2 * D; g.strokeStyle = '#e8ffff'; g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(a) * r * 0.72, y + Math.sin(a) * r * 0.72); g.stroke();
  g.lineCap = 'butt';
}

// the view control: icons for brain, devices, MRI, DJ; "more" steps through brainviz's other modes
const MORE = ['compact', 'connectome', 'storm', ...MODE_GROUPS.brain.filter((x) => !['connectome', 'storm', 'compact', 'mri'].includes(x)), ...MODE_GROUPS.demo].map((x) => `m:${x}`);
const ORDER = [...MAIN.map(([v]) => v), ...MORE];
for (const b of document.querySelectorAll('#views [data-view]')) {
  b.onclick = () => { const v = b.dataset.view; setView(v === 'more' ? MORE[(MORE.indexOf(view) + 1) % MORE.length] : v); b.blur(); };
}
function setView(v) {
  if (!known(v)) v = 'brain';
  view = v; store.set('view', v);
  const heroOn = v === 'brain' && HERO, vizOn = !heroOn && (v === 'brain' || v === 'mri' || v.startsWith('m:'));
  if (v === 'brain' && !HERO) viz.setMode('compact');
  else if (v === 'mri') viz.setMode('mri');
  else if (v.startsWith('m:')) viz.setMode(v.slice(2));
  const mixOn = v === 'dj' && !!mixerStage();            // one WebGL view draws at a time
  $('hero').hidden = !heroOn; $('viz').hidden = !vizOn; $('devices').hidden = v !== 'devices'; djc.hidden = v !== 'dj' || mixOn; $('mixv').hidden = !mixOn;
  document.body.dataset.view = v.startsWith('m:') ? 'more' : v;
  if (heroOn) { fitStage('hero', hero); hero.start(); } else hero.stop();
  if (mixOn) { fitStage('mixv', mixer); mixer.start(); } else if (mixer) mixer.stop();
  if (!vizOn && viz.gl3) viz.gl3.hide();
  for (const b of document.querySelectorAll('#views [data-view]')) b.classList.toggle('on', b.dataset.view === (v.startsWith('m:') ? 'more' : v));
}
setView(view);
// a stage ends above the controls, so what it draws at the bottom stays in view; the mixer also starts below the
// wordmark and the claim (its decks' controls stand at the left edge)
function fitStage(id, stage) {
  const top = id === 'mixv' ? Math.round(document.querySelector('.brand').getBoundingClientRect().bottom + 6) : 0;
  Object.assign($(id).style, { top: `${top}px`, height: `calc(100% - ${top + $('bar').offsetHeight}px)` });
  stage.resize();
}
addEventListener('resize', () => { if (!$('hero').hidden) fitStage('hero', hero); if (mixer && !$('mixv').hidden) fitStage('mixv', mixer); });

function frame() {
  requestAnimationFrame(frame);
  if (document.hidden) return;
  const ms = 15000 / bpm, playing = decks.some((d) => d.playing);
  const phase = playing && clock.step >= 0 ? Math.min(0.999, (clock.step + Math.min(1, (performance.now() - clock.stepAt) / ms)) / 16) : 0;
  if (view === 'devices') { devv.setSecond(secondDeck()); devv.draw(phase); }
  else if (view === 'dj') { if (!mixer) drawDJ(); }
  else if (!(view === 'brain' && HERO)) viz.frame(phase, ms);
}
requestAnimationFrame(frame);

// ---------- the Auto DJ ----------
function say(text) { notes.push({ at: new Date().toISOString(), text }); if (notes.length > 50) notes.shift(); console.info(`radio: ${text}`); }
dj = new AutoDJ({ chans: decks, ctx: () => ctx, ensureAudio, bpm: () => bpm, userBpm: () => null,
  setBpm: (v) => { bpm = +v || bpm; for (const d of decks) if (d.fx) d.fx.setTempo(bpm); if (dj) dj.retime(); },
  status: say, paint: () => {}, readout: () => {}, brainName: () => 'radio' },
// the radio's blend: the second track starts at bar 32, a long blend takes 64 bars with the bass swap in its middle,
// a track lives 128 to 160 bars, a primary plays alone for one section at most (autodj.js, Planner)
{ section: clamp(+q.get('section') || 32, 4, 32), blend: { swaps: [64], swapAt: 32, solo: [1, 1, 2] } });
dj.debug = q.has('djdebug');
window.__radioApp = { dj, decks, viz, hero, mixer: () => mixer, devices: devv, clock, ctx: () => ctx, starts: [], setView, notes, state: () => state(), onPulse: (fn) => pulseTo.push(fn) };

// ---------- the controls ----------
let playingNow = false;
// what the Auto DJ does, in one line (state() only): a track coming in, one leaving, or the primary alone until the next
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
// everything the page does not write, for tests and the console
function state() {
  const f = front(), P = dj.planner;
  return { radio: R.note, live: !!R.live, error: R.error || '', tracks: R.tracks, playing: playingNow, running: !!ctx && ctx.state === 'running', bpm, view,
    front: f && f.track ? { deck: f.d, checkpoint: f.track.checkpoint, track: f.track.no, bar: f.trackAt } : null,
    decks: decks.map((d) => (d.track ? { track: d.track.no, checkpoint: d.track.checkpoint, role: d.role, bar: d.trackAt, fader: d.p.fader, mode: d.mode, trim: d.trim,
      held: d.run ? d.run.list.reduce((b, x) => b + (x.buffer ? x.buffer.length * x.buffer.numberOfChannels * 4 : 0), 0) : d.buffer ? d.buffer.length * d.buffer.numberOfChannels * 4 : 0 } : null)),
    dj: { line: djLine(), primary: P ? P.primary : null, nextStart: P ? P.nextStart : null, active: P ? P.active.map((t) => ({ ...t })) : [] },
    devices: devNote, notes: notes.slice() };
}
function paintControls() {
  $('playB').innerHTML = playingNow && ctx && ctx.state === 'running' ? '&#10074;&#10074;' : '&#9654;';
  $('playB').setAttribute('aria-label', playingNow && ctx && ctx.state === 'running' ? 'pause' : 'play');
}
async function startRadio() {
  if (!R.tracks) { await R.manifest(true); }
  if (!R.tracks) { paintBig(); return; }
  $('bigPlay').hidden = true;
  await ensureAudio();
  if (ctx.state !== 'running') await ctx.resume();
  playingNow = true;
  paintControls();
  if (!dj.on) { await dj.start('approved'); soon(); }
}
// The Auto DJ starts a set on the bar line after the one it is armed at. When it is armed (no deck plays yet), the
// clock is set back so that this line lies in the past and the set's first line comes in a quarter of a second.
function soon() {
  if (!ctx || !dj.on) return;
  const t = ctx.currentTime + 0.25 - barLen();
  if (clock.next > t) clock.next = t;
}
async function togglePlay() {
  if (!playingNow) return startRadio();
  if (ctx.state === 'running') await ctx.suspend(); else await ctx.resume();
  paintControls();
}
async function nextTrack() {
  if (!playingNow) return startRadio();
  dj.stop(); for (const d of decks) d.endTrack();
  await dj.start('approved'); soon();
}
function paintBig() {
  const b = $('bigPlayB');
  const none = R.tracks === 0 && R.listed;
  b.disabled = none;
  b.setAttribute('aria-label', none ? 'play (the store has no tracks yet)' : 'play');
}
$('bigPlayB').onclick = startRadio;
$('playB').onclick = togglePlay;
$('nextB').onclick = nextTrack;
const volume = $('volume');
volume.value = store.get('volume', 0.8);
volume.oninput = () => { store.set('volume', +volume.value); if (vol) vol.gain.setTargetAtTime(+volume.value, ctx.currentTime, 0.02); };
const full = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen?.().catch(() => {}); };
$('fullB').onclick = full;
document.addEventListener('fullscreenchange', () => setTimeout(() => { viz.resize(); if (!$('hero').hidden) fitStage('hero', hero); if (mixer && !$('mixv').hidden) fitStage('mixv', mixer); }, 60));
addEventListener('keydown', (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'f') full();
  else if (e.key === 'm') setView(ORDER[(ORDER.indexOf(view) + 1) % ORDER.length]);
  else if (e.key === 'n') nextTrack();
  else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'Escape') { closePanels(); }
});
// the store: an empty list waits; the list is read again every minute until it has tracks
(async () => {
  await R.manifest(); R.listed = true; paintBig();
  setTimeout(() => R.warm(), 1500);                    // the first track read ahead, once the page is up
  const again = setInterval(async () => { if (R.tracks) return clearInterval(again); await R.manifest(true); paintBig(); }, 60000);
})();

// ---------- the settings (gear) ----------
const settings = $('settings');
function openSettings() {
  closePanels();
  $('apiUrl').value = R.api || '';
  $('radioState').textContent = R.note + (R.error ? ` (${R.error})` : '');
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
function paintCalm() {
  document.querySelector('.scan').hidden = calm(); hero.setCalm(calm()); if (calm()) hero.setScene('wired');
  if (mixer) mixer.setCalm(calm());
  $('calmB').setAttribute('aria-pressed', String(calm()));
  if ($('calmT')) $('calmT').checked = !!store.get('calm', false);
}
$('calmB').onclick = () => { store.set('calm', !store.get('calm', false)); paintCalm(); $('calmB').blur(); };
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
  { mark: ['viz'], html: `<p>A human-like brain that hears and composes minimal techno. It hears, it remembers, it plans, it plays.</p><p>A real brain with a memory and one stream of awareness: not a feed-forward network, not a transformer.</p><p><a href="https://cadence.floatingpragma.ai/" target="_blank" rel="noopener">${TAGLINE}</a></p>` },
  { mark: ['gearB'], html: '<p>A brain can also compose live, on a server of your own. <a href="https://github.com/muellerberndt/cadance-minimal/blob/main/docs/running.md#a-live-composer-for-the-radio-site" target="_blank" rel="noopener">How to run one</a></p><p>The gear holds its address.</p>' },
  { mark: ['bigPlayB', 'gearB'], html: '<div class="row"><button id="tourPlay" class="go">Listen now</button><button id="tourGear">Set up a live brain first</button></div>' },
];
const WHAT = '<p>What you see is the brain itself: its regions and the synapses between them.</p><p>It hears, it remembers, it plans, it plays. It settles until it comes to rest, in one stream of awareness that never resets.</p><p>Not a feed-forward network, not a transformer.</p>';
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
paintControls();
