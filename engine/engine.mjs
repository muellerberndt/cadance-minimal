// cadence-minimal instrument: modelled drum machines (606, 808, 909), a 303-style acid voice, a
// general synth voice, a SID-style chip voice, a one-shot sampler, a tempo-synced ping-pong delay, a
// reverb and a master bus.
// One ES module for Node and the AudioWorklet: no Node imports in this file.
//
// Every patch parameter is a number in 0..1; PARAMS maps it to its physical range. A token is
// {voice, device?, patch, pattern}; a pool holds 4 layers x 33 tokens (index 0 = silence).

const TAU = 2 * Math.PI;
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const mtof = (m) => 440 * Math.pow(2, (m - 69) / 12);
const DC = 1e-15; // keeps decaying effect states out of the denormal range

// ---------------------------------------------------------------- parameters

function P(name, map, lo, hi, def, unit = '') { return { name, map, lo, hi, def, unit }; }
// a register with n settings: unit u selects round(u * (n - 1)); it switches at a bar start and never glides
function Q(name, n, def, unit = '', lo = 0, hi = n - 1) { return { name, map: 'lin', lo, hi, def, unit, q: n }; }

export function phys(p, u) {
  u = clamp(u, 0, 1);
  if (p.map === 'exp') return p.lo * Math.pow(p.hi / p.lo, u);
  if (p.map === 'sq') return p.lo + (p.hi - p.lo) * u * u;
  return p.lo + (p.hi - p.lo) * u;
}

export function unit(p, v) {
  let u;
  if (p.map === 'exp') u = Math.log(v / p.lo) / Math.log(p.hi / p.lo);
  else if (p.map === 'sq') u = Math.sqrt(Math.max(0, (v - p.lo) / (p.hi - p.lo)));
  else u = (v - p.lo) / (p.hi - p.lo);
  return clamp(u, 0, 1);
}

const COMMON = () => [
  P('level', 'sq', 0, 2, 0.55, 'gain'), P('pan', 'lin', -1, 1, 0.5, 'left..right'),
  P('send_delay', 'lin', 0, 1, 0, 'send'), P('send_reverb', 'lin', 0, 1, 0, 'send'),
];

export const DRUM_VOICES = ['kick', 'snare', 'clap', 'ch', 'oh', 'ltom', 'htom', 'rim', 'cow', 'cym'];
const HAT_VOICES = ['ch', 'oh', 'cym'], SNARE_VOICES = ['snare', 'clap', 'rim'];
const fromDb = (db) => Math.pow(10, db / 20);
// The hat modes of the drum dynamics: w = the gain in dB of a plain hat hit on the beat, on the sixteenth after it,
// on the off-beat eighth and on the sixteenth before the next beat (null: the hit is left out); lift: in a line
// the off-beat eighths are played as accents; split: plain hits on the closed hat, accents on the
// line's long sound (open hat or cymbal). With lift, every hit on an off-beat eighth is played as an accent and an
// accent on the beat is eased by onBeatDb.
const HAT_MODES = {
  off: { w: [-7, -6, 4, -5], lift: true, split: true },      // the off-beat carries the line
  push: { w: [-5, -7, 2, -1], lift: true, split: true },     // the beat down, the sixteenth before it lifted
  thin: { w: [null, -12, 4, -12], lift: true, split: true }, // nearly only the off-beat
  flat: { w: [0, 0, 0, 0], lift: false, split: false },      // as written, with the accents' contrast
};
const HAT_ODDS = { off: 0.4, push: 0.25, thin: 0.15, flat: 0.2 };
const mix32 = (a, b) => { let x = (Math.imul(a + 1, 0x9E3779B1) ^ Math.imul(b + 7, 0x85EBCA6B)) >>> 0; x ^= x >>> 15; x = Math.imul(x, 0x2C1B3C6D); x ^= x >>> 12; x = Math.imul(x, 0x297A2D39); x ^= x >>> 15; return (x >>> 0) / 4294967296; };
export const DEVICES = ['606', '808', '909'];

const drumList = [];
for (const v of DRUM_VOICES) {
  drumList.push(
    P(`${v}_tune`, 'lin', 0, 1, 0.5, 'device range'), P(`${v}_decay`, 'lin', 0, 1, 0.5, 'device range'),
    P(`${v}_level`, 'sq', 0, 1.5, 0.75, 'gain'), P(`${v}_tone`, 'lin', 0, 1, 0.5, 'device range'));
}
drumList.push(P('drive', 'lin', 0, 1, 0.1, 'saturation'));

export const PARAMS = {
  drums: [...drumList, ...COMMON()],
  acid: [
    P('wave', 'lin', 0, 1, 0, 'saw..square'), P('tune', 'lin', -12, 12, 0.5, 'semitones'),
    P('cutoff', 'exp', 50, 5000, 0.3, 'Hz'), P('reso', 'lin', 0, 1, 0.6, 'share of self-oscillation'),
    P('envmod', 'lin', 0, 5, 0.4, 'octaves'), P('decay', 'exp', 0.03, 2, 0.4, 's'),
    P('accent_amt', 'lin', 0, 1, 0.5, ''), P('drive', 'lin', 0, 1, 0.2, 'saturation'),
    P('glide', 'exp', 0.015, 0.25, 0.4, 's'), P('gate', 'lin', 0.2, 1, 0.5, 'share of a step'),
    ...COMMON(),
  ],
  sampler: [
    P('tune', 'lin', -12, 12, 0.5, 'semitones'), P('start', 'lin', 0, 0.85, 0, 'source fraction'),
    P('reverse', 'lin', 0, 1, 0, 'reverse'), P('decay', 'exp', 0.025, 2, 0.65, 's'),
    P('cutoff', 'exp', 40, 16000, 0.83, 'Hz'), P('reso', 'exp', 0.5, 4, 0.2, 'Q'),
    ...COMMON(),
  ],
  synth: [
    P('wave', 'lin', 0, 1, 0.5, 'sine 0, saw .5, square 1'), P('tune', 'lin', -24, 24, 0.5, 'semitones'),
    P('fm_ratio', 'exp', 0.25, 12, 0.4, 'modulator / carrier'), P('fm_amt', 'sq', 0, 10, 0, 'index'),
    P('pitch_env', 'lin', -24, 48, 1 / 3, 'semitones'), P('pitch_decay', 'exp', 0.003, 1.5, 0.3, 's'),
    P('cutoff', 'exp', 40, 16000, 0.6, 'Hz'), P('reso', 'exp', 0.5, 20, 0.2, 'Q'),
    P('ftype', 'lin', 0, 1, 0, 'lowpass 0, bandpass .5, highpass 1'), P('f_env', 'lin', -6, 6, 0.5, 'octaves'),
    P('f_decay', 'exp', 0.005, 3, 0.4, 's'), P('attack', 'exp', 0.001, 2, 0.1, 's'),
    P('decay', 'exp', 0.01, 4, 0.5, 's'), P('sustain', 'lin', 0, 1, 0.5, ''),
    P('release', 'exp', 0.005, 4, 0.3, 's'), P('lfo_rate', 'exp', 0.05, 40, 0.5, 'Hz'),
    P('lfo_pitch', 'sq', 0, 12, 0, 'semitones'), P('lfo_cutoff', 'lin', 0, 4, 0, 'octaves'),
    P('lfo_amp', 'lin', 0, 1, 0, 'tremolo depth'), P('drive', 'lin', 0, 1, 0, 'saturation'),
    ...COMMON(),
  ],
  // one SID-style chip per instrument: three oscillators with their own envelopes, one multimode filter,
  // and the three moves a player routine makes at every note (filter sweep, its decay, pulse-width sweep)
  sid: [
    ...[1, 2, 3].flatMap((k) => [
      Q(`wave${k}`, 5, k === 1 ? 0.75 : 0, 'off, triangle, saw, pulse, noise'),
      Q(`tune${k}`, 25, 0.5, 'semitones', -12, 12), P(`fine${k}`, 'lin', -30, 30, 0.5, 'cents'),
      P(`pw${k}`, 'lin', 0.05, 0.95, 0.5, 'duty'),
      Q(`attack${k}`, 16, 0, 'rate nibble'), Q(`decay${k}`, 16, 7 / 15, 'rate nibble'),
      Q(`sustain${k}`, 16, 6 / 15, 'level nibble'), Q(`release${k}`, 16, 5 / 15, 'rate nibble'),
    ]),
    Q('fx', 4, 0, 'none, ring, sync, ring+sync: oscillator 1 by oscillator 3'),
    Q('route', 8, 1, 'bit k-1 set: oscillator k passes the filter'),
    P('cutoff', 'exp', 30, 12000, 1, 'Hz'), P('reso', 'lin', 0, 1, 0, 'resonance'),
    Q('mode', 4, 0, 'lowpass, bandpass, highpass, notch'),
    P('f_env', 'lin', -4, 4, 0.5, 'octaves at note start'), P('f_decay', 'exp', 0.01, 2, 0.4, 's'),
    P('pwm', 'lin', 0, 1, 0, 'pulse-width sweep'),
    ...COMMON(),
  ],
};
export const SID_WAVES = ['off', 'triangle', 'saw', 'pulse', 'noise'];
export const SID_FX = ['none', 'ring', 'sync', 'ring+sync'];
export const SID_MODES = ['lowpass', 'bandpass', 'highpass', 'notch'];

export const INDEX = {};
for (const [v, list] of Object.entries(PARAMS)) { INDEX[v] = {}; list.forEach((p, i) => { INDEX[v][p.name] = i; }); }

export function toUnit(voice, name, value) { return unit(PARAMS[voice][INDEX[voice][name]], value); }

// Per-bar modulation the brain sends with every bar: [filter, space] per layer, each -1..1,
// glided from the previous bar's values across the bar like the patch.
export const MOD_RANGE = {
  levels: [-3, -2, -1, 0, 1, 2, 3], scale: 1 / 3,
  filter: { min: -1, max: 1, label: 'FILTER', cutoff_offset: 0.35, envmod_offset: 0.12, drum_lowpass_hz: [18000, 450], drum_highpass_hz: [20, 140] },
  space: { min: -1, max: 1, label: 'SPACE', send_offset: 0.35 },
};

const FX_DEFAULTS = { delay_steps: 3, delay_feedback: 0.45, delay_tone: 0.5, reverb_size: 0.7, reverb_damp: 0.5, delay_return: 0.8, reverb_return: 0.6, master: 0.9 };

// ---------------------------------------------------------------- tokens

function arr16(a, f) {
  const out = new Array(16);
  for (let i = 0; i < 16; i++) out[i] = f(Array.isArray(a) && a.length ? a[i % a.length] : undefined);
  return out;
}

export function normPattern(voice, pat) {
  pat = pat || {};
  if (voice === 'drums') {
    const hits = {};
    for (const v of DRUM_VOICES) {
      const a = pat.hits && pat.hits[v];
      hits[v] = Array.isArray(a) ? arr16(a, (x) => clamp(+x || 0, 0, 1)) : null;
    }
    return { hits };
  }
  const notes = arr16(pat.notes, (x) => (x == null || !Number.isFinite(+x) ? null : +x));
  const accent = Array.isArray(pat.accent) ? arr16(pat.accent, (x) => (+x ? 1 : 0))
    : Array.isArray(pat.vel) ? arr16(pat.vel, (x) => ((+x || 0) > 0.8 ? 1 : 0)) : new Array(16).fill(0);
  const vel = Array.isArray(pat.vel) ? arr16(pat.vel, (x) => clamp(+x || 0, 0, 1))
    : notes.map((n, i) => (n == null ? 0 : accent[i] ? 1 : 0.7));
  const slide = Array.isArray(pat.slide) ? arr16(pat.slide, (x) => (+x ? 1 : 0)) : new Array(16).fill(0);
  const len = Array.isArray(pat.len) ? arr16(pat.len, (x) => Math.max(0.05, +x || 1)) : new Array(16).fill(1);
  return { notes, vel, accent, slide, len };
}

export function resolveToken(tok) {
  if (!tok || typeof tok !== 'object') return null;
  const voice = PARAMS[tok.voice] ? tok.voice : null;
  if (!voice) return null;
  const list = PARAMS[voice], u = new Float64Array(list.length), patch = tok.patch || {};
  list.forEach((p, i) => { const v = +patch[p.name]; u[i] = Number.isFinite(v) ? clamp(v, 0, 1) : p.def; });
  const device = voice === 'drums' ? (DEVICES.includes(String(tok.device)) ? String(tok.device) : '808') : null;
  // gain: the pool's gain staging (pool/gain_stage.py), a plain multiplier that keeps every sound in an audible range
  const g = +tok.gain; const gain = Number.isFinite(g) && g > 0 ? Math.min(g, 32) : 1;
  const sample = typeof tok.sample === "string" ? tok.sample : null;
  return { voice, device, sample, u, gain, pattern: normPattern(voice, tok.pattern) };
}

// ---------------------------------------------------------------- building blocks

class Noise {
  constructor(seed) { this.s = (seed >>> 0) || 1; }
  next() { let x = this.s; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.s = x >>> 0; return this.s / 2147483648 - 1; }
}

// RBJ peaking filter: the drum bus EQ that masters the kit to the corpus drum spectrum (pool.fx.drum_eq)
class Peak {
  constructor(sr, f, gainDb, q) {
    const A = Math.pow(10, gainDb / 40), w = TAU * Math.min(f, 0.45 * sr) / sr, al = Math.sin(w) / (2 * q), c = Math.cos(w);
    const a0 = 1 + al / A;
    this.b0 = (1 + al * A) / a0; this.b1 = (-2 * c) / a0; this.b2 = (1 - al * A) / a0; this.a1 = (-2 * c) / a0; this.a2 = (1 - al / A) / a0;
    this.x1 = this.x2 = this.y1 = this.y2 = 0;
  }
  run(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = Math.abs(y) < 1e-20 ? 0 : y;
    return this.y1;
  }
}

class SVF {
  constructor() { this.ic1 = 0; this.ic2 = 0; this.k = 1.414; this.a1 = 0; this.a2 = 0; this.a3 = 0; this.lp = 0; this.bp = 0; this.hp = 0; }
  set(fc, q, sr) {
    const g = Math.tan((Math.PI * clamp(fc, 16, sr * 0.45)) / sr), k = 1 / Math.max(q, 0.3);
    this.k = k; this.a1 = 1 / (1 + g * (g + k)); this.a2 = g * this.a1; this.a3 = g * this.a2;
  }
  tick(x) {
    const v3 = x - this.ic2, v1 = this.a1 * this.ic1 + this.a2 * v3, v2 = this.ic2 + this.a2 * this.ic1 + this.a3 * v3;
    this.ic1 = 2 * v1 - this.ic1; this.ic2 = 2 * v2 - this.ic2;
    this.lp = v2; this.bp = v1; this.hp = x - this.k * v1 - v2;
    return v2;
  }
  reset() { this.ic1 = 0; this.ic2 = 0; }
}

function polyblep(t, dt) {
  if (t < dt) { t /= dt; return t + t - t * t - 1; }
  if (t > 1 - dt) { t = (t - 1) / dt; return t * t + t + t + 1; }
  return 0;
}

const kOf = (tau, sr) => Math.exp(-1 / (Math.max(tau, 1e-5) * sr));

// ---------------------------------------------------------------- drum machines

// Per-device ranges: [lo, hi] maps of tune and decay, and voice constants.
const KICK = { '606': { f: [55, 100], tau: [0.02, 0.12], sweep: 2.4, tp: 0.008, click: 0.25, shape: 0.7 },   // punch: deeper sweep, knock, a little saturation
               '808': { f: [38, 68], tau: [0.08, 0.6], sweep: 2.3, tp: 0.012, click: 0.16, shape: 0.5 },
               '909': { f: [42, 72], tau: [0.05, 0.3], sweep: 5.0, tp: 0.018, click: 0.5, shape: 1.8 } };

class Kick {
  constructor(sr) { this.sr = sr; this.on = false; this.n = new Noise(11); this.hp = new SVF(); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const D = KICK[dev], sr = this.sr;
    this.f0 = D.f[0] * Math.pow(D.f[1] / D.f[0], tune);
    this.ka = kOf(D.tau[0] * Math.pow(D.tau[1] / D.tau[0], decay) * (acc ? 1.15 : 1), sr);
    this.ratio = 1 + (D.sweep - 1) * (0.4 + 1.2 * tone);
    this.kp = kOf(D.tp, sr); this.kc = kOf(0.0012, sr);
    this.click = D.click * (0.3 + 1.4 * tone); this.shape = D.shape;
    this.g = gain * (acc ? 1.3 : 1); this.ph = 0; this.env = 1; this.penv = 1; this.cenv = 1; this.att = 0;
    this.hp.set(1500, 0.7, sr); this.on = true;
  }
  tick() {
    const f = this.f0 * (1 + (this.ratio - 1) * this.penv);
    this.penv *= this.kp; this.ph += f / this.sr; if (this.ph >= 1) this.ph -= 1;
    let s = Math.sin(TAU * this.ph);
    if (this.shape > 0) s = Math.tanh(this.shape * s) / Math.tanh(this.shape);
    if (this.att < 1) this.att = Math.min(1, this.att + 2500 / this.sr);
    this.hp.tick(this.n.next());
    const y = s * this.env * this.att + this.click * this.cenv * this.hp.hp;
    this.env *= this.ka; this.cenv *= this.kc;
    if (this.env < 1e-4) this.on = false;
    return y * this.g;
  }
}

const SNARE = { '606': { f: [200, 400], body: 0.04, tau: [0.015, 0.09], hp: 3000, drop: 0.05 },
                '808': { f: [238, 476], body: 0.06, tau: [0.02, 0.14], hp: 1800, drop: 0.0 },
                '909': { f: [185, 330], body: 0.05, tau: [0.02, 0.12], hp: 2500, drop: 0.25 } };

class Snare {
  constructor(sr) { this.sr = sr; this.on = false; this.n = new Noise(23); this.f = new SVF(); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const D = SNARE[dev], sr = this.sr, sc = Math.pow(1.6, tune * 2 - 1);
    this.f1 = D.f[0] * sc; this.f2 = D.f[1] * sc; this.drop = D.drop;
    this.kb = kOf(D.body * (dev === '909' ? 1 : 1.2), sr); this.kd = kOf(0.012, sr);
    this.kn = kOf(D.tau[0] * Math.pow(D.tau[1] / D.tau[0], decay) * (acc ? 1.1 : 1), sr);
    this.nl = 0.25 + 1.1 * tone; this.bl = 1.1 - 0.5 * tone;
    this.f.set(D.hp * (dev === '909' ? 1.6 : 1), dev === '909' ? 1.2 : 0.7, sr); this.dev = dev;
    this.g = gain * (acc ? 1.3 : 1); this.p1 = 0; this.p2 = 0; this.eb = 1; this.en = 1; this.ed = 1; this.on = true;
  }
  tick() {
    const d = 1 + this.drop * this.ed; this.ed *= this.kd;
    this.p1 += (this.f1 * d) / this.sr; this.p2 += (this.f2 * d) / this.sr;
    if (this.p1 >= 1) this.p1 -= 1; if (this.p2 >= 1) this.p2 -= 1;
    const body = (Math.sin(TAU * this.p1) + 0.6 * Math.sin(TAU * this.p2)) * this.eb * this.bl;
    this.f.tick(this.n.next());
    const nz = (this.dev === '909' ? this.f.bp * 0.7 + this.f.hp * 0.5 : this.f.hp) * this.en * this.nl;
    this.eb *= this.kb; this.en *= this.kn;
    if (this.en < 1e-4 && this.eb < 1e-4) this.on = false;
    return (body * 0.6 + nz) * this.g;
  }
}

class Clap {
  constructor(sr) { this.sr = sr; this.on = false; this.n = new Noise(37); this.f = new SVF(); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const sr = this.sr, sc = Math.pow(1.5, tune * 2 - 1);
    const fc = dev === '808' ? 1000 : dev === '909' ? 1300 : 2200;
    this.f.set(fc * sc, 1.2 + 4 * tone, sr);
    this.offs = dev === '606' ? [0] : dev === '909' ? [0, 0.008, 0.016] : [0, 0.009, 0.018, 0.027];
    this.offs = this.offs.map((o) => Math.round(o * sr));
    this.kb = kOf(0.0028, sr);
    const tail = dev === '606' ? [0.02, 0.12] : [0.04, 0.35];
    this.kt = kOf(tail[0] * Math.pow(tail[1] / tail[0], decay), sr);
    this.tl = dev === '606' ? 0.8 : 0.55; this.t = 0; this.eb = 0; this.et = 0;
    this.g = gain * (acc ? 1.3 : 1) * (dev === '606' ? 2.2 : 2.6); this.on = true;
  }
  tick() {
    const t = this.t++;
    for (let i = 0; i < this.offs.length; i++) if (t === this.offs[i]) { this.eb = 1; if (i === this.offs.length - 1) this.et = this.tl; }
    this.f.tick(this.n.next());
    const y = this.f.bp * this.f.k * (this.eb + this.et);
    this.eb *= this.kb; this.et *= this.kt;
    if (t > this.offs[this.offs.length - 1] + 8 && this.et < 1e-4) this.on = false;
    return y * this.g;
  }
}

const METAL = [205.3, 304.4, 369.6, 522.7, 540, 800];
const METAL_DEV = { '606': { scale: 1.08, noise: 0.25, bp: 1.0 }, '808': { scale: 1.0, noise: 0.05, bp: 1.0 }, '909': { scale: 1.32, noise: 0.45, bp: 1.2 } };
const METAL_TAU = { ch: [0.01, 0.06], oh: [0.08, 0.5], cym: [0.25, 1.6] };

class Metal {
  constructor(sr, kind) { this.sr = sr; this.kind = kind; this.on = false; this.n = new Noise(kind === 'ch' ? 41 : kind === 'oh' ? 43 : 47); this.bp = new SVF(); this.hp = new SVF(); this.bp2 = new SVF(); this.ph = new Float64Array(6); this.inc = new Float64Array(6); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const D = METAL_DEV[dev], sr = this.sr, sc = Math.pow(1.4, tune * 2 - 1);
    for (let i = 0; i < 6; i++) this.inc[i] = (METAL[i] * sc * D.scale) / sr;
    this.nm = D.noise;
    const T = METAL_TAU[this.kind];
    this.ke = kOf(T[0] * Math.pow(T[1] / T[0], decay) * (dev === '606' && this.kind === 'ch' ? 0.8 : 1), sr);
    if (this.kind === 'cym') { this.bp.set(3400 * sc, 1.5, sr); this.bp2.set(8000 * sc, 1.2, sr); this.hp.set(2500, 0.7, sr); }
    else { this.bp.set(5500 * Math.pow(2, tone) * D.bp, 1.6, sr); this.hp.set(6000, 0.7, sr); }
    this.tone = tone; this.g = gain * (acc ? 1.3 : 1) * 2.2; this.env = 1; this.att = 0; this.kc = 1; this.on = true;
  }
  choke() { if (this.on) this.kc = kOf(0.006, this.sr); }
  tick() {
    let s = 0;
    for (let i = 0; i < 6; i++) { this.ph[i] += this.inc[i]; if (this.ph[i] >= 1) this.ph[i] -= 1; s += this.ph[i] < 0.5 ? 1 : -1; }
    s = (s / 6) * (1 - this.nm) + this.n.next() * this.nm;
    let y;
    if (this.kind === 'cym') { this.bp.tick(s); this.bp2.tick(s); y = this.bp.bp * (1 - this.tone) + this.bp2.bp * (0.5 + this.tone); this.hp.tick(y); y = this.hp.hp; }
    else { this.bp.tick(s); this.hp.tick(this.bp.bp); y = this.hp.hp; }
    if (this.att < 1) this.att = Math.min(1, this.att + 8000 / this.sr);
    y *= this.env * this.att; this.env *= this.ke; if (this.kc < 1) this.env *= this.kc;
    if (this.env < 1e-4) this.on = false;
    return y * this.g;
  }
}

const TOM = { '606': { f: [90, 160], tau: [0.04, 0.3], drop: 0.15 }, '808': { f: [70, 140], tau: [0.08, 0.6], drop: 0.25 }, '909': { f: [80, 150], tau: [0.06, 0.45], drop: 0.6 } };

class Tom {
  constructor(sr, high) { this.sr = sr; this.high = high; this.on = false; this.n = new Noise(high ? 53 : 59); this.f = new SVF(); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const D = TOM[dev], sr = this.sr, m = this.high ? 1.8 : 1;
    this.f0 = D.f[0] * m * Math.pow(D.f[1] / D.f[0], tune); this.drop = D.drop;
    this.ke = kOf(D.tau[0] * Math.pow(D.tau[1] / D.tau[0], decay), sr); this.kd = kOf(0.03, sr); this.kn = kOf(0.02, sr);
    this.nl = 0.35 * tone; this.f.set(this.f0 * 4, 0.8, sr);
    this.g = gain * (acc ? 1.3 : 1); this.ph = 0; this.env = 1; this.ed = 1; this.en = 1; this.on = true;
  }
  tick() {
    this.ph += (this.f0 * (1 + this.drop * this.ed)) / this.sr; if (this.ph >= 1) this.ph -= 1;
    this.f.tick(this.n.next());
    const y = Math.sin(TAU * this.ph) * this.env + this.f.lp * this.en * this.nl;
    this.env *= this.ke; this.ed *= this.kd; this.en *= this.kn;
    if (this.env < 1e-4) this.on = false;
    return y * this.g;
  }
}

class Rim {
  constructor(sr) { this.sr = sr; this.on = false; this.hp = new SVF(); this.n = new Noise(61); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const sr = this.sr, sc = Math.pow(1.4, tune * 2 - 1);
    this.w1 = (TAU * 1667 * sc * (dev === '606' ? 1.08 : 1)) / sr; this.w2 = (TAU * 455 * sc) / sr;
    this.ke = kOf(0.004 * Math.pow(8, decay), sr); this.a = 0.4 + 0.6 * tone; this.b = 1.2 - 0.8 * tone;
    this.nz = dev === '909' ? 0.3 : 0; this.hp.set(300, 0.7, sr);
    this.g = gain * (acc ? 1.3 : 1) * 1.6; this.t = 0; this.env = 1; this.on = true;
  }
  tick() {
    const t = this.t++;
    let s = Math.sin(this.w1 * t) * this.a + Math.sin(this.w2 * t) * this.b + this.n.next() * this.nz * (t < 60 ? 1 : 0);
    this.hp.tick(s * this.env); this.env *= this.ke;
    if (this.env < 1e-4) this.on = false;
    return this.hp.hp * this.g;
  }
}

class Cow {
  constructor(sr) { this.sr = sr; this.on = false; this.f = new SVF(); }
  trigger(dev, tune, decay, tone, gain, acc) {
    const sr = this.sr, sc = Math.pow(1.3, tune * 2 - 1);
    this.i1 = (540 * sc) / sr; this.i2 = (800 * sc) / sr; this.f.set(800 * Math.pow(3.75, tone), 3, sr);
    this.k1 = kOf(0.01, sr); this.k2 = kOf(0.04 * Math.pow(10, decay), sr);
    this.g = gain * (acc ? 1.3 : 1) * 1.4; this.p1 = 0; this.p2 = 0; this.e1 = 0.7; this.e2 = 0.3; this.on = true;
  }
  tick() {
    this.p1 += this.i1; this.p2 += this.i2; if (this.p1 >= 1) this.p1 -= 1; if (this.p2 >= 1) this.p2 -= 1;
    const s = ((this.p1 < 0.5 ? 1 : -1) + (this.p2 < 0.5 ? 1 : -1)) * 0.5;
    this.f.tick(s); const env = this.e1 + this.e2; this.e1 *= this.k1; this.e2 *= this.k2;
    if (env < 1e-4) this.on = false;
    return this.f.bp * this.f.k * env * this.g;
  }
}

// Physical ranges of the drum knobs per device (for the page): [min, max, unit] at u = 0 and u = 1.
// Decay values are the envelope's time constant (the sound falls by about 26 dB in three of them).
export const DEVICE_VOICES = {
  '606': ['kick', 'snare', 'ltom', 'htom', 'ch', 'oh', 'cym'],
  '808': ['kick', 'snare', 'clap', 'ch', 'oh', 'ltom', 'htom', 'rim', 'cow', 'cym'],
  '909': ['kick', 'snare', 'clap', 'ch', 'oh', 'ltom', 'htom', 'rim', 'cym'],
};
export function drumRanges(dev) {
  const r = (a, b, u) => [Math.round(a * 1000) / 1000, Math.round(b * 1000) / 1000, u];
  const K = KICK[dev], S = SNARE[dev], M = METAL_DEV[dev], T = TOM[dev];
  const clapF = dev === '808' ? 1000 : dev === '909' ? 1300 : 2200, clapT = dev === '606' ? [0.02, 0.12] : [0.04, 0.35];
  const metal = (kind) => ({ tune: r((205.3 * M.scale) / 1.4, 205.3 * M.scale * 1.4, 'Hz (lowest of six)'), decay: r(METAL_TAU[kind][0], METAL_TAU[kind][1], 's'),
    tone: kind === 'cym' ? r(0, 1, 'low..high band') : r(5500 * M.bp, 11000 * M.bp, 'Hz band') });
  return {
    kick: { tune: r(K.f[0], K.f[1], 'Hz'), decay: r(K.tau[0], K.tau[1], 's'), tone: r(0, 1, 'sweep + click') },
    snare: { tune: r(S.f[0] / 1.6, S.f[0] * 1.6, 'Hz'), decay: r(S.tau[0], S.tau[1], 's'), tone: r(0, 1, 'snappy') },
    clap: { tune: r(clapF / 1.5, clapF * 1.5, 'Hz band'), decay: r(clapT[0], clapT[1], 's'), tone: r(1.2, 5.2, 'Q') },
    ch: metal('ch'), oh: metal('oh'), cym: metal('cym'),
    ltom: { tune: r(T.f[0], T.f[1], 'Hz'), decay: r(T.tau[0], T.tau[1], 's'), tone: r(0, 0.35, 'noise') },
    htom: { tune: r(T.f[0] * 1.8, T.f[1] * 1.8, 'Hz'), decay: r(T.tau[0], T.tau[1], 's'), tone: r(0, 0.35, 'noise') },
    rim: { tune: r(1667 / 1.4, 1667 * 1.4, 'Hz'), decay: r(0.004, 0.032, 's'), tone: r(0, 1, 'high..low ping') },
    cow: { tune: r(540 / 1.3, 540 * 1.3, 'Hz (lower of two)'), decay: r(0.04, 0.4, 's'), tone: r(800, 3000, 'Hz band') },
  };
}

// A recorded one-shot of the real machine (Engine.setSamples). Same knobs as the models, all 0..1:
// tune = playback rate (+-12 semitones, 0.5 = original), decay = fade (1 = the full recording, lower =
// exponential fade sooner), tone = tilt (lowpass below 0.5, highpass above, 0.5 = flat), level = gain.
class SampleVoice {
  constructor(sr) { this.sr = sr; this.on = false; this.f = new SVF(); this.mode = 0; this.buf = null; }
  trigger(buf, srcRate, tune, decay, tone, gain, acc) {
    const rate = Math.pow(2, (tune - 0.5) * 2);
    this.buf = buf; this.pos = 0; this.rate = (srcRate / this.sr) * rate;
    const len = buf.length / srcRate / rate;
    this.ka = decay >= 0.98 ? 1 : kOf(len * (0.06 + 0.94 * decay * decay), this.sr);
    this.env = 1; this.g = gain * (acc ? 1.15 : 1);
    if (Math.abs(tone - 0.5) < 0.02) this.mode = 0;
    else if (tone < 0.5) { this.mode = -1; this.f.set(600 * Math.pow(20000 / 600, tone / 0.5), 0.7, this.sr); }
    else { this.mode = 1; this.f.set(20 * Math.pow(800 / 20, (tone - 0.5) / 0.5), 0.7, this.sr); }
    this.f.reset(); this.on = true;
  }
  choke() { if (this.on) this.ka = Math.min(this.ka, kOf(0.006, this.sr)); }
  tick() {
    const b = this.buf, i = this.pos | 0;
    if (i >= b.length - 1) { this.on = false; return 0; }
    let s = b[i] + (b[i + 1] - b[i]) * (this.pos - i);
    this.pos += this.rate;
    s *= this.env; this.env *= this.ka;
    if (this.env < 1e-4) this.on = false;
    if (this.mode !== 0) { this.f.tick(s); s = this.mode < 0 ? this.f.lp : this.f.hp; }
    return s * this.g;
  }
}

const PUNCH_DB = 24;   // a kick hit's level one beat after it starts, below its start, after the drive

export class DrumKit {
  constructor(sr) {
    this.v = { kick: new Kick(sr), snare: new Snare(sr), clap: new Clap(sr), ch: new Metal(sr, 'ch'), oh: new Metal(sr, 'oh'),
      ltom: new Tom(sr, false), htom: new Tom(sr, true), rim: new Rim(sr), cow: new Cow(sr), cym: new Metal(sr, 'cym') };
    this.sv = Object.fromEntries(DRUM_VOICES.map((k) => [k, new SampleVoice(sr)]));
    this.list = DRUM_VOICES.map((k) => this.v[k]).concat(DRUM_VOICES.map((k) => this.sv[k])); this.drive = 0;
    this.rest = this.list.filter((v) => v !== this.v.kick && v !== this.sv.kick);
    this.bank = null; this.bankRate = sr;     // {device: {voice: Float32Array}} from Engine.setSamples
    // THE KICK PULSES (11-23): whatever a kit's decay and drive, each kick hit falls by PUNCH_DB within one beat,
    // AFTER the drive. A fitted kit with a long kick and full drive otherwise saturates into one constant boom
    // that runs into the next hit: a kick in every bar and no audible pulse.
    this.punch = 0; this.kPunch = 1; this.beat = sr * 0.5; this.sr = sr;
    // THE KICK BOOST (11-26, Engine.setDrumDynamics, kickBoost): kb = {att, kAtt, body, kBody, bodyHz, bodyQ, subHz} or
    // null; kg = the level gain of the kit that plays and kc = the ceiling of this hit (Layer.kickGain); ka, kbe = the
    // attack's and the body's envelopes, set to 1 at each kick; kx = what the boost added to this sample (the kit's
    // level reading leaves it out). The level gain and the attack act on the kick above subHz: its sub stays as written.
    this.kb = null; this.kg = 1; this.kc = Infinity; this.ka = 0; this.kbe = 0; this.kx = 0; this.kf = new SVF(); this.ks = new SVF();
  }
  setKickBoost(kb) { this.kb = kb; this.kx = 0; if (kb) { this.kf.set(kb.bodyHz, kb.bodyQ, this.sr); this.ks.set(kb.subHz, 0.6, this.sr); } }
  setBeat(samples) { this.beat = samples; }
  // g: a gain on top of the hit's own; as: the voice that sounds (at the written voice's level), when not the written one
  hit(dev, written, vel, u, g = 1, as = null) {
    const I = INDEX.drums, L = PARAMS.drums, name = as || written;
    const lvl = phys(L[I[`${written}_level`]], u[I[`${written}_level`]]);
    const vg = (0.35 + 0.65 * Math.min(vel, 0.8) / 0.8) * g;
    if (name === 'ch') { this.v.oh.choke(); this.sv.oh.choke(); }
    if (name === 'kick') { this.punch = 1; this.kPunch = Math.pow(10, -PUNCH_DB / 20 / Math.max(1, this.beat)); this.ka = 1; this.kbe = 1; }
    const buf = this.bank && this.bank[dev] && this.bank[dev][name];
    if (buf && buf.length > 1) this.sv[name].trigger(buf, this.bankRate, u[I[`${name}_tune`]], u[I[`${name}_decay`]], u[I[`${name}_tone`]], lvl * vg, vel > 0.8);
    else this.v[name].trigger(dev, u[I[`${name}_tune`]], u[I[`${name}_decay`]], u[I[`${name}_tone`]], lvl * vg, vel > 0.8);
  }
  set(u) { this.drive = u[INDEX.drums.drive]; }
  // the peak of one voice of a kit, hit once with an accent (this kit must be a scratch one: it is reset)
  peakOf(dev, name, u, as = null, seconds = 0.06) { return this.levelOf(dev, name, u, as, seconds).peak; }
  // the same hit's peak and its loudness: the root mean square over `seconds` of the sound as the ear weighs it
  // (lows under about 150 Hz count less, as in loudness meters)
  levelOf(dev, name, u, as = null, seconds = 0.15, vel = 1) {
    this.reset(); this.set(u); this.hit(dev, name, vel, u, 1, as);
    let pk = 0, ss = 0, lp = 0; const n = Math.round(seconds * this.sr), k = 1 - Math.exp(-2 * Math.PI * 150 / this.sr);
    for (let i = 0; i < n; i++) { const x = this.tick(), a = Math.abs(x); if (a > pk) pk = a; lp += k * (x - lp); const h = x - 0.7 * lp; ss += h * h; }
    this.reset();
    return { peak: pk, loud: Math.sqrt(ss / Math.max(1, n)) };
  }
  // the same hit's level as the voice levels hear it (Engine.setVoiceLevels: lows under 100 Hz out, highs over 1.5 kHz
  // 4 dB up), the root mean square over `seconds` (this kit must be a scratch one: it is reset)
  heardOf(dev, name, u, vel = 1, seconds = 0.15, eq = null) {
    this.reset(); this.set(u); this.hit(dev, name, vel, u, 1);
    const n = Math.round(seconds * this.sr), k1 = 1 - Math.exp(-2 * Math.PI * 100 / this.sr), k2 = 1 - Math.exp(-2 * Math.PI * 1500 / this.sr);
    let a = 0, b = 0, ss = 0;
    for (let i = 0; i < n; i++) { let x = this.tick(); if (eq) for (const p of eq) x = p.run(x); a += k1 * (x - a); const h = x - a; b += k2 * (h - b); const w = h + 0.585 * (h - b); ss += w * w; }
    this.reset();
    return Math.sqrt(ss / Math.max(1, n));
  }
  active() { for (const v of this.list) if (v.on) return true; return false; }
  tick() {
    let s = 0, kk = 0;
    for (const v of this.rest) if (v.on) s += v.tick();
    if (this.v.kick.on) kk += this.v.kick.tick();
    if (this.sv.kick.on) kk += this.sv.kick.tick();
    if (this.drive > 0.01) {                  // the kick is driven on its own, so its pulse survives the saturation
      const k = 1 + 4 * this.drive, n = 1 + 1.5 * this.drive;
      s = Math.tanh(s * k) / n; kk = Math.tanh(kk * k) / n;
    }
    kk *= this.punch; this.punch *= this.kPunch;
    if (this.kb !== null) {                   // the kick boost: the attack lifted, the body band added, the kit's level gain
      const K = this.kb, x = kk;
      this.kf.tick(x); const sub = this.ks.tick(x);
      kk = sub + (x - sub) * this.kg * (1 + K.att * this.ka) + K.body * this.kbe * this.kf.bp * this.kf.k * this.kg;
      const a = Math.abs(kk), c = this.kc;                  // the ceiling: over it the boosted kick bends toward 1.25 times it
      if (a > c) kk *= (c + 0.25 * c * Math.tanh((a - c) / (0.25 * c))) / a;
      this.ka *= K.kAtt; this.kbe *= K.kBody; this.kx = kk - x;
    }
    return s + kk;
  }
  reset() { for (const v of this.list) v.on = false; this.punch = 0; this.ka = 0; this.kbe = 0; this.kx = 0; this.kf.reset(); this.ks.reset(); }
}

// ---------------------------------------------------------------- acid (303-style)

class Acid {
  constructor(sr) {
    this.sr = sr; this.on = false; this.ph = 0; this.f = 110; this.ft = 110; this.s1 = 0; this.s2 = 0; this.s3 = 0; this.s4 = 0;
    this.amp = 0; this.fenv = 0; this.accEnv = 0; this.gate = false; this.gateLeft = 0; this.acc = false; this.tun = 0;
    this.kAtt = 1 - Math.exp(-1 / (0.003 * sr)); this.kRel = kOf(0.004, sr); this.kHold = kOf(1.6, sr);
    this.kAccUp = 1 - Math.exp(-1 / (0.012 * sr)); this.kAccDn = 1 - Math.exp(-1 / (0.15 * sr)); this.kAccEnv = kOf(0.2, sr);
    this.set(PARAMS.acid.map((p) => p.def));
  }
  set(u) {
    const L = PARAMS.acid, I = INDEX.acid, sr = this.sr;
    this.wave = u[I.wave]; this.tun = phys(L[I.tune], u[I.tune]); this.cut = phys(L[I.cutoff], u[I.cutoff]);
    this.k = 3.95 * u[I.reso]; this.envmod = phys(L[I.envmod], u[I.envmod]); this.kDec = kOf(phys(L[I.decay], u[I.decay]), sr);
    this.accAmt = u[I.accent_amt]; this.drive = u[I.drive]; this.kGlide = 1 - Math.exp(-1 / (phys(L[I.glide], u[I.glide]) * 0.35 * sr));
    this.gateShare = phys(L[I.gate], u[I.gate]);
  }
  note(midi, accent, slideIn, gateSamples) {
    this.ft = mtof(midi + this.tun); this.acc = !!accent;
    if (!(slideIn && this.gate)) { this.f = this.ft; this.fenv = 1; }
    else if (accent) this.fenv = Math.max(this.fenv, 0.6);
    this.gate = true; this.gateLeft = gateSamples; this.on = true;
  }
  tick() {
    if (this.gate && --this.gateLeft <= 0) this.gate = false;
    if (this.gate) this.amp += (1 - this.amp) * this.kAtt, this.amp *= this.kHold; else this.amp *= this.kRel;
    if (!this.gate && this.amp < 1e-5) { this.on = false; this.amp = 0; return 0; }
    this.f += (this.ft - this.f) * this.kGlide;
    const dt = this.f / this.sr; this.ph += dt; if (this.ph >= 1) this.ph -= 1;
    const saw = 2 * this.ph - 1 - polyblep(this.ph, dt);
    let ph2 = this.ph + 0.5; if (ph2 >= 1) ph2 -= 1;
    const sq = (this.ph < 0.5 ? 1 : -1) + polyblep(this.ph, dt) - polyblep(ph2, dt);
    const osc = saw + (sq - saw) * this.wave;
    this.fenv *= this.acc ? this.kAccEnv : this.kDec;
    const target = this.acc ? this.fenv : 0;
    this.accEnv += (target - this.accEnv) * (target > this.accEnv ? this.kAccUp : this.kAccDn);
    const fc = clamp(this.cut * Math.pow(2, this.envmod * this.fenv * (this.acc ? 0.6 : 1) + this.accAmt * 2.2 * this.accEnv), 20, this.sr * 0.42);
    const g = Math.tan((Math.PI * fc) / this.sr), G = g / (1 + g), b = 1 / (1 + g), G2 = G * G, G4 = G2 * G2;
    const S = G2 * G * b * this.s1 + G2 * b * this.s2 + G * b * this.s3 + b * this.s4;
    const x = osc * (0.7 + 2.5 * this.drive);
    let uin = (x - this.k * S) / (1 + this.k * G4);
    uin = Math.tanh(uin);
    let v = (uin - this.s1) * G, y = v + this.s1; this.s1 = y + v;
    v = (y - this.s2) * G; y = v + this.s2; this.s2 = y + v;
    v = (y - this.s3) * G; y = v + this.s3; this.s3 = y + v;
    v = (y - this.s4) * G; y = v + this.s4; this.s4 = y + v;
    const out = Math.tanh(y * (1 + 0.45 * this.k) * (1 + this.drive)) * 0.55;
    return out * this.amp * (this.acc ? 1 + 0.6 * this.accAmt : 1);
  }
  active() { return this.on; }
  reset() { this.on = false; this.gate = false; this.amp = 0; this.s1 = this.s2 = this.s3 = this.s4 = 0; }
}

// ---------------------------------------------------------------- synth

class SynthVoice {
  constructor(sr, seed) { this.sr = sr; this.on = false; this.f = new SVF(); this.ph = 0; this.mph = 0; this.n = new Noise(seed); this.age = 0; }
  start(midi, vel, lenSamples, S) {
    this.midi = midi; this.vel = vel; this.len = lenSamples; this.t = 0; this.stage = 0; this.env = this.on ? this.env : 0;
    this.aInc = 1 / Math.max(1, S.attack * this.sr); this.kD = kOf(S.decay / 4, this.sr); this.sus = S.sustain; this.kR = kOf(S.release / 4, this.sr);
    this.penv = 1; this.kP = kOf(S.pitch_decay, this.sr); this.pAmt = S.pitch_env;
    this.fenv = 1; this.kF = kOf(S.f_decay, this.sr); this.on = true; this.age = 0; this.ctl = 0; this.pc = 0; this.dt = 0;
  }
  tick(S, lfo) {
    const t = this.t++;
    if (this.stage === 0) { this.env += this.aInc; if (this.env >= 1) { this.env = 1; this.stage = 1; } }
    else if (this.stage === 1) this.env = this.sus + (this.env - this.sus) * this.kD;
    if (this.stage < 2 && t >= this.len) this.stage = 2;
    if (this.stage === 2) { this.env *= this.kR; if (this.env < 1e-4) { this.on = false; return 0; } }
    this.penv *= this.kP;
    if (this.pc-- <= 0) {
      this.pc = 3;
      const pitch = this.midi + S.tune + this.pAmt * this.penv + S.lfo_pitch * lfo;
      this.dt = Math.min(mtof(pitch) / this.sr, 0.45);
    }
    const dt = this.dt;
    this.mph += dt * S.fm_ratio; if (this.mph >= 1) this.mph -= Math.floor(this.mph);
    this.ph += dt; if (this.ph >= 1) this.ph -= 1;
    const pm = S.fm_amt > 0 ? (S.fm_amt * Math.sin(TAU * this.mph)) / TAU : 0;
    let p = this.ph + pm; p -= Math.floor(p);
    const sine = Math.sin(TAU * p);
    let osc;
    if (S.wave < 0.5) { const saw = 2 * this.ph - 1 - polyblep(this.ph, dt); osc = sine + (saw - sine) * (S.wave * 2); }
    else {
      const saw = 2 * this.ph - 1 - polyblep(this.ph, dt);
      let p2 = this.ph + 0.5; if (p2 >= 1) p2 -= 1;
      const sq = (this.ph < 0.5 ? 1 : -1) + polyblep(this.ph, dt) - polyblep(p2, dt);
      osc = saw + (sq - saw) * (S.wave * 2 - 1);
    }
    if (this.ctl-- <= 0) {
      this.ctl = 8;
      this.f.set(S.cutoff * Math.pow(2, S.f_env * this.fenv + S.lfo_cutoff * lfo), S.reso, this.sr);
    }
    this.fenv *= this.kF;
    this.f.tick(osc);
    const ft = S.ftype;
    let y = ft < 0.5 ? this.f.lp + (this.f.bp - this.f.lp) * ft * 2 : this.f.bp + (this.f.hp - this.f.bp) * (ft * 2 - 1);
    if (S.drive > 0.01) y = Math.tanh(y * (1 + 4 * S.drive)) / (1 + 1.5 * S.drive);
    const trem = 1 - S.lfo_amp * (0.5 + 0.5 * lfo);
    return y * this.env * this.vel * trem;
  }
}

class Synth {
  constructor(sr) {
    this.sr = sr; this.voices = [0, 1, 2, 3].map((i) => new SynthVoice(sr, 71 + i)); this.lfoPh = 0; this.S = {};
    this.set(PARAMS.synth.map((p) => p.def));
  }
  set(u) { const L = PARAMS.synth; for (let i = 0; i < L.length; i++) this.S[L[i].name] = L[i].name === 'wave' || L[i].name === 'ftype' || L[i].name === 'sustain' || L[i].name === 'lfo_amp' || L[i].name === 'drive' ? u[i] : phys(L[i], u[i]); }
  note(midi, vel, lenSamples) {
    let v = this.voices.find((x) => !x.on);
    if (!v) { v = this.voices[0]; for (const x of this.voices) if (x.age > v.age) v = x; }
    v.start(midi, vel, lenSamples, this.S);
  }
  active() { for (const v of this.voices) if (v.on) return true; return false; }
  tick() {
    this.lfoPh += this.S.lfo_rate / this.sr; if (this.lfoPh >= 1) this.lfoPh -= 1;
    const lfo = Math.sin(TAU * this.lfoPh); let s = 0;
    for (const v of this.voices) if (v.on) { v.age++; s += v.tick(this.S, lfo); }
    return s * 0.5;
  }
  reset() { for (const v of this.voices) v.on = false; }
}

// ---------------------------------------------------------------- sid (three oscillators, one filter)

// The SID's envelope rates: attack time in ms per nibble; decay and release take three times as long.
const SID_ATTACK_MS = [2, 8, 16, 24, 38, 56, 68, 80, 100, 250, 500, 800, 1000, 3000, 5000, 8000];
const SID_DECAY_MS = [6, 24, 48, 72, 114, 168, 204, 240, 300, 750, 1500, 2400, 3000, 9000, 15000, 24000];
const SID_AT = [1, 2, 3].map((k) => ['wave', 'tune', 'fine', 'pw', 'attack', 'decay', 'sustain', 'release'].map((n) => INDEX.sid[n + k]));
const nibble = (u) => Math.round(clamp(u, 0, 1) * 15);

// What a unit-coded sid patch writes to the chip: `patch` is the parameter array or a {name: value} object.
// The voice and the page read the same decoding.
export function sidRegisters(patch) {
  const L = PARAMS.sid, I = INDEX.sid;
  const at = (i) => { const v = patch && (ArrayBuffer.isView(patch) || Array.isArray(patch) ? patch[i] : patch[L[i].name]); return Number.isFinite(+v) && v != null ? clamp(+v, 0, 1) : L[i].def; };
  const osc = SID_AT.map(([w, t, f, p, a, d, s, r]) => ({
    wave: Math.round(at(w) * 4), tune: Math.round(at(t) * 24) - 12, fine: phys(L[f], at(f)), pw: phys(L[p], at(p)),
    attack: nibble(at(a)), decay: nibble(at(d)), sustain: nibble(at(s)), release: nibble(at(r)),
  }));
  const fx = Math.round(at(I.fx) * 3), route = Math.round(at(I.route) * 7);
  return {
    osc, fx, ring: fx === 1 || fx === 3, sync: fx >= 2, route, filtered: [0, 1, 2].map((k) => !!((route >> k) & 1)),
    cutoff: phys(L[I.cutoff], at(I.cutoff)), reso: at(I.reso), mode: Math.round(at(I.mode) * 3),
    f_env: phys(L[I.f_env], at(I.f_env)), f_decay: phys(L[I.f_decay], at(I.f_decay)), pwm: at(I.pwm),
    attack_ms: SID_ATTACK_MS, decay_ms: SID_DECAY_MS,
  };
}

// One chip, one note at a time. The three oscillators play the note at their own interval and each has its
// own envelope, as the chip's three voices do. Oscillator 3 can ring-modulate oscillator 1's triangle and
// hard-sync oscillator 1, and it keeps running for that while its own output is off. The oscillators run at
// twice the sample rate (ring, sync and noise are not band-limited) and are folded down by a half-band filter.
class Sid {
  constructor(sr) {
    this.sr = sr; this.rate = 2 * sr;
    this.ph = new Float64Array(3); this.dt = new Float64Array(3); this.env = new Float64Array(3); this.stage = new Int8Array(3).fill(2);
    this.wave = new Int8Array(3); this.tune = new Float64Array(3); this.pw = new Float64Array(3); this.duty = new Float64Array(3).fill(0.5);
    this.aInc = new Float64Array(3); this.dK = new Float64Array(3); this.sus = new Float64Array(3); this.rK = new Float64Array(3);
    this.lfsr = new Int32Array(3).fill(0x7ffff8); this.nz = new Float64Array(3); this.nstep = new Int32Array(3);
    this.f = new SVF(); this.z = new Float64Array(7);
    this.on = false; this.gate = false; this.gateLeft = 0; this.pitch = 60; this.target = 60; this.vel = 0; this.amp = 0;
    this.fenv = 0; this.t = 0; this.ctl = 0; this.dc = 0;
    this.kGlide = 1 - Math.exp(-8 / (0.03 * sr)); this.kDc = 1 - Math.exp((-TAU * 12) / sr); this.kAmp = 1 - Math.exp(-1 / (0.003 * sr));
    this.set(PARAMS.sid.map((p) => p.def));
  }
  set(u) {
    const L = PARAMS.sid, I = INDEX.sid, ms = 1000 / this.sr;
    for (let k = 0; k < 3; k++) {
      const [w, t, f, p, a, d, s, r] = SID_AT[k];
      this.wave[k] = Math.round(clamp(u[w], 0, 1) * 4);
      this.tune[k] = Math.round(clamp(u[t], 0, 1) * 24) - 12 + phys(L[f], u[f]) / 100;
      this.pw[k] = phys(L[p], u[p]);
      this.aInc[k] = ms / SID_ATTACK_MS[nibble(u[a])]; this.dK[k] = ms / SID_DECAY_MS[nibble(u[d])];
      this.sus[k] = nibble(u[s]) / 15; this.rK[k] = ms / SID_DECAY_MS[nibble(u[r])];
    }
    const fx = Math.round(clamp(u[I.fx], 0, 1) * 3); this.ring = fx === 1 || fx === 3; this.sync = fx >= 2;
    this.route = Math.round(clamp(u[I.route], 0, 1) * 7); this.cut = phys(L[I.cutoff], u[I.cutoff]);
    this.q = 0.707 * Math.pow(8.5, clamp(u[I.reso], 0, 1)); this.mode = Math.round(clamp(u[I.mode], 0, 1) * 3);
    this.fAmt = phys(L[I.f_env], u[I.f_env]); this.kF = Math.exp(-8 / (phys(L[I.f_decay], u[I.f_decay]) * this.sr));
    this.pwm = clamp(u[I.pwm], 0, 1); this.ctl = 0;
  }
  // slide: the step before asked for a slide; while the gate is open the pitch glides and nothing restarts
  note(midi, vel, lenSamples, slide) {
    const legato = !!slide && this.gate;
    if (!this.on) { this.f.reset(); this.z.fill(0); this.dc = 0; this.amp = vel; }
    this.target = midi; this.vel = vel;
    if (!legato) { this.pitch = midi; this.stage.fill(0); this.fenv = 1; this.t = 0; }
    this.gate = true; this.gateLeft = Math.max(8, lenSamples); this.on = true; this.ctl = 0;
  }
  active() { return this.on; }
  tick() {
    if (this.gate && --this.gateLeft <= 0) { this.gate = false; this.stage.fill(2); }
    let alive = this.gate;
    for (let k = 0; k < 3; k++) {
      let e = this.env[k]; const st = this.stage[k];
      if (st === 0) { e += this.aInc[k]; if (e >= 1) { e = 1; this.stage[k] = 1; } }
      else if (st === 1) { if (e > this.sus[k]) { e -= Math.max(0.05, 3 * e) * this.dK[k]; if (e < this.sus[k]) e = this.sus[k]; } }
      else if (e > 0) { e -= Math.max(0.05, 3 * e) * this.rK[k]; if (e < 0) e = 0; }
      this.env[k] = e; if (e > 0) alive = true;
    }
    if (!alive) { this.on = false; return 0; }
    if (this.ctl-- <= 0) {
      this.ctl = 7;
      this.pitch += (this.target - this.pitch) * this.kGlide;
      for (let k = 0; k < 3; k++) this.dt[k] = Math.min(mtof(this.pitch + this.tune[k]) / this.rate, 0.45);
      this.f.set(clamp(this.cut * Math.pow(2, this.fAmt * this.fenv), 20, 16000), this.q, this.rate);
      const sweep = this.pwm > 0 ? 0.4 * this.pwm * Math.sin((TAU * 1.5 * this.t) / this.sr) : 0;
      for (let k = 0; k < 3; k++) this.duty[k] = clamp(this.pw[k] + sweep, 0.03, 0.97);
      this.fenv *= this.kF;
    }
    this.t++;
    const z = this.z, ph = this.ph, dts = this.dt;
    for (let h = 0; h < 2; h++) {
      let dry = 0, wet = 0, wrapped = false, since = 0;
      for (let k = 2; k >= 0; k--) {            // oscillator 3 first: it modulates oscillator 1
        const dt = dts[k]; let p = ph[k] + dt;
        if (p >= 1) { p -= 1; if (k === 2) { wrapped = true; since = p / dt; } }
        if (k === 0 && this.sync && wrapped) p = since * dt;
        ph[k] = p;
        const w = this.wave[k]; if (w === 0) continue;
        let s;
        if (w === 1) {
          let up = p < 0.5; if (k === 0 && this.ring && ph[2] >= 0.5) up = !up;
          const l = p < 0.5 ? 2 * p : 2 * p - 1; s = up ? 2 * l - 1 : 1 - 2 * l;
        } else if (w === 2) s = 2 * p - 1 - polyblep(p, dt);
        else if (w === 3) {
          const d = this.duty[k]; let p2 = p - d; if (p2 < 0) p2 += 1;
          s = (p < d ? 1 : -1) + polyblep(p, dt) - polyblep(p2, dt);
        } else {
          const step = (p * 16) | 0;
          if (step !== this.nstep[k]) {
            this.nstep[k] = step;
            const l = this.lfsr[k], x = (((l << 1) | (((l >> 22) ^ (l >> 17)) & 1)) & 0x7fffff); this.lfsr[k] = x;
            this.nz[k] = ((((x >> 22) & 1) << 7) | (((x >> 20) & 1) << 6) | (((x >> 16) & 1) << 5) | (((x >> 13) & 1) << 4)
              | (((x >> 11) & 1) << 3) | (((x >> 7) & 1) << 2) | (((x >> 4) & 1) << 1) | ((x >> 2) & 1)) / 127.5 - 1;
          }
          s = this.nz[k];
        }
        s *= this.env[k];
        if ((this.route >> k) & 1) wet += s; else dry += s;
      }
      let out = dry;
      if (this.route) {
        this.f.tick(wet);
        out += this.mode === 0 ? this.f.lp : this.mode === 1 ? this.f.bp : this.mode === 2 ? this.f.hp : this.f.lp + this.f.hp;
      }
      z[0] = z[1]; z[1] = z[2]; z[2] = z[3]; z[3] = z[4]; z[4] = z[5]; z[5] = z[6]; z[6] = out;
    }
    let y = (-z[0] + 9 * z[2] + 16 * z[3] + 9 * z[4] - z[6]) / 32;
    this.dc += this.kDc * (y - this.dc); y -= this.dc;
    this.amp += (this.vel - this.amp) * this.kAmp;
    return y * 0.4 * this.amp;
  }
  reset() { this.on = false; this.gate = false; this.env.fill(0); this.stage.fill(2); this.f.reset(); this.z.fill(0); this.dc = 0; this.amp = 0; }
}

// ---------------------------------------------------------------- effects

class Freeverb {
  constructor(sr) {
    const sc = sr / 44100, combs = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], aps = [556, 441, 341, 225], spread = 23;
    this.cL = combs.map((n) => ({ b: new Float32Array(Math.round(n * sc)), i: 0, f: 0 }));
    this.cR = combs.map((n) => ({ b: new Float32Array(Math.round((n + spread) * sc)), i: 0, f: 0 }));
    this.aL = aps.map((n) => ({ b: new Float32Array(Math.round(n * sc)), i: 0 }));
    this.aR = aps.map((n) => ({ b: new Float32Array(Math.round((n + spread) * sc)), i: 0 }));
    this.fb = 0.86; this.damp = 0.25; this.oL = 0; this.oR = 0;
  }
  set(size, damp) { this.fb = 0.7 + 0.28 * clamp(size, 0, 1); this.damp = 0.4 * clamp(damp, 0, 1); }
  tick(inL, inR) {
    const x = (inL + inR) * 0.015 + DC; let l = 0, r = 0; const fb = this.fb, d = this.damp;
    for (let c = 0; c < 8; c++) {
      let q = this.cL[c]; let y = q.b[q.i]; q.f = y * (1 - d) + q.f * d; q.b[q.i] = x + q.f * fb; if (++q.i >= q.b.length) q.i = 0; l += y;
      q = this.cR[c]; y = q.b[q.i]; q.f = y * (1 - d) + q.f * d; q.b[q.i] = x + q.f * fb; if (++q.i >= q.b.length) q.i = 0; r += y;
    }
    for (let a = 0; a < 4; a++) {
      let q = this.aL[a]; let bo = q.b[q.i]; q.b[q.i] = l + bo * 0.5; if (++q.i >= q.b.length) q.i = 0; l = bo - l;
      q = this.aR[a]; bo = q.b[q.i]; q.b[q.i] = r + bo * 0.5; if (++q.i >= q.b.length) q.i = 0; r = bo - r;
    }
    this.oL = l * 3; this.oR = r * 3;
  }
}

class PingDelay {
  constructor(sr) { this.n = Math.round(sr * 3); this.bL = new Float32Array(this.n); this.bR = new Float32Array(this.n); this.i = 0; this.len = sr * 0.3; this.fb = 0.45; this.tone = 0.45; this.lpL = 0; this.lpR = 0; this.hpL = 0; this.hpR = 0; this.oL = 0; this.oR = 0; }
  tick(x) {
    const n = this.n, len = clamp(this.len, 64, n - 2);
    let rp = this.i - len; if (rp < 0) rp += n;
    const i0 = Math.floor(rp), fr = rp - i0, i1 = i0 + 1 >= n ? 0 : i0 + 1;
    let dl = this.bL[i0] * (1 - fr) + this.bL[i1] * fr, dr = this.bR[i0] * (1 - fr) + this.bR[i1] * fr;
    this.lpL += this.tone * (dl - this.lpL); this.lpR += this.tone * (dr - this.lpR);
    this.hpL += 0.03 * (this.lpL - this.hpL); this.hpR += 0.03 * (this.lpR - this.hpR);
    dl = this.lpL - this.hpL; dr = this.lpR - this.hpR;
    this.bL[this.i] = x + dr * this.fb + DC; this.bR[this.i] = dl * this.fb;
    if (++this.i >= n) this.i = 0;
    this.oL = dl; this.oR = dr;
  }
}

// Mono PCM is embedded in the pool so offline rendering and AudioWorklets
// get the exact same recordings. AudioWorkletGlobalScope does not expose atob.
const BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const SAMPLE_BANKS = new WeakMap();
function sampleBank(pool) {
  if (SAMPLE_BANKS.has(pool)) return SAMPLE_BANKS.get(pool);
  const bank = new Map();
  for (const [id, spec] of Object.entries(pool.sample_bank || {})) {
    const { pcm16, rate, root } = spec;
    if (typeof pcm16 !== 'string' || !Number.isInteger(rate) || rate < 8000 || rate > 48000 ||
        !Number.isInteger(root) || root < 0 || root > 127 || pcm16.length > rate * 8) continue;
    const binary = new Uint8Array(Math.floor(pcm16.length * 3 / 4));
    let bits = 0, value = 0, count = 0;
    for (const ch of pcm16) {
      if (ch === '=') break;
      const digit = BASE64.indexOf(ch);
      if (digit < 0) { count = 0; break; }
      value = (value << 6) | digit;
      bits += 6;
      if (bits >= 8) { bits -= 8; binary[count++] = (value >> bits) & 255; }
    }
    if (count < 4 || count % 2 !== 0) continue;
    const audio = new Float32Array(count / 2);
    const view = new DataView(binary.buffer);
    for (let i = 0; i < audio.length; i++) audio[i] = view.getInt16(2*i, true) / 32768;
    bank.set(id, { audio, rate, root });
  }
  SAMPLE_BANKS.set(pool, bank);
  return bank;
}

class SamplerVoice {
  constructor(sr) {
    this.sr = sr;
    this.f = new SVF();
    this.reset();
  }
  reset() { this.audio = null; }
  note(sample, midi, velocity, duration, u) {
    const I = INDEX.sampler, P = PARAMS.sampler;
    this.audio = sample.audio;
    const reverse = u[I.reverse] >= 0.5;
    const max = this.audio.length - 2;
    const start = phys(P[I.start], u[I.start]);
    this.pos = max * (reverse ? 1 - start : start);
    const semitones = clamp(midi - sample.root + phys(P[I.tune], u[I.tune]), -24, 24);
    this.velocity = velocity;
    this.increment = (reverse ? -1 : 1) * sample.rate / this.sr * Math.pow(2, semitones / 12);
    this.age = 0;
    this.gate = Math.max(1, duration);
    this.fade = Math.max(1, Math.round(this.sr * 0.003));
    this.release = Math.max(1, Math.round(this.sr * 0.012));
    this.decay = phys(P[I.decay], u[I.decay]) * this.sr;
    this.f.set(phys(P[I.cutoff], u[I.cutoff]), phys(P[I.reso], u[I.reso]), this.sr);
    this.f.reset();
  }
  tick() {
    const audio = this.audio;
    if (!audio) return 0;
    const i = Math.floor(this.pos);
    if (i < 0 || i + 1 >= audio.length || this.age >= this.gate + this.release) { this.reset(); return 0; }
    const sample = audio[i] + (audio[i+1] - audio[i]) * (this.pos - i);
    const endDistance = this.increment < 0 ? this.pos : audio.length - 1 - this.pos;
    const env = Math.min(1, this.age / this.fade, endDistance / (Math.abs(this.increment)*this.fade),
                         (this.gate + this.release - this.age) / this.release);
    const level = env * this.velocity * Math.exp(-this.age / this.decay);
    this.age++;
    this.pos += this.increment;
    return this.f.tick(sample * level);
  }
}

class Sampler {
  constructor(sr) { this.voices = Array.from({length: 3}, () => new SamplerVoice(sr)); this.next = 0; this.bank = new Map(); }
  note(id, midi, velocity, duration, controls) {
    const sample = this.bank.get(id);
    if (!sample) return;
    const voice = this.voices.find(v => !v.audio) || this.voices[this.next];
    this.next = (this.next + 1) % this.voices.length;
    voice.note(sample, midi, velocity, duration, controls);
  }
  tick() { return this.voices[0].tick() + this.voices[1].tick() + this.voices[2].tick(); }
  reset() { for (const voice of this.voices) voice.reset(); }
}

// ---------------------------------------------------------------- layers

class Layer {
  constructor(sr) {
    this.sr = sr; this.kit = new DrumKit(sr); this.acid = new Acid(sr); this.synth = new Synth(sr); this.sampler = new Sampler(sr); this.sid = new Sid(sr);
    this.tok = null; this.u = null; this.from = null; this.to = null; this.gain = 0; this.target = 0;
    this.kFade = kOf(0.004, sr); this.level = 0; this.pl = 0.707; this.pr = 0.707; this.sd = 0; this.sv = 0; this.prevSlide = false; this.holdFilter = false;
    this.mFrom = [0, 0]; this.mTo = [0, 0]; this.mod = [0, 0]; this.um = null; this.tilt = new SVF(); this.tiltMode = 0; this.kitOn = false; this.synthOn = false;
    this.dyn = null; this.hatMode = 'flat'; this.dynPattern = null; this.dynRead = null; this.dynTrim = null; this.kx = 0;
  }
  barStart(tok, mod) {
    this.mFrom = this.mod.slice(); this.mTo = [clamp(+(mod && mod[0]) || 0, -1, 1), clamp(+(mod && mod[1]) || 0, -1, 1)];
    if (!tok) { this.target = 0; this.tok = null; this.prevSlide = false; return; }
    const same = this.tok && this.tok.voice === tok.voice && this.tok.device === tok.device;
    if (this.tok?.voice === 'sampler' && tok.voice !== 'sampler') this.sampler.reset();
    if (same) { this.from = Float64Array.from(this.u); this.to = tok.u; }
    else { this.u = Float64Array.from(tok.u); this.from = tok.u; this.to = tok.u; this.prevSlide = false; }
    this.tok = tok; this.target = 1; this.gain = 1;
    this.apply();
  }
  control(frac) {
    this.mod[0] = this.mFrom[0] + (this.mTo[0] - this.mFrom[0]) * frac;
    this.mod[1] = this.mFrom[1] + (this.mTo[1] - this.mFrom[1]) * frac;
    if (!this.tok) return;
    const u = this.u, a = this.from, b = this.to;
    // knobs glide across the bar; a register with a fixed number of settings (a waveform, a rate nibble) switches at its start
    if (a !== b) { const L = PARAMS[this.tok.voice]; for (let i = 0; i < u.length; i++) u[i] = L[i].q ? b[i] : a[i] + (b[i] - a[i]) * frac; }
    this.apply();
  }
  apply() {
    const v = this.tok.voice, L = PARAMS[v], I = INDEX[v], f = this.mod[0], sp = this.mod[1];
    let u = this.u;
    if (f !== 0 && v !== 'drums' && !(this.holdFilter && v === 'synth')) {
      if (!this.um || this.um.length !== u.length) this.um = new Float64Array(u.length);
      this.um.set(u); u = this.um;
      u[I.cutoff] = clamp(u[I.cutoff] + MOD_RANGE.filter.cutoff_offset * f, 0, 1);
      if (v === 'acid') u[I.envmod] = clamp(u[I.envmod] + MOD_RANGE.filter.envmod_offset * f, 0, 1);
    }
    if (v === 'drums') this.kit.set(u);
    else if (v === 'acid') this.acid.set(u);
    else if (v === 'synth') this.synth.set(u);
    else if (v === 'sid') this.sid.set(u);
    else this.sampleControls = u;
    this.level = phys(L[I.level], u[I.level]) * (this.tok.gain || 1);
    const a = ((u[I.pan] * 2 - 1) + 1) * Math.PI / 4; this.pl = Math.cos(a); this.pr = Math.sin(a);
    const so = MOD_RANGE.space.send_offset * sp;
    this.sd = clamp(u[I.send_delay] + so, 0, 1); this.sv = clamp(u[I.send_reverb] + so, 0, 1);
    // drums: FILTER tilts the drum bus (lowpass below zero, highpass above)
    if (v === 'drums' && Math.abs(f) > 0.005) {
      const F = MOD_RANGE.filter;
      if (f < 0) { this.tiltMode = -1; this.tilt.set(F.drum_lowpass_hz[0] * Math.pow(F.drum_lowpass_hz[1] / F.drum_lowpass_hz[0], -f), 0.7, this.sr); }
      else { this.tiltMode = 1; this.tilt.set(F.drum_highpass_hz[0] * Math.pow(F.drum_highpass_hz[1] / F.drum_highpass_hz[0], f), 0.7, this.sr); }
    } else this.tiltMode = 0;
  }
  // What a bar's drum pattern is, for the drum dynamics: the one hat sound a hat line is written on (null when the
  // pattern has none or uses several), whether that line has accents, and which snare, clap or rim rows are a
  // texture (more than 8 hits in the bar).
  drumRead(p) {
    const H = p.hits, hats = HAT_VOICES.filter((v) => H[v] && H[v].some((x) => x > 0));
    const R = { line: hats.length === 1 ? hats[0] : null, accents: false, texture: {}, four: {} };
    if (R.line) R.accents = H[R.line].some((x) => x >= 0.8);
    for (const v of SNARE_VOICES) { let n = 0; if (H[v]) for (const x of H[v]) if (x > 0) n++; R.texture[v] = n > 8; }
    R.four = {};                                    // a snare row with accents on all four beats
    for (const v of SNARE_VOICES) R.four[v] = !!(H[v] && H[v][0] >= 0.8 && H[v][4] >= 0.8 && H[v][8] >= 0.8 && H[v][12] >= 0.8);
    return R;
  }
  // One sixteenth of the drums under the drum dynamics (Engine.setBalance, more.drums). The kick plays as written.
  // What the drum dynamics know of a kit, measured once on a scratch kit: the peak of an accent of each of its
  // sounds, also of a sound played at another sound's level (a hat line's plain hits on the closed hat), and from
  // these one fixed gain per snare and hat sound that brings its accent toward a set loudness against the kit's
  // kick (D.level; pulled D.pull of the way, at most D.reach either way).
  kitTrim(tok) {
    const D = this.dyn;
    let T = D.kits.get(tok.u); if (T && T.bank === this.kit.bank) return T;
    T = { bank: this.kit.bank, device: tok.device, u: tok.u, peaks: {}, gains: {} };
    D.kits.set(tok.u, T);
    return T;
  }
  kitPeak(T, name, as) {
    const key = as && as !== name ? `${name}>${as}` : name;
    if (!(key in T.peaks)) {
      const D = this.dyn; if (!D.scratch) D.scratch = new DrumKit(this.sr);
      D.scratch.bank = this.kit.bank; D.scratch.bankRate = this.kit.bankRate;
      T.peaks[key] = D.scratch.peakOf(T.device, name, T.u, as && as !== name ? as : null);
    }
    return T.peaks[key];
  }
  kitLoud(T, name) {
    const key = `~${name}`;
    if (!(key in T.peaks)) {
      const D = this.dyn; if (!D.scratch) D.scratch = new DrumKit(this.sr);
      D.scratch.bank = this.kit.bank; D.scratch.bankRate = this.kit.bankRate;
      T.peaks[key] = D.scratch.levelOf(T.device, name, T.u).loud;
    }
    return T.peaks[key];
  }
  // The sound that plays for a written snare or hat sound of a kit: itself, unless the kit's own is silent or too
  // quiet to reach its level within D.reach; then the loudest sound of its class in that kit (a kit fitted to a
  // recording without hats has hat sounds near zero, and a hat line written on them would not be heard).
  kitVoice(T, name) {
    if (!T.stand) T.stand = {};
    if (!(name in T.stand)) {
      const D = this.dyn, cls = SNARE_VOICES.includes(name) ? SNARE_VOICES : HAT_VOICES, level = cls === SNARE_VOICES ? D.level.snare : D.level.hats;
      const kick = this.kitLoud(T, 'kick'), own = this.kitLoud(T, name);
      let pick = name;
      if (D.pull > 0 && level != null && kick > 1e-5 && !(own > 1e-6 && level * kick / own <= D.reach)) {
        let best = name, loud = own;
        for (const v of cls) { const l = this.kitLoud(T, v); if (l > loud) { best = v; loud = l; } }
        if (loud > 2 * own) pick = best;
      }
      T.stand[name] = pick;
    }
    return T.stand[name];
  }
  kitGain(T, name) {
    if (!(name in T.gains)) {
      const D = this.dyn, level = SNARE_VOICES.includes(name) ? D.level.snare : HAT_VOICES.includes(name) ? D.level.hats : null;
      const kick = this.kitLoud(T, 'kick'), own = this.kitLoud(T, name);
      T.gains[name] = D.pull > 0 && level != null && kick > 1e-5 && own > 1e-6 ? clamp(Math.pow(level * kick / own, D.pull), 1 / D.reach, D.reach) : 1;
    }
    return T.gains[name];
  }
  // The kick boost's level gain for a kick of velocity v and gain g on kit T: the hit as the voice levels hear it (a
  // plain hit on a scratch kit through the drum EQ, DrumKit.heardOf, scaled to this hit, the layer's level and the kit
  // trim) against the kit's level as heard (Engine.bal.vc.kitw, the level the synth voices are set against; without
  // the voice levels the kit's plain level), both read without the boost's own part, so the mix does not follow the
  // boost. Under the target the kick is raised toward it (K.rate of the way per hit), at most K.max; over it the kick
  // plays as it is. A kit keeps its gain, so a kit that returns starts where it left.
  kickGain(T, v, g) {
    const K = this.dyn.kick, B = this.dyn.bal, V = B && B.vc;
    if (!('~kickH' in T.peaks)) {
      const D = this.dyn; if (!D.scratch) D.scratch = new DrumKit(this.sr);
      D.scratch.bank = this.kit.bank; D.scratch.bankRate = this.kit.bankRate;
      const eq = this.eq ? this.eq.map((p) => Object.assign(Object.create(Peak.prototype), p, { x1: 0, x2: 0, y1: 0, y2: 0 })) : null;
      T.peaks['~kickH'] = D.scratch.heardOf(T.device, 'kick', T.u, 0.8, 0.15, eq);
      const L8 = D.scratch.levelOf(T.device, 'kick', T.u, null, 0.15, 0.8);
      T.peaks['~kick.8'] = L8.loud; T.peaks['^kick.8'] = L8.peak;
    }
    if (T.kickGain == null) T.kickGain = 1;
    const hit = (v > 0.8 ? 1.15 : 0.35 + 0.65 * v / 0.8) * g * this.level;
    const heard = V ? T.peaks['~kickH'] * hit * B.trim[0] : T.peaks['~kick.8'] * hit;
    const kit = V ? (V.kitw > 1e-12 ? Math.sqrt(V.kitw) : 0) : B && B.ms[0] > 1e-9 ? Math.sqrt(B.ms[0]) : 0;
    if (kit > 0 && heard > 1e-9) {
      const want = clamp(K.target * kit / heard, 1, K.max);
      T.kickGain += (want - T.kickGain) * K.rate;
    }
    // the ceiling of this hit: its own peak as written, times the level gain, times K.ceil
    this.kit.kc = T.peaks['^kick.8'] > 1e-6 ? T.peaks['^kick.8'] * hit / this.level * T.kickGain * K.ceil : Infinity;
    return T.kickGain;
  }
  drumStep(tok, p, s) {
    const D = this.dyn;
    if (this.dynPattern !== p) { this.dynPattern = p; this.dynRead = this.drumRead(p); this.dynTrim = this.kitTrim(tok); }
    const T = this.dynTrim;
    const R = this.dynRead, M = D.modes[this.hatMode] || D.modes.flat, at = s & 3;
    for (const name of DRUM_VOICES) {
      const h = p.hits[name]; if (!h || !(h[s] > 0)) continue;
      const vel = h[s]; let g = 1, as = null, v = vel;
      if (name === R.line) {
        let accent = vel >= 0.8, eased = 1;
        if (M.lift && at === 2) accent = true;                            // the off-beat eighth carries the line
        else if (M.lift && accent && at === 0 && R.accents) eased = D.onBeat;   // an accent on the beat is eased
        // a hat line written on the cymbal plays on the hats (D.cymToHats): accents open, plain hits closed
        const onCym = name === 'cym' && D.cymToHats;
        if (accent) { g = D.accent * eased * D.trim.hats; v = 1; if ((M.split && name === 'ch' && at === 2) || onCym) as = 'oh'; }
        else {
          const w = M.w[at]; if (w <= 0) continue;
          g = (vel <= 0.3 ? D.ghost : D.plain) * w * D.trim.hats; v = 0.8;
          if (M.lift && at === 0 && p.hits.kick && p.hits.kick[s] > 0) g *= D.hatOnKick;     // under a kick the tick steps back further
          if ((M.split && name !== 'ch') || onCym) as = 'ch';
        }
      } else if (HAT_VOICES.includes(name)) g = D.trim.hats;
      else if (SNARE_VOICES.includes(name)) {
        g = D.trim.snare;
        if (R.four[name] && vel >= 0.8 && (s === 0 || s === 8)) g *= D.kickBeats;           // accents on all four beats: beats 1 and 3 step back, the backbeat stays
        else if (R.texture[name]) { if (vel < 0.8) g *= D.texture * (at === 2 ? 1.4 : 1); else if (s !== 4 && s !== 12) g *= D.offBackbeat; }
      } else if (name === 'kick') { g = D.trim.kick; if (D.kick) this.kit.kg = this.kickGain(T, v, g); }
      let play = name;
      if (name !== 'kick' && (HAT_VOICES.includes(name) || SNARE_VOICES.includes(name))) {
        play = this.kitVoice(T, name);
        g *= this.kitGain(T, play);
        if (as === play) as = null;
        if (as) {                                   // another sound at this one's level: brought to this one's peak
          const own = this.kitPeak(T, play), other = this.kitPeak(T, play, as);
          if (other > 1e-5 && own > 1e-5) g *= clamp(own / other, 0.05, 20); else as = null;
        }
      }
      this.kit.hit(tok.device, play, v, this.u, g, as); this.kitOn = true;
    }
  }
  step(s, stepLen) {
    this.kit.setBeat(4 * stepLen);
    const tok = this.tok; if (!tok) return;
    const p = tok.pattern;
    if (tok.voice === 'drums') {
      if (this.dyn && this.dyn.on) this.drumStep(tok, p, s);
      else for (const name of DRUM_VOICES) { const h = p.hits[name]; if (h && h[s] > 0) { this.kit.hit(tok.device, name, h[s], this.u); this.kitOn = true; } }
    } else if (tok.voice === 'acid') {
      const n = p.notes[s];
      if (n != null) {
        const holds = p.slide[s] && p.notes[(s + 1) % 16] != null;
        const gs = holds ? Math.round(stepLen * 1.6) : Math.max(8, Math.round(stepLen * this.acid.gateShare));
        this.acid.note(n, p.accent[s], this.prevSlide, gs);
      }
      this.prevSlide = n != null && !!p.slide[s];
    } else {
      const n = p.notes[s];
      if (n != null && p.vel[s] > 0) {
        const duration = Math.round(p.len[s] * stepLen);
        if (tok.voice === 'sampler') this.sampler.note(tok.sample, n, p.vel[s], duration, this.sampleControls);
        else if (tok.voice === 'sid') this.sid.note(n, p.vel[s], duration, this.prevSlide);
        else { this.synth.note(n, p.vel[s], duration); this.synthOn = true; }
      }
      if (tok.voice === 'sid') this.prevSlide = n != null && !!p.slide[s];
    }
  }
  tick() {
    if (this.target === 0) { if (this.gain === 0) return 0; this.gain *= this.kFade; if (this.gain < 1e-4) { this.gain = 0; this.sampler.reset(); return 0; } }
    let s = 0;
    this.kx = 0;
    // kitOn / synthOn: a layer that never hit a drum or played the synth does not walk their voices every sample
    if (this.kitOn) {
      if (this.kit.active()) {
        let k = this.kit.tick();
        if (this.kit.kb !== null) this.kx = this.kit.kx * this.gain * this.level;      // the kick boost's part, before the drum EQ
        if (this.eq) for (const p of this.eq) k = p.run(k);
        if (this.tiltMode !== 0) { this.tilt.tick(k); k = this.tiltMode < 0 ? this.tilt.lp : this.tilt.hp * 1.15; }
        s += k;
      } else this.kitOn = false;
    }
    if (this.acid.on) s += this.acid.tick();
    if (this.synthOn) { if (this.synth.active()) s += this.synth.tick(); else this.synthOn = false; }
    if (this.sid.on) s += this.sid.tick();
    if (this.sampler.bank.size) s += this.sampler.tick();
    if (!Number.isFinite(s)) { this.kit.reset(); this.acid.reset(); this.synth.reset(); this.sampler.reset(); this.sid.reset(); return 0; }
    return s * this.gain * this.level;
  }
}

// ---------------------------------------------------------------- engine

export class Engine {
  // opts: {fx: true, master: true} (the fitter renders with both off)
  constructor(sampleRate, pool, opts = {}) {
    this.sr = sampleRate; this.opts = Object.assign({ fx: true, master: true }, opts);
    this.layers = [0, 1, 2, 3].map(() => new Layer(this.sr));
    this.queue = []; this.cbs = []; this.stepCbs = []; this.frame = 0; this.nextStep = 0; this.stepIdx = -1; this.bar = -1;
    this.bpm = 126; this.tempoSet = false; this.stepLen = (this.sr * 15) / this.bpm; this.barStartFrame = 0; this.barLen = 16 * this.stepLen;
    this.playing = [0, 0, 0, 0]; this.mods = [[0, 0], [0, 0], [0, 0], [0, 0]]; this.ctl = 0; this.solo = -1; this.log = null; this.dyn = null; this.dynBar = -2;
    this.dly = this.opts.fx ? new PingDelay(this.sr) : null; this.verb = this.opts.fx ? new Freeverb(this.sr) : null;
    this.space = null;                                  // the base sends of the synth voices (setSpace), off unless asked for
    this.bassGain = 1;                                  // the bass voice's level against the kit (setBass), 1 unless asked for
    this.holdControls = false;                          // a synth patch played as the pool sound nearest to it (balance.holdControls)
    this.cEnv = 0; this.lim = 1; this.cAtt = 1 - Math.exp(-1 / (0.01 * this.sr)); this.cRel = 1 - Math.exp(-1 / (0.15 * this.sr)); this.limRel = 1 - Math.exp(-1 / (0.06 * this.sr));
    // Loudness balance (off unless asked for, so fits and checks render as written). While it is on:
    //   the kit is brought toward one level (kitDb), so every kit and every track is about equally loud;
    //   every synth voice is trimmed toward a set share of the kit's level while it sounds (ratio);
    //   the kick ducks the synth voices a little (duck), which keeps the low end clear;
    //   the whole mix is brought slowly toward one loudness while the kit plays (outDb), within 4 dB, so one
    //   track is about as loud as the next. A break is not pulled up: without the kit the trim holds.
    // A level is the root mean square of a voice over about a third of a second of its sounding samples, the
    // measure the recordings' levels are read with. A trim goes up slowly and comes down fast, and no boosted
    // sample passes the ceiling above its share.
    const tc = (seconds) => 1 - Math.exp(-1 / (seconds * this.sr));
    this.bal = { on: false, ratio: [1, 0.6, 0.18, 0.22], range: 4, ceiling: 4, ms: [0, 0, 0, 0], trim: [1, 1, 1, 1], gain: [1, 1, 1, 1], kit: 0,
                 k: tc(0.35), down: tc(0.08), up: tc(0.8),
                 kitLevel: 0, kitRange: 4, kdown: tc(0.4), kup: tc(3.0),
                 outLevel: 0, oms: 0, otrim: 1, ok: tc(2.0), oup: tc(14), odown: tc(5), held: 0,
                 vc: null, bus: null, eq: null, eqKey: '[]', duck: 0, share: [0, 1, 0.45, 0.6], lp: 0, lpk: tc(1 / (2 * Math.PI * 140)), env: 0, top: 0, eatt: tc(0.004), erel: tc(0.13), tdecay: Math.exp(-1 / (2.5 * this.sr)) };
    this.setPool(pool || { layers: [] });
  }
  // on: whether the balance works; ratio: [bass, lead, air] as shares of the kit's level (root mean square);
  // rangeDb: how far a trim may move a voice, in dB either way; more: {kitDb: the kit's level in dB (null: as the
  // kit is), kitRangeDb, duck: how far the kick pushes the bass down, 0..0.6 (lead and air less), ceiling}
  setBalance(on, ratio, rangeDb, more) {
    const B = this.bal, m = more || {};
    B.on = !!on;
    if (Array.isArray(ratio) && ratio.length === 3) B.ratio = [1, ...ratio.map((r) => clamp(+r || 0.5, 0.02, 2))];
    if (Number.isFinite(+rangeDb) && +rangeDb > 0) B.range = Math.pow(10, clamp(+rangeDb, 1, 36) / 20);
    B.kitLevel = Number.isFinite(+m.kitDb) && m.kitDb != null ? Math.pow(10, clamp(+m.kitDb, -40, -3) / 20) : 0;
    if (Number.isFinite(+m.kitRangeDb) && +m.kitRangeDb > 0) B.kitRange = Math.pow(10, clamp(+m.kitRangeDb, 1, 30) / 20);
    B.duck = clamp(+m.duck || 0, 0, 0.6);
    B.outLevel = Number.isFinite(+m.outDb) && m.outDb != null ? Math.pow(10, clamp(+m.outDb, -40, -6) / 20) : 0;
    if (Number.isFinite(+m.ceiling) && +m.ceiling >= 1.5) B.ceiling = +m.ceiling;
    if (!B.on) { B.trim = [1, 1, 1, 1]; B.gain = [1, 1, 1, 1]; B.otrim = 1; }
    if ('voice' in m) this.setVoiceLevels(B.on ? m.voice : null);
    if ('bus' in m) this.setBus(B.on ? m.bus : null);
    if ('eq' in m) {     // eq: [[hz, dB, q], ...] (at most four bell filters on the whole mix, before the master's compressor)
      const bands = B.on && Array.isArray(m.eq) ? m.eq.filter((b) => Array.isArray(b) && +b[0] > 10 && Math.abs(+b[1]) > 0.05).slice(0, 4) : [];
      const key = JSON.stringify(bands);
      if (key !== B.eqKey) { B.eqKey = key; B.eq = bands.length ? [0, 1].map(() => bands.map(([f, g, q]) => new Peak(this.sr, +f, clamp(+g, -12, 12), +q || 0.7))) : null; }
    }
    if ('drums' in m) this.setDrumDynamics(B.on ? m.drums : null);   // the level button switches both, for comparing by ear
    if ('space' in m) this.setSpace(B.on ? m.space : null);
    if ('bass' in m) this.setBass(B.on ? m.bass : null);
    if ('holdFilter' in m) for (const l of this.layers) l.holdFilter = B.on && m.holdFilter === true;
    if ('holdControls' in m) this.holdControls = B.on && m.holdControls === true;
  }
  // THE SPACE (11-26, balance.space): the pool's sounds were fitted dry (send_delay and send_reverb 0 in every sound) while
  // the recordings are wet, so the mix adds a base send per synth voice: delay [bass, lead, air] and reverb [bass, lead,
  // air], each 0..1, added to the sound's own send and the bar's space control (both unchanged; the sum held at 1). While it
  // is on, everything sent to the delay and the reverb passes a high-pass at hpHz (default 200 Hz) first, so neither
  // repeats or smears the low end. The drums send nothing here. The defaults are the setting measured against the
  // recordings (engine/space_calibrate.mjs). holdFilter (balance.holdFilter, separate): the synth voice ignores the bar's
  // filter control and plays its patch's own cutoff (its LFOs move it), so a filter the score left open does not jump.
  // THE BASS (11-26, balance.bass {on, gainDb}): the bass voice a fixed gainDb louder than the voice levels set it (after
  // them, so they do not take it back), its sound as fitted (no shelf: each fitted sound carries its recording's sub and low
  // bands, engine/bass_bands.mjs). The default is the gap measured between the recordings' bass against their drums and the
  // renders' (engine/bass_mix.mjs against label_benchmark/results/bass_recordings.json).
  // THE CONTROL HOLD (11-27, balance.holdControls): a score that leaves a synth voice's sound controls open (controls_open,
  // the sound matched by v4's unscored rule: 91 to 98% of the sounding synth bars of v5f) teaches only its voice and register;
  // every other control of the block (gain step, the patch, the four metrical velocities) and the bar's filter and space
  // controls are taken out of the lesson (brain/common.py control_slots), so the brain's values for them are never taught.
  // While this is on, a bar's synth-voice patch plays as the pool sound of its layer nearest to it (the smallest summed
  // distance over the patch's controls): that sound's own patch, gain and metrical velocities; the notes and their register
  // stay the brain's, and the bar's space control is 0 (the filter control is held by holdFilter). The acid voice, whose
  // controls the scored acid reader teaches, and the other voices play as written.
  nearestSound(i, patch) {
    const cache = this.near || (this.near = new Map()), key = i + JSON.stringify(patch.patch || {});
    if (cache.has(key)) return cache.get(key);
    const toks = (this.sounds && this.sounds[i]) || [], mine = patch.patch || {};
    let best = null, bd = Infinity;
    for (const t of toks) {
      let d = 0;
      for (const k in t.patch) if (typeof t.patch[k] === 'number' && typeof mine[k] === 'number') d += Math.abs(t.patch[k] - mine[k]);
      if (d < bd) { bd = d; best = t; }
    }
    let out = null;
    if (best) {
      const hits = [[], [], [], []], pv = (best.pattern && best.pattern.vel) || [], pn = (best.pattern && best.pattern.notes) || [];
      for (let s = 0; s < 16; s++) if (pn[s] != null && pv[s] > 0) hits[s % 4].push(pv[s]);
      const all = hits.flat(), med = (a) => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : null; };
      const fall = med(all) == null ? 0.85 : med(all);
      out = { patch: { ...best.patch }, gain: best.gain, vel: hits.map((h) => (med(h) == null ? fall : med(h))) };
    }
    if (cache.size > 4096) cache.clear();
    cache.set(key, out);
    return out;
  }
  setBass(b) {
    const g = b && b.on !== false && Number.isFinite(+b.gainDb) ? clamp(+b.gainDb, -12, 12) : 0;
    this.bassGain = Math.pow(10, g / 20);
  }
  setSpace(s) {
    if (!s || s.on === false) { this.space = null; return; }
    const num = (x, dflt, lo, hi) => (Number.isFinite(+x) && x != null ? clamp(+x, lo, hi) : dflt);
    const three = (a) => [0, 1, 2].map((k) => num(Array.isArray(a) ? a[k] : 0, 0, 0, 1));
    const hz = num(s.hpHz, 200, 20, 2000), hp = [new SVF(), new SVF()];
    for (const f of hp) f.set(hz, 0.707, this.sr);
    this.space = { d: [0, ...three(s.delay)], r: [0, ...three(s.reverb)], hp };
  }
  // Drum dynamics (off unless asked for; the kick always plays as written). A beat row is written as none, hit or
  // accent on each sixteenth. While this is on:
  //   a hat line's plain hits sound plainDb under its accents (ghostDb for the ghost notes of a full row);
  //   a hat line written on one sound is played on two: plain hits on the closed hat, accents on the open hat or
  //   cymbal (the line's own long sound);
  //   for each section of `section` bars a hat mode is drawn with the odds in `modes` ({off, push, thin, flat}:
  //   HAT_MODES), from the section's number and the kit, so a track plays the same every time;
  //   in a snare, clap or rim row with more than 8 hits (a texture) the plain hits go down by textureDb and the
  //   accents away from the backbeat by offBackbeatDb;
  //   kickBeatsDb: in a snare, clap or rim row with accents on all four beats the accents on beats 1 and 3 go down
  //   by this much (the backbeat stays); hatOnKickDb: in the lifting hat modes a plain hat on the beat that sounds
  //   with a kick goes down by this much more;
  //   trimDb {kick, snare, hats}: a fixed gain for each drum class;
  //   cymToHats (default on): a hat line written on the cymbal plays on the hats (accents open, plain hits closed);
  //   levelDb {snare, hats}: how loud an accent of a kit's snare and hat sounds should be against its kick
  //   (DrumKit.levelOf: weighted root mean square over 0.15 s); each sound of a kit gets one fixed gain that takes
  //   it levelPull (0..1) of the way there, at most levelReachDb either way (engine/kit_levels.mjs surveys a pool).
  //   kickBoost {on, targetDb, maxDb, rate, ceilDb, subHz, punchDb, attackMs, bodyDb, bodyHz, bodyQ, bodyMs} (off
  //   when missing): a kick hit quieter than targetDb against the kit's level as heard (Layer.kickGain: the hit
  //   through the drum EQ, weighted as the voice levels weigh a sound, over the kit's weighted level, which the synth
  //   voices are set against) is raised toward it, at most maxDb, rate of the way per hit; louder kicks are left as
  //   they are. The gain acts on the kick above subHz: the sub stays as written (it fills the level the whole mix is
  //   held at, outDb, and is heard little). The boost is left out of the kit's and the mix's level readings, so the
  //   synth voices, the kit trim and the output trim do not follow it; the master compressor and the bus hear it, so
  //   the kick's peaks stay where the bus puts them and the rest steps back under it. Every kick also gets its attack
  //   lifted by punchDb, falling back with a time constant of attackMs, and a body: the band around bodyHz (bandpass,
  //   bodyQ) added at bodyDb, falling back with a time constant of bodyMs. A hit's peak bends over ceilDb above its
  //   own peak as written times its level gain. The defaults are the setting measured against the recordings' kicks
  //   (quality/loop/kick_check.py).
  setDrumDynamics(d) {
    if (!d || d.on === false) { this.dyn = null; for (const l of this.layers) { l.dyn = null; l.kit.setKickBoost(null); l.kit.kg = 1; l.kit.kc = Infinity; } return; }
    const num = (x, dflt, lo, hi) => (Number.isFinite(+x) && x != null ? clamp(+x, lo, hi) : dflt), t = d.trimDb || {}, lv = d.levelDb || {};
    const odds = Object.assign({}, HAT_ODDS, d.modes || {}), modes = {};
    for (const [k, m] of Object.entries(HAT_MODES)) modes[k] = { w: m.w.map((x) => (x == null ? 0 : fromDb(x))), lift: m.lift, split: m.split && d.split !== false };
    this.dyn = { on: true, modes, odds, section: Math.round(num(d.section, 16, 4, 64)), seed: Math.round(num(d.seed, 0, 0, 1e6)),
                 plain: fromDb(num(d.plainDb, -6, -24, 0)), ghost: fromDb(num(d.ghostDb, -9, -30, 0)), accent: fromDb(num(d.accentDb, 0, -6, 6)),
                 onBeat: fromDb(num(d.onBeatDb, -4, -20, 0)), texture: fromDb(num(d.textureDb, -8, -30, 0)), offBackbeat: fromDb(num(d.offBackbeatDb, -4, -20, 0)),
                 kickBeats: fromDb(num(d.kickBeatsDb, 0, -20, 0)), hatOnKick: fromDb(num(d.hatOnKickDb, 0, -12, 0)),
                 trim: { kick: fromDb(num(t.kick, 0, -12, 12)), snare: fromDb(num(t.snare, 0, -12, 12)), hats: fromDb(num(t.hats, 0, -12, 12)) },
                 level: { snare: lv.snare === null ? null : fromDb(num(lv.snare, -8, -30, 6)), hats: lv.hats === null ? null : fromDb(num(lv.hats, -12, -30, 6)) },
                 cymToHats: d.cymToHats !== false, pull: num(d.levelPull, 0, 0, 1), reach: fromDb(num(d.levelReachDb, 9, 0, 48)), kits: new WeakMap(), scratch: null,
                 kick: null, bal: this.bal };
    const kb = d.kickBoost && d.kickBoost.on !== false ? d.kickBoost : null, ms = (x, dflt, lo, hi) => Math.exp(-1 / (num(x, dflt, lo, hi) / 1000 * this.sr));
    if (kb) this.dyn.kick = { target: fromDb(num(kb.targetDb, 2, -30, 30)), max: fromDb(num(kb.maxDb, 6, 0, 12)), rate: num(kb.rate, 0.25, 0.01, 1), ceil: fromDb(num(kb.ceilDb, 3, 0, 12)) };
    const boost = kb ? { att: fromDb(num(kb.punchDb, 3, 0, 12)) - 1, kAtt: ms(kb.attackMs, 4, 0.5, 30), body: fromDb(num(kb.bodyDb, 6, 0, 12)) - 1,
                         kBody: ms(kb.bodyMs, 60, 5, 300), bodyHz: num(kb.bodyHz, 140, 40, 400), bodyQ: num(kb.bodyQ, 0.9, 0.3, 4), subHz: num(kb.subHz, 70, 20, 200) } : null;
    for (const l of this.layers) { l.dyn = this.dyn; l.hatMode = this.hatModeAt(Math.max(0, this.bar)); l.kit.setKickBoost(boost); l.kit.kg = 1; l.kit.kc = Infinity; }
  }
  // the hat mode of the section a bar lies in
  hatModeAt(bar) {
    const D = this.dyn; if (!D) return 'flat';
    let r = mix32(Math.floor(bar / D.section) + D.seed, this.playing[0] | 0), sum = 0;
    for (const k in D.odds) sum += Math.max(0, +D.odds[k] || 0);
    r *= sum || 1;
    for (const k in D.odds) { r -= Math.max(0, +D.odds[k] || 0); if (r < 0) return k in D.modes ? k : 'flat'; }
    return 'flat';
  }
  // The bus (off unless asked for: setBalance, more.bus), after the master's own compressor and before the limiter:
  // a compressor with a slow attack (a kick's front passes) and a release timed to the beat (releaseBeats), a
  // make-up, and a soft clip from clipKnee up that rounds what the make-up pushes over. It brings the crest factor
  // (peak over level) toward that of the recordings; trimDb gives the added level back after the clip, so the
  // output is as loud as before with lower peaks; the limiter stays last.
  setBus(d) {
    const B = this.bal;
    if (!d || d.on === false) { B.bus = null; return; }
    const num = (x, dflt, lo, hi) => (Number.isFinite(+x) && x != null ? clamp(+x, lo, hi) : dflt), tc = (sec) => 1 - Math.exp(-1 / (sec * this.sr));
    B.bus = { thr: fromDb(num(d.thrDb, -14, -40, 0)), slope: 1 - 1 / num(d.ratio, 2, 1, 10), knee: num(d.kneeDb, 6, 0, 18), att: tc(num(d.attackMs, 30, 1, 200) / 1000),
              beats: num(d.releaseBeats, 0.5, 0.1, 4), rel: tc(0.24), makeup: fromDb(num(d.makeupDb, 4, 0, 12)), clip: num(d.clipKnee, 0.6, 0.3, 0.95), out: fromDb(num(d.trimDb, 0, -12, 0)),
              gsm: tc(0.003), env: B.bus ? B.bus.env : 0, g: B.bus ? B.bus.g : 1, gt: 1, n: 0 };
  }
  // Voice levels with a compressor (off unless asked for: setBalance, more.voice). In place of the plain follower:
  //   a synth voice's level is read as it is heard: weighted like a loudness meter (lows under 100 Hz count less,
  //   highs over 1.5 kHz 4 dB more) and only at its notes (samples within gateDb of its recent peak), so a dull
  //   bass is not left too quiet, a bright lead not too loud, and tails and gaps do not count;
  //   the trim takes that level to the voice's share of the kit's weighted level (the share of the recordings,
  //   moved by brightDb: how much brighter that stem is than the drums in the recordings), at most weighLimitDb
  //   away from where the plain measure would put it;
  //   a feed-forward compressor with a soft knee evens out single notes and filter peaks (ratio, or ratios per
  //   voice [bass, lead, air] with 1 for none; thrDb over the voice's own level, attack, release), and a make-up
  //   brings the level after it back to the share; weighUpDb [bass, lead, air] caps how far the weighted reading
  //   may raise a voice over the plain one (weighLimitDb: how far it may lower it);
  //   a sound that enters is read fast for its first freshMs, and a sound heard before returns at its trim.
  setVoiceLevels(d) {
    const B = this.bal;
    if (!d || d.on === false) { B.vc = null; return; }
    const num = (x, dflt, lo, hi) => (Number.isFinite(+x) && x != null ? clamp(+x, lo, hi) : dflt), tc = (sec) => 1 - Math.exp(-1 / (sec * this.sr));
    const c = d.comp || {}, br = Array.isArray(d.brightDb) ? d.brightDb : [0, 0, 0];
    const old = B.vc;
    B.vc = { k100: 1 - Math.exp(-2 * Math.PI * 100 / this.sr), k1500: 1 - Math.exp(-2 * Math.PI * 1500 / this.sr), ka: 0, kb: 0, kitw: old ? old.kitw : 0,
             bright: [1, 1, 1, 1].map((_, j) => (j ? fromDb(num(br[j - 1], 0, -12, 12)) : 1)), limit: fromDb(num(d.weighLimitDb, 6, 0, 18)),
             up: [1, 1, 1, 1].map((_, j) => (j ? fromDb(num(Array.isArray(d.weighUpDb) ? d.weighUpDb[j - 1] : d.weighLimitDb, 6, 0, 18)) : 1)),   // per voice: how far the weighted reading may raise it over the plain one
             ratios: [1, 1, 1, 1].map((_, j) => (j ? num(Array.isArray(c.ratios) ? c.ratios[j - 1] : c.ratio, 3.5, 1, 12) : 1)),
             gate: fromDb(num(d.gateDb, -20, -40, -6)), eatt: tc(0.003), erel: tc(0.12), pdecay: Math.exp(-1 / (2.5 * this.sr)),
             kfast: tc(0.05), fastRate: tc(0.03), first: Math.round(0.03 * this.sr), fresh: Math.round(num(d.freshMs, 250, 0, 1000) / 1000 * this.sr),
             comp: c.on !== false, ratio: num(c.ratio, 3.5, 1, 12), thr: fromDb(num(c.thrDb, 3, -12, 18)), knee: num(c.kneeDb, 6, 0, 18),
             catt: tc(num(c.attackMs, 5, 0.5, 50) / 1000), crel: tc(num(c.releaseMs, 150, 20, 800) / 1000), gsm: tc(0.002),
             makeup: fromDb(num(c.makeupDb, 8, 0, 14)), mkk: tc(1.0),
             v: old ? old.v : [null, 1, 2, 3].map((j) => (j ? { a: 0, b: 0, e: 0, pk: 0, ms: 0, mp: 0, msy: 0, c: 0, g: 1, gt: 1, n: 0, mk: 1, fresh: 0, cnt: 0, key: null } : null)),
             known: old ? old.known : new Map() };
  }
  // a layer's sound for this bar (barBegin): a change stores what was learned of the old sound and recalls the new one
  voiceSound(j, key) {
    const V = this.bal.vc; if (!V || !j) return;
    const v = V.v[j]; if (v.key === key) return;
    const B = this.bal;
    if (v.key != null && v.ms > 1e-9) { if (V.known.size > 96) V.known.clear(); V.known.set(v.key, [v.ms, v.mp, v.msy, B.trim[j], v.mk]); }
    v.key = key;
    const k = key != null ? V.known.get(key) : null;
    if (k) { v.ms = k[0]; v.mp = k[1]; v.msy = k[2]; B.trim[j] = k[3]; v.mk = k[4]; v.fresh = V.fresh >> 2; v.cnt = V.first; }
    else {                                              // never heard: it starts no louder than the sounds known on this layer, then is read fast
      let low = B.trim[j]; for (const [kk, x] of V.known) if (kk.charCodeAt(0) === 48 + j && x[3] < low) low = x[3];
      v.ms = 0; v.mp = 0; v.msy = 0; v.mk = 1; v.fresh = V.fresh; v.cnt = 0; B.trim[j] = low;
    }
    v.pk = 0; v.e = 0;
  }
  // one sample of synth voice j (1..3) under the voice levels: returns it at its level
  voiceLevel(j, s, p) {
    const B = this.bal, V = B.vc, v = V.v[j];
    v.a += V.k100 * (s - v.a); const h = s - v.a; v.b += V.k1500 * (h - v.b); const w = h + 0.585 * (h - v.b), aw = Math.abs(w);
    v.e += (aw - v.e) * (aw > v.e ? V.eatt : V.erel);
    if (v.e > v.pk) v.pk = v.e; else v.pk *= V.pdecay;
    const note = p > 1e-7 && v.e > v.pk * V.gate;                // inside a note of this voice
    if (note) {
      if (v.cnt < V.first) { v.cnt++; v.ms += (w * w - v.ms) / v.cnt; v.mp += (p - v.mp) / v.cnt; }      // a new sound's first 30 ms: the plain mean
      else { const k = v.fresh > 0 ? V.kfast : B.k; v.ms += (w * w - v.ms) * k; v.mp += (p - v.mp) * k; }
    }
    if (!(V.kitw > 0 && B.kit > 0 && v.ms > 1e-9 && v.mp > 1e-9)) return s;
    const target = B.ratio[j] * V.bright[j] * Math.sqrt(V.kitw);                    // as heard
    const plain = B.ratio[j] * Math.sqrt(B.kit / v.mp);                             // where the plain measure would set the trim
    const want = clamp(clamp(target / Math.sqrt(v.ms), plain / V.limit, plain * V.up[j]), 1 / B.range, B.range);
    if (v.cnt >= V.first) B.trim[j] += (want - B.trim[j]) * (v.fresh > 0 ? V.fastRate : want < B.trim[j] ? B.down : B.up);
    B.gain[j] = B.duck > 0 && B.top > 1e-6 ? 1 - B.duck * B.share[j] * Math.min(1, B.env / B.top) : 1;
    let y = s * B.trim[j] * B.gain[j], G = B.trim[j] * B.gain[j];
    if (V.comp && V.ratios[j] > 1.01) {
      const ay = Math.abs(y);
      v.c += (ay - v.c) * (ay > v.c ? V.catt : V.crel);
      if (--v.n <= 0) {                                           // the gain computer, every 8 samples
        v.n = 8;
        const thr = V.thr * B.trim[j] * Math.sqrt(v.mp), x = v.c > 1e-9 && thr > 1e-9 ? 20 * Math.log10(v.c / thr) : -99, half = V.knee / 2, slope = 1 - 1 / V.ratios[j];
        const cut = x <= -half ? 0 : x >= half ? slope * x : slope * (x + half) * (x + half) / (2 * V.knee || 1);
        v.gt = cut > 0 ? Math.pow(10, -cut / 20) : 1;
      }
      v.g += (v.gt - v.g) * V.gsm;
      y *= v.g; G *= v.g;
      if (note) { const wy = w * G; v.msy = v.msy === 0 ? wy * wy : v.msy + (wy * wy - v.msy) * B.k; }
      if (v.msy > 1e-12) { const mk = clamp(target * B.gain[j] / Math.sqrt(v.msy), 0.7, V.makeup); v.mk += (mk - v.mk) * V.mkk; }
      y *= v.mk;
    }
    if (v.fresh > 0 && note) v.fresh--;
    return y;
  }
  setPool(pool) {
    this.pool = pool;
    const bank = sampleBank(pool);
    for (const layer of this.layers) { layer.sampler.reset(); layer.sampler.bank = bank; }
    const eq = pool.fx && Array.isArray(pool.fx.drum_eq) ? pool.fx.drum_eq : null;   // [[hz, dB, q], ...]
    this.layers[0].eq = eq ? eq.map(([f, g, q]) => new Peak(this.sr, +f, +g, +q || 1.4)) : null;
    this.tokens = [0, 1, 2, 3].map((i) => ((pool.layers && pool.layers[i] && pool.layers[i].tokens) || [null]).map(resolveToken));
    this.sounds = [0, 1, 2, 3].map((i) => ((pool.layers && pool.layers[i] && pool.layers[i].tokens) || []).filter((t) => t && t.voice === 'synth' && t.patch));
    this.near = null;                                     // the control hold's nearest sounds, per pool
    this.fx = Object.assign({}, FX_DEFAULTS, pool.fx || {});
    if (this.verb) this.verb.set(this.fx.reverb_size, this.fx.reverb_damp);
    if (this.dly) { this.dly.fb = clamp(this.fx.delay_feedback, 0, 0.95); this.dly.tone = 0.08 + 0.8 * clamp(this.fx.delay_tone, 0, 1); }
    if (!this.tempoSet && pool.bpm) this.bpm = clamp(+pool.bpm, 40, 220);
  }
  setTempo(bpm) { this.bpm = clamp(+bpm || 126, 40, 220); this.tempoSet = true; }
  // real machine one-shots: kit = {device: {voice: Float32Array (mono)}}, sampleRate of those arrays; null = the models
  setSamples(kit, sampleRate) {
    const bank = {};
    for (const [dev, voices] of Object.entries(kit || {})) {
      bank[dev] = {};
      for (const [voice, a] of Object.entries(voices || {})) if (a && a.length) bank[dev][voice] = a instanceof Float32Array ? a : Float32Array.from(a);
    }
    this.samples = kit ? bank : null; this.sampleRate = +sampleRate || this.sr;
    for (const ly of this.layers) { ly.kit.bank = this.samples; ly.kit.bankRate = this.sampleRate; }
  }
  // tokens[4] (indices into each layer's tokens), mods = [[filter, space] x 4] each -1..1 (missing = 0),
  // patterns[4] optional: a layer's pattern for this bar only (README pattern format), replacing its token's own
  queueBar(tokens, mods, patterns, patches) {
    const t = [0, 1, 2, 3].map((i) => (tokens && Number.isInteger(tokens[i]) && tokens[i] >= 0 ? tokens[i] : 0));
    const m = [0, 1, 2, 3].map((i) => [clamp(+(mods && mods[i] && mods[i][0]) || 0, -1, 1), clamp(+(mods && mods[i] && mods[i][1]) || 0, -1, 1)]);
    const p = [0, 1, 2, 3].map((i) => (patterns && patterns[i] && typeof patterns[i] === 'object' ? patterns[i] : null));
    this.queue.push({ t, m, p, patches });
  }
  // drop the bars queued ahead (a brain switch: the next bar is the new brain's)
  flush() { this.queue.length = 0; }
  // stop: nothing queued, nothing held: the next bar is silent until a new one is queued, and every layer is quiet now
  silence() {
    this.queue.length = 0; this.playing = [0, 0, 0, 0]; this.patterns = null; this.patches = null; this.mods = [[0, 0], [0, 0], [0, 0], [0, 0]];
    for (const layer of this.layers) {
      layer.barStart(null, [0, 0]);
      layer.kit.reset(); layer.acid.reset(); layer.synth.reset(); layer.sampler.reset(); layer.sid.reset();
    }
  }
  onBar(cb) { this.cbs.push(cb); }
  // cb(bar, step 0..15, frame): fired at the first sample of every sixteenth (for step lights)
  onStep(cb) { this.stepCbs.push(cb); }
  // where the engine stands now: {bar, step, frame, stepLen (samples), bpm, playing tokens}
  position() { return { bar: this.bar, step: this.stepIdx, frame: this.frame, stepLen: this.stepLen, bpm: this.bpm, playing: this.playing.slice(), mods: this.layers.map((l) => l.mod.slice()) }; }
  // the step (bar, sixteenth) sounding at a sample frame of this engine's clock, at the current tempo
  stepAt(frame) {
    const d = Math.floor((frame - (this.nextStep - this.stepLen)) / this.stepLen), abs = this.bar * 16 + this.stepIdx + d;
    return { bar: Math.floor(abs / 16), step: ((abs % 16) + 16) % 16 };
  }
  barBegin() {
    this.bar++;
    // an empty queue (the brain's answer is late) plays the last bar again, with its own patterns
    const q = this.queue.length ? this.queue.shift() : { t: this.playing, m: this.mods, p: this.patterns, patches: this.patches };
    const next = q.t; this.playing = next; this.mods = q.m; this.patterns = q.p || null; this.patches = q.patches || null; this.barStartFrame = this.frame; this.barLen = 16 * this.stepLen;
    if (this.dyn && (this.bar % this.dyn.section === 0 || this.dynBar !== this.bar - 1)) { const mode = this.hatModeAt(this.bar); for (const l of this.layers) l.hatMode = mode; }
    this.dynBar = this.bar;
    for (let i = 0; i < 4; i++) {
      let patch = i > 0 && next[i] && this.patches && this.patches[i];
      let t, held = null;
      if (patch && this.holdControls && patch.voice === 'synth' && (held = this.nearestSound(i, patch))) {
        let pat = q.p && q.p[i];
        if (pat && Array.isArray(pat.notes)) {
          const vel = new Array(16).fill(0);
          for (let s = 0; s < 16; s++) if (pat.notes[s] != null && (!Array.isArray(pat.vel) || pat.vel[s] > 0)) vel[s] = held.vel[s % 4];
          pat = { ...pat, vel, accent: undefined };
        }
        patch = { ...patch, patch: held.patch, gain: held.gain };
        t = resolveToken({ ...patch, pattern: pat });
      } else if (patch) {
        t = resolveToken({ ...patch, pattern: q.p && q.p[i] });
      } else {
        t = (this.tokens[i] && this.tokens[i][next[i]]) || null;
      }
      if (!patch && t && q.p && q.p[i]) t = Object.assign({}, t, { pattern: normPattern(t.voice, q.p[i]) });   // the brain's own pattern for this bar
      if (i && this.bal.vc) this.voiceSound(i, t ? i + (patch ? JSON.stringify(patch.patch || patch) : `t${next[i]}`) : null);
      this.layers[i].barStart(this.solo >= 0 && this.solo !== i ? null : t, held ? [q.m[i][0], 0] : q.m[i]);
    }
    for (const cb of this.cbs) cb(this.bar, next.slice(), this.queue.length, q.m.map((x) => x.slice()));
  }
  doStep() {
    this.stepLen = (this.sr * 15) / this.bpm;
    this.stepIdx = (this.stepIdx + 1) % 16;
    if (this.stepIdx === 0) this.barBegin();
    if (this.log) this.log.push(this.frame);
    for (const ly of this.layers) ly.step(this.stepIdx, this.stepLen);
    for (const cb of this.stepCbs) cb(this.bar, this.stepIdx, this.frame);
    this.nextStep += this.stepLen;
    if (this.dly) this.dly.len = this.fx.delay_steps * this.stepLen;
  }
  process(left, right) {
    const n = left.length, fx = this.opts.fx, master = this.opts.master, F = this.fx;
    for (let i = 0; i < n; i++) {
      while (this.frame >= this.nextStep) this.doStep();
      if (this.ctl-- <= 0) {
        this.ctl = 31;
        const frac = clamp((this.frame - this.barStartFrame) / this.barLen, 0, 1);
        for (const ly of this.layers) ly.control(frac);
      }
      let L = 0, R = 0, dS = 0, rS = 0, xL = 0, xR = 0;
      const B = this.bal, SP = this.space;
      for (let j = 0; j < 4; j++) {
        const ly = this.layers[j];
        let s = ly.tick();
        if (B.on && j === 0 && B.duck > 0) {                       // the kick's envelope: the kit's lows, fast up, slower down
          B.lp += (s * B.trim[0] - B.lp) * B.lpk;
          const a = Math.abs(B.lp);
          B.env += (a - B.env) * (a > B.env ? B.eatt : B.erel);
          B.top = Math.max(B.env, B.top * B.tdecay);              // the envelope's recent peak: the duck is relative to it
        }
        if (s === 0) continue;
        if (B.on) {
          const sm = j === 0 ? s - ly.kx : s, p = sm * sm;         // the kit's level is read without the kick boost's part
          if (p > 1e-7) B.ms[j] += (p - B.ms[j]) * B.k;            // the voice's level while it sounds
          if (j === 0) {
            if (B.kitLevel > 0 && B.ms[0] > 1e-7) {                // the kit toward its one level
              const want = clamp(B.kitLevel / Math.sqrt(B.ms[0]), 1 / B.kitRange, B.kitRange);
              B.trim[0] += (want - B.trim[0]) * (want < B.trim[0] ? B.kdown : B.kup);
              s *= B.trim[0];
            }
            if (B.ms[0] > 1e-7) B.kit = B.ms[0] * B.trim[0] * B.trim[0];
            if (p > 1e-7) B.held = this.sr >> 1;                  // the kit sounded within the last half second
            if (B.vc) {                                           // the kit's level as it is heard (weighted)
              const V = B.vc, sk = sm * (B.kitLevel > 0 ? B.trim[0] : 1); V.ka += V.k100 * (sk - V.ka); const h = sk - V.ka; V.kb += V.k1500 * (h - V.kb); const w = h + 0.585 * (h - V.kb);
              if (p > 1e-7) V.kitw += (w * w - V.kitw) * B.k;
            }
          } else if (B.vc) {
            s = this.voiceLevel(j, s, p);
          } else if (B.kit > 0 && B.ms[j] > 1e-7) {
            const want = clamp(B.ratio[j] * Math.sqrt(B.kit / B.ms[j]), 1 / B.range, B.range);
            B.trim[j] += (want - B.trim[j]) * (want < B.trim[j] ? B.down : B.up);
            const top = B.ceiling * B.ratio[j] * Math.sqrt(B.kit), a = Math.abs(s);
            if (a * B.trim[j] > top) B.trim[j] = top / a;         // a sudden loud note never arrives boosted
            B.gain[j] = B.duck > 0 && B.top > 1e-6 ? 1 - B.duck * B.share[j] * Math.min(1, B.env / B.top) : 1;
            s *= B.trim[j] * B.gain[j];
          }
        }
        if (j === 1 && this.bassGain !== 1) s *= this.bassGain;
        L += s * ly.pl; R += s * ly.pr;
        if (SP) { dS += s * Math.min(1, ly.sd + SP.d[j]); rS += s * Math.min(1, ly.sv + SP.r[j]); } else { dS += s * ly.sd; rS += s * ly.sv; }
        if (j === 0 && ly.kx !== 0) { const x = ly.kx * (B.on && B.kitLevel > 0 ? B.trim[0] : 1); xL = x * ly.pl; xR = x * ly.pr; }   // the kick boost's part
      }
      if (fx) {
        if (SP) { SP.hp[0].tick(dS); dS = SP.hp[0].hp; SP.hp[1].tick(rS); rS = SP.hp[1].hp; }
        this.dly.tick(dS); this.verb.tick(rS, rS);
        L += this.dly.oL * F.delay_return + this.verb.oL * F.reverb_return;
        R += this.dly.oR * F.delay_return + this.verb.oR * F.reverb_return;
      }
      if (B.on) {                                                 // the mix toward one loudness, only while the kit plays
        if (B.held > 0) {
          B.held--;
          const ml = L - xL, mr = R - xR;                        // the mix's loudness is read without the kick boost's part
          B.oms += ((ml * ml + mr * mr) / 2 - B.oms) * B.ok;
          if (B.outLevel > 0 && B.oms > 1e-7) {
            const want = clamp(B.outLevel / Math.sqrt(B.oms), 0.63, 1.6);
            B.otrim += (want - B.otrim) * (want < B.otrim ? B.odown : B.oup);
          }
        }
        if (B.outLevel > 0) { L *= B.otrim; R *= B.otrim; }
      }
      if (master) {
        L *= F.master; R *= F.master;
        if (B.on && B.eq) { for (const f of B.eq[0]) L = f.run(L); for (const f of B.eq[1]) R = f.run(R); }     // the mix's tone toward the recordings' bands
        const a = Math.max(Math.abs(L), Math.abs(R));
        this.cEnv += (a - this.cEnv) * (a > this.cEnv ? this.cAtt : this.cRel);
        if (this.cEnv > 0.45) { const g = Math.pow(this.cEnv / 0.45, 1 / 2.5 - 1); L *= g; R *= g; }
        L *= 1.15; R *= 1.15;
        if (B.on && B.bus) {                                      // the bus: glue, make-up, soft clip; the limiter stays last
          const U = B.bus, a2 = Math.max(Math.abs(L), Math.abs(R));
          U.env += (a2 - U.env) * (a2 > U.env ? U.att : U.rel);
          if (--U.n <= 0) {
            U.n = 8;
            U.rel = 1 - Math.exp(-1 / (Math.max(0.05, U.beats * 60 / this.bpm) * this.sr));      // the release goes with the beat
            const x = U.env > 1e-9 ? 20 * Math.log10(U.env / U.thr) : -99, half = U.knee / 2;
            const cut = x <= -half ? 0 : x >= half ? U.slope * x : U.slope * (x + half) * (x + half) / (2 * U.knee || 1);
            U.gt = (cut > 0 ? Math.pow(10, -cut / 20) : 1) * U.makeup;
          }
          U.g += (U.gt - U.g) * U.gsm;
          L *= U.g; R *= U.g;
          const k = U.clip;
          if (L > k) L = k + (1 - k) * Math.tanh((L - k) / (1 - k)); else if (L < -k) L = -k - (1 - k) * Math.tanh((-L - k) / (1 - k));
          if (R > k) R = k + (1 - k) * Math.tanh((R - k) / (1 - k)); else if (R < -k) R = -k - (1 - k) * Math.tanh((-R - k) / (1 - k));
          L *= U.out; R *= U.out;                                 // the level the make-up added is given back, the peaks keep their new place under it
        }
        const pk = Math.max(Math.abs(L), Math.abs(R)), need = pk > 0.97 ? 0.97 / pk : 1;
        this.lim = Math.min(need, this.lim + (1 - this.lim) * this.limRel);
        L *= this.lim; R *= this.lim;
        if (!Number.isFinite(L) || !Number.isFinite(R)) { L = 0; R = 0; this.cEnv = 0; this.lim = 1; }
      }
      left[i] = L; if (right) right[i] = R;
      this.frame++;
    }
  }
}
