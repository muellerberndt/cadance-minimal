#!/usr/bin/env node
// Fit one token (voice/device + patch) to one bar of stem audio, the pattern given.
//   node engine/fit.mjs --layer <0-3> --target bar.wav --pattern pattern.json --bpm <bpm> --out token.json [--budget 1200] [--seed 0] [--loss full|mel] [--lag 2] [--no-refine] [--no-add] [--voice sid]
// --voice sid fits the SID-style chip voice (fitSid) to a synth layer's bar instead of choosing between acid and synth
// --no-refine keeps the pattern as given; --no-add lets the refinement remove and nudge hits/notes but never add new ones
// pattern.json: the pattern in the README format, optionally with bpm, track, bar, tail_s, layer
// (bpm and layer there win over the flags); {"silent": true} writes {"silent": true} and stops.
// The candidate plays two bars of the token dry (no effects, no master bus); its second bar
// (+ tail_s) is compared with the target after both are scaled to equal RMS. The loss is
// multi-resolution log-mel L1 + per-quarter-sixteenth loudness envelope + onset flux. Finally the
// token's level is solved so the render matches the target's RMS.
import fs from 'node:fs';
import { Engine, PARAMS, INDEX, DRUM_VOICES, DEVICES, normPattern, phys, unit } from './engine.mjs';
import { readWav } from './wav.mjs';
import { loadSamples, useSamples } from './samples.mjs';
const SAMPLES = useSamples() ? loadSamples() : null;   // real 606/808/909 one-shots (--no-samples: the models)
import { describe, distance } from './features.mjs';

const SR = 44100;

function mulberry32(a) {
  return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gauss(rnd) { const u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

// render the token for 2 bars + tail; return the mono second bar (+ tail) as Float32Array
export function renderBar(tok, layer, bpm, nOut) {
  const tokens = [[null], [null], [null], [null]]; tokens[layer] = [null, tok];
  const eng = new Engine(SR, { bpm, layers: tokens.map((t) => ({ tokens: t })) }, { fx: false, master: false });
  if (SAMPLES) eng.setSamples(SAMPLES.kit, SAMPLES.sampleRate);
  eng.setTempo(bpm);
  const t = [0, 0, 0, 0]; t[layer] = 1;
  eng.queueBar(t); eng.queueBar(t); eng.queueBar(t);
  const barLen = Math.round((SR * 240) / bpm), n = barLen + nOut;
  const L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) out[i] = 0.5 * (L[barLen + i] + R[barLen + i]);
  return out;
}

function freeParams(voice, pattern) {
  if (voice === 'drums') {
    const used = DRUM_VOICES.filter((v) => pattern.hits[v] && pattern.hits[v].some((x) => x > 0));
    return [...used.flatMap((v) => [`${v}_tune`, `${v}_decay`, `${v}_level`, `${v}_tone`]), 'drive'];
  }
  const fixed = new Set(['tune', 'level', 'pan', 'send_delay', 'send_reverb']);
  return PARAMS[voice].map((p) => p.name).filter((n) => !fixed.has(n));
}

function baseU(voice) { const u = {}; for (const p of PARAMS[voice]) u[p.name] = p.def; u.pan = 0.5; u.send_delay = 0; u.send_reverb = 0; return u; }

function patternFor(voice, raw) {
  const p = normPattern(voice, raw);
  if (voice === 'drums') return { hits: Object.fromEntries(Object.entries(p.hits).filter(([, a]) => a)) };
  if (voice === 'acid') return { notes: p.notes, accent: p.accent, slide: p.slide };
  return { notes: p.notes, vel: p.vel, len: p.len };
}

export function fitToken({ layer, target, pattern, bpm, budget = 1200, seed = 0, weights, lag = 2, refine = true, refineAdd = true, devices = null, log = () => {} }) {
  const rnd = mulberry32(seed + 1), T = describe(target, SR), nOut = target.length;
  // categories: drums = device; synth layers = voice; the octave (tune -12/0/+12) is a discrete choice inside a voice
  const cats = layer === 0 ? (devices && devices.length ? devices : DEVICES).map((d) => ({ voice: 'drums', device: d, oct: [null] }))
    : [{ voice: 'acid', device: null, oct: [0.5, 0, 1] }, { voice: 'synth', device: null, oct: [0.5, 0.25, 0.75] }];
  for (const c of cats) { c.pattern = patternFor(c.voice, pattern); c.free = freeParams(c.voice, c.pattern); c.best = null; }
  let renders = 0; const t0 = Date.now();
  const evaluate = (c, u) => {
    const tok = { voice: c.voice, device: c.device || undefined, patch: u, pattern: c.pattern };
    const x = renderBar(tok, layer, bpm, nOut); renders++;
    let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
    if (e < 1e-12) return { total: 99, mel: 99, env: 99, flux: 99 };
    return distance(T, describe(x, SR), weights, lag);
  };
  const make = (c, oct) => { const u = baseU(c.voice); if (oct != null) u.tune = oct; return u; };
  const consider = (c, u, d) => { if (!c.best || d.total < c.best.d.total) c.best = { u: { ...u }, d }; };

  // 1. random sampling: every category and octave once with the default patch, then random patches
  // spread over the categories in proportion to their free dimensions
  for (const c of cats) for (const o of c.oct) consider(c, make(c, o), evaluate(c, make(c, o)));
  const dims = cats.map((c) => c.free.length), sumD = dims.reduce((a, b) => a + b, 0);
  const nRand = Math.max(renders + cats.length * 4, Math.round(budget * 0.35));
  let i = 0;
  while (renders < nRand) {
    const c = cats.length === 1 ? cats[0] : (() => { let r = rnd() * sumD; for (let j = 0; j < cats.length; j++) { r -= dims[j]; if (r <= 0) return cats[j]; } return cats[cats.length - 1]; })();
    const u = make(c, c.oct[i++ % c.oct.length]);
    for (const k of c.free) u[k] = rnd();
    consider(c, u, evaluate(c, u));
  }
  const ranked = cats.slice().sort((a, b) => a.best.d.total - b.best.d.total);
  log(`random: ${ranked.map((c) => `${c.voice}${c.device || ''}@${c.best.u.tune != null ? (+c.best.u.tune).toFixed(2) : '-'}=${c.best.d.total.toFixed(3)}`).join(' ')}`);

  // 2. (1+lambda) evolution: synth layers always search both voices (60/40), drums the best device
  // (and the runner-up when it is within 15%)
  const left = Math.round((budget - renders) * (refine ? 0.6 : 1)), lanes = [ranked[0]];
  if (ranked[1] && (layer > 0 || ranked[1].best.d.total < ranked[0].best.d.total * 1.15)) lanes.push(ranked[1]);
  const shares = lanes.length === 2 ? [0.6, 0.4] : [1];
  lanes.forEach((c, li) => {
    const stop = renders + Math.round(left * shares[li] * 0.85);
    let sigma = 0.2; const lambda = 8, pm = Math.max(0.2, 3 / c.free.length);
    while (renders < stop) {
      let improved = false;
      for (let j = 0; j < lambda && renders < stop; j++) {
        const u = { ...c.best.u };
        let moved = false;
        for (const k of c.free) if (rnd() < pm) { u[k] = clamp01(u[k] + sigma * gauss(rnd)); moved = true; }
        if (!moved) { const k = c.free[Math.floor(rnd() * c.free.length)]; u[k] = clamp01(u[k] + sigma * gauss(rnd)); }
        if (c.oct.length > 1 && rnd() < 0.04) u.tune = c.oct[Math.floor(rnd() * c.oct.length)];
        const d = evaluate(c, u);
        if (d.total < c.best.d.total) { c.best = { u, d }; improved = true; }
      }
      sigma = improved ? Math.min(0.35, sigma * 1.25) : Math.max(0.01, sigma * 0.82);
    }
  });
  const win = cats.slice().sort((a, b) => a.best.d.total - b.best.d.total)[0];
  // 3. pattern refinement (the corpus pattern is a reading, not the truth): toggle hits/notes, nudge
  // velocities, lengths, accents and slides, keeping every change that lowers the loss; then resume
  // the patch search on the refined pattern
  let refined = 0;
  if (refine) {
    const tryPattern = (mut) => {
      const keep = JSON.stringify(win.pattern); mut(win.pattern);
      const d = evaluate(win, win.best.u);
      if (d.total < win.best.d.total) { win.best = { u: win.best.u, d }; refined++; return true; }
      win.pattern = JSON.parse(keep); return false;
    };
    const stopP = renders + Math.round((budget - renders) * 0.55);
    if (win.voice === 'drums') {
      const voices = Object.keys(win.pattern.hits);
      for (const pass of [0, 1]) for (const v of voices) for (let st = 0; st < 16 && renders < stopP; st++) {
        const h = win.pattern.hits[v][st];
        if (pass === 0) { if (h > 0 || refineAdd) tryPattern((p) => { p.hits[v][st] = h > 0 ? 0 : 0.7; }); }
        else if (h > 0) { if (!tryPattern((p) => { p.hits[v][st] = Math.min(1, h + 0.25); })) tryPattern((p) => { p.hits[v][st] = Math.max(0.1, h - 0.25); }); }
      }
    } else {
      const notes = win.pattern.notes, common = (() => { const c = {}; let b = null; for (const n of notes) if (n != null) { c[n] = (c[n] || 0) + 1; if (b == null || c[n] > c[b]) b = n; } return b; })();
      for (let st = 0; st < 16 && renders < stopP && common != null; st++) {
        const n = win.pattern.notes[st];
        if (n != null) tryPattern((p) => { p.notes[st] = null; if (p.vel) p.vel[st] = 0; });
        else if (refineAdd) { let prev = null; for (let j = st - 1; j >= 0; j--) if (win.pattern.notes[j] != null) { prev = win.pattern.notes[j]; break; }
          tryPattern((p) => { p.notes[st] = prev != null ? prev : common; if (p.vel) p.vel[st] = 0.7; if (p.len) p.len[st] = 1; }); }
      }
      for (let st = 0; st < 16 && renders < stopP; st++) {
        if (win.pattern.notes[st] == null) continue;
        if (win.voice === 'synth') { const l = win.pattern.len[st]; if (!tryPattern((p) => { p.len[st] = Math.min(16, l * 2); })) tryPattern((p) => { p.len[st] = Math.max(0.25, l / 2); }); }
        else { tryPattern((p) => { p.accent[st] = p.accent[st] ? 0 : 1; }); tryPattern((p) => { p.slide[st] = p.slide[st] ? 0 : 1; }); }
      }
    }
    // resume the patch search on the refined pattern
    let sigma = 0.1; const pm = Math.max(0.2, 3 / win.free.length), stop = budget - 2 * win.free.length;
    while (renders < stop) {
      const u = { ...win.best.u };
      for (const k of win.free) if (rnd() < pm) u[k] = clamp01(u[k] + sigma * gauss(rnd));
      const d = evaluate(win, u);
      if (d.total < win.best.d.total) { win.best = { u, d }; sigma = Math.min(0.3, sigma * 1.2); } else sigma = Math.max(0.01, sigma * 0.95);
    }
  }
  // 4. coordinate polish on the winner
  for (const step of [0.06, 0.02]) {
    for (const k of win.free) {
      if (renders >= budget) break;
      for (const s of [-step, step]) {
        const u = { ...win.best.u }; u[k] = clamp01(u[k] + s);
        const d = evaluate(win, u); if (d.total < win.best.d.total) { win.best = { u, d }; break; }
      }
    }
  }
  // 5. level: match the target's RMS (the render is linear in the level gain)
  const P = PARAMS[win.voice][INDEX[win.voice].level], u = { ...win.best.u };
  const x = renderBar({ voice: win.voice, device: win.device || undefined, patch: u, pattern: win.pattern }, layer, bpm, nOut); renders++;
  let cr = 0; for (let i = 0; i < x.length; i++) cr += x[i] * x[i]; cr = Math.sqrt(cr / x.length);
  const g0 = P.lo + (P.hi - P.lo) * u.level * u.level, g = cr > 0 ? (g0 * T.rms) / cr : g0;
  const lv = Math.sqrt(Math.max(0, (g - P.lo) / (P.hi - P.lo)));
  u.level = clamp01(lv);
  const ms = (Date.now() - t0) / renders;
  const round = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Math.round(v * 10000) / 10000]));
  return {
    voice: win.voice, ...(win.device ? { device: win.device } : {}), patch: round(u), pattern: win.pattern,
    fit: { loss: +win.best.d.total.toFixed(4), mel: +win.best.d.mel.toFixed(4), env: +win.best.d.env.toFixed(4), flux: +win.best.d.flux.toFixed(4),
      budget, renders, ms_per_render: +ms.toFixed(1), level_clamped: lv > 1, target_rms: +T.rms.toFixed(5), pattern_changes: refined, shift: win.best.d.shift,
      categories: Object.fromEntries(cats.map((c) => [`${c.voice}${c.device ? ':' + c.device : ''}`, +c.best.d.total.toFixed(4)])) },
  };
}

// ---------------------------------------------------------------- the SID-style chip voice

// The chip pattern of a token of another voice: the same notes on the same steps. An acid line's accents
// become velocities and its gate share and slides become note lengths.
export function sidPattern(source) {
  const p = normPattern(source.voice === 'acid' ? 'acid' : 'synth', source.pattern);
  if (source.voice !== 'acid') return { notes: p.notes, vel: p.vel, len: p.len };
  const share = phys(PARAMS.acid[INDEX.acid.gate], source.patch && Number.isFinite(+source.patch.gate) ? +source.patch.gate : PARAMS.acid[INDEX.acid.gate].def);
  const holds = p.notes.map((n, i) => n != null && !!p.slide[i] && p.notes[(i + 1) % 16] != null);
  return { notes: p.notes, vel: p.notes.map((n, i) => (n == null ? 0 : p.accent[i] ? 1 : 0.7)), len: holds.map((h) => (h ? 1.6 : +share.toFixed(3))), slide: p.slide };
}

const nearest = (table, ms) => { let b = 0; for (let i = 1; i < table.length; i++) if (Math.abs(Math.log(table[i] / ms)) < Math.abs(Math.log(table[b] / ms))) b = i; return b; };
const SID_ATTACK_MS = [2, 8, 16, 24, 38, 56, 68, 80, 100, 250, 500, 800, 1000, 3000, 5000, 8000];
const SID_DECAY_MS = SID_ATTACK_MS.map((x) => 3 * x);

// Where the chip can follow another voice's patch directly: register, envelope rates, filter. The search
// starts here. FM, drive, the LFO and the pitch envelope have no register on the chip.
export function sidStart(source) {
  const S = INDEX.sid, L = PARAMS.sid, u = {}, src = (source && source.patch) || {};
  for (const p of L) u[p.name] = p.def;
  u.pan = 0.5; u.send_delay = 0; u.send_reverb = 0;
  if (!source || !PARAMS[source.voice] || source.voice === 'sid') return u;
  const V = PARAMS[source.voice], at = (n) => { const p = V[INDEX[source.voice][n]], v = +src[n]; return phys(p, Number.isFinite(v) ? v : p.def); };
  const raw = (n) => { const p = V[INDEX[source.voice][n]], v = +src[n]; return Number.isFinite(v) ? v : p.def; };
  const set = (n, v) => { u[n] = clamp01(unit(L[S[n]], v)); };
  u.tune1 = clamp01((Math.round(Math.max(-12, Math.min(12, at('tune')))) + 12) / 24);
  set('cutoff', Math.max(30, Math.min(12000, at('cutoff'))));
  if (source.voice === 'acid') {
    u.wave1 = raw('wave') < 0.5 ? 0.5 : 0.75; u.reso = clamp01(raw('reso'));
    set('f_env', Math.min(4, at('envmod'))); set('f_decay', Math.max(0.01, Math.min(2, at('decay'))));
    u.attack1 = 0; u.decay1 = 11 / 15; u.sustain1 = 1; u.release1 = 0;
  } else {
    u.wave1 = raw('wave') < 0.25 ? 0.25 : raw('wave') < 0.75 ? 0.5 : 0.75;
    u.reso = clamp01(Math.log(Math.max(0.707, at('reso')) / 0.707) / Math.log(8.5));
    u.mode = raw('ftype') < 0.25 ? 0 : raw('ftype') < 0.75 ? 1 / 3 : 2 / 3;
    set('f_env', Math.max(-4, Math.min(4, at('f_env')))); set('f_decay', Math.max(0.01, Math.min(2, at('f_decay'))));
    u.attack1 = nearest(SID_ATTACK_MS, at('attack') * 1000) / 15; u.decay1 = nearest(SID_DECAY_MS, at('decay') * 750) / 15;
    u.sustain1 = Math.round(raw('sustain') * 15) / 15; u.release1 = nearest(SID_DECAY_MS, at('release') * 750) / 15;
  }
  return u;
}

const SID_CHOICES = { tune2: [0, 0.5, 19 / 24, 1], tune3: [0, 0.5, 19 / 24, 1] };   // an octave down, unison, a fifth, an octave up

// Fit the chip voice to `target` (one bar + tail, mono, 44.1 kHz) with the pattern given. `source` (optional)
// is the token whose sound the target is: the search then starts from sidStart(source) and keeps its
// register. Returns {voice: 'sid', patch, pattern, gain?, fit}.
export function fitSid({ layer, target, pattern, bpm, source = null, budget = 220, seed = 0, weights, lag = 2, log = () => {} }) {
  const rnd = mulberry32(seed + 1), T = describe(target, SR), nOut = target.length, L = PARAMS.sid, t0 = Date.now();
  const pat = source ? sidPattern(source) : (() => { const p = normPattern('synth', pattern); return { notes: p.notes, vel: p.vel, len: p.len }; })();
  let renders = 0;
  const evaluate = (u) => {
    const x = renderBar({ voice: 'sid', patch: u, pattern: pat }, layer, bpm, nOut); renders++;
    let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
    if (e < 1e-12) return { total: 99, mel: 99, env: 99, flux: 99 };
    return distance(T, describe(x, SR), weights, lag);
  };
  // oscillator 1 carries the written pitch: it stays a pitched waveform, in the source's register and in tune;
  // oscillators 2 and 3 (noise too) are added to it
  const start = sidStart(source), free = L.map((p) => p.name).filter((n) => !['tune1', 'fine1', 'level', 'pan', 'send_delay', 'send_reverb'].includes(n));
  const defaults = Object.fromEntries(L.map((p) => [p.name, p.def])); defaults.tune1 = start.tune1; defaults.pan = 0.5;
  let before = evaluate(defaults), first = evaluate(start);
  // a bar of a stem names no register: an octave down and an octave up are tried, and the best one is kept
  if (!source) for (const t of [0, 1]) { const d = evaluate({ ...start, tune1: t }); if (d.total < first.total) { start.tune1 = t; defaults.tune1 = t; first = d; before = d; } }
  let best = first.total <= before.total ? { u: { ...start }, d: first } : { u: { ...defaults }, d: before };
  const consider = (u) => { const d = evaluate(u); if (d.total < best.d.total) { best = { u: { ...u }, d }; return true; } return false; };
  const copy = (u, k) => { for (const n of ['attack', 'decay', 'sustain', 'release', 'pw']) u[n + k] = u[n + 1]; };
  // 1. structures on the start patch: each waveform alone, doubled a few cents apart, with a triangle an
  // octave below, with a noise burst, and a ring-modulated triangle
  const lanes = [];
  for (const w of [0.25, 0.5, 0.75]) {
    const a = { ...start, wave1: w }; lanes.push(a);
    const b = { ...a, wave2: w, tune2: a.tune1, fine2: 0.5 + 9 / 60 }; copy(b, 2); lanes.push(b);
    const c = { ...a, wave2: 0.25, tune2: a.tune1 >= 0.5 ? a.tune1 - 0.5 : a.tune1 + 0.5 }; copy(c, 2); lanes.push(c);
  }
  { const n = { ...start, wave3: 1, attack3: 0, decay3: 2 / 15, sustain3: 0, release3: 0, tune3: 1 }; lanes.push(n);
    const r = { ...start, wave1: 0.25, fx: 1 / 3, tune3: 19 / 24 }; lanes.push(r); }
  const scored = lanes.map((u) => ({ u, d: evaluate(u) })).sort((a, b) => a.d.total - b.d.total);
  if (scored[0].d.total < best.d.total) best = { u: { ...scored[0].u }, d: scored[0].d };
  log(`start ${first.total.toFixed(3)} default ${before.total.toFixed(3)} best structure ${best.d.total.toFixed(3)}`);
  // an oscillator that is off (and does not modulate) has no audible registers: leave them alone
  const audible = (u, n) => {
    const k = /[123]$/.test(n) && !n.startsWith('wave') ? +n.slice(-1) : 0;
    if (!k) return true;
    const fx = Math.round(clamp01(u.fx) * 3);
    if (Math.round(clamp01(u[`wave${k}`]) * 4) === 0) return k === 3 && fx > 0 && (n === 'tune3' || n === 'fine3');
    return !(n.startsWith('pw') && Math.round(clamp01(u[`wave${k}`]) * 4) !== 3);
  };
  const mutate = (u, n, sigma) => {
    const p = L[INDEX.sid[n]];
    if (SID_CHOICES[n]) { const c = SID_CHOICES[n]; u[n] = c[Math.floor(rnd() * c.length)]; }
    else if (p.q) { const i = Math.round(clamp01(u[n]) * (p.q - 1)), j = rnd() < 0.7 ? i + (rnd() < 0.5 ? -1 : 1) : Math.floor(rnd() * p.q); u[n] = Math.max(n === 'wave1' ? 1 : 0, Math.min(n === 'wave1' ? 3 : p.q - 1, j)) / (p.q - 1); }
    else u[n] = clamp01(u[n] + sigma * gauss(rnd));
  };
  // 2. (1 + lambda) evolution from the best structure, the runner-up kept for a fifth of the renders
  const polish = Math.min(70, Math.round(budget * 0.25)), stop = Math.max(renders, budget - polish);
  const second = scored.find((x) => JSON.stringify(x.u) !== JSON.stringify(best.u));
  let alt = second ? { u: { ...second.u }, d: second.d } : null;
  let sigma = 0.18, round = 0;
  while (renders < stop) {
    const onAlt = alt && round++ % 5 === 4, from = onAlt ? alt : best;
    let improved = false;
    for (let j = 0; j < 6 && renders < stop; j++) {
      const u = { ...from.u }, names = free.filter((n) => audible(u, n) || n.startsWith('wave')), pm = Math.max(0.1, 2.5 / names.length);
      let moved = false;
      for (const n of names) if (rnd() < pm) { mutate(u, n, sigma); moved = true; }
      if (!moved) mutate(u, names[Math.floor(rnd() * names.length)], sigma);
      const d = evaluate(u);
      if (d.total < from.d.total) { from.u = u; from.d = d; improved = true; }
    }
    if (alt && alt.d.total < best.d.total) { const keep = best; best = alt; alt = keep; }
    if (!onAlt) sigma = improved ? Math.min(0.3, sigma * 1.2) : Math.max(0.02, sigma * 0.85);
  }
  // 3. polish: every audible knob a small step each way, every audible register one setting each way
  for (const step of [0.05, 0.02]) {
    for (const n of free) {
      if (renders >= budget) break;
      if (!audible(best.u, n) || SID_CHOICES[n]) continue;
      const p = L[INDEX.sid[n]], d = p.q ? 1 / (p.q - 1) : step;
      if (p.q && step !== 0.05) continue;
      for (const s of [-d, d]) { const u = { ...best.u }; u[n] = clamp01(u[n] + s); if (n === 'wave1' && (u[n] < 0.125 || u[n] > 0.875)) continue; if (u[n] !== best.u[n] && consider(u)) break; }
    }
  }
  // 4. level: the render is linear in the layer's gain; what the level knob cannot reach goes into the token's gain
  const u = { ...best.u }, P = L[INDEX.sid.level];
  const x = renderBar({ voice: 'sid', patch: u, pattern: pat }, layer, bpm, nOut); renders++;
  let cr = 0; for (let i = 0; i < x.length; i++) cr += x[i] * x[i]; cr = Math.sqrt(cr / x.length);
  const g0 = phys(P, u.level), want = cr > 0 ? (g0 * T.rms) / cr : g0, top = phys(P, 1) * 0.81;
  const gain = want > top ? want / top : 1;
  u.level = clamp01(unit(P, want / gain));
  const r4 = (v) => Math.round(v * 10000) / 10000;
  const patch = Object.fromEntries(L.map((p) => [p.name, r4(p.q ? Math.round(clamp01(u[p.name]) * (p.q - 1)) / (p.q - 1) : u[p.name])]));
  return {
    voice: 'sid', patch, pattern: pat, ...(gain !== 1 ? { gain: r4(gain) } : {}),
    fit: { loss: r4(best.d.total), mel: r4(best.d.mel), env: r4(best.d.env), flux: r4(best.d.flux), default_loss: r4(before.total), start_loss: r4(first.total),
      budget, renders, ms_per_render: +((Date.now() - t0) / renders).toFixed(1), target_rms: +T.rms.toFixed(5) },
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
  const pattern = JSON.parse(fs.readFileSync(opt('--pattern'), 'utf8')), out = opt('--out');
  if (pattern.silent) { fs.writeFileSync(out, JSON.stringify({ silent: true })); console.log(`${out}: silent`); process.exit(0); }
  const layer = Number.isInteger(pattern.layer) ? pattern.layer : +opt('--layer', 0);
  const bpm = +(pattern.bpm || opt('--bpm', 126));
  const w = readWav(opt('--target'));
  if (w.sr !== SR) throw new Error(`target must be ${SR} Hz, got ${w.sr}`);
  const target = new Float32Array(w.L.length); for (let i = 0; i < target.length; i++) target[i] = 0.5 * (w.L[i] + w.R[i]);
  const weights = opt('--loss', 'full') === 'mel' ? { mel: 1, env: 0, flux: 0 } : undefined;
  const tok = opt('--voice') === 'sid' && layer > 0
    ? fitSid({ layer, target, pattern, bpm, budget: +opt('--budget', 600), seed: +opt('--seed', 0), weights, lag: +opt('--lag', 2), log: (s) => console.error(s) })
    : fitToken({ layer, target, pattern, bpm, budget: +opt('--budget', 1200), seed: +opt('--seed', 0), weights, lag: +opt('--lag', 2), refine: !a.includes('--no-refine'), refineAdd: !a.includes('--no-add'), devices: opt('--device') ? [opt('--device')] : null, log: (s) => console.error(s) });
  for (const k of ['track', 'bar', 'tail_s']) if (pattern[k] != null) tok.fit[k] = pattern[k];
  tok.fit.layer = layer; tok.fit.bpm = bpm;
  fs.writeFileSync(out, JSON.stringify(tok));
  console.log(`${out}: ${tok.voice}${tok.device ? ' ' + tok.device : ''} loss ${tok.fit.loss} (mel ${tok.fit.mel}, env ${tok.fit.env}, flux ${tok.fit.flux}) ${tok.fit.renders} renders, ${tok.fit.ms_per_render} ms/render`);
}
