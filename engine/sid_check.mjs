#!/usr/bin/env node
// Checks of the SID-style chip voice: every waveform sounds at the written pitch, the filter follows
// its cutoff, mode and sweep, the envelopes follow the chip's rate nibbles, ring and sync change the
// sound, registers switch at a bar start while knobs glide, and the level is linear.
//   node engine/sid_check.mjs
import fs from 'node:fs';
import { Engine, PARAMS, INDEX, SID_WAVES, sidRegisters } from './engine.mjs';
import { centroid, rms } from './features.mjs';

const SR = 44100, BPM = 120, BAR = Math.round((SR * 240) / BPM), STEP = BAR / 16;
let failed = 0;
const check = (name, ok, detail) => { if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`); };
const NOTE = (midi, len = 8) => ({ notes: [midi, ...Array(15).fill(null)], vel: [1, ...Array(15).fill(0)], len: [len, ...Array(15).fill(1)] });
const tok = (patch, pattern = NOTE(57)) => ({ voice: 'sid', patch: { level: 0.5, ...patch }, pattern });

// dry mono render of `bars` bars of tokens on the bass layer; `mods` is one [filter, space] per bar
function render(tokens, { mods = null, bars = tokens.length } = {}) {
  const eng = new Engine(SR, { bpm: BPM, layers: [{ tokens: [null] }, { tokens: [null, ...tokens] }, { tokens: [null] }, { tokens: [null] }] }, { fx: false, master: false });
  eng.setTempo(BPM);
  for (let b = 0; b < bars; b++) eng.queueBar([0, Math.min(b, tokens.length - 1) + 1, 0, 0], mods ? [[0, 0], mods[b], [0, 0], [0, 0]] : null);
  const n = bars * BAR, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const x = new Float32Array(n); let bad = 0;
  for (let i = 0; i < n; i++) { x[i] = 0.5 * (L[i] + R[i]); if (!Number.isFinite(x[i])) bad++; }
  return { x, bad, eng };
}
const part = (x, fromStep, toStep) => x.subarray(Math.round(fromStep * STEP), Math.round(toStep * STEP));

// fundamental by autocorrelation between 40 and 2000 Hz
function pitch(x) {
  let best = 0, at = 0, e = 0;
  for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  for (let lag = Math.floor(SR / 2000); lag <= Math.floor(SR / 40); lag++) {
    let s = 0; for (let i = 0; i + lag < x.length; i++) s += x[i] * x[i + lag];
    if (s > best * 1.02) { best = s; at = lag; }
  }
  return { hz: at ? SR / at : 0, strength: e ? best / e : 0 };
}

// 0. the parameter list other code builds on: engine/sid_params.json says what the engine holds
{
  const doc = JSON.parse(fs.readFileSync(new URL('./sid_params.json', import.meta.url), 'utf8'));
  const same = doc.params.length === PARAMS.sid.length && doc.params.every((d, i) => { const p = PARAMS.sid[i]; return d.name === p.name && d.map === p.map && d.lo === p.lo && d.hi === p.hi && Math.abs(d.def - p.def) < 1e-5 && (d.q || 0) === (p.q || 0); });
  check('parameter list', same && PARAMS.sid.length === 36, `${PARAMS.sid.length} controls, ${PARAMS.sid.filter((p) => p.q).length} of them registers with a fixed number of settings; sid_params.json ${same ? 'matches' : 'DIFFERS'}`);
}
// 1. every waveform: finite, audible, at the written pitch (noise: audible, no clear pitch)
for (const [w, name] of SID_WAVES.entries()) {
  if (!w) continue;
  const { x, bad } = render([tok({ wave1: w / 4, sustain1: 1, route: 0 })]);
  const seg = part(x, 1, 7), p = pitch(seg), level = rms(seg);
  const ok = bad === 0 && level > 0.01 && (name === 'noise' ? p.strength < 0.5 : Math.abs(p.hz / 220 - 1) < 0.03);
  check(`waveform ${name}`, ok, `rms ${level.toFixed(3)}, ${name === 'noise' ? `periodicity ${p.strength.toFixed(2)} (below 0.5)` : `${p.hz.toFixed(1)} Hz for A3 220 Hz`}, nonfinite ${bad}`);
}
// an oscillator that is off is silent; the default patch plays oscillator 1 as a pulse
{
  const off = render([tok({ wave1: 0 })]), def = render([{ voice: 'sid', patch: {}, pattern: NOTE(57) }]);
  const r = sidRegisters({});
  check('off and default', rms(off.x) < 1e-6 && rms(part(def.x, 0, 4)) > 0.005 && SID_WAVES[r.osc[0].wave] === 'pulse' && r.osc[1].wave === 0 && r.osc[2].wave === 0,
    `off rms ${rms(off.x).toExponential(1)}, default rms ${rms(part(def.x, 0, 4)).toFixed(3)}, default waves ${r.osc.map((o) => SID_WAVES[o.wave]).join(' / ')}`);
}
// 2. tuning: +12 semitones doubles the pitch, a second oscillator a fifth above adds its own partial
{
  const up = pitch(part(render([tok({ wave1: 0.5, tune1: 1, sustain1: 1, route: 0 })]).x, 1, 7)).hz;
  const down = pitch(part(render([tok({ wave1: 0.5, tune1: 0, sustain1: 1, route: 0 })]).x, 1, 7)).hz;
  check('tune by semitones', Math.abs(up / 440 - 1) < 0.03 && Math.abs(down / 110 - 1) < 0.03, `+12 gives ${up.toFixed(1)} Hz (440), -12 gives ${down.toFixed(1)} Hz (110)`);
}
// 3. the filter: the lowpass opens with the cutoff, the modes order low < band < high, resonance stays finite
{
  const cents = [0.2, 0.4, 0.6, 0.8, 1].map((c) => centroid(part(render([tok({ wave1: 0.5, sustain1: 1, route: 1 / 7, cutoff: c })]).x, 1, 7), SR));
  check('lowpass follows cutoff', cents.every((c, i) => !i || c > cents[i - 1]), `spectral centroid ${cents.map((c) => Math.round(c)).join(' < ')} Hz`);
  const modes = [0, 1 / 3, 2 / 3, 1].map((m) => render([tok({ wave1: 0.5, sustain1: 1, route: 1 / 7, cutoff: 0.55, mode: m })]));
  const mc = modes.map((r) => centroid(part(r.x, 1, 7), SR));
  check('filter modes', mc[0] < mc[1] && mc[1] < mc[2] && modes.every((r) => r.bad === 0 && rms(r.x) > 1e-3), `lowpass ${Math.round(mc[0])}, bandpass ${Math.round(mc[1])}, highpass ${Math.round(mc[2])}, notch ${Math.round(mc[3])} Hz`);
  const hot = render([tok({ wave1: 0.5, wave2: 0.75, wave3: 1, sustain1: 1, sustain2: 1, sustain3: 1, route: 1, cutoff: 0.5, reso: 1, f_env: 1, f_decay: 0.3 })]);
  let peak = 0; for (const v of hot.x) peak = Math.max(peak, Math.abs(v));
  check('full resonance, three oscillators', hot.bad === 0 && peak < 4, `peak ${peak.toFixed(2)}, nonfinite ${hot.bad}`);
  const through = render([tok({ wave1: 0.5, sustain1: 1, route: 1 / 7, cutoff: 0.3 })]), past = render([tok({ wave1: 0.5, sustain1: 1, route: 0, cutoff: 0.3 })]);
  check('routing', centroid(part(through.x, 1, 7), SR) < 0.5 * centroid(part(past.x, 1, 7), SR), `through the filter ${Math.round(centroid(part(through.x, 1, 7), SR))} Hz, past it ${Math.round(centroid(part(past.x, 1, 7), SR))} Hz`);
}
// 4. the player routine: a filter sweep that falls after the note start, and a pulse-width sweep
{
  const sweep = render([tok({ wave1: 0.5, sustain1: 1, route: 1 / 7, cutoff: 0.35, f_env: 1, f_decay: 0.45 })]).x;
  const early = centroid(part(sweep, 0.1, 1.5), SR), late = centroid(part(sweep, 5, 7), SR);
  check('filter sweep at the note', early > 1.5 * late, `first steps ${Math.round(early)} Hz, later ${Math.round(late)} Hz`);
  const still = render([tok({ wave1: 0.75, sustain1: 1, route: 0, pwm: 0 })]).x, moving = render([tok({ wave1: 0.75, sustain1: 1, route: 0, pwm: 1 })]).x;
  const spread = (x) => { const c = [1, 3, 5].map((s) => centroid(part(x, s, s + 1.5), SR)); return Math.max(...c) / Math.min(...c); };
  check('pulse-width sweep', spread(moving) > spread(still) * 1.1, `centroid range over the note ${spread(moving).toFixed(2)}x with the sweep, ${spread(still).toFixed(2)}x without`);
}
// 5. envelopes by the chip's rate nibbles
{
  const half = (x) => { let pk = 0; for (const v of x) pk = Math.max(pk, Math.abs(v)); let i = 0; while (i < x.length && Math.abs(x[i]) < 0.5 * pk) i++; return (i / SR) * 1000; };
  const fast = half(render([tok({ wave1: 0.5, attack1: 0, sustain1: 1, route: 0 })]).x), slow = half(render([tok({ wave1: 0.5, attack1: 9 / 15, sustain1: 1, route: 0 })]).x);
  check('attack nibble', fast < 5 && slow > 80 && slow < 200, `nibble 0 reaches half level after ${fast.toFixed(1)} ms, nibble 9 (250 ms) after ${slow.toFixed(0)} ms`);
  const tail = (r) => { const x = render([tok({ wave1: 0.5, sustain1: 1, release1: r / 15, route: 0 }, NOTE(57, 2))], { bars: 1 }).x; return rms(part(x, 3, 5)) / (rms(part(x, 0.5, 1.5)) || 1); };
  check('release nibble', tail(0) < 0.01 && tail(9) > 0.2, `level two steps after the gate closes: nibble 0 ${tail(0).toFixed(4)}, nibble 9 ${tail(9).toFixed(3)} of the held level`);
  const sus = (s) => { const x = render([tok({ wave1: 0.5, decay1: 2 / 15, sustain1: s / 15, route: 0 })]).x; return rms(part(x, 5, 7)) / (rms(part(x, 0, 0.3)) || 1); };
  check('sustain nibble', sus(15) > 1.5 * sus(5) && sus(0) < 0.02, `held level against the attack: sustain 15 ${sus(15).toFixed(2)}, 5 ${sus(5).toFixed(2)}, 0 ${sus(0).toFixed(3)}`);
}
// 6. ring and sync: oscillator 3 changes oscillator 1 while its own output is off
{
  const base = { sustain1: 1, route: 0, tune1: 1, tune3: 0.5 };
  const plainTri = render([tok({ ...base, wave1: 0.25 })]), ring = render([tok({ ...base, wave1: 0.25, fx: 1 / 3, tune3: 19 / 24 })]);
  const fifth = { ...base, tune1: 19 / 24 };        // oscillator 1 a fifth above oscillator 3: no whole-number ratio
  const plainSaw = render([tok({ ...fifth, wave1: 0.5 })]), sync = render([tok({ ...fifth, wave1: 0.5, fx: 2 / 3 })]);
  const diff = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / a.length); };
  const ps = pitch(part(sync.x, 1, 7)).hz;
  check('ring modulation', ring.bad === 0 && diff(plainTri.x, ring.x) > 0.01, `differs from the plain triangle by rms ${diff(plainTri.x, ring.x).toFixed(3)}`);
  check('hard sync', sync.bad === 0 && diff(plainSaw.x, sync.x) > 0.01 && Math.abs(ps / 220 - 1) < 0.03, `the synced saw (written a fifth up) repeats at oscillator 3's ${ps.toFixed(1)} Hz (220)`);
}
// 7. registers switch at the bar start, knobs glide across the bar; the pitch slides on request
{
  const a = tok({ wave1: 0.25, cutoff: 0.2, sustain1: 1 }), b = tok({ wave1: 1, cutoff: 1, sustain1: 1 });
  const eng = new Engine(SR, { bpm: BPM, layers: [{ tokens: [null] }, { tokens: [null, a, b] }, { tokens: [null] }, { tokens: [null] }] }, { fx: false, master: false });
  eng.setTempo(BPM); eng.queueBar([0, 1, 0, 0]); eng.queueBar([0, 2, 0, 0]);
  const buf = new Float32Array(BAR + Math.round(BAR / 4)), right = new Float32Array(buf.length);
  eng.process(buf, right);
  const sid = eng.layers[1].sid, c = PARAMS.sid[INDEX.sid.cutoff], lo = c.lo * Math.pow(c.hi / c.lo, 0.2), hi = c.hi;
  check('registers switch, knobs glide', sid.wave[0] === 4 && sid.cut > lo * 1.5 && sid.cut < hi * 0.6, `a quarter into the bar: waveform ${SID_WAVES[sid.wave[0]]}, cutoff ${Math.round(sid.cut)} Hz on its way from ${Math.round(lo)} to ${hi} Hz`);
  // a slide is asked for on the step before the note it leads into, as on the acid voice
  const line = { notes: [45, 57, ...Array(14).fill(null)], vel: [1, 1, ...Array(14).fill(0)], len: [2, 6, ...Array(14).fill(1)], slide: [1, ...Array(15).fill(0)] };
  const slid = render([tok({ wave1: 0.5, sustain1: 1, route: 0 }, line)]).x;
  const jump = render([tok({ wave1: 0.5, sustain1: 1, route: 0 }, { ...line, slide: Array(16).fill(0) })]).x;
  // mean frequency over the 40 ms after the second note, from the saw's rising zero crossings
  const mean = (x) => { const a = Math.round(STEP), n = Math.round(0.04 * SR); let c = 0; for (let i = a + 1; i < a + n; i++) if (x[i - 1] < 0 && x[i] >= 0) c++; return c / 0.04; };
  check('slide', mean(slid) < 190 && mean(slid) > 115 && Math.abs(mean(jump) / 220 - 1) < 0.15, `40 ms after the second note: ${mean(slid).toFixed(0)} Hz while sliding up from 110 Hz, ${mean(jump).toFixed(0)} Hz without`);
}
// 8. FILTER modulation and level
{
  const t = tok({ wave1: 0.5, sustain1: 1, route: 1 / 7, cutoff: 0.5 });
  const closed = centroid(part(render([t, t], { mods: [[-1, 0], [-1, 0]] }).x, 17, 23), SR), open = centroid(part(render([t, t], { mods: [[1, 0], [1, 0]] }).x, 17, 23), SR);
  check('FILTER modulation', open > 1.5 * closed, `centroid ${Math.round(closed)} Hz at -1, ${Math.round(open)} Hz at +1`);
  const quiet = rms(render([tok({ wave1: 0.5, sustain1: 1, level: 0.3 })]).x), loud = rms(render([tok({ wave1: 0.5, sustain1: 1, level: 0.9 })]).x);
  check('level is linear', Math.abs(loud / quiet / 9 - 1) < 0.02, `level 0.9 against 0.3: ${(loud / quiet).toFixed(2)}x (9x by the squared map)`);
}
// 9. speed: three chips, eight bars
{
  const busy = { notes: Array.from({ length: 16 }, (_, i) => 45 + (i % 4) * 3), vel: Array(16).fill(0.9), len: Array(16).fill(1) };
  const t = { voice: 'sid', patch: { wave1: 0.5, wave2: 0.75, wave3: 0.25, tune2: 1, route: 1, cutoff: 0.6, reso: 0.5, f_env: 0.8, pwm: 0.5 }, pattern: busy };
  const eng = new Engine(SR, { bpm: BPM, layers: [{ tokens: [null] }, { tokens: [null, t] }, { tokens: [null, t] }, { tokens: [null, t] }] });
  eng.setTempo(BPM); for (let b = 0; b < 8; b++) eng.queueBar([0, 1, 1, 1]);
  const n = 8 * BAR, L = new Float32Array(n), R = new Float32Array(n), t0 = performance.now();
  eng.process(L, R);
  const ms = (performance.now() - t0) / 8; let bad = 0; for (const v of L) if (!Number.isFinite(v)) bad++;
  check('speed', ms < 500 && bad === 0, `${ms.toFixed(1)} ms per 2 s bar for three chips with effects (${(2000 / ms).toFixed(0)}x real time)`);
}
console.log(failed ? `${failed} check(s) failed` : 'all checks passed');
process.exitCode = failed ? 1 : 0;
