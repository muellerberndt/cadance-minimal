#!/usr/bin/env node
// Fit the engine's acid voice (303-style) to bars of acid stems, and score sounds on bars. The acid tokens of the pool
// come from here (corpus/acid_tokens.py makes the jobs, the palette and the pool).
//
//   node engine/acid_fit.mjs fit JOBS.json BARS_DIR OUT_DIR SHARD NSHARDS [--budget 700]
//   node engine/acid_fit.mjs score JOBS.json BARS_DIR TOKENS.json OUT.json SHARD NSHARDS [--role check]
//
// A job is one bar: {job, layer (1 bass, 2 lead), row: 16 ints (0 rest, 1 tie, 2 + pitch class), octave (MIDI octave of
// the bar's notes, -1 unknown), t0, t1, role (fit | check)}; its sound is BARS_DIR/<job>.wav (the stem's bar plus a quarter
// second). The loss is fit.mjs's (features.mjs: log-mel at two resolutions, loudness envelope, onset flux, both sounds at
// equal level); the candidate plays two bars dry and its second bar is compared.
//
// fit: the read row as an acid pattern (a tie holds the note before it: a slide into the same pitch), then random
// patches, (1 + 8) evolution of the ten acid controls and the octave, accent and slide toggled per note, the patch
// search resumed, a coordinate polish, the level solved last. Writes OUT_DIR/<job>.json with the patch, the pattern
// (notes, accent, slide) and the loss.
// score: per bar of a role, every token of TOKENS.json ({id, voice, patch, gain?, kind}) plays the bar's row as the
// player plays it (brain/common.py synth_pattern: an acid token plays no accent and no slide, a tie holds the note; a
// synth token plays the notes at velocity 0.85, a tie lengthens the note before it); each token gets its best of the
// seven per-bar filter levels (MOD_RANGE: -1..1 in steps of 1/3), searched for the eight best tokens of each kind.
// --articulate: also the three best acid tokens with the articulation the reader gives a bar (engine/acid_read.mjs):
// hold (a rest after a sounding step becomes a tie), then accent and slide toggled per note, each kept when the loss falls. Writes per bar the best of each kind with the loss and its parts (mel,
// env: loudness envelope, flux) and every token's loss.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Engine, PARAMS, INDEX, normPattern, phys, unit } from './engine.mjs';
import { readWav } from './wav.mjs';
import { describe, distance } from './features.mjs';

const SR = 44100;
const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
function mulberry32(s) { return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(rnd) { const u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

// two bars of one token on one layer, dry; the second bar plus the tail, mono. mod: the layer's filter level (-1..1)
export function render(tok, layer, bpm, nOut, mod = 0) {
  const tokens = [[null], [null], [null], [null]]; tokens[layer] = [null, tok];
  const eng = new Engine(SR, { bpm, layers: tokens.map((t) => ({ tokens: t })) }, { fx: false, master: false });
  eng.setTempo(bpm);
  const t = [0, 0, 0, 0]; t[layer] = 1;
  const m = [[0, 0], [0, 0], [0, 0], [0, 0]]; m[layer] = [mod, 0];
  for (let i = 0; i < 3; i++) eng.queueBar(t, m);
  const barLen = Math.round((SR * 240) / bpm), n = barLen + nOut, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) out[i] = 0.5 * (L[barLen + i] + R[barLen + i]);
  return out;
}
export function lossOf(T, x) {
  let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return e < 1e-12 ? { total: 99, mel: 99, env: 99, flux: 99 } : distance(T, describe(x, SR), undefined, 2);
}

// the MIDI notes of a read row: pitch class + the bar's octave (the reader's octave is the MIDI octave, C4 = 60 -> 4)
export function notesOf(row, octave, fallback) {
  const o = octave >= 0 ? octave : fallback;
  return row.map((v) => (v >= 2 ? 12 * (o + 1) + (v - 2) : null));
}
// a row as an acid pattern: a tie after a note becomes a slide into the same pitch (the acid voice holds a note only
// into a next note); articulations default to none
export function acidPattern(row, octave, fallback) {
  const notes = notesOf(row, octave, fallback), slide = new Array(16).fill(0), accent = new Array(16).fill(0);
  let last = null;
  for (let i = 0; i < 16; i++) {
    if (row[i] >= 2) last = i;
    else if (row[i] === 1 && last !== null) { notes[i] = notes[last]; slide[i - 1] = 1; last = i; }
    else last = null;
  }
  return { notes, accent, slide };
}
// a row as the player gives it to a synth token (brain/common.py synth_pattern without the bar-line hold)
export function synthPattern(row, octave, fallback) {
  const notes = notesOf(row, octave, fallback), vel = new Array(16).fill(0), len = new Array(16).fill(1);
  let last = null;
  for (let i = 0; i < 16; i++) {
    if (row[i] >= 2) { vel[i] = 0.85; last = i; }
    else if (row[i] === 1 && last !== null) len[last] += 1;
    else last = null;
  }
  return { notes, vel, len };
}

const FREE = ['wave', 'cutoff', 'reso', 'envmod', 'decay', 'accent_amt', 'drive', 'glide', 'gate'];
const OCT = [0.5, 0, 1];                         // tune: as read, an octave down, an octave up
const baseU = () => { const u = {}; for (const p of PARAMS.acid) u[p.name] = p.def; u.pan = 0.5; u.send_delay = 0; u.send_reverb = 0; return u; };

export function fitAcid(T, bpm, nOut, layer, pattern, budget, seed) {
  const rnd = mulberry32(seed + 1);
  let renders = 0;
  const pat = JSON.parse(JSON.stringify(pattern));
  const evaluate = (u, p = pat) => { renders++; return lossOf(T, render({ voice: 'acid', patch: u, pattern: p }, layer, bpm, nOut)); };
  let best = null;
  const consider = (u, d) => { if (!best || d.total < best.d.total) { best = { u: { ...u }, d }; return true; } return false; };
  for (const o of OCT) { const u = baseU(); u.tune = o; consider(u, evaluate(u)); }
  const nRand = Math.round(budget * 0.3);
  for (let i = 0; renders < nRand; i++) { const u = baseU(); u.tune = OCT[i % 3]; for (const k of FREE) u[k] = rnd(); consider(u, evaluate(u)); }
  // (1 + 8) evolution
  const evo = (stop, sigma0) => {
    let sigma = sigma0; const pm = 3 / FREE.length;
    while (renders < stop) {
      let improved = false;
      for (let j = 0; j < 8 && renders < stop; j++) {
        const u = { ...best.u }; let moved = false;
        for (const k of FREE) if (rnd() < pm) { u[k] = clamp01(u[k] + sigma * gauss(rnd)); moved = true; }
        if (!moved) { const k = FREE[Math.floor(rnd() * FREE.length)]; u[k] = clamp01(u[k] + sigma * gauss(rnd)); }
        if (rnd() < 0.03) u.tune = OCT[Math.floor(rnd() * 3)];
        if (consider(u, evaluate(u))) improved = true;
      }
      sigma = improved ? Math.min(0.35, sigma * 1.25) : Math.max(0.01, sigma * 0.82);
    }
  };
  evo(Math.round(budget * 0.6), 0.2);
  // articulations: accent and slide toggled per note, kept where the loss falls
  let toggles = 0;
  for (let pass = 0; pass < 2; pass++) for (let s = 0; s < 16; s++) {
    if (pat.notes[s] == null) continue;
    for (const key of ['accent', 'slide']) {
      if (key === 'slide' && pat.notes[(s + 1) % 16] == null) continue;
      pat[key][s] ^= 1;
      if (consider(best.u, evaluate(best.u))) toggles++; else pat[key][s] ^= 1;
    }
  }
  evo(budget - 2 * FREE.length, 0.1);
  for (const step of [0.05, 0.02]) for (const k of FREE) for (const sg of [-step, step]) {
    const u = { ...best.u }; u[k] = clamp01(u[k] + sg);
    if (consider(u, evaluate(u))) break;
  }
  // level: the render is linear in the layer's gain; what the knob cannot reach goes into the token's gain
  const P = PARAMS.acid[INDEX.acid.level], u = { ...best.u };
  const x = render({ voice: 'acid', patch: u, pattern: pat }, layer, bpm, nOut); renders++;
  let cr = 0; for (let i = 0; i < x.length; i++) cr += x[i] * x[i]; cr = Math.sqrt(cr / x.length);
  const g0 = phys(P, u.level), want = cr > 0 ? (g0 * T.rms) / cr : g0, top = phys(P, 1) * 0.81, gain = want > top ? want / top : 1;
  u.level = clamp01(unit(P, want / gain));
  const r4 = (v) => Math.round(v * 10000) / 10000;
  return { voice: 'acid', patch: Object.fromEntries(Object.entries(u).map(([k, v]) => [k, r4(v)])), pattern: pat, ...(gain !== 1 ? { gain: r4(gain) } : {}),
           loss: r4(best.d.total), mel: r4(best.d.mel), env: r4(best.d.env), flux: r4(best.d.flux), toggles, renders };
}

function target(dir, job) {
  const w = readWav(path.join(dir, job.job + '.wav'));
  const x = new Float32Array(w.L.length);
  for (let i = 0; i < x.length; i++) x[i] = w.R ? 0.5 * (w.L[i] + w.R[i]) : w.L[i];
  return { x, bpm: 240 / (job.t1 - job.t0) };
}
const seedOf = (s) => [...s].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7);
const FALLBACK = { 1: 2, 2: 4 };                 // octave when the reader gave none: bass C2.., lead C4..

const MAIN = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
const mode = MAIN ? a[0] : null;
if (mode === 'fit') {
  const [, jobsPath, barsDir, outDir, shard, nshards] = a, budget = +opt('--budget', 700);
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((j) => j.role === 'fit').filter((_, i) => i % +nshards === +shard);
  fs.mkdirSync(outDir, { recursive: true });
  for (const job of jobs) {
    const out = path.join(outDir, job.job + '.json');
    if (fs.existsSync(out) || !fs.existsSync(path.join(barsDir, job.job + '.wav'))) continue;
    const t0 = Date.now(), { x, bpm } = target(barsDir, job), T = describe(x, SR);
    const f = fitAcid(T, bpm, x.length, job.layer, acidPattern(job.row, job.octave, FALLBACK[job.layer]), budget, seedOf(job.job));
    fs.writeFileSync(out, JSON.stringify({ job: job.job, passage: job.passage, layer: job.layer, bpm: +bpm.toFixed(2), target_rms: +T.rms.toFixed(5),
                                           seconds: +((Date.now() - t0) / 1000).toFixed(1), ...f }));
    console.log(`${job.job}: loss ${f.loss} (mel ${f.mel}, env ${f.env}, flux ${f.flux}), ${f.toggles} toggles, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
} else if (mode === 'score') {
  const [, jobsPath, barsDir, tokPath, outPath, shard, nshards] = a, role = opt('--role', 'check');
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((j) => j.role === role).filter((_, i) => i % +nshards === +shard);
  const toks = JSON.parse(fs.readFileSync(tokPath, 'utf8'));
  const levels = [-1, -2 / 3, -1 / 3, 1 / 3, 2 / 3, 1];
  const rows = [];
  for (const job of jobs) {
    if (!fs.existsSync(path.join(barsDir, job.job + '.wav'))) continue;
    const t0 = Date.now(), { x, bpm } = target(barsDir, job), T = describe(x, SR), losses = {}, best = {};
    const patternOf = (tok) => {
      const pattern = tok.voice === 'acid' ? acidPattern(job.row, job.octave, FALLBACK[job.layer]) : synthPattern(job.row, job.octave, FALLBACK[job.layer]);
      if (tok.voice === 'acid') { pattern.accent.fill(0); pattern.slide = pattern.slide.map((v, i) => (job.row[(i + 1) % 16] === 1 ? v : 0)); }
      return pattern;
    };
    const play = (tok, mod, pattern = patternOf(tok)) => lossOf(T, render({ voice: tok.voice, patch: tok.patch, gain: tok.gain, pattern }, job.layer, bpm, x.length, mod));
    const r4 = (d) => ({ loss: +d.total.toFixed(4), mel: +d.mel.toFixed(4), env: +d.env.toFixed(4), flux: +d.flux.toFixed(4) });
    for (const tok of toks) losses[tok.id] = [play(tok, 0), 0];
    for (const kind of [...new Set(toks.map((t) => t.kind))]) {
      const top = toks.filter((t) => t.kind === kind).sort((p, q) => losses[p.id][0].total - losses[q.id][0].total).slice(0, 8);
      for (const tok of top) for (const m of levels) { const d = play(tok, m); if (d.total < losses[tok.id][0].total) losses[tok.id] = [d, Math.round(m * 3)]; }
      const ranked = toks.filter((t) => t.kind === kind).sort((p, q) => losses[p.id][0].total - losses[q.id][0].total);
      if (ranked[0]) best[kind] = { id: ranked[0].id, mod: losses[ranked[0].id][1], ...r4(losses[ranked[0].id][0]) };
      // the acid tokens with the articulation the reader gives a bar: per note accent, then slide toggled (one pass)
      // for the three best at their filter level
      if (kind === 'acid' && a.includes('--articulate')) {
        let art = null;
        for (const tok of ranked.slice(0, 3)) {
          const m = losses[tok.id][1] / 3;
          let row = job.row.slice(), p = patternOf({ voice: 'acid' }), d = losses[tok.id][0];
          const mk = (r) => { const q = acidPattern(r, job.octave, FALLBACK[job.layer]); q.accent.fill(0); q.slide = q.slide.map((v, i) => (r[(i + 1) % 16] === 1 ? v : 0)); return q; };
          // hold: a rest after a sounding step becomes a tie (the note sounds on), kept when the loss falls
          for (let s = 1; s < 16; s++) {
            if (row[s] !== 0 || row[s - 1] === 0) continue;
            const r2 = row.slice(); r2[s] = 1;
            const e = play(tok, m, mk(r2));
            if (e.total < d.total) { d = e; row = r2; }
          }
          p = mk(row);
          for (let s = 0; s < 16; s++) {
            if (p.notes[s] == null) continue;
            for (const key of ['accent', 'slide']) {
              if (key === 'slide' && (s === 15 || p.notes[s + 1] == null || row[s + 1] === 1)) continue;
              p[key][s] ^= 1;
              const e = play(tok, m, p);
              if (e.total < d.total) d = e; else p[key][s] ^= 1;
            }
          }
          if (!art || d.total < art.d.total) art = { id: tok.id, mod: losses[tok.id][1], d, accent: p.accent.slice(), slide: p.slide.slice(), row };
        }
        if (art) best.acid_articulated = { id: art.id, mod: art.mod, ...r4(art.d), accent: art.accent, slide: art.slide, row: art.row };
      }
    }
    for (const k in losses) losses[k] = [+losses[k][0].total.toFixed(4), losses[k][1]];
    rows.push({ job: job.job, passage: job.passage, source: job.source, held_out: job.held_out, best, losses });
    console.log(`${job.job}: ${Object.entries(best).map(([k, v]) => `${k} ${v.id} ${v.loss} (mod ${v.mod})`).join(', ')} ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
  fs.writeFileSync(outPath, JSON.stringify(rows));
} else if (mode) {
  console.error('usage: acid_fit.mjs fit|score ... (see the head of the file)'); process.exit(2);
}
