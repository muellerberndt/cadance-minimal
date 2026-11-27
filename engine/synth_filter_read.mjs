#!/usr/bin/env node
// The filter level of a synth voice per bar, read from its signal with the engine as the model (render and compare), as
// engine/acid_read.mjs reads the acid filter.
//
//   node engine/synth_filter_read.mjs JOBS.json OUT_DIR SHARD NSHARDS [--load-limit L]
//
// With --load-limit a job starts only while the machine's 1-minute load is at most L (checked every 30 s).
// A job is one voice of one passage: {id, wav (the voice's signal: the bass stem, or the lead or air part of "other",
// 44.1 kHz), t0 (the wav's start in the reading's time), bars [[t0, t1], ...], rows (bars x 16: 0 rest, 1 tie, 2 + pitch
// class), octs (bars: MIDI octave, -1 unknown), layer (1 bass, 2 lead, 3 air), tokens (per bar the sound: {voice, patch,
// gain} or null where the voice is silent)}.
// Per sounding bar the bar's sound plays the bar's row (brain/common.py synth_pattern, as the player plays it) at each of
// the seven filter levels, held over the bar and the bar before (no glide), without effects; the loss of each level is
// engine/features.mjs's distance to the signal's bar (both scaled to the same loudness). The levels are decoded from the
// losses afterwards (label_benchmark/controls_bench.py: a change penalty chosen on known clips), so no level depends on
// the one read before it.
// Writes OUT_DIR/<id>.json: {id, layer, bars: [{losses: [7] or null}]}.
// A job with "filter" (per bar the filter level read for it) reads the space level instead: the bar at its filter level
// and each of the seven space levels (the reverb and delay sends), with the engine's effects on (FX_DEFAULTS, or the
// job's "fx"), the bar before at the same levels so the tails are heard.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Engine } from './engine.mjs';
import { readWav } from './wav.mjs';
import { describe, distance } from './features.mjs';
import { synthPattern } from './acid_fit.mjs';

const SR = 44100;
const FALLBACK = { 1: 2, 2: 4, 3: 4 };
const TAIL = 0.25;
const LEVELS = [-3, -2, -1, 0, 1, 2, 3];

// bar 2 of two bars of one sound, both at filter level m (and space level sp, with the effects on when fx is given)
function render(tok, layer, bpm, pat0, pat1, m, nOut, sp = null, fx = null) {
  const tokens = [[null], [null], [null], [null]]; tokens[layer] = [null, tok];
  const pool = { bpm, layers: tokens.map((t) => ({ tokens: t })) };
  if (fx) pool.fx = fx;
  const eng = new Engine(SR, pool, { fx: sp != null, master: false });
  eng.setTempo(bpm);
  const t = [0, 0, 0, 0]; t[layer] = 1;
  const x = [[0, 0], [0, 0], [0, 0], [0, 0]]; x[layer] = [m / 3, sp != null ? sp / 3 : 0];
  const pats = (p) => { const y = [null, null, null, null]; y[layer] = p; return y; };
  eng.queueBar(t, x, pats(pat0)); eng.queueBar(t, x, pats(pat1)); eng.queueBar(t, x, pats(pat1));
  const barLen = Math.round((SR * 240) / bpm), n = barLen + nOut, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) out[i] = 0.5 * (L[barLen + i] + R[barLen + i]);
  return out;
}

export function readVoice(job) {
  const w = readWav(job.wav), mono = new Float32Array(w.L.length);
  for (let i = 0; i < mono.length; i++) mono[i] = w.R ? 0.5 * (w.L[i] + w.R[i]) : w.L[i];
  const out = [];
  let renders = 0;
  for (let b = 0; b < job.bars.length; b++) {
    const tok = job.tokens[b], row = job.rows[b];
    if (!tok || !row.some((v) => v > 0)) { out.push({ losses: null }); continue; }
    const [t0, t1] = job.bars[b], i0 = Math.max(0, Math.round((t0 - job.t0) * SR)), n = Math.round((t1 - t0 + TAIL) * SR);
    const x = new Float32Array(n); x.set(mono.subarray(i0, Math.min(mono.length, i0 + n)));
    let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
    if (e < 1e-10) { out.push({ losses: null }); continue; }
    const T = describe(x, SR), bpm = 240 / (t1 - t0);
    const p1 = synthPattern(row, job.octs[b], FALLBACK[job.layer]);
    const p0 = b > 0 && job.rows[b - 1].some((v) => v > 0) ? synthPattern(job.rows[b - 1], job.octs[b - 1], FALLBACK[job.layer]) : p1;
    const space = Array.isArray(job.filter);
    if (space && job.filter[b] == null) { out.push({ losses: null }); continue; }
    const losses = LEVELS.map((m) => {
      renders++;
      const y = space ? render(tok, job.layer, bpm, p0, p1, job.filter[b], n, m, job.fx || null) : render(tok, job.layer, bpm, p0, p1, m, n);
      let s = 0; for (let i = 0; i < y.length; i++) s += y[i] * y[i];
      return s < 1e-12 ? 99 : +distance(T, describe(y, SR), undefined, 2).total.toFixed(5);
    });
    out.push({ losses });
  }
  return { id: job.id, layer: job.layer, renders, bars: out };
}

if (process.argv[1] && path.resolve(process.argv[1]).endsWith('synth_filter_read.mjs')) {
  const args = process.argv.slice(2), li = args.indexOf('--load-limit'), limit = li >= 0 ? +args[li + 1] : Infinity;
  const [jobsPath, outDir, shard, nshards] = args.filter((x, i) => !x.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
  const pause = new Int32Array(new SharedArrayBuffer(4));
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((_, i) => i % +nshards === +shard);
  fs.mkdirSync(outDir, { recursive: true });
  for (const job of jobs) {
    const out = path.join(outDir, job.id + '.json');
    if (fs.existsSync(out)) continue;
    while (os.loadavg()[0] > limit) Atomics.wait(pause, 0, 0, 30000);
    const t0 = Date.now();
    const r = readVoice(job);
    fs.writeFileSync(out, JSON.stringify(r));
    console.log(`${job.id}: ${job.bars.length} bars, ${r.renders} renders, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  }
}
