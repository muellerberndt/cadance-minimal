#!/usr/bin/env node
// Read an acid line's sound, articulation and filter control from its stem, the engine as the model (render and compare).
//
//   node engine/acid_read.mjs JOBS.json OUT_DIR SHARD NSHARDS --tokens TOKENS.json [--block 8] [--keep 0.03]
//
// A job is one passage: {id, wav (the stem from the passage's first bar on, mono or stereo, 44.1 kHz), t0 (the wav's start
// in the recording's time), bars [[t0, t1], ...] in the recording's time, rows (bars x 16: 0 rest, 1 tie, 2 + pitch class,
// the chain's note rows of the layer), octs (bars: MIDI octave, -1 unknown), layer (1 bass, 2 lead)}. TOKENS.json is a
// list of {id, voice: "acid", patch, gain?}.
//
// Per passage:
//   sound    per block of `block` bars, every token plays the --probes (2) bars of the block with the most notes (no accent,
//            no slide) at each of the seven filter levels; its loss is the best level's, summed over the probes; the block
//            takes the lowest, and keeps the token of the block before when that one is within `keep` (a share) of the best
//            (a sound holds over many bars in this music)
//   filter   per bar, in order: the per-bar filter level (MOD_RANGE, -3..3 in thirds), searched from the level of the bar
//            before a step at a time in the direction that lowers the loss; the engine glides the filter control from the
//            previous bar's level across the bar, so each candidate is heard after the level read for the bar before
//   hold     per rest of the row after a sounding step: the rest becomes a tie (the note sounds on) when the loss falls by more
//            than --hold-margin (0.015 of it); the
//            note reader writes rests where a long-gated or legato 303 line still sounds (corpus/acid_tokens.py check)
//   accent, slide   per note: accent toggled, then slide toggled where the next step has a note (--passes times); a change
//            is kept when the loss falls
// Each bar starts from the bar before's holds, accents and slides where both bars have the notes (a line's pattern repeats).
// The loss is engine/fit.mjs's (features.mjs). A tie in the row is played as a slide into the same pitch (acid_fit.mjs
// acidPattern) and is not toggled.
// Writes OUT_DIR/<id>.json: per bar {token, mod (-3..3), row[16] (the reader's row with the holds), accent[16], slide[16],
// holds, loss}, and per block the losses.
import fs from 'node:fs';
import path from 'node:path';
import { Engine } from './engine.mjs';
import { readWav } from './wav.mjs';
import { describe, distance } from './features.mjs';
import { acidPattern } from './acid_fit.mjs';

const SR = 44100;
const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const FALLBACK = { 1: 2, 2: 4 };
const TAIL = 0.25;
const PROBES = +opt('--probes', 2), PASSES = +opt('--passes', 1), CHOOSE_LEVELS = !a.includes('--choose-at-zero'), HOLD_MARGIN = +opt('--hold-margin', 0.015);

// bar 2 of two bars of one token: bar 1 at filter level m0, bar 2 at m1 (the engine glides from m0 to m1 across bar 2)
function render(tok, layer, bpm, pat0, pat1, m0, m1, nOut) {
  const tokens = [[null], [null], [null], [null]]; tokens[layer] = [null, tok];
  const eng = new Engine(SR, { bpm, layers: tokens.map((t) => ({ tokens: t })) }, { fx: false, master: false });
  eng.setTempo(bpm);
  const t = [0, 0, 0, 0]; t[layer] = 1;
  const mod = (m) => { const x = [[0, 0], [0, 0], [0, 0], [0, 0]]; x[layer] = [m / 3, 0]; return x; };
  const pats = (p) => { const x = [null, null, null, null]; x[layer] = p; return x; };
  // bar 1 glides from 0 to m0, bar 2 from m0 to m1; bar 3 holds m1 for the tail
  eng.queueBar(t, mod(m0), pats(pat0)); eng.queueBar(t, mod(m1), pats(pat1)); eng.queueBar(t, mod(m1), pats(pat1));
  const barLen = Math.round((SR * 240) / bpm), n = barLen + nOut, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) out[i] = 0.5 * (L[barLen + i] + R[barLen + i]);
  return out;
}
function loss(T, x) {
  let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return e < 1e-12 ? 99 : distance(T, describe(x, SR), undefined, 2).total;
}

export function readPassage(job, toks, { block = 8, keep = 0.03, log = () => {} } = {}) {
  const w = readWav(job.wav), mono = new Float32Array(w.L.length);
  for (let i = 0; i < mono.length; i++) mono[i] = w.R ? 0.5 * (w.L[i] + w.R[i]) : w.L[i];
  const nb = job.bars.length;
  const cut = (b) => {
    const [t0, t1] = job.bars[b], i0 = Math.max(0, Math.round((t0 - job.t0) * SR)), n = Math.round((t1 - t0 + TAIL) * SR);
    const x = new Float32Array(n); x.set(mono.subarray(i0, Math.min(mono.length, i0 + n)));
    return { x, bpm: 240 / (t1 - t0) };
  };
  const T = [], B = [];
  for (let b = 0; b < nb; b++) { const c = cut(b); B.push(c); T.push(describe(c.x, SR)); }
  const pat = job.rows.map((row, b) => acidPattern(row, job.octs[b], FALLBACK[job.layer]));
  const hasNotes = (b) => pat[b].notes.some((n) => n != null);
  let renders = 0;
  const L = (tok, b, p, m0, m1) => { renders++; return loss(T[b], render(tok, job.layer, B[b].bpm, b > 0 ? pat[b - 1] : p, p, m0, m1, B[b].x.length)); };
  // sound per block
  const blocks = [];
  let prev = null;
  for (let s = 0; s < nb; s += block) {
    const idx = [...Array(Math.min(block, nb - s)).keys()].map((k) => s + k).filter(hasNotes);
    const probe = idx.sort((p, q) => pat[q].notes.filter((n) => n != null).length - pat[p].notes.filter((n) => n != null).length).slice(0, PROBES);
    if (!probe.length) { blocks.push({ start: s, token: prev, losses: null }); continue; }
    const losses = {};
    // each sound at its best filter level on each probe bar (held over the bar), so a brighter sound does not stand in for
    // a higher level (--choose-at-zero: all at level 0)
    const lv = CHOOSE_LEVELS ? [-3, -2, -1, 0, 1, 2, 3] : [0];
    for (const tok of toks) losses[tok.id] = probe.reduce((acc, b) => acc + Math.min(...lv.map((m) => L(tok, b, pat[b], m, m))), 0) / probe.length;
    let best = toks[0].id; for (const tok of toks) if (losses[tok.id] < losses[best]) best = tok.id;
    const chosen = prev != null && losses[prev] != null && losses[prev] <= losses[best] * (1 + keep) ? prev : best;
    blocks.push({ start: s, token: chosen, best, losses: Object.fromEntries(Object.entries(losses).map(([k, v]) => [k, +v.toFixed(4)])) });
    prev = chosen;
  }
  const first = blocks.find((x) => x.token != null);
  for (const x of blocks) if (x.token == null) x.token = first ? first.token : toks[0].id;
  const byId = Object.fromEntries(toks.map((t) => [t.id, t]));
  // filter level per bar, then articulation
  const out = [];
  let m0 = 0;
  for (let b = 0; b < nb; b++) {
    const tok = byId[blocks[Math.floor(b / block)].token];
    if (!hasNotes(b)) { out.push({ token: tok.id, mod: m0, row: job.rows[b].slice(), accent: new Array(16).fill(0), slide: new Array(16).fill(0), holds: 0, loss: null }); continue; }
    // the bar's row: a rest of the reader's row may become a hold (a tie: the note sounds on), decided below
    const read = job.rows[b];
    let row = read.slice();
    const q = b > 0 ? out[b - 1] : null;
    if (q && q.loss != null && q.row) for (let s = 1; s < 16; s++) if (read[s] === 0 && q.row[s] === 1 && row[s - 1] !== 0) row[s] = 1;
    // a pattern from a row, with accents and free slides (those not made by ties) taken over from another pattern
    const build = (r, from) => {
      const x = acidPattern(r, job.octs[b], FALLBACK[job.layer]);
      if (from) for (let s = 0; s < 16; s++) {
        if (x.notes[s] == null) continue;
        x.accent[s] = from.accent[s];
        if (!(r[(s + 1) % 16] === 1) && s < 15 && x.notes[s + 1] != null) x.slide[s] = from.slide[s];
      }
      return x;
    };
    // the articulation starts from the bar before's where both bars have a note (a line's pattern repeats)
    let p = build(row, q && q.loss != null ? { accent: q.accent, slide: q.slide } : null);
    pat[b] = p;
    // the level: from a start, a step at a time in the direction that lowers the loss
    const level = (start) => {
      const seen = new Map(), at = (m) => { if (!seen.has(m)) seen.set(m, L(tok, b, p, m0, m)); return seen.get(m); };
      let best = start;
      for (const dir of [1, -1]) { let m = best; while (m + dir >= -3 && m + dir <= 3 && at(m + dir) < at(m)) m += dir; if (at(m) < at(best)) best = m; }
      return [best, at(best)];
    };
    let [bestM, bestL] = level(m0);
    const holds = () => {                                  // every rest of the reader's row after a sounding step: hold or not
      for (let s = 1; s < 16; s++) {
        if (read[s] !== 0) continue;
        const r2 = row.slice();
        if (r2[s] === 1) r2[s] = 0; else if (r2[s - 1] !== 0) r2[s] = 1; else continue;
        for (let k = s + 1; k < 16 && read[k] === 0 && r2[k] === 1 && r2[k - 1] === 0; k++) r2[k] = 0;   // a hold ends where its chain is cut
        const p2 = build(r2, p);
        const l = L(tok, b, p2, m0, bestM);
        const added = r2[s] === 1;                         // a new hold must lower the loss by HOLD_MARGIN (a share)
        if (added ? l < bestL * (1 - HOLD_MARGIN) : l < bestL) { bestL = l; row = r2; p = p2; pat[b] = p; }
      }
    };
    const toggle = () => {
      const tied = (i) => row[(i + 1) % 16] === 1;
      for (let s = 0; s < 16; s++) {
        if (p.notes[s] == null) continue;
        for (const key of ['accent', 'slide']) {
          if (key === 'slide' && (tied(s) || p.notes[(s + 1) % 16] == null || s === 15)) continue;
          p[key][s] ^= 1;
          const l = L(tok, b, p, m0, bestM);
          if (l < bestL) bestL = l; else p[key][s] ^= 1;
        }
      }
    };
    holds();
    for (let pass = 0; pass < PASSES; pass++) toggle();
    // the level again with the articulation in place; when it moves, one more pass of holds and toggles
    const [m2, l2] = level(bestM);
    if (m2 !== bestM && l2 < bestL) { bestM = m2; bestL = l2; holds(); toggle(); }
    out.push({ token: tok.id, mod: bestM, row: row.slice(), accent: p.accent.slice(), slide: p.slide.map((v, i) => (row[(i + 1) % 16] === 1 ? 0 : v)),
               holds: row.reduce((n, v, i) => n + (v === 1 && read[i] === 0 ? 1 : 0), 0), loss: +bestL.toFixed(4) });
    m0 = bestM;
    if (b % 16 === 15) log(`${job.id}: bar ${b + 1} of ${nb}, ${renders} renders`);
  }
  return { id: job.id, layer: job.layer, block, keep, renders, blocks, bars: out };
}

if (process.argv[1] && path.resolve(process.argv[1]).endsWith('acid_read.mjs')) {
  const [jobsPath, outDir, shard, nshards] = a.filter((x, i) => !x.startsWith('--') && !(i > 0 && a[i - 1].startsWith('--')));
  const toks = JSON.parse(fs.readFileSync(opt('--tokens'), 'utf8'));
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((_, i) => i % +nshards === +shard);
  fs.mkdirSync(outDir, { recursive: true });
  for (const job of jobs) {
    const out = path.join(outDir, job.id + '.json');
    if (fs.existsSync(out)) continue;
    const t0 = Date.now();
    const r = readPassage(job, toks, { block: +opt('--block', 8), keep: +opt('--keep', 0.03), log: (s) => console.log(s) });
    fs.writeFileSync(out, JSON.stringify(r));
    console.log(`${job.id}: ${job.bars.length} bars, ${r.renders} renders, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
}
