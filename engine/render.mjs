#!/usr/bin/env node
// Offline render: node engine/render.mjs <score.json> <out.wav> [--layer k] [--tail seconds] [--dry] [--no-samples]
// score = {"pool": "pool.json" (relative to the score file) or an inline pool, "bpm": 126,
//          "bars": [[t0,t1,t2,t3] or {"tokens": [t0,t1,t2,t3], "mods": [[filter, space] x 4]}, ...]}
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Engine } from './engine.mjs';
import { writeWav } from './wav.mjs';
import { loadSamples, useSamples } from './samples.mjs';

export function loadScore(scorePath) {
  const score = JSON.parse(fs.readFileSync(scorePath, 'utf8'));
  const pool = typeof score.pool === 'string' ? JSON.parse(fs.readFileSync(path.resolve(path.dirname(scorePath), score.pool), 'utf8')) : score.pool;
  return { score, pool };
}

export function renderScore(score, pool, { sr = 44100, layer = -1, tail = 3, dry = false } = {}) {
  const eng = new Engine(sr, pool, { fx: !dry, master: !dry });
  const smp = useSamples() ? loadSamples() : null;      // real 606/808/909 one-shots (--no-samples: the models)
  if (smp) eng.setSamples(smp.kit, smp.sampleRate);
  eng.setTempo(score.bpm || pool.bpm || 126);
  if (score.balance && score.balance.on !== false) eng.setBalance(true, score.balance.ratio, score.balance.rangeDb, score.balance);   // the player's loudness balance
  eng.solo = layer;
  for (const b of score.bars) Array.isArray(b) ? eng.queueBar(b) : eng.queueBar(b.tokens, b.mods, b.patterns, b.patches);
  const barLen = (sr * 60 * 4) / eng.bpm, n = Math.round(score.bars.length * barLen + tail * sr);
  const L = new Float32Array(n), R = new Float32Array(n);
  eng.queueBar([0, 0, 0, 0]); // the tail: every layer silent, effects ring out
  for (let o = 0; o < n; o += 4096) eng.process(L.subarray(o, Math.min(n, o + 4096)), R.subarray(o, Math.min(n, o + 4096)));
  return { L, R, sr, eng };
}

export function stats(L, R) {
  let pk = 0, ss = 0, bad = 0;
  for (let i = 0; i < L.length; i++) {
    const a = L[i], b = R[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) { bad++; continue; }
    pk = Math.max(pk, Math.abs(a), Math.abs(b)); ss += a * a + b * b;
  }
  return { peak: pk, rms: Math.sqrt(ss / (2 * L.length)), nonfinite: bad };
}

if (process.argv[1] && fs.existsSync(process.argv[1]) &&
    fs.realpathSync(fileURLToPath(import.meta.url)) === fs.realpathSync(process.argv[1])) {
  const args = process.argv.slice(2), pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && ['--layer', '--tail'].includes(args[i - 1])));
  const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? +args[i + 1] : d; };
  const [scorePath, outPath] = pos;
  const t0 = Date.now(), { score, pool } = loadScore(scorePath);
  const { L, R, sr } = renderScore(score, pool, { layer: opt('--layer', -1), tail: opt('--tail', 3), dry: args.includes('--dry') });
  const s = stats(L, R);
  writeWav(outPath, L, R, sr);
  console.log(`${outPath}: ${(L.length / sr).toFixed(1)} s, peak ${s.peak.toFixed(3)}, rms ${s.rms.toFixed(3)}, nonfinite ${s.nonfinite}, rendered in ${((Date.now() - t0) / 1000).toFixed(2)} s`);
}
