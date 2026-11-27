#!/usr/bin/env node
// Fit-quality comparison: loss variants and budgets on known pool_v0 tokens (parameter recovery)
// and on real stem bars (neutral metric only). Single process; prints one JSON line per fit.
//   node engine/fit_eval.mjs [replica_dir] [--budgets 300,900] [--quick]
import fs from 'node:fs';
import path from 'node:path';
import { fitToken, renderBar } from './fit.mjs';
import { readWav } from './wav.mjs';
import { logmel, envelope, rms } from './features.mjs';

const SR = 44100, here = path.dirname(new URL(import.meta.url).pathname);
const pool = JSON.parse(fs.readFileSync(path.join(here, '..', 'pool', 'pool_v0.json'), 'utf8'));
const args = process.argv.slice(2), replica = args.find((a) => !a.startsWith('--'));
const budgets = (args.includes('--budgets') ? args[args.indexOf('--budgets') + 1] : '300,900').split(',').map(Number);

// neutral metric: log-mel at a resolution no loss uses (1024/256, 40 bands) + envelope dB/20, level-normalized
function neutral(a, b) {
  const na = rms(a) || 1, nb = rms(b) || 1, x = a.map((v) => v / na), y = b.map((v) => v / nb);
  const A = logmel(x, SR, 1024, 256, 40), B = logmel(y, SR, 1024, 256, 40);
  let s = 0; const n = Math.min(A.mel.length, B.mel.length); for (let i = 0; i < n; i++) s += Math.abs(A.mel[i] - B.mel[i]);
  const ea = envelope(x, 64), eb = envelope(y, 64); let e = 0; for (let i = 0; i < 64; i++) e += Math.abs(ea[i] - eb[i]);
  return +(s / n + e / 64 / 20).toFixed(4);
}
const variants = [{ name: 'full', weights: undefined }, { name: 'mel', weights: { mel: 1, env: 0, flux: 0 } }];
const jobs = [];
const known = args.includes('--quick') ? [[1, 2], [3, 1]] : [[0, 1], [0, 6], [1, 2], [2, 5], [3, 1], [3, 4]];
for (const [l, t] of known) {
  const tok = pool.layers[l].tokens[t], n = Math.round((SR * 240) / 126) + 4410;
  jobs.push({ kind: 'known', id: `L${l}T${t}`, layer: l, target: renderBar(tok, l, 126, n), pattern: tok.pattern, bpm: 126, truth: tok });
}
if (replica) {
  for (const l of [0, 1, 2, 3]) {
    const files = fs.readdirSync(replica).filter((f) => f.startsWith(`L${l}_B`) && f.endsWith('.pattern.json')).sort();
    for (const f of files) {
      const pat = JSON.parse(fs.readFileSync(path.join(replica, f), 'utf8'));
      if (pat.silent) continue;
      const w = readWav(path.join(replica, f.replace('.pattern.json', '.wav')));
      const x = new Float32Array(w.L.length); for (let i = 0; i < x.length; i++) x[i] = 0.5 * (w.L[i] + w.R[i]);
      if (rms(x) < 1e-3) continue;
      jobs.push({ kind: 'real', id: f.replace('.pattern.json', ''), layer: l, target: x, pattern: pat, bpm: pat.bpm || 126 });
      break;
    }
  }
}
const runs = [...budgets.map((b) => ({ v: variants[0], b })), { v: variants[1], b: budgets[0] }];
for (const j of jobs) for (const r of runs) {
  const t0 = Date.now();
  const fit = fitToken({ layer: j.layer, target: j.target, pattern: j.pattern, bpm: j.bpm, budget: r.b, seed: 0, weights: r.v.weights });
  const x = renderBar(fit, j.layer, j.bpm, j.target.length);
  const row = { kind: j.kind, id: j.id, loss: r.v.name, budget: r.b, voice: fit.voice + (fit.device ? ':' + fit.device : ''), neutral: neutral(j.target, x), fit_loss: fit.fit.loss, s: (Date.now() - t0) / 1000 };
  if (j.truth) {
    row.truth = j.truth.voice + (j.truth.device ? ':' + j.truth.device : '');
    const keys = Object.keys(fit.patch).filter((k) => !['level', 'pan', 'send_delay', 'send_reverb'].includes(k) && (j.layer > 0 || k === 'drive' || Object.keys(fit.pattern.hits || {}).some((v) => k.startsWith(v + '_'))));
    row.param_err = +(keys.reduce((s, k) => s + Math.abs(fit.patch[k] - (j.truth.patch[k] ?? 0.5)), 0) / keys.length).toFixed(3);
  }
  console.log(JSON.stringify(row));
}
