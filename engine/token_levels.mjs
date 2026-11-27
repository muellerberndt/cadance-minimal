// The level of every sound of a pool as the engine plays it: per layer and token, the root mean square over the
// samples where the sound is heard (dB), solo, without effects, master or loudness balance. corpus/level_labels.py
// compares these with the levels measured in the recordings and corrects the taught gains.
//   node engine/token_levels.mjs POOL.json > levels.json
import { readFileSync } from 'node:fs';
import { Engine } from './engine.mjs';

const pool = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const sr = 44100, BARS = 3;
const out = { pool: process.argv[2], bpm: pool.bpm || 126, layers: [] };
for (let layer = 0; layer < 4; layer++) {
  const tokens = (pool.layers[layer] && pool.layers[layer].tokens) || [];
  const row = [];
  for (let t = 0; t < tokens.length; t++) {
    if (!tokens[t]) { row.push(null); continue; }
    const e = new Engine(sr, pool, { fx: false, master: false });
    if (pool.bpm) e.setTempo(pool.bpm);
    const pick = [0, 0, 0, 0]; pick[layer] = t;
    for (let b = 0; b < BARS; b++) e.queueBar(pick, null, null, null);
    const n = Math.round(BARS * 16 * (sr * 15) / e.bpm), skip = Math.round(n / BARS), L = new Float32Array(128), R = new Float32Array(128);
    let sum = 0, count = 0, peak = 0;
    for (let i = 0; i < n; i += 128) {
      e.process(L, R);
      if (i < skip) continue;                           // the first bar settles envelopes and filters
      for (let k = 0; k < 128; k++) { const p = (L[k] * L[k] + R[k] * R[k]) / 2; if (p > 1e-7) { sum += p; count++; } peak = Math.max(peak, Math.abs(L[k]), Math.abs(R[k])); }
    }
    row.push(count ? { db: +(10 * Math.log10(sum / count)).toFixed(2), sounding: +(count / (n - skip)).toFixed(3), peak: +peak.toFixed(4) } : { db: null, sounding: 0, peak: 0 });
  }
  out.layers.push(row);
}
console.log(JSON.stringify(out));
