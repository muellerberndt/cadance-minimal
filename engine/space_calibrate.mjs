// The base sends of balance.space measured against the recordings: every synth sound of the pool was fitted dry to one
// bar of an album stem (corpus_out/medoids: L<layer>_T<token>.wav, its pattern as read). Per such bar, the share of its
// energy in the steps where no note of the pattern sounds (the tails a room or a delay leaves in the gaps) is read on the
// recording and on the engine's render of the fitted sound playing the same pattern (bar 2 of three, effects on, the
// sends under balance.space with the high-pass), for a grid of base sends; the send whose render matches the
// recording's gap share best is the bar's. Per voice the median over its bars is the measured base send.
//   node engine/space_calibrate.mjs MEDOID_DIR POOL.json [--out space_calibration.json]
// The send per voice: bass reverb only, lead delay and reverb alike, air reverb only. Bars with fewer than three gap
// steps or a near-silent gap are left out.
import fs from 'node:fs';
import path from 'node:path';
import { Engine } from './engine.mjs';
import { readWav } from './wav.mjs';

const SR = 44100;
const [dir, poolPath] = process.argv.slice(2);
const oi = process.argv.indexOf('--out'), outPath = oi > 0 ? process.argv[oi + 1] : null;
const pool = JSON.parse(fs.readFileSync(poolPath, 'utf8'));
const GRID = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6];
const KIND = { 1: 'reverb', 2: 'both', 3: 'reverb' };

function gapShare(x, i0, barN, pat) {
  const step = barN / 16, cov = new Array(16).fill(false);
  for (let k = 0; k < 16; k++) if (pat.notes[k] != null) for (let q = k; q < Math.min(16, k + Math.max(1, Math.round(pat.len ? pat.len[k] : 1))); q++) cov[q] = true;
  let gap = 0, tot = 0, nGap = 0;
  for (let k = 0; k < 16; k++) {
    let e = 0; const a = i0 + Math.round(k * step), b = i0 + Math.round((k + 1) * step);
    for (let i = a; i < Math.min(b, x.length); i++) e += x[i] * x[i];
    tot += e; if (!cov[k]) { gap += e; nGap++; }
  }
  return nGap >= 3 && tot > 0 ? gap / tot : null;
}

function render(tok, layer, pat, bpm, send) {
  const tokens = [[null], [null], [null], [null]]; tokens[layer] = [null, tok];
  const eng = new Engine(SR, { bpm, layers: tokens.map((t) => ({ tokens: t })) }, { fx: true, master: false });
  eng.setTempo(bpm);
  const d = [0, 0, 0], r = [0, 0, 0];
  if (KIND[layer] !== 'reverb') d[layer - 1] = send;
  r[layer - 1] = send;
  eng.setSpace({ on: true, delay: d, reverb: r });
  const t = [0, 0, 0, 0]; t[layer] = 1;
  const pats = [null, null, null, null]; pats[layer] = pat;
  for (let b = 0; b < 3; b++) eng.queueBar(t, [[0, 0], [0, 0], [0, 0], [0, 0]], pats);
  const barN = Math.round((SR * 240) / bpm), n = 3 * barN, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = 0.5 * (L[i] + R[i]);
  return { y, barN };
}

const idx = JSON.parse(fs.readFileSync(path.join(dir, 'index.json'), 'utf8'));
const per = { 1: [], 2: [], 3: [] }, rows = [];
for (const m of idx) {
  if (!(m.layer >= 1 && m.layer <= 3)) continue;
  const tok = pool.layers[m.layer].tokens[m.token];
  if (!tok || tok.voice !== 'synth') continue;
  const wav = path.join(dir, m.wav), pj = path.join(dir, m.wav.replace('.wav', '.pattern.json'));
  if (!fs.existsSync(wav) || !fs.existsSync(pj)) continue;
  const pat = JSON.parse(fs.readFileSync(pj, 'utf8')), w = readWav(wav), bpm = m.bpm || pat.bpm;
  const x = new Float32Array(w.L.length); for (let i = 0; i < x.length; i++) x[i] = 0.5 * (w.L[i] + w.R[i]);
  const barN = Math.round((SR * 240) / bpm);
  const rec = gapShare(x, 0, barN, pat);
  if (rec == null) continue;
  const t = { voice: tok.voice, patch: tok.patch, gain: tok.gain };
  const shares = GRID.map((s) => { const r = render(t, m.layer, pat, bpm, s); return gapShare(r.y, r.barN, r.barN, pat); });
  if (shares.some((v) => v == null)) continue;
  let best = 0; for (let k = 1; k < GRID.length; k++) if (Math.abs(shares[k] - rec) < Math.abs(shares[best] - rec)) best = k;
  per[m.layer].push(GRID[best]);
  rows.push({ layer: m.layer, token: m.token, recording: +rec.toFixed(4), dry: +shares[0].toFixed(4), at: shares.map((v) => +v.toFixed(4)), send: GRID[best] });
}
const med = (a) => { const b = a.slice().sort((p, q) => p - q); return b.length ? b[Math.floor(b.length / 2)] : null; };
const q = (a, f) => { const b = a.slice().sort((p, q2) => p - q2); return b.length ? b[Math.floor(f * (b.length - 1))] : null; };
const sum = {};
for (const L of [1, 2, 3]) {
  const R = rows.filter((r) => r.layer === L);
  sum[['bass', 'lead', 'air'][L - 1]] = { bars: R.length, send_median: med(per[L]), send_q25: q(per[L], 0.25), send_q75: q(per[L], 0.75),
    gap_recording_median: med(R.map((r) => r.recording)), gap_dry_median: med(R.map((r) => r.dry)), dry_under_recording: R.filter((r) => r.dry < r.recording).length };
}
console.log(JSON.stringify(sum, null, 1));
if (outPath) fs.writeFileSync(outPath, JSON.stringify({ grid: GRID, kind: KIND, summary: sum, bars: rows }));
