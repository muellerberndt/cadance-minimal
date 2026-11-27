// The low end of the pool's bass sounds against the recordings they were fitted to. Per bass medoid bar (corpus_out/
// medoids/L1_T<token>.wav: the album bass stem bar each pool bass sound was fitted to, its pattern as read): the share of
// the bar's energy in the sub band (35-70 Hz) and the low band (70-140 Hz), on the recording and on the engine's render
// of the fitted sound playing the same pattern (bar 2 of three, no effects, no master).
//   node engine/bass_bands.mjs MEDOID_DIR POOL.json [--out bass_bands.json]
import fs from 'node:fs';
import path from 'node:path';
import { Engine } from './engine.mjs';
import { readWav } from './wav.mjs';
import { fft } from './features.mjs';

const SR = 44100;
export const BANDS = { sub: [35, 70], low: [70, 140] };
export function bandShares(x, i0 = 0, n = x.length - i0, sr = SR) {
  const N = 8192, hop = 4096, win = new Float64Array(N), P = new Float64Array(N / 2 + 1);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  const re = new Float64Array(N), im = new Float64Array(N);
  let frames = 0;
  for (let o = i0; o + N <= i0 + n || frames === 0; o += hop) {
    for (let i = 0; i < N; i++) { const v = o + i < x.length && o + i < i0 + n ? x[o + i] : 0; re[i] = v * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= N / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k];
    frames++;
    if (o + hop + N > i0 + n) break;
  }
  const hz = (k) => (k * sr) / N, sum = (a, b) => { let s = 0; for (let k = 0; k <= N / 2; k++) if (hz(k) >= a && hz(k) < b) s += P[k]; return s; };
  const tot = sum(20, 16000) || 1e-20, out = {};
  for (const [name, [a, b]] of Object.entries(BANDS)) out[name] = 10 * Math.log10(sum(a, b) / tot + 1e-12);
  return out;
}
function render(tok, pat, bpm) {
  const tokens = [[null], [null], [null], [null]]; tokens[1] = [null, tok];
  const eng = new Engine(SR, { bpm, layers: tokens.map((t) => ({ tokens: t })) }, { fx: false, master: false });
  eng.setTempo(bpm);
  for (let b = 0; b < 3; b++) eng.queueBar([0, 1, 0, 0], [[0, 0], [0, 0], [0, 0], [0, 0]], [null, pat, null, null]);
  const barN = Math.round((SR * 240) / bpm), n = 3 * barN, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const y = new Float32Array(n); for (let i = 0; i < n; i++) y[i] = 0.5 * (L[i] + R[i]);
  return { y, barN };
}
if (process.argv[1] && path.resolve(process.argv[1]).endsWith('bass_bands.mjs')) {
  const [dir, poolPath] = process.argv.slice(2), oi = process.argv.indexOf('--out');
  const pool = JSON.parse(fs.readFileSync(poolPath, 'utf8'));
  const idx = JSON.parse(fs.readFileSync(path.join(dir, 'index.json'), 'utf8')), rows = [];
  for (const m of idx) {
    if (m.layer !== 1) continue;
    const tok = pool.layers[1].tokens[m.token]; if (!tok || !['synth', 'acid'].includes(tok.voice)) continue;
    const wav = path.join(dir, m.wav), pj = wav.replace('.wav', '.pattern.json');
    if (!fs.existsSync(wav) || !fs.existsSync(pj)) continue;
    const pat = JSON.parse(fs.readFileSync(pj, 'utf8')), w = readWav(wav), bpm = m.bpm || pat.bpm;
    const x = new Float32Array(w.L.length); for (let i = 0; i < x.length; i++) x[i] = 0.5 * (w.L[i] + w.R[i]);
    const barN = Math.round((SR * 240) / bpm), rec = bandShares(x, 0, barN);
    const r = render({ voice: tok.voice, patch: tok.patch, gain: tok.gain }, pat, bpm), ren = bandShares(r.y, r.barN, r.barN);
    const notes = pat.notes.filter((v) => v != null), lowest = notes.length ? Math.min(...notes) : null;
    rows.push({ token: m.token, voice: tok.voice, lowest_midi: lowest, recording: rec, render: ren });
  }
  const med = (a) => { const b = a.slice().sort((p, q) => p - q); return b.length ? +b[Math.floor(b.length / 2)].toFixed(2) : null; };
  const summary = {};
  for (const band of Object.keys(BANDS)) summary[band] = { recording: med(rows.map((r) => r.recording[band])), render: med(rows.map((r) => r.render[band])),
    render_minus_recording: med(rows.map((r) => r.render[band] - r.recording[band])), render_under: rows.filter((r) => r.render[band] < r.recording[band] - 1).length };
  summary.bars = rows.length;
  console.log(JSON.stringify(summary, null, 1));
  if (oi > 0) fs.writeFileSync(process.argv[oi + 1], JSON.stringify({ bands: BANDS, summary, bars: rows }));
}
