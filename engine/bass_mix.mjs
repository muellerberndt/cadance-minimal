// The bass in the mix, measured as quality/loop/reference.json measures the recordings: per voice the plain and the weighted
// level over its sounding samples (quality/loop/stem_brightness.py: lows under 100 Hz down, highs over 1.5 kHz up), the bass
// against the drums, and the whole mix's share of power in 20-60 Hz (sub) and 60-250 Hz (low) (quality/loop/measure.py).
//   node engine/bass_mix.mjs TRACK.json [more] --set name=JSON [--set ...] [--bars 64]
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Engine } from './engine.mjs';
import { fft } from './features.mjs';
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2), files = [], sets = [];
let BARS = 64;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--bars') BARS = +args[++i];
  else if (args[i] === '--set') { const [name, ...rest] = args[++i].split('='); sets.push([name, JSON.parse(rest.join('='))]); }
  else files.push(args[i]);
}
const sr = 44100, db = (x) => (x > 0 ? 10 * Math.log10(x) : -120);
function weigher() {
  const k1 = 1 - Math.exp(-2 * Math.PI * 100 / sr), k2 = 1 - Math.exp(-2 * Math.PI * 1500 / sr); let a = 0, b = 0;
  return (x) => { a += k1 * (x - a); const h = x - a; b += k2 * (h - b); return h + 0.585 * (h - b); };
}
function shares(x) {
  const N = 8192, hop = 4096, win = new Float64Array(N), P = new Float64Array(N / 2 + 1), re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
  for (let o = 0; o + N <= x.length; o += hop) { for (let i = 0; i < N; i++) { re[i] = x[o + i] * win[i]; im[i] = 0; } fft(re, im); for (let k = 0; k <= N / 2; k++) P[k] += re[k] * re[k] + im[k] * im[k]; }
  const s = (a, b) => { let t = 0; for (let k = 0; k <= N / 2; k++) { const f = (k * sr) / N; if (f >= a && f < b) t += P[k]; } return t; };
  const tot = s(20, 16000) || 1e-20;
  return { sub: db(s(20, 60) / tot), low: db(s(60, 250) / tot) };
}
function render(track, setting) {
  const pool = JSON.parse(readFileSync(resolve(ROOT, 'pool', track.pool), 'utf8'));
  const e = new Engine(sr, pool, { fx: true, master: true });
  if (track.bpm) e.setTempo(track.bpm);
  if (setting.on !== false) e.setBalance(true, setting.ratio, setting.rangeDb, setting);
  const raw = [0, 0, 0, 0];
  e.layers.forEach((ly, j) => { const tick = ly.tick.bind(ly); ly.tick = () => (raw[j] = tick()); });
  const use = track.bars.slice(0, BARS);
  for (const b of use) e.queueBar(b.tokens, b.mods, b.patterns || null, b.patches || null);
  const n = Math.round(use.length * 16 * (sr * 15) / e.bpm), L = new Float32Array(1), R = new Float32Array(1), mix = new Float32Array(n);
  const W = [weigher(), weigher()], acc = { plain: [0, 0], wsum: [0, 0], cnt: [0, 0] }, tap = [new Float32Array(n), new Float32Array(n)];
  for (let i = 0; i < n; i++) {
    raw.fill(0); e.process(L, R); mix[i] = 0.5 * (L[0] + R[0]);
    for (const [k, j] of [[0, 0], [1, 1]]) {           // drums (layer 0) and bass (layer 1), after the balance's trims
      const s = raw[j] * (e.bal.on ? e.bal.trim[j] * (j ? e.bal.gain[j] : 1) : 1) * (j === 1 ? (e.bassGain || 1) : 1), w = W[k](s);
      tap[k][i] = s;
      if (s * s > 1e-7) { acc.plain[k] += s * s; acc.wsum[k] += w * w; acc.cnt[k]++; }
    }
  }
  const lv = (k, key) => (acc.cnt[k] > sr ? db(acc[key][k] / acc.cnt[k]) : null);
  const sp = shares(mix);
  // the kick's band (45-100 Hz): the drums against the bass over 20 ms frames, at the frames where the drums' band is in its
  // top tenth (the kicks): the median of drums minus bass (dB); above 0 the kick wins its band
  const band = (x) => { const lo = 2 * Math.PI * 45 / sr, hi = 2 * Math.PI * 100 / sr; let a = 0, b = 0, c = 0, d = 0; const y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) { a += hi * (x[i] - a); b += hi * (a - b); c += lo * (b - c); d += lo * (c - d); y[i] = b - d; } return y; };
  const fr = Math.round(0.02 * sr), dB = band(tap[0]), bB = band(tap[1]), F = [];
  for (let o = 0; o + fr <= n; o += fr) { let p = 0, q = 0; for (let i = o; i < o + fr; i++) { p += dB[i] * dB[i]; q += bB[i] * bB[i]; } F.push([p, q]); }
  const top = F.map((f) => f[0]).sort((a, b) => a - b)[Math.floor(F.length * 0.9)], wins = F.filter((f) => f[0] >= top && f[0] > 0).map((f) => db(f[0]) - db(f[1] + 1e-12)).sort((a, b) => a - b);
  const kickOverBass = wins.length ? wins[wins.length >> 1] : null;
  return { bass_vs_drums_plain: lv(1, 'plain') != null ? lv(1, 'plain') - lv(0, 'plain') : null, bass_vs_drums_weighted: lv(1, 'wsum') != null ? lv(1, 'wsum') - lv(0, 'wsum') : null,
           sub: sp.sub, low: sp.low, kick_over_bass: kickOverBass, out: db(mix.reduce((a, v) => a + v * v, 0) / n), peak: mix.reduce((a, v) => Math.max(a, Math.abs(v)), 0) };
}
const f1 = (x) => (x == null ? '  - ' : (x >= 0 ? '+' : '') + x.toFixed(1));
for (const [name, setting] of sets) {
  const rows = files.map((f) => render(JSON.parse(readFileSync(f, 'utf8')), setting));
  const med = (k) => { const v = rows.map((r) => r[k]).filter((x) => x != null).sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; };
  console.log(`${name.padEnd(14)} bass - drums plain ${f1(med('bass_vs_drums_plain'))} weighted ${f1(med('bass_vs_drums_weighted'))} dB | mix sub ${f1(med('sub'))} low ${f1(med('low'))} dB | kick over bass in 45-100 Hz at the kicks ${f1(med('kick_over_bass'))} dB | output ${f1(med('out'))} dB peak ${med('peak').toFixed(2)} | per track ` +
              rows.map((r) => `${f1(r.bass_vs_drums_plain)}/${f1(r.sub)}/${f1(r.low)}`).join(' '));
}
