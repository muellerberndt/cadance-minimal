// How a composed track sits in the mix: per voice the level over its sounding samples relative to the kit, the
// kit's own level, and the loudness of the whole output, for one or more balance settings.
//   node engine/mix_check.mjs TRACK.json [more tracks] [--bars 64] [--set name=JSON ...]
// A setting is what site/play-reading.json holds under "balance".
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Engine } from './engine.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2), files = [], sets = [];
let BARS = 64;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--bars') BARS = +args[++i];
  else if (args[i] === '--set') { const [name, ...rest] = args[++i].split('='); sets.push([name, JSON.parse(rest.join('='))]); }
  else files.push(args[i]);
}
if (!sets.length) sets.push(['as written', { on: false }]);
const sr = 44100, db = (x) => (x > 0 ? 10 * Math.log10(x) : -120), pools = {};
export function render(track, setting, bars = BARS) {
  const pool = pools[track.pool] || (pools[track.pool] = JSON.parse(readFileSync(resolve(ROOT, 'pool', track.pool), 'utf8')));
  const e = new Engine(sr, pool, { fx: true, master: true });
  if (track.bpm) e.setTempo(track.bpm);
  if (setting.on !== false) e.setBalance(true, setting.ratio, setting.rangeDb, setting);
  const raw = [0, 0, 0, 0];
  e.layers.forEach((ly, j) => { const tick = ly.tick.bind(ly); ly.tick = () => (raw[j] = tick()); });
  const use = track.bars.slice(0, bars);
  for (const b of use) e.queueBar(b.tokens, b.mods, b.patterns || null, b.patches || null);
  const n = Math.round(use.length * 16 * (sr * 15) / e.bpm), L = new Float32Array(1), R = new Float32Array(1);
  const sum = [0, 0, 0, 0], cnt = [0, 0, 0, 0], barLen = Math.round(16 * (sr * 15) / e.bpm), loud = [];
  let out = 0, peak = 0, acc = 0;
  for (let i = 0; i < n; i++) {
    raw.fill(0); e.process(L, R);
    for (let j = 0; j < 4; j++) { const s = raw[j] * (e.bal.on ? e.bal.trim[j] * (j ? e.bal.gain[j] : 1) : 1); const p = s * s; if (p > 1e-7) { sum[j] += p; cnt[j]++; } }
    const p = (L[0] * L[0] + R[0] * R[0]) / 2; out += p; acc += p; peak = Math.max(peak, Math.abs(L[0]), Math.abs(R[0]));
    if ((i + 1) % barLen === 0) { loud.push(db(acc / barLen)); acc = 0; }
  }
  loud.sort((a, b) => a - b);
  const lv = sum.map((s, j) => (cnt[j] > sr ? db(s / cnt[j]) : null));
  const mixDb = e.bal.oms > 0 ? db(e.bal.oms) : null;
  return { mix: mixDb, otrim: e.bal.otrim, kit: lv[0], rel: [1, 2, 3].map((j) => (lv[j] == null || lv[0] == null ? null : lv[j] - lv[0])), out: db(out / n), peak,
           quiet: loud[Math.floor(loud.length * 0.1)], loudBar: loud[Math.floor(loud.length * 0.9)] };
}
if (process.argv[1] && process.argv[1].endsWith('mix_check.mjs')) {
  const f1 = (x) => (x == null ? '   – ' : (x >= 0 ? '+' : '') + x.toFixed(1).padStart(4));
  for (const [name, setting] of sets) {
    console.log(`== ${name}: ${JSON.stringify(setting)}`);
    const rows = [];
    for (const f of files) {
      const track = JSON.parse(readFileSync(f, 'utf8')), r = render(track, setting);
      rows.push(r);
      console.log(`  ${f.split('/').slice(-2).join('/').slice(-46).padEnd(46)} kit ${f1(r.kit)} dB | bass ${f1(r.rel[0])} lead ${f1(r.rel[1])} air ${f1(r.rel[2])} dB of the kit | output ${f1(r.out)} dB, bars ${f1(r.quiet)} to ${f1(r.loudBar)}, peak ${r.peak.toFixed(2)}${r.mix != null ? `, mix before the master ${f1(r.mix)} dB, trim ${(20 * Math.log10(r.otrim)).toFixed(1)} dB` : ''}`);
    }
    const med = (xs) => { const v = xs.filter((x) => x != null).sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; };
    const spread = (xs) => { const v = xs.filter((x) => x != null); return v.length ? Math.max(...v) - Math.min(...v) : 0; };
    console.log(`  median: kit ${f1(med(rows.map((r) => r.kit)))} | bass ${f1(med(rows.map((r) => r.rel[0])))} lead ${f1(med(rows.map((r) => r.rel[1])))} air ${f1(med(rows.map((r) => r.rel[2])))} | output ${f1(med(rows.map((r) => r.out)))}; output spread over the tracks ${spread(rows.map((r) => r.out)).toFixed(1)} dB, kit spread ${spread(rows.map((r) => r.kit)).toFixed(1)} dB`);
  }
}
