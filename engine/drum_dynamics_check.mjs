#!/usr/bin/env node
// What the drum dynamics do to a track's hat line and snare row, measured on the rendered sound.
//   node engine/drum_dynamics_check.mjs <track.json> [--bars 64] [--set '{"plainDb": -6}']
// The track's drum layer is rendered alone and dry, once with only its hat sounds and once with only its snare
// sounds, as written and under the dynamics. Reported: the mean level (dB) of the sixteenths on the beat, on the
// off-beat eighth and on the other sixteenths, and the spread between the loudest and quietest of the three.
import fs from 'node:fs';
import path from 'node:path';
import { renderScore } from './render.mjs';

const args = process.argv.slice(2), file = args[0];
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const track = JSON.parse(fs.readFileSync(file, 'utf8'));
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const poolPath = typeof track.pool === 'string' ? (fs.existsSync(track.pool) ? track.pool : path.resolve(root, 'pool', path.basename(track.pool))) : null;
const pool = poolPath ? JSON.parse(fs.readFileSync(poolPath, 'utf8')) : track.pool;
const play = JSON.parse(fs.readFileSync(path.resolve(root, 'site/play-reading.json'), 'utf8'));
const setting = Object.assign({ on: true }, (play.balance && play.balance.drums) || {}, JSON.parse(opt('--set', '{}')));
const n = Math.min(+opt('--bars', 64), track.bars.length), start = Math.min(32, Math.max(0, track.bars.length - n));
const only = (names) => track.bars.slice(start, start + n).map((b) => {
  const p = b.patterns && b.patterns[0];
  const hits = p && p.hits ? Object.fromEntries(Object.entries(p.hits).filter(([k]) => names.includes(k))) : null;
  return { tokens: b.tokens, mods: [[0, 0], [0, 0], [0, 0], [0, 0]], patterns: [hits ? { hits } : { hits: {} }, null, null, null], patches: b.patches };
});
function levels(bars, drums) {
  const score = { bpm: track.bpm || 126, bars, balance: { on: true, rangeDb: 1, kitDb: null, outDb: null, duck: 0, drums: drums || { on: false } } };
  const { L, sr, eng } = renderScore(score, pool, { layer: 0, dry: true, tail: 0.1 });
  const step = (sr * 15) / eng.bpm, sum = [0, 0, 0], cnt = [0, 0, 0], modes = {};
  for (let b = 0; b < bars.length; b++) for (let s = 0; s < 16; s++) {
    const p = bars[b].patterns[0].hits; let on = false;
    for (const k in p) if (p[k][s] > 0) on = true;
    if (!on) continue;
    const o = Math.round((b * 16 + s) * step), e = Math.min(L.length, o + Math.round(step * 0.9)); let pk = 0;
    for (let i = o; i < e; i++) pk = Math.max(pk, Math.abs(L[i]));
    const c = s % 4 === 0 ? 0 : s % 4 === 2 ? 1 : 2; sum[c] += pk; cnt[c]++;
  }
  if (drums) for (let b = 0; b < bars.length; b += eng.dyn.section) { const m = eng.hatModeAt(b); modes[m] = (modes[m] || 0) + 1; }
  const db = sum.map((x, i) => (cnt[i] ? 20 * Math.log10(Math.max(1e-6, x / cnt[i])) : null));
  return { db, cnt, modes };
}
const show = (r) => { const f = r.db.filter((x) => x != null); return `on the beat ${r.db[0]?.toFixed(1)} dB (${r.cnt[0]})  off-beat eighth ${r.db[1]?.toFixed(1)} dB (${r.cnt[1]})  other sixteenths ${r.db[2]?.toFixed(1)} dB (${r.cnt[2]})  spread ${(Math.max(...f) - Math.min(...f)).toFixed(1)} dB`; };
console.log(`${path.basename(file)}: bars ${start}..${start + n - 1}, setting ${JSON.stringify(setting)}`);
for (const [label, names] of [['hats ', ['ch', 'oh', 'cym']], ['snare', ['snare', 'clap', 'rim']], ['kick ', ['kick']]]) {
  const bars = only(names), a = levels(bars, null), b = levels(bars, setting);
  console.log(`${label} as written: ${show(a)}`);
  console.log(`${label} dynamics  : ${show(b)}   sections by mode ${JSON.stringify(b.modes)}`);
}
