#!/usr/bin/env node
// The balance inside every kit of a pool: how loud its snare and hat sounds are against its kick.
//   node engine/kit_levels.mjs pool/POOL.json [--no-samples]
// For each kit: the snare-class and hat-class sound its own pattern uses most, the peak and the loudness (weighted
// root mean square over 0.15 s) of one accent of each, in dB against the kit's kick. Then the medians and spreads.
import fs from 'node:fs';
import { Engine, DrumKit } from './engine.mjs';
import { loadSamples } from './samples.mjs';
const args = process.argv.slice(2), pool = JSON.parse(fs.readFileSync(args[0], 'utf8'));
const eng = new Engine(44100, pool, { fx: false, master: false });
const K = new DrumKit(44100);
if (!args.includes('--no-samples')) { try { const s = loadSamples(); eng.setSamples(s.kit, s.sampleRate); K.bank = eng.samples; K.bankRate = eng.sampleRate; } catch { /* the models */ } }
const db = (x) => 20 * Math.log10(Math.max(1e-7, x)), med = (a) => { const s = a.slice().sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const q = (a, f) => { const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(f * s.length))]; };
const rows = [];
for (const tok of eng.tokens[0]) {
  if (!tok || tok.voice !== 'drums') continue;
  const H = (tok.pattern && tok.pattern.hits) || {}, n = (v) => (H[v] || []).filter((x) => x > 0).length;
  const most = (names) => names.reduce((a, b) => (n(b) > n(a) ? b : a));
  const sn = most(['snare', 'clap', 'rim']), hat = most(['ch', 'oh', 'cym']);
  const k = K.levelOf(tok.device, 'kick', tok.u), s = K.levelOf(tok.device, sn, tok.u), h = K.levelOf(tok.device, hat, tok.u);
  rows.push({ device: tok.device, sn, hat, usesSn: n(sn), usesHat: n(hat), snPeak: db(s.peak / k.peak), snLoud: db(s.loud / k.loud), hatPeak: db(h.peak / k.peak), hatLoud: db(h.loud / k.loud) });
}
const used = (key, uses) => rows.filter((r) => r[uses] > 0 && Number.isFinite(r[key]) && r[key] > -100).map((r) => r[key]);
console.log(`${args[0]}: ${rows.length} kits`);
for (const [label, key, uses] of [['snare peak', 'snPeak', 'usesSn'], ['snare loudness', 'snLoud', 'usesSn'], ['hats peak', 'hatPeak', 'usesHat'], ['hats loudness', 'hatLoud', 'usesHat']]) {
  const a = used(key, uses);
  console.log(`  ${label.padEnd(15)} against the kick: median ${med(a).toFixed(1)} dB, middle half ${q(a, 0.25).toFixed(1)} to ${q(a, 0.75).toFixed(1)}, all ${Math.min(...a).toFixed(1)} to ${Math.max(...a).toFixed(1)} dB (${a.length} kits)`);
}
for (const v of ['ch', 'oh', 'cym', 'snare', 'clap', 'rim']) {
  const a = rows.filter((r) => (r.hat === v && r.usesHat > 0) || (r.sn === v && r.usesSn > 0)).map((r) => (r.hat === v ? r.hatLoud : r.snLoud)).filter((x) => x > -100);
  if (a.length) console.log(`  kits on ${v.padEnd(5)}: ${a.length}, loudness against the kick median ${med(a).toFixed(1)} dB, ${Math.min(...a).toFixed(1)} to ${Math.max(...a).toFixed(1)}`);
}
