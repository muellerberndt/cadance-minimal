#!/usr/bin/env node
// Fit drum kits to bars of drum stems, and score kits on bars. The breakbeat kits of the palette come from here.
//
//   node engine/kit_fit.mjs fit JOBS.json BARS_DIR OUT_DIR SHARD NSHARDS [--pool POOL.json] [--budget 700]
//   node engine/kit_fit.mjs score JOBS.json BARS_DIR KITS.json OUT.json SHARD NSHARDS [--pool POOL.json] [--role check]
//
// A job is one bar: {job, rows: [kick, snare, hats] x 16 (0 rest, 1 hit, 2 accent), t0, t1}; its sound is
// BARS_DIR/<job>.wav (mono or stereo, the bar plus a quarter second). The pattern is the bar's rows on a kit's
// voices, with the velocities the player gives them (brain/common.py beat_pattern): the snare row plays on the snare
// or on the clap, the hat row on the closed or the open hat. The loss is fit.mjs's (features.mjs: log-mel, envelope,
// onset flux, both sounds at equal level). Unlike fit.mjs the candidate is rendered through the pool's drum
// equaliser (pool.fx.drum_eq), because the player plays every kit through it.
//
// fit: per bar, the snare row on the snare (606, 808, 909) and on the clap (808, 909), then the winner with the
// hat row on the open hat; the best of the three is written as OUT_DIR/<job>.json.
// score: per bar of a role, the loss of every kit of KITS.json (a list of {id, device, patch, pattern}); a kit plays
// the bar's rows on the voices its own pattern uses most, as in the player.
import fs from 'node:fs';
import path from 'node:path';
import { Engine, PARAMS, INDEX, DEVICES, DEVICE_VOICES } from './engine.mjs';
import { readWav } from './wav.mjs';
import { loadSamples, useSamples } from './samples.mjs';
import { describe, distance } from './features.mjs';

const SR = 44100;
const SAMPLES = useSamples() ? loadSamples() : null;
const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
function mulberry32(s) { return function () { s |= 0; s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(rnd) { const u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

const poolPath = opt('--pool', null);
const EQ = poolPath ? (JSON.parse(fs.readFileSync(poolPath, 'utf8')).fx || {}).drum_eq || null : null;

// the player's velocities: hit 0.7, accent 1.0; in a very busy row (more than 8 hits) plain hits are ghost notes
function vel(row) { const busy = row.filter((v) => v).length > 8; return row.map((v) => (v === 0 ? 0 : v === 1 ? (busy ? 0.25 : 0.7) : busy ? 0.8 : 1.0)); }
export function patternOf(rows, snareVoice, hatVoice) { return { hits: { kick: vel(rows[0]), [snareVoice]: vel(rows[1]), [hatVoice]: vel(rows[2]) } }; }
// the voices a kit plays the snare and the hat row on: those its own pattern uses most
export function voicesOf(kit) {
  const hits = (kit.pattern && kit.pattern.hits) || {};
  const use = (names) => names.reduce((best, v) => ((hits[v] || []).filter((x) => x > 0).length > (hits[best] || []).filter((x) => x > 0).length ? v : best), names[0]);
  return [use(['snare', 'clap']), use(['ch', 'oh', 'cym'])];
}

function render(tok, bpm, nOut) {
  const eng = new Engine(SR, { bpm, layers: [{ tokens: [null, tok] }, { tokens: [null] }, { tokens: [null] }, { tokens: [null] }], fx: EQ ? { drum_eq: EQ } : undefined }, { fx: false, master: false });
  if (SAMPLES) eng.setSamples(SAMPLES.kit, SAMPLES.sampleRate);
  eng.setTempo(bpm);
  for (let i = 0; i < 3; i++) eng.queueBar([1, 0, 0, 0]);
  const barLen = Math.round((SR * 240) / bpm), n = barLen + nOut, L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; i++) out[i] = 0.5 * (L[barLen + i] + R[barLen + i]);
  return out;
}
function loss(T, tok, bpm, nOut) {
  const x = render(tok, bpm, nOut);
  let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return e < 1e-12 ? { total: 99, mel: 99, env: 99, flux: 99 } : distance(T, describe(x, SR), undefined, 2);
}
function target(dir, job) {
  const w = readWav(path.join(dir, job.job + '.wav'));
  const x = new Float32Array(w.L.length);
  for (let i = 0; i < x.length; i++) x[i] = w.R ? 0.5 * (w.L[i] + w.R[i]) : w.L[i];
  return { x, bpm: 240 / (job.t1 - job.t0) };
}
const baseU = () => { const u = {}; for (const p of PARAMS.drums) u[p.name] = p.def; u.pan = 0.5; u.send_delay = 0; u.send_reverb = 0; return u; };

// fit.mjs's search for one pattern: random patches over the devices, then (1 + 8) evolution on the best device (and
// the runner-up when it is within 15%), then a coordinate polish; the level is solved last
function fitPattern(T, bpm, nOut, pattern, devices, budget, seed) {
  const rnd = mulberry32(seed + 1);
  const used = Object.keys(pattern.hits).filter((v) => pattern.hits[v].some((x) => x > 0));
  const free = [...used.flatMap((v) => [`${v}_tune`, `${v}_decay`, `${v}_level`, `${v}_tone`]), 'drive'];
  const cats = devices.filter((d) => used.every((v) => DEVICE_VOICES[d].includes(v))).map((device) => ({ device, best: null }));
  if (!cats.length) return null;
  let renders = 0;
  const evaluate = (c, u) => { renders++; return loss(T, { voice: 'drums', device: c.device, patch: u, pattern }, bpm, nOut); };
  const consider = (c, u, d) => { if (!c.best || d.total < c.best.d.total) c.best = { u: { ...u }, d }; };
  for (const c of cats) consider(c, baseU(), evaluate(c, baseU()));
  const nRand = Math.max(renders + cats.length * 4, Math.round(budget * 0.35));
  for (let i = 0; renders < nRand; i++) { const c = cats[i % cats.length], u = baseU(); for (const k of free) u[k] = rnd(); consider(c, u, evaluate(c, u)); }
  const ranked = cats.slice().sort((x, y) => x.best.d.total - y.best.d.total), lanes = [ranked[0]];
  if (ranked[1] && ranked[1].best.d.total < ranked[0].best.d.total * 1.15) lanes.push(ranked[1]);
  const left = budget - renders - 4 * free.length, shares = lanes.length === 2 ? [0.6, 0.4] : [1];
  lanes.forEach((c, li) => {
    const stop = renders + Math.round(left * shares[li]);
    let sigma = 0.2; const pm = Math.max(0.2, 3 / free.length);
    while (renders < stop) {
      let improved = false;
      for (let j = 0; j < 8 && renders < stop; j++) {
        const u = { ...c.best.u }; let moved = false;
        for (const k of free) if (rnd() < pm) { u[k] = clamp01(u[k] + sigma * gauss(rnd)); moved = true; }
        if (!moved) { const k = free[Math.floor(rnd() * free.length)]; u[k] = clamp01(u[k] + sigma * gauss(rnd)); }
        const d = evaluate(c, u);
        if (d.total < c.best.d.total) { c.best = { u, d }; improved = true; }
      }
      sigma = improved ? Math.min(0.35, sigma * 1.25) : Math.max(0.01, sigma * 0.82);
    }
  });
  const win = cats.slice().sort((x, y) => x.best.d.total - y.best.d.total)[0];
  for (const step of [0.06, 0.02]) for (const k of free) for (const s of [-step, step]) {
    const u = { ...win.best.u }; u[k] = clamp01(u[k] + s);
    const d = evaluate(win, u); if (d.total < win.best.d.total) { win.best = { u, d }; break; }
  }
  const P = PARAMS.drums[INDEX.drums.level], u = { ...win.best.u };
  const x = render({ voice: 'drums', device: win.device, patch: u, pattern }, bpm, nOut);
  let cr = 0; for (let i = 0; i < x.length; i++) cr += x[i] * x[i]; cr = Math.sqrt(cr / x.length);
  const g0 = P.lo + (P.hi - P.lo) * u.level * u.level, g = cr > 0 ? (g0 * T.rms) / cr : g0;
  u.level = clamp01(Math.sqrt(Math.max(0, (g - P.lo) / (P.hi - P.lo))));
  const round = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Math.round(v * 10000) / 10000]));
  return { voice: 'drums', device: win.device, patch: round(u), pattern, loss: +win.best.d.total.toFixed(4), mel: +win.best.d.mel.toFixed(4), env: +win.best.d.env.toFixed(4),
           flux: +win.best.d.flux.toFixed(4), renders, devices: Object.fromEntries(cats.map((c) => [c.device, +c.best.d.total.toFixed(4)])) };
}

const mode = a[0];
if (mode === 'fit') {
  const [, jobsPath, barsDir, outDir, shard, nshards] = a, budget = +opt('--budget', 700);
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((j) => j.role === 'fit').filter((_, i) => i % +nshards === +shard);
  fs.mkdirSync(outDir, { recursive: true });
  if (a.includes('--reverse')) jobs.reverse();          // a second box helps from the other end
  for (const job of jobs) {
    const out = path.join(outDir, job.job + '.json');
    if (fs.existsSync(out) || !fs.existsSync(path.join(barsDir, job.job + '.wav'))) continue;
    const t0 = Date.now(), { x, bpm } = target(barsDir, job), T = describe(x, SR), seed = [...job.job].reduce((s, c) => (s * 31 + c.charCodeAt(0)) | 0, 7);
    const tries = [];
    const onSnare = fitPattern(T, bpm, x.length, patternOf(job.rows, 'snare', 'ch'), DEVICES, budget, seed);
    const onClap = fitPattern(T, bpm, x.length, patternOf(job.rows, 'clap', 'ch'), DEVICES, budget, seed + 1);
    for (const [name, f] of [['snare', onSnare], ['clap', onClap]]) if (f) tries.push({ snare: name, hat: 'ch', ...f });
    tries.sort((p, q) => p.loss - q.loss);
    const open = fitPattern(T, bpm, x.length, patternOf(job.rows, tries[0].snare, 'oh'), [tries[0].device], Math.round(budget / 2), seed + 2);
    if (open) tries.push({ snare: tries[0].snare, hat: 'oh', ...open });
    tries.sort((p, q) => p.loss - q.loss);
    const best = tries[0];
    fs.writeFileSync(out, JSON.stringify({ job: job.job, track: job.track, bar: job.bar, cent_hz: job.cent_hz, bpm: +bpm.toFixed(2), target_rms: +T.rms.toFixed(5), seconds: +((Date.now() - t0) / 1000).toFixed(1),
      ...best, tried: tries.map((t) => ({ snare: t.snare, hat: t.hat, device: t.device, loss: t.loss })) }));
    console.log(`${job.job}: ${best.device} ${best.snare}/${best.hat} loss ${best.loss} (${tries.map((t) => `${t.device} ${t.snare}/${t.hat} ${t.loss}`).join(', ')}) ${((Date.now() - t0) / 1000).toFixed(0)} s`);
  }
} else if (mode === 'score') {
  const [, jobsPath, barsDir, kitsPath, outPath, shard, nshards] = a, role = opt('--role', 'check');
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((j) => j.role === role).filter((_, i) => i % +nshards === +shard);
  const kits = JSON.parse(fs.readFileSync(kitsPath, 'utf8'));
  const rows = [];
  for (const job of jobs) {
    if (!fs.existsSync(path.join(barsDir, job.job + '.wav'))) continue;
    const { x, bpm } = target(barsDir, job), T = describe(x, SR), losses = {};
    for (const kit of kits) {
      const [snare, hat] = voicesOf(kit);
      losses[kit.id] = +loss(T, { voice: 'drums', device: kit.device, patch: kit.patch, pattern: patternOf(job.rows, snare, hat) }, bpm, x.length).total.toFixed(4);
    }
    rows.push({ job: job.job, track: job.track, held_out: job.held_out, kit: job.kit, losses });
    console.log(`${job.job}: ${kits.length} kits scored`);
  }
  fs.writeFileSync(outPath, JSON.stringify(rows));
} else if (mode === 'machines') {
  // every machine with a search of its own on the same bars: is one machine's lead in `fit` real?
  //   node engine/kit_fit.mjs machines JOBS.json BARS_DIR OUT.json COUNT [--pool POOL.json] [--budget 400]
  const [, jobsPath, barsDir, outPath, count] = a, budget = +opt('--budget', 400);
  const all = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).filter((j) => j.role === 'fit' && fs.existsSync(path.join(barsDir, j.job + '.wav')));
  const jobs = Array.from({ length: Math.min(+count, all.length) }, (_, i) => all[Math.floor((i * all.length) / Math.min(+count, all.length))]);
  const rows = [];
  for (const job of jobs) {
    const { x, bpm } = target(barsDir, job), T = describe(x, SR), row = { job: job.job };
    for (const device of DEVICES) for (const snare of ['snare', 'clap']) {
      const f = fitPattern(T, bpm, x.length, patternOf(job.rows, snare, 'ch'), [device], budget, 11);
      if (f) row[`${device} ${snare}`] = f.loss;
    }
    rows.push(row); console.log(JSON.stringify(row));
  }
  fs.writeFileSync(outPath, JSON.stringify(rows));
} else {
  console.error('usage: kit_fit.mjs fit|score|machines ... (see the head of the file)'); process.exit(2);
}
