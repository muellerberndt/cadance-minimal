#!/usr/bin/env node
// Engine checks: finite output and headroom, level per layer, sample-accurate steps, click-free
// glides (sound and FILTER/SPACE modulation), speed per voice, and fit recovery of a known token.
//   node engine/checks.mjs [--no-fit]
import fs from 'node:fs';
import path from 'node:path';
import { Engine } from './engine.mjs';
import { renderScore, stats } from './render.mjs';
import { fitToken, renderBar } from './fit.mjs';
import { logmel, envelope, rms } from './features.mjs';

const SR = 44100, here = path.dirname(new URL(import.meta.url).pathname);
const pool = JSON.parse(fs.readFileSync(path.join(here, '..', 'pool', 'pool_v0.json'), 'utf8'));
const demo = JSON.parse(fs.readFileSync(path.join(here, 'demo_score.json'), 'utf8'));
const results = []; let failed = 0;
const check = (name, ok, detail) => { results.push({ name, ok, detail }); if (!ok) failed++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`); };
const tokensOf = (b) => (Array.isArray(b) ? b : b.tokens);

// 1. the demo: finite, headroom
{
  const { L, R } = renderScore(demo, pool);
  const s = stats(L, R);
  check('demo finite + peak', s.nonfinite === 0 && s.peak <= 1, `nonfinite ${s.nonfinite}, peak ${s.peak.toFixed(3)}, rms ${s.rms.toFixed(3)}, ${(L.length / SR).toFixed(1)} s`);
}

// 2. level per layer over the bars where the layer plays (dry solo renders)
{
  const barLen = (SR * 240) / demo.bpm, ranges = [[0.05, 0.4], [0.02, 0.3], [0.01, 0.3], [0.003, 0.2]];
  for (let l = 0; l < 4; l++) {
    const { L } = renderScore(demo, pool, { layer: l, tail: 0 });
    let ss = 0, n = 0;
    demo.bars.forEach((b, i) => { if (tokensOf(b)[l]) { const a = Math.round(i * barLen), e = Math.round((i + 1) * barLen); for (let j = a; j < e; j++) { ss += L[j] * L[j]; n++; } } });
    const r = Math.sqrt(ss / Math.max(1, n));
    check(`layer ${l} rms while playing`, r >= ranges[l][0] && r <= ranges[l][1], `${r.toFixed(4)} (range ${ranges[l].join('..')})`);
  }
}

// 3. sample-accurate steps: a rim on every sixteenth at two tempos with fractional step lengths
for (const bpm of [126, 133.7]) {
  const tok = { voice: 'drums', device: '808', patch: { rim_level: 0.8 }, pattern: { hits: { rim: Array(16).fill(0.7) } } };
  const eng = new Engine(SR, { bpm, layers: [{ tokens: [null, tok] }] }, { fx: false, master: false });
  eng.setTempo(bpm); eng.log = [];
  for (let i = 0; i < 4; i++) eng.queueBar([1, 0, 0, 0]);
  const n = Math.round(4 * (SR * 240) / bpm), L = new Float32Array(n), R = new Float32Array(n);
  eng.process(L, R);
  const stepLen = (SR * 15) / bpm; let logErr = 0, onsetErr = 0;
  eng.log.forEach((f, k) => {
    logErr = Math.max(logErr, Math.abs(f - Math.ceil(k * stepLen - 1e-9)));
    let o = f; while (o < n && Math.abs(L[o]) < 1e-6) o++;
    onsetErr = Math.max(onsetErr, o - f);
  });
  check(`step timing ${bpm} bpm`, logErr === 0 && onsetErr <= 1, `${eng.log.length} steps, max trigger error ${logErr} samples, max onset delay ${onsetErr} samples`);
}

// 4. glides: switching siblings / modulating must not click more than repeating one sound
function boundaryJump(bars, layer, bpm = 126) {
  const eng = new Engine(SR, pool, { fx: false, master: false }); eng.setTempo(bpm);
  for (const b of bars) eng.queueBar(b.t, b.m);
  const barLen = (SR * 240) / bpm, n = Math.round(bars.length * barLen), L = new Float32Array(n), R = new Float32Array(n);
  eng.solo = layer; eng.process(L, R);
  let worst = 0;
  for (let i = 1; i < bars.length; i++) {
    const c = Math.round(i * barLen);
    for (let j = c - 64; j < c + 64; j++) worst = Math.max(worst, Math.abs(L[j] - L[j - 1]));
  }
  return worst;
}
{
  const same = boundaryJump([1, 1, 1, 1].map((t) => ({ t: [0, t, 0, 0] })), 1);
  const sib = boundaryJump([1, 2, 3, 4].map((t) => ({ t: [0, t, 0, 0] })), 1);
  check('acid sibling glide', sib <= same * 1.3 + 0.01, `max step at bar starts ${sib.toFixed(4)} vs repeated sound ${same.toFixed(4)}`);
  const mod = boundaryJump([-1, -0.33, 0.33, 1].map((f) => ({ t: [0, 1, 0, 0], m: [[0, 0], [f, f], [0, 0], [0, 0]] })), 1);
  check('FILTER/SPACE glide (acid)', mod <= same * 1.3 + 0.01, `${mod.toFixed(4)} vs ${same.toFixed(4)}`);
  const dsame = boundaryJump([1, 1, 1, 1].map((t) => ({ t: [t, 0, 0, 0] })), 0);
  const dmod = boundaryJump([-1, -0.33, 0.33, 1].map((f) => ({ t: [1, 0, 0, 0], m: [[f, f], [0, 0], [0, 0], [0, 0]] })), 0);
  check('FILTER tilt glide (drums)', dmod <= dsame * 1.3 + 0.01, `${dmod.toFixed(4)} vs ${dsame.toFixed(4)}`);
  const pad = boundaryJump([4, 6, 4, 6].map((t) => ({ t: [0, 0, 0, t] })), 3), padSame = boundaryJump([4, 4, 4, 4].map((t) => ({ t: [0, 0, 0, t] })), 3);
  check('pad sibling glide', pad <= padSame * 1.3 + 0.005, `${pad.toFixed(4)} vs ${padSame.toFixed(4)}`);
  const out = boundaryJump([{ t: [0, 1, 0, 0] }, { t: [0, 1, 0, 0] }, { t: [0, 0, 0, 0] }, { t: [0, 1, 0, 0] }], 1);
  check('silence switch', out <= same * 1.3 + 0.01, `${out.toFixed(4)} vs ${same.toFixed(4)}`);
}

// 5. speed: ms per bar per voice, full engine (effects + master), 8 bars each
{
  const rows = [];
  for (const [l, t] of [[0, 1], [0, 4], [0, 6], [1, 2], [1, 5], [2, 7], [3, 1], [3, 4]]) {
    const bars = Array.from({ length: 8 }, () => { const b = [0, 0, 0, 0]; b[l] = t; return b; });
    const t0 = performance.now(); renderScore({ bpm: 126, bars }, pool, { tail: 0 }); const ms = (performance.now() - t0) / 8;
    const tok = pool.layers[l].tokens[t]; rows.push(`${tok.voice}${tok.device ? ':' + tok.device : ''} L${l}T${t} ${ms.toFixed(1)}`);
  }
  const all = performance.now(); renderScore({ bpm: 126, bars: Array(8).fill([3, 4, 2, 4]) }, pool, { tail: 0 }); const ms4 = (performance.now() - all) / 8;
  check('speed (ms per 1.9 s bar)', ms4 < 1905 * 0.25, `${rows.join(', ')}; all four layers ${ms4.toFixed(1)} ms per bar (${(1905 / ms4).toFixed(0)}x real time)`);
}

// perceptual distance used by the recovery check: log-mel at a resolution the fit loss does not use
// (1024/256, 40 bands) + loudness envelope dB/20, both signals scaled to equal RMS
function neutral(a, b) {
  const na = rms(a) || 1, nb = rms(b) || 1, x = a.map((v) => v / na), y = b.map((v) => v / nb);
  const A = logmel(x, SR, 1024, 256, 40), B = logmel(y, SR, 1024, 256, 40);
  let s = 0; const n = Math.min(A.mel.length, B.mel.length); for (let i = 0; i < n; i++) s += Math.abs(A.mel[i] - B.mel[i]);
  const ea = envelope(x, 64), eb = envelope(y, 64); let e = 0; for (let i = 0; i < 64; i++) e += Math.abs(ea[i] - eb[i]);
  return s / n + e / 64 / 20;
}

// 6. fit recovery of a known token
if (!process.argv.includes('--no-fit')) {
  for (const [l, t, budget] of [[1, 2, 400], [0, 4, 400], [3, 1, 400]]) {
    const tok = pool.layers[l].tokens[t], n = Math.round((SR * 240) / 126) + 4410;
    const target = renderBar(tok, l, 126, n);
    const fit = fitToken({ layer: l, target, pattern: tok.pattern, bpm: 126, budget, seed: 0, refine: false }); // the pattern is exact here
    const keys = Object.keys(fit.patch).filter((k) => !['level', 'pan', 'send_delay', 'send_reverb'].includes(k) && (l > 0 || k === 'drive' || Object.keys(fit.pattern.hits || {}).some((v) => k.startsWith(v + '_'))));
    const err = keys.reduce((s, k) => s + Math.abs(fit.patch[k] - (tok.patch[k] ?? 0.5)), 0) / keys.length;
    const same = fit.voice === tok.voice && (fit.device || null) === (tok.device || null);
    const nd = neutral(target, renderBar(fit, l, 126, n));
    check(`fit recovery L${l}T${t}`, same && nd < 0.4, `${fit.voice}${fit.device ? ':' + fit.device : ''} (true ${tok.voice}${tok.device ? ':' + tok.device : ''}), perceptual distance ${nd.toFixed(3)} (< 0.4), loss ${fit.fit.loss}, mean param error ${err.toFixed(3)} over ${keys.length}, ${fit.fit.renders} renders at ${fit.fit.ms_per_render} ms`);
  }
}
console.log(failed ? `${failed} check(s) failed` : 'all checks passed');
process.exitCode = failed ? 1 : 0;
