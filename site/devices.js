// Cadance Radio, the devices view: the drum machine and the three synth voices of the track in front, playing the bar
// that sounds. A canvas in the Pragma palette: cyan lines on black, a machine's own colour only as a thin outline and
// on its knob caps (the 808's orange, the 909's off-white, the 606's red, the acid voice's silver).
//
// The data: the sound each layer's token names in the track's pool (device, voice, patch), or the patch the brain wrote
// for the bar; the bar's patterns (drum hits with their velocity, above 0.75 an accent; a line's notes with length,
// velocity, slide and accent) and its modulation (FILTER and SPACE per layer, -1..1); the step of the bar clock.
// Knobs glide from the bar before to this bar across the bar, as the engine glides within one voice.
//
// No words but the machines' own lettering (names, voices, knobs); no numbers.
//
//   const dev = new Devices(canvas); dev.setPool(pool);
//   dev.setBar(bar, barBefore); dev.setStep(step); dev.setSecond({ bar, level } or null); dev.draw(phase) every frame
import { UI } from './ui.mjs';

const CY = '#4af1f2', INK = '#c9fbfb', DIM = 'rgba(74,241,242,.62)', FAINT = 'rgba(74,241,242,.2)', HOT = '#e8ffff', RED = '#ff2a6d';
const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';
const MACHINE = { '606': { col: '#e0593a', sub: 'drum machine' }, '808': { col: '#ef8a2c', sub: 'rhythm composer' }, '909': { col: '#ece6d8', sub: 'rhythm machine' } };
const VOICE = {
  acid: { col: '#c8ccd2', title: 'acid', sub: '303-style bass line', groups: [['', ['tune', 'cutoff', 'reso', 'envmod', 'decay', 'accent_amt']]] },
  synth: { col: CY, title: 'FM synth', sub: '', groups: [['osc', ['wave', 'tune', 'fm_ratio', 'fm_amt']], ['filter', ['cutoff', 'reso', 'ftype', 'f_env']], ['env', ['attack', 'decay', 'sustain', 'release']]] },
  sid: { col: '#8f85ea', title: 'SID chip', sub: '', groups: [['filter', ['cutoff', 'reso', 'mode', 'f_env']], ['', ['pwm', 'fx', 'tune1', 'tune2']]] },
  sampler: { col: '#db9ae9', title: 'sampler', sub: '', groups: [['', ['tune', 'start', 'decay', 'cutoff', 'reso']]] },
};
const LAYERS = ['beat', 'bass', 'lead', 'air'];
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const labelOf = (voice, name) => ((UI[voice] && UI[voice].knobs.find((k) => k.name === name)) || { label: name.toUpperCase() }).label;

export class Devices {
  constructor(canvas) {
    this.cv = canvas; this.g = canvas.getContext('2d');
    this.pool = null; this.bar = null; this.prev = null; this.second = null;
    this.step = -1; this.stepAt = 0; this.need = 0;
  }
  setPool(pool) { this.pool = pool || null; }
  setBar(bar, prev) { this.bar = bar || null; this.prev = prev || bar || null; }
  setSecond(s) { this.second = s || null; }
  setStep(s) { this.step = s; this.stepAt = performance.now(); }
  // the sound of layer l in a bar: the patch the brain wrote, else the pool's token
  sound(bar, l) {
    if (!bar) return null;
    const w = bar.patches && bar.patches[l];
    if (w) return { ...w, written: true };
    const toks = this.pool && this.pool.layers[l] ? this.pool.layers[l].tokens : null;
    return toks ? toks[(bar.tokens || [])[l]] || null : null;
  }
  draw(phase) {
    const D = Math.min(2, devicePixelRatio || 1), cw = this.cv.clientWidth || 300;
    const W = Math.round(cw * D);
    const H = Math.max(Math.round(200 * D), this.need);
    if (this.cv.width !== W || this.cv.height !== H) { this.cv.width = W; this.cv.height = H; this.cv.style.height = `${H / D}px`; }
    const g = this.g, narrow = cw < 700;
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.textBaseline = 'middle';
    if (!this.bar || !this.pool) { this.need = Math.round(200 * D); return; }      // dark until a track plays and its sounds are read
    const maxW = Math.min(W, 1100 * D), x0 = (W - maxW) / 2;
    let y = 0;
    y = this.drum(g, D, x0, y, maxW, phase, narrow) + 16 * D;
    const gap = 14 * D;
    if (narrow) {
      for (let l = 1; l < 4; l++) y = this.synth(g, D, x0, y, maxW, l, phase, true) + gap;
    } else {
      const w3 = (maxW - 2 * gap) / 3;
      let bottom = y;
      for (let l = 1; l < 4; l++) bottom = Math.max(bottom, this.synth(g, D, x0 + (l - 1) * (w3 + gap), y, w3, l, phase, false));
      y = bottom + gap;
    }
    if (this.second) y = this.strip(g, D, x0, y, maxW) + gap;
    this.need = Math.ceil(y + 8 * D);
  }
  // the struck glow of the step that sounds: bright on the step, fading over it
  struck() { return Math.exp(-(performance.now() - this.stepAt) / 140); }
  font(g, px, D, bold) { g.font = `${bold ? '700 ' : ''}${Math.round(px * D)}px ${MONO}`; }

  drum(g, D, x, y, w, phase, narrow) {
    const s = this.sound(this.bar, 0), old = this.sound(this.prev, 0);
    const dev = (s && s.device) || (old && old.device) || '808', M = MACHINE[dev] || MACHINE['808'];
    const desc = UI.drums.devices[dev] || UI.drums.devices['808'];
    const hits = (this.bar.patterns && this.bar.patterns[0] && this.bar.patterns[0].hits) || (s && s.pattern && s.pattern.hits) || {};
    const patch = (s && s.patch) || {}, before = old && old.device === dev && old.patch ? old.patch : patch;
    const pad = 14 * D, rowH = (narrow ? 19 : 20) * D, head = 44 * D;
    const voices = desc.voices, h = head + voices.length * rowH + 30 * D;
    g.strokeStyle = M.col; g.lineWidth = D; g.globalAlpha = s ? 1 : 0.35;
    g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    g.fillStyle = M.col; g.fillRect(x, y, w, 2 * D);
    // the name plate
    this.font(g, 22, D, true); g.textAlign = 'left'; g.fillStyle = M.col;
    g.fillText(dev, x + pad, y + 24 * D);
    const nameW = g.measureText(dev).width;
    this.font(g, 9, D); g.fillStyle = DIM;
    g.fillText(M.sub.toUpperCase(), x + pad + nameW + 12 * D, y + 26 * D);
    // the brain's modulation of the beat for this bar
    const mods = (this.bar.mods && this.bar.mods[0]) || [0, 0], pm = (this.prev && this.prev.mods && this.prev.mods[0]) || mods;
    if (!narrow) {
      knob(g, D, x + w - pad - 104 * D, y + 24 * D, 10 * D, 0.5 + 0.5 * lerp(pm[0], mods[0], phase), 'FILTER', M.col, true, 'right');
      knob(g, D, x + w - pad - 12 * D, y + 24 * D, 10 * D, 0.5 + 0.5 * lerp(pm[1], mods[1], phase), 'SPACE', M.col, true, 'right');
    }
    // one row per voice: its knobs at the kit's values, then its 16 step buttons
    const labW = 30 * D, kW = narrow ? 0 : 34 * D, stepX = x + pad + labW + 4 * kW + (narrow ? 0 : 10 * D), stepW = x + w - pad - stepX;
    const cell = (stepW - 3 * 6 * D) / 16, hot = this.struck();
    let ry = y + head;
    for (const v of voices) {
      const row = hits[v.name] || (v.name === 'cym' && hits.cy) || null, any = !!row && row.some((q) => q > 0);
      g.globalAlpha = s ? (any ? 1 : 0.55) : 0.35;
      this.font(g, 10, D, any); g.textAlign = 'left'; g.fillStyle = any ? INK : DIM;
      g.fillText(v.label, x + pad, ry + rowH / 2);
      if (!narrow) v.knobs.forEach((k, i) => {
        const val = lerp(+before[k.name] || 0, +patch[k.name] || 0, phase);
        knob(g, D, x + pad + labW + i * kW + kW / 2, ry + rowH / 2, 7 * D, val, null, M.col, false);
      });
      for (let i = 0; i < 16; i++) {
        const q = row ? +row[i] || 0 : 0, cx = stepX + i * cell + Math.floor(i / 4) * 6 * D, cy = ry + 3 * D, ch = rowH - 6 * D;
        const now = i === this.step;
        if (q > 0) {
          const acc = q > 0.75;
          g.fillStyle = now ? HOT : CY; g.globalAlpha = (s ? 1 : 0.35) * (now ? 0.55 + 0.45 * hot : acc ? 0.95 : 0.3 + 0.4 * q);
          if (now) { g.shadowColor = CY; g.shadowBlur = 18 * D * hot; }
          g.fillRect(cx + 1.5 * D, cy, cell - 3 * D, ch);
          g.shadowBlur = 0;
          if (acc) { g.fillStyle = M.col; g.globalAlpha = s ? 1 : 0.35; g.fillRect(cx + 1.5 * D, cy, cell - 3 * D, 2 * D); }
        } else {
          g.strokeStyle = now ? 'rgba(232,255,255,.45)' : FAINT; g.lineWidth = D; g.globalAlpha = s ? 1 : 0.35;
          g.strokeRect(cx + 1.5 * D + 0.5, cy + 0.5, cell - 3 * D - 1, ch - 1);
        }
      }
      ry += rowH;
    }
    // the running light under the steps, and the knob names over the first row's knobs
    g.globalAlpha = s ? 1 : 0.35;
    for (let i = 0; i < 16; i++) {
      const cx = stepX + i * cell + Math.floor(i / 4) * 6 * D, on = i === this.step;
      g.fillStyle = on ? M.col : 'rgba(74,241,242,.12)';
      if (on) { g.shadowColor = M.col; g.shadowBlur = 10 * D; }
      g.fillRect(cx + 1.5 * D, ry + 8 * D, cell - 3 * D, 4 * D);
      g.shadowBlur = 0;
    }
    if (!narrow) {
      this.font(g, 7.5, D); g.fillStyle = DIM; g.textAlign = 'center';
      voices[0].knobs.forEach((k, i) => g.fillText(['TUNE', 'DECAY', 'LEVEL', 'TONE'][i], x + pad + labW + i * kW + kW / 2, y + head - 7 * D));
    }
    g.globalAlpha = 1;
    return y + h;
  }

  synth(g, D, x, y, w, l, phase, narrow) {
    const s = this.sound(this.bar, l), old0 = this.sound(this.prev, l), voice = (s && s.voice) || 'synth';
    const old = old0 && old0.voice === voice ? old0 : s;
    const V = VOICE[voice] || VOICE.synth, pad = 12 * D;
    const pat = (this.bar.patterns && this.bar.patterns[l]) || (s && s.pattern) || null;
    const notes = (pat && pat.notes) || [], sounding = !!s && ((this.bar.tokens || [])[l] > 0) && notes.some((n) => n != null);
    const groups = V.groups, kr = 10 * D, kw = 40 * D, perRow = Math.max(1, Math.floor((w - 2 * pad) / kw));
    const knobs = groups.flatMap(([gname, names]) => names.map((n, i) => ({ n, gname: i === 0 ? gname : '' })));
    const rows = Math.ceil(knobs.length / perRow), rollH = 62 * D, top = 56 * D, h = top + rows * 48 * D + rollH + 44 * D;
    g.globalAlpha = sounding ? 1 : 0.35;
    g.strokeStyle = V.col; g.lineWidth = D; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    g.fillStyle = V.col; g.fillRect(x, y, w, 2 * D);
    this.font(g, 15, D, true); g.textAlign = 'left'; g.fillStyle = INK;
    g.fillText(LAYERS[l], x + pad, y + 20 * D);
    const lw = g.measureText(LAYERS[l]).width;
    this.font(g, 10, D, true); g.fillStyle = V.col;
    g.fillText(V.title, x + pad + lw + 10 * D, y + 21 * D);
    this.font(g, 8, D); g.fillStyle = DIM;
    g.fillText(V.sub, x + pad, y + 37 * D);
    // the knobs at the sound's values, gliding from the bar before
    const patch = (s && s.patch) || {}, before = (old && old.patch) || patch;
    knobs.forEach(({ n, gname }, i) => {
      const r = Math.floor(i / perRow), c = i % perRow, kx = x + pad + c * kw + kw / 2, ky = y + top + 18 * D + r * 48 * D;
      if (gname) { this.font(g, 7.5, D); g.fillStyle = V.col; g.textAlign = 'left'; g.fillText(`${gname.toUpperCase()} ›`, kx - kw / 2 + 2 * D, ky - 17 * D); }
      knob(g, D, kx, ky, kr, lerp(+before[n] || 0, +patch[n] || 0, phase), labelOf(voice, n), V.col, false);
    });
    // the notes of the bar: a roll of 16 steps, a note as long as it is held, a slide as a line into the next
    const ry = y + top + rows * 48 * D, rx = x + pad, rw = w - 2 * pad, cell = rw / 16;
    g.strokeStyle = FAINT; g.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rollH - 1);
    for (let i = 4; i < 16; i += 4) { g.fillStyle = 'rgba(74,241,242,.08)'; g.fillRect(rx + i * cell, ry, D, rollH); }
    if (this.step >= 0) { g.fillStyle = 'rgba(232,255,255,.07)'; g.fillRect(rx + this.step * cell, ry, cell, rollH); }
    const pitched = notes.filter((n) => n != null), lo = pitched.length ? Math.min(...pitched) : 0, hi = pitched.length ? Math.max(...pitched) : 0;
    const yOf = (n) => ry + rollH - 8 * D - (hi > lo ? ((n - lo) / (hi - lo)) * (rollH - 18 * D) : (rollH - 18 * D) / 2);
    const lens = (pat && pat.len) || [], vel = (pat && pat.vel) || [], slide = (pat && pat.slide) || [], accent = (pat && pat.accent) || [];
    const hot = this.struck();
    for (let i = 0; i < 16; i++) {
      const n = notes[i];
      if (n == null || !sounding) continue;
      const len = clamp(+lens[i] || 1, 1, 16 - i), ny = yOf(n), on = this.step >= i && this.step < i + len, acc = (+accent[i] || 0) > 0 || (+vel[i] || 0) >= 0.99;
      g.globalAlpha = on ? 1 : 0.45 + 0.5 * clamp(+vel[i] || 0.7, 0, 1);
      g.fillStyle = on ? HOT : CY;
      if (on) { g.shadowColor = CY; g.shadowBlur = 16 * D * (this.step === i ? hot : 0.4); }
      g.fillRect(rx + i * cell + 1.5 * D, ny - 2.5 * D, len * cell - 3 * D, 5 * D);
      g.shadowBlur = 0;
      if (acc) { g.fillStyle = V.col; g.globalAlpha = 1; g.beginPath(); g.arc(rx + i * cell + 4 * D, ny - 6 * D, 1.6 * D, 0, Math.PI * 2); g.fill(); }
      const next = notes.findIndex((m, k) => k > i && m != null);
      if ((+slide[i] || 0) > 0 && next > 0) {
        g.strokeStyle = CY; g.globalAlpha = 0.7; g.lineWidth = 1.2 * D; g.beginPath();
        g.moveTo(rx + (i + len) * cell - 1.5 * D, ny); g.lineTo(rx + next * cell + 1.5 * D, yOf(notes[next])); g.stroke();
      }
    }
    g.globalAlpha = sounding ? 1 : 0.35;
    // FILTER and SPACE: the brain's modulation of this layer for the bar
    const mods = (this.bar.mods && this.bar.mods[l]) || [0, 0], pm = (this.prev && this.prev.mods && this.prev.mods[l]) || mods;
    [['FILTER', lerp(pm[0], mods[0], phase)], ['SPACE', lerp(pm[1], mods[1], phase)]].forEach(([name, v], k) => {
      const bx = rx + k * (rw / 2 + 4 * D), bw = rw / 2 - 4 * D, by = ry + rollH + 22 * D;
      this.font(g, 7.5, D); g.fillStyle = DIM; g.textAlign = 'left'; g.fillText(name, bx, by - 9 * D);
      g.fillStyle = 'rgba(74,241,242,.12)'; g.fillRect(bx, by, bw, 4 * D);
      const mid = bx + bw / 2, end = mid + (bw / 2) * clamp(v, -1, 1);
      g.fillStyle = V.col; g.fillRect(Math.min(mid, end), by, Math.abs(end - mid) || D, 4 * D);
      g.fillStyle = HOT; g.fillRect(mid - 0.5 * D, by - 2 * D, D, 8 * D);
    });
    g.globalAlpha = 1;
    return y + h;
  }

  // the deck that blends in: its drums and lines as one small strip, a line along its top as far as its fader is up
  strip(g, D, x, y, w) {
    const { bar, level } = this.second, pad = 12 * D, h = 84 * D;
    g.strokeStyle = FAINT; g.lineWidth = D; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    g.fillStyle = CY; g.fillRect(x, y, w * clamp(level || 0, 0, 1), 2 * D);          // how far the deck is up
    if (!bar) return y + h;
    const lanes = 6, top = y + 12 * D, lh = (h - 18 * D) / lanes, sx = x + pad + 44 * D, cell = (w - 2 * pad - 44 * D) / 16;
    const hits = (bar.patterns && bar.patterns[0] && bar.patterns[0].hits) || {};
    const drum = [['BD', hits.kick], ['SD', hits.snare || hits.clap], ['HH', hits.ch || hits.oh]];
    this.font(g, 8, D);
    drum.forEach(([name, row], r) => {
      g.fillStyle = DIM; g.fillText(name, x + pad, top + (r + 0.5) * lh);
      for (let i = 0; i < 16; i++) if (row && row[i] > 0) { g.fillStyle = i === this.step ? HOT : CY; g.globalAlpha = 0.35 + 0.6 * row[i]; g.fillRect(sx + i * cell + D, top + r * lh + D, cell - 2 * D, lh - 2 * D); }
      g.globalAlpha = 1;
    });
    for (let l = 1; l < 4; l++) {
      const p = bar.patterns && bar.patterns[l], notes = (p && p.notes) || [], on = (bar.tokens || [])[l] > 0, r = 2 + l;
      g.fillStyle = DIM; g.fillText(LAYERS[l], x + pad, top + (r + 0.5) * lh);
      for (let i = 0; i < 16; i++) if (on && notes[i] != null) { const len = clamp((p.len && p.len[i]) || 1, 1, 16 - i); g.fillStyle = i === this.step ? HOT : CY; g.globalAlpha = 0.7; g.fillRect(sx + i * cell + D, top + r * lh + lh * 0.3, len * cell - 2 * D, lh * 0.4); }
      g.globalAlpha = 1;
    }
    return y + h;
  }
}

// a knob: the track from 7 to 5 o'clock, the value as a lit arc (bipolar: from the top), a cap in the device's colour
function knob(g, D, x, y, r, v, label, col, bip, align) {
  const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25, a = a0 + (a1 - a0) * clamp(v, 0, 1), mid = (a0 + a1) / 2;
  g.lineWidth = Math.max(1.5 * D, r * 0.22); g.lineCap = 'round';
  g.strokeStyle = 'rgba(74,241,242,.16)'; g.beginPath(); g.arc(x, y, r, a0, a1); g.stroke();
  g.strokeStyle = CY; g.beginPath(); if (bip) g.arc(x, y, r, Math.min(mid, a), Math.max(mid, a) + 0.001); else g.arc(x, y, r, a0, a + 0.001); g.stroke();
  g.lineCap = 'butt';
  g.fillStyle = '#000'; g.strokeStyle = col; g.lineWidth = D; g.beginPath(); g.arc(x, y, r * 0.58, 0, Math.PI * 2); g.fill(); g.stroke();
  g.strokeStyle = HOT; g.lineWidth = 1.4 * D; g.beginPath(); g.moveTo(x + Math.cos(a) * r * 0.15, y + Math.sin(a) * r * 0.15); g.lineTo(x + Math.cos(a) * r * 0.58, y + Math.sin(a) * r * 0.58); g.stroke();
  if (label) {
    g.font = `${Math.round(7.5 * D)}px ${MONO}`; g.fillStyle = 'rgba(201,251,251,.8)'; g.textAlign = align === 'right' ? 'right' : 'center';
    g.fillText(label, align === 'right' ? x - r - 4 * D : x, align === 'right' ? y : y + r + 9 * D);
  }
}
