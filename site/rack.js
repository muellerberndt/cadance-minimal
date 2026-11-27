// DEVICES view: the instrument as it plays. One drum machine (606 / 808 / 909, its own voices and
// faceplate; the recorded one-shots of the machine where the page has them, the model otherwise) and three
// synth units, each showing the voice the engine plays for that layer in this bar: the SID-style chip with
// its three oscillators and filter, the 303-style acid voice, the FM synth or the one-shot sampler. Each
// bar every unit loads a sound (patch + one-bar pattern); knobs glide from the old to the new patch across
// the bar as the engine does and registers switch at the bar start, FILTER / SPACE are the brain's per-bar
// modulation. Step lights follow the engine's sixteenths.
import { UI, DRUM_LABELS } from './ui.mjs';

const LOOK = {
  '606': { body: 'linear-gradient(180deg,#d8d7d1,#b4b3ac)', ink: '#18181a', steps: ['#d8492a'], off: 'rgba(0,0,0,.16)', sub: 'drum machine model', led: '#d8492a' },
  '808': { body: 'linear-gradient(180deg,#2a2a2d,#19191b)', ink: '#ece6d8', steps: ['#e0412f', '#ef8a2c', '#f2d23c', '#ece6d8'], off: 'rgba(255,255,255,.07)', sub: 'rhythm composer model', led: '#ff3b2f' },
  '909': { body: 'linear-gradient(180deg,#b9bab8,#979896)', ink: '#141414', steps: ['#ee8420'], off: 'rgba(0,0,0,.15)', sub: 'rhythm model', led: '#ee8420' },
};
const UNIT = {
  acid: { body: 'linear-gradient(180deg,#cfd0cb,#adaea9)', ink: '#141414', led: '#e8342a', track: 'rgba(0,0,0,.18)', name: 'acid', sub: '303 model',
          knobs: ['cutoff', 'reso', 'envmod', 'decay'] },
  sampler: { body: 'linear-gradient(180deg,#29232e,#19151d)', ink: '#e6d8eb', led: '#db9ae9', track: 'rgba(255,255,255,.09)', name: 'sampler', sub: 'one-shot',
             knobs: ['cutoff', 'start', 'reverse', 'decay'] },
  synth: { body: 'linear-gradient(180deg,#171c26,#10141b)', ink: '#c7dcf5', led: '#5fd3c6', track: 'rgba(255,255,255,.09)', name: 'synth', sub: 'fm · lfo',
           knobs: ['cutoff', 'reso', 'pitch_env', 'lfo_rate'] },
  // the chip: a dark ceramic package with the home computer's beige print and its light-blue screen colour
  sid: { body: 'linear-gradient(180deg,#2b2621,#1a1613)', ink: '#d9caa4', led: '#8f85ea', track: 'rgba(255,255,255,.09)', name: 'sid', sub: 'chip',
         knobs: ['cutoff', 'reso', 'f_env', 'pwm'] },
};
const NS = 'http://www.w3.org/2000/svg';
const MODE_SHORT = ['LP', 'BP', 'HP', 'NOTCH'];
const NOISE_Y = [7, 2, 11, 5, 12, 3, 9, 1, 8, 13, 4, 10, 2, 12, 6, 9, 3, 11, 5, 13, 7];
// one oscillator's waveform, two cycles in a 40 x 14 box
function wavePath(wave, duty) {
  if (wave === 1) return 'M0,7 L5,1 L15,13 L25,1 L35,13 L40,7';
  if (wave === 2) return 'M0,13 L13,1 L13,13 L26,1 L26,13 L39,1 L39,13';
  if (wave === 3) { const d = 20 * Math.max(0.08, Math.min(0.92, duty)); return `M0,13 L0,1 L${d.toFixed(1)},1 L${d.toFixed(1)},13 L20,13 L20,1 L${(20 + d).toFixed(1)},1 L${(20 + d).toFixed(1)},13 L40,13`; }
  if (wave === 4) return 'M' + NOISE_Y.map((y, i) => `${i * 2},${y}`).join(' L');
  return 'M0,7 L40,7';
}
// the envelope the four rate nibbles draw: slower rates are wider
function adsrPath(o) {
  const a = 2 + (o.attack / 15) * 9, d = 2 + (o.decay / 15) * 9, r = 2 + (o.release / 15) * 9, y = 13 - (o.sustain / 15) * 12;
  return `M0,13 L${a.toFixed(1)},1 L${(a + d).toFixed(1)},${y.toFixed(1)} L${(a + d + 7).toFixed(1)},${y.toFixed(1)} L${(a + d + 7 + r).toFixed(1)},13`;
}
function glyph(parent, cls) {
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', '-1 0 42 14'); svg.setAttribute('class', cls);
  const path = document.createElementNS(NS, 'path'); svg.appendChild(path); parent.appendChild(svg);
  return path;
}
const hz = (f) => (f >= 1000 ? `${(f / 1000).toFixed(f >= 10000 ? 0 : 1)} kHz` : `${Math.round(f)} Hz`);
const MOD_COL = '#f2a65a';
const LABEL = Object.fromEntries([...UI.acid.knobs, ...UI.synth.knobs, ...UI.sampler.knobs, ...UI.sid.knobs].map((k) => [k.name, k.label]));

const el = (tag, cls, parent, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
};
const lerp = (a, b, t) => a + (b - a) * t;
const num = (p, k, d) => (p && typeof p[k] === 'number' ? p[k] : d);

function knob(parent, label) {
  const k = el('div', 'knob', parent);
  const dial = el('div', 'dial', k);
  el('div', 'kl', k, label);
  return dial;
}
// value ring: 270 degrees from bottom-left; mods (-1..1) draw from the top centre
function ring(dial, u, col, track, bipolar) {
  u = Math.max(0, Math.min(1, u));
  if (bipolar) {
    const a = (u - 0.5) * 270;
    const [s, e] = a >= 0 ? [0, a] : [360 + a, 360];
    dial.style.background = `conic-gradient(${a >= 0 ? `${col} 0deg ${e}deg, ${track} ${e}deg 135deg, transparent 135deg 225deg, ${track} 225deg 360deg`
      : `${track} 0deg 135deg, transparent 135deg 225deg, ${track} 225deg ${s}deg, ${col} ${s}deg 360deg`})`;
  } else {
    const d = u * 270;
    dial.style.background = `conic-gradient(from 225deg, ${col} 0deg ${d}deg, ${track} ${d}deg 270deg, transparent 270deg 360deg)`;
  }
}

export class Rack {
  constructor(root) { this.root = root; this.pool = null; this.units = []; this.step = -1; this.kit = null; }

  // the recorded one-shots the page loaded (site/kit.js): {device: {voice: samples}}, or null for the models
  setKit(kit) {
    this.kit = kit ? Object.fromEntries(Object.entries(kit).map(([dev, voices]) => [dev, new Set(Object.keys(voices || {}))])) : null;
    for (const u of this.units) if (u.kind === 'drum' && u.device) this._buildDrum(u, u.device);
  }

  setPool(pool) {
    this.pool = pool;
    this.root.innerHTML = '';
    this.units = pool.layers.map((layer, l) => (l === 0 ? this._drum() : this._synth(layer.name)));
    // before the first bar a unit shows the instrument this pool holds for its layer
    this.units.forEach((u, l) => {
      const first = (pool.layers[l].tokens || []).find((t) => t);
      if (u.kind === 'drum') this._buildDrum(u, (first && LOOK[first.device] && first.device) || '808');
      else this._buildSynth(u, (first && UNIT[first.voice] && first.voice) || 'synth');
    });
    this.cur = [0, 0, 0, 0]; this.prev = [0, 0, 0, 0];
    this.mods = [[0, 0], [0, 0], [0, 0], [0, 0]]; this.prevMods = this.mods;
  }

  _drum() {
    const box = el('div', 'unit drum', this.root);
    const face = el('div', 'face', box);
    const model = el('span', 'model', face);
    const sub = el('span', 'sub', face);
    const tok = el('span', 'tok', face);
    const grid = el('div', 'grid', box);
    const lights = el('div', 'lights', box);
    el('span', '', lights);
    const leds = Array.from({ length: 16 }, (_, i) => el('span', `led${i % 4 === 0 && i ? ' q' : ''}`, lights));
    const knobs = el('div', 'knobs', box);
    return { kind: 'drum', box, model, sub, tok, grid, leds, knobs, device: null, rows: {}, dials: [], mod: null };
  }

  _buildDrum(u, device) {
    const look = LOOK[device] || LOOK['808'];
    u.device = device; u.look = look;
    u.box.style.background = look.body; u.box.style.color = look.ink;
    // what plays: the recorded one-shots of this machine where the page has them, the model for the rest
    const recorded = (this.kit && this.kit[device]) || new Set(), voices = UI.drums.devices[device].voices;
    const all = voices.every((v) => recorded.has(v.name));
    u.model.textContent = device;
    u.sub.textContent = !recorded.size ? look.sub : all ? 'recorded one-shots' : 'recorded one-shots · model where marked';
    u.grid.innerHTML = ''; u.rows = {};
    for (const v of voices) {
      const row = el('div', 'row', u.grid);
      const rl = el('span', 'rl', row, DRUM_LABELS[v.name] || v.name);
      if (recorded.size && !recorded.has(v.name)) { rl.classList.add('modelled'); rl.title = 'played by the model: no recording of this voice'; }
      u.rows[v.name] = { row, cells: Array.from({ length: 16 }, (_, i) => el('span', `cell${i % 4 === 0 && i ? ' q' : ''}`, row)) };
    }
    u.leds.forEach((d) => { d.style.background = ''; });
    u.knobs.innerHTML = '';
    u.mod = [knob(u.knobs, 'FILTER'), knob(u.knobs, 'SPACE')];
    u.dials = [];
    u.track = device === '808' ? 'rgba(255,255,255,.1)' : 'rgba(0,0,0,.18)';
  }

  _synth(name) {
    const box = el('div', 'unit synthunit', this.root);
    const face = el('div', 'face', box);
    const model = el('span', 'model', face);
    const sub = el('span', 'sub', face);
    const tok = el('span', 'tok', face);
    const roll = el('div', 'roll', box);
    const cols = Array.from({ length: 16 }, (_, i) => { const c = el('div', `col${i % 4 === 0 ? ' q' : ''}`, roll); return { c, note: el('div', 'note', c) }; });
    const chip = el('div', 'chip', box);
    const knobs = el('div', 'knobs', box);
    return { kind: 'synth', name, box, model, sub, tok, cols, chip, knobs, voice: null, dials: [], mod: null };
  }

  _buildSynth(u, voice) {
    const look = UNIT[voice] || UNIT.synth;
    u.voice = voice; u.look = look;
    u.box.style.background = look.body; u.box.style.color = look.ink;
    u.model.textContent = u.name; u.sub.textContent = `${look.name} · ${look.sub}`;
    u.chip.innerHTML = ''; u.osc = null;
    u.box.classList.toggle('sid', voice === 'sid');
    if (voice === 'sid') {
      // one row per oscillator: waveform, interval, pulse width, envelope, and whether it passes the filter
      u.osc = [1, 2, 3].map((k) => {
        const row = el('div', 'osc', u.chip);
        el('span', 'on', row, String(k));
        const wave = glyph(row, 'wv'), text = el('span', 'ot', row), env = glyph(row, 'ev'), pass = el('span', 'fl', row, 'F');
        pass.title = 'passes the filter';
        return { row, wave, text, env, pass };
      });
      const line = el('div', 'flt', u.chip);
      u.flt = { mode: el('span', 'fm', line), freq: el('span', '', line), fx: el('span', 'fx', line) };
    }
    u.knobs.innerHTML = '';
    u.mod = [knob(u.knobs, 'FILTER'), knob(u.knobs, 'SPACE')];
    u.dials = look.knobs.map((k) => [k, knob(u.knobs, LABEL[k] || k.toUpperCase())]);
    u.track = look.track;
  }

  // a new bar starts playing: tokens, the bar before, the brain's modulation
  setBar(tokens, prev, mods, prevMods, patterns, patches, prevPatches) {
    if (!this.pool) return;
    this.patches = patches; this.prevPatches = prevPatches;
    this.cur = tokens.slice(); this.prev = prev.slice(); this.mods = mods || this.mods; this.prevMods = prevMods || this.mods;
    this.units.forEach((u, l) => {
      let t = patches?.[l] || this.pool.layers[l].tokens[tokens[l]] || null;
      if (t && patterns && patterns[l]) t = Object.assign({}, t, { pattern: patterns[l] });   // the brain wrote this bar's pattern
      // dark when the layer plays nothing this bar: no sound, or a synth line without a note (a rest row)
      const quiet = !t || (patterns && patterns[l] && Array.isArray(patterns[l].notes) && !patterns[l].notes.some((n) => n != null));
      // a beat sound whose pattern holds no hit plays nothing either
      const mute = u.kind === 'drum' && t && t.pattern && t.pattern.hits && !Object.values(t.pattern.hits).some((a) => Array.isArray(a) && a.some((x) => x > 0));
      u.box.classList.toggle('silent', !!(quiet || mute));
      // a patch the brain wrote control by control is no numbered sound of the pool
      u.tok.textContent = !t ? 'silent' : patches?.[l] && u.kind !== 'drum' ? 'written patch' : `sound ${String(tokens[l]).padStart(2, '0')}`;
      if (u.kind === 'drum') this._patternDrum(u, t);
      else this._patternSynth(u, t);
    });
  }

  _patternDrum(u, t) {
    const device = (t && t.device) || u.device || '808';
    if (device !== u.device) this._buildDrum(u, device);
    const hits = (t && t.pattern && t.pattern.hits) || {};
    // the knobs worth watching: decay of every voice that plays, then drive
    const voices = Object.keys(u.rows).filter((v) => (hits[v] || []).some((x) => x > 0));
    const want = [...voices.map((v) => `${v}_decay`), 'drive'].slice(0, 7);
    if (want.join() !== (u.want || []).join()) {
      u.want = want;
      u.dials.forEach(([, d]) => d.parentElement.remove());
      u.dials = want.map((k) => [k, knob(u.knobs, k === 'drive' ? 'DRIVE' : `${DRUM_LABELS[k.split('_')[0]]} DEC`)]);
    }
    for (const [v, r] of Object.entries(u.rows)) {
      const h = hits[v];
      r.vals = h || null;
      r.row.classList.toggle('none', !h || !h.some((x) => x > 0));
      r.cells.forEach((c, i) => {
        const x = h ? h[i] || 0 : 0;
        const col = u.look.steps[Math.floor(i / 4) % u.look.steps.length];
        c.style.backgroundColor = x > 0 ? col : u.look.off;
        c.style.opacity = x > 0 ? String(0.42 + 0.58 * Math.min(1, x)) : '1';
        c.style.boxShadow = 'none';
      });
    }
  }

  _patternSynth(u, t) {
    const voice = (t && t.voice) || u.voice || 'synth';
    if (voice !== u.voice) this._buildSynth(u, voice);
    if (t?.sample) u.tok.textContent = t.sample.replaceAll("_", " ");
    if (voice === 'sid' && u.osc) this._chip(u, t);
    const p = (t && t.pattern) || {};
    const notes = p.notes || [];
    const pitched = notes.filter((n) => n != null);
    const lo = pitched.length ? Math.min(...pitched) : 48, hi = pitched.length ? Math.max(...pitched) : 60;
    u.cols.forEach(({ note }, i) => {
      const n = notes[i];
      if (n == null) { note.style.display = 'none'; return; }
      note.style.display = 'block';
      note.style.bottom = `${6 + (hi === lo ? 0.5 : (n - lo) / (hi - lo)) * 76}%`;
      note.style.background = u.look.led; note.style.color = u.look.led;
      const acc = (p.accent && p.accent[i]) || (p.vel && p.vel[i] > 0.8);
      note.classList.toggle('acc', !!acc);
      note.classList.toggle('slide', !!(p.slide && p.slide[i]));
      note.style.opacity = acc ? '1' : '.62';
      note.style.boxShadow = 'none';
    });
    u.notes = notes;
  }

  // the chip's registers for this bar
  _chip(u, t) {
    const r = UI.sid.registers((t && t.patch) || {});
    r.osc.forEach((o, k) => {
      const e = u.osc[k], off = o.wave === 0, modulates = k === 2 && r.fx > 0;
      e.row.classList.toggle('off', off && !modulates);
      e.wave.setAttribute('d', wavePath(o.wave, o.pw));
      e.env.setAttribute('d', off ? '' : adsrPath(o));
      const tune = o.tune ? `${o.tune > 0 ? '+' : '−'}${Math.abs(o.tune)}` : '0';
      e.text.textContent = off ? (modulates ? `mod ${tune}` : 'off') : `${UI.sid.waves[o.wave].slice(0, 3)} ${tune}${o.wave === 3 ? ` · ${Math.round(o.pw * 100)}%` : ''}`;
      e.row.title = off ? (modulates ? `oscillator 3 is silent and modulates oscillator 1, ${tune} semitones` : `oscillator ${k + 1} is off`)
        : `oscillator ${k + 1}: ${UI.sid.waves[o.wave]}, ${tune} semitones, ${o.fine.toFixed(0)} cents${o.wave === 3 ? `, pulse width ${Math.round(o.pw * 100)}%` : ''}; attack ${o.attack}, decay ${o.decay}, sustain ${o.sustain}, release ${o.release}`;
      e.pass.classList.toggle('lit', r.filtered[k] && !off);
    });
    const routed = r.filtered.some((f, k) => f && r.osc[k].wave !== 0);
    u.flt.mode.textContent = routed ? MODE_SHORT[r.mode] : 'no filter';
    u.flt.freq.textContent = routed ? `${hz(r.cutoff)} · res ${Math.round(r.reso * 15)}` : '';
    u.flt.fx.textContent = r.fx ? `${UI.sid.fx[r.fx]} 3→1` : '';
  }

  // the sixteenth sounding now
  setStep(step) {
    if (!this.pool) return;
    const last = this.step; this.step = step;
    this.units.forEach((u) => {
      if (u.kind === 'drum') {
        if (u.leds[last]) u.leds[last].classList.remove('on');
        if (u.leds[step]) { u.leds[step].classList.add('on'); u.leds[step].style.setProperty('--c', u.look.led); }
        for (const r of Object.values(u.rows)) {
          const a = r.cells[last], b = r.cells[step];
          if (a) a.style.boxShadow = 'none';
          if (b && r.vals && r.vals[step] > 0) b.style.boxShadow = `0 0 10px 2px ${b.style.backgroundColor}`;
        }
      } else {
        const a = u.cols[last], b = u.cols[step];
        if (a) { a.c.classList.remove('now'); a.note.style.boxShadow = 'none'; }
        if (b) { b.c.classList.add('now'); if (u.notes && u.notes[step] != null) b.note.style.boxShadow = `0 0 12px 3px ${u.look.led}`; }
      }
    });
  }

  // every frame: phase 0..1 through the bar (knobs and modulation glide)
  update(phase) {
    if (!this.pool) return;
    this.units.forEach((u, l) => {
      const toks = this.pool.layers[l].tokens;
      const cur = this.patches?.[l] || toks[this.cur[l]] || null;
      let old = this.prevPatches?.[l] || toks[this.prev[l]] || cur;
      // the engine glides within one voice (and one drum machine) only: a new voice starts at its own settings
      if (old && cur && (old.voice !== cur.voice || old.device !== cur.device)) old = cur;
      const col = u.kind === 'drum' ? (u.look ? u.look.steps[0] : '#ccc') : (u.look ? u.look.led : '#ccc');
      for (const [k, dial] of u.dials) {
        const a = num(old && old.patch, k, 0), b = num(cur && cur.patch, k, a);
        ring(dial, lerp(a, b, phase), col, u.track);
      }
      if (u.mod) u.mod.forEach((dial, i) => {
        const a = this.prevMods ? this.prevMods[l][i] : 0, b = this.mods ? this.mods[l][i] : 0;
        ring(dial, 0.5 + 0.5 * lerp(a, b, phase), MOD_COL, u.track, true);
      });
    });
  }
}
