// Cadance radio, the brain stage: one picture of a deep brain that hears, plans and plays.
//
// Everything that moves is driven by data: the brain's packed 3D file (neurons, sampled synapses), a track's layout,
// and per bar the brain's recorded view (48 activation values per region, the recall heads' reads, the settle's
// steps and residual) with the bar's own tokens and patterns. The clock times the playing; an audio analyser, if
// given, only scales the overall level.
//
//   HEARS   five blocks of sensory neurons: the bar just heard and the bars 2, 4, 8 and 16 back, lit one neuron
//           per slot exactly as those bars' one-hot observation was (kit, voices, levels, drum steps, line steps)
//   PLANS   the learned code, each instrument's plan and detail regions, the coordinator, the upper coordinator:
//           every neuron at its recorded value, all of them disturbed at the bar line and relaxing together
//   PLAYS   the motor neurons laid out as the bar they write, and over them the bar as an instrument panel:
//           three drum rows (kick, snare, hats) with a pad per hit that flashes on its step, an accent larger and
//           brighter, and three lines (bass, lead, air) with a bar per note, as long as the note and as high as
//           its pitch class, under a playhead
//   MEMORY  a ring of the remembered bars; the recall heads' beams reach the cells they read in this bar
//
// The loop: each mark leaves the hands at its strike, travels under the brain and lands at the bar line in the same
// cell of the ear's first block, so the bar just played is what the ear's first block holds in the next bar; the
// older blocks step back. A second copy goes to the memory ring's newest cell, which fills as the bar is played.
//
// Colour: cyan is the base. Each instrument pathway (beat, bass, lead, air) has a tint, carried by its plan and
// detail neurons, its synapses, its rows of the hands and of the ear, and its marks. Each recall head has a tint,
// carried by its neurons, its beams and the cells it reads. The copy arcs are a warm white. Red and blue appear
// only as the split of the picture on a kick, a cut or a drop.
//
// Scenes: 'wired' (the whole brain), 'memory', 'stream', and two for a party: 'party' (the brain folded into a
// kaleidoscope) and 'tunnel' (a flight through the brain from the ear to the hands and back over the memory).
// In both the picture closes in and splits on the kick steps of the bar that sounds. On a section line (every
// 32 bars) the neurons go dark for a beat and only the bare wiring shows. No pulse comes within 0.36 s of the
// last one; calm mode has none, no drop, no fold and no zoom.
//
// Derived, not recorded: the light that travels along a synapse (the brain's file holds the synapse, the view
// holds its two regions' activity, not the current through it), the agitation's fine grain during a settle (the
// view holds the settle's steps and residual and each region's change, not its sweeps), a cell's tint in the
// memory ring (a hash of that bar's tokens and hits), the way and the timing of a mark's flight (the data says
// that the played bar is the next bar's heard bar and the memory's newest entry, not how it gets there), the
// partial step back of the deeper ear blocks, and every tint.
//
// One ES module, no dependency, no request. WebGL (1 or 2); without it a plain 2D schematic.
//
//   const stage = new BrainStage(canvas, { calm: false });
//   stage.setBrain(arrayBufferOrNull); stage.setLayout(layout); stage.setClock({ ctx, bpm, barZeroTime });
//   stage.setBar(index, bar); stage.setAudio(analyser); stage.setScene('wired'); stage.start();
//   also: stage.scenes(), setCalm(bool), setLabels(bool), resize(), stop(), stats(), and stage.ok (false: no canvas context)

const SCENES = ['wired', 'memory', 'stream', 'party', 'tunnel'];
// colour with meaning, all of it tints of the cyan and the ink: the base, then one per instrument pathway (beat, bass,
// lead, air), one per recall head, and a warm white for the copy arcs. Red and blue appear only in the split.
const LANE_TINT = [[0.29, 0.945, 0.949], [0.42, 1.0, 0.6], [0.22, 0.62, 1.0], [0.72, 0.9, 0.88], [0.74, 0.6, 1.0]];
const HEAD_TINT = [[0.45, 1.0, 0.82], [0.3, 0.78, 1.0], [0.55, 0.62, 1.0], [0.82, 0.86, 1.0]];
const WARM = [1.0, 0.9, 0.74];
const LANE_OF = { beat: 1, rhythm: 1, bass: 2, lead: 3, melody: 3, air: 4 };
const MONO = 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, "Liberation Mono", monospace';
const BINS = 48, VW = 64, VH = 64;            // the value texture: a row per region, a column per recorded value
const EW = 128, EH = 128;                     // the ear texture: one texel per sensory neuron
const MW = 64, MH = 32;                       // the motor texture: one texel per motor neuron
const RING = 128;
// a bar's columns as brain/common.py lays them out for the kept brains (1,720 values): the kit, three voices on or
// off, eight levels of seven, 48 drum steps of three values, 48 line steps of fourteen, then the sound controls
const BAR = { width: 1720, kit: 65, voices: 65, mods: 71, drums: 127, lines: 271, controls: 943, extras: 8 };
const LAGS = [1, 2, 4, 8, 16];

// ---------- small matrix kit (column major) ----------
const M = {
  persp(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  look(e, a, u) {
    let zx = e[0] - a[0], zy = e[1] - a[1], zz = e[2] - a[2];
    let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = u[1] * zz - u[2] * zy, xy = u[2] * zx - u[0] * zz, xz = u[0] * zy - u[1] * zx;
    l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
      -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  },
  rotZ(t) { const c = Math.cos(t), s = Math.sin(t); return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); },
};
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const mixv = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

// the bytes of a packed brain (site/brainview.py pack(), read as site/brainviz.js parseBrain3d reads them):
// "B3D1", the header's length, a JSON header, then arrays at 4-byte boundaries
function parseBrain(buf) {
  const dv = new DataView(buf);
  if (buf.byteLength < 8 || String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'B3D1') throw Error('not a packed brain');
  const hl = dv.getUint32(4, true), base = 8 + hl;
  // Python writes a missing number as NaN, which JSON does not have (the compact view's lobes of some brains)
  const text = new TextDecoder().decode(new Uint8Array(buf, 8, hl)).replace(/\bNaN\b|-?\bInfinity\b/g, 'null');
  const head = JSON.parse(text);
  const kinds = { '<f4': Float32Array, '|u1': Uint8Array, '<u2': Uint16Array, '<u4': Uint32Array, '|i1': Int8Array };
  const arr = (name) => { const [off, len, dt] = head.arrays[name], K = kinds[dt]; return new K(buf, base + off, len / K.BYTES_PER_ELEMENT); };
  return { head, pos: arr('position'), bin: arr('bin'), pre: arr('pre'), post: arr('post'), weight: arr('weight') };
}

// ---------- a bar as the brain holds it: one lit column per slot ----------
// Returns { cols: Uint8Array(1720) 1 where the slot's value sits, onset: Int8Array(1720) the step a motor neuron
// fires on (-1 never, -2 held for the whole bar), power, hold, kicks: the steps with a kick, tint, density }
function readBar(bar) {
  const W = BAR.width, cols = new Uint8Array(W), onset = new Int8Array(W).fill(-1), power = new Uint8Array(W), hold = new Uint8Array(W);
  const tok = (bar && bar.tokens) || [0, 0, 0, 0], pats = (bar && bar.patterns) || [];
  const held = (c, p) => { if (c >= 0 && c < W) { cols[c] = 1; onset[c] = -2; power[c] = p; } };
  if (tok[0] >= 0 && tok[0] < 65) held(tok[0], 150);
  for (let i = 1; i < 4; i++) held(BAR.voices + (i - 1) * 2 + (tok[i] > 0 ? 1 : 0), 150);
  const mods = (bar && bar.mods) || [];
  for (let l = 0; l < 4; l++) for (let k = 0; k < 2; k++) {
    const m = mods[l] ? +mods[l][k] || 0 : 0;
    held(BAR.mods + (l * 2 + k) * 7 + clamp(Math.round((m + 1) * 3), 0, 6), 110);
  }
  let hitsOf = null; const lines = [];
  for (const p of pats) { if (p && p.hits) hitsOf = p.hits; else if (p && p.notes) lines.push(p); }
  const rows = hitsOf ? [hitsOf.kick, hitsOf.snare || hitsOf.clap, hitsOf.oh || hitsOf.ch] : [];
  const kicks = [];
  let hash = (tok[0] * 131 + tok[1] * 17 + tok[2] * 7 + tok[3]) >>> 0, density = 0;
  for (let r = 0; r < 3; r++) for (let t = 0; t < 16; t++) {
    let h = rows[r] ? +rows[r][t] || 0 : 0;
    if (r === 2 && hitsOf && hitsOf.oh && hitsOf.ch) h = Math.max(h, +hitsOf.ch[t] || 0);
    const v = h > 0.75 ? 2 : h > 0 ? 1 : 0, c = BAR.drums + (r * 16 + t) * 3 + v;
    cols[c] = 1;
    if (v) { onset[c] = t; power[c] = v === 2 ? 255 : 170; hold[c] = 1; density++; if (r === 0) kicks.push(t); }
    else { onset[c] = -3; power[c] = 60; }
    hash = (Math.imul(hash, 31) + v * (r + 1)) >>> 0;
  }
  for (let l = 0; l < 3; l++) {
    const p = lines[l], sounding = tok[l + 1] > 0;
    if (!p || !sounding) continue;                       // a voice that does not sound is not heard
    let tie = 0;
    for (let t = 0; t < 16; t++) {
      const n = p.notes ? p.notes[t] : null;
      let v = 0;
      if (n !== null && n !== undefined) { v = 2 + (((n % 12) + 12) % 12); tie = Math.max(0, ((p.len && p.len[t]) || 1) - 1); }
      else if (tie > 0) { v = 1; tie--; }
      const c = BAR.lines + (l * 16 + t) * 14 + v;
      cols[c] = 1;
      if (v >= 2) { onset[c] = t; power[c] = 255; hold[c] = clamp((p.len && p.len[t]) || 1, 1, 16); density++; }
      else if (v === 1) { onset[c] = t; power[c] = 90; hold[c] = 1; }
      else { onset[c] = -3; power[c] = 60; }
      hash = (Math.imul(hash, 33) + v) >>> 0;
    }
  }
  // the tint of this bar's cell in the memory ring: a hash, inside the palette
  const a = (hash & 255) / 255, b = ((hash >>> 8) & 255) / 255;
  const base = mixv([0.29, 0.945, 0.949], [0.36, 0.76, 1.0], a * 0.5);
  const tint = mixv(base, [0.788, 0.984, 0.984], b * 0.5);
  return { cols, onset, power, hold, kicks, tint, density };
}

// where a bar's column sits on a block: (u, v) in 0..1 by -0.15..1; the ear's blocks and the hands share it
function slotUV(col) {
  if (col < BAR.voices) return [0.02 + (col / 65) * 0.5, 0.985];
  if (col < BAR.mods) return [0.56 + ((col - BAR.voices) / 6) * 0.07, 0.985];
  if (col < BAR.drums) return [0.66 + ((col - BAR.mods) / 56) * 0.32, 0.985];
  if (col < BAR.lines) {
    const k = col - BAR.drums, s = (k / 3) | 0, v = k % 3, r = (s / 16) | 0, t = s % 16;
    return [(t + 0.5) / 16, 0.9 - r * 0.075 + (v - 1) * 0.02];
  }
  if (col < BAR.controls) {
    const k = col - BAR.lines, s = (k / 14) | 0, v = k % 14, l = (s / 16) | 0, t = s % 16;
    return [(t + 0.5) / 16 + (v === 1 ? 0.008 : 0), 0.5 - l * 0.2 + (v / 13) * 0.16];
  }
  if (col < BAR.width) {
    const k = col - BAR.controls, l = (k / 259) | 0, j = k % 259;
    return [((j % 37) + 0.5) / 37, 0.035 - l * 0.065 - ((j / 37) | 0) * 0.008];
  }
  return [0.02 + ((col - BAR.width) / 8) * 0.12, 1.03];            // the mood and the bar clock, beside the kit
}
// the pathway a slot belongs to: the drum steps to the beat, a line's steps and its sound controls to that line
const slotLane = (col) => (col < BAR.drums ? 0 : col < BAR.lines ? 1 : col < BAR.controls ? 2 + ((((col - BAR.lines) / 14) | 0) / 16 | 0) : col < BAR.width ? 2 + (((col - BAR.controls) / 259) | 0) : 0);
const slotClass = (col) => (col < BAR.drums ? 0 : col < BAR.lines ? 1 : col < BAR.controls ? 2 : col < BAR.width ? 3 : 4);

// ---------- shaders (GLSL ES 1.00, so WebGL 1 and 2 both take them) ----------
const VS_POINT = `
attribute vec3 aPos; attribute vec4 aInfo; attribute vec4 aAux;
uniform mat4 uMVP; uniform sampler2D uVal, uEar, uMotor, uRing;
uniform float uTime, uPhase, uSettle, uEarIn, uEarShift, uLevel, uKick, uPx, uRingShift, uParty, uOn;
uniform vec3 uGlow, uInk; uniform vec3 uLane[5]; uniform vec3 uHead[4];
varying vec4 vCol;
float sq(float x) { return x * x; }
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
void main() {
  float kind = aInfo.x, seed = aInfo.w;
  float lane = kind < 2.5 ? floor(aAux.w / 10.0 + 0.01) : 0.0, cls = aAux.w - lane * 10.0;
  vec3 tint = uLane[int(clamp(lane, 0.0, 4.0) + 0.5)];
  vec3 pos = aPos; float b = 0.0; vec3 col = tint; float size = aAux.x;
  if (kind < 0.5 || (kind > 2.5 && kind < 3.5)) {                 // a hidden region or a recall head: its recorded values
    vec4 t = texture2D(uVal, vec2((aInfo.z + 0.5) / ${VW}.0, (aAux.y + 0.5) / ${VH}.0));
    float cur = t.r * t.r, prev = t.g * t.g, v = mix(prev, cur, uSettle);
    float grain = hash(seed * 91.0 + floor(uTime * 24.0)) * (1.0 - uSettle) * (0.25 + 0.9 * t.b);
    b = 0.15 + 2.0 * pow(v, 0.62) + grain * 0.9 + uKick * 0.06;
    col = mix(tint, uInk, clamp(v * 2.2 + grain, 0.0, 1.0) * (lane > 0.5 ? 0.35 : 1.0));
    if (kind > 2.5) { col = mix(uHead[int(clamp(aAux.z, 0.0, 3.0) + 0.5)], uInk, clamp(v, 0.0, 0.4)); b *= 0.34; }
    size *= 1.0 + 1.3 * v + grain * 0.8;
  } else if (kind < 1.5) {                                        // the ear: one neuron per slot value of a heard bar
    vec4 t = texture2D(uEar, vec2((mod(aInfo.y, ${EW}.0) + 0.5) / ${EW}.0, (floor(aInfo.y / ${EW}.0) + 0.5) / ${EH}.0));
    float lit = mix(t.g, t.r, uEarIn), near = 1.0 - 0.17 * aAux.z;
    float shimmer = 0.75 + 0.25 * sin(uTime * 3.0 + seed * 40.0);
    b = (cls > 2.5 ? 0.06 : 0.11) * near + lit * near * (1.7 * shimmer + 0.7 * (1.0 - uEarIn) * t.r);
    col = mix(tint, uInk, lit * 0.4);
    size *= 1.0 + lit * 1.6;
    // at the bar line every heard bar is one bar older: a block's content steps back from where it was a bar ago
    // (the block of two bars back takes the whole place of the first, the deeper blocks a half, a quarter, an eighth)
    if (aAux.z > 0.5) pos += vec3(0.2, -0.44, 0.6) * exp2(1.0 - aAux.z) * uEarShift;
  } else if (kind < 2.5) {                                        // the hands: a slot's winning neuron fires on its step
    vec4 t = texture2D(uMotor, vec2((mod(aInfo.y, ${MW}.0) + 0.5) / ${MW}.0, (floor(aInfo.y / ${MW}.0) + 0.5) / ${MH}.0));
    float code = floor(t.r * 255.0 + 0.5), power = t.g, holdN = max(1.0, floor(t.b * 255.0 + 0.5));
    vec4 rv = texture2D(uVal, vec2((aInfo.z + 0.5) / ${VW}.0, (aAux.y + 0.5) / ${VH}.0));
    float rec = mix(rv.g * rv.g, rv.r * rv.r, uSettle);
    b = (cls > 2.5 ? 0.07 : 0.12) + 0.45 * rec;
    if (code > 199.5) { b += (code > 205.0 ? 1.1 : 1.5) * power * (0.7 + 0.3 * uSettle); col = mix(tint, uInk, code > 205.0 ? 0.0 : 0.4); size *= code > 205.0 ? 0.9 : 1.4; }   // a rest, or held all bar
    else if (code > 0.5) {
      float d = uPhase * 16.0 - (code - 1.0);
      float pre = (d < 0.0 ? 0.8 : 1.1) * power * uSettle;        // decided at the bar line: to come, then played
      float fire = d >= 0.0 ? power * (exp(-d * 1.6 / holdN) * 2.4 + (d < holdN ? 0.6 : 0.0)) : 0.0;
      b += pre + fire;
      col = mix(tint, uInk, clamp(fire * 0.5 + 0.1, 0.0, 0.8));
      size *= 1.0 + min(fire, 2.2) * 0.6 + pre * 0.5;
    }
  } else if (kind < 4.5) {                                        // a cell of the memory ring
    float cell = aInfo.y;
    vec4 t = texture2D(uRing, vec2((cell + 0.5) / ${RING}.0, 0.25)), rd = texture2D(uRing, vec2((cell + 0.5) / ${RING}.0, 0.75));
    float read = rd.a, flare = read * (0.5 + 0.5 * uSettle) * (1.0 + 0.6 * sin(uTime * 9.0 + seed * 6.0));
    b = 0.2 + 1.1 * dot(t.rgb, vec3(0.34)) * (1.0 - 0.4 * cell / ${RING}.0);
    if (cell < 0.5) b *= 0.25 + 0.75 * uPhase;                   // the newest cell fills as its bar is played
    b += 3.4 * flare;
    col = mix(t.rgb, rd.rgb, clamp(flare * 1.8, 0.0, 1.0));      // a cell that is read takes its reader's tint
    size *= 1.0 + 2.6 * flare + (cell < 0.5 ? 1.2 : 0.0);
    float th = aInfo.z - uRingShift * ${(Math.PI * 2 / RING).toFixed(6)};
    pos = vec3(aAux.y + cos(th) * aAux.z * (1.0 + seed * 0.05), pos.y - sin(th) * 0.5 + seed * 0.02, sin(th) * aAux.w * (1.0 + seed * 0.05));
  }
  b *= uLevel * (1.0 - uParty * 0.35) * uOn;
  vec4 p = uMVP * vec4(pos, 1.0);
  gl_Position = p;
  float ps = size * uPx / max(p.w, 0.2) * (1.0 + uKick * 0.25);
  gl_PointSize = clamp(ps, 1.0, 72.0);
  vCol = vec4(col, min(b, 3.5) * clamp(34.0 / ps, 0.2, 1.0));    // a neuron close to the eye is large, not blinding
}`;
const FS_POINT = `
precision mediump float; varying vec4 vCol;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0, a = max(0.0, 1.0 - d); a *= a;
  gl_FragColor = vec4(vCol.rgb * vCol.a * a, 1.0);
}`;
const VS_LINE = `
attribute vec3 aPos; attribute vec4 aL; attribute vec2 aM;
uniform mat4 uMVP; uniform sampler2D uVal;
uniform float uTime, uPhase, uSettle, uLevel, uKick, uLines, uParty, uBare;
uniform vec3 uGlow, uInk, uWarm; uniform vec3 uLane[5]; uniform vec3 uHead[4];
varying vec4 vCol;
float sq(float x) { return x * x; }
void main() {
  float t = aL.x, type = aL.y, seed = aL.z, w = aL.w;
  vec3 col = uGlow; float b = 0.0;
  if (type < 0.5) {                               // a synapse of the sample: light runs from its sender to its receiver
    vec3 tint = uLane[int(clamp(aM.y, 0.0, 4.0) + 0.5)];          // a synapse of an instrument's pathway carries its tint
    vec4 m = texture2D(uVal, vec2(63.5 / ${VW}.0, (aM.x + 0.5) / ${VH}.0));
    float act = mix(m.g, m.r, uSettle);
    float p = fract(t * 0.9 - uTime * (0.33 + 0.25 * seed) + seed * 7.0);
    float bump = exp(-sq((p - 0.5) * 7.0));
    b = w * (0.016 + (0.16 + 1.4 * act) * bump * (0.55 + 0.45 * uSettle) + (1.0 - uSettle) * 0.07);
    b = mix(b, w * 0.24, uBare);
    col = mix(tint, uInk, bump * 0.5 * (1.0 - uBare));
  } else if (type < 1.5) {                        // a copy arc: straight from a heard value to the same played value
    float near = aM.y < 0.5 ? 1.0 : 0.42;
    float p = uSettle * 1.25 - 0.12, bump = exp(-sq((t - p) * 5.0)) * (1.0 - uSettle * uSettle);
    float flow = exp(-sq((fract(t - uTime * 0.21 + seed) - 0.5) * 6.0));
    b = near * w * (0.045 + 0.62 * bump + 0.14 * flow);
    b = mix(b, near * w * 0.22, uBare);
    col = uWarm;
  } else if (type < 2.5) {                        // the rails of the stream: the way the played bar takes back to the ear
    b = 0.016 + 0.03 * step(0.5, fract(t * 26.0 + seed)); col = uGlow;
  } else if (type < 3.5) {                        // the rails into the memory ring
    b = 0.016 + 0.03 * step(0.5, fract(t * 20.0 + seed)); col = uGlow;
  } else if (type < 4.5) {                        // a recall head reading a remembered bar: from the cell to the head
    float p = fract(t * 0.6 - uTime * 0.9 + seed);
    float bump = exp(-sq((p - 0.5) * 5.0));
    b = w * (0.4 + 2.3 * bump) * (0.35 + 0.65 * uSettle);
    col = mix(uHead[int(clamp(aM.y, 0.0, 3.0) + 0.5)], uInk, bump * 0.55);
  } else {                                        // plain: frames, guides, the residual meter, the playhead
    b = w; col = aM.y > 1.5 ? uLane[int(clamp(aM.y - 1.0, 0.0, 4.0) + 0.5)] : (aM.y > 0.5 ? uInk : uGlow);
  }
  gl_Position = uMVP * vec4(aPos, 1.0);
  vCol = vec4(col, min(b, 3.0) * uLevel * uLines * (1.0 - uParty * 0.2 + uKick * 0.15));
}`;
const FS_LINE = `precision mediump float; varying vec4 vCol; void main() { gl_FragColor = vec4(vCol.rgb * vCol.a, 1.0); }`;
// the marks: what the hands play, drawn as a drum machine and three lines. A pad per drum hit, a bar per line note
// (as long as the note, as high as its pitch class). Mode 0 sits on the hands; mode 1 is the same mark leaving at
// its strike for the same cell of the ear's first block, where it lands at the bar line; mode 2 goes to the
// memory ring's newest cell.
const VS_MARK = `
attribute vec3 aA; attribute vec3 aC; attribute vec3 aB; attribute vec3 aD; attribute vec4 aK; attribute vec4 aN;
uniform mat4 uMVP; uniform float uPhase, uOnly, uLevel, uStream, uShrink;
uniform vec3 uLane[5];
varying vec4 vCol; varying vec4 vT; varying vec3 vQ;
void main() {
  float on = aK.x / 16.0, len = aK.y / 16.0, mode = aN.z;
  vec3 pos = aA; float b = 1.0;
  if (mode > 0.5) {
    float arrive = mode < 1.5 ? 1.0 + 0.07 * on : on + 0.4;
    float f = clamp((uPhase - on) / (arrive - on), 0.0, 1.0), s = f * f * (3.0 - 2.0 * f);
    vec3 c = (1.0 - s) * (1.0 - s) * aC + 2.0 * (1.0 - s) * s * aD + s * s * aB;
    float shrink = mode < 1.5 ? mix(1.0, uShrink, s) * (1.0 + (0.25 + 0.9 * uStream) * sin(3.14159 * s)) : mix(1.0, 0.2, s);
    pos = c + (aA - aC) * shrink;
    float gone = uPhase > arrive ? exp(-(uPhase - arrive) * (mode < 1.5 ? 24.0 : 30.0)) : 1.0;
    b = step(on, uPhase) * (mode < 1.5 ? 0.7 + 0.4 * uStream + 0.6 * sin(3.14159 * f) : 0.7) * gone;
    if (uOnly > 0.5 && mode > 1.5) b = 0.0;
  } else if (uOnly > 0.5) b = 0.0;
  gl_Position = uMVP * vec4(pos, 1.0);
  vCol = vec4(uLane[int(clamp(aK.w, 0.0, 4.0) + 0.5)], b * uLevel);
  vT = vec4(on, len, mode, aK.z); vQ = vec3(aN.xy, uPhase);
}`;
const FS_MARK = `
precision mediump float; varying vec4 vCol; varying vec4 vT; varying vec3 vQ;
uniform vec3 uInk;
void main() {
  float on = vT.x, len = vT.y, mode = vT.z, power = vT.w, phase = vQ.z;
  float shape = (1.0 - smoothstep(0.6, 1.0, abs(vQ.y))) * (1.0 - smoothstep(0.88, 1.0, abs(vQ.x * 2.0 - 1.0)));
  float b = vCol.a; vec3 col = vCol.rgb;
  if (mode < 0.5) {
    float since = phase - on;
    if (since < 0.0) b *= 0.17 + 0.1 * power;                      // decided, still to come
    else {
      float flash = exp(-since * 24.0) * (1.2 + 2.4 * power);      // the strike: a flash and a short decay
      float body = 0.42 + 0.3 * power, head = 0.0;
      if (len > 0.001) {                                           // a line note: lit up to the playhead while it sounds
        float at = on + vQ.x * len, held = step(phase, on + len);
        body = at <= phase ? mix(0.5, 1.0, held) : 0.2;
        head = held * exp(-abs(at - phase) * 70.0) * 1.5;
      }
      b *= body + flash + head;
      col = mix(col, uInk, clamp(flash * 0.4 + head * 0.6, 0.0, 0.85));
    }
  }
  gl_FragColor = vec4(col * b * shape, 1.0);
}`;
const VS_QUAD = `attribute vec2 aP; varying vec2 vUV; void main() { vUV = aP * 0.5 + 0.5; gl_Position = vec4(aP, 0.0, 1.0); }`;
const FS_BLUR = `
precision mediump float; varying vec2 vUV; uniform sampler2D uTex; uniform vec2 uDir;
void main() {
  vec3 c = texture2D(uTex, vUV).rgb * 0.227;
  c += (texture2D(uTex, vUV + uDir * 1.385).rgb + texture2D(uTex, vUV - uDir * 1.385).rgb) * 0.316;
  c += (texture2D(uTex, vUV + uDir * 3.231).rgb + texture2D(uTex, vUV - uDir * 3.231).rgb) * 0.070;
  gl_FragColor = vec4(c, 1.0);
}`;
// the frame kept between frames: the new picture over a fading, slowly turning copy of the last one (the trails)
const FS_MIX = `
precision mediump float; varying vec2 vUV; uniform sampler2D uScene, uBloom, uPrev;
uniform float uDecay, uZoom, uTurn, uBloomGain;
void main() {
  vec2 c = vUV - 0.5; float s = sin(uTurn), k = cos(uTurn);
  vec2 q = vec2(c.x * k - c.y * s, c.x * s + c.y * k) * uZoom + 0.5;
  vec3 prev = texture2D(uPrev, q).rgb * uDecay;
  vec3 now = texture2D(uScene, vUV).rgb + texture2D(uBloom, vUV).rgb * uBloomGain;
  gl_FragColor = vec4(max(now, prev), 1.0);
}`;
const FS_OUT = `
precision mediump float; varying vec2 vUV; uniform sampler2D uTex;
uniform float uFold, uMirror, uSplit, uGhost, uAspect, uScan, uTime, uGain;
uniform vec3 uRed, uBlue;
vec2 fold(vec2 uv) {
  vec2 c = (uv - 0.5) * vec2(uAspect, 1.0);
  float r = length(c), a = atan(c.y, c.x), seg = 3.14159265 / 6.0;
  a = mod(a + uTime * 0.05, 2.0 * seg); a = abs(a - seg);
  // drawn larger, and on an upright canvas the folded slice runs along the long side, where the brain is
  vec2 f = (uAspect < 1.0 ? vec2(sin(a), cos(a)) : vec2(cos(a), sin(a))) * r * 0.6;
  return f / vec2(uAspect, 1.0) + 0.5;
}
void main() {
  vec2 uv = mix(vUV, fold(vUV), uFold);
  vec3 c = texture2D(uTex, uv).rgb;
  if (uMirror > 0.01) c = max(c, texture2D(uTex, vec2(1.0 - uv.x, uv.y)).rgb * uMirror);     // the tunnel: left and right mirror each other
  if (uGhost > 0.004) {                           // the split: the picture's red and blue ghosts part, on a kick or a cut
    vec2 d = (uv - 0.5) * uSplit + vec2(uSplit * 0.4, 0.0);
    float l1 = dot(texture2D(uTex, uv + d).rgb, vec3(0.3, 0.45, 0.25)), l2 = dot(texture2D(uTex, uv - d).rgb, vec3(0.3, 0.45, 0.25));
    c = c * (1.0 - 0.25 * uGhost) + (uRed * l1 + uBlue * l2) * uGhost;
  }
  float v = 1.0 - 0.55 * dot(vUV - 0.5, vUV - 0.5) * 2.0;
  c *= v * uGain * (1.0 - uScan * 0.07 * step(0.5, fract(gl_FragCoord.y * 0.5)));
  c = c / (1.0 + 0.5 * c);                        // a soft shoulder: the glow does not clip to flat white
  gl_FragColor = vec4(c * 1.38, 1.0);
}`;
const VS_LABEL = `attribute vec4 aQ; varying vec2 vUV; void main() { vUV = aQ.zw; gl_Position = vec4(aQ.xy, 0.0, 1.0); }`;
const FS_LABEL = `precision mediump float; varying vec2 vUV; uniform sampler2D uTex; uniform float uA; uniform vec3 uCol;
void main() { float a = texture2D(uTex, vUV).a * uA; gl_FragColor = vec4(uCol * a, a); }`;

export class BrainStage {
  constructor(canvas, { calm = false } = {}) {
    this.canvas = canvas;
    this.calm = !!calm || (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.scene = 'wired'; this.from = null; this.sceneAt = 0;
    this.labels = true;
    this.clock = { ctx: null, bpm: 122, barZeroTime: null };
    this.bars = new Map();                       // the bars this stage was given, by index: what the ear and the ring show
    this.cur = null; this.curIndex = -1; this.barAt = 0;
    this.settleWin = 0.23; this.residual = 0; this.tolerance = 0.003;
    this.kick = 0; this.lastFlash = -9; this.split = 0; this.playedKick = -1;
    this.dropping = false; this.thin = 0; this.partyK = 0; this.streamK = 0; this.tunnelK = 0;
    this.level = 1; this.analyser = null; this.spec = null;
    this.brain = null; this.layout = null; this.geo = null;
    this.running = false; this.raf = 0; this.frameMs = 16; this.scale = 1; this.slow = 0; this.frames = 0;
    this.gl = null; this.ctx2d = null;
    this._loop = this._loop.bind(this);
    this._init();
  }

  scenes() { return SCENES.slice(); }
  get ok() { return !!(this.gl || this.ctx2d); }

  setBrain(buf) {
    try { this.brain = buf ? parseBrain(buf) : null; this.brainError = null; } catch (e) { this.brain = null; this.brainError = String(e && e.message || e); }
    this._build();
  }

  setLayout(layout) {
    const changed = JSON.stringify(layout && layout.regions) !== JSON.stringify(this.layout && this.layout.regions);
    this.layout = layout || null;
    if (changed) { this.bars.clear(); this.cur = null; this.curIndex = -1; this._build(); }
  }

  setClock({ ctx = null, bpm = 122, barZeroTime = null } = {}) { this.clock = { ctx, bpm: bpm || 122, barZeroTime }; }
  setAudio(analyser) { this.analyser = analyser || null; this.spec = analyser ? new Uint8Array(analyser.frequencyBinCount) : null; }
  setCalm(on) { this.calm = !!on; }
  setLabels(on) { this.labels = !!on; }
  setScene(name) {
    if (!SCENES.includes(name) || name === this.scene) return;
    this.from = this._camera(this.scene, this._now()); this.scene = name; this.sceneAt = this._now();
    this._pulse(this._now(), 0, 1);
  }

  // every full-picture pulse goes through here: never two within 0.36 s (under three a second), none in calm mode
  _pulse(now, kick, split) {
    if (this.calm || Math.abs(now - this.lastFlash) < 0.36) return false;
    this.lastFlash = now; if (kick) this.kick = kick; this.split = Math.max(this.split, split);
    return true;
  }

  // the bar that sounds now
  setBar(index, bar) {
    if (!bar) return;
    if (this.cur && index === this.curIndex && bar === this.cur.bar) return;
    if (index < this.curIndex - 1 || index > this.curIndex + 64) this.bars.clear();       // another track, or a jump
    const read = readBar(bar);
    this.bars.set(index, read);
    for (const k of this.bars.keys()) if (k < index - 160) this.bars.delete(k);
    const same = index === this.curIndex;
    this.cur = { index, bar, read, view: bar.view || null };
    this.curIndex = index;
    if (!same) { this.barAt = this._now(); this.playedKick = -1; }
    const s = (bar.view && bar.view.settle) || {};
    const share = s.budget > 0 ? clamp((s.steps || 0) / s.budget, 0, 1) : 0.03;
    this.settleWin = clamp(0.14 + 0.5 * Math.sqrt(share), 0.14, 0.6);       // the settle's length on screen follows its steps
    this.residual = +s.residual || 0; this.tolerance = +s.tolerance || 0.003;
    if (!same && index % 8 === 7) this._pulse(this._now(), 0, 0.6);         // the bar before a phrase turns
    this._uploadBar();
  }

  resize() {
    const c = this.canvas, dpr = Math.min(2, (typeof devicePixelRatio === 'number' ? devicePixelRatio : 1)) * this.scale;
    const w = Math.max(2, Math.round((c.clientWidth || c.width || 2) * dpr)), h = Math.max(2, Math.round((c.clientHeight || c.height || 2) * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (this.gl) this._targets();
  }

  start() { if (this.running) return; this.running = true; this.last = performance.now(); this.raf = requestAnimationFrame(this._loop); }
  stop() { this.running = false; if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }
  stats() { return { frameMs: this.frameMs, scale: this.scale, thin: this.thin, points: this.geo ? this.geo.nPoints : 0, lines: this.geo ? this.geo.nLineVerts / 2 : 0, webgl: !!this.gl }; }

  // ---------- time ----------
  _now() { const c = this.clock.ctx; return c && typeof c.currentTime === 'number' ? c.currentTime : performance.now() / 1000; }
  _barDur() { return 240 / (this.clock.bpm || 122); }
  _phase(now) {
    const z = this.clock.barZeroTime, since = (now - this.barAt) / this._barDur();
    if (z !== null && z !== undefined && this.curIndex >= 0) {
      const p = (now - z) / this._barDur() - this.curIndex;
      if (p > -0.05 && p < 1.25) return clamp(p, 0, 1.25);
    }
    return clamp(since, 0, 1.25);
  }

  // ---------- set up ----------
  _init() {
    const c = this.canvas, opts = { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: 'high-performance' };
    let gl = null;
    try { gl = c.getContext('webgl2', opts) || c.getContext('webgl', opts) || c.getContext('experimental-webgl', opts); } catch (e) { gl = null; }
    if (gl && gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS) < 4) gl = null;
    if (!gl) { try { this.ctx2d = c.getContext('2d'); } catch (e) { this.ctx2d = null; } return; }
    this.gl = gl;
    c.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });
    c.addEventListener('webglcontextrestored', () => { this.lost = false; this._gpu(); this._upload(); this._uploadBar(); });
    this._gpu();
  }

  _program(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s) || 'shader');
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p) || 'link');
    const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, name); }
    const a = {}, m = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES);
    for (let i = 0; i < m; i++) { const name = gl.getActiveAttrib(p, i).name; a[name] = gl.getAttribLocation(p, name); }
    return { p, u, a };
  }

  _texture(w, h, filter) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  _gpu() {
    const gl = this.gl;
    this.prog = { point: this._program(VS_POINT, FS_POINT), line: this._program(VS_LINE, FS_LINE), mark: this._program(VS_MARK, FS_MARK), blur: this._program(VS_QUAD, FS_BLUR),
      mix: this._program(VS_QUAD, FS_MIX), out: this._program(VS_QUAD, FS_OUT), label: this._program(VS_LABEL, FS_LABEL) };
    this.buf = { quad: gl.createBuffer(), point: gl.createBuffer(), line: gl.createBuffer(), beam: gl.createBuffer(), mark: gl.createBuffer(), label: gl.createBuffer() };
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    this.tex = { val: this._texture(VW, VH, gl.NEAREST), ear: this._texture(EW, EH, gl.NEAREST), motor: this._texture(MW, MH, gl.NEAREST),
      ring: this._texture(RING, 2, gl.NEAREST), label: this._texture(2, 2, gl.LINEAR) };
    this.px = { val: new Uint8Array(VW * VH * 4), ear: new Uint8Array(EW * EH * 4), motor: new Uint8Array(MW * MH * 4), ring: new Uint8Array(RING * 2 * 4) };
    this.markCap = 0; this.markVerts = [0, 0];
    this.laneFlat = new Float32Array(LANE_TINT.flat()); this.headFlat = new Float32Array(HEAD_TINT.flat());
    this.beamData = new Float32Array(64 * 16 * 2 * 9); this.beamVerts = 0;
    this.labelData = new Float32Array(96 * 6 * 4);
    this.fbo = null; this.size = [0, 0];
    this._atlas();
    this.resize();
  }

  _target(w, h) {
    const gl = this.gl, tex = this._texture(w, h, gl.LINEAR), fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    return { fb, tex, w, h };
  }

  _targets() {
    const gl = this.gl, w = this.canvas.width, h = this.canvas.height;
    if (this.fbo && this.size[0] === w && this.size[1] === h) return;
    if (this.fbo) for (const t of Object.values(this.fbo)) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
    const q = (n) => Math.max(2, Math.round(n / 4));
    this.fbo = { scene: this._target(w, h), b1: this._target(q(w), q(h)), b2: this._target(q(w), q(h)), keepA: this._target(w, h), keepB: this._target(w, h) };
    this.size = [w, h];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // the labels: drawn once into a texture in the site's mono type
  _atlas() {
    const gl = this.gl, c = document.createElement('canvas'); c.width = 1024; c.height = 512;
    const g = c.getContext('2d'); this.atlas = {};
    const big = ['HEARS', 'PLANS', 'PLAYS', 'MEMORY'];
    const small = ['the code', 'coordinator', 'upper coordinator', 'prefrontal', 'beat plan', 'bass plan', 'lead plan', 'air plan',
      'rhythm plan', 'melody plan', 'beat detail', 'bass detail', 'lead detail', 'air detail', 'rhythm detail', 'melody detail', 'plan', 'detail',
      'beat', 'bass', 'lead', 'air', 'rhythm', 'melody', 'kick', 'snare', 'hats'];
    let x = 4, y = 4, rowH = 0;
    const put = (text, px, track) => {
      g.font = `${px >= 30 ? 500 : 400} ${px}px ${MONO}`;
      const chars = text.split(''), widths = chars.map((ch) => g.measureText(ch).width + track), w = Math.ceil(widths.reduce((a, b) => a + b, 0)) + 6, h = Math.ceil(px * 1.4);
      if (x + w > c.width - 4) { x = 4; y += rowH + 4; rowH = 0; }
      g.fillStyle = '#fff'; g.textBaseline = 'middle';
      let cx = x + 3; for (let i = 0; i < chars.length; i++) { g.fillText(chars[i], cx, y + h / 2); cx += widths[i]; }
      this.atlas[text] = { u0: x / c.width, v0: y / c.height, u1: (x + w) / c.width, v1: (y + h) / c.height, w, h, px };
      x += w + 6; rowH = Math.max(rowH, h);
    };
    for (const t of big) put(t, 34, 9);
    for (const t of small) put(t, 22, 1.5);
    gl.bindTexture(gl.TEXTURE_2D, this.tex.label);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  }

  // ---------- the geometry: every drawn neuron and the sampled wiring, in this stage's own arrangement ----------
  // Neurons keep the shape their region has in the packed brain; the regions are placed so the picture reads left
  // to right: hears, plans, plays, with the memory above. The ear and the hands are laid out as the bar they hold.
  _build() {
    const L = this.layout, B = this.brain;
    if (!L && !B) { this.geo = null; return; }
    const R = rng(7);
    // the regions to draw: from the packed brain when there is one, else from the layout
    let regs;
    if (B) regs = B.head.regions.map((r) => ({ name: r.name, start: r.start, count: r.count, neurons: r.neurons, centre: r.centre, radius: r.radius }));
    else { let at = 0; regs = L.regions.map((r) => { const count = Math.min(r.neurons, r.name === 'sensory' ? 12000 : 2400), o = { name: r.name, start: at, count, neurons: r.neurons, centre: [0, 0, 0], radius: 0.2 }; at += count; return o; }); }
    const rowOf = {}; (L ? L.regions : regs).forEach((r, i) => { if (i < VH) rowOf[r.name] = i; });
    this.rowOf = rowOf;
    const names = regs.map((r) => r.name);
    const lanes = [];                               // the pathways: a detail and a plan region each
    for (const key of ['beat', 'rhythm', 'bass', 'lead', 'melody', 'air']) if (names.includes(key + '_detail') || names.includes(key + '_plan')) lanes.push(key);
    for (const n of names) { const m = /^(.*)_(detail|plan)$/.exec(n); if (m && !lanes.includes(m[1])) lanes.push(m[1]); }
    const HEAR = { x: -5.0, y: -1.3, w: 1.3, h: 2.4 }, PLAY = { x: 3.3, y: -1.75, w: 2.3, h: 3.25 };
    const rowY = (v) => PLAY.y + ((v + 0.15) / 1.2) * PLAY.h;
    this.rowY = rowY;
    const laneIdx = lanes.map((k, i) => LANE_OF[k] || 1 + (i % 4));
    const laneOf = (name) => { const m = /^(.*)_(detail|plan)$/.exec(name), i = m ? lanes.indexOf(m[1]) : -1; return i >= 0 ? laneIdx[i] : 0; };
    // a pathway sits level with the rows of the hands it plays: the drum rows, then the three lines
    const laneRow = [0, 0.825, 0.58, 0.38, 0.18];
    const laneY = (i) => (lanes.length <= 1 ? 0 : lanes.length === 4 ? rowY(laneRow[laneIdx[i]]) * 0.96 : 0.72 - (1.46 * i) / (lanes.length - 1));
    const heads = names.filter((n) => /^recall(_\d+)?$/.test(n));
    const upper = names.includes('phrase_context');
    const anchor = {}; const scale = {};
    this.HEAR = HEAR; this.PLAY = PLAY;
    anchor.code = [-2.7, -0.15, 0]; scale.code = 1.3;
    anchor.prefrontal = [-2.55, 1.3, 0]; scale.prefrontal = 1.15;
    anchor.association = [-0.15, 1.5, 0.1]; scale.association = 1.25;
    anchor.phrase_context = [0.95, 1.82, -0.1]; scale.phrase_context = 1.25;
    lanes.forEach((k, i) => { anchor[k + '_plan'] = [-0.75, laneY(i) + 0.04, 0.3 * (i % 2 ? -1 : 1)]; anchor[k + '_detail'] = [1.15, laneY(i), 0.12 * (i % 2 ? 1 : -1)]; scale[k + '_plan'] = 1.1; scale[k + '_detail'] = 1.1; });
    const modules = names.filter((n) => /^module_\d+$/.test(n));
    modules.forEach((n, i) => { anchor[n] = [-1.7 + (2.4 * i) / Math.max(1, modules.length - 1), 0.1 * (i % 2 ? 1 : -1), 0]; scale[n] = 0.9; });
    heads.forEach((n, i) => { anchor[n] = [-0.7 + (heads.length > 1 ? (3.1 * i) / (heads.length - 1) : 1.55), 2.16, 0.55]; scale[n] = 0.5; });
    const ring = { c: [0.85, 2.72, 0], rx: 4.1, rz: 1.25, cap: (L && L.capacity) || (B && B.head.capacity) || 0 };
    this.ringGeo = ring;
    const centre = {};                             // where a region ended up, for labels and beams
    const P = [], I = [], A = [];                   // positions, info (kind, a, b, seed), aux (size, row, block, class)
    const where = new Float32Array(regs.reduce((s, r) => Math.max(s, r.start + r.count), 0) * 3).fill(NaN);
    const add = (x, y, z, kind, a, b, size, row, blk, cls) => { P.push(x, y, z); I.push(kind, a, b, R()); A.push(size, row, blk, cls); };
    const exact = regs.some((r) => r.name === 'motor' && r.count === BAR.width && r.neurons === BAR.width);
    const sens = regs.find((r) => r.name === 'sensory');
    const lagBlocks = sens && sens.count === sens.neurons && (sens.neurons - BAR.width - BAR.extras) % BAR.width === 0 ? 1 + (sens.neurons - BAR.width - BAR.extras) / BAR.width : 0;
    this.lagBlocks = exact ? lagBlocks : 0; this.exact = exact;
    const blockOrigin = (b) => ({ x: HEAR.x - 0.2 * b, y: HEAR.y + 0.44 * b, z: 0.45 - 0.6 * b, s: 1 - 0.1 * b });
    this.blockOrigin = blockOrigin;
    for (const r of regs) {
      const row = rowOf[r.name] === undefined ? VH - 1 : rowOf[r.name];
      if (/^next/.test(r.name) || /_when$/.test(r.name) || r.name === 'other') continue;    // the successor ports and the two-neuron clocks are not drawn
      if (r.name === 'sensory' && this.lagBlocks) {
        for (let i = 0; i < r.count; i++) {
          const first = BAR.width + BAR.extras, b = i < first ? 0 : 1 + (((i - first) / BAR.width) | 0), col = i < first ? i : (i - first) % BAR.width;
          const [u, v] = slotUV(col), o = blockOrigin(b);
          const x = o.x + u * HEAR.w * o.s, y = o.y + ((v + 0.15) / 1.2) * HEAR.h * o.s, z = o.z;
          where[(r.start + i) * 3] = x; where[(r.start + i) * 3 + 1] = y; where[(r.start + i) * 3 + 2] = z;
          add(x, y, z, 1, i, B ? B.bin[r.start + i] : 0, slotClass(col) === 3 ? 1.6 : 2.7, row, b, slotClass(col) + 10 * slotLane(col));
        }
        centre.sensory = [HEAR.x + HEAR.w / 2, HEAR.y + HEAR.h / 2, 0.45];
        continue;
      }
      if (r.name === 'motor' && exact) {
        for (let i = 0; i < r.count; i++) {
          const [u, v] = slotUV(i);
          const x = PLAY.x + u * PLAY.w, y = PLAY.y + ((v + 0.15) / 1.2) * PLAY.h, z = 0;
          where[(r.start + i) * 3] = x; where[(r.start + i) * 3 + 1] = y; where[(r.start + i) * 3 + 2] = z;
          add(x, y, z, 2, i, B ? B.bin[r.start + i] : Math.floor((i * BINS) / r.count), slotClass(i) === 3 ? 2.0 : 3.4, row, 0, slotClass(i) + 10 * slotLane(i));
        }
        centre.motor = [PLAY.x + PLAY.w / 2, PLAY.y + PLAY.h / 2, 0];
        continue;
      }
      // every other region: its own shape from the packed brain, scaled, at this stage's place for it
      let a = anchor[r.name];
      if (!a) {
        if (r.name === 'sensory') a = [HEAR.x + 0.4, -0.2, 0];
        else if (r.name === 'motor') a = [PLAY.x + 0.5, -0.2, 0];
        else a = [-0.6 + 1.6 * R(), -1.2 - 0.3 * R(), 0.6 * (R() - 0.5)];
      }
      const s = scale[r.name] || (r.name === 'sensory' || r.name === 'motor' ? 0.6 : 0.85), head = heads.indexOf(r.name);
      const stride = head >= 0 ? 3 : 1;                         // a recall head holds as many neurons as the code: one in three is drawn
      for (let i = 0; i < r.count; i += stride) {
        let lx, ly, lz;
        if (B) { lx = B.pos[(r.start + i) * 3] - r.centre[0]; ly = B.pos[(r.start + i) * 3 + 1] - r.centre[1]; lz = B.pos[(r.start + i) * 3 + 2] - r.centre[2]; }
        else {
          const rad = clamp(0.042 * Math.cbrt(r.neurons), 0.08, 0.5) * Math.cbrt(R());
          const th = R() * Math.PI * 2, z0 = R() * 2 - 1, q = Math.sqrt(1 - z0 * z0);
          lx = rad * q * Math.cos(th); ly = rad * z0; lz = rad * q * Math.sin(th);
        }
        const x = a[0] + lx * s, y = a[1] + ly * s, z = a[2] + lz * s;
        where[(r.start + i) * 3] = x; where[(r.start + i) * 3 + 1] = y; where[(r.start + i) * 3 + 2] = z;
        const bin = B ? Math.min(B.bin[r.start + i], BINS - 1) : Math.floor((i * BINS) / r.count);
        add(x, y, z, head >= 0 ? 3 : 0, 0, bin, r.name === 'code' ? 3.3 : head >= 0 ? 2.3 : 4.8, row, Math.max(head, 0), head >= 0 ? 0 : 10 * laneOf(r.name));
      }
      centre[r.name] = a;
    }
    // the memory ring: a tick of points per remembered bar, the newest in front
    this.ringStart = P.length / 3;
    if (ring.cap) {
      for (let c = 0; c < Math.min(ring.cap, RING); c++) {
        const th = Math.PI / 2 + (c / RING) * Math.PI * 2;
        for (let k = 0; k < 9; k++) add(0, ring.c[1] + (k - 4) * 0.016, 0, 4, c, th, 4.2 - Math.abs(k - 4) * 0.3, ring.c[0], ring.rx + (k - 4) * 0.018, ring.rz + (k - 4) * 0.008);
      }
    }
    // ----- the wiring: sampled synapses as curves between the two neurons they join -----
    const LP = [], LL = [], LM = [];
    const curve = (p0, p1, ctl, type, seed, w, row, flag, segs) => {
      let px = p0[0], py = p0[1], pz = p0[2];
      for (let i = 1; i <= segs; i++) {
        const t = i / segs, u = 1 - t;
        const x = u * u * p0[0] + 2 * u * t * ctl[0] + t * t * p1[0], y = u * u * p0[1] + 2 * u * t * ctl[1] + t * t * p1[1], z = u * u * p0[2] + 2 * u * t * ctl[2] + t * t * p1[2];
        LP.push(px, py, pz, x, y, z); LL.push((i - 1) / segs, type, seed, w, t, type, seed, w); LM.push(row, flag, row, flag);
        px = x; py = y; pz = z;
      }
    };
    this._curve = curve;
    const regionAt = (idx) => { for (const r of regs) if (idx >= r.start && idx < r.start + r.count) return r; return null; };
    const lists = new Map();
    if (B) {
      for (let k = 0; k < B.pre.length; k++) {
        const a = B.pre[k], b = B.post[k];
        if (Number.isNaN(where[a * 3]) || Number.isNaN(where[b * 3])) continue;
        const ra = regionAt(a), rb = regionAt(b); if (!ra || !rb) continue;
        const key = ra.name + '>' + rb.name; let list = lists.get(key);
        if (!list) { list = { ra, rb, syn: [] }; lists.set(key, list); }
        list.syn.push(k);
      }
    } else {
      // no packed brain: the pathways a layout of this family has, drawn between random members of the two regions
      const pairs = [['sensory', 'code'], ['sensory', 'motor'], ['code', 'association'], ['prefrontal', 'code'], ['association', 'motor'], ['motor', 'association']];
      for (const k of lanes) pairs.push(['code', k + '_detail'], [k + '_detail', k + '_plan'], [k + '_plan', k + '_detail'], [k + '_plan', 'association'], ['association', k + '_plan'], ['association', k + '_detail'], [k + '_detail', 'motor'], ['motor', k + '_detail']);
      for (const h of heads) pairs.push([h, upper ? 'phrase_context' : 'association']);
      if (upper) pairs.push(['phrase_context', 'association'], ['association', 'phrase_context'], ['phrase_context', 'motor'], ['motor', 'phrase_context']);
      for (const [x, y] of pairs) {
        const ra = regs.find((r) => r.name === x), rb = regs.find((r) => r.name === y); if (!ra || !rb) continue;
        const syn = []; for (let i = 0; i < 40; i++) syn.push([ra.start + Math.floor(R() * ra.count), rb.start + Math.floor(R() * rb.count)]);
        lists.set(x + '>' + y, { ra, rb, syn, made: true });
      }
    }
    // how many of a pathway's synapses are drawn; each list leads with its strongest, so a short draw keeps the pathway
    const quota = (key) => (key === 'sensory>motor' ? 320 : key === 'sensory>code' ? 230 : key === 'code>code' ? 30 : /^recall/.test(key) ? 20 : /association|phrase_context/.test(key) ? 16 : 26);
    const copies = lists.get('sensory>motor');
    if (copies && !copies.made && this.lagBlocks) {              // of the sampled arcs, those on the diagonal: heard value to the same played value
      const first = BAR.width + BAR.extras, colOf = (a) => { const i = a - copies.ra.start; return i < first ? i : (i - first) % BAR.width; };
      const diag = copies.syn.filter((k) => colOf(B.pre[k]) === B.post[k] - copies.rb.start && slotClass(colOf(B.pre[k])) < 3);
      diag.sort((x, y) => (B.post[x] - B.post[y]) || (B.pre[x] - B.pre[y]));
      if (diag.length >= 40) copies.pick = diag;
    }
    const fan = lists.get('sensory>code');
    if (fan && !fan.made) fan.pick = fan.syn.slice().sort((x, y) => B.pre[x] - B.pre[y]);
    const rounds = []; let more = true, round = 0, halfAt = -1;
    while (more) {                                              // round robin over the pathways: a shortened draw thins all of them alike
      more = false;
      for (const [key, list] of lists) {
        const q = quota(key), per = Math.ceil(q / 6), from = round * per, to = Math.min(list.syn.length, q, from + per);
        if (from < to) { rounds.push([key, list, from, to]); more = true; }
      }
      round++;
      if (round === 3) halfAt = rounds.length;
    }
    let halfVerts = 0, nth = 0;
    for (const [key, list, from, to] of rounds) {
      if (nth++ === halfAt) halfVerts = LP.length / 3;       // a thin draw stops here: three of the six passes, every pathway kept
      const lane = laneOf(list.ra.name) || laneOf(list.rb.name);
      const copy = key === 'sensory>motor', rowA = rowOf[list.ra.name] === undefined ? VH - 1 : rowOf[list.ra.name];
      for (let j = from; j < to; j++) {
        let a, b, w;
        if (list.made) { [a, b] = list.syn[j]; w = 0.7; }
        else {
          let k = list.syn[j];
          if (copy || key === 'sensory>code') { const pick = list.pick || list.syn; k = pick[Math.floor((j * (pick.length - 1)) / Math.max(1, Math.min(pick.length, quota(key)) - 1))]; }   // spread over the list, not its strongest end alone
          a = B.pre[k]; b = B.post[k]; w = 0.45 + 0.55 * Math.sqrt(Math.abs(B.weight[k]) / 127);
        }
        const p0 = [where[a * 3], where[a * 3 + 1], where[a * 3 + 2]], p1 = [where[b * 3], where[b * 3 + 1], where[b * 3 + 2]];
        if (Number.isNaN(p0[0]) || Number.isNaN(p1[0])) continue;
        const seed = R();
        if (copy) {
          const blk = this.lagBlocks ? (a - list.ra.start < BAR.width + BAR.extras ? 0 : 1 + (((a - list.ra.start - BAR.width - BAR.extras) / BAR.width) | 0)) : 0;
          const ctl = [(p0[0] + p1[0]) / 2 - 0.2, -3.0 - 0.13 * blk - 0.25 * seed, 0.9 + 0.5 * seed];        // under the deep path, in front of it
          curve(p0, p1, ctl, 1, seed, w, rowA, blk, 20);
        } else {
          const dx = p1[0] - p0[0], dy = p1[1] - p0[1], len = Math.hypot(dx, dy) || 1, bow = (0.1 + 0.16 * seed) * (key.split('>')[0] < key.split('>')[1] ? 1 : -1);
          const ctl = [(p0[0] + p1[0]) / 2 - (dy / len) * bow * len * 0.5, (p0[1] + p1[1]) / 2 + (dx / len) * bow * len * 0.5, (p0[2] + p1[2]) / 2 + (seed - 0.5) * 0.5];
          curve(p0, p1, ctl, 0, seed, w * (key === 'sensory>code' ? 0.7 : 1), rowA, lane, key === 'code>code' ? 5 : 10);
        }
      }
    }
    const synVerts = LP.length / 3;
    // the rails of the stream: from the hands back to the ear, and into the memory ring
    const play = [PLAY.x + PLAY.w * 0.5, PLAY.y + 0.05, 0], hear = [HEAR.x + HEAR.w * 0.5, HEAR.y - 0.02, 0.45];
    for (let i = 0; i < 14; i++) {
      const s = R(), off = (i / 13 - 0.5);
      curve([play[0] + off * PLAY.w * 0.9, play[1], 0], [hear[0] + off * HEAR.w * 0.9, hear[1], hear[2]], [-0.4 + off * 0.6, -4.3 - 0.6 * off - 0.1 * s, 1.6], 2, s, 1, 0, 0, 30);
    }
    if (ring.cap) {
      const front = [ring.c[0], ring.c[1] - 0.5, ring.rz];
      for (let i = 0; i < 8; i++) {
        const s = R(), off = (i / 7 - 0.5);
        curve([PLAY.x + PLAY.w * (0.5 + off * 0.8), PLAY.y + PLAY.h + 0.04, 0], [front[0] + off * 0.3, front[1], front[2]], [PLAY.x + PLAY.w * 0.5 - 0.3 + off * 0.4, 2.95 + 0.2 * s, 1.2], 3, s, 1, 0, 0, 22);
      }
    }
    const plain = (a, b, w) => curve(a, b, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2], 5, 0, w, 0, 0, 1);
    if (ring.cap) for (let c = 0; c < RING; c++) {
      const t0 = Math.PI / 2 + (c / RING) * Math.PI * 2, t1 = Math.PI / 2 + ((c + 1) / RING) * Math.PI * 2;
      plain([ring.c[0] + Math.cos(t0) * ring.rx, ring.c[1] - Math.sin(t0) * 0.5, Math.sin(t0) * ring.rz], [ring.c[0] + Math.cos(t1) * ring.rx, ring.c[1] - Math.sin(t1) * 0.5, Math.sin(t1) * ring.rz], 0.07);
    }
    const frame = (x, y, w, h, z, a) => { const m = 0.05; plain([x - m, y - m, z], [x + w + m, y - m, z], a); plain([x + w + m, y - m, z], [x + w + m, y + h + m, z], a); plain([x + w + m, y + h + m, z], [x - m, y + h + m, z], a); plain([x - m, y + h + m, z], [x - m, y - m, z], a); };
    if (exact) {
      frame(PLAY.x, PLAY.y, PLAY.w, PLAY.h * 0.99, 0, 0.16);
      // the hands as an instrument panel: a rail per drum row, the edges of each line's pitch strip, a tick per beat
      const X = (u) => PLAY.x + u * PLAY.w, rule = (a, b, w, flag) => curve(a, b, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0], 5, 0, w, 0, flag, 1);
      for (let r = 0; r < 3; r++) rule([X(0), rowY(0.9 - r * 0.075), 0], [X(1), rowY(0.9 - r * 0.075), 0], 0.1, 2);
      for (let l = 0; l < 3; l++) for (const v of [1.45, 13.55]) { const y = rowY(0.5 - l * 0.2 + (v / 13) * 0.16); rule([X(0), y, 0], [X(1), y, 0], 0.085, 3 + l); }
      for (let t = 4; t < 16; t += 4) rule([X(t / 16), rowY(0.09), 0], [X(t / 16), rowY(0.945), 0], 0.06, 0);
    }
    if (this.lagBlocks) for (let b = 0; b < this.lagBlocks; b++) { const o = blockOrigin(b); frame(o.x, o.y, HEAR.w * o.s, HEAR.h * o.s * 0.99, o.z, 0.15 - 0.022 * b); }
    this.geo = { pos: new Float32Array(P), info: new Float32Array(I), aux: new Float32Array(A), nPoints: P.length / 3,
      lpos: new Float32Array(LP), ll: new Float32Array(LL), lm: new Float32Array(LM), nLineVerts: LP.length / 3, synVerts, halfVerts: halfVerts || synVerts, centre, lanes, laneIdx, heads, upper };
    for (const rd of this.bars.values()) rd.marks = null;
    this.headAt = heads.map((n) => anchor[n]);
    this.readFrom = upper ? anchor.phrase_context : anchor.association;
    this._labelsFor();
    this._upload();
    this._uploadBar();
  }

  _labelsFor() {
    const g = this.geo, c = g.centre, H = this.HEAR, Pl = this.PLAY, out = [], rowY = this.rowY;
    const big = (t, p) => out.push({ t, p, big: true }), small = (t, p, side, more) => out.push(Object.assign({ t, p, side: side || 0, lane: 0 }, more || {}));
    big('HEARS', [H.x + H.w * 0.3, -3.08, 0.4]); big('PLANS', [0.2, -3.08, 0.4]); big('PLAYS', [Pl.x + Pl.w * 0.5, -3.08, 0.4]);
    for (const l of out) l.ty = -2.86;                 // upright, the captions stand closer to the brain
    if (this.ringGeo.cap) big('MEMORY', [this.ringGeo.c[0], this.ringGeo.c[1] + 0.78, 0]);
    // first come, first placed: a label that would cover an earlier one is left out
    if (this.exact) {
      const rows = [['kick', 0.9, 1, 0], ['snare', 0.825, 1, 1.05], ['hats', 0.75, 1, 0], ['bass', 0.58, 2, 0], ['lead', 0.38, 3, 0], ['air', 0.18, 4, 0]];
      for (const [t, v, lane, lift] of rows) small(t, [Pl.x - 0.1, rowY(v), 0], -1, { lane, up: true, lift });
    }
    if (c.code) small('the code', [c.code[0], c.code[1] - 0.9, 0]);
    if (c.prefrontal) small('prefrontal', [c.prefrontal[0], c.prefrontal[1] + 0.36, 0]);
    if (c.association) small('coordinator', [c.association[0] - 0.32, c.association[1] + 0.06, 0], -1);
    if (c.phrase_context) small('upper coordinator', [c.phrase_context[0] + 0.32, c.phrase_context[1] - 0.02, 0], 1);
    let lowest = null;
    g.lanes.forEach((k, i) => {
      const lane = g.laneIdx[i], pl = c[k + '_plan'], de = c[k + '_detail'];
      if (pl) { small(k + ' plan', [pl[0] - 0.3, pl[1] + 0.02, 0], -1, { lane, wide: true }); small(k, [pl[0] - 0.34, pl[1], 0], 0, { lane, tall: true }); }
      if (de) small(k + ' detail', [de[0] + 0.3, de[1] - 0.02, 0], 1, { lane, wide: true });
      if (pl && de && (!lowest || pl[1] < lowest[0][1])) lowest = [pl, de];
    });
    if (lowest) { small('plan', [lowest[0][0], lowest[0][1] - 0.42, 0], -1, { tall: true }); small('detail', [lowest[1][0], lowest[1][1] - 0.42, 0], -1, { tall: true }); }
    this.labelList = out;
  }

  // the bar as marks on the hands: a pad per drum hit (an accent larger and brighter), a bar per line note. Every
  // mark has two copies that leave it at its strike: one for the same cell of the ear's first block, one for the
  // memory ring's newest cell. 20 numbers a corner: where it is, the mark's centre, the copy's goal, the bend of
  // its way, (step, length in steps, power, pathway), (along, across, mode, seed).
  _marks(rd) {
    if (rd.marks) return rd.marks;
    const out = [], Pl = this.PLAY, H = this.HEAR, o = this.blockOrigin(0), rg = this.ringGeo, rowY = this.rowY;
    const X = (u) => Pl.x + u * Pl.w, eX = (u) => o.x + u * H.w * o.s, eY = (v) => o.y + ((v + 0.15) / 1.2) * H.h * o.s;
    const front = [rg.c[0], rg.c[1] - 0.5, rg.rz], midY = Pl.y + Pl.h / 2, z = 0.03;
    const mark = (u, v, hw, hh, on, len, power, lane) => {
      const cx = X(u), cy = rowY(v), seed = (on * 0.37 + v * 5.1 + lane * 0.21) % 1;
      const modes = [[0, [cx, cy, z], [cx, cy, z]]];
      if (this.lagBlocks) modes.push([1, [eX(u), eY(v), o.z], [(cx + eX(u)) / 2, -4.25 - 1.7 * ((cy - midY) / Pl.h), 1.6]]);
      if (rg.cap) modes.push([2, [front[0] + (seed - 0.5) * 0.08, front[1], front[2]], [cx - 0.3, 2.95, 1.2]]);
      for (const [mode, goal, bend] of modes) {
        for (const [qx, qy] of [[0, -1], [1, -1], [0, 1], [0, 1], [1, -1], [1, 1]]) {
          out.push(cx + (qx * 2 - 1) * hw, cy + qy * hh, z, cx, cy, z, goal[0], goal[1], goal[2], bend[0], bend[1], bend[2], on, len, power, lane, qx, qy, mode, seed);
        }
      }
    };
    for (let r = 0; r < 3; r++) for (let t = 0; t < 16; t++) {
      const c = BAR.drums + (r * 16 + t) * 3, acc = rd.cols[c + 2];
      if (rd.cols[c + 1] || acc) mark((t + 0.5) / 16, 0.9 - r * 0.075, (Pl.w / 16) * (acc ? 0.44 : 0.33), (0.075 / 1.2) * Pl.h * (acc ? 0.42 : 0.31), t, 0, acc ? 1 : 0.5, 1);
    }
    for (let l = 0; l < 3; l++) for (let t = 0; t < 16; t++) {
      const c = BAR.lines + (l * 16 + t) * 14;
      for (let v = 2; v < 14; v++) if (rd.cols[c + v]) {
        const len = Math.min(16 - t, Math.max(1, rd.hold[c + v]));
        mark((t + len / 2) / 16, 0.5 - l * 0.2 + (v / 13) * 0.16, (Pl.w / 32) * len - 0.014, 0.036, t, len, 0.75, 2 + l);
      }
    }
    rd.marks = new Float32Array(out);
    return rd.marks;
  }

  _upload() {
    const gl = this.gl, g = this.geo; if (!gl || !g) return;
    const point = new Float32Array(g.nPoints * 11);
    for (let i = 0; i < g.nPoints; i++) { point.set(g.pos.subarray(i * 3, i * 3 + 3), i * 11); point.set(g.info.subarray(i * 4, i * 4 + 4), i * 11 + 3); point.set(g.aux.subarray(i * 4, i * 4 + 4), i * 11 + 7); }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.point); gl.bufferData(gl.ARRAY_BUFFER, point, gl.STATIC_DRAW);
    const line = new Float32Array(g.nLineVerts * 9);
    for (let i = 0; i < g.nLineVerts; i++) { line.set(g.lpos.subarray(i * 3, i * 3 + 3), i * 9); line.set(g.ll.subarray(i * 4, i * 4 + 4), i * 9 + 3); line.set(g.lm.subarray(i * 2, i * 2 + 2), i * 9 + 7); }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.line); gl.bufferData(gl.ARRAY_BUFFER, line, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.beam); gl.bufferData(gl.ARRAY_BUFFER, this.beamData, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.label); gl.bufferData(gl.ARRAY_BUFFER, this.labelData, gl.DYNAMIC_DRAW);
  }

  // ---------- a bar's data into the textures ----------
  _uploadBar() {
    const gl = this.gl, g = this.geo, cur = this.cur; if (!gl || !g || !cur || this.lost) return;
    const px = this.px, i = cur.index;
    // the regions' recorded values: this bar's beside the bar before's, and how much each region changed
    for (let k = 0; k < VW * VH; k++) { px.val[k * 4 + 1] = px.val[k * 4]; }
    const view = cur.view;
    if (view && view.regions) {
      for (const r of view.regions) {
        const row = this.rowOf[r.name]; if (row === undefined) continue;
        const vals = r.values || [], n = vals.length || 1, change = Math.sqrt(Math.max(0, +r.change || 0)) * 4;
        for (let b = 0; b < BINS; b++) {
          const v = clamp(+vals[Math.min(n - 1, Math.floor((b * n) / BINS))] || 0, 0, 1), o = (row * VW + b) * 4;
          px.val[o] = Math.round(Math.sqrt(v) * 255); px.val[o + 2] = Math.round(clamp(change, 0, 1) * 255);
        }
        px.val[(row * VW + 63) * 4] = Math.round(clamp(Math.sqrt(Math.max(0, +r.mean || 0)), 0, 1) * 255);
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.val);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, VW, VH, 0, gl.RGBA, gl.UNSIGNED_BYTE, px.val);
    // the ear: block b holds the bar LAGS[b] back, lit where that bar's slot values sit; dark where this stage was not given that bar
    for (let k = 0; k < EW * EH; k++) { px.ear[k * 4 + 1] = px.ear[k * 4]; px.ear[k * 4] = 0; }
    if (this.lagBlocks) {
      for (let b = 0; b < this.lagBlocks; b++) {
        const heard = this.bars.get(i - (LAGS[b] || b + 1)); if (!heard) continue;
        const base = b === 0 ? 0 : BAR.width + BAR.extras + (b - 1) * BAR.width;
        for (let c = 0; c < BAR.width; c++) if (heard.cols[c]) px.ear[(base + c) * 4] = 255;
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.ear);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, EW, EH, 0, gl.RGBA, gl.UNSIGNED_BYTE, px.ear);
    // the hands: the step each motor neuron fires on in the bar that sounds
    px.motor.fill(0);
    const rd = cur.read;
    for (let c = 0; c < BAR.width; c++) {
      const on = rd.onset[c]; if (on === -1) continue;
      px.motor[c * 4] = on === -2 ? 200 : on === -3 ? 210 : on + 1; px.motor[c * 4 + 1] = rd.power[c]; px.motor[c * 4 + 2] = rd.hold[c];
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.motor);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, MW, MH, 0, gl.RGBA, gl.UNSIGNED_BYTE, px.motor);
    // the memory ring: the cell at age a holds the bar a back; the heads' reads flare their cells in the reader's tint
    const reads = new Float32Array(RING), readTint = new Float32Array(RING * 3);
    const heads = (view && view.heads) || [];
    heads.forEach((h, k) => { const ht = HEAD_TINT[k % 4]; for (const r of h || []) if (r && r.age !== null && r.age >= 0 && r.age < RING) { const w = +r.w || 0; reads[r.age] += w; for (let q = 0; q < 3; q++) readTint[r.age * 3 + q] += ht[q] * w; } });
    for (let a = 0; a < RING; a++) {
      const held = this.bars.get(i - a), t = held ? held.tint : [0.16, 0.5, 0.52], d = held ? 0.6 + 0.4 * clamp(held.density / 40, 0, 1) : 0.42;
      px.ring[a * 4] = Math.round(t[0] * d * 255); px.ring[a * 4 + 1] = Math.round(t[1] * d * 255); px.ring[a * 4 + 2] = Math.round(t[2] * d * 255);
      px.ring[a * 4 + 3] = 255;
      const o2 = (RING + a) * 4, sum = reads[a] || 1;
      for (let q = 0; q < 3; q++) px.ring[o2 + q] = Math.round(clamp(readTint[a * 3 + q] / sum, 0, 1) * 255);
      px.ring[o2 + 3] = Math.round(clamp(Math.sqrt(reads[a]), 0, 1) * 255);
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.ring);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, RING, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, px.ring);
    // the heads' beams: from each cell a head reads to that head, as bright as the read's weight; then on to where the heads feed
    const D = this.beamData; let n = 0;
    const seg = (p0, p1, ctl, type, seed, w, segs, flag = 0) => {
      let x0 = p0[0], y0 = p0[1], z0 = p0[2];
      for (let s = 1; s <= segs && n + 18 <= D.length; s++) {
        const t = s / segs, u = 1 - t;
        const x = u * u * p0[0] + 2 * u * t * ctl[0] + t * t * p1[0], y = u * u * p0[1] + 2 * u * t * ctl[1] + t * t * p1[1], z = u * u * p0[2] + 2 * u * t * ctl[2] + t * t * p1[2];
        D.set([x0, y0, z0, (s - 1) / segs, type, seed, w, 0, flag, x, y, z, t, type, seed, w, 0, flag], n); n += 18;
        x0 = x; y0 = y; z0 = z;
      }
    };
    const rg = this.ringGeo;
    heads.forEach((h, k) => {
      const at = this.headAt[k]; if (!at) return;
      for (const r of h || []) {
        if (!r || r.age === null || r.age < 0 || r.age >= RING || !rg.cap) continue;
        const th = Math.PI / 2 + (r.age / RING) * Math.PI * 2;
        const cell = [rg.c[0] + Math.cos(th) * rg.rx, rg.c[1] - Math.sin(th) * 0.5, Math.sin(th) * rg.rz];
        const top = [at[0], at[1] + 0.2, at[2]];
        const w = clamp(Math.sqrt(+r.w || 0) * 1.25, 0.12, 1.3), strands = w > 0.5 ? 3 : w > 0.25 ? 2 : 1;
        for (let q = 0; q < strands; q++) {
          const o = (q - (strands - 1) / 2) * 0.022;
          seg([cell[0] + o, cell[1], cell[2]], [top[0] + o, top[1], top[2]], [(cell[0] + top[0]) / 2 + o, Math.max(cell[1], top[1]) + 0.1, (cell[2] + top[2]) / 2 + 0.35], 4, (k * 0.23 + r.age * 0.011 + q * 0.31) % 1, w, 14, k % 4);
        }
      }
    });
    // the residual meter: where this bar's settle ended against its tolerance, on a log scale of five decades
    const m0 = [-1.0, -1.34, 0.2], m1 = [1.7, -1.34, 0.2];
    seg(m0, m1, [(m0[0] + m1[0]) / 2, m0[1], m0[2]], 5, 0, 0.16, 1);
    this.meter = { at: n, m0, m1 };
    seg(m0, m0, m0, 5, 0, 1.0, 1);                 // the moving part: rewritten every frame
    seg(m0, m0, m0, 5, 0, 0.9, 1);                 // the tolerance tick
    for (let q = 0; q < 3; q++) seg(m0, m0, m0, 5, 0, 0.5, 1);       // the playhead on the hands: three strands
    // the marks of the bar that sounds, then those of the bar before (its copies land in the ear as this bar begins)
    this.markVerts = [0, 0];
    if (this.exact) {
      const a = this._marks(cur.read), before = this.bars.get(i - 1), b = before ? this._marks(before) : null, total = a.length + (b ? b.length : 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.mark);
      if (total > this.markCap) { this.markCap = Math.max(40000, total * 2); gl.bufferData(gl.ARRAY_BUFFER, this.markCap * 4, gl.DYNAMIC_DRAW); }
      if (a.length) gl.bufferSubData(gl.ARRAY_BUFFER, 0, a);
      if (b && b.length) gl.bufferSubData(gl.ARRAY_BUFFER, a.length * 4, b);
      this.markVerts = [a.length / 20, b ? b.length / 20 : 0];
      for (const [k, rd] of this.bars) if (k < i - 1 && rd.marks) rd.marks = null;      // only two bars keep their marks
    }
    this.beamVerts = n / 9;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.beam);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, D.subarray(0, n));
  }

  // ---------- camera ----------
  _camera(scene, now) {
    const calm = this.calm, t = now, bars = this.curIndex + this._phase(now);
    const sway = calm ? 0.25 : 1;
    if (scene === 'memory') {
      const a = Math.sin(t * 0.11) * 0.22 * sway;
      return { eye: [0.85 + Math.sin(a) * 6.6, 4.9 + 0.25 * Math.sin(t * 0.07) * sway, Math.cos(a) * 6.6], at: [0.85, 1.75, 0], fov: 0.8, fit: 0 };
    }
    if (scene === 'stream') {
      const turn = (bars / 8) * Math.PI * 2 * (calm ? 0.25 : 1), a = Math.sin(turn) * 0.5, r = 10.6 - 1.4 * Math.cos(turn * 2);
      return { eye: [-0.17 + Math.sin(a) * r, -1.9 + 1.7 * Math.cos(turn), Math.cos(a) * r], at: [-0.17 + Math.sin(a) * 2.6, -0.45, 0], fov: 0.8, fit: 1 };
    }
    if (scene === 'party') {
      const a = t * (calm ? 0.03 : 0.17), r = 4.9 + 1.5 * Math.sin(t * 0.23);
      return { eye: [Math.sin(a) * r, 0.7 + 1.2 * Math.sin(t * 0.13), Math.cos(a) * r], at: [-0.2, 0.3, 0], fov: 0.95, fit: 0, free: 1 };
    }
    if (scene === 'tunnel') {
      // a flight through the brain along its own axis, from behind the ear through the plans and out past the hands,
      // then back over the memory ring: one round in eight bars
      const th = (bars / 8) * Math.PI * 2 * (calm ? 0.25 : 1);
      // (between the blocks of two and four bars back, through the code, then along the face of the hands)
      const path = (q) => { const s = Math.sin(q), up = s < 0 ? s * s : 0, x = -0.2 - 7.6 * Math.cos(q); return [x, 0.25 + 3.3 * up + 0.2 * Math.sin(q * 3), -0.45 + 1.1 * ease((x + 3.9) / 5.5) + 2.2 * up]; };
      const eye = path(th), ahead = path(th + 0.4), s = Math.sin(th), up = s < 0 ? s * s : 0;
      return { eye, at: mixv([ahead[0], ahead[1], ahead[2] * 0.6 - 0.7 * ease((eye[0] + 2.5) / 3)], [-0.2, 0.4, 0], 0.55 * up), fov: 1.25, fit: 0, free: 1 };
    }
    const a = Math.sin(t * 0.09) * 0.045 * sway, b = Math.sin(t * 0.063 + 1.0) * 0.02 * sway;
    return { eye: [-0.17 + Math.sin(a) * 9.4, 0.3 + b * 9.4, Math.cos(a) * 9.4], at: [-0.17, 0.12, 0], fov: 0.64, fit: 1 };
  }

  _mvp(now) {
    const w = this.canvas.width, h = this.canvas.height, aspect = w / Math.max(1, h);
    let cam = this._camera(this.scene, now);
    if (this.from) {
      const k = ease((now - this.sceneAt) / 1.4);
      if (k >= 1) this.from = null;
      else cam = { eye: mixv(this.from.eye, cam.eye, k), at: mixv(this.from.at, cam.at, k), fov: this.from.fov + (cam.fov - this.from.fov) * k, fit: this.from.fit + (cam.fit - this.from.fit) * k, free: (this.from.free || 0) + ((cam.free || 0) - (this.from.free || 0)) * k };
    }
    const tall = aspect < 0.95;                       // a phone held upright: the picture reads top to bottom
    const model = tall ? M.rotZ(-Math.PI / 2) : M.rotZ(0);
    // pull the camera back until the whole brain fits the frame
    // in the party scenes the picture closes in on every kick of the bar
    const punch = this.calm ? 0 : 0.075 * this.kick * Math.max(this.partyK, this.tunnelK), fov = cam.fov * (1 - punch), free = cam.free || 0;
    const halfW = tall ? 3.32 : 5.9, halfH = tall ? 5.95 : 3.52, tanv = Math.tan(cam.fov / 2);
    const need = Math.max(halfH / tanv, halfW / (tanv * aspect)) * 1.05;
    const dir = [cam.eye[0] - cam.at[0], cam.eye[1] - cam.at[1], cam.eye[2] - cam.at[2]], dist = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const fit = tall ? Math.max(cam.fit, 0.86) : cam.fit;
    const fitted = fit > 0 ? dist + (need - dist) * fit : Math.max(dist, need * 0.72), want = fitted + (dist - fitted) * free;
    // upright: the camera keeps its place relative to the brain and rolls a quarter turn, so hears is at the top
    const turn = (v) => (tall ? [v[1], -v[0], v[2]] : v);
    const at = turn(tall ? mixv(cam.at, [-0.17, 0.12, 0], 0.7 * (1 - free)) : cam.at), d = turn(dir);
    let eye = [at[0] + (d[0] / dist) * want, at[1] + (d[1] / dist) * want, at[2] + (d[2] / dist) * want];
    if (!this.calm && this.kick > 0.02) { eye = [eye[0], eye[1] + this.kick * 0.012, eye[2] - this.kick * 0.05]; }
    const view = M.look(eye, at, [0, 1, 0]), proj = M.persp(fov, aspect, 0.1, 80);
    return { mvp: M.mul(proj, M.mul(view, model)), aspect, tall };
  }

  // ---------- a frame ----------
  _loop(ts) {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this._loop);
    const dt = Math.min(100, ts - (this.last || ts)); this.last = ts;
    this.frameMs += (dt - this.frameMs) * 0.05; this.frames++;
    // too slow: draw at a lower resolution; the data on screen stays the same
    // and when half the resolution is still slow (a software renderer, an old phone): half the wiring, a shorter glow
    if (this.frames > 90 && this.frameMs > 26 && !(this.thin && this.scale <= 0.5)) {
      const crawl = this.frameMs > 60;
      if (++this.slow > (crawl ? 20 : 60)) { if (this.scale > 0.5) { this.scale = Math.max(0.5, this.scale * (crawl ? 0.6 : 0.8)); this.resize(); } else this.thin = 1; this.slow = 0; this.frames = 0; }
    } else this.slow = 0;
    if (this.gl && !this.lost) this._draw(dt / 1000); else if (this.ctx2d) this._draw2d();
  }

  _draw(dt) {
    const gl = this.gl, g = this.geo, P = this.prog;
    const c = this.canvas; if (c.clientWidth && (Math.abs(c.clientWidth * Math.min(2, devicePixelRatio || 1) * this.scale - c.width) > 2)) this.resize();
    if (!this.fbo) this._targets();
    const now = this._now(), phase = this._phase(now), settle = ease(phase / this.settleWin), earIn = ease(phase / 0.1), ph = Math.min(phase, 1);
    // a section line (every 32 bars): for a beat only the bare wiring shows, then everything is back at once
    const dropping = !this.calm && this.curIndex > 0 && this.curIndex % 32 === 0 && phase * this._barDur() < clamp(this._barDur() / 4, 0.4, 0.6);
    if (this.dropping && !dropping && !this.calm) { this.kick = 1; this.split = 1; this.lastFlash = now; }
    this.dropping = dropping;
    const on = dropping ? 0 : 1, wild = Math.max(this.partyK, this.tunnelK);
    // the kick of the bar that sounds: a pulse and a split on its steps, never more than three a second
    if (this.cur && !this.calm) {
      const step = Math.floor(ph * 16);
      if (step !== this.playedKick) { this.playedKick = step; if (!dropping && this.cur.read.kicks.includes(step)) this._pulse(now, 1, 0.16 + 0.7 * wild); }
    }
    this.kick *= Math.exp(-dt * 7); this.split *= Math.exp(-dt * 3.2);
    if (this.analyser && this.spec) { this.analyser.getByteFrequencyData(this.spec); let s = 0; for (let i = 0; i < this.spec.length; i += 4) s += this.spec[i]; const lv = s / (this.spec.length / 4) / 255; this.level += ((0.8 + 0.5 * clamp(lv * 2.2, 0, 1)) - this.level) * 0.1; } else this.level += (1 - this.level) * 0.1;
    const party = this.scene === 'party' ? 1 : 0, calm = this.calm ? 1 : 0;
    this.partyK = (this.partyK || 0) + (party - (this.partyK || 0)) * Math.min(1, dt * 2.5);
    this.tunnelK += ((this.scene === 'tunnel' ? 1 : 0) - this.tunnelK) * Math.min(1, dt * 2.5);
    this.streamK = (this.streamK || 0) + ((this.scene === 'stream' ? 1 : 0) - (this.streamK || 0)) * Math.min(1, dt * 2.5);
    const { mvp, aspect } = this._mvp(now), pal = { g: [0.29, 0.945, 0.949], i: [0.788, 0.984, 0.984], r: [1.0, 0.165, 0.427], b: [0.165, 0.427, 1.0] };
    const F = this.fbo;
    gl.bindFramebuffer(gl.FRAMEBUFFER, F.scene.fb); gl.viewport(0, 0, F.scene.w, F.scene.h);
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    const bind = (unit, tex) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); };
    if (g) {
      // the wiring
      let p = P.line; gl.useProgram(p.p);
      gl.uniformMatrix4fv(p.u.uMVP, false, mvp); bind(0, this.tex.val); gl.uniform1i(p.u.uVal, 0);
      gl.uniform1f(p.u.uTime, now); gl.uniform1f(p.u.uPhase, ph); gl.uniform1f(p.u.uSettle, settle); gl.uniform1f(p.u.uLevel, this.level);
      gl.uniform1f(p.u.uKick, this.kick); const ppu = c.width < c.height * 0.95 ? c.width / 6.97 : Math.min(c.width / 12.4, c.height / 7.4), fine = clamp(Math.pow(ppu / 129, 0.6), 0.6, 1.1);
      gl.uniform1f(p.u.uLines, (this.scene === 'memory' ? 0.55 : 1 - 0.5 * this.tunnelK) * fine); gl.uniform1f(p.u.uParty, this.partyK); gl.uniform1f(p.u.uBare, 1 - on);
      gl.uniform3fv(p.u.uGlow, pal.g); gl.uniform3fv(p.u.uInk, pal.i); gl.uniform3fv(p.u.uWarm, WARM); gl.uniform3fv(p.u.uLane, this.laneFlat); gl.uniform3fv(p.u.uHead, this.headFlat);
      const lines = (buffer, verts, first = 0) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(p.a.aPos); gl.vertexAttribPointer(p.a.aPos, 3, gl.FLOAT, false, 36, 0);
        gl.enableVertexAttribArray(p.a.aL); gl.vertexAttribPointer(p.a.aL, 4, gl.FLOAT, false, 36, 12);
        gl.enableVertexAttribArray(p.a.aM); gl.vertexAttribPointer(p.a.aM, 2, gl.FLOAT, false, 36, 28);
        gl.drawArrays(gl.LINES, first, verts);
      };
      if (this.thin && g.halfVerts < g.synVerts) { lines(this.buf.line, g.halfVerts); lines(this.buf.line, g.nLineVerts - g.synVerts, g.synVerts); }
      else lines(this.buf.line, g.nLineVerts);
      if (this.beamVerts && this.meter) {
        // the residual meter falls during the settle to where this bar's settle ended, on a log scale under its tolerance
        const D = this.beamData, m = this.meter, tol = this.tolerance || 0.003, end = clamp(1 - Math.log10(Math.max(this.residual, 1e-9) / tol) / -5, 0, 1);
        const lvl = 1 - (1 - end) * settle, x = m.m0[0] + (m.m1[0] - m.m0[0]) * lvl, tick = m.m0[0] + (m.m1[0] - m.m0[0]) * 1.0;
        D.set([m.m0[0], m.m0[1], m.m0[2], 0, 5, 0, 0.9, 0, 1, x, m.m0[1], m.m0[2], 1, 5, 0, 0.9, 0, 1], m.at);
        D.set([tick, m.m0[1] - 0.05, m.m0[2], 0, 5, 0, 0.6, 0, 0, tick, m.m0[1] + 0.05, m.m0[2], 1, 5, 0, 0.6, 0, 0], m.at + 18);
        const Pl = this.PLAY, hx = Pl.x + Pl.w * ph, lit = this.exact ? on : 0, y0 = Pl.y + Pl.h * 0.19, y1 = Pl.y + Pl.h * 0.925;
        for (let q = 0; q < 3; q++) { const o = (q - 1) * 0.013, w = lit * (q === 1 ? 1.6 : 0.55); D.set([hx + o, y0, 0.04, 0, 5, 0, w, 0, 1, hx + o, y1, 0.04, 1, 5, 0, w, 0, 1], m.at + 36 + q * 18); }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.beam); gl.bufferSubData(gl.ARRAY_BUFFER, m.at * 4, D.subarray(m.at, m.at + 90));
        lines(this.buf.beam, this.beamVerts);
      }
      // the neurons
      p = P.point; gl.useProgram(p.p);
      gl.uniformMatrix4fv(p.u.uMVP, false, mvp);
      bind(0, this.tex.val); gl.uniform1i(p.u.uVal, 0); bind(1, this.tex.ear); gl.uniform1i(p.u.uEar, 1);
      bind(2, this.tex.motor); gl.uniform1i(p.u.uMotor, 2); bind(3, this.tex.ring); gl.uniform1i(p.u.uRing, 3);
      gl.uniform1f(p.u.uTime, now); gl.uniform1f(p.u.uPhase, ph); gl.uniform1f(p.u.uSettle, settle); gl.uniform1f(p.u.uEarIn, earIn);
      gl.uniform1f(p.u.uLevel, this.level); gl.uniform1f(p.u.uKick, this.kick); gl.uniform1f(p.u.uPx, F.scene.h * 0.0105);
      gl.uniform1f(p.u.uRingShift, 1 - ease(phase / 0.3)); gl.uniform1f(p.u.uParty, this.partyK + 1.5 * this.tunnelK); gl.uniform1f(p.u.uOn, on); gl.uniform1f(p.u.uEarShift, 1 - ease(phase / 0.24));
      gl.uniform3fv(p.u.uGlow, pal.g); gl.uniform3fv(p.u.uInk, pal.i); gl.uniform3fv(p.u.uLane, this.laneFlat); gl.uniform3fv(p.u.uHead, this.headFlat);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.point);
      gl.enableVertexAttribArray(p.a.aPos); gl.vertexAttribPointer(p.a.aPos, 3, gl.FLOAT, false, 44, 0);
      gl.enableVertexAttribArray(p.a.aInfo); gl.vertexAttribPointer(p.a.aInfo, 4, gl.FLOAT, false, 44, 12);
      gl.enableVertexAttribArray(p.a.aAux); gl.vertexAttribPointer(p.a.aAux, 4, gl.FLOAT, false, 44, 28);
      gl.drawArrays(gl.POINTS, 0, g.nPoints);
      // what the hands play, and its way back into the ear and into the memory
      const [mv, mb] = this.markVerts;
      if (mv + mb > 0 && on) {
        p = P.mark; gl.useProgram(p.p);
        gl.uniformMatrix4fv(p.u.uMVP, false, mvp); gl.uniform1f(p.u.uLevel, this.level); gl.uniform1f(p.u.uStream, this.streamK);
        gl.uniform1f(p.u.uShrink, (this.HEAR.w * this.blockOrigin(0).s) / this.PLAY.w);
        gl.uniform3fv(p.u.uLane, this.laneFlat); gl.uniform3fv(p.u.uInk, pal.i);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.mark);
        const at = (loc, n, off) => { if (loc >= 0) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 80, off); } };
        at(p.a.aA, 3, 0); at(p.a.aC, 3, 12); at(p.a.aB, 3, 24); at(p.a.aD, 3, 36); at(p.a.aK, 4, 48); at(p.a.aN, 4, 64);
        if (mv) { gl.uniform1f(p.u.uPhase, phase); gl.uniform1f(p.u.uOnly, 0); gl.drawArrays(gl.TRIANGLES, 0, mv); }
        if (mb && phase < 0.3) { gl.uniform1f(p.u.uPhase, phase + 1); gl.uniform1f(p.u.uOnly, 1); gl.drawArrays(gl.TRIANGLES, mv, mb); }
      }
    }
    gl.disable(gl.BLEND);
    const quad = (p) => { gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.quad); gl.enableVertexAttribArray(p.a.aP); gl.vertexAttribPointer(p.a.aP, 2, gl.FLOAT, false, 0, 0); gl.drawArrays(gl.TRIANGLES, 0, 6); };
    // the glow: the picture at a quarter size, blurred both ways twice
    let p = P.blur; gl.useProgram(p.p); gl.uniform1i(p.u.uTex, 0);
    const pass = (src, dst, dx, dy) => { gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb); gl.viewport(0, 0, dst.w, dst.h); bind(0, src.tex); gl.uniform2f(p.u.uDir, dx / dst.w, dy / dst.h); quad(p); };
    pass(F.scene, F.b1, 1.2, 0); pass(F.b1, F.b2, 0, 1.2); if (!this.thin) { pass(F.b2, F.b1, 2.4, 0); pass(F.b1, F.b2, 0, 2.4); }
    // the trails
    p = P.mix; gl.useProgram(p.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, F.keepB.fb); gl.viewport(0, 0, F.keepB.w, F.keepB.h);
    bind(0, F.scene.tex); gl.uniform1i(p.u.uScene, 0); bind(1, F.b2.tex); gl.uniform1i(p.u.uBloom, 1); bind(2, F.keepA.tex); gl.uniform1i(p.u.uPrev, 2);
    const trail = calm ? 0.5 : dropping ? 0.3 : 0.6 + 0.2 * this.streamK + 0.27 * this.partyK + 0.27 * this.tunnelK;
    gl.uniform1f(p.u.uDecay, Math.pow(trail, dt * 60));
    gl.uniform1f(p.u.uZoom, 1 - (calm ? 0 : 0.0025 * this.streamK + (0.012 + 0.016 * this.kick) * this.partyK + (0.013 + 0.022 * this.kick) * this.tunnelK));
    gl.uniform1f(p.u.uTurn, calm ? 0 : 0.004 * this.partyK + 0.0025 * Math.sin(now * 0.4) * this.tunnelK);
    gl.uniform1f(p.u.uBloomGain, 0.9 + (calm ? 0 : 0.35 * this.kick) - 0.25 * this.partyK - 0.2 * this.tunnelK);
    quad(p);
    // to the screen
    p = P.out; gl.useProgram(p.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, c.width, c.height);
    bind(0, F.keepB.tex); gl.uniform1i(p.u.uTex, 0);
    gl.uniform1f(p.u.uFold, this.partyK * (calm ? 0.0 : 1.0)); gl.uniform1f(p.u.uMirror, this.tunnelK); gl.uniform1f(p.u.uSplit, calm ? 0 : 0.016 * this.split); gl.uniform1f(p.u.uGhost, calm ? 0 : clamp(this.split * 0.9, 0, 0.7));
    gl.uniform3fv(p.u.uRed, pal.r); gl.uniform3fv(p.u.uBlue, pal.b);
    gl.uniform1f(p.u.uAspect, aspect); gl.uniform1f(p.u.uScan, 1); gl.uniform1f(p.u.uTime, now); gl.uniform1f(p.u.uGain, 1.0);
    quad(p);
    const t = F.keepA; F.keepA = F.keepB; F.keepB = t;
    if (this.labels && g && wild < 0.5 && on) this._drawLabels(mvp, 1 - wild * 2);
  }

  _drawLabels(mvp, alpha) {
    const gl = this.gl, p = this.prog.label, D = this.labelData, W = this.canvas.width, H = this.canvas.height;
    let n = 0; const tall = W < H * 0.95, unit = Math.max(0.55, Math.min(1.25, tall ? W / 880 : Math.min(W, H * 1.7) / 1500));
    const placed = [];                                // the labels drawn so far: a later one that would cover one of them is left out
    const batch = (big, lane) => {
      n = 0;
      for (const l of this.labelList) {
        if (!!l.big !== big || (!big && l.lane !== lane) || (l.tall && !tall) || (l.wide && tall)) continue; const a = this.atlas[l.t]; if (!a) continue;
        const x = l.p[0], y = tall && l.ty !== undefined ? l.ty : l.p[1], z = l.p[2];
        const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15]; if (cw <= 0.1) continue;
        const cx = (mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]) / cw, cy = (mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]) / cw;
        const k = unit * (big ? 0.62 : 0.52), w = (a.w * k) / W * 2, h = (a.h * k) / H * 2;
        let x0 = l.side === -1 ? cx - w : l.side === 1 ? cx : cx - w / 2, y0 = cy - h / 2;
        if (tall && l.up) { x0 = cx - w / 2; y0 = cy + h * (0.1 + (l.lift || 0)); }       // upright: a row's name stands over its row
        if (x0 < -0.985 || x0 + w > 0.985 || y0 < -0.985 || y0 + h > 0.985) continue;      // a label is whole or not there
        const mx = w * 0.04, my = h * 0.12;
        if (placed.some((q) => x0 + mx < q[2] && x0 + w - mx > q[0] && y0 + my < q[3] && y0 + h - my > q[1])) continue;
        if (n + 24 > D.length) break;
        placed.push([x0, y0, x0 + w, y0 + h]);
        D.set([x0, y0, a.u0, a.v1, x0 + w, y0, a.u1, a.v1, x0, y0 + h, a.u0, a.v0, x0, y0 + h, a.u0, a.v0, x0 + w, y0, a.u1, a.v1, x0 + w, y0 + h, a.u1, a.v0], n); n += 24;
      }
      if (!n) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.label); gl.bufferSubData(gl.ARRAY_BUFFER, 0, D.subarray(0, n));
      gl.enableVertexAttribArray(p.a.aQ); gl.vertexAttribPointer(p.a.aQ, 4, gl.FLOAT, false, 0, 0);
      gl.uniform1f(p.u.uA, (big ? 0.9 : lane ? 0.95 : 0.6) * alpha);
      gl.uniform3fv(p.u.uCol, big ? [0.788, 0.984, 0.984] : LANE_TINT[lane]);
      gl.drawArrays(gl.TRIANGLES, 0, n / 4);
    };
    gl.useProgram(p.p); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex.label); gl.uniform1i(p.u.uTex, 0);
    batch(true, 0); for (let lane = 1; lane < 5; lane++) batch(false, lane); batch(false, 0);
    gl.disable(gl.BLEND);
  }

  // without WebGL: the regions as discs at their recorded mean, the heads' reads as lines, the labels
  _draw2d() {
    const g = this.ctx2d, c = this.canvas, geo = this.geo; if (!g) return;
    if (c.clientWidth && c.width !== c.clientWidth) { c.width = c.clientWidth; c.height = c.clientHeight || c.height; }
    g.fillStyle = '#000'; g.fillRect(0, 0, c.width, c.height);
    if (!geo) return;
    const sx = c.width / 9.2, sy = c.height / 6.8, s = Math.min(sx, sy), X = (x) => c.width / 2 + (x + 0.3) * s, Y = (y) => c.height / 2 - (y - 0.5) * s;
    const means = {}; const view = this.cur && this.cur.view;
    if (view && view.regions) for (const r of view.regions) means[r.name] = +r.mean || 0;
    g.strokeStyle = 'rgba(74,241,242,.25)'; g.lineWidth = 1;
    for (const [name, p] of Object.entries(geo.centre)) {
      const m = clamp(Math.sqrt(means[name] || 0.02) * 1.6, 0.1, 1);
      g.fillStyle = `rgba(74,241,242,${m})`; g.beginPath(); g.arc(X(p[0]), Y(p[1]), 6 + 14 * m, 0, 7); g.fill();
    }
    g.fillStyle = '#c9fbfb'; g.font = `12px ${MONO}`; g.textAlign = 'center';
    for (const l of this.labelList || []) if (this.labels && !l.tall) g.fillText(l.t, X(l.p[0]), Y(l.p[1]));
  }
}

export default BrainStage;
