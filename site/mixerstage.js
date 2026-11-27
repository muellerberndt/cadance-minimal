// Cadance radio, the mixer stage: the Auto DJ at work, without one letter or digit.
//
// Three decks, one lane each. A lane is the deck's track as a strip that streams from right to left through a fixed
// play line. Right of the line is what is coming, drawn in depth (the far bars are small and close together, so a
// whole section ahead is in view); left of it is what went out. The strip's shape is the track: three bands stacked
// from the lane's axis outward, low, mid, high. Its light is the mixer: left of the play line every band is as
// bright as the deck let it through at that moment (EQ, the two filters, the fader, a cut), so a killed bass leaves
// a hollow core and a closing lowpass eats the rim; right of the line the bands are faint, and fainter where the
// mixer would take them out now.
//
// Beside each lane stands the deck's channel strip as glowing forms that move as the planner moves them: a level
// meter, the fader, three knobs for the EQ (in the bands' tints), a ring whose lit arc is the band the two filters
// let pass, and a ring for the effect that swells with its amount.
//
// The Auto DJ: the primary deck carries a warm mark on the play line and warm rails. A track that will come in
// shows on the idle lane as a ring that closes over the section before it, and its strip's first bar travels toward
// the play line. The bar where the bass changes decks travels as a pair of gates on the two lanes with a link
// between them; when it reaches the play line a beam joins the two lanes, the mark changes lanes and the picture
// pulses once. The bar where a track leaves is the end of its strip. An effect bends, repeats, chops or blurs the
// played part of its deck's strip; a cut leaves a dark gap in it.
//
// Section lines (every 32 bars) cross all lanes at once. The rays behind the lanes are the master's spectrum.
//
// Recorded: the knob positions, roles, transitions and bars the reader gives, the deck meters, the master
// spectrum, the track's waveform (from its audio, three bands by two one-pole filters, or from its note file).
// Derived, and said so here: what a knob position takes from a band (the mixer's knob laws and the filters'
// slopes at three band centres), the effect shapes on the strip (one shape per family of effects, scaled by the
// recorded amount), the wave drawn from a note file (envelopes per hit and note, not the sound), the beat pulse
// (a rise of the primary's low band at the play line), the meter of a deck without an analyser (its wave times
// its level), the height of an audio wave's bands (each against its own loudest slot in the track), and every tint.
//
// One ES module, no dependency, no request. WebGL 1 or 2; without it `ok` is false and nothing is drawn.
//
//   const mixer = new MixerStage(canvas, { decks: 3, calm: false });
//   mixer.ok                                    false: no WebGL, the page falls back
//   mixer.setClock({ ctx, bpm, barZeroTime })   global bar k starts at barZeroTime + k * 240 / bpm; without ctx: performance.now()
//   mixer.setAudio(analyser)                    the master analyser (optional)
//   mixer.setMeter(deck, analyser)              a deck's analyser (optional)
//   mixer.setTrack(deck, { id, bars, startBar, loopFrom } | null)     another id clears the deck's wave
//   mixer.setWave(deck, audioBuffer | { bars: [...] } | { peaks, perBar } | null)     the whole track's wave: its audio, its note file, or peaks
//   mixer.addWave(deck, { fromBar, audio } | { fromBar, peaks, perBar })              a piece of it, as segments are decoded
//   mixer.setReader(() => ({ bar, decks: [{ playing, role, knobs: { fader, lo, mid, hi, hp, lp, fx, cut }, fxType }],
//                            dj: { on, primary, section, nextStart, transitions: [{ in, out, start, swap, end }] } }))
//   mixer.pulse(kind, deck)                     'cue', 'swap', 'leave', 'gate' (optional: the stage also finds them in the reader's data)
//   mixer.setCalm(on); mixer.resize(); mixer.start(); mixer.stop(); mixer.stats()

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const CYAN = [0.29, 0.945, 0.949], INK = [0.788, 0.984, 0.984], RED = [1.0, 0.165, 0.427], BLUE = [0.165, 0.427, 1.0];
// the bands' tints (low, mid, high), the primary's warm white
const LOW = [0.22, 0.62, 1.0], MIDT = [0.38, 1.0, 0.66], HIGH = [0.78, 0.66, 1.0], WARM = [1.0, 0.9, 0.74];
const COLS = 1024;                 // columns of a lane's strip, refilled every frame
const HS = 32, HN = 1024;          // what went out is remembered per 1/32 bar, for 32 bars
const NS = 64;                     // a wave's slots per bar (from notes and from audio alike)
// the mixer's knob laws (position 0..1 to value), as the radio's decks use them
const LAW = {
  eq: (p) => (p <= 0.015 ? -40 : p < 0.5 ? (-26 * (0.5 - p)) / 0.485 : (6 * (p - 0.5)) / 0.5),      // dB
  hp: (p) => (p <= 0.005 ? 0 : 20 * Math.pow(125, p)),                                             // Hz, 0: open
  lp: (p) => (p >= 0.995 ? 0 : 180 * Math.pow(20000 / 180, p)),                                    // Hz, 0: open
  fader: (p) => Math.pow(Math.max(0, p) / 0.8, 2.5),
};
const BAND_HZ = [70, 900, 7000];   // where the three bands sit, for what a filter takes from each
const NEUTRAL = { fader: 0, lo: 0.5, mid: 0.5, hi: 0.5, hp: 0, lp: 1, fx: 0, cut: 1 };
// one drawn shape per family of effects: 1 a halo (reverb, vanish, sink), 2 repeats (echo), 3 a sway (flanger,
// phaser, sweep), 4 blocks (crush), 5 chops (gater, roll)
const FX_CLASS = { reverb: 1, vanish: 1, sink: 1, echo: 2, flanger: 3, phaser: 3, sweep: 3, crush: 4, gater: 5, roll: 5 };

// what the knobs let through: [low, mid, high, level], 1 = untouched (a boost goes to 1.27 at most)
function gainsOf(k, out) {
  const eq = [k.lo, k.mid, k.hi], hp = LAW.hp(k.hp), lp = LAW.lp(k.lp);
  for (let i = 0; i < 3; i++) {
    let g = Math.pow(10, LAW.eq(eq[i]) / 20);
    if (hp > 0) { const r = BAND_HZ[i] / hp, r2 = r * r; g *= r2 / Math.sqrt(1 + r2 * r2); }
    if (lp > 0) { const r = BAND_HZ[i] / lp, r2 = r * r; g *= 1 / Math.sqrt(1 + r2 * r2); }
    out[i] = Math.min(1.27, g);
  }
  out[3] = Math.min(1.27, LAW.fader(k.fader) * clamp(k.cut === undefined ? 1 : k.cut, 0, 1));
  return out;
}

// a note file's bars as a wave: per bar 64 slots of (low, mid, high). The drums are transients (kick low, snare mid
// and high, hats high), the three lines are beds (bass low, lead mid, air high) as long as their notes. Envelopes
// drawn from the notes, not the sound: where the audio's wave is there, it is shown instead.
function notesWave(bars) {
  const n = bars.length, data = new Uint8Array(n * NS * 3), hit = new Float32Array(NS * 3), bed = new Float32Array(NS * 3);
  const KICK = [1, 0.84, 0.64, 0.46, 0.31, 0.2, 0.12, 0.06], SNARE = [1, 0.74, 0.52, 0.35, 0.22, 0.12, 0.06], NOISE = [0.9, 0.72, 0.5, 0.32, 0.18, 0.09];
  const CLOSED = [1, 0.5, 0.2], OPEN = [1, 0.9, 0.76, 0.62, 0.48, 0.36, 0.25, 0.15, 0.08];
  const burst = (band, slot, level, shape) => { for (let k = 0; k < shape.length && slot + k < NS; k++) { const o = (slot + k) * 3 + band; hit[o] = Math.max(hit[o], level * shape[k]); } };
  for (let b = 0; b < n; b++) {
    hit.fill(0); bed.fill(0);
    const bar = bars[b] || {}, tok = bar.tokens || [1, 1, 1, 1], pats = bar.patterns || [];
    let hits = null; const lines = [];
    for (const p of pats) { if (p && p.hits) hits = p.hits; else if (p && p.notes) lines.push(p); }
    if (hits) for (let t = 0; t < 16; t++) {
      const kick = +((hits.kick || [])[t]) || 0, snare = Math.max(+((hits.snare || [])[t]) || 0, +((hits.clap || [])[t]) || 0);
      const ch = +((hits.ch || [])[t]) || 0, oh = +((hits.oh || [])[t]) || 0;
      if (kick) { burst(0, t * 4, 0.6 + 0.4 * kick, KICK); burst(1, t * 4, 0.34 * kick, CLOSED); burst(2, t * 4, 0.2 * kick, [1, 0.3]); }
      if (snare) { burst(1, t * 4, 0.5 + 0.45 * snare, SNARE); burst(2, t * 4, 0.3 + 0.4 * snare, NOISE); }
      if (ch) burst(2, t * 4, 0.3 + 0.45 * ch, CLOSED);
      if (oh) burst(2, t * 4, 0.4 + 0.5 * oh, OPEN);
    }
    for (let l = 0; l < 3 && l < lines.length; l++) {
      if (!(tok[l + 1] > 0)) continue;                         // a voice that does not sound
      const p = lines[l];
      for (let t = 0; t < 16; t++) {
        const note = p.notes[t]; if (note === null || note === undefined) continue;
        const vel = p.vel ? +p.vel[t] || 0.6 : 0.6, len = clamp((p.len && p.len[t]) || 1, 1, 16) * 4, acc = p.accent && p.accent[t] ? 1.15 : 1;
        for (let k = 0; k < len + 2 && t * 4 + k < NS; k++) {
          // an attack, a body that sinks a little, a short release; a slow ripple by the note's pitch so a held note is not a flat block
          const env = k >= len ? 0.45 / (k - len + 1.4) : (k === 0 ? 1 : 0.86 - 0.3 * (k / len)) * (0.93 + 0.07 * Math.sin(k * (0.9 + (note % 12) * 0.13)));
          const o = (t * 4 + k) * 3, v = env * acc;
          if (l === 0) { bed[o] = Math.max(bed[o], (0.3 + 0.36 * vel) * v); bed[o + 1] = Math.max(bed[o + 1], 0.2 * vel * v); }
          else if (l === 1) { bed[o + 1] = Math.max(bed[o + 1], (0.26 + 0.42 * vel) * v); bed[o + 2] = Math.max(bed[o + 2], 0.2 * vel * v); }
          else { bed[o + 2] = Math.max(bed[o + 2], (0.26 + 0.4 * vel) * v); bed[o + 1] = Math.max(bed[o + 1], 0.24 * vel * v); }
        }
      }
    }
    // a transient stands on its band's bed
    for (let i = 0; i < NS * 3; i++) data[b * NS * 3 + i] = Math.round(clamp(bed[i] + hit[i] * (1 - 0.45 * bed[i]), 0, 1) * 255);
  }
  return { n: n * NS, data };
}

// ---------- shaders (GLSL ES 1.00) ----------
const HP = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
`;
const VS_FULL = `attribute vec2 aP; varying vec2 vUV; void main() { vUV = aP * 0.5 + 0.5; gl_Position = vec4(aP, 0.0, 1.0); }`;
// a lane: the strip, its rails, the bar lines, the play line, the gates that travel on it
const FS_STRIP = `${HP}
uniform sampler2D uWave;
uniform vec2 uRes;
uniform vec4 uGeo;      // the strip's left edge, the play line, the depth's pull, the height all lanes run toward
uniform vec4 uLane;     // the lane's height on the canvas, its half width, the wave's two rows (shape, light)
uniform vec4 uView;     // bars in view up to the middle of the depth, bars behind the play line, the bar position (mod 32), time
uniform vec4 uShape;    // lanes run together (1) or stay level (0), how much a lane narrows with depth, unused, unused
uniform vec4 uState;    // playing, primary, level, effect amount
uniform vec4 uFx;       // effect family, cut (1 open, 0 shut), kick, role
uniform vec3 uGain;     // what the mixer lets through now
uniform vec4 uMark[4];  // a gate: bars from the play line, kind, strength
uniform vec3 uLow, uMid, uHigh, uInk, uCyan, uWarm, uRed;
float sq(float x) { return x * x; }
void main() {
  vec2 p = gl_FragCoord.xy / uRes;
  float x0 = uGeo.x, xp = uGeo.y, q = uGeo.z;
  if (p.x < x0) discard;
  float ahead = step(xp, p.x);
  float s = max(0.0, (p.x - xp) / (1.0 - xp)), k = 1.0 - q * s;
  float yc = mix(uLane.x, uGeo.w + (uLane.x - uGeo.w) * k, uShape.x), h = uLane.y * mix(1.0, k, uShape.y);
  float v = (p.y - yc) / h, d = abs(v);
  if (d > 1.4) discard;
  float past = uView.y, B0 = uView.x, t = uView.w;
  float b = ahead > 0.5 ? B0 * s / k : (p.x - xp) / (xp - x0) * past;
  float perPx = ahead > 0.5 ? B0 / (k * k) / ((1.0 - xp) * uRes.x) : past / ((xp - x0) * uRes.x);     // bars per pixel here
  float u = (p.x - x0) / (1.0 - x0);
  float amt = uState.w * (1.0 - ahead), cls = uFx.x, barPos = uView.z + b, chop = 1.0;
  if (amt > 0.01) {                                // the effect's hand on what was played
    if (cls > 2.5 && cls < 3.5) v += amt * 0.3 * sin(barPos * 12.566 + t * 2.5);
    else if (cls > 3.5 && cls < 4.5) { float n = mix(420.0, 60.0, amt), m = mix(40.0, 6.0, amt); u = (floor(u * n) + 0.5) / n; v = (floor(v * m) + 0.5) / m; }
    else if (cls > 4.5) chop = mix(1.0, step(fract(barPos * 8.0), 0.5), min(1.0, amt * 1.8));
    d = abs(v);
  }
  vec4 pk = texture2D(uWave, vec2(u, uLane.z)), gn = texture2D(uWave, vec2(u, uLane.w));
  if (amt > 0.01 && cls > 1.5 && cls < 2.5) {      // an echo: three sixteenths later it comes again, and again
    float e = 0.1875 / past * (xp - x0) / (1.0 - x0), a = min(1.0, amt * 1.8);
    pk = max(pk, texture2D(uWave, vec2(u - e, uLane.z)) * 0.7 * a);
    pk = max(pk, texture2D(uWave, vec2(u - 2.0 * e, uLane.z)) * 0.5 * a);
    pk = max(pk, texture2D(uWave, vec2(u - 3.0 * e, uLane.z)) * 0.32 * a);
  }
  float b1 = pk.r * 0.42, b2 = b1 + pk.g * 0.31, b3 = b2 + pk.b * 0.22;
  float px = 1.2 / (h * uRes.y);
  float iL = 1.0 - smoothstep(b1 - px, b1 + px, d), iM = 1.0 - smoothstep(b2 - px, b2 + px, d), iH = 1.0 - smoothstep(b3 - px, b3 + px, d);
  vec3 g = mix(gn.rgb * 1.275, uGain, ahead);
  float level = mix(gn.a * 1.275, 1.0, ahead), far = 0.4 + 0.6 * k;
  vec3 lit = ahead > 0.5 ? (0.11 + 0.36 * g) * far : g * level * chop * 0.74;
  float sheen = 0.6 + 0.4 * d / max(b3, 0.05);
  vec3 col = (uLow * iL * lit.r + uMid * (iM - iL) * lit.g + uHigh * (iH - iM) * lit.b) * sheen * pk.a * 1.15;
  if (amt > 0.01 && cls < 1.5) col = col * (1.0 - 0.35 * amt) + mix(uHigh, uInk, 0.4) * exp(-abs(d - b3) * 5.0) * amt * 0.42 * level * pk.a;       // a halo: the strip dissolves outward
  float outline = exp(-sq((d - b3) / px * 0.9)) * 0.5 + (exp(-sq((d - b1) / px)) + exp(-sq((d - b2) / px))) * 0.15;
  col += mix(uCyan, uInk, 0.4) * outline * pk.a * (ahead > 0.5 ? 0.5 * far : (0.25 + 0.75 * level) * chop);
  col *= 1.0 + (1.0 - ahead) * exp(min(b, 0.0) * 7.0) * 1.3 * uState.z;           // just played: as loud as the deck's meter
  float inLane = step(d, 1.14), rail = exp(-sq((d - 1.14) / px * 0.8));
  col += mix(uCyan, uWarm, uState.y) * rail * (0.09 + 0.5 * uState.y + 0.1 * uState.x) * (ahead > 0.5 ? far : 1.0);
  col += uCyan * exp(-sq(d / px)) * 0.06;
  float fb = fract(barPos), db = min(fb, 1.0 - fb);
  float barLine = (1.0 - smoothstep(0.5, 1.5, db / perPx)) * clamp((1.0 / perPx - 3.0) / 9.0, 0.0, 1.0);
  float f4 = fract(barPos / 4.0), d4 = min(f4, 1.0 - f4) * 4.0;
  float four = (1.0 - smoothstep(0.6, 1.8, d4 / perPx)) * clamp((4.0 / perPx - 3.0) / 9.0, 0.0, 1.0);
  col += uCyan * inLane * (barLine * 0.09 + four * 0.15) * (0.55 + 0.45 * uState.x);
  float dx = abs(p.x - xp) * uRes.x;
  float head = exp(-sq(dx / (1.2 + 1.8 * uState.y))) * step(d, 1.3);
  col += mix(uInk, uWarm, uState.y) * head * (0.22 + 0.8 * uState.x * (0.4 + 0.9 * uState.z) + 0.6 * uState.y);
  col += uRed * head * (1.0 - uFx.y) * uState.x * 1.2;                              // a cut: the deck is shut
  for (int i = 0; i < 4; i++) {
    vec4 m = uMark[i];
    if (m.y < 0.5) continue;
    float sm = m.x / (B0 + q * m.x);
    float xm = m.x >= 0.0 ? xp + sm * (1.0 - xp) : xp + m.x / past * (xp - x0);
    float km = m.x >= 0.0 ? 1.0 - q * sm : 1.0;
    float dm = abs(p.x - xm) * uRes.x, w = 1.2 + 1.8 * km;
    float bar = exp(-sq(dm / w)) * step(d, 1.3);
    vec3 mc = m.y < 2.5 ? uLow : m.y < 3.5 ? uInk : uCyan;
    float pat = m.y < 1.5 ? 1.0 : m.y < 2.5 ? 0.35 + 0.65 * step(0.5, fract(v * 3.0 + 0.25)) : m.y < 3.5 ? 0.75 : 0.8;
    col += mc * bar * pat * m.z * (0.45 + 0.55 * km) * 1.5;
    if (m.y < 2.5) {                               // the side of the gate on which this deck holds the bass
      float side = m.y < 1.5 ? step(xm, p.x) : step(p.x, xm);
      col += uLow * side * exp(-dm / (40.0 * km)) * step(d, 0.46) * 0.3 * m.z;
    }
  }
  col *= smoothstep(x0, x0 + 0.015, p.x);
  gl_FragColor = vec4(col, 1.0);
}`;
// behind the lanes: the section lines across all of them, and the master's spectrum as rays from the point the tracks come from
const FS_BACK = `${HP}
uniform sampler2D uSpec;
uniform vec2 uRes;
uniform vec4 uGeo, uView;
uniform vec4 uFan;      // the lanes' upper and lower edge at the play line, lanes run together, level
uniform vec3 uCyan, uInk;
void main() {
  vec2 p = gl_FragCoord.xy / uRes;
  float x0 = uGeo.x, xp = uGeo.y, q = uGeo.z, y0 = uGeo.w;
  float ahead = step(xp, p.x), s = max(0.0, (p.x - xp) / (1.0 - xp)), k = 1.0 - q * s;
  float top = mix(uFan.x, y0 + (uFan.x - y0) * k, uFan.z), bot = mix(uFan.y, y0 + (uFan.y - y0) * k, uFan.z);
  float inFan = step(bot, p.y) * step(p.y, top) * step(x0, p.x);
  float b = ahead > 0.5 ? uView.x * s / k : (p.x - xp) / (xp - x0) * uView.y;
  float perPx = ahead > 0.5 ? uView.x / (k * k) / ((1.0 - xp) * uRes.x) : uView.y / ((xp - x0) * uRes.x);
  float f = fract((uView.z + b) / 32.0), ds = min(f, 1.0 - f) * 32.0 / perPx;       // pixels to the nearest section line
  vec3 col = mix(uCyan, uInk, 0.6) * inFan * ((1.0 - smoothstep(0.8, 2.6, ds)) * 0.8 + exp(-ds / 16.0) * 0.14) * (0.35 + 0.65 * k);
  float f16 = fract((uView.z + b) / 16.0), d16 = min(f16, 1.0 - f16) * 16.0 / perPx;
  col += uCyan * inFan * (1.0 - smoothstep(0.6, 1.8, d16)) * 0.14 * k;
  vec2 dv = (p - vec2(xp + (1.0 - xp) / q, y0)) * vec2(uRes.x / uRes.y, 1.0);
  float ang = atan(dv.y, -dv.x), a01 = clamp(abs(ang) / 1.25, 0.0, 1.0);
  float sp = texture2D(uSpec, vec2(a01, 0.5)).r;
  col += mix(uCyan, uInk, sp) * sp * sp * (0.45 + 0.55 * sin(ang * 150.0)) * exp(-length(dv) * 1.5) * 0.55 * uFan.w;
  gl_FragColor = vec4(col, 1.0);
}`;
// the controls and the Auto DJ's signs: each a small quad with a form drawn into it
const VS_FORM = `
attribute vec2 aP; uniform vec2 uC, uS, uRes; varying vec2 vQ; varying vec2 vS;
void main() { vQ = aP * uS; vS = uS; gl_Position = vec4((uC + aP * uS) / uRes * 2.0 - 1.0, 0.0, 1.0); }`;
const FS_FORM = `${HP}
varying vec2 vQ; varying vec2 vS;
uniform float uKind, uTime; uniform vec4 uVal; uniform vec3 uCol, uCol2, uRed;
float sq(float x) { return x * x; }
float along(vec2 q) { return (atan(q.x, q.y) + 2.35619) / 4.71239; }       // 0..1 along a knob's sweep: lower left, over the top, lower right
vec2 onArc(float t, float R) { float a = -2.35619 + t * 4.71239; return R * vec2(sin(a), cos(a)); }
void main() {
  vec2 q = vQ; float R = min(vS.x, vS.y) * 0.84, r = length(q), w = max(1.1, R * 0.075);
  vec3 col = vec3(0.0);
  if (uKind < 0.5) {                               // a knob: value, how fast it is being turned, killed
    float t = along(q), on = step(0.0, t) * step(t, 1.0), ring = exp(-sq((r - R) / w));
    col += uCol * ring * on * (0.13 + 0.95 * step(t, uVal.x));
    vec2 e = q - onArc(uVal.x, R);
    col += mix(uCol, uCol2, 0.7) * exp(-dot(e, e) / sq(R * 0.17)) * 1.4;
    col += uCol * exp(-sq(r / (R * 0.36))) * (0.08 + 0.5 * uVal.x + 1.3 * uVal.y);
    col += uRed * exp(-sq(r / (R * 0.22))) * uVal.z * 0.9;
    col += uCol2 * exp(-sq(q.x / 1.1)) * step(R * 1.08, q.y) * 0.35;                // the mark of the flat position
  } else if (uKind < 1.5) {                        // the fader: value, speed, the mark of full level, cut
    float H = vS.y * 0.9, yv = -H + 2.0 * H * uVal.x, yu = -H + 2.0 * H * uVal.z;
    col += uCol * exp(-sq(q.x / 1.1)) * step(abs(q.y), H) * (0.16 + 0.8 * step(q.y, yv));
    col += mix(uCol2, uRed, 1.0 - uVal.w) * exp(-sq((q.y - yv) / 2.2)) * step(abs(q.x), vS.x * 0.85) * (1.0 + 1.6 * uVal.y);
    col += uCol * exp(-sq(q.y - yu)) * step(vS.x * 0.5, abs(q.x)) * 0.35;
  } else if (uKind < 2.5) {                        // the meter: level, its held peak, cut
    float H = vS.y * 0.9, y01 = (q.y + H) / (2.0 * H), n = 16.0, inY = step(0.0, y01) * step(y01, 1.0) * step(abs(q.x), vS.x * 0.7);
    float lit = step((floor(y01 * n) + 0.5) / n, uVal.x);
    col += mix(mix(uCol, uCol2, smoothstep(0.5, 1.0, y01)), uRed, 1.0 - uVal.z) * inY * step(0.2, fract(y01 * n)) * (0.07 + 0.95 * lit);
    col += uCol2 * inY * exp(-sq((y01 - uVal.y) * n / 0.45)) * 0.7 * step(0.02, uVal.y);
  } else if (uKind < 3.5) {                        // the two filters as one ring: the lit arc is what passes (from, to, speed)
    float t = along(q), on = step(0.0, t) * step(t, 1.0), ring = exp(-sq((r - R) / (w * 1.3)));
    col += uCol * ring * on * (0.12 + 0.95 * step(uVal.x, t) * step(t, uVal.y));
    vec2 e1 = q - onArc(uVal.x, R), e2 = q - onArc(uVal.y, R);
    col += uCol2 * (exp(-dot(e1, e1) / sq(R * 0.15)) + exp(-dot(e2, e2) / sq(R * 0.15))) * (0.8 + 1.6 * uVal.z);
    float xin = q.x / (R * 1.24) + 0.5;
    col += uCol * step(abs(q.y), max(1.0, R * 0.06)) * step(uVal.x, xin) * step(xin, uVal.y) * step(0.0, xin) * step(xin, 1.0) * 0.55;
  } else if (uKind < 4.5) {                        // the effect: amount, family, a turning phase, speed
    float ang = atan(q.x, q.y), ring = exp(-sq((r - R) / w));
    float petal = 0.5 + 0.5 * cos(ang * (2.0 + uVal.y) + uVal.z);
    col += uCol * ring * (0.13 + 0.85 * step(ang / 6.28318 + 0.5, uVal.x));
    col += uCol * exp(-sq((r - R * (0.28 + 0.56 * uVal.x * petal)) / (w * 1.2))) * (0.16 + 1.4 * uVal.x);
    col += uCol2 * exp(-sq(r / (R * 0.3))) * uVal.x * (0.8 + 0.9 * uVal.w);
  } else if (uKind < 5.5) {                        // the next track's entry: a ring that closes over the section before it
    float ang = atan(q.x, q.y), t01 = ang / 6.28318 + 0.5;
    col += uCol * exp(-sq((r - R) / w)) * (0.16 + 0.95 * step(t01, uVal.x));
    float a6 = mod(ang + uVal.x * 2.1, 1.0472) - 0.5236;
    col += uCol2 * exp(-sq((r - R * 0.8 * (1.0 - uVal.x) * 0.866 / cos(a6)) / w)) * (0.25 + 0.7 * uVal.x);
    float tk = fract(t01 * 8.0);
    col += uCol * step(R * 1.06, r) * step(r, R * 1.17) * (1.0 - smoothstep(0.02, 0.05, min(tk, 1.0 - tk))) * 0.5;
    col += uCol2 * exp(-sq(r / (R * 0.12))) * sq(uVal.x) * 1.2;
  } else if (uKind < 6.5) {                        // the link between the two gates of a bass swap: arrows from the deck that gives to the deck that takes
    float c = fract(q.y / 18.0 * uVal.x - uTime * 1.4 + abs(q.x) / 18.0);
    col += uCol * (exp(-sq(q.x / 1.2)) * 0.45 + exp(-sq((c - 0.5) * 5.0)) * exp(-sq(q.x / (vS.x * 0.6))) * 0.8) * uVal.y;
  } else if (uKind < 7.5) {                        // an event's ring: progress, strength
    col += uCol * exp(-sq((r - R * uVal.x) / (w * 2.0))) * sq(1.0 - uVal.x) * uVal.y * 2.0;
  } else if (uKind < 8.5) {                        // the beam of a bass swap: strength, shake
    float j = (sin(q.y * 0.07 + uTime * 31.0) * 5.0 + sin(q.y * 0.19 - uTime * 47.0) * 3.0) * uVal.y;
    col += mix(uCol, uCol2, uVal.x) * exp(-sq((q.x + j) / (2.0 + 8.0 * uVal.x))) * uVal.x * 2.4;
    col += uCol * exp(-abs(q.x) / (vS.x * 0.4)) * uVal.x * 0.3;
  } else {                                         // the primary's mark on the play line: strength, level
    float dia = abs(q.x) / (vS.x * 0.92) + abs(q.y) / (vS.y * 0.96);
    col += uCol * exp(-sq((dia - 1.0) * min(vS.x, vS.y) * 0.4)) * uVal.x * (0.7 + 0.8 * uVal.y);
    col += uCol * exp(-sq((dia - 0.55) * min(vS.x, vS.y) * 0.5)) * uVal.x * 0.3;
  }
  gl_FragColor = vec4(col, 1.0);
}`;
const FS_BLUR = `
precision mediump float; varying vec2 vUV; uniform sampler2D uTex; uniform vec2 uDir;
void main() {
  vec3 c = texture2D(uTex, vUV).rgb * 0.227;
  c += (texture2D(uTex, vUV + uDir * 1.385).rgb + texture2D(uTex, vUV - uDir * 1.385).rgb) * 0.316;
  c += (texture2D(uTex, vUV + uDir * 3.231).rgb + texture2D(uTex, vUV - uDir * 3.231).rgb) * 0.070;
  gl_FragColor = vec4(c, 1.0);
}`;
// the frame kept between frames: the new picture over a fading copy of the last one that drifts out of the depth
const FS_MIX = `
precision mediump float; varying vec2 vUV; uniform sampler2D uScene, uBloom, uPrev;
uniform float uDecay, uZoom, uBloomGain, uFrom; uniform vec2 uCentre;
void main() {
  // the trails belong to the lanes: the controls beside them stay sharp
  vec3 prev = texture2D(uPrev, (vUV - uCentre) * uZoom + uCentre).rgb * uDecay * smoothstep(uFrom - 0.01, uFrom + 0.03, vUV.x);
  vec3 now = texture2D(uScene, vUV).rgb + texture2D(uBloom, vUV).rgb * uBloomGain;
  gl_FragColor = vec4(max(now, prev), 1.0);
}`;
const FS_OUT = `
precision mediump float; varying vec2 vUV; uniform sampler2D uTex;
uniform float uSplit, uGhost, uPunch; uniform vec2 uCentre; uniform vec3 uRed, uBlue;
void main() {
  vec2 uv = (vUV - uCentre) * (1.0 - uPunch) + uCentre;
  vec3 c = texture2D(uTex, uv).rgb;
  if (uGhost > 0.004) {                            // the split: the picture's red and blue ghosts part, on a swap or a cue
    vec2 d = (uv - 0.5) * uSplit + vec2(uSplit * 0.4, 0.0);
    float l1 = dot(texture2D(uTex, uv + d).rgb, vec3(0.3, 0.45, 0.25)), l2 = dot(texture2D(uTex, uv - d).rgb, vec3(0.3, 0.45, 0.25));
    c = c * (1.0 - 0.25 * uGhost) + (uRed * l1 + uBlue * l2) * uGhost;
  }
  c *= (1.0 - 0.55 * dot(vUV - 0.5, vUV - 0.5) * 2.0) * (1.0 - 0.07 * step(0.5, fract(gl_FragCoord.y * 0.5)));
  c = c / (1.0 + 0.5 * c);
  gl_FragColor = vec4(c * 1.38, 1.0);
}`;

const KIND = { knob: 0, fader: 1, meter: 2, pass: 3, effect: 4, entry: 5, link: 6, ring: 7, beam: 8, mark: 9 };

export class MixerStage {
  constructor(canvas, { decks = 3, calm = false } = {}) {
    this.canvas = canvas; this.n = clamp(decks | 0 || 3, 1, 4);
    this.calm = !!calm || (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.clock = { ctx: null, bpm: 145, barZeroTime: null };
    this.reader = null; this.master = null; this.masterBuf = null;
    this.decks = Array.from({ length: this.n }, () => ({
      track: null, notes: null, audio: null, job: 0, meter: null, meterBuf: null, level: 0, peak: 0,
      knobs: { ...NEUTRAL }, prev: { ...NEUTRAL }, speed: { fader: 0, lo: 0, mid: 0, hi: 0, hp: 0, lp: 0, fx: 0 },
      gain: new Float32Array([1, 1, 1, 0]), playing: false, role: 0, fxClass: 1, primary: 0, endBar: null,
      hist: new Uint8Array(HN * 4), histAt: new Float64Array(HN).fill(-1), lastSlot: -1, marks: new Float32Array(16), low: 0,
    }));
    this.pos = 0; this.lastBar = -1; this.barSeen = 0; this.dj = null; this.primary = -1; this.markY = null;
    this.rings = []; this.beam = null; this.lastPulse = {}; this.kick = 0; this.split = 0; this.lastFlash = -9; this.lowWas = 0;
    this.running = false; this.raf = 0; this.frameMs = 16; this.scale = 1; this.slow = 0; this.frames = 0; this.lost = false;
    this.gl = null; this._loop = this._loop.bind(this);
    this._init();
  }

  get ok() { return !!this.gl; }
  setClock({ ctx = null, bpm = 145, barZeroTime = null } = {}) { this.clock = { ctx, bpm: bpm || 145, barZeroTime }; }
  setAudio(analyser) { this.master = analyser || null; this.masterBuf = analyser ? new Uint8Array(analyser.frequencyBinCount) : null; }
  setMeter(deck, analyser) { const D = this.decks[deck]; if (!D) return; D.meter = analyser || null; D.meterBuf = analyser ? new Uint8Array(analyser.fftSize || 512) : null; }
  setCalm(on) { this.calm = !!on; }
  setReader(fn) { this.reader = typeof fn === 'function' ? fn : null; }

  // a deck's track: its length in bars, the global bar its bar 0 falls on, the bar its end loops back to
  setTrack(deck, t) {
    const D = this.decks[deck]; if (!D) return;
    if (!t) { D.track = null; D.endBar = null; return; }
    if (!D.track || D.track.id !== t.id) { D.notes = null; D.audio = null; D.job++; }          // another track: its wave is not this one's
    const bars = Math.max(1, t.bars | 0);
    D.track = { id: t.id, bars, startBar: +t.startBar || 0, loopFrom: t.loopFrom === undefined || t.loopFrom === null ? Math.max(0, bars - 32) : clamp(t.loopFrom | 0, 0, bars - 1) };
  }

  // a deck's wave, in one of three forms (any of them at any time, the audio shown where it is there):
  //   an AudioBuffer of the whole track: analysed here in short slices (three bands)
  //   { bars: [...] }: the track's note file, the wave drawn from its notes
  //   { peaks, perBar }: the whole track's peaks if the page has them (see addWave for their form)
  //   null: nothing
  setWave(deck, source) {
    const D = this.decks[deck]; if (!D) return;
    if (!source) { D.notes = null; D.audio = null; D.job++; return; }
    if (typeof source.getChannelData === 'function') { D.audio = null; D.job++; this._audio(D, source, 0, true); }
    else if (Array.isArray(source.bars)) D.notes = notesWave(source.bars);
    else if (source.peaks) { D.audio = null; D.job++; this.addWave(deck, { fromBar: 0, peaks: source.peaks, perBar: source.perBar }); }
  }

  // a piece of a deck's wave, as the page decodes the track in segments:
  //   { fromBar, audio }: an AudioBuffer that holds the track from that bar of the track on; analysed here in slices
  //   { fromBar, peaks, perBar }: peaks the page made itself: triples (low, mid, high) per slot, `perBar` slots a bar
  //     (64 if not said), from that bar on; a Uint8Array counts 0..255, any other array 0..1
  // Pieces may come in any order and may overlap; the bars without a piece yet show the note wave.
  addWave(deck, piece) {
    const D = this.decks[deck]; if (!D || !piece) return;
    const from = Math.max(0, +piece.fromBar || 0);
    if (piece.audio && typeof piece.audio.getChannelData === 'function') { this._audio(D, piece.audio, from, false); return; }
    const pk = piece.peaks; if (!pk || pk.length < 3) return;
    const per = piece.perBar || NS, count = Math.floor(pk.length / 3), scale = pk instanceof Uint8Array ? 1 : 255;
    const s0 = Math.floor(from * NS), s1 = Math.floor((from + count / per) * NS), A = this._store(D, s1);
    for (let s = s0; s < s1 && s < A.n; s++) {
      const i0 = clamp(Math.floor((s / NS - from) * per), 0, count - 1), i1 = clamp(Math.floor(((s + 1) / NS - from) * per), i0 + 1, count);
      for (let q = 0; q < 3; q++) { let v = 0; for (let i = i0; i < i1; i++) v = Math.max(v, pk[i * 3 + q] * scale); A.data[s * 3 + q] = v = clamp(Math.round(v), 0, 255); if (v > A.top[q]) A.top[q] = v; }
      if (!A.done[s]) { A.done[s] = 1; A.have++; }
    }
  }
  // the store of a deck's audio wave: three bands per 1/64 bar of the track
  _store(D, need) {
    const want = Math.max(need, D.track ? D.track.bars * NS : 0, NS);
    if (D.audio && D.audio.n >= want) return D.audio;
    // top: each band's loudest slot so far; a band is drawn against 0.62 of it (a quiet master fills its lane like a loud one, the few loudest slots reach the band's full height)
    const A = { n: want, data: new Uint8Array(want * 3), done: new Uint8Array(want), have: 0, busy: 0, top: [40, 40, 40] };
    if (D.audio) { A.data.set(D.audio.data); A.done.set(D.audio.done); A.have = D.audio.have; A.busy = D.audio.busy; A.top = D.audio.top; }
    D.audio = A;
    return A;
  }

  // a one-shot event, if the page wants to say it itself: 'cue' (a track starts on that deck), 'swap' (that deck takes
  // the bass and becomes the primary), 'leave' (that deck's track ends), 'gate' (that deck is cut). The stage finds
  // the same events in the reader's data; an event said twice within 0.8 s is shown once.
  pulse(kind, deck) { this._event(kind, deck | 0, undefined, true); }

  resize() {
    const c = this.canvas, dpr = Math.min(2, typeof devicePixelRatio === 'number' ? devicePixelRatio : 1) * this.scale;
    const w = Math.max(2, Math.round((c.clientWidth || c.width || 2) * dpr)), h = Math.max(2, Math.round((c.clientHeight || c.height || 2) * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (this.gl) { this._targets(); this._layout(); }
  }
  start() { if (this.running || !this.gl) return; this.running = true; this.last = performance.now(); this.raf = requestAnimationFrame(this._loop); }
  stop() { this.running = false; if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }
  stats() {
    // waves: per deck 'none', 'notes', 'audio' (all of the track) or 'audio 40%' (that share of its bars so far); busy: an analysis is running
    const wave = (D) => { const A = D.audio; if (!A || !A.have) return D.notes ? 'notes' : 'none'; const all = D.track ? D.track.bars * NS : A.n; return A.have >= all ? 'audio' : 'audio ' + Math.round((100 * A.have) / all) + '%'; };
    return { frameMs: this.frameMs, scale: this.scale, webgl: !!this.gl, bar: this.pos, waves: this.decks.map(wave), busy: this.decks.some((D) => D.audio && D.audio.busy > 0) };
  }

  // ---------- time ----------
  _now() { const c = this.clock.ctx; return c && typeof c.currentTime === 'number' ? c.currentTime : performance.now() / 1000; }
  _barDur() { return 240 / (this.clock.bpm || 145); }

  // ---------- the wave from audio: two one-pole filters split three bands, a slot is 1/64 bar at the clock's tempo ----------
  // (never one long task: five milliseconds of work, then the page gets its turn)
  _audio(D, buf, fromBar, whole) {
    const sr = buf.sampleRate, len = (this._barDur() * sr) / NS, slots = Math.floor(buf.length / len), job = D.job, base = Math.floor(fromBar * NS);
    if (slots < 1) return;
    this._store(D, base + slots).busy++;
    const ch0 = buf.getChannelData(0), ch1 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : null;
    const aL = 1 - Math.exp((-2 * Math.PI * 170) / sr), aM = 1 - Math.exp((-2 * Math.PI * 2600) / sr);
    let at = 0;                                               // a whole track starts where the deck plays now, so the bars in view come first
    if (whole && D.track) { const i = this.pos - D.track.startBar; if (i > 0 && i < D.track.bars) at = clamp(Math.floor(i * NS) - 8, 0, slots - 1); }
    const first = at; let todo = slots, l1 = 0, l2 = 0, m1 = 0;
    const slice = () => {
      if (D.job !== job || !D.audio) return;                  // another wave took this deck
      const A = D.audio, t0 = performance.now();
      while (todo > 0 && performance.now() - t0 < 5) {
        for (let k = 0; k < 16 && todo > 0; k++, todo--) {
          const a = Math.round(at * len), b = Math.min(buf.length, Math.round((at + 1) * len));
          let pk = 0, sm = 0, sh = 0;
          for (let i = a; i < b; i++) {
            const x = ch1 ? (ch0[i] + ch1[i]) * 0.5 : ch0[i];
            l1 += aL * (x - l1); l2 += aL * (l1 - l2); m1 += aM * (x - m1);
            const lo = l2 < 0 ? -l2 : l2, mid = m1 - l2, hi = x - m1;
            if (lo > pk) pk = lo; sm += mid * mid; sh += hi * hi;
          }
          const s = base + at, cnt = Math.max(1, b - a);
          if (s < A.n) {
            A.data[s * 3] = Math.round(255 * Math.min(1, Math.pow(pk * 1.3, 0.8)));
            A.data[s * 3 + 1] = Math.round(255 * Math.min(1, Math.pow(Math.sqrt(sm / cnt) * 4.4, 0.8)));
            A.data[s * 3 + 2] = Math.round(255 * Math.min(1, Math.pow(Math.sqrt(sh / cnt) * 9.5, 0.8)));
            for (let q = 0; q < 3; q++) if (A.data[s * 3 + q] > A.top[q]) A.top[q] = A.data[s * 3 + q];
            if (!A.done[s]) { A.done[s] = 1; A.have++; }
          }
          if (++at >= slots) { at = 0; l1 = l2 = m1 = 0; }
          if (at === first && todo > 1) todo = 1;
        }
      }
      if (todo > 0) setTimeout(slice, 0); else A.busy--;
    };
    setTimeout(slice, 0);
  }

  // a deck's three bands between two global bar positions; false where its track is not
  _bands(D, gA, gB, out) {
    const T = D.track; if (!T) return false;
    if (gB <= T.startBar || (D.endBar !== null && gA >= D.endBar)) return false;
    let iA = Math.max(0, gA - T.startBar), iB = Math.max(iA, gB - T.startBar);
    if (iA >= T.bars) { const span = T.bars - T.loopFrom, w = iB - iA; iA = T.loopFrom + ((iA - T.bars) % span); iB = Math.min(T.bars, iA + w); }
    else if (iB > T.bars) iB = T.bars;
    out[0] = out[1] = out[2] = 0;
    const A = D.audio;
    if (A) {
      const fA = iA * NS, fB = iB * NS, a = Math.floor(fA), b = Math.min(A.n - 1, Math.floor(fB));
      if (a < A.n && A.done[a] && A.done[b] && A.done[Math.min(A.n - 1, a + 1)]) { this._range(A, fA, fB, out); for (let q = 0; q < 3; q++) out[q] = Math.min(255, (out[q] * 228) / (A.top[q] * 0.62)); return true; }
    }
    if (D.notes) { this._range(D.notes, iA * NS, iB * NS, out); return true; }
    out[0] = 10; out[1] = 6; out[2] = 4;                      // a track without a wave yet: a thin line
    return true;
  }
  _range(W, fA, fB, out) {
    const d = W.data, n = W.n; let a = Math.floor(fA), b = Math.floor(fB);
    if (b - a < 1) {                                          // under one slot: between its two neighbours, so the strip glides
      const f = (fA + fB) / 2 - 0.5, i = clamp(Math.floor(f), 0, n - 2), w = clamp(f - i, 0, 1);
      for (let q = 0; q < 3; q++) out[q] = d[i * 3 + q] * (1 - w) + d[(i + 1) * 3 + q] * w;
      return;
    }
    a = clamp(a, 0, n - 1); b = clamp(b, a, n - 1);
    for (let i = a; i <= b; i++) for (let q = 0; q < 3; q++) if (d[i * 3 + q] > out[q]) out[q] = d[i * 3 + q];
  }

  // ---------- set up ----------
  _init() {
    const c = this.canvas, opts = { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: 'high-performance' };
    let gl = null;
    try { gl = c.getContext('webgl2', opts) || c.getContext('webgl', opts) || c.getContext('experimental-webgl', opts); } catch (e) { gl = null; }
    if (!gl) return;
    this.gl = gl;
    c.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });
    c.addEventListener('webglcontextrestored', () => { this.lost = false; try { this._gpu(); } catch (e) { this.gl = null; } });
    try { this._gpu(); } catch (e) { this.error = String(e && e.message || e); this.gl = null; }
  }
  _program(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s) || 'shader');
      gl.attachShader(p, s);
    }
    gl.bindAttribLocation(p, 0, 'aP');
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p) || 'link');
    const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, name); }
    return { p, u };
  }
  _texture(w, h, filter, data) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data || null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  _gpu() {
    const gl = this.gl;
    this.prog = { strip: this._program(VS_FULL, FS_STRIP), back: this._program(VS_FULL, FS_BACK), form: this._program(VS_FORM, FS_FORM),
      blur: this._program(VS_FULL, FS_BLUR), mix: this._program(VS_FULL, FS_MIX), out: this._program(VS_FULL, FS_OUT) };
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    this.rows = this.n * 2;
    this.wavePx = new Uint8Array(COLS * this.rows * 4);
    this.waveTex = this._texture(COLS, this.rows, gl.LINEAR, this.wavePx);
    this.specPx = new Uint8Array(64 * 4);
    this.specTex = this._texture(64, 1, gl.LINEAR, this.specPx);
    this.fbo = null; this.size = [0, 0];
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

  // where everything stands: wide, the controls beside each lane and the lanes running together into the depth;
  // upright (a phone), the decks stacked, each lane level with its controls in a row under it
  _layout() {
    const W = this.canvas.width, H = this.canvas.height, n = this.n, tall = W < H * 0.9;
    const L = { tall, W, H, yL: [], forms: [] };
    if (!tall) {
      Object.assign(L, { x0: 0.18, xp: 0.37, q: 0.8, y0: 0.5, conv: 1, taper: 1, B0: 7, past: 3.5 });
      const slot = 0.92 / n; L.h0 = slot * 0.36;
      for (let d = 0; d < n; d++) L.yL.push(0.96 - (d + 0.5) * slot);
      const hp = L.h0 * H, r1 = Math.min(hp * 0.3, W * 0.02), r2 = Math.min(hp * 0.44, W * 0.03);
      for (let d = 0; d < n; d++) {
        const y = L.yL[d] * H;
        L.forms.push({ meter: [W * 0.024, y, W * 0.007, hp], fader: [W * 0.048, y, W * 0.012, hp], hi: [W * 0.085, y + hp * 0.66, r1, r1], mid: [W * 0.085, y, r1, r1], lo: [W * 0.085, y - hp * 0.66, r1, r1],
          pass: [W * 0.135, y + hp * 0.5, r2, r2], effect: [W * 0.135, y - hp * 0.5, r2, r2] });
      }
    } else {
      Object.assign(L, { x0: 0, xp: 0.26, q: 0.8, y0: 0.5, conv: 0, taper: 0.45, B0: 3.2, past: 1.6 });
      const slot = 0.94 / n; L.h0 = slot * 0.21;
      for (let d = 0; d < n; d++) {
        const top = 0.97 - d * slot; L.yL.push(top - slot * 0.31);
        const y = (top - slot * 0.8) * H, rh = slot * H * 0.13, r = Math.min(rh * 0.9, W * 0.062);
        L.forms.push({ meter: [W * 0.06, y, W * 0.016, rh], fader: [W * 0.15, y, W * 0.026, rh], lo: [W * 0.29, y, r, r], mid: [W * 0.43, y, r, r], hi: [W * 0.57, y, r, r],
          pass: [W * 0.72, y, r * 1.12, r * 1.12], effect: [W * 0.88, y, r * 1.12, r * 1.12] });
      }
    }
    // the bar offset (from the play line) at every column edge of the strip
    L.colB = new Float32Array(COLS + 1);
    for (let c = 0; c <= COLS; c++) {
      const x = L.x0 + (c / COLS) * (1 - L.x0);
      if (x < L.xp) L.colB[c] = ((x - L.xp) / (L.xp - L.x0)) * L.past;
      else { const s = (x - L.xp) / (1 - L.xp); L.colB[c] = (L.B0 * s) / (1 - L.q * s); }
    }
    L.reach = L.B0 / (1 - L.q);
    this.L = L;
  }
  // where a bar offset sits: [x in pixels, the depth factor there]
  _xOf(b) { const L = this.L; if (b < 0) return [(L.xp + (b / L.past) * (L.xp - L.x0)) * L.W, 1]; const s = b / (L.B0 + L.q * b); return [(L.xp + s * (1 - L.xp)) * L.W, 1 - L.q * s]; }
  _yOf(d, k) { const L = this.L; return (L.conv ? L.y0 + (L.yL[d] - L.y0) * k : L.yL[d]) * L.H; }

  // every full-picture pulse goes through here: never two within 0.36 s (under three a second), none in calm mode
  _pulse(kick, split) {
    const now = performance.now() / 1000;
    if (this.calm || Math.abs(now - this.lastFlash) < 0.36) return false;
    this.lastFlash = now; this.kick = Math.max(this.kick, kick); this.split = Math.max(this.split, split);
    return true;
  }
  _event(kind, deck, from, said) {
    const now = performance.now() / 1000, key = kind + ':' + deck;
    if (this.lastPulse[key] !== undefined && now - this.lastPulse[key] < 0.8) return;        // said by the page and found in the data: once
    this.lastPulse[key] = now;
    if (deck < 0 || deck >= this.n) return;
    if (kind === 'swap') {
      const a = from !== undefined && from >= 0 ? from : this.primary !== deck ? this.primary : this.primaryWas;       // said by the page before the reader shows it: the deck that held the bass until now
      if (a !== undefined && a >= 0 && a !== deck) this.beam = { from: a, to: deck, t0: now };
      this.rings.push({ deck, t0: now, strength: 1.4, warm: 1 });
      this._pulse(1, 1);
    } else if (kind === 'cue') { this.rings.push({ deck, t0: now, strength: 1, warm: 0 }); this._pulse(0.6, 0.45); }
    else if (kind === 'leave') { this.rings.push({ deck, t0: now, strength: 0.7, warm: 0, inward: 1 }); this._pulse(0.3, 0.2); }
    else if (kind === 'gate') this.rings.push({ deck, t0: now, strength: 0.5, warm: 0, red: 1 });
    if (said && kind === 'swap') this.saidSwap = now;
  }

  // ---------- the reader's data into the stage's state ----------
  _read(dt) {
    let r = null;
    if (this.reader) { try { r = this.reader(); } catch (e) { r = null; } }
    const now = this._now(), dur = this._barDur(), z = this.clock.barZeroTime;
    const bar = r && typeof r.bar === 'number' ? r.bar : null;
    if (bar !== null && bar !== this.lastBar) { this.lastBar = bar; this.barSeen = now; }
    let pos = z !== null && z !== undefined ? (now - z) / dur : (bar !== null ? bar + clamp((now - this.barSeen) / dur, 0, 1) : this.pos);
    if (bar !== null && (pos < bar - 0.5 || pos > bar + 1.5)) pos = bar + clamp((now - this.barSeen) / dur, 0, 0.999);       // the clock and the reader disagree: the reader's bar counts
    this.pos = pos;
    const dj = (r && r.dj) || null; this.dj = dj;
    const list = (r && r.decks) || [];
    let primary = dj && typeof dj.primary === 'number' ? dj.primary : -1;
    for (let d = 0; d < this.n; d++) {
      const D = this.decks[d], src = list[d] || {}, k = src.knobs || {};
      const was = D.playing;
      D.playing = !!src.playing;
      D.role = src.role === 'primary' ? 1 : src.role === 'in' ? 2 : src.role === 'out' ? 3 : D.playing ? 2 : 0;
      if (primary < 0 && D.role === 1) primary = d;
      const cutWas = D.knobs.cut;
      const ease = 1 - Math.exp(-dt * 14);                    // the page may say the knobs ten times a second: the forms glide between
      for (const key in NEUTRAL) {
        const v = typeof k[key] === 'number' ? clamp(k[key], 0, 1) : NEUTRAL[key], cur = D.knobs[key];
        if (key in D.speed) D.speed[key] += (Math.min(1, Math.abs(v - cur) * 9) - D.speed[key]) * 0.25;       // how fast the invisible hand turns it
        D.knobs[key] = key === 'cut' || Math.abs(v - cur) < 0.002 || (D.playing && !was) ? v : cur + (v - cur) * ease;
      }
      if (!D.playing) D.knobs.fader = Math.min(D.knobs.fader, 0);
      gainsOf(D.knobs, D.gain);
      D.fxClass = FX_CLASS[src.fxType] || 1;
      D.endBar = null;
      if (D.playing && !was) this._event('cue', d);
      if (!D.playing && was) this._event('leave', d);
      if (D.playing && cutWas >= 0.5 && D.knobs.cut < 0.5) this._event('gate', d);
      // what went out is remembered at the play line
      const slot = Math.floor(pos * HS);
      if (slot < D.lastSlot || slot - D.lastSlot > HN) D.lastSlot = slot - 1;
      for (let s = D.lastSlot + 1; s <= slot; s++) {
        const o = (((s % HN) + HN) % HN);
        for (let q = 0; q < 4; q++) D.hist[o * 4 + q] = Math.round(clamp(D.gain[q] * (D.playing ? 1 : q === 3 ? 0 : 1), 0, 1.275) * 200);
        D.histAt[o] = s;
      }
      D.lastSlot = slot;
      D.marks.fill(0);
    }
    if (primary !== this.primary) {
      if (this.primary >= 0 && primary >= 0) this._event('swap', primary, this.primary);
      this.primaryWas = this.primary; this.primary = primary;
    }
    // the gates on the lanes: where a track enters, where the bass changes decks, where a track ends
    const put = (d, b, kind, strength) => { const D = this.decks[d]; if (!D || b < -this.L.past || b > this.L.reach) return; for (let i = 0; i < 4; i++) if (!D.marks[i * 4 + 1]) { D.marks.set([b, kind, strength, 0], i * 4); return; } };
    this.links = []; this.entry = null;
    if (dj && dj.on !== false) {
      for (const t of dj.transitions || []) {
        if (typeof t.swap === 'number') {
          const b = t.swap - pos;
          if (b > -this.L.past) { put(t.in, b, 1, 1); put(t.out, b, 2, 1); if (b > 0.02 && b < this.L.reach) this.links.push({ b, from: t.out, to: t.in }); }
        }
        if (typeof t.end === 'number' && this.decks[t.out]) { this.decks[t.out].endBar = t.end; put(t.out, t.end - pos, 3, 0.8); }
        if (typeof t.start === 'number') put(t.in, t.start - pos, 4, 0.7);
      }
      if (typeof dj.nextStart === 'number') {
        const idle = this.decks.findIndex((D) => !D.playing), rem = dj.nextStart - pos, sec = dj.section || 32;
        if (idle >= 0 && rem > -0.25) { this.entry = { deck: idle, closed: clamp(1 - rem / sec, 0, 1) }; put(idle, rem, 4, 0.9); }
      }
    }
    return pos;
  }

  // ---------- a frame ----------
  _loop(ts) {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this._loop);
    const dt = Math.min(100, ts - (this.last || ts)); this.last = ts;
    this.frameMs += (dt - this.frameMs) * 0.05; this.frames++;
    if (this.frames > 90 && this.frameMs > 26 && this.scale > 0.5) {         // too slow: a lower resolution, the same picture
      if (++this.slow > (this.frameMs > 60 ? 20 : 60)) { this.scale = Math.max(0.5, this.scale * (this.frameMs > 60 ? 0.6 : 0.8)); this.slow = 0; this.frames = 0; this.resize(); }
    } else this.slow = 0;
    if (this.gl && !this.lost) this._draw(dt / 1000);
  }

  _draw(dt) {
    const gl = this.gl, c = this.canvas, P = this.prog;
    if (c.clientWidth && Math.abs(c.clientWidth * Math.min(2, devicePixelRatio || 1) * this.scale - c.width) > 2) this.resize();
    if (!this.fbo) this._targets();
    if (!this.L || this.L.W !== c.width || this.L.H !== c.height) this._layout();
    const L = this.L, W = L.W, H = L.H, n = this.n, real = performance.now() / 1000, calm = this.calm;
    const pos = this._read(dt);
    // the strips' columns: shape (three bands, is there a track) and light (what went out, or what would go out now)
    const px = this.wavePx, out = [0, 0, 0];
    for (let d = 0; d < n; d++) {
      const D = this.decks[d], r0 = d * 2 * COLS * 4, r1 = (d * 2 + 1) * COLS * 4;
      const g = [Math.round(clamp(D.gain[0], 0, 1.275) * 200), Math.round(clamp(D.gain[1], 0, 1.275) * 200), Math.round(clamp(D.gain[2], 0, 1.275) * 200)];
      for (let col = 0; col < COLS; col++) {
        const bA = L.colB[col], bB = L.colB[col + 1], o = r0 + col * 4, o2 = r1 + col * 4;
        if (this._bands(D, pos + bA, pos + bB, out)) { px[o] = out[0]; px[o + 1] = out[1]; px[o + 2] = out[2]; px[o + 3] = 255; }
        else { px[o] = px[o + 1] = px[o + 2] = px[o + 3] = 0; }
        if (bB <= 0) {
          const s = Math.floor((pos + (bA + bB) / 2) * HS), h = ((s % HN) + HN) % HN;
          if (D.histAt[h] === s) { px[o2] = D.hist[h * 4]; px[o2 + 1] = D.hist[h * 4 + 1]; px[o2 + 2] = D.hist[h * 4 + 2]; px[o2 + 3] = D.hist[h * 4 + 3]; continue; }
          px[o2 + 3] = Math.round(clamp(D.gain[3], 0, 1.275) * 200);
        } else px[o2 + 3] = 200;
        px[o2] = g[0]; px[o2 + 1] = g[1]; px[o2 + 2] = g[2];
      }
      // the deck's level: from its meter, else from its wave at the play line and its level
      this._bands(D, pos - 0.02, pos, out);
      const here = D.track && D.playing ? Math.max(out[0] * D.gain[0], out[1] * D.gain[1], out[2] * D.gain[2]) / 255 * Math.min(1, D.gain[3]) : 0;
      D.lowNow = D.track && D.playing ? (out[0] / 255) * D.gain[0] * Math.min(1, D.gain[3]) : 0;
      let lv = here;
      if (D.meter && D.meterBuf) { D.meter.getByteTimeDomainData(D.meterBuf); let sum = 0; for (let i = 0; i < D.meterBuf.length; i++) { const v = (D.meterBuf[i] - 128) / 128; sum += v * v; } lv = clamp(Math.sqrt(sum / D.meterBuf.length) * 3.2, 0, 1); }
      D.level += (lv - D.level) * (lv > D.level ? 0.6 : 0.16);
      D.peak = Math.max(D.level, D.peak - dt * 0.35);
      D.primary += ((d === this.primary ? 1 : 0) - D.primary) * Math.min(1, dt * 8);
    }
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.waveTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, COLS, this.rows, gl.RGBA, gl.UNSIGNED_BYTE, px);
    // the master's spectrum, low to high; without an analyser: the decks' three bands
    const sp = this.specPx; let total = 0;
    if (this.master && this.masterBuf) {
      this.master.getByteFrequencyData(this.masterBuf); const m = this.masterBuf.length;
      for (let i = 0; i < 64; i++) { const a = Math.floor(Math.pow(i / 64, 2.2) * (m * 0.7)), b = Math.max(a + 1, Math.floor(Math.pow((i + 1) / 64, 2.2) * (m * 0.7))); let v = 0; for (let j = a; j < b; j++) v = Math.max(v, this.masterBuf[j]); sp[i * 4] = v; total += v; }
    } else {
      const band = [0, 0, 0];
      for (const D of this.decks) if (D.playing) { this._bands(D, pos - 0.02, pos, out); for (let q = 0; q < 3; q++) band[q] = Math.max(band[q], out[q] * Math.min(1, D.gain[q]) * Math.min(1, D.gain[3])); }
      for (let i = 0; i < 64; i++) { const v = band[i < 20 ? 0 : i < 44 ? 1 : 2] * (0.75 + 0.25 * Math.sin(i * 2.4 + real * 3)); sp[i * 4] += (v - sp[i * 4]) * 0.5; total += sp[i * 4]; }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.specTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 64, 1, gl.RGBA, gl.UNSIGNED_BYTE, sp);
    // the beat: the primary's low band rises at the play line
    const low = this.primary >= 0 ? this.decks[this.primary].lowNow || 0 : 0;
    if (low > 0.34 && low - this.lowWas > 0.14) this._pulse(0.5, 0);
    this.lowWas += (low - this.lowWas) * 0.5;
    this.kick *= Math.exp(-dt * 7); this.split *= Math.exp(-dt * 3.4);

    const F = this.fbo;
    gl.bindFramebuffer(gl.FRAMEBUFFER, F.scene.fb); gl.viewport(0, 0, W, H);
    gl.disable(gl.SCISSOR_TEST); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const bind = (unit, tex) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); };
    const geo = [L.x0, L.xp, L.q, L.y0], view = [L.B0, L.past, ((pos % 32) + 32) % 32, real % 1000];
    // behind the lanes
    let p = P.back; gl.useProgram(p.p);
    bind(0, this.specTex); gl.uniform1i(p.u.uSpec, 0); gl.uniform2f(p.u.uRes, W, H); gl.uniform4fv(p.u.uGeo, geo); gl.uniform4fv(p.u.uView, view);
    gl.uniform4f(p.u.uFan, L.yL[0] + L.h0 * 1.3, L.yL[n - 1] - L.h0 * 1.3, L.conv, (L.tall ? 0.4 : 1) * (0.6 + 0.4 * clamp(total / (64 * 90), 0, 1)));
    gl.uniform3fv(p.u.uCyan, CYAN); gl.uniform3fv(p.u.uInk, INK);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // the lanes
    p = P.strip; gl.useProgram(p.p);
    bind(0, this.waveTex); gl.uniform1i(p.u.uWave, 0); gl.uniform2f(p.u.uRes, W, H); gl.uniform4fv(p.u.uGeo, geo); gl.uniform4fv(p.u.uView, view);
    gl.uniform4f(p.u.uShape, L.conv, L.taper, 0, 0);
    gl.uniform3fv(p.u.uLow, LOW); gl.uniform3fv(p.u.uMid, MIDT); gl.uniform3fv(p.u.uHigh, HIGH); gl.uniform3fv(p.u.uInk, INK); gl.uniform3fv(p.u.uCyan, CYAN); gl.uniform3fv(p.u.uWarm, WARM); gl.uniform3fv(p.u.uRed, RED);
    gl.enable(gl.SCISSOR_TEST);
    for (let d = 0; d < n; d++) {
      const D = this.decks[d], y = L.yL[d], hh = L.h0 * 1.45;
      const lo = Math.min(y - hh, L.conv ? L.y0 + (y - hh - L.y0) * (1 - L.q) : y - hh), hi = Math.max(y + hh, L.conv ? L.y0 + (y + hh - L.y0) * (1 - L.q) : y + hh);
      gl.scissor(Math.floor(L.x0 * W), Math.max(0, Math.floor(lo * H)), W - Math.floor(L.x0 * W), Math.min(H, Math.ceil((hi - lo) * H) + 2));
      gl.uniform4f(p.u.uLane, y, L.h0, (d * 2 + 0.5) / this.rows, (d * 2 + 1.5) / this.rows);
      gl.uniform4f(p.u.uState, D.playing ? 1 : 0, D.primary, D.level, D.playing ? D.knobs.fx : 0);
      gl.uniform4f(p.u.uFx, D.fxClass, D.playing ? D.knobs.cut : 1, this.kick, D.role);
      gl.uniform3f(p.u.uGain, Math.min(1, D.gain[0]), Math.min(1, D.gain[1]), Math.min(1, D.gain[2]));
      gl.uniform4fv(p.u.uMark, D.marks);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.disable(gl.SCISSOR_TEST);
    // the controls and the Auto DJ's signs
    p = P.form; gl.useProgram(p.p);
    gl.uniform2f(p.u.uRes, W, H); gl.uniform1f(p.u.uTime, real % 1000); gl.uniform3fv(p.u.uRed, RED);
    const form = (kind, box, col, col2, a, b, cc, e) => {
      gl.uniform1f(p.u.uKind, kind); gl.uniform2f(p.u.uC, box[0], box[1]); gl.uniform2f(p.u.uS, box[2], box[3]);
      gl.uniform3fv(p.u.uCol, col); gl.uniform3fv(p.u.uCol2, col2); gl.uniform4f(p.u.uVal, a, b || 0, cc || 0, e || 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    const xp = L.xp * W, hp = L.h0 * H;
    for (let d = 0; d < n; d++) {
      const D = this.decks[d], f = L.forms[d], k = D.knobs, sp2 = D.speed, live = D.playing ? 1 : 0.45;
      const dim = (col) => [col[0] * live, col[1] * live, col[2] * live];
      form(KIND.meter, f.meter, dim(CYAN), INK, D.level, D.peak, D.playing ? k.cut : 1);
      form(KIND.fader, f.fader, dim(CYAN), INK, clamp(k.fader, 0, 1), sp2.fader, 0.8, D.playing ? k.cut : 1);
      form(KIND.knob, f.lo, dim(LOW), INK, k.lo, sp2.lo, k.lo <= 0.015 && D.playing ? 1 : 0);
      form(KIND.knob, f.mid, dim(MIDT), INK, k.mid, sp2.mid, k.mid <= 0.015 && D.playing ? 1 : 0);
      form(KIND.knob, f.hi, dim(HIGH), INK, k.hi, sp2.hi, k.hi <= 0.015 && D.playing ? 1 : 0);
      form(KIND.pass, f.pass, dim(CYAN), INK, 0.699 * k.hp, 0.318 + 0.682 * k.lp, Math.max(sp2.hp, sp2.lp));         // both filters on one axis, 20 Hz to 20 kHz
      form(KIND.effect, f.effect, dim(WARM), INK, D.playing ? k.fx : 0, D.fxClass, real * (1.2 + D.fxClass * 0.5), sp2.fx);
    }
    if (this.entry) form(KIND.entry, [xp, L.yL[this.entry.deck] * H, hp * 1.0, hp * 1.0], CYAN, INK, this.entry.closed);
    for (const l of this.links || []) {
      const [x, k] = this._xOf(l.b), ya = this._yOf(l.from, k), yb = this._yOf(l.to, k);
      form(KIND.link, [x, (ya + yb) / 2, 12 * k + 3, Math.abs(ya - yb) / 2], LOW, INK, yb > ya ? 1 : -1, 0.5 + 0.5 * k);
    }
    if (this.primary >= 0) {                        // the primary's mark slides to the deck that took the bass
      const target = L.yL[this.primary] * H;
      this.markY = this.markY === null || this.markAt !== L.H ? target : this.markY + (target - this.markY) * Math.min(1, dt * 9);
      this.markAt = L.H;
      form(KIND.mark, [xp, this.markY, hp * 0.3, hp * 1.22], WARM, INK, 1, this.decks[this.primary].level);
    }
    this.rings = this.rings.filter((r) => real - r.t0 < 0.9);
    for (const r of this.rings) {
      const t = (real - r.t0) / 0.9, size = hp * (1.6 + r.strength);
      form(KIND.ring, [xp, L.yL[r.deck] * H, size, size], r.red ? RED : r.warm ? WARM : CYAN, INK, r.inward ? 1 - t : t, r.strength * (calm ? 0.5 : 1) * (r.inward ? 0.6 + t : 1));
    }
    if (this.beam) {
      const age = real - this.beam.t0, s = Math.exp(-age * 2.6);
      if (s < 0.02) this.beam = null;
      else { const ya = L.yL[this.beam.from] * H, yb = L.yL[this.beam.to] * H; form(KIND.beam, [xp, (ya + yb) / 2, 40, Math.abs(ya - yb) / 2 + hp], LOW, INK, s * (calm ? 0.5 : 1), calm ? 0 : 1); }
    }
    gl.disable(gl.BLEND);
    // the glow, the trails, the screen
    const quad = () => gl.drawArrays(gl.TRIANGLES, 0, 6);
    p = P.blur; gl.useProgram(p.p); gl.uniform1i(p.u.uTex, 0);
    const pass = (src, dst, dx, dy) => { gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb); gl.viewport(0, 0, dst.w, dst.h); bind(0, src.tex); gl.uniform2f(p.u.uDir, dx / dst.w, dy / dst.h); quad(); };
    pass(F.scene, F.b1, 1.2, 0); pass(F.b1, F.b2, 0, 1.2); pass(F.b2, F.b1, 2.4, 0); pass(F.b1, F.b2, 0, 2.4);
    const cx = Math.min(1.3, L.xp + (1 - L.xp) / L.q), cy = L.y0;
    p = P.mix; gl.useProgram(p.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, F.keepB.fb); gl.viewport(0, 0, W, H);
    bind(0, F.scene.tex); gl.uniform1i(p.u.uScene, 0); bind(1, F.b2.tex); gl.uniform1i(p.u.uBloom, 1); bind(2, F.keepA.tex); gl.uniform1i(p.u.uPrev, 2);
    gl.uniform1f(p.u.uDecay, Math.pow(calm ? 0.45 : 0.68 + 0.14 * this.kick, dt * 60)); gl.uniform1f(p.u.uZoom, 1 - (calm ? 0 : 0.0012 + 0.02 * this.kick));
    gl.uniform1f(p.u.uFrom, L.tall ? -1 : L.x0);
    gl.uniform1f(p.u.uBloomGain, 0.85 + (calm ? 0 : 0.4 * this.kick)); gl.uniform2f(p.u.uCentre, cx, cy);
    quad();
    p = P.out; gl.useProgram(p.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, c.width, c.height);
    bind(0, F.keepB.tex); gl.uniform1i(p.u.uTex, 0);
    gl.uniform1f(p.u.uSplit, calm ? 0 : 0.016 * this.split); gl.uniform1f(p.u.uGhost, calm ? 0 : clamp(this.split * 0.9, 0, 0.7)); gl.uniform1f(p.u.uPunch, calm ? 0 : 0.03 * this.kick);
    gl.uniform2f(p.u.uCentre, Math.min(1, cx), cy); gl.uniform3fv(p.u.uRed, RED); gl.uniform3fv(p.u.uBlue, BLUE);
    quad();
    const t = F.keepA; F.keepA = F.keepB; F.keepB = t;
  }
}

export default MixerStage;
