// Cadance radio, the screen stage: a light sculpture. A sphere of small icosahedral lamps turns slowly in a dark
// room, and the three synth lines of the track that plays write its colours.
//
// The object. The cells stand on a geodesic sphere (92 of them, 42 on a small canvas): twelve cells have five
// neighbours, all others six. Every cell is an icosahedron: twelve ports at its corners, thirty edges, twenty faces.
// Neighbouring cells are joined port to port by a seam, and a port carries at most one seam. A cell is a lamp:
// twenty frosted panels lit from inside, a dark frame on the edges, a small bright point at every port. Light
// spills from a cell onto its neighbours' frames and glass, runs along the seams, glows in the air around the
// sphere and lies on the floor under it.
//
// The light program. One rule for all three lines, with a few numbers per line:
//   a note lights a spot at the cell of its pitch class. The twelve five-neighbour cells are the twelve pitch
//   classes: a tritone is the opposite side of the sphere, a ring of five is a chain of fifths;
//   from there the spot wanders over the neighbour graph, cell by cell along a shortest way, toward the cell of the
//   line's next note, and where the next note is known it arrives as that note starts;
//   a held note keeps its spot alive; a note marked as a slide makes no new spot: the line's spot is dragged to
//   the new place; an accent is brighter and sends a ring over the sphere;
//   every cell remembers the light that passed (an afterglow), so a moving spot draws a tail.
//   The bass is slow and wide and sends a ring with every note, the lead is fast and narrow, the air is a wide,
//   slow wash. Pitch class picks the hue inside the line's family (bass blue to magenta, lead green to sky, air
//   rose to peach), the octave its lightness. Where lines meet, the stronger one sets the hue and the others tint
//   it; brightness is bounded, so the sphere is never white all over. A dim tide in the page's cyan keeps it from
//   going dark. A kick is a small breath of the whole sculpture's brightness (never two within 0.36 s), the hats
//   make ports twinkle. Calm mode: no breath, no twinkle, slower rings, a slower turn.
//
// Read from the bar: tokens (which line sounds), patterns (the drum hits; per line the notes with pitch, velocity,
// length, slide and accent). Not read: mods, patches, view. Derived, and said so here: everything else. Which
// cell is which pitch class, the ways the spots take, their widths, speeds and colours, the afterglow, the rings,
// the tide, the breath and the twinkle are this module's own program; the track's data gives the notes only.
//
// One ES module, no dependency, no request. WebGL 1 or 2; without it `ok` is false and nothing is drawn.
//
//   const screen = new ScreenStage(canvas, { calm: false });
//   screen.ok                                   false: no WebGL, the page falls back
//   screen.setClock({ ctx, bpm, barZeroTime }); bar k starts at barZeroTime + k * 240 / bpm; without ctx: performance.now()
//   screen.setBar(index, bar);                  the bar that sounds now (tokens, patterns)
//   screen.setAudio(analyser);                  the master analyser, optional: it scales the overall level a little
//   screen.setCalm(on); screen.resize(); screen.start(); screen.stop(); screen.stats();
//   also: screen.setView({ turn, zoom })        for stills and close-ups: a fixed turn in radians (null: it turns by itself), a zoom (1: the sphere fits)

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const TAU = Math.PI * 2;
// per line: the hue family (degrees, centre and half span), saturation, lightness, the spot's width (radians on the
// unit sphere), its fastest travel (radians a bar), attack and release (bars), the afterglow's time (bars), the
// ring a note sends (strength, speed in radians a bar, width, life in bars), gain, how the travel is eased
const LINES = [
  { hue: 262, span: 52, sat: 0.96, light: 0.5, sigma: 0.31, speed: 3.2, attack: 0.02, release: 0.2, glow: 0.26, ring: 0.2, ringSpeed: 3.6, ringWidth: 0.18, ringLife: 0.28, gain: 0.9, ease: 1 },
  { hue: 168, span: 38, sat: 0.95, light: 0.5, sigma: 0.17, speed: 16, attack: 0.008, release: 0.24, glow: 0.22, ring: 0, ringSpeed: 9, ringWidth: 0.15, ringLife: 0.14, gain: 1.3, ease: 1.1 },
  { hue: 6, span: 20, sat: 0.92, light: 0.58, sigma: 0.8, speed: 1.5, attack: 0.14, release: 0.8, glow: 0.9, ring: 0, ringSpeed: 2, ringWidth: 0.4, ringLife: 0.6, gain: 0.52, ease: 1 },
];
// the twelve pitch classes on the twelve five-neighbour cells (the pole, the upper ring, the lower ring, the other
// pole): opposite cells are a tritone apart, the upper ring is a chain of fifths
const POLE_OF_PC = (() => { const at = [0, 7, 2, 9, 4, 11, 10, 5, 1, 8, 3, 6], out = new Array(12); at.forEach((pc, cell) => { out[pc] = cell; }); return out; })();

// ---------- small vector and matrix kit (column major) ----------
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const M = {
  persp(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  },
  look(e, a, u) {
    const z = norm([e[0] - a[0], e[1] - a[1], e[2] - a[2]]), x = norm(cross(u, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    return o;
  },
  rotY(t) { const c = Math.cos(t), s = Math.sin(t); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); },
  rotX(t) { const c = Math.cos(t), s = Math.sin(t); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); },
  rotZ(t) { const c = Math.cos(t), s = Math.sin(t); return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); },
};
function hsl(h, s, l) {
  h = (((h % 360) + 360) % 360) / 60; const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs((h % 2) - 1)), m = l - c / 2;
  const [r, g, b] = h < 1 ? [c, x, 0] : h < 2 ? [x, c, 0] : h < 3 ? [0, c, x] : h < 4 ? [0, x, c] : h < 5 ? [x, 0, c] : [c, 0, x];
  return [r + m, g + m, b + m];
}

// ---------- the geometry ----------
// an icosahedron with one corner on the axis `up` of a frame: 12 unit corners (the corner on the axis, a ring of
// five, a ring of five turned by 36 degrees, the opposite corner) and its 20 faces, wound outward
function icosahedron() {
  const c = 1 / Math.sqrt(5), s = 2 / Math.sqrt(5), v = [[0, 0, 1]];
  for (let k = 0; k < 5; k++) v.push([s * Math.cos(k * 1.2566371), s * Math.sin(k * 1.2566371), c]);
  for (let k = 0; k < 5; k++) v.push([s * Math.cos(k * 1.2566371 + 0.6283185), s * Math.sin(k * 1.2566371 + 0.6283185), -c]);
  v.push([0, 0, -1]);
  const faces = [], near = (a, b) => { const d = [v[a][0] - v[b][0], v[a][1] - v[b][1], v[a][2] - v[b][2]]; return dot(d, d) < 1.3; };
  for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) for (let k = j + 1; k < 12; k++) {
    if (!near(i, j) || !near(j, k) || !near(i, k)) continue;
    const n = cross([v[j][0] - v[i][0], v[j][1] - v[i][1], v[j][2] - v[i][2]], [v[k][0] - v[i][0], v[k][1] - v[i][1], v[k][2] - v[i][2]]);
    faces.push(dot(n, [v[i][0] + v[j][0] + v[k][0], v[i][1] + v[j][1] + v[k][1], v[i][2] + v[j][2] + v[k][2]]) > 0 ? [i, j, k] : [i, k, j]);
  }
  return { v, faces };
}
// the sphere: the corners of an icosahedron whose faces are divided `f` times (42 cells for 2, 92 for 3), each
// cell an icosahedron turned so that ports face its neighbours, the seams between those ports, the ways over the
// neighbour graph to each of the twelve five-neighbour cells
function buildSphere(f) {
  const ico = icosahedron(), base = ico.v.map(([x, y, z]) => [x, z, -y]);           // the axis upright
  const pos = base.map((p) => p.slice()), key = new Map(), tris = [];
  const name = (p) => p.map((x) => Math.round(x * 1e5)).join(',');
  pos.forEach((p, i) => key.set(name(p), i));
  const id = (p) => { const q = norm(p), k = name(q); if (!key.has(k)) { key.set(k, pos.length); pos.push(q); } return key.get(k); };
  for (const [a, b, c] of ico.faces) {
    const g = (i, j) => { const k = f - i - j; return id([0, 1, 2].map((q) => (base[a][q] * i + base[b][q] * j + base[c][q] * k) / f)); };
    for (let i = 0; i < f; i++) for (let j = 0; j < f - i; j++) { tris.push([g(i + 1, j), g(i, j + 1), g(i, j)]); if (i + j < f - 1) tris.push([g(i + 1, j), g(i + 1, j + 1), g(i, j + 1)]); }
  }
  const n = pos.length, nbr = Array.from({ length: n }, () => new Set());
  for (const [a, b, c] of tris) { nbr[a].add(b).add(c); nbr[b].add(a).add(c); nbr[c].add(a).add(b); }
  const nb = nbr.map((s) => [...s]);
  let hop = 0, edges = 0, least = 9;
  for (let a = 0; a < n; a++) for (const b of nb[a]) { const ang = Math.acos(clamp(dot(pos[a], pos[b]), -1, 1)); hop += ang; edges++; least = Math.min(least, ang); }
  hop /= edges;
  const rc = least * 0.37;                                    // a cell's radius: the cells stand apart, the far side shows through the gaps
  // a cell's own icosahedron: on a five-neighbour cell a corner faces outward and the inner ring of five ports faces
  // the neighbours; on a six-neighbour cell a face is outward and the six ports round its waist face the neighbours
  const five = ico.v, g0 = norm([0, 1, 2].map((q) => five[0][q] + five[1][q] + five[2][q])), a0 = norm([0, 1, 2].map((q) => five[0][q] - g0[q] * dot(five[0], g0))), b0 = cross(g0, a0);
  const six = five.map((p) => [dot(p, a0), dot(p, b0), dot(p, g0)]);
  const wrap = (x) => Math.atan2(Math.sin(x), Math.cos(x));
  const corners = new Float32Array(n * 36), portOf = nb.map(() => new Map());
  for (let c = 0; c < n; c++) {
    const N = pos[c], e1 = norm(cross(Math.abs(N[1]) > 0.95 ? [1, 0, 0] : [0, 1, 0], N)), e2 = cross(N, e1);
    const local = nb[c].length === 5 ? five : six, cand = [];
    for (let i = 0; i < 12; i++) if (nb[c].length === 5 ? i >= 6 && i <= 10 : Math.abs(local[i][2]) < 0.3) cand.push([Math.atan2(local[i][1], local[i][0]), i]);
    cand.sort((x, y) => x[0] - y[0]);
    const around = nb[c].map((k) => [Math.atan2(dot(pos[k], e2), dot(pos[k], e1)), k]).sort((x, y) => x[0] - y[0]);
    const m = around.length; let best = null;
    for (let s = 0; s < m; s++) {                             // which port faces which neighbour, and the turn that fits them best
      let sx = 0, sy = 0; for (let k = 0; k < m; k++) { const d = around[k][0] - cand[(k + s) % m][0]; sx += Math.cos(d); sy += Math.sin(d); }
      const turn = Math.atan2(sy, sx); let cost = 0; for (let k = 0; k < m; k++) cost += wrap(around[k][0] - cand[(k + s) % m][0] - turn) ** 2;
      if (!best || cost < best.cost) best = { cost, turn, s };
    }
    const ct = Math.cos(best.turn), st = Math.sin(best.turn);
    for (let i = 0; i < 12; i++) {
      const x = local[i][0] * ct - local[i][1] * st, y = local[i][0] * st + local[i][1] * ct, z = local[i][2];
      for (let q = 0; q < 3; q++) corners[(c * 12 + i) * 3 + q] = N[q] + rc * (x * e1[q] + y * e2[q] + z * N[q]);
    }
    for (let k = 0; k < m; k++) portOf[c].set(around[k][1], cand[(k + best.s) % m][1]);       // one port per neighbour, no port twice
  }
  const seams = [];
  for (let a = 0; a < n; a++) for (const b of nb[a]) if (a < b) seams.push([a, b, portOf[a].get(b), portOf[b].get(a)]);
  // the way to each of the twelve pitch cells: from any cell, the neighbour that is one step closer and most in line
  const next = [];
  for (let p = 0; p < 12; p++) {
    const dist = new Int16Array(n).fill(-1), to = new Int16Array(n).fill(-1), queue = [p]; dist[p] = 0;
    for (let h = 0; h < queue.length; h++) for (const k of nb[queue[h]]) if (dist[k] < 0) { dist[k] = dist[queue[h]] + 1; queue.push(k); }
    for (let c = 0; c < n; c++) { let bestDot = -9; for (const k of nb[c]) if (dist[k] === dist[c] - 1) { const d = dot(pos[k], pos[p]); if (d > bestDot) { bestDot = d; to[c] = k; } } }
    next.push(to);
  }
  return { n, pos, nb, rc, hop, corners, faces: ico.faces, seams, next };
}

// ---------- shaders (GLSL ES 1.00) ----------
const HP = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
`;
// a lamp's face: a frosted panel lit from inside by its programmed colour, a dark frame along its three edges
const VS_CELL = `
attribute vec3 aPos, aNor, aB, aEmit, aSpill;
uniform mat4 uVP, uModel; uniform vec3 uEye; uniform vec4 uLens;      // pixels per unit at distance one, a face's inner radius, where the dimming with distance starts and ends
varying vec3 vB, vEmit, vSpill, vK;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0); vec3 n = (uModel * vec4(aNor, 0.0)).xyz;
  vec4 p = uVP * w; gl_Position = p;
  float dist = length(uEye - w.xyz);
  vK = vec3(dot(n, normalize(uEye - w.xyz)), 1.0 - 0.6 * smoothstep(uLens.z, uLens.w, dist), 0.45 * p.w / (uLens.x * uLens.y));
  vB = aB; vEmit = aEmit; vSpill = aSpill;
}`;
const FS_CELL = `${HP}
varying vec3 vB, vEmit, vSpill, vK;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  float b3 = 1.0 - vB.x - vB.y, e = min(vB.x, min(vB.y, b3));
  float panel = smoothstep(0.05, 0.05 + vK.z, e);                 // 0 on the frame, 1 on the glass
  float core = clamp((e - 0.05) / 0.2833, 0.0, 1.0);              // 0 at the frame, 1 over the lamp in the middle
  float glass = (0.5 + 0.5 * pow(core, 0.6)) * (0.93 + 0.07 * hash(floor(vB.xy * 64.0) + vB.z));       // frosted: brightest over the lamp, a fine grain
  float facing = clamp(vK.x, 0.0, 1.0);
  vec3 col = (vEmit * glass * (0.82 + 0.18 * facing) + (vec3(0.016, 0.02, 0.023) + vSpill * 0.42) * (0.45 + 0.55 * glass)) * panel;
  float sheen = pow(1.0 - facing, 2.0);
  col += (vec3(0.005, 0.006, 0.007) + (vEmit * 0.045 + vSpill * 0.16) * (0.35 + 0.65 * sheen)) * (1.0 - panel);      // the frame: dark, with a sheen of the light around it
  gl_FragColor = vec4(col * vK.y, 1.0);
}`;
// a port: a small bright point at a corner
const VS_PORT = `
attribute vec3 aPos, aCol;
uniform mat4 uVP, uModel; uniform vec3 uEye; uniform vec4 uLens; uniform float uSize;
varying vec3 vCol;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0); vec4 p = uVP * w; gl_Position = p;
  gl_PointSize = clamp(uSize * uLens.x / p.w, 1.5, 14.0);
  vCol = aCol * (1.0 - 0.6 * smoothstep(uLens.z, uLens.w, length(uEye - w.xyz)));
}`;
const FS_PORT = `
precision mediump float; varying vec3 vCol;
void main() { float d = length(gl_PointCoord - 0.5) * 2.0, a = max(0.0, 1.0 - d); gl_FragColor = vec4(vCol * a * a, 1.0); }`;
// a seam: a thin link between two ports, in the colours of its two cells, with the light of a spot that crosses it
const VS_SEAM = `
attribute vec3 aA, aB, aCa, aCb; attribute vec2 aT, aPulse;
uniform mat4 uVP, uModel; uniform vec3 uEye; uniform vec4 uLens; uniform vec2 uRes; uniform float uWidth;
varying vec3 vCa, vCb; varying vec4 vS;
void main() {
  vec4 a = uVP * (uModel * vec4(aA, 1.0)), b = uVP * (uModel * vec4(aB, 1.0));
  vec2 d = (b.xy / b.w - a.xy / a.w) * uRes; d = normalize(d + vec2(1e-5, 0.0));
  vec4 p = mix(a, b, aT.x);
  float wpx = clamp(uWidth * uLens.x / p.w, 1.2, 9.0);
  p.xy += vec2(-d.y, d.x) * aT.y * wpx / uRes * 2.0 * p.w;
  gl_Position = p;
  vec3 w = mix((uModel * vec4(aA, 1.0)).xyz, (uModel * vec4(aB, 1.0)).xyz, aT.x);
  vCa = aCa; vCb = aCb; vS = vec4(aT.x, aT.y, aPulse.x, aPulse.y * (1.0 - 0.6 * smoothstep(uLens.z, uLens.w, length(uEye - w))));
  vCa *= 1.0 - 0.6 * smoothstep(uLens.z, uLens.w, length(uEye - w)); vCb *= 1.0 - 0.6 * smoothstep(uLens.z, uLens.w, length(uEye - w));
}`;
const FS_SEAM = `
precision mediump float; varying vec3 vCa, vCb; varying vec4 vS;
void main() {
  float across = 1.0 - vS.y * vS.y, t = vS.x;
  vec3 col = mix(vCa, vCb, t) * 0.4 + vec3(0.012, 0.015, 0.017);
  float pulse = exp(-(t - vS.z) * (t - vS.z) * 18.0) * vS.w;
  col += (mix(vCa, vCb, vS.z) + 0.25) * pulse * 1.6;
  gl_FragColor = vec4(col * across, 1.0);
}`;
const VS_FULL = `attribute vec2 aP; varying vec2 vUV; void main() { vUV = aP * 0.5 + 0.5; gl_Position = vec4(aP, 0.0, 1.0); }`;
// the room: the sphere's light on the floor under it and, faintly, on the wall behind it
const FS_ROOM = `${HP}
varying vec2 vUV;
uniform vec3 uEye, uF, uR, uU, uAvg; uniform vec4 uCam;        // the lens's half width and half height, the picture's upward shift, the floor's height
void main() {
  vec2 ndc = vUV * 2.0 - 1.0; ndc.y -= uCam.z;
  vec3 dir = normalize(uF + uR * ndc.x * uCam.x + uU * ndc.y * uCam.y), col = vec3(0.0);
  if (dir.y < -0.002) {
    vec3 h = uEye + dir * ((uCam.w - uEye.y) / dir.y); float d2 = dot(h, h);
    col += uAvg * 1.5 * (-uCam.w) / (d2 * sqrt(d2)) * exp(-length(h.xz) * 0.22);
  }
  if (dir.z < -0.05) { vec3 h = uEye + dir * ((-4.5 - uEye.z) / dir.z); col += uAvg * 0.1 / (1.0 + dot(h.xy, h.xy) * 0.16) * smoothstep(uCam.w, uCam.w + 1.6, h.y); }
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
// to the screen: the picture, the glow close round the lamps, the haze in the air
const FS_OUT = `
precision mediump float; varying vec2 vUV; uniform sampler2D uScene, uBloom, uHaze;
uniform vec2 uTexel; uniform vec4 uMix;          // glow, haze, unused, brightness
void main() {
  vec3 c = (texture2D(uScene, vUV + uTexel * vec2(0.25, 0.25)).rgb + texture2D(uScene, vUV + uTexel * vec2(-0.25, 0.25)).rgb
          + texture2D(uScene, vUV + uTexel * vec2(0.25, -0.25)).rgb + texture2D(uScene, vUV + uTexel * vec2(-0.25, -0.25)).rgb) * 0.25;
  c += texture2D(uBloom, vUV).rgb * uMix.x + texture2D(uHaze, vUV).rgb * uMix.y;

  c *= uMix.w * (1.0 - 0.5 * dot(vUV - 0.5, vUV - 0.5) * 2.0);
  c = c / (1.0 + 0.45 * c);
  gl_FragColor = vec4(c * 1.4, 1.0);
}`;

export class ScreenStage {
  constructor(canvas, { calm = false } = {}) {
    this.canvas = canvas;
    this.calm = !!calm || (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.clock = { ctx: null, bpm: 145, barZeroTime: null };
    this.score = new Map(); this.curIndex = -1; this.barAt = 0; this.pos = 0; this.posWas = null;
    this.heads = []; this.rings = []; this.kick = 0; this.lastFlash = -9; this.spark = 0; this.sparkSeed = 0; this.stepWas = -1;
    this.level = 1; this.analyser = null; this.spec = null; this.view = { turn: null, zoom: 1 }; this.turn = 0.6;
    this.running = false; this.raf = 0; this.frameMs = 16; this.scale = 1; this.slow = 0; this.frames = 0; this.lost = false;
    this.gl = null; this.S = null; this._loop = this._loop.bind(this);
    this._init();
  }

  get ok() { return !!this.gl; }
  setClock({ ctx = null, bpm = 145, barZeroTime = null } = {}) { this.clock = { ctx, bpm: bpm || 145, barZeroTime }; }
  setAudio(analyser) { this.analyser = analyser || null; this.spec = analyser ? new Uint8Array(analyser.frequencyBinCount) : null; }
  setCalm(on) { this.calm = !!on; }
  setView({ turn = null, zoom = 1 } = {}) { this.view = { turn: typeof turn === 'number' ? turn : null, zoom: clamp(+zoom || 1, 0.4, 4) }; }

  // the bar that sounds now: its three lines' notes, and the steps of its kick and its hats
  setBar(index, bar) {
    if (!bar) return;
    if (index < this.curIndex - 1 || index > this.curIndex + 8) { this.score.clear(); this.heads.length = 0; this.rings.length = 0; this.posWas = null; }       // another track, or a jump
    const tok = bar.tokens || [1, 1, 1, 1], pats = bar.patterns || [], lines = [[], [], []], sets = [];
    let hits = null;
    for (const p of pats) { if (p && p.hits) hits = p.hits; else sets.push(p && p.notes ? p : null); }       // a line keeps its place when another line's pattern is missing
    for (let l = 0; l < 3 && l < sets.length; l++) {
      if (!sets[l] || !(tok[l + 1] > 0)) continue;             // a voice that does not sound writes no light
      const p = sets[l];
      for (let t = 0; t < 16; t++) {
        const midi = p.notes[t]; if (midi === null || midi === undefined) continue;
        lines[l].push({ t: index + t / 16, dur: clamp((p.len && p.len[t]) || 1, 1, 16) / 16, midi, vel: p.vel ? clamp(+p.vel[t] || 0.6, 0.2, 1) : 0.7, slide: !!(p.slide && p.slide[t]), accent: !!(p.accent && p.accent[t]) });
      }
    }
    const row = (k) => { const out = []; for (let t = 0; t < 16; t++) if (hits && hits[k] && +hits[k][t] > 0) out.push(t); return out; };
    const same = index === this.curIndex;
    this.score.set(index, { lines, kicks: row('kick'), hats: [...row('ch'), ...row('oh')] });
    for (const k of this.score.keys()) if (k < index - 2) this.score.delete(k);
    if (!same) { this.curIndex = index; this.barAt = this._now(); }
    // a spot that waited for its line's next note now knows where to go
    for (const h of this.heads) if (h.waiting && lines[h.line].length) { const nx = lines[h.line][0]; if (nx.t > h.t0 && !nx.slide) { this._send(h, nx, Math.max(this.pos, h.t0)); h.waiting = false; } }
  }

  resize() {
    const c = this.canvas, dpr = Math.min(2, typeof devicePixelRatio === 'number' ? devicePixelRatio : 1) * this.scale;
    const w = Math.max(2, Math.round((c.clientWidth || c.width || 2) * dpr)), h = Math.max(2, Math.round((c.clientHeight || c.height || 2) * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    if (!this.gl) return;
    // fewer, larger cells on a small canvas
    const small = Math.min(c.clientWidth || w, c.clientHeight || h) < 520, f = small ? 2 : 3;
    if (!this.S || this.S.f !== f) this._build(f);
    this._targets();
  }
  start() { if (this.running || !this.gl) return; this.running = true; this.last = performance.now(); this.raf = requestAnimationFrame(this._loop); }
  stop() { this.running = false; if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }
  stats() { return { frameMs: this.frameMs, scale: this.scale, webgl: !!this.gl, cells: this.S ? this.S.n : 0, seams: this.S ? this.S.seams.length : 0, spots: this.heads.length, rings: this.rings.length, bar: this.pos }; }

  // ---------- time ----------
  _now() { const c = this.clock.ctx; return c && typeof c.currentTime === 'number' ? c.currentTime : performance.now() / 1000; }
  _barDur() { return 240 / (this.clock.bpm || 145); }
  _pos(now) {
    if (this.curIndex < 0) return 0;
    const z = this.clock.barZeroTime, since = (now - this.barAt) / this._barDur();
    if (z !== null && z !== undefined) { const p = (now - z) / this._barDur() - this.curIndex; if (p > -0.05 && p < 1.25) return this.curIndex + clamp(p, 0, 1.25); }
    return this.curIndex + clamp(since, 0, 1.25);
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
  _program(vs, fs, names) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s) || 'shader');
      gl.attachShader(p, s);
    }
    names.forEach((name, i) => gl.bindAttribLocation(p, i, name));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p) || 'link');
    const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, name); }
    return { p, u, attrs: names.length };
  }
  _texture(w, h) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  _gpu() {
    const gl = this.gl;
    this.prog = { cell: this._program(VS_CELL, FS_CELL, ['aPos', 'aNor', 'aB', 'aEmit', 'aSpill']), port: this._program(VS_PORT, FS_PORT, ['aPos', 'aCol']),
      seam: this._program(VS_SEAM, FS_SEAM, ['aA', 'aB', 'aT', 'aCa', 'aCb', 'aPulse']), room: this._program(VS_FULL, FS_ROOM, ['aP']),
      blur: this._program(VS_FULL, FS_BLUR, ['aP']), out: this._program(VS_FULL, FS_OUT, ['aP']) };
    this.buf = {}; for (const k of ['quad', 'cell', 'light', 'port', 'portCol', 'seam', 'seamCol']) this.buf[k] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    this.fbo = null; this.size = [0, 0, 0]; this.S = null;
    this.resize();
  }
  _target(w, h, depth) {
    const gl = this.gl, tex = this._texture(w, h), fb = gl.createFramebuffer(); let rb = null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (depth) { rb = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, rb); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h); gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb); }
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    return { fb, tex, rb, w, h };
  }
  _targets() {
    const gl = this.gl, w = this.canvas.width, h = this.canvas.height;
    // on a screen of one pixel per point the picture is drawn at one and a half times its size and scaled down: smooth edges without multisampling
    const ss = (typeof devicePixelRatio === 'number' ? devicePixelRatio : 1) * this.scale >= 1.7 ? 1 : 1.5;
    if (this.fbo && this.size[0] === w && this.size[1] === h && this.size[2] === ss) return;
    if (this.fbo) for (const t of Object.values(this.fbo)) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); if (t.rb) gl.deleteRenderbuffer(t.rb); }
    const part = (n, k) => Math.max(2, Math.round(n / k));
    this.fbo = { scene: this._target(Math.round(w * ss), Math.round(h * ss), true), q1: this._target(part(w, 4), part(h, 4)), q2: this._target(part(w, 4), part(h, 4)), e1: this._target(part(w, 8), part(h, 8)), e2: this._target(part(w, 8), part(h, 8)) };
    this.size = [w, h, ss];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // the sphere's buffers: per face corner its place, the face's normal, its place on the face; per port a point; per seam a strip
  _build(f) {
    const gl = this.gl, S = buildSphere(f); S.f = f;
    const nF = S.n * 20, cell = new Float32Array(nF * 3 * 9);
    S.faceAt = new Float32Array(nF * 3); S.faceDir = new Float32Array(nF * 3);
    let o = 0;
    for (let c = 0; c < S.n; c++) for (let k = 0; k < 20; k++) {
      const [i0, i1, i2] = S.faces[k], P = [i0, i1, i2].map((i) => [S.corners[(c * 12 + i) * 3], S.corners[(c * 12 + i) * 3 + 1], S.corners[(c * 12 + i) * 3 + 2]]);
      const nrm = norm(cross([P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]], [P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]]));
      const mid = [0, 1, 2].map((q) => (P[0][q] + P[1][q] + P[2][q]) / 3), dir = norm(mid), fi = c * 20 + k, seed = (fi * 0.618034) % 1;
      // for the light program a face stands further from its cell's middle than it does: a spot that passes lights the
      // near faces first, so the colour runs through a cell face by face
      S.faceAt.set([0, 1, 2].map((q) => S.pos[c][q] + (mid[q] - S.pos[c][q]) * 2.4), fi * 3); S.faceDir.set(dir, fi * 3);
      for (let v = 0; v < 3; v++) { cell.set(P[v], o); cell.set(nrm, o + 3); cell.set([v === 0 ? 1 : 0, v === 1 ? 1 : 0, seed], o + 6); o += 9; }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.cell); gl.bufferData(gl.ARRAY_BUFFER, cell, gl.STATIC_DRAW);
    S.light = new Float32Array(nF * 3 * 6);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.light); gl.bufferData(gl.ARRAY_BUFFER, S.light, gl.DYNAMIC_DRAW);
    const port = new Float32Array(S.n * 12 * 3);
    for (let c = 0; c < S.n; c++) for (let i = 0; i < 12; i++) for (let q = 0; q < 3; q++) port[(c * 12 + i) * 3 + q] = S.pos[c][q] + (S.corners[(c * 12 + i) * 3 + q] - S.pos[c][q]) * 1.03;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.port); gl.bufferData(gl.ARRAY_BUFFER, port, gl.STATIC_DRAW);
    S.portCol = new Float32Array(S.n * 12 * 3);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.portCol); gl.bufferData(gl.ARRAY_BUFFER, S.portCol, gl.DYNAMIC_DRAW);
    const seam = new Float32Array(S.seams.length * 6 * 8), ends = [[0, -1], [1, -1], [0, 1], [0, 1], [1, -1], [1, 1]];
    S.seamOf = new Map();
    S.seams.forEach(([a, b, pa, pb], s) => {
      S.seamOf.set(a * S.n + b, s);
      for (let v = 0; v < 6; v++) { const at = (s * 6 + v) * 8; for (let q = 0; q < 3; q++) { seam[at + q] = S.corners[(a * 12 + pa) * 3 + q]; seam[at + 3 + q] = S.corners[(b * 12 + pb) * 3 + q]; } seam[at + 6] = ends[v][0]; seam[at + 7] = ends[v][1]; }
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.seam); gl.bufferData(gl.ARRAY_BUFFER, seam, gl.STATIC_DRAW);
    S.seamCol = new Float32Array(S.seams.length * 6 * 8); S.seamPulse = new Float32Array(S.seams.length * 2);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.seamCol); gl.bufferData(gl.ARRAY_BUFFER, S.seamCol, gl.DYNAMIC_DRAW);
    // the light program's memory: per line and cell the afterglow and its colour; per cell its colour now
    S.glowI = [0, 1, 2].map(() => new Float32Array(S.n)); S.glowC = [0, 1, 2].map(() => new Float32Array(S.n * 3));
    S.cellCol = new Float32Array(S.n * 3); S.near = new Float32Array(S.n);
    this.S = S; this.heads.length = 0; this.rings.length = 0;
  }

  // ---------- the light program ----------
  _colour(line, note) {
    const L = LINES[line], pc = ((note.midi % 12) + 12) % 12, fifths = (pc * 7) % 12, oct = clamp((note.midi - (line === 0 ? 30 : line === 1 ? 62 : 56)) / 24, -0.5, 0.5);
    const c = hsl(L.hue + (fifths / 11 - 0.5) * 2 * L.span, L.sat, L.light + 0.1 * oct);
    return note.accent ? c.map((x) => x + (1 - x) * 0.22) : c;
  }
  // the way from a cell to a pitch cell, over the neighbour graph
  _way(from, pole) { const S = this.S, out = [from]; let c = from; for (let i = 0; i < 64 && c !== pole; i++) { c = S.next[pole][c]; if (c < 0) break; out.push(c); } return out; }
  // send a spot toward the place of a coming note: it leaves now and, where its line's speed allows, arrives as that note starts
  _send(h, note, from) {
    const pole = POLE_OF_PC[((note.midi % 12) + 12) % 12], path = this._way(h.cell, pole);
    if (path.length > 1) h.moves.push({ path, t0: from, t1: Math.max(note.t, from + 0.01), ease: LINES[h.line].ease });
  }
  _note(line, note, nextNote) {
    const L = LINES[line], S = this.S, pc = ((note.midi % 12) + 12) % 12, pole = POLE_OF_PC[pc], col = this._colour(line, note), level = (0.55 + 0.45 * note.vel) * (note.accent ? 1.4 : 1);
    let h = null;
    if (note.slide) for (const x of this.heads) if (x.line === line && !x.gone && note.t <= x.end + L.release * 1.5 && (!h || x.t0 > h.t0)) h = x;       // a slide: the line's spot is dragged, no new one is lit
    if (h) {
      const path = this._way(h.cell, pole), glide = clamp([0.05, 0.03, 0.14][line] * (path.length - 1), 0.03, [0.2, 0.14, 0.9][line]);       // the air is dragged slowly, the lead at once
      h.moves = path.length > 1 ? [{ path, t0: note.t, t1: note.t + glide, ease: 1, drag: true }] : [];
      h.from = h.col.slice(); h.to = col; h.turnAt = note.t; h.turnFor = glide;
      h.end = note.t + note.dur; h.level = Math.max(h.level * 0.92, level); h.waiting = false; h.dragged = true;
    } else {
      for (const x of this.heads) if (x.line === line && x.end > note.t) x.end = Math.max(x.t0, note.t);       // the line's last note ends where the new one begins
      h = { line, t0: note.t, end: note.t + note.dur, level, col, from: col, to: col, turnAt: 0, turnFor: 0, cell: pole, moves: [], p: S.pos[pole].slice(), seg: null, I: 0, waiting: false, gone: false };
      this.heads.push(h);
      const ring = L.ring > 0 ? L.ring : note.accent ? 0.42 : 0;
      if (ring > 0) this.rings.push({ line, dir: S.pos[pole], t0: note.t, col, strength: ring * level * (note.accent && L.ring > 0 ? 2.2 : 1) });
    }
    // where it goes from here: toward the line's next note, if that note is known and is not itself a slide (a slide will drag it anyway)
    const after = h.moves.length ? h.moves[h.moves.length - 1].t1 : note.t;
    h.cellAfter = pole;
    if (nextNote && !nextNote.slide) { const keep = h.cell; h.cell = pole; this._send(h, nextNote, after); h.cell = keep; }
    else if (!nextNote) h.waiting = true;
  }
  _lights(pos, dBars) {
    const S = this.S, n = S.n, heads = this.heads, calm = this.calm;
    // the notes whose time has come
    if (this.posWas === null || pos < this.posWas || pos - this.posWas > 2) { this.posWas = pos - 1e-6; }
    for (let b = Math.floor(this.posWas); b <= Math.floor(pos); b++) {
      const sc = this.score.get(b); if (!sc) continue;
      for (let l = 0; l < 3; l++) { const list = sc.lines[l]; for (let i = 0; i < list.length; i++) { const nt = list[i]; if (nt.t > this.posWas && nt.t <= pos) { const up = this.score.get(b + 1); this._note(l, nt, list[i + 1] || (up && up.lines[l][0]) || null); } } }
    }
    this.posWas = pos;
    // every spot: how bright, which colour, where
    for (const h of heads) {
      const L = LINES[h.line]; let env;
      if (pos < h.t0 + L.attack) env = Math.max(0, (pos - h.t0) / L.attack);
      else if (pos <= h.end) env = 1 - 0.2 * clamp((pos - h.t0) / Math.max(h.end - h.t0, 0.06), 0, 1);
      else env = 0.8 * Math.exp(-(pos - h.end) / L.release);
      h.I = h.level * env * L.gain;
      if (pos > h.end && env < 0.012) h.gone = true;
      if (h.turnFor > 0) { const k = clamp((pos - h.turnAt) / h.turnFor, 0, 1); for (let q = 0; q < 3; q++) h.col[q] = h.from[q] + (h.to[q] - h.from[q]) * k; }
      let mv = null; for (const m of h.moves) { if (pos >= m.t0) mv = m; }
      h.seg = null;
      if (mv) {
        const hops = mv.path.length - 1, f = Math.pow(clamp((pos - mv.t0) / Math.max(mv.t1 - mv.t0, 1e-4), 0, 1), mv.ease);
        const most = mv.drag ? hops : ((pos - mv.t0) * L.speed * (calm ? 0.6 : 1)) / S.hop;        // a dragged spot keeps up with its note; a wandering one is no faster than its line
        const g = Math.min(hops * f, most, hops - 1e-4), i = Math.floor(g), fr = g - i;
        const A = S.pos[mv.path[i]], B = S.pos[mv.path[i + 1]], p = norm([A[0] + (B[0] - A[0]) * fr, A[1] + (B[1] - A[1]) * fr, A[2] + (B[2] - A[2]) * fr]);
        h.p = p; h.cell = fr < 0.5 ? mv.path[i] : mv.path[i + 1]; h.seg = [mv.path[i], mv.path[i + 1], fr];
      }
    }
    this.heads = heads.filter((h) => !h.gone);
    this.rings = this.rings.filter((r) => pos - r.t0 < LINES[r.line].ringLife * 3.5);
    for (const r of this.rings) { const L = LINES[r.line], age = pos - r.t0; r.rho = age * L.ringSpeed * (calm ? 0.55 : 1); r.I = r.strength * Math.exp(-age / L.ringLife) * (calm ? 0.6 : 1) * Math.min(1, age / 0.01); }
    // the afterglow: a cell keeps the light of the spots that passed, per line
    const hd = this.heads, nh = hd.length;
    for (let l = 0; l < 3; l++) {
      const gI = S.glowI[l], gC = S.glowC[l], decay = Math.exp(-dBars / LINES[l].glow), inv = 1 / (2 * LINES[l].sigma * LINES[l].sigma);
      for (let c = 0; c < n; c++) {
        const P = S.pos[c]; let best = 0, bh = null;
        for (let k = 0; k < nh; k++) { const h = hd[k]; if (h.line !== l || h.I < 0.004) continue; const dx = P[0] - h.p[0], dy = P[1] - h.p[1], dz = P[2] - h.p[2], v = h.I * Math.exp(-(dx * dx + dy * dy + dz * dz) * inv); if (v > best) { best = v; bh = h; } }
        const g = gI[c] * decay;
        if (bh && best >= g * 0.7) { const k = best >= g ? 0.5 : 0.15; for (let q = 0; q < 3; q++) gC[c * 3 + q] += (bh.col[q] - gC[c * 3 + q]) * k; }
        gI[c] = Math.max(g, best);
      }
    }
    // the tide: a dim wash in the page's cyan that travels round the sphere once in sixteen bars
    const ta = (pos / 16) * TAU, tide = [Math.cos(ta) * 0.94, 0.34 * Math.sin(ta * 2), Math.sin(ta) * 0.94];
    // every face: the three lines at its place, mixed; then the cell's colour, the ports, the light it spills
    const light = S.light, cellCol = S.cellCol, I = [0, 0, 0], C = [0, 0, 0, 0, 0, 0, 0, 0, 0], rings = this.rings, nr = rings.length;
    for (let c = 0; c < n; c++) {
      let cr = 0, cg = 0, cb = 0;
      for (let k = 0; k < 20; k++) {
        const fi = c * 20 + k, fx = S.faceAt[fi * 3], fy = S.faceAt[fi * 3 + 1], fz = S.faceAt[fi * 3 + 2], ux = S.faceDir[fi * 3], uy = S.faceDir[fi * 3 + 1], uz = S.faceDir[fi * 3 + 2];
        for (let l = 0; l < 3; l++) { I[l] = S.glowI[l][c] * 0.7; C[l * 3] = S.glowC[l][c * 3]; C[l * 3 + 1] = S.glowC[l][c * 3 + 1]; C[l * 3 + 2] = S.glowC[l][c * 3 + 2]; }
        for (let j = 0; j < nh; j++) {
          const h = hd[j]; if (h.I < 0.004) continue;
          const l = h.line, s = LINES[l].sigma, dx = fx - h.p[0], dy = fy - h.p[1], dz = fz - h.p[2], v = h.I * Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * s * s));
          if (v > I[l]) { I[l] = v; C[l * 3] = h.col[0]; C[l * 3 + 1] = h.col[1]; C[l * 3 + 2] = h.col[2]; }
        }
        for (let j = 0; j < nr; j++) {
          const r = rings[j]; if (r.I < 0.01) continue;
          const l = r.line, ang = Math.acos(clamp(ux * r.dir[0] + uy * r.dir[1] + uz * r.dir[2], -1, 1)), d = (ang - r.rho) / LINES[l].ringWidth, v = r.I * Math.exp(-d * d);
          if (v > I[l]) { const k = I[l] / (v + 1e-6); C[l * 3] = r.col[0] + (C[l * 3] - r.col[0]) * k * 0.6; C[l * 3 + 1] = r.col[1] + (C[l * 3 + 1] - r.col[1]) * k * 0.6; C[l * 3 + 2] = r.col[2] + (C[l * 3 + 2] - r.col[2]) * k * 0.6; I[l] = v; }
        }
        // where lines meet the strongest sets the hue and the others tint it; the brightness is bounded
        const w0 = I[0] * I[0] * I[0], w1 = I[1] * I[1] * I[1], w2 = I[2] * I[2] * I[2], ws = w0 + w1 + w2 + 1e-6;
        const top = Math.max(I[0], I[1], I[2]), T = Math.min(1.2, top + 0.2 * (I[0] + I[1] + I[2] - top));
        const td = 0.07 + 0.13 * Math.max(0, ux * tide[0] + uy * tide[1] + uz * tide[2]) ** 2, quiet = 1 - Math.min(1, T * 3.6);
        let r = (C[0] * w0 + C[3] * w1 + C[6] * w2) / ws, g = (C[1] * w0 + C[4] * w1 + C[7] * w2) / ws, b = (C[2] * w0 + C[5] * w1 + C[8] * w2) / ws;
        const grey = 0.3 * r + 0.5 * g + 0.2 * b, E = T * (0.55 + 0.75 * T);                 // a mixed colour keeps its strength of hue; a bright spot is brighter than twice a dim one
        r = Math.max(0, grey + (r - grey) * 1.3) * E + 0.2 * td * (0.12 + 0.88 * quiet); g = Math.max(0, grey + (g - grey) * 1.3) * E + 0.86 * td * (0.12 + 0.88 * quiet); b = Math.max(0, grey + (b - grey) * 1.3) * E + 0.9 * td * (0.12 + 0.88 * quiet);
        for (let v = 0; v < 3; v++) { const o = (fi * 3 + v) * 6; light[o] = r; light[o + 1] = g; light[o + 2] = b; }
        cr += r; cg += g; cb += b;
      }
      cellCol[c * 3] = cr / 20; cellCol[c * 3 + 1] = cg / 20; cellCol[c * 3 + 2] = cb / 20;
    }
    let ar = 0, ag = 0, ab = 0;
    const spark = calm ? 0 : this.spark, seed = this.sparkSeed;
    for (let c = 0; c < n; c++) {
      let sr = 0, sg = 0, sb = 0; const nb = S.nb[c];
      for (const k of nb) { sr += cellCol[k * 3]; sg += cellCol[k * 3 + 1]; sb += cellCol[k * 3 + 2]; }
      sr /= nb.length; sg /= nb.length; sb /= nb.length;
      for (let v = 0; v < 60; v++) { const o = (c * 60 + v) * 6 + 3; light[o] = sr; light[o + 1] = sg; light[o + 2] = sb; }
      for (let i = 0; i < 12; i++) {
        const o = (c * 12 + i) * 3, tw = spark > 0.02 ? spark * (((Math.sin((c * 12 + i) * 12.9898 + seed * 78.233) * 43758.5453) % 1 + 1) % 1 > 0.82 ? 1 : 0) : 0;
        S.portCol[o] = cellCol[c * 3] * 0.85 + 0.05 + tw * 0.7; S.portCol[o + 1] = cellCol[c * 3 + 1] * 0.85 + 0.06 + tw * 0.75; S.portCol[o + 2] = cellCol[c * 3 + 2] * 0.85 + 0.06 + tw * 0.75;
      }
      ar += cellCol[c * 3]; ag += cellCol[c * 3 + 1]; ab += cellCol[c * 3 + 2];
    }
    this.avg = [ar / n, ag / n, ab / n];
    // the seams: each in the colours of its two cells; a spot on its way from one cell to the next lights the seam between them
    const pulse = S.seamPulse; pulse.fill(0);
    for (const h of this.heads) if (h.seg && h.I > 0.03) {
      const [a, b, fr] = h.seg, s = S.seamOf.get(a < b ? a * n + b : b * n + a); if (s === undefined) continue;
      const along = (fr - 0.32) / 0.36;                       // the seam spans the middle third of the way from cell to cell
      if (along > -0.6 && along < 1.6) { pulse[s * 2] = a < b ? along : 1 - along; pulse[s * 2 + 1] = Math.max(pulse[s * 2 + 1], Math.min(1.2, h.I * 1.3)); }
    }
    const sc = S.seamCol;
    for (let s = 0; s < S.seams.length; s++) {
      const a = S.seams[s][0], b = S.seams[s][1];
      for (let v = 0; v < 6; v++) { const o = (s * 6 + v) * 8; sc[o] = cellCol[a * 3]; sc[o + 1] = cellCol[a * 3 + 1]; sc[o + 2] = cellCol[a * 3 + 2]; sc[o + 3] = cellCol[b * 3]; sc[o + 4] = cellCol[b * 3 + 1]; sc[o + 5] = cellCol[b * 3 + 2]; sc[o + 6] = pulse[s * 2]; sc[o + 7] = pulse[s * 2 + 1]; }
    }
  }

  // every full-picture pulse goes through here: never two within 0.36 s (under three a second), none in calm mode
  _pulse(amount) {
    const now = performance.now() / 1000;
    if (this.calm || Math.abs(now - this.lastFlash) < 0.36) return false;
    this.lastFlash = now; this.kick = Math.max(this.kick, amount);
    return true;
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
    if (!this.fbo || !this.S) this.resize();
    const S = this.S, F = this.fbo, W = c.width, H = c.height, calm = this.calm, aspect = W / Math.max(1, H);
    const pos = this._pos(this._now()), dBars = this.pos2 === undefined ? 0 : clamp(pos - this.pos2, 0, 0.5);
    this.pos2 = pos; this.pos = pos;
    // the drums: a kick is a breath of the whole sculpture, the hats make ports twinkle
    const sc = this.score.get(Math.floor(pos)), step = Math.floor((pos % 1) * 16) + Math.floor(pos) * 16;
    if (sc && step !== this.stepWas) { this.stepWas = step; const t = ((step % 16) + 16) % 16; if (sc.kicks.includes(t)) this._pulse(1); if (sc.hats.includes(t)) { this.spark = 1; this.sparkSeed = step; } }
    this.kick *= Math.exp(-dt * 6.5); this.spark *= Math.exp(-dt * 14);
    if (this.analyser && this.spec) { this.analyser.getByteFrequencyData(this.spec); let s = 0; for (let i = 0; i < this.spec.length; i += 4) s += this.spec[i]; this.level += ((0.85 + 0.4 * clamp((s / (this.spec.length / 4) / 255) * 2.2, 0, 1)) - this.level) * 0.1; } else this.level += (1 - this.level) * 0.1;
    this._lights(pos, dBars);
    // the camera: the sphere fits the canvas, a little above its middle; on an upright canvas it fills the width
    this.turn += dt * (calm ? 0.045 : 0.075);
    const turn = this.view.turn === null ? this.turn : this.view.turn, fov = 0.56, tanF = Math.tan(fov / 2), outer = 1 + S.rc * 1.05;
    const tall = aspect < 0.9, shift = 0.1, fit = Math.min(0.78, 0.93 * aspect) * this.view.zoom;
    const dist = outer / Math.sin(Math.atan(Math.min(fit * tanF, 5))), eye = [0, 0.42, Math.sqrt(Math.max(0.1, dist * dist - 0.42 * 0.42))];
    const view = M.look(eye, [0, 0, 0], [0, 1, 0]), proj = M.persp(fov, aspect, 0.3, 30); proj[9] = -shift;
    const vp = M.mul(proj, view), model = M.mul(M.rotZ(0.14), M.mul(M.rotX(0.3), M.rotY(turn)));
    const ss = this.size[2], lens = [(F.scene.h / 2) / tanF, 0.3035 * S.rc, dist - 0.55, dist + 1.25];
    const fwd = norm([-eye[0], -eye[1], -eye[2]]), right = norm(cross(fwd, [0, 1, 0])), up = cross(right, fwd), floor = -(outer + 0.34);
    gl.bindFramebuffer(gl.FRAMEBUFFER, F.scene.fb); gl.viewport(0, 0, F.scene.w, F.scene.h);
    gl.clearColor(0, 0, 0, 1); gl.clearDepth(1); gl.depthMask(true); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const attribs = (p, list) => { for (let i = 0; i < 8; i++) { if (i < p.attrs) gl.enableVertexAttribArray(i); else gl.disableVertexAttribArray(i); } for (const [buffer, loc, size, stride, off] of list) { gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, off); } };
    // the room
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE);
    let p = P.room; gl.useProgram(p.p); attribs(p, [[this.buf.quad, 0, 2, 0, 0]]);
    gl.uniform3fv(p.u.uEye, eye); gl.uniform3fv(p.u.uF, fwd); gl.uniform3fv(p.u.uR, right); gl.uniform3fv(p.u.uU, up); gl.uniform3fv(p.u.uAvg, this.avg || [0, 0, 0]);
    gl.uniform4f(p.u.uCam, tanF * aspect, tanF, shift, floor);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // the lamps
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK); gl.frontFace(gl.CCW);
    p = P.cell; gl.useProgram(p.p);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.light); gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.light);
    attribs(p, [[this.buf.cell, 0, 3, 36, 0], [this.buf.cell, 1, 3, 36, 12], [this.buf.cell, 2, 3, 36, 24], [this.buf.light, 3, 3, 24, 0], [this.buf.light, 4, 3, 24, 12]]);
    gl.uniformMatrix4fv(p.u.uVP, false, vp); gl.uniformMatrix4fv(p.u.uModel, false, model); gl.uniform3fv(p.u.uEye, eye); gl.uniform4fv(p.u.uLens, lens);
    gl.drawArrays(gl.TRIANGLES, 0, S.n * 60);
    // the seams and the ports: light, added to what is there, hidden by what is in front
    gl.disable(gl.CULL_FACE); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    p = P.seam; gl.useProgram(p.p);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.seamCol); gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.seamCol);
    attribs(p, [[this.buf.seam, 0, 3, 32, 0], [this.buf.seam, 1, 3, 32, 12], [this.buf.seam, 2, 2, 32, 24], [this.buf.seamCol, 3, 3, 32, 0], [this.buf.seamCol, 4, 3, 32, 12], [this.buf.seamCol, 5, 2, 32, 24]]);
    gl.uniformMatrix4fv(p.u.uVP, false, vp); gl.uniformMatrix4fv(p.u.uModel, false, model); gl.uniform3fv(p.u.uEye, eye); gl.uniform4fv(p.u.uLens, lens);
    gl.uniform2f(p.u.uRes, F.scene.w, F.scene.h); gl.uniform1f(p.u.uWidth, S.rc * 0.075);
    gl.drawArrays(gl.TRIANGLES, 0, S.seams.length * 6);
    p = P.port; gl.useProgram(p.p);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.portCol); gl.bufferSubData(gl.ARRAY_BUFFER, 0, S.portCol);
    attribs(p, [[this.buf.port, 0, 3, 0, 0], [this.buf.portCol, 1, 3, 0, 0]]);
    gl.uniformMatrix4fv(p.u.uVP, false, vp); gl.uniformMatrix4fv(p.u.uModel, false, model); gl.uniform3fv(p.u.uEye, eye); gl.uniform4fv(p.u.uLens, lens); gl.uniform1f(p.u.uSize, S.rc * 0.34);
    gl.drawArrays(gl.POINTS, 0, S.n * 12);
    gl.depthMask(true); gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    // the glow round the lamps and the haze in the air
    p = P.blur; gl.useProgram(p.p); attribs(p, [[this.buf.quad, 0, 2, 0, 0]]); gl.uniform1i(p.u.uTex, 0); gl.activeTexture(gl.TEXTURE0);
    const pass = (src, dst, dx, dy) => { gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb); gl.viewport(0, 0, dst.w, dst.h); gl.bindTexture(gl.TEXTURE_2D, src.tex); gl.uniform2f(p.u.uDir, dx / dst.w, dy / dst.h); gl.drawArrays(gl.TRIANGLES, 0, 6); };
    pass(F.scene, F.q1, 1.2, 0); pass(F.q1, F.q2, 0, 1.2); pass(F.q2, F.q1, 2.2, 0); pass(F.q1, F.q2, 0, 2.2);
    pass(F.q2, F.e1, 1.6, 0); pass(F.e1, F.e2, 0, 1.6); pass(F.e2, F.e1, 3.2, 0); pass(F.e1, F.e2, 0, 3.2);
    // to the screen
    p = P.out; gl.useProgram(p.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, F.scene.tex); gl.uniform1i(p.u.uScene, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, F.q2.tex); gl.uniform1i(p.u.uBloom, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, F.e2.tex); gl.uniform1i(p.u.uHaze, 2);
    gl.uniform2f(p.u.uTexel, 1 / W, 1 / H);
    gl.uniform4f(p.u.uMix, 0.75, 0.62, 0, 1.2 * this.level * (1 + (calm ? 0 : 0.13 * this.kick)));
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.activeTexture(gl.TEXTURE0);
  }
}

export default ScreenStage;
