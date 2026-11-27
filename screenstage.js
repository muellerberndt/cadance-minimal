// Cadance radio, the screen stage: a light sculpture. A sphere of small icosahedral lamps turns slowly in a dark
// room; the three synth lines of the track that plays write its colours and the drums strike it.
//
// The object. The cells stand on a geodesic sphere (92 of them, 42 on a small canvas): twelve cells have five
// neighbours, all others six. Every cell is an icosahedron: twelve ports at its corners, thirty edges, twenty faces.
// Neighbouring cells are joined port to port by a seam, and a port carries at most one seam. A cell is a lamp:
// twenty frosted panels lit from inside, a dark frame on the edges, a small bright point at every port. Light
// spills from a cell onto its neighbours' frames and glass, runs along the seams, glows in the air around the
// sphere and lies on the floor under it.
//
// The light program, the synths. One rule for all three lines, with a few numbers per line:
//   a note lights a spot at the cell of its pitch class. The twelve pitch classes stand as the twelve corners of an
//   icosahedron (a tritone is the opposite side, a ring of five a chain of fifths). Each line holds this figure
//   in a turn of its own, and at every bar line it turns the figure by up to a radian toward the part of the
//   sphere that has had the least light, counted over the notes the bar is going to play; within a bar a pitch
//   keeps its cell;
//   every cell knows how much it wants light: how long it has been dark, and how little light it has had in the
//   last bars against the others. Between two notes a spot does not rest and does not take the shortest way: it
//   wanders over the neighbour graph by way of the cell that wants light most among those it can reach in time,
//   and arrives at the next note's cell as that note starts. On its way it stays lit, so the eye can follow it;
//   a held note keeps its spot alive; a note marked as a slide makes no new spot: the line's spot is dragged to
//   the new place; an accent is brighter and sends a ring over the sphere;
//   every cell remembers the light that passed (an afterglow), so a moving spot draws a tail.
//   The bass is slow and wide and sends a ring with every note, the lead is fast and narrow, the air is a wide
//   wash that drifts round the sphere. Pitch class picks the hue inside the line's family (bass blue to magenta,
//   lead green to sky, air rose to peach), the octave its lightness. Where lines meet, the stronger one sets the
//   hue and the others tint it; brightness is bounded, so the sphere is never white all over. A dim tide in the
//   page's cyan keeps it from going dark.
//
// The light program, the drums. The bar's sixteen steps lie evenly from left to right across the side of the sphere
// that faces the room, and once more across the far side, whatever the sphere's turn. A thin line of light is the
// playhead: it crosses the near side once a bar. The kick lives on a low ring, the snare on a ring above the
// middle. From the bar line on, the bar's pattern stands there to read: a warm panel and warm ports where a kick
// falls, a short cold line of them where a snare falls. When the playhead reaches a hit, the panels round it
// light: a kick is a warm pulse that spreads a short way, a snare a sharp cold streak (an accent taller and
// brighter). The hats make ports
// sparkle round the playhead. A kick is also a small breath of the whole sculpture (7 percent, never two within
// 0.36 s); all other drum light is local.
//
// Calm mode: no breath, no strikes, no playhead and no sparkle (the bar's hits are steady dim marks), slower
// spots and rings, a slower turn. When nobody plays (no bar yet, or the clock stands for more than a bar), three
// slow lights of the three families wander by themselves.
//
// Read from the bar: tokens (which line sounds), patterns (the drum hits with their levels; per line the notes with
// pitch, velocity, length, slide and accent). Not read: mods, patches, view. Derived, and said so here: everything
// else. Which cell is which pitch class and how a line turns that figure, the ways the spots take, their widths, speeds
// and colours, the afterglow, the rings, the tide, where the steps and the drum voices lie, the shapes and colours
// of the strikes, the playhead, the breath, the sparkle and the idle lights are this module's own program; the
// track's data gives the notes and the hits only.
//
// One ES module, no dependency, no request. WebGL 1 or 2; without it `ok` is false and nothing is drawn.
//
//   const screen = new ScreenStage(canvas, { calm: false });
//   screen.ok                                   false: no WebGL, the page falls back
//   screen.setClock({ ctx, bpm, barZeroTime }); bar k starts at barZeroTime + k * 240 / bpm; without ctx: performance.now()
//   screen.setBar(index, bar);                  the bar that sounds now (tokens, patterns)
//   screen.setBar(index, bar, nextBar);         the same with the bar after it, if the page knows it: the spots then travel across the bar line
//   screen.setAudio(analyser);                  the master analyser, optional: it scales the overall level a little
//   screen.setCalm(on); screen.resize(); screen.start(); screen.stop(); screen.stats();
//   also: screen.setView({ turn, zoom })        for stills and close-ups: a fixed turn in radians (null: it turns by itself), a zoom (1: the sphere fits)

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const TAU = Math.PI * 2;
// per line: the hue family (degrees, centre and half span), saturation, lightness, the spot's width (radians on the
// unit sphere), its fastest travel (radians a bar), attack and release (bars), the afterglow's time (bars), the
// ring a note sends (strength, speed in radians a bar, width, life in bars), gain, how the travel is eased, how
// bright a spot stays on its way (a share of its note's level)
const LINES = [
  { hue: 262, span: 52, sat: 0.96, light: 0.5, sigma: 0.31, speed: 3.6, attack: 0.02, release: 0.2, glow: 0.26, ring: 0.2, ringSpeed: 3.6, ringWidth: 0.18, ringLife: 0.28, gain: 0.9, ease: 1, floor: 0.72 },
  { hue: 168, span: 38, sat: 0.95, light: 0.5, sigma: 0.17, speed: 16, attack: 0.008, release: 0.24, glow: 0.22, ring: 0, ringSpeed: 9, ringWidth: 0.15, ringLife: 0.14, gain: 1.3, ease: 1.1, floor: 0.62 },
  { hue: 6, span: 20, sat: 0.92, light: 0.58, sigma: 0.6, speed: 2.4, attack: 0.14, release: 0.6, glow: 0.45, ring: 0, ringSpeed: 2, ringWidth: 0.4, ringLife: 0.6, gain: 0.6, ease: 1, floor: 0.8 },
];
// the drums' places and colours: the kick's ring below the middle, the snare's above it (latitudes in radians)
const KICK_LAT = -0.32, SNARE_LAT = 0.5, KICK_RGB = [1.0, 0.8, 0.55], SNARE_RGB = [0.72, 0.88, 1.0];
// where a moment of the bar (0 to 1) lies on the side that faces the room: evenly from left to right as the room
// sees it (an angle round the sphere's axis, in the room's frame; the far side holds the same again, half a turn on)
const stepLon = (phase) => Math.acos(clamp(-0.94 + 1.88 * phase, -1, 1));
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

// a rotation as nine numbers, row by row: about an axis by an angle; one after another; applied to a direction
function turnAbout(a, ang) {
  const c = Math.cos(ang), s = Math.sin(ang), t = 1 - c, [x, y, z] = a;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}
function turnAfter(A, B) {                       // first B, then A; made true again, so that many turns in a row stay a rotation
  const m = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) m[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  const r0 = norm([m[0], m[1], m[2]]), k = dot(r0, [m[3], m[4], m[5]]), r1 = norm([m[3] - k * r0[0], m[4] - k * r0[1], m[5] - k * r0[2]]), r2 = cross(r0, r1);
  return [...r0, ...r1, ...r2];
}
// how each line's figure of the twelve pitch classes stands at first, and the turns a line may give it at a bar line
// (none, or half a radian or one radian about one of six axes, either way)
const FIGURE0 = [turnAbout(norm([0.42, 0.86, 0.29]), 0.4), turnAbout(norm([-0.55, 0.52, 0.65]), 2.1), turnAbout(norm([0.7, 0.15, -0.7]), 4.0)];
const STEPS = (() => { const out = [{ m: null, cost: 0 }]; for (const a of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]) for (const ang of [0.5, -0.5, 1, -1]) out.push({ m: turnAbout(norm(a), ang), cost: 0.05 * Math.abs(ang) }); return out; })();

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
// neighbour graph between any two cells
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
  // the hops between any two cells over the neighbour graph
  const D = new Uint8Array(n * n).fill(255);
  for (let a = 0; a < n; a++) { const queue = [a]; D[a * n + a] = 0; for (let h = 0; h < queue.length; h++) for (const k of nb[queue[h]]) if (D[a * n + k] === 255) { D[a * n + k] = D[a * n + queue[h]] + 1; queue.push(k); } }
  return { n, pos, nb, rc, hop, corners, faces: ico.faces, seams, D };
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
    this.heads = []; this.rings = []; this.idlers = []; this.kick = 0; this.lastFlash = -9; this.spark = 0; this.sparkSeed = 0; this.sparkRate = 14; this.stepWas = -1;
    this.kb = new Float32Array(24 * 8); this.sb = new Float32Array(24 * 4); this.nk = 0; this.ns = 0; this.idleK = 1; this.movedAt = -1e9; this.lt = 0;
    this.figure = [new Map(), new Map(), new Map()]; this.figureNow = FIGURE0.map((q) => q.slice());
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

  // the bar that sounds now: its three lines' notes and its drum hits. `next`, if the page knows it, is the bar after
  // it: the spots then travel across the bar line toward its first notes
  setBar(index, bar, next) {
    if (!bar) return;
    if (index < this.curIndex - 1 || index > this.curIndex + 8) { this.score.clear(); this.heads.length = 0; this.rings.length = 0; this.posWas = null; for (const m of this.figure) m.clear(); }       // another track, or a jump
    const same = index === this.curIndex;
    this.score.set(index, this._read(index, bar));
    if (next) this.score.set(index + 1, this._read(index + 1, next));
    for (const k of this.score.keys()) if (k < index - 2) this.score.delete(k);
    if (!same) { this.curIndex = index; this.barAt = this._now(); }
    // a spot that wandered without a goal now knows its line's next note
    const lines = this.score.get(index).lines;
    for (const h of this.heads) if (h.waiting && lines[h.line].length) {
      const nx = lines[h.line][0];
      if (nx.t > h.t0 && !nx.slide && this.S) { const now = Math.max(this.pos, h.t0); h.moves = h.moves.filter((m) => m.t0 < now); this._send(h, h.cell, nx, now); }
      h.waiting = false;
    }
  }
  _read(index, bar) {
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
    const row = (...names) => { const out = new Float32Array(16); if (hits) for (const k of names) if (hits[k]) for (let t = 0; t < 16; t++) out[t] = Math.max(out[t], clamp(+hits[k][t] || 0, 0, 1)); return out; };
    return { lines, kick: row('kick'), snare: row('snare', 'clap'), hat: row('ch', 'oh', 'cym'), open: row('oh', 'cym') };
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
  stats() { return { frameMs: this.frameMs, scale: this.scale, webgl: !!this.gl, cells: this.S ? this.S.n : 0, seams: this.S ? this.S.seams.length : 0, spots: this.heads.length, rings: this.rings.length, bar: this.pos, idle: this.idleK > 0.5 }; }

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
    S.cellCol = new Float32Array(S.n * 3);
    // for the drums: where a face and a port lie round the sphere's axis; for the wandering: how long a cell has been dark (bars)
    S.faceLon = new Float32Array(nF); S.faceLat = new Float32Array(nF); S.faceCos = new Float32Array(nF);
    for (let i = 0; i < nF; i++) { const y = clamp(S.faceDir[i * 3 + 1], -1, 1); S.faceLat[i] = Math.asin(y); S.faceCos[i] = Math.sqrt(1 - y * y); S.faceLon[i] = Math.atan2(S.faceDir[i * 3 + 2], S.faceDir[i * 3]); }
    S.portLon = new Float32Array(S.n * 12); S.portCos = new Float32Array(S.n * 12); S.portLat = new Float32Array(S.n * 12); S.portDir = new Float32Array(S.n * 36);
    for (let i = 0; i < S.n * 12; i++) { const d = norm([port[i * 3], port[i * 3 + 1], port[i * 3 + 2]]); S.portDir.set(d, i * 3); S.portLon[i] = Math.atan2(d[2], d[0]); S.portCos[i] = Math.sqrt(1 - d[1] * d[1]); S.portLat[i] = Math.asin(clamp(d[1], -1, 1)); }
    S.dark = new Float32Array(S.n).fill(3); S.fed = new Float32Array(S.n); S.want = new Float32Array(S.n).fill(1);
    this.S = S; this.heads.length = 0; this.rings.length = 0; this.idlers.length = 0;
  }

  // ---------- the light program ----------
  _colour(line, note) {
    const L = LINES[line], pc = ((note.midi % 12) + 12) % 12, fifths = (pc * 7) % 12, oct = clamp((note.midi - (line === 0 ? 30 : line === 1 ? 62 : 56)) / 24, -0.5, 0.5);
    const c = hsl(L.hue + (fifths / 11 - 0.5) * 2 * L.span, L.sat, L.light + 0.1 * oct);
    return note.accent ? c.map((x) => x + (1 - x) * 0.22) : c;
  }
  // the cell where a pitch class stands when the figure of the twelve is held in the turn Q
  _cell(Q, midi) {
    const S = this.S, d = S.pos[POLE_OF_PC[((midi % 12) + 12) % 12]], x = Q[0] * d[0] + Q[1] * d[1] + Q[2] * d[2], y = Q[3] * d[0] + Q[4] * d[1] + Q[5] * d[2], z = Q[6] * d[0] + Q[7] * d[1] + Q[8] * d[2];
    let best = 0, top = -2;
    for (let i = 0; i < S.n; i++) { const P = S.pos[i], v = P[0] * x + P[1] * y + P[2] * z; if (v > top) { top = v; best = i; } }
    return best;
  }
  // a line's figure in a bar: the turn it had, turned at the bar line toward where the sphere wants light, over the notes of the bar
  _figure(line, bar) {
    const kept = this.figure[line]; let Q = kept.get(bar); if (Q) return Q;
    let from = -1; for (const k of kept.keys()) if (k < bar && k > from) from = k;
    Q = from >= 0 ? kept.get(from) : this.figureNow[line];
    const sc = this.score.get(bar); if (!sc) return Q;        // a bar that is not known yet: nothing is decided
    const notes = sc.lines[line], S = this.S;
    if (notes.length) {
      let best = Q, top = -9;
      for (const step of STEPS) {
        const R = step.m ? turnAfter(step.m, Q) : Q; let sum = 0, len = 0;
        for (const nt of notes) { const c = this._cell(R, nt.midi); let w = 0; for (const k of S.nb[c]) w += S.want[k]; sum += nt.dur * (0.5 * S.want[c] + (0.5 * w) / S.nb[c].length); len += nt.dur; }
        const score = sum / len - step.cost;
        if (score > top) { top = score; best = R; }
      }
      Q = best;
    }
    kept.set(bar, Q); this.figureNow[line] = Q;
    for (const k of kept.keys()) if (k < bar - 3) kept.delete(k);
    return Q;
  }
  // where a pitch class lives at a bar position
  _home(line, midi, pos) { return this._cell(this._figure(line, Math.floor(pos + 1e-6)), midi); }
  // how much a cell wants light, 0 to 1
  _hunger(c) { return this.S.want[c]; }
  // a shortest way from cell to cell that prefers the hungry cells (appended to `out`, without its first cell)
  _leg(from, to, out) {
    const S = this.S, n = S.n, D = S.D; let c = from;
    while (c !== to) {
      const need = D[c * n + to] - 1; let best = -1, top = -9;
      for (const k of S.nb[c]) if (D[k * n + to] === need) { const sc = this._hunger(k) + 0.3 * dot(S.pos[k], S.pos[to]); if (sc > top) { top = sc; best = k; } }
      if (best < 0) break;
      out.push(best); c = best;
    }
    return out;
  }
  // the way a spot takes in the hops it has time for: by way of the hungriest cell in reach (to < 0: no goal, it ends there)
  _route(from, to, budget) {
    const S = this.S, n = S.n, D = S.D, direct = to < 0 ? 0 : D[from * n + to]; let via = -1, top = to < 0 ? -1 : 0.2;
    if (budget > direct) for (let w = 0; w < n; w++) {
      const out = D[from * n + w]; if (w === from || out + (to < 0 ? 0 : D[w * n + to]) > budget) continue;
      let h = 0; for (const k of S.nb[w]) h += this._hunger(k);
      const sc = 0.55 * this._hunger(w) + (0.45 * h) / S.nb[w].length + 0.02 * out;
      if (sc > top) { top = sc; via = w; }
    }
    const path = [from];
    if (via >= 0 && via !== to) { this._leg(from, via, path); if (to >= 0) this._leg(via, to, path); } else if (to >= 0) this._leg(from, to, path);
    for (const c of path) S.dark[c] *= 0.4;                   // these cells are about to be lit: the next spot looks elsewhere
    return path;
  }
  _hops(line, bars) { return Math.floor((LINES[line].speed * (this.calm ? 0.6 : 1) * bars * 0.9) / this.S.hop); }
  // send a spot toward the place of a coming note: it leaves now and arrives as that note starts
  _send(h, start, note, t0) {
    const t1 = Math.max(note.t, t0 + 0.01), path = this._route(start, this._home(h.line, note.midi, note.t), this._hops(h.line, t1 - t0));
    if (path.length > 1) h.moves.push({ path, t0, t1, ease: LINES[h.line].ease });
  }
  // the line's next note is not known yet: the spot wanders until the bar line
  _wander(h, start, t0, until) {
    if (until - t0 < 0.03) return;
    const path = this._route(start, -1, this._hops(h.line, until - t0));
    if (path.length > 1) h.moves.push({ path, t0, t1: until, ease: 1 });
  }
  _note(line, note, nextNote) {
    const L = LINES[line], S = this.S, home = this._home(line, note.midi, note.t), col = this._colour(line, note), level = (0.55 + 0.45 * note.vel) * (note.accent ? 1.4 : 1);
    let h = null;
    if (note.slide) for (const x of this.heads) if (x.line === line && !x.gone && note.t <= x.end + L.release * 1.5 && (!h || x.t0 > h.t0)) h = x;       // a slide: the line's spot is dragged, no new one is lit
    if (h) {
      const path = this._leg(h.cell, home, [h.cell]), glide = clamp([0.05, 0.03, 0.14][line] * (path.length - 1), 0.03, [0.2, 0.14, 0.9][line]);       // the air is dragged slowly, the lead at once
      h.moves = path.length > 1 ? [{ path, t0: note.t, t1: note.t + glide, ease: 1, drag: true }] : [];
      h.from = h.col.slice(); h.to = col; h.turnAt = note.t; h.turnFor = glide;
      h.end = note.t + note.dur; h.level = Math.max(h.level * 0.92, level); h.waiting = false;
    } else {
      for (const x of this.heads) if (x.line === line && x.end > note.t) x.end = Math.max(x.t0, note.t);       // the line's last note ends where the new one begins
      h = { line, t0: note.t, end: note.t + note.dur, level, col, from: col, to: col, turnAt: 0, turnFor: 0, cell: home, moves: [], p: S.pos[home].slice(), seg: null, I: 0, waiting: false, gone: false };
      this.heads.push(h);
      const ring = L.ring > 0 ? L.ring : note.accent ? 0.42 : 0;
      if (ring > 0) this.rings.push({ line, dir: S.pos[home], t0: note.t, col, strength: ring * level * (note.accent && L.ring > 0 ? 2.2 : 1) });
    }
    // where it goes from here: toward the line's next note, if that note is known and is not itself a slide (a slide will drag it anyway)
    const after = h.moves.length ? h.moves[h.moves.length - 1].t1 : note.t;
    if (nextNote && !nextNote.slide) this._send(h, home, nextNote, after);
    else if (!nextNote) { h.waiting = true; this._wander(h, home, after, Math.floor(note.t + 1e-6) + 1); }
  }
  // nobody plays: three slow lights of the three families wander by themselves, each to the hungriest cell in its reach
  _idle(dt, k) {
    const S = this.S, t = performance.now() / 1000;
    if (!this.idlers.length) for (let l = 0; l < 3; l++) { const cell = POLE_OF_PC[[0, 4, 7][l]]; this.idlers.push({ line: l, cell, path: null, at: 0, I: 0, p: S.pos[cell].slice(), col: [0, 0, 0], seg: null }); }
    for (const h of this.idlers) {
      const L = LINES[h.line];
      if (!h.path || h.at >= h.path.length - 1) { if (h.path) h.cell = h.path[h.path.length - 1]; h.path = this._route(h.cell, -1, 4 + 2 * h.line); h.at = 0; }
      if (h.path.length > 1) {
        h.at = Math.min(h.path.length - 1, h.at + dt * [1.0, 2.2, 0.6][h.line]);
        const i = clamp(Math.floor(h.at), 0, h.path.length - 2), fr = h.at - i, A = S.pos[h.path[i]], B = S.pos[h.path[i + 1]];
        h.p = norm([A[0] + (B[0] - A[0]) * fr, A[1] + (B[1] - A[1]) * fr, A[2] + (B[2] - A[2]) * fr]); h.cell = fr < 0.5 ? h.path[i] : h.path[i + 1]; h.seg = [h.path[i], h.path[i + 1], fr];
      }
      h.I = k * [0.62, 0.82, 0.5][h.line];
      h.col = hsl(L.hue + L.span * Math.sin(t * 0.11 + h.line * 2.1), L.sat, L.light);
    }
  }
  // the drums of this frame, in the sphere's own frame. Per kick: its place, the strength and width of its glow on the
  // panels, its ring, its mark on the ports. Per snare: the longitude of its streak, its strength, its height, its mark.
  // A step lies at the same place in the room whatever the turn. The marks of the bar's hits stand from the bar line
  // on (the pattern is there to read); a strike is the light on the panels when the playhead reaches the hit.
  _drums(pos, turn) {
    const kb = this.kb, sb = this.sb, calm = this.calm, bar = Math.floor(pos), ck = Math.cos(KICK_LAT), sk = Math.sin(KICK_LAT); let nk = 0, ns = 0;
    for (let back = 0; back < (calm ? 1 : 2); back++) {
      const sc = this.score.get(bar - back); if (!sc) continue;
      const phase = Math.min(pos - bar, 1) + back, shown = back ? 0 : Math.min(1, phase / 0.04);
      for (let s = 0; s < 16; s++) {
        const k = sc.kick[s], sn = sc.snare[s]; if (!k && !sn) continue;
        const age = phase - s / 16, lon = stepLon((s + 0.5) / 16) + turn;
        if (k && nk < 24) {
          let glow = 0, ring = 0;
          if (!calm && age >= 0) { glow = k * (0.95 * Math.exp(-age / 0.07) + 0.1 * Math.exp(-age / 0.25)); ring = 0.5 * k * Math.exp(-age / 0.17); }
          const mark = (0.45 + 0.4 * k) * shown * (calm ? 1.6 : 1);
          if (glow > 0.004 || ring > 0.004 || mark > 0.004) { const o = nk * 8; kb[o] = ck * Math.cos(lon); kb[o + 1] = sk; kb[o + 2] = ck * Math.sin(lon); kb[o + 3] = glow; kb[o + 4] = 12.5; kb[o + 5] = ring; kb[o + 6] = Math.min(0.62, 3.4 * Math.max(age, 0)); kb[o + 7] = mark; nk++; }
        }
        if (sn && ns < 24) {
          let v = 0; const tall = sn > 0.75 ? 0.5 : 0.34;
          if (!calm && age >= 0) v = sn * (1.15 * Math.exp(-age / 0.05) + 0.1 * Math.exp(-age / 0.25));
          const mark = (0.45 + 0.4 * sn) * shown * (calm ? 1.6 : 1);
          if (v > 0.004 || mark > 0.004) { const o = ns * 4; sb[o] = lon; sb[o + 1] = v; sb[o + 2] = 1 / (2 * tall * tall); sb[o + 3] = mark; ns++; }
        }
      }
    }
    this.nk = nk; this.ns = ns;
  }
  _lights(pos, dBars, turn) {
    const S = this.S, n = S.n, heads = this.heads, calm = this.calm;
    // the notes whose time has come
    if (this.posWas === null || this.posWas === undefined || pos < this.posWas || pos - this.posWas > 2) { this.posWas = pos - 1e-6; }
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
      let mv = null; for (const m of h.moves) { if (pos >= m.t0) mv = m; }
      const onWay = mv && !mv.drag && pos < mv.t1;
      if (onWay) env = Math.max(env, L.floor * Math.min(1, (pos - h.t0) / 0.04));        // a spot on its way stays lit
      const want = h.level * env * L.gain;
      h.I = want >= h.I ? want : Math.max(want, h.I * Math.exp(-dBars / 0.06));
      if (pos > h.end && h.I < 0.012 && !onWay) h.gone = true;
      if (h.turnFor > 0) { const k = clamp((pos - h.turnAt) / h.turnFor, 0, 1); for (let q = 0; q < 3; q++) h.col[q] = h.from[q] + (h.to[q] - h.from[q]) * k; }
      h.seg = null;
      if (mv) {
        const hops = mv.path.length - 1, f = Math.pow(clamp((pos - mv.t0) / Math.max(mv.t1 - mv.t0, 1e-4), 0, 1), mv.ease);
        const g = Math.min(hops * f, hops - 1e-4), i = Math.floor(g), fr = g - i;
        const A = S.pos[mv.path[i]], B = S.pos[mv.path[i + 1]], p = norm([A[0] + (B[0] - A[0]) * fr, A[1] + (B[1] - A[1]) * fr, A[2] + (B[2] - A[2]) * fr]);
        h.p = p; h.cell = fr < 0.5 ? mv.path[i] : mv.path[i + 1]; h.seg = [mv.path[i], mv.path[i + 1], fr];
      }
    }
    this.heads = heads.filter((h) => !h.gone);
    this.rings = this.rings.filter((r) => pos - r.t0 < LINES[r.line].ringLife * 3.5);
    for (const r of this.rings) { const L = LINES[r.line], age = pos - r.t0; r.rho = age * L.ringSpeed * (calm ? 0.55 : 1); r.I = r.strength * Math.exp(-age / L.ringLife) * (calm ? 0.6 : 1) * Math.min(1, age / 0.01); }
    const hd = this.idleK > 0.01 && this.idlers.length ? this.heads.concat(this.idlers) : this.heads, nh = hd.length;
    // the afterglow: a cell keeps the light of the spots that passed, per line
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
    const ta = (this.lt / 16) * TAU, tide = [Math.cos(ta) * 0.94, 0.34 * Math.sin(ta * 2), Math.sin(ta) * 0.94], tideUp = 1 + 1.1 * this.idleK;
    // the drums of this frame, and the playhead: the line that crosses the near side once a bar
    this._drums(pos, turn);
    const kb = this.kb, sb = this.sb, nk = this.nk, ns = this.ns, fLon = S.faceLon, fLat = S.faceLat, fCos = S.faceCos;
    const playOn = !calm && this.idleK < 0.5 && this.score.has(Math.floor(pos)), playLon = stepLon(Math.min(pos - Math.floor(pos), 1)) + turn;
    // every face: the three lines at its place, mixed, and the drums over them; then the cell's colour, the ports, the light it spills
    const light = S.light, cellCol = S.cellCol, I = [0, 0, 0], C = [0, 0, 0, 0, 0, 0, 0, 0, 0], rings = this.rings, nr = rings.length, fedKeep = Math.exp(-dBars / 6);
    let fedSum = 0; if (this.fedMean === undefined) this.fedMean = 0;
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
        const td = (0.07 + 0.13 * Math.max(0, ux * tide[0] + uy * tide[1] + uz * tide[2]) ** 2) * tideUp, quiet = 1 - Math.min(1, T * 3.6);
        let r = (C[0] * w0 + C[3] * w1 + C[6] * w2) / ws, g = (C[1] * w0 + C[4] * w1 + C[7] * w2) / ws, b = (C[2] * w0 + C[5] * w1 + C[8] * w2) / ws;
        const grey = 0.3 * r + 0.5 * g + 0.2 * b, E = T * (0.55 + 0.75 * T);                 // a mixed colour keeps its strength of hue; a bright spot is brighter than twice a dim one
        r = Math.max(0, grey + (r - grey) * 1.3) * E + 0.2 * td * (0.12 + 0.88 * quiet); g = Math.max(0, grey + (g - grey) * 1.3) * E + 0.86 * td * (0.12 + 0.88 * quiet); b = Math.max(0, grey + (b - grey) * 1.3) * E + 0.9 * td * (0.12 + 0.88 * quiet);
        // the drums: the kick's warm glow and ring at the nearer of its step's two places, the snare's cold streak along its step's line
        if (nk || ns || playOn) {
          let kv = 0, sv = 0, pv = 0;
          for (let j = 0; j < nk; j++) {
            const o = j * 8, a = ux * kb[o] + uz * kb[o + 2], dd = 2 - 2 * (uy * kb[o + 1] + (a > 0 ? a : -a)); if (dd > 1.7) continue;
            kv += kb[o + 3] * Math.exp(-dd * kb[o + 4]);
            if (dd < 0.05) kv += 0.22 * kb[o + 7] * Math.exp(-dd * 89);                           // its mark: the panel at the hit's place
            if (kb[o + 5] > 0.01) { const d = Math.sqrt(dd) - kb[o + 6]; kv += kb[o + 5] * Math.exp(-d * d * 51); }
          }
          const lon = fLon[fi], cl = fCos[fi], dy = fLat[fi] - SNARE_LAT;
          for (let j = 0; j < ns; j++) {
            const o = j * 4; let dl = lon - sb[o]; dl -= Math.PI * Math.round(dl / Math.PI); const x = dl * cl; if (x > 0.4 || x < -0.4) continue;
            sv += sb[o + 1] * Math.exp(-x * x * 69.2 - dy * dy * sb[o + 2]) + 0.22 * sb[o + 3] * Math.exp(-x * x * 139 - dy * dy * 16);       // the streak, and its mark
          }
          if (playOn) { let dl = lon - playLon; dl -= Math.PI * Math.round(dl / Math.PI); const x = dl * cl; if (x < 0.2 && x > -0.2) pv = 0.11 * Math.exp(-x * x * 139); }
          if (kv > 1.3) kv = 1.3; if (sv > 1.3) sv = 1.3;
          r += kv * KICK_RGB[0] + sv * SNARE_RGB[0] + pv * 0.7; g += kv * KICK_RGB[1] + sv * SNARE_RGB[1] + pv * 0.9; b += kv * KICK_RGB[2] + sv * SNARE_RGB[2] + pv;
        }
        for (let v = 0; v < 3; v++) { const o = (fi * 3 + v) * 6; light[o] = r; light[o + 1] = g; light[o + 2] = b; }
        cr += r; cg += g; cb += b;
      }
      cr /= 20; cg /= 20; cb /= 20; cellCol[c * 3] = cr; cellCol[c * 3 + 1] = cg; cellCol[c * 3 + 2] = cb;
      // a cell is lit when its mean panel colour shows at half of full scale or more. It wants light the more, the
      // longer it has been dark and the less lit time it has had in the last bars against the others
      const lit = Math.max(cr, cg, cb) >= 0.3546;
      if (lit) S.dark[c] = 0; else S.dark[c] += dBars;
      S.fed[c] = S.fed[c] * fedKeep + (lit ? dBars : 0); fedSum += S.fed[c];
      S.want[c] = clamp(0.4 * Math.min(1, S.dark[c] / 3) + 0.6 * clamp(1.25 - S.fed[c] / (this.fedMean + 0.02), 0, 1), 0, 1);
    }
    this.fedMean = fedSum / n;
    let ar = 0, ag = 0, ab = 0;
    const spark = calm ? 0 : this.spark, seed = this.sparkSeed, pLon = S.portLon, pCos = S.portCos, pLat = S.portLat, pDir = S.portDir, marks = nk || ns;
    for (let c = 0; c < n; c++) {
      let sr = 0, sg = 0, sb2 = 0; const nb = S.nb[c];
      for (const k of nb) { sr += cellCol[k * 3]; sg += cellCol[k * 3 + 1]; sb2 += cellCol[k * 3 + 2]; }
      sr /= nb.length; sg /= nb.length; sb2 /= nb.length;
      for (let v = 0; v < 60; v++) { const o = (c * 60 + v) * 6 + 3; light[o] = sr; light[o + 1] = sg; light[o + 2] = sb2; }
      for (let i = 0; i < 12; i++) {
        const pi = c * 12 + i, o = pi * 3; let tw = 0, ph = 0, km = 0, sm = 0;
        if (marks) {                                          // the bar's pattern, to read: a warm point where a kick falls, a short cold line where a snare falls
          const px = pDir[o], py = pDir[o + 1], pz = pDir[o + 2];
          for (let j = 0; j < nk; j++) { const q = j * 8; if (kb[q + 7] < 0.004) continue; const a = px * kb[q] + pz * kb[q + 2], dd = 2 - 2 * (py * kb[q + 1] + (a > 0 ? a : -a)); if (dd < 0.12) km += 1.7 * kb[q + 7] * Math.exp(-dd * 42); }
          const dy = pLat[pi] - SNARE_LAT;
          if (dy < 0.75 && dy > -0.75) for (let j = 0; j < ns; j++) { const q = j * 4; if (sb[q + 3] < 0.004) continue; let dl = pLon[pi] - sb[q]; dl -= Math.PI * Math.round(dl / Math.PI); const x = dl * pCos[pi]; if (x < 0.2 && x > -0.2) sm += 1.7 * sb[q + 3] * Math.exp(-x * x * 100 - dy * dy * 6); }
        }
        if (playOn) {                                         // the playhead is a line of lit ports; the hats make the ports round it sparkle
          let dl = pLon[pi] - playLon; dl -= Math.PI * Math.round(dl / Math.PI); const x = dl * pCos[pi], xx = x * x;
          if (xx < 0.5) { ph = 0.34 * Math.exp(-xx * 165); if (spark > 0.02 && ((Math.sin(pi * 12.9898 + seed * 78.233) * 43758.5453) % 1 + 1) % 1 > 0.5) tw = spark * Math.exp(-xx * 9); }
        }
        S.portCol[o] = cellCol[c * 3] * 0.85 + 0.05 + tw * 0.7 + ph * 0.8 + km * KICK_RGB[0] + sm * SNARE_RGB[0]; S.portCol[o + 1] = cellCol[c * 3 + 1] * 0.85 + 0.06 + tw * 0.75 + ph * 0.95 + km * KICK_RGB[1] + sm * SNARE_RGB[1]; S.portCol[o + 2] = cellCol[c * 3 + 2] * 0.85 + 0.06 + tw * 0.75 + ph + km * KICK_RGB[2] + sm * SNARE_RGB[2];
      }
      ar += cellCol[c * 3]; ag += cellCol[c * 3 + 1]; ab += cellCol[c * 3 + 2];
    }
    this.avg = [ar / n, ag / n, ab / n];
    // the seams: each in the colours of its two cells; a spot on its way from one cell to the next lights the seam between them
    const pulse = S.seamPulse; pulse.fill(0);
    for (const h of hd) if (h.seg && h.I > 0.03) {
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
    const dt = clamp(ts - (this.last || ts), 0, 100); this.last = ts;      // the first frame's time may lie before the call of start()
    this.frameMs += (dt - this.frameMs) * 0.05; this.frames++;
    if (this.frames > 90 && this.frameMs > 26 && this.scale > 0.5) {         // too slow: a lower resolution, the same picture
      if (++this.slow > (this.frameMs > 60 ? 20 : 60)) { this.scale = Math.max(0.5, this.scale * (this.frameMs > 60 ? 0.6 : 0.8)); this.slow = 0; this.frames = 0; this.resize(); }
    } else this.slow = 0;
    if (this.gl && !this.lost) this._draw(dt / 1000);
  }

  // one step of the program: the time, the drums' breath and sparkle, the turn, the lights. No drawing.
  _advance(dt) {
    const calm = this.calm, real = performance.now() / 1000, barDur = this._barDur();
    const pos = this._pos(this._now()), dBars = this.pos2 === undefined ? 0 : clamp(pos - this.pos2, 0, 0.5);
    if (this.curIndex >= 0 && (this.pos2 === undefined || pos > this.pos2 + 1e-7)) this.movedAt = real;
    this.pos2 = pos; this.pos = pos;
    // nobody plays: before the first bar, or when the clock has stood for more than a bar
    this.idleK += ((this.curIndex < 0 || real - this.movedAt > 1.3 * barDur ? 1 : 0) - this.idleK) * Math.min(1, dt * 1.5);
    // the drums' part in the whole picture: a kick is a small breath of the sculpture, the hats make ports sparkle
    const sc = this.score.get(Math.floor(pos)), step = Math.floor((pos % 1) * 16) + Math.floor(pos) * 16;
    if (sc && step !== this.stepWas) { this.stepWas = step; const t = ((step % 16) + 16) % 16; if (sc.kick[t] > 0) this._pulse(1); if (sc.hat[t] > 0) { this.spark = 0.5 + 0.5 * sc.hat[t]; this.sparkSeed = step; this.sparkRate = sc.open[t] > 0 ? 5 : 14; } }
    this.kick *= Math.exp(-dt * 6.5); this.spark *= Math.exp(-dt * this.sparkRate);
    if (this.analyser && this.spec) { this.analyser.getByteFrequencyData(this.spec); let s = 0; for (let i = 0; i < this.spec.length; i += 4) s += this.spec[i]; this.level += ((0.85 + 0.4 * clamp((s / (this.spec.length / 4) / 255) * 2.2, 0, 1)) - this.level) * 0.1; } else this.level += (1 - this.level) * 0.1;
    this.turn += dt * (calm ? 0.045 : 0.075);
    const turn = this.view.turn === null ? this.turn : this.view.turn, own = dBars + (this.idleK * dt) / barDur;       // the lights' own time runs on when nobody plays
    this.lt += own;
    if (this.idleK > 0.01) this._idle(dt, this.idleK);
    this._lights(pos, own, turn);
    return turn;
  }

  _draw(dt) {
    const gl = this.gl, c = this.canvas, P = this.prog;
    if (c.clientWidth && Math.abs(c.clientWidth * Math.min(2, devicePixelRatio || 1) * this.scale - c.width) > 2) this.resize();
    if (!this.fbo || !this.S) this.resize();
    const S = this.S, F = this.fbo, W = c.width, H = c.height, calm = this.calm, aspect = W / Math.max(1, H);
    const turn = this._advance(dt);
    // the camera: the sphere fits the canvas, a little above its middle; on an upright canvas it fills the width
    const fov = 0.56, tanF = Math.tan(fov / 2), outer = 1 + S.rc * 1.05;
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
    gl.uniform4f(p.u.uMix, 0.75, 0.62, 0, 1.2 * this.level * (1 + (calm ? 0 : 0.07 * this.kick)));
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.activeTexture(gl.TEXTURE0);
  }
}

export default ScreenStage;
