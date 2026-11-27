// BRAIN view: the Cadence brain drawn from what the server reads after each `live` moment (one per bar):
// settled activation per region (downsampled), the four recall heads' reads (age in bars, share), the apical
// pools' gains, the settle's steps, the arousal. Everything moves on the music's clock: dots swell on the
// kick, decay over the beat, glide between bars. Six demo-scene modes (cadance, plasma, copper, tunnel, vectors,
// acid: Amiga style, in the demo-scene section below) and four diagrams:
//   orbit   regions as rings of neurons (inside out: hears, meaning, modules, association, prefrontal, plays),
//           memory reads reaching out to a ring of past bars, apical halos, settle ripples, the brain's core at the centre
//   flow    the signal path left to right; ribbons between regions = activity of one region times the next
//           (drawn from region activity, not measured per synapse)
//   memory  a spiral of past bars (newest at the centre, the four layers dotted), the four heads reaching back
//   scope   one oscilloscope line per region under a playhead running through the bar
// demo-scene modes first (cadance, plasma, copper, tunnel, vectors, acid: see the demo-scene section above), then the four diagrams
const LAYER_NAMES = ['BEAT', 'BASS', 'LEAD', 'AIR'];
const CALM = [95, 211, 198], HOT = [242, 166, 90], APICAL = [180, 156, 255];
const HEAD = ['#5fd3c6', '#8fb3ff', '#f2a65a', '#e58fb0'];
const LAYER = [[201, 206, 214], [95, 211, 198], [242, 166, 90], [143, 179, 255]];
const ROLE = { sensory: 'hears', code: 'meaning', association: 'association', prefrontal: 'prefrontal', motor: 'plays' };

const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${clamp(a)})`;
const roleOf = (n) => ROLE[n] || (n.startsWith('module_') ? `module ${n.split('_')[1]}` : n.startsWith('recall') ? `read ${n === 'recall' ? 1 : +n.split('_')[1] + 1}` : n);

// ======================= demo-scene modes =======================
// Demo-scene modes of the BRAIN view, Amiga style: everything is drawn into a small pixel screen (320 px wide)
// and blown up without smoothing. Every motion comes from the Cadence brain (region activity, the four memory
// heads' reads, arousal), the step clock and the audio; with no brain loaded they live from audio and steps alone.
//   cadance  the letters C A D A N C E, one per part of the brain, bouncing on the kick, copper gradient, sine scroller
//   plasma   sine plasma: frequencies = region activity, phases = memory heads, acid palette stepping on the kick
//   copper   raster bars, one per region: thickness = its activity, texture = its neurons, bounce on the kick
//   tunnel   texture tunnel: tiles = the four layers' sounds, speed = audio level, twist = SPACE
//   vectors  vector balls = association/prefrontal/module neurons, rings of past bars, memory reads as lines
//   acid     a 303 scope (Lissajous of the lead sound, motor activity as ripple) in a zooming feedback trail
const DEMO_MODES = ['cadance', 'plasma', 'copper', 'tunnel', 'vectors', 'acid', 'bobs', 'twister', 'fire', 'metaballs'];

const RGB = (r, g, b) => (255 << 24) | (clamp(b | 0, 0, 255) << 16) | (clamp(g | 0, 0, 255) << 8) | clamp(r | 0, 0, 255);
const unpack = (c) => [c & 255, (c >> 8) & 255, (c >> 16) & 255];

function ramp(stops) {                       // 256-entry palette from [pos, [r, g, b]] stops
  const p = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let k = 0; while (k < stops.length - 2 && i > stops[k + 1][0]) k++;
    const [a, ca] = stops[k], [b, cb] = stops[k + 1], t = clamp((i - a) / Math.max(1, b - a));
    p[i] = RGB(ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, ca[2] + (cb[2] - ca[2]) * t);
  }
  return p;
}
const ACID = ramp([[0, [6, 4, 14]], [40, [10, 60, 20]], [85, [57, 255, 20]], [125, [245, 255, 59]], [165, [255, 120, 20]], [205, [255, 43, 214]], [235, [60, 10, 90]], [255, [6, 4, 14]]]);
const GOLD = ramp([[0, [40, 8, 60]], [50, [180, 30, 120]], [100, [255, 120, 40]], [150, [255, 230, 90]], [190, [255, 255, 220]], [220, [90, 200, 255]], [255, [40, 8, 60]]]);
const SIN = new Float32Array(1024); for (let i = 0; i < 1024; i++) SIN[i] = Math.sin((i / 1024) * Math.PI * 2);
const hue = (h, s = 1, v = 1) => {             // h 0..1 -> [r, g, b]
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const c = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][((i % 6) + 6) % 6];
  return c.map((x) => Math.round(x * 255));
};

// ---------- 5x7 bitmap font ----------
const GLYPHS = {
  A: ' ### #   ##   #######   ##   ##   #', B: '#### #   ##   ##### #   ##   ##### ', C: ' ### #   ##    #    #    #   # ### ',
  D: '#### #   ##   ##   ##   ##   ##### ', E: '######    #    #### #    #    #####', F: '######    #    #### #    #    #    ',
  G: ' ### #   ##    # ####   ##   # ####', H: '#   ##   ##   #######   ##   ##   #', I: ' ###   #    #    #    #    #   ### ',
  J: '  ###   #    #    #    ##  #  ##  ', K: '#   ##  # # #  ##   # #  #  # #   #', L: '#    #    #    #    #    #    #####',
  M: '#   ### ### # ## # ##   ##   ##   #', N: '#   ###  ## # ##  ###   ##   ##   #', O: ' ### #   ##   ##   ##   ##   # ### ',
  P: '#### #   ##   ##### #    #    #    ', Q: ' ### #   ##   ##   ## # ##  #  ## #', R: '#### #   ##   ##### # #  #  # #   #',
  S: ' #####    #     ###     #    #####', T: '#####  #    #    #    #    #    #  ', U: '#   ##   ##   ##   ##   ##   # ### ',
  V: '#   ##   ##   ##   ##   # # #   #  ', W: '#   ##   ##   ## # ## # ## # # # # ', X: '#   ##   # # #   #   # # #   ##   #',
  Y: '#   ##   # # #   #    #    #    #  ', Z: '#####    #   #   #   #   #    #####',
  0: ' ### #   ##  ### # ###  ##   # ### ', 1: '  #   ##    #    #    #    #   ### ', 2: ' ### #   #    #   #   #   #   #####',
  3: '#####   #   #     #    ##   # ### ', 4: '   #   ##  # # #  # #####   #    # ', 5: '######    ####     #    ##   # ### ',
  6: '  ##  #   #    #### #   ##   # ### ', 7: '#####    #   #   #   #    #    #   ', 8: ' ### #   ##   # ### #   ##   # ### ',
  9: ' ### #   ##   # ####    #   #  ##  ', '.': '                          ##   ## ', ':': '      ##   ##          ##   ##      ',
  '-': '               #####               ', '>': '#     #     #     #   #   #   #    ', '+': '       #    #  #####  #    #       ',
  '/': '    #    #   #   #   #   #    #    ', ' ': '                                   ',
};
const FONT = {};
for (const [k, s] of Object.entries(GLYPHS)) { const a = []; for (let i = 0; i < 35; i++) if (s[i] === '#') a.push([i % 5, (i / 5) | 0]); FONT[k] = a; }

// ---------- the pixel screen ----------
class DemoScreen {
  constructor() {
    this.cv = document.createElement('canvas'); this.g = this.cv.getContext('2d');
    this.w = 0; this.h = 0; this.img = null; this.buf = null;
    this.fb = document.createElement('canvas'); this.fg = this.fb.getContext('2d');      // acid feedback
    this.tmp = document.createElement('canvas'); this.tg = this.tmp.getContext('2d');
    this.stars = Array.from({ length: 160 }, (_, i) => ({ x: Math.sin(i * 12.9898) * 1.0, y: Math.cos(i * 78.233) * 1.0, z: ((i * 0.618) % 1) + 0.02 }));
    this.scroll = { text: 'CADANCE :: A CADENCE BRAIN PLAYS MINIMAL TECHNO :: ', x: 0 };
    this.tex = new Uint32Array(64 * 64); this.texKey = '';
    this.tun = null; this.uoff = 0; this.voff = 0; this.rot = [0.3, 0.6, 0]; this.palOff = 0;
  }
  size(W, H) {
    const w = 320, h = Math.round(clamp((320 * H) / Math.max(1, W), 120, 640));
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h; this.cv.width = w; this.cv.height = h;
    this.img = this.g.createImageData(w, h); this.buf = new Uint32Array(this.img.data.buffer);
    for (const c of [this.fb, this.tmp]) { c.width = w; c.height = h; }
    this.tun = null;
  }
  clear(c = RGB(6, 5, 12)) { this.buf.fill(c); }
  px(x, y, c) { x |= 0; y |= 0; if (x >= 0 && y >= 0 && x < this.w && y < this.h) this.buf[y * this.w + x] = c; }
  add(x, y, r, g, b) {                       // additive pixel
    x |= 0; y |= 0; if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    const i = y * this.w + x, c = this.buf[i];
    this.buf[i] = RGB((c & 255) + r, ((c >> 8) & 255) + g, ((c >> 16) & 255) + b);
  }
  rect(x, y, w, h, c) { for (let j = Math.max(0, y | 0); j < Math.min(this.h, (y + h) | 0); j++) this.buf.fill(c, j * this.w + Math.max(0, x | 0), j * this.w + Math.min(this.w, (x + w) | 0)); }
  line(x0, y0, x1, y1, r, g, b) {
    const n = Math.max(1, Math.ceil(Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0))));
    for (let i = 0; i <= n; i++) this.add(x0 + ((x1 - x0) * i) / n, y0 + ((y1 - y0) * i) / n, r, g, b);
  }
  text(s, x, y, scale, colorAt) {            // colorAt(charIndex, px, py) -> packed colour
    let cx = x;
    for (let k = 0; k < s.length; k++) {
      const gl = FONT[s[k]] || FONT[' '];
      for (const [gx, gy] of gl) {
        const c = colorAt(k, cx + gx * scale, y + gy * scale);
        this.rect(cx + gx * scale, y + gy * scale, scale, scale, c);
      }
      cx += 6 * scale;
    }
  }
  blit(g, W, H) {
    this.g.putImageData(this.img, 0, 0);
    g.imageSmoothingEnabled = false; g.drawImage(this.cv, 0, 0, W, H); g.imageSmoothingEnabled = true;
  }
}

// ---------- shared backdrops ----------
function starfield(S, X, speed) {
  const { w, h } = X;
  for (const s of X.stars) {
    s.z -= speed * S.dt; if (s.z <= 0.02) { s.z += 1; }
    const k = 0.5 / s.z, x = w / 2 + s.x * w * k, y = h / 2 + s.y * h * k;
    const b = clamp((1 - s.z) * 1.2) * 200;
    X.add(x, y, b, b, b * 1.1);
  }
}

// ---------- cadance: the letters ----------
const PARTS = ['hears', 'meaning', 'association', 'prefrontal', 'plays', 'memory', 'arousal'];
function cadance(X, S) {
  const { w, h } = X; X.clear(RGB(5, 4, 12));
  // copper sky: three slow raster bands behind everything
  for (let y = 0; y < h; y++) {
    let r = 5, g = 4, b = 12;
    for (let k = 0; k < 3; k++) {
      const cy = h * (0.5 + 0.42 * Math.sin(S.t * 0.00045 * (k + 1) + k * 2.1)), d = Math.abs(y - cy) / (5 + 6 * S.act[k + 2]);
      if (d < 1) { const c = hue((0.78 + 0.08 * k + S.heat * 0.15) % 1, 0.8, 0.35 * (1 - d * d)); r += c[0]; g += c[1]; b += c[2]; }
    }
    X.buf.fill(RGB(r, g, b), y * w, (y + 1) * w);
  }
  starfield(S, X, 0.08 + 0.5 * S.level);
  // the logo
  const word = 'CADANCE', s = Math.max(2, Math.floor(Math.min((w * 0.92) / (7 * 6), (h * 0.36) / 7)));
  const x0 = Math.round((w - (7 * 6 - 1) * s) / 2), baseY = Math.round(h * 0.2);
  const roll = (S.t * 0.06) | 0;
  for (let i = 0; i < 7; i++) {
    const a = S.act[i];
    const wave = Math.sin(S.t * 0.003 + i * 0.75) * s * 1.5 * (0.4 + a);
    const jump = -S.kickAt(i) * s * 3.2;                     // a kick runs through the letters one after another
    const yy = baseY + wave + jump + (S.flashSign < 0 ? Math.sin(S.t * 0.09 + i) * S.flash * s : 0);
    const sy = Math.round(s * (1 + 0.55 * a));                // a busy part stretches its letter
    const lit = 0.62 + 0.55 * a + 0.25 * S.kick;
    for (const [gx, gy] of FONT[word[i]]) {
      const px0 = x0 + (i * 6 + gx) * s, py0 = yy + gy * sy;
      const pink = S.flashSign > 0 ? S.flash : 0;
      const f = (v, j) => clamp((v * (1 - pink) + [255, 111, 170][j] * pink) * lit, 0, 255);
      for (let r = 0; r < sy; r++) {                         // copper: one colour per scanline, rolling through the logo
        const y = Math.round(py0 + r), c = unpack(GOLD[((y * 2) + roll) & 255]);
        const edge = r === 0 ? 1.45 : r === sy - 1 ? 0.35 : 1;
        X.rect(px0, y, s, 1, RGB(f(c[0] * edge, 0), f(c[1] * edge, 1), f(c[2] * edge, 2)));
      }
      X.rect(px0 + s - 1, py0 + 1, 1, sy - 1, RGB(20, 10, 30));                                             // bevel shadow
    }
    // the part's name under its letter, tiny
    const lab = PARTS[i].toUpperCase().slice(0, 4), lx = x0 + i * 6 * s + Math.round((5 * s - 4 * 6) / 2);
    X.text(lab, lx, baseY + 7 * s * 1.55 + 6, 1, () => RGB(60 + 140 * a, 70 + 120 * a, 90 + 90 * a));
  }
  // sine scroller: what the brain is doing, bar by bar
  const sc = X.scroll, speed = 30 + 40 * S.level;
  sc.x += speed * S.dt;
  while (sc.x > 12 && sc.text.length > 1) { sc.x -= 12; sc.text = sc.text.slice(1); }
  if (sc.text.length < 80) sc.text += S.news();
  const sy0 = Math.round(h * 0.82), cw = 12;
  for (let k = 0; k * cw - sc.x < w + cw && k < sc.text.length; k++) {
    const xk = k * cw - sc.x, yk = sy0 + Math.sin((xk + S.t * 0.12) * 0.035) * h * 0.05;
    X.text(sc.text[k], xk, yk, 2, (ci, px, py) => ACID[(((px + py * 2) | 0) + ((S.t * 0.1) | 0)) & 255]);
  }
  return 'cadance: C hears · A meaning · D association · A prefrontal · N plays · C memory reads · E arousal — a letter swells with its part, the kick runs through them, the scroller reports the bar';
}

// ---------- plasma ----------
function plasma(X, S) {
  const { w, h, buf } = X, t = S.t * 0.001;
  const f1 = 0.02 + 0.06 * S.act[0], f2 = 0.02 + 0.07 * S.act[1], f3 = 0.015 + 0.05 * S.act[2], f4 = 0.03 + 0.08 * S.act[3];
  const p1 = t * (0.7 + S.headShare[0]), p2 = t * (0.9 + S.headShare[1]), p3 = t * (0.5 + S.headShare[2]), p4 = t * (1.1 + S.headShare[3]);
  const warp = 18 * S.filter[2] + 10 * S.filter[1];          // FILTER on lead/bass bends the field
  if (S.kickEdge) X.palOff = (X.palOff + 32) & 255;           // the palette steps on every kick
  X.palOff = (X.palOff + S.dt * 40) % 256;
  const off = X.palOff | 0, cx = w / 2, cy = h / 2, bright = 0.75 + 0.35 * S.level + 0.2 * S.kick;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = x - cx, dy = y - cy, r = Math.sqrt(dx * dx + dy * dy);
      const xw = x + Math.sin(y * 0.05 + t) * warp;
      const v = SIN[((xw * f1 + p1) * 163) & 1023] + SIN[((y * f2 + p2) * 163) & 1023] + SIN[(((x + y) * f3 + p3) * 163) & 1023] + SIN[((r * f4 - p4) * 163) & 1023];
      const c = ACID[(((v + 4) * 32) | 0) + off & 255];
      buf[y * w + x] = bright >= 0.99 ? c : RGB((c & 255) * bright, ((c >> 8) & 255) * bright, ((c >> 16) & 255) * bright);
    }
  }
  return 'plasma: frequencies = activity of hears, meaning, association, prefrontal; phases = the four memory heads; FILTER bends it; the palette steps on the kick';
}

// ---------- copper bars ----------
function copper(X, S) {
  const { w, h, buf } = X; X.clear(RGB(4, 3, 10));
  const water = Math.round(h * 0.74), bars = S.bars;
  const list = bars.map((b, i) => {
    const ph = S.t * 0.0012 * (0.6 + 0.12 * i) + i * 0.83;
    return { ...b, i, y: water * (0.5 + 0.42 * Math.sin(ph)) - S.kick * 8 * (1 + b.a), z: Math.cos(ph), th: 2.5 + 7 * b.a };
  }).sort((a, b) => a.z - b.z);
  const col = (b, d) => {                     // copper gradient: dark edges, saturated body, a white-hot core
    const k = 1 - Math.abs(d), base = hue((b.hue + S.heat * 0.12 + S.mood[1] * 0.1) % 1, 0.9, 0.2 + 0.8 * k), hot = Math.pow(k, 6) * 0.75;
    return base.map((v) => v * (1 - hot) + 255 * hot);
  };
  for (const b of list) {
    for (let y = Math.max(0, Math.floor(b.y - b.th)); y < Math.min(water, Math.ceil(b.y + b.th)); y++) {
      const d = (y - b.y) / b.th, c = col(b, d), vals = b.vals, n = vals.length;
      for (let x = 0; x < w; x++) {
        const v = n ? vals[((x * n) / w) | 0] : 0.5, k = 0.8 + 0.3 * v;    // the region's neurons along the bar
        buf[y * w + x] = RGB(c[0] * k, c[1] * k, c[2] * k);
      }
    }
  }
  for (let y = water; y < h; y++) {           // water: mirror with ripple
    const j = y - water, src = water - 1 - Math.round(j * 1.7);
    if (src < 0) break;
    const off = Math.round(Math.sin(S.t * 0.005 + j * 0.6) * (1 + j * 0.15));
    for (let x = 0; x < w; x++) {
      const c = buf[src * w + clamp(x + off, 0, w - 1)], k = 0.42 * (1 - j / (h - water));
      buf[y * w + x] = RGB((c & 255) * k, ((c >> 8) & 255) * k * 1.05, ((c >> 16) & 255) * k * 1.25 + 8);
    }
  }
  X.rect(0, water, w, 1, RGB(80, 90, 120));
  return 'copper: one raster bar per brain region — thickness = its activity, texture = its neurons, bounce on the kick; mirrored in the water';
}

// ---------- tunnel ----------
function tunnel(X, S) {
  const { w, h, buf } = X;
  if (!X.tun) {                               // angle / depth tables, once per screen size
    X.tun = { u: new Uint8Array(w * h), v: new Uint16Array(w * h), s: new Float32Array(w * h) };
    const R = Math.hypot(w, h) / 2;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dx = x - w / 2, dy = (y - h / 2) * 1.2, d = Math.max(1, Math.hypot(dx, dy)), i = y * w + x;
      X.tun.u[i] = (((Math.atan2(dy, dx) / (Math.PI * 2)) * 256) | 0) & 255; X.tun.v[i] = ((32 * 64) / d) | 0; X.tun.s[i] = clamp(d / (R * 0.55));
    }
  }
  const key = S.tokens.join(',');
  if (key !== X.texKey) {                     // the texture: four sectors, one per layer, coloured by its sound
    X.texKey = key;
    for (let v = 0; v < 64; v++) for (let u = 0; u < 64; u++) {
      const L = u >> 4, tok = S.tokens[L] || 0, chk = ((u >> 3) ^ (v >> 3)) & 1;
      const c = tok ? hue(((tok * 0.137) + L * 0.25) % 1, 0.85, chk ? 1 : 0.45) : (chk && (u + v) % 7 === 0 ? [40, 40, 60] : [8, 8, 14]);
      X.tex[v * 64 + u] = RGB(c[0], c[1], c[2]);
    }
  }
  X.voff += S.dt * (10 + 70 * S.level + 25 * S.kick);
  X.uoff += S.dt * (6 + 30 * (S.filter[0] + S.filter[2]));
  const twist = 0.6 * (S.space[1] + S.space[2] + S.space[3]), vo = X.voff | 0, uo = X.uoff | 0, { u, v, s } = X.tun;
  const boost = 1 + 0.6 * S.kick;
  for (let i = 0; i < w * h; i++) {
    const vv = v[i], uu = (u[i] + uo + ((vv * twist) | 0)) & 63, c = X.tex[((vv + vo) & 63) * 64 + uu], k = s[i] * boost;
    buf[i] = RGB((c & 255) * k, ((c >> 8) & 255) * k, ((c >> 16) & 255) * k);
  }
  return 'tunnel: the walls are the four layers (beat, bass, lead, air) coloured by their sounds — speed = audio level, twist = SPACE, spin = FILTER';
}

// ---------- vector balls ----------
function vectors(X, S) {
  const { w, h } = X; X.clear(RGB(3, 3, 9));
  starfield(S, X, 0.15 + 0.6 * S.level);
  X.rot[0] += S.dt * (0.25 + 0.6 * S.level); X.rot[1] += S.dt * (0.17 + 0.3 * S.act[3]);
  const [ax, ay] = X.rot, ca = Math.cos(ax), sa = Math.sin(ax), cb = Math.cos(ay), sb = Math.sin(ay);
  const R = Math.min(w, h) * 0.3, cx = w / 2, cy = h * 0.47;
  // rings of past bars receding behind the ball cloud; memory reads drawn as lines to them
  const ring = (age) => ({ z: 1 + age * 0.22, r: R * 1.3 });
  for (let age = 15; age >= 1; age--) {
    const bar = S.history[S.history.length - 1 - age]; if (!bar) continue;
    const { z, r } = ring(age), k = 1 / z;
    for (let l = 0; l < 4; l++) {
      const tok = bar.tokens[l]; if (!tok) continue;
      const c = hue((tok * 0.137 + l * 0.25) % 1, 0.8, 0.8 * k);
      for (let q = 0; q < 6; q++) { const a = (l / 4 + q / 24) * Math.PI * 2 + age * 0.3; X.add(cx + Math.cos(a) * r * k, cy + Math.sin(a) * r * k * 0.6, c[0], c[1], c[2]); }
    }
  }
  S.heads.forEach((reads, hd) => {
    const c = [[95, 211, 198], [143, 179, 255], [242, 166, 90], [229, 143, 176]][hd];
    for (const rd of reads) {
      if (rd.age == null || rd.age < 1 || rd.age > 15) continue;
      const { z, r } = ring(rd.age), k = 1 / z, a = hd / 4 * Math.PI * 2 + rd.age * 0.3, m = clamp(rd.w * 1.5);
      X.line(cx, cy, cx + Math.cos(a) * r * k, cy + Math.sin(a) * r * k * 0.6, c[0] * m, c[1] * m, c[2] * m);
    }
  });
  // the balls: neurons on a sphere, size and brightness = activation
  const pts = [];
  S.cloud.forEach((v, i) => {
    const n = S.cloud.length, yy = 1 - (2 * (i + 0.5)) / n, rr = Math.sqrt(1 - yy * yy), th = i * 2.39996;
    let x = Math.cos(th) * rr, y = yy, z = Math.sin(th) * rr;
    const x1 = x * cb - z * sb, z1 = x * sb + z * cb, y1 = y * ca - z1 * sa, z2 = y * sa + z1 * ca;
    const pr = R * (1 + 0.12 * S.kick) * (1 + 0.2 * v), k = 2.4 / (2.4 + z2);
    pts.push({ x: cx + x1 * pr * k, y: cy + y1 * pr * k, z: z2, v, k, g: S.cloudGroup[i] });
  });
  pts.sort((a, b) => b.z - a.z);
  for (const p of pts) {
    const r = (1.2 + 4.2 * p.v) * p.k, base = hue((0.48 + 0.18 * p.g + S.heat * 0.15) % 1, 0.75, 1), bright = 0.35 + 0.65 * p.v;
    for (let dy = -Math.ceil(r); dy <= Math.ceil(r); dy++) for (let dx = -Math.ceil(r); dx <= Math.ceil(r); dx++) {
      const d2 = (dx * dx + dy * dy) / (r * r); if (d2 > 1) continue;
      const hl = clamp(1 - ((dx + r * 0.35) ** 2 + (dy + r * 0.35) ** 2) / (r * r * 0.9)), sh = (0.35 + 0.65 * (1 - d2)) * bright;
      X.px(p.x + dx, p.y + dy, RGB(base[0] * sh + 255 * hl * 0.6 * bright, base[1] * sh + 255 * hl * 0.6 * bright, base[2] * sh + 255 * hl * 0.6 * bright));
    }
  }
  return 'vectors: balls = neurons of the modules, association and prefrontal (size = activation); rings = past bars by layer sound; lines = the four memory heads reading back';
}

// ---------- acid scope with feedback ----------
function acid(X, S, out) {
  const { w, h, fb, fg, tmp, tg } = X;
  const zoom = 1.018 + 0.04 * S.kick + 0.012 * S.level, rot = 0.008 + 0.04 * S.filter[2];
  tg.setTransform(1, 0, 0, 1, 0, 0); tg.globalCompositeOperation = 'source-over'; tg.fillStyle = '#06040e'; tg.fillRect(0, 0, w, h);
  tg.save(); tg.translate(w / 2, h / 2); tg.rotate(rot); tg.scale(zoom, zoom); tg.translate(-w / 2, -h / 2);
  try { tg.filter = `hue-rotate(${(6 + 30 * S.kick) | 0}deg)`; } catch { /* no canvas filters */ }
  tg.globalAlpha = 0.9; tg.drawImage(fb, 0, 0); tg.globalAlpha = 1; tg.filter = 'none'; tg.restore();
  fg.setTransform(1, 0, 0, 1, 0, 0); fg.globalCompositeOperation = 'copy'; fg.drawImage(tmp, 0, 0); fg.globalCompositeOperation = 'lighter';
  const tok = S.tokens[2] || S.tokens[1] || 1, a = 1 + (tok % 4), b = a + 1 + ((tok >> 2) % 3), vals = S.motor, n = vals.length;   // never a plain ellipse
  const amp = Math.min(w, h) * (0.18 + 0.22 * S.act[4] + 0.15 * S.level), ph = S.phase * Math.PI * 2 + S.filter[2] * 2;
  const cols = ['#39ff14', '#f5ff3b', '#ff2bd6', '#ff7814'];
  fg.lineWidth = 1; fg.globalAlpha = 0.75;
  for (const [mx, my, col] of [[1, 1, (S.bar + (S.step >> 2)) & 3], [-1, 1, (S.bar + (S.step >> 2) + 2) & 3]]) {   // the figure and its mirror image
    fg.strokeStyle = cols[col]; fg.beginPath();
    for (let i = 0; i <= 320; i++) {
      const th = (i / 320) * Math.PI * 2, rip = n ? vals[Math.min(n - 1, ((i / 320) * n) | 0)] : 0;
      const r = amp * (1 + 0.45 * rip);
      const x = w / 2 + mx * Math.sin(a * th + ph) * r, y = h / 2 + my * Math.sin(b * th) * r * 0.8;
      i ? fg.lineTo(x, y) : fg.moveTo(x, y);
    }
    fg.stroke();
  }
  fg.globalAlpha = 1;
  fg.globalCompositeOperation = 'source-over';
  out.imageSmoothingEnabled = false; out.drawImage(fb, 0, 0, out.canvas.width, out.canvas.height); out.imageSmoothingEnabled = true;
  return 'acid: a 303 scope — the lead sound sets the figure, plays-region activity ripples it, FILTER twists the feedback, the kick zooms';
}

// ---------- bobs: Amiga blitter objects on a Lissajous, leaving trails ----------
function bobs(X, S) {
  const { w, h, buf } = X;
  for (let i = 0; i < buf.length; i++) {      // the trails: every frame the screen fades a little (blitter feedback)
    const c = buf[i]; buf[i] = RGB((c & 255) * 0.78, ((c >> 8) & 255) * 0.78, ((c >> 16) & 255) * 0.82);
  }
  const a = 2 + Math.round(3 * S.act[2]), b = 3 + Math.round(3 * S.act[3]);          // the figure: association x prefrontal
  const n = 28, speed = 0.0006 + 0.0012 * S.level, size = 3 + 5 * S.act[4] + 3 * S.kick;   // size: plays region + kick
  const cx = w / 2, cy = h / 2, ax = w * 0.4, ay = h * 0.38, t = S.t * speed;
  for (let k = n - 1; k >= 0; k--) {
    const th = t - k * 0.09, L = k % 4, tok = S.tokens[L] || 0;
    const x = cx + Math.sin(a * th + S.headShare[0] * 2) * ax, y = cy + Math.sin(b * th) * ay;
    const c = tok ? hue((tok * 0.137 + L * 0.25) % 1, 0.85, 1) : [70, 70, 90], r = size * (0.6 + 0.4 * (1 - k / n));
    for (let dy = -Math.ceil(r); dy <= Math.ceil(r); dy++) for (let dx = -Math.ceil(r); dx <= Math.ceil(r); dx++) {
      const d2 = (dx * dx + dy * dy) / (r * r); if (d2 > 1) continue;
      const hl = clamp(1 - ((dx + r * 0.4) ** 2 + (dy + r * 0.4) ** 2) / (r * r * 0.7)), sh = 0.3 + 0.7 * (1 - d2);
      X.px(x + dx, y + dy, RGB(c[0] * sh + 255 * hl * 0.7, c[1] * sh + 255 * hl * 0.7, c[2] * sh + 255 * hl * 0.7));
    }
  }
  return 'bobs: a snake of blitter objects on a Lissajous — its figure = association x prefrontal activity, size = plays region + kick, colours = the four layers\' sounds';
}

// ---------- twister: a rotating copper column ----------
function twister(X, S) {
  const { w, h } = X; X.clear(RGB(5, 4, 12));
  starfield(S, X, 0.05 + 0.3 * S.level);
  const twist = 0.008 + 0.05 * (S.headShare[0] + S.headShare[1] + S.headShare[2] + S.headShare[3]) / 4;   // memory reads twist it
  const spin = S.t * (0.0012 + 0.003 * S.level) + S.kick * 0.4, cx = w / 2, R = w * (0.14 + 0.05 * S.act[4]);
  const faces = [0, 1, 2, 3].map((L) => { const tok = S.tokens[L] || 0; return tok ? hue((tok * 0.137 + L * 0.25) % 1, 0.85, 1) : [60, 60, 80]; });
  for (let y = 0; y < h; y++) {
    const th = spin + y * twist + Math.sin(y * 0.02 + S.t * 0.001) * 0.6 * S.space[2];
    const xs = [0, 1, 2, 3].map((k) => cx + Math.sin(th + (k * Math.PI) / 2) * R + Math.sin(y * 0.05 + S.t * 0.002) * 6 * S.filter[1]);
    for (let k = 0; k < 4; k++) {
      const x0 = xs[k], x1 = xs[(k + 1) % 4];
      if (x1 <= x0) continue;                 // a face we see: its two edges in order
      const light = clamp((x1 - x0) / (R * 1.45)), c = faces[k];
      for (let x = Math.ceil(x0); x < x1; x++) {
        const u = (x - x0) / (x1 - x0), edge = u < 0.08 || u > 0.92 ? 1.35 : 1, stripe = ((y + (S.t * 0.04) | 0) >> 3) & 1 ? 1 : 0.82;
        X.px(x, y, RGB(c[0] * light * edge * stripe, c[1] * light * edge * stripe, c[2] * light * edge * stripe));
      }
    }
  }
  return 'twister: a rotating copper column — its faces are the four layers\' sounds, the twist = how strongly the memory heads read, spin = audio level, SPACE bends it';
}

// ---------- fire: classic demo fire fed by the plays region ----------
const FIRE = ramp([[0, [0, 0, 0]], [40, [60, 0, 10]], [90, [200, 30, 0]], [150, [255, 140, 0]], [210, [255, 240, 120]], [255, [255, 255, 255]]]);
function fire(X, S) {
  const { w, h, buf } = X;
  if (!X.fire || X.fire.length !== w * h) X.fire = new Uint8Array(w * h);
  const F = X.fire, m = S.motor, n = m.length || 1;
  for (let x = 0; x < w; x++) {               // the bottom rows: heat from the plays region's neurons, a burst on the kick
    const v = m.length ? m[((x * n) / w) | 0] : S.level, heat = clamp(0.25 + 0.75 * v + 0.6 * S.kick) * 255;
    F[(h - 1) * w + x] = heat * (0.7 + 0.3 * Math.abs(SIN[(x * 37 + ((S.t * 0.6) | 0)) & 1023]));
    F[(h - 2) * w + x] = F[(h - 1) * w + x];
  }
  const cool = (0.35 + 1.1 * (1 - S.level) - 0.5 * S.act[2]) * (190 / h);      // subtractive cooling per row: association lets it climb
  for (let y = 0; y < h - 2; y++) for (let x = 0; x < w; x++) {
    const sx = x + (((S.t * 0.01 + y * 0.07) | 0) % 3) - 1;
    const v = (F[(y + 1) * w + clamp(sx - 1, 0, w - 1)] + F[(y + 1) * w + clamp(sx, 0, w - 1)] + F[(y + 1) * w + clamp(sx + 1, 0, w - 1)] + F[(y + 2) * w + clamp(sx, 0, w - 1)]) / 4;
    F[y * w + x] = Math.max(0, v - cool);
  }
  for (let i = 0; i < w * h; i++) buf[i] = FIRE[F[i]];
  return 'fire: the bottom flames are the plays region\'s neurons (what the brain is about to play), the kick throws a burst, association activity lets it climb higher';
}

// ---------- metaballs: one ball per layer and per memory head ----------
function metaballs(X, S) {
  const { w, h, buf } = X, t = S.t * 0.001, balls = [];
  for (let L = 0; L < 4; L++) {               // four layers: present when sounding, size = audio level and FILTER
    if (!S.tokens[L]) continue;
    const r = 11 + 7 * S.level + 5 * (S.filter[L] + 1) / 2;
    balls.push({ x: w / 2 + Math.cos(t * (0.4 + 0.1 * L) + L * 1.6) * w * 0.28, y: h / 2 + Math.sin(t * (0.55 + 0.07 * L) + L) * h * 0.28, r2: r * r, hue: L / 4 });
  }
  for (let hd = 0; hd < 4; hd++) {            // four memory heads: size = how strongly they read
    const r = 3 + 13 * S.headShare[hd];
    balls.push({ x: w / 2 + Math.cos(-t * 0.7 + hd * 1.57) * w * 0.18, y: h / 2 + Math.sin(-t * 0.9 + hd * 1.57) * h * 0.22, r2: r * r, hue: 0.5 + hd / 8 });
  }
  const off = ((S.t * 0.05) | 0) + (S.kickEdge ? 64 : 0);
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {       // computed at half resolution, chunky
    let f = 0;
    for (const b of balls) { const dx = x - b.x, dy = y - b.y; f += b.r2 / (dx * dx + dy * dy + 1); }
    const c = f > 1 ? ACID[(Math.min(150, ((f - 1) * 22) | 0) + off) & 255] : f > 0.9 ? RGB(240, 240, 255) : RGB(6 + 40 * f * f, 4 + 12 * f * f, 14 + 60 * f * f);
    const i = y * w + x; buf[i] = c; buf[i + 1] = c; if (y + 1 < h) { buf[i + w] = c; buf[i + w + 1] = c; }
  }
  return 'metaballs: the big balls are the sounding layers (size = level and FILTER), the small ones the four memory heads (size = read strength); they merge when both are strong';
}

const DEMO = { cadance, plasma, copper, tunnel, vectors, acid, bobs, twister, fire, metaballs };

function demoCaption(g, W, H, D, text) {
  g.font = `${9 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  const words = text.split(' '), lines = []; let cur = '';
  for (const wd of words) { const t = cur ? `${cur} ${wd}` : wd; if (cur && g.measureText(t).width > W - 32 * D) { lines.push(cur); cur = wd; } else cur = t; }
  if (cur) lines.push(cur);
  const top = H - 6 * D - lines.length * 12 * D - 4 * D;
  g.fillStyle = 'rgba(6,6,10,0.55)'; g.fillRect(0, top, W, H - top);
  g.fillStyle = 'rgba(150,158,170,.95)';
  lines.forEach((l, i) => g.fillText(l, 16 * D, H - 6 * D - (lines.length - 1 - i) * 12 * D));
}

// ======================= the 3D brain (WebGL2) =======================
// The whole brain in 3D: every neuron (as the server places them, /api/brain3d: hearing at one end, the code, the
// association hub with prefrontal above it, the rhythm pathway on one side and the melody pathway on the other,
// the plays region at the far end, one double helix per memory head below, the notebook's slots as a ring under
// them) and a sample of its synapses (the strongest of every pathway). Live: a neuron's brightness and size follow
// its activation (every drawn neuron per bar when the page asks for it, else the region's 48 values), glided
// between bars; a synapse glows with its presynaptic activation times |weight| and carries sparks from pre to
// post in proportion; a sweep runs along the signal path once per bar. Drawn on a WebGL2 canvas beneath the 2D
// one (which keeps the captions); no libraries.
//   connectome  the anatomy: neurons first, fine strands, short trails
//   storm       the synapses as a storm of light: sparks, long trails with a feedback zoom, kicks punch through
//   compact     the regions as the lobes of one compact brain (/api/brain3d's compact part), the dense sample's
//               synapses bundled along every pathway, in the trippy style (glow, trails, colours turning with the kick)
//   mri         the compact brain as an MRI: axial, sagittal and coronal slices of a volume made of the network (grey =
//               neurons per voxel, white = the synapse bundles) sweeping slowly, an fMRI-like heat overlay (each voxel's
//               neurons relative to their region's range, smoothed over space and time), lobe names, a glass brain, an
//               EEG strip of the bars' settles
//   mri human   the same viewer with the neurons placed on a human brain's anatomy
const GL_MODES = ['connectome', 'storm', 'compact', 'mri'];      // 'mri human' (a human anatomy) was taken out on the owner's word, 26 Nov
const HEAD_RGB = [[95, 211, 198], [143, 179, 255], [242, 166, 90], [229, 143, 176]];

const M4 = {
  persp(fovy, asp, n, f) {
    const t = 1 / Math.tan(fovy / 2), nf = 1 / (n - f);
    return new Float32Array([t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) * nf, -1, 0, 0, 2 * f * n * nf, 0]);
  },
  look(e, c, u) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], nrm = (a) => { const l = Math.hypot(...a) || 1; return a.map((x) => x / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = nrm(sub(e, c)), x = nrm(cross(u, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[i + 4 * k] * b[k + 4 * j]; o[i + 4 * j] = s; }
    return o;
  },
};
const hash01 = (i) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };

// the bytes /api/brain3d serves (brainview.pack): "B3D1", header length, JSON header, arrays at 4-byte boundaries
function parseBrain3d(buf) {
  const dv = new DataView(buf);
  if (buf.byteLength < 8 || String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'B3D1') throw Error('not a 3D brain');
  const hl = dv.getUint32(4, true), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl))), base = 8 + hl;
  const kinds = { '<f4': Float32Array, '|u1': Uint8Array, '<u2': Uint16Array, '<u4': Uint32Array, '|i1': Int8Array };
  const arr = (name) => { const [off, len, dt] = head.arrays[name], K = kinds[dt]; return new K(buf, base + off, len / K.BYTES_PER_ELEMENT); };
  const opt = (name) => (head.arrays[name] ? arr(name) : null);       // the MRI part (version 2 of the view)
  return { head, pos: arr('position'), region: arr('region'), bin: arr('bin'), pre: arr('pre'), post: arr('post'), weight: arr('weight'),
    mriPos: opt('mri_position'), mriBook: opt('mri_notebook'), anatomy: opt('anatomy'), surface: opt('surface'),
    compactPos: opt('compact_position'), compactBook: opt('compact_notebook'), compactT1: opt('compact_t1'), compactLobes: opt('compact_lobes'),
    densePre: opt('dense_pre'), densePost: opt('dense_post'), denseWeight: opt('dense_weight') };
}

// a calm stand-in when the server has no 3D brain to give (an older server, or no brain was ever loaded): the same
// anatomy in fewer neurons, with made-up wiring; the audio lights it
function placeholderBrain() {
  let s = 7; const rnd = () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const G = Math.PI * (3 - Math.sqrt(5));
  const shape = {
    ball: (i, n, R) => { const y = 1 - (2 * (i + 0.5)) / n, r = Math.sqrt(Math.max(0, 1 - y * y)), th = G * i, k = R * Math.cbrt((i * 0.618034 + 0.5) % 1); return [Math.cos(th) * r * k, y * k, Math.sin(th) * r * k]; },
    coil: (i, n, R) => { const t = (i + 0.5) / n, th = 2 * Math.PI * 2.6 * t, rho = R * (1 - 0.72 * t), tb = 0.17 * R * (1 - 0.5 * t), ph = G * i; return [0.5 * R * t + tb * Math.sin(ph), (rho + tb * Math.cos(ph)) * Math.cos(th), (rho + tb * Math.cos(ph)) * Math.sin(th)]; },
    ring: (i, n, R) => { const u = (2 * Math.PI * (i + 0.5)) / n, v = G * i * 3; return [0.12 * Math.sin(v), (R + 0.12 * Math.cos(v)) * Math.cos(u), (R + 0.12 * Math.cos(v)) * Math.sin(u)]; },
    helix: (i, n, R) => { const t = (i + 0.5) / n, th = 2 * Math.PI * 3.5 * t + (i % 2) * Math.PI, ph = G * i; return [(0.17 + 0.04 * Math.cos(ph)) * Math.cos(th), -0.95 * t, (0.17 + 0.04 * Math.cos(ph)) * Math.sin(th)]; },
  };
  const defs = [['sensory', 'hears', 900, [-2.25, 0.05, 0], 'coil', 0.56, 0], ['code', 'meaning', 900, [-1.2, 0.12, 0], 'ball', 0.5, 0.12],
    ['association', 'association', 120, [0, 0.18, 0], 'ball', 0.18, 0.38], ['prefrontal', 'prefrontal', 80, [0.12, 0.9, 0], 'ball', 0.16, 0.42],
    ['rhythm', 'rhythm pathway', 160, [0.85, 0.1, 0.58], 'ball', 0.22, 0.54], ['melody', 'melody pathway', 160, [0.85, 0.1, -0.58], 'ball', 0.22, 0.54],
    ['motor', 'plays', 600, [1.8, 0.05, 0], 'ring', 0.5, 0.68], ...[0, 1, 2, 3].map((h) => [`recall_${h}`, `memory read ${h + 1}`, 300, [-1 + (2 * h) / 3, -0.72, 0], 'helix', 0.2, 0.88, h])];
  const n = defs.reduce((a, d) => a + d[2], 0), pos = new Float32Array(n * 3), region = new Uint8Array(n), bin = new Uint8Array(n).fill(255), regions = [];
  let at = 0;
  defs.forEach(([name, role, count, c, sh, R, flow, head], r) => {
    regions.push({ name, role, start: at, count, neurons: count, centre: sh === 'helix' ? [c[0], c[1] - 0.47, c[2]] : c, radius: R, flow, head: head ?? -1, snapshot: false });
    for (let i = 0; i < count; i++, at++) { const p = shape[sh](i, count, R); pos.set([p[0] + c[0], p[1] + c[1], p[2] + c[2]], at * 3); region[at] = r; }
  });
  const links = [[0, 1], [1, 2], [3, 2], [2, 4], [2, 5], [4, 6], [5, 6], [7, 2], [8, 2], [9, 2], [10, 2], [3, 1]], pre = [], post = [], weight = [];
  for (const [a, b] of links) for (let k = 0; k < 450; k++) {
    const A = regions[a], B = regions[b];
    pre.push(A.start + ((rnd() * A.count) | 0)); post.push(B.start + ((rnd() * B.count) | 0)); weight.push(((rnd() < 0.25 ? -1 : 1) * (20 + rnd() * 107)) | 0);
  }
  return { head: { checkpoint: null, placeholder: true, drawn: n, neurons: n, drawn_synapses: pre.length, synapses_total: pre.length, regions, heads: 4, capacity: 64,
    notebook: { centre: [0, -2.1, 0], radius: 1.3 }, bounds: { centre: [-0.2, -0.33, 0], radius: 2.35 }, first: 12 },
    pos, region, bin, pre: Uint16Array.from(pre), post: Uint16Array.from(post), weight: Int8Array.from(weight) };
}

const GLSL_HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D u_pos, u_act;
uniform mat4 u_mvp;
uniform vec3 u_centre;
uniform vec4 u_reg[64], u_ctr[64];
uniform float u_mix, u_time, u_phase, u_kick, u_beat, u_hue, u_px, u_gain, u_bend, u_storm, u_hot, u_bundle;
ivec2 at(int i) { return ivec2(i & 255, i >> 8); }
float act(int i) { vec2 a = texelFetch(u_act, at(i), 0).rg; return mix(a.r, a.g, u_mix); }
vec3 hsv(float h, float s, float v) { vec3 k = clamp(abs(fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0); return v * mix(vec3(1.0), k, s); }
float sweep(float flow) { float d = fract(u_phase - flow + 0.5) - 0.5; return exp(-d * d * 110.0); }
float hash(float n) { return fract(sin(n * 12.9898 + 78.233) * 43758.5453); }
// a synapse's path: a quadratic curve bowed away from the centre; with u_bundle (the compact view) its middle is
// pulled toward the axis between the two regions' centres, so a pathway's strands run as one bundle (the server
// voxelises the same paths as the compact MRI's white matter)
vec3 bez(vec4 A, vec4 B, float t) {
  vec3 a = A.xyz, b = B.xyz, m = 0.5 * (a + b), o = m - u_centre; o.y += 0.25;
  vec3 c = m + normalize(o + vec3(1e-4)) * distance(a, b) * u_bend;
  if (u_bundle > 0.0 && int(A.w) != int(B.w)) {
    vec3 axis = 0.5 * (u_ctr[int(A.w)].xyz + u_ctr[int(B.w)].xyz);
    c = mix(c, axis + (axis - u_centre) * 0.08, u_bundle);
  }
  return mix(mix(a, c, t), mix(c, b, t), t);
}
`;
const GL_SRC = {
  neuronV: GLSL_HEAD + `
out vec3 v_col;
void main() {
  int i = gl_VertexID;
  vec4 p = texelFetch(u_pos, at(i), 0);
  vec4 R = u_reg[int(p.w)];
  float a = act(i), s = sweep(R.y);
  vec3 q = p.xyz + (p.xyz - u_centre) * (0.035 * u_kick);
  gl_Position = u_mvp * vec4(q, 1.0);
  float size = u_px * (0.5 + 1.7 * a + 1.1 * s * a + 0.8 * u_kick * a + 0.35 * u_beat * a) / gl_Position.w;
  gl_PointSize = clamp(size, 1.0, 56.0);
  float lit = (0.05 + a * (0.6 + 1.1 * s) + 0.25 * u_beat * a + 0.1 * s) * min(1.0, size * size);
  v_col = hsv(R.x + u_hue + 0.07 * a + 0.04 * s, 0.85 - 0.5 * a * a, 1.0) * lit * u_gain;
  if (u_hot > 0.5) {                     // the glass brain: quiet grey at rest, hot colours with activity
    vec3 hc = clamp(vec3(a * 3.0, a * 3.0 - 1.0, a * 3.0 - 2.0), 0.0, 1.0);
    v_col = mix(vec3(0.045, 0.06, 0.08), hc * (0.45 + a), smoothstep(0.12, 0.45, a)) * min(1.0, size * size) * u_gain;
  }
}`,
  pointF: `#version 300 es
precision highp float;
in vec3 v_col; out vec4 o;
void main() { vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard; o = vec4(v_col * (exp(-r2 * 4.0) + 0.5 * smoothstep(0.3, 0.0, r2)), 1.0); }`,
  strandV: GLSL_HEAD + `
layout(location = 0) in uint a_pre; layout(location = 1) in uint a_post; layout(location = 2) in float a_w; layout(location = 3) in float a_k;
uniform float u_seg, u_base;
out vec3 v_col;
void main() {
  int i = int(a_pre), j = int(a_post);
  vec4 A = texelFetch(u_pos, at(i), 0), B = texelFetch(u_pos, at(j), 0);
  vec4 RA = u_reg[int(A.w)], RB = u_reg[int(B.w)];
  float t = float(gl_VertexID) / u_seg, w = abs(a_w), flow = act(i) * w * w;
  float s = sweep(mix(RA.y, RB.y, t));
  gl_Position = u_mvp * vec4(bez(A, B, t), 1.0);
  float lit = (u_base * (0.3 + w) + flow * (0.04 + 0.32 * s + 0.1 * u_beat) * (0.45 + 0.55 * t)) * a_k;
  v_col = hsv(mix(RA.x, RB.x, t) + u_hue + (a_w < 0.0 ? 0.42 : 0.0), a_w < 0.0 ? 0.5 : 0.85, 1.0) * lit * u_gain;
}`,
  lineF: `#version 300 es
precision highp float;
in vec3 v_col; out vec4 o;
void main() { o = vec4(v_col, 1.0); }`,
  sparkV: GLSL_HEAD + `
layout(location = 0) in uint a_pre; layout(location = 1) in uint a_post; layout(location = 2) in float a_w; layout(location = 3) in float a_k;
uniform float u_k;
out vec3 v_col;
void main() {
  int i = int(a_pre), j = int(a_post);
  float k = float(gl_VertexID), w = abs(a_w), flow = act(i) * w * w;
  float on = smoothstep(k / u_k, k / u_k + 0.12, flow * 2.2 + 0.05 * u_storm);
  if (on < 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; v_col = vec3(0.0); return; }
  vec4 A = texelFetch(u_pos, at(i), 0), B = texelFetch(u_pos, at(j), 0);
  vec4 RA = u_reg[int(A.w)], RB = u_reg[int(B.w)];
  float t = fract(u_time * (0.18 + 0.7 * flow + 0.25 * u_storm) + hash(float(gl_InstanceID)) + k / u_k);
  gl_Position = u_mvp * vec4(bez(A, B, t), 1.0);
  float s = sweep(mix(RA.y, RB.y, t));
  gl_PointSize = clamp(u_px * 0.55 * (0.5 + 1.6 * flow + 0.6 * u_kick) / gl_Position.w, 1.0, 22.0);
  v_col = hsv(mix(RA.x, RB.x, t) + u_hue + 0.12, 0.45, 1.0) * on * (0.12 + 0.9 * flow + 0.6 * s * flow) * sqrt(a_k) * sin(t * 3.14159) * u_gain;
}`,
  dynV: `#version 300 es
precision highp float;
layout(location = 0) in vec3 a_p; layout(location = 1) in vec4 a_c;
uniform mat4 u_mvp; uniform float u_px;
out vec3 v_col;
void main() { gl_Position = u_mvp * vec4(a_p, 1.0); gl_PointSize = clamp(u_px * a_c.a / gl_Position.w, 1.0, 40.0); v_col = a_c.rgb; }`,
  quadV: `#version 300 es
out vec2 v_uv;
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); v_uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`,
  fadeF: `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform float u_decay, u_zoom, u_twist, u_aspect;
in vec2 v_uv; out vec4 o;
void main() {
  vec2 c = (v_uv - 0.5) * vec2(u_aspect, 1.0);
  c = mat2(cos(u_twist), sin(u_twist), -sin(u_twist), cos(u_twist)) * c / u_zoom;
  vec2 uv = c / vec2(u_aspect, 1.0) + 0.5;
  vec3 col = texture(u_tex, uv).rgb * u_decay;
  o = vec4(col * step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0), 1.0);
}`,
  downF: `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_texel; uniform float u_thr;
in vec2 v_uv; out vec4 o;
void main() {
  vec3 c = 0.25 * (texture(u_tex, v_uv + u_texel * vec2(-1.0, -1.0)).rgb + texture(u_tex, v_uv + u_texel * vec2(1.0, -1.0)).rgb
                 + texture(u_tex, v_uv + u_texel * vec2(-1.0, 1.0)).rgb + texture(u_tex, v_uv + u_texel * vec2(1.0, 1.0)).rgb);
  o = vec4(max(c - u_thr, 0.0), 1.0);
}`,
  blurF: `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_dir;
in vec2 v_uv; out vec4 o;
void main() {
  vec3 c = texture(u_tex, v_uv).rgb * 0.227027;
  c += (texture(u_tex, v_uv + u_dir * 1.3846).rgb + texture(u_tex, v_uv - u_dir * 1.3846).rgb) * 0.316216;
  c += (texture(u_tex, v_uv + u_dir * 3.2308).rgb + texture(u_tex, v_uv - u_dir * 3.2308).rgb) * 0.070270;
  o = vec4(c, 1.0);
}`,
  sliceF: `#version 300 es
precision highp float; precision highp sampler3D;
uniform sampler3D u_anat, u_heat;
uniform int u_axis;
uniform vec3 u_cut;
uniform vec2 u_span, u_pix;
uniform float u_mix, u_lvl;
in vec2 v_uv; out vec4 o;
vec3 hot(float h) { return clamp(vec3(h * 3.0, h * 3.0 - 1.0, h * 3.0 - 2.0), 0.0, 1.0); }
void main() {
  // radiological convention: the patient's right on the image's left; axial seen from below with the front up,
  // sagittal with the front on the left, coronal from the front
  vec2 q = (v_uv - 0.5) * u_span + 0.5;
  vec3 p = u_axis == 0 ? vec3(1.0 - q.x, u_cut.y, q.y) : u_axis == 1 ? vec3(u_cut.x, q.y, 1.0 - q.x) : vec3(1.0 - q.x, q.y, u_cut.z);
  float inside = step(0.0, min(min(p.x, p.y), p.z)) * step(max(max(p.x, p.y), p.z), 1.0);
  float g = texture(u_anat, p).r * inside;
  vec3 col = pow(vec3(g), vec3(1.25)) * vec3(0.66, 0.7, 0.76);
  vec2 hh = texture(u_heat, p).rg;
  float h = mix(hh.r, hh.g, u_mix) * inside;
  float a = smoothstep(u_lvl, u_lvl + 0.22, h) * smoothstep(0.08, 0.2, g);
  col = mix(col, hot(0.22 + 0.95 * (h - u_lvl) / (1.0 - u_lvl)), a * 0.9);
  vec2 c = u_axis == 0 ? vec2(1.0 - u_cut.x, u_cut.z) : u_axis == 1 ? vec2(1.0 - u_cut.z, u_cut.y) : vec2(1.0 - u_cut.x, u_cut.y);
  float lx = (1.0 - smoothstep(0.0, 1.2, abs(q.x - c.x) / u_pix.x)) * step(0.5, fract(q.y / u_pix.y / 9.0));
  float ly = (1.0 - smoothstep(0.0, 1.2, abs(q.y - c.y) / u_pix.y)) * step(0.5, fract(q.x / u_pix.x / 9.0));
  col += vec3(0.2, 0.5, 0.55) * 0.3 * max(lx, ly) * inside;
  o = vec4(col, 1.0);
}`,
  outF: `#version 300 es
precision highp float;
uniform sampler2D u_tex, u_bloom; uniform float u_bloomK, u_aber, u_exposure;
in vec2 v_uv; out vec4 o;
void main() {
  vec2 d = (v_uv - 0.5) * u_aber;
  vec3 c = vec3(texture(u_tex, v_uv + d).r, texture(u_tex, v_uv).g, texture(u_tex, v_uv - d).b) + texture(u_bloom, v_uv).rgb * u_bloomK;
  c = 1.0 - exp(-c * u_exposure);
  float vig = smoothstep(1.3, 0.35, length((v_uv - 0.5) * vec2(1.6, 1.25)));
  o = vec4(pow(c, vec3(0.92)) * vig, 1.0);
}`,
};

const shownGL = new Set();               // GL canvases on screen: hidden when their BrainViz stops drawing (another view took over)
function watchGL() {
  const now = performance.now();
  for (const b of shownGL) if (now - b.drawnAt > 120) b.hide();
  if (shownGL.size) requestAnimationFrame(watchGL);
}

class Brain3D {
  constructor(under, compact) {
    this.under = under; this.compact = compact; this.shown = false; this.drawnAt = 0; this.onscreen = true;
    this.cam = { yaw: 0.25, pitch: 0.3, held: 0 }; this.lod = compact ? 0.45 : 1; this.scale = 1; this.ft = 16.7; this.slowFor = 0; this.fastFor = 0; this.lastT = 0;
    this.data = null; this.n = 0; this.prev = null; this.cur = null; this.mixT = 0; this.mixDur = 0.3; this.refs = null; this.trail = 0;
    const cv = this.cv = document.createElement('canvas');
    cv.setAttribute('aria-hidden', 'true');
    Object.assign(cv.style, { position: 'absolute', pointerEvents: 'none', display: 'none', left: '0px', top: '0px' });
    under.parentNode.insertBefore(cv, under);
    try { new IntersectionObserver((es) => { this.onscreen = es[es.length - 1].isIntersecting; }).observe(under); } catch { /* always drawn */ }
    cv.addEventListener('webglcontextlost', (e) => { e.preventDefault(); this.lost = true; });
    cv.addEventListener('webglcontextrestored', () => { this.lost = false; this.init(); if (this.data) this.upload(this.data); });
    this.gl = cv.getContext('webgl2', { antialias: false, alpha: false, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: 'high-performance' });
    if (!this.gl) { this.failed = 'WebGL2 is not available in this browser'; return; }
    this.init();
  }

  init() {
    const gl = this.gl;
    const prog = (vs, fs) => {
      const p = gl.createProgram();
      for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
        const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s));
        gl.attachShader(p, s);
      }
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p));
      const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) { const name = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, ''); u[name] = gl.getUniformLocation(p, name); }
      return { p, u };
    };
    try {
      this.P = {
        neuron: prog(GL_SRC.neuronV, GL_SRC.pointF), strand: prog(GL_SRC.strandV, GL_SRC.lineF), spark: prog(GL_SRC.sparkV, GL_SRC.pointF),
        dyn: prog(GL_SRC.dynV, GL_SRC.pointF), dynLine: prog(GL_SRC.dynV, GL_SRC.lineF), fade: prog(GL_SRC.quadV, GL_SRC.fadeF),
        down: prog(GL_SRC.quadV, GL_SRC.downF), blur: prog(GL_SRC.quadV, GL_SRC.blurF), out: prog(GL_SRC.quadV, GL_SRC.outF),
        slice: prog(GL_SRC.quadV, GL_SRC.sliceF),
      };
    } catch (e) { this.failed = `the 3D shaders did not compile: ${e.message}`; console.warn(this.failed); return; }
    this.half = !!gl.getExtension('EXT_color_buffer_float');
    this.empty = gl.createVertexArray();
    this.synVao = gl.createVertexArray(); this.synBufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
    this.dynVao = gl.createVertexArray(); this.dynBuf = gl.createBuffer();
    gl.bindVertexArray(this.dynVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
    gl.bindVertexArray(null);
    this.posTex = gl.createTexture(); this.actTex = gl.createTexture();
    this.fbo = null; this.size = [0, 0];
  }

  // ---------- data ----------
  upload(d) {
    const gl = this.gl; if (!gl || this.failed) return;
    this.data = d; const n = this.n = d.head.drawn, rows = Math.max(1, Math.ceil(n / 256));
    this.mriSets = {}; this.cbrain = null;           // built at the first frame that needs them
    const pos = new Float32Array(256 * rows * 4);
    for (let i = 0; i < n; i++) { pos[i * 4] = d.pos[i * 3]; pos[i * 4 + 1] = d.pos[i * 3 + 1]; pos[i * 4 + 2] = d.pos[i * 3 + 2]; pos[i * 4 + 3] = Math.min(63, d.region[i]); }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.posTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, rows, 0, gl.RGBA, gl.FLOAT, pos);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    this.actBytes = new Uint8Array(256 * rows * 2); this.rows = rows;
    gl.bindTexture(gl.TEXTURE_2D, this.actTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, 256, rows, 0, gl.RG, gl.UNSIGNED_BYTE, this.actBytes);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    const wide = d.pre instanceof Uint32Array;
    gl.bindVertexArray(this.synVao);
    [[d.pre, 0], [d.post, 1]].forEach(([a, loc]) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.synBufs[loc]); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc); gl.vertexAttribIPointer(loc, 1, wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0, 0); gl.vertexAttribDivisor(loc, 1);
    });
    this.strands(this.synVao, this.synBufs, d.pre, d.post, d.weight);
    gl.bindVertexArray(null);
    const regs = d.head.regions;                    // per region: hue along the signal path, flow, memory head
    this.reg = new Float32Array(64 * 4);
    regs.slice(0, 64).forEach((r, i) => { this.reg.set([(0.55 + 0.85 * r.flow) % 1, r.flow, r.head, r.count], i * 4); });
    this.prev = new Float32Array(n); this.cur = new Float32Array(n); this.refs = new Float32Array(regs.length).fill(0.05);
    this.noise = Float32Array.from({ length: n }, (_, i) => hash01(i));
    this.byName = new Map(regs.map((r, i) => [r.name, i]));
    this.headBottom = [];
    for (const r of regs) if (r.head >= 0 && r.name.startsWith('recall') && !r.name.endsWith('_when')) this.headBottom[r.head] = [r.centre[0], r.centre[1] - 0.5, r.centre[2]];
    this.mixT = performance.now();
  }

  // a synapse list into a vertex array: pre, post (integers), weight, and the bundle factor
  strands(vao, bufs, pre, post, weight) {
    const gl = this.gl, d = this.data, wide = pre instanceof Uint32Array;
    gl.bindVertexArray(vao);
    [[pre, 0], [post, 1]].forEach(([a, loc]) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, bufs[loc]); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc); gl.vertexAttribIPointer(loc, 1, wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0, 0); gl.vertexAttribDivisor(loc, 1);
    });
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs[2]); gl.bufferData(gl.ARRAY_BUFFER, weight, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.BYTE, true, 0, 0); gl.vertexAttribDivisor(2, 1);
    // a pathway drawn with many strands lights each one less (1 / sqrt of its strands), and so does a region many
    // pathways converge on: a big bundle does not burn white, a pathway of a few dozen strands still shows
    const R = d.head.regions.length, count = new Map(), into = new Float32Array(R), key = new Uint32Array(pre.length), bundle = new Uint8Array(pre.length);
    for (let e = 0; e < pre.length; e++) { key[e] = d.region[pre[e]] * R + d.region[post[e]]; count.set(key[e], (count.get(key[e]) || 0) + 1); into[d.region[post[e]]]++; }
    const k1 = 14 * Math.sqrt(pre.length / 48000), k2 = 45 * Math.sqrt(pre.length / 48000);   // the same light per pathway at any sample size
    for (let e = 0; e < pre.length; e++) bundle[e] = 255 * Math.min(1, k1 / Math.sqrt(count.get(key[e])), k2 / Math.sqrt(into[d.region[post[e]]]));
    gl.bindBuffer(gl.ARRAY_BUFFER, bufs[3]); gl.bufferData(gl.ARRAY_BUFFER, bundle, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.UNSIGNED_BYTE, true, 0, 0); gl.vertexAttribDivisor(3, 1);
    gl.bindVertexArray(null);
  }

  // the compact brain's positions, centres and dense sample, at its first frame
  initCompact() {
    const d = this.data, gl = this.gl, h = d.head.compact;
    if (!h || !d.compactPos || !d.densePre) return false;
    const pos = new Float32Array(256 * this.rows * 4);
    for (let i = 0; i < this.n; i++) pos.set([d.compactPos[i * 3], d.compactPos[i * 3 + 1], d.compactPos[i * 3 + 2], Math.min(63, d.region[i])], i * 4);
    const posTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, posTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, this.rows, 0, gl.RGBA, gl.FLOAT, pos);
    for (const q of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, q, gl.NEAREST);
    const vao = gl.createVertexArray(), bufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
    this.strands(vao, bufs, d.densePre, d.densePost, d.denseWeight);
    const ctr = new Float32Array(64 * 4);
    h.lobes.slice(0, 64).forEach((l, i) => ctr.set([...l.centre, 0], i * 4));
    this.cbrain = { posTex, vao, n: d.densePre.length, ctr, head: h };
    return true;
  }

  // new activity targets (0..1 per drawn neuron), glided to over `ease` seconds
  target(values, ease) {
    if (!this.n) return;
    const now = performance.now(), m = this.mixNow(now), p = this.prev, c = this.cur, b = this.actBytes;
    for (let i = 0; i < this.n; i++) {
      const shown = p[i] + (c[i] - p[i]) * m;
      p[i] = shown; c[i] = values[i];
      b[i * 2] = shown * 255 + 0.5; b[i * 2 + 1] = values[i] * 255 + 0.5;
    }
    this.mixT = now; this.mixDur = ease;
    for (const M of Object.values(this.mriSets || {})) if (M && now - M.shownAt < 1000) this.heatFrom(M, values);
    const gl = this.gl; if (!gl || this.failed) return;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_2D, this.actTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, this.rows, gl.RG, gl.UNSIGNED_BYTE, b);
  }
  mixNow(now) { const x = Math.min(1, (now - this.mixT) / 1000 / this.mixDur); return x * x * (3 - 2 * x); }

  // a bar's view: every drawn neuron (view.neurons, when it belongs to this layout), else the region values
  fromView(v, checkpoint) {
    if (!this.n || !v) return false;
    const d = this.data, regs = d.head.regions, out = new Float32Array(this.n);
    let source = 'regions';
    if (v.neurons && checkpoint && checkpoint === d.head.checkpoint) {
      const raw = atob(v.neurons);
      if (raw.length === this.n) {
        source = 'neurons';
        regs.forEach((r, ri) => {
          let mx = 0;
          for (let i = r.start; i < r.start + r.count; i++) { const x = raw.charCodeAt(i) / 255; out[i] = x * x; if (out[i] > mx) mx = out[i]; }
          const ref = this.refs[ri] = Math.max(0.03, mx, this.refs[ri] * 0.85);
          for (let i = r.start; i < r.start + r.count; i++) out[i] = Math.pow(Math.min(1, out[i] / ref), 0.8);
        });
      }
    }
    if (source === 'regions') {
      const vals = new Map((v.regions || []).map((r) => [r.name, r.values]));
      const shares = (v.heads || []).map((reads) => reads.reduce((m, x) => Math.max(m, x.w || 0), 0));
      regs.forEach((r, ri) => {
        const vv = vals.get(r.name);
        if (vv && vv.length) {
          let mx = 0; for (const x of vv) mx = Math.max(mx, Math.abs(x));
          const ref = this.refs[ri] = Math.max(0.03, mx, this.refs[ri] * 0.85);
          for (let i = r.start; i < r.start + r.count; i++) { const b = d.bin[i]; out[i] = Math.pow(Math.min(1, Math.abs(vv[b < vv.length ? b : 0]) / ref), 0.8) * (0.75 + 0.25 * this.noise[i]); }
        } else {                                   // no values (the next-bar ports): the head's read share, speckled
          const s = r.head >= 0 ? Math.min(1, (shares[r.head] || 0) * 1.6) : 0.1;
          for (let i = r.start; i < r.start + r.count; i++) out[i] = s * (this.noise[i] > 0.55 ? this.noise[i] : 0.15);
        }
      });
    }
    this.target(out, 0.45);
    this.source = source;
    return true;
  }

  // no brain view: the audio lights the regions along the signal path (low bands at the hearing end), twinkling per step
  fromAudio(spec, level, step) {
    if (!this.n) return;
    const regs = this.data.head.regions, out = new Float32Array(this.n), nb = spec ? spec.length : 0;
    regs.forEach((r) => {
      let e = level * 0.3;
      if (nb) { const c = Math.min(nb - 3, Math.floor((0.02 + 0.6 * Math.min(1, r.flow)) * nb)); e = (spec[c] + spec[c + 1] + spec[c + 2]) / 765; }
      const lvl = Math.min(1, e * 1.5);
      for (let i = r.start; i < r.start + r.count; i++) { const k = hash01(i * 0.37 + step * 3.1); out[i] = lvl * (k > 0.6 ? k : 0.25 * k); }
    });
    this.target(out, 0.12);
    this.source = 'audio';
  }

  // ---------- camera ----------
  turn(dx, dy) { this.cam.yaw += dx * 0.006; this.cam.pitch = clamp(this.cam.pitch + dy * 0.005, -1.2, 1.2); this.cam.held = performance.now(); }
  matrices(W, H, t, s, bounds) {
    const b = bounds || this.data.head.bounds, c = b.centre, storm = s.storm;
    if (performance.now() - this.cam.held > 2500) {
      this.cam.yaw += s.dt * (storm ? 0.16 : 0.07) * (1 + 1.5 * s.level);
      const want = (storm ? 0.22 : 0.3) + 0.14 * Math.sin(t / 9000);
      this.cam.pitch += (want - this.cam.pitch) * Math.min(1, s.dt * 0.4);
    }
    const asp = W / H, fov = 0.74;
    const fit = b.radius / Math.tan(fov / 2) / Math.min(1, asp * 0.62);      // the bounding sphere, farther in a narrow box
    const dist = fit * (storm ? 0.82 + 0.06 * Math.sin(t / 3100) : 0.92) * (1 - 0.035 * s.kick);
    const y = this.cam.yaw, p = this.cam.pitch;
    const eye = [c[0] + dist * Math.cos(p) * Math.sin(y), c[1] + dist * Math.sin(p), c[2] + dist * Math.cos(p) * Math.cos(y)];
    const sway = storm ? 0.06 * Math.sin(t / 1700) : 0;
    const view = M4.look(eye, c, [Math.sin(sway), Math.cos(sway), 0]);
    this.mvp = M4.mul(M4.persp(fov, asp, 0.1, dist * 3), view);
    this.dist = dist;
    return this.mvp;
  }
  project(p, W, H) {
    const m = this.mvp, x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    return w > 0.05 ? [(x / w * 0.5 + 0.5) * W, (1 - (y / w * 0.5 + 0.5)) * H, w] : null;
  }

  // ---------- frame buffers ----------
  surface(w, h) {
    const gl = this.gl, tex = () => {
      const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, this.half ? gl.RGBA16F : gl.RGBA8, w, h, 0, gl.RGBA, this.half ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      return { t, f, w, h };
    };
    return tex();
  }
  buffers(w, h) {
    if (this.size[0] === w && this.size[1] === h && this.fbo) return;
    const gl = this.gl;
    if (this.fbo) for (const x of this.fbo.all) { gl.deleteTexture(x.t); gl.deleteFramebuffer(x.f); }
    const bw = Math.max(1, w >> 2), bh = Math.max(1, h >> 2);
    const trail = [this.surface(w, h), this.surface(w, h)], bloom = [this.surface(bw, bh), this.surface(bw, bh)];
    this.fbo = { trail, bloom, all: [...trail, ...bloom] }; this.size = [w, h]; this.trail = 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ---------- on screen ----------
  show() {
    const u = this.under, cs = getComputedStyle(u);
    if (!this.kept) this.kept = { position: u.style.position, background: u.style.background };
    if (cs.position === 'static') u.style.position = 'relative';      // stacked above the GL canvas, in tree order
    u.style.background = 'transparent';
    this.cv.style.display = 'block'; this.shown = true;
    if (!shownGL.size) requestAnimationFrame(watchGL);
    shownGL.add(this);
  }
  hide() {
    if (!this.shown) return;
    this.cv.style.display = 'none'; this.shown = false; shownGL.delete(this);
    if (this.kept) { this.under.style.position = this.kept.position; this.under.style.background = this.kept.background; this.kept = null; }
  }
  place() {                                       // the GL canvas over the 2D canvas's content box
    const u = this.under, st = this.cv.style;
    const x = `${u.offsetLeft + u.clientLeft}px`, y = `${u.offsetTop + u.clientTop}px`, w = `${u.clientWidth}px`, h = `${u.clientHeight}px`;
    if (st.left !== x) st.left = x; if (st.top !== y) st.top = y; if (st.width !== w) st.width = w; if (st.height !== h) st.height = h;
    const r = getComputedStyle(u).borderRadius; if (st.borderRadius !== r) st.borderRadius = r;
  }

  // one frame; s: {t, dt, phase, kick, beat, level, storm}
  draw(s) {
    const gl = this.gl;
    this.drawnAt = performance.now();
    if (!this.shown) this.show();
    this.place();
    if (!gl || this.failed || this.lost || !this.n || !this.onscreen || !this.under.clientWidth) return;
    // pace: the interval between frames; slow frames drop synapses and resolution, fast ones bring them back
    const now = this.drawnAt, gap = this.lastT ? now - this.lastT : 16.7; this.lastT = now;
    if (gap < 250) this.ft += (gap - this.ft) * 0.05;
    if (this.ft > 21) { this.slowFor += gap; this.fastFor = 0; } else if (this.ft < 17.6) { this.fastFor += gap; this.slowFor = 0; } else { this.slowFor = this.fastFor = 0; }
    const top = this.compact ? 0.45 : 1;
    if (this.slowFor > 1200) { this.slowFor = 0; if (this.lod > 0.12) this.lod = Math.max(0.12, this.lod * 0.7); else this.scale = Math.max(0.5, this.scale * 0.85); }
    if (this.fastFor > 4000) { this.fastFor = 0; if (this.scale < 1) this.scale = Math.min(1, this.scale / 0.85); else this.lod = Math.min(top, this.lod / 0.8); }
    const D = devicePixelRatio || 1, cw = this.under.clientWidth, ch = this.under.clientHeight;
    let k = D * this.scale; const px = cw * ch * k * k, cap = this.compact ? 0.4e6 : 2.2e6;
    if (px > cap) k *= Math.sqrt(cap / px);
    const W = Math.max(2, Math.round(cw * k)), H = Math.max(2, Math.round(ch * k));
    if (this.cv.width !== W || this.cv.height !== H) { this.cv.width = W; this.cv.height = H; }
    if (s.mri) { this.drawMRI(s, W, H); return; }
    this.buffers(W, H);
    const C = s.compact && (this.cbrain || (this.initCompact() && this.cbrain));
    if (s.compact && !C) { this.compactMissing = true; s = { ...s, compact: false }; }
    const b = C ? C.head.bounds : this.data.head.bounds, mvp = this.matrices(W, H, s.t, s, b);
    const storm = s.storm ? 1 : 0, wild = storm || C ? 1 : 0, hueT = s.t * (storm ? 0.00004 : C ? 0.00003 : 0.000015) + (this.hueKick || 0);
    if (s.kickEdge && wild) this.hueKick = ((this.hueKick || 0) + (C ? 0.035 : 0.06)) % 1;
    const mix = this.mixNow(now);
    const common = (P) => {
      const u = P.u;
      gl.uniformMatrix4fv(u.u_mvp, false, mvp);
      if (u.u_pos) { gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, C ? C.posTex : this.posTex); gl.uniform1i(u.u_pos, 0); }
      if (u.u_ctr && C) gl.uniform4fv(u.u_ctr, C.ctr);
      if (u.u_act) { gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.actTex); gl.uniform1i(u.u_act, 1); }
      if (u.u_reg) gl.uniform4fv(u.u_reg, this.reg);
      if (u.u_centre) gl.uniform3fv(u.u_centre, b.centre);
      const set = { u_mix: mix, u_time: s.t / 1000, u_phase: s.phase, u_kick: s.kick, u_beat: s.beat, u_hue: hueT, u_px: H * 0.0032 * this.dist,
        u_gain: storm ? 0.27 : C ? 0.26 : 0.75, u_bend: C ? C.head.bend : storm ? 0.42 : 0.28, u_storm: storm, u_bundle: C ? C.head.bundle : 0, u_hot: 0 };
      for (const [k2, v] of Object.entries(set)) if (u[k2]) gl.uniform1f(u[k2], v);
    };
    const [src, dst] = [this.fbo.trail[this.trail], this.fbo.trail[1 - this.trail]];
    // 1. the trail: last frame, faded (and in the storm zoomed and turned), the new scene added on top
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.f); gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND); gl.bindVertexArray(this.empty);
    gl.useProgram(this.P.fade.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src.t); gl.uniform1i(this.P.fade.u.u_tex, 0);
    gl.uniform1f(this.P.fade.u.u_decay, storm ? 0.84 + 0.05 * s.level : C ? 0.74 + 0.06 * s.level : 0.6);
    gl.uniform1f(this.P.fade.u.u_zoom, storm ? 1.006 + 0.03 * s.kick : C ? 1.0 + 0.012 * s.kick : 1.0);
    gl.uniform1f(this.P.fade.u.u_twist, storm ? 0.0025 + 0.01 * s.kick : C ? 0.004 * s.kick : 0);
    gl.uniform1f(this.P.fade.u.u_aspect, W / H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    const hd = this.data.head, total = C ? C.n : this.data.pre.length, nSyn = Math.min(total, Math.max((hd.pathways || 64) * (hd.first || 12), Math.round(total * this.lod)));
    gl.useProgram(this.P.strand.p); common(this.P.strand);
    gl.uniform1f(this.P.strand.u.u_seg, storm ? 8 : 10); gl.uniform1f(this.P.strand.u.u_base, storm ? 0.02 : C ? 0.05 : 0.05);
    gl.bindVertexArray(C ? C.vao : this.synVao);
    gl.drawArraysInstanced(gl.LINE_STRIP, 0, (storm ? 8 : 10) + 1, nSyn);
    gl.useProgram(this.P.spark.p); common(this.P.spark);
    const K = storm ? 4 : C ? 3 : 2; gl.uniform1f(this.P.spark.u.u_k, K);
    gl.drawArraysInstanced(gl.POINTS, 0, K, nSyn);
    gl.bindVertexArray(this.empty);
    gl.useProgram(this.P.neuron.p); common(this.P.neuron);
    if (storm || C) gl.uniform1f(this.P.neuron.u.u_gain, storm ? 0.3 : 0.4);   // the compact view: the bundles lead
    gl.drawArrays(gl.POINTS, 0, this.n);
    if (C) this.drawSlots(s, mvp, H, this.data.compactBook); else this.drawNotebook(s, mvp, H);
    // 2. bloom: a quarter-size copy of the bright parts, blurred twice each way
    const [ba, bb] = this.fbo.bloom;
    gl.disable(gl.BLEND); gl.bindVertexArray(this.empty);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ba.f); gl.viewport(0, 0, ba.w, ba.h);
    gl.useProgram(this.P.down.p); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dst.t); gl.uniform1i(this.P.down.u.u_tex, 0);
    gl.uniform2f(this.P.down.u.u_texel, 1 / W, 1 / H); gl.uniform1f(this.P.down.u.u_thr, storm ? 0.25 : 0.25);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(this.P.blur.p); gl.uniform1i(this.P.blur.u.u_tex, 0);
    for (const spread of [1, 2.2]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, bb.f); gl.bindTexture(gl.TEXTURE_2D, ba.t); gl.uniform2f(this.P.blur.u.u_dir, spread / ba.w, 0); gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, ba.f); gl.bindTexture(gl.TEXTURE_2D, bb.t); gl.uniform2f(this.P.blur.u.u_dir, 0, spread / ba.h); gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    // 3. out: trail + bloom, the kick splits the colours, tone mapped, vignette
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
    gl.useProgram(this.P.out.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dst.t); gl.uniform1i(this.P.out.u.u_tex, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, ba.t); gl.uniform1i(this.P.out.u.u_bloom, 1);
    gl.uniform1f(this.P.out.u.u_bloomK, (storm ? 1.2 : 0.8) * (1 + 0.8 * s.kick));
    gl.uniform1f(this.P.out.u.u_aber, (storm ? 0.03 : 0.008) * s.kick);
    gl.uniform1f(this.P.out.u.u_exposure, 1.0 + 0.3 * s.kick);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.trail = 1 - this.trail;
    this.drawn = { synapses: nSyn, neurons: this.n, W, H };
    if (now - (this.reported || 0) > 1000) {         // for a look from the console or a test: pace, detail, source
      this.reported = now;
      (window.__brain3d ||= {})[this.compact ? 'mini' : 'main'] = { frameMs: +this.ft.toFixed(2), lod: +this.lod.toFixed(2), scale: +this.scale.toFixed(2),
        synapses: nSyn, neurons: this.n, size: [W, H], source: this.source || null, checkpoint: this.data.head.checkpoint, half: this.half, mode: C ? 'compact' : storm ? 'storm' : 'connectome' };
    }
  }

  // the notebook: one slot per bar it keeps, a ring that turns one slot per bar (newest in front); the four heads'
  // reads as threads from the slots they read to their helix, a spark running up each
  drawNotebook(s, mvp, H) {
    const hd = this.data.head, nb = hd.notebook, cap = Math.min(256, hd.capacity || 0);
    if (!nb || !cap) return;
    const gl = this.gl, pts = [], lines = [], hist = s.history || [], c = nb.centre || nb.center, R = nb.radius;
    const slot = (age) => { const a = Math.PI / 2 - ((age + s.phase) / cap) * Math.PI * 2; return [c[0] + Math.cos(a) * R, c[1], c[2] + Math.sin(a) * R]; };
    for (let age = 0; age < cap; age++) {
      const bar = hist[hist.length - 1 - age], prev = hist[hist.length - 2 - age];
      const changed = bar && prev && bar.tokens.some((x, l) => x !== prev.tokens[l]);
      const k = bar ? (age === 0 ? 1.4 : changed ? 0.9 : 0.5) * (1 - 0.5 * age / cap) : 0.12;
      const col = hsv01(((0.3 + age / cap * 0.4 + s.hue) % 1), 0.6, k);
      pts.push(...slot(age), col[0], col[1], col[2], age === 0 ? 5 + 3 * s.beat : changed ? 3.2 : 2.4);
    }
    (s.heads || []).forEach((reads, h) => {
      const end = this.headBottom[h]; if (!end) return;
      for (const r of reads) {
        if (r.age == null || r.age < 0 || r.age >= cap || r.w < 0.03) continue;
        const a = slot(r.age), rgb = HEAD_RGB[h % 4].map((x) => (x / 255) * Math.min(1, 0.35 + r.w * 1.5));
        const mid = [(a[0] + end[0]) / 2, (a[1] + end[1]) / 2 - 0.25, (a[2] + end[2]) / 2];
        let last = a;
        for (let q = 1; q <= 12; q++) {
          const t = q / 12, u = 1 - t, p = [u * u * a[0] + 2 * u * t * mid[0] + t * t * end[0], u * u * a[1] + 2 * u * t * mid[1] + t * t * end[1], u * u * a[2] + 2 * u * t * mid[2] + t * t * end[2]];
          lines.push(...last, ...rgb, 1, ...p, ...rgb, 1); last = p;
        }
        const t = (s.t / 1000 * 0.6 + h * 0.25) % 1, u = 1 - t;
        pts.push(u * u * a[0] + 2 * u * t * mid[0] + t * t * end[0], u * u * a[1] + 2 * u * t * mid[1] + t * t * end[1], u * u * a[2] + 2 * u * t * mid[2] + t * t * end[2], rgb[0] * 2, rgb[1] * 2, rgb[2] * 2, 5);
      }
    });
    const all = new Float32Array(pts.length + lines.length); all.set(pts); all.set(lines, pts.length);
    gl.bindVertexArray(this.dynVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf); gl.bufferData(gl.ARRAY_BUFFER, all, gl.STREAM_DRAW);
    gl.useProgram(this.P.dyn.p); gl.uniformMatrix4fv(this.P.dyn.u.u_mvp, false, mvp); gl.uniform1f(this.P.dyn.u.u_px, H * 0.0032 * this.dist);
    gl.drawArrays(gl.POINTS, 0, pts.length / 7);
    if (lines.length) { gl.useProgram(this.P.dynLine.p); gl.uniformMatrix4fv(this.P.dynLine.u.u_mvp, false, mvp); gl.drawArrays(gl.LINES, pts.length / 7, lines.length / 7); }
    gl.bindVertexArray(null);
  }
}
// ---------- the MRI views (modes mri and mri human) ----------
// kind 'cadence': the compact brain, its volume made of the network; kind 'human': the neurons on a human anatomy
Brain3D.prototype.initMRI = function (kind) {
  const d = this.data, gl = this.gl, cad = kind === 'cadence';
  const m = cad ? d.head.compact : d.head.mri, P = cad ? d.compactPos : d.mriPos, bookPos = cad ? d.compactBook : d.mriBook, vol = cad ? d.compactT1 : d.anatomy;
  if (!m || !P || !vol) return null;
  const [nx, ny, nz] = m.shape, tex3 = (fmt, kindOf, w, h, dd, data) => {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_3D, t);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, fmt, w, h, dd, 0, kindOf, gl.UNSIGNED_BYTE, data);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_3D, k, v);
    return t;
  };
  // the heat volume at half the volume's resolution (about 3.3 mm voxels in the human brain); every drawn neuron and
  // every notebook slot in its voxel; in the Cadence brain every heat voxel knows its lobe
  const f = 2, hx = Math.ceil(nx / f), hy = Math.ceil(ny / f), hz = Math.ceil(nz / f), hv = m.voxel * f, [[x0], [y0], [z0]] = m.box;
  const cell = (x, y, z) => { const i = Math.floor((x - x0) / hv), j = Math.floor((y - y0) / hv), k = Math.floor((z - z0) / hv); return i < 0 || j < 0 || k < 0 || i >= hx || j >= hy || k >= hz ? -1 : (k * hy + j) * hx + i; };
  const idx = new Int32Array(this.n);
  for (let i = 0; i < this.n; i++) idx[i] = cell(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]);
  const nb = bookPos ? bookPos.length / 3 : 0, book = new Int32Array(nb);
  for (let j = 0; j < nb; j++) book[j] = cell(bookPos[j * 3], bookPos[j * 3 + 1], bookPos[j * 3 + 2]);
  const N = hx * hy * hz, lobeOf = cad && d.compactLobes ? new Uint8Array(N) : null;
  if (lobeOf) for (let k = 0; k < hz; k++) for (let j = 0; j < hy; j++) for (let i = 0; i < hx; i++)
    lobeOf[(k * hy + j) * hx + i] = d.compactLobes[(Math.min(nz - 1, k * f) * ny + Math.min(ny - 1, j * f)) * nx + Math.min(nx - 1, i * f)];
  const pos = new Float32Array(256 * this.rows * 4);
  for (let i = 0; i < this.n; i++) pos.set([P[i * 3], P[i * 3 + 1], P[i * 3 + 2], Math.min(63, d.region[i])], i * 4);
  const posTex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, posTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, this.rows, 0, gl.RGBA, gl.FLOAT, pos);
  for (const q of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, q, gl.NEAREST);
  let surface = cad ? null : d.surface;
  if (cad && d.compactLobes) {                      // the compact brain's surface: lobe voxels beside the outside
    const L = d.compactLobes, out = [], at = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 255 : L[(k * ny + j) * nx + i]);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (at(i, j, k) === 255) continue;
      if (at(i - 1, j, k) === 255 || at(i + 1, j, k) === 255 || at(i, j - 1, k) === 255 || at(i, j + 1, k) === 255 || at(i, j, k - 1) === 255 || at(i, j, k + 1) === 255)
        out.push(x0 + (i + 0.5) * m.voxel, y0 + (j + 0.5) * m.voxel, z0 + (k + 0.5) * m.voxel);
    }
    const keep = [];                                // every third surface voxel: a glass, not a shell
    for (let i = 0; i < out.length / 3; i += 3) keep.push(out[i * 3], out[i * 3 + 1], out[i * 3 + 2]);
    surface = Float32Array.from(keep);
  }
  const sn = surface ? surface.length / 3 : 0, surf = new Float32Array(sn * 7);
  const sc = cad ? 0.6 : 1;
  for (let i = 0; i < sn; i++) surf.set([surface[i * 3], surface[i * 3 + 1], surface[i * 3 + 2], 0.05 * sc, 0.065 * sc, 0.085 * sc, 1.3], i * 7);
  const surfVao = gl.createVertexArray(), surfBuf = gl.createBuffer();
  gl.bindVertexArray(surfVao); gl.bindBuffer(gl.ARRAY_BUFFER, surfBuf); gl.bufferData(gl.ARRAY_BUFFER, surf, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 28, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 12);
  gl.bindVertexArray(null);
  const set = { kind, m, hx, hy, hz, N, idx, book, bookPos, posTex, surfVao, sn, anat: tex3(gl.R8, gl.RED, nx, ny, nz, vol),
    heat: tex3(gl.RG8, gl.RG, hx, hy, hz, new Uint8Array(N * 2)), sum: new Float32Array(N), cnt: new Float32Array(N), tmp: new Float32Array(N),
    prev: new Float32Array(N), next: new Float32Array(N), bytes: new Uint8Array(N * 2), ref: 0.1, mixT: 0, mixDur: 1.2, shownAt: performance.now(), bar: 0,
    lobes: cad ? d.compactLobes : null, lobeOf, lo: new Float32Array(256), hi: new Float32Array(256) };
  this.mriSets[kind] = set;
  this.heatFrom(set, this.cur);
  return set;
};
// the heat. Human: per voxel the mean activity of its neurons, smoothed (two [1 2 1] passes per axis), scaled by the
// brightest voxel. Cadence brain: per voxel the mean activity of the neurons near it (values and counts smoothed
// alike, so a voxel between neurons takes their mean, not a dilution), then every lobe's voxels over that lobe's own
// range this bar (lowest 0, highest 1), so every region shows its pattern. Both glide over about a second, like a
// BOLD signal; the notebook's slots by how recent their bar is
Brain3D.prototype.heatFrom = function (M, values) {
  if (!M) return;
  const { hx, hy, hz, N, idx, book, sum, cnt, tmp } = M, cad = M.kind === 'cadence';
  sum.fill(0); cnt.fill(0);
  for (let i = 0; i < this.n; i++) { const k = idx[i]; if (k >= 0) { sum[k] += values[i]; cnt[k]++; } }
  for (let j = 0; j < book.length; j++) {
    const k = book[j]; if (k < 0) continue;
    const age = ((M.bar - j) % book.length + book.length) % book.length;
    sum[k] += Math.exp(-age / 6); cnt[k]++;
  }
  const pass = (a, b, stride, len) => {            // [1 2 1] / 4 along one axis
    for (let k = 0; k < N; k++) {
      const c = Math.floor(k / stride) % len;
      b[k] = (2 * a[k] + (c > 0 ? a[k - stride] : a[k]) + (c < len - 1 ? a[k + stride] : a[k])) / 4;
    }
  };
  const blur = (a) => { for (let r = 0; r < 2; r++) { pass(a, tmp, 1, hx); pass(tmp, a, hx, hy); pass(a, tmp, hx * hy, hz); a.set(tmp); } };
  if (cad) {
    blur(sum); blur(cnt);
    const { lobeOf, lo, hi } = M;
    lo.fill(Infinity); hi.fill(-Infinity);
    for (let k = 0; k < N; k++) {
      sum[k] = cnt[k] > 0.02 ? sum[k] / cnt[k] : 0;
      const l = lobeOf ? lobeOf[k] : 0; if (l === 255 || cnt[k] <= 0.02) continue;
      if (sum[k] < lo[l]) lo[l] = sum[k]; if (sum[k] > hi[l]) hi[l] = sum[k];
    }
    for (let k = 0; k < N; k++) {
      const l = lobeOf ? lobeOf[k] : 0;
      if (l === 255 || cnt[k] <= 0.02) { sum[k] = 0; continue; }
      const span = hi[l] - lo[l];
      sum[k] = span > 1e-4 ? (sum[k] - lo[l]) / span : 0;
    }
    M.ref = 1;
  } else {
    for (let k = 0; k < N; k++) sum[k] = cnt[k] ? sum[k] / cnt[k] : 0;
    blur(sum);
    let mx = 0; for (let k = 0; k < N; k++) if (sum[k] > mx) mx = sum[k];
    M.ref = Math.max(0.04, mx, M.ref * 0.85);
  }
  const now = performance.now(), x = Math.min(1, (now - M.mixT) / 1000 / M.mixDur), mixed = x * x * (3 - 2 * x);
  for (let k = 0; k < N; k++) {
    const shown = M.prev[k] + (M.next[k] - M.prev[k]) * mixed, v = Math.min(1, sum[k] / M.ref);
    M.prev[k] = shown; M.next[k] = v; M.bytes[k * 2] = shown * 255 + 0.5; M.bytes[k * 2 + 1] = v * 255 + 0.5;
  }
  M.mixT = now;
  const gl = this.gl;
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.bindTexture(gl.TEXTURE_3D, M.heat);
  gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, hx, hy, hz, gl.RG, gl.UNSIGNED_BYTE, M.bytes);
};
// s.mri = { kind, panels: [{kind, x, y, w, h}] as fractions of the box (y from the top), cuts: [x, y, z] in 0..1 }
Brain3D.prototype.drawMRI = function (s, W, H) {
  const gl = this.gl, kind = s.mri.kind;
  const M = this.mriSets[kind] || (this.mriSets[kind] === undefined ? this.initMRI(kind) : null);
  if (!M) {
    this.mriSets[kind] = null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); gl.clearColor(0.012, 0.014, 0.018, 1); gl.clear(gl.COLOR_BUFFER_BIT); return;
  }
  const m = M.m, now = performance.now();
  if (now - M.shownAt > 1000) this.heatFrom(M, this.cur);    // back to this view: the heat of the activity as it is
  M.shownAt = now;
  M.bar = s.bar || 0;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.disable(gl.BLEND);
  gl.viewport(0, 0, W, H); gl.clearColor(0.012, 0.014, 0.018, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  const E = m.box.map(([a, b]) => b - a), heatMix = Math.min(1, (now - M.mixT) / 1000 / M.mixDur), hm = heatMix * heatMix * (3 - 2 * heatMix);
  for (const p of s.mri.panels) {
    const px = Math.round(p.x * W), pw = Math.round(p.w * W), ph = Math.round(p.h * H), py = Math.round(H - (p.y + p.h) * H);
    if (pw < 4 || ph < 4) continue;
    gl.viewport(px, py, pw, ph); gl.enable(gl.SCISSOR_TEST); gl.scissor(px, py, pw, ph);
    if (p.kind === 'glass') { this.drawGlass(s, pw, ph, M); continue; }
    const axis = { axial: 0, sagittal: 1, coronal: 2 }[p.kind], span = sliceSpan(axis, pw, ph, E);
    const P = this.P.slice, u = P.u;
    gl.useProgram(P.p); gl.bindVertexArray(this.empty);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, M.anat); gl.uniform1i(u.u_anat, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_3D, M.heat); gl.uniform1i(u.u_heat, 1);
    gl.uniform1i(u.u_axis, axis); gl.uniform3fv(u.u_cut, s.mri.cuts); gl.uniform2f(u.u_span, span[0], span[1]);
    gl.uniform2f(u.u_pix, span[0] / pw, span[1] / ph); gl.uniform1f(u.u_mix, hm); gl.uniform1f(u.u_lvl, M.kind === 'cadence' ? 0.28 : 0.3);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  gl.disable(gl.SCISSOR_TEST); gl.viewport(0, 0, W, H);
  if (now - (this.reported || 0) > 1000) {
    this.reported = now;
    let inside = 0, lit = 0;
    if (M.lobeOf) for (let k = 0; k < M.N; k++) if (M.lobeOf[k] !== 255) { inside++; if (M.next[k] > 0.28) lit++; }
    (window.__brain3d ||= {})[this.compact ? 'mini' : 'main'] = { mode: `mri ${kind}`, frameMs: +this.ft.toFixed(2), size: [W, H], neurons: this.n,
      heat: [M.hx, M.hy, M.hz], lit: inside ? +(lit / inside).toFixed(2) : null, source: this.source || null, checkpoint: this.data.head.checkpoint };
  }
};
// the slice's span in texture units along its two image axes, so the volume keeps its proportions in the panel
function sliceSpan(axis, pw, ph, E) {
  const [eA, eB] = [[E[0], E[2]], [E[2], E[1]], [E[0], E[1]]][axis], scale = Math.max(eA / pw, eB / ph) * 1.04;
  return [pw * scale / eA, ph * scale / eB];
}
// the glass brain: the brain's surface as faint points, every neuron at its place (hot with activity), the three
// slice planes and the notebook's slots; turning slowly (a drag turns it)
Brain3D.prototype.drawGlass = function (s, pw, ph, M) {
  const gl = this.gl, m = M.m, c = m.box.map(([a, b]) => (a + b) / 2);
  const asp = pw / ph, fov = 0.7, dist = (M.kind === 'cadence' ? 1.25 : 1.12) / Math.tan(fov / 2) / Math.min(1, asp * 0.95);
  const yaw = this.cam.yaw + s.t * 0.00012, p = 0.18;
  const eye = [c[0] + dist * Math.cos(p) * Math.sin(yaw), c[1] + dist * Math.sin(p), c[2] + dist * Math.cos(p) * Math.cos(yaw)];
  const mvp = M4.mul(M4.persp(fov, asp, 0.05, dist * 3), M4.look(eye, c, [0, 1, 0]));
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  gl.useProgram(this.P.dyn.p); gl.uniformMatrix4fv(this.P.dyn.u.u_mvp, false, mvp); gl.uniform1f(this.P.dyn.u.u_px, ph * 0.0028 * dist);
  gl.bindVertexArray(M.surfVao); gl.drawArrays(gl.POINTS, 0, M.sn);
  const P = this.P.neuron, u = P.u;
  gl.useProgram(P.p); gl.bindVertexArray(this.empty);
  gl.uniformMatrix4fv(u.u_mvp, false, mvp);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, M.posTex); gl.uniform1i(u.u_pos, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.actTex); gl.uniform1i(u.u_act, 1);
  gl.uniform4fv(u.u_reg, this.reg); gl.uniform3fv(u.u_centre, c);
  for (const [k, v] of [['u_mix', this.mixNow(performance.now())], ['u_time', s.t / 1000], ['u_phase', 0.5], ['u_kick', 0], ['u_beat', 0], ['u_hue', 0],
    ['u_px', ph * 0.0024 * dist], ['u_gain', 0.55], ['u_hot', 1]]) if (u[k]) gl.uniform1f(u[k], v);
  gl.drawArrays(gl.POINTS, 0, this.n);
  gl.uniform1f(u.u_hot, 0);
  // the slice planes (outlines) and the notebook's slots
  const [[x0, x1], [y0, y1], [z0, z1]] = m.box, [cx, cy, cz] = s.mri.cuts.map((f, i) => m.box[i][0] + f * (m.box[i][1] - m.box[i][0]));
  const lines = [], col = [0.12, 0.3, 0.33, 1];
  const quad = (a, b, cc, d) => { for (const [e, f] of [[a, b], [b, cc], [cc, d], [d, a]]) lines.push(...e, ...col, ...f, ...col); };
  quad([x0, cy, z0], [x1, cy, z0], [x1, cy, z1], [x0, cy, z1]);
  quad([cx, y0, z0], [cx, y1, z0], [cx, y1, z1], [cx, y0, z1]);
  quad([x0, y0, cz], [x1, y0, cz], [x1, y1, cz], [x0, y1, cz]);
  const pts = [], book = M.bookPos, nb = book ? book.length / 3 : 0;
  for (let j = 0; j < nb; j++) {
    const age = ((M.bar - j) % nb + nb) % nb, k = Math.exp(-age / 6);
    pts.push(book[j * 3], book[j * 3 + 1], book[j * 3 + 2], 0.1 + 0.9 * k, 0.08 + 0.5 * k * k, 0.05, 1.6 + 2.5 * k);
  }
  const all = new Float32Array(pts.length + lines.length); all.set(pts); all.set(lines, pts.length);
  gl.bindVertexArray(this.dynVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf); gl.bufferData(gl.ARRAY_BUFFER, all, gl.STREAM_DRAW);
  gl.useProgram(this.P.dyn.p);
  if (pts.length) gl.drawArrays(gl.POINTS, 0, pts.length / 7);
  gl.useProgram(this.P.dynLine.p); gl.uniformMatrix4fv(this.P.dynLine.u.u_mvp, false, mvp);
  gl.drawArrays(gl.LINES, pts.length / 7, lines.length / 7);
  gl.bindVertexArray(null); gl.disable(gl.BLEND);
};
// a lobe's short name: the region's role, the memory ports by head
function lobeLabel(l) {
  const n = l.name || '', h = n.startsWith('recall') || n.startsWith('next') ? (/_(\d+)/.exec(n) ? +/_(\d+)/.exec(n)[1] : 0) : -1;
  if (h >= 0) return `${n.startsWith('recall') ? 'read' : 'next'} ${h + 1}${n.endsWith('_when') ? ' time' : ''}`;
  return l.role || n;
}
// the notebook's slots as points where a layout placed them (the compact brain), the newest bar's slot brightest
Brain3D.prototype.drawSlots = function (s, mvp, H, book) {
  const gl = this.gl, nb = book ? book.length / 3 : 0;
  if (!nb) return;
  const pts = new Float32Array(nb * 7), bar = s.bar || 0;
  for (let j = 0; j < nb; j++) {
    const age = ((bar - j) % nb + nb) % nb, k = Math.exp(-age / 8), c = hsv01((0.3 + 0.3 * age / nb + s.hue) % 1, 0.6, 0.25 + 0.9 * k);
    pts.set([book[j * 3], book[j * 3 + 1], book[j * 3 + 2], c[0], c[1], c[2], 1.5 + 3 * k + 2 * k * s.beat], j * 7);
  }
  gl.bindVertexArray(this.dynVao); gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf); gl.bufferData(gl.ARRAY_BUFFER, pts, gl.STREAM_DRAW);
  gl.useProgram(this.P.dyn.p); gl.uniformMatrix4fv(this.P.dyn.u.u_mvp, false, mvp); gl.uniform1f(this.P.dyn.u.u_px, H * 0.0032 * this.dist);
  gl.drawArrays(gl.POINTS, 0, nb);
  gl.bindVertexArray(null);
};
const hsv01 = (h, s, v) => hue(h, s, v).map((x) => x / 255);

// ======================= the brain view =======================
const BRAIN_MODES = ['orbit', 'flow', 'memory', 'scope', 'matrix', 'timeline', 'decision', 'settle', ...GL_MODES];
const MODES = [...DEMO_MODES, ...BRAIN_MODES];
export const MODE_GROUPS = { demo: DEMO_MODES, brain: BRAIN_MODES };

export class BrainViz {
  // opts.compact: a small always-visible box (devices view): the same pictures without captions
  constructor(canvas, opts = {}) {
    this.cv = canvas; this.g = canvas.getContext('2d'); this.compact = !!opts.compact;
    this.captions = opts.captions !== false;     // false: the pictures alone (the radio page)
    this.scroller = opts.scroller || null;       // the demo scroller's text, round and round, in place of the bar's news (the radio page)
    this.mode = 0; this.layout = null; this.view = null; this.arousal = null; this.checkpoint = null;
    this.regions = new Map();          // name -> {disp, target, mean, dmean, max}
    this.heads = [[], [], [], []];     // displayed reads per head: [{age, w, dw}]
    this.apical = []; this.apMax = 0.3;
    this.history = []; this.bar = 0; this.barT = 0;
    this.kick = 0; this.tick = 0; this.flash = 0; this.flashSign = 0; this.ripples = []; this.heat = 0;
    this.last = performance.now();
    this.screen = new DemoScreen(); if (this.scroller) this.screen.scroll.text = this.scroller; this.level = 0; this.lastKickT = -1e9; this.kickEdge = false; this.stepIdx = 0;
    this.now = null; this.news = []; this.meanMax = new Map(); this.an = null; this.wave = null; this.spec = null;
    this.matrixRows = []; this.settleHist = []; this.mxCanvas = null;
    this.gl3 = null; this.b3d = { key: undefined, state: 'idle', busy: false, next: 0, note: '' }; this.lastStep = 0; this.drag = null;
    this.eeg = []; this.eegAudio = []; this.eegRef = 0; this.eegPrevVals = null;      // the MRI view's EEG: one record per bar
    this.resize(); addEventListener('resize', () => this.resize());
    // the 3D modes turn with a drag (the full view only: the small box opens the full view on a click)
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.is3d || this.compact || e.button !== 0) return;
      this.drag = { x: e.clientX, y: e.clientY }; try { canvas.setPointerCapture(e.pointerId); } catch { /* released */ }
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!this.drag || !this.gl3) return;
      this.gl3.turn(e.clientX - this.drag.x, e.clientY - this.drag.y); this.drag = { x: e.clientX, y: e.clientY };
    });
    for (const ev of ['pointerup', 'pointercancel']) canvas.addEventListener(ev, () => { this.drag = null; });
  }
  get is3d() { return GL_MODES.includes(MODES[this.mode]); }

  // the audio: an analyser beside the speakers (level for the demo modes, spectrum when no brain is loaded)
  listen(ctx, node) {
    try {
      this.an = ctx.createAnalyser(); this.an.fftSize = 1024; this.an.smoothingTimeConstant = 0.6;
      node.connect(this.an); this.wave = new Float32Array(this.an.fftSize); this.spec = new Uint8Array(this.an.frequencyBinCount);
    } catch { this.an = null; }
  }

  get modeName() { return MODES[this.mode]; }
  nextMode() { this.setMode(this.mode + 1); }
  setMode(name) { if (name === 'mri human') name = 'mri'; const i = typeof name === 'number' ? name : MODES.indexOf(name); this.mode = i >= 0 ? i % MODES.length : 0; }

  resize() {
    const r = this.cv.getBoundingClientRect(), D = devicePixelRatio || 1;
    this.cv.width = Math.max(1, Math.round(r.width * D)); this.cv.height = Math.max(1, Math.round(r.height * D));
  }

  setLayout(layout) { this.layout = layout || null; this.regions.clear(); }

  // a bar starts playing: the brain's moment that chose it
  setBar(now, history) {
    this.news.push(this.newsOf(now, history));
    if (this.news.length > 3) this.news.shift();
    this.now = now;
    this.history = history; this.bar = now.bar || 0; this.barT = performance.now();
    this.arousal = now.arousal || null; this.checkpoint = now.checkpoint || null;
    const v = now.view;
    this.view = v || null;
    if (this.gl3 && this.gl3.n && v && this.is3d) this.gl3.fromView(v, now.checkpoint);
    if (this.is3d) this.eegBar(now);
    if (!v) return;
    for (const r of v.regions || []) {
      let s = this.regions.get(r.name);
      if (!s || s.disp.length !== r.values.length) {
        s = { disp: Float32Array.from(r.values), target: Float32Array.from(r.values), mean: r.mean, dmean: r.mean, max: 0.1 };
        this.regions.set(r.name, s);
      }
      s.target.set(r.values); s.mean = r.mean;
      const m = r.values.reduce((a, x) => Math.max(a, Math.abs(x)), 0);
      s.max = Math.max(0.06, Math.max(m, s.max * 0.9));
    }
    const regs = (this.layout && this.layout.regions || []).map((r) => r.name).filter((n) => this.regions.has(n));
    if (regs.length) {                            // one row of the neuron matrix per bar: every region's activations
      const row = [], spans = [];
      for (const n of regs) { const r = this.regions.get(n); spans.push([n, r.target.length]); for (const x of r.target) row.push(clamp(Math.abs(x) / r.max)); }
      this.matrixRows.push({ spans, row: Float32Array.from(row), bar: now.bar || 0 });
      if (this.matrixRows.length > 128) this.matrixRows.shift();
    }
    this.settleHist.push({ steps: (v.settle && v.settle.steps) || 0, residual: (v.settle && v.settle.residual) || 0, heat: this.heat, bar: now.bar || 0, changed: (history.length > 1 ? (now.tokens || []).filter((t, l) => history[history.length - 2].tokens[l] !== t).length : 0) });
    if (this.settleHist.length > 96) this.settleHist.shift();
    (v.heads || []).forEach((reads, h) => {
      const old = new Map((this.heads[h] || []).map((x) => [x.age, x]));
      this.heads[h] = reads.filter((x) => x.age != null).map((x) => ({ age: x.age, w: x.w, dw: old.has(x.age + 1) ? old.get(x.age + 1).dw : 0 }));
    });
    if (v.apical && v.apical.length) {
      this.apical = v.apical;
      this.apMax = Math.max(0.05, Math.max(...v.apical), this.apMax * 0.95);
    }
    const steps = (v.settle && v.settle.steps) || 0;
    this.ripples.push({ t0: performance.now(), n: clamp(Math.round(steps / 16), 1, 4) });
    if (this.ripples.length > 6) this.ripples.shift();
  }

  step(step, kick) {
    const p = this.now && this.now.patterns && this.now.patterns[0];       // the beat the brain wrote, when it writes one
    if (p && p.hits && p.hits.kick) kick = p.hits.kick[step] > 0;
    this.stepIdx = step; this.tick = step % 4 === 0 ? 1 : 0.45;
    if (kick) { this.kick = step === 0 ? 1 : 0.8; this.lastKickT = performance.now(); this.kickEdge = true; }
    this.lastStep = performance.now();
    if (this.gl3 && this.gl3.n && this.is3d && !this.view) this.gl3.fromAudio(this.spec, this.level, (this.bar << 4) + step);
  }

  // one factual line for the scroller: what changed this bar, how long things hold, how far the memory reached
  newsOf(now, history) {
    const out = [`BAR ${now.bar || 0}`];
    const prevBar = history.length > 1 ? history[history.length - 2] : null;
    const changed = [];
    (now.tokens || []).forEach((t, l) => { if (prevBar && prevBar.tokens[l] !== t) changed.push(`${LAYER_NAMES[l]} ${prevBar.tokens[l]}>${t}`); });
    if (changed.length) out.push(changed.join(' '));
    else {
      let hold = 1; for (let i = history.length - 2; i >= 0 && history[i].tokens.join() === (now.tokens || []).join(); i--) hold++;
      out.push(`ALL SOUNDS HOLD ${hold} BAR${hold > 1 ? 'S' : ''}`);
    }
    const v = now.view;
    if (v && v.heads) {
      let best = null;
      v.heads.forEach((reads, h) => reads.forEach((r) => { if (r.age != null && r.age > 0 && (!best || r.w > best.w)) best = { h, age: r.age, w: r.w }; }));
      if (best) out.push(`HEAD ${best.h + 1} READS ${best.age} BAR${best.age > 1 ? 'S' : ''} BACK`);
    }
    (now.mods || []).forEach((m, l) => { const f = Math.round((m[0] || 0) * 3); if (f) out.push(`FILTER ${f > 0 ? '+' : '-'}${Math.abs(f)} ${LAYER_NAMES[l]}`); });
    if (now.mode === 'brain') out.push(now.arousal && now.arousal.mode ? String(now.arousal.mode).toUpperCase() : 'PLAYING');
    else out.push('NO BRAIN LOADED');
    return out.join(' :: ') + ' :: ';
  }

  // everything a demo mode reads, normalised to 0..1
  demoState(t, dt, phase) {
    const R = (n) => this.regions.get(n);
    const norm = (n) => {
      const s = R(n); if (!s) return 0;
      const mx = Math.max(0.02, Math.max(Math.abs(s.dmean), (this.meanMax.get(n) || 0) * 0.995)); this.meanMax.set(n, mx);
      return clamp(s.dmean / mx);
    };
    const vals = (n) => { const s = R(n); return s ? Array.from(s.disp, (v) => clamp(v / s.max)) : []; };
    this.hear();
    const brain = !!(this.view && this.layout);
    const now = this.now || { tokens: [0, 0, 0, 0], mods: [[0, 0], [0, 0], [0, 0], [0, 0]] };
    const mods = now.mods || [[0, 0], [0, 0], [0, 0], [0, 0]];
    const spec = this.spec ? Array.from(this.spec.subarray(2, 98), (x) => x / 255) : new Array(96).fill(0.2 * this.level);
    const core = brain ? this.core() : [], modules = core.filter((n) => n.startsWith('module_')), reads = brain ? this.reads() : [];
    const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
    const act = brain
      ? [norm('sensory'), norm('code'), norm('association'), norm('prefrontal'), norm('motor'), avg(reads.map(norm)), clamp(this.heat + (this.view.settle ? Math.min(1, (this.view.settle.steps || 0) / 64) * 0.4 : 0))]
      : [this.level, avg(now.tokens.map((x) => (x ? 1 : 0))), this.kick, this.tick, this.level, 0.35 + 0.25 * Math.sin(t / 1700), 0];
    const bars = brain
      ? core.concat(reads).map((n, i, all) => ({ a: norm(n), vals: vals(n), hue: (0.55 + i / all.length * 0.9) % 1 }))
      : [0, 1, 2, 3, 4, 5].map((i) => ({ a: clamp(avg(spec.slice(i * 16, i * 16 + 16)) * 1.6), vals: spec.slice(i * 16, i * 16 + 16), hue: (0.55 + i / 6 * 0.9) % 1 }));
    const groups = brain ? [...modules.map((n) => [n, 0]), ['association', 1], ['prefrontal', 2]] : [];
    let cloud = [], cloudGroup = [];
    for (const [n, gi] of groups) { const v = vals(n); cloud = cloud.concat(v); cloudGroup = cloudGroup.concat(v.map(() => gi)); }
    if (!cloud.length) { cloud = spec; cloudGroup = spec.map((_, i) => (i * 3 / spec.length) | 0); }
    const sinceKick = (t - this.lastKickT) / 1000;
    const S = {
      t, dt, phase, brain, act, bars, cloud, cloudGroup, motor: brain ? vals('motor') : spec.slice(0, 48),
      heads: this.heads.map((hs) => hs.map((x) => ({ age: x.age, w: x.dw }))), headShare: this.heads.map((hs) => clamp(hs.reduce((m, x) => Math.max(m, x.dw), 0))),
      tokens: now.tokens || [0, 0, 0, 0], filter: mods.map((m) => m[0] || 0), space: mods.map((m) => m[1] || 0), history: this.history,
      bar: this.bar, step: this.stepIdx, kick: this.kick, kickEdge: this.kickEdge, level: this.level, heat: this.heat,
      flash: this.flash, flashSign: this.flashSign, mood: this.moodNow(),
      kickAt: (i) => { const d = sinceKick - i * 0.035; return d < 0 ? 0 : Math.exp(-d / 0.13); },
      news: () => this.scroller || this.news.shift() || `BAR ${this.bar} :: `,
    };
    this.kickEdge = false;
    return S;
  }
  hear() {                                     // the audio level (rms, fast up, slow down) and the spectrum
    if (this.an) {
      this.an.getFloatTimeDomainData(this.wave); let e = 0; for (const x of this.wave) e += x * x;
      const rms = Math.min(1, Math.sqrt(e / this.wave.length) * 3.2);
      this.level += (rms - this.level) * (rms > this.level ? 0.5 : 0.06);
      this.an.getByteFrequencyData(this.spec);
    } else this.level = Math.max(this.level * 0.95, 0.25 + 0.5 * this.kick);
  }
  moodNow() {                                  // the mood pad's dot (energy, brightness)
    const d = document.getElementById('dot');
    if (!d) return [0.5, 0.5];
    return [1 - (parseFloat(d.style.top) || 50) / 100, (parseFloat(d.style.left) || 50) / 100];
  }
  feedback(sign) { this.flash = 1; this.flashSign = sign; }

  frame(phase, stepMs) {
    if (document.hidden) return;
    const t = performance.now(), dt = Math.min(0.1, (t - this.last) / 1000); this.last = t;
    if (stepMs) this.stepMs = stepMs;
    const W = this.cv.width, H = this.cv.height, g = this.g, D = devicePixelRatio || 1;
    if (!W || !H) return;
    const ease = 1 - Math.exp(-dt / 0.22);
    for (const s of this.regions.values()) {
      for (let i = 0; i < s.disp.length; i++) s.disp[i] += (s.target[i] - s.disp[i]) * ease;
      s.dmean += (s.mean - s.dmean) * ease;
    }
    for (const hs of this.heads) for (const x of hs) x.dw += (x.w - x.dw) * ease;
    const decay = Math.exp(-dt / Math.max(0.08, (stepMs || 120) / 1000 * 1.6));
    this.kick *= decay; this.tick *= decay; this.flash *= Math.exp(-dt / 0.5);
    const a = this.arousal, target = a ? clamp((a.level || 0) * 1.6 + (a.mode === 'routine' ? 0 : 0.25)) : 0;
    this.heat += (target - this.heat) * (1 - Math.exp(-dt / 0.8));
    this.tone = mix(CALM, HOT, this.heat);
    g.clearRect(0, 0, W, H);
    this.phaseNow = phase || 0;
    const ctx = { g, W, H, D, t, phase, cx: W / 2, cy: H / 2 };
    if (this.is3d) { this.frame3d(ctx, dt); this.feedbackFlash(ctx); return; }
    if (this.gl3) this.gl3.hide();
    if (DEMO_MODES.includes(MODES[this.mode])) {          // demo-scene modes live from audio and steps even without a brain
      const S = this.demoState(t, dt, phase), X = this.screen;
      X.size(W, H);
      const cap = DEMO[MODES[this.mode]](X, S, g);
      if (MODES[this.mode] !== 'acid') X.blit(g, W, H);
      this.feedbackFlash(ctx);
      if (!this.compact && this.captions) demoCaption(g, W, H, D, cap + (S.brain ? '' : ' · no brain loaded: driven by the audio and the steps'));
      return;
    }
    if (!this.view || !this.layout) return this.idle(ctx);
    this[MODES[this.mode]](ctx);
    this.feedbackFlash(ctx);
  }

  // ---------- the 3D modes (Brain3D above) ----------
  want3d() {                                   // the 3D layout of the brain that plays (the newest one kept when none plays)
    const ck = this.now && this.now.mode === 'brain' && this.now.checkpoint ? this.now.checkpoint : '', B = this.b3d;
    if (B.busy || (B.key === ck && B.state !== 'retry') || (B.state === 'retry' && B.key === ck && performance.now() < B.next)) return;
    B.busy = true; B.key = ck;
    fetch(`/api/brain3d${ck ? `?checkpoint=${encodeURIComponent(ck)}` : ''}`, { cache: 'no-store' })
      .then(async (r) => {
        if (r.status === 202) { B.state = 'retry'; B.next = performance.now() + 1500; B.note = `building the 3D view of ${ck || 'the brain'}`; return; }
        if (!r.ok) { let m = `${r.status}`; try { m = (await r.json()).error || m; } catch { /* not JSON */ } throw Error(m); }
        const d = parseBrain3d(await r.arrayBuffer());
        this.gl3.upload(d); B.state = 'ready'; B.note = '';
        if (this.view) this.gl3.fromView(this.view, this.now && this.now.checkpoint);
      })
      .catch((e) => {
        B.state = 'failed'; B.note = `no 3D view (${e.message})`;
        if (!this.gl3.n) { this.gl3.upload(placeholderBrain()); B.note = 'a stand-in brain lit by the audio (the server has no 3D view)'; }
      })
      .finally(() => { B.busy = false; });
  }
  frame3d(ctx, dt) {
    const { g, W, H, D, t, phase } = ctx, name = MODES[this.mode], storm = name === 'storm';
    if (!this.gl3) this.gl3 = new Brain3D(this.cv, this.compact);
    const G = this.gl3;
    if (G.failed) { G.hide(); this.idle(ctx); this.caption(ctx, `${name}: ${G.failed}`); return; }
    this.want3d();
    this.hear();
    const playing = performance.now() - this.lastStep < 1500;
    if (!playing && G.n && performance.now() - (this.calmAt || 0) > 140) {   // nothing plays: a calm shimmer from the silence
      this.calmAt = performance.now(); G.fromAudio(this.spec, this.level * 0.5, (t / 140) | 0);
    }
    if (name === 'mri' || name === 'mri human') return this.frameMRI(ctx, dt, G, playing ? phase : (t / 5000) % 1, name === 'mri' ? 'cadence' : 'human');
    const compact = name === 'compact';
    G.draw({ t, dt, phase: playing ? phase : (t / 5000) % 1, kick: this.kick, kickEdge: this.kickEdge, beat: playing ? this.tick : 0, level: this.level,
      storm, compact, bar: this.bar, history: this.history, heads: this.heads.map((hs) => hs.map((x) => ({ age: x.age, w: x.dw }))), hue: t * 0.000015 });
    this.kickEdge = false;
    if (!G.n) { if (!this.compact) this.caption(ctx, `${name}: ${this.b3d.note || 'loading the 3D brain'}`); return; }
    const hd = G.data.head;
    if (compact) return this.captionCompact(ctx, G);
    if (!this.compact && !storm && G.mvp) {          // region names at their places
      g.font = `${9 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
      const seen = new Set();
      for (const r of hd.regions) {
        if (r.name.endsWith('_when') || r.name === 'other') continue;
        const label = r.head >= 0 ? `memory ${r.head + 1}` : r.role;
        if (seen.has(label)) continue; seen.add(label);
        const p = G.project(r.head >= 0 ? [r.centre[0], r.centre[1] - 0.62, r.centre[2]] : [r.centre[0], r.centre[1] + r.radius + 0.12, r.centre[2]], W, H);
        if (!p || p[1] > H - 44 * D || p[1] < 12 * D) continue;          // never over the caption
        g.fillStyle = `rgba(200,206,214,${clamp(0.85 - (p[2] - G.dist) * 0.25, 0.25, 0.85).toFixed(2)})`; g.fillText(label, p[0], p[1]);
      }
      const nc = hd.notebook && (hd.notebook.centre || hd.notebook.center);
      if (nc && hd.capacity) { const p = G.project([nc[0] + hd.notebook.radius + 0.3, nc[1], nc[2]], W, H); if (p && p[1] < H - 44 * D) { g.fillStyle = 'rgba(200,206,214,.6)'; g.fillText('notebook', p[0], p[1]); } }
      g.textBaseline = 'alphabetic';
    }
    if (this.compact) return;
    const n = (x) => Number(x || 0).toLocaleString('en-US');
    const lit = G.source === 'neurons' ? 'every neuron live, one value per neuron per bar' : G.source === 'regions' ? 'activity from the bar\'s region values (48 per region)' : 'lit by the audio (no brain view for this bar)';
    const who = hd.placeholder ? 'a stand-in brain (the server sent no 3D view)' : `${hd.checkpoint || 'the last brain'}${this.now && this.now.mode === 'brain' ? '' : ' (no brain plays: its layout, lit by the audio)'}`;
    if (hd.placeholder) {
      this.caption(ctx, `${name}: a stand-in brain (the server sent no 3D view): the regions of a Cadence brain with made-up wiring, lit by the audio; drag to turn`);
      return;
    }
    this.caption(ctx, storm
      ? `storm: ${who}, ${n(G.drawn && G.drawn.synapses)} synapses as light; sparks run pre to post with activation x |weight|, trails zoom on the kick, colours turn with every kick; ${lit}; drag to turn`
      : `connectome: ${who}, ${n(hd.drawn)} neurons placed by region along the signal path (hears, code, association with prefrontal above, rhythm and melody pathways, plays; memory heads and notebook below) and ${n(G.drawn && G.drawn.synapses)} of its ${n(hd.synapses_total)} synapses (the strongest of every pathway); brightness = activation, sparks = activation x |weight| from pre to post, a sweep runs the signal path once per bar; ${lit}; drag to turn`);
  }

  // the compact mode's lobe names and caption
  captionCompact(ctx, G) {
    const { g, W, H, D } = ctx, hd = G.data.head, C = G.cbrain;
    if (!C) { if (!this.compact) this.caption(ctx, 'compact: this server sends no compact brain yet (it needs a restart for the new layout)'); return; }
    if (!this.compact && G.mvp) {
      g.font = `${9 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
      const seen = new Set();
      for (const l of C.head.lobes) {
        if (l.name.endsWith('_when')) continue;
        const label = lobeLabel(l);
        if (seen.has(label)) continue; seen.add(label);
        const p = G.project(l.centre, W, H);
        if (!p || p[1] > H - 44 * D || p[1] < 12 * D) continue;
        g.fillStyle = `rgba(214,220,228,${clamp(0.9 - (p[2] - G.dist) * 0.3, 0.3, 0.9).toFixed(2)})`; g.fillText(label, p[0], p[1]);
      }
      g.textBaseline = 'alphabetic';
    }
    if (this.compact) return;
    const n = (x) => Number(x || 0).toLocaleString('en-US');
    const lit = G.source === 'neurons' ? 'every neuron live' : G.source === 'regions' ? 'activity from the bar\'s region values' : 'lit by the audio (no brain view for this bar)';
    this.caption(ctx, `compact: ${hd.checkpoint || 'the last brain'} as one compact brain: every region a solid lobe sized by its neurons, its neighbours by how strongly they are wired; ${n(G.drawn && G.drawn.synapses)} synapses (every pathway a bundle, then the strongest), sparks run pre to post with activation x |weight|, colours turn with the kick; ${lit}; drag to turn`);
  }

  // ---------- the MRI views ----------
  // one EEG record per bar: the settle's sweeps, budget, residual and tolerance, and per region its mean activation
  // change this bar per sweep (the server's region change, else the change of its 48 values since the last bar)
  eegBar(now) {
    const v = now.view, rec = { t0: performance.now(), bar: now.bar || 0, barSec: (this.stepMs || 120) * 16 / 1000 };
    if (v && v.regions && v.regions.length) {
      const st = v.settle || {}, prev = this.eegPrevVals || new Map(), change = new Map(), mean = new Map();
      rec.steps = st.steps || 0; rec.budget = st.budget || Math.max(32, rec.steps); rec.residual = st.residual || 0; rec.tol = st.tolerance || 0;
      for (const r of v.regions) {
        if (r.name.endsWith('_when')) continue;
        let c = r.change;
        if (c == null) {
          const p = prev.get(r.name); rec.derived = true;
          c = p && p.length === r.values.length ? r.values.reduce((a, x, i) => a + Math.abs(x - p[i]), 0) / r.values.length : 0;
        }
        change.set(r.name, c / Math.max(1, rec.steps)); mean.set(r.name, Math.abs(r.mean || 0));
      }
      this.eegPrevVals = new Map(v.regions.map((r) => [r.name, r.values]));
      let mx = 0; for (const x of change.values()) mx = Math.max(mx, x);
      this.eegRef = Math.max(1e-7, mx, this.eegRef * 0.8);
      let mm = 0; for (const x of mean.values()) mm = Math.max(mm, x);
      rec.names = [...change.keys()];
      rec.amp = rec.names.map((n) => Math.pow(change.get(n) / this.eegRef, 0.35));
      rec.mean = rec.names.map((n) => (mm ? mean.get(n) / mm : 0));
      // how fast it settled: decades of residual per sweep (from 1 to the residual reached): a fast repair rings
      // fast (beta), a slow one slow (theta); the burst lasts longer the more sweeps the settle took
      rec.rate = -Math.log10(Math.max(1e-14, rec.residual || 1)) / Math.max(1, rec.steps);
      rec.f = clamp(3 + 60 * rec.rate, 3, 30);
      rec.D = rec.barSec * Math.min(0.75, 0.1 + rec.steps / 128);
    } else rec.audio = true;
    this.eeg.push(rec); if (this.eeg.length > 24) this.eeg.shift();
  }
  eegChannels() {
    const last = [...this.eeg].reverse().find((r) => r.names);
    if (last) return last.names;
    const regs = this.gl3 && this.gl3.data ? this.gl3.data.head.regions.filter((r) => r.snapshot && !r.name.endsWith('_when')).map((r) => r.name) : [];
    return regs.length ? regs : ['sensory', 'code', 'association', 'prefrontal', 'motor', 'recall'];
  }
  eegLabel(n) {
    if (MODES[this.mode] === 'mri') { const l = this.gl3 && this.gl3.data && this.gl3.data.head.regions.find((r) => r.name === n); return l ? lobeLabel(l) : n; }
    const heads = (this.gl3 && this.gl3.data && this.gl3.data.head.heads) || 4, h = n.startsWith('recall') ? (n === 'recall' ? 0 : +n.split('_')[1]) : -1;
    if (h >= 0) return `HC-${h < heads / 2 ? 'L' : 'R'}${h + 1} memory`;
    if (n.startsWith('module_')) return `TP module ${n.split('_')[1]}`;
    return { sensory: 'STG hears', code: 'TPJ code', association: 'PAR assoc', prefrontal: 'PFC', motor: 'M1 plays', rhythm_plan: 'BG rhythm',
      rhythm_detail: 'CB drums', melody_plan: 'rTL melody', melody_detail: 'rTL notes', phrase_context: 'CING phrase' }[n] || n;
  }
  // a region's trace at time tt (ms): every bar's settle as a burst at the bar's start (amplitude = the region's change
  // per sweep, decaying over the settle's share of the bar, frequency from its sweeps), over a slow rhythm whose size
  // is the region's mean activation
  eegAt(c, name, tt) {
    let v = 0, base = null;
    for (let k = this.eeg.length - 1; k >= 0; k--) {
      const r = this.eeg[k], tau = (tt - r.t0) / 1000;
      if (tau < 0) continue;
      if (tau > 4) break;
      if (r.audio) continue;
      const i = r.names.indexOf(name); if (i < 0) continue;
      if (base == null) base = r.mean[i];
      const env = Math.exp(-tau / (r.D * 0.4)) * (1 - Math.exp(-tau / 0.015));
      if (env > 0.003) v += r.amp[i] * env * Math.sin(2 * Math.PI * r.f * (0.9 + 0.2 * hash01(c)) * tau + c * 1.7);
    }
    const s = tt / 1000;
    return v + 0.12 * (base ?? 0.3) * (0.6 * Math.sin(2 * Math.PI * 10.2 * s + c) + 0.4 * Math.sin(2 * Math.PI * 6.3 * s + 2 * c));
  }
  // the residual on a log scale: 0 at 1, 1 at 1e-8; it falls in a straight line over the settle's share of the bar to
  // the residual the settle reached (the compiled act keeps no per-sweep trace)
  eegResidual(tt) {
    for (let k = this.eeg.length - 1; k >= 0; k--) {
      const r = this.eeg[k]; if (r.t0 > tt) continue;
      if (r.audio || !r.residual) return null;
      const end = Math.log10(Math.max(1e-14, r.residual)), tau = (tt - r.t0) / 1000;
      return clamp((tau < r.D ? end * (tau / r.D) : end) / -8, 0, 1);
    }
    return null;
  }
  mriLayout(W, H, D, t) {
    const cap = this.compact ? 0 : 50 * D, gap = (this.compact ? 3 : 6) * D, eegH = Math.round(this.compact ? H * 0.32 : Math.max(150 * D, (H - cap) * 0.3));
    const top = gap, ph = H - cap - eegH - top - gap, kinds = ['axial', 'sagittal', 'coronal', 'glass'];
    let panels;
    if (this.compact) panels = [{ kind: kinds[Math.floor(t / 6000) % 4], x: 0, y: 0, w: W, h: ph + top }];
    else if (W / ph > 2.4) { const pw = (W - 5 * gap) / 4; panels = kinds.map((kind, i) => ({ kind, x: gap + i * (pw + gap), y: top, w: pw, h: ph })); }
    else { const pw = (W - 3 * gap) / 2, hh = (ph - gap) / 2; panels = kinds.map((kind, i) => ({ kind, x: gap + (i % 2) * (pw + gap), y: top + Math.floor(i / 2) * (hh + gap), w: pw, h: hh })); }
    return { panels, eeg: { x: 0, y: H - cap - eegH, w: W, h: eegH } };
  }
  frameMRI(ctx, dt, G, phase, kind) {
    const { g, W, H, D, t } = ctx, L = this.mriLayout(W, H, D, t), cad = kind === 'cadence';
    // the slices sweep slowly: sagittal across both hemispheres, axial and coronal through the cerebrum
    const cuts = [0.5 + 0.3 * Math.sin(t / 26000 + 1.3), 0.58 + 0.2 * Math.sin(t / 21000), 0.5 + 0.3 * Math.sin(t / 24000 + 2.6)];
    G.draw({ t, dt, phase, kick: 0, kickEdge: false, beat: 0, level: this.level, storm: false, bar: this.bar,
      mri: { kind, panels: L.panels.map((p) => ({ kind: p.kind, x: p.x / W, y: p.y / H, w: p.w / W, h: p.h / H })), cuts } });
    if (!this.view) {                                // no brain view: the audio's bands, kept for the strip
      const sp = this.spec, ch = this.eegChannels(), nb = sp ? sp.length : 0;
      this.eegAudio.push({ t, v: ch.map((_, c) => (nb ? sp[Math.min(nb - 1, 2 + Math.floor((c / ch.length) * nb * 0.6))] / 255 : this.level)) });
      while (this.eegAudio.length && this.eegAudio[0].t < t - 9000) this.eegAudio.shift();
    }
    const name = cad ? 'mri' : 'mri human', M = G.mriSets && G.mriSets[kind];
    if (!G.n) { if (!this.compact) this.caption(ctx, `${name}: ${this.b3d.note || 'loading the 3D brain'}`); return; }
    if (!M) { if (!this.compact) this.caption(ctx, `${name}: this server sends no ${cad ? 'compact brain' : 'MRI anatomy'} yet (it needs a restart for the new layout)`); return; }
    const m = M.m, E = m.box.map(([a, b]) => b - a), mm = (i, f) => Math.round((m.box[i][0] + f * E[i] - (m.box[i][0] + m.box[i][1]) / 2) * m.mm_per_unit);
    if (cad && !this.compact) this.lobeNotes(g, L.panels, M, cuts, D, t);
    g.font = `${(this.compact ? 8 : 9) * D}px ui-monospace, Menlo, monospace`; g.textBaseline = 'alphabetic';
    for (const p of L.panels) {
      g.strokeStyle = 'rgba(48,56,64,.7)'; g.lineWidth = D; g.strokeRect(p.x + 0.5, p.y + 0.5, p.w - 1, p.h - 1);
      const title = { axial: `AXIAL ${mm(1, cuts[1]) >= 0 ? 'S' : 'I'} ${Math.abs(mm(1, cuts[1]))} mm`, sagittal: `SAGITTAL ${mm(0, cuts[0]) >= 0 ? 'R' : 'L'} ${Math.abs(mm(0, cuts[0]))} mm`,
        coronal: `CORONAL ${mm(2, cuts[2]) >= 0 ? 'A' : 'P'} ${Math.abs(mm(2, cuts[2]))} mm`, glass: 'GLASS BRAIN' }[p.kind];
      g.textAlign = 'left'; g.fillStyle = 'rgba(150,162,172,.9)'; g.fillText(title, p.x + 8 * D, p.y + 14 * D);
      if (this.compact || p.kind === 'glass') continue;
      const [l, r, tp, bt] = { axial: ['R', 'L', 'A', 'P'], sagittal: ['A', 'P', 'S', 'I'], coronal: ['R', 'L', 'S', 'I'] }[p.kind];
      g.fillStyle = 'rgba(110,122,132,.85)'; g.textAlign = 'center';
      g.fillText(l, p.x + 10 * D, p.y + p.h / 2); g.fillText(r, p.x + p.w - 10 * D, p.y + p.h / 2);
      g.fillText(tp, p.x + p.w / 2, p.y + 26 * D); g.fillText(bt, p.x + p.w / 2, p.y + p.h - 8 * D);
    }
    if (!this.compact) {                             // the heat scale
      const gp = L.panels.find((p) => p.kind === 'glass') || L.panels[0], x = gp.x + gp.w - 96 * D, y = gp.y + gp.h - 16 * D;
      const gr = g.createLinearGradient(x, 0, x + 70 * D, 0);
      [[0, '#300000'], [0.33, '#c01000'], [0.66, '#ff9000'], [1, '#fffbe0']].forEach(([o, c]) => gr.addColorStop(o, c));
      g.fillStyle = gr; g.fillRect(x, y, 70 * D, 5 * D);
      g.fillStyle = 'rgba(120,130,140,.9)'; g.textAlign = 'left'; g.fillText('activity', x - 54 * D, y + 6 * D);
    }
    this.drawEEG(g, L.eeg, D, t);
    if (this.compact) return;
    const hd = G.data.head, brain = this.now && this.now.mode === 'brain', last = this.eeg[this.eeg.length - 1];
    const waves = !brain || !last || last.audio ? 'EEG: no brain plays, so the traces follow the audio\'s bands'
      : `EEG per region: each bar's settle as a burst; amplitude = its activation change per sweep${last.derived ? ' (from its 48 values)' : ''} (compressed), frequency = residual decades per sweep (now ${last.rate.toFixed(2)}: ${last.f.toFixed(0)} Hz), length = sweeps (now ${last.steps}); residual: a straight log fall from 1 to the ${last.residual.toExponential(1)} reached${last.tol ? ` (dashed: tolerance ${last.tol})` : ''}; the compiled act keeps no per-sweep trace`;
    this.caption(ctx, cad
      ? `mri: ${hd.checkpoint || 'the last brain'} as one compact brain, each region a solid lobe sized by its neurons and placed by its wiring; grey = neurons per voxel, white = the synapse bundles (summed |weight| along their paths), dark lines = lobe borders; heat = per voxel its neurons' activation relative to their region's range this bar, smoothed over space and about a second. ${waves}`
      : `mri human: ${hd.checkpoint || 'the last brain'} on a human brain (hearing STG, code temporo-parietal, association parietal, prefrontal frontal, plays motor cortex, rhythm basal ganglia and cerebellum, melody right temporal, memory hippocampi, notebook entorhinal); synthetic T1 slices sweep slowly; heat = mean activity of the neurons per voxel, smoothed over space and about a second. ${waves}`);
  }
  // the lobes' names on the slices, as radiology annotations: per panel at the centre of each lobe the slice cuts
  // (worked out from the lobe volume a few times a second)
  lobeNotes(g, panels, M, cuts, D, t) {
    if (!this.notes || t - this.notes.t > 250) {
      const [nx, ny, nz] = M.m.shape, E = M.m.box.map(([a, b]) => b - a), out = [];
      for (const p of panels) {
        if (p.kind === 'glass') continue;
        const axis = { axial: 0, sagittal: 1, coronal: 2 }[p.kind], span = sliceSpan(axis, p.w, p.h, E), acc = new Map(), N = 72;
        for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
          const q = [(a + 0.5) / N, (b + 0.5) / N];      // image coordinates over the volume's extent
          const tx = axis === 1 ? cuts[0] : 1 - q[0], ty = axis === 0 ? cuts[1] : q[1], tz = axis === 0 ? q[1] : axis === 1 ? 1 - q[0] : cuts[2];
          const i = Math.floor(tx * nx), j = Math.floor(ty * ny), k = Math.floor(tz * nz);
          if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
          const l = M.lobes[(k * ny + j) * nx + i]; if (l === 255) continue;
          const e = acc.get(l) || [0, 0, 0]; e[0] += q[0]; e[1] += q[1]; e[2]++; acc.set(l, e);
        }
        for (const [l, [sa, sb, c]] of acc) {
          if (c < 20) continue;
          const qa = sa / c, qb = sb / c, u = (qa - 0.5) / span[0] + 0.5, v = (qb - 0.5) / span[1] + 0.5;
          out.push({ x: p.x + u * p.w, y: p.y + (1 - v) * p.h, text: lobeLabel(M.m.lobes[l] || { role: '?' }) });
        }
      }
      this.notes = { t, out };
    }
    g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
    for (const n of this.notes.out) {
      g.fillStyle = 'rgba(0,0,0,.55)'; const w = g.measureText(n.text).width + 6 * D; g.fillRect(n.x - w / 2, n.y - 6 * D, w, 12 * D);
      g.fillStyle = 'rgba(150,220,215,.9)'; g.fillText(n.text, n.x, n.y);
    }
    g.textBaseline = 'alphabetic';
  }
  drawEEG(g, R, D, t) {
    const names = this.eegChannels(), n = names.length + 1, head = this.compact ? 0 : 12 * D, lane = (R.h - head) / n, span = this.compact ? 4000 : 8000;
    const lw = this.compact ? 4 * D : 112 * D, x0 = R.x + lw, x1 = R.x + R.w - 10 * D, X = (tt) => x1 - ((t - tt) / span) * (x1 - x0);
    g.fillStyle = 'rgba(5,7,9,.97)'; g.fillRect(R.x, R.y, R.w, R.h);
    g.lineWidth = Math.max(1, 0.6 * D);
    for (let s = Math.ceil((t - span) / 1000) * 1000; s <= t; s += 1000) { g.strokeStyle = 'rgba(34,42,50,.8)'; g.beginPath(); g.moveTo(X(s), R.y); g.lineTo(X(s), R.y + R.h); g.stroke(); }
    g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'left';
    for (const r of this.eeg) {
      if (r.t0 < t - span) continue;
      g.strokeStyle = 'rgba(90,140,150,.5)'; g.beginPath(); g.moveTo(X(r.t0), R.y); g.lineTo(X(r.t0), R.y + R.h); g.stroke();
      if (!this.compact) { g.fillStyle = 'rgba(100,150,160,.8)'; g.fillText(`bar ${r.bar}`, X(r.t0) + 3 * D, R.y + 9 * D); }
    }
    const step = 2 * D, audio = !this.view;
    names.forEach((nm, c) => {
      const mid = R.y + head + lane * (c + 0.5);
      if (!this.compact) { g.fillStyle = 'rgba(128,140,150,.9)'; g.fillText(this.eegLabel(nm), R.x + 8 * D, mid + 3 * D); }
      g.strokeStyle = 'rgba(176,196,186,.82)'; g.lineWidth = Math.max(1, 0.8 * D); g.beginPath();
      let first = true, ai = 0;
      for (let x = x0; x <= x1; x += step) {
        const tt = t - ((x1 - x) / (x1 - x0)) * span;
        let v;
        if (audio) {
          while (ai + 1 < this.eegAudio.length && this.eegAudio[ai + 1].t <= tt) ai++;
          const a = this.eegAudio[ai]; v = a && a.t <= tt + 50 ? (a.v[c] || 0) * Math.sin(2 * Math.PI * (6 + c) * tt / 1000) : 0;
        } else v = this.eegAt(c, nm, tt);
        const y = mid - clamp(v, -1.3, 1.3) * lane * 0.42;
        if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y);
      }
      g.stroke();
    });
    const top = R.y + head + lane * (n - 1) + 2 * D, hh = lane - 4 * D;      // the residual lane
    if (!this.compact) { g.fillStyle = 'rgba(170,150,230,.9)'; g.fillText('residual', R.x + 8 * D, top + hh / 2 + 3 * D); }
    const tol = [...this.eeg].reverse().find((r) => r.tol), ty = top + clamp(Math.log10(tol ? tol.tol : 1e-3) / -8, 0, 1) * hh;
    g.setLineDash([4 * D, 3 * D]); g.strokeStyle = 'rgba(120,110,170,.6)'; g.beginPath(); g.moveTo(x0, ty); g.lineTo(x1, ty); g.stroke(); g.setLineDash([]);
    g.strokeStyle = 'rgba(180,156,255,.9)'; g.lineWidth = Math.max(1, D); g.beginPath();
    let open = false;
    for (let x = x0; x <= x1; x += step) {
      const v = this.eegResidual(t - ((x1 - x) / (x1 - x0)) * span);
      if (v == null) { open = false; continue; }
      const y = top + v * hh;
      if (!open) { g.moveTo(x, y); open = true; } else g.lineTo(x, y);
    }
    g.stroke();
  }

  // ---------- helpers ----------
  core() { return (this.layout.regions || []).filter((r) => !r.name.startsWith('recall')).map((r) => r.name).filter((n) => this.regions.has(n)); }
  reads() { return (this.layout.regions || []).filter((r) => r.name.startsWith('recall')).map((r) => r.name).filter((n) => this.regions.has(n)); }
  norm(s, v) { return clamp(v / s.max); }
  pastBar(age) { return this.history[this.history.length - 1 - age] || null; }
  apicalPools() {
    return (this.layout.apical || []).map((p, i) => ({ ...p, g: this.apical[i] != null ? this.apical[i] / this.apMax : 0 }));
  }
  caption(ctx, text) {
    if (this.compact || !this.captions) return;
    const { g, D, H, W } = ctx;
    g.font = `${9 * D}px ui-monospace, Menlo, monospace`; g.fillStyle = 'rgba(75,82,91,.9)'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    const lines = [];
    let cur = '';
    for (const w of text.split(' ')) {
      const t = cur ? `${cur} ${w}` : w;
      if (cur && g.measureText(t).width > W - 32 * D) { lines.push(cur); cur = w; } else cur = t;
    }
    if (cur) lines.push(cur);
    lines.forEach((l, i) => g.fillText(l, 16 * D, H - 6 * D - (lines.length - 1 - i) * 12 * D));
  }
  orb(ctx, x, y, r) {
    const { g } = ctx;
    const rr = r * (1 + 0.32 * this.kick + 0.1 * this.flash);
    const fc = this.flashSign > 0 ? [239, 111, 138] : HOT;
    const c = this.flash > 0.05 ? mix(this.tone, fc, this.flash) : this.tone;
    const jx = this.flashSign < 0 ? this.flash * (Math.random() - 0.5) * 6 * ctx.D : 0;
    const gr = g.createRadialGradient(x + jx, y, 0, x + jx, y, rr * 2.4);
    gr.addColorStop(0, rgba(c, 0.95)); gr.addColorStop(0.35, rgba(c, 0.25)); gr.addColorStop(1, rgba(c, 0));
    g.fillStyle = gr; g.beginPath(); g.arc(x + jx, y, rr * 2.4, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#08090b'; g.beginPath(); g.arc(x + jx, y, rr * 0.5, 0, Math.PI * 2); g.fill();
  }
  feedbackFlash(ctx) {
    if (this.flash < 0.03) return;
    const { g, W, H, cx, cy } = ctx, c = this.flashSign > 0 ? [239, 111, 138] : HOT, R = Math.min(W, H) * 0.5;
    const gr = g.createRadialGradient(cx, cy, 0, cx, cy, R);
    gr.addColorStop(0, rgba(c, this.flash * 0.10)); gr.addColorStop(1, rgba(c, 0));
    g.fillStyle = gr; g.fillRect(cx - R, cy - R, 2 * R, 2 * R);
  }
  idle(ctx) {
    const { g, cx, cy, W, H, D, t } = ctx, R = Math.min(W, H) * 0.42;
    for (let i = 0; i < 6; i++) {
      const r = R * (0.2 + 0.13 * i) * (1 + 0.01 * Math.sin(t / 1400 + i));
      g.strokeStyle = 'rgba(42,47,54,.6)'; g.lineWidth = D; g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.stroke();
    }
    this.orb(ctx, cx, cy, R * 0.07);
    g.font = `${10 * D}px ui-monospace, Menlo, monospace`; g.fillStyle = 'rgba(75,82,91,.95)'; g.textAlign = 'center';
    g.fillText('no brain loaded', cx, cy + R * 1.08);
  }


  // ---------- matrix: the brain's activity as a spectrogram, one row per bar ----------
  matrix(ctx) {
    const { g, W, H, D } = ctx, rows = this.matrixRows;
    if (!rows.length) return this.idle(ctx);
    const last = rows[rows.length - 1], cols = last.row.length, n = Math.min(96, Math.max(16, rows.length + 1));
    if (!this.mxCanvas) { this.mxCanvas = document.createElement('canvas'); this.mxPal = ramp([[0, [6, 6, 12]], [55, [16, 48, 70]], [130, [95, 211, 198]], [200, [242, 166, 90]], [255, [255, 252, 235]]]); }
    const mc = this.mxCanvas; if (mc.width !== cols || mc.height !== n) { mc.width = cols; mc.height = n; this.mxImg = mc.getContext('2d').createImageData(cols, n); }
    const buf = new Uint32Array(this.mxImg.data.buffer); buf.fill(RGB(6, 6, 12));
    const live = [];                               // the newest row is live: the activations as they glide now
    for (const [name] of last.spans) { const r = this.regions.get(name); if (r) for (const x of r.disp) live.push(clamp(Math.abs(x) / r.max)); }
    const all = rows.slice(-(n - 1)).map((r) => r.row).concat([Float32Array.from(live)]);
    all.forEach((row, j) => { const y = n - all.length + j; for (let x = 0; x < Math.min(cols, row.length); x++) buf[y * cols + x] = this.mxPal[(row[x] * 255) | 0]; });
    mc.getContext('2d').putImageData(this.mxImg, 0, 0);
    const L = (this.compact ? 4 : 92) * D, T = (this.compact ? 4 : 26) * D, B = (this.compact ? 4 : 30) * D, ww = W - L - 12 * D, hh = H - T - B;
    g.imageSmoothingEnabled = false; g.drawImage(mc, L, T, ww, hh); g.imageSmoothingEnabled = true;
    let x0 = 0; g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'left';
    for (const [name, len] of last.spans) {        // region boundaries and names
      const x = L + (x0 / cols) * ww; g.fillStyle = 'rgba(8,9,11,.9)'; g.fillRect(x - D * 0.5, T, D, hh);
      if (!this.compact && len / cols * ww > 26 * D) { g.fillStyle = 'rgba(141,149,159,.9)'; g.fillText(roleOf(name).toUpperCase().slice(0, 10), x + 3 * D, T - 8 * D); }
      x0 += len;
    }
    if (!this.compact) {
      g.fillStyle = 'rgba(141,149,159,.9)'; g.textAlign = 'right';
      g.fillText('now', L - 6 * D, T + hh - 2 * D); g.fillText(`${Math.min(n, all.length) - 1} bars ago`, L - 6 * D, T + hh - (all.length / n) * hh + 9 * D);
      const y = T + hh - (hh / n) * (1 - ((this.kick * 2) % 1)); g.fillStyle = rgba(this.tone, 0.35 * this.kick); g.fillRect(L, T + hh - hh / n, ww, hh / n);
    }
    this.caption(ctx, 'matrix: every neuron of every region (columns) bar by bar (rows, newest at the bottom, live) — a spectrogram of the brain; brightness = activation');
  }

  // ---------- timeline: the last 64 bars, the four layers, the memory heads reaching back ----------
  timeline(ctx) {
    const { g, W, H, D } = ctx, hist = this.history.slice(-64);
    if (!hist.length) return this.idle(ctx);
    const span = Math.min(64, Math.max(16, hist.length));
    const L = (this.compact ? 6 : 64) * D, R = 14 * D, lane = (this.compact ? 16 : 30) * D, top = H * (this.compact ? 0.42 : 0.46), ww = W - L - R, cw = ww / span;
    const xOf = (age) => L + ww - (age + 0.5) * cw;   // age 0 = the bar playing, at the right edge
    g.font = `${9 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'right';
    for (let l = 0; l < 4; l++) {
      const y = top + l * lane;
      if (!this.compact) { g.fillStyle = rgba(LAYER[l], 0.7); g.fillText(LAYER_NAMES[l], L - 8 * D, y + lane * 0.62); }
      hist.forEach((b, j) => {
        const age = hist.length - 1 - j, tok = b.tokens[l], x = xOf(age) - cw / 2;
        if (!tok) { g.fillStyle = '#0f1115'; g.fillRect(x + 0.5, y + 2 * D, cw - 1, lane - 4 * D); return; }
        const f = b.mods && b.mods[l] ? b.mods[l][0] : 0, c = hue((tok * 0.137 + l * 0.25) % 1, 0.7, 0.42 + 0.3 * (f + 1) / 2);
        g.fillStyle = rgba(c, 1); g.fillRect(x + 0.5, y + 2 * D, cw - 1, lane - 4 * D);
        const prev = hist[j - 1];
        if (prev && prev.tokens[l] !== tok) { g.fillStyle = 'rgba(255,255,255,.85)'; g.fillRect(x, y, Math.max(1, 1.5 * D), lane); }   // a change
      });
    }
    const rulerY = top + 4 * lane + 6 * D;          // the bar clock: ticks per bar, taller every 4, 8, 16
    hist.forEach((b, j) => {
      const age = hist.length - 1 - j, k = b.bar || 0, hgt = k % 16 === 0 ? 12 : k % 8 === 0 ? 8 : k % 4 === 0 ? 5 : 2;
      g.fillStyle = k % 8 === 0 ? 'rgba(242,166,90,.8)' : 'rgba(75,82,91,.8)'; g.fillRect(xOf(age) - cw / 2, rulerY, Math.max(1, D), hgt * D);
    });
    const nowX = xOf(0), baseY = top - 6 * D;       // the four heads: arcs from now back to the bars they read
    this.heads.forEach((reads, hd) => {
      for (const r of reads) {
        if (r.age == null || r.age < 1 || r.age > span - 1 || r.dw < 0.03) continue;
        const x1 = xOf(r.age), lift = Math.min(top - 18 * D, (nowX - x1) * 0.45 + 12 * D);
        g.strokeStyle = HEAD[hd]; g.globalAlpha = clamp(0.25 + r.dw); g.lineWidth = (0.6 + 5 * r.dw) * D;
        g.beginPath(); g.moveTo(nowX, baseY); g.quadraticCurveTo((nowX + x1) / 2, baseY - lift, x1, baseY); g.stroke();
      }
    });
    g.globalAlpha = 1;
    g.fillStyle = rgba(this.tone, 0.5 + 0.5 * this.kick); g.fillRect(nowX - cw / 2, top - 3 * D, cw, 4 * lane + 6 * D);
    this.caption(ctx, 'timeline: the last 64 bars, newest at the right — the four layers coloured by their sound (brightness = FILTER), white ticks = a change; arcs = the four memory heads reading back (thickness = share); orange ticks = every 8th bar');
  }

  // ---------- decision: what held, what changed, where in the phrase ----------
  decision(ctx) {
    const { g, W, H, D, t } = ctx, hist = this.history;
    if (!hist.length) return this.idle(ctx);
    const cur = hist[hist.length - 1], cx = this.compact ? W * 0.28 : W * 0.3, cy = H * 0.47, R = Math.min(W * 0.22, H * 0.36);
    const pos = (bar) => (bar % 32) / 32 * Math.PI * 2 - Math.PI / 2;
    g.strokeStyle = 'rgba(42,47,54,.9)'; g.lineWidth = D; g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
    for (let k = 0; k < 32; k++) {                    // the 32-bar ring, phrase marks at 8 and 16
      const a = pos(k), long = k % 8 === 0 ? 0.16 : k % 4 === 0 ? 0.08 : 0.03;
      g.strokeStyle = k % 8 === 0 ? 'rgba(242,166,90,.75)' : 'rgba(75,82,91,.8)'; g.lineWidth = (k % 8 === 0 ? 1.6 : 1) * D;
      g.beginPath(); g.moveTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); g.lineTo(cx + Math.cos(a) * R * (1 - long), cy + Math.sin(a) * R * (1 - long)); g.stroke();
    }
    const counts = new Array(32).fill(0);
    for (let j = 1; j < hist.length; j++) {             // where in the phrase the brain changed sounds, by layer
      const b = hist[j], p = hist[j - 1];
      for (let l = 0; l < 4; l++) if (b.tokens[l] !== p.tokens[l]) {
        const k = (b.bar || 0) % 32, a = pos(b.bar || 0), rr = R * (1.08 + 0.07 * l + 0.06 * counts[k]);
        g.fillStyle = rgba(LAYER[l], 0.35 + 0.65 * (j / hist.length)); g.beginPath(); g.arc(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, 2.6 * D, 0, Math.PI * 2); g.fill();
      }
      if (b.tokens.some((x, l) => x !== p.tokens[l])) counts[(b.bar || 0) % 32]++;
    }
    const a = pos((cur.bar || 0) + this.phaseNow), hx = cx + Math.cos(a) * R, hy = cy + Math.sin(a) * R;   // where we are now
    g.strokeStyle = rgba(this.tone, 0.9); g.lineWidth = 2 * D; g.beginPath(); g.moveTo(cx, cy); g.lineTo(hx, hy); g.stroke();
    this.orb(ctx, cx, cy, R * 0.12);
    const since = [0, 1, 2, 3].map((l) => { let n = 0; for (let j = hist.length - 1; j > 0 && hist[j].tokens[l] === hist[j - 1].tokens[l]; j--) n++; return n + 1; });
    const changedNow = hist.length > 1 ? [0, 1, 2, 3].map((l) => cur.tokens[l] !== hist[hist.length - 2].tokens[l]) : [false, false, false, false];
    const x0 = this.compact ? W * 0.56 : W * 0.6, fs = (this.compact ? 9 : 12) * D;
    g.textAlign = 'left'; g.font = `${fs}px ui-monospace, Menlo, monospace`;
    for (let l = 0; l < 4; l++) {
      const y = cy - 1.6 * fs * 1.6 + l * fs * 2.1;
      g.fillStyle = rgba(LAYER[l], 0.9); g.fillText(LAYER_NAMES[l], x0, y);
      g.fillStyle = changedNow[l] ? rgba(HOT, 1) : 'rgba(141,149,159,.95)';
      g.fillText(cur.tokens[l] ? (changedNow[l] ? `changed → ${cur.tokens[l]}` : `holds ${cur.tokens[l]} · ${since[l]} bar${since[l] > 1 ? 's' : ''}`) : 'silent', x0 + fs * 4.2, y);
    }
    if (!this.compact) {
      const total = counts.reduce((x, y) => x + y, 0), on8 = counts.filter((_, k) => k % 8 === 0).reduce((x, y) => x + y, 0);
      g.fillStyle = 'rgba(141,149,159,.85)'; g.font = `${10 * D}px ui-monospace, Menlo, monospace`;
      g.fillText(`bar ${cur.bar || 0} · ${(cur.bar || 0) % 32 + 1} of 32 · changes on 8-bar marks ${total ? Math.round((100 * on8) / total) : 0}%`, x0, cy + fs * 5.2);
    }
    this.caption(ctx, 'decision: right = what each layer does this bar (holds how long, or changed); the ring = the 32-bar cycle, dots = where in it the brain changed sounds (by layer), orange marks = every 8th bar');
  }

  // ---------- settle: how hard the brain works each bar ----------
  settle(ctx) {
    const { g, W, H, D, t } = ctx, hs = this.settleHist.slice(-64);
    if (!hs.length) return this.idle(ctx);
    const span = Math.min(64, Math.max(16, hs.length));
    const L = (this.compact ? 6 : 40) * D, R = 16 * D, base = H * (this.compact ? 0.86 : 0.78), top = H * 0.18, ww = W - L - R, cw = ww / span;
    const maxSteps = Math.max(32, ...hs.map((x) => x.steps));
    hs.forEach((x, j) => {
      const age = hs.length - 1 - j, xx = L + ww - (age + 1) * cw, hh = (x.steps / maxSteps) * (base - top) * 0.82;
      const c = mix(CALM, HOT, clamp(x.heat)); g.fillStyle = rgba(c, 0.35 + 0.5 * (j / hs.length)); g.fillRect(xx + 1, base - hh, cw - 2, hh);
      if (x.changed) { g.fillStyle = 'rgba(255,255,255,.8)'; g.fillRect(xx + 1, base + 4 * D, cw - 2, (2 + 2 * x.changed) * D); }
    });
    g.strokeStyle = 'rgba(180,156,255,.85)'; g.lineWidth = 1.4 * D; g.beginPath();   // the residual, log scale
    hs.forEach((x, j) => {
      const age = hs.length - 1 - j, xx = L + ww - (age + 0.5) * cw, lr = clamp((Math.log10(Math.max(x.residual, 1e-14)) + 14) / 14);
      const y = base - lr * (base - top); j ? g.lineTo(xx, y) : g.moveTo(xx, y);
    });
    g.stroke();
    const last = hs[hs.length - 1], since = (performance.now() - this.barT) / 1000;
    for (let k = 0; k < 3; k++) {                       // the newest settle as rings spreading out
      const rr = (8 + (since * 60 + k * 18) % 60) * D, al = clamp(1 - ((since * 60 + k * 18) % 60) / 60) * 0.5;
      g.strokeStyle = rgba(this.tone, al); g.lineWidth = D; g.beginPath(); g.arc(L + (this.compact ? 20 : 300) * D, top - 4 * D, rr * (last.steps / maxSteps + 0.3) * 0.6, 0, Math.PI * 2); g.stroke();
    }
    if (!this.compact) {
      g.font = `${9 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'left'; g.fillStyle = 'rgba(141,149,159,.9)';
      g.fillText(`settle steps ${last.steps} · residual ${last.residual.toExponential(1)}`, L, top - 10 * D);
    }
    this.caption(ctx, 'settle: each bar the brain settles once to answer it — bar height = settle steps, violet line = remaining residual (log scale), colour = arousal, white marks = how many layers changed');
  }

  // ---------- orbit: everything at once ----------
  orbit(ctx) {
    const { g, cx, cy, W, H, D, phase } = ctx;
    const R = Math.min(W, H) * 0.47, names = this.core(), k = names.length;
    const r0 = R * 0.17, r1 = R * 0.68, gap = (r1 - r0) / Math.max(1, k - 1), Rm = R * 0.9;
    const turn = (this.bar + phase);
    // settle ripples: one per bar, as many rings as the settle needed steps
    const now = performance.now();
    for (const rp of this.ripples) {
      const age = (now - rp.t0) / 2400;
      if (age > 1) continue;
      for (let j = 0; j < rp.n; j++) {
        const f = clamp(age - j * 0.08);
        g.strokeStyle = rgba(this.tone, (1 - f) * 0.12); g.lineWidth = D;
        g.beginPath(); g.arc(cx, cy, r0 + (Rm - r0) * f, 0, Math.PI * 2); g.stroke();
      }
    }
    // regions as rings of neurons
    const ringR = {};
    names.forEach((n, i) => {
      const s = this.regions.get(n), r = r0 + gap * i, m = this.norm(s, s.dmean);
      ringR[n] = r;
      const breathe = r * (1 + 0.025 * m * (0.6 + this.kick));
      g.strokeStyle = rgba([42, 47, 54], 0.7); g.lineWidth = D; g.beginPath(); g.arc(cx, cy, breathe, 0, Math.PI * 2); g.stroke();
      const rot = (i % 2 ? -1 : 1) * turn * Math.PI * 2 / (24 + 8 * i);
      const N = s.disp.length;
      for (let j = 0; j < N; j++) {
        const v = this.norm(s, s.disp[j]), ang = rot + (j / N) * Math.PI * 2;
        const rr = breathe + v * gap * 0.22 * (0.7 + 0.6 * this.kick);
        g.fillStyle = rgba(this.tone, 0.14 + 0.86 * v * (0.75 + 0.25 * this.tick));
        g.beginPath(); g.arc(cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr, (0.9 + 2.6 * v) * D, 0, Math.PI * 2); g.fill();
      }
    });
    // apical halos: an arc beside every region an apical pool sits in, thickness = the pool's gain
    this.apicalPools().forEach((p, pi) => {
      for (const [n, cnt] of Object.entries(p.regions || {})) {
        if (ringR[n] == null) continue;
        const span = clamp(cnt / Math.max(1, p.neurons), 0.15, 1) * Math.PI * 0.9, c0 = -Math.PI / 2 + pi * Math.PI + turn * 0.05;
        g.strokeStyle = rgba(APICAL, 0.18 + 0.6 * clamp(p.g)); g.lineWidth = (1 + 4 * clamp(p.g)) * D; g.lineCap = 'round';
        g.beginPath(); g.arc(cx, cy, ringR[n] + gap * 0.42, c0, c0 + span); g.stroke();
      }
    });
    // ring of past bars, newest at the top, older counter-clockwise
    const AGES = 40, angOf = (age) => -Math.PI / 2 - ((age + phase) / AGES) * Math.PI * 2 * 0.92;
    g.strokeStyle = 'rgba(42,47,54,.5)'; g.lineWidth = D; g.beginPath(); g.arc(cx, cy, Rm, 0, Math.PI * 2); g.stroke();
    for (let age = 0; age < AGES; age++) {
      const b = this.pastBar(age), pb = this.pastBar(age + 1);
      if (!b) break;
      const ang = angOf(age);
      for (let l = 0; l < 4; l++) {
        if (!b.tokens[l]) continue;
        const changed = pb && pb.tokens[l] !== b.tokens[l];
        const rr = Rm + (l - 1.5) * 4.5 * D;
        g.fillStyle = rgba(LAYER[l], changed ? 0.95 : 0.4);
        g.beginPath(); g.arc(cx + Math.cos(ang) * rr, cy + Math.sin(ang) * rr, (changed ? 2.2 : 1.2) * D, 0, Math.PI * 2); g.fill();
      }
    }
    // memory reads: from the playing point on the outer region ring out to the bars read
    const top = [cx, cy - (r1 + gap * 0.2)];
    this.heads.forEach((hs, h) => hs.forEach((x) => {
      if (x.dw < 0.03) return;
      const ang = angOf(x.age + 1), ex = cx + Math.cos(ang) * (Rm - 8 * D), ey = cy + Math.sin(ang) * (Rm - 8 * D);
      const mx = (top[0] + ex) / 2 + (cx - (top[0] + ex) / 2) * 0.35, my = (top[1] + ey) / 2 + (cy - (top[1] + ey) / 2) * 0.35;
      g.strokeStyle = HEAD[h]; g.globalAlpha = 0.12 + 0.75 * x.dw; g.lineWidth = (0.6 + 3.5 * x.dw) * D;
      g.beginPath(); g.moveTo(top[0], top[1]); g.quadraticCurveTo(mx, my, ex, ey); g.stroke();
      const f = (phase * 4) % 1, px = (1 - f) * (1 - f) * top[0] + 2 * (1 - f) * f * mx + f * f * ex, py = (1 - f) * (1 - f) * top[1] + 2 * (1 - f) * f * my + f * f * ey;
      g.fillStyle = HEAD[h]; g.beginPath(); g.arc(px, py, (1 + 2 * x.dw) * D, 0, Math.PI * 2); g.fill();
      g.globalAlpha = 1;
    }));
    this.orb(ctx, cx, cy, r0 * 0.42);
    this.caption(ctx, 'rings inside out: hears · meaning · modules · association · prefrontal · plays   outer ring: past bars   arcs: memory reads' + (this.apical.length ? '   violet: apical' : ''));
  }

  // ---------- flow: the signal path ----------
  flow(ctx) {
    const { g, W, H, D, phase } = ctx;
    const names = this.core(), k = names.length, reads = this.reads();
    const x0 = W * 0.08, x1 = W * 0.92, top = H * 0.16, bot = H * 0.66, colW = Math.max(6 * D, Math.min(16 * D, W / (k * 4)));
    const xs = names.map((_, i) => x0 + ((x1 - x0) * i) / Math.max(1, k - 1));
    const means = names.map((n) => { const s = this.regions.get(n); return this.norm(s, s.dmean); });
    const ribbon = (xa, ya, xb, yb, w, c, alpha, lift) => {
      const cx1 = xa + (xb - xa) * 0.45, cx2 = xa + (xb - xa) * 0.55;
      g.strokeStyle = rgba(c, alpha); g.lineWidth = w; g.lineCap = 'round';
      g.beginPath(); g.moveTo(xa, ya); g.bezierCurveTo(cx1, ya + lift, cx2, yb + lift, xb, yb); g.stroke();
      return (f) => {
        const u = 1 - f;
        return [u * u * u * xa + 3 * u * u * f * cx1 + 3 * u * f * f * cx2 + f * f * f * xb, u * u * u * ya + 3 * u * u * f * (ya + lift) + 3 * u * f * f * (yb + lift) + f * f * f * yb];
      };
    };
    const mid = (top + bot) / 2;
    // ribbons between consecutive regions (and back from prefrontal to association)
    for (let i = 0; i + 1 < k; i++) {
      const inten = clamp(Math.sqrt(means[i] * means[i + 1]));
      const back = names[i] === 'association' && names[i + 1] === 'prefrontal';
      const lifts = back ? [-(bot - top) * 0.18, (bot - top) * 0.18] : [0];
      lifts.forEach((lift, j) => {
        const path = j === 0 ? ribbon(xs[i] + colW / 2, mid, xs[i + 1] - colW / 2, mid, (1 + 10 * inten) * D, this.tone, 0.06 + 0.4 * inten, lift)
          : ribbon(xs[i + 1] - colW / 2, mid, xs[i] + colW / 2, mid, (1 + 10 * inten) * D, this.tone, 0.06 + 0.4 * inten, lift);
        for (let p = 0; p < 3; p++) {
          const [px, py] = path(((phase * 4 + p / 3) % 1));
          g.fillStyle = rgba(this.tone, 0.3 + 0.7 * inten); g.beginPath(); g.arc(px, py, (1 + 2.5 * inten) * D, 0, Math.PI * 2); g.fill();
        }
      });
    }
    // memory read ports below, flowing up into the association
    const ai = names.indexOf('association');
    reads.forEach((n, h) => {
      const s = this.regions.get(n), m = this.norm(s, s.dmean), rx = x0 + (x1 - x0) * (0.3 + 0.4 * (h / Math.max(1, reads.length - 1))), ry = H * 0.84;
      const inten = clamp(Math.sqrt(m * (ai >= 0 ? means[ai] : 0.5)));
      const c = HEAD[h % 4].match(/\w\w/g).map((x) => parseInt(x, 16));
      if (ai >= 0) {
        const path = ribbon(rx, ry - 6 * D, xs[ai], bot + 6 * D, (0.8 + 5 * inten) * D, c, 0.08 + 0.45 * inten, -(H * 0.05));
        const [px, py] = path((phase * 4) % 1);
        g.fillStyle = rgba(c, 0.4 + 0.6 * inten); g.beginPath(); g.arc(px, py, (1 + 2 * inten) * D, 0, Math.PI * 2); g.fill();
      }
      const N = s.disp.length, w = Math.min(W * 0.12, N * 2.5 * D);
      for (let j = 0; j < N; j++) {
        const v = this.norm(s, s.disp[j]);
        g.fillStyle = rgba(c, 0.08 + 0.9 * v); g.fillRect(rx - w / 2 + (j * w) / N, ry - 4 * D, Math.max(1, w / N - D * 0.5), 8 * D);
      }
      g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.fillStyle = 'rgba(75,82,91,.9)'; g.textAlign = 'center';
      g.fillText(roleOf(n), rx, ry + 16 * D);
    });
    // region columns: one cell per (downsampled) neuron group
    names.forEach((n, i) => {
      const s = this.regions.get(n), N = s.disp.length, ch = (bot - top) / N;
      for (let j = 0; j < N; j++) {
        const v = this.norm(s, s.disp[j]);
        g.fillStyle = rgba(this.tone, 0.06 + 0.94 * v * (0.8 + 0.2 * this.tick));
        g.fillRect(xs[i] - colW / 2, top + j * ch, colW, Math.max(1, ch - D * 0.6));
      }
      g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.fillStyle = 'rgba(75,82,91,.95)'; g.textAlign = 'center';
      g.fillText(roleOf(n), xs[i], bot + 22 * D);
    });
    // apical arcs over the regions their pools sit in
    this.apicalPools().forEach((p) => {
      for (const n of Object.keys(p.regions || {})) {
        const i = names.indexOf(n); if (i < 0) continue;
        g.strokeStyle = rgba(APICAL, 0.2 + 0.6 * clamp(p.g)); g.lineWidth = (1 + 3 * clamp(p.g)) * D;
        g.beginPath(); g.arc(xs[i], top - 10 * D, colW * 0.9, Math.PI * 1.1, Math.PI * 1.9); g.stroke();
      }
    });
    this.caption(ctx, 'flow: ribbons = activity of a region times the next (drawn from region activity, not measured per synapse)');
  }

  // ---------- memory: spiral of past bars ----------
  memory(ctx) {
    const { g, cx, cy, W, H, D, phase } = ctx;
    const R = Math.min(W, H) * 0.46, R0 = R * 0.08, AGES = 64;
    const slide = clamp(phase * 4);                         // the spiral settles into place in the first beat
    const pos = (age) => {
      const a = age + slide;
      const th = -Math.PI / 2 + a * 0.38, r = R0 + (R - R0) * Math.sqrt(a / AGES);
      return [cx + Math.cos(th) * r, cy + Math.sin(th) * r, th, r];
    };
    g.strokeStyle = 'rgba(42,47,54,.55)'; g.lineWidth = D; g.beginPath();
    for (let a = 0; a <= AGES; a += 0.25) { const [x, y] = pos(a); a ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
    for (let age = 0; age < AGES; age++) {
      const b = this.pastBar(age), pb = this.pastBar(age + 1);
      if (!b) break;
      const [x, y, th] = pos(age);
      for (let l = 0; l < 4; l++) {
        if (!b.tokens[l]) continue;
        const changed = pb && pb.tokens[l] !== b.tokens[l], off = (l - 1.5) * 5 * D;
        const fm = b.mods ? b.mods[l][0] : 0;
        g.fillStyle = rgba(LAYER[l], (changed ? 0.95 : 0.35 + 0.25 * (fm + 1) / 2) * (age === 0 ? 0.6 + 0.4 * this.tick : 1));
        g.beginPath(); g.arc(x + Math.cos(th) * off, y + Math.sin(th) * off, (changed ? 2.6 : 1.4) * D, 0, Math.PI * 2); g.fill();
      }
    }
    this.heads.forEach((hs, h) => hs.forEach((x) => {
      if (x.dw < 0.03) return;
      const [ex, ey] = pos(x.age + 1);
      g.strokeStyle = HEAD[h]; g.globalAlpha = 0.15 + 0.8 * x.dw; g.lineWidth = (0.6 + 3 * x.dw) * D;
      g.beginPath(); g.moveTo(cx, cy); g.quadraticCurveTo((cx + ex) / 2 + (ey - cy) * 0.2, (cy + ey) / 2 - (ex - cx) * 0.2, ex, ey); g.stroke();
      g.globalAlpha = 1;
    }));
    this.orb(ctx, cx, cy, R0 * 0.6);
    g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.textAlign = 'left';
    HEAD.forEach((c, h) => { g.fillStyle = c; g.fillText(`head ${h + 1}`, 6 * D, 16 * D + h * 12 * D); });
    ['beat', 'bass', 'lead', 'air'].forEach((n, l) => { g.fillStyle = rgba(LAYER[l], 0.8); g.fillText(n, W - 40 * D, 16 * D + l * 12 * D); });
    this.caption(ctx, 'memory: newest bar at the centre, bright dots = a layer changed its sound, arcs = the four recall heads reading back');
  }

  // ---------- scope: a line per region ----------
  scope(ctx) {
    const { g, W, H, D, phase } = ctx;
    const names = [...this.core(), ...this.reads()], k = names.length;
    const x0 = W * 0.14, x1 = W * 0.94, top = H * 0.08, bot = H * 0.9, gapY = (bot - top) / Math.max(1, k);
    const ph = x0 + (x1 - x0) * phase;
    g.strokeStyle = rgba(this.tone, 0.18); g.lineWidth = D; g.beginPath(); g.moveTo(ph, top - gapY * 0.3); g.lineTo(ph, bot); g.stroke();
    const pools = this.apicalPools();
    names.forEach((n, i) => {
      const s = this.regions.get(n), N = s.disp.length, y = top + gapY * (i + 0.5);
      const head = n.startsWith('recall') ? HEAD[(n === 'recall' ? 0 : +n.split('_')[1]) % 4].match(/\w\w/g).map((x) => parseInt(x, 16)) : this.tone;
      const m = this.norm(s, s.dmean);
      g.strokeStyle = rgba(head, 0.3 + 0.6 * m); g.lineWidth = 1.2 * D; g.lineJoin = 'round'; g.beginPath();
      const pts = [];
      for (let j = 0; j < N; j++) {
        const x = x0 + ((x1 - x0) * j) / Math.max(1, N - 1), v = this.norm(s, s.disp[j]);
        const near = Math.exp(-Math.pow((x - ph) / ((x1 - x0) * 0.06), 2));
        pts.push([x, y - v * gapY * 0.42 * (0.75 + 0.25 * this.kick + 0.3 * near)]);
      }
      g.moveTo(pts[0][0], pts[0][1]);
      for (let j = 1; j < pts.length - 1; j++) g.quadraticCurveTo(pts[j][0], pts[j][1], (pts[j][0] + pts[j + 1][0]) / 2, (pts[j][1] + pts[j + 1][1]) / 2);
      g.lineTo(pts[pts.length - 1][0], pts[pts.length - 1][1]);
      g.stroke();
      g.font = `${8 * D}px ui-monospace, Menlo, monospace`; g.fillStyle = 'rgba(75,82,91,.95)'; g.textAlign = 'right';
      g.fillText(roleOf(n), x0 - 10 * D, y + 3 * D);
      pools.forEach((p, pi) => {
        if (!p.regions || !p.regions[n]) return;
        g.fillStyle = rgba(APICAL, 0.3 + 0.6 * clamp(p.g));
        const hgt = gapY * 0.6 * clamp(p.g);
        g.fillRect(x1 + (6 + pi * 6) * D, y - hgt, 3 * D, hgt);
      });
    });
    this.caption(ctx, 'scope: one line per region, the playhead runs through the bar' + (pools.length ? '   violet bars: apical gain' : ''));
  }
}
