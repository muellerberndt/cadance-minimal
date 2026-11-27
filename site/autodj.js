// Cadance Auto DJ: an autopilot for the mixer. It plays whole tracks composed by one brain (the server's track
// factory) on the three decks and mixes from one to the next with programmed transitions and some chance:
// which transition, how long a track plays alone, which effect, a bass break now and then. Nothing here learns.
//
// The set (owner, 26 Nov 09:36 to 09:55): a live track of the chosen brain opens it. The brain composes the next live
// track as soon as the one before it is composed (one ahead; the factory primes it on a new passage: a new
// composition, no reload). Two sections before the next track's line the autopilot decides, in this order:
//   1. the next live track is buffered (its opening bars are composed and the measured rate covers the rest,
//      Track.ready): it mixes in, live to live;
//   2. else a finished track of the same brain from the shelf (the factory's kept tracks) mixes in;
//   3. else a repertoire piece as the transition: it comes in under the outgoing track, carries the set while the
//      live track is not buffered, and the live track then mixes in over it.
// A repertoire piece plays only between two tracks of the brain: never first, never two in a row, and after a piece
// only the live track comes. Every track plays at the set's tempo. With an empty repertoire a late brain makes the
// track on the decks play on.
//
// One deck is the PRIMARY: it plays in full. A new track comes in slowly beside it (its highs first, or through an
// opening lowpass, or its mids under the primary) and then plays in full beside the primary, without its bass, for
// one or two sections: both tracks' patterns are heard together. It takes over the bass on the first beat of a
// 32-bar section and becomes the primary; the old track stays beside it for another section and then leaves.
// Effects are used on the decks beside the primary only, on about half of them.
//
// The Planner is plain logic (bars in, lanes and commands out) and is tested without audio (test_autodj.mjs).
export const SECTION = 32;
// knob positions of a deck that plays in full (the mixer's laws: fader 0.8 = 0 dB, EQ 0.5 = flat, 0 = kill)
export const NEUTRAL = { fader: 0.8, lo: 0.5, mid: 0.5, hi: 0.5, hp: 0, lp: 1, fx: 0, cut: 1 };
export const KEYS = Object.keys(NEUTRAL);

export function mulberry(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const pick = (r, list) => list[Math.min(list.length - 1, Math.floor(r() * list.length))];
const sway = (r, v, d = 0.04) => Math.max(0, Math.min(1, v + (r() * 2 - 1) * d));

// A lane is a list of [bar, value] points in rising bar order. Between points the value moves in a straight line;
// two points on one bar are a jump that lands exactly on that bar line. Before the first point the lane has the
// first value, after the last point the last.
export function at(points, x) {
  if (!points || !points.length) return undefined;
  if (x < points[0][0]) return points[0][1];
  let i = points.length - 1;
  while (i > 0 && points[i][0] > x) i--;
  if (i === points.length - 1) return points[i][1];
  const [a, va] = points[i], [b, vb] = points[i + 1];
  return b === a ? vb : va + ((vb - va) * (x - a)) / (b - a);
}
export function jumps(points, x0, x1) {
  const out = [];
  for (let i = 1; points && i < points.length; i++) {
    const b = points[i][0];
    if (b === points[i - 1][0] && points[i][1] !== points[i - 1][1] && b > x0 && b <= x1) out.push({ bar: b, before: points[i - 1][1], after: points[i][1] });
  }
  return out;
}

// The lane with every jump turned into a short straight line that ends on the jump's bar line (`edge` bars long):
// a filter or a gain that changes in one sample makes a loud click.
export function soft(points, edge) {
  const out = [];
  for (let i = 0; points && i < points.length; i++) {
    const [b, v] = points[i], last = out[out.length - 1];
    if (last && last[0] === b) { last[1] = v; continue; }                       // several points on one bar: the last value counts
    if (i + 1 < points.length && points[i + 1][0] === b) {                      // a jump starts here
      let j = i + 1; while (j + 1 < points.length && points[j + 1][0] === b) j++;
      const after = points[j][1];
      if (after !== v) { const a = Math.max(last ? last[0] : -Infinity, b - edge); if (a < b) out.push([a, v]); }
      out.push([b, after]); i = j; continue;
    }
    out.push([b, v]);
  }
  return out;
}

// ---------- the transitions ----------
// Bars count from the transition's start, a section line. `swap` is the bar where the bass changes decks and the
// incoming deck becomes the primary; at `len` the old deck stops. `in` are the incoming deck's lanes, `out` the
// lanes of the deck that was primary. An effect belongs to a deck while it is not the primary.
export const TRANSITIONS = {
  // The three long blends. The new track comes in filtered over 16 to 32 bars, then plays in full beside the primary,
  // without its bass, until the bass changes decks on a section line. The old track then stays beside the new
  // primary for 32 bars, without its bass, and leaves over the 32 after that. Both tracks' patterns are heard
  // together for 64 to 96 bars.
  // the new track's highs first, then its body, then the bass
  tops(r) {
    const swap = pick(r, [64, 64, 96]), len = swap + 64, body = sway(r, 0.42, 0.02);
    return { name: 'highs first', swap, len,
      in: { fader: [[0, 0], [16, sway(r, 0.6)], [32, 0.74], [swap, 0.78], [swap, 0.8]], hp: [[0, 1], [16, 1], [32, body], [swap, body], [swap, 0]], lo: [[0, 0], [swap, 0], [swap, 0.5]] },
      out: { lo: [[swap, 0.5], [swap, 0]], fader: [[swap, 0.8], [swap + 8, 0.74], [swap + 32, 0.74], [len, 0]], lp: [[swap + 32, 1], [len, sway(r, 0.32)]] },
      fxIn: { type: pick(r, ['reverb', 'flanger', 'phaser', 'sweep']), lane: [[0, sway(r, 0.26)], [24, 0.1], [32, 0]] },
      fxOut: pick(r, [0, 1, 2]) ? { type: pick(r, ['vanish', 'sink']), keep: true, lane: [[swap + 32, 0], [len - 4, sway(r, 0.85, 0.05)], [len, 0.95]] } : { type: pick(r, ['reverb', 'phaser', 'crush']), lane: [[swap + 32, 0], [swap + 48, sway(r, 0.34)], [len, 0.4]] } };
  },
  // the new track through a lowpass that opens: its warmth first
  lowpass(r) {
    const swap = pick(r, [64, 64, 96]), len = swap + 64;
    return { name: 'opening lowpass', swap, len,
      in: { fader: [[0, 0], [12, sway(r, 0.62)], [32, 0.76], [swap, 0.8]], lp: [[0, sway(r, 0.22)], [8, 0.24], [32, 1]], lo: [[0, 0], [swap, 0], [swap, 0.5]] },
      out: { lo: [[swap, 0.5], [swap, 0]], fader: [[swap, 0.8], [swap + 8, 0.74], [swap + 32, 0.72], [len, 0]], hp: [[swap + 24, 0], [len, sway(r, 0.7)]] },
      fxIn: { type: pick(r, ['reverb', 'sweep', 'phaser']), lane: [[0, sway(r, 0.24)], [16, 0.08], [32, 0]] },
      fxOut: pick(r, [0, 1]) ? { type: 'vanish', keep: true, lane: [[swap + 32, 0], [len - 4, sway(r, 0.85, 0.05)], [len, 0.95]] } : { type: pick(r, ['gater', 'roll', 'sweep']), lane: [[swap + 32, 0], [swap + 44, sway(r, 0.34)], [len, 0.38]] } };
  },
  // the new track's mids under the primary, then its highs, then the bass
  mids(r) {
    const swap = pick(r, [64, 96]), len = swap + 64;
    return { name: 'mids under', swap, len,
      in: { fader: [[0, 0], [16, sway(r, 0.68)], [40, 0.76], [swap, 0.8]], hp: [[0, 0.5], [swap, 0.5], [swap, 0]], lp: [[0, sway(r, 0.5)], [16, 0.5], [40, 1]],
            hi: [[0, 0.15], [16, 0.15], [40, 0.5]], lo: [[0, 0], [swap, 0], [swap, 0.5]] },
      out: { lo: [[swap, 0.5], [swap, 0]], mid: [[swap, 0.5], [len, 0.4]], fader: [[swap, 0.8], [swap + 8, 0.74], [swap + 32, 0.72], [len, 0]], hp: [[swap + 32, 0], [len, sway(r, 0.8)]] },
      fxIn: { type: pick(r, ['flanger', 'phaser', 'sweep']), lane: [[0, sway(r, 0.3)], [32, 0]] },
      fxOut: { type: pick(r, ['gater', 'reverb', 'flanger', 'crush', 'roll']), lane: [[swap + 32, 0], [swap + 40, sway(r, 0.36)], [len, 0.36]] } };
  },
  // the three bands change decks one after the other, each on its own line: the highs, then the mids, then the bass;
  // the old track then dissolves through vanish or sink
  swap3(r) {
    const swap = pick(r, [64, 96]), len = swap + 48, a = swap - 32, b = swap - 16;
    return { name: 'three-band swap', swap, len,
      in: { fader: [[0, 0], [16, sway(r, 0.66)], [a, 0.78], [swap, 0.8]], hi: [[0, 0.12], [a, 0.12], [a, 0.5]], mid: [[0, 0.15], [b, 0.15], [b, 0.5]], lo: [[0, 0], [swap, 0], [swap, 0.5]] },
      out: { hi: [[a, 0.5], [a, 0.12]], mid: [[b, 0.5], [b, 0.15]], lo: [[swap, 0.5], [swap, 0]], fader: [[swap, 0.8], [len - 8, 0.76], [len, 0]] },
      fxIn: null,
      fxOut: { type: pick(r, ['vanish', 'sink']), keep: true, lane: [[swap + 8, 0], [len - 4, sway(r, 0.85, 0.05)], [len, 0.95]] } };
  },
  // the bass changes decks first, on the one; the new track then opens from below while the old one thins out
  bass(r) {
    const swap = 32, len = 64;
    return { name: 'bass first', swap, len,
      in: { fader: [[0, sway(r, 0.76, 0.02)], [swap, 0.8]], lp: [[0, 0.05], [8, 0.07], [48, 1]] },
      out: { lo: [[0, 0.5], [0, 0]], hp: [[16, 0], [56, sway(r, 0.75)]], fader: [[32, 0.8], [len, 0]] },
      fxIn: null,
      fxOut: pick(r, [0, 1]) ? { type: pick(r, ['sink', 'vanish']), keep: true, lane: [[swap, 0], [len - 4, sway(r, 0.8, 0.05)], [len, 0.9]] } : { type: pick(r, ['reverb', 'phaser']), lane: [[swap, 0], [56, sway(r, 0.38)], [len, 0.38]] } };
  },
  // highs in, then a break: no bass anywhere for some bars, and the new track drops on the one. The old track
  // stays for eight bars as echoing highs.
  drop(r) {
    const swap = 32, hole = pick(r, [4, 8]), len = 40, a = swap - hole;
    return { name: `break and drop (${hole} bars without bass)`, swap, len,
      in: { fader: [[0, 0], [16, sway(r, 0.56)], [a, 0.6], [swap, 0.62], [swap, 0.8]], hp: [[0, 1], [swap, 0.9], [swap, 0]], lo: [[0, 0], [swap, 0], [swap, 0.5]] },
      out: { lo: [[a, 0.5], [a, 0]], hp: [[a, 0], [swap, sway(r, 0.5)], [swap, 0.9]], fader: [[swap, 0.8], [swap, 0.62], [len, 0]] },
      fxIn: { type: 'reverb', lane: [[0, 0.2], [a, 0.12], [swap, 0]] },
      fxOut: { type: 'echo', keep: true, lane: [[swap, 0], [swap, sway(r, 0.45)], [len, 0.45]] } };      // the echo is this transition's exit
  },
};
const ODDS = [['tops', 4], ['lowpass', 3], ['mids', 3], ['swap3', 3], ['bass', 0.6], ['drop', 1]];       // mostly the long blends
function draw(r) {
  const total = ODDS.reduce((s, [, w]) => s + w, 0);
  let x = r() * total;
  for (const [name, w] of ODDS) { x -= w; if (x <= 0) return TRANSITIONS[name](r); }
  return TRANSITIONS.tops(r);
}
// the same transition with its bass swap and everything after it d bars (32-bar units) later
function delayed(t, d) {
  const shift = (pts) => pts.map(([x, v]) => [x >= t.swap ? x + d : x, v]);
  const lanes = (o) => Object.fromEntries(Object.entries(o).map(([key, pts]) => [key, shift(pts)]));
  const out = { ...t, in: lanes(t.in), out: lanes(t.out), swap: t.swap + d, len: t.len + d };
  if (t.fxIn) out.fxIn = { ...t.fxIn, lane: shift(t.fxIn.lane) };
  if (t.fxOut) out.fxOut = { ...t.fxOut, lane: shift(t.fxOut.lane) };
  return out;
}
// now and then the old track stays for a while as a thin layer of highs with an effect: three decks can sound
function linger(r, t) {
  if (t.len - t.swap < 32 || r() > 0.2) return t;
  const { swap, len } = t, end = len + 32;
  t.out.hp = [[swap + 32, 0], [swap + 48, 0.92]];
  t.out.fader = [[swap, 0.8], [swap + 8, 0.74], [swap + 32, 0.72], [swap + 48, sway(r, 0.5)], [len + 16, 0.5], [end, 0]];
  delete t.out.lp; delete t.out.mid;
  t.fxOut = { type: pick(r, ['gater', 'flanger', 'phaser', 'roll', 'crush', 'echo']), keep: true, lane: [[swap + 40, 0], [swap + 52, sway(r, 0.34)], [len + 16, 0.34], [end, 0.1]] };
  t.len = end; t.name += ', the old track stays as highs';
  return t;
}

// ---------- which track comes next: live, a stand-in of the same brain, or a transition piece ----------
// Plain logic, tested without audio. Kinds: 'live' (composed for this set), 'shelf' (a finished track of the brain),
// 'bridge' (a repertoire piece).
export class Transitions {
  constructor({ repertoire = true } = {}) { this.repertoire = repertoire; this.last = null; this.plan = null; this.why = ''; this.kinds = []; }
  // at the look-ahead line: liveSoon = the live track will be buffered at the line; shelf = the shelf may have a
  // track (false once the factory said it has none)
  decide({ liveSoon, shelf }) {
    if (liveSoon || this.last === 'bridge' || this.last === null) this.plan = 'live';      // after a piece only the live track
    else if (shelf) this.plan = 'shelf';
    else this.plan = this.repertoire ? 'bridge' : 'live';
    return this.plan;
  }
  // at the line, with what is ready now: the kind to cue, or null (the primary plays on for one more section)
  choose({ live, shelf, bridge }) {
    if (live) return 'live';                          // a buffered live track always comes first
    if (this.last === 'bridge' || this.last === null) return null;
    if (shelf && this.plan !== 'bridge') return 'shelf';
    if (bridge && this.repertoire && this.plan !== 'shelf') return 'bridge';
    return shelf ? 'shelf' : bridge && this.repertoire ? 'bridge' : null;
  }
  cued(kind) { this.last = kind; this.kinds.push(kind); this.plan = null; }
}

// ---------- the planner ----------
export class Planner {
  constructor({ rand = Math.random, section = SECTION, decks = 3 } = {}) {
    this.r = rand; this.sec = section; this.k = section / SECTION; this.n = decks; this.cutOdds = 0.3;
    this.doubleOdds = 0.2; this.doubleBass = []; this.rideOdds = 0.3;
    this.lanes = Array.from({ length: decks }, () => ({}));
    this.role = Array(decks).fill('idle');          // idle | primary | in | out
    this.events = []; this.primary = -1; this.active = []; this.nextStart = null; this.upcoming = null; this.origin = 0;
    this.needTrack = null; this.breakUntil = -1; this.waited = 0;
    this.fxOdds = { in: 0.4, out: 0.5 };            // how often the deck coming in and the deck going out get an effect
  }
  scale(points) { return points.map(([b, v]) => [b * this.k, v]); }
  // set a deck's lane from bar `from` on; the value it had there is held up to the first new point
  add(deck, key, from, points) {
    const lane = this.lanes[deck][key] || (this.lanes[deck][key] = []);
    const first = from + points[0][0], held = lane.length ? at(lane, first) : NEUTRAL[key];
    while (lane.length && lane[lane.length - 1][0] > first) lane.pop();
    lane.push([first, held]);
    for (const [b, v] of points) lane.push([from + b, v]);
  }
  fresh(deck, from, values) {
    this.lanes[deck] = {};
    for (const key of KEYS) this.lanes[deck][key] = [[from, key in values ? values[key] : NEUTRAL[key]]];
  }
  value(deck, key, pos) { const v = at(this.lanes[deck][key], pos); return v === undefined ? NEUTRAL[key] : v; }
  // the first track starts on deck 0 at bar G
  begin(G) {
    this.origin = G; this.primary = 0; this.role.fill('idle'); this.role[0] = 'primary';
    this.fresh(0, G, {});
    this.needTrack = this.plan(G, 1);          // the first transition starts after one section
  }
  // after a track became the primary: how long it plays alone and how the next one will come in
  plan(from, sections) {
    const t = this.upcoming = linger(this.r, draw(this.r));
    this.nextStart = from + sections * this.sec;
    // how long the next track will play alone: two to four sections, rarely one after a long tail (three decks overlap)
    // counted from the bass swap: with two sections the next track starts as the old one has left, with more the
    // primary plays alone in between; one section (rare, after a long tail) lets three decks overlap
    t.solo = t.len - t.swap > 64 ? pick(this.r, [1, 2, 3]) : pick(this.r, [2, 2, 2, 3]);       // mostly the next track enters as the old one has left
    this.waited = 0;
    return this.life(t);
  }
  // the bars a track needs: coming in, alone, and going out (three sections; a longer tail repeats its last section)
  life(t) { return Math.ceil((t.swap * this.k + (t.solo + 3) * this.sec) / 16) * 16 + 16; }
  idle() { return this.role.findIndex((x) => x === 'idle'); }
  // called once per bar, when bar G has just begun. ready() says whether the next track can start.
  tick(G, ready) {
    const cmds = [];
    for (const e of this.events.filter((e) => e.at <= G)) {
      if (e.cmd === 'primary') { this.role[this.primary] = 'out'; this.primary = e.deck; this.role[e.deck] = 'primary'; }
      if (e.cmd === 'stop') { this.role[e.deck] = 'idle'; this.active = this.active.filter((t) => t.out !== e.deck); }
      cmds.push(e);
    }
    this.events = this.events.filter((e) => e.at > G);
    for (const lanes of this.lanes) for (const lane of Object.values(lanes)) {      // what lies more than eight bars back is forgotten
      let k = 0; while (k + 1 < lane.length && lane[k + 1][0] < G - 8) k++;
      if (k) lane.splice(0, k);
    }
    if (this.needTrack) { cmds.push({ cmd: 'want', bars: this.needTrack }); this.needTrack = null; }
    if (this.nextStart != null && G === this.nextStart - 1) {
      const deck = this.idle();
      if (deck >= 0 && ready()) this.commit(G + 1, deck, cmds);
      else { this.nextStart += this.sec; this.waited++; cmds.push({ cmd: 'say', text: deck < 0 ? 'all decks are busy: one more section' : 'the next track is still being composed: one more section' }); }
    }
    // a cut: one deck falls silent for the last one or two bars of a 16-bar phrase and comes back on its line,
    // while both tracks play; mostly the deck that is not the primary, the primary only when the other plays in full
    const inPhrase = (((G - this.origin) % 16) + 16) % 16;
    if (inPhrase === 0 && G > this.origin && this.r() < this.cutOdds) {
      const n = pick(this.r, [1, 1, 2]), a = G + 16 - n, b = G + 16;
      const t = this.active.find((x) => x.start <= a - 4 && x.end >= b + 4);
      const clash = this.breakFrom != null && this.breakUntil > a - 1 && this.breakFrom < b + 1;
      if (t && !clash && this.value(t.in, 'fader', a) >= 0.55 && this.value(t.out, 'fader', a) >= 0.55) {
        const other = this.primary === t.in ? t.out : t.in;
        const full = this.value(other, 'fader', a) >= 0.7 && this.value(other, 'hp', a) <= 0.3 && this.value(other, 'lp', a) >= 0.9 && this.value(other, 'lo', a) >= 0.3;
        const deck = full && this.r() < 0.35 ? this.primary : other;
        this.add(deck, 'cut', 0, [[a, 1], [a, 0], [b, 0], [b, 1]]);
        this.lastCut = { deck, a, b, n };
      }
    }
    // an EQ ride: on an 8-bar line, while two tracks play, the deck that is not the primary dips its mids or lifts
    // its highs for a few bars and comes back; only on a band that has no planned move ahead
    if (inPhrase % 8 === 0 && G > this.origin && this.r() < this.rideOdds) {
      const t = this.active.find((x) => x.start <= G && x.end >= G + 10);
      if (t) {
        const other = this.primary === t.in ? t.out : t.in;
        const key = pick(this.r, ['mid', 'mid', 'hi']);
        const lane = this.lanes[other][key] || [];
        const free = !lane.length || lane[lane.length - 1][0] <= G;
        const now = this.value(other, key, G);
        if (free && this.value(other, 'fader', G) >= 0.55 && now >= 0.35) {
          const to = key === 'mid' ? Math.max(0.2, now - sway(this.r, 0.22, 0.05)) : Math.min(0.68, now + sway(this.r, 0.12, 0.03));
          const hold = pick(this.r, [2, 4]);
          this.add(other, key, 0, [[G, now], [G + 1, to], [G + 1 + hold, to], [G + 2 + hold, now]]);
          this.lastRide = { deck: other, key, at: G, to };
        }
      }
    }
    const inSection = (((G - this.origin) % this.sec) + this.sec) % this.sec;
    if (inSection === 0 && !this.active.length && this.nextStart != null && this.nextStart - G >= this.sec && G > this.origin && this.r() < 0.25) {
      const hole = pick(this.r, [4, 8]) * this.k, a = G + this.sec - hole, b = G + this.sec;      // a bass break before the next section
      this.add(this.primary, 'lo', 0, [[a, 0.5], [a, 0], [b, 0], [b, 0.5]]);
      this.add(this.primary, 'hp', 0, [[a, 0], [b, sway(this.r, 0.3)], [b, 0]]);
      this.breakUntil = b; this.breakFrom = a;
    }
    return cmds;
  }
  commit(S, deck, cmds) {
    const old = this.primary, k = this.k;
    let t = this.upcoming;
    // a transition that holds the new track's bass back until the swap needs a bass in the old track meanwhile:
    // when the old track has bass in under half of those bars (a piece without bass, a long break), the bass comes
    // in first, from the new track (bassShare is given by the controller; without it every transition stays)
    if (this.bassShare && t.in.lo) {
      const share = this.bassShare(S, S + t.swap * k);
      if (share != null && share < 0.5) {
        const b = TRANSITIONS.bass(this.r);
        b.solo = t.solo; b.name += ' (the old track has little bass)';
        t = this.upcoming = b;
      }
    }
    // now and then both basslines play in full for the last 8 units (8 bars at the 32-bar pace) before the swap
    if (this.r() < this.doubleOdds && t.in.lo && t.swap >= 32 && !/little bass|drop/.test(t.name)) {
      const share = this.bassShare ? this.bassShare(S + (t.swap - 8) * k, S + t.swap * k) : 1;
      if (share == null || share >= 0.75) {
        const from = t.swap - 8;
        t = this.upcoming = { ...t, in: { ...t.in, lo: [[0, 0], [from, 0], [from, 0.5]] }, name: `${t.name}, both basslines for ${8 * k} bars before the swap` };
        this.doubleBass.push({ deck, from: S + from * k, to: S + t.swap * k });
      }
    }
    // the other way round: when the new track has bass in under half of the section after the swap while the old
    // track has it, the swap waits one 32-bar unit (the old track keeps the bass a while longer)
    if (this.newBassShare && this.bassShare && t.in.lo && !/little bass/.test(t.name)) {
      const sw = S + t.swap * k, mine = this.newBassShare(S, sw, sw + this.sec), theirs = this.bassShare(sw, sw + this.sec);
      if (mine != null && mine < 0.5 && theirs != null && theirs >= 0.5) { t = this.upcoming = delayed(t, 32); t.name += ' (the bass changes later: the new track has little bass yet)'; }
    }
    // effects are used sparingly: many transitions have none on one deck or on both
    if (t.fxIn && this.r() >= this.fxOdds.in) t.fxIn = null;
    if (t.fxOut && !t.fxOut.keep && this.r() >= this.fxOdds.out) t.fxOut = null;
    const first = Object.fromEntries(Object.entries(t.in).map(([key, pts]) => [key, pts[0][1]]));
    this.fresh(deck, S, { fader: 0, ...first, fx: t.fxIn ? t.fxIn.lane[0][1] : 0 });
    for (const [key, pts] of Object.entries(t.in)) this.add(deck, key, S, this.scale(pts));
    for (const [key, pts] of Object.entries(t.out)) this.add(old, key, S, this.scale(pts));
    if (t.fxIn) { this.add(deck, 'fx', S, this.scale(t.fxIn.lane)); cmds.push({ cmd: 'fx', at: S - 1, deck, type: t.fxIn.type }); }
    if (t.fxOut) {
      this.add(old, 'fx', S, this.scale(t.fxOut.lane));
      this.events.push({ cmd: 'fx', at: S + t.swap * k - 1, deck: old, type: t.fxOut.type });      // set while its amount is still zero
    }
    const swap = S + t.swap * k, end = S + t.len * k;
    this.role[deck] = 'in';
    cmds.push({ cmd: 'cue', at: S - 1, deck });
    this.events.push({ cmd: 'primary', at: swap, deck }, { cmd: 'stop', at: end, deck: old });
    this.events.sort((a, b) => a.at - b.at);
    this.active.push({ name: t.name, start: S, swap, end, in: deck, out: old });
    // the track after this one is asked for as soon as this one starts: the brain has the whole transition to compose it
    this.events.push({ cmd: 'want', at: S, bars: this.plan(swap, t.solo) });
    this.events.sort((a, b) => a.at - b.at);
  }
  // what the DJ is doing at bar G, for the readout
  describe(G) {
    const d = (i) => `deck ${i + 1}`;
    const { parts, lastOut } = this.moving(G);
    if (this.nextStart != null && this.nextStart > G && !this.active.some((t) => G < t.swap)) {
      const alone = this.nextStart - lastOut;
      parts.push(`${alone > 0 ? `then ${d(this.primary)} plays alone for ${alone} bars, ` : ''}the next track enters in ${this.nextStart - G} bars (${this.upcoming.name})`);
    }
    return { primary: this.primary, section: Math.floor((G - this.origin) / this.sec) + 1, bar: ((((G - this.origin) % this.sec) + this.sec) % this.sec) + 1, text: parts.join(' · ') };
  }
  // the transitions under way at bar G and a bass break, in words; lastOut: the bar the last outgoing track leaves
  moving(G) {
    const d = (i) => `deck ${i + 1}`;
    const parts = [];
    let lastOut = G;
    for (const t of this.active) {
      if (G < t.swap) parts.push(`${d(t.in)} is coming in (${t.name}): both play side by side, the bass changes decks in ${t.swap - G} bars`);
      else { parts.push(`${d(t.out)} stays beside it and fades out over the next ${t.end - G} bars`); lastOut = Math.max(lastOut, t.end); }
    }
    if (G >= this.breakFrom && G < this.breakUntil) parts.push(`bass break, ${this.breakUntil - G} bars`);
    return { parts, lastOut };
  }
}

// ---------- the page side: tracks, decks, the clock and the automation ----------
const LOOK = 1.4, GRAIN = 0.05;      // seconds scheduled ahead (a hidden tab's timers fire once a second), and the grain
const EDGE = 0.03;                   // seconds a jump takes, ending on its bar line

const BRIDGE_BARS = 352;             // a bridge is asked for this long (the factory repeats its piece): it carries the set while the brain is late
const AHEAD = 2;                     // sections before the line at which the next track is decided
const shortBrain = (checkpoint) => String(checkpoint || '').replace(/^slate_/, '').replace(/_(e\d+(b\d+)?|l\d{8}|latest)$/, '');

class Track {
  constructor(id, want, kind) { this.id = id; this.want = want; this.kind = kind; this.bars = []; this.done = false; this.pool = null; this.poolName = null; this.bpm = null; this.error = null; this.state = ''; this.checkpoint = null; }
  // barSec: how long a bar plays (s). A finished track is ready; a track still being composed may start once
  // 48 bars lie ahead and the brain composes at least 1.15 times as fast as the deck plays (the deck then never
  // catches up; if the brain stalls, the engine holds the last bar until the next one comes)
  // A track still being composed may start when the deck will not run out of it: the bars composed so far plus
  // what the brain composes while the deck plays (at 85% of its measured pace) cover the bars the deck needs of it.
  // A brain a little slower than the deck still qualifies once enough bars are banked.
  // seconds until ready(barSec) holds at the measured rate (null while no rate is measured)
  secondsToReady(barSec) {
    if (this.done || !this.t0 || !barSec) return this.done ? 0 : null;
    const seconds = (performance.now() - this.t0) / 1000, rate = (this.bars.length - this.n0) / Math.max(1, seconds);
    if (seconds < 12 || rate <= 0) return null;
    const need = this.want || this.total || 128, bank = Math.max(32, need - 0.85 * rate * need * barSec);
    return Math.max(0, (Math.min(need, bank) - this.bars.length) / rate, 12 - seconds);
  }
  ready(barSec, ahead = 0) {
    if (this.error || !this.pool || this.bars.length < 16) return false;
    if (this.done) return this.bars.length >= (this.total || 0);
    if (!barSec || !this.t0 || this.bars.length < 32) return false;
    const seconds = (performance.now() - this.t0) / 1000;
    if (seconds < 12) return false;
    // ahead: seconds until the line; the bars the brain composes meanwhile count as banked
    const rate = (this.bars.length - this.n0) / seconds, need = this.want || this.total || 128;
    return Math.min(need, this.bars.length + rate * ahead) + 0.85 * rate * need * barSec >= need;
  }
}

export class AutoDJ {
  // desk: { chans, ctx(), ensureAudio(), bpm(), userBpm(), setBpm(v), status(text), paint(), brainName() }
  constructor(desk, { rand = Math.random, section = SECTION } = {}) {
    this.desk = desk; this.rand = rand; this.section = section; this.on = false; this.planner = null; this.clock = null;
    this.tracks = []; this.deckTrack = [null, null, null]; this.scheduledTo = 0; this.last = {}; this.model = 'leader';
    this.live = null; this.shelf = null; this.bridge = null;    // the next live track, a stand-in of the brain, a transition piece: not on a deck yet
    this.order = new Transitions();
    this.say = ''; this.session = 0; this.painted = 0; this.debug = false;
    this.sid = Math.random().toString(36).slice(2, 12);      // this page's name at the track factory: pages do not disturb each other
  }
  async post(path, body) { return (await fetch(path, { method: 'POST', body: JSON.stringify(body) })).json(); }
  // a live track: composed new by the chosen brain (the factory primes it on a new passage first); a bridge: a piece
  // of the repertoire, picked by the factory by its stars and not one of the last few, played from its start
  // shelf: a finished track of the brain (kept_only: the factory composes none in its place; a server without the
  // switch starts one, and follow() drops it)
  async request(bars, kind) {
    const session = this.session;
    let j;
    const body = kind === 'bridge' ? { model: 'repertoire', bars, quick: true, fallback: this.model, session: this.sid }
      : kind === 'shelf' ? { model: this.model, bars, quick: true, kept_only: true, session: this.sid }
        : { model: this.model, bars, quick: false, session: this.sid };
    try { j = await this.post('/api/autodj/track', body); } catch { this.note('no answer from the server'); return null; }
    if (session !== this.session || !j.id) return null;
    const t = new Track(j.id, bars, kind);
    this.tracks.push(t); this.follow(t, session);
    return t;
  }
  // the bars a live track is asked for: the longest life a track can have in the set (planner.life)
  liveBars() { return Math.ceil((96 * (this.section / SECTION) + 6 * this.section) / 16) * 16 + 16; }
  // the next live track is asked for once the live track on the decks is composed (or none plays): one at a time
  ensureLive(changed = false) {
    if (this.live || this.liveAsking || performance.now() < (this.liveRetryAt || 0)) return;
    if (!changed && this.deckTrack.some((t) => t && t.kind === 'live' && !t.done && !t.error)) return;
    this.liveAsking = true;
    this.request(this.liveBars(), 'live').then((tr) => { this.liveAsking = false; if (tr) this.live = tr; else this.liveRetryAt = performance.now() + 30000; });
  }
  // the brain was changed in the picker: the next track is the new brain's (the factory loads it while the decks
  // play on); a live track of the old brain that is not on a deck yet is dropped
  brainChanged() {
    if (!this.on) return;
    if (this.live) { this.live = null; this.keep(); }
    this.shelf = null; this.noShelf = false; this.liveRetryAt = 0; this.brain = null; this.decidedFor = null; this.order.plan = null;
    this.ensureLive(true);
  }
  // the factory has no finished track of the brain: a transition piece instead (or, without a repertoire, the wait)
  shelfEmpty() {
    this.noShelf = true;
    if (this.order.plan !== 'shelf') return;
    this.order.plan = this.order.repertoire ? 'bridge' : 'live';
    this.decision = this.decision.replace(/: an earlier track by .*$/, this.order.repertoire ? ': no earlier track, a transition piece' : ': no earlier track, the track on the decks plays on');
    this.ensureStandIn();
  }
  // the stand-in the decision asked for: a track of the brain from the shelf, else a transition piece
  ensureStandIn() {
    const plan = this.order.plan;
    if (plan === 'shelf' && !this.shelf && !this.shelfAsking) {
      this.shelfAsking = true;
      this.request(this.liveBars(), 'shelf').then((tr) => {
        this.shelfAsking = false;
        if (tr) this.shelf = tr;
        else this.shelfEmpty();
      });
    }
    if (plan === 'bridge' && !this.bridge && !this.bridgeAsking) {
      this.bridgeAsking = true;
      this.request(BRIDGE_BARS, 'bridge').then((tr) => { this.bridgeAsking = false; if (tr) this.bridge = tr; });
    }
  }
  // fetch a track's bars as the factory composes them, and its sounds once
  async follow(t, session) {
    while (session === this.session && this.on && !(t.done && t.bars.length >= t.total)) {
      let j;
      try { j = await (await fetch(`/api/autodj/track?id=${t.id}&from=${t.bars.length}`)).json(); } catch { await new Promise((r) => setTimeout(r, 2000)); continue; }
      if (j.error) {                                       // the brain failed or the server lost the track (a restart)
        t.error = j.error; t.done = true;
        if (t === this.live) {                             // the track on the decks plays on; the brain is asked again in 30 s
          this.live = null; this.liveRetryAt = performance.now() + 30000;
          this.note(`the brain did not compose the next track (${j.error}): asking again`);
        }
        if (t === this.bridge) this.bridge = null;
        if (t === this.shelf) this.shelf = null;
        if (t === this.first && !this.planner) this.note(`track: ${j.error}`);
        break;
      }
      for (const b of j.bars || []) t.bars.push(b);
      if (!t.t0 && t.bars.length) { t.t0 = performance.now(); t.n0 = t.bars.length; }      // the composing rate is read from here on
      t.total = j.total; t.done = j.done; t.state = j.state || ''; t.bpm = j.bpm; t.checkpoint = j.checkpoint; t.kept = j.kept; t.seed = j.seed; t.name = j.name || null; if (j.layout) t.layout = j.layout;
      if (j.stars != null) t.stars = j.stars;
      if (t.kind === 'shelf' && !j.kept) {                // a server without kept_only composes in its place: not a stand-in
        if (t === this.first && !this.planner) t.kind = 'live';          // the opening: that composition is the first live track
        else {
          if (t === this.shelf) { this.shelf = null; this.shelfEmpty(); }
          t.error = 'no finished track'; t.done = true; this.keep(); break;
        }
      }
      if (t.kind === 'live' && j.checkpoint) this.brain = shortBrain(j.checkpoint);
      if (j.pool && !t.pool && !t.poolBusy) {
        t.poolBusy = true; t.poolName = j.pool;
        try { t.pool = await (await fetch(`/api/autodj/pool?id=${t.id}`)).json(); } catch { /* asked again on the next round */ }
        t.poolBusy = false;
      }
      this.paint();
      await new Promise((r) => setTimeout(r, (j.bars || []).length >= 64 ? 50 : 900));
    }
  }
  note(text) { this.say = text; this.paint(); }
  async start(model) {
    if (this.on) return;
    const d = this.desk;
    this.model = model || 'leader'; this.on = true; this.session++; this.planner = null; this.clock = null; this.tracks = []; this.deckTrack = [null, null, null];
    this.live = null; this.shelf = null; this.bridge = null; this.liveAsking = this.shelfAsking = this.bridgeAsking = false; this.liveRetryAt = 0; this.brain = null;
    this.noShelf = false; this.decidedFor = null; this.decision = '';
    this.note('starting…'); d.paint();
    await this.post('/api/autodj/stop', { session: this.sid }).catch(() => {});     // what this page left behind
    let pieces = 0;
    try { pieces = (await (await fetch('/api/autodj/models')).json()).repertoire || 0; } catch { /* no bridges then */ }
    this.order = new Transitions({ repertoire: pieces > 0 });
    await d.ensureAudio();
    const ctx = d.ctx();
    if (ctx.state !== 'running') await ctx.resume();
    for (const ch of d.chans) { if (ch.playing) ch.stop(); ch.autoBegin(); }
    const session = this.session;
    // the set starts at once with a finished track of the chosen brain from the shelf when there is one (the
    // brain composes its first live track meanwhile); without one the first live track opens it, once enough of it
    // is composed that the deck does not catch up with the brain (Track.ready), with the loading bar and the time
    // it still needs in the readout
    this.first = await this.request(this.liveBars(), 'shelf');
    if (session !== this.session) return;
    if (!this.first) this.first = await this.request(this.liveBars(), 'live');
    if (!this.first) { this.stop('the server did not start a track'); return; }
    this.note('the brain composes the first track…');
    while (session === this.session && this.on && !(this.first.pool && this.first.ready(240 / d.bpm()))) {
      if (this.first.error) {
        if (this.first.kind !== 'shelf') { this.stop(this.first.error); return; }
        this.first = await this.request(this.liveBars(), 'live');      // nothing on the shelf after all
        if (!this.first) { this.stop('the server did not start a track'); return; }
        continue;
      }
      const f = this.first, eta = f.secondsToReady(240 / d.bpm());
      const composing = f.bars.length ? `${this.brainLabel()} composes the first track · ${f.bars.length} of ${f.want} bars, ${eta != null ? `starts in about ${Math.max(1, Math.round(eta))} s` : 'starts once the brain stays ahead of the decks'}` : '';
      const st = f.state || '';                          // what the brain does first: an earlier track, a load, or ours
      this.note(composing || (/^composing /.test(st) && !st.includes(f.id) ? `the brain finishes an earlier track first (${st.replace(/^composing /, '')}), then composes yours…`
        : st && !/^idle/.test(st) ? `${st}…` : 'waiting for the first track…'));
      await new Promise((r) => setTimeout(r, 500));
    }
    if (session !== this.session || !this.on) return;
    // the set's tempo: the one the listener set, else the first live track's own; every later track, the bridges
    // included, plays at it (one tempo for all decks)
    if (d.userBpm() == null && this.first.bpm) d.setBpm(this.first.bpm, false);
    this.say = '';
    this.armed = true;                                   // the next bar line starts the set
  }
  stop(why) {
    if (!this.on) return;
    this.on = false; this.session++; this.armed = false; this.planner = null; this.clock = null; this.live = null; this.shelf = null; this.bridge = null;
    for (const ch of this.desk.chans) { ch.endTrack(); ch.autoEnd(); ch.setRole(''); }
    this.post('/api/autodj/stop', { session: this.sid }).catch(() => {});
    this.say = why ? `stopped: ${why}` : ''; this.desk.paint(); this.paint();
  }
  // the share of the bars a..b-1 (the set's clock) in which the primary deck's track has a bass note; null when
  // the track or its place is not known. Past a track's end its last bar stands for the section that goes round.
  bassShare(a, b) {
    const p = this.planner ? this.planner.primary : -1, tr = p >= 0 ? this.deckTrack[p] : null, from = this.deckFrom ? this.deckFrom[p] : null;
    if (!tr || from == null || !tr.bars || !tr.bars.length) return null;
    let n = 0, yes = 0;
    for (let g = a; g < b; g++) {
      const i = Math.min(g - from, tr.bars.length - 1);
      if (i < 0) continue;
      const bar = tr.bars[i], pat = bar && bar.patterns && bar.patterns[1];
      const has = pat ? (pat.notes || []).some((x) => x != null) : !!(bar && bar.tokens && bar.tokens[1] > 0);
      n++; if (has) yes++;
    }
    return n ? yes / n : null;
  }
  // the same for the track the next cue will take (the finished or ready new track, else the stand-in), placed
  // to start at bar S; null while those bars are not composed yet
  newBassShare(S, a, b) {
    const bs = this.clock ? this.clock.len : 2;
    const kind = this.order.choose(this.readyNow(bs));
    const tr = kind ? this[kind] : null;
    if (!tr || !tr.bars || tr.bars.length < b - S) return null;
    let n = 0, yes = 0;
    for (let g = a; g < b; g++) {
      const bar = tr.bars[g - S], pat = bar && bar.patterns && bar.patterns[1];
      const has = pat ? (pat.notes || []).some((x) => x != null) : !!(bar && bar.tokens && bar.tokens[1] > 0);
      n++; if (has) yes++;
    }
    return n ? yes / n : null;
  }
  timeOf(pos) { return this.clock.t + (pos - this.clock.G) * this.clock.len; }
  posOf(t) { return this.clock.G + (t - this.clock.t) / this.clock.len; }
  // deck 0's engine began bar G at time t (the context's clock, exact); every deck is on the same bar line
  onBar(G, t) {
    if (!this.on) return;
    const d = this.desk, len = 240 / d.bpm();
    this.clock = { G, t, len };
    if (this.armed) {                                    // the set starts on the next bar line
      this.armed = false;
      this.planner = new Planner({ rand: this.rand, section: this.section });
      this.planner.bassShare = (a, b) => this.bassShare(a, b);
      this.planner.newBassShare = (S, a, b) => this.newBassShare(S, a, b);
      this.planner.begin(G + 1);
      this.deckFrom = [G + 1, null, null];
      this.deckTrack[0] = this.first; d.chans[0].cueTrack(this.first); d.chans[0].setRole('primary');
      this.order.cued(this.first.kind);
      this.scheduledTo = 0; this.last = {};
    }
    if (!this.planner) return;
    const P = this.planner;
    const bs = 240 / d.bpm();
    this.ensureLive();
    // the look-ahead: two sections before the line the next track is decided (again after every postponed line)
    if (P.nextStart != null && G >= P.nextStart - AHEAD * this.section && this.decidedFor !== P.nextStart) {
      this.decidedFor = P.nextStart;
      const lv = this.live, soon = !!(lv && lv.ready(bs, (P.nextStart - G) * bs));
      const plan = this.order.decide({ liveSoon: soon, shelf: !this.noShelf });
      const buffered = lv ? `${lv.bars.length} of ${lv.want} bars` : 'not started';
      this.decision = plan === 'live' ? (soon ? `next track buffered ${buffered}` : `waiting for the next track (${buffered})`)
        : plan === 'shelf' ? `the brain is late (${buffered}): an earlier track by ${this.brainLabel()}`
          : `the brain is late (${buffered}): a transition piece`;
    }
    this.ensureStandIn();
    const now = this.readyNow(bs);
    for (const c of P.tick(G, () => this.order.choose(now) != null)) {
      if (this.debug) console.log(`dj bar ${G}: ${JSON.stringify(c)}`);
      const ch = c.deck != null ? d.chans[c.deck] : null;
      if (c.cmd === 'want') this.ensureLive();
      else if (c.cmd === 'cue') {
        for (const key of KEYS) delete this.last[c.deck + key];
        const kind = this.order.choose(now), tr = this[kind];
        // the reason: the decision taken ahead, or a live track that was buffered by the line after all
        const why = kind === 'live' && this.order.plan !== 'live' ? `next track buffered ${tr.bars.length} of ${tr.want} bars` : this.decision;
        this[kind] = null;
        this.order.cued(kind); this.decidedFor = null; this.lastCue = { kind, why };
        this.deckTrack[c.deck] = tr; ch.cueTrack(tr); ch.setRole('in'); this.say = '';
        if (this.deckFrom) this.deckFrom[c.deck] = G + 1;                // the track's first bar sounds on the next bar line
      }
      else if (c.cmd === 'primary') { for (const o of d.chans) if (o.role === 'primary') o.setRole('out'); ch.setRole('primary'); }
      else if (c.cmd === 'stop') { ch.endTrack(); ch.setRole(''); ch.autoRest(); this.deckTrack[c.deck] = null; this.keep(); }
      else if (c.cmd === 'fx') ch.setFx(c.type, { echoSteps: [3, 3, 4, 6][Math.floor(this.rand() * 4)], gate: Math.floor(this.rand() * 4), rollSteps: [1, 1, 2][Math.floor(this.rand() * 3)] });
      else if (c.cmd === 'say') this.say = /busy/.test(c.text) ? c.text : '';     // a late live track: the readout says it
    }
    for (let i = 0; i < d.chans.length; i++) {           // the gater's pattern for the bar after this one
      if (P.role[i] !== 'idle') d.chans[i].gate(t + len, len, P.value(i, 'fx', G + 1));
    }
    this.pump(); this.paint(); d.paint();
  }
  readyNow(bs) { return { live: !!(this.live && this.live.ready(bs)), shelf: !!(this.shelf && this.shelf.ready(bs)), bridge: !!(this.bridge && this.bridge.ready(bs)) }; }
  keep() { this.post('/api/autodj/keep', { session: this.sid, ids: [...this.deckTrack, this.live, this.shelf, this.bridge].filter((t) => t && !t.error).map((t) => t.id) }).catch(() => {}); }
  // schedule every automated value from where the schedule ends to LOOK seconds ahead; jumps land on their bar line
  pump() {
    if (!this.on || !this.planner || !this.clock) return;
    const d = this.desk, ctx = d.ctx(), P = this.planner, now = ctx.currentTime, horizon = now + LOOK;
    let from = Math.max(this.scheduledTo, now + 0.01);
    if (this.scheduledTo < now) this.last = {};          // fell behind (a sleeping tab): start the lines again
    const edge = EDGE / this.clock.len;
    const lanes = d.chans.map((_, i) => (P.role[i] === 'idle' ? null : Object.fromEntries(KEYS.map((key) => [key, soft(P.lanes[i][key], edge)]))));
    while (from + GRAIN <= horizon) {
      const to = from + GRAIN, p0 = this.posOf(from), p1 = this.posOf(to);
      for (let i = 0; i < d.chans.length; i++) {
        if (!lanes[i]) continue;
        const ch = d.chans[i];
        for (const key of KEYS) {
          const lane = lanes[i][key], id = i + key, v = at(lane, p1), seen = this.last[id];
          const corners = lane.filter(([b]) => b > p0 && b <= p1);
          if (!corners.length && seen && seen.v === v) { seen.t = to; seen.idle = true; continue; }
          if (!seen || seen.idle) ch.autoSet(key, seen ? seen.v : at(lane, p0), from);      // anchor: a line starts here
          for (const [b, value] of corners) ch.autoLine(key, value, this.timeOf(b));       // every corner on its own time: a change ends on its bar line
          ch.autoLine(key, v, to);
          this.last[id] = { v, t: to, idle: false };
        }
      }
      from = to;
    }
    this.scheduledTo = from;
    if (now - this.painted > 0.1) {                      // the knobs follow
      this.painted = now;
      const pos = this.posOf(now);
      for (let i = 0; i < d.chans.length; i++) if (P.role[i] !== 'idle') d.chans[i].autoShow(Object.fromEntries(KEYS.map((k) => [k, P.value(i, k, pos)])));
    }
  }
  retime() { this.scheduledTo = 0; if (this.clock) this.clock.len = 240 / this.desk.bpm(); }
  brainLabel() { return this.brain || shortBrain((this.desk.brainName && this.desk.brainName()) || '') || 'the brain'; }
  pieceName(t) {
    const name = String(t.name || 'without a name').replace(/\s+/g, ' ').trim(), short = name.length > 40 ? `${name.slice(0, 38).replace(/[ ,·]+$/, '')}…` : name;
    return `repertoire piece "${short}"${t.stars ? ` (${t.stars} star${t.stars === 1 ? '' : 's'})` : ''}`;
  }
  // in time order: what plays now, the transition under way (and which case brought the track in), what comes next
  readout() {
    if (!this.on) return this.say || '';
    if (!this.planner || !this.clock) return this.say || 'starting…';
    const P = this.planner, G = this.clock.G, s = P.describe(G), brain = this.brainLabel();
    const cur = this.deckTrack[P.primary], lv = this.live, by = (t) => shortBrain(t && t.checkpoint) || brain;     // a track names its own brain
    const of = (t) => `${t.bars.length} of ${t.want} bars`;
    const parts = [`section ${s.section}, bar ${s.bar}/${this.section}`];
    const composing = lv && lv.bars.length ? `composes the next track · ${of(lv)}` : lv && lv.state && /^(finding|loading) /.test(lv.state) ? `prepares the next track: ${lv.state}` : 'composes the next track';
    if (cur && cur.kind === 'bridge') parts.push(`transition: ${this.pieceName(cur)} while ${brain} ${composing}`);
    else if (cur && cur.kind === 'shelf') parts.push(`earlier track by ${by(cur)} while ${brain} ${composing}`);
    else if (cur) parts.push(`live: new track by ${by(cur)}${cur.done ? '' : ` · ${of(cur)} composed`}`);
    const moving = P.moving(G).parts;
    parts.push(...moving);
    if (this.lastCue && P.active.some((t) => G < t.swap) && this.lastCue.why) parts.push(`it came in because: ${this.lastCue.why}`);
    if (P.nextStart != null && P.nextStart > G && !P.active.some((t) => G < t.swap)) {
      const n = Math.ceil((P.nextStart - G) / this.section), when = `in ${n} section${n === 1 ? '' : 's'}`, plan = this.order.plan;
      const what = plan === 'shelf' ? `earlier track by ${this.shelf ? by(this.shelf) : brain}` : plan === 'bridge' ? `transition piece${this.bridge ? `, ${this.pieceName(this.bridge)},` : ''}` : `new track by ${brain}`;
      const why = plan ? this.decision : lv ? `${of(lv)} composed, decided ${AHEAD} sections before the line` : '';
      parts.push(`next: ${what} ${when}${why ? ` · ${why}` : ''}`);
    }
    if (!this.order.repertoire) parts.push('the repertoire is empty: no transition pieces');
    if (this.say) parts.push(this.say);
    return parts.join(' · ');
  }
  paint() { if (this.desk.readout) this.desk.readout(this.readout()); }
}
