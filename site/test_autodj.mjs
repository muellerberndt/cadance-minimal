// The Auto DJ's planner without audio: lanes, jumps and a long simulated set.
//   node site/test_autodj.mjs
import assert from 'node:assert/strict';
import { Planner, Transitions, AutoDJ, at, jumps, soft, mulberry, KEYS, NEUTRAL, TRANSITIONS, SECTION } from './autodj.js';

// lanes
const lane = [[0, 0], [16, 0.6], [32, 0.6], [32, 0.8], [48, 0]];
assert.equal(at(lane, -5), 0); assert.equal(at(lane, 8), 0.3); assert.equal(at(lane, 31.999) > 0.59, true);
assert.equal(at(lane, 32), 0.8); assert.equal(at(lane, 40), 0.4); assert.equal(at(lane, 99), 0);
assert.deepEqual(jumps(lane, 31.5, 32), [{ bar: 32, before: 0.6, after: 0.8 }]);
assert.deepEqual(jumps(lane, 32, 33), []);
// a jump becomes a short line that ends on its bar line; nothing else moves
assert.deepEqual(soft(lane, 0.02), [[0, 0], [16, 0.6], [31.98, 0.6], [32, 0.8], [48, 0]]);
assert.deepEqual(soft([[8, 0.5], [8, 0.5], [8, 0], [9, 0]], 0.02), [[7.98, 0.5], [8, 0], [9, 0]]);
assert.deepEqual(soft([[8, 0.5], [8, 0.5], [12, 0]], 0.02), [[8, 0.5], [12, 0]]);
assert.equal(at(soft(lane, 0.02), 32), 0.8); assert.equal(at(soft(lane, 0.02), 31.97), 0.6);

// every transition: lanes in rising order, values in range, the bass changes decks on a section line
for (const [name, make] of Object.entries(TRANSITIONS)) {
  for (let seed = 1; seed <= 40; seed++) {
    const t = make(mulberry(seed));
    assert.equal(t.swap % 32, 0, `${name}: swap on a section line`);
    assert.ok(t.len > t.swap || name === 'bass');
    for (const side of [t.in, t.out]) for (const [key, pts] of Object.entries(side)) {
      assert.ok(KEYS.includes(key), `${name}: ${key}`);
      for (let i = 0; i < pts.length; i++) {
        assert.ok(pts[i][1] >= 0 && pts[i][1] <= 1, `${name}.${key} value`);
        if (i) assert.ok(pts[i][0] >= pts[i - 1][0], `${name}.${key} order`);
      }
    }
    // effects stay in the background; vanish and sink are the old track's exit and may take it all the way
    for (const fx of [t.fxIn, t.fxOut]) if (fx) for (const [, v] of fx.lane) assert.ok(v >= 0 && v <= (/vanish|sink/.test(fx.type) && fx === t.fxOut ? 1 : 0.6), `${name}: an effect stays in the background`);
  }
}

// a long set: one primary at all times, the bass in one deck only on section lines, never more than three decks
for (const SEC of [SECTION, 16]) for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
  const P = new Planner({ rand: mulberry(seed), section: SEC });
  let fxUsed = 0, together = 0;
  let next = false, wants = 0, cues = 0, three = 0, primaries = new Set(), late = seed % 3 === 0 ? 40 : 0, askedAt = 0;
  const startAt = [0, Infinity, Infinity];          // a deck sounds from the bar its track starts on
  P.begin(0);
  for (let G = -1; G < 3000; G++) {
    if (G < 0) continue;
    for (const c of P.tick(G, () => next && G - askedAt >= late)) {
      if (c.cmd === 'want') { wants++; next = true; askedAt = G; assert.ok(c.bars >= 48 && c.bars <= 352, 'a track of a sane length'); }
      if (c.cmd === 'cue') { cues++; next = false; startAt[c.deck] = G + 1; assert.equal((G + 1 - P.origin) % SEC, 0, 'a track starts on a section line'); }
      if (c.cmd === 'primary') { assert.equal((G - P.origin) % SEC, 0, 'the primary changes on a section line'); primaries.add(c.deck); }
      if (c.cmd === 'stop') { assert.equal((G - P.origin) % 4, 0); startAt[c.deck] = Infinity; }
      if (c.cmd === 'fx') fxUsed++;
    }
    assert.equal(P.role.filter((r) => r === 'primary').length, 1, 'one primary');
    const sounding = P.role.map((r, i) => (r !== 'idle' && G >= startAt[i] && P.value(i, 'fader', G + 0.5) > 0.02 && P.value(i, 'cut', G + 0.5) > 0 ? i : -1)).filter((i) => i >= 0);
    if (sounding.length === 3) three++;
    // both tracks in full range at a good level: the patterns of two compositions are heard together
    if (sounding.filter((i) => P.value(i, 'fader', G + 0.5) >= 0.7 && P.value(i, 'hp', G + 0.5) <= 0.55 && P.value(i, 'lp', G + 0.5) >= 0.95).length >= 2) together++;
    // the primary plays in full, apart from a bass break and a moment after a bass-first swap
    const p = P.primary;
    assert.ok(P.value(p, 'fader', G + 0.5) >= 0.75, `primary fader at bar ${G} (seed ${seed})`);
    assert.equal(P.value(p, 'fx', G + 0.5), 0, 'no effect on the primary');
    // two full basses never sound together
    const bass = sounding.filter((i) => P.value(i, 'lo', G + 0.5) > 0.3 && P.value(i, 'hp', G + 0.5) < 0.3 && P.value(i, 'fader', G + 0.5) > 0.3);
    const doubled = P.doubleBass.some((w) => G + 0.5 >= w.from && G + 0.5 < w.to);
    assert.ok(bass.length <= 1 || (doubled && bass.length === 2), `one bass at bar ${G} (seed ${seed}): decks ${bass}`);
    for (let i = 0; i < 3; i++) for (const key of KEYS) { const v = P.value(i, key, G + 0.25); assert.ok(v >= 0 && v <= 1 && Number.isFinite(v), `${key} in range`); assert.ok((P.lanes[i][key] || []).length < 60, 'lanes stay short'); }
  }
  assert.ok(cues >= 8, `a set of 3000 bars has transitions (seed ${seed}: ${cues})`);
  assert.ok(together / 3000 > 0.3, `two tracks are heard together for a good part of the set (seed ${seed}: ${together} bars)`);
  assert.ok(fxUsed < 2 * cues * 0.85, `effects are not on every deck of every transition (seed ${seed}: ${fxUsed} of ${2 * cues})`);
  assert.ok(primaries.size >= 2, 'the primary moves between decks');
  console.log(`sections of ${SEC}, seed ${seed}: ${cues} transitions, ${wants} tracks asked for, effects on ${fxUsed} of ${2 * cues} decks, two tracks together in full for ${together} bars, three decks in ${three} of 3000 bars`);
}
// a primary without bass: every transition that holds the new bass back becomes "bass first"
{
  const P = new Planner({ rand: mulberry(3), section: 16 });
  P.bassShare = () => 0;
  let commits = 0, swapped = 0;
  P.begin(0);
  for (let G = 0; G < 1500; G++) for (const c of P.tick(G, () => true)) if (c.cmd === 'cue') commits++;
  for (const t of P.active) if (/little bass/.test(t.name)) swapped++;
  const names = new Set(); const Q = new Planner({ rand: mulberry(3), section: 16 }); Q.bassShare = () => 0; Q.begin(0);
  for (let G = 0; G < 1500; G++) { Q.tick(G, () => true); for (const t of Q.active) names.add(t.name); }
  assert.ok(commits > 10, 'transitions happen');
  assert.ok([...names].every((n) => /bass first|break and drop/.test(n) === true || !/highs first|lowpass|mids under/.test(n)), `no transition holds the bass back without a bass in the old track: ${[...names].join('; ')}`);
  console.log(`primary without bass: ${commits} transitions, names ${[...names].slice(0, 3).join('; ')}`);
}
// a new track without bass while the old one has it: the swap waits, and still lands on a section line
{
  const P = new Planner({ rand: mulberry(5), section: 16 });
  P.bassShare = () => 1; P.newBassShare = () => 0;
  P.begin(0);
  let later = 0, swaps = 0;
  for (let G = 0; G < 1500; G++) for (const c of P.tick(G, () => true)) if (c.cmd === 'primary') { swaps++; assert.equal((G - P.origin) % 16, 0, 'a delayed swap is on a section line'); }
  for (const t of P.active) if (/changes later/.test(t.name)) later++;
  assert.ok(swaps > 5, 'swaps happen');
  console.log(`new track without bass: ${swaps} swaps, the last active ones delayed: ${later}`);
}
// cuts: one deck silent for the last one or two bars of a 16-bar phrase, only while two tracks play
{
  const P = new Planner({ rand: mulberry(11), section: 16 });
  P.begin(0);
  const cuts = [];
  let last = null;
  for (let G = 0; G < 3000; G++) {
    P.tick(G, () => true);
    if (P.lastCut && P.lastCut !== last) {              // checked as it is planned (lanes further back are forgotten)
      last = P.lastCut; cuts.push(last);
      assert.equal(P.value(last.deck, 'cut', last.a + 0.5), 0, 'the deck is silent during its cut');
      assert.equal(P.value(last.deck, 'cut', last.b + 0.5), 1, 'and back on the line');
      assert.equal(P.value(last.deck, 'cut', last.a - 0.5), 1, 'and playing before it');
    }
  }
  assert.ok(cuts.length >= 5, `cuts happen (${cuts.length})`);
  for (const c of cuts) {
    assert.equal((c.b - P.origin) % 16, 0, 'a cut ends on a 16-bar line');
    assert.ok(c.b - c.a === 1 || c.b - c.a === 2, 'a cut lasts one or two bars');
  }
  console.log(`cuts: ${cuts.length} in 3000 bars, ${cuts.filter((c) => c.b - c.a === 2).length} of two bars`);
}
// the new moves: three-band swaps, both basslines, EQ rides
{
  const P = new Planner({ rand: mulberry(21), section: 16 });
  P.begin(0);
  const names = new Set(); let rides = 0, lastRide = null;
  for (let G = 0; G < 3000; G++) {
    P.tick(G, () => true);
    for (const t of P.active) names.add(t.name.split(',')[0]);
    if (P.lastRide && P.lastRide !== lastRide) { lastRide = P.lastRide; rides++; }
  }
  assert.ok([...names].some((n) => n.startsWith('three-band swap')), 'three-band swaps happen');
  assert.ok(P.doubleBass.length > 0, 'both basslines play now and then');
  assert.ok(rides > 5, `EQ rides happen (${rides})`);
  for (const w of P.doubleBass) assert.ok(w.to - w.from <= 8, 'a double bass lasts at most 8 bars');
  console.log(`new moves: ${[...names].join(' | ')}; double basses ${P.doubleBass.length}, rides ${rides}`);
}
// which track comes next: the buffered live track; else a finished track of the brain from the shelf; else a
// transition piece, after which only the live track comes; a live track opens the set
{
  const O = new Transitions();
  assert.equal(O.decide({ liveSoon: false, shelf: true }), 'live', 'a live track opens the set');
  assert.equal(O.choose({ live: false, shelf: true, bridge: true }), null, 'the opening waits for the live track');
  assert.equal(O.choose({ live: true }), 'live'); O.cued('live');
  assert.equal(O.decide({ liveSoon: true, shelf: true }), 'live', 'buffered: live to live');
  assert.equal(O.choose({ live: true, shelf: true, bridge: true }), 'live'); O.cued('live');
  assert.equal(O.decide({ liveSoon: false, shelf: true }), 'shelf', 'late: a finished track of the brain');
  assert.equal(O.choose({ live: false, shelf: true, bridge: true }), 'shelf'); O.cued('shelf');
  assert.equal(O.decide({ liveSoon: false, shelf: false }), 'bridge', 'late and nothing on the shelf: a transition piece');
  assert.equal(O.choose({ live: false, shelf: false, bridge: true }), 'bridge'); O.cued('bridge');
  assert.equal(O.decide({ liveSoon: false, shelf: true }), 'live', 'after a piece only the live track');
  assert.equal(O.choose({ live: false, shelf: true, bridge: true }), null, 'the piece carries the set while the live track is late');
  assert.equal(O.choose({ live: true }), 'live');
  const E = new Transitions({ repertoire: false }); E.cued('live');
  assert.equal(E.decide({ liveSoon: false, shelf: false }), 'live');
  assert.equal(E.choose({ live: false, shelf: false, bridge: true }), null, 'no repertoire: no transition piece');
}
for (const SEC of [16, SECTION]) for (const [seed, delay, shelfOn] of [[1, 8, true], [2, 400, true], [3, 400, false], [4, 900, false], [5, 150, false]]) {
  // a live track is asked for once the live track before it is composed (one ahead), and is buffered `delay` bars
  // later; the shelf has finished tracks of the brain or none
  const P = new Planner({ rand: mulberry(seed), section: SEC }), O = new Transitions();
  let live = null, lastDone = 0, decidedFor = null, carried = 0;
  const kinds = [], deckKind = ['live', null, null];
  let primaryKind = 'live';
  P.begin(0); O.cued('live');
  for (let G = 0; G < 6000; G++) {
    if (!live && G >= lastDone) live = { askedAt: G, readyAt: G + delay };
    if (P.nextStart != null && G >= P.nextStart - 2 * SEC && decidedFor !== P.nextStart) {
      decidedFor = P.nextStart;
      O.decide({ liveSoon: live.readyAt <= P.nextStart, shelf: shelfOn });
    }
    const now = { live: G >= live.readyAt, shelf: shelfOn && O.plan === 'shelf', bridge: O.plan === 'bridge' };
    const waited = P.waited;
    for (const c of P.tick(G, () => O.choose(now) != null)) {
      if (c.cmd === 'cue') {
        const kind = O.choose(now);
        O.cued(kind); kinds.push(kind); deckKind[c.deck] = kind; decidedFor = null;
        if (kind === 'live') { lastDone = live.readyAt; live = null; }
      }
      if (c.cmd === 'primary') primaryKind = deckKind[c.deck];
    }
    if (P.waited > waited && primaryKind === 'bridge') carried++;
  }
  const seq = ['live', ...kinds];
  for (let i = 1; i < seq.length; i++) {
    if (seq[i - 1] === 'bridge') assert.equal(seq[i], 'live', `after a transition piece the live track (seed ${seed}: ${seq.slice(0, i + 1).join(' ')})`);
  }
  if (shelfOn) assert.ok(!seq.includes('bridge'), `with a shelf no transition piece (seed ${seed})`);
  if (delay <= 8) assert.ok(seq.every((k) => k === 'live'), `a fast brain: live to live (seed ${seed})`);
  if (!shelfOn && delay >= 400) assert.ok(seq.includes('bridge') && carried > 0, `a slow brain without a shelf: transition pieces carry the set (seed ${seed})`);
  console.log(`next tracks, sections of ${SEC}, live track buffered ${delay} bars after asking, ${shelfOn ? 'a shelf' : 'no shelf'}: ${seq.slice(0, 12).join(' ')}${seq.length > 12 ? ' ...' : ''} (${kinds.length} cues, piece carried ${carried} late lines)`);
}
// the page side bar by bar with stand-in decks and tracks: onBar never throws, a live track opens the set, the
// readout names the case of every next track (26 Nov: a readout error stopped the decks)
for (const shelf of [false, true]) {
  const deck = () => ({ cueTrack() {}, setRole() {}, gate() {}, autoSet() {}, autoLine() {}, autoShow() {}, endTrack() {}, autoRest() {}, setFx() {} });
  const texts = [];
  const desk = { chans: [deck(), deck(), deck()], ctx: () => ({ currentTime: 0 }), bpm: () => 126, userBpm: () => null, setBpm() {}, status() {}, paint() {},
    readout: (t) => texts.push(t), brainName: () => 'split_s1_s1lag' };
  const dj = new AutoDJ(desk, { section: 16, rand: mulberry(4) });
  let G = 0;
  const made = [];
  const track = (kind, want) => {
    const t = { kind, want, bars: [], done: false, pool: {}, name: kind === 'bridge' ? `piece ${made.length}` : null, stars: kind === 'bridge' ? 4 : undefined,
      state: '', id: `t${made.length}`, born: G, ready() { return this.kind !== 'live' || G - this.born > 260; } };
    if (kind !== 'live') { t.bars = Array(want).fill({}); t.done = true; }
    made.push(t);
    return t;
  };
  dj.request = async (bars, kind) => (kind === 'shelf' && !shelf ? null : track(kind, bars));
  dj.post = async () => ({});
  dj.on = true; dj.order = new Transitions({ repertoire: true }); dj.first = track('live', 160); dj.first.bars = Array(160).fill({}); dj.first.done = true; dj.armed = true;
  for (G = 0; G < 2500; G++) {
    for (const t of made) if (t.kind === 'live' && !t.done) { t.bars.length = Math.min(t.want, Math.floor((G - t.born) / 2)); t.done = t.bars.length >= t.want; }
    dj.onBar(G, G * 1.9);
    await Promise.resolve(); await Promise.resolve();
  }
  const kinds = dj.order.kinds;
  assert.equal(kinds[0], 'live', 'a live track opens the set');
  for (let i = 1; i < kinds.length; i++) if (kinds[i - 1] === 'bridge') assert.equal(kinds[i], 'live', `after a piece the live track: ${kinds.join(' ')}`);
  assert.ok(kinds.includes(shelf ? 'shelf' : 'bridge'), `a late brain brings ${shelf ? 'a shelf track' : 'a transition piece'}: ${kinds.join(' ')}`);
  assert.ok(texts.some((t) => (shelf ? /the brain is late \((\d+ of \d+ bars|not started)\): an earlier track by split_s1_s1lag/ : /the brain is late \((\d+ of \d+ bars|not started)\): no earlier track, a transition piece/).test(t)), 'the readout says why a stand-in comes');
  if (!shelf) assert.ok(texts.some((t) => /transition: repertoire piece "piece \d+" \(4 stars\) while split_s1_s1lag composes the next track · \d+ of \d+ bars/.test(t)), 'the readout says the piece and the composing');
  console.log(`page side, ${shelf ? 'a shelf' : 'no shelf'}: ${kinds.join(' ')}; e.g. "${texts.find((t) => /next: .* · /.test(t))}"`);
}
console.log('auto dj planner: all checks passed');
