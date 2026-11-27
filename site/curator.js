// Cadance curator: listen to the tracks the track factory composed, mark sections, keep them as the Auto DJ's
// repertoire. A site of its own (curator.py); the instrument is the player's (engine.mjs in worklet.js).
import { loadKit } from './kit.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const LAYERS = ['beat', 'bass', 'lead', 'air'], SECTION = 16, AHEAD = 3;
const get = async (path) => (await fetch(path)).json();
const post = async (path, body) => (await fetch(path, { method: 'POST', body: JSON.stringify(body) })).json();
const status = (text) => { $('status').textContent = text || ''; };

// ---------- what is shown ----------
let tracks = [], items = [], track = null, trackId = null;      // the open track: {bars, pool, bpm, seed, checkpoint}
let source = null;                                             // what plays: {kind: 'track' | 'item', id, bars, loop: [a, b] | null}
let selected = new Set();                                      // marked sections of the open track (index of 16 bars)
let anchor = null, bpm = 126, userBpm = null;
let cursor = 0;                                                // the playhead in the open track: where play starts
let suggest = null;                                            // the open track's passage scores and suggestions (site/suggest.py)
const KIND = { like: ['like the material', '--s-like'], synth: ['synth line', '--s-synth'], break: ['breakbeat', '--s-break'] };
const GROUP = {};                       // display names of the score's material groups, by group id; none: the id is shown
const pools = {};

// ---------- audio: one instrument, bars queued a few ahead ----------
let ctx = null, node = null, playing = false, pos = 0, mirror = [], now = -1, poolName = null;
async function ensureAudio() {
  if (ctx) return;
  ctx = new AudioContext({ sampleRate: 44100, latencyHint: 'playback' });
  await ctx.audioWorklet.addModule('worklet.js');
  node = new AudioWorkletNode(ctx, 'minimal', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
  const lim = ctx.createDynamicsCompressor();
  lim.threshold.value = -1.5; lim.knee.value = 0; lim.ratio.value = 20; lim.attack.value = 0.002; lim.release.value = 0.12;
  node.connect(lim); lim.connect(ctx.destination);
  await loadKit(ctx, node);
  try { const j = await get('/play-reading.json'); if (j && j.balance) node.port.postMessage({ type: 'balance', ...j.balance, on: j.balance.on !== false }); } catch { /* as written */ }
  node.port.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'bar') {
      while (mirror.length > m.queued) now = mirror.shift();
      top(); paintNow();
    } else if (m.type === 'error') status('engine: ' + m.message.split('\n')[0]);
  };
}
async function pool(name) {
  if (!pools[name]) pools[name] = await get(`/api/pool?name=${encodeURIComponent(name)}`);
  return pools[name];
}
function nextIndex() {                                           // the bar after `pos` for what plays
  const n = source.bars.length, [a, b] = source.loop || [0, n];
  if (pos < a || pos >= b) pos = a;
  const k = pos; pos = pos + 1 >= b ? (source.loop ? a : b) : pos + 1;
  return k;
}
function top() {
  if (!playing || !source) return;
  while (mirror.length < AHEAD) {
    if (!source.loop && pos >= source.bars.length) { if (!mirror.length) stop(); return; }
    const k = nextIndex(), j = source.bars[k];
    mirror.push(k);
    node.port.postMessage({ type: 'queue', tokens: j.tokens, mods: j.mods, patterns: j.patterns || null, patches: j.patches || null });
  }
}
async function playFrom(src, at) {
  await ensureAudio();
  if (ctx.state !== 'running') await ctx.resume();
  source = src; pos = at;
  const p = await pool(src.pool);
  if (poolName !== src.pool) { node.port.postMessage({ type: 'pool', pool: p }); poolName = src.pool; }
  if (userBpm == null && src.bpm) setBpm(src.bpm, false);
  node.port.postMessage({ type: 'tempo', bpm });
  node.port.postMessage({ type: 'flush' });                      // the next bar line is the new place
  mirror = []; playing = true; top(); paintTransport(); paintItems();
}
function stop() {
  playing = false; mirror = []; now = -1;
  if (node) node.port.postMessage({ type: 'silence' });
  paintTransport(); paintNow(); paintItems();
}
function setBpm(v, mine = true) {
  bpm = clamp(Math.round(+v || 126), 60, 200);
  if (mine) userBpm = bpm;
  if (node) node.port.postMessage({ type: 'tempo', bpm });
  if (document.activeElement !== $('bpm')) $('bpm').value = bpm;
}

// ---------- the track list ----------
function when(iso) { return iso ? iso.replace('T', ' ').slice(5, 16) : ''; }
let armedTrack = null, armedAll = false;
async function deleteTracks(body, label) {
  try {
    const j = await post('/api/tracks/remove', body);
    const n = (j.deleted || []).length;
    tracks = tracks.filter((t) => !(j.deleted || []).includes(t.id));
    if (trackId && (j.deleted || []).includes(trackId)) { trackId = null; }
    paintTracks(); status(`${label}: ${n} track${n === 1 ? '' : 's'} deleted`);
  } catch (e) { status(`could not delete: ${e.message}`); }
}
function paintTracks() {
  const box = $('tracks'); box.innerHTML = '';
  const unused = tracks.filter((t) => !t.marks).length;
  const all = $('delUnused');
  if (all) { all.hidden = !unused; all.textContent = armedAll ? `click again: delete ${unused}` : `delete unused (${unused})`; all.classList.toggle('armed', armedAll); }
  if (!tracks.length) { box.appendChild(el('div', 'empty', 'no composed tracks yet: compose one below, or run the Auto DJ in the mixer')); return; }
  let group = null;
  for (const t of tracks) {
    if (t.checkpoint !== group) { group = t.checkpoint; box.appendChild(el('div', 'group', `composed by ${group}`)); }
    const line = el('div', 'trow');
    const row = el('button', 'row' + (t.id === trackId ? ' on' : ''));
    const text = el('div'); text.append(el('div', 't', `${when(t.created)} · ${t.bars} bars`), el('div', 's', t.seed ? `opening context: ${t.seed}` : ''));
    const right = el('span');
    if (t.suggested) { const c = el('span', 'sgn', `${t.suggested} ✦`); c.title = `${t.suggested} suggested passage${t.suggested === 1 ? '' : 's'} (open the track to see them above its timeline)`; right.appendChild(c); }
    right.appendChild(el('span', 'n' + (t.marks ? ' has' : ''), t.marks ? `${t.marks} kept` : ''));
    row.append(text, right);
    row.title = `composed by the brain ${t.checkpoint}\n${t.id}${t.seed ? `\nits opening context was eight bars of: ${t.seed}` : ''}`;
    row.onclick = () => open(t.id);
    line.appendChild(row);
    if (!t.marks) {                                   // only a track no repertoire piece came from can go
      const x = el('button', 'tdel' + (armedTrack === t.id ? ' armed' : ''), '✕');
      x.title = armedTrack === t.id ? 'click again to delete this track' : 'delete this track (nothing of it is in the repertoire; click twice)';
      x.onclick = (e) => {
        e.stopPropagation();
        if (armedTrack !== t.id) { armedTrack = t.id; paintTracks(); setTimeout(() => { if (armedTrack === t.id) { armedTrack = null; paintTracks(); } }, 4000); return; }
        armedTrack = null; deleteTracks({ id: t.id }, 'deleted');
      };
      line.appendChild(x);
    } else line.appendChild(el('span', 'tdel-none'));
    box.appendChild(line);
  }
}
if ($('delUnused')) $('delUnused').onclick = () => {
  if (!armedAll) { armedAll = true; paintTracks(); setTimeout(() => { if (armedAll) { armedAll = false; paintTracks(); } }, 5000); return; }
  armedAll = false; deleteTracks({ all_unused: true }, 'every unused track');
};
async function loadTracks() {
  try { tracks = (await get('/api/tracks')).tracks || []; $('orb').className = 'orb on'; } catch { $('orb').className = 'orb'; status('the curator server does not answer'); return; }
  paintTracks(); countsLater();
}
// the server scores the tracks in the background: their suggestion counts come in as they are done
let countTimer = null, countTries = 0;
function countsLater() {
  if (countTimer || !tracks.some((t) => t.suggested == null) || countTries > 20) return;
  countTimer = setTimeout(async () => {
    countTimer = null; countTries += 1;
    let j; try { j = await get('/api/suggested'); } catch { return; }
    let changed = false;
    for (const t of tracks) { const n = (j.counts || {})[t.id]; if (n != null && t.suggested !== n) { t.suggested = n; changed = true; } }
    if (changed) paintTracks();
    countsLater();
  }, 2500);
}

// ---------- the open track: a timeline of 16-bar sections ----------
function stepsOf(bar, layer) {
  const tok = bar.tokens[layer], on = new Array(16).fill(false);
  if (!tok) return on;
  const pat = (bar.patterns && bar.patterns[layer]) || (pools[track.pool] && pools[track.pool].layers[layer] && pools[track.pool].layers[layer].tokens[tok] && pools[track.pool].layers[layer].tokens[tok].pattern) || null;
  if (pat && pat.hits) { for (const a of Object.values(pat.hits)) if (Array.isArray(a)) a.forEach((v, i) => { if (i < 16 && +v > 0) on[i] = true; }); }
  else if (pat && pat.notes) pat.notes.forEach((v, i) => { if (i < 16 && v != null) on[i] = true; });
  return on;
}
function keptRanges() { return items.filter((i) => i.source === trackId).map((i) => [i.from, i.to]); }

// ---------- suggestions: three scores per passage, and runs of passages where one stays high ----------
const num = (v) => (v == null ? '-' : Number.isInteger(v) ? String(v) : (+v).toFixed(2));
function scoresTitle(s) {
  const p = s.parts || {}, out = [];
  const like = p.like || {};
  out.push(`like the material ${num(s.like)}: nearest ${GROUP[like.group] || like.group} (distance ${num(like.distance)} in its spreads)` +
    ((like.farthest || []).length ? `; farthest: ${like.farthest.map(([k, v, typ]) => `${k} ${num(v)} (typical ${num(typ)})`).join(', ')}` : ''));
  const syn = p.synth || {}, line = (name) => { const x = syn[name]; return x ? `${name} ${x.pitches} pitches over ${x.range} semitones, ${num(x['notes a bar'])} notes a bar, ${num(x['bars a few steps changed'])} of the bars a few steps changed, ${num(x['bars rewritten'])} rewritten${x['longest on one note'] >= 8 ? `, ${x['longest on one note']} bars on one note` : ''}` : `no ${name}`; };
  out.push(`synth line ${num(s.synth)}: ${line('lead')}; ${line('bass')}`);
  out.push(`breakbeat ${num(s.break)}: ${Object.entries(p.break || {}).map(([k, v]) => `${k} ${num(v)}`).join(', ')}`);
  return out.join('\n');
}
function suggestionAt(q) { return suggest && suggest.suggestions ? suggest.suggestions.find((s) => s.sections.includes(q)) : null; }
function paintSuggest() {
  const box = $('suggest'), list = $('sugList');
  box.hidden = !track; list.innerHTML = '';
  if (!track) return;
  const say = (text) => { const n = el('span', '', text); n.id = 'sugNone'; list.appendChild(n); };
  if (!suggest || suggest.id !== trackId) { say('scoring the passages…'); return; }
  if (suggest.error) { say(suggest.error); return; }
  if (!suggest.suggestions.length) { say(`none: no run of 32 bars or more after bar ${suggest.skip} stays high (the lines under each passage show its scores)`); return; }
  const cur = [...selected].sort((a, b) => a - b).join(',');
  for (const s of suggest.suggestions) {
    const b = el('button', 'sug' + (s.sections.join(',') === cur ? ' on' : ''));
    const tag = el('b', 'tag', KIND[s.kind][0]); tag.style.setProperty('--c', `var(${KIND[s.kind][1]})`);
    b.append(tag, el('span', '', `bars ${s.from + 1} to ${s.to}`));
    for (const k of s.kinds.slice(1)) { const a = el('span', 'also'), i = el('i', '', `· ${KIND[k][0]}`); i.style.setProperty('--c', `var(${KIND[k][1]})`); a.appendChild(i); b.appendChild(a); }
    b.title = `${s.label}\nmean scores: like the material ${num(s.scores.like)}, synth line ${num(s.scores.synth)}, breakbeat ${num(s.scores.break)}\n` +
      `click: marks these ${s.sections.length} passages (${s.to - s.from} bars) and puts the playhead on bar ${s.from + 1}; space plays them, "add to the repertoire" takes them`;
    b.onclick = () => takeSuggestion(s);
    list.appendChild(b);
  }
}
function takeSuggestion(s) {                                     // marks the passages as shift-click does; plays nothing
  selected = new Set(s.sections); anchor = s.sections[0]; cursor = s.from;
  paintTimeline();
  const first = document.querySelector(`.sec[data-q="${s.sections[0]}"]`);
  if (first) first.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
async function loadSuggest(id) {
  let j;
  try { j = await get(`/api/track/suggest?id=${encodeURIComponent(id)}`); } catch { j = { error: 'the suggestions did not load' }; }
  if (id !== trackId) return;
  suggest = j.error ? { id, error: j.error } : { ...j, id };
  const t = tracks.find((x) => x.id === id);
  if (t && !j.error && t.suggested !== j.suggestions.length) { t.suggested = j.suggestions.length; paintTracks(); }
  paintTimeline();
}

function paintTimeline() {
  const box = $('timeline'); box.innerHTML = '';
  paintSuggest();
  if (!track) return;
  const n = track.bars.length, sections = Math.ceil(n / SECTION), kept = keptRanges();
  const scored = suggest && suggest.id === trackId && suggest.sections ? suggest : null;
  for (let s = 0; s < sections; s += 2) {
    const line = el('div', 'line');
    line.appendChild(el('div', 'at', `${Math.floor((s * SECTION * 240) / bpm / 60)}:${String(Math.floor(((s * SECTION * 240) / bpm) % 60)).padStart(2, '0')}`));
    for (const q of [s, s + 1]) {
      if (q >= sections) { line.appendChild(el('div')); continue; }
      const a = q * SECTION, b = Math.min(n, a + SECTION);
      const isKept = kept.some(([x, y]) => a < y && b > x);
      const sec = el('div', 'sec' + (selected.has(q) ? ' sel' : '') + (isKept ? ' kept' : '')); sec.dataset.q = q;
      const where = el('span', '', `bars ${a + 1} to ${b}`), sug = scored && suggestionAt(q);
      if (sug) { const star = el('span', 'sg', '✦'); star.style.setProperty('--c', `var(${KIND[sug.kind][1]})`); star.title = `suggested: ${sug.label}`; where.appendChild(star); }
      const head = el('div', 'sh'); head.append(where, el('span', 'k', isKept ? 'in the repertoire' : selected.has(q) ? 'marked' : ''));
      head.title = 'click to mark these 16 bars, shift-click to mark every passage up to here';
      head.onclick = (e) => { e.stopPropagation(); mark(q, e.shiftKey); };
      sec.appendChild(head);
      const row = el('div', 'barsrow');
      for (let k = a; k < b; k++) {
        const bar = el('div', 'bar'); bar.dataset.k = k; bar.title = `bar ${k + 1}: click puts the playhead here (space plays)`;
        for (let l = 0; l < 4; l++) {
          const lane = el('i', 'steps'); lane.style.setProperty('--c', `var(--${LAYERS[l]})`);
          const on = stepsOf(track.bars[k], l);
          for (let i = 0; i < 16; i++) lane.appendChild(el('b', on[i] ? 'on' : ''));
          bar.appendChild(lane);
        }
        // a click moves the playhead and plays nothing; with shift it marks up to this section
        bar.onclick = (e) => { e.stopPropagation(); if (e.shiftKey) { mark(q, true); return; } cursor = k; if (playing && source && source.kind === 'track') playFrom(trackSource(null), k); paintNow(); paintSelection(); };
        row.appendChild(bar);
      }
      sec.appendChild(row);
      const sc = scored && scored.sections[q];
      if (sc) {                                                  // the passage's three scores, as lines of their length
        const meter = el('div', 'sc'); meter.title = scoresTitle(sc);
        for (const k of Object.keys(KIND)) {
          const line = el('i', sc[k] >= scored.kinds[k].mark ? 'hi' : ''); line.style.setProperty('--c', `var(${KIND[k][1]})`); line.style.width = `${Math.round(100 * clamp(sc[k], 0, 1))}%`;
          meter.appendChild(line);
        }
        sec.appendChild(meter);
      }
      sec.onclick = (e) => { if (e.shiftKey) mark(q, true); };
      line.appendChild(sec);
    }
    box.appendChild(line);
  }
  paintNow(); paintSelection();
}
function trackSource(loop) { return { kind: 'track', id: trackId, bars: track.bars, pool: track.pool, bpm: track.bpm, loop }; }
function paintNow() {
  for (const b of document.querySelectorAll('.bar.now, .bar.cur')) b.classList.remove('now', 'cur');
  const c = document.querySelector(`.bar[data-k="${cursor}"]`);
  if (c) c.classList.add('cur');
  if (!playing || !source || source.kind !== 'track' || source.id !== trackId || now < 0) return;
  const b = document.querySelector(`.bar[data-k="${now}"]`);
  if (b) b.classList.add('now');
}
function span() {                                                // the marked sections as one range of bars
  if (!selected.size || !track) return null;
  const qs = [...selected].sort((a, b) => a - b);
  return [qs[0] * SECTION, Math.min(track.bars.length, (qs[qs.length - 1] + 1) * SECTION)];
}
function mark(q, extend) {
  if (extend && anchor != null) { const [a, b] = [Math.min(anchor, q), Math.max(anchor, q)]; selected = new Set(); for (let i = a; i <= b; i++) selected.add(i); }
  else if (selected.has(q) && selected.size === 1) { selected.clear(); anchor = null; }
  else { selected = new Set([q]); anchor = q; }
  paintTimeline();
}
function paintSelection() {
  const r = span(), has = !!r;
  for (const id of ['name', 'loopB', 'addB', 'clearB']) $(id).hidden = !has;
  const secs = has ? Math.round(((r[1] - r[0]) * 240) / bpm) : 0;
  $('selSay').textContent = has ? `marked: bars ${r[0] + 1} to ${r[1]} (${(r[1] - r[0]) / SECTION} passages, ${r[1] - r[0]} bars, ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')} at ${bpm} bpm)`
    : `playhead at bar ${cursor + 1}: space plays and stops. Click a passage’s header to mark it, shift-click to mark every passage up to there.`;
}
async function open(id) {
  status('loading the track…');
  let t;
  try { t = await get(`/api/track?id=${encodeURIComponent(id)}`); } catch { status('could not load the track'); return; }
  if (t.error) { status(t.error); return; }
  await pool(t.pool);
  track = t; trackId = id; selected = new Set(); anchor = null; cursor = 0;
  if (!suggest || suggest.id !== id) suggest = null;
  if (userBpm == null && t.bpm && !playing) setBpm(t.bpm, false);
  $('title').textContent = `composed by ${t.checkpoint} · ${t.bars.length} bars`;
  $('sub').textContent = `${when(t.created)}${t.seed ? ` · the brain's opening context: eight bars of ${t.seed}` : ''}`;
  status(''); paintTracks(); paintTimeline();
  loadSuggest(id);
}

// ---------- the repertoire ----------
function paintItems() {
  const box = $('items'); box.innerHTML = '';
  $('repN').textContent = items.length ? `· ${items.length} sections, ${items.reduce((s, i) => s + i.length, 0)} bars` : '';
  if (!items.length) { box.appendChild(el('div', 'empty', 'empty: mark sections of a track and add them')); return; }
  for (const i of items) {
    const row = el('div', 'row');
    const on = playing && source && source.kind === 'item' && source.id === i.id;
    const pb = el('button', 'ib' + (on ? ' on' : ''), on ? '■' : '▶'); pb.title = on ? 'stop' : 'play this section in a loop';
    pb.onclick = async () => {
      if (on) { stop(); return; }
      const full = await get(`/api/repertoire/item?id=${i.id}`);
      playFrom({ kind: 'item', id: i.id, bars: full.bars, pool: full.pool, bpm: full.bpm, loop: [0, full.bars.length] }, 0);
    };
    const text = el('div'); text.style.minWidth = 0;
    const name = el('div', 't', i.name); name.title = 'double-click to rename';
    name.ondblclick = async () => { const v = prompt('name of the section', i.name); if (v && v.trim()) { await post('/api/repertoire/rename', { id: i.id, name: v.trim() }); loadItems(); } };
    const stars = el('div', 'stars'); stars.title = 'rate this piece: the Auto DJ plays pieces with more stars more often (click a star again to clear)';
    for (let n = 1; n <= 5; n++) {
      const st = el('button', 'star' + (n <= (i.stars || 0) ? ' on' : ''), '★'); st.setAttribute('aria-label', `${n} star${n > 1 ? 's' : ''}`);
      st.onclick = async () => { await post('/api/repertoire/rate', { id: i.id, stars: i.stars === n ? 0 : n }); loadItems(); };
      stars.appendChild(st);
    }
    const sub = el('div', 's', `${i.length} bars · ${i.bpm ? Math.round(i.bpm) + ' bpm · ' : ''}${when(i.created)} · `);
    if (i.mp3) { const a = el('a', 'mp3', 'mp3'); a.href = `/repertoire/${i.mp3}`; a.target = '_blank'; a.title = `the piece as a 192 kbit/s MP3 (${i.seconds ? Math.round(i.seconds) + ' s' : ''})`; sub.appendChild(a); }
    else sub.appendChild(el('span', i.mp3_error ? 'bad' : '', i.mp3_error ? 'mp3 failed' : 'rendering the mp3…'));
    text.append(name, stars, sub);
    const sidB = el('button', 'ib', 'S'); sidB.title = 'make a SID chip version of this piece (the same notes, every synth sound in its SID fit)'; sidB.hidden = !!i.sid;
    sidB.onclick = async () => { const j = await post('/api/repertoire/sid', { id: i.id }); if (j.error) status(j.error); else status(`added: ${j.name}`); loadItems(); };
    const x = el('button', 'ib x', '✕'); x.title = 'remove from the repertoire';
    x.onclick = async () => { if (!confirm(`Remove "${i.name}" from the repertoire?`)) return; if (on) stop(); const j = await post('/api/repertoire/remove', { id: i.id }); if (j.error) status(j.error); loadItems(); };
    const side = el('div'); side.style.display = 'flex'; side.style.gap = '4px'; side.append(sidB, x);
    row.append(pb, text, side); row.title = `${i.name}\ncomposed by ${i.checkpoint || 'a brain'}: bars ${i.from + 1} to ${i.to} of ${i.source}${i.mp3_error ? `\nmp3: ${i.mp3_error}` : ''}`;
    box.appendChild(row);
  }
}
async function loadItems() {
  let j;
  try { j = await get('/api/repertoire'); } catch { return; }
  items = j.items || [];
  const s = j.sync || {}, sync = $('sync');
  sync.textContent = s.state === 'synced' ? `bucket: saved ${s.at || ''}` : s.state === 'offline' ? 'bucket: off' : `bucket: ${s.state}`;
  if (items.some((i) => !i.mp3 && !i.mp3_error)) setTimeout(loadItems, 2500);          // an MP3 is being rendered
  sync.className = s.state === 'synced' ? 'ok' : s.state === 'not synced' ? 'bad' : '';
  sync.title = s.error ? `${j.bucket}\n${s.error}` : j.bucket || '';
  paintItems(); if (track) paintTimeline(); paintTracks();
}
$('addB').onclick = async () => {
  const r = span(); if (!r) return;
  $('addB').disabled = true;
  const j = await post('/api/repertoire/add', { track: trackId, from: r[0], to: r[1], name: $('name').value, bpm });
  $('addB').disabled = false;
  if (j.error) { status(j.error); return; }
  status(`added: ${j.name}. Its MP3 (192 kbit/s, ${bpm} bpm) is being rendered and saved to the bucket.`); $('name').value = ''; selected.clear(); anchor = null;
  for (const t of tracks) if (t.id === trackId) t.marks += 1;
  await loadItems();
};
$('loopB').onclick = () => { const r = span(); if (r) playFrom(trackSource(r), r[0]); };
$('clearB').onclick = () => { selected.clear(); anchor = null; paintTimeline(); };

// ---------- composing more ----------
async function loadComposer() {
  let j;
  try { j = await get('/api/composer'); } catch { return; }
  const say = $('composeSay'), sel = $('model');
  if (j.error) { say.textContent = j.error; $('composeB').disabled = true; return; }
  $('composeB').disabled = false;
  if (document.activeElement !== sel) {
    const want = sel.value || 'leader';
    sel.innerHTML = '';
    sel.add(new Option(j.models.leader ? `ranking leader: ${j.models.leader}` : 'ranking leader', 'leader'));
    for (const m of j.models.models || []) if (m.id !== 'repertoire') sel.add(new Option(m.label, m.id));
    sel.value = [...sel.options].some((o) => o.value === want) ? want : 'leader';
  }
  const busy = (j.status.tracks || []).filter((t) => !t.done);
  say.textContent = busy.length ? `the brain is ${j.status.state}: ${busy.map((t) => `${t.total}/${t.want}`).join(', ')} bars` : '';
  if (busy.length || composeWatch) { composeWatch = busy.length > 0; loadTracks(); }
}
let composeWatch = false;
$('composeB').onclick = async () => {
  const j = await post('/api/compose', { model: $('model').value, bars: 288 });
  if (j.error) { status(j.error); return; }
  status('asked the brain for a new track: it appears in the list when it is finished (about six minutes)'); composeWatch = true; loadComposer();
};

// ---------- transport ----------
function paintTransport() {
  $('play').textContent = playing ? '■' : '▶'; $('play').title = playing ? 'stop (space)' : 'play from the playhead (space)';
  $('stop').classList.toggle('off', !playing);
}
$('play').onclick = () => {                                       // start at the playhead, or stop
  if (playing) { stop(); return; }
  if (!track) { status('pick a track first'); return; }
  const r = span();
  playFrom(trackSource(r && cursor >= r[0] && cursor < r[1] ? r : null), cursor);      // inside the marked passages: they go round
};
$('stop').onclick = () => stop();
{
  const f = $('bpm');
  f.addEventListener('change', () => { setBpm(f.value); if (track) paintTimeline(); });
  f.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') f.blur(); });
  f.addEventListener('wheel', (e) => { e.preventDefault(); setBpm(bpm + (e.deltaY < 0 ? 1 : -1)); f.value = bpm; }, { passive: false });
  setBpm(126, false);
}
addEventListener('keydown', (e) => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
});

await loadItems(); await loadTracks(); loadComposer();
if (tracks.length) open(tracks[0].id);
setInterval(loadComposer, 5000); setInterval(loadItems, 15000); setInterval(loadTracks, 60000);
