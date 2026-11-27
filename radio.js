// Cadance Radio (the static site, tools/build_radio.py): the track store and the page's /api calls. A classic script,
// loaded before radio-app.js. The Auto DJ (autodj.js) asks for tracks through /api/autodj/*; here those calls are
// answered from the track store: tracks.json at the store's base URL ({"tracks": [{id, file, brain, bars, bpm, seconds,
// score?}]}), the MP3 of a track fetched and decoded when the Auto DJ takes it, and its note file ("score": the bars
// as the brain wrote them, with its recorded view) for the views. The brains take turns by their weight, and a brain's
// tracks come at random by their weight (the pick, below); the manifest is read again as the set goes, so new tracks
// and weights count without a reload. The setting "Live brain (server
// URL)" is checked here at load: in this version a live brain is reported, and the radio plays on.
(() => {
  const STORE = 'https://cadance-radio.s3.amazonaws.com';       // the public track store (tracks.json, tracks/<id>.mp3)
  const KEY = 'cadance.api';
  const OFFLINE = 'Cadance brain offline, radio mode active';
  const realFetch = window.fetch.bind(window);
  const q = new URLSearchParams(location.search);
  const read = () => { try { return (localStorage.getItem(KEY) || '').trim().replace(/\/+$/, ''); } catch { return ''; } };
  const api = read();
  // ?tracks=<base url or manifest url> for tests
  const given = (q.get('tracks') || '').trim();
  const manifestUrl = given ? (/\.json(\?|$)/.test(given) ? given : `${given.replace(/\/+$/, '')}/tracks.json`) : `${STORE}/tracks.json`;
  const base = manifestUrl.replace(/[^/]*$/, '');
  const state = { mode: 'radio', api, note: OFFLINE, detail: '', played: [], tracks: 0, store: base, error: '' };
  window.__radio = state;

  // a live brain: the URL answers /api/status within 4 s (in this version it is reported; the radio plays)
  state.ready = (async () => {
    if (!api) return 'radio';
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 4000);
    try {
      const r = await realFetch(`${api}/api/status`, { cache: 'no-store', signal: ctl.signal });
      if (r.ok) { state.live = true; state.note = `Live brain found at ${api}. Live playback arrives with the next update; the radio plays.`; }
      else state.detail = `${api} answered ${r.status}`;
    } catch (e) { state.detail = `${api} does not answer (${e.name === 'AbortError' ? 'no reply in 4 s' : 'not reachable'})`; }
    finally { clearTimeout(timer); }
    return 'radio';
  })();

  // ---------- the store ----------
  let list = null, last = null, serial = 0, listAt = 0, brainWeights = {};
  const handed = new Map();            // the Auto DJ's track id -> {entry, kept, audio, notes, error, at}
  const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } });
  async function manifest(force) {
    if (list && !force) return list;
    if (force && Date.now() - listAt < 20000 && list) return list;      // read again at most every 20 s
    listAt = Date.now();
    try {
      const j = await (await realFetch(manifestUrl, { cache: 'no-store' })).json();
      list = (j.tracks || []).filter((t) => t && t.id && t.file && t.bpm && t.bars);
      brainWeights = j.brain_weights && typeof j.brain_weights === 'object' ? j.brain_weights : {};
      state.error = '';
    } catch (e) { list = list || []; state.error = 'the track store does not answer'; }
    state.tracks = list.length;
    return list;
  }
  state.manifest = manifest;
  // ---------- the pick ----------
  // The brains take turns. A brain's share of the turns is its weight: the manifest's "brain_weights" maps a name
  // prefix to a weight, the longest matching prefix decides, none means 1, and 0 takes a brain out. The turns spread
  // evenly: before each pick every brain's credit grows by its weight, the brain with the most credit plays and gives
  // up the sum of the weights (the brains start in a shuffled order). Within a brain its tracks come in rounds by their
  // own "weight" (default 1; 0: never): a track of weight w >= 1 round(w) times a round, one below 1 once with chance
  // w; the copies of a track spread over the round (each falls at random in its own part of it), the brain's list is
  // through before a track repeats, and no track comes twice in a row. The manifest is read again before a pick (at
  // most every 20 s): a track that left it, or whose weight became 0, is skipped; new tracks count from their brain's
  // next round, new weights at once.
  const weightOf = (t) => { const w = t.weight == null ? 1 : +t.weight; return Number.isFinite(w) ? Math.max(0, w) : 1; };
  function brainWeight(b) {
    let best = -1, w = 1;
    for (const [prefix, v] of Object.entries(brainWeights)) if (b.startsWith(prefix) && prefix.length > best) { best = prefix.length; w = +v; }
    return Number.isFinite(w) ? Math.max(0, w) : 1;
  }
  const rounds = new Map(), credit = new Map(), lastOf = new Map();       // per brain: its round, its credit, its last track
  function round(entries, before) {
    let best = null;
    for (let tries = 0; tries < 24; tries++) {   // a few draws: the first in which no track meets itself
      const keyed = [];
      for (const t of entries) {
        const w = weightOf(t), k = w >= 1 ? Math.round(w) : Math.random() < w ? 1 : 0;
        for (let i = 0; i < k; i++) keyed.push([(i + Math.random()) / k, t.id]);
      }
      if (!keyed.length) {                        // only light tracks, none drawn: one of them, by weight
        const sum = entries.reduce((s, t) => s + weightOf(t), 0);
        let x = Math.random() * sum;
        for (const t of entries) { x -= weightOf(t); if (x <= 0) { keyed.push([0, t.id]); break; } }
      }
      keyed.sort((p, q) => p[0] - q[0]);
      const a = [];                               // in key order, each next copy the first that differs from the one before
      let prev = before;
      while (keyed.length) { const i = Math.max(0, keyed.findIndex((p) => p[1] !== prev)); prev = keyed.splice(i, 1)[0][1]; a.push(prev); }
      const meets = a.filter((id, i) => id === (i ? a[i - 1] : before)).length;
      if (!best || meets < best.meets) best = { a, meets };
      if (!meets) break;
    }
    return best.a;
  }
  function fromBrain(b, entries) {                // the brain's next track that is still listed and not the one just played
    const ids = new Set(entries.map((t) => t.id));
    let q = (rounds.get(b) || []).filter((id) => ids.has(id));
    if (!q.length || (q.length === 1 && q[0] === last)) q = q.concat(round(entries, q.length ? q[q.length - 1] : lastOf.get(b)));
    const i = q.findIndex((id) => id !== last);
    if (i < 0) { rounds.set(b, q); return null; }
    const id = q.splice(i, 1)[0];
    rounds.set(b, q); lastOf.set(b, id);
    return entries.find((t) => t.id === id);
  }
  async function nextEntry() {
    const all = await manifest(true), by = new Map();
    for (const t of all) {
      const b = String(t.brain || '');
      if (weightOf(t) > 0 && brainWeight(b) > 0) (by.get(b) || by.set(b, []).get(b)).push(t);
    }
    for (const b of [...credit.keys()]) if (!by.has(b)) credit.delete(b);
    let sum = 0;
    for (const b of by.keys()) { sum += brainWeight(b); credit.set(b, (credit.get(b) ?? Math.random() * 0.01) + brainWeight(b)); }
    for (const b of [...by.keys()].sort((x, y) => credit.get(y) - credit.get(x))) {
      const e = fromBrain(b, by.get(b));
      if (!e) continue;                           // only the track that just played: the next brain plays
      credit.set(b, credit.get(b) - sum);
      last = e.id; state.played.push(e.id);
      return e;
    }
    const only = all.find((t) => weightOf(t) > 0 && brainWeight(String(t.brain || '')) > 0);      // one track in all: it repeats
    if (only) { last = only.id; state.played.push(only.id); }
    return only || null;
  }
  state.entries = () => list || [];
  const url = (file) => (/^https?:/.test(file) ? file : base + file.replace(/^\/+/, ''));
  // when the Auto DJ takes a track its note file is read (about 0.5 MB gzipped); the MP3 is fetched and decoded (an
  // AudioBuffer at 44.1 kHz, 188 MB for 534 s) only when a deck cues it, and let go when the deck stops, so no more
  // tracks are fetched and decoded than decks play
  function release(id) { const h = handed.get(id); if (h) { h.audio = null; h.decoding = null; h.notes = null; } }
  state.release = release;
  state.track = (id) => handed.get(id) || null;
  async function load(h) {
    h.notes = h.entry.score ? await realFetch(url(h.entry.score)).then((r) => (r.ok ? r.json() : null)).catch(() => null) : null;
    h.ready = true;
  }
  state.decode = (id) => {
    const h = handed.get(id);
    if (!h) return Promise.reject(new Error('no such track'));
    if (h.audio) return Promise.resolve(h.audio);
    if (!h.decoding) {
      h.decoding = (async () => {
        let bytes;
        try {
          const r = await realFetch(url(h.entry.file));
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          bytes = await r.arrayBuffer();
        } catch (e) { throw new Error('the track could not be loaded'); }
        try {
          h.audio = await new OfflineAudioContext(2, 1, 44100).decodeAudioData(bytes);
          return h.audio;
        } catch (e) {
          throw new Error(/memory|allocation|range/i.test(String(e && (e.name + ' ' + e.message))) ? 'not enough memory to decode the track' : 'the track could not be decoded');
        }
      })();
    }
    return h.decoding;
  };
  function bars(h) {                                              // the bars the Auto DJ plans with: the notes, or stand-ins
    const n = h.entry.bars;
    if (h.notes && Array.isArray(h.notes.bars) && h.notes.bars.length) return h.notes.bars.slice(0, n);
    return Array.from({ length: n }, (_, i) => ({ bar: i + 1, tokens: [1, 1, 1, 1] }));
  }
  async function answer(u, init) {
    const path = u.pathname, method = ((init && init.method) || 'GET').toUpperCase();
    let body = {};
    try { body = init && init.body ? JSON.parse(init.body) : {}; } catch { body = {}; }
    if (path === '/api/autodj/models' || path === '/api/autodj/approved') {
      return json({ leader: null, models: [], repertoire: 0, approved: ['checkpoint:radio'], passed: null, leader_gate: null });
    }
    if (path === '/api/autodj/track' && method === 'POST') {
      if (body.model === 'repertoire') return json({ id: null });
      const entry = await nextEntry();
      if (!entry) return json({ id: null });
      const id = `r${++serial}`, h = { id, entry, kept: !!body.kept_only, audio: null, notes: null, error: '', at: performance.now() };
      handed.set(id, h);
      h.loading = load(h);
      return json({ id });
    }
    if (path === '/api/autodj/track') {
      const h = handed.get(u.searchParams.get('id'));
      if (!h) return json({ error: 'no such track' });
      if (h.error) return json({ error: h.error });
      const n = h.entry.bars;
      if (!h.ready) return json({ id: h.id, bars: [], want: n, total: n, done: false, error: null, state: 'reading the track' });
      const from = Math.max(0, +u.searchParams.get('from') || 0);
      return json({ id: h.id, bars: bars(h).slice(from), want: n, total: n, done: true, error: null, pool: 'radio', bpm: h.entry.bpm,
        checkpoint: h.entry.brain, kept: h.kept, state: 'idle', layout: (h.notes && h.notes.layout) || null });
    }
    if (path === '/api/autodj/pool') return json({});                        // a file deck needs no pool
    if (path === '/api/autodj/keep') {
      const keep = new Set(body.ids || []);
      for (const [id, h] of handed) if (!keep.has(id) && !(state.onDeck || new Set()).has(id)) { release(id); handed.delete(id); }
      return json({ ok: true });
    }
    if (path === '/api/autodj/stop') { for (const id of [...handed.keys()]) if (!(state.onDeck || new Set()).has(id)) { release(id); handed.delete(id); } return json({ ok: true }); }
    if (path === '/api/brain3d') return realFetch('brain3d/brain.b3d');      // the one brain the 3D modes draw
    if (path === '/api/status') return json({ ok: true, loading: null, ready: true, active: null, radio: true });
    return json({ error: 'radio mode' }, 404);
  }
  window.fetch = async (input, init) => {
    const u = new URL(typeof input === 'string' ? input : input.url, location.href);
    if (u.origin !== location.origin || !u.pathname.startsWith('/api/')) return realFetch(input, init);
    return answer(u, init);
  };
})();
