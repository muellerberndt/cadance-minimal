// Cadance Radio (the static site, tools/build_radio.py): the track store and the page's /api calls. A classic script,
// loaded before radio-app.js. The Auto DJ (autodj.js) asks for tracks through /api/autodj/*; here those calls are
// answered from the track store: tracks.json at the store's base URL ({"tracks": [{id, file, brain, bars, bpm, seconds,
// score?}]}), the MP3 of a track fetched and decoded when the Auto DJ takes it, and its note file ("score": the bars
// as the brain wrote them, with its recorded view) for the views. Picks are random, with no repeat until the list is
// through; then the manifest is read again, so new tracks join without a reload. The setting "Live brain (server
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
  let list = null, order = [], last = null, serial = 0, listAt = 0;
  const handed = new Map();            // the Auto DJ's track id -> {entry, kept, audio, notes, error, at}
  const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } });
  async function manifest(force) {
    if (list && !force) return list;
    if (force && Date.now() - listAt < 20000 && list) return list;      // read again at most every 20 s
    listAt = Date.now();
    try {
      const j = await (await realFetch(manifestUrl, { cache: 'no-store' })).json();
      list = (j.tracks || []).filter((t) => t && t.id && t.file && t.bpm && t.bars);
      state.error = '';
    } catch (e) { list = list || []; state.error = 'the track store does not answer'; }
    state.tracks = list.length;
    return list;
  }
  state.manifest = manifest;
  function shuffled(ids) {
    const a = ids.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    if (a.length > 1 && a[0] === last) a.push(a.shift());          // not the track that just played, at the turn
    return a;
  }
  async function nextEntry() {                                    // at random, no repeat until every track has played
    let all = await manifest();
    if (!order.length) { all = await manifest(true); order = shuffled(all.map((t) => t.id)); }
    if (!order.length) return null;
    last = order.shift();
    state.played.push(last);
    return all.find((t) => t.id === last) || null;
  }
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
