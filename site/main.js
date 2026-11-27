// Cadance: the page. Audio runs in an AudioWorklet (engine.mjs); the Cadence brain lives in the local server and
// answers one bar per request. Mood pad = the wanted mood of the next bar. Stars (1-5, keys 1-5) rate the brain that
// plays. (Heart moments are off until they are designed properly; the server still takes /api/feedback.)
import { Rack } from './rack.js';
import { BrainViz, MODE_GROUPS } from './brainviz.js';
import { loadKit } from './kit.js';
import { Brains, trained } from './brains.js';
import { Stats } from './stats.js';

const $ = (id) => document.getElementById(id);
const rack = new Rack($('rack'));
const viz = new BrainViz($('brain'));
const mini = new BrainViz($('mini'), { compact: true });      // the brain, always visible beside the devices
const AHEAD = 2;                                  // bars requested ahead of the one playing
const store = {
  get(k, d) { try { const v = localStorage.getItem('cadance.' + k); return v == null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem('cadance.' + k, v); } catch { /* private window */ } },
};

async function requestJSON(path, options = {}, timeout = 12000) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const response = await fetch(path, { ...options, signal: ctl.signal });
    const result = await response.json();
    if (!response.ok || result.error) {
      const error = Error(result.error || `Request failed (${response.status})`);
      error.status = response.status; error.body = result;           // 409: the server loads another brain (switchTo)
      throw error;
    }
    return result;
  } catch (error) {
    if (ctl.signal.aborted) throw Error('The server did not finish the request in time');
    throw error;
  } finally { clearTimeout(timer); }
}

function postJSON(path, body, timeout) {
  return requestJSON(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) }, timeout);
}

let ctx = null, node = null, playing = false, pool = null, bpm = 126, pending = 0;
let transportVersion = 0, barRequest = null;
let mood = [0.45, 0.4];
const queue = [];
let now = { tokens: [0, 0, 0, 0], mods: [[0, 0], [0, 0], [0, 0], [0, 0]], bar: 0, mode: null, arousal: null };
let prev = now;
const history = [];
let step = -1, stepAt = 0, stepMs = 60000 / bpm / 4, layoutFor = undefined;
// the waveform below the devices (wave.js): what played, bars marked liked or disliked for training. Loaded on its own,
// so the page plays on when a server without it (one started before wave.js existed) cannot serve the file.
let wave = null;
import('./wave.js').then(({ Wave }) => {
  wave = new Wave($('wave'), { tempo: () => bpm, balance: () => balance, onReplay: () => { if (playing) pause(); } });
  if (node) wave.attach(ctx, node);
}).catch((error) => console.warn('the waveform did not load:', error.message));

// ---------- views ----------
function setView(v) {
  document.body.classList.toggle('v-brain', v === 'brain');
  document.querySelectorAll('.tab[data-v]').forEach((t) => t.classList.toggle('on', t.dataset.v === v));
  store.set('view', v);
  if (v === 'brain') viz.resize(); else mini.resize();
}
function setMode(m) { viz.setMode(m); mini.setMode(viz.mode); $('mode').value = viz.modeName; store.set('mode', viz.modeName); }
$('mini').onclick = () => setView('brain');
document.querySelectorAll('.tab[data-v]').forEach((t) => (t.onclick = () => setView(t.dataset.v)));
(() => {                                         // the visualization dropdown: demo pieces, then pieces that show the brain
  const sel = $('mode');
  for (const [group, names] of Object.entries(MODE_GROUPS)) {
    const og = document.createElement('optgroup'); og.label = group;
    for (const n of names) og.appendChild(new Option(n, n));
    sel.appendChild(og);
  }
  sel.onchange = () => { setMode(sel.value); sel.blur(); };
})();
$('miniMax').onclick = (e) => { e.stopPropagation(); setView('brain'); };
$('backDev').onclick = () => setView('devices');
setView(store.get('view', 'devices'));
setMode(store.get('mode', 'cadance'));

// ---------- Cadence brains ----------
const CHECKPOINT_SCOPE = '__checkpoint_scope__';
let showAllCheckpoints = store.get('all_checkpoints', '0') === '1';
function checkpointLabel(checkpoint) {
  let duration = '';
  if (Number.isFinite(checkpoint.minutes)) {
    const minutes = Math.round(checkpoint.minutes);
    const hours = Math.floor(minutes / 60);
    duration = hours ? `${hours}h ${minutes % 60}m` : `${minutes}m`;
  }
  return [checkpoint.label || checkpoint.name, duration, trained(checkpoint, {epoch: true})].filter(Boolean).join(' · ');
}

function latestPerCandidate(checkpoints) {
  const latest = new Map();
  for (const checkpoint of checkpoints) {
    const family = checkpoint.candidate || checkpoint.name;
    const variant = checkpoint.playback_variant ? `${family}/${checkpoint.playback_variant}` : family;
    const previous = latest.get(variant);
    const created = Date.parse(checkpoint.created) || 0;
    const previousCreated = Date.parse(previous?.created) || 0;
    if (!previous || created > previousCreated ||
        (created === previousCreated && (checkpoint.epoch ?? -1) > (previous.epoch ?? -1))) {
      latest.set(variant, checkpoint);
    }
  }
  return new Set(latest.values());
}

async function loadCheckpoints(force = false) {
  const j = await requestJSON('/api/checkpoints');
  const sel = $('checkpoint');
  if (!force && document.activeElement === sel) return;
  const selected = pendingLoad?.name || loadingName || j.active || (j.demo ? `demo:${j.demo}` : 'demo');
  const latest = latestPerCandidate(j.checkpoints);
  sel.innerHTML = '';
  for (const checkpoint of j.checkpoints) {
    if (showAllCheckpoints || checkpoint.featured || latest.has(checkpoint) || checkpoint.name === selected) {
      sel.add(new Option(checkpointLabel(checkpoint), checkpoint.name));
    }
  }
  // a brain fetched for a listen plays without a place in the menu
  if (j.active && !j.checkpoints.some((checkpoint) => checkpoint.name === j.active)) sel.add(new Option(`${j.active} · from Stats`, j.active));
  sel.add(new Option('no brain (instrument check)', 'demo'));
  sel.add(new Option('no brain (SID chip check)', 'demo:sid'));
  if (latest.size < j.checkpoints.length) {
    sel.add(new Option(showAllCheckpoints ? 'Show latest per variant' : 'Show all checkpoints…', CHECKPOINT_SCOPE));
  }
  sel.value = selected;
  brains.setActive(j.active, j.checkpoints);
}
setInterval(() => loadCheckpoints().catch(() => {}), 30000);
// load a brain (name) or, with no name, take up the one the server switched to; then the pool, layout and menu follow
let switching = false, loadRequested = false, loadError = '', loadingName = null;
let checkpointReady = false, activeCheckpoint = null, polling = false, checkpointRevision = 0;
function readyCheckpoint(name) {
  checkpointReady = true;
  activeCheckpoint = name ?? null;
  light('ok');
  $('transport-status').dataset.state = playing ? '' : 'ready';
  if (!playing) status(loadError || `Ready to play · ${activeCheckpoint || 'Instrument check'}`);    // one message at a time
  paintTransport();
  settleJoined(activeCheckpoint);
  if (pendingLoad) setTimeout(tryPending, 0);       // the brain asked for meanwhile loads now
}
function invalidateBars() {
  transportVersion++;
  barRequest?.abort();
  queue.length = 0;
}
// While a brain loads, a click on that brain (or on its run) joins the load, and a click on another brain waits its
// turn: the newest such click loads right after the current load (the line names it as next). The server answers a
// request during a load with 409 (it loads one brain at a time); that is never an error here. A server started
// before the progress route behaves the same way: the page asks again every 3 s until the server takes the request.
let pendingLoad = null, pendingTimer = null, loadingCandidate = null;
let joined = [];                                   // clicks waiting for the load in progress: { name, candidate, resolve }
const shortName = (text) => String(text || '').replace(/\bslate_/g, '');
function sameLoad(name, candidate, busy) {
  busy = String(busy || '');
  if (candidate) return busy === candidate || busy.startsWith(`${candidate}_`) || busy.startsWith(`${candidate} `);
  return !!name && busy === name;
}
function joinLoad(name, candidate) {
  return new Promise((resolve) => joined.push({ name, candidate, resolve }));
}
function settleJoined(active, failed = false) {
  const waiting = joined; joined = [];
  for (const w of waiting) w.resolve(!failed && sameLoad(w.name, w.candidate, active));
}
function queueLoad(name, candidate) {
  if (pendingLoad) pendingLoad.resolve(false);       // the newest click wins
  return new Promise((resolve) => {
    pendingLoad = { name: name || null, candidate: candidate || null, resolve };
    if (!pendingTimer) pendingTimer = setInterval(tryPending, 3000);
    if (lightState === 'loading') paintLoad();
    $('checkpoint').value = pendingLoad.name || $('checkpoint').value;
  });
}
async function tryPending() {
  const p = pendingLoad;
  if (!p) { clearInterval(pendingTimer); pendingTimer = null; return; }
  if (loadRequested) return;
  const st = await serverStatus().catch(() => null);          // asked only once the server is free
  if (!st || st.loading || pendingLoad !== p || loadRequested) return;
  await switchTo(p.name, p.candidate, p);
}
// /api/status; a server with the progress route names the player's load record in it (`progress`)
async function serverStatus() {
  const j = await requestJSON('/api/status', {}, 6000);
  loading.supported = 'progress' in j;
  return j;
}
// `candidate` (from Stats) names a training run in place of a checkpoint: the server makes that run's newest
// checkpoint loadable, fetching it when the player does not have it, and switches to it. `queued` is the waiting
// click this call serves (tryPending). Resolves true once the brain asked for plays.
async function switchTo(name, candidate, queued = null) {
  if (loadRequested) {                             // this page's own load is in progress
    if (sameLoad(name, candidate, loadingCandidate || loadingName)) return joinLoad(name, candidate);
    return queued ? false : queueLoad(name, candidate);
  }
  if (!queued && pendingLoad) { pendingLoad.resolve(false); pendingLoad = null; }   // a direct click replaces the waiting one
  loadRequested = true;
  checkpointRevision++;
  loadError = '';
  loadingName = name || null;
  loadingCandidate = candidate || null;
  paintTransport();
  let busy = null;
  try {
    let selected = activeCheckpoint;
    if (name || candidate) {
      switching = !playing;
      // a load the server runs already (another tab, its start, an earlier click) is joined or waited for, unasked
      busy = (await serverStatus().catch(() => ({}))).loading || null;
      let j;
      if (!busy) {
        // the waiting click is served now: it resolves with the brain that plays (readyCheckpoint) after this load
        if (queued && pendingLoad === queued) { pendingLoad = null; joined.push({ name, candidate, resolve: queued.resolve }); }
        light('loading', candidate ? `Fetching the newest checkpoint of ${candidate}…` : `Loading ${name}…`);
        // a split brain takes 30 to 160 s on the Mac and a SID brain longer: the bar shows the load meanwhile
        try {
          j = await postJSON('/api/load', candidate ? { candidate } : { name, mix: playing }, candidate ? 900000 : 600000);
        } catch (error) {
          if (error.status !== 409) throw error;
          busy = error.body?.loading || (await serverStatus().catch(() => ({}))).loading || 'a brain';
        }
      }
      if (busy) {
        switching = false;
        checkpointReady = false;
        light('loading', `Loading ${busy}…`);
        if (queued) {                                  // a race: the server took another load first
          const k = joined.findIndex((w) => w.resolve === queued.resolve);
          if (k >= 0) { joined.splice(k, 1); if (pendingLoad) queued.resolve(false); else pendingLoad = queued; }
        }
        if (sameLoad(name, candidate, busy)) {         // the server loads this very brain: the click joins that load
          if (!queued) return joinLoad(name, candidate);
          if (pendingLoad === queued) { pendingLoad = null; joinLoad(name, candidate).then(queued.resolve); }
          return false;
        }
        return queued ? false : queueLoad(name, candidate);    // another brain: this one waits its turn
      }
      if (j.how === 'mix') {
        settleJoined(name);
        light('ok', `Mixing into ${name}…`);
        $('transport-status').dataset.state = '';
        return true;
      }
      selected = j.active;
    }
    switching = true;
    invalidateBars();
    node?.port.postMessage({ type: 'flush' });
    await fetchPool();
    if (node) {
      node.port.postMessage({ type: 'pool', pool });
      node.port.postMessage({ type: 'tempo', bpm });
    }
    readyCheckpoint(selected);
  } catch (error) {
    checkpointReady = false;
    loadError = `Could not load ${name || candidate || 'the checkpoint'}: ${error.message}`;
    if (queued && pendingLoad === queued) { pendingLoad = null; queued.resolve(false); }
    light('error', loadError);
    settleJoined(null, true);
    return false;
  } finally {
    switching = loadRequested = false;
    loadingName = loadingCandidate = null;
    if (checkpointReady && !playing) $('checkpoint').value = activeCheckpoint || 'demo';
    paintTransport();
    loadCheckpoints().catch(() => {});
    if (checkpointReady) refreshLayout().catch(() => {});
    else pollStatus();
  }
  if (playing) for (let i = 0; i < AHEAD; i++) await requestBar();
  return true;
}
$('checkpoint').addEventListener('change', (e) => {
  if (e.target.value === CHECKPOINT_SCOPE) {
    showAllCheckpoints = !showAllCheckpoints;
    store.set('all_checkpoints', showAllCheckpoints ? '1' : '0');
    e.target.value = loadingName || activeCheckpoint || 'demo';
    loadCheckpoints(true).catch(() => {});
  } else {
    switchTo(e.target.value);
  }
});
// Stats' ▶ beside a brain: its newest checkpoint into the player, the devices in view, and the brain composing.
// Loudness balance: the engine keeps every synth voice at a set share of the kit's level (site/play-reading.json
// names the shares). A mixing aid until the brains hear their own mix; the switch compares with and without.
let balance = { on: true, ratio: [0.75, 0.22, 0.31], rangeDb: 30, kitDb: -14, duck: 0.3, outDb: -17 };
// The kick boost (balance.drums.kickBoost: quiet kicks raised, every kick punchier) has its own switch, kept in this
// browser; it works while the level balance is on.
const kickOn = () => balance.drums?.kickBoost?.on !== false && !!balance.drums?.kickBoost;
const setKick = (on) => { balance = { ...balance, drums: { ...(balance.drums || {}), kickBoost: { ...(balance.drums?.kickBoost || {}), on } } }; };
// The space (balance.space: a base reverb and delay send per synth voice, the sounds being fitted dry) has its own switch
// too, kept in this browser; it works while the level balance is on.
const spaceOn = () => balance.space?.on !== false && !!balance.space;
const setSpace = (on) => { balance = { ...balance, space: { ...(balance.space || {}), on } }; };
// The bass (balance.bass: the bass voice a measured gain louder against the kit) likewise
const bassOn = () => balance.bass?.on !== false && !!balance.bass;
const setBass = (on) => { balance = { ...balance, bass: { ...(balance.bass || {}), on } }; };
const sendBalance = () => {
  node?.port.postMessage({ type: 'balance', ...balance });
  $('levelBtn').classList.toggle('on', balance.on);
  $('levelBtn').setAttribute('aria-pressed', String(balance.on));
  $('kickBtn').classList.toggle('on', kickOn());
  $('kickBtn').setAttribute('aria-pressed', String(kickOn()));
  $('spaceBtn')?.classList.toggle('on', spaceOn());
  $('spaceBtn')?.setAttribute('aria-pressed', String(spaceOn()));
  $('bassBtn')?.classList.toggle('on', bassOn());
  $('bassBtn')?.setAttribute('aria-pressed', String(bassOn()));
};
fetch('/play-reading.json', { cache: 'no-store' }).then((r) => (r.ok ? r.json() : null)).then((j) => {
  if (j?.balance) balance = { ...balance, ...j.balance };
  const kept = store.get('kickBoost', null);
  if (kept !== null && balance.drums?.kickBoost) setKick(kept === '1');
  const keptSpace = store.get('space', null);
  if (keptSpace !== null && balance.space) setSpace(keptSpace === '1');
  const keptBass = store.get('bass', null);
  if (keptBass !== null && balance.bass) setBass(keptBass === '1');
  sendBalance();
}).catch(() => sendBalance());
$('levelBtn').onclick = () => { balance.on = !balance.on; sendBalance(); };
$('kickBtn').onclick = () => { setKick(!kickOn()); store.set('kickBoost', kickOn() ? '1' : '0'); sendBalance(); };
if ($('spaceBtn')) $('spaceBtn').onclick = () => { setSpace(!spaceOn()); store.set('space', spaceOn() ? '1' : '0'); sendBalance(); };
if ($('bassBtn')) $('bassBtn').onclick = () => { setBass(!bassOn()); store.set('bass', bassOn() ? '1' : '0'); sendBalance(); };
const stats = new Stats({ panel: $('stats'), button: $('statsBtn'),
  onPlay: async (candidate) => {
    if (!playing) unlockAudio().catch(() => {});
    setView('devices');
    if ((await switchTo(null, candidate)) && !playing) await start();
  } });
const requestedStats = new URLSearchParams(location.search).get('stats');
if (requestedStats !== null) stats.open(requestedStats || undefined);
// the editor's ▶ plays that brain (and starts the sound if it was off); ■ on the brain that plays stops the sound
const brains = new Brains({ panel: $('brains'), stars: $('stars'), button: $('brainsBtn'), status: (m) => status(m),
  transport: () => ({ playing, stopped }),
  onStats: (name) => stats.open(name),
  onPlay: async (n) => {
    if (!playing) unlockAudio().catch(() => {});
    if ((await switchTo(n)) && n && !playing) await start();
  },
  onPause: () => pause(), onResume: () => start(), onStop: () => stop(),
  onOpen: async (n) => {
    if (!playing) unlockAudio().catch(() => {});
    if ((n === activeCheckpoint || await switchTo(n)) && !playing) await start();
  } });
// the tempo: the pool's (the corpus median) unless the listener set one (kept in this browser); double-click resets
let userBpm = null;
try { const v = +localStorage.getItem('cadance_bpm'); if (v >= 60 && v <= 200) userBpm = v; } catch {}
function applyBpm(v) {
  bpm = Math.max(60, Math.min(200, Math.round(v))); stepMs = 60000 / bpm / 4;
  if (node) node.port.postMessage({ type: 'tempo', bpm });
  const f = $('bpm'); if (f && document.activeElement !== f) f.value = bpm;
  $('bpmBox').classList.toggle('mine', userBpm != null);
  $('bpmBox').title = userBpm != null ? `your tempo (the brain's corpus: ${(pool && pool.bpm) || '–'} bpm); double-click for that` : 'tempo: drag up/down, scroll, or type';
}
function setUserBpm(v) {
  userBpm = v == null ? null : Math.max(60, Math.min(200, Math.round(v)));
  try { userBpm == null ? localStorage.removeItem('cadance_bpm') : localStorage.setItem('cadance_bpm', String(userBpm)); } catch {}
  applyBpm(userBpm ?? ((pool && pool.bpm) || 126));
  $('bpm').value = bpm;                            // set by hand: the field shows it even while focused
}
async function fetchPool(revision = checkpointRevision) {
  const nextPool = await requestJSON('/api/pool');
  if (!nextPool.layers?.length) throw Error('The checkpoint instrument pool could not be loaded');
  if (revision !== checkpointRevision) return;
  pool = nextPool;
  applyBpm(userBpm ?? (pool.bpm || 126));
  rack.setPool(pool);
  rack.setBar(now.tokens, prev.tokens, now.mods, prev.mods);
}
async function refreshLayout() {
  const j = await requestJSON('/api/layout');
  layoutFor = j.checkpoint || null;
  viz.setLayout(j.layout); mini.setLayout(j.layout);
}

// ---------- audio ----------
const latency = () => (ctx ? (ctx.outputLatency || 0) + (ctx.baseLatency || 0) : 0);
const later = (t, fn) => { const ms = Math.max(0, (t + latency() - (ctx ? ctx.currentTime : 0)) * 1000); ms > 4 ? setTimeout(fn, ms) : fn(); };

// ▶ plays (from a pause: the same composition goes on; after ■ or on a fresh page: a new one), ⏸ pauses, ■ stops and
// resets the brain, so the next ▶ starts an entirely new composition
let stopped = true, starting = false;
async function unlockAudio() {
  if (!ctx) ctx = new AudioContext({ sampleRate: 44100, latencyHint: 'playback' });
  return ctx.resume();
}

async function boundedAudio(operation, message) {
  let timer;
  try {
    return await Promise.race([operation, new Promise((_, reject) => {
      timer = setTimeout(() => reject(Error(message)), 12000);
    })]);
  } finally { clearTimeout(timer); }
}

async function start() {
  wave?.stopReplay();                              // a mark's replay gives way to the brain
  if (stopped && !playing && !starting) wave?.clear();   // after ■ the next ▶ is a new composition: a fresh waveform
  if (starting || playing || switching || loadRequested || !checkpointReady) return;
  starting = true;
  paintTransport();
  const version = transportVersion;
  try {
    await boundedAudio(unlockAudio(), 'Audio is blocked. Press Play again to enable it.');
    if (version === transportVersion) await startPlayback(version);
  } catch (error) {
    if (version === transportVersion) {
      stopped = true;
      playing = false;
      invalidateBars();
      node?.port.postMessage({ type: 'silence' });
      light('error', `Could not start playback. ${error.message}`);
    }
  } finally {
    starting = false;
    paintTransport();
  }
}
async function startPlayback(version) {
  if (!node) {
    await boundedAudio(ctx.audioWorklet.addModule('worklet.js'), 'The audio engine did not finish loading.');
    const nextNode = new AudioWorkletNode(ctx, 'minimal', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
    rack.setKit(await boundedAudio(loadKit(ctx, nextNode), 'The drum samples did not finish loading.'));   // the rack names what plays: recordings or models
    node = nextNode;
    node.connect(ctx.destination);
    viz.listen(ctx, node); mini.listen(ctx, node);
    node.port.onmessage = (e) => {
      const m = e.data;
      const version = transportVersion;
      if (m.type === 'step') later(m.t, () => { if (!stopped && version === transportVersion) onStep(m.step); });
      else if (m.type === 'bar') {
        wave?.bar(m);                              // every bar the engine starts, with its place in the audio
        later(ctx.currentTime, () => { if (!stopped && version === transportVersion) onBar(); });
      } else if (m.type === 'wave') wave?.columns(m);
      else if (m.type === 'error') status('engine: ' + m.message.split('\n')[0]);
    };
    wave?.attach(ctx, node);
    node.port.postMessage({ type: 'pool', pool });
    node.port.postMessage({ type: 'tempo', bpm });
    sendBalance();
  }
  if (version !== transportVersion) return;
  if (stopped) {                                   // a new composition: the brain primed again from another passage
    const j = await postJSON('/api/new', {});
    if (version !== transportVersion) return;
    if (j.loading) {
      checkpointReady = false;
      light('loading', `Loading ${j.loading}…`);
      pollStatus();
      return;
    }
    queue.length = 0; history.length = 0;
    node.port.postMessage({ type: 'silence' });
    stopped = false; playing = true;
    for (let i = 0; i < AHEAD; i++) await requestBar();
    if (!playing || version !== transportVersion) return;
  }
  if (version !== transportVersion || !ctx || ctx.state !== 'running') return;
  if (version !== transportVersion) return;
  playing = true;
  if (queue.length < AHEAD && pending === 0) requestBar();
  paintTransport();
}
async function pause() {
  if (ctx) await ctx.suspend();
  playing = false;
  paintTransport();
}
async function stop() {
  playing = false; stopped = true; step = -1;
  invalidateBars();
  if (node) node.port.postMessage({ type: 'silence' });
  if (ctx) setTimeout(() => { if (!playing) ctx.suspend(); }, 60);   // let the silence reach the output first
  paintTransport();
  status('stopped: ▶ starts a new composition');
}
function paintTransport() {
  if (typeof brains !== 'undefined') brains.transportChanged();
  $('play').disabled = !playing && (!checkpointReady || loadRequested || starting);
  $('play').textContent = playing ? '⏸' : '▶';
  $('play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
  $('play').title = playing ? 'pause (space): the composition goes on where it stopped' : (stopped ? 'play (space): a new composition' : 'play (space): go on');
  $('stop').classList.toggle('off', stopped && !playing);
}
async function requestBar() {
  if (switching || !playing || pending || queue.length >= AHEAD) return;
  const version = transportVersion;
  const ctl = new AbortController();
  barRequest = ctl;
  pending++;
  const asked = mood.slice();
  try {
    const timer = setTimeout(() => ctl.abort(), 8000);
    let j;
    try {
      // brain3d: a 3D mode shows, so the server adds every drawn neuron's activation to the bar's view
      j = await (await fetch('/api/next', { method: 'POST', body: JSON.stringify({ mood: asked, reward: 0, brain3d: viz.is3d }), signal: ctl.signal })).json();
    } catch (e) {
      if (version !== transportVersion) return;
      light('down', 'no answer from the server: is it running?'); setTimeout(() => { if (playing && queue.length < AHEAD && pending === 0) requestBar(); }, 2000); return;
    } finally { clearTimeout(timer); }
    if (version !== transportVersion) return;
    if (j.loading) { light('loading', `Loading ${j.loading}…`); setTimeout(() => { if (playing && queue.length < AHEAD && pending === 0) requestBar(); }, 700); return; }
    if (j.error) { status(j.error); light('error', j.error); return; }
    light('ok');
    queue.push(j);
    node.port.postMessage({ type: 'queue', tokens: j.tokens, mods: j.mods, patterns: j.patterns || null, patches: j.patches || null,
      seq: wave?.queued(j, asked) });
  } finally {
    barRequest = null;
    pending--;
    if (version !== transportVersion && playing && !switching) requestBar();
  }
}
// the light at the top left: green = the brain answers, amber = a brain is loading, red = no server
let lightState = 'ok';
function light(state, msg) {
  lightState = state;
  const o = $('orb');
  const transport = $('transport-status');
  o.classList.toggle('down', state === 'down' || state === 'error');
  o.classList.toggle('loading', state === 'loading');
  o.title = state === 'ok' ? 'connected' : (msg || state);
  if (state === 'loading') transport.dataset.state = 'loading';
  if (state === 'down' || state === 'error') transport.dataset.state = 'error';
  if (state === 'ok' && !loadRequested && transport.dataset.state === 'loading') transport.dataset.state = '';
  if (state === 'loading') { watchLoad(msg); return; }
  if (loading.timer) endLoad(state === 'ok');
  if (msg) status(msg);
}

// ---------- a brain loading: a bar with the percent and the stage in words (the server's /api/load/progress) ----------
// While the light is amber the page asks every 500 ms; the message that started the wait stays the bar's text. A server
// started before the route answers 404: then a stripe runs along the bar and the message stands alone.
const loading = { base: '', rec: null, timer: null, supported: null, done: null };
// the time a load still takes in words (the server sends them as left_text; the same wording here)
function leftWords(s) {
  if (!(s >= 1)) return '';
  if (s < 55) return `about ${Math.max(5, Math.round(s / 5) * 5)} s left`;
  const m = Math.floor((s + 5) / 60), rest = Math.round((s - 60 * m) / 10) * 10;
  return rest > 0 ? `about ${m} min ${rest} s left` : `about ${m} min left`;
}
function loadText() {
  const asked = pendingLoad && (pendingLoad.name || pendingLoad.candidate);
  const next = asked ? ` · next: ${{ demo: 'no brain (instrument check)', 'demo:sid': 'no brain (SID chip check)' }[asked] || asked}` : '';
  const r = loading.rec;
  if (!r) return shortName(`${loading.base}${next}`);
  const left = r.finished || r.percent >= 99 ? '' : (r.left_text ?? leftWords(r.left));
  return shortName(`${loading.base.replace(/…$/, '')} · ${r.text}${left ? ` · ${left}` : ''}${next}`);
}
function paintLoad() {
  const r = loading.rec, known = !!r && Number.isFinite(r.percent), o = $('orb'), bar = $('load-bar');
  const pct = known ? Math.floor(r.percent) : null, fill = $('load-fill');
  if (known && bar.classList.contains('indet')) {    // the running stripe gives way to the percent without sliding back
    fill.style.transition = 'none'; bar.classList.remove('indet'); fill.style.width = `${r.percent}%`;
    void fill.offsetWidth; fill.style.transition = '';
  }
  bar.classList.toggle('indet', !known);
  fill.style.width = known ? `${r.percent}%` : '';
  $('load-pct').textContent = known ? `${pct}%` : '';
  if (known) bar.setAttribute('aria-valuenow', String(pct)); else bar.removeAttribute('aria-valuenow');
  bar.setAttribute('aria-valuetext', loadText());
  o.classList.toggle('pct', known);
  o.style.setProperty('--load', known ? r.percent : 0);
  o.title = `${loadText()}${known ? ` · ${pct}%` : ''}`;
  status(loadText());
}
async function pollLoad() {
  if (lightState !== 'loading' || !loading.supported) return;      // known from /api/status (serverStatus)
  try {
    const response = await fetch('/api/load/progress', { cache: 'no-store' });
    if (response.status === 404) { loading.supported = false; return; }
    const r = (await response.json()).player;
    // this load's record: the one in progress, or the one of the brain waited for that went through in the last seconds
    loading.rec = r && !r.error && (!r.finished || (r.ago < 4 && loading.base.includes(r.name))) ? r : null;
  } catch { return; }                              // the next tick asks again
  if (lightState === 'loading') paintLoad();
}
function watchLoad(msg) {
  loading.base = msg || loading.base;
  clearTimeout(loading.done); loading.done = null;
  $('load-bar').classList.remove('done'); $('load-pct').classList.remove('done');
  if (!loading.timer) { loading.timer = setInterval(pollLoad, 500); pollLoad(); }
  paintLoad();
}
// the load is over: a brain that came in shows 100% for a moment, then the bar goes
function endLoad(ok) {
  clearInterval(loading.timer); loading.timer = null; loading.rec = null;
  const o = $('orb'); o.classList.remove('pct'); o.style.removeProperty('--load');
  if (!ok) return;
  const bar = $('load-bar'), pct = $('load-pct');
  bar.classList.remove('indet'); $('load-fill').style.width = '100%'; pct.textContent = '100%';
  bar.setAttribute('aria-valuenow', '100');
  bar.classList.add('done'); pct.classList.add('done');
  loading.done = setTimeout(() => { bar.classList.remove('done'); pct.classList.remove('done'); loading.done = null; }, 900);
}
async function pollStatus() {
  if (polling || loadRequested) return;
  polling = true;
  const revision = checkpointRevision;
  try {
    const j = await serverStatus();
    if (revision !== checkpointRevision) return;
    if (j.loading) {
      checkpointReady = false;
      light('loading', `Loading ${j.loading}…`);
      paintTransport();
    } else if (!checkpointReady || (j.active ?? null) !== activeCheckpoint) {
      await fetchPool(revision);
      if (revision !== checkpointRevision) return;
      node?.port.postMessage({ type: 'pool', pool });
      node?.port.postMessage({ type: 'tempo', bpm });
      readyCheckpoint(j.active);
      loadCheckpoints().catch(() => {});
      refreshLayout().catch(() => {});
    } else if (lightState === 'down') {
      readyCheckpoint(j.active);
    }
  } catch (error) {
    if (revision === checkpointRevision && !loadError) light('down', `Could not check playback. ${error.message}`);
  } finally { polling = false; }
}
setInterval(pollStatus, 2000);
function onBar() {
  const next = queue.shift();
  prev = now;
  if (next) now = next;
  history.push({ bar: now.bar, tokens: now.tokens.slice(), mods: now.mods });
  if (history.length > 96) history.shift();
  rack.setBar(now.tokens, prev.tokens, now.mods, prev.mods, now.patterns, now.patches, prev.patches);
  viz.setBar(now, history); mini.setBar(now, history);
  if (now.mode === 'brain' && now.checkpoint !== layoutFor) refreshLayout().catch(() => {});
  if (now.mode !== 'brain' && layoutFor !== null) { layoutFor = null; viz.setLayout(null); mini.setLayout(null); }
  if (playing && queue.length < AHEAD && pending === 0) requestBar();
  orb(); status();
}
function onStep(s) {
  step = s; stepAt = performance.now();
  rack.setStep(s);
  const beat = pool && pool.layers[0].tokens[now.tokens[0]];
  const kick = beat && beat.pattern && beat.pattern.hits && beat.pattern.hits.kick ? beat.pattern.hits.kick[s] > 0 : s % 4 === 0;
  viz.step(s, kick); mini.step(s, kick);
}


// ---------- controls ----------
$('play').onclick = () => (playing ? pause() : start());
$('stop').onclick = () => stop();
{
  const f = $('bpm');
  f.addEventListener('change', () => setUserBpm(+f.value || bpm));
  f.addEventListener('keydown', (e) => { e.stopPropagation(); if (e.key === 'Enter') f.blur(); });
  f.addEventListener('wheel', (e) => { e.preventDefault(); setUserBpm(bpm + (e.deltaY < 0 ? 1 : -1)); }, { passive: false });
  $('bpmBox').addEventListener('dblclick', (e) => { e.preventDefault(); setUserBpm(null); f.blur(); });
  let drag = null;
  f.addEventListener('pointerdown', (e) => { drag = { y: e.clientY, v: bpm, moved: false }; f.setPointerCapture(e.pointerId); });
  f.addEventListener('pointermove', (e) => { if (!drag) return; const d = Math.round((drag.y - e.clientY) / 4); if (d) { drag.moved = true; setUserBpm(drag.v + d); } });
  f.addEventListener('pointerup', (e) => { if (drag && drag.moved) { e.preventDefault(); f.blur(); } drag = null; });
}
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey) return;
  if (e.code === 'Space') { e.preventDefault(); playing ? pause() : start(); }
  if (e.key === 's' || e.key === 'S') { if (!/INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || '')) stop(); }
  else if (e.key >= '1' && e.key <= '5') brains.rateActive(+e.key);
  else if (e.key === 'v') setView(document.body.classList.contains('v-brain') ? 'devices' : 'brain');
  else if (e.key === 'm') setMode(viz.mode + 1);
});
const pad = $('pad'), dot = $('dot');
function placeDot() { dot.style.left = `${mood[1] * 100}%`; dot.style.top = `${(1 - mood[0]) * 100}%`; }
function padMove(e) {
  const r = pad.getBoundingClientRect();
  mood = [1 - Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1), Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1)];
  placeDot();
}
pad.addEventListener('pointerdown', (e) => { pad.setPointerCapture(e.pointerId); padMove(e); });
pad.addEventListener('pointermove', (e) => { if (e.buttons) padMove(e); });
placeDot();

function orb() {
  if (lightState !== 'ok') return;
  const a = now.arousal, heat = a ? Math.min(1, (a.level || 0) * 1.6 + (a.mode === 'routine' ? 0 : 0.25)) : 0;
  const c = `rgb(${[95, 211, 198].map((v, i) => Math.round(v + ([242, 166, 90][i] - v) * heat)).join(',')})`;
  $('orb').style.background = c; $('orb').style.boxShadow = `0 0 8px ${c}`; $('dot').style.boxShadow = `0 0 10px ${c}`;
}
function status(msg) {
  const line = $('status');
  const transport = $('transport-status');
  if (msg) {
    if (line.textContent !== msg) line.textContent = msg;
    return;
  }
  if (transport.dataset.state === 'loading' || (!playing && transport.dataset.state === 'ready')) return;
  if (!playing && !now.bar) { line.textContent = ''; return; }
  transport.dataset.state = '';
  const a = now.arousal;
  const who = now.mix && now.mix.from ? `mixing ${now.mix.from} → ${now.mix.to} (${now.mix.bar}/${now.mix.of})` : (now.mode === 'brain' ? now.checkpoint : 'instrument check, no brain');
  line.textContent = `bar ${now.bar} · ${who}${a ? ` · ${a.mode === 'routine' ? 'calm' : 'aroused'}` : ''}`;
}

// ---------- the frame clock ----------
function frame() {
  requestAnimationFrame(frame);
  if (document.hidden) return;
  const phase = playing && step >= 0 ? Math.min(0.999, (step + Math.min(1, (performance.now() - stepAt) / stepMs)) / 16) : 0;
  const f0 = performance.now();
  if (document.body.classList.contains('v-brain')) viz.frame(phase, stepMs);
  else { rack.update(phase); mini.frame(phase, stepMs); }
  const ft = (window.__frameMs ||= []); ft.push(performance.now() - f0); if (ft.length > 300) ft.shift();   // work per frame (ms)
}
requestAnimationFrame(frame);
paintTransport();
light('loading', 'Checking checkpoint…');
loadCheckpoints().catch(() => {});
pollStatus();
