// The brain editor: a modal over the page with the curated brains (pinned ones and remarkable new checkpoints).
// Per brain: your order (drag the handle, or the arrows), play, a name and a note you can edit, 1-5 stars, the QA
// verdict (hover for the failed lines), delete (two clicks). A double-click on a row closes the editor and plays that brain. One-click sorts (stars, QA, newest) set
// the order once; the order is kept on the server and is the order of the menu.
// Plus a small 5-star widget for the brain that is playing.
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const post = async (path, body) => (await fetch(path, { method: 'POST', body: JSON.stringify(body) })).json();
const QA_RANK = { PASS: 2, 'HOLDS A BEAT': 1, fail: 0 };
// when a checkpoint arrived (its card's creation, shown in local time) and how far it trained
export function arrived(b) {
  const t = b.created || b.added; if (!t) return '';
  const d = new Date(/Z$|[+-]\d\d:?\d\d$/.test(t) ? t : t + 'Z');
  if (isNaN(d)) return '';
  const today = new Date().toDateString() === d.toDateString();
  return (today ? '' : d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ') + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
export function trained(b, {epoch = false} = {}) {
  const m = /_e(\d+)(?:b(\d+))?$/.exec(b.name || ''), e = b.epoch;
  if (epoch && m) return `epoch ${+m[1]}` + (m[2] ? ` · batch ${+m[2]}` : '');
  if (typeof e === 'number' && e < 1 && e > 0) return `${Math.round(e * 100)}% of a pass` + (m && m[2] ? ` · b${+m[2]}` : '');
  if (m) return `${+m[1]} pass${+m[1] === 1 ? '' : 'es'}` + (m[2] ? ` + batch ${+m[2]}` : '');
  return e != null ? `epoch ${e}` : '';
}

export class Brains {
  constructor({ panel, stars, button, onPlay, onPause, onResume, onStop, onOpen, onStats, transport, status }) {
    this.panel = panel; this.starsBox = stars; this.onPlay = onPlay; this.onOpen = onOpen || onPlay; this.onStats = onStats;
    this.onPause = onPause || (() => {}); this.onResume = onResume || (() => {}); this.onStop = onStop || (() => {});
    this.transport = transport || (() => ({ playing: false, stopped: true }));
    this.status = status || (() => {});
    this.list = []; this.active = null; this.drag = null;
    button.onclick = () => this.open();
    panel.addEventListener('mousedown', (e) => { if (e.target === panel) this.close(); });   // a click outside the box
    addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) { e.stopPropagation(); this.close(); }
      else if ((e.key === 'e' || e.key === 'E') && !e.metaKey && !e.ctrlKey && !/INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || '')) { this.isOpen() ? this.close() : this.open(); }
    }, true);
    this.starsBox.innerHTML = '';
    this.activeStars = [1, 2, 3, 4, 5].map((n) => {
      const s = el('button', 'st', '★'); s.title = `${n} star${n > 1 ? 's' : ''} for this brain (${n})`;
      s.onclick = () => this.rateActive(n); this.starsBox.appendChild(s); return s;
    });
    this.paintActive();
  }

  isOpen() { return this.panel.classList.contains('open'); }
  transportChanged() { if (this.isOpen() && !this.panel.contains(document.activeElement) && this.drag == null) this.render(); }
  open() { this.panel.classList.add('open'); this.refresh(); }
  close() { this.panel.classList.remove('open'); }

  // the menu reloaded: who plays now and the curated list as the server sorts it
  setActive(name, list) {
    this.active = name; if (list) this.list = list;
    this.paintActive();
    if (this.isOpen() && !this.panel.contains(document.activeElement) && !this.drag) this.render();
  }

  starsOf(name) { const b = this.list.find((x) => x.name === name); return b ? b.stars || 0 : 0; }

  paintActive() {
    const brain = this.active && this.active !== 'demo', n = brain ? this.starsOf(this.active) : 0;
    this.starsBox.style.visibility = brain ? 'visible' : 'hidden';
    this.activeStars.forEach((s, i) => s.classList.toggle('on', i < n));
  }

  async rate(name, n) {
    const cur = this.starsOf(name), stars = cur === n ? 0 : n;      // clicking the current rating clears it
    const j = await post('/api/brains/update', { name, stars });
    if (j.error) { this.status(j.error); return; }
    const b = this.list.find((x) => x.name === name); if (b) b.stars = stars;
    this.paintActive(); if (this.isOpen()) this.render();
  }

  rateActive(n) { if (this.active && this.active !== 'demo') this.rate(this.active, n); }

  async refresh() {
    try {
      const j = await (await fetch('/api/brains')).json();
      this.list = j.brains || []; this.active = j.active || this.active;
      this.paintActive(); this.render();
    } catch { this.status('brain list: server unreachable'); }
  }

  async saveOrder() {
    const j = await post('/api/brains/order', { names: this.list.map((b) => b.name) });
    if (j.error) this.status(j.error);
    this.render();
  }

  move(i, d) {
    const k = i + d; if (k < 0 || k >= this.list.length) return;
    const [b] = this.list.splice(i, 1); this.list.splice(k, 0, b); this.saveOrder();
  }

  sortBy(what) {
    const key = {
      stars: (b) => [b.stars || 0, QA_RANK[b.qa] ?? -1, b.created || ''],
      qa: (b) => [QA_RANK[b.qa] ?? -1, -(b.qa_failed || []).length, b.stars || 0],
      newest: (b) => [b.created || b.added || ''],
    }[what];
    this.list.sort((a, b) => { const x = key(a), y = key(b); for (let i = 0; i < x.length; i++) { if (x[i] < y[i]) return 1; if (x[i] > y[i]) return -1; } return 0; });
    this.saveOrder();
  }

  render() {
    const box = this.panel; box.innerHTML = '';
    const card = el('div', 'ed'); box.appendChild(card);
    const head = el('div', 'edh');
    head.appendChild(el('div', 'edt', 'brains'));
    const sorts = el('div', 'eds');
    sorts.appendChild(el('span', 'edl', 'sort once by'));
    [['stars', 'stars'], ['qa', 'QA'], ['newest', 'newest']].forEach(([k, t]) => { const b = el('button', 'sb', t); b.onclick = () => this.sortBy(k); sorts.appendChild(b); });
    head.appendChild(sorts);
    const x = el('button', 'edx', '✕'); x.title = 'close (Esc)'; x.onclick = () => this.close(); head.appendChild(x);
    card.appendChild(head);
    card.appendChild(el('div', 'edn', 'Newest on top. Drag a row (or use the arrows) to place it yourself: placed brains keep your order below the new arrivals, and the menu follows. Edit a name or note in place; Enter saves.'));
    const rows = el('div', 'edr'); card.appendChild(rows);
    if (!this.list.length) { rows.appendChild(el('div', 'empty', 'no brains listed yet')); return; }
    this.list.forEach((b, i) => {
      const row = el('div', 'brow' + (b.name === this.active ? ' on' : ''));
      row.draggable = true;
      row.addEventListener('dragstart', (e) => { this.drag = i; row.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move'; });
      row.addEventListener('dragend', () => { this.drag = null; row.classList.remove('dragging'); });
      row.addEventListener('dragover', (e) => { e.preventDefault(); row.classList.add('over'); });
      row.addEventListener('dragleave', () => row.classList.remove('over'));
      row.addEventListener('drop', (e) => {
        e.preventDefault(); row.classList.remove('over');
        if (this.drag == null || this.drag === i) return;
        const [m] = this.list.splice(this.drag, 1); this.list.splice(i, 0, m); this.drag = null; this.saveOrder();
      });
      const grip = el('div', 'gr', `${i + 1}`); grip.title = 'drag to move';
      const arrows = el('div', 'ar');
      const up = el('button', 'mv', '↑'); up.title = 'move up'; up.onclick = () => this.move(i, -1);
      const dn = el('button', 'mv', '↓'); dn.title = 'move down'; dn.onclick = () => this.move(i, 1);
      arrows.append(up, dn);
      // the same transport as the page: ▶ / ⏸ (pause: the composition goes on where it was) and ■ (stop: the brain
      // is reset, ▶ then starts a new composition); ■ only on the brain that is loaded
      const t = this.transport(), mine = b.name === this.active, sounding = mine && t.playing;
      const play = el('button', 'pl', sounding ? '⏸' : '▶');
      play.title = sounding ? 'pause' : (mine && !t.stopped ? 'go on (paused)' : (mine ? 'play: a new composition' : 'play this brain'));
      play.onclick = async () => {
        if (sounding) { await this.onPause(); this.render(); return; }
        if (mine) { await this.onResume(); this.render(); return; }
        play.textContent = '…'; this.active = b.name;
        await this.onPlay(b.name); this.render();
      };
      const stopB = el('button', 'pl sp' + (mine && !t.stopped ? '' : ' off'), '■');
      stopB.title = mine ? 'stop: resets the brain; ▶ then starts a new composition' : '';
      stopB.onclick = async () => { if (!mine || t.stopped) return; await this.onStop(); this.render(); };
      const ctl = el('div', 'tc'); ctl.append(play, stopB);
      const main = el('div', 'bm');
      const label = this.editable('lab', b.label || '', b.candidate || b.name, (v) => { b.label = v; return post('/api/brains/update', { name: b.name, label: v }); });
      const note = this.editable('nt', b.note || '', 'note', (v) => { b.note = v; return post('/api/brains/update', { name: b.name, note: v }); });
      const meta = el('div', 'mt', `${b.name} · ${b.reason || ''}`);
      if (this.onStats) {
        const stats = el('button', 'brain-stat-link', 'Stats');
        stats.title = `Training stats for ${b.name}`;
        stats.onclick = () => { this.close(); this.onStats(b.name); };
        meta.prepend(stats, ' · ');
      }
      main.append(label, note, meta);
      const when = el('div', 'wh'); when.append(el('div', 'wt', arrived(b)), el('div', 'wp', trained(b)));
      when.title = `arrived ${b.created || b.added || ''} (UTC) · ${b.order == null ? 'new: on top until you place it' : 'placed by you'}`;
      const qa = el('div', 'qa q' + (QA_RANK[b.qa] ?? 'n'), b.qa ? (b.qa === 'HOLDS A BEAT' ? 'beat' : b.qa) : '–');
      qa.title = b.qa ? `QA ${b.qa}${(b.qa_failed || []).length ? ': open: ' + b.qa_failed.join('; ') : ''}` : 'not checked yet';
      const stars = el('div', 'rs');
      [1, 2, 3, 4, 5].forEach((n) => { const s = el('button', 'st' + (n <= (b.stars || 0) ? ' on' : ''), '★'); s.title = `${n} star${n > 1 ? 's' : ''}`; s.onclick = () => this.rate(b.name, n); stars.appendChild(s); });
      const del = el('button', 'dl', '✕'); del.title = 'delete from the list';
      del.onclick = async () => {
        if (!del.classList.contains('confirm')) { del.classList.add('confirm'); del.textContent = 'delete?'; setTimeout(() => { del.classList.remove('confirm'); del.textContent = '✕'; }, 3000); return; }
        const j = await post('/api/brains/delete', { name: b.name });
        if (j.error) { this.status(j.error); return; }
        if (j.active !== this.active) this.onPlay(null, j.active);
        this.refresh();
      };
      row.title = 'double-click: play this brain';
      row.addEventListener('dblclick', async (e) => {
        if (e.target.closest('input, button, .rs')) return;      // editing a name, a star, a button: not an open
        this.close(); await this.onOpen(b.name);
      });
      row.append(grip, arrows, ctl, when, main, qa, stars, del); rows.appendChild(row);
    });
  }

  editable(cls, value, placeholder, save) {
    const e = el('input', cls); e.value = value; e.placeholder = placeholder; e.spellcheck = false;
    let last = value;
    const commit = async () => { const v = e.value.trim(); if (v === last) return; last = v; const j = await save(v); if (j && j.error) this.status(j.error); };
    e.addEventListener('keydown', (ev) => { ev.stopPropagation(); if (ev.key === 'Enter') e.blur(); if (ev.key === 'Escape') { e.value = last; e.blur(); } });
    e.addEventListener('blur', commit);
    e.addEventListener('mousedown', (ev) => ev.stopPropagation());
    return e;
  }
}
