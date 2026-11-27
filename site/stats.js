// Checkpoint learning curves and the latest status from the local watcher.
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const NS = 'http://www.w3.org/2000/svg';
const sv = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const pct = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '–';
const PARTS = [['arrangement', 'Arrangement'], ['drum_rows', 'Drum patterns'],
               ['synth_line', 'Synth lines'], ['synth_controls', 'Synth controls']];
const axisLabel = (v, composer) => composer ? (v >= 60 ? `${(v / 60).toFixed(1)} h` : `${Math.round(v)} min`) : `${v.toFixed(2)}`;
const music = (point) => [
  Number.isFinite(point.gains?.[0]) && `Memory path strength ${point.gains[0].toFixed(3)}`,
  Number.isFinite(point.composition) && `Music selection score ${pct(point.composition)}`,
  Number.isFinite(point.symbolic) && `Pattern continuation ${pct(point.symbolic)}`,
  Number.isFinite(point.hold) && `Repeated-bar pattern baseline ${pct(point.hold)}`,
  Number.isFinite(point.phrase_events?.score) && `Phrase event score ${pct(point.phrase_events.score)} · repeated bar ${pct(point.phrase_events.hold)}`,
  Number.isFinite(point.development?.score) && `Buildup, release and motif handoff ${pct(point.development.score)} · repeated bar ${pct(point.development.hold)}`,
  ...[['drum_hits', 'Drum hits'], ['drum_accents', 'Drum accents'], ['synth_attacks', 'Synth onset timing'], ['synth_notes', 'Pitched synth onsets'], ['synth_ties', 'Synth ties']]
    .flatMap(([key, label]) => {
      const detail = point.test_parts?.detail?.[key];
      if (!Number.isFinite(detail?.f1)) return [];
      const baseline = point.copy_baseline?.test?.detail?.[key]?.f1;
      return [`${label} held-out F1 ${pct(detail.f1)} · repeated bar ${pct(baseline)}`];
    }),
].filter(Boolean);

function select(label, id, values, current, set) {
  const field = el('label', 'stats-field', label);
  const input = el('select');
  input.id = id;
  for (const [value, title] of values) {
    const opt = el('option', '', title);
    opt.value = value;
    input.append(opt);
  }
  input.value = current;
  input.onchange = () => set(input.value);
  field.append(input);
  return field;
}

const get = (object, path) => path.split('.').reduce((value, key) => value?.[key], object);
const count = (value) => Number(value).toLocaleString('en-US');
const cap = (text) => text[0].toUpperCase() + text.slice(1);
// every part a curve can show: the four groups of the average, then each instrument's own slots
const partsOf = (points) => [...PARTS.map(([key, label]) => [key, label]),
  ...Object.keys(points.find((p) => p.test_parts?.instruments)?.test_parts.instruments || {})
    .map((name) => [`instruments.${name}`, `${cap(name)} · every slot it plays`])];

// ---------------------------------------------------------------- brain layout
// The drawing is laid out from the layout the server reads off the brain's genome (brain/anatomy.py):
// what the brain hears on top, the regions in the middle, what each pathway writes at the bottom.
// ---------- does the coordinator contribute (generation 2) ----------
// From the trainer's metrics rows: hidden_credit (Cadence's teaching contrast per region, the mean over the lessons
// since the evaluation before), coordinator_cut (the held-out error and the free rollout with the coordinator cut)
// and rollout (the free rollout intact). A share is a region's teaching contrast over the motor's. The coordinator
// is the region "association" in the pathway brains (brain/hierarchy.py: every pathway's plan meets there).
// credit_warnings carries the warnings Cadence gave (it warns when a region's contrast stays at or under a thousandth
// of the motor's); a row written before the trainer kept them is judged by that thousandth.
const creditShare = (value) => !Number.isFinite(value) ? '–' : value >= 0.1 ? `${(value * 100).toFixed(0)}%`
  : value >= 0.01 ? `${(value * 100).toFixed(1)}%` : `${(value * 100).toFixed(2)}%`;
const creditMean = (entry) => Number.isFinite(entry?.mean) ? entry.mean : Number.isFinite(entry) ? entry : null;
const creditRegion = (name) => name.replace(/[_/]/g, ' ');
function controller(point) {
  if (!point) return null;
  const hc = point.hidden_credit, cut = point.coordinator_cut;
  let credit = null;
  if (hc) {
    const motor = creditMean(hc.credit_motor_contrast);
    const regions = Object.keys(hc).map((key) => /^credit_(.+)_contrast$/.exec(key))
      .filter((m) => m && m[1] !== 'motor' && !m[1].endsWith('_voltage'))
      .map((m) => ({name: m[1], value: creditMean(hc[m[0]])})).filter((r) => Number.isFinite(r.value));
    if (Number.isFinite(motor) && motor > 0 && regions.length) {
      for (const r of regions) r.share = r.value / motor;
      const coordinator = regions.find((r) => r.name === 'association') || regions.find((r) => /coordinator/.test(r.name));
      const weakest = regions.reduce((a, b) => (b.share < a.share ? b : a));
      const warnings = (point.credit_warnings || []).map(String);
      const kept = Array.isArray(point.credit_warnings);
      credit = {coordinator, weakest, regions, warnings, warned: kept ? warnings.length > 0 : regions.some((r) => r.share <= 1e-3),
        lessons: hc.credit_motor_contrast?.lessons};
    }
  }
  const intact = point.test, cutError = cut?.test_error?.balanced;
  const cutting = Number.isFinite(cutError) && Number.isFinite(intact)
    ? {intact, cut: cutError, points: (cutError - intact) * 100, rollout: cut.rollout || null} : null;
  return {credit, cut: cutting, rollout: point.rollout || null};
}
// the newest evaluation that carries any of the three
const controlled = (points) => [...points].reverse().find((p) => p.hidden_credit || p.coordinator_cut || p.rollout) || null;
const cutPoints = (points) => `${points > 0 ? '+' : points < 0 ? '−' : '±'}${Math.abs(points).toFixed(1)} points`;
const cutVerdict = (c) => Math.abs(c.points) < 0.1 ? 'no difference' : c.points > 0 ? 'the coordinator helps' : 'the brain does better without it';
const rolled = (r) => !r ? '–' : ['balanced', 'arrangement', 'change'].map((key) => `${key} ${Number.isFinite(r[key]) ? (Math.abs(r[key]) <= 1 ? pct(r[key]) : r[key].toFixed(2)) : '–'}`).join(' · ');
function warnBadge(credit) {
  const badge = el('span', 'stats-warn', 'warned');
  const weak = credit.regions.filter((r) => r.share <= 1e-3).map((r) => creditRegion(r.name));
  badge.title = credit.warnings.length ? credit.warnings.join('\n')
    : `Cadence warns when a region's teaching contrast stays at or under a thousandth of the motor's: ${weak.join(', ')}`;
  return badge;
}
function controllerBlock(point) {
  const c = controller(point);
  const box = el('div', 'stats-controller');
  box.append(el('div', 'lh', 'Does the coordinator contribute'));
  const line = (label) => { const row = el('div', 'edn'); row.append(el('span', 'stats-number', label)); box.append(row); return row; };
  const credit = line('Teaching contrast · ');
  if (c?.credit) {
    const k = c.credit;
    credit.append(`${k.coordinator ? `coordinator (${creditRegion(k.coordinator.name)}) ${creditShare(k.coordinator.share)}` : 'no coordinator region'} of the motor's · weakest region ` +
      `${creditRegion(k.weakest.name)} ${creditShare(k.weakest.share)}${Number.isFinite(k.lessons) ? ` · mean over ${k.lessons} lessons` : ''} `);
    if (k.warned) credit.append(warnBadge(k));
    credit.title = k.regions.map((r) => `${creditRegion(r.name)} ${creditShare(r.share)}`).join('\n');
  } else credit.append('–');
  const cut = line('Coordinator cut · ');
  if (c?.cut) cut.append(`held-out error ${pct(c.cut.cut)} with the coordinator cut, ${pct(c.cut.intact)} intact: ${cutPoints(c.cut.points)}, ${cutVerdict(c.cut)}`);
  else cut.append('–');
  const roll = line('Free rollout · ');
  roll.append(c?.rollout || c?.cut?.rollout ? `${rolled(c?.rollout)}${c?.cut?.rollout ? ` · with the coordinator cut: ${rolled(c.cut.rollout)}` : ''}` : '–');
  return box;
}

const STYLE = `
  #stats .stats-actions { display: inline-flex; align-items: baseline; gap: 12px; }
  #stats .stats-warn { display: inline-block; margin-left: 4px; padding: 0 4px; border: 1px solid #6b5a2a; border-radius: 3px; color: #f2c14e; font-size: 9px; letter-spacing: .04em; vertical-align: 1px; }
  #stats .stats-controller { margin: 10px 0 4px; }
  #stats .stats-controller .lh { margin-top: 4px; }
  #stats .stats-training { display: inline-flex; align-items: center; gap: 6px; font-size: 10px; color: var(--dim); white-space: nowrap; }
  #stats .stats-training .sw { position: relative; width: 18px; height: 10px; border-radius: 5px; border: 1px solid var(--dim); flex: none; }
  #stats .stats-training .sw::after { content: ''; position: absolute; top: 1px; left: 1px; width: 6px; height: 6px; border-radius: 50%; background: var(--dim); }
  #stats .stats-training.on { color: var(--fg); }
  #stats .stats-training.on .sw { border-color: var(--calm); }
  #stats .stats-training.on .sw::after { left: 9px; background: var(--calm); box-shadow: 0 0 5px var(--calm); }
  #stats .stats-training.unknown .sw { border-style: dashed; }
  #stats .stats-check { display: inline-flex; align-items: baseline; gap: 5px; color: var(--mid); white-space: nowrap; }
  #stats .stats-check .g { color: var(--dim); width: 9px; text-align: center; flex: none; }
  #stats .stats-check.good .g { color: var(--calm); } #stats .stats-check.good { color: var(--fg); }
  #stats .stats-check.bad .g { color: var(--heart); } #stats .stats-check.bad { color: var(--fg); }
  #stats .stats-check.wait .g { color: var(--aroused); }
  #stats .stats-slate .lt td { vertical-align: top; }
  #stats .stats-candidate-head h2 { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; }
  #stats .av { stroke: var(--mid); stroke-width: 1; stroke-dasharray: 1 4; }
  #stats .k.av { border-top: 1px dotted var(--mid); height: 0; }
  #stats .stats-candidate-head .stats-load { text-decoration: none; color: var(--fg); }
  #stats .stats-layout { margin: 8px 0 10px; color: var(--mid); }
  #stats .stats-layout > summary { cursor: pointer; color: var(--mid); width: fit-content; }
  #stats .stats-layout > summary:hover { color: var(--fg); }
  #stats .stats-layout .ly { width: 100%; max-width: 960px; height: auto; display: block; margin-top: 8px; }
  #stats .stats-layout .bx { fill: rgba(255, 255, 255, .025); stroke: var(--dim); stroke-width: 1; }
  #stats .stats-layout .bx.hears { stroke: var(--line); fill: none; }
  #stats .stats-layout .bx.coordinator { stroke: var(--calm); }
  #stats .stats-layout .bx.writes { fill: rgba(242, 166, 90, .07); stroke: var(--aroused); }
  #stats .stats-layout text { font-size: 9px; fill: var(--mid); }
  #stats .stats-layout text.t1 { fill: var(--fg); }
  #stats .stats-layout text.er { fill: var(--s-test); }
  #stats .stats-layout text.rd { fill: var(--dim); }
  #stats .stats-layout .dr { fill: none; stroke: var(--mid); stroke-width: 1.2; }
  #stats .stats-layout .fb { fill: none; stroke: var(--mid); stroke-width: 1; stroke-dasharray: 3 3; opacity: .75; }
  #stats .stats-layout .ah { fill: var(--mid); }
  #stats .stats-layout .k.dr { background: var(--mid); } #stats .stats-layout .k.fb { border-top: 1px dashed var(--mid); height: 0; }
  #stats .stats-history { margin: 2px 0 18px; }
  #stats .stats-history > summary { cursor: pointer; width: fit-content; }
  #stats .stats-history > summary:hover { color: var(--fg); }
  #stats .stats-history .hx-title { color: var(--fg); font-size: 10px; margin: 12px 0 2px; }
  #stats .stats-history .edn { margin: 2px 0 6px; }
  #stats .stats-history .hx-pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 4px 28px; max-width: 960px; }
  @media (max-width: 700px) { #stats .stats-history .hx-pair { grid-template-columns: minmax(0, 1fr); } }
  #stats .stats-history .hx-legend { flex-wrap: wrap; gap: 2px 0; }
  #stats .stats-history .hx-item { display: inline-flex; align-items: center; white-space: nowrap; margin-right: 6px; }
  #stats .stats-history .hx-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  #stats .stats-history .hx-guide { stroke: var(--dim); stroke-width: 1; opacity: .35; }
  #stats .stats-history .hx-guide.on { stroke: var(--mid); opacity: 1; }
  #stats .stats-history .hx-mark { cursor: default; outline: none; }
  #stats .stats-history .hx-mark circle { fill: var(--bg); stroke: var(--mid); stroke-width: 1; }
  #stats .stats-history .hx-mark text { fill: var(--fg); font-size: 8px; text-anchor: middle; pointer-events: none; }
  #stats .stats-history .hx-mark:hover circle, #stats .stats-history .hx-mark:focus-visible circle, #stats .stats-history .hx-mark.on circle { stroke: var(--fg); fill: var(--faint); }
  #stats .stats-history .hx-end { fill: var(--fg); font-size: 9px; }
  #stats .stats-history .hx-off { fill: var(--mid); font-size: 8px; }
  #stats .stats-history .tp.hx-wide { white-space: normal; width: max-content; max-width: 340px; }
  #stats .stats-history .tp .v { color: var(--fg); font-weight: 600; }
  #stats .stats-history .tp .d { color: var(--mid); }
  #stats .stats-history .hx-list { list-style: none; padding: 0; margin: 8px 0 4px; max-width: 960px; font-size: 10px; }
  #stats .stats-history .hx-list li { display: grid; grid-template-columns: 18px 88px minmax(0, 1fr); gap: 1px 8px; padding: 4px 2px; border-bottom: 1px solid var(--line); color: var(--mid); }
  #stats .stats-history .hx-list li.on { background: rgba(255, 255, 255, .03); }
  #stats .stats-history .hx-list .hx-no { color: var(--fg); border: 1px solid var(--mid); border-radius: 50%; width: 15px; height: 15px; line-height: 13px; text-align: center; font-size: 8px; }
  #stats .stats-history .hx-list .hx-when { color: var(--dim); font-variant-numeric: tabular-nums; white-space: nowrap; }
  #stats .stats-history .hx-list .hx-what { color: var(--fg); }
  #stats .stats-history .hx-list .hx-ba { grid-column: 3; }
  #stats .stats-history .lt { max-width: 960px; }
`;
if (!document.getElementById('stats-layout-style')) {
  const style = el('style', '', STYLE);
  style.id = 'stats-layout-style';
  document.head.append(style);
}

const REGION = {
  association: 'Coordinator', phrase_context: 'Phrase context', prefrontal: 'Working trace', code: 'Learned code',
};
const regionName = (region, stages) => {
  if (region.role === 'stage' || (region.role === 'coordinator' && stages > 1)) return `Stage ${region.stage} of ${stages}`;
  if (region.pathway) return `${cap(region.pathway)} ${region.role}`;
  return REGION[region.name] || cap(region.name.replaceAll('_', ' '));
};
const WRITES = {
  kit: (n) => 'kit', gate: (n) => `${n} on/off`, sound: (n) => `${n} sound choices`, onoff: (n) => `${n} on/off`,
  mix: (n) => `${n} mix controls`, drum: (n) => n > 3 ? `${n} drum steps` : `${n} drum rows`,
  line: (n) => n % 16 ? `${n} line slots` : `${n} note steps`, control: (n) => `${n} sound controls`,
};
const OUTPUT = { all: 'Writes every part', coordinator: 'Arrangement', rhythm: 'Rhythm', melody: 'Melody' };

function wrap(words, width) {
  const lines = [];
  for (const word of words) {
    if (lines.length && (lines.at(-1) + ' · ' + word).length <= width) lines[lines.length - 1] += ' · ' + word;
    else lines.push(word);
  }
  return lines;
}

let drawings = 0;
function layoutDrawing(layout, point) {
  const W = 720, PAD = 10, LINE = 12, CHAR = 5.6;
  const boxes = {}, wires = [];
  const box = (id, x, y, w, cls, lines) => (boxes[id] = {id, x, y, w, h: 10 + LINE * lines.length, cls, lines});
  const chars = (w) => Math.floor((w - 14) / CHAR);
  const byName = Object.fromEntries(layout.regions.map((region) => [region.name, region]));
  const stageList = layout.regions.filter((region) => region.role === 'stage' || region.role === 'coordinator');
  stageList.forEach((region, index) => { region.stage = index + 1; });
  const chain = layout.architecture === 'chain';
  const stages = chain ? stageList.length : 1;
  const recall = layout.recall;
  // what a region reads beside the regions drawn around it
  const reads = (region, width) => {
    const out = [];
    if (recall) {
      const heads = recall.lands.map((name, head) => name === region.name ? head + 1 : 0).filter(Boolean);
      if (heads.length === recall.lands.length) out.push(`recall × ${heads.length}`);
      else if (heads.length) out.push(`recall head ${heads.join(', ')}`);
    }
    if (region.name === 'association' && byName.prefrontal) out.push('working trace');
    if (region.role === 'detail' && layout.reads) out.push('the code');
    if (!out.length) return [];
    out[0] = `reads ${out[0]}`;
    return wrap(out, chars(width)).map((text) => [text, 'rd']);
  };
  const regionBox = (region, x, y, w) => box(region.name, x, y, w, region.role,
    [[regionName(region, stages), 't1'], [`${count(region.size)} neurons`], ...reads(region, w)]);
  const outputBox = (output, x, y, w) => {
    const title = OUTPUT[output.name] || cap(output.name);
    const parts = Object.entries(output.parts).map(([kind, n]) => (WRITES[kind] || ((m) => `${m} ${kind}`))(n));
    const lines = [[`${title} · ${output.slots} ${output.slots === 1 ? 'slot' : 'slots'}`, 't1'],
                   ...wrap(parts, chars(w)).map((text) => [text])];
    for (const [label, path] of output.errors) {
      const test = get(point?.test_parts, path), base = get(point?.copy_baseline?.test, path);
      if (!Number.isFinite(test)) continue;
      const name = output.errors.length > 1 ? `${label}: ` : '';
      const words = [`${name}test error ${pct(test)}`, ...(Number.isFinite(base) ? [`repeat ${pct(base)}`] : [])];
      lines.push(...wrap(words, chars(w)).map((text) => [text, 'er']));
    }
    return box(`out:${output.name}`, x, y, w, 'writes', lines);
  };

  // what the brain hears, and its learned code
  const hears = byName.sensory, code = byName.code;
  const top = code ? (W - 2 * PAD - 40) / 2 : W - 2 * PAD;
  box('hears', PAD, 8, top, 'hears', [['Hears', 't1'],
    ...wrap(['the bar just played', 'the mood', 'the bar clock', `${count(layout.inputs)} inputs`], chars(top)).map((text) => [text])]);
  if (code) {
    box('code', PAD + top + 40, 8, top, 'code', [['Learned code', 't1'], [`${count(code.size)} units`]]);
    boxes.code.h = boxes.hears.h = Math.max(boxes.code.h, boxes.hears.h);
    wires.push({from: 'hears', to: 'code', kind: 'one-way', side: 'h'});
  }
  const source = code ? 'code' : 'hears';
  let y = 8 + boxes.hears.h + 26;

  if (chain) {
    const w = 420, x = (W - w) / 2;
    let previous = source;
    for (const region of stageList) {
      regionBox(region, x, y, w);
      wires.push({from: previous, to: region.name, kind: previous === source ? 'one-way' : layout.apical ? 'feedback' : 'mutual'});
      previous = region.name;
      y += boxes[region.name].h + 22;
    }
    const output = outputBox(layout.outputs[0], x, y, w);
    wires.push({from: previous, to: output.id, kind: layout.apical ? 'feedback' : 'mutual'});
    y += output.h;
  } else {
    // The coordinator spans every column, so each connection runs straight down: its own output first,
    // then one column per pathway with the plan above the detail above what the pathway writes.
    const hub = byName.association, context = byName.phrase_context;
    if (context) {
      regionBox(context, PAD, y, 220);
      y += boxes[context.name].h + 22;
    }
    regionBox(hub, PAD, y, W - 2 * PAD);
    wires.push({from: source, to: hub.name, kind: 'one-way'});
    if (context) wires.push({from: context.name, to: hub.name, kind: layout.apical ? 'feedback' : 'mutual', up: true});
    const hubBottom = y + boxes[hub.name].h;
    const pair = layout.apical ? 'feedback' : 'mutual';
    const gap = 10, gutter = 22, k = layout.outputs.length;
    const colW = (W - 2 * PAD - (k - 1) * gap) / k;
    const pathways = layout.outputs.filter((output) => output.name !== 'coordinator');
    const tall = (suffix) => Math.max(...pathways.map((path) =>
      10 + LINE * (2 + reads(byName[`${path.name}_${suffix}`], colW - (suffix === 'plan' ? gutter : 0)).length)));
    const planY = hubBottom + 26, detailY = planY + tall('plan') + 24, outY = detailY + tall('detail') + 24;
    let bottom = outY;
    layout.outputs.forEach((path, index) => {
      const x = PAD + index * (colW + gap), middle = x + colW / 2;
      const output = outputBox(path, x, outY, colW);
      bottom = Math.max(bottom, outY + output.h);
      if (path.name === 'coordinator') {      // the coordinator writes these slots itself
        wires.push({kind: pair, points: [[middle, hubBottom], [middle, outY]]});
        return;
      }
      const plan = regionBox(byName[`${path.name}_plan`], x + gutter, planY, colW - gutter);
      const detail = regionBox(byName[`${path.name}_detail`], x, detailY, colW);
      const planX = plan.x + plan.w / 2;
      // the detail's summary climbs through the plan; what comes back down that way is feedback
      wires.push({kind: pair, points: [[planX, planY], [planX, hubBottom]]});
      wires.push({kind: pair, points: [[planX, detailY], [planX, planY + plan.h]]});
      // the coordinator drives the detail directly, past the plan
      wires.push({kind: 'one-way', points: [[x + gutter / 2, hubBottom], [x + gutter / 2, detailY]]});
      wires.push({kind: pair, points: [[middle, detailY + detail.h], [middle, outY]]});
    });
    y = bottom;
  }

  const H = y + 10;
  const id = `ly${drawings++}`;
  const svg = sv('svg', {viewBox: `0 0 ${W} ${H}`, class: 'ly', role: 'img',
                         'aria-label': 'Brain layout: what it hears, its regions and what each pathway writes'});
  const defs = sv('defs', {});
  const marker = sv('marker', {id, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse'});
  marker.append(sv('path', {d: 'M0,0 L8,4 L0,8 z', class: 'ah'}));
  defs.append(marker);
  svg.append(defs);
  const line = (points, cls, shift) => {
    // a wire and its return run side by side: move the points sideways from their own direction
    const [a, b] = [points[0], points.at(-1)];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const dx = -(b[1] - a[1]) / length * shift, dy = (b[0] - a[0]) / length * shift;
    svg.append(sv('polyline', {points: points.map(([px, py]) => `${px + dx},${py + dy}`).join(' '), class: cls,
                               'marker-end': `url(#${id})`}));
  };
  for (const wire of wires) {
    let points = wire.points;
    if (!points) {
      const a = boxes[wire.from], b = boxes[wire.to];
      if (wire.side === 'h') {
        const level = a.y + Math.min(a.h, b.h) / 2;
        points = a.x < b.x ? [[a.x + a.w, level], [b.x, level]] : [[a.x, level], [b.x + b.w, level]];
      } else {
        const x = (Math.max(a.x, b.x) + Math.min(a.x + a.w, b.x + b.w)) / 2;
        points = wire.up ? [[x, b.y], [x, a.y + a.h]] : [[x, a.y + a.h], [x, b.y]];
      }
    }
    if (wire.kind === 'one-way') line(points, 'dr', 0);
    else {
      line(points, 'dr', -3);
      line([...points].reverse(), wire.kind === 'feedback' ? 'fb' : 'dr', -3);
    }
  }
  for (const item of Object.values(boxes)) {
    svg.append(sv('rect', {x: item.x, y: item.y, width: item.w, height: item.h, rx: 4, class: `bx ${item.cls}`}));
    item.lines.forEach(([text, cls], index) => {
      const label = sv('text', {x: item.x + 7, y: item.y + 15 + index * LINE, class: cls || ''});
      label.textContent = text;
      svg.append(label);
    });
  }
  return svg;
}

function layoutFacts(layout) {
  const facts = [`${count(layout.neurons)} neurons`, `${count(layout.synapses)} trainable synapses`];
  if (layout.recall) {
    facts.push(`recall memory: ${layout.recall.heads} ${layout.recall.heads === 1 ? 'head' : 'heads'}` +
               (layout.recall.bars ? ` over the last ${layout.recall.bars} bars` : ''));
  }
  return facts.join(' · ');
}

// ---------------------------------------------------------------- matched comparisons
// Runs named <wiring>_s<seed>_<tag> are one comparison: the same lessons and evaluations for every wiring,
// several seeds each. Series colors: the validated categorical slots 1-5 on the dark surface, in fixed order.
const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'];
const COMPARED = /^(.+)_s(\d+)_([A-Za-z0-9]+)$/;
function wiringName(brain, stem) {
  const layout = brain.layout;
  if (layout?.architecture === 'chain') return 'Shared chain';
  if (layout?.architecture === 'rhythm-melody-v1') return 'Rhythm and melody pathways';
  if (layout?.architecture === 'trio-v1') return 'Beat, bass and melody pathways';
  if (layout?.architecture === 'parts-v1') return 'Notes and sound apart';
  if (layout?.architecture === 'instruments-v1') {
    const own = layout.recall?.lands.some((name) => name.endsWith('_plan'));
    const deep = layout.regions?.some((region) => region.name === 'phrase_context') ? ', two coordinator layers' : '';
    return `Instrument pathways${layout.reads && own ? ', direct reads and own recall' : layout.reads ? ', direct reads' : own ? ', own recall' : ''}${deep}`;
  }
  return stem;
}
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
// The ranking orders brains that write the same slots: errors of different layouts do not compare.
const RANKS = [['train', 'Lowest training error'], ['test', 'Lowest test error']];
const KEPT = 'cadance-stats';          // the panel's state across a reload of the page
const NEW_LOW = .002;        // an error counts as a new low when it beats the lowest so far by this much (the trainer's rule)
const clock = (seconds) => new Date(seconds * 1000).toLocaleTimeString([], {hour12: false});
// The switch beside a brain: on while a box trains it, off when none does, dashed when the mirror cannot say.
function trainingMark(brain, long) {
  const box = brain.training;
  const mark = el('span', `stats-training ${box ? 'on' : box === null ? 'unknown' : 'off'}`);
  mark.append(el('span', 'sw'));
  let text = box ? 'training' : box === null ? 'unknown' : 'not training';
  if (box && long) {
    const since = Date.parse(box.since);
    text += ` · ${box.type} ${box.instance}` + (Number.isFinite(since) ? ` · since ${new Date(since).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hour12: false})}` : '');
  }
  mark.append(document.createTextNode(text));
  mark.title = box ? `A box is training this brain: ${box.type} ${box.instance}` : box === null
    ? 'The mirror that watches the boxes stopped reporting, so the state is unknown' : 'No box is training this brain';
  return mark;
}
const duration = (minutes) => !Number.isFinite(minutes) ? '–' : minutes >= 90 ? `${(minutes / 60).toFixed(1)} h` : `${Math.round(minutes)} min`;
// The epochs behind a reading, from its name: _e012 is twelve whole epochs, _e012b003 the third batch of the twelfth.
function epochsOf(point, lessons) {
  const tag = /_e(\d+)(?:b(\d+))?$/.exec(point.name);
  if (!tag) return null;
  if (tag[2] == null) return Number(tag[1]);
  const batches = lessons?.batch ? Math.ceil(lessons.epoch_windows / lessons.batch) : null;
  return Number(tag[1]) - 1 + (batches ? Math.min(1, Number(tag[2]) / batches) : 0);
}
const slotsOf = (brain) => brain.layout?.outputs.reduce((sum, output) => sum + output.slots, 0) || 0;
// the lowest reading of one error, where it was read, and the evaluations since it last set a new low
function lows(points, key) {
  let best = null, mark = Infinity, since = 0;
  points.forEach((point) => {
    if (!best || point[key] < best[key]) best = point;
    if (point[key] < mark - NEW_LOW) { mark = point[key]; since = 0; } else since += 1;
  });
  return {best, since};
}

// How far a brain has come, from its lowest test error against two fixed levels: writing the commonest bar
// whatever was heard, and repeating the bar just heard. Way is the share of the distance between the two.
const HOLDS = .005;          // within half a point of repeating counts as holding
const LISTENS = .02;         // two points under the commonest-bar level: a reading at the level is no listening
const LEVELS = {
  deaf: ['✕', 'does not listen', 'bad', 'Its test error has not been below the level of writing the commonest bar: it does not use what it hears'],
  listens: ['◐', 'listens', 'wait', 'Its test error is below the level of writing the commonest bar: it uses what it hears'],
  holds: ['●', 'holds', 'good', 'Its test error is at the level of repeating the bar it heard'],
  beyond: ['★', 'beyond repeating', 'good', 'Its test error is below the level of repeating the bar it heard: it writes changes before they come'],
};
function levelOf(brain, points) {
  const blind = brain.blind?.test?.balanced;
  const repeat = points.map((p) => p.copy_baseline?.test?.balanced).find(Number.isFinite);
  // the reading at birth counts: a brain born with its memory path on holds what it hears before any lesson
  if (!Number.isFinite(blind) || !points.length) return null;
  const low = Math.min(...points.map((p) => p.test));
  const key = low >= blind - LISTENS ? 'deaf' : !Number.isFinite(repeat) || low > repeat + HOLDS ? 'listens'
    : low >= repeat - NEW_LOW ? 'holds' : 'beyond';
  return {key, low, blind, repeat, way: Number.isFinite(repeat) && blind > repeat ? (blind - low) / (blind - repeat) : null};
}
// A mark and a label: the mark carries the colour, the label is plain text.
function mark(kind, glyph, text, title) {
  const span = el('span', `stats-check ${kind}`);
  span.append(el('span', 'g', glyph), document.createTextNode(text));
  if (title) span.title = title;
  return span;
}
function levelMark(level, training) {
  if (!level) return mark('none', '–', 'no reading yet');
  const [glyph, label, kind, title] = LEVELS[level.key];
  const text = level.key === 'deaf' && training ? 'not listening yet' : label;
  return mark(level.key === 'deaf' && training ? 'wait' : kind, glyph, text,
    `${title}. Lowest test error ${pct(level.low)}; commonest bar ${pct(level.blind)}` +
    (Number.isFinite(level.repeat) ? `; repeating ${pct(level.repeat)}` : ''));
}
const wayText = (level) => level?.way == null ? '–' : `${Math.round(level.way * 100)}%`.replace('-', '−');

function delta(now, first) {
  if (!Number.isFinite(now) || !Number.isFinite(first)) return '–';
  const step = Math.round((now - first) * 1000) / 10;
  return `${step > 0 ? '+' : ''}${step.toFixed(1)} pp`;
}

// ---------------------------------------------------------------- training history
// quality/history/history.py rebuilds site/history.json every ten minutes: per score (the labelled material the runs
// learn) the frontier of every run's held-out readings over time, and the curated breakthroughs
// (quality/history/breakthroughs.json). A score keeps its colour: its place in the order the scores were made.
const SCORE_ORDER = ['first', 'v2', 'v4', 'v5a', 'v5b'];
const HISTORY_EVERY = 5 * 60e3;      // the open panel asks for the file again this often
const EARLY = 15 * 60e3;             // a score's first quarter hour: birth readings far off that would squeeze its chart
const signed = (v, digits = 1) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`;
const hhmm = (t) => new Date(t).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hour12: false});
const dayTime = (t) => { const d = new Date(t); return `${d.getDate()} ${d.toLocaleString('en-GB', {month: 'short'})} ${hhmm(t)}`; };
const runName = (run) => run.replace(/^slate_/, '');
function scoreStyle(key, keys) {
  let at = SCORE_ORDER.indexOf(key);
  if (at < 0) at = SCORE_ORDER.length + keys.filter((k) => !SCORE_ORDER.includes(k)).indexOf(key);
  return {color: SERIES[at % SERIES.length], dashed: at >= SERIES.length};
}
function lineKey(style) {
  const key = el('span', 'k');
  key.style.background = style.dashed ? `repeating-linear-gradient(90deg, ${style.color} 0 4px, transparent 4px 7px)` : style.color;
  return key;
}
// the frontier at time t: the last new best at or before it
function frontAt(points, t) {
  let found = null;
  for (const point of points) { if (point.time <= t) found = point; else break; }
  return found;
}
function niceRange(lo, hi, ticks) {
  const step = [0.25, 0.5, 1, 2, 5, 10, 20, 50].find((s) => (hi - lo || 1) / s <= ticks) || 100;
  lo = Math.floor(lo / step + 1e-9) * step;
  hi = Math.ceil(hi / step - 1e-9) * step;
  return {lo, hi: hi > lo ? hi : lo + step, step};
}
const MEASURES = {
  gap: {tick: (v) => signed(v, 0), value: (v) => `${signed(v, 2)} points`, end: (v) => signed(v, 1), reference: 0},
  low: {tick: (v) => `${v}%`, value: (v) => `${v.toFixed(2)}%`, end: (v) => `${v.toFixed(1)}%`},
  notes: {tick: (v) => `${v}%`, value: (v) => `${v.toFixed(1)}%`, end: (v) => `${v.toFixed(1)}%`},
};

// One step chart over time: per score its frontier (a step at each new best) from its first reading to its last, or
// to the file's time while a run of that score trains. Readings of a score's first quarter hour that lie far off are
// drawn at the chart's edge and named there. With marks: the breakthroughs as numbered marks under the time axis.
function historyChart(history, measure, {W, plotH, P, xticks, yticks, marks = [], name}) {
  const spec = MEASURES[measure];
  const keys = history.scores.map((s) => s.key);
  const end = Date.parse(history.generated);
  const series = history.scores.filter((s) => s[measure]?.length).map((s) => ({
    ...scoreStyle(s.key, keys), score: s, points: s[measure].map((p) => ({...p, time: Date.parse(p.t)})),
    end: s.training?.length ? end : Date.parse(s.last)}));
  const wrap = el('div', 'cc stats-chart');
  if (!series.length) { wrap.append(el('div', 'edn', 'No reading yet.')); return {wrap, mark: () => {}}; }
  const start = new Date(Math.min(...history.scores.map((s) => Date.parse(s.first))));
  start.setMinutes(0, 0, 0);
  const t0 = start.getTime(), t1 = Math.max(end, ...series.map((s) => s.end));

  // the value range: every reading, unless a score's first quarter hour stretches it far beyond the rest
  const all = [], core = [];
  for (const s of series) {
    const settled = s.points[0].time + EARLY;
    for (const p of s.points) { all.push(p.v); if (p.time >= settled) core.push(p.v); }
    core.push(frontAt(s.points, settled).v);
  }
  if (spec.reference != null) { all.push(spec.reference); core.push(spec.reference); }
  const [aLo, aHi, cLo, cHi] = [Math.min(...all), Math.max(...all), Math.min(...core), Math.max(...core)];
  const {lo, hi, step} = aHi - aLo > 1.6 * Math.max(cHi - cLo, .1) ? niceRange(cLo, cHi, yticks) : niceRange(aLo, aHi, yticks);

  // the breakthroughs under the axis; marks close in time step down one row each, so they read in order
  const X = (t) => P.l + (t - t0) / (t1 - t0) * (W - P.l - P.r);
  const placed = marks.map((b, index) => ({b, index, x: Math.max(P.l, Math.min(W - P.r, X(Date.parse(b.time))))}))
    .sort((a, b) => a.x - b.x);
  placed.forEach((item, i) => { item.lane = i && item.x - placed[i - 1].x < 16 ? placed[i - 1].lane + 1 : 0; });
  const lanes = placed.length ? Math.max(...placed.map((item) => item.lane)) + 1 : 0;
  const bottom = 22 + (lanes ? 8 + lanes * 17 : 0);
  const H = P.t + plotH + bottom;
  const Y = (v) => P.t + (hi - Math.max(lo, Math.min(hi, v))) / (hi - lo) * plotH;
  const svg = sv('svg', {viewBox: `0 0 ${W} ${H}`, class: 'cs', role: 'img', 'aria-label': name});
  const text = (content, x, y, cls, anchor = 'start') => {
    const t = sv('text', {x, y, class: cls, 'text-anchor': anchor});
    t.textContent = content;
    svg.append(t);
    return t;
  };
  for (let v = lo; v <= hi + 1e-9; v += step) {
    svg.append(sv('line', {x1: P.l, x2: W - P.r, y1: Y(v), y2: Y(v), class: 'gl'}));
    text(spec.tick(Math.round(v * 100) / 100), P.l - 6, Y(v) + 3, 'al', 'end');
  }
  // the time axis: whole hours, the date at midnight
  const hour = 3600e3;
  const every = [1, 2, 3, 4, 6, 12, 24].find((h) => (t1 - t0) / (h * hour) <= xticks) || 24;
  for (let t = t0; t <= t1; t += hour) {
    const d = new Date(t);
    if (d.getHours() % every) continue;
    text(d.getHours() ? hhmm(t) : `${d.getDate()} ${d.toLocaleString('en-GB', {month: 'short'})}`, X(t), P.t + plotH + 14, 'al', 'middle');
  }
  if (spec.reference != null) {
    svg.append(sv('line', {x1: P.l, x2: W - P.r, y1: Y(spec.reference), y2: Y(spec.reference), class: 'bl'}));
    text('repeating the bar heard', P.l + 4, Y(spec.reference) - 4, 'al');
  }
  const guides = placed.map(({x}) => {
    const guide = sv('line', {x1: x, x2: x, y1: P.t, y2: P.t + plotH, class: 'hx-guide'});
    svg.append(guide);
    return guide;
  });
  const ends = [];
  for (const s of series) {
    const d = s.points.map((p, i) => i ? `H${X(p.time)}V${Y(p.v)}` : `M${X(p.time)},${Y(p.v)}`).join('') + `H${X(s.end)}`;
    const line = sv('path', {d, class: 'hx-line', stroke: s.color});
    if (s.dashed) line.setAttribute('stroke-dasharray', '7 4');
    svg.append(line);
    let last = -Infinity;
    s.points.forEach((p, i) => {      // a dot at each new best, where the dots do not crowd
      const x = X(p.time);
      if (x - last < 9 && i < s.points.length - 1) return;
      last = x;
      svg.append(sv('circle', {cx: x, cy: Y(p.v), r: 4, fill: s.color, class: 'dt'}));
    });
    const below = s.points.filter((p) => p.v < lo), above = s.points.filter((p) => p.v > hi);
    if (below.length) {
      const p = below.reduce((a, b) => b.v < a.v ? b : a);
      text(`▼ ${spec.end(p.v)}`, X(p.time) + 4, P.t + plotH - 4, 'hx-off');
    }
    if (above.length) {
      const p = above.reduce((a, b) => b.v > a.v ? b : a);
      text(`▲ ${spec.end(p.v)}`, X(p.time) + 4, P.t - 3, 'hx-off');      // above the plot, clear of the clamped line
    }
    if (s.end >= t1) ends.push({s, y: Y(s.points.at(-1).v)});
  }
  // the scores still training are named at the right end, unless their labels would touch
  ends.sort((a, b) => a.y - b.y);
  if (ends.every((item, i) => !i || item.y - ends[i - 1].y >= 11)) {
    for (const {s, y} of ends) text(`${s.score.key === 'first' ? 'first' : s.score.key} ${spec.end(s.points.at(-1).v)}`, W - P.r + 5, y + 3, 'hx-end');
  }

  // the legend: each key with its name, kept together on one line
  const legend = el('div', 'lg hx-legend');
  const item = (key, name) => { const span = el('span', 'hx-item'); span.append(key, document.createTextNode(name)); legend.append(span); };
  for (const s of series) {
    item(lineKey(s), s.score.name + (s.score.training_bars ? ` · ${count(s.score.training_bars)} bars` : '') +
      (s.score.training?.length ? ' · training' : ''));
  }
  if (spec.reference != null) item(el('span', 'k bl'), 'Repeating the bar heard');
  wrap.append(legend);

  // hover: a hairline at the nearest new best of any score, every score's frontier there
  const tip = el('div', 'tp');
  const cursor = sv('line', {y1: P.t, y2: P.t + plotH, class: 'cx'});
  const hit = sv('rect', {x: P.l, y: P.t, width: W - P.l - P.r, height: plotH, class: 'hit'});
  svg.append(cursor, hit);
  const times = [...new Set(series.flatMap((s) => s.points.map((p) => p.time)))].sort((a, b) => a - b);
  // beside the hairline at the plot's top; above a mark, centred on it (x and y in the drawing's units)
  const place = (x, y) => {
    const box = svg.getBoundingClientRect(), px = x / W * box.width, room = wrap.clientWidth - tip.offsetWidth;
    if (y == null) {
      tip.style.left = `${Math.max(0, px + 10 <= room ? px + 10 : px - 10 - tip.offsetWidth)}px`;
      tip.style.top = `${svg.offsetTop + 2}px`;
    } else {
      tip.style.left = `${Math.max(0, Math.min(room, px - tip.offsetWidth / 2))}px`;
      tip.style.top = `${svg.offsetTop + y / H * box.height - tip.offsetHeight - 10}px`;
    }
  };
  hit.addEventListener('pointermove', (event) => {
    const box = svg.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width * W;
    const t = times.reduce((best, time) => Math.abs(X(time) - x) < Math.abs(X(best) - x) ? time : best);
    cursor.setAttribute('x1', X(t)); cursor.setAttribute('x2', X(t)); cursor.style.opacity = 1;
    const rows = series.filter((s) => s.points[0].time <= t && t <= s.end).map((s) => {
      const p = frontAt(s.points, t);
      const row = el('div');
      row.append(lineKey(s), el('span', 'v', spec.value(p.v)),
        el('span', 'd', ` · ${s.score.name} · ${runName(p.run)} ${p.checkpoint === '000' ? 'at birth' : p.checkpoint}` +
          (p.time < t ? ` · since ${hhmm(p.time)}` : '')));
      return row;
    });
    tip.classList.remove('hx-wide');
    tip.replaceChildren(el('strong', '', dayTime(t)), ...rows);
    tip.style.opacity = 1;
    place(X(t));
  });
  hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; cursor.style.opacity = 0; });

  // the breakthroughs: a numbered mark each, with what changed on hover or focus
  const groups = placed.map(({b, index, x, lane}) => {
    const y = P.t + plotH + 30 + lane * 17;
    const g = sv('g', {class: 'hx-mark', tabindex: 0, role: 'img', 'aria-label': `${index + 1}: ${b.title}. ${b.before} to ${b.after}`});
    g.append(sv('circle', {cx: x, cy: y, r: 7}));
    const number = sv('text', {x, y: y + 3});
    number.textContent = String(index + 1);
    g.append(number);
    svg.append(g);
    return {g, x, y, b, index};
  });
  const mark = (index, on) => {
    const item = groups.find((g) => g.index === index);
    if (!item) return;
    item.g.classList.toggle('on', on);
    guides[groups.indexOf(item)].classList.toggle('on', on);
    if (!on) { tip.style.opacity = 0; return; }
    const {b} = item;
    tip.classList.add('hx-wide');
    tip.replaceChildren(el('strong', '', `${index + 1} · ${dayTime(Date.parse(b.time))}`), el('div', 'v', b.title),
      el('div', '', `${b.before} → ${b.after}`), el('div', 'd', b.change), el('div', 'd', `Source: ${b.plan}`));
    tip.style.opacity = 1;
    place(item.x, item.y - 8);
  };
  for (const item of groups) {
    for (const [type, on] of [['pointerenter', true], ['focus', true], ['pointerleave', false], ['blur', false]]) {
      item.g.addEventListener(type, () => { mark(item.index, on); wrap.dispatchEvent(new CustomEvent('hx-mark', {detail: {index: item.index, on}})); });
    }
  }
  wrap.append(svg, tip);
  return {wrap, mark};
}

export class Stats {
  constructor({ panel, button, onPlay }) {
    this.panel = panel;
    this.button = button;
    this.onPlay = onPlay;
    this.data = {brains: []};
    this.layouts = new Set();      // the candidates whose brain layout is unfolded
    this.comparePart = '';         // the part the matched comparison shows
    this.compareTag = '';
    this.rankBy = 'train';         // the ranking's order: the lowest training error, or the lowest test error
    this.rankSlots = null;         // the layout the ranking shows, by its number of motor slots (null: the commonest)
    this.rankAll = false;          // every brain of the layout, or the first twelve
    // only the runs training on a box now (owner, 26 Nov 09:45); kept in the browser
    this.trainingOnly = (() => { try { return localStorage.getItem('statsTrainingOnly') === '1'; } catch { return false; } })();
    this.part = '';                // the part of the next-bar error the selected candidate's curve shows
    // the training history at the panel's top: open unless folded (kept in the browser)
    this.historyOpen = (() => { try { return localStorage.getItem('statsHistoryOpen') !== '0'; } catch { return true; } })();
    button.onclick = () => this.open();
    // A reload of the page comes back to the open panel, the chosen candidate and the place in it.
    addEventListener('pagehide', () => {
      if (!this.isOpen()) return sessionStorage.removeItem(KEPT);
      sessionStorage.setItem(KEPT, JSON.stringify({
        scroll: this.panel.scrollTop, layouts: [...this.layouts], part: this.part, comparePart: this.comparePart,
        compareTag: this.compareTag, rankBy: this.rankBy, rankSlots: this.rankSlots, rankAll: this.rankAll,
        candidate: this.shown || '', checkpoint: this.checkpoint || ''}));
    });
    // While a list is being chosen from, a live update waits: it would close the open list under the pointer.
    for (const type of ['pointerdown', 'keydown']) {
      panel.addEventListener(type, (e) => { if (e.target.tagName === 'SELECT') this.heldUntil = Date.now() + 8000; }, true);
    }
    panel.addEventListener('change', () => { this.heldUntil = 0; }, true);
    panel.addEventListener('mousedown', (e) => {
      // a press on the panel's own scrollbar is not a click on the backdrop
      this.pressed = e.target === panel && e.offsetX < panel.clientWidth && e.offsetY < panel.clientHeight;
    });
    panel.addEventListener('click', (e) => { if (e.target === panel && this.pressed) this.close(); this.pressed = false; });
    addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) { e.stopPropagation(); this.close(); }
    }, true);
  }

  isOpen() { return this.panel.classList.contains('open'); }

  // The address names the open panel and its candidate, so a reload returns to them.
  remember(candidate) {
    const url = new URL(location.href);
    if (candidate == null) url.searchParams.delete('stats'); else url.searchParams.set('stats', candidate);
    if (url.href !== location.href) history.replaceState(null, '', url);
  }

  close() {
    this.disconnect();
    clearInterval(this.historyTimer);
    this.historyTimer = null;
    this.remember(null);
    this.panel.classList.remove('open');
    this.button.focus({preventScroll: true});
  }

  open(checkpoint) {
    this.disconnect();
    this.target = checkpoint;
    this.candidate = null;
    this.checkpoint = null;
    this.error = null;
    try {
      const kept = JSON.parse(sessionStorage.getItem(KEPT) || 'null');
      sessionStorage.removeItem(KEPT);
      if (kept && (kept.candidate || '') === (checkpoint || '')) {      // the same panel as before the reload
        this.layouts = new Set(kept.layouts);
        Object.assign(this, {part: kept.part, comparePart: kept.comparePart, compareTag: kept.compareTag,
          rankBy: kept.rankBy, rankSlots: kept.rankSlots, rankAll: kept.rankAll, checkpoint: kept.checkpoint || null});
        this.scrollBack = kept.scroll;
      }
    } catch { /* a reload without kept state opens the panel at its top */ }
    this.connection = 'Connecting…';
    this.panel.classList.add('open');
    this.render();
    this.loadHistory();
    clearInterval(this.historyTimer);
    this.historyTimer = setInterval(() => this.loadHistory(), HISTORY_EVERY);
    if (!window.EventSource) {
      this.poll();
      return;
    }
    const stream = new EventSource('/api/curves/events');
    this.stream = stream;
    stream.onmessage = (event) => {
      if (this.stream !== stream) return;
      clearInterval(this.timer);
      this.timer = null;
      this.request?.abort();
      this.setConnection('Live');
      this.receive(JSON.parse(event.data));
    };
    stream.onerror = () => {
      if (this.stream !== stream) return;
      this.setConnection('Reconnecting…');
      this.poll();
    };
  }

  disconnect() {
    this.stream?.close();
    this.stream = null;
    this.request?.abort();
    this.request = null;
    clearInterval(this.timer);
    this.timer = null;
  }

  // When the boxes were last looked at. Older than five minutes means the mirror is not running.
  mirrorText() {
    const at = this.mirror?.at;
    if (!at) return '';
    return Date.now() / 1000 - at < 300 ? `AWS checked ${clock(at)}` : `AWS not checked since ${clock(at)}`;
  }

  setConnection(state) {
    this.connection = state;
    const label = this.panel.querySelector('.stats-connection');
    if (label && label.textContent !== state) label.textContent = state;
  }

  poll() {
    if (this.timer) return;
    this.refresh();
    this.timer = setInterval(() => this.refresh(), 10000);
  }

  async refresh() {
    if (this.request) return;
    const request = new AbortController();
    this.request = request;
    const timeout = setTimeout(() => request.abort(), 5000);
    try {
      const response = await fetch('/api/curves', {cache: 'no-store', signal: request.signal});
      if (!response.ok) throw Error(response.status);
      const data = await response.json();
      if (this.request === request && !request.signal.aborted) {
        this.setConnection('Polling');
        this.receive(data);
      }
    } catch {
      if (this.request === request && this.timer && !this.error) {
        this.error = 'Could not refresh training stats.';
        this.render(true);
      }
    } finally {
      clearTimeout(timeout);
      if (this.request === request) this.request = null;
    }
  }

  receive(data) {
    // the mirror's clock ticks every minute: it updates its own label and is no reason to draw the panel again
    const signature = JSON.stringify(data, (key, value) => key === 'checked_at' || key === 'mirror' ? undefined : value);
    this.mirror = data.mirror;
    const label = this.panel.querySelector('.stats-mirror');
    if (label) label.textContent = this.mirrorText();
    if (signature !== this.signature || this.error) {
      this.data = data;
      this.signature = signature;
      this.error = null;
      this.updated = new Date();
      const wait = (this.heldUntil || 0) - Date.now();
      clearTimeout(this.later);
      if (wait > 0) this.later = setTimeout(() => { if (this.isOpen()) this.render(true); }, wait + 50);
      else this.render(true);
    }
    if (data.brains?.some((brain) => brain.kind === 'legacy') && !this.listen && !this.listeningRequest) {
      this.listeningRequest = fetch('/api/listen')
        .then((response) => response.ok ? response.json() : null)
        .then((result) => {
          if (result) {
            this.listen = result;
            if (this.isOpen()) this.render(true);
          }
        }).catch(() => {}).finally(() => { this.listeningRequest = null; });
    }
  }

  render(keepScroll = false) {
    const box = this.panel;
    const panelScroll = keepScroll ? box.scrollTop : 0;
    const scroll = keepScroll ? box.querySelector('.ed')?.scrollTop : null;
    const focus = box.contains(document.activeElement) ? document.activeElement.id : '';
    const card = el('div', 'ed');
    const head = el('div', 'edh');
    head.append(el('div', 'edt', 'Training stats'));
    const connection = el('span', 'stats-connection', this.connection);
    connection.setAttribute('role', 'status');
    head.append(connection);
    head.append(el('span', 'stats-connection stats-mirror', this.mirrorText()));
    if (this.updated) {
      head.append(el('span', 'stats-connection', `last change ${this.updated.toLocaleTimeString([], {hour12: false})}`));
    }
    const close = el('button', 'edx', '✕');
    close.id = 'closeStats';
    close.setAttribute('aria-label', 'Close training stats');
    close.onclick = () => this.close();
    head.append(close);
    card.append(head);
    if (this.error) card.append(el('div', 'edn', this.error));

    const all = this.data.brains || [];
    const brains = this.trainingOnly ? all.filter((b) => b.training || b.candidate === this.candidate) : all;
    const brain = brains.find((b) => b.candidate === this.candidate) ||
                  brains.find((b) => b.candidate === this.target) ||
                  brains.find((b) => b.points.some((p) => p.name === this.target));
    this.shown = brain?.candidate || '';
    if (this.isOpen() && (brains.length || !this.target)) this.remember(this.shown);
    if (!brain) this.historySection(card);
    if (!brain) this.slate(card, this.data.slate, brains);
    if (all.length) {
      const controls = el('div', 'stats-controls');
      controls.append(select('Candidate', 'statsBrain',
        [['', 'All candidates'], ...brains.map((b) => [b.candidate, b.candidate])], brain?.candidate || '', (value) => {
          this.candidate = value; this.target = null; this.checkpoint = null; this.render();
        }));
      const live = all.filter((b) => b.training).length;
      const only = el('button', 'stats-load', this.trainingOnly ? `Show all ${all.length} brains` : `In training only: ${live} runs, all layouts`);
      only.id = 'statsTrainingOnly';
      only.setAttribute('aria-pressed', String(this.trainingOnly));
      only.title = this.trainingOnly ? 'Show every brain again' : 'Show only the runs that train on a box now';
      only.onclick = () => {
        this.trainingOnly = !this.trainingOnly;
        try { localStorage.setItem('statsTrainingOnly', this.trainingOnly ? '1' : '0'); } catch { /* private window */ }
        this.render(true);
      };
      controls.append(only);
      card.append(controls);
      if (brain) this.details(card, brain);
      else {
        this.ranking(card, brains);
        this.compare(card, brains);
        const order = (candidate) => candidate.training ? 0 : 1;
        for (const candidate of [...brains].sort((a, b) => order(a) - order(b))) this.overview(card, candidate);
      }
      if (this.listen) {
        const detail = el('details', 'stats-listening');
        detail.open = Boolean(box.querySelector('.stats-listening:not(.stats-attention)')?.open);
        detail.append(el('summary', '', 'Listening checks'));
        this.listening(detail);
        card.append(detail);
      }
    } else {
      card.append(el('div', 'edn', !this.signature ? 'Loading…'
        : 'No run yet. A layout is trained once it passed its checks, and its curve appears here with its first evaluation.'));
    }
    box.replaceChildren(card);
    box.scrollTop = panelScroll;
    if (this.scrollBack != null && brains.length) {      // the place before a reload of the page
      box.scrollTop = this.scrollBack;
      this.scrollBack = null;
    }
    if (scroll != null) card.scrollTop = scroll;
    if (focus) document.getElementById(focus)?.focus({preventScroll: true});
    else if (!keepScroll) close.focus({preventScroll: true});      // only when the panel opens or its view changes
  }

  measured(brain) {
    return brain.points.filter((p) => Number.isFinite(p.train) && Number.isFinite(p.test));
  }

  // The brain's layout, folded until asked for. `point` supplies the errors written into the outputs.
  layout(brain, point) {
    if (!brain.layout) return null;
    const detail = el('details', 'stats-layout');
    detail.open = this.layouts.has(brain.candidate);
    detail.append(el('summary', '', 'Brain layout'));
    const draw = () => {
      if (detail.childElementCount > 1) return;
      const legend = el('div', 'lg');
      legend.append(el('span', 'k dr'), document.createTextNode('drives'), el('span', 'k fb'),
        document.createTextNode(brain.layout.apical
          ? `feedback, at most ${Math.round(brain.layout.apical * 100)}% of the drive it joins`
          : 'feedback'));
      detail.append(el('div', 'edn', layoutFacts(brain.layout)), legend, layoutDrawing(brain.layout, point));
      if (point && !point.initial) detail.append(el('div', 'edn', `Errors in the outputs: ${point.name}.`));
    };
    if (detail.open) draw();
    detail.addEventListener('toggle', () => {
      if (detail.open) { this.layouts.add(brain.candidate); draw(); } else this.layouts.delete(brain.candidate);
    });
    return detail;
  }

  // ▶ beside a brain: its newest checkpoint goes into the player, the devices come into view and it composes.
  playButton(brain, compact = false) {
    const button = el('button', 'stats-load', compact ? '▶' : '▶ Play');
    button.id = `stats-play-${brain.candidate}`;
    button.disabled = !this.onPlay || !brain.points.some((point) => !point.initial);
    button.title = button.disabled ? 'No checkpoint yet' : 'Load its newest checkpoint into the player and let it compose';
    button.setAttribute('aria-label', `Play ${brain.candidate}`);
    button.onclick = () => {
      this.close();
      this.onPlay(brain.candidate);
    };
    return button;
  }

  // DJ beside ▶ (owner, 26 Nov 08:35): the Auto DJ composes with this run's newest checkpoint from its next track;
  // it replaces the brain chosen in the mixer's "brain" field.
  djButton(brain, compact = false) {
    const button = el('button', 'stats-load stats-dj', compact ? 'DJ' : 'DJ · Auto DJ');
    button.disabled = !brain.points.some((point) => !point.initial);
    button.title = button.disabled ? 'No checkpoint yet' : 'Give this run to the Auto DJ: its newest checkpoint composes the next tracks';
    button.setAttribute('aria-label', `Use ${brain.candidate} in the Auto DJ`);
    button.onclick = async () => {
      const id = `candidate:${brain.candidate}`, label = button.textContent;
      const send = async (model, on) => (await fetch('/api/autodj/approved', {method: 'POST', body: JSON.stringify({model, on})})).json();
      button.disabled = true;
      try {
        let j = await send(id, true);
        for (const other of j.approved || []) if (other !== id) j = await send(other, false);
        button.textContent = compact ? 'DJ ✓' : 'DJ ✓ next track';
        button.title = 'The Auto DJ composes with this run from its next track';
        location.href = '/mixer?radio=1';      // owner, 26 Nov 10:07: straight to the mixer with the Auto DJ mixing this brain
      } catch {
        button.textContent = compact ? 'DJ ✕' : 'DJ ✕ no answer';
      }
      setTimeout(() => { button.textContent = label; button.disabled = false; }, 4000);
    };
    return button;
  }

  // A run of the slate is named by its layout; any other run by the wiring its layout drawing shows.
  titleOf(brain, fallback) {
    const entry = (this.data.slate || []).find((item) => item.layout === brain.slate);
    return entry ? entry.title : wiringName(brain, fallback);
  }

  // The layouts under comparison: what each is, which checks it passed, and how far its runs have come.
  slate(card, slate, brains) {
    if (!slate?.length || !brains.length) return;       // clean start: the layouts show once a generation-2 run reports
    const section = el('section', 'stats-latest stats-slate');
    const title = el('h2', 'lh', 'Layouts and checks');
    title.id = 'statsSlateHeading';
    section.setAttribute('aria-labelledby', title.id);
    section.append(title);
    section.append(el('div', 'edn', 'Generation 2 brains have no route from memory to the motor: the coordinator writes the ' +
      'arrangement and reads the four recall heads, and the pathways learn every bar from the motor labels alone, taught before ' +
      'the act records the moment. A layout trains after three checks on its code. Builds: it builds and reloads on Cadence ' +
      '0.104.0 with no route. Listens: it learns real v5f passages below the commonest-bar level within eight lesson batches. ' +
      'Bench, holds: a quarter-size copy learns to hold a bench pattern at least a quarter of the way from the commonest bar to ' +
      'repeating. Bench, rule: how far it learns a rule on top of holding, a reading that does not block a layout. Furthest is ' +
      'the best level a run of the layout has reached, with its way from the commonest bar (0%) to repeating the bar heard (100%).'));
    const table = el('table', 'lt');
    const header = el('tr');
    for (const text of ['Layout', 'Pathways', 'Builds', 'Bench: holds', 'Bench: rule', 'Listens', 'Runs', 'Furthest', 'Way']) {
      const cell = el('th', '', text);
      cell.scope = 'col';
      header.append(cell);
    }
    for (const index of [2, 3, 4, 5, 7]) header.children[index].style.textAlign = 'left';
    table.append(header);
    const cell = (node) => { const td = el('td', 'nm'); td.style.maxWidth = 'none'; td.append(node); return td; };
    const stage = (passed, good, bad) => passed == null ? mark('none', '–', 'not run')
      : passed ? mark('good', '✓', good) : mark('bad', '✕', 'failed', bad);
    const task = (bench, name) => {
      const reading = bench?.[name];
      if (!reading) return bench?.passed === false ? mark('bad', '✕', 'failed', (bench.problems || [])[0]) : mark('none', '–', 'not run');
      const passed = reading.error <= reading.bar;
      const facts = `Error ${pct(reading.error)} after ${count(reading.lessons)} lessons. The commonest answer gives ${pct(reading.commonest)}`;
      if (name === 'hold') {
        return mark(passed ? 'good' : 'bad', passed ? '✓' : '✕', pct(reading.error),
          `${facts}, repeating the bar ${pct(reading.repeat)}. Passes at ${pct(reading.bar)} or lower.`);
      }
      // the rule is a reading, not a gate: learned, partly or not learned
      const partly = reading.reading === 'partly';
      return mark(passed ? 'good' : partly ? 'wait' : 'bad', passed ? '✓' : partly ? '◐' : '✕',
        `${pct(reading.error)}${passed ? '' : partly ? ' · partly' : ' · not learned'}`,
        `${facts}, and repeating the bar is always wrong here. Learned at ${pct(reading.bar)} or lower.`);
    };
    for (const item of slate) {
      const runs = brains.filter((brain) => brain.slate === item.layout);
      const levels = runs.map((brain) => ({brain, level: levelOf(brain, this.measured(brain))})).filter((row) => row.level);
      const best = levels.sort((a, b) => a.level.low - b.level.low)[0];
      const build = item.stages.build, bench = item.stages.bench, listen = item.stages.listen;
      const name = el('td', 'nm');
      name.style.maxWidth = 'none';
      name.style.whiteSpace = 'normal';
      name.style.minWidth = '210px';
      name.append(document.createTextNode(item.title), el('div', 'edn', item.wiring));
      name.lastChild.style.margin = '1px 0 0';
      for (const [label, failed] of [['Build', build], ['Bench', bench], ['Listening', listen]]) {
        if (failed?.passed === false && failed.problems?.length) {      // why a stage failed, in words
          name.append(el('div', 'edn', `${label} failed: ${failed.problems[0]}`));
          name.lastChild.style.margin = '1px 0 0';
          name.lastChild.style.color = 'var(--mid)';
        }
      }
      let heard;
      if (best && best.level.key !== 'deaf') {
        heard = mark('good', '✓', `in training · ${pct(best.level.low)}`, `${best.brain.candidate}: lowest test error ${pct(best.level.low)}, commonest bar ${pct(best.level.blind)}`);
      } else if (listen) {
        const last = listen.curve.at(-1), level = listen.levels?.commonest?.test;
        const text = last ? `${pct(last.test)} after ${last.batches} lesson batches` : 'started';
        const tip = `On this computer. Test error ${last ? pct(last.test) : '–'}; commonest bar ${pct(level)}; repeating ${pct(listen.levels?.repeat?.test)}.`;
        heard = listen.passed ? mark('good', '✓', text, tip) : listen.passed === false && !listen.running
          ? mark('bad', '✕', listen.problems?.length ? 'failed' : text, tip) : mark('wait', '…', text, tip);
      } else if (best) {
        heard = levelMark(best.level, Boolean(best.brain.training));
      } else {
        heard = mark('none', '–', 'with its first run');
      }
      const training = runs.filter((brain) => brain.training).length;
      const row = el('tr');
      row.append(name, el('td', 'n', String(item.pathways)),
        cell(stage(build?.passed, build?.neurons ? `${count(build.neurons)} neurons` : 'passed', (build?.problems || [])[0])),
        cell(task(bench, 'hold')), cell(task(bench, 'rule')), cell(heard),
        el('td', 'n', runs.length ? `${runs.length}${training ? ` · ${training} training` : ''}` : '–'),
        cell(best ? levelMark(best.level, Boolean(best.brain.training)) : mark('none', '–', 'no run')),
        el('td', 'n', wayText(best?.level)));
      table.append(row);
    }
    const scroll = el('div', 'stats-latest-table');
    scroll.append(table);
    section.append(scroll);
    card.append(section);
  }

  // The training history file; the panel is drawn again only when the file changed or could not be read.
  async loadHistory() {
    if (this.historyRequest) return;
    this.historyRequest = true;
    const before = `${this.historyState}:${this.history?.generated}`;
    try {
      const response = await fetch('/history.json', {cache: 'no-store'});
      if (response.status === 404) {      // the player runs without the route yet: asked again when the panel opens again
        this.historyState = 'missing';
        clearInterval(this.historyTimer);
        this.historyTimer = null;
      } else if (!response.ok) throw Error(response.status);
      else { this.history = await response.json(); this.historyState = 'ok'; }
    } catch {
      this.historyState = 'error';
    } finally {
      this.historyRequest = false;
    }
    if (this.isOpen() && `${this.historyState}:${this.history?.generated}` !== before) this.render(true);
  }

  // Training history: per score the best gap over repeating so far, the curated breakthroughs as numbered marks and a
  // list, and two small charts of the lowest held-out error and the best synth note F1 so far.
  historySection(card) {
    const section = el('details', 'stats-history');
    section.open = this.historyOpen;
    section.addEventListener('toggle', () => {
      this.historyOpen = section.open;
      try { localStorage.setItem('statsHistoryOpen', section.open ? '1' : '0'); } catch { /* private window */ }
    });
    const summary = el('summary', 'lh', 'Training history');
    summary.id = 'statsHistoryHeading';
    section.append(summary);
    card.append(section);
    const h = this.history;
    if (!h) {
      section.append(el('div', 'edn', this.historyState === 'missing' ? 'Training history appears after the player\'s next restart.'
        : this.historyState === 'error' ? 'The training history could not be read.' : 'Loading the training history…'));
      return;
    }
    if (!h.scores.length) {            // generation 2, before its first evaluation: nothing to draw yet
      section.append(el('div', 'edn', 'No readings yet. The history starts with the first evaluation of the generation-2 brains ' +
        '(clean start of 26 November, Cadence 0.104.0).'));
      return;
    }
    const ahead = h.scores.length ? Math.min(...h.scores.map((s) => Date.parse(s.first))) : null;
    section.append(el('div', 'edn', `Every run's held-out readings since the first generation-2 brains${ahead ? ` (${dayTime(ahead)})` : ''}, ` +
      'grouped by the score they learn; the four-passage memorisation probes are left out. ' +
      'The gap over repeating is how many points a brain\'s held-out error lies under the error of repeating the bar it heard, ' +
      'read on the same passages: above zero the brain writes changes before they come. It is the one number that compares ' +
      'across scores. ' + `${count(h.readings)} readings of ${h.runs} runs, as of ${hhmm(Date.parse(h.generated))}; rebuilt every ten minutes.`));

    const marks = h.breakthroughs || [];
    section.append(el('div', 'hx-title', 'Best gap over repeating the bar so far, per score (points, higher is better)'));
    const gap = historyChart(h, 'gap', {W: 720, plotH: 190, P: {l: 40, r: 58, t: 12}, xticks: 10, yticks: 4, marks,
      name: 'Best gap over repeating the bar per score over time, with the breakthroughs as numbered marks'});
    section.append(gap.wrap);
    if (marks.length) {
      section.append(el('div', 'edn', 'The numbered marks under the time axis are the breakthroughs below. Hover a mark, a line or a row for the numbers.'));
      const list = el('ol', 'hx-list');
      list.setAttribute('aria-label', 'Breakthroughs in training');
      const rows = marks.map((b, index) => {
        const row = el('li');
        row.append(el('span', 'hx-no', String(index + 1)), el('span', 'hx-when', dayTime(Date.parse(b.time))),
          el('span', 'hx-what', b.title), el('span', 'hx-ba', `${b.before} → ${b.after}`));
        row.addEventListener('pointerenter', () => { gap.mark(index, true); row.classList.add('on'); });
        row.addEventListener('pointerleave', () => { gap.mark(index, false); row.classList.remove('on'); });
        list.append(row);
        return row;
      });
      gap.wrap.addEventListener('hx-mark', (event) => rows[event.detail.index]?.classList.toggle('on', event.detail.on));
      section.append(list);
    }

    const pair = el('div', 'hx-pair');
    for (const [measure, title, note] of [
      ['low', 'Lowest held-out error so far (lower is better)',
       'Each score has its own held-out passages and labels: errors of different scores do not compare.'],
      ['notes', 'Best synth note F1 so far (higher is better)',
       'Pitched synth onsets on each score\'s own held-out passages: F1 of different scores does not compare.']]) {
      const box = el('div');
      box.append(el('div', 'hx-title', title), el('div', 'edn', note),
        historyChart(h, measure, {W: 360, plotH: 120, P: {l: 34, r: 62, t: 12}, xticks: 5, yticks: 4, name: title}).wrap);
      pair.append(box);
    }
    section.append(pair);

    // the same in numbers: one row per score
    const table = el('table', 'lt');
    const header = el('tr');
    for (const text of ['Score', 'Training bars', 'Runs', 'Training now', 'Readings', 'From', 'To', 'Best gap', 'Lowest held-out error',
                        'Best synth note F1', 'Best drum hit F1']) {
      const cell = el('th', '', text);
      cell.scope = 'col';
      header.append(cell);
    }
    table.append(header);
    const keys = h.scores.map((s) => s.key);
    const best = (s, key, format) => {
      const p = s[key]?.at(-1);
      const cell = el('td', 'n', p ? format(p.v) : '–');
      if (p) cell.title = `${runName(p.run)} ${p.checkpoint === '000' ? 'at birth' : p.checkpoint} · ${dayTime(Date.parse(p.t))}`;
      return cell;
    };
    for (const s of h.scores) {
      const row = el('tr');
      const name = el('td', 'nm');
      const key = lineKey(scoreStyle(s.key, keys));
      key.style.marginLeft = '0';
      name.append(key, document.createTextNode(s.name));
      row.append(name, el('td', 'n', s.training_bars ? count(s.training_bars) : '–'), el('td', 'n', String(s.runs)),
        el('td', 'n', String(s.training?.length || 0)), el('td', 'n', String(s.readings)),
        el('td', 'n', dayTime(Date.parse(s.first))), el('td', 'n', dayTime(Date.parse(s.last))),
        best(s, 'gap', (v) => signed(v, 2)), best(s, 'low', (v) => `${v.toFixed(2)}%`),
        best(s, 'notes', (v) => `${v.toFixed(1)}%`), best(s, 'drums', (v) => `${v.toFixed(1)}%`));
      table.append(row);
    }
    const scroll = el('div', 'stats-latest-table');
    scroll.append(table);
    section.append(scroll);
  }

  // The running ranking: every brain of one layout, ordered by its lowest training error (or test error),
  // with both errors side by side and how long ago each brain last set a new low.
  ranking(card, brains) {
    const measured = brains.filter((brain) => brain.kind === 'composer' && this.measured(brain).length);
    if (measured.length < 2) return;
    const sizes = new Map();
    for (const brain of measured) sizes.set(slotsOf(brain), (sizes.get(slotsOf(brain)) || 0) + 1);
    const common = [...sizes.entries()].sort((a, b) => b[1] - a[1] || b[0] - a[0])[0][0];
    const slots = this.rankSlots != null && sizes.has(this.rankSlots) ? this.rankSlots : common;
    const by = this.rankBy;
    const rows = measured.filter((brain) => slotsOf(brain) === slots).map((brain) => {
      const points = this.measured(brain);
      const last = points.at(-1), heard = brain.lessons, epochs = epochsOf(last, heard);
      // how often the brain has heard its training set: the bars of its lessons over the bars of the set
      const passes = heard?.training_bars && epochs != null ? epochs * heard.epoch_windows * heard.window / heard.training_bars : null;
      return {brain, points, train: lows(points, 'train'), test: lows(points, 'test'), last, epochs, passes,
        level: levelOf(brain, points)};
    }).sort((a, b) => a[by].best[by] - b[by].best[by] || a.brain.candidate.localeCompare(b.brain.candidate));
    if (!rows.length) return;

    const section = el('section', 'stats-latest stats-ranking');
    const title = el('h2', 'lh', 'Ranking');
    title.id = 'statsRankingHeading';
    section.setAttribute('aria-labelledby', title.id);
    section.append(title);
    const controls = el('div', 'stats-controls');
    controls.append(select('Rank by', 'statsRankBy', RANKS, by, (value) => { this.rankBy = value; this.render(true); }));
    if (sizes.size > 1) {
      controls.append(select('Layout', 'statsRankSlots',
        [...sizes.entries()].sort((a, b) => b[0] - a[0]).map(([n, k]) => [String(n), n ? `${n} motor slots · ${k} brains` : `Unknown layout · ${k} brains`]),
        String(slots), (value) => { this.rankSlots = Number(value); this.render(true); }));
    }
    section.append(controls);
    const repeat = rows.map((row) => row.last.copy_baseline).find((baseline) => Number.isFinite(baseline?.train?.balanced));
    const blind = rows.map((row) => row.brain.blind).find((baseline) => Number.isFinite(baseline?.train?.balanced));
    section.append(el('div', 'edn', 'One row per brain, the lowest first. Brains of one layout write the same slots, so their errors compare. ' +
      `A check is one evaluation. A new low is ${by === 'train' ? 'a training' : 'a test'} error at least 0.2 points under the lowest before it. ` +
      'Time is the training time. Epochs are the trainer\'s rounds. Set heard is the bars of all lessons so far over the bars ' +
      'of the training set, so it compares runs whose rounds differ in size. Since low counts the checks since the last new low. ' +
      'Drum hits and synth notes are F1 scores on the held-out passages: how many of the reference\'s drum hits and pitched ' +
      'synth notes the brain writes, and how many of its own are in the reference. Plays like the corpus counts the checks ' +
      'its own free playing passes against the corpus: how often each part changes and how long sounds hold. ' +
      'Coordinator credit is the coordinator\'s teaching contrast as a share of the motor\'s (Cadence\'s credit report); "warned" marks ' +
      'a region whose share stays at or under a thousandth. Coordinator cut is the held-out error with the coordinator cut minus ' +
      'the error intact, in points: positive means the coordinator helps.' +
      (repeat ? ` Repeating the previous bar gives ${pct(repeat.train.balanced)} training and ${pct(repeat.test.balanced)} test error.` : '') +
      (blind ? ` Writing the commonest bar whatever was heard gives ${pct(blind.train.balanced)} training and ${pct(blind.test.balanced)} test error: a brain above that level has not learned to use what it hears.` : '')));
    const table = el('table', 'lt rank');
    const header = el('tr');
    for (const text of ['#', 'Brain', 'Training', 'Wiring', 'Level', 'Way', 'Lowest training', 'Lowest test', 'Drum hits F1',
                        'Synth notes F1', 'Plays like the corpus', 'Coordinator credit', 'Coordinator cut', 'Latest train · test', 'Time', 'Epochs', 'Set heard', 'Since low', 'Listen']) {
      const cell = el('th', '', text);
      cell.scope = 'col';
      header.append(cell);
    }
    for (const index of [1, 2, 3, 4]) header.children[index].style.textAlign = 'left';
    table.append(header);
    const shown = this.rankAll ? rows : rows.slice(0, 12);
    shown.forEach((row, index) => {
      const tr = el('tr');
      const name = el('td', 'stats-latest-name');
      const button = el('button', '', row.brain.candidate);
      button.id = `stats-rank-${row.brain.candidate}`;
      button.onclick = () => { this.candidate = row.brain.candidate; this.target = null; this.checkpoint = null; this.render(); };
      name.append(button);
      const full = this.titleOf(row.brain, '–');
      // short in the table, whole on hover
      const wiring = el('td', 'nm', full.replace('Instrument pathways', 'Instruments').replace('Rhythm and melody pathways', 'Rhythm + melody')
        .replace(', direct reads and own recall', ', reads, own recall').replace(', direct reads', ', reads'));
      wiring.title = full;
      wiring.style.maxWidth = 'none';
      wiring.style.color = 'var(--mid)';
      const since = row[by].since;
      const state = since ? String(since) : 'new low';
      const low = (key) => {
        const cell = el('td', 'n', pct(row[key].best[key]));
        if (key === by) cell.style.color = 'var(--fg)';
        cell.title = `${row[key].best.initial ? 'before any lessons' : row[key].best.name} · ` +
          `training ${pct(row[key].best.train)} · test ${pct(row[key].best.test)}`;
        return cell;
      };
      const listen = el('td', 'n');
      listen.append(this.playButton(row.brain, true), this.djButton(row.brain, true));
      const heard = row.brain.lessons;
      const rounds = el('td', 'n', row.epochs == null ? '–' : String(Math.round(row.epochs * 10) / 10));
      const passes = el('td', 'n', row.passes == null ? '–' : `${row.passes.toFixed(1)} ×`);
      if (heard) {
        rounds.title = passes.title = `One epoch: ${heard.epoch_windows} windows of ${heard.window} bars` +
          (heard.training_bars ? `, ${(heard.epoch_windows * heard.window / heard.training_bars).toFixed(2)} times the ${count(heard.training_bars)} training bars` : '');
      }
      const training = el('td', 'nm');
      training.append(trainingMark(row.brain, false));
      const level = el('td', 'nm');
      level.style.maxWidth = 'none';
      level.append(levelMark(row.level, Boolean(row.brain.training)));
      const way = el('td', 'n', wayText(row.level));
      way.title = 'The way from the commonest-bar level (0%) to repeating the bar heard (100%), by the lowest test error';
      // how much of the reference's drum hits and pitched synth notes the latest reading writes (held-out F1)
      const f1 = (key) => {
        const reading = row.last.test_parts?.detail?.[key]?.f1;
        const repeat = row.last.copy_baseline?.test?.detail?.[key]?.f1;
        const cell = el('td', 'n', Number.isFinite(reading) ? pct(reading) : '–');
        cell.title = 'Held-out F1 of the latest reading' + (Number.isFinite(repeat) ? `. Repeating the bar heard gives ${pct(repeat)}` : '');
        return cell;
      };
      // its own playing against the corpus: how many of the free-play checks the latest checkpoint passes
      // (how often each part changes, how long sounds hold), each with its value and the corpus range on hover
      const checks = [...row.points].reverse().find((point) => point.gate_checks?.length)?.gate_checks;
      const plays = el('td', 'n', checks ? `${checks.filter((check) => check.pass).length} of ${checks.length}` : '–');
      if (checks) {
        plays.title = checks.map((check) => `${check.pass ? '✓' : '✕'} ${check.check}: ${check.value}` +
          (Array.isArray(check.range) ? ` (corpus range ${check.range[0]} to ${check.range[1]})` : '')).join('\n');
      }
      // does the coordinator contribute: its teaching contrast as a share of the motor's, and the held-out error
      // with the coordinator cut against intact, in points (positive: the coordinator helps)
      const held = controller(controlled(row.points));
      const credit = el('td', 'n', held?.credit ? creditShare(held.credit.coordinator?.share) : '–');
      if (held?.credit) {
        credit.title = `Teaching contrast as a share of the motor's: coordinator ${creditShare(held.credit.coordinator?.share)}, ` +
          `weakest region ${creditRegion(held.credit.weakest.name)} ${creditShare(held.credit.weakest.share)}`;
        if (held.credit.warned) credit.append(warnBadge(held.credit));
      }
      const cut = el('td', 'n', held?.cut ? cutPoints(held.cut.points).replace(' points', '') : '–');
      if (held?.cut) {
        cut.title = `Held-out error ${pct(held.cut.cut)} with the coordinator cut, ${pct(held.cut.intact)} intact: ${cutVerdict(held.cut)}`;
        cut.style.color = held.cut.points >= 0.1 ? 'var(--calm, #5fd3c6)' : held.cut.points <= -0.1 ? '#f2858b' : '';
      }
      tr.append(el('td', 'n', String(index + 1)), name, training, wiring, level, way, low('train'), low('test'),
        f1('drum_hits'), f1('synth_notes'), plays, credit, cut,
        el('td', 'n', `${pct(row.last.train)} · ${pct(row.last.test)}`),
        el('td', 'n', duration(row.last.minutes)), rounds, passes, el('td', 'n', state), listen);
      table.append(tr);
    });
    const scroll = el('div', 'stats-latest-table');
    scroll.append(table);
    section.append(scroll);
    if (rows.length > 12) {
      const more = el('button', 'stats-load', this.rankAll ? 'Show the first twelve' : `Show all ${rows.length}`);
      more.id = 'statsRankAll';
      more.onclick = () => { this.rankAll = !this.rankAll; this.render(true); };
      section.append(more);
    }
    card.append(section);
  }

  // One chart for a matched comparison: the mean test error of every wiring over its seeds, evaluation by
  // evaluation, with each seed as a thin line behind it and the table of the latest readings below.
  compare(card, brains) {
    const tags = new Map();
    for (const brain of brains) {
      const match = COMPARED.exec(brain.candidate);
      if (!match || brain.kind !== 'composer' || !this.measured(brain).length) continue;
      const wirings = tags.get(match[3]) || tags.set(match[3], new Map()).get(match[3]);
      (wirings.get(match[1]) || wirings.set(match[1], []).get(match[1])).push(brain);
    }
    const known = [...tags.keys()].filter((tag) => tags.get(tag).size > 1);
    if (!known.length) return;
    const tag = known.includes(this.compareTag) ? this.compareTag : known.at(-1);
    // Every layout keeps one place, its place in the slate, so its colour does not change when runs come and go.
    // Five hues are validated on this surface; the sixth to eighth layout take the first hues again with a dashed line.
    const slateOrder = (this.data.slate || []).map((item) => item.layout);
    const place = (runs, stem) => { const at = slateOrder.indexOf(runs[0].slate); return at < 0 ? slateOrder.length + stem.charCodeAt(0) : at; };
    const wirings = [...tags.get(tag).entries()].sort(([a, runsA], [b, runsB]) => place(runsA, a) - place(runsB, b) || a.localeCompare(b))
      .slice(0, 8);
    const members = wirings.flatMap(([, runs]) => runs);
    const parts = partsOf(members.flatMap((brain) => brain.points));
    const part = parts.some(([key]) => key === this.comparePart) ? this.comparePart : '';
    const read = (p) => part ? get(p.test_parts, part) : p.test;
    const series = wirings.map(([stem, runs], index) => {
      const lines = runs.map((brain) => this.measured(brain).map(read).filter(Number.isFinite));
      const length = Math.max(...lines.map((line) => line.length));
      const means = Array.from({length}, (_, i) => mean(lines.filter((line) => i < line.length).map((line) => line[i])));
      const last = lines.filter((line) => line.length).map((line) => line.at(-1));
      const at = slateOrder.indexOf(runs[0].slate) < 0 ? index : slateOrder.indexOf(runs[0].slate);
      return {stem, runs, lines, means, last, color: SERIES[at % SERIES.length], dashed: at >= SERIES.length,
        name: this.titleOf(runs[0], stem)};
    }).filter((item) => item.means.length);
    if (!series.length) return;
    const baseline = get(members.flatMap((brain) => brain.points).find((p) => Number.isFinite(get(p.copy_baseline?.test, part || 'balanced')))?.copy_baseline.test, part || 'balanced');

    const section = el('section', 'stats-candidate stats-compare');
    const title = el('div', 'stats-candidate-head');
    title.append(el('h2', '', `Matched comparison ${tag}`));
    section.append(title);
    const controls = el('div', 'stats-controls');
    if (known.length > 1) {
      controls.append(select('Comparison', 'statsCompareTag', known.map((name) => [name, name]), tag,
        (value) => { this.compareTag = value; this.render(true); }));
    }
    controls.append(select('Curve', 'statsComparePart', [['', 'All parts, averaged'], ...parts], part,
      (value) => { this.comparePart = value; this.render(true); }));
    section.append(controls);
    const shown = parts.find(([key]) => key === part)?.[1];
    section.append(el('div', 'edn', `Test error${shown ? ` of ${shown}` : ''} after the same lessons for every wiring. ` +
      'Thick line: mean over the seeds. Thin lines: single seeds. Lower is better.'));

    const wrap = el('div', 'cc stats-chart');
    const legend = el('div', 'lg');
    for (const item of series) {
      const key = el('span', 'k');
      key.style.background = item.dashed
        ? `repeating-linear-gradient(90deg, ${item.color} 0 4px, transparent 4px 7px)` : item.color;
      legend.append(key, document.createTextNode(item.name));
    }
    if (Number.isFinite(baseline)) legend.append(el('span', 'k bl'), document.createTextNode('Repeat previous bar'));
    const average = get(members.find((brain) => brain.blind)?.blind.test, part || 'balanced');
    // The chart keeps to the range the runs are in. The commonest-bar level is drawn while the runs are near it,
    // and named with its value once they are far below, so the curves are not squeezed against the bottom.
    const drawn = series.flatMap((item) => item.lines.flat());
    const near = Number.isFinite(average) && average <= Math.max(...drawn) + .12;
    if (Number.isFinite(average)) {
      legend.append(el('span', 'k av'), document.createTextNode(near ? 'Always the commonest bar'
        : `Always the commonest bar: ${pct(average)}, above this chart`));
    }
    legend.style.flexWrap = 'wrap';
    wrap.append(legend);
    const W = 720, H = 260, P = {l: 43, r: 20, t: 14, b: 41};
    const steps = Math.max(1, ...series.map((item) => item.means.length - 1));
    const values = series.flatMap((item) => item.lines.flat());
    if (Number.isFinite(baseline)) values.push(baseline);
    if (near) values.push(average);
    const lo = Math.max(0, Math.floor((Math.min(...values) - .01) * 50) / 50);
    const hi = Math.min(1, Math.ceil((Math.max(...values) + .01) * 50) / 50);
    const X = (i) => P.l + i / steps * (W - P.l - P.r);
    const Y = (v) => P.t + (hi - v) / (hi - lo) * (H - P.t - P.b);
    const svg = sv('svg', {viewBox: `0 0 ${W} ${H}`, class: 'cs', role: 'img',
                           'aria-label': `Matched comparison ${tag}: mean test error per wiring over the evaluations`});
    const label = (text, x, y, anchor = 'middle') => {
      const t = sv('text', {x, y, class: 'al', 'text-anchor': anchor});
      t.textContent = text; svg.append(t);
    };
    for (let i = 0; i <= 4; i++) {
      const value = lo + i * (hi - lo) / 4, y = Y(value);
      svg.append(sv('line', {x1: P.l, x2: W - P.r, y1: y, y2: y, class: 'gl'}));
      label(pct(value), P.l - 6, y + 3, 'end');
    }
    const every = Math.max(1, Math.ceil(steps / 8));
    for (let i = 0; i <= steps; i += every) label(i ? String(i) : 'start', X(i), H - P.b + 15);
    label('Evaluation', W / 2, H - 5);
    if (Number.isFinite(baseline)) svg.append(sv('line', {x1: P.l, x2: W - P.r, y1: Y(baseline), y2: Y(baseline), class: 'bl'}));
    if (near) svg.append(sv('line', {x1: P.l, x2: W - P.r, y1: Y(average), y2: Y(average), class: 'av'}));
    const path = (line) => line.map((v, i) => `${i ? 'L' : 'M'}${X(i)},${Y(v)}`).join(' ');
    for (const item of series) {
      for (const line of item.lines) {
        if (line.length > 1) svg.append(sv('path', {d: path(line), fill: 'none', stroke: item.color, 'stroke-width': 1, opacity: .35}));
      }
    }
    for (const item of series) {
      const line = sv('path', {d: path(item.means), fill: 'none', stroke: item.color, 'stroke-width': 2});
      if (item.dashed) line.setAttribute('stroke-dasharray', '7 4');
      svg.append(line);
      item.means.forEach((v, i) => svg.append(sv('circle', {cx: X(i), cy: Y(v), r: 3, fill: item.color, class: 'dt'})));
    }
    const tip = el('div', 'tp');
    const hit = sv('rect', {x: P.l, y: P.t, width: W - P.l - P.r, height: H - P.t - P.b, class: 'hit'});
    const cursor = sv('line', {y1: P.t, y2: H - P.b, class: 'cx'});
    svg.append(cursor, hit);
    hit.style.cursor = 'default';
    hit.addEventListener('pointermove', (event) => {
      const r = svg.getBoundingClientRect();
      const i = Math.max(0, Math.min(steps, Math.round(((event.clientX - r.left) / r.width * W - P.l) / (W - P.l - P.r) * steps)));
      cursor.setAttribute('x1', X(i)); cursor.setAttribute('x2', X(i)); cursor.style.opacity = 1;
      const rows = series.filter((item) => i < item.means.length).map((item) => {
        const row = el('div');
        const key = el('span', 'k');
        key.style.background = item.dashed
        ? `repeating-linear-gradient(90deg, ${item.color} 0 4px, transparent 4px 7px)` : item.color;
        const seeds = item.lines.filter((line) => i < line.length).length;
        row.append(key, document.createTextNode(`${item.name} ${pct(item.means[i])} · ${seeds} ${seeds === 1 ? 'seed' : 'seeds'}`));
        return row;
      });
      tip.replaceChildren(el('strong', '', i ? `Evaluation ${i}` : 'Before any lessons'), ...rows);
      tip.style.left = `${Math.max(0, Math.min(wrap.clientWidth - tip.offsetWidth, X(i) / W * wrap.clientWidth + 8))}px`;
      tip.style.opacity = 1;
    });
    hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; cursor.style.opacity = 0; });
    wrap.append(svg, tip);
    section.append(wrap);

    const table = el('table', 'lt');
    const header = el('tr');
    for (const text of ['Wiring', 'Seeds', 'Trainable synapses', 'Evaluations', 'Before lessons', 'Best mean', 'Latest mean', 'Latest range']) {
      header.append(el('th', '', text));
    }
    table.append(header);
    for (const item of series) {
      const row = el('tr');
      const name = el('td', 'nm');
      const key = el('span', 'k');
      key.style.background = item.dashed
        ? `repeating-linear-gradient(90deg, ${item.color} 0 4px, transparent 4px 7px)` : item.color;
      key.style.marginLeft = '0';
      name.append(key, document.createTextNode(item.name));
      name.style.maxWidth = 'none';
      const synapses = item.runs[0].layout?.synapses;
      row.append(name, el('td', 'n', String(item.lines.length)), el('td', 'n', synapses ? count(synapses) : '–'),
        el('td', 'n', String(Math.min(...item.lines.map((line) => line.length - 1)))),
        el('td', 'n', pct(item.means[0])), el('td', 'n', pct(Math.min(...item.means))), el('td', 'n', pct(mean(item.last))),
        el('td', 'n', `${pct(Math.min(...item.last))} to ${pct(Math.max(...item.last))}`));
      table.append(row);
    }
    if (Number.isFinite(baseline)) {
      const row = el('tr', 'ref');
      row.append(el('td', 'nm', 'Repeat previous bar'), el('td', 'n', ''), el('td', 'n', ''), el('td', 'n', ''),
        el('td', 'n', pct(baseline)), el('td', 'n', pct(baseline)), el('td', 'n', pct(baseline)), el('td', 'n', ''));
      table.append(row);
    }
    section.append(table);
    card.append(section);
  }

  overview(card, brain) {
    const section = el('section', 'stats-candidate');
    const title = el('div', 'stats-candidate-head');
    const name = el('h2', '', brain.candidate);
    name.append(trainingMark(brain, true));
    title.append(name);
    const inspect = el('button', '', 'Inspect checkpoints');
    inspect.onclick = () => { this.candidate = brain.candidate; this.render(); };
    const actions = el('span', 'stats-actions');
    actions.append(this.playButton(brain), this.djButton(brain), inspect);
    title.append(actions);
    section.append(title);
    this.status(section, brain.status);
    const points = this.measured(brain);
    const layout = this.layout(brain, points.at(-1));
    if (layout) section.append(layout);
    if (points.length) {
      section.append(this.chart(brain, points, points.find((p) => p.name === this.data.active)));
      const last = points.at(-1);
      section.append(el('div', 'edn', `Latest evaluation · training ${pct(last.train)} · test ${pct(last.test)}`));
      const notes = music(last);
      if (notes.length) section.append(el('div', 'edn', notes.join(' · ')));
      if (brain.kind === 'composer') section.append(controllerBlock(controlled(points)));
    } else {
      section.append(el('div', 'edn', 'Waiting for the first evaluation.'));
    }
    card.append(section);
  }

  details(card, brain) {
    const options = brain.points;
    const points = this.measured(brain);
    const pinned = options.find((p) => p.name === (this.checkpoint || this.target));
    const selected = pinned || points.at(-1) || options.at(-1);
    if (selected) {
      const controls = el('div', 'stats-controls');
      controls.append(select('Checkpoint', 'statsCheckpoint',
        [['', 'Latest evaluation (live)'], ...options.map((p) => [p.name, p.initial ? 'Before any lessons' : p.name])],
        pinned?.name || '', (name) => { this.checkpoint = name; this.target = null; this.render(); }));
      controls.append(this.playButton(brain), this.djButton(brain));
      card.append(controls);
    }
    card.append(trainingMark(brain, true));
    this.status(card, brain.status);
    this.monitor(card, brain.monitor);
    const composer = brain.kind === 'composer';
    const layout = this.layout(brain, Number.isFinite(selected?.test) ? selected : points.at(-1));
    if (layout) card.append(layout);
    const parts = composer ? partsOf(points) : [];
    const part = parts.some(([key]) => key === this.part) ? this.part : '';
    if (parts.length) {
      const controls = el('div', 'stats-controls');
      controls.append(select('Curve', 'statsPart', [['', 'All parts, averaged'], ...parts], part,
        (value) => { this.part = value; this.render(true); }));
      card.append(controls);
    }
    const shown = parts.find(([key]) => key === part)?.[1];
    card.append(el('div', 'edn', shown ? `Next-bar error of one part: ${shown}. Lower is better.` : composer ?
      'Next-bar error, averaged over arrangement, drum patterns, sounding synth lines and sounding synth controls. Lower is better.' :
      'Next-bar sound error. Keeping and changing count equally. Repeating the previous bar gives 50% error.'));
    if (points.length) {
      card.append(this.chart(brain, points, selected, part));
      if (composer) card.append(controllerBlock(selected && (selected.hidden_credit || selected.coordinator_cut || selected.rollout) ? selected : controlled(points)));
      if (selected) card.append(this.readings(selected, points[0], composer));
    } else {
      card.append(el('div', 'edn', 'Waiting for the first evaluation.'));
    }
  }

  status(card, status) {
    if (!status?.phase) return;
    const content = [status.phase[0].toUpperCase() + status.phase.slice(1)];
    if (status.epoch) content.push(`Epoch ${status.epoch} · batch ${status.batch}/${status.batches}`);
    if (status.last_progress_at) {
      const date = new Date(status.last_progress_at);
      content.push(`Progress reported ${date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}`);
    }
    const line = el('div', 'stats-status', content.join(' · '));
    line.setAttribute('role','status');
    card.append(line);
    if (status.detail) card.append(el('div', 'edn', status.detail));
    if (status.allocation) {
      const {state, epochs, detail} = status.allocation;
      card.append(el('div', 'edn', `Training review · ${state} · ${epochs} full ${epochs === 1 ? 'epoch' : 'epochs'}. ${detail}`));
    }
  }

  monitor(card, monitor) {
    if (!monitor) return;
    const active = monitor.events.filter((event) => ['queued', 'reviewing'].includes(event.status));
    if (active.length) {
      for (const event of active) {
        const prefix = event.status === 'reviewing' ? 'Reviewing' : 'Queued for review';
        card.append(el('div', 'edn', `${prefix} · ${event.message}`));
      }
    }
    if (monitor.groove) {
      card.append(el('div', 'edn', `Groove test · quarter kicks ${pct(monitor.groove.kick)} · simultaneous rewrites ${pct(monitor.groove.rewrites)}`));
    }
    if (monitor.last_review) {
      const heading = monitor.last_review.status === 'reviewed' ? 'Latest callback report' : 'Callback needs attention';
      const detail = el('details');
      detail.append(el('summary', 'edn', heading), el('div', 'edn', monitor.last_review.summary));
      card.append(detail);
    }
  }

  // the listening check: each brain's newest checkpoint composing on its own (3 x 256 bars, read as the player reads it),
  // measured bar by bar beside the real tracks played on the same instrument (brain/listen.py)
  listening(card) {
    const L = this.listen;
    if (!L || !L.brains || !L.brains.length) return;
    card.appendChild(el('div', 'lh', 'listening check'));
    card.appendChild(el('div', 'edn', 'Each brain composes 3 × 256 bars on its own; every bar is rendered and measured, next to the real tracks on the same instrument (first row). Closeness: 100% = like the tracks in all of these. Newest checkpoint per brain; best first.'));
    const t = el('table', 'lt'); card.appendChild(t);
    const head = ['brain', 'closeness', 'drums alone', 'bass · lead · air heard', 'beats / 16 bars', 'bass notes · lines', 'lead notes · lines'];
    const tr = el('tr'); head.forEach((h) => tr.appendChild(el('th', '', h))); t.appendChild(tr);
    const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`), f1 = (v) => (v == null ? '–' : v.toFixed(1));
    const row = (r, label, ref) => {
      const x = el('tr', ref ? 'ref' : '');
      const cells = [label, ref ? '' : pct(r.listen_error == null ? null : 1 - r.listen_error), pct(r.drums_only),
        ['bass', 'lead', 'air'].map((n) => pct(r.heard[n])).join(' · '), f1(r.beat_kinds),
        `${f1(r.bass.onsets)} · ${f1(r.bass.distinct_per_16)}`, `${f1(r.lead.onsets)} · ${f1(r.lead.distinct_per_16)}`];
      cells.forEach((c, i) => { const d = el('td', i ? 'n' : 'nm', c); if (!i) d.title = r.name; x.appendChild(d); });
      t.appendChild(x);
    };
    const ref = (L.refs || {}).ref_streams_r6_z_nd;
    if (ref) row(ref, 'real tracks', true);
    for (const r of L.brains) row(r, r.name, false);
  }
  chart(brain, points, selected, part = '') {
    const composer = brain.kind === 'composer';
    const axis = composer ? 'minutes' : 'epoch';
    const read = (p, key) => part ? get(p[`${key}_parts`], part) : p[key];
    const pts = points.filter((p) => Number.isFinite(p[axis]) && Number.isFinite(read(p, 'train')) && Number.isFinite(read(p, 'test')));
    const wrap = el('div', 'cc stats-chart');
    if (!pts.length) return wrap;
    const W = 720, H = 238, P = {l: 43, r: 20, t: 14, b: 41};
    const maxX = Math.max(composer ? 1 : .1, ...pts.map((p) => p[axis]));
    const repeat = (p) => get(p.copy_baseline?.test, part || 'balanced');
    const baseline = composer ? repeat(pts.find((p) => Number.isFinite(repeat(p))) || {}) : .5;
    const errs = pts.flatMap((p) => [read(p, 'train'), read(p, 'test')]);
    if (Number.isFinite(baseline)) errs.push(baseline);
    // what writing the commonest bar scores on the training bars, whatever was heard
    const average = composer ? get(brain.blind?.train, part || 'balanced') : null;
    const near = Number.isFinite(average) &&
      average <= Math.max(...pts.flatMap((p) => [read(p, 'train'), read(p, 'test')]).filter(Number.isFinite)) + .12;
    if (near) errs.push(average);
    const lo = Math.max(0, Math.floor((Math.min(...errs) - .01) * 50) / 50);
    const hi = Math.min(1, Math.ceil((Math.max(...errs) + .01) * 50) / 50);
    const X = (v) => P.l + v / maxX * (W - P.l - P.r);
    const Y = (v) => P.t + (hi - v) / (hi - lo) * (H - P.t - P.b);
    const legend = el('div','lg');
    for (const [css,text] of [['tr','Training error'], ['te','Test error'], ...(Number.isFinite(baseline) ? [['bl','Repeat previous bar (test)']] : []),
                              ...(Number.isFinite(average) ? [['av', near ? 'Always the commonest bar (training)'
                                : `Always the commonest bar (training): ${pct(average)}, above this chart`]] : [])]) {
      legend.append(el('span', `k ${css}`), document.createTextNode(text));
    }
    wrap.append(legend);
    const svg = sv('svg', {viewBox: `0 0 ${W} ${H}`, class: 'cs', role: 'img',
                           'aria-label': `${brain.candidate} training and test error over time`});
    function label(text, x, y, anchor='middle') {
      const t = sv('text', {x, y, class: 'al', 'text-anchor': anchor});
      t.textContent = text; svg.append(t);
    }
    for (let i=0; i<=4; i++) {
      const value = lo + i*(hi-lo)/4, y = Y(value);
      svg.append(sv('line', {x1:P.l, x2:W-P.r, y1:y, y2:y, class:'gl'}));
      label(pct(value), P.l-6, y+3, 'end');
      label(axisLabel(maxX*i/4, composer), X(maxX*i/4), H-P.b+15, i===4 ? 'end' : 'middle');
    }
    label(composer ? 'Elapsed run time' : 'Passes through training data', W/2, H-5);
    if (Number.isFinite(baseline)) svg.append(sv('line', {x1:P.l, x2:W-P.r, y1:Y(baseline), y2:Y(baseline), class:'bl'}));
    if (near) svg.append(sv('line', {x1:P.l, x2:W-P.r, y1:Y(average), y2:Y(average), class:'av'}));
    if (selected && Number.isFinite(selected[axis])) {
      svg.append(sv('line', {x1:X(selected[axis]), x2:X(selected[axis]), y1:P.t, y2:H-P.b, class:'selected-line'}));
    }
    for (const [key, cls] of [['train','tr'], ['test','te']]) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[axis])},${Y(read(p, key))}`).join(' ');
      svg.append(sv('path', {d, class:`ln ${cls}`}));
      for (const p of pts) {
        const dot = sv('circle', {cx:X(p[axis]), cy:Y(read(p, key)), r:p.name === selected?.name ? 5 : 3, class:`dt ${cls}`});
        const title = sv('title', {});
        title.textContent = `${p.name} · ${pct(read(p, key))} ${key === 'train' ? 'training' : 'test'} error`;
        dot.append(title);
        svg.append(dot);
      }
    }
    const tip = el('div', 'tp');
    const hit = sv('rect', {x:P.l, y:P.t, width:W-P.l-P.r, height:H-P.t-P.b, class:'hit'});
    const cursor = sv('line', {y1:P.t, y2:H-P.b, class:'cx'});
    svg.append(cursor, hit);
    function nearest(event) {
      const r = svg.getBoundingClientRect();
      const x = (event.clientX - r.left) / r.width * W;
      return pts.reduce((best,p) => Math.abs(X(p[axis])-x) < Math.abs(X(best[axis])-x) ? p : best);
    }
    hit.addEventListener('pointermove', (event) => {
      const p = nearest(event), x = X(p[axis]);
      cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); cursor.style.opacity = 1;
      tip.replaceChildren(el('strong', '', p.initial ? 'Before any lessons' : p.name), el('div','',`Training ${pct(read(p, 'train'))} · test ${pct(read(p, 'test'))}`));
      tip.style.left = `${Math.max(0, Math.min(wrap.clientWidth-tip.offsetWidth, x/W*wrap.clientWidth+8))}px`;
      tip.style.opacity = 1;
    });
    hit.addEventListener('pointerleave', () => { tip.style.opacity = 0; cursor.style.opacity = 0; });
    hit.addEventListener('click', (event) => {
      this.candidate = brain.candidate; this.checkpoint = nearest(event).name; this.render();
    });
    wrap.append(svg, tip);
    return wrap;
  }

  readings(point, first, composer) {
    const box = el('div', 'stats-reading');
    box.append(el('div', 'lh', point.initial ? 'Before any lessons' :
      point.starting_checkpoint ? `Starting weights from ${point.starting_checkpoint}` : point.name));
    if (!Number.isFinite(point.train) || !Number.isFinite(point.test)) {
      box.append(el('div', 'edn', 'Evaluation pending.'));
      return box;
    }
    const summary = el('div', 'stats-controls');
    summary.append(el('div', 'stats-number', `Training error ${pct(point.train)} (${delta(point.train,first.train)})`),
                   el('div', 'stats-number', `Test error ${pct(point.test)} (${delta(point.test,first.test)})`));
    box.append(summary, el('div', 'edn', 'Change in percentage points since the first evaluation.'));
    if (composer) {
      const notes = [point.initial ? 'Untrained baseline' : 'Evaluation'];
      if (Number.isFinite(point.eta)) notes.push(`Learning rate ${Number(point.eta.toPrecision(3))}`);
      if (Number.isFinite(point.next_eta) && point.next_eta !== point.eta) notes.push(`Next rate ${Number(point.next_eta.toPrecision(3))}`);
      if (point.gate_pass != null) notes.push(`Music gate ${point.gate_pass ? 'passed' : 'failed'}`);
      if (point.source_checkpoint) notes.push(`Training stats from ${point.source_checkpoint}`);
      if (Number.isFinite(point.playback_review?.detail)) {
        notes.push(`Playback detail ${pct(point.playback_review.detail)} · repeated bar ${pct(point.playback_review.held_detail)}`);
      }
      if (Number.isFinite(point.playback_review?.development)) {
        notes.push(`Playback phrase development ${pct(point.playback_review.development)}`);
      }
      notes.push(...music(point));
      if (point.prompt_protocol === 2) notes.push('Prompts reserve rare musical events');
      box.append(el('div', 'edn', notes.join(' · ')));
      if (point.modules?.length) {
        const architecture = point.architecture === 'rhythm-melody-v1'
          ? [`Rhythm ${point.modules.slice(0, 2).join(' / ')}`,
             `Melody ${point.modules.slice(2, 4).join(' / ')}`,
             `${point.coordinator_depth ?? 1} coordinator layer(s) × ${point.modules[4]}`]
          : [`${point.apical > 0 ? 'Apical' : 'Hidden'} layers ${point.modules.join(' / ')}`];
        if (point.recall_heads > 0) architecture.push(`Recall heads ${point.recall_heads}`);
        if (point.learned_code > 0) architecture.push(`Learned code width ${point.learned_code}`);
        box.append(el('div', 'edn', architecture.join(' · ')));
      }
      const recall = point.recall_probe;
      if (Number.isFinite(recall?.intact) && Number.isFinite(recall.cleared)) {
        const reads = [`Recall intact ${pct(recall.intact)}`, `History cleared ${pct(recall.cleared)}`];
        if (Number.isFinite(recall.muted)) reads.push(`Recall drive muted ${pct(recall.muted)}`);
        box.append(el('div', 'edn', reads.join(' · ')));
      }
      if (point.copy_baseline?.test) {
        box.append(el('div', 'edn',
          `Repeating the previous bar gives ${pct(point.copy_baseline.train.balanced)} training error and ${pct(point.copy_baseline.test.balanced)} test error.`));
      }
      const table = el('table', 'lt');
      const header = el('tr');
      for (const text of ['Prediction', 'Training error', 'Test error', 'Repeat previous bar (test)']) header.append(el('th', '', text));
      table.append(header);
      for (const [key, label] of partsOf([point])) {
        const tr = el('tr');
        tr.append(el('td','nm',label), el('td','n',pct(get(point.train_parts, key))), el('td','n',pct(get(point.test_parts, key))),
                  el('td','n',pct(get(point.copy_baseline?.test, key))));
        table.append(tr);
      }
      box.append(table);
      if (point.recall_attention?.length) box.append(this.attention(point.recall_attention));
    }
    return box;
  }

  attention(heads) {
    const detail = el('details', 'stats-listening stats-attention');
    detail.open = Boolean(this.panel.querySelector('.stats-attention')?.open);
    detail.append(el('summary', '', 'Recall head focus'),
                  el('div', 'edn', 'Peak share is the average attention on the top bar. Paired values compare actual attention with a uniform read.'));
    const t = el('table', 'lt');
    const header = el('tr');
    for (const label of ['Head', 'Mean lookback', 'Peak share', '8 or 16 bars back', 'Kick entry', 'Synth entry']) {
      header.append(el('th', '', label));
    }
    t.append(header);
    for (const head of heads) {
      const row = el('tr');
      row.append(el('td', 'nm', String(head.head)), el('td', 'n', `${head.mean_lag.toFixed(1)} bars`),
                 el('td', 'n', pct(head.peak_share)));
      for (const field of ['eight_or_sixteen', 'kick_entrance', 'synth_entrance']) {
        row.append(el('td', 'n', `${pct(head[field])} / ${pct(head[field + '_uniform'])}`));
      }
      t.append(row);
    }
    detail.append(t);
    return detail;
  }
}
