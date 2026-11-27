// DJ effects for one mixer channel: echo, reverb, flanger, gater, phaser, roll, crush, sweep. One effect is selected
// per channel and one amount (0..1) sets how much of it is heard. Echo, reverb, flanger, phaser and sweep are sends
// beside the dry signal, so their tails ring on when the send closes; the gater chops the dry signal in time with
// the bar; roll (a beat repeat) and crush (fewer bits) take over part of the dry signal as their amount rises.
export const FX = ['vanish', 'sink', 'reverb', 'flanger', 'gater', 'phaser', 'roll', 'crush', 'sweep', 'echo'];
export const FX_LABEL = { echo: 'ECHO', reverb: 'VERB', flanger: 'FLANGE', gater: 'GATE', phaser: 'PHASE', roll: 'ROLL', crush: 'CRUSH', sweep: 'SWEEP', vanish: 'VANISH', sink: 'SINK' };
const DUCK = { roll: 0.7, crush: 0.75, vanish: 0.9, sink: 0.9 };          // how far the dry signal gives way at full amount
// vanish and sink: the track passes through a resonant filter that sweeps away (vanish: a high-pass rising to about
// 2.4 kHz; sink: a low-pass falling to about 200 Hz) into a reverb, while its dry signal gives way: it dissolves
const FILT = {
  vanish: (a) => ({ f: 30 * Math.pow(80, a), q: 0.7 + 5.5 * a, verb: 0.75 * a }),
  sink: (a) => ({ f: 18000 * Math.pow(0.011, a), q: 0.7 + 5 * a, verb: 0.6 * a }),
};
const GATES = ['x.x.x.x.x.x.x.x.', 'x..x..x.x..x..x.', 'xx.xxx.xxx.xxx.x', 'x...x.x.x...x.x.'];

function impulse(ctx, seconds, seed) {
  const n = Math.floor(ctx.sampleRate * seconds), buf = ctx.createBuffer(2, n, ctx.sampleRate);
  let s = seed >>> 0;
  const rnd = () => { s = (s + 0x6d2b79f5) >>> 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296 * 2 - 1; };
  for (let c = 0; c < 2; c++) {
    const d = buf.getChannelData(c);
    let low = 0;
    for (let i = 0; i < n; i++) {
      const t = i / n, env = Math.pow(1 - t, 2.4) * (i < 400 ? i / 400 : 1);
      low += (rnd() - low) * (0.75 - 0.45 * t);         // the tail gets darker as it decays
      d[i] = low * env;
    }
  }
  return buf;
}

export class FxUnit {
  constructor(ctx) {
    this.ctx = ctx; this.type = 'echo'; this.amount = 0; this.bpm = 126; this.echoSteps = 3; this.gate = 0;
    const g = (v) => { const n = ctx.createGain(); n.gain.value = v; return n; };
    const bq = (type, f, q) => { const b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; if (q != null) b.Q.value = q; return b; };
    this.input = g(1); this.dry = g(1); this.output = g(1);
    this.input.connect(this.dry); this.dry.connect(this.output);
    // echo: a tempo delay whose repeats lose their lows and highs
    this.eSend = g(0); this.eDelay = ctx.createDelay(2.5); this.eFb = g(0.52); const eHp = bq('highpass', 380, 0.5), eLp = bq('lowpass', 3800, 0.5);
    this.input.connect(this.eSend); this.eSend.connect(this.eDelay); this.eDelay.connect(eHp); eHp.connect(eLp);
    eLp.connect(this.output); eLp.connect(this.eFb); this.eFb.connect(this.eDelay);
    // reverb: a dark room without lows
    this.rSend = g(0); const conv = ctx.createConvolver(); conv.buffer = impulse(ctx, 2.8, 7 + Math.floor(Math.random() * 1000)); const rHp = bq('highpass', 320, 0.5);
    this.input.connect(this.rSend); this.rSend.connect(conv); conv.connect(rHp); rHp.connect(this.output);
    // vanish and sink share one resonant filter, which also feeds the room
    this.vSend = g(0); this.vFilt = bq('highpass', 30, 0.7); this.vVerb = g(0);
    this.input.connect(this.vSend); this.vSend.connect(this.vFilt); this.vFilt.connect(this.output); this.vFilt.connect(this.vVerb); this.vVerb.connect(conv);
    // flanger: a short delay swept slowly, fed back
    this.fSend = g(0); this.fDelay = ctx.createDelay(0.05); this.fDelay.delayTime.value = 0.0046; this.fFb = g(0.55);
    this.fLfo = ctx.createOscillator(); this.fLfo.type = 'sine'; this.fLfo.frequency.value = 0.07; this.fDepth = g(0.0016);
    this.fLfo.connect(this.fDepth); this.fDepth.connect(this.fDelay.delayTime); this.fLfo.start();
    this.input.connect(this.fSend); this.fSend.connect(this.fDelay); this.fDelay.connect(this.output); this.fDelay.connect(this.fFb); this.fFb.connect(this.fDelay);
    // phaser: four all-pass stages swept slowly; beside the dry signal its notches move through the sound
    this.pSend = g(0); this.pStages = [0, 1, 2, 3].map(() => bq('allpass', 900, 0.7));
    this.pLfo = ctx.createOscillator(); this.pLfo.type = 'sine'; this.pLfo.frequency.value = 0.1; this.pDepth = g(700);
    this.pLfo.connect(this.pDepth); for (const st of this.pStages) this.pDepth.connect(st.frequency); this.pLfo.start();
    this.pFb = g(0.35);
    this.input.connect(this.pSend); this.pSend.connect(this.pStages[0]);
    for (let i = 0; i < 3; i++) this.pStages[i].connect(this.pStages[i + 1]);
    this.pStages[3].connect(this.output); this.pStages[3].connect(this.pFb); this.pFb.connect(this.pStages[0]);
    // roll: the last sixteenth (or eighth) repeated, bright, while the dry signal gives way
    this.oSend = g(0); this.oDelay = ctx.createDelay(1.0); this.oFb = g(0.55); const oHp = bq('highpass', 180, 0.6);
    this.input.connect(this.oSend); this.oSend.connect(this.oDelay); this.oDelay.connect(oHp); oHp.connect(this.output); oHp.connect(this.oFb); this.oFb.connect(this.oDelay);
    this.rollSteps = 1;
    // crush: the signal at fewer bits, through a low-pass that keeps it from turning to fizz
    this.cSend = g(0); const crush = ctx.createWaveShaper(); crush.curve = (() => { const n = 4096, c = new Float32Array(n), steps = 12; for (let i = 0; i < n; i++) { const x = i / (n - 1) * 2 - 1; c[i] = Math.round(x * steps) / steps; } return c; })();
    const cLp = bq('lowpass', 6500, 0.7);
    this.input.connect(this.cSend); this.cSend.connect(crush); crush.connect(cLp); cLp.connect(this.output);
    // sweep: a resonant band swept over four bars
    this.wSend = g(0); this.wBand = bq('bandpass', 800, 4.5); this.wLfo = ctx.createOscillator(); this.wLfo.type = 'triangle'; this.wLfo.frequency.value = 0.12; this.wDepth = g(650);
    this.wLfo.connect(this.wDepth); this.wDepth.connect(this.wBand.frequency); this.wLfo.start();
    this.input.connect(this.wSend); this.wSend.connect(this.wBand); this.wBand.connect(this.output);
    this.sends = { echo: this.eSend, reverb: this.rSend, flanger: this.fSend, phaser: this.pSend, roll: this.oSend, crush: this.cSend, sweep: this.wSend, vanish: this.vSend, sink: this.vSend };
    this.scale = { echo: 0.85, reverb: 0.9, flanger: 0.8, phaser: 0.9, roll: 0.55, crush: 0.8, sweep: 1.1, vanish: 1, sink: 1 };   // roll: its repeats add up to about 2.2 times the send
    this.setTempo(126);
  }
  setTempo(bpm) {
    this.bpm = bpm;
    const t = this.ctx.currentTime;
    this.eDelay.delayTime.setTargetAtTime(Math.min(2.4, this.echoSteps * 15 / bpm), t, 0.05);
    this.fLfo.frequency.setTargetAtTime(bpm / (60 * 4 * 8), t, 0.1);      // one sweep in eight bars
    this.pLfo.frequency.setTargetAtTime(bpm / (60 * 4 * 4), t, 0.1);      // the phaser: one sweep in four bars
    this.wLfo.frequency.setTargetAtTime(bpm / (60 * 4 * 4) / 2, t, 0.1);  // the band: up and down in eight bars
    this.oDelay.delayTime.setTargetAtTime(Math.min(0.9, this.rollSteps * 15 / bpm), t, 0.02);
  }
  // select the effect; the other sends close and their tails ring out
  setType(type, opts = {}) {
    if (!FX.includes(type)) return;
    this.type = type;
    if (opts.echoSteps) this.echoSteps = opts.echoSteps;
    if (opts.rollSteps) this.rollSteps = opts.rollSteps;
    this.gate = Number.isInteger(opts.gate) ? opts.gate % GATES.length : this.gate;
    const t = this.ctx.currentTime;
    for (const [name, send] of Object.entries(this.sends)) if (send !== this.sends[type]) { send.gain.cancelScheduledValues(t); send.gain.setTargetAtTime(0, t, 0.03); }
    if (FILT[type]) this.vFilt.type = type === 'vanish' ? 'highpass' : 'lowpass';
    this.setTempo(this.bpm);
    this.set(this.amount, t);
  }
  value(a) { return this.type === 'gater' ? 0 : Math.max(0, Math.min(1, a)) * this.scale[this.type]; }
  dryOf(a) { const d = DUCK[this.type]; return d ? 1 - d * Math.max(0, Math.min(1, a)) : 1; }
  // the amount now (a knob move)
  set(a, t = this.ctx.currentTime) {
    this.amount = a;
    const send = this.sends[this.type];
    if (send) { send.gain.cancelScheduledValues(t); send.gain.setTargetAtTime(this.value(a), t, 0.02); }
    if (this.type !== 'gater' || a <= 0.001) { this.dry.gain.cancelScheduledValues(t); this.dry.gain.setTargetAtTime(this.dryOf(a), t, 0.01); }
    const F = FILT[this.type];
    if (F) { const v = F(a); for (const [p, x] of [[this.vFilt.frequency, v.f], [this.vFilt.Q, v.q], [this.vVerb.gain, v.verb]]) { p.cancelScheduledValues(t); p.setTargetAtTime(x, t, 0.03); } }
  }
  // automation: reach the amount at time t by a straight line from the last scheduled point, or jump to it at t
  ramp(a, t) {
    this.amount = a; const send = this.sends[this.type]; if (send) send.gain.linearRampToValueAtTime(this.value(a), t);
    if (DUCK[this.type]) this.dry.gain.linearRampToValueAtTime(this.dryOf(a), t);
    const F = FILT[this.type];
    if (F) { const v = F(a); this.vFilt.frequency.exponentialRampToValueAtTime(v.f, t); this.vFilt.Q.linearRampToValueAtTime(v.q, t); this.vVerb.gain.linearRampToValueAtTime(v.verb, t); }
  }
  step(a, t) {
    this.amount = a; const send = this.sends[this.type]; if (send) send.gain.setValueAtTime(this.value(a), t);
    if (DUCK[this.type]) this.dry.gain.setValueAtTime(this.dryOf(a), t);
    const F = FILT[this.type];
    if (F) { const v = F(a); this.vFilt.frequency.setValueAtTime(v.f, t); this.vFilt.Q.setValueAtTime(v.q, t); this.vVerb.gain.setValueAtTime(v.verb, t); }
  }
  hold(t) {
    for (const p of [this.vFilt.frequency, this.vFilt.Q, this.vVerb.gain]) { p.cancelScheduledValues(t); p.setValueAtTime(p.value, t); }
    for (const send of new Set(Object.values(this.sends))) { send.gain.cancelScheduledValues(t); send.gain.setValueAtTime(send.gain.value, t); }
    if (DUCK[this.type]) { this.dry.gain.cancelScheduledValues(t); this.dry.gain.setValueAtTime(this.dry.gain.value, t); }
  }
  // the gater's sixteen steps of the bar that starts at t0 (seconds on the context clock)
  scheduleGate(t0, barLen, depth) {
    const p = this.dry.gain, step = barLen / 16, pat = GATES[this.gate];
    if (this.type !== 'gater' || depth <= 0.001) { p.setTargetAtTime(1, t0, 0.01); return; }
    const low = 1 - Math.max(0, Math.min(0.9, depth));
    for (let i = 0; i < 16; i++) p.setTargetAtTime(pat[i] === 'x' ? 1 : low, t0 + i * step, 0.004);
  }
}
