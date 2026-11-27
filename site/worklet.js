// AudioWorklet: runs the instrument (engine.mjs); reports each bar start and each sixteenth to the page.
// With the scope on (the player's waveform, site/wave.js) it also sends the min and max of the output per `per`
// samples, ten columns at a time, while a bar from the page sounds; it does nothing else with the audio.
import { Engine } from './engine.mjs';

const SCOPE_BATCH = 10;                            // columns per message (100 ms at 10 ms per column)

class MinimalProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.engine = null;
    this.seqs = [];                                // the page's ids of the bars in the engine's queue, in its order
    this.held = -1;                                // the id of the bar that sounds (-1: silence or a bar without an id)
    this.scope = null;
    this.blockFrame = 0;                           // the engine frame at the start of the block being rendered
    this.port.onmessage = (e) => {
      const m = e.data;
      try {
        if (m.type === 'pool') {
          if (!this.engine) {
            this.engine = new Engine(sampleRate, m.pool);
            // a bar starts: which queued bar it is (seq; null = the engine plays the bar before again), its frame,
            // and with the scope on its place in the scope's columns
            this.engine.onBar((bar, tokens, queued, mods) => {
              let popped = false, seq;
              while (this.seqs.length > queued) { seq = this.seqs.shift(); popped = true; }
              if (popped) this.held = Number.isInteger(seq) ? seq : -1;
              const s = this.scope;
              this.port.postMessage({ type: 'bar', bar, tokens, queued, mods, frame: this.engine.frame,
                seq: popped ? this.held : null, held: this.held,
                col: s ? (s.at + this.engine.frame - this.blockFrame) / s.per : null });
            });
            this.engine.onStep((bar, step) => this.port.postMessage({ type: 'step', bar, step, t: currentTime }));
            if (this.kit) this.engine.setSamples(this.kit.kit, this.kit.sampleRate);
            if (this.balance) this.engine.setBalance(this.balance.on, this.balance.ratio, this.balance.rangeDb, this.balance);
          } else {
            this.engine.setPool(m.pool);
          }
        } else if (m.type === 'samples') {          // real 606/808/909 one-shots (site/kit.js)
          this.kit = { kit: m.kit, sampleRate: m.sampleRate };
          if (this.engine) this.engine.setSamples(m.kit, m.sampleRate);
        } else if (m.type === 'balance') {          // loudness balance of the synth voices against the kit
          this.balance = m;
          if (this.engine) this.engine.setBalance(m.on, m.ratio, m.rangeDb, m);
        } else if (m.type === 'tempo' && this.engine) {
          this.engine.setTempo(m.bpm);
        } else if (m.type === 'flush' && this.engine) {
          this.engine.flush();
          this.seqs.length = 0;
        } else if (m.type === 'silence' && this.engine) {
          this.engine.silence();
          this.seqs.length = 0; this.held = -1;
        } else if (m.type === 'queue' && this.engine) {
          this.engine.queueBar(m.tokens, m.mods, m.patterns, m.patches);
          this.seqs.push(m.seq);
        } else if (m.type === 'scope') {
          this.scope = m.on ? { per: Math.max(32, m.per | 0 || 441), at: 0, col: 0, n: 0, lo: Infinity, hi: -Infinity,
            buf: new Float32Array(2 * SCOPE_BATCH), k: 0 } : null;
        }
      } catch (err) {
        this.port.postMessage({ type: 'error', message: String((err && err.stack) || err) });
      }
    };
  }

  // min and max of both channels per column. The columns count only the blocks in which a bar from the page sounds
  // (`at` = samples taken so far), so a silent stretch (stopped, nothing queued) takes no room in the page's buffer.
  collect(l, r) {
    const s = this.scope, n = l.length;
    for (let i = 0; i < n; i++) {
      const a = l[i], b = r[i];
      if (a < s.lo) s.lo = a; if (a > s.hi) s.hi = a;
      if (b < s.lo) s.lo = b; if (b > s.hi) s.hi = b;
      if (++s.n < s.per) continue;
      s.buf[s.k++] = s.lo; s.buf[s.k++] = s.hi; s.lo = Infinity; s.hi = -Infinity; s.n = 0;
      if (s.k === s.buf.length) {
        // offset: the context frame minus the scope's sample count, so the page finds the column it hears now
        this.port.postMessage({ type: 'wave', col: s.col, data: s.buf, offset: currentFrame - s.at }, [s.buf.buffer]);
        s.col += SCOPE_BATCH; s.buf = new Float32Array(2 * SCOPE_BATCH); s.k = 0;
      }
    }
    s.at += n;
  }

  process(inputs, outputs) {
    const out = outputs[0];
    if (this.engine && out && out[0]) {
      this.blockFrame = this.engine.frame;
      this.engine.process(out[0], out[1] || out[0]);
      if (this.scope && this.held >= 0) this.collect(out[0], out[1] || out[0]);
    }
    return true;
  }
}

registerProcessor('minimal', MinimalProcessor);
