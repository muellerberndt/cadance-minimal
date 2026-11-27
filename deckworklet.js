// AudioWorklet of the mixer view: three instruments (engine.mjs), one per deck, on ONE clock. All three engines are
// made together and step in the same render quanta at the same tempo, so their bars start on the same sample: a deck
// that starts later comes in on the next bar line, in time with the others. Each deck has its own stereo output,
// which the page runs through its channel strip (EQ, filters, fader). The radio page (radio-app.js) sets where the
// engines' bars begin: 'align' puts the next bar line of every engine on a frame of the context's clock, its own.
import { Engine } from './engine.mjs';

const DECKS = 3;

class DeckProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.engines = [];
    for (let d = 0; d < DECKS; d++) {
      const e = new Engine(sampleRate, { layers: [] });
      e.setTempo(126);
      // t: the bar line's time on the context clock, to the sample (the autopilot schedules its moves against it)
      e.onBar((bar, tokens, queued) => this.port.postMessage({ type: 'bar', deck: d, bar, tokens, queued, t: currentTime + (e.frame - (e.q0 ?? e.frame)) / sampleRate }));
      if (d === 0) e.onStep((bar, step) => this.port.postMessage({ type: 'step', bar, step, t: currentTime }));
      this.engines.push(e);
    }
    this.port.onmessage = (e) => {
      const m = e.data;
      try {
        const eng = Number.isInteger(m.deck) ? this.engines[m.deck] : null;
        if (m.type === 'samples') {                       // the real 606/808/909 one-shots, for every deck
          for (const g of this.engines) g.setSamples(m.kit, m.sampleRate);
        } else if (m.type === 'balance') {                // loudness balance of the synth voices against the kit, as in the player
          for (const g of this.engines) g.setBalance(m.on, m.ratio, m.rangeDb, m);
        } else if (m.type === 'tempo') {                  // one tempo for all decks (they share the clock)
          for (const g of this.engines) g.setTempo(m.bpm);
        } else if (m.type === 'align') {                  // the next bar of every engine begins at this context frame
          for (const g of this.engines) { g.nextStep = m.frame - (this.base ?? currentFrame); g.stepIdx = 15; }
        } else if (eng && m.type === 'pool') {
          eng.setPool(m.pool);
        } else if (eng && m.type === 'queue') {
          eng.queueBar(m.tokens, m.mods, m.patterns, m.patches);
        } else if (eng && m.type === 'flush') {
          eng.flush();
        } else if (eng && m.type === 'silence') {
          eng.silence();
        }
      } catch (err) {
        this.port.postMessage({ type: 'error', message: String((err && err.stack) || err) });
      }
    };
  }

  process(inputs, outputs) {
    this.base = currentFrame - this.engines[0].frame;    // a context frame minus the engines' frame: the same for all, for good
    for (let d = 0; d < DECKS; d++) {
      const out = outputs[d];
      this.engines[d].q0 = this.engines[d].frame;
      if (out && out[0]) this.engines[d].process(out[0], out[1] || out[0]);
    }
    return true;
  }
}

registerProcessor('decks', DeckProcessor);
