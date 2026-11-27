// Node only: the real machine one-shots in engine/samples/<device>/<voice>.wav as Engine.setSamples wants them.
//   import { loadSamples } from './samples.mjs';  const s = loadSamples();  if (s) engine.setSamples(s.kit, s.sampleRate);
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { readWav } from './wav.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
let cache;

export function loadSamples(dir = path.join(HERE, 'samples')) {
  if (cache !== undefined && dir === path.join(HERE, 'samples')) return cache;
  const index = path.join(dir, 'index.json');
  if (!fs.existsSync(index)) { cache = null; return null; }
  const kit = {}; let sampleRate = 44100;
  for (const [dev, voices] of Object.entries(JSON.parse(fs.readFileSync(index, 'utf8')))) {
    kit[dev] = {};
    for (const voice of voices) {
      const w = readWav(path.join(dir, dev, `${voice}.wav`));
      sampleRate = w.sr;
      const a = new Float32Array(w.L.length);
      for (let i = 0; i < a.length; i++) a[i] = w.R ? 0.5 * (w.L[i] + w.R[i]) : w.L[i];
      kit[dev][voice] = a;
    }
  }
  cache = { kit, sampleRate };
  return cache;
}

// true unless the command line asks for the old synthesized models
export const useSamples = () => !process.argv.includes('--no-samples');
