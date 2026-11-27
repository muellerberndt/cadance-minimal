// Minimal WAV reading and writing (PCM 16/24/32-bit and float32), for Node.
import fs from 'node:fs';

export function readWav(path) {
  const b = fs.readFileSync(path);
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw new Error('not a WAV file: ' + path);
  let o = 12, fmt = null, data = null;
  while (o + 8 <= b.length) {
    const id = b.toString('ascii', o, o + 4), size = b.readUInt32LE(o + 4);
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(o + 8), ch: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22) };
    if (id === 'data') { data = b.subarray(o + 8, Math.min(b.length, o + 8 + size)); break; }
    o += 8 + size + (size & 1);
  }
  if (!fmt || !data) throw new Error('WAV without fmt or data: ' + path);
  const bytes = fmt.bits / 8, n = Math.floor(data.length / (bytes * fmt.ch));
  const chans = []; for (let c = 0; c < fmt.ch; c++) chans.push(new Float32Array(n));
  for (let i = 0; i < n; i++) for (let c = 0; c < fmt.ch; c++) {
    const p = (i * fmt.ch + c) * bytes; let v;
    if (fmt.tag === 3 && fmt.bits === 32) v = data.readFloatLE(p);
    else if (fmt.bits === 16) v = data.readInt16LE(p) / 32768;
    else if (fmt.bits === 24) v = data.readIntLE(p, 3) / 8388608;
    else if (fmt.bits === 32) v = data.readInt32LE(p) / 2147483648;
    else throw new Error('unsupported WAV sample format in ' + path);
    chans[c][i] = v;
  }
  return { sr: fmt.sr, L: chans[0], R: chans[1] || chans[0] };
}

export function writeWav(path, L, R, sr) {
  const n = L.length, b = Buffer.alloc(44 + n * 4);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 4, 4); b.write('WAVE', 8); b.write('fmt ', 12); b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20); b.writeUInt16LE(2, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 4, 28); b.writeUInt16LE(4, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) {
    b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(L[i] * 32767))), 44 + i * 4);
    b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(R[i] * 32767))), 46 + i * 4);
  }
  fs.writeFileSync(path, b);
}
