// Audio features for the fitter and the checks: radix-2 FFT, log-mel spectrogram, spectral flux,
// a per-sixteenth loudness envelope and the spectral centroid. Plain JS, no Node imports.

export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
const mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
const banks = new Map();

function melBank(nfft, sr, nmel, fmin, fmax) {
  const key = `${nfft}:${sr}:${nmel}:${fmin}:${fmax}`;
  if (banks.has(key)) return banks.get(key);
  const m0 = hz2mel(fmin), m1 = hz2mel(fmax), pts = [];
  for (let i = 0; i < nmel + 2; i++) pts.push((mel2hz(m0 + ((m1 - m0) * i) / (nmel + 1)) * nfft) / sr);
  const bank = [];
  for (let m = 0; m < nmel; m++) {
    const a = pts[m], b = pts[m + 1], c = pts[m + 2], lo = Math.max(1, Math.floor(a)), hi = Math.min(nfft / 2, Math.ceil(c)), w = [];
    for (let k = lo; k <= hi; k++) w.push(k < b ? Math.max(0, (k - a) / Math.max(1e-9, b - a)) : Math.max(0, (c - k) / Math.max(1e-9, c - b)));
    const s = w.reduce((x, y) => x + y, 0) || 1;
    bank.push({ lo, w: Float64Array.from(w.map((x) => x / s)) });
  }
  banks.set(key, bank);
  return bank;
}

const windows = new Map();
function hann(n) {
  if (!windows.has(n)) { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); windows.set(n, w); }
  return windows.get(n);
}

// log-mel frames (frames x nmel, row-major) + linear-magnitude spectral flux per frame
export function logmel(x, sr, nfft = 2048, hop = 512, nmel = 48, fmin = 30, fmax = 16000) {
  const bank = melBank(nfft, sr, nmel, fmin, fmax), win = hann(nfft);
  const frames = Math.max(1, Math.floor((x.length - nfft) / hop) + 1), out = new Float32Array(frames * nmel), flux = new Float32Array(frames);
  const re = new Float64Array(nfft), im = new Float64Array(nfft), prev = new Float64Array(nfft / 2 + 1), mag = new Float64Array(nfft / 2 + 1);
  // floor about 30 dB under the band level of unit-RMS white noise (about 60 dB under a loud tonal
  // band): differences between near-silences do not count (the fitter compares RMS-normalized audio)
  const floor = 3.75e-4 * nfft;
  for (let f = 0; f < frames; f++) {
    const o = f * hop;
    for (let i = 0; i < nfft; i++) { re[i] = (o + i < x.length ? x[o + i] : 0) * win[i]; im[i] = 0; }
    fft(re, im);
    let fl = 0;
    for (let k = 0; k <= nfft / 2; k++) { mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]); const d = mag[k] - prev[k]; if (d > 0 && f > 0) fl += d; prev[k] = mag[k]; }
    flux[f] = fl;
    for (let m = 0; m < nmel; m++) {
      const { lo, w } = bank[m]; let s = 0;
      for (let j = 0; j < w.length; j++) { const k = lo + j; s += w[j] * mag[k] * mag[k]; }
      out[f * nmel + m] = Math.log(floor + s);
    }
  }
  return { mel: out, flux, frames, nmel, hop };
}

export function rms(x) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s / Math.max(1, x.length)); }

// loudness in dB over `segs` equal segments (default: 64 = four per sixteenth of a bar)
export function envelope(x, segs = 64) {
  const out = new Float32Array(segs), n = Math.floor(x.length / segs);
  for (let s = 0; s < segs; s++) { let a = 0; for (let i = s * n; i < (s + 1) * n; i++) a += x[i] * x[i]; out[s] = 10 * Math.log10(1e-8 + a / Math.max(1, n)); }
  return out;
}

export function centroid(x, sr, nfft = 4096) {
  const hop = nfft, re = new Float64Array(nfft), im = new Float64Array(nfft), win = hann(nfft);
  let num = 0, den = 0;
  for (let o = 0; o + nfft <= x.length; o += hop) {
    for (let i = 0; i < nfft; i++) { re[i] = x[o + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 1; k <= nfft / 2; k++) { const m = re[k] * re[k] + im[k] * im[k]; num += m * ((k * sr) / nfft); den += m; }
  }
  return den > 0 ? num / den : 0;
}

// Everything the fitter compares, computed once for the target and per candidate.
export function describe(x, sr, segs = 64) {
  const r = rms(x) || 1e-9, y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] / r;
  return { rms: r, a: logmel(y, sr, 2048, 512), b: logmel(y, sr, 512, 256, 32), env: envelope(y, segs) };
}

function normFlux(f) {
  let m = 0; for (let i = 0; i < f.length; i++) m += f[i];
  m = m / Math.max(1, f.length) || 1;
  const out = new Float32Array(f.length); for (let i = 0; i < f.length; i++) out[i] = Math.log1p(f[i] / m);
  return out;
}

// weights: multi-resolution log-mel L1 + loudness envelope (dB / 20) + onset flux (log-normalized).
// lag: the candidate may sit up to `lag` short frames (256 samples, 5.8 ms each) early or late; the
// best shift counts, so a stem bar cut a few milliseconds off the grid does not penalise the right sound.
export function distance(T, C, w = { mel: 1, env: 0.5, flux: 0.5 }, lag = 2) {
  const shifted = (p, q, n, cols, sh) => {
    let s = 0, c = 0;
    const f0 = Math.max(0, -sh), f1 = Math.min(n, n - sh);
    for (let f = f0; f < f1; f++) for (let k = 0; k < cols; k++) { s += Math.abs(p[f * cols + k] - q[(f + sh) * cols + k]); c++; }
    return s / Math.max(1, c);
  };
  if (!T.fluxN) T.fluxN = normFlux(T.b.flux);
  const cf = normFlux(C.b.flux);
  const na = Math.min(T.a.frames, C.a.frames), nb = Math.min(T.b.frames, C.b.frames);
  let env = 0; for (let i = 0; i < T.env.length; i++) env += Math.abs(T.env[i] - C.env[i]); env = env / T.env.length / 20;
  let best = null;
  for (let sh = -lag; sh <= lag; sh++) {
    const mel = 0.5 * (shifted(T.a.mel, C.a.mel, na, T.a.nmel, Math.round(sh / 2)) + shifted(T.b.mel, C.b.mel, nb, T.b.nmel, sh));
    const flux = shifted(T.fluxN, cf, nb, 1, sh);
    const total = w.mel * mel + w.env * env + w.flux * flux;
    if (!best || total < best.total) best = { total, mel, env, flux, shift: sh };
  }
  return best;
}
