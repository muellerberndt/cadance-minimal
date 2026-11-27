// What the page needs to draw the instrument: knob lists with labels and physical ranges per voice
// and per drum device, the drum voices each device has, and the per-bar modulation ranges.
// Every knob value in a patch is 0..1; `display(knob, u)` gives the physical value for a readout.
import { PARAMS, INDEX, DEVICES, DEVICE_VOICES, DRUM_VOICES, MOD_RANGE, SID_WAVES, SID_FX, SID_MODES, drumRanges, phys, sidRegisters } from './engine.mjs';

const LABELS = {
  wave: 'WAVE', tune: 'TUNE', cutoff: 'CUTOFF', reso: 'RESO', envmod: 'ENV MOD', decay: 'DECAY', accent_amt: 'ACCENT',
  drive: 'DRIVE', glide: 'SLIDE', gate: 'GATE', level: 'LEVEL', pan: 'PAN', send_delay: 'DELAY', send_reverb: 'REVERB',
  fm_ratio: 'FM RATIO', fm_amt: 'FM AMT', pitch_env: 'P.ENV', pitch_decay: 'P.DECAY', ftype: 'TYPE', f_env: 'F.ENV',
  f_decay: 'F.DECAY', attack: 'ATTACK', sustain: 'SUSTAIN', release: 'RELEASE', lfo_rate: 'LFO RATE', lfo_pitch: 'LFO>PITCH',
  lfo_cutoff: 'LFO>CUT', lfo_amp: 'LFO>AMP',
  fx: 'RING/SYNC', route: 'ROUTE', mode: 'MODE', pwm: 'PWM',
  ...Object.fromEntries([1, 2, 3].flatMap((k) => [[`wave${k}`, `WAVE ${k}`], [`tune${k}`, `TUNE ${k}`], [`fine${k}`, `FINE ${k}`], [`pw${k}`, `PW ${k}`],
    [`attack${k}`, `A ${k}`], [`decay${k}`, `D ${k}`], [`sustain${k}`, `S ${k}`], [`release${k}`, `R ${k}`]])),
};
export const DRUM_LABELS = { kick: 'BD', snare: 'SD', clap: 'CP', ch: 'CH', oh: 'OH', ltom: 'LT', htom: 'HT', rim: 'RS', cow: 'CB', cym: 'CY' };
const DRUM_KNOB = { tune: 'TUNE', decay: 'DECAY', level: 'LEVEL', tone: 'TONE' };
const TONE_LABEL = { snare: 'SNAPPY', clap: 'Q', kick: 'ATTACK' };

const knob = (voice, p) => ({ name: p.name, label: LABELS[p.name] || p.name.toUpperCase(), min: +phys(p, 0).toFixed(4), max: +phys(p, 1).toFixed(4), unit: p.unit, map: p.map, def: p.def, index: INDEX[voice][p.name] });

function drumDevice(dev) {
  const R = drumRanges(dev), lvl = PARAMS.drums[INDEX.drums.kick_level];
  return {
    label: dev,
    voices: DEVICE_VOICES[dev].map((v) => ({
      name: v, label: DRUM_LABELS[v],
      knobs: ['tune', 'decay', 'level', 'tone'].map((k) => {
        const [min, max, unit] = k === 'level' ? [phys(lvl, 0), phys(lvl, 1), 'gain'] : R[v][k];
        return { name: `${v}_${k}`, label: k === 'tone' && TONE_LABEL[v] ? TONE_LABEL[v] : DRUM_KNOB[k], min, max, unit, map: k === 'level' ? 'sq' : k === 'decay' || k === 'tune' || (k === 'tone' && ['cow', 'ch', 'oh'].includes(v)) ? 'exp' : 'lin', def: PARAMS.drums[INDEX.drums[`${v}_${k}`]].def, index: INDEX.drums[`${v}_${k}`] };
      }),
    })),
  };
}

export const UI = {
  steps: 16,
  layers: ['beat', 'bass', 'lead', 'air'],
  acid: { label: 'ACID', knobs: PARAMS.acid.map((p) => knob('acid', p)) },
  synth: { label: 'SYNTH', knobs: PARAMS.synth.map((p) => knob('synth', p)) },
  sampler: { label: 'SAMPLER', knobs: PARAMS.sampler.map((p) => knob('sampler', p)) },
  // the chip voice: its knobs, the names of its register settings and the decoding of a patch into registers
  sid: { label: 'SID', knobs: PARAMS.sid.map((p) => knob('sid', p)), waves: SID_WAVES, fx: SID_FX, modes: SID_MODES, registers: sidRegisters },
  drums: {
    common: ['drive', 'level', 'pan', 'send_delay', 'send_reverb'].map((n) => knob('drums', PARAMS.drums[INDEX.drums[n]])),
    devices: Object.fromEntries(DEVICES.map((d) => [d, drumDevice(d)])),
    all_voices: DRUM_VOICES,
  },
  mods: MOD_RANGE,
};

// physical value of a knob at u (linear between min and max for 'lin', geometric for 'exp')
export function display(k, u) {
  u = Math.min(1, Math.max(0, u));
  if (k.map === 'exp' && k.min > 0) return k.min * Math.pow(k.max / k.min, u);
  if (k.map === 'sq') return k.min + (k.max - k.min) * u * u;
  return k.min + (k.max - k.min) * u;
}
