"""Absolute instrument controls. Fitted patches are teachers, never playback presets."""
import numpy as np

# The order and bin counts are part of the v1 checkpoint format. Unit coordinates
# follow engine/PARAMS, including its logarithmic frequency and envelope ranges.
VERSION = "dsp-v2"
SAMPLE_VERSION = "dsp-v3"
VOICES = ("synth", "acid", "sampler")
DEFAULTS = dict(wave=.5, tune=.5, fm_ratio=.4, fm_amt=0., pitch_env=1/3,
                pitch_decay=.3, cutoff=.6, reso=.2, ftype=0., f_env=.5, f_decay=.4,
                attack=.1, decay=.5, sustain=.5, release=.3, lfo_rate=.5,
                lfo_pitch=0., lfo_cutoff=0., lfo_amp=0., drive=0., level=.55,
                pan=.5, send_delay=0., send_reverb=0., envmod=.4, accent_amt=.5,
                glide=.4, gate=.5)
ACID = dict(wave=0., tune=.5, cutoff=.3, reso=.6, envmod=.4, decay=.4,
            accent_amt=.5, drive=.2, glide=.4, gate=.5)
NAMES = tuple(DEFAULTS)
# voice, MIDI register (midi // 12), gain in powers of two, then patch controls
PATCH_SIZES = (2, 10, 11) + tuple(13 if k in ("tune", "pitch_env") else 7 for k in NAMES)

# Four metrical velocities, one for each sixteenth position within a beat.
SIZES = PATCH_SIZES + (7,) * 4


def sizes(book):
    version = book["version"]
    if version == "dsp-v1":
        return PATCH_SIZES
    if version == VERSION:
        return SIZES
    if version == SAMPLE_VERSION:
        return (len(VOICES), *SIZES[1:], len(book["samples"]), 7, 2)
    raise ValueError(f"unknown DSP checkpoint format {version!r}")


def width(book):
    return len(sizes(book))


def velocities(token):
    pattern = token.get("pattern", {})
    hits = [[] for _ in range(4)]
    for step, note in enumerate(pattern.get("notes", [])):
        if note is None:
            continue
        if token.get("voice") == "acid":
            value = 1.0 if (pattern.get("accent") or [0]*16)[step] else 4/7
        else:
            value = (pattern.get("vel") or [0.85]*16)[step]
        if value is not None and value > 0:
            hits[step % 4].append(value)
    all_hits = [v for group in hits for v in group]
    fallback = float(np.median(all_hits)) if all_hits else 0.85
    return [int(np.clip(round(7 * (float(np.median(group)) if group else fallback)) - 1, 0, 6))
            for group in hits]


def encode(token, layer, book=None):
    tok = token or {}
    voice = tok.get("voice", "synth")
    if voice not in VOICES or (voice == "sampler" and (not book or not book.get("samples"))):
        raise ValueError(f"unknown synth voice {voice!r}")
    patch = {**DEFAULTS, **(ACID if voice == "acid" else {}), **tok.get("patch", {})}
    notes = [n for n in tok.get("pattern", {}).get("notes", []) if n is not None]
    register = int(np.median(notes) // 12) if notes else (3, 4, 5)[layer]
    gain = float(tok.get("gain", 1.0))
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("expected a positive synth gain")
    values = [VOICES.index(voice), int(np.clip(register, 0, 9)),
              int(np.clip(round(np.log2(gain)) + 5, 0, 10))]
    for name, size in zip(NAMES, SIZES[3:]):
        value = float(patch[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid unit control {name}={value}")
        values.append(round(value * (size - 1)))
    encoded = values + velocities(tok)
    if book and book.get("samples"):
        if voice == "sampler":
            controls = tok.get("patch", {})
            start = float(controls.get("start", 0))
            reverse = float(controls.get("reverse", 0))
            if not 0 <= start <= 1 or not 0 <= reverse <= 1:
                raise ValueError("Sampler start and reverse must be unit controls")
            encoded += [book["samples"].index(tok["sample"]), round(start*6), int(reverse >= .5)]
        else:
            encoded += [0, 0, 0]
    return encoded


def teachers(pool):
    layers = pool["layers"]
    if [layer["name"] for layer in layers] != ["beat", "bass", "lead", "air"]:
        raise ValueError("expected beat, bass, lead and air in the fitted pool")
    samples = sorted(pool.get("sample_bank") or {})
    if samples and not 2 <= len(samples) <= 12:
        raise ValueError("Sampler lessons need between two and twelve one-shots")
    book = {"version": SAMPLE_VERSION if samples else VERSION, "hearing": "sounding"}
    if samples:
        book["samples"] = samples
    book["layers"] = [[encode(tok, i, book) if tok or j == 0 else None
                       for j, tok in enumerate(layer["tokens"])]
                      for i, layer in enumerate(layers[1:])]
    return book


def track_controls(tokens, octaves, book):
    """Carry the last patch through silence, using neutral controls before entry."""
    parts = []
    for layer, table in enumerate(book["layers"]):
        ids = tokens[:, layer + 1]
        if np.any(ids < 0) or np.any(ids >= len(table)) or any(table[i] is None for i in np.unique(ids)):
            raise ValueError("corpus synth token missing from its fitted pool")
        indices = np.maximum.accumulate(np.where(ids > 0, np.arange(len(ids)) + 1, 0))
        values = np.array([table[i] for i in np.r_[0, ids]], dtype=int)
        if octaves is not None:
            octave = octaves[:, layer]
            known = (ids > 0) & (octave >= 0)
            values[1:, 1][known] = np.clip(octave[known] + 1, 0, 9)
        parts.append(values[indices])
    return np.concatenate(parts, axis=1)


def instrument(row, book=None):
    """Decode from motor slots alone, also when the combination is new."""
    voice, register, gain = map(int, row[:3])
    patch = {name: int(v) / (size - 1) for name, v, size in zip(NAMES, row[3:], SIZES[3:])}
    token = {"voice": VOICES[voice], "patch": patch, "gain": 2.0 ** (gain - 5)}
    if token["voice"] == "sampler":
        sample, start, reverse = map(int, row[len(SIZES):])
        token["sample"] = book["samples"][sample]
        patch.update(start=start/6, reverse=reverse)
    return token, register


# ---------------------------------------------------------------- reserved: the SID chip voice
# A brain of the stage-1 shape (PLAN.md) is born with one block of motor slots per synth for the engine's SID
# voice (engine/engine.mjs: PARAMS.sid, sidRegisters): a switch, then one slot per register. Value 0 of every
# slot is "off". With the switch on, a register's value v plays its level v - 1. A register the chip holds
# as a number has the chip's own count of levels: five waveforms, 25 semitone offsets, the sixteen values
# of an envelope nibble, four ring/sync settings, eight filter routings, four filter modes. A register the
# engine takes as a unit value has seven levels like the other sound controls, and thirteen for the two a
# SID player sweeps: the pulse width and the filter cutoff. Until a later stage teaches the chip, every
# lesson labels every slot of the block "off".
SID_OSCILLATOR = ("wave", "tune", "fine", "pw", "attack", "decay", "sustain", "release")
SID_SHARED = ("fx", "route", "cutoff", "reso", "mode", "f_env", "f_decay", "pwm",
              "level", "pan", "send_delay", "send_reverb")
SID_LEVELS = dict(wave=5, tune=25, fine=7, pw=13, attack=16, decay=16, sustain=16, release=16,
                  fx=4, route=8, cutoff=13, reso=7, mode=4, f_env=7, f_decay=7, pwm=7,
                  level=7, pan=7, send_delay=7, send_reverb=7)
SID_NAMES = tuple(f"{name}{k}" for k in (1, 2, 3) for name in SID_OSCILLATOR) + SID_SHARED
SID_SIZES = (2,) + tuple(SID_LEVELS[name.rstrip("123")] + 1 for name in SID_NAMES)
SID_WIDTH = len(SID_SIZES)
# The engine's own defaults (PARAMS.sid), for a register left "off" while the chip plays.
SID_DEFAULTS = {**{f"{name}{k}": value for k in (1, 2, 3) for name, value in dict(
                    wave=0.75 if k == 1 else 0., tune=.5, fine=.5, pw=.5, attack=0., decay=7/15, sustain=6/15,
                    release=5/15).items()},
                "fx": 0., "route": 1., "cutoff": 1., "reso": 0., "mode": 0., "f_env": .5, "f_decay": .4, "pwm": 0.,
                "level": .55, "pan": .5, "send_delay": 0., "send_reverb": 0.}


def sid_off():
    """The block of a synth that does not play the chip."""
    return [0] * SID_WIDTH


def sid_encode(token):
    """The block of a synth that plays this fitted SID sound (a pool token with voice ``sid``)."""
    if (token or {}).get("voice") != "sid":
        raise ValueError("expected a fitted SID sound")
    patch = {**SID_DEFAULTS, **token.get("patch", {})}
    values = [1]
    for name, size in zip(SID_NAMES, SID_SIZES[1:]):
        value = float(patch[name])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid unit control {name}={value}")
        values.append(1 + round(value * (size - 2)))
    return values


def sid_decode(block):
    """The SID sound a block writes, or None while its switch is off."""
    block = [int(v) for v in block]
    if len(block) != SID_WIDTH:
        raise ValueError("expected one SID block")
    if not block[0]:
        return None
    patch = {name: (value - 1) / (size - 2) if value else SID_DEFAULTS[name]
             for name, value, size in zip(SID_NAMES, block[1:], SID_SIZES[1:])}
    return {"voice": "sid", "patch": patch}


def sid_book(pool):
    """The SID fits of a pool's sounds as blocks: per synth layer and sound, the block that plays it on the chip
    (pool/pool_underground_sid.json holds the fits of the pool's sounds under the same sound numbers)."""
    layers = pool["layers"]
    if [layer["name"] for layer in layers] != ["beat", "bass", "lead", "air"]:
        raise ValueError("expected beat, bass, lead and air in the SID pool")
    return {"layers": [[sid_encode(tok) if tok else None for tok in layer["tokens"]] for layer in layers[1:]]}


def sid_controls(tokens, book, layers=(0, 1, 2)):
    """The SID blocks of a track: the fit of every bar's sound with the chip's switch on, carried through
    silence as the other sound controls are, and "off" before a synth's first sounding bar. ``layers``: the
    synths (0 bass, 1 lead, 2 air) that carry a block."""
    parts = []
    for layer, table in enumerate(book["layers"]):
        if layer not in layers:
            continue
        ids = tokens[:, layer + 1]
        if np.any(ids < 0) or np.any(ids >= len(table)) or any(table[i] is None for i in np.unique(ids[ids > 0])):
            raise ValueError("corpus synth sound missing from the SID pool")
        indices = np.maximum.accumulate(np.where(ids > 0, np.arange(len(ids)) + 1, 0))
        values = np.array([sid_off()] + [table[i] or sid_off() for i in ids], dtype=int)
        parts.append(values[indices])
    return np.concatenate(parts, axis=1)
