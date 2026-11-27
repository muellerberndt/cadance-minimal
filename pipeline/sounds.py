"""Stage 6: the sounds. A pool of fitted sounds per layer, the sound of every bar, and its filter and space levels.

The brain does not write audio. Per bar it names a kit and switches three synth layers on or off, and for every
sounding synth layer it writes the sound's controls (oscillator, filter, envelopes, level, sends). The pool
supplies the labels: every sounding bar of a recording is matched to one fitted sound of its layer, and that
sound's controls are what a lesson teaches for the bar.
"""
from __future__ import annotations

# the unit controls of a synth sound (each 0..1), in the order brain/timbre.py teaches them
CONTROLS = ("wave", "tune", "fm_ratio", "fm_amt", "pitch_env", "pitch_decay", "cutoff", "reso", "ftype", "f_env",
            "f_decay", "attack", "decay", "sustain", "release", "lfo_rate", "lfo_pitch", "lfo_cutoff", "lfo_amp",
            "drive", "level", "pan", "send_delay", "send_reverb")
ACID_CONTROLS = ("envmod", "accent_amt", "glide", "gate")       # the acid voice has these four as well
KIT_VOICES = ("kick", "snare", "clap", "rim", "cow", "ch", "oh", "cym", "ltom", "htom")
KIT_KNOBS = ("tune", "decay", "tone", "level")


def synth_token(notes, voice="synth", gain=1.0, **controls):
    """A pool sound of a synth layer. ``notes``: the 16 MIDI numbers (or None) of the bar the sound was fitted
    to; their median sets the sound's register. ``controls`` override the neutral setting of 0.5."""
    names = CONTROLS + (ACID_CONTROLS if voice == "acid" else ())
    unknown = set(controls) - set(names)
    if unknown:
        raise ValueError(f"unknown controls: {sorted(unknown)}")
    patch = {name: float(controls.get(name, 0.5)) for name in names}
    return {"voice": voice, "patch": patch, "gain": float(gain),
            "pattern": {"notes": list(notes), "vel": [0.85 if n is not None else 0 for n in notes],
                        "len": [1] * len(notes)}}


def kit_token(device, hits, gain=1.0, **knobs):
    """A pool kit: a drum machine model ("606", "808" or "909"), its knobs per voice (``kick_tune=0.4`` ...,
    each 0..1, 0.5 where not given) and the hits of the bar it was fitted to ({voice: 16 levels 0..1})."""
    patch = {f"{voice}_{knob}": float(knobs.get(f"{voice}_{knob}", 0.5)) for voice in KIT_VOICES for knob in KIT_KNOBS}
    patch.update(level=float(knobs.get("level", 0.8)), pan=0.5, drive=float(knobs.get("drive", 0.0)),
                 send_delay=0.0, send_reverb=0.0)
    return {"voice": "drums", "device": str(device), "patch": patch, "gain": float(gain),
            "pattern": {"hits": {voice: list(row) for voice, row in hits.items()}}}


def pool_of(kits, bass, lead, air, bpm=126.0):
    """Four lists of sounds -> a pool file's content. Sound 0 of every layer is silence."""
    layers = [{"name": name, "tokens": [None, *sounds]}
              for name, sounds in (("beat", kits), ("bass", bass), ("lead", lead), ("air", air))]
    return {"version": 1, "bpm": float(bpm), "layers": layers,
            "fx": {"delay_steps": 3, "delay_feedback": 0.45, "delay_tone": 0.5, "reverb_size": 0.7, "reverb_damp": 0.5}}


def fit_sound(stem_bar, sample_rate, layer, render):
    """One bar of a stem -> the synth sound whose render is closest to it.

    Contract: ``render(token)`` plays a candidate through the engine (engine/render.mjs); the fit compares
    spectra and envelopes, not waveforms. Fit a few hundred bars per layer and reduce them to a pool of a size
    the kit slot and the lessons can carry (64 per layer here) by clustering the fitted controls."""
    raise NotImplementedError("search the controls so that the engine's render matches the bar")


def match_bars(stem_bars, pool_layer):
    """Every sounding bar of a stem -> the id of a pool sound, and its filter and space level (-3..3).

    Contract: a bar far from every pool sound is better marked ``controls_open`` than taught the nearest sound.
    The level against the sound's own setting did not beat "always the sound's own setting" when it was measured
    (docs/score.md); smooth it over runs of one sound and score it against a truth before you teach it."""
    raise NotImplementedError("compare each bar's spectrum, density and register with the pool's sounds")
