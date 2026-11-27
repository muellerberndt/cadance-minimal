"""A small synthetic score and sound pool in the real format: programmed drum patterns and lines, no recording.

    python -m pipeline.toy_score OUT_DIR [--tracks 8] [--bars 128] [--seed 0] [--check]

writes OUT_DIR/toy_score.json and OUT_DIR/toy_pool.json, validates both (score_format.py) and, with --check,
loads them the way the trainer does (brain/common.py load_streams with the settings of the kept brains: drum
steps, line steps, sound controls from the pool, the bar clock). It is a format example and a smoke test of
your setup. Its music is a loop with fills, a breakdown and returns; a brain trained on it has learned nothing
about real material.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import score_format, sounds
from .lines import encode_line

SCALE = (0, 3, 5, 7, 10)                        # minor pentatonic, as steps above the track's root


def _kit_rows(rng, kind):
    """The home bar of a track's drums: kick, snare, hats x 16."""
    kick = np.zeros(16, dtype=int)
    kick[[0, 4, 8, 12]] = 1
    kick[0] = 2
    if kind % 3 == 1:
        kick[11] = 1                             # a pushed kick before the last beat
    snare = np.zeros(16, dtype=int)
    if kind % 4 != 3:
        snare[[4, 12]] = 2 if kind % 2 else 1    # the backbeat
    hats = np.zeros(16, dtype=int)
    if kind % 2:
        hats[:] = 1
        hats[[2, 6, 10, 14]] = 2                 # sixteenths, the off-beats accented
    else:
        hats[[2, 6, 10, 14]] = 2                 # off-beat hats
    if rng.random() < 0.5:
        hats[15] = 1
    return np.stack([kick, snare, hats])


def _motif(rng, root, octave, density, held=False):
    """One bar of a line as encode_line reads it: 16 entries of None, "tie" or a MIDI number."""
    notes = [None] * 16
    if held:
        notes[0] = 12 * (octave + 1) + root + int(rng.choice(SCALE[:4]))
        notes[1:] = ["tie"] * 15
        return notes
    steps = sorted(rng.choice(16, size=density, replace=False).tolist())
    for step in steps:
        notes[step] = 12 * (octave + 1) + root + int(rng.choice(SCALE))
        if step + 1 < 16 and step + 1 not in steps and rng.random() < 0.4:
            notes[step + 1] = "tie"
    return notes


def _vary(rng, notes, root, octave):
    """A variant of a bar: one note moved in pitch and one step switched."""
    out = list(notes)
    played = [i for i, v in enumerate(out) if isinstance(v, int)]
    if played:
        out[int(rng.choice(played))] = 12 * (octave + 1) + root + int(rng.choice(SCALE))
    step = int(rng.integers(16))
    out[step] = None if out[step] is not None else 12 * (octave + 1) + root + int(rng.choice(SCALE))
    return out


def track(index, bars, rng, sounds_per_layer=4):
    """One synthetic recording of ``bars`` bars: a loop that builds up, with a fill every eighth bar, a varied bar
    every fourth, a breakdown in each block of 64 bars and the home bar returning after each of them."""
    root = int(rng.integers(12))
    home = _kit_rows(rng, index)
    bass_a = _motif(rng, root, 1, density=int(rng.integers(4, 9)))
    bass_b = _vary(rng, bass_a, root, 1)
    lead_a = _motif(rng, root, 4, density=int(rng.integers(3, 7)))
    lead_b = _vary(rng, lead_a, root, 4)
    air = [_motif(rng, root, 3, 1, held=True) for _ in range(2)]
    ids = [1 + (index + shift) % sounds_per_layer for shift in range(4)]       # kit, bass, lead, air
    state = np.zeros((bars, 4), dtype=int)
    drums = np.zeros((bars, 3, 16), dtype=int)
    synth = np.zeros((bars, 3, 16), dtype=int)
    octave = np.full((bars, 3), -1, dtype=int)
    mods = np.zeros((bars, 8), dtype=int)
    section = np.zeros(bars, dtype=int)
    for b in range(bars):
        place = b % 64
        breakdown, quiet = 48 <= place < 56, 48 <= place < 52
        section[b] = int(breakdown)
        if not quiet:                                                          # the beat
            state[b, 0] = ids[0]
            rows = home.copy()
            if place < 8:
                rows[1:] = 0                                                   # the kick alone
            elif place < 16:
                rows[1] = 0                                                    # kick and hats
            if breakdown:
                rows[0] = 0                                                    # the kick is out
                rows[1, 8:] = 1                                                # a snare roll into the return
            if b % 8 == 7 and not breakdown:
                rows[1, [13, 14, 15]] = 1                                      # a fill, the same each time
                rows[0, 14] = 1
            drums[b] = rows
        lines = [None, None, None]
        if place >= 8 and not quiet:
            lines[0] = bass_b if b % 4 == 3 else bass_a
        if place >= 16:
            lines[1] = lead_b if b % 8 >= 6 else lead_a
        if place >= 32:
            lines[2] = air[(b // 8) % 2]
        for layer, notes in enumerate(lines):
            if notes is None:
                continue
            state[b, 1 + layer] = ids[1 + layer]
            synth[b, layer], octave[b, layer] = encode_line(notes)
            sweep = int(round(2 * np.sin(2 * np.pi * b / 32))) if layer == 0 else int(place >= 56) * 2
            mods[b, 2 * (layer + 1)] = sweep                                   # the layer's filter level
    sounding = (state > 0).sum(axis=1)
    hat_hits = (drums[:, 2] > 0).sum(axis=1)
    mood = np.column_stack([np.clip(0.2 + 0.18 * sounding, 0, 1), np.clip(0.25 + 0.03 * hat_hits + 0.15 * (state[:, 2] > 0), 0, 1)])
    name = f"toy_{index:02d}"
    return dict(id=name, title=name, bpm=126.0, held_out=False, group="reference" if index % 2 == 0 else "second",
                style="reference", bars=state, mood=np.round(mood, 3), mods=mods, drums=drums, synth=synth,
                synth_oct=octave, section=section, drums_sure=np.ones((bars, 3), dtype=int),
                synth_sure=np.ones((bars, 3), dtype=int), drums_open=np.zeros((bars, 3, 16), dtype=int),
                synth_open=np.zeros((bars, 3, 16), dtype=int), controls_open=np.zeros((bars, 3), dtype=int))


def toy_pool(rng, per_layer=4):
    """Four kits and four sounds per synth layer, with controls drawn around neutral settings."""
    devices = ("606", "808", "909", "808")
    kits = [sounds.kit_token(devices[i % 4], {"kick": [1, 0, 0, 0] * 4, "snare": [0, 0, 0, 0, 1, 0, 0, 0] * 2,
                                              "ch": [0, 0, 1, 0] * 4},
                             kick_tune=round(float(rng.uniform(0.3, 0.7)), 3), kick_decay=round(float(rng.uniform(0.3, 0.8)), 3))
            for i in range(per_layer)]

    def layer(octave, voices, **fixed):
        return [sounds.synth_token([12 * (octave + 1)] + [None] * 15, voice=voices[i % len(voices)],
                                   cutoff=round(float(rng.uniform(0.2, 0.9)), 3), reso=round(float(rng.uniform(0.1, 0.7)), 3),
                                   decay=round(float(rng.uniform(0.2, 0.8)), 3), **fixed)
                for i in range(per_layer)]

    return sounds.pool_of(kits, layer(1, ("synth", "acid")), layer(4, ("synth",)), layer(3, ("synth",), attack=0.6, release=0.8))


def build(out, tracks=8, bars=128, seed=0):
    rng = np.random.default_rng(seed)
    pool = toy_pool(rng)
    made = [track(i, bars, rng) for i in range(tracks)]
    for held in made[-2:]:                                                     # one recording of each group
        held["held_out"] = True
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "toy_pool.json").write_text(json.dumps(pool))
    score = score_format.write_score(out / "toy_score.json", made, tokens_per_layer=len(pool["layers"][0]["tokens"]),
                                     made_by="pipeline/toy_score.py", seed=seed)
    problems = score_format.validate_pool(pool) + score_format.validate_score(score, pool)
    return out / "toy_score.json", out / "toy_pool.json", problems


def check(score_path, pool_path, brain_dir):
    """Load the two files as the trainer does and say what it sees. No brain is composed."""
    sys.path.insert(0, str(brain_dir))
    import common
    import timbre
    book = timbre.teachers(json.loads(Path(pool_path).read_text()))
    common.SYNTH_VOCAB.update(kind="steps")
    common.SYNTH_FILL = True
    tracks = common.load_streams(score_path, steps=True, clock=True, synth=True, patch_book=book)
    assert all(t["bars"].shape == (len(t["mood"]), common.SLOTS) for t in tracks)
    train = [t for t in tracks if not t["held_out"]]
    windows = common.windows(train, 64, 16)
    width = (common.LIVE - common.L - common.MODS - common.STEPS - common.SYN) // 3
    return {"slots": common.SLOTS, "sound slots": common.L, "level slots": common.MODS, "drum steps": common.STEPS,
            "line steps": common.SYN, "controls per synth": width, "motor width": common.ACTIONS,
            "observation width": common.INPUTS, "recordings": len(tracks), "held out": len(tracks) - len(train),
            "bars": int(sum(len(t["bars"]) for t in tracks)), "lesson windows of 64 bars": len(windows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tracks", type=int, default=8)
    ap.add_argument("--bars", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="load the files as the trainer does")
    ap.add_argument("--brain", default=str(Path(__file__).resolve().parents[1] / "brain"), help="the folder of common.py")
    args = ap.parse_args()
    score_path, pool_path, problems = build(args.out, args.tracks, args.bars, args.seed)
    print(score_path, pool_path)
    for line in problems:
        print("problem:", line)
    if args.check and not problems:
        for key, value in check(score_path, pool_path, args.brain).items():
            print(f"  {key}: {value}")
    sys.exit(1 if problems else 0)
