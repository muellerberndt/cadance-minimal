"""The score and the sound pool as the trainer reads them (brain/common.py load_streams, brain/timbre.py teachers):
their shapes and value ranges, a writer, a validator, and the split into training and held-out recordings.

A score is one JSON file. Per recording it holds one row per bar: which sounds play, the drum steps, the line
steps and what the reader could not decide. Nothing in it is audio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAYERS = ("beat", "bass", "lead", "air")        # the four sound slots of a bar, in this order
DRUM_ROWS = ("kick", "snare", "hats")           # snare row: snare or clap; hat row: closed or open hat or cymbal
SYNTH_LAYERS = ("bass", "lead", "air")
STEPS = 16                                      # sixteenths per bar
MODS = 8                                        # filter and space level per layer: [f_beat, s_beat, f_bass, ...]
REST, TIE, FIRST_PITCH = 0, 1, 2                # a line step: rest, tie, or 2 + pitch class (2..13) for a new note
MARKS = ("drums_sure", "synth_sure", "drums_open", "synth_open", "controls_open")


def write_score(path, tracks, tokens_per_layer, mod_levels=7, window_shares=None, **notes):
    """Write a score file. ``tracks`` are dicts as ``validate_score`` describes; arrays may be NumPy arrays.
    ``tokens_per_layer`` is the number of kit ids, silence (0) included. ``notes`` are extra top-level entries
    (how the score was made); the trainer ignores them."""
    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    score = {"version": 1, "tokens_per_layer": int(tokens_per_layer), "mod_levels": int(mod_levels),
             "tracks": [plain(track) for track in tracks], **plain(notes)}
    if window_shares:
        score["window_shares"] = plain(window_shares)
    Path(path).write_text(json.dumps(score))
    return score


def _array(track, key, problems, where, shape, low, high, kind=int):
    try:
        value = np.asarray(track[key], dtype=float)
    except (TypeError, ValueError):
        problems.append(f"{where}: {key} is not a rectangular array of numbers")
        return None
    if value.shape != shape:
        problems.append(f"{where}: {key} has shape {value.shape}, expected {shape}")
        return None
    if kind is int and not np.array_equal(value, np.round(value)):
        problems.append(f"{where}: {key} must hold whole numbers")
    if value.size and (value.min() < low or value.max() > high):
        problems.append(f"{where}: {key} leaves {low}..{high} (found {value.min():g}..{value.max():g})")
    return value


def validate_score(score, pool=None, window=64):
    """The problems of a score, as a list of sentences (empty: the trainer's loader takes it).

    Top level: ``tokens_per_layer`` (kit ids, 0 = no beat), ``mod_levels`` (odd, 7), ``tracks``.
    Per track, with n bars:
      id          a string, unique in the score
      held_out    true for a recording kept for testing; whole recordings, never single bars
      bars        n x 4   the sound of beat, bass, lead, air; 0 = the layer is silent in this bar
      mood        n x 2   energy and brightness, each 0..1
      mods        n x 8   optional; filter and space level per layer around the sound's own setting, -3..3
      drums       n x 3 x 16   kick, snare, hats; 0 none, 1 hit, 2 accent
      synth       n x 3 x 16   bass, lead, air; 0 rest, 1 tie, 2..13 a new note at pitch class 0..11
      synth_oct   n x 3   optional; the octave of each line in the bar (MIDI // 12 - 1), -1 unknown
      group       optional; the material group, for lesson shares and for REFERENCE_GROUP
      style       optional; one style in the whole score adds no input, several add a one-hot style input
      drums_sure, synth_sure   n x 3   optional; 1 where the reader was sure of the row or line
      drums_open, synth_open   n x 3 x 16   optional; 1 where a step was left undecided (no lesson there)
      controls_open            n x 3   optional; 1 where a layer's sound controls were not read
      section     n       optional; 0 normal, 1 breakdown, 2 build
    With ``pool`` the sound ids are checked against it: every id a track uses must be a fitted sound.
    ``window`` is the lesson window in bars; a shorter recording gives no lesson."""
    problems = []
    tokens, levels = score.get("tokens_per_layer"), score.get("mod_levels", 7)
    if not isinstance(tokens, int) or tokens < 2:
        problems.append("tokens_per_layer must be a whole number of at least 2")
        tokens = 2
    if not isinstance(levels, int) or levels < 3 or levels % 2 == 0:
        problems.append("mod_levels must be odd and at least 3")
        levels = 7
    tracks = score.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        return problems + ["tracks must be a list with at least one recording"]
    span = (levels - 1) // 2
    table = [layer["tokens"] for layer in pool["layers"]] if pool else None
    seen, groups, styles = set(), set(), set()
    for index, track in enumerate(tracks):
        where = f"track {index}"
        if not isinstance(track.get("id"), str) or track["id"] in seen:
            problems.append(f"{where}: id must be a string that no other track has")
        seen.add(track.get("id"))
        for key in ("bars", "mood", "drums", "synth"):
            if key not in track:
                problems.append(f"{where}: {key} is missing")
        if "bars" not in track:
            continue
        n = len(track["bars"])
        if n < window + 1:
            problems.append(f"{where}: {n} bars give no lesson window of {window} bars")
        bars = _array(track, "bars", problems, where, (n, len(LAYERS)), 0, 10 ** 6)
        if bars is not None and bars[:, 0].max(initial=0) >= tokens:
            problems.append(f"{where}: a kit id reaches tokens_per_layer ({tokens})")
        if bars is not None and table is not None:
            for layer in range(len(LAYERS)):
                for sound in np.unique(bars[:, layer]).astype(int):
                    if sound >= len(table[layer]) or (sound > 0 and table[layer][sound] is None):
                        problems.append(f"{where}: {LAYERS[layer]} sound {sound} is not in the pool")
        if "mood" in track:
            _array(track, "mood", problems, where, (n, 2), 0.0, 1.0, kind=float)
        if track.get("mods") is not None:
            _array(track, "mods", problems, where, (n, MODS), -span, span)
        if "drums" in track:
            _array(track, "drums", problems, where, (n, len(DRUM_ROWS), STEPS), 0, 2)
        if "synth" in track:
            _array(track, "synth", problems, where, (n, len(SYNTH_LAYERS), STEPS), 0, FIRST_PITCH + 11)
        if track.get("synth_oct") is not None:
            _array(track, "synth_oct", problems, where, (n, len(SYNTH_LAYERS)), -1, 8)
        for key, shape in (("drums_sure", (n, 3)), ("synth_sure", (n, 3)), ("drums_open", (n, 3, STEPS)),
                           ("synth_open", (n, 3, STEPS)), ("controls_open", (n, 3))):
            if track.get(key) is not None:
                _array(track, key, problems, where, shape, 0, 1)
        if track.get("section") is not None and len(track["section"]) != n:
            problems.append(f"{where}: section needs one entry per bar")
        groups.add(track.get("group"))
        if track.get("style"):
            styles.add(track["style"])
    held = [bool(track.get("held_out")) for track in tracks]
    if all(held) or not any(held):
        problems.append("the score needs training recordings and held-out recordings")
    unknown = set(score.get("window_shares") or {}) - groups
    if unknown:
        problems.append(f"window_shares names groups no track has: {sorted(unknown)}")
    return problems


def validate_pool(pool):
    """The problems of a sound pool, as a list of sentences.

    ``pool["layers"]`` is four dicts named beat, bass, lead and air, each with ``tokens``: a list whose entry 0 is
    null (silence) and whose other entries are sounds, or null for an id no bar uses. A synth sound has ``voice``
    ("synth" or "acid"), ``patch`` (unit controls, each 0..1), ``gain`` (positive) and ``pattern`` with ``notes``
    (16 MIDI numbers or nulls). A kit has ``voice`` "drums", ``device``, ``patch`` and ``pattern.hits``."""
    problems = []
    layers = pool.get("layers") or []
    if [layer.get("name") for layer in layers] != list(LAYERS):
        return ["the pool's layers must be named beat, bass, lead and air, in this order"]
    for layer in layers:
        tokens = layer.get("tokens") or [None]
        if tokens[0] is not None:
            problems.append(f"{layer['name']}: sound 0 is silence and must be null")
        for index, token in enumerate(tokens[1:], start=1):
            if token is None:
                continue
            where = f"{layer['name']} sound {index}"
            gain = token.get("gain", 1.0)
            if not isinstance(gain, (int, float)) or not np.isfinite(gain) or gain <= 0:
                problems.append(f"{where}: gain must be a positive number")
            if layer["name"] == "beat":
                if not isinstance((token.get("pattern") or {}).get("hits"), dict):
                    problems.append(f"{where}: a kit needs pattern.hits")
                continue
            if token.get("voice", "synth") not in ("synth", "acid"):
                problems.append(f"{where}: voice must be synth or acid")
            for name, value in (token.get("patch") or {}).items():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    problems.append(f"{where}: control {name} must lie in 0..1")
            notes = (token.get("pattern") or {}).get("notes")
            if notes is not None and len(notes) != STEPS:
                problems.append(f"{where}: pattern.notes needs {STEPS} entries")
    return problems


def split_held_out(tracks, share=0.15, seed=0, key="source"):
    """Mark whole sources as held out: every track of a held-out source gets ``held_out`` true. ``key`` names the
    field that says which recording a track was cut from (pieces of one long recording share it); a track without
    it is its own source. About ``share`` of the bars are held out, at least one source. Returns the tracks."""
    rng = np.random.default_rng(seed)
    sources = {}
    for index, track in enumerate(tracks):
        sources.setdefault(track.get(key, track["id"]), []).append(index)
    total = sum(len(track["bars"]) for track in tracks)
    names = sorted(sources, key=str)
    held, bars = set(), 0
    for position in rng.permutation(len(names)):
        name = names[position]
        if bars >= share * total and held:
            break
        if len(held) + 1 == len(sources):                 # never hold out everything
            break
        held.add(name)
        bars += sum(len(tracks[i]["bars"]) for i in sources[name])
    for name, members in sources.items():
        for i in members:
            tracks[i]["held_out"] = name in held
    return tracks
