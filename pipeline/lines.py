"""Stage 4: the lines. Per bar and synth layer (bass, lead, air) sixteen steps: rest, tie or a new note at a
pitch class, and the octave of the line in the bar.

A step holds 0 (rest), 1 (tie: the note before it goes on) or 2 + pitch class (a new note, 2..13). The octave is
kept per bar and layer, not per note. A tie after a note makes it longer and legato; accents of a line are not in
the score.
"""
from __future__ import annotations

import numpy as np

REST, TIE, FIRST_PITCH = 0, 1, 2


def encode_line(notes):
    """One bar of a line -> (row of 16 step values, octave).

    ``notes`` has 16 entries: None for a rest, "tie" where the note before goes on, or a MIDI number for a new
    note. The octave is the median of the bar's notes (MIDI // 12 - 1), or -1 for a bar without a note."""
    if len(notes) != 16:
        raise ValueError("a bar has 16 steps")
    row, pitches = [], []
    for value in notes:
        if value is None:
            row.append(REST)
        elif value == "tie":
            row.append(TIE)
        else:
            row.append(FIRST_PITCH + int(value) % 12)
            pitches.append(int(value))
    octave = int(np.median(pitches)) // 12 - 1 if pitches else -1
    return row, octave


def decode_line(row, octave):
    """The inverse for listening checks: a step row and its octave -> 16 MIDI numbers, "tie" or None."""
    return [None if value == REST else "tie" if value == TIE else 12 * (octave + 1) + value - FIRST_PITCH
            for value in row]


def read_line(stem, lattice, sample_rate, reference_stem=None):
    """One pitched stem and the lattice -> (rows n x 16, octaves n, class probabilities n x 16 x 3).

    What was built: a classifier of rest, tie and new note per step from the stem's spectral flux, level before
    and after the step and spectral change (a slide changes pitch without an attack), with the drum stem as a
    leakage reference; the pitch measured, not learned (the waveform's period for the bass, the strongest
    pitch class for the lead). The air layer was read as one held note per bar."""
    raise NotImplementedError("classify each step as rest, tie or new note and measure the pitch of new notes")
