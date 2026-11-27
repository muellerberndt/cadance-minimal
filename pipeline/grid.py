"""Stage 2: the grid. Tempo, beats, the first beat of the bar, and from them the sixteenth lattice every reader
uses. A grid that is right in tempo and wrong in bar start gives a score whose every bar is rotated: the readers
still score well step by step and the brain learns bars that begin on beat 2.
"""
from __future__ import annotations

import numpy as np


def estimate_tempo_and_beats(drum_stem, sample_rate, low=85.0, high=185.0):
    """A drum stem -> (bpm, beat times in seconds).

    Contract: one constant tempo per passage (cut a DJ set into passages first), beat times within about 20 ms of
    the kick's attack. Check the result against a click you can hear or against clips whose tempo you know."""
    raise NotImplementedError("score candidate tempos between low and high by how well a beat lattice explains "
                              "the stem's onsets, then place the beats by onset phase")


def find_bar_start(beats, stems, drum_probabilities=None):
    """Beat times and the stems -> (index 0..3 of the beat that starts the bar, confidence 0..1).

    Evidence that worked: where stems enter and leave, where the spectrum changes, snare or clap on beats 2 and
    4, kick fills at a bar's end, the kick returning on a 1. Combine them and keep the confidence: a passage
    whose bar start is unsure is better left out than taught rotated."""
    raise NotImplementedError("score the four candidate phases from the evidence above; fit the weights on "
                              "clips whose bar start you know")


def bar_starts(beats, phase, beats_per_bar=4):
    """Beat times and the bar phase -> the start time of every whole bar."""
    beats = np.asarray(beats, dtype=float)
    return beats[int(phase) % beats_per_bar::beats_per_bar]


def sixteenth_lattice(starts, steps=16):
    """Bar start times (n + 1 values for n bars) -> the time of every step, n x steps."""
    starts = np.asarray(starts, dtype=float)
    if len(starts) < 2:
        raise ValueError("a lattice needs the start of at least two bars")
    fractions = np.arange(steps) / steps
    return starts[:-1, None] + np.diff(starts)[:, None] * fractions[None, :]


def quantise_onsets(times, levels, starts, steps=16, tolerance=0.4, accent_quantile=0.75):
    """Onsets of one voice -> its step row per bar: n x steps of 0 (none), 1 (hit), 2 (accent).

    ``times`` are onset times in seconds, ``levels`` their strengths (any monotone measure, for example dB).
    An onset goes to the nearest step when it lies within ``tolerance`` of a step length, else it is dropped.
    Two onsets on one step keep the stronger. An accent is a hit at or above the ``accent_quantile`` of the
    levels of all placed hits of this voice in the recording."""
    lattice = sixteenth_lattice(starts, steps)
    n = len(lattice)
    flat = lattice.reshape(-1)
    step_length = np.repeat(np.diff(np.asarray(starts, dtype=float)) / steps, steps)
    best = np.full(n * steps, -np.inf)
    for time, level in zip(np.asarray(times, dtype=float), np.asarray(levels, dtype=float)):
        index = int(np.argmin(np.abs(flat - time)))
        if abs(flat[index] - time) <= tolerance * step_length[index] and level > best[index]:
            best[index] = level
    placed = np.isfinite(best)
    rows = placed.astype(int)
    if placed.any():
        rows[placed & (best >= np.quantile(best[placed], accent_quantile))] = 2
    return rows.reshape(n, steps)
