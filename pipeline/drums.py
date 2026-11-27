"""Stage 3: the drum rows. Per bar and voice (kick, snare or clap, hats) sixteen steps of 0, 1 or 2.

Two readers were built. The first places detected onsets on the lattice (``rows_from_onsets``). The second
classifies every step of the lattice from features of the drum stem, with the bass and other stems as leakage
references (``read_drum_rows``). The second is the one that was kept. It needs clips whose rows are known for
its training.
"""
from __future__ import annotations

import numpy as np

from .grid import quantise_onsets

VOICES = ("kick", "snare", "hats")


def rows_from_onsets(onsets, starts):
    """{"kick": (times, levels), "snare": ..., "hats": ...} and the bar starts -> n x 3 x 16 drum rows.

    The simple reader: one onset detector per voice in its band of the drum stem (kick below about 150 Hz,
    snare and clap in the mid band, hats above about 6 kHz), each onset on its nearest sixteenth."""
    rows = [quantise_onsets(*onsets[voice], starts) for voice in VOICES]
    return np.stack(rows, axis=1)


def silence_rule(rows, bar_level_db, margin_db=27.0, floor_db=-60.0):
    """Zero the rows of bars whose drum stem is silent: a bar more than ``margin_db`` under the recording's loud
    bars (95th percentile), or under ``floor_db``. A reader finds hits in the noise of a silent stem otherwise."""
    rows = np.array(rows, copy=True)
    level = np.asarray(bar_level_db, dtype=float)
    quiet = level < max(np.percentile(level, 95) - margin_db, floor_db)
    rows[quiet] = 0
    return rows, ~quiet


def step_features(stems, lattice, sample_rate):
    """The stems and the lattice -> one feature vector per step and voice.

    What carried the classifier: per band the level at the step against the recording's loud steps, the rise
    over the last 10 and 30 ms, the decay after 50 and 100 ms, the offset of the peak from the lattice point;
    for the kick the bass stem's level and rise at the same place; for the snare the "other" stem's level in the
    snare band; the same quantities at the neighbouring steps. No step index: a position prior teaches the
    reader the commonest pattern instead of the recording."""
    raise NotImplementedError("compute per-step features from short-time spectra of the stems")


def read_drum_rows(features, model, thresholds):
    """Per-step features -> (rows n x 3 x 16, probabilities n x 3 x 16).

    Contract: the model was fitted on clips whose rows are known and scored on other clips of other material
    (see docs/score.md, "A truth to score against"). Keep the probabilities: the pooled reading and the open
    marks need them."""
    raise NotImplementedError("classify each step per voice and threshold the probability")
