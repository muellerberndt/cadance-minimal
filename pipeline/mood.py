"""Stage 7: the mood rows. Two numbers per bar, energy and brightness, each 0..1. The brain hears the mood wanted
for the bar it writes, and the player's mood pad sets the same two numbers."""
from __future__ import annotations

import numpy as np


def mood_rows(bar_level_db, bar_brightness, reference_db=None, low=None, high=None, span_db=24.0):
    """Per-bar loudness (dB) and a brightness measure (for example the spectral centroid) -> n x 2 mood rows.

    Energy: the bar's level on a scale of ``span_db`` under ``reference_db`` (default: the 95th percentile of
    the levels given). Brightness: between ``low`` and ``high`` (default: the 5th and 95th percentile). Take
    the three references over the whole material, not per recording, so that a quiet recording reads quiet."""
    level = np.asarray(bar_level_db, dtype=float)
    bright = np.asarray(bar_brightness, dtype=float)
    reference = np.percentile(level, 95) if reference_db is None else reference_db
    low = np.percentile(bright, 5) if low is None else low
    high = np.percentile(bright, 95) if high is None else high
    energy = np.clip((level - reference + span_db) / span_db, 0, 1)
    brightness = np.clip((bright - low) / max(0.1, high - low), 0, 1)
    return np.column_stack([energy, brightness])
