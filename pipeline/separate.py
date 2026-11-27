"""Stage 1: source separation. A recording becomes four stems: drums, bass, other and vocals.

Every later stage reads one stem, with another stem as a reference for what leaked into it. A later stage can
discount a leak it has a reference for and cannot undo it: a kick's tail in the bass stem reads as a bass note,
a snare in the "other" stem as a lead note.
"""
from __future__ import annotations


def separate(audio, sample_rate):
    """A mono or stereo recording -> {"drums", "bass", "other", "vocals"}, arrays of the same length.

    Contract the later stages rely on: the stems sum to about the recording, all at ``sample_rate``, cut and
    normalised the same way for every recording (the readers' thresholds are relative to a recording's own
    levels, so a per-recording normalisation is fine and a per-chunk one is not)."""
    raise NotImplementedError("run a source separation model of your choice here and return its four stems; "
                              "process long recordings in chunks with overlap and keep the levels of the whole")
