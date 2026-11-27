"""Run every implemented part of the skeleton once on synthetic input.

    python -m pipeline.selftest

It needs NumPy alone. It does not touch a recording, a stem or a brain.
"""
from __future__ import annotations

import copy
import json
import tempfile

import numpy as np

from . import drums, grid, lines, mood, pooling, returns, score_format, sounds, toy_score


def main():
    with tempfile.TemporaryDirectory() as folder:
        score_path, pool_path, problems = toy_score.build(folder)
        score, pool = json.loads(score_path.read_text()), json.loads(pool_path.read_text())
    assert not problems, problems
    damaged = copy.deepcopy(score)
    damaged["tracks"][0]["drums"][3][1][2] = 5                    # a step value that does not exist
    damaged["tracks"][1]["bars"][0][1] = 9                        # a bass sound the pool does not hold
    del damaged["tracks"][2]["mood"]
    found = score_format.validate_score(damaged, pool)
    assert len(found) == 3, found
    print("score_format: the toy score validates, a damaged copy gives", len(found), "problems")

    starts = np.arange(5) * 2.0                                   # four bars of two seconds
    step = 2.0 / 16
    times = np.array([0.0, 0.51, 0.98, 1.5, 2.0, 2.0 + 0.5 * step, 3.26])
    levels = np.array([-6.0, -12, -12, -12, -5, -30, -11])
    rows = grid.quantise_onsets(times, levels, starts)
    assert rows[0].tolist() == [2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert rows[1, 0] == 2 and rows[1, 10] == 1 and rows[1].sum() == 3      # the onset between two steps is dropped
    three = drums.rows_from_onsets({"kick": (times, levels), "snare": (times[:2], levels[:2]),
                                    "hats": (times[3:], levels[3:])}, starts)
    silenced, sounding = drums.silence_rule(three, [-10, -12, -80, -11])
    assert three.shape == (4, 3, 16) and not silenced[2].any() and sounding.tolist() == [True, True, False, True]
    print("grid, drums: onsets land on their sixteenths, a silent bar is zeroed")

    row, octave = lines.encode_line([48, "tie", None, 51, None, None, 55, "tie", "tie", None, None, None, 48, None, None, None])
    assert row[:4] == [2, 1, 0, 5] and octave == 3 and lines.decode_line(row, octave)[0] == 48
    print("lines: a bar of notes and ties encodes to", row, "at octave", octave)

    rng = np.random.default_rng(0)                                # a four-bar loop read with noise near the threshold
    truth = np.tile(np.array([[1, 0, 0, 0] * 4] * 3 + [[1, 0, 1, 0] * 4]), (8, 1)).astype(bool)
    probabilities = np.clip(np.where(truth, 0.62, 0.38) + rng.normal(0, 0.12, truth.shape), 0, 1)
    alone = probabilities >= 0.5
    pooled, opened = pooling.read(probabilities, 0.5, reach=8, similar=0.2, open_band=(0.45, 0.55))
    assert (pooled != truth).sum() < (alone != truth).sum()
    print(f"pooling: {int((alone != truth).sum())} wrong steps read alone, {int((pooled != truth).sum())} pooled, "
          f"{int(opened.sum())} of {truth.size} left open")

    rows = mood.mood_rows([-30, -18, -12, -10], [800, 1500, 2500, 4000])
    assert rows.shape == (4, 2) and rows.min() >= 0 and rows.max() <= 1
    report = returns.lag_returns(score["tracks"])
    assert report["drums"]["returns"] > 0.5
    print(f"returns: {report['drums']['returns']:.0%} of the toy score's drum changes and "
          f"{report['lines']['returns']:.0%} of its line changes are returns")

    tracks = [dict(id=f"t{i}", source=f"s{i // 2}", bars=[[0] * 4] * (50 + 10 * i)) for i in range(10)]
    held = {t["source"] for t in score_format.split_held_out(tracks, share=0.2, seed=1) if t["held_out"]}
    assert held and len(held) < 5 and all(t["held_out"] == (t["source"] in held) for t in tracks)
    assert len(sounds.synth_token([36] + [None] * 15, voice="acid", cutoff=0.3)["patch"]) == 28
    print("score_format: whole sources are held out together:", sorted(held))
    print("ok")


if __name__ == "__main__":
    main()
