"""A check on a finished score: how many of its changes are returns.

A slot changes when its value differs from the bar before. A change is a return when the new value is the one
the slot held 2, 4, 8 or 16 bars earlier. A score of loop-based music has many returns, and a brain that hears
those earlier bars can answer them (docs/brains.md). A score with few returns is either other music or a reader
that flickers.

    python -m pipeline.returns SCORE.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

LAGS = (2, 4, 8, 16)


def lag_returns(tracks, lags=LAGS):
    """Score tracks -> per slot group ("drums", "lines") the share of changed step values that equal the value at
    one of ``lags``, the share of step values that change at all, and the error of copying each lag.
    Drum steps count in bars whose kit sounds, line steps in bars whose layer sounds."""
    reach = max(lags)
    count = {name: dict(steps=0, changed=0, returned=0, copy={k: 0 for k in (1, *lags)}) for name in ("drums", "lines")}
    for track in tracks:
        bars = np.asarray(track["bars"])
        for name, rows, sounding in (("drums", np.asarray(track["drums"]), bars[:, :1] > 0),
                                     ("lines", np.asarray(track["synth"]), bars[:, 1:] > 0)):
            n = len(rows)
            if n <= reach:
                continue
            now, before = rows[reach:], rows[reach - 1:n - 1]
            mask = np.broadcast_to(sounding[reach:, :, None], now.shape)
            changed = (now != before) & mask
            back = np.zeros_like(changed)
            for lag in lags:
                back |= now == rows[reach - lag:n - lag]
            c = count[name]
            c["steps"] += int(mask.sum()); c["changed"] += int(changed.sum()); c["returned"] += int((changed & back).sum())
            for lag in (1, *lags):
                c["copy"][lag] += int(((now != rows[reach - lag:n - lag]) & mask).sum())
    return {name: {"changed": c["changed"] / max(1, c["steps"]),
                   "returns": c["returned"] / max(1, c["changed"]),
                   "copy_error": {lag: wrong / max(1, c["steps"]) for lag, wrong in c["copy"].items()}}
            for name, c in count.items()}


if __name__ == "__main__":
    report = lag_returns(json.loads(open(sys.argv[1]).read())["tracks"])
    for name, row in report.items():
        print(f"{name}: {row['changed']:.1%} of step values change; {row['returns']:.1%} of the changes return to "
              f"a value {', '.join(map(str, LAGS))} bars back; copying lag 1 is wrong on {row['copy_error'][1]:.1%}")
