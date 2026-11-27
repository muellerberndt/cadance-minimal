"""Stage 5: the pooled reading. Use the repetition the music has.

A reader decides every sixteenth of every bar alone, so a step whose probability sits near the threshold flips
from bar to bar although the record plays the same loop. Here a bar's step probabilities are pooled with those
of the bars at the same place of the loop around it, as long as those bars read like it. A step whose own
probability is decisive keeps its own decision, so a fill or a change of pattern survives. A step that stays
undecided can be left open: the trainer teaches no lesson on an open step.
"""
from __future__ import annotations

import numpy as np

LAGS = (1, 2, 4, 8)


def loop_lengths(probabilities, reach=16, lags=LAGS, slack=0.01):
    """bars x 16 step probabilities of one row -> per bar the loop length in bars: the shortest lag at which the
    row agrees with itself about as well as at the best lag, over the bars around it."""
    P = np.asarray(probabilities, dtype=float)
    n = len(P)
    cost = np.full((len(lags), n), np.inf)
    for i, lag in enumerate(lags):
        if n <= lag:
            continue
        full = np.full(n, np.nan)
        full[lag:] = np.abs(P[lag:] - P[:-lag]).mean(axis=1)
        sums = np.nancumsum(np.r_[0.0, full])
        counts = np.cumsum(np.r_[0, np.isfinite(full).astype(int)])
        low, high = np.maximum(0, np.arange(n) - reach), np.minimum(n, np.arange(n) + reach + 1)
        k = counts[high] - counts[low]
        cost[i] = np.where(k > 0, (sums[high] - sums[low]) / np.maximum(k, 1), np.inf)
    best = cost.min(axis=0)
    out = np.ones(n, dtype=int)
    for b in range(n):
        for i, lag in enumerate(lags):
            if cost[i, b] <= best[b] + slack:
                out[b] = lag
                break
    return out


def pool(probabilities, reach=8, similar=0.1, lags=LAGS):
    """bars x 16 step probabilities -> (pooled probabilities, number of bars pooled with each bar). A bar is
    pooled with the bars whole loops before and after it, up to ``reach`` bars away, whose probabilities differ
    from its own by at most ``similar`` on average."""
    P = np.asarray(probabilities, dtype=float)
    n = len(P)
    if n < 2:
        return P.copy(), np.zeros(n, dtype=int)
    loop = loop_lengths(P, lags=lags)
    out, size = P.copy(), np.zeros(n, dtype=int)
    for b in range(n):
        length = int(loop[b])
        near = [b + k * length for k in range(-(reach // length), reach // length + 1)
                if k != 0 and 0 <= b + k * length < n]
        near = [c for c in near if np.abs(P[c] - P[b]).mean() <= similar]
        if near:
            out[b] = (P[b] + P[near].sum(axis=0)) / (1 + len(near))
            size[b] = len(near)
    return out, size


def read(probabilities, threshold, reach=8, similar=0.1, low=0.1, high=0.9, open_band=None):
    """One row's step probabilities -> (hits bars x 16 bool, open steps bars x 16 bool).

    A step at or under ``low`` is no hit and at or over ``high`` a hit, whatever the pool says. In between the
    pooled probability decides against ``threshold``. With ``open_band`` (a pair of probabilities) an in-between
    step whose pooled probability lies inside the band is marked open. Choose the parameters per row on clips
    whose rows you know, and score on other clips."""
    P = np.asarray(probabilities, dtype=float)
    pooled, _ = pool(P, reach=reach, similar=similar) if reach else (P.copy(), None)
    undecided = (P > low) & (P < high)
    hits = np.where(P >= high, True, np.where(P <= low, False, pooled >= threshold))
    opened = np.zeros_like(hits)
    if open_band is not None:
        opened = undecided & (pooled > open_band[0]) & (pooled < open_band[1])
    return hits, opened
