"""Read-only summaries of which audible bars each recall head visits."""
from collections import defaultdict

import numpy as np

import common


class RecallTrace:
    def __init__(self, keep_reads=False):
        self.totals = defaultdict(lambda: defaultdict(float))
        self.reads = [] if keep_reads else None
        self.max_slot_share = 0.0

    def prompt(self, track, position, seed):
        self.track, self.position = track, position
        self.events = []
        self.previous = None
        for bar in seed:
            self._heard(bar)

    def _heard(self, state):
        gates = common.sounds(state) > 0
        sounding = np.concatenate([[gates[0] and common.decode_beat(state)[0].any()],
                                   gates[1:] & common.decode_synth(state).any(axis=1)])
        enters = np.zeros_like(sounding) if self.previous is None else sounding & ~self.previous
        self.events.append((bool(enters[0]), bool(enters[1:].any())))
        self.previous = sounding

    def read(self, recall, state):
        if recall.strength:
            raise ValueError("Tracing absolute bars requires recall to keep its slots in write order")
        cue_bar = self.position + len(self.events)
        heads = recall.attended_heads or ()
        reads = [head for head in heads if head is not None]
        if reads:
            self.max_slot_share = max(self.max_slot_share, float(np.sum(reads, axis=0).max()))
        for index, head in enumerate(heads):
            if head is None:
                continue
            bars_held = min(len(self.events), head.shape[1])
            if not bars_held:
                continue
            weights = head[0, -bars_held:]
            mass = float(weights.sum())
            if not mass:
                continue
            shares = weights / mass
            lags = np.arange(len(shares), 0, -1)
            heard = np.asarray(self.events[-len(shares):])
            features = {"eight_or_sixteen": (lags == 8) | (lags == 16),
                        "kick_entrance": heard[:, 0], "synth_entrance": heard[:, 1]}
            total = self.totals[index]
            total["reads"] += 1
            total["mean_lag"] += float(shares @ lags)
            total["peak_share"] += float(shares.max())
            total["peak_uniform"] += 1 / len(shares)
            for name, mask in features.items():
                total[name] += float(shares[mask].sum())
                total[name + "_uniform"] += float(mask.mean())
            if self.reads is not None:
                top = np.argsort(-shares, kind="stable")[:3]
                self.reads.append({"track": self.track, "cue_bar": cue_bar, "head": index + 1,
                                   "retrieved": [{"bar": cue_bar - int(lags[i]), "bars_ago": int(lags[i]),
                                                  "share": float(shares[i]), "events": [key for key in features if features[key][i]]}
                                                 for i in top]})
        self._heard(state)

    def summary(self):
        return [{"head": index + 1, "reads": int(total["reads"]),
                 **{key: value / total["reads"] for key, value in total.items() if key != "reads"}}
                for index, total in sorted(self.totals.items())]
