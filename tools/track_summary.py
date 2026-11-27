"""Counts of a composed shelf track (site/tracks/<state>/*.json): kicks, snare, hats, bass and lead notes per bar by
segment, drum bars equal to the bar 1 / 2 / 4 / 8 back, drum bars changing per 16, the longest held bass and lead bar.
    python tools/track_summary.py site/tracks/<state>/<file>.json [...]
"""
import json, sys
import numpy as np

SEGS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 304))

def counts(bar):
    pats = bar.get("patterns") or []
    tok = bar.get("tokens") or [0, 0, 0, 0]
    hits = (pats[0] or {}).get("hits", {}) if pats else {}
    def n(row): return int(sum(1 for v in hits.get(row, []) if v and v > 0))
    def notes(i): return int(sum(1 for v in ((pats[i] or {}).get("notes") or []) if v is not None)) if len(pats) > i else 0
    drums = tuple(tuple(1 if v and v > 0 else 0 for v in hits.get(r, [])) for r in ("kick", "snare", "oh"))
    line = lambda i: tuple((pats[i] or {}).get("notes") or []) if len(pats) > i else ()
    return {"kick": n("kick") if tok[0] > 0 else None, "snare": n("snare") if tok[0] > 0 else None, "hats": n("oh") if tok[0] > 0 else None,
            "bass": notes(1) if tok[1] > 0 else None, "lead": notes(2) if tok[2] > 0 else None,
            "drums": drums, "bassline": line(1), "leadline": line(2), "voices": sum(1 for t in tok if t > 0)}

def held(lines):
    best = run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b and any(v is not None for v in a) else 1
        best = max(best, run)
    return best

for path in sys.argv[1:]:
    t = json.load(open(path)); rows = [counts(b) for b in t["bars"]]
    print(f"{path.split('/')[-1]}  bars {len(rows)}  passage {t.get('seed')!r}")
    for a, b in SEGS:
        part = rows[a:b]
        if not part: continue
        def mean(k):
            v = [r[k] for r in part if r[k] is not None]
            return f"{np.mean(v):4.1f}" if v else "  - "
        print(f"   bars {a+1:>3}-{min(b, len(rows)):>3}: kick {mean('kick')} snare {mean('snare')} hats {mean('hats')} bass {mean('bass')} lead {mean('lead')} voices {np.mean([r['voices'] for r in part]):.1f}")
    d = [r["drums"] for r in rows]
    same = lambda k: float(np.mean([d[i] == d[i - k] for i in range(k, len(d))]))
    changing = float(np.mean([len(set(d[i:i + 16])) for i in range(0, len(d) - 15, 16)]))
    capped = {}
    for b in t["bars"]:
        for c in b.get("capped") or []: capped[c] = capped.get(c, 0) + 1
    print(f"   drums equal 1/2/4/8 back: {same(1):.2f} {same(2):.2f} {same(4):.2f} {same(8):.2f}; different drum bars per 16: {changing:.1f}; "
          f"longest held bass bar {held([r['bassline'] for r in rows])}, lead bar {held([r['leadline'] for r in rows])}; capped bars {capped}")
