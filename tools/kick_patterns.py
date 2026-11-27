"""The kick patterns a composed track plays: the commonest step sets with their bar counts and the long runs.
    python tools/kick_patterns.py site/tracks/<state>/*.json
A first-hour echo state holds one kick pattern for 100 to 300 bars, whatever it was at about bar 50; a track
whose long run lacks one of the quarters 0, 4, 8, 12 has lost a kick."""
import collections
import json
import sys

for f in sys.argv[1:]:
    pats = []
    for b in json.load(open(f))["bars"]:
        k = None
        for p in (b.get("patterns") or []):
            if isinstance(p, dict) and isinstance(p.get("hits"), dict):
                k = p["hits"].get("kick")
        pats.append(tuple(i for i, v in enumerate(k or []) if v and v > 0))
    runs, cur, n, start = [], None, 0, 0
    for i, p in enumerate(pats):
        if p == cur:
            n += 1
        else:
            if cur is not None:
                runs.append((start + 1, n, cur))
            cur, n, start = p, 1, i
    runs.append((start + 1, n, cur))
    four = sum(1 for p in pats if {0, 4, 8, 12} <= set(p)) / max(1, len(pats))
    print("/".join(f.split("/")[-2:]), f"bars with all four quarters {four:.0%}; longest run {max(n for _, n, _ in runs)} bars")
    print("   commonest:", [(list(p), c) for p, c in collections.Counter(pats).most_common(4)])
    print("   runs of 12 or more (from bar, bars, steps):", [(s, c, list(p)) for s, c, p in runs if c >= 12][:8])
