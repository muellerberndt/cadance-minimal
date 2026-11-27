"""Whole-track means of the shelf's tracks, one line per track, from track_summary.py's output on stdin:
    python tools/track_summary.py site/tracks/<state>/*.json | python tools/track_means.py
Columns: stamp, kick / snare / hats / bass / lead per bar over the whole track, drums equal to the previous bar,
different drum bars per 16, longest held bass / lead bar, passage."""
import re
import sys

cur, acc = None, {}
for line in sys.stdin:
    m = re.match(r"(\S+\.json)\s+bars \d+\s+passage (.*)", line)
    if m:
        cur = m.group(1)
        acc[cur] = {"p": m.group(2)[:44], "n": 0, "s": {}, "d": ("-",) * 4}
        continue
    m = re.match(r"\s+bars\s+(\d+)-\s*(\d+): kick\s+([\d.]+) snare\s+([\d.]+) hats\s+([\d.]+) bass\s+([\d.]+) lead\s+([\d.]+)", line)
    if m and cur:
        w = int(m.group(2)) - int(m.group(1)) + 1
        a = acc[cur]
        a["n"] += w
        for k, v in zip(("kick", "snare", "hats", "bass", "lead"), m.groups()[2:]):
            a["s"][k] = a["s"].get(k, 0) + w * float(v)
    m = re.match(r"\s+drums equal 1/2/4/8 back: ([\d.]+).*per 16: ([\d.]+); longest held bass bar (\d+), lead bar (\d+)", line)
    if m and cur:
        acc[cur]["d"] = m.groups()
for k, a in acc.items():
    means = " ".join(f"{x} {a['s'][x] / max(1, a['n']):4.1f}" for x in ("kick", "snare", "hats", "bass", "lead"))
    print(k, means, "eq1 %s diff16 %s held %s/%s" % a["d"], a["p"])
