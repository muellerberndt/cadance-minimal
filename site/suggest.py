"""Passage suggestions for the repertoire editor (curator.py, GET /api/track/suggest?id=<track id>).

Per 16-bar passage of a composed track, three scores in 0..1, each with the numbers it is made of:

  like     like the material: the passage's measures (drum hits per bar, hats off the beat, snare on and off the
           backbeat, broken kick, voices present, bass and lead pitches and ranges, different bars per 16) against
           the training material's 16-bar passages by group (the groups the score names; the v2 score the
           brains learn from, training and test passages). The distance to a group is the root mean square of the
           differences, each scaled by that group's spread; the score is the share of the group's own passages that
           lie farther from the group than this one (0.5: as typical as the group's median passage). The nearest
           group counts.
  synth    a synth line: lead and bass that move and mutate (distinct pitches, range, notes per bar neither empty
           nor a wall, bars that differ from the bar before by one to three steps, few rewrites, no voice on one
           note for the passage).
  break    a breakbeat: kick not four on the floor, snare hits off the backbeat and ghost notes, hats busy and
           accented off the beat.

The measures are those of the play report (quality/loop/measure.py play_row and content, the functions
quality/loop/play/material.json is built from), read the same way on the material and on a track.

Suggestions: runs of 2 to 8 passages (32 to 128 bars) where a score stays at or above its mark (a fixed mark, or the
track's own 60th percentile when that is higher), best first, at most six per track; runs that overlap are merged. The first 32 bars are left out (they are nearest to the eight bars of
material the brain was given as its opening context).

    python site/suggest.py site/tracks/<brain>/<track>.json      the scores and suggestions of a track
    python site/suggest.py --material                            the three scores on the material, by group
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
sys.path.insert(0, str(ROOT / "quality" / "loop"))
import measure as M                          # noqa: E402  the play report's measures
from play_report import SETS                 # noqa: E402  the two DJ sets of the material, one group

TRACKS = SITE / "tracks"
SCORE = ROOT / "corpus" / "streams_underground_v2.json"     # the score the brains learn from
REFERENCE = SITE / "suggest_material.json"                    # the material's 16-bar passages, measured (built here)
SECTION, SKIP, SHORTEST, LONGEST, MOST = 16, 2, 2, 8, 6
KINDS = {"like": "like the material", "synth": "synth line", "break": "breakbeat"}
# the measures "like the material" compares, with their names in the editor
LIKE = [("kick_per_bar", "kick hits a bar"), ("snare_per_bar", "snare hits a bar"), ("hats_per_bar", "hat hits a bar"),
        ("hats_offbeat_share", "hats off the beat"), ("snare_backbeat_share", "snare on the backbeat"), ("snare_off_per_bar", "snare off the backbeat a bar"),
        ("kick_broken_bars", "broken kick"), ("with_beat", "beat present"), ("with_bass", "bass present"), ("with_lead_or_pad", "lead or pad present"),
        ("bass_distinct_pitches_per_16", "bass pitches"), ("bass_range_per_16", "bass range"),
        ("lead_distinct_pitches_per_16", "lead pitches"), ("lead_range_per_16", "lead range"),
        ("distinct_drum_bars_per_16", "different drum bars"), ("distinct_bass_bars_per_16", "different bass bars"), ("distinct_lead_bars_per_16", "different lead bars")]
OFF_BACKBEAT = np.ones(16, bool)
OFF_BACKBEAT[[4, 12]] = False
LOCK = threading.Lock()
CACHE = {}                                   # track id -> ((mtime, size, reference mark), result)
_REF = None


def cut(R, a, b):
    return {"drums": R["drums"][a:b], "pitch": R["pitch"][a:b], "sound": R["sound"][a:b]}


def measures(rows):
    """Every number the three scores read, for one passage of rows (measure.rows_of_track / rows_of_passage)."""
    pr, c = M.play_row(rows), M.content(rows)
    n = max(1, len(rows["sound"]))
    D = rows["drums"]
    snare = D[:, 1]
    m = {k: pr.get(k) for k, _ in LIKE if k in pr}
    m["kick_per_bar"] = c.get("kick_per_bar")
    m["snare_off_per_bar"] = float((snare[:, OFF_BACKBEAT] > 0).sum() / n)
    m["ghosts_per_bar"] = float((snare[:, OFF_BACKBEAT] == 1).sum() / n)
    m["snare_busy_bars"] = c.get("snare_busy_bars") or 0.0
    m["kick_present"] = c.get("kick_present", 0.0)
    kick = D[:, 0] > 0
    kb = kick.any(axis=1)                    # not four on the floor: a quarter without a kick (an extra kick is not enough)
    m["kick_misses_quarter"] = float((~kick[kb][:, [0, 4, 8, 12]].all(axis=1)).mean()) if kb.any() else 0.0
    m["snare_backbeat_bars"] = float((snare[:, [4, 12]] > 0).any(axis=1).mean()) if len(snare) else 0.0
    m["hat_hits_per_bar"] = float((D[:, 2] > 0).sum() / n)
    for k in ("hats_accent_offbeat", "hats_accent_onbeat"):
        m[k] = c.get(k)
    for v, name in ((1, "bass"), (2, "lead")):
        P = rows["pitch"][:, v - 1]
        on = (P >= 0).any(axis=1)
        m[f"{name}_present"] = float(on.mean()) if len(on) else 0.0
        m[f"{name}_notes_per_bar"] = float((P[on] >= 0).sum(axis=1).mean()) if on.any() else 0.0
        notes = P[P >= 0]                    # a burst in one or two bars does not make a passage's line: pitches heard in
        bars_of = {}                         # two bars or more, range from the 10th to the 90th percentile of the notes
        for row in P:
            for p in set(row[row >= 0].tolist()):
                bars_of[p] = bars_of.get(p, 0) + 1
        m[f"{name}_pitches"] = sum(1 for c in bars_of.values() if c >= 2)
        m[f"{name}_range"] = int(round(np.percentile(notes, 90) - np.percentile(notes, 10))) if len(notes) else 0
        for k in ("same", "subtle", "rewritten"):
            m[f"{name}_{k}"] = pr.get(f"{name}_{k}")
        m[f"{name}_one_note"] = (pr.get("longest_one_note") or {}).get(name, 0)
    return m


def ramp(x, a, b):
    """0 at a, 1 at b, straight between."""
    return float(min(1.0, max(0.0, (x - a) / (b - a))))


def line(m, name, pitches, span):
    """One synth voice: pitches heard in two bars or more, range (10th to 90th percentile), notes per bar, kept and
    varied, not stuck; 0..1 and its parts."""
    if m[f"{name}_present"] <= 0:
        return 0.0, None
    a = m[f"{name}_notes_per_bar"]
    density = a / 3 if a < 3 else 1.0 if a <= 10 else 1 - 0.7 * (a - 10) / 6        # empty .. a wall of sixteenths
    subtle, rewritten = m[f"{name}_subtle"] or 0.0, m[f"{name}_rewritten"] or 0.0
    motif = 0.6 * min(1.0, subtle / 0.35) + 0.4 * (1 - rewritten)                     # a few steps changed, few rewrites
    one = m[f"{name}_one_note"]
    stuck = 0.25 if one >= SECTION else 0.7 if one >= SECTION // 2 else 1.0
    s = min(1.0, m[f"{name}_present"] / 0.75) * np.mean([ramp(m[f"{name}_pitches"], 1, pitches), ramp(m[f"{name}_range"], 0, span), density, motif]) * stuck
    return float(s), {"pitches": m[f"{name}_pitches"], "range": m[f"{name}_range"], "notes a bar": round(a, 1),
                      "bars a few steps changed": round(subtle, 2), "bars rewritten": round(rewritten, 2), "longest on one note": one}


def synth_score(m):
    lead, lp = line(m, "lead", 6, 12)
    bass, bp = line(m, "bass", 4, 7)
    return 0.7 * max(lead, bass) + 0.3 * min(lead, bass), {"lead": lp, "bass": bp, "lead score": round(lead, 2), "bass score": round(bass, 2)}


def break_score(m):
    """Kick missing quarters, the snare on the backbeat with hits and ghost notes around it, busy hats accented off
    the beat; snare rolls (more than 8 hits in a bar) count against it. On the development material (median of the 16-bar
    passages) the group of breakbeat recordings read 0.67 and the others 0.47 and 0.51."""
    broken, backbeat = m["kick_misses_quarter"], m["snare_backbeat_bars"]
    off, ghosts, hats = ramp(m["snare_off_per_bar"], 0, 2.5), ramp(m["ghosts_per_bar"], 0, 2), ramp(m["hat_hits_per_bar"], 4, 10)
    acc_off, acc_on = m.get("hats_accent_offbeat"), m.get("hats_accent_onbeat")
    accent = 0.5 if acc_off is None or acc_on is None else float(min(1.0, max(0.0, 0.5 + acc_off - acc_on)))
    hats_off = 0.5 * ramp(m.get("hats_offbeat_share") or 0.0, 0, 0.3) + 0.5 * accent
    s = min(1.0, m["kick_present"] / 0.5) * (0.3 * broken + 0.15 * backbeat + 0.15 * off + 0.15 * ghosts + 0.1 * hats + 0.15 * hats_off) * (1 - 0.6 * m["snare_busy_bars"])
    return float(s), {"kick misses a quarter": round(broken, 2), "snare on the backbeat": round(backbeat, 2),
                      "snare off the backbeat a bar": round(m["snare_off_per_bar"], 1), "ghost notes a bar": round(m["ghosts_per_bar"], 1),
                      "hat hits a bar": round(m["hat_hits_per_bar"], 1), "hats off the beat": round(m.get("hats_offbeat_share") or 0.0, 2),
                      "hat accents off the beat": None if acc_off is None else round(acc_off, 2), "hat accents on the beat": None if acc_on is None else round(acc_on, 2),
                      "snare rolls": round(m["snare_busy_bars"], 2)}


# ------------------------------------------------------------------ the material

def windows(R, step=8):
    n = len(R["sound"])
    return [cut(R, a, a + SECTION) for a in range(0, n - SECTION + 1, step)]


def build_reference():
    """The material's 16-bar passages (every 8 bars) by group: per measure the median and the spread (interquartile
    range / 1.349, at least a quarter of the spread over all groups), each passage's own distance to its group, and
    the synth and break scores of the passages."""
    score = json.loads(SCORE.read_bytes())
    per = {}
    for t in score["tracks"]:
        g = "sets" if t["group"] in SETS else t["group"]
        per.setdefault(g, []).extend(measures(w) for w in windows(M.rows_of_passage(t)))
    pooled = {}
    for k, _ in LIKE:
        v = np.array([m[k] for lst in per.values() for m in lst if m.get(k) is not None], float)
        q = np.percentile(v, [25, 75]) if len(v) else [0, 0]
        pooled[k] = max(float(q[1] - q[0]) / 1.349, 1e-3)
    groups = {}
    for g, lst in per.items():
        stats = {}
        for k, _ in LIKE:
            v = np.array([m[k] for m in lst if m.get(k) is not None], float)
            if not len(v):
                continue
            q25, q50, q75 = np.percentile(v, [25, 50, 75])
            stats[k] = [round(float(q50), 4), round(max(float(q75 - q25) / 1.349, 0.25 * pooled[k], 0.02), 4)]
        groups[g] = {"passages": len(lst), "measures": stats}
        groups[g]["distances"] = sorted(round(like_distance(m, stats)[0], 4) for m in lst)
        groups[g]["synth"] = [round(float(x), 3) for x in np.percentile([synth_score(m)[0] for m in lst], [10, 25, 50, 75, 90])]
        groups[g]["break"] = [round(float(x), 3) for x in np.percentile([break_score(m)[0] for m in lst], [10, 25, 50, 75, 90])]
    ref = {"score": SCORE.name, "score_mtime": SCORE.stat().st_mtime, "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "section": SECTION, "groups": groups}
    part = REFERENCE.with_suffix(".part")
    part.write_text(json.dumps(ref))
    part.replace(REFERENCE)
    return ref


def reference():
    global _REF
    if _REF is None:
        try:
            ref = json.loads(REFERENCE.read_bytes())
            if ref.get("score_mtime") != SCORE.stat().st_mtime or ref.get("section") != SECTION:
                raise ValueError("the score changed")
        except (OSError, ValueError):
            ref = build_reference()
        _REF = ref
    return _REF


def like_distance(m, stats):
    """Root mean square of the differences to a group's medians, each in the group's spreads (cut at 4)."""
    z = {}
    for k, (med, spread) in stats.items():
        if m.get(k) is not None:
            z[k] = float(np.clip((m[k] - med) / spread, -4, 4))
    return (float(np.sqrt(np.mean([x * x for x in z.values()]))) if z else 9.0), z


def like_score(m):
    best = None
    for g, ref in reference()["groups"].items():
        d, z = like_distance(m, ref["measures"])
        dist = ref["distances"]
        share = 1.0 - float(np.searchsorted(dist, d, side="left")) / max(1, len(dist))    # the group's passages farther away
        if best is None or share > best[0]:
            best = (share, g, d, z)
    share, g, d, z = best
    names = dict(LIKE)
    stats = reference()["groups"][g]["measures"]
    far = sorted(z, key=lambda k: -abs(z[k]))[:3]
    return share, {"group": g, "distance": round(d, 2), "farthest": [[names[k], round(m[k], 2), stats[k][0]] for k in far if abs(z[k]) >= 1]}


# ------------------------------------------------------------------ a track

MARK = {"like": 0.25, "synth": 0.6, "break": 0.6}     # a passage's score at or above this counts as high ...
WITHIN = 0.6                                          # ... and at or above this share of the track's own passages


def marks(vals):
    """Per kind, the mark of this track: the fixed mark, or the track's 60th percentile when that is higher (a
    track that is good all through is narrowed to its best stretches)."""
    out = {}
    for k, v in vals.items():
        own = [x for x in v[SKIP:] if x is not None]
        out[k] = round(max(MARK[k], float(np.quantile(own, WITHIN)) if own else 0.0), 3)
    return out


def runs(vals, mark):
    """Runs of passages at or above the mark, from the third passage on; one passage up to 0.1 under the mark
    between two high ones does not break a run. A run starts and ends on a passage at or above the mark."""
    strict = [v is not None and v >= mark for v in vals]
    hi = list(strict)
    for q in range(1, len(vals) - 1):
        if not hi[q] and strict[q - 1] and strict[q + 1] and vals[q] >= mark - 0.1:
            hi[q] = True
    out, a = [], None
    for q in range(SKIP, len(vals) + 1):
        h = q < len(vals) and hi[q]
        if h and a is None:
            a = q
        elif not h and a is not None:
            b = q
            while a < b and not strict[a]:
                a += 1
            while b > a and not strict[b - 1]:
                b -= 1
            if b > a:
                out.append((a, b))
            a = None
    return out


def free_parts(a, b, kept):
    """The stretches of [a, b) no kept suggestion covers."""
    out, x = [], a
    for s in sorted(kept):
        if s[1] <= x or s[0] >= b:
            continue
        if s[0] > x:
            out.append((x, s[0]))
        x = max(x, s[1])
    if x < b:
        out.append((x, b))
    return out


def suggestions(sections, mark):
    vals = {k: [s[k] if s else None for s in sections] for k in KINDS}
    strength = lambda k, a, b: (float(np.mean(vals[k][a:b])) - MARK[k]) / (1 - MARK[k])     # ranked on the fixed marks

    cands = []
    for k in KINDS:
        todo = runs(vals[k], mark[k])
        while todo:
            a, b = todo.pop()
            if b - a < SHORTEST:
                continue
            if b - a > LONGEST:                                 # the best 8 passages; what is left over stays a candidate
                x = max(range(a, b - LONGEST + 1), key=lambda x: np.mean(vals[k][x:x + LONGEST]))
                todo += [(a, x), (x + LONGEST, b)]
                a, b = x, x + LONGEST
            cands.append([strength(k, a, b), k, a, b])
    cands.sort(key=lambda c: -c[0])
    kept = []                                                   # [a, b] in passages, strongest first
    for _, k, a, b in cands:
        over = [s for s in kept if a < s[1] and b > s[0]]
        if len(over) == 1:                                      # overlaps one: merged when the union is 8 passages or fewer
            s = over[0]
            ua, ub = min(a, s[0]), max(b, s[1])
            if ub - ua <= LONGEST and not any(t is not s and ua < t[1] and ub > t[0] for t in kept):
                s[0], s[1] = ua, ub
                continue
        if over:                                                # else the longest part outside the others, if long enough
            free = max(free_parts(a, b, kept), key=lambda p: p[1] - p[0], default=None)
            if not free or free[1] - free[0] < SHORTEST:
                continue
            a, b = free
        kept.append([a, b])
    out = []
    for a, b in kept:
        means = {k: round(float(np.mean(vals[k][a:b])), 3) for k in KINDS}
        lead = max(KINDS, key=lambda k: strength(k, a, b))
        also = [k for k in KINDS if k != lead and means[k] >= mark[k]]
        out.append({"from": a * SECTION, "to": b * SECTION, "sections": list(range(a, b)), "kind": lead, "kinds": [lead] + also, "scores": means,
                    "strength": round(strength(lead, a, b) + 0.05 * len(also), 3),
                    "label": f"bars {a * SECTION + 1} to {b * SECTION} · " + " · ".join(KINDS[k] for k in [lead] + also)})
    out.sort(key=lambda s: -s["strength"])
    return out[:MOST]


def score_track(track):
    t0 = time.time()
    R = M.rows_of_track(track)
    n = len(R["sound"])
    sections = []
    for q in range(-(-n // SECTION)):
        a, b = q * SECTION, min(n, (q + 1) * SECTION)
        if b - a < SECTION:
            sections.append(None)                                # a short last passage is not scored
            continue
        m = measures(cut(R, a, b))
        like, lp = like_score(m)
        synth, sp = synth_score(m)
        brk, bp = break_score(m)
        sections.append({"q": q, "from": a, "to": b, "like": round(like, 3), "synth": round(synth, 3), "break": round(brk, 3),
                         "parts": {"like": lp, "synth": sp, "break": bp}})
    mark = marks({k: [s[k] if s else None for s in sections] for k in KINDS})
    return {"bars": n, "section": SECTION, "skip": SKIP * SECTION, "kinds": {k: {"label": v, "mark": mark[k]} for k, v in KINDS.items()},
            "sections": sections, "suggestions": suggestions(sections, mark), "seconds": round(time.time() - t0, 3)}


def for_track(tid, path):
    """The scores and suggestions of one track file, from the cache while the file is unchanged."""
    st = path.stat()
    key = (st.st_mtime_ns, st.st_size, reference().get("made"))
    hit = CACHE.get(tid)
    if hit and hit[0] == key:
        return hit[1]
    with LOCK:
        hit = CACHE.get(tid)
        if hit and hit[0] == key:
            return hit[1]
        out = dict(score_track(json.loads(path.read_bytes())), id=tid)
        CACHE[tid] = (key, out)
        return out


def count(tid, path):
    """The number of suggestions of a track when its scores are in the cache, else None."""
    hit = CACHE.get(tid)
    try:
        st = path.stat()
    except OSError:
        return None
    if hit and hit[0][:2] == (st.st_mtime_ns, st.st_size):
        return len(hit[1]["suggestions"])
    return None


_WARMING = set()


def warm(pairs):
    """Score these (track id, path) pairs in the background, one after the other."""
    todo = [(t, p) for t, p in pairs if count(t, p) is None and t not in _WARMING]
    if not todo:
        return
    _WARMING.update(t for t, _ in todo)

    def run():
        for tid, path in todo:
            try:
                for_track(tid, path)
            except Exception:
                pass
            finally:
                _WARMING.discard(tid)
    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    if sys.argv[1:] == ["--material"]:
        for g, ref in reference()["groups"].items():
            print(f"{g:8s} {ref['passages']:5d} passages  synth p10..p90 {ref['synth']}  break p10..p90 {ref['break']}  own distance median {np.median(ref['distances']):.2f}")
        raise SystemExit
    for arg in sys.argv[1:]:
        p = Path(arg).resolve()
        r = for_track(f"{p.parent.name}/{p.name}", p)
        print(f"{p.parent.name}/{p.name}: {r['seconds']} s")
        for s in r["sections"]:
            if s:
                print(f"  bars {s['from'] + 1:3d}-{s['to']:3d}  like {s['like']:.2f} ({s['parts']['like']['group']})  synth {s['synth']:.2f}  break {s['break']:.2f}")
        for s in r["suggestions"]:
            print(f"  -> {s['label']}  {s['scores']}  strength {s['strength']}")
