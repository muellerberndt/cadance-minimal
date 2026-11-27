"""Finished Auto DJ tracks composed ahead of time: N whole tracks of one brain state, written to the shelf the Auto DJ
reads (site/tracks/<checkpoint>/, in the format the TrackFactory keeps), so a set with that brain starts at once
with a track of it and a late live track has a stand-in.

    nice -n 10 python site/shelf.py <checkpoint> [--tracks N] [--bars 304] [--out DIR] [--seed S]

<checkpoint> is the name the player lists (a pinned state, a run's checkpoint, or a run's newest state as the player
fetched it, <run>_l<stamp>). The brain is loaded and played as the Auto DJ factory plays it: serve.Creature with the
per-brain play settings of site/play-reading.json, the brain view per bar (the mixer's visuals), the factory's mood
over a track (it rises toward the middle and comes down again), a new passage for every track. Everything runs in
this process: the live player's brains, its play cache, its load-time rates and its selection are not touched (this
tool keeps its own play copies and rates in site/.shelf-cache/). A finished track is written as <stamp>_<bars>.json
(a .part first, then renamed) and the folder keeps the newest DJ_KEPT tracks, as the factory keeps it. --bars 304
is what the Auto DJ asks for at 32-bar sections; it takes a kept track of at least three quarters of that. --seed
fixes every unseeded generator (a test). Every track's record carries a "summary" (bass and lead notes per bar while
they sound, as the line caps count them, and kicks per bar while the beat sounds; per segment and over the track).
--require "bass>=3,lead>=3,kick>=3" (whole-track means; any subset; >= <= > <): a track that misses is kept in
<shelf>/<checkpoint>/rejected/ (the Auto DJ reads the checkpoint folder only) and another is composed from a new
passage, up to --tries per wanted track (default 3).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SITE = Path(__file__).resolve().parent
CACHE = SITE / ".shelf-cache"
# the live player's thread settings (launchd: OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1), set before NumPy loads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("CADANCE_PLAY_CACHE_DIR", str(CACHE))
os.environ.setdefault("CADANCE_LOAD_RATES", str(CACHE / "load-rates.json"))
CACHE.mkdir(exist_ok=True)

import numpy as np  # noqa: E402

SEGMENTS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, None))


def summary(rows):
    """rows: per bar (bass notes or None while silent, lead notes or None, kicks or None while the beat is off)."""
    def means(part):
        out = {}
        for i, key in enumerate(("bass", "lead", "kick")):
            vals = [r[i] for r in part if r[i] is not None]
            out[key] = round(float(np.mean(vals)), 2) if vals else None
        return out
    segs = [{"bars": f"{a + 1}-{b if b else len(rows)}", **means(rows[a:b])} for a, b in SEGMENTS if a < len(rows)]
    return {"whole": means(rows), "segments": segs}


def requirement(text):
    """'bass>=3,lead>=3,kick>=3' -> [(key, op, value)]."""
    import re
    out = []
    for part in [p.strip() for p in (text or "").split(",") if p.strip()]:
        m = re.fullmatch(r"(bass|lead|kick)\s*(>=|<=|>|<)\s*([0-9.]+)", part)
        if not m:
            raise SystemExit(f"--require: {part!r} is not bass|lead|kick with >= <= > < and a number")
        out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def misses(whole, needs):
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b}
    return [f"{k} {whole.get(k)} not {op} {v:g}" for k, op, v in needs if whole.get(k) is None or not ops[op](whole[k], v)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--tracks", type=int, default=1)
    ap.add_argument("--bars", type=int, default=304)
    ap.add_argument("--out", type=Path, default=None, help="the shelf folder (default: site/tracks)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--require", default="", help='whole-track means a track must meet, e.g. "bass>=3,lead>=3,kick>=3"')
    ap.add_argument("--tries", type=int, default=3, help="compositions per wanted track at most (with --require)")
    args = ap.parse_args()
    if os.nice(0) < 10:
        os.nice(10 - os.nice(0))
    if args.seed is not None:
        real = np.random.default_rng
        np.random.default_rng = lambda s=None, _r=real: _r(args.seed if s is None else s)
    sys.path.insert(0, str(SITE))
    import serve

    shelf = args.out or serve.TRACKS
    name = args.checkpoint
    c = serve.Creature()
    c.follow_latest, c.with_view = False, True
    started = time.time()
    c.load(name)
    print(f"{name}: loaded in {time.time() - started:.0f}s; play reading {serve.play_reading(c.card)}", flush=True)
    pool_name = c.pool_name()
    try:
        bpm = json.loads((serve.ROOT / "pool" / pool_name).read_bytes()).get("bpm")
    except (OSError, ValueError):
        bpm = None
    base = [0.45, 0.4]                                   # the Auto DJ's mood (TrackFactory.request without a mood)
    needs = requirement(args.require)
    import common
    fresh = True                                         # the load primed the first composition
    for n in range(args.tracks):
        for attempt in range(max(1, args.tries) if needs else 1):
            if not fresh:
                c.renew()                                # a new composition from another passage, as the factory does
            fresh = False
            rng = np.random.default_rng()
            lift, drift = float(rng.uniform(0.08, 0.2)), float(rng.uniform(-0.1, 0.1))
            want, bars, rows, began = args.bars, [], [], time.time()
            for k in range(want):
                arc = np.sin(np.pi * k / max(1, want - 1))
                mood = [float(np.clip(base[0] + lift * (arc - 0.5), 0.0, 1.0)),
                        float(np.clip(base[1] + drift * (k / want - 0.5), 0.0, 1.0))]
                reply = c.next(mood, 0.0)
                bars.append({"tokens": reply.get("tokens"), "mods": reply.get("mods"), "patterns": reply.get("patterns"),
                             "patches": reply.get("patches"), "bar": k + 1, "view": reply.get("view")})
                bars[-1].update({key: reply[key] for key in ("capped", "row_capped", "line_capped") if key in reply})
                tok, lines = reply.get("tokens") or [0, 0, 0, 0], common.decode_synth(c.state)
                beat = common.decode_beat(c.state) if common.STEPS else None
                rows.append((int((np.asarray(lines[0]) >= 2).sum()) if lines is not None and tok[1] > 0 else None,
                             int((np.asarray(lines[1]) >= 2).sum()) if lines is not None and tok[2] > 0 else None,
                             int((np.asarray(beat[0]) > 0).sum()) if beat is not None and tok[0] > 0 else None))
                if (k + 1) % 32 == 0 or k + 1 == want:
                    took = time.time() - began
                    print(f"  track {n + 1}: {k + 1} of {want} bars, {took / (k + 1):.2f} s a bar, about "
                          f"{took / (k + 1) * (want - k - 1) / 60:.0f} min left", flush=True)
            made = summary(rows)
            missed = misses(made["whole"], needs)
            segs = " | ".join(f"{s['bars']}: bass {s['bass']} lead {s['lead']} kick {s['kick']}" for s in made["segments"])
            print(f"  summary of track {n + 1}{f' (try {attempt + 1})' if needs else ''}, passage {getattr(c, 'seed_title', None)!r}: "
                  f"whole bass {made['whole']['bass']} lead {made['whole']['lead']} kick {made['whole']['kick']} | {segs}"
                  f"{' | MISSES ' + '; '.join(missed) if missed else ''}", flush=True)
            folder = shelf / name / "rejected" if missed else shelf / name
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"{time.strftime('%Y%m%dT%H%M%S')}_{len(bars)}.json"
            part = target.with_suffix(".part")
            part.write_text(json.dumps({"checkpoint": name, "pool": pool_name, "bpm": bpm, "seed": getattr(c, "seed_title", None),
                                        "bars": bars, "layout": getattr(c, "view_layout", None), "summary": made,
                                        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, separators=(",", ":")))
            part.replace(target)
            if missed:
                print(f"track {n + 1} of {args.tracks}, try {attempt + 1}: rejected, kept in {target}", flush=True)
                continue
            for old in sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[serve.DJ_KEPT:]:
                old.unlink()
            print(f"track {n + 1} of {args.tracks}: {target} ({len(bars)} bars in {(time.time() - began) / 60:.1f} min, "
                  f"passage {getattr(c, 'seed_title', None)!r})", flush=True)
            break
        else:
            print(f"track {n + 1} of {args.tracks}: none of {args.tries} tries met {args.require!r}", flush=True)


if __name__ == "__main__":
    main()
