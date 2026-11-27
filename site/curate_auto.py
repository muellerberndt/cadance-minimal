"""Pre-select repertoire pieces by measurement, and keep the library growing.

The owner rates and deletes; this only proposes. It asks the player's track factory for tracks of the approved
brains, measures every composed track, takes the best long stretch of a good track as a piece, names it from what
was measured, and adds it through the repertoire site (which renders the MP3 and saves to the bucket). It cannot
hear. What it measures:

  per 16-bar passage   a solid beat (kick and hats or snare in the bar), bass present, lead or pad present,
                       how the stream mutates (some change from bar to bar, few outright rewrites)
  per piece            the mean and the weakest passage; 160 to 256 bars, starting on a 32-bar line
  from the rendered    loudness steadiness over eight-bar blocks, the share of lows, mids and highs, silence
  MP3

    python site/curate_auto.py --report            measure the composed tracks and print, change nothing
    python site/curate_auto.py --hours 9 --pieces 30    run: compose, measure, add
    python site/curate_auto.py --spread            what the kept pieces are, row by row, and how far apart
    python site/curate_auto.py --refresh-mp3       render again every MP3 made under older player settings

The repertoire factory on a box (the Mac's player then composes for listeners only): the player's server runs
headless on a GPU box (~/minimal/player/run_player.sh), a tunnel brings its port to the Mac
(site/repertoire_tunnel.sh), and the job asks it for tracks, copies each finished track into site/tracks/ and goes on
as before:

    python site/curate_auto.py --hours 9 --keep 24 --player http://127.0.0.1:8796 \
        --sync USER@BOX:minimal/player/site/tracks --key KEYFILE \
        --models checkpoint:slate_split_s1_n25d_e001 --prefer slate_split_s1_n25d_e001 --session curate-box --moods
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
TRACKS = SITE / "tracks"
STATE = TRACKS / ".curated.json"
PLAYER, CURATOR = "http://127.0.0.1:8792", "http://127.0.0.1:8794"
ASKS = TRACKS / ".asks.json"        # what each remotely composed track was asked with: model, mood, seconds
TRACK_RE = re.compile(r"[A-Za-z0-9_]{1,80}/[A-Za-z0-9_]{1,60}\.json")
# The moods the tracks are asked with, in turn ([energy, brightness], 0..1), so the pieces do not all come from one
# corner; the factory lets the energy rise toward a track's middle on top of it.
MOODS = [[0.45, 0.4], [0.7, 0.55], [0.3, 0.3], [0.6, 0.25], [0.35, 0.6], [0.8, 0.4], [0.5, 0.7], [0.25, 0.45],
         [0.65, 0.65], [0.4, 0.2], [0.75, 0.3], [0.55, 0.5]]
SECTION = 16
GOOD, WEAKEST = 0.74, 0.5            # a piece's mean passage and its weakest passage
POOLS = {}


def call(base, path, body=None, timeout=60):
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())


def pool(name):
    if name not in POOLS:
        POOLS[name] = json.loads((ROOT / "pool" / name).read_bytes())
    return POOLS[name]


def rows(track):
    """Per bar: the drum rows (kick, snare or clap, hats) and the note steps of bass and lead, the air's presence, the kit."""
    p = pool(track["pool"])
    out = []
    for bar in track["bars"]:
        tok = bar["tokens"]
        pats = bar.get("patterns") or [None] * 4

        def pattern(layer):
            if not tok[layer]:
                return None
            return pats[layer] or ((p["layers"][layer]["tokens"][tok[layer]] or {}).get("pattern") if tok[layer] < len(p["layers"][layer]["tokens"]) else None)
        drums = np.zeros((3, 16), bool)
        pat = pattern(0)
        if pat and pat.get("hits"):
            for voice, steps in pat["hits"].items():
                row = 0 if voice == "kick" else 1 if voice in ("snare", "clap", "rim") else 2 if voice in ("ch", "oh", "cym") else None
                if row is not None:
                    drums[row] |= np.array([(v or 0) > 0 for v in (list(steps) + [0] * 16)[:16]])
        notes = []
        for layer in (1, 2, 3):
            pat = pattern(layer)
            notes.append(tuple((pat.get("notes") or [None] * 16)[:16]) if pat else None)
        patch = (bar.get("patches") or [None] * 4)
        out.append(dict(drums=drums, notes=notes, kit=tok[0], bass_voice=(patch[1] or {}).get("voice") if patch[1] else None))
    return out


def passage_scores(R):
    """One score per 16-bar passage, and what it is made of."""
    scores, parts = [], []
    for a in range(0, len(R) - SECTION + 1, SECTION):
        seg = R[a:a + SECTION]
        beat = np.mean([b["drums"][0].sum() >= 2 and (b["drums"][2].sum() >= 2 or b["drums"][1].sum() >= 1) for b in seg])
        bass = np.mean([b["notes"][0] is not None and sum(n is not None for n in b["notes"][0]) >= 1 for b in seg])
        top = np.mean([any(b["notes"][v] is not None and any(n is not None for n in b["notes"][v]) for v in (1, 2)) for b in seg])
        changed, rewrites = [], []
        for x, y in zip(seg, seg[1:]):
            d = int((x["drums"] != y["drums"]).sum())
            n = [sum(p != q for p, q in zip(x["notes"][v] or [None] * 16, y["notes"][v] or [None] * 16)) for v in (0, 1)]
            changed.append(d > 0 or any(k > 0 for k in n))
            rewrites.append(d > 16 or any(k > 8 for k in n))
        change, rewrite = float(np.mean(changed)), float(np.mean(rewrites))
        mutation = 1.0 if 0.15 <= change <= 0.7 else max(0.0, change / 0.15) if change < 0.15 else max(0.0, (1 - change) / 0.3)
        score = 0.35 * beat + 0.2 * bass + 0.15 * top + 0.15 * mutation + 0.15 * (1 - min(1.0, rewrite / 0.4))
        scores.append(float(score))
        parts.append(dict(beat=float(beat), bass=float(bass), top=float(top), change=change, rewrite=rewrite))
    return scores, parts


def best_piece(scores, parts):
    """The longest good stretch: 256, 224, 192 or 160 bars from a 32-bar line, by its mean and its weakest passage.
    The first 32 bars of a track are left out (they are nearest to the passage the brain was given as context), and a
    piece must have its beat in at least 90% of its bars, a bass in 60% and a lead or pad in 60%."""
    best = None
    for bars in (256, 224, 192, 160):
        n = bars // SECTION
        for a in range(2, len(scores) - n + 1, 2):
            window, made = scores[a:a + n], parts[a:a + n]
            mean, weakest = float(np.mean(window)), float(min(window))
            whole = all(np.mean([p[k] for p in made]) >= floor for k, floor in (("beat", 0.9), ("bass", 0.6), ("top", 0.6)))
            if whole and mean >= GOOD and weakest >= WEAKEST and (best is None or mean > best["mean"] + 0.01):
                best = dict(start=a * SECTION, bars=bars, mean=mean, weakest=weakest)
        if best:
            return best
    return None


def describe(track, R, piece):
    seg = R[piece["start"]:piece["start"] + piece["bars"]]
    p = pool(track["pool"])
    kits = [b["kit"] for b in seg if b["kit"]]
    device = None
    if kits:
        kit = max(set(kits), key=kits.count)
        device = (p["layers"][0]["tokens"][kit] or {}).get("device")
    with_kick = [b for b in seg if b["drums"][0].any()]
    quarters = np.mean([b["drums"][0][[0, 4, 8, 12]].all() for b in with_kick]) if with_kick else 0
    backbeat = np.mean([b["drums"][1][4] and b["drums"][1][12] for b in seg])
    hats = np.mean([b["drums"][2].sum() for b in seg])
    bass_on = [sum(n is not None for n in b["notes"][0]) for b in seg if b["notes"][0] is not None]
    lead_on = [sum(n is not None for n in b["notes"][1]) for b in seg if b["notes"][1] is not None]
    air = np.mean([b["notes"][2] is not None and any(n is not None for n in b["notes"][2]) for b in seg])
    acid = np.mean([b["bass_voice"] == "acid" for b in seg])
    beat = "four-on-the-floor" if quarters >= 0.7 else "broken kick" if quarters <= 0.3 and with_kick else "shifting kick"
    words = [f"{device} {beat}" if device else beat]
    if backbeat >= 0.4:
        words.append("backbeat")
    words.append("busy hats" if hats >= 9 else "sparse hats" if hats < 4 else "steady hats")
    if bass_on:
        words.append("acid bass" if acid >= 0.5 else "rolling bass" if np.mean(bass_on) >= 6 else "sparse bass" if np.mean(bass_on) <= 2.5 else "bass line")
    if lead_on and len(lead_on) >= len(seg) * 0.3:
        words.append("busy lead" if np.mean(lead_on) >= 6 else "lead figure")
    if air >= 0.5:
        words.append("pads")
    brain = (track.get("checkpoint") or "").replace("slate_", "").replace("_s1", "").replace("_n25d", "")
    return ", ".join(words) + f" · {brain}", dict(quarters=float(quarters), backbeat=float(backbeat), hats=float(hats), air=float(air))


def audio_check(mp3):
    """The rendered piece: loudness per eight-bar block, the shares of lows, mids and highs, and silence."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", "22050", "-f", "f32le", "-"], capture_output=True, timeout=300).stdout
    x = np.frombuffer(raw, "<f4")
    if len(x) < 22050 * 20:
        return dict(ok=False, why="the render is shorter than twenty seconds")
    block = 22050 * 15
    level = np.array([10 * np.log10(np.mean(x[i:i + block] ** 2) + 1e-12) for i in range(0, len(x) - block + 1, block)])
    spec = np.zeros(2049)
    for i in range(0, len(x) - 4096, 4096 * 8):
        spec += np.abs(np.fft.rfft(x[i:i + 4096] * np.hanning(4096))) ** 2
    f = np.fft.rfftfreq(4096, 1 / 22050)
    total = spec.sum() + 1e-12
    low, mid, high = (float(spec[(f >= a) & (f < b)].sum() / total) for a, b in ((20, 150), (150, 2500), (2500, 11025)))
    quiet = float(np.mean(level < level.max() - 14))
    out = dict(level_db=float(np.median(level)), swing_db=float(level.max() - np.percentile(level, 10)), low=low, mid=mid, high=high, quiet=quiet)
    # nearly all low end: the kick and bass bury the rest (the other pieces hold 42 to 51% under 150 Hz; an 808 piece
    # rendered on 26 November held 88%, with 2% above 2.5 kHz)
    why = ("hardly any low end" if low < 0.12 else "nearly all low end" if low > 0.75 else "mostly highs" if high > 0.45 else "long quiet stretches" if quiet > 0.2
           else "loudness swings of more than 9 dB" if out["swing_db"] > 9 else None)
    return dict(out, ok=why is None, why=why)


def measure(track_id):
    track = json.loads((TRACKS / track_id).read_bytes())
    R = rows(track)
    scores, parts = passage_scores(R)
    piece = best_piece(scores, parts)
    name, extra = (describe(track, R, piece) if piece else (None, {}))
    return track, scores, parts, piece, name, extra


def log(text):
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def features(source, start=0, bars=None):
    """What makes one piece differ from another, as numbers in 0..1: broken-kick share, kick, snare and hat hits per
    bar, the hats' off-beat share, the snare's backbeat share, bass and lead attacks per bar and their ranges, then
    the pitch classes of the bass and of the lead (two histograms of 12). `source`: a track or a repertoire piece."""
    R = rows(source)[start:start + bars if bars else None]
    if not R:
        return None
    D = np.array([b["drums"] for b in R])                       # bars x 3 x 16
    kick, snare, hats = D[:, 0], D[:, 1], D[:, 2]
    kb = kick.any(axis=1)
    four = kick[:, [0, 4, 8, 12]].all(axis=1) & (kick.sum(axis=1) == 4)
    vec = [float(1 - four[kb].mean()) if kb.any() else 0.0, float(kick.sum() / (16 * len(R))), float(snare.sum() / (16 * len(R))), float(hats.sum() / (16 * len(R))),
           float(hats[:, [2, 6, 10, 14]].sum() / max(1, hats.sum())), float(snare[:, [4, 12]].sum() / max(1, snare.sum()))]
    hist = []
    for v in (0, 1):
        notes = [n for b in R if b["notes"][v] is not None for n in b["notes"][v] if n is not None]
        vec.append(min(1.0, len(notes) / (16 * len(R))))
        vec.append(min(1.0, (max(notes) - min(notes)) / 24) if notes else 0.0)
        h = np.bincount(np.asarray(notes, int) % 12, minlength=12).astype(float) if notes else np.zeros(12)
        hist += list(h / h.sum() if h.sum() else h)
    return [round(float(x), 4) for x in vec + hist]


def distance(a, b):
    """0 = the same piece by its features; the ten shares and rates as a Euclidean distance, plus half the share of
    bass notes and of lead notes that would have to move to another pitch class."""
    if not a or not b:
        return 1.0
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(((a[:10] - b[:10]) ** 2).sum()) + 0.25 * np.abs(a[10:22] - b[10:22]).sum() + 0.25 * np.abs(a[22:34] - b[22:34]).sum())


_KEPT = {}


def kept_features(skip=()):
    """{repertoire id: (name, features)} of every piece in the repertoire except SID chip versions (they are their
    source piece again) and the ids in `skip`."""
    out = {}
    for f in sorted((SITE / "repertoire").glob("r*.json")):
        if f.stem in skip:
            continue
        key = (f.stem, f.stat().st_size)
        if key not in _KEPT:
            try:
                item = json.loads(f.read_bytes())
                _KEPT[key] = None if item.get("sid") or "SID chip" in str(item.get("name")) else (item.get("name"), features(item))
            except Exception:
                _KEPT[key] = None
        if _KEPT[key]:
            out[f.stem] = _KEPT[key]
    return out


def nearest(feats, skip=()):
    """(distance, name) of the kept piece most like these features."""
    best = (9.0, None)
    for _, (name, other) in kept_features(skip).items():
        d = distance(feats, other)
        if d < best[0]:
            best = (d, name)
    return best


def standing(state, prefer):
    """The pre-selected pieces still in the repertoire, weakest first: [(key, track id, entry)]. A piece the owner
    took out is marked gone. The order of giving up: pieces he rated 1 or 2 stars, then unrated pieces of brains
    that are not preferred, then unrated pieces of preferred brains, each by the lowest mean passage. A piece with
    3 or more stars is never given up."""
    out = []
    for tid, v in state.items():
        if not v.get("item"):
            continue
        f = SITE / "repertoire" / f"{v['item']}.json"
        if not f.is_file():
            v["gone"], v["item"] = v["item"], None           # removed by hand in the repertoire site
            continue
        try:
            stars = int(json.loads(f.read_bytes()).get("stars") or 0)
        except Exception:
            stars = 0
        if stars >= 3:
            continue
        liked = any(p in tid for p in prefer)
        out.append(((0 if 1 <= stars <= 2 else 1 if not liked else 2, float((v.get("piece") or {}).get("mean") or 0.0)), tid, v))
    out = sorted(out, key=lambda x: x[0])
    # Variety: among the pieces of the first class that measure within 0.03 of its weakest, the one most like another
    # kept piece goes first (two alike pieces give the Auto DJ less than two different ones).
    if len(out) > 1:
        group = [x for x in out if x[0][0] == out[0][0][0] and x[0][1] <= out[0][0][1] + 0.03]
        if len(group) > 1:
            def alike(x):
                v = x[2]
                if "features" not in v:
                    try:
                        v["features"] = features(json.loads((SITE / "repertoire" / f"{v['item']}.json").read_bytes()))
                    except Exception:
                        v["features"] = None
                return nearest(v["features"], skip=(v["item"], v.get("sid_item") or ""))[0] if v["features"] else 9.0
            first = min(group, key=alike)
            out.remove(first); out.insert(0, first)
    return out


def remote_tracks(args):
    """The finished track files on the box, as track ids."""
    host, path = args.sync.split(":", 1)
    done = subprocess.run(["ssh", "-i", args.key, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, f"cd {path} && ls */*.json 2>/dev/null"],
                          capture_output=True, text=True, timeout=40)
    return [x.strip() for x in done.stdout.splitlines() if TRACK_RE.fullmatch(x.strip())]


def fetch_tracks(args, state, known_only=None):
    """Copy the box's finished tracks that are neither measured nor here yet into site/tracks/; the new track ids."""
    host, path = args.sync.split(":", 1)
    new = []
    try:                                                        # tracks deleted by hand in the editor stay deleted
        deleted = set((TRACKS / ".deleted").read_text().split())
    except OSError:
        deleted = set()
    for tid in remote_tracks(args):
        dest = TRACKS / tid
        if tid in state or dest.is_file() or tid in deleted:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        got = subprocess.run(["scp", "-i", args.key, "-q", "-o", "ConnectTimeout=10", f"{host}:{path}/{tid}", str(part)], capture_output=True, text=True, timeout=180)
        try:
            if got.returncode:
                raise ValueError(got.stderr.strip()[-120:])
            json.loads(part.read_bytes())                       # a whole file, not one still being written
            part.replace(dest)
            new.append(tid)
        except Exception as e:
            part.unlink(missing_ok=True)
            log(f"copy of {tid} put off: {e}")
    return new


def box_load(args):
    """The box's one-minute load, or None when it cannot be read."""
    try:
        host = args.sync.split(":", 1)[0]
        done = subprocess.run(["ssh", "-i", args.key, "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, "cat /proc/loadavg"], capture_output=True, text=True, timeout=30)
        return float(done.stdout.split()[0])
    except Exception:
        return None


def refresh_mp3(limit=None):
    """Render again, one at a time, every piece whose MP3 was made under other player settings than the present ones
    (the repertoire site marks each MP3 with the settings and engine it was rendered with)."""
    mark = call(CURATOR, "/api/repertoire/settings")["mark"]
    stale = []
    for f in sorted((SITE / "repertoire").glob("r*.json")):
        item = json.loads(f.read_bytes())
        if item.get("mp3_settings") != mark:
            stale.append((f, item.get("name")))
    log(f"{len(stale)} of {len(list((SITE / 'repertoire').glob('r*.json')))} pieces have an MP3 from other settings (present mark {mark})")
    for f, name in stale[:limit]:
        started = time.time()
        call(CURATOR, "/api/repertoire/mp3", dict(id=f.stem))
        while time.time() - started < 900:
            time.sleep(5)
            try:
                item = json.loads(f.read_bytes())
            except Exception:
                continue
            if item.get("mp3_settings") == mark or item.get("mp3_error"):
                break
        log(f"{f.stem} '{name}': " + (f"MP3 rendered again in {time.time() - started:.0f} s" if item.get("mp3_settings") == mark else f"not rendered: {item.get('mp3_error') or 'no answer in 15 minutes'}"))
    return len(stale)


def spread():
    """What the kept pieces are: one row per piece from the listening worker's play report (quality/loop/measure.py),
    the range of every column across the pieces, and how far each piece is from its nearest neighbour."""
    sys.path.insert(0, str(ROOT / "quality" / "loop"))
    import measure as M
    cols = ("kick_broken_bars", "hats_offbeat_share", "snare_backbeat_share", "distinct_drum_bars_per_16", "distinct_bass_bars_per_16", "distinct_lead_bars_per_16",
            "beat_same", "bass_same", "lead_same", "entries_exits_per_32", "bass_distinct_pitches_per_16", "bass_range_per_16", "lead_distinct_pitches_per_16", "lead_range_per_16", "snare_busy_rows")
    table = []
    for f in sorted((SITE / "repertoire").glob("r*.json")):
        item = json.loads(f.read_bytes())
        if item.get("sid") or "SID chip" in str(item.get("name")):
            continue
        row = M.play_row(M.rows_of_track(item))
        near = nearest(features(item), skip=(f.stem,))
        table.append((item, row, near))
    print(f"{'piece':44s} {'bars':>4s} {'stars':>5s} " + " ".join(f"{c[:11]:>11s}" for c in cols) + "  nearest piece")
    for item, row, near in table:
        print(f"{str(item.get('name'))[:44]:44s} {len(item['bars']):4d} {int(item.get('stars') or 0):5d} " + " ".join(f"{row.get(c):11.2f}" if isinstance(row.get(c), (int, float)) else f"{'-':>11s}" for c in cols)
              + f"  {near[0]:.2f} {str(near[1])[:30]}")
    if len(table) > 1:
        print(f"{'range across the pieces':55s} " + " ".join((lambda v: f"{min(v):5.2f}-{max(v):<5.2f}" if v else f"{'-':>11s}")([r.get(c) for _, r, _ in table if isinstance(r.get(c), (int, float))]) for c in cols))


def give_up(state, tid, v, why):
    for key in ("item", "sid_item"):
        if v.get(key):
            try:
                call(CURATOR, "/api/repertoire/remove", dict(id=v[key]))
            except Exception as e:
                log(f"remove {v[key]}: {e}")
    log(f"{tid}: '{v.get('name')}' taken out of the repertoire, {why}")
    v["replaced"], v["item"], v["sid_item"] = v.get("item"), None, None


def factory_step(args, state, ctx):
    """Keep the factory composing: note a finished track, rest, wait for a quiet box, ask for the next track. Returns
    True when it is busy or was just asked (nothing to wait for here)."""
    status = call(args.player, "/api/autodj/status")
    pending = ctx.get("pending")
    if status["state"] != "idle" or any(not t["done"] for t in status["tracks"]):
        return True                                   # the factory is busy (a set is running, or our track is in work)
    if pending:                                       # our track is done (its file is copied by the sync, or it failed)
        took = (pending.get("done_at") or time.time()) - pending["at"]
        failed = next((t.get("error") for t in status["tracks"] if t.get("id") == pending["id"] and t.get("error")), None)
        ctx["composed"] += 0 if failed else 1
        hours = (time.time() - ctx["started"]) / 3600
        log((f"the track failed at the factory: {failed}" if failed else f"track composed in {took:.0f} s")
            + f"; {ctx['composed']} tracks in {hours:.2f} h = {ctx['composed'] / max(1e-6, hours):.1f} per hour")
        ctx["rest_until"], ctx["pending"] = (pending.get("done_at") or time.time()) + args.rest * took, None
        return False
    if time.time() < ctx["rest_until"]:
        return False
    if args.sync and args.box_load:
        load = box_load(args)
        if load is not None and load > args.box_load:
            if time.time() - ctx["waited_load"] > 600:
                log(f"the box's load is {load:.1f}, above {args.box_load:g}: waiting with the next track")
                ctx["waited_load"] = time.time()
            ctx["rest_until"] = time.time() + 60
            return False
    approved = args.models or (call(args.player, "/api/autodj/models").get("approved") or ["leader"])
    if args.follow:          # the brain the owner picked for his Auto DJ (the Mac's dj-approved.json): its tracks fill the shelf
        try:
            picked = [str(m) for m in json.loads(Path(args.follow).read_text()) if str(m).startswith(("candidate:", "checkpoint:"))]
        except (OSError, ValueError):
            picked = []
        approved = picked or approved
    approved = approved + [m for m in args.also if m not in approved]      # brains whose tracks have passed before

    # two tracks of three from the brain whose tracks have passed most often, the third from each in turn
    def rate(m):
        key = m.split(":", 1)[-1]
        tries = [v for t, v in state.items() if key in t]
        return (sum(1 for v in tries if v.get("item") or v.get("replaced")) + 1) / (len(tries) + 2)
    turn = ctx["turn"]
    model = max(approved, key=rate) if turn % 3 else approved[(turn // 3) % len(approved)]
    mood = MOODS[turn % len(MOODS)] if args.moods else None
    ctx["turn"] += 1
    known = set(remote_tracks(args)) if args.sync else set()
    answer = call(args.player, "/api/autodj/track", dict(model=model, bars=args.bars, session=args.session, **({"mood": mood} if mood else {})))
    ctx["pending"] = dict(id=answer.get("id"), model=model, mood=mood, at=time.time(), known=known)
    log(f"asked the brain {model} for a track of {args.bars} bars" + (f", mood {mood}" if mood else ""))
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--pieces", type=int, default=30, help="stop when the repertoire holds this many pre-selected pieces")
    ap.add_argument("--bars", type=int, default=288)
    ap.add_argument("--sid-every", type=int, default=2, help="also add a SID chip version of every Nth piece (0 = never)")
    ap.add_argument("--also", nargs="*", default=[], help="brains to compose with besides the approved ones (model ids)")
    ap.add_argument("--keep", type=int, default=0, help="hold at most this many pre-selected pieces: past it a better piece replaces the weakest (0 = stop at --pieces)")
    ap.add_argument("--prefer", nargs="*", default=[], help="pieces of brains whose name contains one of these are given up last")
    ap.add_argument("--player", default=PLAYER, help="the track factory's address (a tunnel to the box's player); default the Mac's player")
    ap.add_argument("--sync", default=None, help="HOST:PATH of the box's site/tracks folder: finished tracks are copied from there into site/tracks/")
    ap.add_argument("--key", default=os.environ.get("CADANCE_SSH_KEY"), help="the ssh key for --sync (default: CADANCE_SSH_KEY)")
    ap.add_argument("--models", nargs="*", default=[], help="compose with exactly these model ids (not the player's approved list)")
    ap.add_argument("--follow", default=None, help="a dj-approved.json to follow: compose with the brains it lists, re-read before every track")
    ap.add_argument("--session", default="curate-auto", help="the session name at the factory ('curate-auto' is composed at background pace)")
    ap.add_argument("--moods", action="store_true", help="ask each track with another mood, in turn")
    ap.add_argument("--rest", type=float, default=0.0, help="after a track, leave the factory idle this share of the time the track took")
    ap.add_argument("--box-load", type=float, default=0.0, help="with --sync: ask for no track while the box's one-minute load is above this (0 = always ask)")
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--refresh-mp3", action="store_true")
    args = ap.parse_args()
    if args.sync and not args.key:
        ap.error("--sync needs an ssh key: --key KEYFILE or CADANCE_SSH_KEY")
    if args.spread:
        spread()
        raise SystemExit
    if args.refresh_mp3:
        refresh_mp3()
        raise SystemExit
    if args.report:
        for f in sorted(TRACKS.glob("*/*.json")):
            tid = f"{f.parent.name}/{f.name}"
            track, scores, parts, piece, name, extra = measure(tid)
            print(f"{tid[-58:]:58s} passages {' '.join(f'{s:.2f}' for s in scores)}")
            print(f"{'':58s} beat {np.mean([p['beat'] for p in parts]):.2f} bass {np.mean([p['bass'] for p in parts]):.2f} top {np.mean([p['top'] for p in parts]):.2f} "
                  f"change {np.mean([p['change'] for p in parts]):.2f} rewrite {np.mean([p['rewrite'] for p in parts]):.2f} -> "
                  + (f"{piece['bars']} bars from bar {piece['start'] + 1}, mean {piece['mean']:.2f}, weakest {piece['weakest']:.2f}: {name}" if piece else "no piece"))
        raise SystemExit
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    asks = json.loads(ASKS.read_text()) if ASKS.exists() else {}
    until, turn = time.time() + args.hours * 3600, 0
    ctx = dict(pending=None, rest_until=0.0, composed=0, started=time.time(), waited_load=0.0, turn=len(asks))   # the mood cycle goes on over a restart
    log(f"pre-selecting for {args.hours} h" + (f", holding {args.keep} pieces" if args.keep else f" or until {args.pieces} pieces")
        + (f"; factory {args.player}, tracks from {args.sync}" if args.sync else ""))
    while time.time() < until:
        try:
            done = sum(1 for v in state.values() if v.get("item"))
            if not args.keep and done >= args.pieces:
                log(f"{done} pieces pre-selected: done")
                break
            if args.keep:                                    # better pieces in, weaker pieces out
                order = standing(state, args.prefer)
                held = sum(1 for v in state.values() if v.get("item"))
                while held > args.keep and order:
                    _, old_tid, old = order.pop(0)
                    give_up(state, old_tid, old, f"the repertoire holds {held} pre-selected pieces, {args.keep} are kept")
                    held -= 1
                    STATE.write_text(json.dumps(state, indent=1))
            if args.sync:
                for tid in fetch_tracks(args, state):
                    pending = ctx.get("pending")
                    if pending and tid not in pending["known"]:      # the track just asked for: its mood and time travel with it
                        pending["done_at"] = time.time()
                        asks[tid] = dict(model=pending["model"], mood=pending["mood"], seconds=round(time.time() - pending["at"]))
                        ASKS.write_text(json.dumps(asks, indent=1))
            busy = factory_step(args, state, ctx)            # the factory composes on while the Mac measures and renders
            fresh = [f"{f.parent.name}/{f.name}" for f in sorted(TRACKS.glob("*/*.json"), key=lambda p: p.stat().st_mtime) if f"{f.parent.name}/{f.name}" not in state]
            if fresh:
                tid = fresh[0]
                track, scores, parts, piece, name, extra = measure(tid)
                order = standing(state, args.prefer) if args.keep else []
                full = args.keep and sum(1 for v in state.values() if v.get("item")) >= args.keep
                feats = features(track, piece["start"], piece["bars"]) if piece else None
                near = nearest(feats) if piece else (9.0, None)
                # Full: the piece that would go is order[0]. The new one comes in when it measures better, or measures
                # the same within 0.03 and is less like the kept pieces than the one that would go.
                if piece and full and order and order[0][0][0] >= 1:
                    going = order[0]
                    mean_out = going[0][1]
                    alike_out = nearest(going[2].get("features"), skip=(going[2]["item"], going[2].get("sid_item") or ""))[0] if going[2].get("features") else 9.0
                    ahead = (any(p in tid for p in args.prefer) and going[0][0] == 1) or piece["mean"] > mean_out + 0.01
                    other = piece["mean"] >= mean_out - 0.03 and near[0] > alike_out + 0.1
                else:
                    ahead = other = True
                if piece and not (ahead or other):
                    state[tid] = dict(item=None, why=f"the repertoire is full: this one measures {piece['mean']:.2f} against {mean_out:.2f} for the piece that would go, "
                                                     f"and is as like the kept pieces ({near[0]:.2f} from '{str(near[1])[:40]}')")
                    log(f"{tid}: {state[tid]['why']}")
                elif piece and near[0] < 0.15:               # a near copy of a kept piece adds nothing to a set (two tracks of one brain lie 0.6 to 0.95 apart, two halves of one track 0.27 and more)
                    state[tid] = dict(item=None, why=f"nearly the same piece as '{str(near[1])[:50]}' (distance {near[0]:.2f})")
                    log(f"{tid}: {state[tid]['why']}")
                elif not piece:
                    state[tid] = dict(item=None, why=f"no stretch of 160 bars is good enough (mean passage {np.mean(scores):.2f}; beat in {np.mean([p['beat'] for p in parts]):.0%} of the bars, "
                                                     f"bass in {np.mean([p['bass'] for p in parts]):.0%}, lead or pad in {np.mean([p['top'] for p in parts]):.0%})")
                    log(f"{tid}: {state[tid]['why']}")
                else:
                    item = call(CURATOR, "/api/repertoire/add", dict(track=tid, **{"from": piece["start"], "to": piece["start"] + piece["bars"]}, name=name,
                                                                     note=f"pre-selected by measurement: mean passage {piece['mean']:.2f}, weakest {piece['weakest']:.2f}"))
                    if item.get("error"):
                        raise RuntimeError(item["error"])
                    mp3 = SITE / "repertoire" / f"{item['id']}.mp3"
                    for _ in range(120):                      # the site renders the MP3
                        if mp3.is_file():
                            break
                        time.sleep(2)
                    heard = audio_check(mp3) if mp3.is_file() else dict(ok=True, why=None)
                    if not heard["ok"]:
                        call(CURATOR, "/api/repertoire/remove", dict(id=item["id"]))
                        state[tid] = dict(item=None, why=f"its render has {heard['why']}", audio=heard)
                        log(f"{tid}: taken out again, {state[tid]['why']}")
                    else:
                        state[tid] = dict(item=item["id"], name=name, piece=piece, audio=heard, features=feats, nearest=round(near[0], 3),
                                          mood=(asks.get(tid) or {}).get("mood"))
                        if args.sid_every and sum(1 for v in state.values() if v.get("item")) % args.sid_every == 0:
                            try:                              # every few pieces also as the SID chip plays it
                                chip = call(CURATOR, "/api/repertoire/sid", dict(id=item["id"]))
                                state[tid]["sid_item"] = chip.get("id")
                                log(f"{tid}: SID chip version added")
                            except Exception as e:
                                log(f"SID version: {e}")
                        log(f"{tid}: added '{name}', {piece['bars']} bars from bar {piece['start'] + 1} (mean {piece['mean']:.2f}); "
                            f"lows {heard.get('low', 0):.2f} mids {heard.get('mid', 0):.2f} highs {heard.get('high', 0):.2f}")
                STATE.write_text(json.dumps(state, indent=1))
                continue
            time.sleep(10 if args.sync else (20 if busy else 30))
        except Exception as e:
            log(f"{type(e).__name__}: {e}")
            time.sleep(30)
    log("finished")
