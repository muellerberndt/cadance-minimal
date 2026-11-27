"""Cadance curator: a site of its own for building the Auto DJ's repertoire.

It shows the tracks the brains composed (the track factory's, site/tracks/), plays them, and lets the owner mark
successive 16-bar passages and add them to the repertoire as one piece. A piece is one file with its bars
(site/repertoire/<id>.json), a 192 kbit/s MP3 of it as the engine plays it (<id>.mp3), a name and a rating of one
to five stars; all of it is also in the bucket under minimal/repertoire/. The Auto DJ of the mixer page plays the
repertoire when it is chosen as a source and prefers the pieces with more stars. It plays a piece from its bars,
on a deck's instrument, so the piece follows the set's tempo; the MP3 is the piece as a sound file.

Above a track's timeline the editor lists suggested passages, and under each 16-bar passage it shows three scores:
like the material, synth line, breakbeat (site/suggest.py, from the play report's measures).

This server loads no brain and imports no Cadence. It is not linked from the player or the mixer.

    python site/curator.py            http://127.0.0.1:8794/
"""
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
PORT = int(os.environ.get("CADANCE_CURATOR_PORT", "8794"))
PLAYER = os.environ.get("CADANCE_PLAYER", "http://127.0.0.1:8792")       # the player's server: it composes new tracks
TRACKS = SITE / "tracks"
REPERTOIRE = Path(os.environ.get("CADANCE_REPERTOIRE") or SITE / "repertoire")
def _local(name, default=""):
    """A setting of this computer alone: the environment, else site/local.json (kept out of the repository)."""
    if os.environ.get(name) is not None:
        return os.environ[name]
    try:
        return json.loads((SITE / "local.json").read_text()).get(name, default)
    except (OSError, ValueError):
        return default


_BUCKET = _local("CADANCE_BUCKET")                                        # "s3://bucket/prefix"; empty: no bucket
BUCKET = _BUCKET + "/repertoire"
OFFLINE = os.environ.get("CADANCE_CURATOR_OFFLINE") == "1" or not _BUCKET  # tests, or no bucket: never touch one
STATIC = {"/": SITE / "curator.html", "/curator.js": SITE / "curator.js", "/engine.mjs": ROOT / "engine" / "engine.mjs",
          "/worklet.js": SITE / "worklet.js", "/kit.js": SITE / "kit.js", "/play-reading.json": SITE / "play-reading.json",
          "/list": SITE / "repertoire-list.html"}      # the plain list: listen, rate, sort, rename, delete
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"}
TRACK_ID = re.compile(r"[A-Za-z0-9_]{1,80}/[A-Za-z0-9_]{1,60}\.json")
ITEM_ID = re.compile(r"r[0-9]{8}T[0-9]{6}_[0-9a-f]{4}")
LOCK = threading.RLock()
SYNC = {"state": "offline" if OFFLINE else "starting", "error": None, "at": None}


def aws(*args, timeout=120):
    exe = shutil.which("aws") or "/usr/local/bin/aws"
    done = subprocess.run([exe, "s3", *args, "--only-show-errors"], capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(done.stderr.strip()[-300:] or "aws failed")


def sync(job):
    """Run one bucket operation in the background and record how it went."""
    if OFFLINE:
        return

    def run():
        SYNC.update(state="syncing", error=None)
        try:
            job()
            SYNC.update(state="synced", at=time.strftime("%H:%M:%S"))
        except Exception as e:
            SYNC.update(state="not synced", error=str(e)[:300])
    threading.Thread(target=run, daemon=True).start()


def meta(item):
    return {k: v for k, v in item.items() if k not in ("bars", "layout")}


def items():
    out = []
    for f in sorted(REPERTOIRE.glob("r*.json")) if REPERTOIRE.exists() else []:
        try:
            item = json.loads(f.read_bytes())
        except ValueError:
            continue
        if ITEM_ID.fullmatch(item.get("id", "")) and isinstance(item.get("bars"), list):
            out.append(item)
    return sorted(out, key=lambda i: i.get("created", ""), reverse=True)


def write_index():
    index = [meta(i) for i in items()]
    part = REPERTOIRE / "index.json.part"
    part.write_text(json.dumps(index, indent=1))
    part.replace(REPERTOIRE / "index.json")
    return index


def tracks():
    marks = {}
    for i in items():
        marks[i.get("source")] = marks.get(i.get("source"), 0) + 1
    out = []
    for f in sorted(TRACKS.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if TRACKS.exists() else []:
        tid = f"{f.parent.name}/{f.name}"
        try:
            data = json.loads(f.read_bytes())
        except ValueError:
            continue
        out.append({"id": tid, "checkpoint": data.get("checkpoint") or f.parent.name, "created": data.get("created"), "bars": len(data.get("bars") or []),
                    "seed": data.get("seed"), "bpm": data.get("bpm"), "pool": data.get("pool"), "marks": marks.get(tid, 0)})
    counts = suggested_counts()
    for t in out:
        t["suggested"] = counts.get(t["id"])
    return out


# Passage suggestions (site/suggest.py): per 16-bar passage three scores (like the material, synth line, breakbeat)
# and the runs where they stay high. Loaded on first use; when it cannot be loaded the editor shows no suggestions.
_SUGGEST = {}


def suggester():
    if "module" not in _SUGGEST:
        try:
            import suggest
            _SUGGEST["module"] = suggest
        except Exception as e:
            _SUGGEST["module"], _SUGGEST["error"] = None, f"{type(e).__name__}: {e}"
    return _SUGGEST["module"]


def suggested_counts():
    """{track id: number of suggestions, or None while it is being scored}; scores the tracks not yet scored in the
    background."""
    S = suggester()
    if not S or not TRACKS.exists():
        return {}
    pairs = [(f"{f.parent.name}/{f.name}", f) for f in TRACKS.glob("*/*.json")]
    S.warm(pairs)
    return {tid: S.count(tid, f) for tid, f in pairs}


def suggest_track(tid):
    if not TRACK_ID.fullmatch(tid or ""):
        raise ValueError("unknown track")
    f = (TRACKS / tid).resolve()
    if TRACKS.resolve() not in f.parents or not f.is_file():
        raise ValueError("unknown track")
    S = suggester()
    if not S:
        return {"error": "no suggestions: " + _SUGGEST.get("error", "site/suggest.py did not load")}
    return S.for_track(tid, f)


# A name for a piece added without one: two words drawn from lists in the spirit of minimal techno, so pieces are
# told apart at a glance (the opening context's title, the old default, named every piece of a track alike).
_NAME_A = ["Velvet", "Rubber", "Hollow", "Static", "Sleepy", "Copper", "Neon", "Paper", "Liquid", "Dusty", "Tiny", "Midnight",
           "Plastic", "Quiet", "Crooked", "Glass", "Lazy", "Magnetic", "Foggy", "Orbital", "Pocket", "Silent", "Wobbly", "Frozen",
           "Analog", "Woolly", "Tangled", "Distant", "Humming", "Sticky", "Polar", "Soft", "Crystal", "Haunted", "Elastic", "Lunar"]
_NAME_B = ["Hamster", "Basement", "Pulse", "Tram", "Lantern", "Ferret", "Teacup", "Comet", "Pickle", "Oscillator", "Ghost",
           "Spoon", "Turntable", "Mole", "Kettle", "Heron", "Satellite", "Waffle", "Loop", "Raccoon", "Submarine", "Biscuit",
           "Tunnel", "Moth", "Metronome", "Badger", "Elevator", "Pebble", "Walrus", "Parking Lot", "Goldfish", "Ferry",
           "Thermostat", "Noodle", "Crane", "Sidewalk"]


def invent_name():
    taken = {i.get("name") for i in items()}
    for _ in range(50):
        name = f"{secrets.choice(_NAME_A)} {secrets.choice(_NAME_B)}"
        if name not in taken:
            return name
    return f"{secrets.choice(_NAME_A)} {secrets.choice(_NAME_B)} {secrets.randbelow(90) + 10}"


def remove_tracks(ids, unused_only=True):
    """Delete composed tracks from site/tracks/. A track a repertoire piece came from is kept unless unused_only is
    off. Deleted ids go to site/tracks/.deleted, so the repertoire job does not copy them from the box again."""
    used = {i.get("source") for i in items()}
    gone = []
    with LOCK:
        for tid in ids:
            if not TRACK_ID.fullmatch(tid or "") or (unused_only and tid in used):
                continue
            f = (TRACKS / tid).resolve()
            if TRACKS.resolve() not in f.parents or not f.is_file():
                continue
            f.unlink()
            gone.append(tid)
        if gone:
            with open(TRACKS / ".deleted", "a") as out:
                out.write("".join(f"{tid}\n" for tid in gone))
    return {"deleted": gone, "kept_in_use": [t for t in ids if t in used]}


def read_track(tid):
    if not TRACK_ID.fullmatch(tid or ""):
        raise ValueError("unknown track")
    f = (TRACKS / tid).resolve()
    if TRACKS.resolve() not in f.parents or not f.is_file():
        raise ValueError("unknown track")
    return json.loads(f.read_bytes())


def play_settings_mark():
    """A short mark of the player's sound settings (the balance in site/play-reading.json) and of the engine: an MP3
    carries the mark it was rendered with, so a piece whose MP3 no longer matches what the decks play can be found
    (site/curate_auto.py --refresh-mp3 renders those again)."""
    import hashlib
    h = hashlib.sha256()
    try:
        h.update(json.dumps(json.loads((SITE / "play-reading.json").read_bytes()).get("balance"), sort_keys=True).encode())
    except (OSError, ValueError):
        h.update(b"as written")
    try:
        h.update((ROOT / "engine" / "engine.mjs").read_bytes())
    except OSError:
        pass
    return h.hexdigest()[:12]


def render_mp3(item_id):
    """The section as sound: the engine renders it with the player's loudness balance, ffmpeg makes a 192 kbit/s MP3."""
    import tempfile
    f = REPERTOIRE / f"{item_id}.json"
    item = json.loads(f.read_bytes())
    try:
        balance = json.loads((SITE / "play-reading.json").read_bytes()).get("balance")
    except (OSError, ValueError):
        balance = None
    settings = play_settings_mark()                 # which player settings this MP3 is rendered with
    with tempfile.TemporaryDirectory() as work:
        score, wav, mp3 = Path(work) / "score.json", Path(work) / "out.wav", REPERTOIRE / f"{item_id}.mp3"
        score.write_text(json.dumps({"pool": str(ROOT / "pool" / item["pool"]), "bpm": item.get("bpm") or 126, "bars": item["bars"], "balance": balance}))
        node = shutil.which("node") or "/opt/homebrew/bin/node"
        done = subprocess.run(["nice", "-n", "15", node, str(ROOT / "engine/render.mjs"), str(score), str(wav), "--tail", "2"], capture_output=True, text=True, timeout=900)
        if done.returncode or not wav.is_file():
            raise RuntimeError("render failed: " + (done.stderr or done.stdout).strip()[-200:])
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        part = mp3.with_suffix(".part.mp3")
        done = subprocess.run(["nice", "-n", "15", ffmpeg, "-v", "error", "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "192k",
                               "-metadata", f"title={item['name']}", "-metadata", "artist=Cadance", str(part)], capture_output=True, text=True, timeout=300)
        if done.returncode:
            raise RuntimeError("mp3 failed: " + done.stderr.strip()[-200:])
        part.replace(mp3)
    with LOCK:
        item = json.loads(f.read_bytes())
        item.update(mp3=mp3.name, mp3_kbps=192, seconds=round(len(item["bars"]) * 240 / (item.get("bpm") or 126), 1),
                    mp3_settings=settings, mp3_rendered=time.strftime("%Y-%m-%dT%H:%M:%S"))
        item.pop("mp3_error", None)
        f.write_text(json.dumps(item, separators=(",", ":")))
        write_index()
    if not OFFLINE:
        aws("cp", str(mp3), f"{BUCKET}/{mp3.name}")


def export(item_id):
    """Render the section's MP3 and put the section, its MP3 and the index into the bucket (in the background)."""
    def run():
        SYNC.update(state="rendering", error=None)
        f = REPERTOIRE / f"{item_id}.json"
        try:
            render_mp3(item_id)
        except Exception as e:
            with LOCK:
                if f.is_file():
                    item = json.loads(f.read_bytes())
                    item["mp3_error"] = str(e)[:200]
                    f.write_text(json.dumps(item, separators=(",", ":")))
                    write_index()
            SYNC.update(state="not synced", error=str(e)[:300])
            if OFFLINE:
                return
        if OFFLINE:
            SYNC.update(state="offline")
            return
        try:
            aws("cp", str(f), f"{BUCKET}/{f.name}")
            aws("cp", str(REPERTOIRE / "index.json"), f"{BUCKET}/index.json")
            if SYNC.get("state") != "not synced":
                SYNC.update(state="synced", at=time.strftime("%H:%M:%S"))
        except Exception as e:
            SYNC.update(state="not synced", error=str(e)[:300])
    threading.Thread(target=run, daemon=True).start()


def add(body):
    track = read_track(body.get("track"))
    a, b = int(body.get("from", 0)), int(body.get("to", 0))
    bars = track["bars"]
    if not (0 <= a < b <= len(bars)) or b - a < 8 or b - a > 352:
        raise ValueError("a piece is 8 to 352 bars inside its track")
    name = str(body.get("name") or "").strip()[:80] or invent_name()
    bpm = body.get("bpm")
    bpm = float(bpm) if isinstance(bpm, (int, float)) and 60 <= bpm <= 200 else track.get("bpm")
    item = {"id": f"r{time.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(2)}", "name": name, "note": str(body.get("note") or "")[:300], "stars": 0,
            "checkpoint": track.get("checkpoint"), "pool": track.get("pool"), "bpm": bpm, "seed": track.get("seed"),
            "source": body["track"], "from": a, "to": b, "length": b - a, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "layout": track.get("layout"),
            "bars": bars[a:b]}
    with LOCK:
        REPERTOIRE.mkdir(parents=True, exist_ok=True)
        f = REPERTOIRE / f"{item['id']}.json"
        part = f.with_suffix(".part")
        part.write_text(json.dumps(item, separators=(",", ":")))
        part.replace(f)
        write_index()
    export(item["id"])
    return meta(item)


SID_POOL = "pool_underground_sid.json"     # every sound of the pool fitted on the engine's SID chip voice


def sid_version(item_id):
    """The same piece played by the SID chip: the brain's notes and choice of sounds, every synth sound in its SID
    fit. The brain did not write the chip's registers; the fits did. A piece of its own in the repertoire."""
    if not ITEM_ID.fullmatch(item_id or ""):
        raise ValueError("unknown piece")
    with LOCK:
        f = REPERTOIRE / f"{item_id}.json"
        if not f.is_file() or not (ROOT / "pool" / SID_POOL).is_file():
            raise ValueError("unknown piece, or no SID pool")
        item = json.loads(f.read_bytes())
        if item.get("sid"):
            raise ValueError("this piece is a SID version already")
        new = dict(item, id=f"r{time.strftime('%Y%m%dT%H%M%S')}_{secrets.token_hex(2)}", name=(item["name"][:68] + " · SID chip"), pool=SID_POOL, sid=True,
                   of=item["id"], stars=0, created=time.strftime("%Y-%m-%dT%H:%M:%S"), note="the SID chip plays this piece: the SID fits of its sounds, the brain's notes",
                   bars=[dict(b, patches=None) for b in item["bars"]])
        for key in ("mp3", "mp3_kbps", "seconds", "mp3_error"):
            new.pop(key, None)
        g = REPERTOIRE / f"{new['id']}.json"
        g.write_text(json.dumps(new, separators=(",", ":")))
        write_index()
    export(new["id"])
    return meta(new)


def remove(item_id):
    if not ITEM_ID.fullmatch(item_id or ""):
        raise ValueError("unknown section")
    with LOCK:
        f = REPERTOIRE / f"{item_id}.json"
        if not f.is_file():
            raise ValueError("unknown section")
        f.unlink()
        (REPERTOIRE / f"{item_id}.mp3").unlink(missing_ok=True)
        write_index()
    sync(lambda: (aws("rm", f"{BUCKET}/{item_id}.json"), aws("rm", f"{BUCKET}/{item_id}.mp3"), aws("cp", str(REPERTOIRE / "index.json"), f"{BUCKET}/index.json")))


def rename(item_id, name, note):
    if not ITEM_ID.fullmatch(item_id or ""):
        raise ValueError("unknown section")
    with LOCK:
        f = REPERTOIRE / f"{item_id}.json"
        item = json.loads(f.read_bytes())
        item["name"] = str(name or item["name"]).strip()[:80] or item["name"]
        if note is not None:
            item["note"] = str(note)[:300]
        f.write_text(json.dumps(item, separators=(",", ":")))
        write_index()
    sync(lambda: (aws("cp", str(f), f"{BUCKET}/{f.name}"), aws("cp", str(REPERTOIRE / "index.json"), f"{BUCKET}/index.json")))
    return meta(item)


def rate(item_id, stars):
    if not ITEM_ID.fullmatch(item_id or ""):
        raise ValueError("unknown section")
    with LOCK:
        f = REPERTOIRE / f"{item_id}.json"
        item = json.loads(f.read_bytes())
        item["stars"] = int(min(max(int(stars), 0), 5))
        f.write_text(json.dumps(item, separators=(",", ":")))
        write_index()
    sync(lambda: (aws("cp", str(f), f"{BUCKET}/{f.name}"), aws("cp", str(REPERTOIRE / "index.json"), f"{BUCKET}/index.json")))
    return meta(item)


def pull():
    """The bucket's sections this machine does not have yet (nothing local is removed)."""
    REPERTOIRE.mkdir(parents=True, exist_ok=True)
    aws("sync", BUCKET + "/", str(REPERTOIRE), "--exclude", "*", "--include", "r*.json", "--include", "r*.mp3", timeout=600)
    with LOCK:
        write_index()


def player(path, body=None):
    request = urllib.request.Request(PLAYER + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    return json.loads(urllib.request.urlopen(request, timeout=30).read())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_range(self, data, ctype):
        """A file with HTTP ranges (a browser seeks in an MP3 only when the server answers a range request)."""
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if not m or (not m.group(1) and not m.group(2)):
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            part = data
        else:
            n = len(data)
            if m.group(1):
                a, b = int(m.group(1)), int(m.group(2)) if m.group(2) else n - 1
            else:
                a, b = max(0, n - int(m.group(2))), n - 1
            if a >= n:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{n}")
                self.end_headers()
                return
            b = min(b, n - 1)
            part = data[a:b + 1]
            self.send_response(206)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {a}-{b}/{n}")
            self.send_header("Content-Length", str(len(part)))
        self.send_header("Cache-Control", "no-store")
        try:
            self.end_headers()
            self.wfile.write(part)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        from urllib.parse import parse_qs
        path, _, query = self.path.partition("?")
        q = parse_qs(query)
        try:
            if path.startswith("/kit/"):
                kit = ROOT / "engine" / "samples"
                f = (kit / path[len("/kit/"):]).resolve()
                if f.is_file() and kit.resolve() in f.parents and f.suffix in (".wav", ".json"):
                    return self._send(200, f.read_bytes(), "audio/wav" if f.suffix == ".wav" else "application/json")
                return self._send(404, {"error": "not found"})
            if path in STATIC and STATIC[path].exists():
                f = STATIC[path]
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            if path == "/api/tracks":
                return self._send(200, {"tracks": tracks()})
            if path == "/api/track":
                return self._send(200, read_track(q.get("id", [""])[0]))
            if path == "/api/track/suggest":          # per passage three scores, and the suggested runs of passages
                return self._send(200, suggest_track(q.get("id", [""])[0]))
            if path == "/api/suggested":              # the number of suggestions of each track scored so far
                return self._send(200, {"counts": suggested_counts()})
            if path == "/api/pool":
                name = q.get("name", [""])[0]
                if not re.fullmatch(r"[A-Za-z0-9_]{1,60}\.json", name) or not (ROOT / "pool" / name).is_file():
                    return self._send(404, {"error": "no such pool"})
                return self._send(200, (ROOT / "pool" / name).read_bytes())
            if path == "/api/repertoire/settings":     # the mark an MP3 rendered now would carry
                return self._send(200, {"mark": play_settings_mark()})
            if path == "/api/repertoire":
                return self._send(200, {"items": [meta(i) for i in items()], "sync": SYNC, "bucket": BUCKET})
            if path == "/api/repertoire/item":
                item_id = q.get("id", [""])[0]
                f = REPERTOIRE / f"{item_id}.json"
                if not ITEM_ID.fullmatch(item_id) or not f.is_file():
                    return self._send(404, {"error": "unknown section"})
                return self._send(200, f.read_bytes())
            if path.startswith("/repertoire/") and path.endswith(".mp3"):
                name = path[len("/repertoire/"):]
                if not ITEM_ID.fullmatch(name[:-4]) or not (REPERTOIRE / name).is_file():
                    return self._send(404, {"error": "no such file"})
                return self._send_range((REPERTOIRE / name).read_bytes(), "audio/mpeg")
            if path == "/api/composer":            # what the player's track factory is doing, and the brains it can use
                try:
                    return self._send(200, {"models": player("/api/autodj/models"), "status": player("/api/autodj/status")})
                except Exception as e:
                    return self._send(200, {"error": f"the player's server does not answer: {e}"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/repertoire/add":
                return self._send(200, add(body))
            if self.path == "/api/tracks/remove":          # composed tracks no repertoire piece came from
                ids = body.get("ids") or ([body.get("id")] if body.get("id") else [])
                if body.get("all_unused"):
                    ids = [t["id"] for t in tracks() if not t["marks"]]
                return self._send(200, remove_tracks([str(x) for x in ids]))
            if self.path == "/api/repertoire/remove":
                remove(body.get("id"))
                return self._send(200, {"ok": True})
            if self.path == "/api/repertoire/rename":
                return self._send(200, rename(body.get("id"), body.get("name"), body.get("note")))
            if self.path == "/api/repertoire/sid":
                return self._send(200, sid_version(body.get("id")))
            if self.path == "/api/repertoire/rate":
                return self._send(200, rate(body.get("id"), body.get("stars", 0)))
            if self.path == "/api/repertoire/mp3":      # render a section's MP3 again
                if not ITEM_ID.fullmatch(body.get("id") or "") or not (REPERTOIRE / f"{body['id']}.json").is_file():
                    raise ValueError("unknown section")
                export(body["id"])
                return self._send(200, {"ok": True})
            if self.path == "/api/repertoire/pull":
                sync(pull)
                return self._send(200, {"ok": True})
            if self.path == "/api/compose":       # ask the player's track factory for one more track of a brain
                bars = int(min(max(int(body.get("bars", 288)), 64), 352))
                return self._send(200, player("/api/autodj/track", {"model": str(body.get("model") or "leader"), "bars": bars}))
        except (ValueError, KeyError) as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    REPERTOIRE.mkdir(parents=True, exist_ok=True)
    sync(pull)
    print(f"http://127.0.0.1:{PORT}/  repertoire in {REPERTOIRE} and {BUCKET}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
