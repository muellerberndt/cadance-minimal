#!/usr/bin/env python
"""The radio's track list on this computer: listen, veto, rate, publish.

    python tools/track_viewer.py            http://127.0.0.1:8795/

It lists the tracks of the radio's store (the store's tracks.json), plays each from the store, and keeps a
rating per track in a file on this computer (quality/radio_ratings.json): 1 to 5 stars, or a veto. A track
nobody rated counts as 3 stars. The ratings go to the radio by themselves: a quarter of a minute after the last
change the store's tracks.json is written again with the ratings folded in ("publish now" does it at once). A
vetoed track leaves the list, and the others carry "stars" and a "weight" the radio's random pick honours
(1 star 0.25, 2 stars 0.5, 3 stars 1, 4 stars 2, 5 stars 4). Publishing uses the AWS command line of this
computer, so only a computer that may write the store can publish. The server listens on 127.0.0.1 only.

Where a table of the composed tracks exists (quality/loop/radio_tracks.jsonl, one line per composition with
its counts), the list shows a track's counts beside it.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("track_viewer.html")
STORE = os.environ.get("CADANCE_RADIO_STORE", "https://cadance-radio.s3.amazonaws.com").rstrip("/")
BUCKET = os.environ.get("CADANCE_RADIO_BUCKET", "s3://cadance-radio")
RATINGS = Path(os.environ.get("CADANCE_RADIO_RATINGS") or ROOT / "quality" / "radio_ratings.json")
TABLE = ROOT / "quality" / "loop" / "radio_tracks.jsonl"
WEIGHT = {1: 0.25, 2: 0.5, 3: 1.0, 4: 2.0, 5: 4.0}
LOCK = threading.RLock()
QUIET = float(os.environ.get("CADANCE_VIEWER_QUIET", "15"))    # seconds after the last rating before it is published
STATE = {"changed": 0.0, "published": None, "said": "", "error": ""}
COUNTS = ("kick", "snare", "hats", "bass", "lead", "drums_equal_previous", "drum_bars_per_16", "held_bass", "held_lead",
          "four_quarters", "loudness")


def manifest():
    with urllib.request.urlopen(f"{STORE}/tracks.json?t={int(time.time())}", timeout=20) as r:
        return json.loads(r.read())


def ratings():
    try:
        return json.loads(RATINGS.read_text())
    except (OSError, ValueError):
        return {}


def save_ratings(value):
    RATINGS.parent.mkdir(parents=True, exist_ok=True)
    part = RATINGS.with_suffix(".json.part")
    part.write_text(json.dumps(value, indent=1, sort_keys=True))
    part.replace(RATINGS)


def table():
    """The composed tracks' counts by published id, where the table names one."""
    out = {}
    try:
        lines = TABLE.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for key in ("id", "id_140", "public_id"):
            if row.get(key):
                out[str(row[key])] = {k: row[k] for k in COUNTS if isinstance(row.get(k), (int, float))}
    return out


def tracks():
    rated, counts = ratings(), table()
    listed = manifest().get("tracks") or []
    out = []
    for n, t in enumerate(listed, 1):
        r = rated.get(t["id"]) or {}
        out.append({**t, "n": n, "url": f"{STORE}/{t['file']}", "stars": r.get("stars"), "veto": bool(r.get("veto")),
                    "note": r.get("note") or "", "counts": counts.get(t["id"]) or {}})
    listed_ids = {t["id"] for t in listed}
    # vetoed tracks that left the published list stay visible here, so a veto can be taken back
    for tid, r in sorted(rated.items()):
        if r.get("veto") and tid not in listed_ids and r.get("entry"):
            out.append({**r["entry"], "n": None, "url": f"{STORE}/{r['entry']['file']}", "stars": r.get("stars"), "veto": True,
                        "note": r.get("note") or "", "counts": counts.get(tid) or {}, "unlisted": True})
    return out


def publish():
    """The store's tracks.json with the ratings folded in. Vetoed tracks leave the list (their entries are kept in
    the ratings file, so a veto taken back brings them back), the others carry stars and weight."""
    rated = ratings()
    current = manifest()
    entries = {t["id"]: t for t in current.get("tracks") or []}
    for tid, r in rated.items():                       # a veto taken back: the entry returns
        if not r.get("veto") and r.get("entry") and tid not in entries:
            entries[tid] = r["entry"]
    kept, out_of_list = [], 0
    for tid, t in entries.items():
        r = rated.get(tid) or {}
        base = {k: v for k, v in t.items() if k not in ("stars", "weight")}
        if r.get("veto"):
            rated[tid] = {**r, "entry": base}
            out_of_list += 1
            continue
        stars = int(r.get("stars") or 3)
        kept.append({**base, "stars": stars, "weight": WEIGHT.get(stars, 1.0)})
    if not kept:
        raise RuntimeError("every track is vetoed: nothing published")
    save_ratings(rated)
    new = {**{k: v for k, v in current.items() if k != "tracks"}, "tracks": kept,
           "rated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "tracks.json"
        path.write_text(json.dumps(new, indent=1))
        aws = shutil.which("aws") or next((p for p in ("/opt/homebrew/bin/aws", "/usr/local/bin/aws") if os.path.exists(p)), "aws")
        done = subprocess.run([aws, "s3", "cp", str(path), f"{BUCKET}/tracks.json", "--content-type", "application/json",
                               "--cache-control", "public, max-age=60", "--only-show-errors"], capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise RuntimeError("the store refused the list: " + (done.stderr.strip()[-200:] or "aws s3 cp failed"))
    STATE.update(published=time.strftime("%H:%M:%S"), said=f"{len(kept)} tracks on the radio, {out_of_list} vetoed and out", error="")
    return {"published": len(kept), "vetoed": out_of_list}


def in_step():
    """Whether the store's list already carries the ratings: no vetoed track in it, every rated track at its stars."""
    rated = ratings()
    listed = {t["id"]: t for t in manifest().get("tracks") or []}
    for tid, r in rated.items():
        if r.get("veto"):
            if tid in listed:
                return False
        elif r.get("stars") and tid in listed and listed[tid].get("stars") != r["stars"]:
            return False
        elif not r.get("veto") and r.get("entry") and tid not in listed:
            return False
    return True


def keeper():
    """Publish by itself: QUIET seconds after the last rating, and whenever the store's list has fallen out of step
    with the ratings (another writer added tracks and wrote an older state)."""
    checked = 0.0
    while True:
        time.sleep(3)
        try:
            due = STATE["changed"] and time.time() - STATE["changed"] >= QUIET
            if not due and time.time() - checked >= 120:
                checked = time.time()
                due = not in_step()
            if due:
                with LOCK:
                    STATE["changed"] = 0.0
                    publish()
        except Exception as e:  # noqa: BLE001 - shown on the page, tried again at the next change or check
            STATE["error"] = str(e)[:200]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, status, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _state():
        wait = max(0.0, QUIET - (time.time() - STATE["changed"])) if STATE["changed"] else None
        return {"pending": wait is not None, "in": None if wait is None else round(wait), "published": STATE["published"],
                "said": STATE["said"], "error": STATE["error"]}

    def do_GET(self):
        try:
            if self.path.split("?")[0] == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path.split("?")[0] == "/api/tracks":
                return self._send(200, {"tracks": tracks(), "store": STORE, "weights": WEIGHT, "state": self._state()})
            if self.path.split("?")[0] == "/api/state":
                return self._send(200, self._state())
            return self._send(404, {"error": "no such page"})
        except Exception as e:  # noqa: BLE001 - the page shows the reason
            return self._send(500, {"error": str(e)[:300]})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/api/rate":
                tid = str(body.get("id") or "")
                if not tid:
                    return self._send(400, {"error": "no track id"})
                with LOCK:
                    rated = ratings()
                    r = dict(rated.get(tid) or {})
                    if "stars" in body:
                        stars = body["stars"]
                        if stars is None:
                            r.pop("stars", None)
                        elif int(stars) in WEIGHT:
                            r["stars"] = int(stars)
                    if "veto" in body:
                        r["veto"] = bool(body["veto"])
                    if "note" in body:
                        r["note"] = str(body["note"])[:300]
                    r["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    if r.get("stars") is None and not r.get("veto") and not r.get("note") and not r.get("entry"):
                        rated.pop(tid, None)
                    else:
                        rated[tid] = r
                    save_ratings(rated)
                    STATE["changed"] = time.time()
                return self._send(200, {"ok": True, "rating": r})
            if self.path == "/api/publish":
                with LOCK:
                    return self._send(200, publish())
            return self._send(404, {"error": "no such call"})
        except Exception as e:  # noqa: BLE001 - the page shows the reason
            return self._send(500, {"error": str(e)[:300]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("CADANCE_VIEWER_PORT", "8795")))
    args = ap.parse_args()
    threading.Thread(target=keeper, daemon=True).start()
    print(f"http://127.0.0.1:{args.port}/  tracks of {STORE}, ratings in {RATINGS}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
