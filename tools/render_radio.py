#!/usr/bin/env python
"""Render Auto DJ tracks for the radio's track store: each finished track (site/tracks/<checkpoint>/*.json, as
site/shelf.py and the factory keep them) becomes <out>/tracks/<id>.mp3 (44.1 kHz stereo, 192 kbit/s CBR, the player's
loudness balance, as the curator renders its sections) and its note file <out>/tracks/<id>.json: the brain's name, the
tempo, the pool, the view layout, and per bar the tokens, mods, patterns, patches, the brain's recorded view and the
bar number (the page's Brain and Instruments views read it). The first --skip bars are left out (a track's opening
stays close to the training passage it continued); the passage's title ("seed") is never written, and the MP3 carries
no title but "Cadance radio <id>". tools/build_radio.py manifest <out> then writes <out>/tracks.json.

    python tools/render_radio.py <track file or folder> [...] [--out build/radio-store] [--skip 32] [--bpm 145]

One track at a time, at low priority. --bpm sets the tempo of the rendered score, the note file and the manifest
(default 145, the radio's tempo); the id includes it: the first 12 hex of sha1("<brain>/<file name>|<bpm>"), the
tempo written as 145, not 145.0.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def track_files(items):
    found = []
    for item in items:
        p = Path(item)
        found += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return list(dict.fromkeys(found))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "radio-store")
    ap.add_argument("--skip", type=int, default=32)
    ap.add_argument("--min-bars", type=int, default=64)
    ap.add_argument("--bpm", type=float, default=145.0, help="the tempo the score is rendered at (default 145)")
    args = ap.parse_args()
    (args.out / "tracks").mkdir(parents=True, exist_ok=True)
    try:
        balance = json.loads((SITE / "play-reading.json").read_bytes()).get("balance")
    except (OSError, ValueError):
        balance = None
    node = shutil.which("node") or "/opt/homebrew/bin/node"
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    for f in track_files(args.tracks):
        t = json.loads(f.read_bytes())
        bars = (t.get("bars") or [])[args.skip:]
        brain, bpm, pool = str(t.get("checkpoint") or ""), args.bpm, t.get("pool")
        bpm = int(bpm) if float(bpm).is_integer() else bpm
        if len(bars) < args.min_bars or not bpm or not pool:
            print(f"left out: {f.name} ({len(bars)} bars after the skip)", flush=True)
            continue
        tid = hashlib.sha1(f"{brain}/{f.name}|{bpm}".encode()).hexdigest()[:12]     # the tempo is part of the id
        mp3, note = args.out / "tracks" / f"{tid}.mp3", args.out / "tracks" / f"{tid}.json"
        note.write_text(json.dumps({"brain": brain, "bpm": bpm, "pool": pool, "layout": t.get("layout"),
                                    "bars": [{**{k: b.get(k) for k in ("tokens", "mods", "patterns", "patches", "view") if k in b}, "bar": i + 1}
                                             for i, b in enumerate(bars)]}, separators=(",", ":")))
        if mp3.is_file():
            print(f"{tid}: its MP3 is there already; the note file written", flush=True)
            continue
        score_bars = [{k: b.get(k) for k in ("tokens", "mods", "patterns", "patches") if k in b} for b in bars]
        with tempfile.TemporaryDirectory() as work:
            score, wav = Path(work) / "score.json", Path(work) / "out.wav"
            score.write_text(json.dumps({"pool": str(ROOT / "pool" / pool), "bpm": bpm, "bars": score_bars, "balance": balance}))
            done = subprocess.run(["nice", "-n", "15", node, str(ROOT / "engine" / "render.mjs"), str(score), str(wav), "--tail", "2"],
                                  capture_output=True, text=True, timeout=1800)
            if done.returncode or not wav.is_file():
                sys.exit(f"{f.name}: render failed: {(done.stderr or done.stdout).strip()[-300:]}")
            part = mp3.with_suffix(".part.mp3")
            done = subprocess.run(["nice", "-n", "15", ffmpeg, "-v", "error", "-y", "-i", str(wav), "-map_metadata", "-1",
                                   "-codec:a", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                                   "-metadata", f"title=Cadance radio {tid}", "-metadata", "artist=Cadance", str(part)],
                                  capture_output=True, text=True, timeout=600)
            if done.returncode:
                sys.exit(f"{f.name}: mp3 failed: {done.stderr.strip()[-300:]}")
            part.replace(mp3)
        seconds = round(len(bars) * 240 / float(bpm), 2)
        print(f"{tid}: {mp3.name} {mp3.stat().st_size / 1e6:.1f} MB, {len(bars)} bars, {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
