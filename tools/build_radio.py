#!/usr/bin/env python
"""Build the Cadance Radio site (cadance.floatingpragma.ai) and its track manifest.

    python tools/build_radio.py page [--out build/radio] [--brain <checkpoint>] [--cname cadance.floatingpragma.ai]
    python tools/build_radio.py manifest <store folder>

page: the gh-pages folder: index.html (site/radio.html), radio.css, the scripts (radio.js: the track store and the
page's /api calls; radio-app.js: the page; autodj.js, brainstage.js, brainviz.js, djfx.js, mixerstage.js, devices.js
with the engine's ui.mjs; engine.mjs with deckworklet.js, the instrument that plays a live brain's tracks), the favicon and the consent script (site/brand/), the preview
image og.png (site/brand/og.png, made from site/brand/social-card.html), the one brain file the 3D views draw (site/.brain3d/<brain>.v*.b3d, made by the player, as
brain3d/brain.b3d), CNAME and .nojekyll. The page plays MP3s; it carries no score, pool or sample (--engine, the
decks that render the notes in the browser, comes with the next update).

manifest: tracks.json for a store folder with tracks/<id>.mp3 and their note files tracks/<id>.json (written by
tools/render_radio.py): {"tracks": [{id, file, brain, bars, bpm, seconds, score}], "made"}. The note files are checked
too: nothing but the brain's name, tempo, pool, layout and per bar tokens, mods, patterns, patches, the view and the bar.
Every file written is checked for strings that look private (a bucket path, a home folder, a host, an audio title).
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PAGE = {"index.html": SITE / "radio.html", "radio.css": SITE / "radio.css", "radio.js": SITE / "radio.js", "radio-app.js": SITE / "radio-app.js",
        "autodj.js": SITE / "autodj.js", "brainstage.js": SITE / "brainstage.js", "brainviz.js": SITE / "brainviz.js", "djfx.js": SITE / "djfx.js",
        "devices.js": SITE / "devices.js", "mixerstage.js": SITE / "mixerstage.js", "deckworklet.js": SITE / "deckworklet.js",
        "ui.mjs": ROOT / "engine" / "ui.mjs", "engine.mjs": ROOT / "engine" / "engine.mjs",
        "favicon.svg": SITE / "brand" / "favicon.svg", "assets/consent.js": SITE / "brand" / "consent.js"}
BINARY = {"og.png": SITE / "brand" / "og.png"}         # the preview image of a shared link (made from site/brand/social-card.html)
NOTE_KEYS = {"brain", "bpm", "pool", "layout", "bars"}
BAR_KEYS = {"tokens", "mods", "patterns", "patches", "view", "bar"}
CNAME = "cadance.floatingpragma.ai"
DEFAULT_BRAIN = "slate_inst_reads_s2_echo2_e001b002"
PRIVATE = [re.compile(p, re.I) for p in (r"s3://", r"/Users/", r"\bec2-", r"\.wav\b", r"\.flac\b", r"\bseed\b")]


def strings(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(str(k))
            strings(v, out)
    elif isinstance(x, list):
        for v in x:
            strings(v, out)
    elif isinstance(x, str):
        out.add(x)
    return out


def checked(what, texts, patterns=PRIVATE):
    bad = sorted(t for t in texts if any(p.search(t) for p in patterns))
    if bad:
        sys.exit(f"{what}: refusing to publish strings that look private: {bad[:5]}")


def clean_b3d(data):
    """The packed brain with a header the browser's JSON.parse reads: brainview.py writes NaN for a lobe of the compact
    brain without voxels (27 Nov: the learned code's lobe), which JSON.parse refuses, so the page fell back to its
    stand-in brain. The NaNs become 0; the header is padded with spaces so the arrays after it stay on 4-byte bounds
    (their offsets count from the header's end)."""
    import struct
    if data[:4] != b"B3D1":
        sys.exit("the brain file is not a packed 3D brain (B3D1)")
    n = struct.unpack("<I", data[4:8])[0]
    head = re.sub(r"-?\bNaN\b|-?\bInfinity\b", "0", data[8:8 + n].decode())
    json.loads(head)
    raw = head.encode()
    raw += b" " * ((-(8 + len(raw))) % 4)
    return b"B3D1" + struct.pack("<I", len(raw)) + raw + data[8 + n:]


def nan_positions(data):
    """The float arrays of a packed brain that hold NaN: the views that draw from them show nothing there."""
    import struct
    from array import array
    n = struct.unpack("<I", data[4:8])[0]
    head = json.loads(re.sub(r"-?\bNaN\b|-?\bInfinity\b", "0", data[8:8 + n].decode()))
    out = []
    for k, (off, ln, dt) in head["arrays"].items():
        if dt == "<f4":
            a = array("f", data[8 + n + off:8 + n + off + ln])
            if sys.byteorder != "little":
                a.byteswap()
            if any(x != x for x in a):
                out.append(k)
    return out


def page(args):
    out = args.out.resolve()
    if out == ROOT or SITE in [out, *out.parents]:
        sys.exit(f"{out}: the output must be a folder of its own, not the repository or site/")
    if args.engine:
        sys.exit("--engine: the decks that render the notes come with the next update; this build plays MP3s")
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    (out / "brain3d").mkdir()
    for name, src in PAGE.items():
        text = src.read_text()
        checked(name, {text}, [re.compile(p) for p in (r"/Users/", r"s3://", r"\bec2-")])
        (out / name).write_text(text)
    for name, src in BINARY.items():
        shutil.copyfile(src, out / name)
    made = sorted((SITE / ".brain3d").glob(f"{args.brain}.v*.b3d"), key=lambda p: p.stat().st_mtime)
    if not made:
        sys.exit(f"no 3D view of {args.brain} in site/.brain3d (the player makes it the first time a page shows that brain in 3D)")
    bad = nan_positions(made[-1].read_bytes())
    if bad:
        sys.exit(f"{made[-1].name} holds NaN in {bad}: the 3D views that use them draw nothing; rebuild it with the "
                 "fixed site/brainview.py (delete the file, the player makes it again)")
    (out / "brain3d" / "brain.b3d").write_bytes(clean_b3d(made[-1].read_bytes()))
    if args.cname:
        (out / "CNAME").write_text(args.cname + "\n")
    (out / ".nojekyll").write_text("")
    if args.cname:  # search engines: one page, the radio
        site = f"https://{args.cname}/"
        (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {site}sitemap.xml\n")
        (out / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                                         '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                         f"  <url><loc>{site}</loc><lastmod>{time.strftime('%Y-%m-%d')}</lastmod></url>\n</urlset>\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}: the radio page, {len(PAGE) + len(BINARY)} files and the brain {args.brain} for the 3D views; {size / 1e6:.1f} MB")


def manifest(args):
    folder = args.folder.resolve()
    tracks = []
    for mp3 in sorted((folder / "tracks").glob("*.mp3")):
        tid = mp3.stem
        note = mp3.with_suffix(".json")
        if not note.is_file():
            print(f"{tid}: no note file, left out")
            continue
        t = json.loads(note.read_bytes())
        if set(t) - NOTE_KEYS or any(set(b) - BAR_KEYS for b in t.get("bars") or []):
            sys.exit(f"{note.name}: keys beyond the note format: {sorted(set(t) - NOTE_KEYS)}")
        checked(note.name, strings(t, set()))
        bars, bpm = len(t["bars"]), float(t["bpm"])
        tracks.append({"id": tid, "file": f"tracks/{mp3.name}", "brain": t["brain"], "bars": bars, "bpm": bpm,
                       "seconds": round(bars * 240 / bpm, 1), "score": f"tracks/{note.name}"})
    data = {"tracks": tracks, "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    checked("tracks.json", strings(data, set()))
    (folder / "tracks.json").write_text(json.dumps(data, indent=1))
    print(f"{folder / 'tracks.json'}: {len(tracks)} tracks, {sum(t['seconds'] for t in tracks) / 3600:.2f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("page")
    p.add_argument("--out", type=Path, default=ROOT / "build" / "radio")
    p.add_argument("--brain", default=DEFAULT_BRAIN, help="the brain whose 3D view the page draws")
    p.add_argument("--cname", default=CNAME)
    p.add_argument("--engine", action="store_true", help="decks that render the notes (the next update)")
    m = sub.add_parser("manifest")
    m.add_argument("folder", type=Path)
    args = ap.parse_args()
    page(args) if args.cmd == "page" else manifest(args)


if __name__ == "__main__":
    main()
