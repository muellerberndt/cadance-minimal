"""Liked and disliked selections from the player's waveform (site/wave.js), kept for training rounds.

The page posts one mark per selection of whole bars: POST /api/feedback {"mark": "liked" | "disliked", ...}.
The server writes one JSON file per mark and appends one line to the index. Nothing here trains or scores:
a trainer or critic reads the files. GET /api/feedback lists the marks (newest first), GET /api/feedback?id=<id>
returns one mark, DELETE /api/feedback?id=<id> removes it (its file and its index line).

Files in site/feedback/:
    <id>.json      one mark: the record below, UTF-8 JSON
    index.jsonl    one line per mark present, in the order they were made: the record without "bars",
                   "brains", "play_reading" and "balance" (the summary the page lists), plus "file"
    feedback.jsonl the heart log of Creature.feedback (rating per bar window), another format, untouched here

A mark, format "cadance-mark/1":
    id             "m20261126-031502-3f9a": the server's local time of the mark and four hex digits; the file name
    format         "cadance-mark/1"
    mark           "liked" | "disliked"
    note           the listener's note ("" when none), at most 500 characters
    time           the server's local time of the mark, "2026-11-26T03:15:02"
    page_time      the page's clock at the mark, ISO 8601 UTC
    bpm            the tempo in force at the mark (every bar also carries the tempo it started with)
    bars_count     the number of bars
    first_bar, last_bar   the server bar numbers of the first and the last bar
    seconds        the selection's length as played
    states         how many of the bars carry the brain's slot vector (see bars: state)
    checkpoints    the checkpoints that wrote the bars, in the order they first appear (null: the instrument check)
    brains         {checkpoint: {candidate, epoch, label, created, minutes, pool, pool_sha, streams,
                    tokens_per_layer, mod_levels, layout}} from the installed cards; layout (common.layout_key)
                    only for the checkpoint that played at the mark
    layout, tokens_per_layer, mod_levels, style
                   as in the heart log: the brain that played at the mark, when it wrote the bars (else null)
    play_reading   the content of site/play-reading.json (CADANCE_PLAY_READING on a test instance) when the mark
                   arrived; null when it could not be read
    balance        the loudness balance the page had sent to the engine (play_reading's "balance" with the
                   level switch as set)
    session, ring  the page session the selection was made in and the selection's first and last index in that
                   page's buffer of played bars (the page draws its own marks from them)
    bars           the bars in play order, each:
        bar          the server's bar number (counts from 1 in each composition)
        life         the page's composition counter (a new composition or another brain starts a new life)
        checkpoint   the checkpoint that wrote the bar; null for the instrument check (no brain)
        mode         "brain" | "demo"
        pool         the instrument pool file (pool/<name>) the tokens index, as the server resolved it
        tokens       [4] the sound per layer (0 = silent): the engine's queueBar input
        mods         [[filter, space] x 4], each -1..1
        patterns     [4] the layer's pattern for this bar, or null (README pattern format)
        patches      [4] the layer's synth patch, or null; only for patch-book brains
        mix          the hand-over between two brains when the bar was mixed; absent otherwise
        mood         [energy, brightness] the page asked for with this bar
        repeat       true when the engine played the bar before again (the brain's answer came late)
        bpm          the tempo at the bar's start
        seconds      the bar's length as played
        started      the page's clock at the bar's start, ISO 8601 UTC
        state        the brain's slot vector of the bar, when the server still held it (its last 64 bars of
                     the composition that played at the mark); absent otherwise
        position     the bar's place on the brain's composition clock, with state
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent / "feedback"
INDEX = DIR / "index.jsonl"
FORMAT = "cadance-mark/1"
MARKS = ("liked", "disliked")
MAX_BARS = 4000                     # 30 minutes at 200 bpm is 1500 bars
NOTE_CHARS = 500
LAYERS = 4
_ID = re.compile(r"^m\d{8}-\d{6}-[0-9a-f]{4}$")
_LOCK = threading.Lock()
_HEAVY = ("bars", "brains", "play_reading", "balance")


def _number(v, default=None):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def _tokens(v):
    if not isinstance(v, list) or len(v) != LAYERS or not all(isinstance(t, int) and not isinstance(t, bool) for t in v):
        raise ValueError("every bar needs tokens: four whole numbers")
    return v


def _plays(state, tokens):
    """The state sounds as these tokens: the first slots, or with on/off slots the product of both."""
    if not isinstance(state, list) or len(state) < 2 * LAYERS:
        return False
    head = [int(s) for s in state[:LAYERS]]
    gated = [int(state[k]) * int(state[len(state) - LAYERS + k]) for k in range(LAYERS)]
    return tokens in (head, gated)


def _bar(raw, pool_of, held, life_now, active):
    if not isinstance(raw, dict):
        raise ValueError("a bar is an object")
    b = {"bar": int(raw.get("bar") or 0), "life": int(raw.get("life") or 0),
         "checkpoint": raw.get("checkpoint") if isinstance(raw.get("checkpoint"), str) else None,
         "mode": "brain" if raw.get("mode") == "brain" else "demo"}
    b["pool"] = pool_of(b["checkpoint"])
    b["tokens"] = _tokens(raw.get("tokens"))
    for key in ("mods", "patterns", "patches", "mix"):
        if raw.get(key) is not None:
            b[key] = raw[key]
    mood = raw.get("mood")
    if isinstance(mood, list) and all(_number(v) is not None for v in mood):
        b["mood"] = [float(v) for v in mood]
    b["repeat"] = bool(raw.get("repeat"))
    for key in ("bpm", "seconds"):
        if _number(raw.get(key)) is not None:
            b[key] = _number(raw.get(key))
    if isinstance(raw.get("started"), str):
        b["started"] = raw["started"][:40]
    # the brain's own slot vector, when the server still holds this bar of the composition playing now
    h = held.get(b["bar"])
    if (h and b["life"] == life_now and b["checkpoint"] == active and b["mode"] == "brain" and "mix" not in b
            and _plays(h.get("state"), b["tokens"])):
        b["state"] = [int(s) for s in h["state"]]
        if h.get("position") is not None:
            b["position"] = int(h["position"])
    return b


def _summary(rec):
    return {k: v for k, v in rec.items() if k not in _HEAVY}


def add(body, *, card, history, installed, pool_of, style, layout, reading_path):
    """Store one mark. card: the brain playing now (its full card, {} for none); history: its answered bars
    (Creature.history); installed: the installed cards by name; pool_of(checkpoint): the pool file name."""
    mark = body.get("mark")
    if mark not in MARKS:
        raise ValueError("mark is liked or disliked")
    raw = body.get("bars")
    if not isinstance(raw, list) or not raw:
        raise ValueError("a mark needs at least one bar")
    if len(raw) > MAX_BARS:
        raise ValueError(f"at most {MAX_BARS} bars per mark")
    active = card.get("name")
    held = {h["bar"]: h for h in history if isinstance(h, dict) and "state" in h}
    life_now = int(body.get("life_now") or 0)
    bars = [_bar(b, pool_of, held, life_now, active) for b in raw]
    checkpoints = list(dict.fromkeys(b["checkpoint"] for b in bars))
    brains = {}
    for name in checkpoints:
        s = (installed.get(name) or (card if name == active else None)) if name else None   # a brain from Stats: its card
        if s:
            brains[name] = {k: s.get(k) for k in ("candidate", "epoch", "label", "created", "minutes", "pool",
                                                  "pool_sha", "streams", "tokens_per_layer", "mod_levels")}
            brains[name]["layout"] = layout if name == active else None
    try:
        reading = json.loads(Path(reading_path).read_text())
    except (OSError, ValueError):
        reading = None
    note = body.get("note") if isinstance(body.get("note"), str) else ""
    ring = body.get("ring")
    now = time.localtime()
    rec = {"id": f"m{time.strftime('%Y%m%d-%H%M%S', now)}-{secrets.token_hex(2)}", "format": FORMAT, "mark": mark,
           "note": note.strip()[:NOTE_CHARS], "time": time.strftime("%Y-%m-%dT%H:%M:%S", now),
           "page_time": str(body.get("page_time") or "")[:40] or None, "bpm": _number(body.get("bpm")),
           "bars_count": len(bars), "first_bar": bars[0]["bar"], "last_bar": bars[-1]["bar"],
           "seconds": round(sum(b.get("seconds") or 0.0 for b in bars), 3), "checkpoints": checkpoints,
           "states": sum("state" in b for b in bars),
           "layout": layout if active in checkpoints else None,
           "tokens_per_layer": card.get("tokens_per_layer") if active in checkpoints else None,
           "mod_levels": card.get("mod_levels") if active in checkpoints else None,
           "style": style, "session": str(body.get("session") or "")[:40] or None,
           "ring": [int(v) for v in ring][:2] if isinstance(ring, list) and all(isinstance(v, int) for v in ring) else None,
           "brains": brains, "play_reading": reading,
           "balance": body.get("balance") if isinstance(body.get("balance"), dict) else None, "bars": bars}
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / f"{rec['id']}.json"
    with _LOCK:
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(rec, separators=(",", ":")))
        os.replace(tmp, path)
        with open(INDEX, "a") as f:
            f.write(json.dumps({**_summary(rec), "file": path.name}) + "\n")
    return {**_summary(rec), "file": path.name}


def listing():
    """The marks present, newest first (the index lines whose file is there)."""
    out = []
    try:
        lines = INDEX.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if _ID.match(str(row.get("id"))) and (DIR / f"{row['id']}.json").is_file():
            out.append(row)
    return out[::-1]


def read(mark_id):
    if not _ID.match(str(mark_id or "")):
        return None
    try:
        return json.loads((DIR / f"{mark_id}.json").read_text())
    except (OSError, ValueError):
        return None


def remove(mark_id):
    """Delete the mark's file and its index line. False when there is no such mark."""
    if not _ID.match(str(mark_id or "")):
        return False
    with _LOCK:
        path = DIR / f"{mark_id}.json"
        found = path.is_file()
        try:
            lines = INDEX.read_text().splitlines()
        except OSError:
            lines = []
        kept = []
        for line in lines:
            try:
                if json.loads(line).get("id") == mark_id:
                    found = True
                    continue
            except ValueError:
                pass
            kept.append(line)
        if len(kept) != len(lines):
            tmp = INDEX.with_suffix(".jsonl.part")
            tmp.write_text("".join(line + "\n" for line in kept))
            os.replace(tmp, INDEX)
        if path.is_file():
            path.unlink()
    return found


def pool_path(name, pool_dir):
    """An instrument pool file by name, for replaying a mark; None unless it is a pool file in pool_dir."""
    name = str(name or "")
    if Path(name).name != name or not name.endswith(".json") or not name.startswith("pool"):
        return None
    f = Path(pool_dir) / name
    return f if f.is_file() else None
