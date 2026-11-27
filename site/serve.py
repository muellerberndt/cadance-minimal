"""Local website for the minimal-techno creature: http://127.0.0.1:8792/

The server holds one living creature (a Cadence brain loaded from a checkpoint) and answers one `live`
moment per bar: the page sends the wanted mood and the listener's reward, the creature answers the next
bar's tokens. Before any gated checkpoint exists it plays the instrument check (hand-sequenced, no brain).

    python site/serve.py
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import threading
import time
import warnings
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
sys.path.insert(0, str(ROOT / "brain"))
# Prefer the pinned post-audit wheel when installed. Keep the legacy player
# usable in existing workspaces and honor its explicit library override.
from importlib.metadata import PackageNotFoundError, distribution
try:
    _dist = distribution("cadence-net")
    _released = Path(_dist.locate_file("")) if tuple(int(x) for x in _dist.version.split(".")[:3]) >= (0, 103, 0) else None
except (PackageNotFoundError, ValueError):
    _released = None
_recall_runtime = ROOT / ".cadence103-recall"
_official_runtime = ROOT / ".cadence1031"
# Cadence 0.104.0 (pip --target .cadence104 cadence-net==0.104.0, no runtime patch stamp): the generation-2 brains
# (slate_*_s2_*) name 0.104 features in their checkpoints, which an older library refuses
_runtime_104 = ROOT / ".cadence104"
# 0.104.0 plus quality/library/sensory-reflex-0.104.0.patch (a one-way sensory projection beside a learned code; the
# sparse-reflex brains slate_*_s2_sref need it, every other 0.104.0 brain runs the same code): preferred when present
_runtime_104_reflex = ROOT / ".cadence104-reflex"
# 0.104.0 plus quality/library/reflex-slot-surprise-0.104.0.patch (the reflex change and the per-slot surprise option;
# brains taught with learning_surprise_slots need it, and it covers the reflex-only patch: cadence/.patch-covers)
_runtime_104_slots = ROOT / ".cadence104-slots"
# 0.104.0 plus quality/library/reflex-slot-surprise-sampled-0.104.0.patch (the two above and the sampled judge of
# surprise lessons; brains taught with learning_surprise_sampled need it; it covers both earlier patch stamps)
_runtime_104_sampled = ROOT / ".cadence104-sampled"
CADENCE_SRC = Path(os.environ.get("CADANCE_CADENCE") or
                   (_runtime_104_sampled if (_runtime_104_sampled / "cadence/.patch-sha256").is_file() else None) or
                   (_runtime_104_slots if (_runtime_104_slots / "cadence/.patch-sha256").is_file() else None) or
                   (_runtime_104_reflex if (_runtime_104_reflex / "cadence/.patch-sha256").is_file() else None) or
                   (_runtime_104 if (_runtime_104 / "cadence/__init__.py").is_file() else None) or
                   (_official_runtime if (_official_runtime / "cadence/__init__.py").is_file() else None) or
                   (_recall_runtime if (_recall_runtime / "cadence/.patch-sha256").is_file() else None) or _released or
                   (ROOT / ".cadence103" if (ROOT / ".cadence103/cadence").is_dir() else None) or
                   (ROOT / ".cadence102" if (ROOT / ".cadence102/cadence").is_dir() else ROOT.parent / "cadence/src"))
sys.path.insert(0, str(CADENCE_SRC))
import cadence  # noqa: E402
import common  # noqa: E402
import groove  # noqa: E402
import timbre  # noqa: E402
from common import L, observe  # noqa: E402
# how a brain is read (common.sample_slots), unless its card names its own (training runs with --entry-scale/--line-scale).
# 11-24 listening check against the tracks: a silent layer's comeback at 2x the temperature (drums alone in 4-7% of the bars
# instead of 21-34%; tracks 1%), drum rows and synth lines at 1.2x (beat kinds 11 per 16 bars, lead 5 notes per bar, 9-10 lines
# per 16 bars; tracks 10.6, 5.9, 9.2; at 1x the commonest = the plainest line won: 1-4 notes, 2-5 lines)
READING = {"entry_scale": 2.0, "line_scale": 1.2}


# How the slate's brains are played, read from site/play-reading.json when it exists: {"temperature": ..,
# "line_scale": .., "entry_scale": ..}. Their trainer scans one temperature only, which plays them frozen; the
# values in the file come from the free-play probe against the corpus (aws/freeplay_probe.py). It applies to a
# brain raised hearing every bar first (card field hear_first) and to the generation-2 brains (runs named
# slate_*_s2_*: taught before the act, hear_first off), and to no other brain.

def _takes_play_reading(card):
    card = card or {}
    if card.get("hear_first"):
        return True
    return any("_s2_" in str(card.get(k) or "") for k in ("name", "candidate"))

def play_reading(card):
    # a test instance may name its own copy (CADANCE_PLAY_READING), so a setting can be measured without changing
    # the file the owner's player reads
    path = Path(os.environ.get("CADANCE_PLAY_READING") or Path(__file__).resolve().parent / "play-reading.json")
    if not _takes_play_reading(card) or not path.is_file():
        return {}
    try:
        values = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    # per-brain settings: "brains": {"<name prefix>": {"temperature": 0.16, ...}}; the longest prefix of the card's
    # name (or of its run's candidate name) wins, and its keys replace the shared ones for that brain only
    per = values.get("brains")
    if isinstance(per, dict):
        names = [str((card or {}).get("name") or ""), str((card or {}).get("candidate") or "")]
        hits = [k for k in per if isinstance(per[k], dict) and any(n.startswith(k) for n in names if n)]
        if hits:
            values = {**values, **per[max(hits, key=len)]}
    out = {k: float(v) for k, v in values.items()
           if k in ("temperature", "line_scale", "entry_scale", "control_scale", "loudness_scale", "synth_line_scale", "kick_scale", "kick_cap", "acid_start", "play_tolerance") and v}
    # play_tolerance: the residual each bar's settle stops at when the brain plays (Brain.retune(learning_tolerance=..)
    # at load, Creature._read); absent, the brain keeps its own (the trainer's 0.003)
    if not 0 < out.get("play_tolerance", 1) <= 1:
        del out["play_tolerance"]
    mode = values.get("cap_mode")            # "calm" (default): a capped bar takes the calm copy; "thin": thinned in place
    if mode in ("calm", "thin"):
        out["cap_mode"] = mode
    caps = values.get("row_caps")            # the most hits per bar of each drum row (Creature._cap_rows)
    if isinstance(caps, list) and len(caps) == len(common.DRUM_ROWS) and all(
            c is None or (isinstance(c, (int, float)) and not isinstance(c, bool) and c >= 0) for c in caps):
        if any(caps):
            out["row_caps"] = [float(c) if c else None for c in caps]
    caps = values.get("line_caps")           # the most notes per bar of each synth layer (Creature._cap_lines)
    if isinstance(caps, list) and len(caps) == len(common.SYNTH_LAYERS) and all(
            c is None or (isinstance(c, (int, float)) and not isinstance(c, bool) and c >= 0) for c in caps):
        if any(caps):
            out["line_caps"] = [float(c) if c else None for c in caps]
    dwell = values.get("voice_dwell")
    if isinstance(dwell, list) and len(dwell) == L and all(isinstance(n, int) and n >= 2 for n in dwell):
        out["voice_dwell"] = dwell       # bars a voice keeps a major change before the next one (groove.VoiceDwell)
    go = values.get("dwell_lets_go")
    if isinstance(go, list) and len(go) == L and all(isinstance(x, bool) for x in go):
        out["dwell_lets_go"] = go        # per voice: the dwell lets a sounding voice leave (it still holds its sound)
    # route_scale: a factor on the strengths of the brain's direct route (its read ports: recall and next per head),
    # applied when a brain is loaded; a number for every port, or a list of one factor per head. Under 1 the brain
    # copies the bars it heard less and its own answer decides more. Brains without a route are left alone.
    rs = values.get("route_scale")
    if isinstance(rs, (int, float)) and not isinstance(rs, bool) and 0 <= rs <= 1.5:
        out["route_scale"] = float(rs)
    elif isinstance(rs, list) and rs and all(isinstance(x, (int, float)) and 0 <= x <= 1.5 for x in rs):
        out["route_scale"] = [float(x) for x in rs]
    return out


def apply_route_scale(brain, card):
    """Multiply the loaded brain's direct-route strengths by the play reading's route_scale (see play_reading)."""
    rs = play_reading(card).get("route_scale")
    recall = getattr(brain, "recall", None)
    if rs is None or recall is None or getattr(recall, "gains", None) is None:
        return None
    if not getattr(recall, "gain", 0) or not np.asarray(recall.gains).size:     # no recall-to-motor route: left alone
        return None
    gains = np.asarray(recall.gains, dtype=float).copy()
    if isinstance(rs, list):
        for head, factor in enumerate(rs[: len(gains) // 2]):
            gains[2 * head: 2 * head + 2] *= factor
    else:
        gains *= rs
    recall.gains = gains
    return gains


def play_temperatures(card, base):
    """One temperature per slot for a slate brain: `base` for sounds and mix, and the play reading's scales on
    the sound controls. control_scale applies to every sound control; loudness_scale to the slots that set how
    loud a voice is (register, gain step, level, the four velocities), so loudness moves only when the brain is
    sure. One gain step is a factor of two."""
    over = play_reading(card)
    if not np.isscalar(base) or not common.PATCH_BOOK or not ("control_scale" in over or "loudness_scale" in over):
        return base
    width = timbre.width(common.PATCH_BOOK)
    start = common.SLOTS - 3 * width - (L if common.ONOFF else 0)
    temps = np.full(common.SLOTS, float(base))
    control = over.get("control_scale", 1.0)
    temps[start:start + 3 * width] *= control
    loud = [1, 2, 3 + timbre.NAMES.index("level"), *range(len(timbre.PATCH_SIZES), len(timbre.PATCH_SIZES) + 4)]
    for layer in range(3):
        temps[start + layer * width + np.asarray(loud)] = float(base) * over.get("loudness_scale", control)
    return temps


def reading(card):
    card = card or {}
    over = play_reading(card)
    return tuple(over[k] if k in over else card.get(k) if card.get(k) is not None else READING[k]
                 for k in ("entry_scale", "line_scale"))
sys.path.insert(0, str(SITE))
import anatomy  # noqa: E402
import check  # noqa: E402
import layouts  # noqa: E402
import brainview  # noqa: E402
import training_stats  # noqa: E402
import feedback_marks  # noqa: E402   liked/disliked selections from the waveform; the file format is in its docstring

_CURVES = {}                       # the last curves read and the signature of the files they came from
_CURVES_LOCK = threading.Lock()

CHECKPOINTS = SITE / "checkpoints"
_runtime_stamp = Path(cadence.__file__).with_name(".patch-sha256")
RUNTIME_PATCH = _runtime_stamp.read_text().strip() if _runtime_stamp.is_file() else None
SUPPORTED_CHECKPOINT_PATCH = RUNTIME_PATCH or (
    common.AUDITED_RECALL_PATCH if cadence.__version__ == common.AUDITED_RECALL_RELEASE else None)
# every patch stamp this runtime can read: its own and the ones it contains (one sha per line in .patch-covers)
_runtime_covers = _runtime_stamp.with_name(".patch-covers")
# patch stamps of development libraries whose changes a release ships: a brain trained on 0.104.0 plus the sensory
# projection beside a learned code, or plus that and surprise by slot, loads on 0.104.1, which has both
RELEASE_COVERS = {(0, 104, 1): ("6b3eb1d9703248c52d3555cfb7e302d72d360d02ba5b07a6df293ce07d7f0c96",
                                "e8cf41673a43b79b75188a052c5f2a610f8a8de0059f6bcd300ec975d10abaf8")}
try:
    _library_version = tuple(int(x) for x in cadence.__version__.split(".")[:3])
except ValueError:
    _library_version = ()
SUPPORTED_CHECKPOINT_PATCHES = frozenset(
    ([SUPPORTED_CHECKPOINT_PATCH] if SUPPORTED_CHECKPOINT_PATCH else [])
    + (_runtime_covers.read_text().split() if _runtime_covers.is_file() else [])
    + [stamp for release, stamps in RELEASE_COVERS.items() if _library_version >= release for stamp in stamps])
PINNED = SITE / "pinned"            # kept by hand (early favourites, remarkable brains); the watcher never touches it
FEEDBACK = SITE / "feedback" / "feedback.jsonl"     # every heart moment with the bars it marks, kept for training rounds
WINDOW = 16
LIVES = SITE / "lives"
PLAY_MODE = "imitate"   # imitate: the learned policy plays, nothing learns (listening tests); live: the creature (brain.live + reward)
NEED = 0.05          # reward per bar the creature wants; a quiet room leaves it restless, a heart lets it settle
SAVE_EVERY = 32
PORT = int(os.environ.get("CADANCE_PORT", "8792"))   # a second instance for tests: CADANCE_PORT=8793
STATIC = {
    "/": SITE / "index.html", "/index.html": SITE / "index.html", "/main.js": SITE / "main.js",
    "/worklet.js": SITE / "worklet.js", "/rack.js": SITE / "rack.js", "/engine.mjs": ROOT / "engine" / "engine.mjs",
    "/brainviz.js": SITE / "brainviz.js", "/ui.mjs": ROOT / "engine" / "ui.mjs", "/kit.js": SITE / "kit.js", "/brains.js": SITE / "brains.js", "/stats.js": SITE / "stats.js",
    "/mixer": SITE / "mixer.html", "/mixer.html": SITE / "mixer.html", "/mixer.js": SITE / "mixer.js", "/deckworklet.js": SITE / "deckworklet.js",
    "/autodj.js": SITE / "autodj.js", "/djfx.js": SITE / "djfx.js", "/wave.js": SITE / "wave.js",
    "/play-reading.json": Path(os.environ.get("CADANCE_PLAY_READING") or SITE / "play-reading.json"),
    "/history.json": SITE / "history.json",      # the training history (quality/history/history.py, every ten minutes)
}
SAMPLES = ROOT / "samples"
SAMPLE_TITLES = {
    "test_atmos": "instrument test · atmos: long pads with slow FILTER/SPACE breaths, delayed blips",
    "test_bass": "instrument test · bass: acid siblings + filter sweeps, 16th runs, sub notes",
    "test_patterns": "instrument test · patterns: 606 + repeating acid lead, delay throws, chirps",
    "instrument_check_v0": "instrument check v0 · all four layers, hand-sequenced",
    "kit_606": "kit · 606 model", "kit_808": "kit · 808 model", "kit_909": "kit · 909 model",
}
try:        # titles of demo scores made on this computer (samples/titles.json, kept out of the repository)
    SAMPLE_TITLES.update(json.loads((SAMPLES / "titles.json").read_text()))
except (OSError, ValueError):
    pass


def R3_TITLES():
    p = ROOT / "corpus" / "streams_r3.json"
    try:
        return {t["id"]: t.get("title", t["id"]) for t in json.loads(p.read_text())["tracks"]}
    except Exception:
        return {}


TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"}


# Checkpoints fetched for a listen from Stats: loadable by name, never listed in the menu.
AUDITION = SITE / "audition"
# where training runs keep their checkpoints and the repertoire ("s3://bucket/prefix"); empty: local files only
BUCKET = common.local_setting("CADANCE_BUCKET")
SID_POOL = "pool_underground_sid.json"      # the present sounds fitted on the SID chip, for the chip's instrument check
AUTO_LIST = SITE / ".auto-list"             # with this file present, remarkable new checkpoints join the menu by themselves


def _dir(where):
    return PINNED if where == "pinned" else AUDITION if where == "audition" else CHECKPOINTS


_RESOLVED = {}          # candidate -> (time looked up, checkpoint name): the bucket is listed again after RESOLVE_TTL
LISTEN_STATES = SITE / "listen-states.json"   # {"<run>": "<checkpoint>"}: the state a run plays as (listen_state)
_LISTEN = [None, {}]    # (the file's mtime, its mapping)


def listen_states():
    """site/listen-states.json: {"<run>": "<checkpoint>"}, the state each named run PLAYS as (a hand-kept state, such
    as a run's first-hour evaluation state when its later states play worse). Read again when the file changes; no
    file, an empty mapping."""
    try:
        stamp = LISTEN_STATES.stat().st_mtime
    except OSError:
        return {}
    if _LISTEN[0] != stamp:
        try:
            data = json.loads(LISTEN_STATES.read_text())
        except (OSError, ValueError):
            data = {}
        _LISTEN[0] = stamp
        _LISTEN[1] = {str(k): v for k, v in data.items() if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", v)} \
            if isinstance(data, dict) else {}
    return _LISTEN[1]


def listen_state(run):
    """The checkpoint `run` plays as (listen_states), None for a run without an entry. For play only: the player,
    the decks and the Auto DJ factory resolve a run through it (audition_checkpoint, TrackFactory.kept); Stats'
    curves and the trainer's files do not read it."""
    return listen_states().get(str(run)) if run else None


RESOLVE_TTL = 900.0


def short_brain(model):
    """A model id or checkpoint as the pages name it: candidate:slate_split_s1_lag4tr -> split_s1_lag4tr."""
    text = str(model or "").split(":", 1)[-1]
    return text[len("slate_"):] if text.startswith("slate_") else text


def audition_checkpoint(candidate, progress=None):
    """Make a candidate's newest checkpoint loadable and return its name: a checkpoint the player has, one a
    local run keeps beside its curve, or one fetched from the run's folder in the bucket. The answer is kept for
    RESOLVE_TTL seconds while its files are on disk: listing the bucket for every Auto DJ track took minutes.
    `progress` (a LoadProgress) shows the listing and the fetch as the first stages of the load that follows."""
    state = listen_state(candidate)
    if state is not None:                   # the run's listening state (site/listen-states.json), kept on this computer
        if not any(card["name"] == state for card, _ in _available_checkpoints()):
            raise ValueError(f"listen-states.json names {state} for {candidate}, which is not on this computer")
        return state
    hit = _RESOLVED.get(candidate)
    if hit and time.time() - hit[0] < RESOLVE_TTL and any(
            (folder / f"{hit[1]}.npz").is_file() for folder in (AUDITION, CHECKPOINTS, PINNED)):
        return hit[1]
    name = _audition_checkpoint(candidate, progress)
    _RESOLVED[candidate] = (time.time(), name)
    return name


def _fetch(source, part, size, progress):
    """aws s3 cp of one file to `part`, its bytes reported against `size` (the listing's) while it runs. The CLI
    writes to a file of its own beside `part` (<part>.<8 hex>) and renames it when done."""
    import shutil
    import subprocess
    aws = shutil.which("aws") or "/usr/local/bin/aws"
    started = time.time()
    run = subprocess.Popen([aws, "s3", "cp", source, str(part), "--only-show-errors"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    while True:
        try:
            run.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            pass
        if time.time() - started > 900:
            run.kill()
            raise ValueError(f"could not fetch {Path(source).name}: no copy after 900 s")
        if progress is not None and size:
            got = 0
            for f in part.parent.glob(part.name + "*"):
                try:
                    if f.stat().st_mtime >= started - 2:
                        got = max(got, f.stat().st_size)
                except OSError:
                    pass
            progress.at("fetching", got, size, f"{got / 1e6:.0f} of {size / 1e6:.0f} MB")
    if run.returncode:
        raise ValueError(f"could not fetch {Path(source).name}: {(run.stderr.read() if run.stderr else '').strip()[-200:]}")


def _audition_checkpoint(candidate, progress=None):
    import re
    import shutil
    import subprocess

    if not isinstance(candidate, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", candidate):
        raise ValueError("unknown candidate")
    found = []                                   # (time, checkpoint, where it is: None = already loadable)
    sizes = {}                                   # checkpoint -> bytes of its npz in the bucket (the listing's)
    if progress is not None:
        progress.begin(candidate, carry=True)
        progress.expect("finding")
        for stage in ("building", "priming"):              # a split brain, until the checkpoint is known
            progress.expect(stage, nbytes=1.6e9)
        progress.at("finding", detail=short_brain(candidate))
    for card, _ in _available_checkpoints():
        if card.get("candidate") == candidate:
            try:
                made = time.mktime(time.strptime(card.get("created") or "", "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            except ValueError:
                made = 0.0
            found.append((made, card["name"], None))
    # a run's checkpoints, and its rolling state <run>_latest (written at every reading; runs on the s1c
    # trainer keep _e checkpoints only at their marks)
    for weights in [*(ROOT / "quality/composer" / candidate).glob(f"{candidate}_e*.npz"),
                    *(ROOT / "quality/composer" / candidate).glob(f"{candidate}_latest.npz")]:
        if weights.with_suffix(".json").is_file():
            found.append((weights.stat().st_mtime, weights.stem, weights))
    aws = shutil.which("aws") or "/usr/local/bin/aws"
    for prefix in ((f"{BUCKET}/composer/{candidate}/checkpoints/", f"{BUCKET}/brains_r4/{candidate}_e") if BUCKET else ()):
        try:
            listing = subprocess.run([aws, "s3", "ls", prefix], capture_output=True, text=True, timeout=40).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in listing.splitlines():
            fields = line.split()
            if len(fields) == 4 and (fields[3].startswith(f"{candidate}_e") or fields[3] == f"{candidate}_latest.npz") and fields[3].endswith(".npz"):
                stamp = time.mktime(time.strptime(f"{fields[0]} {fields[1]}", "%Y-%m-%d %H:%M:%S"))    # listed in local time
                found.append((stamp, fields[3][:-4], prefix.rsplit("/", 1)[0] + "/"))
                sizes[fields[3][:-4]] = int(fields[2]) if fields[2].isdigit() else 0
    if not found:
        raise ValueError(f"{candidate} has no checkpoint yet (a run's rolling state comes with its first reading, "
                         f"its checkpoints at its marks)")
    stamp, name, source = max(found, key=lambda item: item[0])
    if source is None or any(other == name and where is None for _, other, where in found):
        return name
    AUDITION.mkdir(exist_ok=True)
    # a rolling state keeps its name in the bucket while its contents change: here it is kept under a name with
    # its time, so a newer state is fetched and loaded as a brain of its own
    local = f"{candidate}_l{time.strftime('%m%d%H%M', time.localtime(stamp))}" if name.endswith("_latest") else name
    if not all((AUDITION / f"{local}{suffix}").is_file() for suffix in (".json", ".npz")):
        if progress is not None:
            for stage in ("building", "priming"):          # planned again after the copy, here estimated from its size
                progress.drop(stage)
            packed = (source.with_suffix(".npz").stat().st_size if isinstance(source, Path) else sizes.get(name)) or 8e8
            if isinstance(source, Path):
                progress.expect("copying", seconds=2.0)
            else:
                progress.expect("fetching", nbytes=packed)
            for stage in ("unpacking", "building", "priming"):         # a checkpoint unpacks to about 2.5 times its size
                progress.expect(stage, nbytes=2.5 * packed)
        for suffix in (".json", ".npz"):
            target = AUDITION / f"{local}{suffix}"
            part = AUDITION / f"{name}{suffix}.part"
            if isinstance(source, Path):
                if progress is not None:
                    progress.at("copying", detail=f"{name}{suffix}")
                shutil.copyfile(source.with_suffix(suffix), part)
            elif suffix == ".npz":
                _fetch(f"{source}{name}{suffix}", part, sizes.get(name), progress)
            else:
                done = subprocess.run([aws, "s3", "cp", f"{source}{name}{suffix}", str(part), "--only-show-errors"],
                                      capture_output=True, text=True, timeout=900)
                if done.returncode:
                    raise ValueError(f"could not fetch {name}: {done.stderr.strip()[-200:]}")
            if suffix == ".json" and local != name:
                card = json.loads(part.read_bytes()); card["name"] = local; card["rolling"] = name
                part.write_text(json.dumps(card))
            part.replace(target)
    for old in sorted(AUDITION.glob("*.npz"), key=lambda file: file.stat().st_mtime, reverse=True)[4:]:
        if old.stem != local:                    # the four newest stay for a quick return
            old.unlink()
            old.with_suffix(".json").unlink(missing_ok=True)
    return local


BRAINS = Path(os.environ.get("CADANCE_BRAINS") or SITE / "brains.json")   # the curated list (order, stars, labels, notes, deletions); a test instance names its own copy
PLAYER_SELECTION = BRAINS.with_name(f".{BRAINS.stem}-selection")
BRAINS_LOCK = threading.RLock()
AUTO_CAP = 40                           # at most this many unrated auto-added brains; the oldest is superseded
_AUTO_EVERY = 60.0
_last_auto = [0.0]


def _summary(c, where):
    """What the player needs of one installed checkpoint card."""
    s = {"dir": where, "label": c.get("label"), "pinned": where == "pinned",
         "featured": bool(c.get("featured")), "playback_variant": c.get("playback_variant"),
         "name": c["name"], "candidate": c.get("candidate"), "epoch": c.get("epoch"), "minutes": c.get("minutes"),
         "created": c.get("created"), "best": c.get("best"), "pool": c.get("pool", "pool.json"),
         "gate": bool((c.get("gate") or {}).get("pass")), "score": (c.get("held_out") or {}).get("score"),
         "checks": (c.get("gate") or {}).get("checks", []), "start": (c.get("free_run") or [None])[0],
         "tokens_per_layer": c.get("tokens_per_layer") or (c.get("settings", {}).get("slots") or [33])[0],
         "mod_levels": c.get("mod_levels", 7), "styles": c.get("styles") or [], "beat_rows": c.get("beat_rows"), "streams": c.get("streams"), "streams_sha": c.get("streams_sha"), "clock": c.get("clock", False), "synth_rows": c.get("synth_rows"), "steps": c.get("steps", False), "content": c.get("content", False), "pool_sha": c.get("pool_sha"),
         "imitation": c.get("imitation", False), "temperature": c.get("temperature"), "replay": ((c.get("held_out") or {}).get("replay") or {}).get("mean_all"),
         "hold": ((c.get("held_out") or {}).get("replay") or {}).get("hold_all"),
         "composition_score": (c.get("composition") or {}).get("selection_score"),
         "cadence_patch_sha256": c.get("cadence_patch_sha256"),
         }
    s["quality"] = quality(s)
    return s


def _available_checkpoints():
    """Return installed cards in index order, with pinned copies last."""
    index = CHECKPOINTS / "index.json"
    found = []
    try:
        found = [(c, "checkpoints") for c in (json.loads(index.read_text()) if index.exists() else [])]
    except ValueError:
        pass
    for p in sorted(PINNED.glob("*.json")) if PINNED.exists() else []:
        try:
            c = json.loads(p.read_text())
        except ValueError:
            continue
        c.setdefault("name", p.stem)
        found.append((c, "pinned"))
    for c, where in found:
        if "name" not in c or not (_dir(where) / f"{c['name']}.npz").exists():
            continue
        if c.get("cadence_patch_sha256") and c["cadence_patch_sha256"] not in SUPPORTED_CHECKPOINT_PATCHES:
            continue
        yield c, where


def _installed():
    """Summarize installed player cards. A pinned copy takes precedence."""
    return {card["name"]: _summary(card, where) for card, where in _available_checkpoints()}


def _round(s) -> str:
    import re
    m = re.match(r"r(\d+)", s["name"])
    if m:
        return f"r{m.group(1)}"
    return "r1" if s.get("pool") == "pool_grid1.json" else "r2"


def _beat_share(s):
    v = {x.get("check"): x.get("value") for x in s.get("checks") or []}
    return float(v.get("share of beat bars with four kicks") or 0), float(v.get("share with snare/clap on 2 and 4") or 0)


def _writes_beat(s) -> bool:
    k, sn = _beat_share(s)
    return k >= 0.6 and sn >= 0.4


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _entry(s, reason, auto):
    return {"name": s["name"], "dir": "pinned", "label": s.get("label") or "", "note": "", "stars": 0, "added": _now(),
            "deleted": False, "reason": reason, "auto": auto, "round": _round(s), "quality": s["quality"],
            "beat": _writes_beat(s), "candidate": s.get("candidate"), "created": s.get("created")}


def _keep(name):
    """Keep a curated checkpoint where the watcher never prunes: site/pinned (a hard link; S3 holds the original)."""
    PINNED.mkdir(exist_ok=True)
    src_npz, dst_npz = CHECKPOINTS / f"{name}.npz", PINNED / f"{name}.npz"
    if src_npz.exists() and not dst_npz.exists():
        try:
            os.link(src_npz, dst_npz)
        except OSError:
            import shutil
            shutil.copy2(src_npz, dst_npz)
    src_json, dst_json = CHECKPOINTS / f"{name}.json", PINNED / f"{name}.json"
    if not dst_json.exists():
        card = None
        if src_json.exists():
            try:
                card = json.loads(src_json.read_text())
            except ValueError:
                card = None
        if card is None:                     # fall back to the watcher's index entry
            try:
                card = next((c for c in json.loads((CHECKPOINTS / "index.json").read_text()) if c.get("name") == name), None)
            except Exception:
                card = None
        if card is not None:
            card["free_run"] = (card.get("free_run") or [None])[:8 if card.get("patch_book") else 1]
            dst_json.write_text(json.dumps(card))


def _remove(name):
    for d in (PINNED, CHECKPOINTS):
        for ext in ("npz", "json"):
            f = d / f"{name}.{ext}"
            if f.exists():
                f.unlink()


def _save(entries):
    tmp = BRAINS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=1))
    os.replace(tmp, BRAINS)


def _best_by_test():
    """Per brain the checkpoint with the lowest test error (quality/curves.jsonl): kept in the list beside the newest,
    as some brains get worse with more training (11-23)."""
    best = {}
    try:
        for line in (ROOT / "quality" / "curves.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("failed"):
                continue
            c = r.get("candidate")
            if c not in best or r["test_score"] > best[c][1]:
                best[c] = (r["name"], r["test_score"])
    except (FileNotFoundError, ValueError):
        pass
    return {n: s for n, s in best.values()}


def _auto_add(entries, installed, preserve=None):
    """Add only remarkable checkpoints: the first brain of a new round, the first of a round that writes a beat
    (four kicks >= 0.6, snare/clap on 2 and 4 >= 0.4), or a new best of its round (quality >= best ever listed
    + 0.05). Names ever listed (deleted ones too) never come back. A newer auto entry supersedes older unrated
    auto entries of the same candidate; more than AUTO_CAP unrated auto entries drop the oldest."""
    known = {e["name"] for e in entries}
    changed = False
    best_test = _best_by_test()
    composer_best = {}
    for item in installed.values():
        score = item.get("composition_score")
        if score is not None:
            candidate = item.get("candidate")
            composer_best[candidate] = max(composer_best.get(candidate, (-1, "")), (score, item["name"]))
    best_compositions = {name for _, name in composer_best.values()}
    fresh = sorted((s for n, s in installed.items() if n not in known
                    and (s["dir"] == "checkpoints" or s.get("featured") or n == preserve)),
                   key=lambda s: (s.get("created") or "", s.get("epoch") or 0))
    for s in fresh:
        if s.get("featured") or s["name"] == preserve:
            reason = "Featured listening checkpoint" if s.get("featured") else "Current player selection"
            entries.append({**_entry(s, reason, auto=False), "dir": s["dir"]})
            changed = True
            continue
        if s.get("composition_score") is not None:
            reason = ("Best held-out composition" if s["name"] in best_compositions else "Training checkpoint")
            entries.append({**_entry(s, reason, auto=True), "dir": "checkpoints"})
            changed = True
            continue
        rnd = _round(s)
        same = [e for e in entries if e.get("round") == rnd]
        reason = None
        if rnd >= "r4":                         # owner 11-23: the newest checkpoint of every brain, as soon as it is here (its
            # QA verdict follows within minutes); it replaces the older entry of the same brain unless that one is rated,
            # placed, renamed or noted by the owner
            reason = f"lowest test error of {s.get('candidate')}" if s["name"] in best_test else f"newest of {s.get('candidate')}"
            if s["name"] in best_compositions:
                reason = f"best held-out composition of {s.get('candidate')}"
            for e in entries:
                if (e.get("auto") and not e.get("deleted") and e.get("candidate") == s.get("candidate") and not e.get("stars")
                        and e["name"] not in best_test and e["name"] not in best_compositions                  # the brain's best by test error stays
                        and e.get("order") is None and not e.get("note") and e.get("reason", "").startswith(("newest", "added for comparison", "first brain", "new best", "best held-out composition"))
                        and e["name"] not in (s["name"], preserve) and (e.get("created") or "") < (s.get("created") or "")):
                    e["deleted"] = True; e["reason"] += f" · superseded by {s['name']}"; _remove(e["name"])
            _keep(s["name"])
            entries.append(_entry(s, reason, auto=True)); changed = True
            continue
        elif not same:
            reason = f"first brain of round {rnd[1:]}"
        elif _writes_beat(s) and not any(e.get("beat") for e in same):
            reason = f"first brain of round {rnd[1:]} that writes a beat"
        else:
            best = max(e.get("quality", -9.0) for e in same)
            if s["quality"] >= best + 0.05:
                reason = f"new best of round {rnd[1:]} (quality {s['quality']:.2f}, was {best:.2f})"
        if not reason:
            continue
        _keep(s["name"])
        for e in entries:                    # an older unrated auto entry of the same candidate is superseded
            if (e.get("auto") and not e.get("deleted") and not e.get("stars") and e.get("candidate") == s.get("candidate")
                    and e["name"] != preserve):
                e["deleted"] = True; e["reason"] += f" · superseded by {s['name']}"; _remove(e["name"])
        entries.append(_entry(s, reason, auto=True)); changed = True
    auto = [e for e in entries if e.get("auto") and not e.get("deleted") and not e.get("stars")
            and e["name"] != preserve and not installed.get(e["name"], {}).get("featured")]
    while len(auto) > AUTO_CAP:
        old = min(auto, key=lambda e: e.get("added") or "")
        old["deleted"] = True; old["reason"] += " · superseded"; _remove(old["name"]); auto.remove(old); changed = True
    return changed


def _brains(installed):
    """The curated list, seeded from the kept (pinned) brains when missing or unreadable, with remarkable
    new checkpoints added at most every minute."""
    entries = None
    if BRAINS.exists():
        try:
            entries = json.loads(BRAINS.read_text())
            if not (isinstance(entries, list) and all(isinstance(e, dict) and "name" in e for e in entries)):
                entries = None
        except ValueError:
            entries = None
    changed = False
    if entries is None:
        entries = []
        for name, s in sorted(installed.items(), key=lambda kv: kv[1].get("created") or ""):
            if s["dir"] != "pinned":
                continue
            lab = s.get("label") or ""
            reason = "pinned by owner" if "promising" in lab else (f"pinned: {lab}" if lab else "pinned")
            entries.append(_entry(s, reason, auto=False))
        changed = True
        _last_auto[0] = 0.0
    if time.time() - _last_auto[0] >= _AUTO_EVERY:
        _last_auto[0] = time.time()
        try:
            selected = PLAYER_SELECTION.read_text().strip()
        except OSError:
            selected = None
        for entry in entries:
            name = entry["name"]
            item = installed.get(name, {})
            reason = entry.get("reason", "")
            recovered = name == selected or (item.get("composition_score") is not None and " · superseded by " in reason)
            if (item and recovered and entry.get("deleted") and entry.get("auto")
                    and " · superseded" in reason and "deleted by owner" not in reason):
                entry.update(deleted=False, reason="Training checkpoint", dir=item["dir"])
                changed = True
        if AUTO_LIST.is_file():          # the menu holds the brains chosen for it; nothing joins by itself otherwise
            changed = _auto_add(entries, installed, preserve=selected) or changed
    if changed:
        _save(entries)
    return entries


def _hearts():
    out = {}
    if FEEDBACK.exists():
        for line in FEEDBACK.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("rating", 0) > 0:
                out[row.get("checkpoint")] = out.get(row.get("checkpoint"), 0) + 1
    return out


def cards():
    """The menu: the curated brains whose checkpoint is present, rated ones first (stars), then by quality."""
    with BRAINS_LOCK:
        installed = _installed()
        entries = _brains(installed)
        hearts = _hearts()
        verdict = QA_RESULTS()
        out = []
        for e in entries:
            s = installed.get(e["name"])
            if e.get("deleted") or not s:
                continue
            c = dict(s)
            c.update(label=e.get("label") or s.get("label"), note=e.get("note", ""), stars=int(e.get("stars") or 0),
                     reason=e.get("reason"), added=e.get("added"), hearts=hearts.get(e["name"], 0), round=e.get("round"))
            v = verdict.get(e["name"])
            c.update(qa=v[0] if v else None, qa_failed=v[1] if v else [], order=e.get("order"))
            out.append(c)
    # the owner's stars first, then the QA verdict (PASS > HOLDS A BEAT > unchecked/fail), then the quality number
    rnd = lambda c: int((c.get("round") or "r0")[1:] or 0)
    # ... then the newest round (its labels and fixes), then the quality number
    out.sort(key=lambda c: (c["stars"], QA_RANK.get(c["qa"] or "", 0), rnd(c), -len(c.get("qa_failed") or []) if c.get("qa") else -99, c["quality"]), reverse=True)
    # the owner (11-23): the newest brains on top by default, then the ones placed by hand in the editor, in that order
    new = sorted((c for c in out if c.get("order") is None), key=lambda c: c.get("created") or c.get("added") or "", reverse=True)
    mine = sorted((c for c in out if c.get("order") is not None), key=lambda c: c["order"])
    return new + mine


def update_brain(name, **fields):
    with BRAINS_LOCK:
        entries = _brains(_installed())
        e = next((x for x in entries if x["name"] == name and not x.get("deleted")), None)
        if e is None:
            raise ValueError(f"no brain {name} in the list")
        if "stars" in fields and fields["stars"] is not None:
            e["stars"] = max(0, min(5, int(fields["stars"])))
        if fields.get("label") is not None:
            e["label"] = str(fields["label"]).strip()[:60]
        if fields.get("note") is not None:
            e["note"] = str(fields["note"]).strip()[:140]
        _save(entries)
        return e


def order_brains(names):
    """The listener's order of the list: names first to last; brains not named keep no order and follow, ranked as
    before (stars, QA, round, quality)."""
    with BRAINS_LOCK:
        entries = _brains(_installed())
        pos = {n: i for i, n in enumerate(names)}
        for e in entries:
            if e["name"] in pos:
                e["order"] = pos[e["name"]]
            else:
                e.pop("order", None)
        _save(entries)
        return {"ordered": len(pos)}


def delete_brain(name):
    with BRAINS_LOCK:
        entries = _brains(_installed())
        e = next((x for x in entries if x["name"] == name and not x.get("deleted")), None)
        if e is None:
            raise ValueError(f"no brain {name} in the list")
        e["deleted"] = True; e["reason"] = (e.get("reason") or "") + " · deleted by owner"
        _remove(name)
        _save(entries)
        return e


def quality(c) -> float:
    """Music quality for the order among unrated brains: the share of stream checks against the corpus the
    brain's free play passes, how much better than holding the opening bar it replays the tracks, and a beat
    bonus (free-play bars with four kicks and with snare/clap on 2 and 4). Stars set by the listener rank first."""
    checks = c.get("checks") or []
    gate = sum(1 for x in checks if x.get("pass")) / len(checks) if checks else 0.0
    margin = (c.get("replay") or 0) - (c.get("hold") or 0)
    k, sn = _beat_share(c)
    base = gate + 2.0 * margin + 0.5 * k + 0.5 * sn
    q = QUALITY_SCORES().get(c.get("name"))          # the overnight quality check (free runs vs the corpus) weighs most
    return round(base + (3.0 * q if q is not None else 0.0), 4)


QA_RANK = {"PASS": 2, "HOLDS A BEAT": 1, "fail": 0}


def QA_RESULTS():
    """The QA protocol's verdict per checkpoint (brain/quality.py QA_BEAT / QA_FULL, quality/history.jsonl):
    name -> ("PASS" | "HOLDS A BEAT" | "fail", [failed lines]). Unchecked checkpoints are missing."""
    p = ROOT / "quality" / "history.jsonl"
    out = {}
    try:
        for line in p.read_text().splitlines():
            h = json.loads(line)
            q = h.get("qa")
            if q:
                out[h["name"]] = ("PASS" if q.get("pass") else ("HOLDS A BEAT" if q.get("beat") else "fail"), q.get("failed") or [])
    except Exception:
        pass
    return out


def QUALITY_SCORES():
    """Scores of the overnight quality check per checkpoint (quality/history.jsonl, synced by the watcher)."""
    p = ROOT / "quality" / "history.jsonl"
    out = {}
    try:
        for line in p.read_text().splitlines():
            h = json.loads(line)
            out[h["name"]] = float(h["score"])
    except Exception:
        pass
    return out


# ---------- what a brain load is doing: one progress record per holder, for the pages' progress bars ----------
# A load runs through these stages, the first three only when they are needed: finding a run's newest checkpoint in
# the bucket, fetching it, unpacking it into the play cache (once per checkpoint), then building the brain from the
# cache (Brain.load), retuning it, its slot temperatures (or the temperature scan of a brain without one), priming it
# on a starting passage and drawing its layout. Every stage has a time expected from the rates measured on earlier
# loads (LOAD_RATES); a stage that counts its work (bytes fetched or unpacked, bars primed, temperatures scanned)
# reports how far it is, the others count their time against the expected one. The percent is the expected time
# done over the expected time of all stages, and it never goes back.
LOAD_RATES = Path(os.environ.get("CADANCE_LOAD_RATES") or SITE / ".load-rates.json")     # a test instance names its own
# measured 26 Nov on the Mac at load 18 to 40 (a split brain, 1.48 GB unpacked): Brain.load 32 to 39 s from the
# checkpoint, 23 to 25 s from the play cache; priming 72 to 167 s, most of it the library's first-use setup of a
# loaded brain (np.unique over the synapses) and seven settled bars
RATE_DEFAULTS = {
    "finding": 4.0,                 # seconds: listing the run's folders in the bucket
    "fetching": 40e6,               # bytes per second from the bucket
    "unpacking": 150e6,             # unpacked bytes per second: the checkpoint inflated and written as the play cache
    "building": 60e6,               # unpacked bytes per second: Brain.load from the play cache
    "building_packed": 42e6,        # unpacked bytes per second: Brain.load from the compressed checkpoint
    "priming": 15e6,                # unpacked bytes per second: the first settled bars of a loaded brain scale with its size
    "retune": 1.0, "temperatures": 1.0, "calibrating": 60.0, "view": 1.0,     # seconds
}
STAGE_WORDS = {                     # short: the status line also carries the time left and the brain asked for next
    "finding": "finding the newest checkpoint",
    "fetching": "downloading",
    "copying": "copying the run's checkpoint",
    "unpacking": "unpacking (once per checkpoint)",
    "waiting": "waiting for the other load of it",
    "building": "building the brain",
    "retune": "retuning",
    "temperatures": "slot temperatures",
    "calibrating": "scanning temperatures",
    "priming": "priming",
    "view": "drawing the layout",
}


def left_words(seconds):
    """The time a load still takes, in the words the pages show: about 40 s left, about 1 min 20 s left."""
    if seconds is None or not seconds >= 1:
        return ""
    if seconds < 55:
        return f"about {max(5, int(round(seconds / 5.0)) * 5)} s left"
    minutes = int((seconds + 5) // 60)
    rest = int(round((seconds - 60 * minutes) / 10.0)) * 10
    return f"about {minutes} min {rest} s left" if rest > 0 else f"about {minutes} min left"
_RATES_LOCK = threading.Lock()


def load_rates():
    """The kept rates (RATE_DEFAULTS where the file has none)."""
    try:
        saved = json.loads(LOAD_RATES.read_text())
    except (OSError, ValueError):
        saved = {}
    return {k: float(saved[k]) if isinstance(saved.get(k), (int, float)) and saved[k] > 0 else v
            for k, v in RATE_DEFAULTS.items()}


def _learn_rates(measured):
    """Fold the rates of one finished load into the kept ones (half the old value, half the new one)."""
    if not measured:
        return
    with _RATES_LOCK:
        rates = load_rates()
        for k, v in measured.items():
            if k in rates and v > 0 and np.isfinite(v):
                rates[k] = 0.5 * rates[k] + 0.5 * float(v)
        try:
            part = LOAD_RATES.with_suffix(".part")
            part.write_text(json.dumps({k: round(v, 3) for k, v in rates.items()}, indent=1))
            part.replace(LOAD_RATES)
        except OSError as e:
            print(f"load rates: {e}", flush=True)


def _eased(x):
    """A stage timed against its expected length: even up to 80%, then closing in on 100% without reaching it."""
    return x if x <= 0.8 else 1.0 - 0.2 * float(np.exp(-(x - 0.8) / 0.2))


class LoadProgress:
    """The load one holder runs (the player, a deck, the Auto DJ's factory): its stages in order with the seconds
    each is expected to take, the stage in progress and how far it is. read() gives the record the pages draw."""

    def __init__(self, who):
        self.who = who
        self.lock = threading.Lock()
        self.rec = None

    def begin(self, name, carry=False):
        """A new load of `name`; with carry, a load in progress (finding, fetching) goes on under this name."""
        with self.lock:
            if carry and self.rec is not None and not self.rec["finished"]:
                self.rec["name"] = name
                return
            now = time.time()
            self.rec = {"name": name, "plan": {}, "rate_of": {}, "stage": None, "done": None, "total": None,
                        "detail": None, "started": now, "stage_started": now, "finished": None, "error": None,
                        "shown": 0.0, "took": {}, "marks": []}

    def expect(self, stage, seconds=None, nbytes=None, rate=None):
        """Add a stage to the plan (or set it again): `seconds`, or `nbytes` at the kept rate `rate` (the stage's own
        by default); a stage planned by its bytes teaches its rate when the load ends."""
        rates = load_rates()
        if seconds is None:
            key = rate or stage
            seconds = (nbytes or 0) / rates[key] if nbytes else rates.get(key, 1.0)
        with self.lock:
            r = self.rec
            if r is None or r["finished"]:
                return
            r["plan"][stage] = max(0.05, float(seconds))
            if nbytes:
                r["rate_of"][stage] = (rate or stage, float(nbytes))

    def drop(self, stage):
        with self.lock:
            if self.rec is not None and stage != self.rec["stage"]:
                self.rec["plan"].pop(stage, None)

    def at(self, stage, done=None, total=None, detail=None):
        """The load is in `stage` (the stages planned before it are done); done of total counts its work."""
        with self.lock:
            r = self.rec
            if r is None or r["finished"]:
                return
            now = time.time()
            if r["stage"] != stage:
                if r["stage"] is not None:
                    r["took"][r["stage"]] = now - r["stage_started"]
                r["stage"], r["stage_started"], r["marks"] = stage, now, []
                if stage not in r["plan"]:
                    r["plan"][stage] = load_rates().get(stage, 1.0) if stage not in ("fetching", "unpacking", "building", "priming") else 1.0
            if done and done != r["done"]:
                r["marks"].append((done, now))                     # when each counted unit was reached
            r["done"], r["total"], r["detail"] = done, total, detail

    def end(self, error=None):
        """The load is over: the record shows 100% (or the error), and a load that went through teaches its rates."""
        with self.lock:
            r = self.rec
            if r is None or r["finished"]:
                return
            now = time.time()
            if r["stage"] is not None:
                r["took"][r["stage"]] = now - r["stage_started"]
            r["finished"], r["error"] = now, error
            learned = {}
            if error is None:
                for stage, took in r["took"].items():
                    if took < 0.05:
                        continue
                    if stage in r["rate_of"]:
                        key, nbytes = r["rate_of"][stage]
                        learned[key] = nbytes / took
                    elif stage in RATE_DEFAULTS and stage not in ("fetching", "unpacking", "building", "building_packed", "priming"):
                        learned[stage] = took
            took = ", ".join(f"{stage} {t:.1f}s" for stage, t in r["took"].items())
            print(f"load of {r['name']} ({self.who}): {now - r['started']:.1f}s{' failed' if error else ''} · {took}", flush=True)
        _learn_rates(learned)

    def active(self):
        with self.lock:
            return self.rec is not None and not self.rec["finished"]

    @staticmethod
    def _part(r, now, expected):
        """How far the stage in progress is (0..1) and the seconds it still takes, by its count or its time.
        A count of bars (priming, the temperature scan) is read per bar: the first bar carries the brain's first-use
        setup (half of the priming or more), so the later bars are timed from the first one's end."""
        elapsed = now - r["stage_started"]
        done, total, marks = r["done"] or 0, r["total"], r["marks"]
        counted = r["stage"] in ("priming", "calibrating")
        if counted and done <= 0:                                   # before the first bar: at most its share
            return min(0.5, _eased(elapsed / expected)), max(expected - elapsed, 0.4 * expected)
        if not total:
            part = _eased(elapsed / expected)
            return part, max(0.0, expected - elapsed) if part < 0.8 else expected * (1.0 - part) / 0.8
        if not counted:                                             # bytes: an even count
            if done <= 0:
                return 0.0, expected
            return min(1.0, done / total), elapsed * (total - done) / done
        (d0, t0), (d1, t1) = marks[0], marks[-1]
        # measured 26 Nov: a later bar takes 0.07 (load 9) to 0.33 (load 20) of the first one
        per = (t1 - t0) / (d1 - d0) if d1 > d0 else 0.2 * (t0 - r["stage_started"])
        remaining = max(0.1 * per, (total - done) * per - (now - t1))
        return min(1.0, elapsed / (elapsed + remaining)), remaining

    def read(self):
        with self.lock:
            r = self.rec
            if r is None:
                return None
            now = time.time()
            plan, stage = r["plan"], r["stage"]
            order = list(plan)
            whole = sum(plan.values()) or 1.0
            if r["finished"]:
                percent = 100.0 if r["error"] is None else r["shown"]
                left = 0.0
            else:
                at = order.index(stage) if stage in order else len(order)
                spent = sum(plan[s] for s in order[:at])
                later = sum(plan[s] for s in order[at + 1:])
                part, left = 0.0, later
                if stage in plan:
                    part, here = self._part(r, now, plan[stage])
                    spent += part * plan[stage]
                    left += here
                # the share of the stages done, and never more than the time spent over the time spent and left:
                # a stage that overruns its estimate holds the bar instead of pushing it to 99%
                took = now - r["started"]
                percent = min(99.0, 100.0 * spent / whole, 100.0 * took / max(1e-6, took + left))
            r["shown"] = max(r["shown"], percent)
            words = STAGE_WORDS.get(stage, stage or "starting")
            if r["finished"]:
                words = "ready" if r["error"] is None else f"failed: {r['error']}"
            sep = " " if stage == "fetching" else ": "                   # "downloading 120 of 365 MB"
            return {"who": self.who, "name": r["name"], "stage": stage, "text": words + (f"{sep}{r['detail']}" if r["detail"] and not r["finished"] else ""),
                    "percent": round(r["shown"], 1), "done": r["done"], "total": r["total"], "left": round(left, 1),
                    "left_text": left_words(left) if not r["finished"] and r["shown"] < 99 else "",
                    "elapsed": round((r["finished"] or now) - r["started"], 1), "finished": r["finished"] is not None,
                    "ago": round(now - r["finished"], 1) if r["finished"] else None, "error": r["error"]}


LOAD_PROGRESS = {who: LoadProgress(who) for who in ("player", "deck0", "deck1", "deck2", "dj")}


def all_progress():
    return {"player": LOAD_PROGRESS["player"].read(), "decks": [LOAD_PROGRESS[f"deck{i}"].read() for i in range(3)],
            "dj": LOAD_PROGRESS["dj"].read()}


# ---------- the play cache: a checkpoint as a plain archive, so a load reads its arrays without inflating them ----------
# A checkpoint is a zip of deflated arrays (0.36 to 1.96 GB on disk, 1.5 to 4.8 GB unpacked). Brain.load reads every
# array once and the learner's arrays (the synapses and their optimizer state) a second time, and each read of a
# deflated array inflates it again. The play cache holds the same .npy members stored without compression (the
# archive Brain.save(compressed=False) writes), written once per checkpoint beside the others in .playcache/; a load
# then reads plain bytes. The cache is checked against the checkpoint's size and time and kept within PLAY_CACHE_GB.
PLAY_CACHE = Path(os.environ.get("CADANCE_PLAY_CACHE_DIR") or SITE / ".playcache")
PLAY_CACHE_ON = os.environ.get("CADANCE_PLAY_CACHE", "1") != "0"
PLAY_CACHE_BYTES = float(os.environ.get("CADANCE_PLAY_CACHE_GB", "16")) * 1e9
_CACHE_LOCKS: dict = {}
_CACHE_GUARD = threading.Lock()


def unpacked_bytes(npz):
    """The size of a checkpoint's arrays unpacked (read from the zip's directory, no array is read)."""
    import zipfile
    try:
        with zipfile.ZipFile(npz) as z:
            return sum(i.file_size for i in z.infolist())
    except (OSError, zipfile.BadZipFile):
        return 0


def _play_paths(npz):
    import hashlib
    src = Path(npz).resolve()
    key = hashlib.sha1(str(src).encode()).hexdigest()[:8]
    target = PLAY_CACHE / f"{src.stem}.{key}.npz"
    return src, target, target.with_suffix(".json")


def _play_stamp(src):
    st = src.stat()
    return {"source": str(src), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def play_cache_hit(npz):
    """The play cache of this checkpoint when it is written and current, else None."""
    if not PLAY_CACHE_ON:
        return None
    try:
        src, target, stamp = _play_paths(npz)
        if target.is_file() and json.loads(stamp.read_text()) == _play_stamp(src):
            os.utime(target)                  # the newest use: the prune keeps it longest
            return target
    except (OSError, ValueError):
        pass
    return None


def write_play_cache(npz, progress=None):
    """Write the play cache of this checkpoint (each member inflated once and stored as it is, so the arrays are
    the checkpoint's to the byte) and return its path; None when it cannot be written (the load then reads the
    checkpoint itself). A second load of the same checkpoint waits for the first one's cache."""
    import shutil
    import zipfile
    if not PLAY_CACHE_ON:
        return None
    src, target, stamp = _play_paths(npz)
    with _CACHE_GUARD:
        lock = _CACHE_LOCKS.setdefault(str(target), threading.Lock())
    if not lock.acquire(blocking=False):
        if progress is not None:
            progress.at("waiting")
        lock.acquire()
    try:
        hit = play_cache_hit(npz)
        if hit is not None:
            return hit
        PLAY_CACHE.mkdir(exist_ok=True)
        with zipfile.ZipFile(src) as zin:
            infos = zin.infolist()
            total = sum(i.file_size for i in infos)
            if shutil.disk_usage(PLAY_CACHE).free < total + 8e9:        # the disk keeps 8 GB free beside the cache
                print(f"play cache: not enough free disk for {src.name} ({total / 1e9:.2f} GB)", flush=True)
                return None
            part = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.part")
            started, done = time.time(), 0
            try:
                with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
                    for k, info in enumerate(infos):
                        member = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                        member.compress_type = zipfile.ZIP_STORED
                        member.file_size = info.file_size
                        with zin.open(info) as fin, zout.open(member, "w", force_zip64=info.file_size > 2**31 - 1) as fout:
                            while True:
                                chunk = fin.read(8 << 20)
                                if not chunk:
                                    break
                                fout.write(chunk)
                                done += len(chunk)
                                if progress is not None:
                                    progress.at("unpacking", done, total, f"{done / 1e6:.0f} of {total / 1e6:.0f} MB")
                stamp.write_text(json.dumps(_play_stamp(src)))
                part.replace(target)
            finally:
                part.unlink(missing_ok=True)
        print(f"play cache: {src.name} unpacked in {time.time() - started:.1f}s ({total / 1e9:.2f} GB)", flush=True)
        _prune_play_cache(keep=target)
        return target
    except Exception as e:                 # the load goes on from the checkpoint itself
        print(f"play cache of {src.name}: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        lock.release()


# What a loaded brain works out on its first settled bars and keeps (cadence.blocks.Layout's cached properties: whether
# synapses share an entry, the entries' order, each synapse's block) depends on its layout only. The first load from
# a play copy keeps what the library computed beside the copy (<copy>.layouts); later loads give it back to the new
# brain's layouts before the first bar. A layout takes the kept values only when its arrays hash to the same digest.
LAYOUT_CACHED = ("parallel_synapses", "_flat_order", "block_of_edge")


def _brain_layouts(brain):
    """The brain's layouts by name: its learned code's graph and its learner's network (None where it has none)."""
    out = {}
    code = getattr(getattr(getattr(brain, "learned_code", None), "graph", None), "layout", None)
    net = getattr(getattr(getattr(brain, "learner", None), "brain", None), "layout", None)
    for name, lay in (("code", code), ("net", net)):
        if lay is not None and all(hasattr(lay, f) for f in ("starts", "pair_pre", "pair_post", "offset", "edge_index")):
            out[name] = lay
    return out


def _layout_digest(lay):
    import hashlib
    h = hashlib.sha1()
    for f in ("starts", "pair_pre", "pair_post", "offset", "edge_index"):
        a = np.ascontiguousarray(getattr(lay, f))
        h.update(f"{f}:{a.dtype.str}:{a.shape}".encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def keep_layout_caches(brain, copy, echo_lags=None):
    """After a load from the play copy `copy`: keep the brain's layout caches beside it, once."""
    side = Path(copy).with_suffix(".layouts")
    if side.exists():
        return
    arrays = {}
    for name, lay in _brain_layouts(brain).items():
        held = lay.__dict__
        if "parallel_synapses" not in held:
            continue
        arrays[f"{name}/digest"] = np.array(_layout_digest(lay))
        arrays[f"{name}/parallel_synapses"] = np.array(bool(held["parallel_synapses"]))
        if "_flat_order" in held:
            order = held["_flat_order"]
            arrays[f"{name}/flat_order_none"] = np.array(order is None)
            if order is not None:
                arrays[f"{name}/flat_order"], arrays[f"{name}/flat_sorted"] = order
        if "block_of_edge" in held:
            arrays[f"{name}/block_of_edge"] = held["block_of_edge"]
    if not arrays:
        return
    arrays["library"] = np.array(cadence.__version__)       # caches of another library version are not given back
    arrays["echo_lags"] = np.array(json.dumps(list(echo_lags or [])))     # nor those of another observation
    part = side.with_name(f"{side.name}.{os.getpid()}.part")
    try:
        with open(part, "wb") as f:
            np.savez(f, **arrays)
        part.replace(side)
    except OSError as e:
        print(f"layout caches of {Path(copy).name}: {e}", flush=True)
        part.unlink(missing_ok=True)


def fill_layout_caches(brain, copy, echo_lags=None):
    """Give a brain just loaded from `copy` the layout caches kept beside it; returns the names filled."""
    side = Path(copy).with_suffix(".layouts")
    if not side.is_file():
        return []
    filled = []
    try:
        with np.load(side, allow_pickle=False) as kept:
            if "library" not in kept.files or str(kept["library"]) != cadence.__version__:
                return []
            if (str(kept["echo_lags"]) if "echo_lags" in kept.files else "[]") != json.dumps(list(echo_lags or [])):
                return []
            for name, lay in _brain_layouts(brain).items():
                if f"{name}/digest" not in kept.files or str(kept[f"{name}/digest"]) != _layout_digest(lay):
                    continue
                held = lay.__dict__
                held.setdefault("parallel_synapses", bool(kept[f"{name}/parallel_synapses"]))
                if f"{name}/flat_order_none" in kept.files:
                    held.setdefault("_flat_order", None if bool(kept[f"{name}/flat_order_none"]) else
                                    (np.asarray(kept[f"{name}/flat_order"]), np.ascontiguousarray(kept[f"{name}/flat_sorted"])))
                if f"{name}/block_of_edge" in kept.files:
                    held.setdefault("block_of_edge", np.asarray(kept[f"{name}/block_of_edge"]))
                filled.append(name)
    except (OSError, ValueError, KeyError) as e:
        print(f"layout caches of {Path(copy).name}: {type(e).__name__}: {e}", flush=True)
    return filled


def _prune_play_cache(keep=None):
    """Drop caches whose checkpoint is gone or changed, and the least recently used ones beyond PLAY_CACHE_BYTES."""
    try:
        rows = []
        for f in PLAY_CACHE.glob("*.npz"):
            stamp = f.with_suffix(".json")
            try:
                saved = json.loads(stamp.read_text())
                current = Path(saved["source"]).is_file() and _play_stamp(Path(saved["source"])) == saved
            except (OSError, ValueError, KeyError):
                current = False
            if not current and f != keep:
                f.unlink(missing_ok=True)
                stamp.unlink(missing_ok=True)
                f.with_suffix(".layouts").unlink(missing_ok=True)
                continue
            rows.append((f.stat().st_mtime, f.stat().st_size, f))
        kept = 0
        for _, size, f in sorted(rows, reverse=True):
            kept += size
            if kept > PLAY_CACHE_BYTES and f != keep:
                f.unlink(missing_ok=True)
                f.with_suffix(".json").unlink(missing_ok=True)
                f.with_suffix(".layouts").unlink(missing_ok=True)
        for f in PLAY_CACHE.glob("*.part"):                          # a write a restart cut short
            if time.time() - f.stat().st_mtime > 3600:
                f.unlink(missing_ok=True)
    except OSError as e:
        print(f"play cache prune: {e}", flush=True)


LAYOUT = threading.RLock()            # common.configure is module state: one brain's layout at a time (DJ mix: two brains)
_LAYOUT_OWNER = [None]


class Creature:
    def __init__(self):
        self.lock = threading.RLock()       # one lock for loading and playing: the layout lives in module state
        self.brain = None
        self.card = None
        self.state = None
        self.bar = 0
        self.first = True
        self.pending = 0.0                  # listener reward held for the next moment
        self.history = deque(maxlen=64)     # (bar, state or tokens, mods, mood) as answered
        self.heard = []                     # the bars heard so far, oldest first, each (1, SLOTS): what an echo brain hears back
        self.mood = [0.5, 0.5]
        self.view_layout = None
        self.play_mode = PLAY_MODE
        self.temperature = 0.05
        self.rng = np.random.default_rng()
        self.groove = None
        self.style = common.REFERENCE_STYLE  # the style the listener asks for (brains trained on several styles)
        self.follow_latest = True           # the newest checkpoint plays unless the listener picked one
        self.load_lock = threading.Lock()
        self.loading = None                 # the brain being loaded: /api/next answers "loading" instead of waiting
        self.last_request = 0.0             # time of the page's last bar request: the follow thread waits while it plays
        self.kickless = 0                   # bars since the last audible kick (the kick hotfix, _keep_kick)
        self.held, self.held_tok = [None] * 3, [None] * L
        self.last_kick = None               # (beat token, kick row slot value) of the last bar with a kick
        self.kick_fixes = 0
        self.demo = self._demo_bars()
        self.demo_pool = None               # the pool the instrument check plays, when not the default one
        self.with_view = True               # the brain view (brainview.snapshot) for the main page; the mixer's decks skip it
        self.want3d = 0.0                   # when a page last asked for every drawn neuron per bar (its 3D view is on)
        self.progress = LOAD_PROGRESS["player"]     # where this holder's loads report (a deck's and the Auto DJ's own)
        self.report = None                          # the load a stand-in Creature (Creature._load) reports its steps to

    @staticmethod
    def _demo_bars():
        score = ROOT / "engine" / "demo_score.json"
        if score.exists():
            bars = json.loads(score.read_text()).get("bars") or []
            if bars:
                return bars
        return [[1, 0, 0, 0]] * 8 + [[1, 1, 0, 0]] * 16 + [[1, 2, 1, 0]] * 16 + [[2, 3, 1, 1]] * 16 + [[0, 3, 2, 1]] * 8

    def pool_name(self):
        if self.brain is None and self.demo_pool and (ROOT / "pool" / self.demo_pool).exists():
            return self.demo_pool
        name = (self.card or {}).get("pool", "pool.json")
        return name if self.brain is not None and (ROOT / "pool" / name).exists() else "pool_v0.json"

    def load(self, name):
        with self.load_lock:                  # one load at a time; the stream keeps answering meanwhile
            self.loading = name
            self.progress.begin(name, carry=True)       # a fetch for this load (finding, fetching) goes on in the same record
            error = None
            try:
                pre = self._read(name)
                with LAYOUT, self.lock:
                    # A failed load can leave common configured for the candidate.
                    # The next player must restore its own layout before composing.
                    _LAYOUT_OWNER[0] = None
                    out = self._load(name, pre)
                    _LAYOUT_OWNER[0] = self
                if getattr(self, "copy_path", None) is not None and not self.copy_path.with_suffix(".layouts").exists():
                    # kept once per copy, after priming worked them out; written beside the copy in the background
                    threading.Thread(target=keep_layout_caches, args=(self.brain, self.copy_path, (self.card or {}).get("echo_lags")), daemon=True).start()
                return out
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                self.loading = None
                self.progress.end(error)

    def renew(self):
        """A new composition by the brain that plays: primed again from another passage in full flow."""
        with LAYOUT, self.lock:
            if self.brain is not None:
                self.ensure_layout()
            if self.brain is None:
                return
            if self.first and self.bar == 0:
                return  # loading already prepared this opening
            self.state = self._prime(self.brain, self.card)
            self.bar, self.first, self.pending = 0, True, 0.0
            self.history.clear()

    def _read(self, name):
        """The checkpoint read from disk (the slow part of a load; needs no layout, so the music goes on meanwhile)."""
        from cadence import Brain

        available = {card["name"]: (card, where) for card, where in _available_checkpoints()}
        source = available.get(name)
        if source is None and all((AUDITION / f"{name}{suffix}").is_file() for suffix in (".json", ".npz")):
            source = (json.loads((AUDITION / f"{name}.json").read_text()), "audition")      # fetched for a listen
        if source is None:
            raise ValueError(f"no checkpoint {name}")
        card = _summary(*source)
        # The menu index trims free runs and layout metadata. Load the full training card.
        sidecar = _dir(card["dir"]) / f"{name}.json"
        full = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        card = {**card, **full, "dir": card["dir"]}
        if card.get("cadence_patch_sha256") and card["cadence_patch_sha256"] not in SUPPORTED_CHECKPOINT_PATCHES:
            raise ValueError("This checkpoint requires a different Cadence recall patch")
        if card.get("patch_book"):
            if tuple(map(int, cadence.__version__.split(".")[:3])) < (0, 103, 0):
                raise ValueError("Composer checkpoints require cadence-net 0.103.0 or newer")
        life = LIVES / f"{name}.life.npz"
        if self.play_mode != "live":
            life = Path("/nonexistent")
        # the stages of this load with the time each is expected to take (LoadProgress); the checkpoint is read
        # from its play cache, written first when this checkpoint has none
        npz = life if life.exists() else _dir(card["dir"]) / f"{name}.npz"
        size = unpacked_bytes(npz)
        cache = None if life.exists() else play_cache_hit(npz)
        p = self.progress
        for stage in ("unpacking", "building", "priming"):  # the stand-ins a listing or a download planned before the size was known
            p.drop(stage)
        if cache is None and not life.exists() and PLAY_CACHE_ON:
            p.expect("unpacking", nbytes=size)
        p.expect("building", nbytes=size, rate="building" if cache is not None or (PLAY_CACHE_ON and not life.exists()) else "building_packed")
        p.expect("retune")
        if not (play_reading(card).get("temperature") or card.get("temperature")):
            p.expect("calibrating")
        p.expect("temperatures")
        p.expect("priming", nbytes=size)
        if self.with_view:
            p.expect("view")
        if cache is None and not life.exists():
            cache = write_play_cache(npz, p)
            if cache is None:                               # read from the checkpoint itself, at the packed rate
                p.drop("unpacking")
                p.expect("building", nbytes=size, rate="building_packed")
        p.at("building", detail=f"{size / 1e9:.1f} GB{' from the play copy' if cache is not None else ''}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # CADANCE_BACKEND=torch puts the brain on the GPU (the repertoire factory on a box); the Mac plays on the CPU
            brain = Brain.load(cache or npz, backend=os.environ.get("CADANCE_BACKEND", "cpu"))
            # the layout caches an earlier load from this copy kept: the first bars then skip working them out
            # (26 Nov, split brain, load 10 to 15: about 10 s less; the first 16 bars identical to a load without them)
            if cache is not None and fill_layout_caches(brain, cache, card.get("echo_lags")):
                print(f"layout caches of {name} given from its play copy", flush=True)
            self.copy_path = cache
            p.at("retune")
            if not life.exists():
                brain.reset()
            try:
                brain.retune(arousal_need=NEED)
            except ValueError:                    # a brain built without arousal (Cadence's default composed brain): plain retune
                brain.retune()
            # the play tolerance (play_reading "play_tolerance"), for play only: a brain in live mode learns from the
            # listener and its state is saved, so it keeps its own; a checkpoint on disk is never changed by it
            tolerance = play_reading(card).get("play_tolerance") if self.play_mode != "live" else None
            if tolerance is not None:
                brain.retune(learning_tolerance=tolerance)
                print(f"play tolerance of {name}: {tolerance:g}", flush=True)
        scaled = apply_route_scale(brain, card)            # after the retune: the play reading's weaker copy route
        if scaled is not None:
            print(f"route scaled for {name}: {np.round(scaled, 3).tolist()}", flush=True)
        return card, brain

    def _load(self, name, pre=None):
        card, brain = pre or self._read(name)
        candidate = Creature()
        candidate.style = self.style
        candidate.report = self.progress                  # its temperature scan and priming report to this load
        candidate._configure(card)
        temperature = play_reading(card).get("temperature") or card.get("temperature")
        if not temperature:
            temperature = candidate._calibrate(brain, card)
        self.progress.at("temperatures")
        temperature = candidate._slot_temperatures(brain, card, temperature)
        temperature = play_temperatures(card, temperature)
        candidate._configure(card)
        self.progress.at("priming")
        state = candidate._prime(brain, card)
        if self.with_view:
            self.progress.at("view")
        try:
            view = brainview.layout(brain) if self.with_view else None
        except Exception as e:
            view = {"error": str(e)}
        self.brain, self.card, self.state = brain, card, state
        self.pool, self.style, self.temperature = candidate.pool, candidate.style, temperature
        self.groove = candidate.groove
        self.view_layout = view
        self.bar, self.first, self.pending = 0, True, 0.0
        self.bar_pos, self.seed_title = candidate.bar_pos, candidate.seed_title
        self.held, self.held_tok = candidate.held, candidate.held_tok
        self.heard = candidate.heard
        self.kickless, self.last_kick, self.kick_fixes = candidate.kickless, candidate.last_kick, candidate.kick_fixes
        self.calm_rows, self.row_capped = {}, {}            # the row caps (_cap_rows) start anew with the brain
        self.calm_lines, self.line_capped = {}, {}          # and the line caps (_cap_lines)
        self.history.clear()

    def _slot_temperatures(self, brain, card, base):
        """One temperature per voice (common.calibrate_slots; kept beside the checkpoint as <name>.slots.json), so
        every voice changes as often as in the tracks; the one number `base` when the corpus is not here."""
        name = card.get("streams") or ("streams_r2_drums.json" if card.get("steps") else "streams.json")
        path = ROOT / "corpus" / name
        if not common.SLOT_CALIBRATION or not path.exists():     # never fitted while the player starts: it blocked the page for minutes
            return base
        try:
            if card.get("content"):
                common.set_content(self.pool)
            cache = _dir(card["dir"]) / f"{card['name']}.slots.json"
            sid = ({"sid_book": timbre.sid_book(json.loads((ROOT / "pool" / card["sid_pool"]).read_bytes()))}
                   if card.get("sid_live") and card.get("sid_pool") else {})      # a SID brain's bars carry the chip's settings
            tracks = None if cache.exists() else common.load_streams(path, steps=card.get("steps", False), rows=card.get("beat_rows"), clock=card.get("clock", False), synth=card.get("synth_rows"), patch_book=card.get("patch_book"), **sid)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temps = common.slot_temperatures(brain, tracks, {"temperature": base}, cache=cache, style=self.style)
            return temps if len(temps) == common.SLOTS else base
        except Exception as e:
            print(f"slot temperatures of {card.get('name')}: {type(e).__name__}: {e}; one temperature for all", flush=True)
            return base

    def ensure_layout(self):
        """Set this brain's layout when another brain's is in place (a DJ mix runs two brains bar by bar)."""
        if _LAYOUT_OWNER[0] is not self and self.card is not None:
            self._configure(self.card)
            if self.card.get("content"):
                common.set_content(self.pool)
        _LAYOUT_OWNER[0] = self

    def _configure(self, card):
        """Layout, pool and heard content of the brain this card describes."""
        common.ONOFF = bool(card.get("onoff"))
        common.HEAR_FIRST = bool(card.get("hear_first"))      # the brain was raised hearing every bar before answering it
        if hasattr(common, "shape_from"):
            common.shape_from(card)                           # the reserved inputs and SID slots a stage-1 brain was born with
        if card.get("reserve") and not card.get("patch_book") and card.get("pool"):
            # a run's rolling card (<run>_latest) carries no patch book: it is the teachers of the pool the run
            # was trained with, as train.py builds it
            pool_file = ROOT / "pool" / Path(str(card["pool"])).name
            if pool_file.is_file():
                card["patch_book"] = timbre.teachers(json.loads(pool_file.read_bytes()))
        common.configure(card["tokens_per_layer"], card["mod_levels"], steps=card.get("steps", False), styles=card.get("styles"), rows=card.get("beat_rows"), clock=card.get("clock", False), synth=card.get("synth_rows"), patch_book=card.get("patch_book"))
        if common.STYLES and self.style not in common.STYLES:
            self.style = next((s for s in common.STYLES if common.REFERENCE_STYLE in s), common.STYLES[0])
        raw = (ROOT / "pool" / card.get("pool", "pool.json")).read_bytes()
        self.pool = json.loads(raw)
        if card.get("content"):              # this brain hears its sounds' content: build it from the pool it plays
            common.set_content(self.pool)
        else:
            common.set_content(None)
        if card.get("pool_sha"):
            import hashlib
            if hashlib.sha256(raw).hexdigest()[:12] != card["pool_sha"]:
                if (card.get("patch_book") or {}).get("samples"):
                    raise ValueError("This sampler checkpoint needs its original sample pack")
                print(f"note: {card['name']} was trained on pool {card['pool_sha']}, the site's pool differs")

    def _seed(self, card, n=8):
        """n bars (states + mood) of a passage in full flow with a simple beat from a training track of the corpus
        this brain learned from, the listener's style first: the brain continues a real passage instead of starting
        cold. The passages come from a cache (seed_passages); while it is being built, the brain's own free run
        from its card stands in."""
        pick = seed_passages(card, n)
        rng = np.random.default_rng()
        if pick:
            pool = [p for p in pick if p[4] == self.style] or pick
            # acid start (owner, 26 Nov): a brain that learned the acid voice plays it only from an acid context, so this
            # share of its compositions starts from an acid passage (play setting "acid_start", per brain)
            share = float(play_reading(card).get("acid_start") or 0.0)
            acid = [p for p in pick if len(p) > 5 and p[5] == "acid"]
            if acid and share and rng.random() < share:
                bars, mood, title, start = acid[rng.integers(len(acid))][:4]
                return [list(map(int, b)) for b in bars], mood, title, start
            # HOTFIX (11-23 night): brains keep the layers the opening passage has on and rarely bring one in, so a
            # composition starts from a passage where the bass and the lead both play in most bars (when there is one)
            def full(p):
                b = common.sounds(np.asarray(p[0]))
                return (b[:, 1] > 0).mean() >= 0.75 and (b[:, 2] > 0).mean() >= 0.75
            pool = [p for p in pool if full(p)] or pool
            bars, mood, title, start = pool[rng.integers(len(pool))][:4]      # passages carry style and group after these four
            return [list(map(int, b)) for b in bars], mood, title, start
        fr = card.get("free_run")
        if isinstance(fr, list) and len(fr) >= n and all(isinstance(r, list) and len(r) == common.SLOTS for r in fr[:n]):
            return [list(map(int, r)) for r in fr[:n]], common.mood_rows([[0.5, 0.5]] * (n + 1), self.style).tolist(), "its own free run", 0
        return None

    def _prime(self, brain, card, seed=None):
        seed = seed or self._seed(card)
        groove_seed = int(self.rng.integers(2**32)) if card.get("groove") else None
        dwell = play_reading(card).get("voice_dwell")
        go = play_reading(card).get("dwell_lets_go")
        self.groove = groove.player_controller({**card, "voice_dwell": dwell, **({"dwell_lets_go": go} if go else {})} if dwell else card, groove_seed)
        if card.get("content"):
            common.set_content(self.pool)
        brain.reset()
        self.bar_pos = 0
        self.seed_title = None
        self.held, self.held_tok = [None] * 3, [None] * L
        self.kickless, self.last_kick, self.kick_fixes = 0, None, 0
        if seed is None:
            mid = (common.M - 1) // 2
            start = [int(t) for t in card["start"]] if card.get("start") and len(card["start"]) == common.SLOTS else [0] * L + [mid] * common.MODS + [0] * (common.SLOTS - L - common.MODS)
            self.heard = [np.array([start])]
            return start
        bars, mood, title, start = seed
        self.bar_pos = start + len(bars)          # the bar count of the next bar the brain writes (bar clock)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heard = []
            for t in range(len(bars) - 1):
                heard.append(bars[t:t + 1])
                common.answer(brain, heard_observation(heard, mood[t + 1:t + 2]))
                if self.report is not None:              # a load's priming (a new composition's is not reported)
                    self.report.at("priming", t + 1, len(bars) - 1, f"bar {t + 1} of {len(bars) - 1}")
        self.seed_title = title
        self.heard = keep_heard([bars[t:t + 1] for t in range(len(bars))])      # the passage, its last bar the one heard now
        if self.groove is not None:
            self.groove.prime(bars)
        for b in bars:
            self._keep_kick([int(v) for v in b], watch=False)
        return [int(v) for v in bars[-1]]

    def _cap_kick(self, st):
        """HOTFIX (owner 26 Nov 10:20), player side: the lag-shape brains copy a busy kick once they sampled one (8 to
        15 kicks per bar to the end of a track; the material plays 3.3, the reference brain 3.2 to 4.1). A bar whose
        kick row has more than kick_cap hits (play setting "kick_cap", per brain) is played with the kick row of the
        last bar at or under it, and the brain hears it so, which ends the copy loop. Training on the brain's own bars
        (own play) has to make this unnecessary; /api/next counts the bars capped."""
        cap = getattr(self, "kick_cap", None)
        if not cap or not common.STEPS:
            return st
        k = L + common.MODS
        width = 1 if common.ROWS else common.STEPS // len(common.DRUM_ROWS)
        if int((np.asarray(common.decode_beat(st)[0]) > 0).sum()) <= cap:
            self.calm_kick = list(st[k:k + width])
            return st
        st = list(st)
        if getattr(self, "cap_mode", None) == "thin":       # thinned in place, the quarter-note steps kept longest
            if common.ROWS:
                return st                                   # a row token has no steps to thin: left as written
            steps = list(st[k:k + width])
            on = sorted((i for i in range(width) if steps[i] > 0), key=lambda i: (i % 4 == 0, steps[i], -i))
            for i in on[:max(0, len(on) - int(cap))]:
                steps[i] = 0
            st[k:k + width] = steps
            self.kick_capped = getattr(self, "kick_capped", 0) + 1
            return st
        if getattr(self, "calm_kick", None) is not None:
            st[k:k + width] = self.calm_kick
        elif common.ROWS:                                  # none yet: the vocabulary's most four-on-the-floor kick row
            rows = common.ROWS[common.DRUM_ROWS[0]]
            st[k] = max(range(len(rows)), key=lambda i: (np.asarray(rows[i])[[0, 4, 8, 12]] > 0).sum() - 0.01 * (np.asarray(rows[i]) > 0).sum())
        else:
            st[k:k + width] = [1 if i % 4 == 0 else 0 for i in range(width)]
        self.kick_capped = getattr(self, "kick_capped", 0) + 1
        return st

    def _cap_rows(self, st):
        """The kick cap for the other drum rows (play setting "row_caps", per brain; the kick row is _cap_kick's): a
        bar whose row has more hits than its cap is played with that row as it was in the last bar at or under the
        cap with at least one hit (an empty row is never the calm copy: 27 Nov, an empty first snare row silenced the
        snare for 16 bars), each row its own, and the brain hears it so. Before any such bar a row in step form is
        thinned to the cap (plain hits before accents, off-quarter steps and later steps first); a row token is left
        as it is."""
        caps = getattr(self, "row_caps", None)
        self.capped_now = []
        if not caps or not common.STEPS:
            return st
        k = L + common.MODS
        width = 1 if common.ROWS else common.STEPS // len(common.DRUM_ROWS)
        beat = common.decode_beat(st)
        calm = self.__dict__.setdefault("calm_rows", {})
        counts = self.__dict__.setdefault("row_capped", {})
        thin = getattr(self, "cap_mode", None) == "thin"    # thinned in place, no calm copy
        st = list(st)
        for r, name in enumerate(common.DRUM_ROWS):
            cap = caps[r] if r < len(caps) else None
            if r == 0 or not cap:
                continue
            a, b = k + r * width, k + (r + 1) * width
            hits = int((np.asarray(beat[r]) > 0).sum())
            if hits <= cap:
                if hits:
                    calm[r] = list(st[a:b])
                continue
            if r in calm and not thin:
                st[a:b] = calm[r]
            elif not common.ROWS:
                steps = list(st[a:b])
                on = sorted((i for i in range(width) if steps[i] > 0), key=lambda i: (steps[i], i % 4 == 0, -i))
                for i in on[:max(0, len(on) - int(cap))]:
                    steps[i] = 0
                st[a:b] = steps
            else:
                continue
            counts[name] = counts.get(name, 0) + 1
            self.capped_now.append(name)
        return st

    def _cap_lines(self, st):
        """The row caps for the synth lines (play setting "line_caps", per brain): a bar whose layer has more notes
        (attacks in common.decode_synth's row) than its cap is played with that layer's line as it was in the last bar
        at or under the cap that had a note, each layer its own; the layer's sound and controls stay the bar's own.
        Before such a bar a line in step form is thinned to the cap (the dropped attacks become ties, so the rhythm
        grid stays and the note before sounds on; off-quarter and later steps first; a step line holds no accents)
        and the thinned line is the first calm copy; a line of tokens (factored, vocabulary) is left as it is. The
        brain hears the bar as played."""
        caps = getattr(self, "line_caps", None)
        self.lines_capped_now = []
        if not caps or not common.SYN:
            return st
        rows = common.decode_synth(st)
        if rows is None:
            return st
        base = L + common.MODS + common.STEPS
        width = common.SYN // len(common.SYNTH_LAYERS)
        calm = self.__dict__.setdefault("calm_lines", {})
        counts = self.__dict__.setdefault("line_capped", {})
        thin = getattr(self, "cap_mode", None) == "thin"    # thinned in place, no calm copy
        st = list(st)
        for v, name in enumerate(common.SYNTH_LAYERS):
            cap = caps[v] if v < len(caps) else None
            if not cap:
                continue
            a, b = base + v * width, base + (v + 1) * width
            notes = int((np.asarray(rows[v]) >= 2).sum())
            if notes <= cap:
                if notes:
                    calm[v] = list(st[a:b])
                continue
            if v in calm and not thin:
                st[a:b] = calm[v]
            elif common.SYNTH.get("kind") == "steps":
                steps = list(st[a:b])
                on = sorted((i for i in range(width) if steps[i] >= 2), key=lambda i: (i % 4 == 0, -i))
                for i in on[:max(0, len(on) - int(cap))]:
                    steps[i] = 1                 # the attack becomes a tie: the note before it sounds on
                st[a:b] = steps
                if not thin:
                    calm[v] = list(steps)
            else:
                continue
            counts[name] = counts.get(name, 0) + 1
            self.lines_capped_now.append(name)
        return st

    def _keep_kick(self, st, watch=True):
        """HOTFIX (owner 11-23), player side: a brain may go KICK_WAIT bars without an audible kick (the corpus:
        95% of the kickless stretches inside the reference tracks are <= 28 bars); then the player plays its last
        kick row (and its beat sound if that went silent) in every bar, and the brain hears those bars as its own,
        until the brain writes a kick itself: then it leads again, breakdowns included. Training has to make this
        unnecessary; /api/next reports the bars the player held the kick."""
        if not common.ROWS:
            return st
        k = L + common.MODS
        if common.sounds(st)[0] > 0 and common.decode_beat(st)[0].any():
            self.kickless, self.last_kick = 0, (int(common.sounds(st)[0]), st[k])
            return st
        if not watch:
            return st
        self.kickless += 1
        if self.kickless <= KICK_WAIT:
            return st
        self.kickless = KICK_WAIT                            # held: the next kickless bar gets the kick too
        if self.last_kick is None:                         # never heard one: the vocabulary's most four-on-the-floor kick row
            rows = common.ROWS[common.DRUM_ROWS[0]]
            row = max(range(len(rows)), key=lambda i: (np.asarray(rows[i])[[0, 4, 8, 12]] > 0).sum() - 0.01 * (np.asarray(rows[i]) > 0).sum())
            self.last_kick = (st[0] or 1, row)
        st = list(st)
        if common.sounds(st)[0] == 0:
            common.set_sound(st, 0, int(self.last_kick[0]))
        st[k] = int(self.last_kick[1])
        self.kick_fixes += 1
        if self.kick_fixes % 32 == 1:
            print(f"kick hotfix: {self.card and self.card.get('name')} bar {self.bar} ({self.kick_fixes} bars held)", flush=True)
        return st

    def _calibrate(self, brain, card, bars=48):
        """For a checkpoint without a calibrated temperature: the sampling temperature whose free run changes
        sounds about as often as the corpus does (corpus_reference.json from the training run)."""
        ref_path = ROOT / "brains" / "corpus_reference.json"
        target = 0.11
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())
            target = float(np.mean([ref[n]["change"] for n in ("beat", "bass", "lead", "air") if n in ref]))
        seed = self._seed(card)
        common.ENTRY_SCALE, common.LINE_SCALE = reading(card)
        common.SYNTH_SCALE = play_reading(card).get("synth_line_scale")
        common.KICK_SCALE = play_reading(card).get("kick_scale")
        if card.get("content"):
            common.set_content(json.loads((ROOT / "pool" / card.get("pool", "pool.json")).read_text()))
        best, scan = 0.05, []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            steps = (0.02, 0.05, 0.07, 0.085, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3)
            for k, t in enumerate(steps):
                if self.report is not None:
                    self.report.at("calibrating", k, len(steps), f"{k} of {len(steps)} temperatures")
                brain.reset()
                pos = 0
                heard = []
                if seed is not None:
                    for k in range(len(seed[0]) - 1):
                        heard.append(seed[0][k:k + 1])
                        common.answer(brain, heard_observation(heard, seed[1][k + 1:k + 2]))
                    cur = seed[0][-1:].copy()
                    pos = seed[3] + len(seed[0])
                else:
                    cur = np.array([[0] * common.SLOTS])
                heard.append(cur)
                changes = 0
                rng = np.random.default_rng(7)
                for _ in range(bars):
                    a = np.asarray(sample_bar(brain, heard_observation(heard, common.mood_rows([[0.5, 0.5]], self.style, bars=[pos])), cur[0], t, rng)).reshape(1, -1)
                    pos += 1
                    changes += int((common.sounds(a[0]) != common.sounds(cur[0])).sum()); cur = a
                    keep_heard(heard).append(cur)
                scan.append((t, changes / (bars * L)))
            brain.reset()
        best = min(scan, key=lambda r: abs(r[1] - target))[0]
        print(f"calibrated {card['name']}: T={best} (target change {target:.3f}; scan {[(t, round(r, 3)) for t, r in scan]})", flush=True)
        return best

    def unload(self):
        with self.lock:
            self.brain, self.card, self.bar, self.view_layout = None, None, 0, None
            self.history.clear()

    def feedback(self, rating, playing_bar):
        """A heart (+1) or cross (-1) for the bar playing: reward for the next moment, and a log row with the
        bars it rates (the last WINDOW answered up to the playing bar) for later training rounds."""
        rating = 1.0 if rating > 0 else -1.0
        with self.lock:
            self.pending = max(-1.0, min(1.0, self.pending + rating))
            rows = [h for h in self.history if playing_bar is None or h["bar"] <= playing_bar][-WINDOW:]
            card = self.card or {}
            row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "rating": rating, "checkpoint": card.get("name", "demo"),
                   "layout": common.layout_key(card), "style": self.style,
                   "life_bar": self.bar, "playing_bar": playing_bar, "mood": self.mood,
                   "tokens_per_layer": card.get("tokens_per_layer"), "mod_levels": card.get("mod_levels"), "bars": rows}
        FEEDBACK.parent.mkdir(exist_ok=True)
        with open(FEEDBACK, "a") as f:
            f.write(json.dumps(row) + "\n")
        return {"ok": True, "logged_bars": len(rows), "pending": self.pending}

    def next(self, mood, reward):
        with LAYOUT, self.lock:
            if self.brain is not None:
                self.ensure_layout()
            if self.brain is None:
                bar = self.demo[self.bar % len(self.demo)]
                tokens, mods = (bar.get("tokens"), bar.get("mods") or [[0, 0]] * L) if isinstance(bar, dict) else (bar, [[0, 0]] * L)
                if self.pool_name() == self.demo_pool:
                    # The chip check walks through the fitted sounds with every chip playing from the first bar:
                    # a new bass and lead sound every four bars, a new pad every eight, a new kit every sixteen.
                    step = self.bar // 4
                    tokens = [1 + (self.bar // 16) % 8, 1 + step % 64, 1 + (step * 5 + 20) % 64, 1 + (step // 2 * 7 + 40) % 64]
                self.bar += 1
                self.mood = mood
                self.history.append({"bar": self.bar, "tokens": tokens, "mods": mods, "mood": mood})
                return {"tokens": tokens, "mods": mods, "bar": self.bar, "mode": "demo", "arousal": None}
            common.ENTRY_SCALE, common.LINE_SCALE = reading(self.card)      # each brain (player, decks) read its own way
            common.SYNTH_SCALE = play_reading(self.card).get("synth_line_scale")   # the synth lines alone, beside line_scale
            common.KICK_SCALE = play_reading(self.card).get("kick_scale")           # the kick row alone, beside line_scale
            heard = self.heard if self.heard and list(np.asarray(self.heard[-1]).reshape(-1)) == list(self.state) else [np.array([self.state])]
            x = heard_observation(heard, common.mood_rows([mood], self.style, bars=[getattr(self, "bar_pos", 0)]))
            self.bar_pos = getattr(self, "bar_pos", 0) + 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.play_mode == "live":
                    r = max(-1.0, min(1.0, float(reward) + self.pending))
                    self.pending = 0.0
                    a = self.brain.live(x, reward=None if self.first else [r])
                else:
                    self.pending = 0.0
                    a = sample_bar(self.brain, x, self.state, self.temperature, self.rng)   # coherent changes at the calibrated temperature
                view = brainview.snapshot(self.brain, x, neurons=time.time() - self.want3d < 30) if self.with_view else None
            self.first = False
            self.mood = mood
            proposed = np.asarray(a).reshape(common.SLOTS)
            if self.groove is not None and self.play_mode == "imitate":
                # the voice dwell decides what plays; the busy-kick cap then applies to that bar too, and the brain hears
                # the capped bar (26 Nov: the cap was skipped here, the only branch the owner's settings use)
                self.row_caps = play_reading(self.card).get("row_caps")         # every drum row's cap (_cap_rows)
                self.line_caps = play_reading(self.card).get("line_caps")       # every synth layer's cap (_cap_lines)
                self.cap_mode = play_reading(self.card).get("cap_mode")         # "thin": the caps thin in place
                self.kick_cap = (self.row_caps or [None])[0] or play_reading(self.card).get("kick_cap")
                kicked = getattr(self, "kick_capped", 0)
                # and the kickless rescue (_keep_kick, the 11-23 hotfix) runs here too: under the dwell it never ran, so a
                # brain could go 81 bars without a kick (26 Nov, P8)
                self.state = self._keep_kick(self._cap_lines(self._cap_rows(self._cap_kick(self.groove.play(self.state, proposed, self.bar_pos - 1).tolist()))))
            else:
                previous = self.state
                self.row_caps = play_reading(self.card).get("row_caps")         # every drum row's cap (_cap_rows)
                self.line_caps = play_reading(self.card).get("line_caps")       # every synth layer's cap (_cap_lines)
                self.cap_mode = play_reading(self.card).get("cap_mode")         # "thin": the caps thin in place
                self.kick_cap = (self.row_caps or [None])[0] or play_reading(self.card).get("kick_cap")   # the busy-kick hotfix
                kicked = getattr(self, "kick_capped", 0)
                self.state = self._keep_kick(self._cap_lines(self._cap_rows(self._cap_kick([int(t) for t in proposed]))))
                if self.groove is not None:
                    self.groove.hear(previous, self.state)
            self.heard = keep_heard(heard + [np.array([self.state])])      # the bar played is the bar heard next
            self.bar += 1
            self.history.append({"bar": self.bar, "state": self.state, "position": self.bar_pos - 1, "mood": mood})
            info = (self.brain.last_arousal or {}) if self.play_mode == "live" else {"mode": "imitating"}
            if self.play_mode == "live" and self.bar % SAVE_EVERY == 0:
                LIVES.mkdir(exist_ok=True)
                self.brain.save(LIVES / f"{self.card['name']}.life.npz")
            score = common.score_bar(self.state, self.pool, self.held, self.held_tok)
            if self.row_caps or self.line_caps:  # the bars the row and line caps played (the listening worker counts them)
                capped = (["kick"] if self.row_caps and getattr(self, "kick_capped", 0) > kicked else []) + self.capped_now
                if self.row_caps:
                    score = {**score, "row_capped": {**getattr(self, "row_capped", {}), "kick": getattr(self, "kick_capped", 0)}}
                if self.line_caps:
                    score = {**score, "line_capped": dict(getattr(self, "line_capped", {}))}
                    capped = capped + self.lines_capped_now
                score = {**score, "capped": capped}
            return {**score, "bar": self.bar, "mode": "brain", "checkpoint": self.card["name"], "view": view, "kick_fixes": self.kick_fixes,
                    "arousal": {k: (float(info[k]) if isinstance(info.get(k), (int, float, np.floating)) else info.get(k))
                                for k in ("mode", "level", "want", "surprise") if k in info}}


KICK_WAIT = 32      # bars without an audible kick before the player's kick hotfix steps in (Creature._keep_kick)


def heard_observation(heard, mood):
    """The observation of the bar heard now, heard[-1] (an array or list of shape (1, SLOTS)), for the brain whose layout
    is configured: observe() for a plain brain; for an echo brain (card "echo_lags", common.ECHO_LAGS after shape_from)
    common.lagged_observation over the bars heard so far, oldest first, as train.py builds it: a lag reaching before
    the first bar heard is an all-zero block."""
    if getattr(common, "ECHO_LAGS", ()):
        return common.lagged_observation(heard, mood)
    return observe(heard[-1], mood)


def keep_heard(heard):
    """The bars a brain still needs to hear back: the longest echo lag (one bar for a plain brain)."""
    keep = max(getattr(common, "ECHO_LAGS", ()) or (1,))
    del heard[:-keep]
    return heard


def sample_bar(brain, x, state, temperature, rng):
    """One bar from the settled brain with coherent changes (common.sample_slots): the brain settles once (greedy
    act, which advances its stream and memory); every slot is read at its own temperature (one number per slot
    from common.calibrate_slots, or one for all), and one shared draw per group decides which slots change."""
    common.answer(brain, x)
    return np.array([common.sample_slots(common.motor_activity(brain)[0], state, temperature, rng)])


CREATURE = Creature()


SEEDS: dict = {}                      # (corpus, layout) -> passages; built once per corpus, kept in corpus/.seeds/
_SEED_BUILDING: set = set()


def _seed_key(card) -> str:
    import hashlib
    lay = json.dumps([card.get("streams"), bool(card.get("steps")), card.get("beat_rows"), card.get("clock", False), card.get("synth_rows"), card.get("styles")] + ([True] if card.get("synth_fill") else []) + (["onoff"] if card.get("onoff") else []), sort_keys=True)
    if card.get("patch_book"):
        lay += common.layout_key(card)
    lay += "|groups"      # 26 Nov: the passages carry their group (acid start), so older caches are rebuilt
    if card.get("section") or card.get("lookahead"):      # score v5f: a section slot after the plan slots
        lay += f"|section={bool(card.get('section'))}|lookahead={bool(card.get('lookahead'))}"
    if card.get("echo_lags"):                             # the echo shape: the passages are read with its switches
        lay += f"|echo={list(card.get('echo_lags'))}"
    return f"{(card.get('streams') or 'streams').replace('.json', '')}_{hashlib.sha256(lay.encode()).hexdigest()[:10]}"


def seed_passages(card, n=8):
    """The cached starting passages of this brain's corpus and layout, or None while they are built (in the
    background, once: parsing the corpus takes about 90 s)."""
    key = _seed_key(card)
    if key in SEEDS:
        return SEEDS[key]
    f = ROOT / "corpus" / ".seeds" / f"{key}.json"
    if f.exists():
        try:
            SEEDS[key] = json.loads(f.read_text())
            return SEEDS[key]
        except ValueError:
            pass
    if key not in _SEED_BUILDING:
        _SEED_BUILDING.add(key)
        threading.Thread(target=_build_seeds, args=(dict(card), key, n), daemon=True).start()
    return None


def _build_seeds(card, key, n):
    """Parse the corpus once in a child process (its own layout state) and keep up to 120 passages per style."""
    try:
        name = card.get("streams") or ("streams_r2_drums.json" if card.get("steps") else "streams.json")
        path = ROOT / "corpus" / name
        if not path.exists():
            return
        if card.get("streams_sha"):
            import hashlib
            if hashlib.sha256(path.read_bytes()).hexdigest() != card["streams_sha"]:
                return
        out = ROOT / "corpus" / ".seeds" / f"{key}.json"; out.parent.mkdir(exist_ok=True)
        spec = ROOT / "corpus" / ".seeds" / f"{key}.card.json"
        # the shape switches travel with the card (reserved inputs, SID blocks, ear): without them a SID brain's
        # starting passages have the plain shape's slots and no chip settings, and the brain cannot be primed
        spec.write_text(json.dumps({k: card.get(k) for k in ("streams", "steps", "beat_rows", "clock", "synth_rows", "synth_fill", "onoff", "patch_book", "styles", "tokens_per_layer", "mod_levels",
                                                               "reserve", "sid_live", "sid_pool", "ear", "hear_parts", "lookahead", "section", "echo_lags")}))
        code = (
            "import json, os, sys; import numpy as np; sys.path.insert(0, sys.argv[4]); import common, timbre\n"
            "c = json.load(open(sys.argv[2])); n = int(sys.argv[5]); common.SYNTH_FILL = bool(c.get('synth_fill')); common.ONOFF = bool(c.get('onoff'))\n"
            "if hasattr(common, 'shape_from'): common.shape_from(c)\n"
            "more = {'sid_book': timbre.sid_book(json.load(open(os.path.join(sys.argv[6], c['sid_pool']))))} if c.get('sid_live') and c.get('sid_pool') else {}\n"
            "common.configure(c['tokens_per_layer'], c['mod_levels'], steps=c.get('steps', False), styles=c.get('styles'), rows=c.get('beat_rows'), clock=c.get('clock', False), synth=c.get('synth_rows'), patch_book=c.get('patch_book'))\n"
            "tr = [t for t in common.load_streams(sys.argv[1], steps=c.get('steps', False), rows=c.get('beat_rows'), clock=c.get('clock', False), synth=c.get('synth_rows'), patch_book=c.get('patch_book'), **more) if not t['held_out'] and len(t['bars']) > n + 2]\n"
            "rng = np.random.default_rng(0); per = {}; acid = []\n"
            "for i in rng.permutation(len(tr)):\n"
            "    t = tr[i]; st = t.get('style'); g = t.get('group')\n"
            "    full = len(per.get(st, [])) >= 120\n"
            "    if full and (g != 'acid' or len(acid) >= 40): continue\n"
            "    b = np.asarray(t['bars']); beat = common.sounds(b)[:, 0] > 0; ok = common.simple_beat_bars(b)\n"
            "    s = [k for k in range(0, len(b) - n - 1) if beat[k:k + n].all() and ok[k:k + n].sum() >= n - 1]\n"
            "    for k in ([s[j] for j in rng.permutation(len(s))[:3]] if s else []):\n"
            "        e = [b[k:k + n].tolist(), [list(map(float, m)) for m in t['mood'][k:k + n + 1]], t['title'], int(k), st, g]\n"
            "        (acid if full else per.setdefault(st, [])).append(e)\n"
            "json.dump([p for v in per.values() for p in v] + acid, open(sys.argv[3], 'w'))\n"
        )
        import subprocess
        r = subprocess.run([sys.executable, "-c", code, str(path), str(spec), str(out), str(ROOT / "brain"), str(n), str(ROOT / "pool")], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(CADENCE_SRC)})
        if r.returncode == 0 and out.exists():
            SEEDS[key] = json.loads(out.read_text())
            print(f"starting passages ready: {name} ({len(SEEDS[key])})", flush=True)
        else:
            print(f"starting passages of {name} failed: {(r.stderr or r.stdout)[-300:]}", flush=True)
    finally:
        _SEED_BUILDING.discard(key)


MIX = {"incoming": None, "bar": 0}
PLAYER_LOAD = threading.Lock()
PLAYER_REQUEST = None


def player_loading():
    return PLAYER_REQUEST or CREATURE.loading


def remember_selection(name):
    try:
        temporary = PLAYER_SELECTION.with_suffix(".part")
        temporary.write_text(name or "demo")
        temporary.replace(PLAYER_SELECTION)
    except OSError as error:
        print(f"could not save player selection: {error}", flush=True)


def select_checkpoint(body):
    global PLAYER_REQUEST
    if CREATURE.loading or not PLAYER_LOAD.acquire(blocking=False):
        # the page joins a load of the brain it asked for and asks again for another one (main.js switchTo)
        return 409, {"error": f"Still preparing {player_loading() or 'the current checkpoint'}.", "loading": player_loading(),
                     "progress": CREATURE.progress.read()}
    try:
        name = body.get("name")
        if body.get("candidate"):
            # A listen from Stats: the newest checkpoint of that run, fetched when the player does not have it (or
            # the run's listening state, site/listen-states.json)
            state = listen_state(body["candidate"])
            PLAYER_REQUEST = f"{body['candidate']} · {f'its listening state {state}' if state else 'fetching its newest checkpoint'}"
            name = audition_checkpoint(body["candidate"], CREATURE.progress)
        PLAYER_REQUEST = name or "Instrument check"
        if name in ("latest", "best"):
            cancel_mix()
            CREATURE.follow_latest = True
            newest = default_checkpoint()
            CREATURE.load(newest) if newest else CREATURE.unload()
        elif name in (None, "", "demo", "demo:sid"):
            with LAYOUT:
                cancel_mix()
                CREATURE.follow_latest = False
                CREATURE.unload()
                CREATURE.demo_pool = SID_POOL if name == "demo:sid" else None
        elif name != (CREATURE.card or {}).get("name") or MIX["incoming"] is not None:
            CREATURE.follow_latest = False
            if body.get("mix"):
                how = start_mix(name)
                remember_selection(name)
                return 200, {"active": name, "pool": CREATURE.pool_name(), "how": how}
            cancel_mix()
            CREATURE.load(name)
        active = (CREATURE.card or {}).get("name")
        remember_selection(active)
        return 200, {"active": active, "pool": CREATURE.pool_name(), "ready": True}
    except Exception as e:
        CREATURE.progress.end(f"{type(e).__name__}: {e}")          # a failed fetch ends its record (a load ends its own)
        raise
    finally:
        CREATURE.progress.end()                                    # nothing was loaded after the listing: the record closes
        PLAYER_REQUEST = None
        PLAYER_LOAD.release()
# a DJ handover over 12 bars in steps of 4 (owner 11-23): which brain plays each layer (beat, bass, lead, air)
MIX_PLAN = [("old", "old", "old", "new"), ("old", "old", "new", "new"), ("old", "new", "new", "new")]
MIX_STEP = 4


def cancel_mix():
    with LAYOUT:
        MIX.update(incoming=None, bar=0)


def start_mix(name):
    """Load `name` as a second brain beside the one that plays; /api/next then hands over layer by layer. Falls back
    to a plain switch when the two brains play different pools (their sounds are numbered differently)."""
    global CREATURE
    old = CREATURE
    if old.brain is None or MIX["incoming"] is not None:
        cancel_mix()
        old.load(name); return "switch"
    new = Creature()
    new.play_mode, new.style, new.follow_latest = old.play_mode, old.style, False
    new.progress = old.progress                 # the incoming brain's load shows as the player's
    new.last_request = old.last_request
    card = next((c for c in cards() if c["name"] == name), None)
    if card is None or card.get("pool", "pool.json") != (old.card or {}).get("pool", "pool.json"):
        old.load(name); return "switch"
    new.load(name)                              # the old brain keeps playing meanwhile
    MIX.update(incoming=new, bar=0)
    return "mix"


def mixed_next(mood, reward):
    """One bar of the handover: both brains compose (each on its own path), the plan picks each layer's source."""
    global CREATURE
    new = MIX["incoming"]
    a = CREATURE.next(mood, reward)
    b = new.next(mood, reward)
    step = MIX["bar"] // MIX_STEP
    MIX["bar"] += 1
    if step >= len(MIX_PLAN):                   # handed over: the new brain plays alone from here
        new.last_request = CREATURE.last_request
        CREATURE = new
        MIX.update(incoming=None, bar=0)
        b["mix"] = "done"
        return b
    plan = MIX_PLAN[step]
    out = dict(b)
    out["tokens"] = [(a if plan[l] == "old" else b)["tokens"][l] for l in range(L)]
    out["mods"] = [(a if plan[l] == "old" else b)["mods"][l] for l in range(L)]
    out["patterns"] = [((a if plan[l] == "old" else b).get("patterns") or [None] * L)[l] for l in range(L)]
    out["patches"] = [((a if plan[l] == "old" else b).get("patches") or [None] * L)[l] for l in range(L)]
    out["mix"] = {"from": a.get("checkpoint"), "to": b.get("checkpoint"), "bar": MIX["bar"], "of": MIX_STEP * len(MIX_PLAN), "layers": list(plan)}
    return out


# ---------- the mixer view: three decks, each its own brain and composition, beside the main player ----------
DECKS = 3
_DECKS = [None] * DECKS
DECK_IDLE = 900.0       # a deck nobody asked for a bar for this long gives its brain's memory back


def deck(i):
    i = int(i)
    if not 0 <= i < DECKS:
        raise ValueError(f"deck is 0..{DECKS - 1}")
    if _DECKS[i] is None:
        c = Creature()
        c.follow_latest, c.with_view = False, False
        c.progress = LOAD_PROGRESS[f"deck{i}"]
        _DECKS[i] = c
    return _DECKS[i]


def decks_status():
    out = []
    for i in range(DECKS):
        c = _DECKS[i]
        out.append({"deck": i, "active": (c.card or {}).get("name") if c else None, "loading": c.loading if c else None,
                    "pool": c.pool_name() if c and c.brain is not None else None, "progress": LOAD_PROGRESS[f"deck{i}"].read()})
    return out


def deck_idle():
    """Unload the brain of a deck that has been idle for DECK_IDLE seconds (the mixer page closed or stopped)."""
    while True:
        time.sleep(60)
        for c in _DECKS:
            try:
                if c is not None and c.brain is not None and not c.loading and time.time() - max(c.last_request, getattr(c, "loaded_at", 0)) > DECK_IDLE:
                    c.unload()
            except Exception as e:
                print(f"deck idle: {e}", flush=True)


# ---------- the Auto DJ's track factory: one brain composes whole tracks ahead, the mixer's decks play them ----------
DJ_MAX_BARS = 352       # a track's longest life in a mix: coming in, playing alone, going out
DJ_IDLE = 600.0         # the factory's brain gives its memory back when nobody asked for a track for this long
TRACKS = SITE / "tracks"   # finished tracks, kept per checkpoint: a set starts at once and a late track has a stand-in
REPERTOIRE = SITE / "repertoire"   # the sections the owner kept (site/curator.py): the Auto DJ's repertoire, also in the bucket
_REPERTOIRE_PULL = [0.0]


def repertoire():
    """The kept sections, read from disk; the bucket's copy is fetched at most every ten minutes, in the background."""
    if time.time() - _REPERTOIRE_PULL[0] > 600:
        _REPERTOIRE_PULL[0] = time.time()

        def pull():
            import shutil
            import subprocess
            if not BUCKET:                               # no bucket named on this computer: the local folder is all
                return
            try:
                REPERTOIRE.mkdir(exist_ok=True)
                subprocess.run([shutil.which("aws") or "/usr/local/bin/aws", "s3", "sync", f"{BUCKET}/repertoire/", str(REPERTOIRE),
                                "--exclude", "*", "--include", "r*.json", "--only-show-errors"], capture_output=True, timeout=300)
            except Exception as e:
                print(f"repertoire: {e}", flush=True)
        threading.Thread(target=pull, daemon=True).start()
    out = []
    for f in sorted(REPERTOIRE.glob("r*.json")) if REPERTOIRE.exists() else []:
        try:
            item = json.loads(f.read_bytes())
        except (OSError, ValueError):
            continue
        if isinstance(item.get("bars"), list) and len(item["bars"]) >= 8 and item.get("pool"):
            out.append(item)
    return out
DJ_KEPT = 24            # tracks kept per checkpoint, the newest
DJ_LEADER_MINUTES = 120.0   # training time before a run can be the ranking leader for the Auto DJ


DJ_APPROVED = Path(os.environ.get("CADANCE_DJ_APPROVED") or SITE / "dj-approved.json")     # the brains the Auto DJ may compose with (a list of model ids), set in the mixer; a test instance names its own copy


def dj_approved(change=None):
    """The approved brains; change = (model id, on) adds or removes one. An empty list means the ranking leader."""
    with BRAINS_LOCK:
        try:
            found = [str(x) for x in json.loads(DJ_APPROVED.read_text()) if isinstance(x, str)]
        except (OSError, ValueError):
            found = ["leader"]
        if change is not None:
            model, on = str(change[0]), bool(change[1])
            if not re.fullmatch(r"leader|(candidate|checkpoint):[A-Za-z0-9_]{1,80}", model):
                raise ValueError("unknown brain")
            found = [x for x in found if x != model] + ([model] if on else [])
            DJ_APPROVED.write_text(json.dumps(found, indent=1))
        return found or ["leader"]


DJ_PASSED = Path(__file__).resolve().parent / "dj-passed.json"


def dj_models():
    """What the Auto DJ may play: the run that leads the ranking (lowest training error, as Stats orders it),
    every run with a curve by its newest checkpoint, and the checkpoints kept by hand."""
    brains = []
    try:
        with _CURVES_LOCK:
            brains = list(_CURVES.get("brains") or [])
    except Exception:
        pass
    rows = []
    for b in brains:
        if b.get("kind") != "composer":
            continue
        points = [pt for pt in b.get("points") or []
                  if isinstance(pt.get("train"), (int, float)) and isinstance(pt.get("test"), (int, float))]
        lows = [pt["train"] for pt in points]
        # A brain leads only after two hours of lessons: a newborn on an easier score reads low and plays thin.
        trained = max((float(pt.get("minutes") or 0) for pt in points), default=0.0)
        if lows and trained >= DJ_LEADER_MINUTES:
            rows.append((min(lows), b["candidate"]))
    rows.sort()
    states = listen_states()
    models = [{"id": f"candidate:{c}", "label": f"{c} ({f'plays {states[c]}' if c in states else 'newest checkpoint'})", "train": low}
              for low, c in rows]
    # A run leads for the Auto DJ only when it has passed the free-play probes and the listening loop (PLAN.md,
    # "Selection", point 5): site/dj-passed.json lists the run names or patterns that have. Errors of runs on
    # different scores do not compare, so the lowest number alone must not pick the brain a set is composed with.
    try:
        passed = [str(x) for x in json.loads(DJ_PASSED.read_text())] if DJ_PASSED.is_file() else None
    except Exception:
        passed = None
    trained = len(rows)
    if passed is not None:
        rows = [(low, c) for low, c in rows if any(fnmatch.fnmatch(c, pat) for pat in passed)]
    for card, where in _available_checkpoints():
        if where == "pinned":
            models.append({"id": f"checkpoint:{card['name']}", "label": f"{card['name']} (kept)"})
    # why there is no leader, in the words the mixer shows
    gate = (None if rows else "no brain has passed the play checks yet" if passed is not None and (not passed or trained)
            else "no run has trained two hours yet")
    out = {"leader": rows[0][1] if rows else None, "models": models, "repertoire": len(repertoire()), "approved": dj_approved(),
           "passed": passed, "leader_gate": gate}
    if states:
        out["listen_states"] = dict(states)  # the runs that play as a kept state (the mixer names it in place of "newest")
    return out


class TrackFactory:
    """One brain composes whole tracks ahead of time, one after the other, each a composition of its own.

    Three decks with a live brain each hold three brains in memory. A track composed ahead needs one brain for
    all decks, and a deck plays its track without asking the server bar by bar."""
    def __init__(self):
        self.creature = None
        self.tracks = {}            # id -> the track: model, bars composed so far, want, done, error, pool, seed
        self.order = []             # the ids still to compose, oldest first
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.serial = 0
        self.last = time.time()
        self.thread = None
        self._state = "idle"
        self.served = []            # the kept tracks handed out, oldest first
        self.played = []            # the repertoire's sections handed out, oldest first
        self.current = None         # the approved brain that composed last

    @property
    def state(self):
        """What the factory does; while it finds or loads a brain, with the load's stage and percent (LoadProgress):
        the Auto DJ's readout shows this text."""
        raw = self._state
        if raw.startswith(("finding ", "loading ")):
            r = LOAD_PROGRESS["dj"].read()
            if r is not None and not r["finished"]:
                left = left_words(r["left"]) if r["percent"] < 99 else ""
                return f"{raw} · {r['text']} · {r['percent']:.0f}%" + (f" · {left}" if left else "")
        return raw

    @state.setter
    def state(self, value):
        self._state = value

    def kept(self, model, bars):
        """A finished track of this model from disk: the one served longest ago that is long enough."""
        kind, _, value = str(model).partition(":")
        if kind == "leader":
            kind, value = "candidate", dj_models()["leader"]
        if not value or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", value) or not TRACKS.exists():
            return None
        # a run's checkpoints (_e...) and its rolling states (_l<stamp>): the tracks of a run that is still training are kept
        # under the rolling name, so they count as the run's tracks too
        state = listen_state(value) if kind != "checkpoint" else None     # a run that plays as its listening state:
        folders = [TRACKS / value] if kind == "checkpoint" else [TRACKS / state] if state else sorted(  # its tracks only
            [*TRACKS.glob(f"{value}_e*"), *TRACKS.glob(f"{value}_l[0-9]*")], key=lambda d: d.stat().st_mtime, reverse=True)
        for folder in folders[:4]:                     # the newest checkpoint first, then older ones of the same run
            files = sorted(folder.glob("*.json"), key=lambda f: (self.served.index(str(f)) if str(f) in self.served else -1, -f.stat().st_mtime))
            for f in files:
                try:
                    data = json.loads(f.read_bytes())
                except (OSError, ValueError):
                    continue
                if len(data.get("bars") or []) >= 0.75 * bars:
                    self.served = [x for x in self.served if x != str(f)] + [str(f)]
                    return data
        return None

    def section(self, bars):
        """A kept section of the repertoire as a track: the one played longest ago, repeated to the length asked for."""
        found = repertoire()
        if not found:
            return None
        # not the last few played; among the others by chance, a piece with more stars more often (no stars counts as two)
        recent = set(self.played[-min(3, max(0, len(found) - 1)):]) if len(found) > 1 else set()
        fresh = [i for i in found if i["id"] not in recent] or found
        weights = np.array([float((int(i.get("stars") or 0) or 2) ** 2) for i in fresh])
        item = fresh[int(np.random.default_rng().choice(len(fresh), p=weights / weights.sum()))]
        self.played = [x for x in self.played if x != item["id"]] + [item["id"]]
        once = item["bars"]
        rounds = max(1, -(-bars // len(once)))
        return {"bars": [dict(b, bar=k + 1) for k, b in enumerate(once * rounds)], "pool": item["pool"], "bpm": item.get("bpm"),
                "seed": item.get("seed"), "checkpoint": item.get("checkpoint"), "name": item.get("name"), "layout": item.get("layout"),
                "stars": item.get("stars")}

    def request(self, model, bars, mood, quick=False, fallback=None, session=None, kept_only=False):
        bars = int(min(max(int(bars), 16), DJ_MAX_BARS))
        self.forget()
        base = [min(max(float(v), 0.0), 1.0) for v in (mood or [0.45, 0.4])][:2]
        if str(model) == "approved":               # one of the approved brains: mostly the one that composed last (a change reloads)
            allowed = dj_approved()
            if self.current not in allowed or (len(allowed) > 1 and np.random.default_rng().random() < 0.35):
                others = [m for m in allowed if m != self.current] or allowed
                self.current = others[int(np.random.default_rng().integers(len(others)))]
            model = self.current
        if fallback == "approved":
            fallback = self.current or dj_approved()[0]
        if str(model) == "repertoire":
            data = self.section(bars)
            if data is None:                       # an empty repertoire: the brain composes instead
                model, quick = fallback or "leader", True
                data = self.kept(model, bars)
        else:
            data = self.kept(model, bars) if quick else None
        if data is None and kept_only:              # a finished track only (the Auto DJ's stand-in): none to compose in its place
            return None
        if data is not None:
            with self.lock:
                self.serial += 1
                tid = f"t{self.serial}"
                self.tracks[tid] = {"id": tid, "model": str(model), "want": len(data["bars"]), "mood": base, "bars": data["bars"], "done": True,
                                    "error": None, "pool": data.get("pool"), "bpm": data.get("bpm"), "seed": data.get("seed"),
                                    "checkpoint": data.get("checkpoint"), "kept": True, "name": data.get("name"), "layout": data.get("layout"),
                                    "stars": data.get("stars"), "session": session, "touched": time.time()}
                self.last = time.time()
            return tid
        with self.lock:
            self.serial += 1
            tid = f"t{self.serial}"
            self.tracks[tid] = {"id": tid, "model": str(model), "want": bars, "mood": base, "bars": [], "done": False,
                                "error": None, "pool": None, "bpm": None, "seed": None, "checkpoint": None,
                                "session": session, "touched": time.time()}
            self.order.append(tid)
            self.last = time.time()
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._work, daemon=True)
                self.thread.start()
        self.wake.set()
        return tid

    def read(self, tid, start):
        with self.lock:
            self.last = time.time()
            t = self.tracks.get(tid)
            if t is None:
                return {"error": "no such track"}
            t["touched"] = time.time()
            start = max(0, int(start))
            return {"id": tid, "from": start, "bars": t["bars"][start:start + 64], "total": len(t["bars"]), "want": t["want"],
                    "done": t["done"], "error": t["error"], "pool": t["pool"], "bpm": t["bpm"], "seed": t["seed"],
                    "checkpoint": t["checkpoint"], "state": self.state, "kept": bool(t.get("kept")), "name": t.get("name"), "stars": t.get("stars"),
                    "layout": t.get("layout") if start == 0 else None,
                    "progress": LOAD_PROGRESS["dj"].read() if LOAD_PROGRESS["dj"].active() else None}

    def keep(self, ids, session=None):
        """A page names the tracks it still plays or waits for; its other tracks are forgotten. Every open mixer
        page is a session of its own: one page never takes another page's tracks away."""
        with self.lock:
            self.last = time.time()
            for tid in [k for k, t in self.tracks.items() if t.get("session") == session and k not in ids]:
                del self.tracks[tid]
            self.order = [k for k in self.order if k in self.tracks]

    def stop(self, session=None):
        with self.lock:
            for tid in [k for k, t in self.tracks.items() if t.get("session") == session]:
                del self.tracks[tid]
            self.order = [k for k in self.order if k in self.tracks]
        self.wake.set()

    def forget(self):
        """Tracks no page has read for half an hour (a closed tab) are dropped."""
        with self.lock:
            old = time.time() - 1800
            for tid in [k for k, t in self.tracks.items() if t.get("touched", 0) < old]:
                del self.tracks[tid]
            self.order = [k for k in self.order if k in self.tracks]

    def status(self):
        with self.lock:
            return {"state": self.state, "tracks": [{"id": t["id"], "total": len(t["bars"]), "want": t["want"], "done": t["done"],
                                                     "error": t["error"]} for t in self.tracks.values()],
                    "active": (self.creature.card or {}).get("name") if self.creature and self.creature.brain is not None else None,
                    "progress": LOAD_PROGRESS["dj"].read()}

    def _resolve(self, model):
        kind, _, value = model.partition(":")
        if kind == "checkpoint":
            return value
        if kind == "leader":
            models = dj_models()
            value = models["leader"]
            if not value:
                raise ValueError(f"{models['leader_gate'] or 'no leader'}: pick a run in the brain field")
        return audition_checkpoint(value, LOAD_PROGRESS["dj"])     # the run's newest checkpoint, fetched when the player lacks it

    def _work(self):
        while True:
            with self.lock:
                mine = [k for k in self.order if k in self.tracks]    # a listener's track comes before the repertoire job's
                tid = next((k for k in mine if self.tracks[k].get("session") != "curate-auto"), mine[0] if mine else None)
                idle = time.time() - self.last
            if tid is None:
                self.state = "idle"
                if self.creature is not None and self.creature.brain is not None and idle > DJ_IDLE:
                    try:
                        self.creature.unload()
                    except Exception as e:
                        print(f"auto dj: unload: {e}", flush=True)
                self.wake.wait(30.0)
                self.wake.clear()
                continue
            try:
                self._compose(tid)
            except Exception as e:
                print(f"auto dj: track {tid}: {type(e).__name__}: {e}", flush=True)
                with self.lock:
                    if tid in self.tracks:
                        self.tracks[tid]["error"] = f"{type(e).__name__}: {e}"
            with self.lock:
                if tid in self.tracks:
                    self.tracks[tid]["done"] = True
                self.order = [k for k in self.order if k != tid]

    def _compose(self, tid):
        with self.lock:
            t = self.tracks.get(tid)
            if t is None:
                return
            model, want, base = t["model"], t["want"], t["mood"]
            background = t.get("session") == "curate-auto"      # the repertoire job: composed at a third of the time, the Mac stays quiet
        self.state = f"finding {short_brain(model)}"
        try:
            name = self._resolve(model)
        except Exception as e:
            LOAD_PROGRESS["dj"].end(f"{type(e).__name__}: {e}")
            raise
        if self.creature is None:
            c = Creature()
            c.follow_latest, c.with_view = False, True      # the brain's view per bar: the mixer's visuals show it
            c.progress = LOAD_PROGRESS["dj"]
            self.creature = c
        c = self.creature
        if c.brain is None or (c.card or {}).get("name") != name:
            self.state = f"loading {short_brain(name)}"
            c.load(name)
        else:
            LOAD_PROGRESS["dj"].end()                # a listing found the brain that is loaded: no load follows
            c.renew()                                # a new composition from another passage
        pool_name = c.pool_name()
        bpm = None
        try:
            bpm = json.loads((ROOT / "pool" / pool_name).read_bytes()).get("bpm")
        except (OSError, ValueError):
            pass
        with self.lock:
            if tid not in self.tracks:
                return
            self.tracks[tid].update(pool=pool_name, bpm=bpm, seed=getattr(c, "seed_title", None), checkpoint=name,
                                    layout=getattr(c, "view_layout", None))
        self.state = f"composing {tid}"
        rng = np.random.default_rng()
        lift, drift = float(rng.uniform(0.08, 0.2)), float(rng.uniform(-0.1, 0.1))
        for k in range(want):
            arc = np.sin(np.pi * k / max(1, want - 1))        # a track rises toward its middle and comes down again
            mood = [float(np.clip(base[0] + lift * (arc - 0.5), 0.0, 1.0)), float(np.clip(base[1] + drift * (k / want - 0.5), 0.0, 1.0))]
            started = time.time()
            reply = c.next(mood, 0.0)
            took = time.time() - started
            bar = {"tokens": reply.get("tokens"), "mods": reply.get("mods"), "patterns": reply.get("patterns"),
                   "patches": reply.get("patches"), "bar": k + 1, "view": reply.get("view")}
            bar.update({key: reply[key] for key in ("capped", "row_capped", "line_capped") if key in reply})   # the caps' bars
            with self.lock:
                if tid not in self.tracks:           # the page stopped or dropped the track
                    return
                self.tracks[tid]["bars"].append(bar)
                waiting = any(not x["done"] and x.get("session") != "curate-auto" for x in self.tracks.values())
                # a track whose page is gone (a reload or a closed tab: no read for a minute) gives the brain up as soon as
                # another page's set waits (owner, 26 Nov 10:12: the brain finished an abandoned track first)
                me = self.tracks[tid]
                orphan = (me.get("session") != "curate-auto" and time.time() - me.get("touched", 0) > 60
                          and any(k != tid and not x["done"] and x.get("session") != me.get("session") for k, x in self.tracks.items()))
                if orphan:
                    del self.tracks[tid]
                    self.order = [k for k in self.order if k != tid]
                    print(f"auto dj: {tid} dropped: no page read it for a minute and another set waits", flush=True)
                    return
            if background and waiting:               # a listener's set waits for the brain: the repertoire job's track is given up
                raise RuntimeError("set aside: a listener's set needs the brain")
            # the player and the decks get the layout in between; a background track rests two thirds of the time
            time.sleep(min(6.0, 2.0 * took) if background else 0.005)
        with self.lock:
            done = dict(self.tracks[tid]) if tid in self.tracks else None
        if done and len(done["bars"]) >= 64:         # keep the finished track
            try:
                folder = TRACKS / name
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{time.strftime('%Y%m%dT%H%M%S')}_{len(done['bars'])}.json"
                part = target.with_suffix(".part")
                part.write_text(json.dumps({"checkpoint": name, "pool": pool_name, "bpm": bpm, "seed": done["seed"], "bars": done["bars"],
                                            "layout": done.get("layout"),
                                            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, separators=(",", ":")))
                part.replace(target)
                for old in sorted(folder.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[DJ_KEPT:]:
                    old.unlink()
            except OSError as e:
                print(f"auto dj: keeping {tid}: {e}", flush=True)


DJ = TrackFactory()



def default_checkpoint():
    """The best brain by music quality (cards() is sorted best first)."""
    cs = cards()
    ready = [c for c in cs if not c.get("cadence_patch_sha256") or c.get("gate") or c.get("stars")]
    return (ready or cs)[0]["name"] if cs else None


def follow_latest():
    """While the listener follows 'latest', switch to each newer checkpoint as the watcher installs it."""
    while True:
        time.sleep(30)
        try:
            newest = default_checkpoint()
            idle = False      # 11-23: never switch brains on its own (it surprised the listener); the menu orders, the listener picks
            if idle and CREATURE.follow_latest and newest and newest != (CREATURE.card or {}).get("name"):
                CREATURE.load(newest)
                print(f"following best: {newest}", flush=True)
        except Exception as e:
            print(f"follow latest: {e}", flush=True)


CORS_ORIGINS = {"https://cadance.floatingpragma.ai",
                *(o.strip().rstrip("/") for o in os.environ.get("CADANCE_CORS_ORIGINS", "").split(",") if o.strip())}


def cors_origin(origin):
    """The Origin a response may name in Access-Control-Allow-Origin: the radio site's (CORS_ORIGINS) or a local test
    page's (http://localhost or http://127.0.0.1, any port); None for any other, and for a request without one."""
    if not origin:
        return None
    if origin in CORS_ORIGINS or re.fullmatch(r"http://(localhost|127\.0\.0\.1)(:\d{1,5})?", origin):
        return origin
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        # the radio site's live mode (site/radio.js) calls this server from its own origin
        origin = cors_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self):
        """The preflight of a cross-origin call (the radio site's live mode); Private Network Access when asked."""
        origin = cors_origin(self.headers.get("Origin"))
        self.send_response(204 if origin else 403)
        if origin:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers") or "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            if (self.headers.get("Access-Control-Request-Private-Network") or "").lower() == "true":
                self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

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
            pass  # the browser left or cancelled a superseded request

    def _curves(self):
        # An open Stats panel asks every second. The curves are read again only when a file they come from changed.
        composer_only = (SITE / ".composer-only").is_file()
        signature = (str(ROOT), training_stats.signature(ROOT, CHECKPOINTS, PINNED), composer_only,
                     SUPPORTED_CHECKPOINT_PATCHES)
        with _CURVES_LOCK:
            if _CURVES.get("signature") != signature:
                installed = [card["name"] for card, _ in _available_checkpoints()]
                _CURVES["brains"] = training_stats.curves(ROOT, CHECKPOINTS, PINNED, composer_only,
                                                          SUPPORTED_CHECKPOINT_PATCHES, installed_names=installed,
                                                          describe=anatomy.describe)
                _CURVES["signature"] = signature
            brains = _CURVES["brains"]
        # which runs have a box, as the mirror last saw it: a dict while training, None when unknown, False when not
        states, checked = training_stats.training(ROOT)
        brains = [{**brain, "training": states.get(brain["candidate"], False)} for brain in brains]
        return {"brains": brains, "active": (CREATURE.card or {}).get("name"), "mirror": {"at": checked},
                "slate": training_stats.slate(ROOT, layouts.LAYOUTS, check.code_sha(), check.judge)}

    def _curve_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.connection.settimeout(5)
        previous = None
        last_sent = time.monotonic()
        try:
            while True:
                curves = self._curves()
                # the time of the watcher's last look is no news; the cached curves stay as they are
                curves["brains"] = [
                    {**brain, "status": {k: v for k, v in brain["status"].items() if k != "checked_at"}}
                    if brain.get("status") else brain for brain in curves["brains"]]
                data = json.dumps(curves, separators=(",", ":")).encode()
                now = time.monotonic()
                if data != previous:
                    self.wfile.write(b"data: " + data + b"\n\n")
                    previous = data
                    last_sent = now
                elif now - last_sent >= 10:
                    self.wfile.write(b": keepalive\n\n")
                    last_sent = now
                time.sleep(1)
        except (ConnectionError, TimeoutError):
            pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/kit/"):          # the real 606/808/909 one-shots (engine/samples)
            kit = ROOT / "engine" / "samples"
            f = (kit / path[len("/kit/"):]).resolve()
            if f.is_file() and kit.resolve() in f.parents and f.suffix in (".wav", ".json"):
                return self._send(200, f.read_bytes(), "audio/wav" if f.suffix == ".wav" else "application/json")
            return self._send(404, {"error": "not found"})
        if path in STATIC and STATIC[path].exists():
            f = STATIC[path]
            return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
        if path == "/api/samples":
            import wave
            out = []
            for f in sorted(SAMPLES.glob("*.wav"), key=lambda p: -p.stat().st_mtime):
                try:
                    with wave.open(str(f)) as w:
                        secs = w.getnframes() / w.getframerate()
                except Exception:
                    secs = None
                title = SAMPLE_TITLES.get(f.stem)
                if title is None and f.stem.startswith("quality_"):
                    title = f"quality check · {f.stem[len('quality_'):]} · a free run (2 min)"
                if title is None and f.stem.startswith("replica_r3_"):
                    tid = f.stem[len("replica_r3_"):].removesuffix("_tokens")
                    names = R3_TITLES()
                    kind = "through the round-3 vocabulary" if f.stem.endswith("_tokens") else "every bar fitted on its own"
                    title = f"round 3 replica · {names.get(tid, tid)} · {kind}"
                out.append({"name": f.name, "title": title or f.stem, "seconds": secs})
            return self._send(200, {"samples": out})
        if path.startswith("/samples/") and path.endswith(".wav"):
            f = SAMPLES / Path(path).name
            if not f.exists():
                return self._send(404, {"error": "not found"})
            size = f.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if (a and b) else size - 1
            with open(f, "rb") as fh:
                fh.seek(start)
                data = fh.read(end - start + 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/pool":
            return self._send(200, (ROOT / "pool" / CREATURE.pool_name()).read_bytes())
        if path == "/api/layout":
            return self._send(200, {"layout": CREATURE.view_layout, "checkpoint": (CREATURE.card or {}).get("name")})
        if path == "/api/brain3d":                 # the 3D brain of a checkpoint (brainview.get3d): the player's when none is named
            from urllib.parse import parse_qs
            name = parse_qs(self.path.partition("?")[2]).get("checkpoint", [""])[0] or (CREATURE.card or {}).get("name")
            if name and not re.fullmatch(r"[A-Za-z0-9_]{1,80}", name):
                return self._send(400, {"error": "bad checkpoint name"})
            holders = [CREATURE, MIX["incoming"], DJ.creature, *_DECKS]
            brain = next((c.brain for c in holders if c is not None and c.brain is not None and (c.card or {}).get("name") == name), None)
            npz = next((_dir(where) / f"{name}.npz" for card, where in _available_checkpoints() if card["name"] == name), AUDITION / f"{name}.npz") if name else None
            code, body = brainview.get3d(name or None, brain, npz)
            return self._send(code, body, "application/octet-stream" if isinstance(body, bytes) else "application/json")
        if path == "/api/curves":
            return self._send(200, self._curves())
        if path == "/api/curves/events":
            return self._curve_events()
        if path == "/api/decks":
            return self._send(200, {"decks": decks_status()})
        if path == "/api/autodj/models":
            self._curves()                             # the ranking reads the curves as Stats does
            return self._send(200, dj_models())
        if path == "/api/autodj/status":
            return self._send(200, DJ.status())
        if path in ("/api/autodj/track", "/api/autodj/pool"):
            from urllib.parse import parse_qs
            q = parse_qs(self.path.partition("?")[2])
            reply = DJ.read(q.get("id", [""])[0], q.get("from", ["0"])[0])
            if path == "/api/autodj/track" or reply.get("error") or not reply.get("pool"):
                return self._send(200, reply)
            return self._send(200, (ROOT / "pool" / reply["pool"]).read_bytes())
        if path == "/api/deck/pool":
            from urllib.parse import parse_qs
            q = parse_qs(self.path.partition("?")[2])
            return self._send(200, (ROOT / "pool" / deck(q.get("deck", ["0"])[0]).pool_name()).read_bytes())
        if path == "/api/listen":                  # the listening check (brain/listen.py in the QA loop): the newest checkpoint per brain
            rows, refs = {}, {}
            try:
                for line in (ROOT / "quality" / "listen" / "listen.jsonl").read_text().splitlines():
                    r = json.loads(line)
                    if not r.get("summary"):
                        continue
                    if r["name"].startswith("ref_"):
                        refs[r["name"]] = r
                        continue
                    if (r.get("entry_scale"), r.get("line_scale")) != (2.0, 1.2):     # as the player reads them
                        continue
                    c = r.get("candidate") or r.get("brain")
                    if c not in rows or (r.get("time") or "") > (rows[c].get("time") or ""):
                        rows[c] = r
            except FileNotFoundError:
                pass
            def slim(r):
                s = r["summary"]; y = s.get("synth") or {}
                return {"name": r.get("brain") or r["name"], "candidate": r.get("candidate"), "streams": r.get("streams"), "listen_error": r.get("listen_error"),
                        "drums_only": s["drums_only"], "heard": s["heard"], "beat_kinds": s["beat_distinct_per_16"],
                        "bass": {k: (y.get("bass") or {}).get(k) for k in ("onsets", "distinct_per_16")},
                        "lead": {k: (y.get("lead") or {}).get(k) for k in ("onsets", "distinct_per_16")}}
            from listen import listen_error                 # rows written before the score existed get it here
            for r in rows.values():
                ref = refs.get(f"ref_{Path(r.get('streams') or '').stem}")
                if r.get("listen_error") is None and ref:
                    r["listen_error"] = listen_error(r["summary"], ref["summary"])
            out = sorted((slim(r) for r in rows.values()), key=lambda r: (r["listen_error"] is None, r["listen_error"] or 0))
            return self._send(200, {"brains": out, "refs": {k: slim(v) for k, v in refs.items()}})
        if path == "/api/feedback":                # the waveform's marks (feedback_marks.py): the list, or one by id
            from urllib.parse import parse_qs
            mark_id = parse_qs(self.path.partition("?")[2]).get("id", [""])[0]
            if mark_id:
                rec = feedback_marks.read(mark_id)
                return self._send(200, rec) if rec else self._send(404, {"error": "no such mark"})
            return self._send(200, {"marks": feedback_marks.listing()})
        if path == "/api/feedback/pool":           # the instrument pool a mark's bars index, for its replay
            from urllib.parse import parse_qs
            f = feedback_marks.pool_path(parse_qs(self.path.partition("?")[2]).get("name", [""])[0], ROOT / "pool")
            return self._send(200, f.read_bytes()) if f else self._send(404, {"error": "no such pool"})
        if path == "/api/load/progress":           # every holder's load: stage, words, percent (LoadProgress.read)
            return self._send(200, all_progress())
        if path == "/api/status":
            loading = player_loading()
            return self._send(200, {"ok": True, "loading": loading, "ready": loading is None,
                                    "progress": LOAD_PROGRESS["player"].read(),
                                    "active": (CREATURE.card or {}).get("name"),
                                    "pool": CREATURE.pool_name(), "cadence_version": cadence.__version__,
                                    "cadence_patch_sha256": RUNTIME_PATCH})
        if path == "/api/brains":
            return self._send(200, {"brains": cards(), "active": (CREATURE.card or {}).get("name"), "follow_latest": CREATURE.follow_latest})
        if path == "/api/checkpoints":
            return self._send(200, {"checkpoints": cards(), "active": (CREATURE.card or {}).get("name"), "play_mode": CREATURE.play_mode,
                                    "follow_latest": CREATURE.follow_latest,
                                    "demo": "sid" if CREATURE.brain is None and CREATURE.demo_pool else None})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/next":
                mood = [min(max(float(v), 0.0), 1.0) for v in body.get("mood", [0.5, 0.5])][:2]
                reward = min(max(float(body.get("reward", 0.0)), -1.0), 1.0)
                CREATURE.last_request = time.time()
                if body.get("brain3d"):              # the page shows the 3D brain: every drawn neuron per bar
                    for c in (CREATURE, MIX["incoming"]):
                        if c is not None:
                            c.want3d = time.time()
                if CREATURE.loading or not LAYOUT.acquire(timeout=2.0):
                    return self._send(200, {"loading": CREATURE.loading or "busy"})
                try:
                    if CREATURE.loading:
                        reply = {"loading": CREATURE.loading}
                    elif MIX["incoming"] is not None:
                        reply = mixed_next(mood, reward)
                    else:
                        reply = CREATURE.next(mood, reward)
                finally:
                    LAYOUT.release()
                return self._send(200, reply)
            if self.path == "/api/autodj/track":
                return self._send(200, {"id": DJ.request(body.get("model") or "leader", body.get("bars", 256), body.get("mood"), bool(body.get("quick")), body.get("fallback"), body.get("session"),
                                                         kept_only=bool(body.get("kept_only")))})
            if self.path == "/api/autodj/approved":
                dj_approved((body.get("model"), body.get("on", True)))
                return self._send(200, dj_models())
            if self.path == "/api/autodj/keep":
                DJ.keep(set(body.get("ids") or []), body.get("session"))
                return self._send(200, DJ.status())
            if self.path == "/api/autodj/stop":
                DJ.stop(body.get("session"))
                return self._send(200, {"ok": True})
            if self.path.startswith("/api/deck/"):
                d = deck(body.get("deck", 0))
                what = self.path[len("/api/deck/"):]
                if what == "next":
                    mood = [min(max(float(v), 0.0), 1.0) for v in body.get("mood", [0.5, 0.5])][:2]
                    d.last_request = time.time()
                    if d.loading or not LAYOUT.acquire(timeout=2.0):
                        return self._send(200, {"loading": d.loading or "busy"})
                    try:
                        if d.loading:
                            reply = {"loading": d.loading}
                        elif d.brain is None:
                            reply = {"empty": True}
                        else:
                            reply = dict(d.next(mood, 0.0), deck=int(body.get("deck", 0)))
                    finally:
                        LAYOUT.release()
                    return self._send(200, reply)
                if what == "load":
                    name = body.get("name")
                    if name in (None, "", "none"):
                        d.unload()
                    elif d.brain is not None and (d.card or {}).get("name") == name and not d.loading:
                        d.renew()                                  # already this deck's brain (a page reload): a new composition
                        d.loaded_at = time.time()
                    else:
                        d.load(name)
                        d.loaded_at = time.time()
                    return self._send(200, {"deck": int(body.get("deck", 0)), "active": (d.card or {}).get("name"), "pool": d.pool_name() if d.brain is not None else None})
                if what == "new":
                    if d.loading:
                        return self._send(200, {"loading": d.loading})
                    d.renew()
                    return self._send(200, {"deck": int(body.get("deck", 0)), "active": (d.card or {}).get("name"), "seed": getattr(d, "seed_title", None)})
                return self._send(404, {"error": "not found"})
            if self.path == "/api/new":
                with LAYOUT:
                    if CREATURE.loading:
                        reply = {"loading": CREATURE.loading}
                    else:
                        cancel_mix()
                        CREATURE.renew()
                        reply = {"active": (CREATURE.card or {}).get("name"), "seed": getattr(CREATURE, "seed_title", None)}
                return self._send(200, reply)
            if self.path == "/api/mode":
                mode = body.get("mode")
                if mode not in ("imitate", "live"):
                    raise ValueError("mode is imitate or live")
                CREATURE.play_mode = mode
                if CREATURE.card:
                    CREATURE.load(CREATURE.card["name"])
                return self._send(200, {"mode": mode})
            if self.path == "/api/brains/update":
                return self._send(200, update_brain(body.get("name"), stars=body.get("stars"), label=body.get("label"), note=body.get("note")))
            if self.path == "/api/brains/order":
                return self._send(200, order_brains([str(n) for n in body.get("names") or []]))
            if self.path == "/api/brains/delete":
                e = delete_brain(body.get("name"))
                if (CREATURE.card or {}).get("name") == e["name"]:       # the deleted brain was playing: the best one takes over
                    best = default_checkpoint()
                    CREATURE.follow_latest = True
                    CREATURE.load(best) if best else CREATURE.unload()
                return self._send(200, {"deleted": e["name"], "active": (CREATURE.card or {}).get("name")})
            if self.path == "/api/feedback" and "mark" in body:   # a liked or disliked selection (feedback_marks.py)
                player = CREATURE
                with player.lock:
                    card, history, playing_pool = dict(player.card or {}), list(player.history), player.pool_name()
                installed = _installed()

                def pool_of(name):
                    if name == card.get("name"):
                        return playing_pool
                    p = (installed.get(name) or {}).get("pool") or "pool_v0.json" if name else "pool_v0.json"
                    return p if (ROOT / "pool" / p).exists() else "pool_v0.json"
                return self._send(200, feedback_marks.add(
                    body, card=card, history=history, installed=installed, pool_of=pool_of, style=player.style,
                    layout=common.layout_key(card) if card else None, reading_path=STATIC["/play-reading.json"]))
            if self.path == "/api/feedback":
                pb = body.get("bar")
                return self._send(200, CREATURE.feedback(float(body.get("rating", 0)), int(pb) if pb is not None else None))
            if self.path == "/api/load":
                return self._send(*select_checkpoint(body))
        except Exception as e:  # report to the page, keep serving
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})
        return self._send(404, {"error": "not found"})

    def do_DELETE(self):
        path = self.path.split("?")[0]
        if path == "/api/feedback":                # one waveform mark: its file and its index line
            from urllib.parse import parse_qs
            mark_id = parse_qs(self.path.partition("?")[2]).get("id", [""])[0]
            if feedback_marks.remove(mark_id):
                return self._send(200, {"deleted": mark_id})
            return self._send(404, {"error": "no such mark"})
        return self._send(404, {"error": "not found"})


def main():
    name = default_checkpoint()
    try:
        saved = PLAYER_SELECTION.read_text().strip()
        if saved == "demo" or any(c["name"] == saved for c in cards()):
            name = None if saved == "demo" else saved
    except OSError:
        pass

    def first():
        if name:
            try:
                CREATURE.load(name)
                print(f"loaded {name}", flush=True)
            except Exception as e:              # never let one checkpoint keep the site down
                print(f"could not load {name}: {type(e).__name__}: {e}", flush=True)
        threading.Thread(target=follow_latest, daemon=True).start()
        threading.Thread(target=deck_idle, daemon=True).start()

    for c in cards()[:6]:                       # the starting passages of the listed brains' corpora, in the background
        try:
            seed_passages(c)
        except Exception as e:
            print(f"starting passages: {e}", flush=True)
    CREATURE.loading = name                     # the page sees "loading" until the brain is in
    threading.Thread(target=first, daemon=True).start()
    print(f"http://127.0.0.1:{PORT}/  loading {name or 'no brain (instrument check)'}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
