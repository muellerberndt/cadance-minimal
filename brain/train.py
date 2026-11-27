"""Childhood of a minimal-techno creature: teacher lessons on the corpus token streams.

Each window of a training track is one continuing stream (no reset inside it); per bar the brain is taught
the corpus's next bar, then acts on the present bar (teach first, then act, so the recall memory holds the
moment as the act leaves it). Checkpoints with a card (settings, held-out reading, free-run gate) go to
--out every --save-every epochs and whenever composition or held-out prediction improves.

    PYTHONPATH=cadence/src python brain/train.py --streams corpus/streams.json --preset deep --name deep_s0
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import shutil
import tempfile
import time
import warnings
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composition
import groove
import anatomy
import hierarchy
import layouts
import timbre
import common  # noqa: E402
from common import L, compose, corpus_reference, load_streams, observe, windows  # noqa: E402
from gate import calibrate, commonest_bar, density_rollout, free_run, gate, held_out, prediction_error, replay, rollouts  # noqa: E402
from recall_trace import RecallTrace


# Library patches (sha256) and the earlier patches each contains: the combined reflex + per-slot-surprise patch holds
# the sensory-reflex patch whole, so a state trained on the latter continues on the former.
PATCH_LINEAGE = {
    "e8cf41673a43b79b75188a052c5f2a610f8a8de0059f6bcd300ec975d10abaf8":
        ("6b3eb1d9703248c52d3555cfb7e302d72d360d02ba5b07a6df293ce07d7f0c96",),
    # reflex + per-slot surprise + sampled judge
    "f87487bbabc36137d0841efb0a4da78bf878786713449aa7c6c8c9038c9e9600":
        ("e8cf41673a43b79b75188a052c5f2a610f8a8de0059f6bcd300ec975d10abaf8",
         "6b3eb1d9703248c52d3555cfb7e302d72d360d02ba5b07a6df293ce07d7f0c96"),
}


def card_for(args, settings, epoch, elapsed, reading, gate_sampled, gate_greedy, path, best):
    return {
        "name": path.stem, "candidate": args.name, "preset": args.preset, "layout": args.layout, "seed": args.seed, "epoch": epoch,
        "hear_first": common.HEAR_FIRST, "gain_bound": args.gain_bound or None, "reserve": common.RESERVE or False,
        "sid_live": bool(common.SID_LIVE), "sid_pool": Path(args.sid_lessons).name if args.sid_lessons else None,
        "ear": bool(common.EAR_LIVE), "hear_parts": bool(common.HEAR_PARTS), "lookahead": bool(common.LOOKAHEAD),
        "section": bool(common.SECTION), "clock_offsets": args.clock_offsets_card,
        "echo_lags": list(common.ECHO_LAGS) or None,
        "minutes": round(elapsed / 60, 1), "best": best,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": settings, "held_out": reading,
        "gate": gate_sampled, "gate_greedy": {"pass": gate_greedy["pass"], "checks": gate_greedy["checks"]},
        "pool": args.pool, "streams_sha": args.streams_sha, "tokens_per_layer": common.T, "mod_levels": common.M, "steps": bool(args.steps), "styles": common.STYLES, "streams": Path(args.streams).name, "beat_rows": common.ROWS, "synth_rows": common.SYNTH, "synth_fill": bool(common.SYNTH_FILL), "onoff": bool(common.ONOFF), "entry_scale": common.ENTRY_SCALE, "line_scale": common.LINE_SCALE, "own_sample": args.own_sample, "clock": bool(common.CLOCK), "content": bool(args.content), "pool_sha": args.pool_sha, "imitation": bool(args.all_tracks),
    }


def eval_set(tracks, k, nbars, seed, interior=False):
    """A fixed evaluation set: up to k test tracks and k training tracks (seeded choice), each cut to nbars."""
    if not k and not nbars:
        return tracks
    r = np.random.default_rng(1000 + seed)
    def pick(lst):
        if not k or len(lst) <= k:
            return lst
        return [lst[i] for i in sorted(r.choice(len(lst), size=k, replace=False))]
    training = [t for t in tracks if not t["held_out"]]
    testing = [t for t in tracks if t["held_out"]]
    if interior:
        training = [t for t in training if len(t["bars"]) >= 17]
        testing = [t for t in testing if len(t["bars"]) >= 25]
    groups = sorted({t["group"] for t in training if t.get("window_share")})
    if groups and k >= len(groups):
        # Every reference recording group appears in the temperature scan.
        by_group = [[t for t in training if t.get("group") == g and len(t["bars"]) >= 17] for g in groups]
        first = [items[int(r.integers(len(items)))] for items in by_group if items]
        selected = {t["id"] for t in first}
        extra = [t for t in training if t["id"] not in selected and len(t["bars"]) >= 17]
        training = first + [extra[i] for i in r.permutation(len(extra))[:max(0,k-len(first))]]
    else:
        training = pick(training)
    sel = pick(testing) + training
    if nbars:
        excerpts = []
        for t in sel:
            # Keep the absolute six-bit clock phase during the separate mood tour.
            blocks = max(0, len(t["bars"]) - nbars) // 64
            start = 64 * int(r.integers(blocks + 1)) if interior else 0
            excerpts.append(dict(common.passage(t, start, nbars), offset=start))
        sel = excerpts
    return sel


def load_liked(paths, layout):
    """Liked windows from the site's feedback logs, grouped by length: {n: [(states (n, slots), mood (n, 2)), ...]}."""
    groups = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("rating", 0) <= 0 or row.get("tokens_per_layer") != common.T or row.get("mod_levels") != common.M:
                continue
            if row.get("layout") != layout and (common.PATCH_BOOK or row.get("layout")):
                continue
            bars = [b for b in row.get("bars", []) if len(b.get("state") or []) == common.SLOTS]
            if len(bars) < 2:
                continue
            states = np.array([b["state"] for b in bars], dtype=int)
            mood = np.array([b["mood"] for b in bars], dtype=float)
            if mood.shape[1] != common.MOOD:
                if mood.shape[1] != 2:
                    continue
                mood = common.mood_rows(mood, row.get("style"), bars=[b.get("position", b["bar"]) for b in bars])
            groups.setdefault(len(bars), []).append((states, mood))
    return groups


def interest_weights(streams, train, weight):
    """{id of a training track's bar array: lesson weight}: ``weight`` for the passages whose interest reading
    (the score's "interest" score per passage) is at or above the median of the training passages, 1 for the
    rest. A lesson window is a view of its track's bars, so its weight is found through the array it views."""
    reading = {t["id"]: (t.get("interest") or {}).get("score") for t in json.loads(Path(streams).read_bytes())["tracks"]}
    scores = {id(t["bars"]): reading.get(t["id"]) for t in train}
    known = [v for v in scores.values() if v is not None]
    if not known:
        return None
    median = float(np.median(known))
    return {key: (weight if v is not None and v >= median else 1.0) for key, v in scores.items()}


def window_weight(table, window_bars):
    """A window's weight in interest_weights' table (1 when its track has none)."""
    base = window_bars.base if window_bars.base is not None else window_bars
    return table.get(id(base), 1.0)


def transpose_lines(bars, shifts):
    """Bass and lead lines of lesson windows (rows, bars, slots) moved by ``shifts`` semitones, one per row.

    A synth line is 16 steps per bar, 0 a rest, 1 a tie and 2..13 an onset at a pitch class; a shift turns the
    pitch classes round the octave (the sound's register stays, as it does for every line of the score), so the
    intervals and the rhythm of a line stay and its key changes. The air line, the sounds and every other slot
    stay as written."""
    if not (common.SYNTH and common.SYNTH.get("kind") == "steps") or not np.any(shifts):
        return bars
    per = common.SYN // len(common.SYNTH_LAYERS)
    lines = slice(common.L + common.MODS + common.STEPS, common.L + common.MODS + common.STEPS + 2 * per)
    out = np.array(bars, copy=True)
    notes = out[:, :, lines]
    moved = (notes - 2 + np.asarray(shifts)[:, None, None]) % 12 + 2
    out[:, :, lines] = np.where(notes >= 2, moved, notes)
    return out


MARK_FORMAT = "cadance-mark/1"          # site/feedback_marks.py


def load_marks(paths, layout):
    """The owner's marks from the player's waveform (site/feedback_marks.py: one <id>.json per mark, in folders or
    named singly): liked marks as liked windows, {n: [(states (n, slots), mood (n, MOOD)), ...]} as load_liked
    gives them, and disliked marks set aside as [(id, states, mood), ...] (kept for the critic, never taught).

    A mark counts when it is of this format, its brain's layout key is the run's and its sizes are the run's.
    Its windows are the runs of consecutive bars (by the server's bar number, one life) that carry the brain's
    slot vector; a run of one bar teaches nothing."""
    liked, disliked = {}, []
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        try:
            mark = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(mark, dict) or mark.get("format") != MARK_FORMAT or mark.get("mark") not in ("liked", "disliked"):
            continue
        if mark.get("layout") != layout or mark.get("tokens_per_layer") != common.T or mark.get("mod_levels") != common.M:
            continue
        runs, run = [], []
        for b in mark.get("bars") or []:
            usable = len(b.get("state") or []) == common.SLOTS and len(b.get("mood") or []) in (2, common.MOOD)
            if usable and run and (b.get("bar") != run[-1].get("bar", 0) + 1 or b.get("life") != run[-1].get("life")):
                runs.append(run)
                run = []
            if usable:
                run.append(b)
            elif run:
                runs.append(run)
                run = []
        runs.append(run)
        for bars in (r for r in runs if len(r) >= 2):
            states = np.array([b["state"] for b in bars], dtype=int)
            mood = np.array([b["mood"] for b in bars], dtype=float)
            if mood.shape[1] != common.MOOD:
                mood = common.mood_rows(mood, mark.get("style"), bars=[b.get("position", b["bar"]) for b in bars])
            if mark["mark"] == "liked":
                liked.setdefault(len(bars), []).append((states, mood))
            else:
                disliked.append((mark.get("id"), states, mood))
    return liked, disliked


def learning_rate(config, eta):
    """Keep bias and synapse updates in the same ratio when retuning."""
    bias = config.eta_bias if config.eta == 0 else config.eta_bias * eta / config.eta
    return replace(config, eta=eta, eta_bias=bias)


class Plateau:
    """Reduce the lesson step only when validation error and continuations both stall."""
    def __init__(self, eta, patience, floor=0.25):
        self.floor = eta * floor
        self.patience = patience
        self.best_error = float("inf")
        self.best_music = float("-inf")
        self.stalled = 0

    def next_rate(self, eta, error, music):
        improved = error < self.best_error - .002 or music > self.best_music + .005
        if error < self.best_error - .002:
            self.best_error = error
        if music > self.best_music + .005:
            self.best_music = music
        self.stalled = 0 if improved else self.stalled + 1
        if self.stalled >= self.patience and eta > self.floor:
            self.stalled = 0
            return max(eta*.7, self.floor)
        return eta


class SourceWindows:
    """Deal windows without replacement within each group, across epoch boundaries."""
    def __init__(self, tracks, length, stride, rng):
        self.rng = rng
        shares = {t["group"]: t["window_share"] for t in tracks}
        self.groups = {g: windows([t for t in tracks if t["group"] == g], length, stride) for g in shares}
        if any(not self.groups[g] or not 0 < share <= 1 for g, share in shares.items()) or not np.isclose(sum(shares.values()), 1):
            raise ValueError("Each training group needs windows and its share of the epoch")
        total = min(len(self.groups[g]) / share for g, share in shares.items())
        self.counts = {g: max(1, round(total*share)) for g, share in shares.items()}
        self.remaining = {g: [] for g in shares}

    def draw(self, with_groups=False):
        out = []
        for group, count in self.counts.items():
            pool = self.groups[group]
            for _ in range(count):
                if not self.remaining[group]:
                    self.remaining[group] = self.rng.permutation(len(pool)).tolist()
                window = pool[self.remaining[group].pop()]
                out.append((group, window) if with_groups else window)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", required=True)
    ap.add_argument("--preset", default="deep")
    ap.add_argument("--layout", choices=tuple(layouts.LAYOUTS) + tuple(layouts.S1_LAYOUTS),
                    help="a layout of the comparison slate (layouts.py): sets the preset and its wiring")
    ap.add_argument("--hear-first", action="store_true",
                    help="every bar is heard as a moment of its own before it is taught or answered, so the recall "
                         "memory holds the bar the answer is asked for (common.HEAR_FIRST)")
    ap.add_argument("--hear-parts", action="store_true",
                    help="with --hear-first: a bar is heard as two moments, its drum rows and synth lines alone and "
                         "then the whole bar, so heads born at lags beyond 1 bring those rows only "
                         "(common.HEAR_PARTS, common.born_lags)")
    ap.add_argument("--lookahead", action="store_true",
                    help="every bar carries the plan slots, the score's summary of the 8 bars after it, which the brain "
                         "hears with the bar heard and writes with the bar it writes (common.LOOKAHEAD, plan_rows)")
    ap.add_argument("--clock-offsets", default=None, metavar="FILE",
                    help="per track id the bar offset of its lesson clock (JSON: {\"clock_offset\": {id: offset}}, "
                         "the structure lane's one-bar grid): the clock of bar b of the track reads b + offset; "
                         "no new slot, and in play the clock counts from bar 0 (needs --clock)")
    ap.add_argument("--section", action="store_true",
                    help="every bar carries the section slot (normal, breakdown, build) from the score's per-bar "
                         "\"section\" (score v5f), heard and written as the plan is (common.SECTION)")
    ap.add_argument("--read-share", type=float, default=0.0,
                    help="light readings take at most this share of the run's time: after a reading that took c "
                         "seconds the next one waits for c (1 - share) / share seconds of lessons (the first comes "
                         "after the first batch). In place of --read-batches")
    ap.add_argument("--eval-minutes", default="",
                    help="full evaluations at these minutes of the run (comma-separated, e.g. 120,240,480) and at "
                         "its end, and none at the end of a pass or inside one")
    ap.add_argument("--keep-latest", action="store_true",
                    help="every light reading leaves its weights as <name>_latest.npz with a card: the newest "
                         "state, to branch from or to restore")
    ap.add_argument("--read-batches", type=int, default=0,
                    help="between the full evaluations, read the next-bar error every N batches: a point on the "
                         "curve, no checkpoint (0 = never)")
    ap.add_argument("--gain-bound", type=float, default=0.0,
                    help="the bound on the step of the value path's gains (Cadence: 5). A lesson's step is this over "
                         "the energy of what the memory read, about 8,700 at the composer's size, so the gains of a "
                         "full-size brain learn sixty times slower than a small brain's (0 = the library's bound)")
    ap.add_argument("--gain-cap", type=float, default=0.0,
                    help="the most a gain of the direct route may reach: after every lesson a gain above it is set "
                         "back to it; below it the gains learn as ever (0 = no cap)")
    ap.add_argument("--listen-patience", type=int, default=0,
                    help="stop a run whose held-out error is not below the commonest-bar level after this many "
                         "evaluations: it does not use what it hears (0 = never)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without a better score (0 = never)")
    ap.add_argument("--gate-bars", type=int, default=64, help="free-run bars per mood corner")
    ap.add_argument("--change-weight", type=float, default=1.0, help="lesson weight of a stream row whose next bar changes a layer")
    ap.add_argument("--mask-unheard-targets", action="store_true",
                    help="Leave inactive instrument controls open while teaching next-bar labels")
    ap.add_argument("--synth-lesson-group", action="append",
                    help="Use exact synth labels only for these source groups. Every group still teaches arrangement and drums.")
    ap.add_argument("--sampler-weight", type=float, default=3., help="Relative lesson weight for labeled one-shot gestures")
    ap.add_argument("--plateau", type=int, default=0, help="Reduce learning rate after this many stagnant validation checks")
    ap.add_argument("--evals-per-epoch", type=int, default=0,
                    help="this many full evaluations per pass at even spacing, the last at its end; replaces --save-batches")
    ap.add_argument("--unsettled-patience", type=int, default=3,
                    help="after this many lesson windows or readings in which the brain did not settle, without a clean "
                         "evaluation between them, the last saved weights come back and the learning rate is halved")
    ap.add_argument("--gpu-cap", type=float, default=0.0,
                    help="the most GPU memory this trainer may take, in GB: beyond it this trainer fails and no other "
                         "process on the GPU does (0 = no cap)")
    ap.add_argument("--plateau-floor", type=float, default=0.25,
                    help="the lowest learning rate the plateau rule goes to, as a share of the starting rate")
    ap.add_argument("--reserve", nargs="?", const="sid", default=None, choices=("inputs", "kits", "sid", "sidlead"),
                    help="the stage-1 shape (PLAN.md, common.RESERVE), so later stages change the lessons and not the "
                         "brain's shape: 'inputs' = idle ear and style inputs; 'kits' = those and room for 64 more kits "
                         "in the kit slot; 'sid' = those and one block of SID register slots per synth")
    ap.add_argument("--ear", action="store_true",
                    help="with --reserve: the ear inputs carry the measured levels of the bar just heard "
                         "(common.ear_inputs; the score's levels and bar_db), from the first lesson")
    ap.add_argument("--sid-lessons", metavar="POOL",
                    help="with --reserve sid: teach the SID blocks from the first lesson, each bar's sound as its SID fit "
                         "in this pool (pool/pool_underground_sid.json), the chip's switch on. Without it the blocks are "
                         "labelled off")
    ap.add_argument("--eta-scale", type=float, default=1.0, help="multiply the brain's learning rate (learner eta) by this, after compose or --init")
    ap.add_argument("--interest-weight", type=float, default=1.0,
                    help="lesson weight of windows from the more interesting half of the training passages (score "
                         "v4's interest reading: at or above the median of the training passages); the other half "
                         "keeps weight 1")
    ap.add_argument("--transpose", type=int, default=0,
                    help="each lesson window's bass and lead lines shifted by one random number of semitones in "
                         "-N..N, the same for both (transpose_lines); 0 = as written")
    # The settle of a lesson (Cadence LearnerConfig, set as an expert after compose or --init; none = the brain's own)
    ap.add_argument("--beta", type=float, help="the nudge strength of a lesson (the composed brain's constant: 0.1)")
    ap.add_argument("--nudged-steps", type=int, help="the sweeps of each nudged phase (teaching 'deep': 32, run whole)")
    ap.add_argument("--free-steps", type=int, help="the most sweeps of a lesson's free phase (100)")
    ap.add_argument("--credit-window", type=int, help="report hidden credit and warn after this many quiet lessons")
    ap.add_argument("--drive-budget", type=float,
                    help="Cadence's learning_drive_budget: a bound on the learned basal projections into each hidden region "
                         "on the lesson's own activations (0.104.0)")
    ap.add_argument("--learner-set", action="append", default=[], metavar="FIELD=JSON",
                    help="a LearnerConfig field set by retune after birth or --init (e.g. a newer library's per-slot "
                         "surprise): the switch a resumed brain can take without a new birth")
    ap.add_argument("--credit-transport", type=float,
                    help="experimental maximum teaching-only apical feedback gain (0 disables)")
    ap.add_argument("--tolerance", type=float, help="the movement at which a lesson's free phase stops (1e-4)")
    ap.add_argument("--normalize", type=float, help="the forgetting factor of each synapse's RMS that divides its step "
                                                     "(teaching 'deep': 0.99; 0 = raw contrast steps)")
    ap.add_argument("--momentum", type=float, help="the running average a synapse steps on (the composed brain's 0.9)")
    ap.add_argument("--rehearsal", type=float, help="after each lesson replay this many times its rows from a reservoir of "
                                                     "past lesson rows (Cadence learning rehearsal; with surprise learning "
                                                     "only rows the brain now answers wrong move it; 0 = none)")
    ap.add_argument("--onoff", action="store_true", help="each layer an on/off slot; its sound slot keeps the last sound while off (common.ONOFF)")
    ap.add_argument("--synth-fill", action="store_true", help="while a synth layer is silent, teach its line as the last line that sounded (no rest majority)")
    ap.add_argument("--calibration-bars", type=int, default=32, help="free-run bars per mood corner during temperature calibration")
    ap.add_argument("--temps", default="", help="temperatures the checkpoint's calibration scans, comma-separated (default: gate.TEMPS, 12 of them)")
    ap.add_argument("--synth-vocab", default="rhythm:128", help="steps, splitsteps (separate timing and pitch), factored:R[:C], rhythm:K or rows:K")
    ap.add_argument("--focus-style", help="style whose training statistics and held-out tracks guide selection")
    ap.add_argument("--focus-share", type=float, default=0., help="minimum share of epoch windows in that style; every original window is still visited")
    ap.add_argument("--min-kick", type=float, default=0.0, help="teach only tracks whose beat bars carry a kick in at least this share (drum rows): steady-beat tracks, every kickless stretch short")
    ap.add_argument("--event-weight", type=float, default=1.0, help="lesson weight of a bar where one of --event-slots changes (rare events: the kick leaves or comes back)")
    ap.add_argument("--event-slots", default="kick", help="kick (beat sound + kick row) | beat (beat sound + all drum rows) | tokens (the four sounds) | sounds+kick | arrangement (layer entrances/exits and audible kick departures/returns) | section (the bars on either side of a change of section: the bar before it and the first bar of the new section; needs --section)")
    ap.add_argument("--all-tracks", action="store_true", help="imitation: train on every track (held-out ones too); best = replay from memory")
    ap.add_argument("--replay-bars", type=int, default=128)
    ap.add_argument("--out", default="brains")
    ap.add_argument("--pool", default="pool.json")
    ap.add_argument("--set", action="append", default=[], help="extra compose setting key=json")
    ap.add_argument("--synth-controls", metavar="POOL", help="learn absolute oscillator, filter, envelope and mix controls from this fitted pool")
    ap.add_argument("--recall-probe", action="store_true", help="compare intact recall, cleared history, and muted recall drive on saved checkpoints")
    ap.add_argument("--audio-eval", action="store_true", help="compare rendered stem audibility on every validation prompt and keep the best mix")
    ap.add_argument("--composition-bars", type=int, default=64, help="held-out continuation length for DSP checkpoints")
    ap.add_argument("--synth-rows", action="store_true", help="the brain writes the synth lines: one 16-step note row per synth layer (needs \"synth\" in the streams)")
    ap.add_argument("--clock", action="store_true", help="the brain hears the bar clock (bar count of the bar it writes, 6 bits)")
    ap.add_argument("--save-first-batch", action="store_true", help="evaluate and save an early listening baseline after the first batch")
    ap.add_argument("--save-batches", type=int, default=0, help="also evaluate and save a checkpoint every N batches inside an epoch")
    ap.add_argument("--max-batches", type=int, default=0, help="stop after this many total batches with a final evaluation (0 = unlimited)")
    ap.add_argument("--eval-tracks", type=int, default=0, help="evaluate on at most this many test and this many training tracks (0 = all)")
    ap.add_argument("--eval-bars", type=int, default=0, help="evaluate on at most this many bars per track (0 = all)")
    ap.add_argument("--beat-rows", action="store_true", help="the brain writes the beat as one row per drum voice (row vocabularies from the training tracks)")
    ap.add_argument("--steps", action="store_true", help="the brain writes the beat: kick/snare/hats on 16 steps (needs \"drums\" in the streams)")
    ap.add_argument("--backend", default="cpu", help="cpu or torch (CUDA also supports learned code)")
    ap.add_argument("--content", help="pool.json whose sounds the brain hears (pattern + timbre per layer) next to their ids")
    ap.add_argument("--own-sample", type=float, default=0.0, help="on its own path the brain hears a SAMPLED answer read the way the player reads it (this temperature, --entry-scale, --line-scale) instead of its greedy answer (the commonest value: plain lines, layers off); 0 = greedy")
    ap.add_argument("--entry-scale", type=float, default=None, help="common.ENTRY_SCALE for the own path and the card (the player reads the brain with it)")
    ap.add_argument("--line-scale", type=float, default=None, help="common.LINE_SCALE for the own path and the card (the player reads the brain with it)")
    ap.add_argument("--own-path", type=float, default=0.0, help="share of moments the brain hears its own answer instead of the corpus bar (labels stay the corpus); ramps up over --own-ramp epochs")
    ap.add_argument("--reference-edits", action="store_true", help="on sampled bars teach stable, sounding patterns to carry until the reference teaches an edit")
    ap.add_argument("--motif-weight", type=float, default=1.0,
                    help="minimum lesson weight when a sounding reference motif returns after eight or sixteen bars")
    ap.add_argument("--development-weight", type=float, default=1.0,
                    help="minimum lesson weight on reference buildups, releases and changes to a new motif")
    ap.add_argument("--detail-focus", type=float, default=0.0,
                    help="fraction of ordinary teacher bars devoted to drum accents and pitched attacks")
    ap.add_argument("--arrangement-focus", type=float, default=0.0,
                    help="fraction of stable entrances and exits taught with just their arrangement slots")
    ap.add_argument("--rollout-bars", type=int, choices=(1, 2, 4, 8, 16, 32), default=1,
                    help="when choosing its own path, play that many consecutive bars aligned to the track clock")
    ap.add_argument("--own-ramp", type=int, default=20)
    ap.add_argument("--latest-every", type=int, default=0, metavar="BATCHES",
                    help="with --keep-latest: also keep the newest state every this many batches between readings")
    ap.add_argument("--teach-returns", action="store_true",
                    help="returns only: a changed slot whose label none of the bars 2, 4, 8, 16 back holds is left open")
    ap.add_argument("--rollout-prime", type=int, default=8,
                    help="the held-out bars heard before each free-running continuation of the rollout reading")
    ap.add_argument("--rollout-length", type=int, default=16, help="the bars of each free-running continuation")
    ap.add_argument("--rollout-continuations", type=int, default=0, metavar="N",
                    help="full evaluations: the most continuations of each rollout reading (0: every opening)")
    ap.add_argument("--density-passages", type=int, default=4,
                    help="full evaluations: the held-out passages of the 64-bar free play sampled at 0.11")
    ap.add_argument("--density-greedy", choices=("every", "end"), default="every",
                    help="the greedy 64-bar free play at every full evaluation or at the run's end only")
    ap.add_argument("--lag-cuts", choices=("every", "end"), default="every",
                    help="the echo brain's per-lag reflex cuts at every full evaluation or at the run's end only")
    ap.add_argument("--schedule-file", default="", metavar="JSON",
                    help="read at every batch end (default <out>/schedule.json, when it exists): eval_minutes (the "
                         "marks ahead of the clock), rollout_continuations, density_passages, density_greedy, "
                         "lag_cuts; a running trainer takes a new evaluation schedule from it without a restart")
    ap.add_argument("--own-after", type=float, default=0.0, metavar="MINUTES",
                    help="no own play before this many minutes of the life; then --own-path with its ramp")
    ap.add_argument("--init", help="continue from this checkpoint (.npz) instead of a newborn brain")
    ap.add_argument("--cadence-modules", type=int, nargs="+", metavar="WIDTH",
                    help="compose Cadence's own default brain (Brain.compose with these module widths and the library's "
                         "defaults: no hierarchy, no direct route, no apical feedback) in place of the preset; the "
                         "observation and the slots stay the score's (learn_probe.py P5, the generalization test P8)")
    ap.add_argument("--cadence-apical", type=float, default=0.0,
                    help="with --cadence-modules: the apical compartment's share (0 = Cadence's default, none)")
    ap.add_argument("--new-score", action="store_true",
                    help="with --init: the checkpoint may come from another score file when its pool (hash) and its "
                         "slots are this run's (a re-read of the same material, e.g. v5c to v5d)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run of --init in place: its epochs, run time, curve and best readings go on")
    ap.add_argument("--error-patience", type=int, default=0,
                    help="stop after this many end-of-epoch evaluations in which neither the training error nor the "
                         "held-out error set a new low (0 = never)")
    ap.add_argument("--error-margin", type=float, default=0.002,
                    help="an error is a new low when it beats the lowest so far by this much")
    ap.add_argument("--migrate-recall-patch", action="store_true",
                    help="continue from the approved 0.103.0 recall repair on the official 0.103.1 release")
    ap.add_argument("--learn-groove", action="store_true",
                    help="learn when to replace each pattern and use those decisions on generated rollouts")
    ap.add_argument("--voice-dwell", type=int, nargs=4, metavar=("BEAT", "BASS", "LEAD", "AIR"),
                    help="minimum bars between major edits for each voice during self-play")
    ap.add_argument("--feedback", action="append", default=[], help="feedback.jsonl from the site: liked windows become extra lessons")
    ap.add_argument("--feedback-marks", action="append", default=[], metavar="DIR",
                    help="folders (or files) of the player's waveform marks (site/feedback/<id>.json): liked marks of "
                         "this brain's layout become extra lessons at --feedback-weight, disliked ones are counted and "
                         "set aside")
    ap.add_argument("--feedback-weight", type=float, default=3.0)
    args = ap.parse_args()
    layout_over = {}
    if args.layout:
        args.preset, layout_over = layouts.layout(args.layout)
    if args.preset in common.OPEN_LABEL_PRESETS:
        args.mask_unheard_targets = True
    if args.hear_first:
        # the bar is heard (acted on) before its lesson: Cadence warns that the lesson reads the moment itself
        warnings.filterwarnings("ignore", message="lesson on the observations the brain just acted on")

    if args.rollout_bars > 1 and not args.clock:
        ap.error("phrase-length rollouts need --clock")
    if args.reference_edits and not (args.synth_controls and args.own_path and args.own_sample):
        ap.error("--reference-edits requires the DSP composer and sampled own-path training")
    if args.synth_lesson_group and not (args.synth_controls and args.mask_unheard_targets):
        ap.error("--synth-lesson-group requires the DSP composer and open labels")
    if args.synth_vocab == "splitsteps" and not (args.synth_rows and args.mask_unheard_targets and args.synth_controls):
        ap.error("--synth-vocab splitsteps needs DSP synth rows and open labels")
    if args.learn_groove and not (args.reference_edits and args.clock and args.synth_lesson_group):
        ap.error("--learn-groove needs a bar clock, reference edits and explicit synth lesson groups")
    if args.voice_dwell and (min(args.voice_dwell) < 2 or not args.reference_edits or args.learn_groove):
        ap.error("--voice-dwell needs at least two bars per voice and sampled reference-edit lessons")
    if args.resume and not args.init:
        ap.error("--resume continues the checkpoint named by --init")
    if args.error_patience < 0 or args.error_margin < 0:
        ap.error("--error-patience and --error-margin cannot be negative")
    if args.migrate_recall_patch and not args.init:
        ap.error("--migrate-recall-patch requires a warm-start checkpoint")
    if not np.isfinite(args.motif_weight) or args.motif_weight < 1 or (args.motif_weight > 1 and not args.synth_controls):
        ap.error("--motif-weight requires the DSP composer and a finite multiplier of at least one")
    if not np.isfinite(args.development_weight) or args.development_weight < 1 or (
            args.development_weight > 1 and not (args.synth_controls and args.clock and args.window >= 16)):
        ap.error("--development-weight needs a finite multiplier of at least one, clocked DSP and windows of 16 bars or more")
    if not 0 <= args.detail_focus <= 1 or (args.detail_focus and not (
            args.steps and args.synth_vocab == "steps" and args.synth_controls and args.mask_unheard_targets)):
        ap.error("--detail-focus needs a fraction from zero to one, step-level drum and synth labels and open DSP lessons")
    if args.max_batches < 0:
        ap.error("--max-batches must be nonnegative")
    if not 0 <= args.arrangement_focus <= 1 or (args.arrangement_focus and not (args.synth_controls and args.mask_unheard_targets)):
        ap.error("--arrangement-focus needs a fraction from zero to one, DSP controls and open labels")
    over = dict(layout_over)
    for kv in args.set:
        k, v = kv.split("=", 1)
        over[k] = json.loads(v)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common.SYNTH_FILL = bool(args.synth_fill)
    common.HEAR_FIRST = bool(args.hear_first)
    if args.hear_parts and not args.hear_first:
        ap.error("--hear-parts belongs to --hear-first")
    common.HEAR_PARTS = bool(args.hear_parts)
    common.LOOKAHEAD = bool(args.lookahead)
    common.SECTION = bool(args.section)
    args.clock_offsets_card = None
    if args.clock_offsets:
        if not args.clock:
            ap.error("--clock-offsets needs --clock")
        raw = Path(args.clock_offsets).read_bytes()
        table = json.loads(raw)
        table = table.get("clock_offset", table)
        if not all(isinstance(v, int) and v >= 0 for v in table.values()):
            ap.error("--clock-offsets: every offset must be a whole number of bars, 0 or more")
        common.CLOCK_OFFSETS = {str(k): int(v) for k, v in table.items() if int(v)}
        args.clock_offsets_card = {"file": Path(args.clock_offsets).name, "sha": hashlib.sha256(raw).hexdigest()[:12],
                                   "shifted": len(common.CLOCK_OFFSETS)}
    common.ONOFF = bool(args.onoff)
    if args.reserve and not args.synth_controls:
        ap.error("--reserve belongs to the composer with --synth-controls")
    sid_book = None
    if args.sid_lessons:
        if args.reserve not in ("sid", "sidlead"):
            ap.error("--sid-lessons teaches the SID blocks of --reserve sid")
        sid_book = timbre.sid_book(json.loads(Path(args.sid_lessons).read_bytes()))
    if args.ear and not args.reserve:
        ap.error("--ear fills the ear inputs of --reserve")
    common.RESERVE, common.SID_LIVE, common.EAR_LIVE = args.reserve or False, bool(sid_book), bool(args.ear)
    if args.synth_controls:
        args.entry_scale = 1.0 if args.entry_scale is None else args.entry_scale
        args.line_scale = 1.0 if args.line_scale is None else args.line_scale
    common.ENTRY_SCALE, common.LINE_SCALE = args.entry_scale, args.line_scale
    kind, _, kk = args.synth_vocab.partition(":")            # rows:K | rhythm:K | factored:K_RHYTHM[:K_CONTOUR]
    kk, _, kc = kk.partition(":")
    common.SYNTH_VOCAB.update(kind=kind, k=int(kk or (256 if kind in ("rhythm", "factored") else 64)), contours=int(kc or 128))
    if args.audio_eval and not args.synth_controls:
        ap.error("--audio-eval requires --synth-controls")
    patch_book = None
    if args.synth_controls:
        if not args.synth_rows or args.beat_rows == args.steps or args.content or args.onoff or args.all_tracks:
            ap.error("--synth-controls requires --synth-rows, either --beat-rows or --steps, and separate held-out tracks. It uses its own gates and timbre inputs.")
        if Path(args.synth_controls).name != args.pool:
            ap.error("--pool must name the same file as --synth-controls")
        if args.calibration_bars < 5:
            ap.error("--calibration-bars must provide at least 17 bars across four corners")
        if args.composition_bars < 17:
            ap.error("--composition-bars must be at least 17")
        pool_bytes = Path(args.synth_controls).read_bytes()
        patch_pool = json.loads(pool_bytes)
        patch_book = timbre.teachers(patch_pool)
        if patch_book.get("samples"):
            expected = json.loads(Path(args.streams).read_bytes()).get("pool_sha256")
            if expected != hashlib.sha256(pool_bytes).hexdigest():
                ap.error("Sampler score and sample pack must match")
    synth_lesson_groups = set(args.synth_lesson_group or ())
    if synth_lesson_groups:
        sources = json.loads(Path(args.streams).read_bytes()).get("window_shares", {})
        if synth_lesson_groups - sources.keys():
            ap.error("--synth-lesson-group must name curated training sources")
    tracks = load_streams(args.streams, steps=args.steps, rows=True if args.beat_rows else None, clock=args.clock,
                          synth=True if args.synth_rows else None, patch_book=patch_book,
                          synth_lesson_groups=synth_lesson_groups or None, sid_book=sid_book)
    if common.CLOCK_OFFSETS:
        unknown = set(common.CLOCK_OFFSETS) - {t["id"] for t in tracks}
        if unknown:
            ap.error(f"--clock-offsets names {len(unknown)} passages the score does not carry, e.g. {sorted(unknown)[0]}")
        print(f"clock offsets: {len(common.CLOCK_OFFSETS)} passages shifted, "
              f"{sum(len(t['bars']) for t in tracks if t['id'] in common.CLOCK_OFFSETS)} bars", flush=True)
    pool_sha = hashlib.sha256(Path(args.synth_controls).read_bytes()).hexdigest()[:12] if patch_book else None
    if args.content:
        raw = Path(args.content).read_bytes()
        pool_sha = hashlib.sha256(raw).hexdigest()[:12]
        common.set_content(json.loads(raw))
    args.pool_sha = pool_sha
    args.streams_sha = hashlib.sha256(Path(args.streams).read_bytes()).hexdigest() if patch_book else None
    focus = [t for t in tracks if t["style"] == args.focus_style] if args.focus_style else tracks
    if not any(not t["held_out"] for t in focus) or not 0 <= args.focus_share < 1 or (args.focus_share and not args.focus_style):
        ap.error("focus style must have training tracks and its window share must be in [0,1)")
    # the reference reads the mood columns alone: with the ear a track's mood rows carry what the ear heard as well
    ref = corpus_reference([dict(t, mood=np.asarray(t["mood"])[:, :common.MOOD]) for t in focus])
    (out / "corpus_reference.json").write_text(json.dumps({k: v for k, v in ref.items() if k not in ("token_mood", "token_count")}, indent=1))
    train = tracks if args.all_tracks else [t for t in tracks if not t["held_out"]]
    if args.min_kick > 0 and common.ROWS:
        off = np.array([not np.asarray(r).any() for r in common.ROWS["kick"]])
        def kick_share(t):
            b = np.asarray(t["bars"]); beat = common.sounds(b)[:, 0] > 0
            return float((~off[b[:, L + common.MODS]] & beat).sum() / max(beat.sum(), 1))
        kept = [t for t in train if kick_share(t) >= args.min_kick]
        print(f"--min-kick {args.min_kick}: {len(kept)} of {len(train)} training tracks, {sum(len(t['bars']) for t in kept)} bars", flush=True)
        train = kept
    wins = windows(train, args.window, args.stride)
    focused_windows = windows([t for t in train if t["style"] == args.focus_style], args.window, args.stride)
    if args.focus_share and not focused_windows:
        ap.error("no full training windows in the requested focus style")
    extra = max(0, int(np.ceil((args.focus_share * len(wins) - len(focused_windows)) / (1 - args.focus_share))))
    rng = np.random.default_rng(args.seed)
    grouped = None
    if any(t.get("window_share") for t in train):
        if args.focus_share or args.all_tracks:
            ap.error("The curated source schedule requires its separate training split")
        grouped = SourceWindows(train, args.window, args.stride, rng)
    if not wins or args.batch < 1 or args.window < 2:
        ap.error("need training windows of at least two bars and a positive batch size")
    if args.mask_unheard_targets:
        if not patch_book:
            ap.error("--mask-unheard-targets requires composer DSP controls")
        if any(k in over and over[k] != v for k, v in common.AUDIBLE_RECALL.items()):
            ap.error("Open labels require recall without a direct token-to-motor route")
        over.update(common.AUDIBLE_RECALL)
    groove_fit = groove.fit(train, synth_lesson_groups) if args.learn_groove else None
    if args.init:
        from cadence import Brain, __version__
        init_card = Path(args.init).with_suffix(".json")
        previous = json.loads(init_card.read_text()) if init_card.exists() else {}
        old_settings = previous.get("settings") or {}
        is_hierarchy = args.preset in hierarchy.PRESETS
        if hierarchy.PRESETS.get(args.preset) != old_settings.get("architecture"):
            ap.error("Warm starts must keep the same architecture. Use a new brain to change the wiring.")
        if is_hierarchy and any(k in over and over[k] != old_settings.get(k)
                                for k in ("modules", "coordinator_depth", "learned_code",
                                          "pathway_reads", "pathway_recall")):
            ap.error("The hierarchical checkpoint already fixes its widths and coordinator depth.")
        # the coordinator layers and where the recall heads read, compared at their defaults too: a layout that
        # does not name them asks for one layer with the heads on it
        asked = {**common.PRESETS.get(args.preset, {}), **over}
        if is_hierarchy and any(asked.get(k, default) != old_settings.get(k, default)
                                for k, default in (("coordinator_depth", 1), ("context_recall", False))):
            ap.error("The hierarchical checkpoint already fixes its coordinator layers and where the recall heads read.")
        previous_patch = previous.get("cadence_patch_sha256")
        migrated = (args.migrate_recall_patch and previous.get("cadence_version") == "0.103.0"
                    and previous_patch == common.AUDITED_RECALL_PATCH
                    and __version__ == common.AUDITED_RECALL_RELEASE and not os.environ.get("CADANCE_CADENCE_PATCH_SHA"))
        if args.migrate_recall_patch and not migrated:
            ap.error("The source patch or installed release does not match the approved migration")
        # a later library patch that contains an earlier one takes the earlier one's checkpoints (PATCH_LINEAGE)
        current_patch = os.environ.get("CADANCE_CADENCE_PATCH_SHA")
        contained = previous_patch in PATCH_LINEAGE.get(current_patch, ())
        if previous_patch and previous_patch != current_patch and not migrated and not contained:
            ap.error("The warm-start checkpoint requires its recorded Cadence recall patch")
        brain = Brain.load(args.init, backend=args.backend, device=over.get("device"), precision=over.get("precision"))
        if "recall_key_norm" in over and over["recall_key_norm"] is not getattr(brain.recall, "key_norm", False):
            ap.error("Warm starts keep their recall key geometry. Start a new brain to change recall_key_norm.")
        settings = previous.get("settings", {})
        settings = {**settings, "init": Path(args.init).name}
        if migrated:
            settings["migrated_recall_patch_sha256"] = previous_patch
        recall_tuning = {key: over[key] for key in ("recall_amplitude", "recall_temperature") if key in over}
        if recall_tuning:
            brain.retune(**recall_tuning)
            settings.update(recall_tuning)
    elif args.cadence_modules:
        from cadence import Brain
        device = {"device": over["device"]} if args.backend == "torch" and over.get("device") else {}
        apical = {"apical": args.cadence_apical} if args.cadence_apical else {}
        recall_geometry = {key: over[key] for key in ("recall_key_norm",) if key in over}
        brain = Brain.compose(common.INPUTS, common.ACTIONS, slots=tuple(common.SIZES), modules=tuple(args.cadence_modules),
                              seed=args.seed, backend=args.backend, **device, **apical, **recall_geometry)
        settings = {"cadence_default": list(args.cadence_modules), **apical, **recall_geometry, "inputs": common.INPUTS,
                    "actions": common.ACTIONS, "slots": list(common.SIZES)}
    else:
        brain, settings = compose(args.preset, seed=args.seed, backend=args.backend, **over)
    if args.init and previous.get("echo_lags"):
        common.ECHO_LAGS = tuple(previous["echo_lags"])
    brain = common.echo(brain)                  # the echo shape hears the lag blocks through the wrapper
    if args.mask_unheard_targets and brain.recall is not None and brain.recall.tokens is not None:
        ap.error("This warm-start checkpoint uses token recall, which requires complete labels")
    if args.gain_bound:
        if args.gain_bound < 0 or brain.recall is None or not len(np.asarray(brain.recall.gains)):
            ap.error("--gain-bound needs a positive bound and a brain with the value path")
        # On the class: a brain rebuilt inside this trainer keeps it. Evaluation copies read it and never learn.
        type(brain.recall).GAIN_BOUND = float(args.gain_bound)
    if args.gain_cap and (args.gain_cap < 0 or brain.recall is None or not len(np.asarray(brain.recall.gains))):
        ap.error("--gain-cap needs a positive cap and a brain with the value path")
    if args.eta_scale != 1.0:                                  # 11-23: curves flat after 6-18% of a pass; is the step too small?
        cfg = brain.learner.config
        brain.retune(learning=learning_rate(cfg, cfg.eta * args.eta_scale))
        settings = {**settings, "eta_scale": args.eta_scale, "eta": brain.learner.config.eta}
    learner = {name: value for name, value in (("beta", args.beta), ("nudged_steps", args.nudged_steps),
                                               ("free_steps", args.free_steps), ("tolerance", args.tolerance),
                                               ("credit_window", args.credit_window),
                                               ("credit_transport", args.credit_transport),
                                               ("drive_budget", args.drive_budget),
                                               ("normalize", args.normalize), ("momentum", args.momentum),
                                               ("rehearsal", args.rehearsal))
               if value is not None}
    for item in args.learner_set:
        field_name, _, value = item.partition("=")
        if not hasattr(brain.learner.config, field_name):
            ap.error(f"--learner-set: this Cadence has no learner field {field_name}")
        learner[field_name] = json.loads(value)
    if learner.get("normalize") == 0:
        learner["normalize_relative"] = 0.0          # the projection-relative floor belongs to RMS steps
    if learner:
        for setting in ("credit_window", "credit_transport", "drive_budget"):
            if setting in learner and not hasattr(brain.learner.config, setting):
                ap.error(f"--{setting.replace('_', '-')} requires a Cadence build supporting it")
        brain.retune(learning=replace(brain.learner.config, **learner))
        settings = {**settings, "learner": learner}
    # the learner as it will teach, after any retune: its step proven in the log
    cfg = brain.learner.config
    print(f"{datetime.now().strftime('%H:%M:%S')} {args.name} learner eta {cfg.eta:g}, normalize {cfg.normalize:g}, "
          f"surprise {getattr(cfg, 'surprise', None)}, slots {getattr(cfg, 'surprise_slots', None)}, share "
          f"{getattr(cfg, 'surprise_share', None)}, drive budget {getattr(cfg, 'drive_budget', None)}"
          + (f"; set {json.dumps(learner)}" if learner else ""), flush=True)
    if not np.isfinite(args.sampler_weight) or args.sampler_weight < 1:
        ap.error("--sampler-weight must be a finite multiplier of at least one")
    if args.plateau < 0 or (args.plateau and not patch_book):
        ap.error("--plateau requires the composer score and a positive patience")
    if not 0 < args.plateau_floor <= 1:
        ap.error("--plateau-floor is a share of the starting rate, above zero and at most one")
    schedule = Plateau(brain.learner.config.eta, args.plateau, args.plateau_floor) if args.plateau else None
    layout = dict(streams=Path(args.streams).name, tokens_per_layer=common.T, mod_levels=common.M,
                  styles=common.STYLES, clock=bool(common.CLOCK), onoff=common.ONOFF, steps=args.steps,
                  beat_rows=common.ROWS, synth_rows=common.SYNTH, synth_fill=common.SYNTH_FILL, patch_book=patch_book)
    if args.streams_sha:
        layout["streams_sha"] = args.streams_sha
    if common.RESERVE:
        layout["reserve"] = common.RESERVE
    if common.SID_LIVE:
        layout["sid_live"] = True
    if common.EAR_LIVE:
        layout["ear"] = True
    if common.HEAR_PARTS:
        layout["hear_parts"] = True
    if common.LOOKAHEAD:
        layout["lookahead"] = True
    if common.SECTION:
        layout["section"] = True
    if common.ECHO_LAGS:
        layout["echo_lags"] = list(common.ECHO_LAGS)
    if args.init:
        old = json.loads(init_card.read_text()) if init_card.exists() else {}
        if (args.new_score and old.get("pool_sha") == args.pool_sha
                and list((old.get("settings") or {}).get("slots") or []) == list(common.SIZES)):
            # the same pool and the same slots: the score file may be a newer reading of the same material
            settings = {**settings, "score_change": {"from": old.get("streams"), "to": layout["streams"]}}
            old = {**old, "streams": layout["streams"], "streams_sha": layout.get("streams_sha")}
        if old.get("latest") and "patch_book" not in old and old.get("pool_sha") == args.pool_sha:
            # a light reading's card (<run>_latest) names its pool by its hash, not by the book
            old = {**old, "patch_book": patch_book}
        if (patch_book or old.get("patch_book")) and common.layout_key(old) != common.layout_key(layout):
            ap.error("warm-start checkpoint uses a different motor or corpus layout")
    liked = load_liked(args.feedback, common.layout_key(layout))
    marked, disliked = load_marks(args.feedback_marks, common.layout_key(layout)) if args.feedback_marks else ({}, [])
    for n, group in marked.items():
        liked.setdefault(n, []).extend(group)
    if args.feedback_marks:
        settings = {**settings, "feedback_marks": {"liked_windows": sum(len(v) for v in marked.values()),
                                                   "liked_bars": sum(n * len(v) for n, v in marked.items()),
                                                   "disliked_windows": len(disliked)}}
    log = open(out / f"{args.name}.log", "a")

    def say(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')} {args.name} {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    say(f"preset={args.preset} windows={len(wins)} batch={args.batch} liked windows={sum(len(v) for v in liked.values())} settings={json.dumps(over)}")
    if synth_lesson_groups:
        say(f"exact synth lessons from {', '.join(sorted(synth_lesson_groups))}")
    t0 = time.time()
    since_best = 0
    start_epoch = 1
    skip_batches = 0            # a resumed run inside a pass: the batches its state already had
    resume_fraction = 0.0
    evals = eval_set(focus, args.eval_tracks, args.eval_bars, 0, interior=bool(patch_book))
    state = {"best": -1.0, "best_test_error": float("inf"), "best_train_error": float("inf"), "agreed": 0, "seen": 0}
    # A brain that does not settle on a bar raises and issues no action. One such bar must not end a run that
    # has trained for hours (run n25d, trio: one held-out bar of a reading, 4.8 hours in). The window or the
    # reading is skipped and counted; when it keeps happening the last saved weights come back at half the rate.
    unsettled = {"lessons": 0, "readings": 0, "restores": 0, "streak": 0}

    def unsettled_error(error):
        return isinstance(error, RuntimeError) and "did not settle" in str(error)

    def unsettled_counts():
        counts = {key: unsettled[key] for key in ("lessons", "readings", "restores")}
        return counts if any(counts.values()) else None
    # Progress: the lowest training and held-out errors that counted, and the evaluations since either fell.
    # A brain that still learns its lessons keeps its place, whether or not the held-out tracks follow yet.
    progress = {"test": float("inf"), "train": float("inf"), "since": 0}
    # Listening: the held-out error of always writing the commonest bar, the evaluations so far, and whether
    # the run has been below that level. A run that stays above it does not use what it hears.
    hearing = {"level": None, "evaluations": 0, "listens": False}
    LISTEN_MARGIN = 0.02      # below the commonest-bar level by two points: a reading at the level is no listening

    def count_progress(test, train):
        lower = False
        for name, error in (("test", test), ("train", train)):
            if error is not None and error < progress[name] - args.error_margin:
                progress[name], lower = error, True
        progress["since"] = 0 if lower else progress["since"] + 1

    if args.resume:
        # The run goes on where its checkpoint stopped: the same clock, the same curve, and inside a pass the
        # batches after the state's own (its tag <pass>b<batch>: a newest state or a checkpoint inside a pass);
        # a state from a pass's end goes on with the next pass.
        start_epoch = int(float(previous.get("epoch") or 0)) + 1
        resume_fraction = float(previous.get("epoch") or 0) % 1      # a card without a tag: its pass's share
        inside = re.fullmatch(r"(\d+)b(\d+)", str(previous.get("latest") or "")) or \
            re.search(r"_e(\d+)b(\d+)\.npz$", str(args.init))
        if inside:
            start_epoch, skip_batches, resume_fraction = int(inside.group(1)), int(inside.group(2)), 0.0
        t0 -= 60 * float(previous.get("minutes") or 0)
        earlier = out / "metrics.jsonl"
        rows = [json.loads(line) for line in earlier.read_text().splitlines()] if earlier.is_file() else []
        for row in rows:
            error = (row.get("test_error") or {}).get("balanced")
            lessons = (row.get("train_error") or {}).get("balanced")
            if error is not None:
                state["best_test_error"] = min(state["best_test_error"], error)
            if lessons is not None:
                state["best_train_error"] = min(state["best_train_error"], lessons)
            if row.get("composition") is not None:
                state["best"] = max(state["best"], row["composition"])
            if "b" not in str(row.get("checkpoint")):          # readings between epochs do not count as checks
                count_progress(error, lessons)
            if row.get("blind_baseline"):
                hearing["level"] = row["blind_baseline"]["test"]["balanced"]
            elif error is not None and not row.get("light"):
                hearing["evaluations"] += 1
        if hearing["level"] is not None:
            hearing["listens"] = state["best_test_error"] < hearing["level"] - LISTEN_MARGIN
        if schedule is not None and rows:
            # The step's floor stays a quarter of the run's first step, and the schedule knows the bests so far.
            first = next((row["eta"] for row in rows if row.get("eta")), None)
            if first:
                schedule.floor = first / 4
            schedule.best_error = min(schedule.best_error, state["best_test_error"])
            schedule.best_music = max(schedule.best_music, state["best"])
    train_eval = [t for t in evals if not t["held_out"]]
    test_eval = [t for t in evals if t["held_out"]]
    # whole held-out tracks for the long free play (the evaluation passages are 48 bars): the longest, in a fixed order
    long_held = sorted((t for t in tracks if t.get("held_out")), key=lambda t: (-len(t["bars"]), t["id"]))
    copy_baseline = dict(train=prediction_error(None, train_eval), test=prediction_error(None, test_eval)) if patch_book else None

    if patch_book:
        if not any(len(t["bars"]) >= 17 for t in train_eval):
            ap.error("need training tracks with at least 17 evaluation bars")
        if not any(len(t["bars"]) >= 25 for t in test_eval):
            ap.error("need held-out tracks with at least 25 evaluation bars")
    metrics_path = out / "metrics.jsonl"
    prompt_protocol = 3 if args.development_weight > 1 else 2

    on_gpu = False
    if args.backend == "torch":
        import torch
        on_gpu = torch.cuda.is_available() and not str(over.get("device") or "").startswith("cpu")
        if on_gpu and args.gpu_cap > 0:
            share = args.gpu_cap * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, share))
            say(f"GPU memory capped at {args.gpu_cap:g} GB")

    def release():
        """Free what the copies of an evaluation left on the GPU.

        Every reading answers on a copy of the brain loaded from a snapshot. A dropped copy keeps its GPU memory
        until Python's collector finds it (the copies sit in reference cycles), and torch's allocator keeps
        every block it ever took. The copies of one evaluation therefore piled up at random, and the piles
        set new high marks for hours: a trainer went from 8.2 to 11 GB in five hours (run n25d). Measured on
        one brain (brain/memory_growth.py): each dropped copy left 0.2 to 0.3 GB until a collection, and the
        allocator held 2.7 GB where 1.4 GB was in use. Collecting and handing the free blocks back after
        every copy keeps both flat."""
        gc.collect()
        if on_gpu:
            torch.cuda.empty_cache()

    def gpu_memory():
        """What torch holds of the GPU, in GB: in use, held by its allocator (what nvidia-smi shows, less the
        CUDA context), and the most in use since the last reading."""
        if not on_gpu:
            return None
        row = {"in_use": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
               "held": round(torch.cuda.memory_reserved() / 2 ** 30, 2),
               "peak": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        torch.cuda.reset_peak_memory_stats()
        return row

    def load_snapshot(snapshot):
        from cadence import Brain
        release()           # the copy before this one is gone before the next arrives
        return common.echo(Brain.load(snapshot, backend=args.backend, device=over.get("device"), precision=over.get("precision")))

    def restore():
        """The last saved weights at half the present learning rate. Without a saved checkpoint the present
        weights stay and only the rate is halved."""
        nonlocal brain
        eta = brain.learner.config.eta * 0.5
        saved = state.get("last_saved")
        found = bool(saved) and Path(saved).is_file()
        if found:
            brain = load_snapshot(saved)
        brain.reset()
        brain.retune(learning=learning_rate(brain.learner.config, eta))
        unsettled["restores"] += 1
        unsettled["streak"] = 0
        say(f"the brain did not settle {args.unsettled_patience} times in a row: "
            + (f"weights of {Path(saved).name} restored" if found else "nothing saved yet, weights kept")
            + f", learning rate halved to {eta:.5g}")

    def note_unsettled(kind, where, error):
        unsettled[kind] += 1
        unsettled["streak"] += 1
        say(f"the brain did not settle in {where}; skipped. So far {unsettled['lessons']} lesson windows and "
            f"{unsettled['readings']} readings were skipped. {str(error)[:140]}")
        if unsettled["streak"] >= max(1, args.unsettled_patience):
            restore()
        else:
            brain.reset()

    def cap_gains():
        """The direct route's gains held at or under --gain-cap."""
        gains = np.asarray(brain.recall.gains, dtype=float)
        if (gains > args.gain_cap).any():
            brain.recall.gains = np.minimum(gains, args.gain_cap)

    def path_gains():
        """The value path's gains, one per read port: the first is the strength of the path that holds a bar."""
        gains = getattr(getattr(brain, "recall", None), "gains", None)
        return None if gains is None else [round(float(g), 4) for g in np.asarray(gains, dtype=float)]

    def turns(reading):
        """The error on the slots that change and on those that stay, for the log line."""
        change, stay = (reading.get("change") or {}).get("balanced"), (reading.get("stay") or {}).get("balanced")
        return "" if change is None or stay is None else f" (change {change:.3f}, stay {stay:.3f})"

    # The schedule of readings. With --read-share the light readings keep to their share of the run's time, and
    # with --eval-minutes the full evaluations come at the marks and at the end only.
    reading_clock = {"cost": None, "since": time.time()}
    try:
        marks = sorted(float(m) for m in args.eval_minutes.split(",") if m.strip())
    except ValueError:
        ap.error("--eval-minutes takes minutes, comma-separated")
    if not 0 <= args.read_share < 1:
        ap.error("--read-share is a share of the run's time, below 1")
    by_marks = bool(marks)

    def mark_due(elapsed):
        """Whether a full evaluation's minute has come; the marks behind the clock are used up."""
        due = False
        while marks and elapsed >= marks[0] * 60:
            marks.pop(0)
            due = True
        return due

    schedule_seen = {"mtime": None}

    def take_schedule(elapsed):
        """The schedule file, when it changed since the last look: the marks ahead of the clock and the full
        evaluation's options. True when it set the marks."""
        path = Path(args.schedule_file) if args.schedule_file else out / "schedule.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        if mtime == schedule_seen["mtime"]:
            return False
        schedule_seen["mtime"] = mtime
        try:
            asked = json.loads(path.read_text())
            took = {}
            for key, kind, allowed in (("rollout_continuations", int, None), ("density_passages", int, None),
                                       ("density_greedy", str, ("every", "end")), ("lag_cuts", str, ("every", "end"))):
                if key in asked:
                    value = kind(asked[key])
                    if allowed and value not in allowed or kind is int and value < 0:
                        raise ValueError(f"{key} {value!r}")
                    took[key] = value
            if "eval_minutes" in asked:
                took["eval_minutes"] = sorted(float(m) for m in asked["eval_minutes"] if float(m) * 60 > elapsed)
        except (OSError, ValueError, TypeError) as error:
            say(f"schedule file {path.name} not taken: {error}")
            return False
        for key, value in took.items():
            if key == "eval_minutes":
                marks[:] = value
            else:
                setattr(args, key, value)
        say(f"schedule file {path.name}: {json.dumps(took)}")
        return "eval_minutes" in took

    def reading_due():
        cost = reading_clock["cost"]
        return cost is None or time.time() - reading_clock["since"] >= cost * (1 - args.read_share) / args.read_share

    def keep_latest(snapshot, epoch, part, elapsed):
        """The weights of a light reading stay as the run's newest state, with the card a checkpoint has."""
        path = out / f"{args.name}_latest.npz"
        scratch = out / f".{args.name}_latest.npz"
        shutil.copyfile(snapshot, scratch)
        os.replace(scratch, path)
        card = card_for(args, settings, round(epoch - 1 + part / max(n_batches, 1), 2), elapsed, None, None,
                        {"pass": None, "checks": []}, path, False)
        card["latest"] = f"{epoch:03d}b{part:03d}"
        card["patch_book"] = patch_book          # a branch from this state checks its layout against the book
        metadata = out / f".{args.name}_latest.json"
        metadata.write_text(json.dumps(card, allow_nan=False))
        os.replace(metadata, path.with_suffix(".json"))
        state["last_saved"] = str(path)

    def rollout_reading(reader):
        """16-bar free-running continuations from the held-out openings: balanced, arrangement and changed-slot error."""
        pieces, played = rollouts(reader, test_eval, prime=args.rollout_prime, length=args.rollout_length,
                                  most=args.rollout_continuations)
        if not pieces:
            return None
        r = prediction_error(None, pieces, answers=played)
        return {"balanced": r["balanced"], "arrangement": r["arrangement"], "change": (r.get("change") or {}).get("balanced"),
                "stay": (r.get("stay") or {}).get("balanced"), "continuations": len(pieces), "bars": args.rollout_length}

    def coordination_readings(load, lags=True, spent=None):
        """The held-out readings of the intact brain's continuations, and of a copy with the coordinator's outgoing
        links to the pathways muted (its own arrangement slots kept): rollout and coordinator_cut. ``spent`` takes
        the seconds of each reading."""
        spent = {} if spent is None else spent
        began = time.time()
        out = {"rollout": rollout_reading(load())}
        spent["rollout"] = round(time.time() - began, 1)
        began = time.time()
        cut = load()
        if common.mute_coordination(cut):
            muted = load()
            common.mute_coordination(muted)
            out["coordinator_cut"] = {"test_error": prediction_error(cut, test_eval), "rollout": rollout_reading(muted)}
            spent["coordinator_cut"] = round(time.time() - began, 1)
        # two coordinator layers: every output of the upper coordinator muted (upper_cut), and its link down to the
        # phrase coordinator alone, its own arrangement slots kept (upper_to_phrase_cut)
        populations = brain.connectome.populations      # no copies loaded for a brain without an upper coordinator
        upper = "phrase_context" in populations and "association" in populations
        for reading, phrase_only in (("upper_cut", False), ("upper_to_phrase_cut", True)) if upper else ():
            began = time.time()
            cut = load()
            if common.mute_upper(cut, phrase_only):
                muted = load()
                common.mute_upper(muted, phrase_only)
                out[reading] = {"test_error": prediction_error(cut, test_eval), "rollout": rollout_reading(muted)}
                spent[reading] = round(time.time() - began, 1)
        began = time.time()
        cut = load()
        if common.mute_reflex(cut):            # the reflex arc muted: what the pathways hold without it
            muted = load()
            common.mute_reflex(muted)
            out["reflex_cut"] = {"test_error": prediction_error(cut, test_eval), "rollout": rollout_reading(muted)}
            spent["reflex_cut"] = round(time.time() - began, 1)
        began = time.time()
        if common.ECHO_LAGS and lags:           # the echo brain: each lag's arcs muted alone, teacher-forced
            out["reflex_cut_lags"] = {}
            for k in common.ECHO_LAGS:
                cut = load()
                common.mute_reflex(cut, lag=k)
                r = prediction_error(cut, test_eval)
                out["reflex_cut_lags"][str(k)] = {"balanced": r["balanced"], "stay": (r.get("stay") or {}).get("balanced"),
                                                 "change": (r.get("change") or {}).get("balanced")}
            spent["reflex_cut_lags"] = round(time.time() - began, 1)
        return out

    lesson_rows = {"attempted": 0.0, "agreed": 0.0, "slots_agreed": 0.0, "slots_taught": 0.0,
                   "returns_opened": 0.0, "returns_labelled": 0.0}

    def rows_taught():
        """The lesson rows since the last row of the curve and the share of them taught, then the count starts over."""
        rows = lesson_rows["attempted"]
        out = {"rows": int(rows), "answered_as_shown": int(lesson_rows["agreed"]),
               "taught_share": (1 - lesson_rows["agreed"] / rows) if rows else None,
               # the labelled slots the free answer already wrote as shown, per row (agreed_slots over the rows)
               "slots_as_shown_per_row": (lesson_rows["slots_agreed"] / rows) if rows else None,
               # with a per-slot surprise (a newer library's taught_slots): the slots each row was taught
               "taught_slots_per_row": (lesson_rows["slots_taught"] / rows) if rows and lesson_rows["slots_taught"] else None}
        if lesson_rows["returns_labelled"]:      # --teach-returns: the share of labelled slots left open as new values
            out["returns_opened_share"] = lesson_rows["returns_opened"] / lesson_rows["returns_labelled"]
        lesson_rows.update(attempted=0.0, agreed=0.0, slots_agreed=0.0, slots_taught=0.0, returns_opened=0.0,
                           returns_labelled=0.0)
        return out

    def light_reading(epoch, part, elapsed):
        """The next-bar error alone, between the full evaluations: a point on the curve and no checkpoint."""
        failed = None
        began = time.time()
        with tempfile.TemporaryDirectory(dir=out) as scratch:
            snapshot = Path(scratch) / "reading.npz"
            brain.save(snapshot)
            if args.keep_latest:
                keep_latest(snapshot, epoch, part, elapsed)
            row = dict(checkpoint=f"{epoch:03d}b{part:03d}", minutes=round(elapsed / 60, 2),
                       eta=brain.learner.config.eta, light=True, gains=path_gains())
            try:
                row.update(train_error=prediction_error(load_snapshot(snapshot), train_eval),
                           test_error=prediction_error(load_snapshot(snapshot), test_eval),
                           copy_baseline=copy_baseline)
                # a short form of the density reading: two held-out passages, 32 bars of free play
                row["density"] = density_rollout(load_snapshot(snapshot), long_held, prime=8, length=32, passages=2,
                                                 temperature=0.11)
            except RuntimeError as error:
                if not unsettled_error(error):
                    raise
                failed = error
        release()
        reading_clock["cost"], reading_clock["since"] = time.time() - began, time.time()
        row["gpu"] = gpu_memory()
        row["reading_seconds"] = round(reading_clock["cost"], 1)
        row["lesson_rows"] = rows_taught()
        if credit_sums:              # the teaching credit since the last full evaluation, read without clearing it
            m = credit_sums.get("credit_motor_contrast", 0) / max(credit_counts.get("credit_motor_contrast", 1), 1)
            if m:
                row["credit_so_far"] = {k[len("credit_"):]: s / credit_counts[k] / (m if k.endswith("_contrast") else 1)
                                        for k, s in credit_sums.items()
                                        if k.startswith("credit_") and (k.endswith("_contrast") or k.endswith("_slope"))
                                        and not k.endswith("voltage_contrast")}
        if failed is not None:
            note_unsettled("readings", f"the reading {row['checkpoint']}", failed)
            row["unsettled"] = True
        if unsettled_counts():
            row["unsettled_counts"] = unsettled_counts()
        with metrics_path.open("a") as report:
            report.write(json.dumps(row, allow_nan=False) + "\n")
        if failed is not None:
            return
        say(f"reading {row['checkpoint']} train {row['train_error']['balanced']:.3f}{turns(row['train_error'])}, "
            f"test {row['test_error']['balanced']:.3f}{turns(row['test_error'])}, path {(row['gains'] or [None])[0]}")

    if patch_book and not args.resume:
        with tempfile.TemporaryDirectory(dir=out) as scratch:
            initial = Path(scratch) / "initial.npz"
            brain.save(initial)
            baseline = dict(checkpoint="000", tag="warm_start" if args.init else "untrained", minutes=0,
                            train_error=prediction_error(load_snapshot(initial), train_eval),
                            test_error=prediction_error(load_snapshot(initial), test_eval),
                            copy_baseline=copy_baseline)
            if args.init:
                baseline["starting_checkpoint"] = Path(args.init).stem
            # the brain's layout travels with its curve, so Stats can draw it before a checkpoint is installed
            drawn = anatomy.describe({**layout, "settings": settings, "seed": args.seed})
            if drawn:
                baseline["anatomy"] = drawn
            # what one epoch holds, so Stats can say how often the run has heard its training set
            # the error of writing the commonest bar whatever was heard: a brain above it does not use what it hears
            average = commonest_bar([t for t in tracks if args.all_tracks or not t.get("held_out")])
            baseline["blind_baseline"] = dict(train=prediction_error(None, train_eval, constant=average),
                                              test=prediction_error(None, test_eval, constant=average))
            hearing["level"] = baseline["blind_baseline"]["test"]["balanced"]
            if args.layout:
                baseline["layout"] = args.layout
            if os.environ.get("CADANCE_BIRTH_INTACT") == "1":       # a birth probe: the intact rollout only, no cuts
                baseline["rollout"] = rollout_reading(load_snapshot(initial))
            else:
                baseline.update(coordination_readings(lambda: load_snapshot(initial), lags=False))   # per-lag cuts at evaluations
            baseline["hear_first"] = common.HEAR_FIRST
            baseline["gains"] = path_gains()
            release()
            baseline["gpu"] = gpu_memory()
            if args.gain_bound:
                baseline["gain_bound"] = args.gain_bound
            baseline["lessons"] = dict(
                window=args.window, batch=args.batch, unique_windows=len(wins),
                epoch_windows=sum(grouped.counts.values()) if grouped else len(wins) + extra,
                training_bars=sum(len(t["bars"]) for t in tracks if args.all_tracks or not t.get("held_out")))
        if baseline["test_error"]["balanced"] is not None:
            state["best_test_error"] = baseline["test_error"]["balanced"]
        with metrics_path.open("a") as report:
            report.write(json.dumps(baseline, allow_nan=False) + "\n")
        count_progress(baseline["test_error"]["balanced"], baseline["train_error"]["balanced"])
        say(f"starting error train {baseline['train_error']['balanced']:.3f}{turns(baseline['train_error'])}, "
            f"test {baseline['test_error']['balanced']:.3f}{turns(baseline['test_error'])}")
    if args.resume:
        say(f"resuming at epoch {start_epoch} after {(time.time() - t0) / 60:.1f} min, lowest errors train "
            f"{state['best_train_error']:.3f} and test {state['best_test_error']:.3f}, "
            f"{progress['since']} evaluations since either fell")

    credit_sums = {}
    credit_counts = {}
    credit_warnings = {"count": 0, "lessons": 0, "messages": []}
    lesson_rows.update(attempted=0.0, agreed=0.0, slots_agreed=0.0, slots_taught=0.0, returns_opened=0.0, returns_labelled=0.0)
    # Cadence's warning that motor teaching has not reached a region over the credit window: every one is shown in
    # the log and counted for the next evaluation's hidden_credit
    warnings.filterwarnings("always", message="motor teaching contrast has not measurably reached")
    shown = warnings.showwarning

    def counted(message, category, filename, lineno, file=None, line=None):
        if str(message).startswith("motor teaching contrast has not measurably reached"):
            credit_warnings["count"] += 1
            credit_warnings["messages"].append(str(message))
        return shown(message, category, filename, lineno, file, line)
    warnings.showwarning = counted

    def note_credit(report):
        # the rows the lessons met and the rows they already answered as shown (learning_surprise: no lesson there)
        lesson_rows["attempted"] += float(report.get("attempted_presentations") or 0)
        lesson_rows["agreed"] += float(report.get("agreed_presentations") or 0)
        lesson_rows["slots_agreed"] += float(report.get("agreed_slots") or 0)
        lesson_rows["slots_taught"] += float(report.get("taught_slots") or 0)
        if any(key.startswith("credit_") for key in report):
            credit_warnings["lessons"] += 1
        for key, value in report.items():
            if (key.startswith("credit_") or key == "drive_limited_projections") and np.isfinite(value):
                credit_sums[key] = credit_sums.get(key, 0.0) + float(value)
                credit_counts[key] = credit_counts.get(key, 0) + 1

    def evaluate_and_save(epoch, part, elapsed, final=False):
        # Even greedy acts update the learned code. Evaluate independent copies of
        # the pre-evaluation snapshot so validation cannot teach the next batch.
        failed = None
        with tempfile.TemporaryDirectory(dir=out) as scratch:
            snapshot = Path(scratch) / "training.npz"
            brain.save(snapshot)
            try:
                better, metrics = evaluate_snapshot(snapshot, epoch, part, elapsed, final)
            except Exception as error:
                if unsettled_error(error):
                    failed, better, metrics = error, False, None
                elif snapshot.is_file():
                    recovery = out / "evaluation-recovery.npz"
                    metadata = snapshot.with_suffix(".json")
                    metadata.write_text(json.dumps({**layout, "settings": settings}, allow_nan=False))
                    os.replace(snapshot, recovery)
                    os.replace(metadata, recovery.with_suffix(".json"))
                    say(f"evaluation failed; weights kept at {recovery.name}")
                    raise
                else:
                    raise
        release()
        if failed is not None:
            # No checkpoint and no verdict from an evaluation that did not finish: a row on the curve says so.
            tag = f"{epoch:03d}" if part is None else f"{epoch:03d}b{part:03d}"
            note_unsettled("readings", f"the evaluation {tag}", failed)
            with metrics_path.open("a") as report:
                report.write(json.dumps(dict(checkpoint=tag, minutes=round(elapsed / 60, 2), eta=brain.learner.config.eta,
                                             unsettled=True, unsettled_counts=unsettled_counts(), gains=path_gains(),
                                             gpu=gpu_memory()), allow_nan=False) + "\n")
            return False
        unsettled["streak"] = 0         # a clean evaluation: the brain settles on every bar it was asked
        if metrics is not None:
            # Cadence's low-contrast credit warnings since the last evaluation, as written (empty when none)
            metrics["credit_warnings"] = list(credit_warnings["messages"])
            metrics["lesson_rows"] = rows_taught()
            credit_warnings["messages"].clear()
            if credit_sums:
                metrics["hidden_credit"] = {
                    k: {"mean": s / credit_counts[k], "lessons": credit_counts[k]}
                    for k, s in credit_sums.items()
                }
                # the two coordinators at a glance: each one's share of the motor's teaching contrast (the ratio
                # of the means over the measured lessons) and its mean slope in the lessons' free phase
                motor_contrast = credit_sums.get("credit_motor_contrast", 0.0) / max(credit_counts.get("credit_motor_contrast", 0), 1)
                coordinators = {}
                for role, region in (("upper", "phrase_context"), ("phrase", "association")):
                    key = f"credit_{region}_contrast"
                    if key in credit_sums and motor_contrast > 0:
                        slope = f"credit_{region}_slope"
                        coordinators[role] = {
                            "share": credit_sums[key] / credit_counts[key] / motor_contrast,
                            "slope": credit_sums[slope] / credit_counts[slope] if slope in credit_sums else None}
                if coordinators:
                    metrics["coordinator_credit"] = coordinators
                metrics["hidden_credit"]["warnings"] = credit_warnings["count"]
                metrics["hidden_credit"]["measured_lessons"] = credit_warnings["lessons"]
                credit_warnings["count"] = credit_warnings["lessons"] = 0
                calibration = getattr(brain.learner, "last_credit_calibration", {})
                if calibration and "credit_calibration_applied" in credit_sums:
                    metrics["credit_calibration"] = calibration
                credit_sums.clear()
                credit_counts.clear()
            if unsettled_counts():
                metrics["unsettled_counts"] = unsettled_counts()
            metrics["gpu"] = gpu_memory()
            if metrics["gpu"]:
                say(f"GPU memory in use {metrics['gpu']['in_use']} GB, held {metrics['gpu']['held']} GB, "
                    f"peak in the evaluation {metrics['gpu']['peak']} GB")
            if schedule is not None:
                cfg = brain.learner.config
                eta = schedule.next_rate(cfg.eta, metrics["test_error"]["balanced"], metrics["composition"])
                if eta != cfg.eta:
                    brain.reset()
                    brain.retune(learning=learning_rate(cfg, eta))
                    say(f"validation plateau, learning rate {cfg.eta:.5g} to {eta:.5g}")
                metrics["next_eta"] = eta
            hearing["evaluations"] += 1
            if (hearing["level"] is not None and not hearing["listens"]
                    and state["best_test_error"] < hearing["level"] - LISTEN_MARGIN):
                hearing["listens"] = True
                metrics["listens"] = True
                say(f"listens: held-out error {state['best_test_error']:.3f} is below the commonest-bar level "
                    f"{hearing['level']:.3f} after {hearing['evaluations']} evaluations")
            with metrics_path.open("a") as report:
                report.write(json.dumps(metrics, allow_nan=False) + "\n")
        return better

    def deaf():
        """The run had its evaluations and never got below the commonest-bar level."""
        if not args.listen_patience or hearing["level"] is None or hearing["listens"]:
            return False
        if hearing["evaluations"] < args.listen_patience:
            return False
        say(f"stop: does not listen. Held-out error {state['best_test_error']:.3f} is not below the commonest-bar "
            f"level {hearing['level']:.3f} after {hearing['evaluations']} evaluations")
        return True

    def evaluate_snapshot(snapshot, epoch, part, elapsed, final):
        """Held-out reading, replay, free-run gate on the evaluation set; save when better or on schedule."""
        def load():
            return load_snapshot(snapshot)
        spent = {}                       # the seconds of each reading, on the curve (evaluation_seconds)
        clock = {"at": time.time()}

        def lap(name):
            now = time.time()
            spent[name] = round(now - clock["at"], 1)
            clock["at"] = now
        reading = held_out(load(), evals)
        lap("held_out")
        rp = replay(load(), evals, bars=args.replay_bars)
        lap("replay")
        reading["replay"] = rp
        metric = rp["mean_all"] if args.all_tracks else reading["score"]
        def calibration():
            return calibrate(load, [t for t in evals if not t["held_out"]], ref, bars_per_corner=args.calibration_bars,
                             groove_fit=groove_fit, dwell_bars=args.voice_dwell,
                             **({"temps": tuple(float(t) for t in args.temps.split(","))} if args.temps else {}))
        metrics = None
        best_prediction = False
        if patch_book:
            temp, temp_scan = calibration()
            lap("calibration")
            audition = composition.rendered_comparison({"pool": args.pool}, patch_pool, snapshot.parent) if args.audio_eval else None
            trace = RecallTrace() if args.recall_probe else None
            continuation = composition.compare(load, focus, temp, args.composition_bars, audition=audition,
                                               recall_trace=trace, groove_fit=groove_fit, dwell_bars=args.voice_dwell,
                                               prompt_protocol=prompt_protocol)
            if trace is not None:
                continuation["recall_attention"] = trace.summary()
                continuation["recall_max_slot_share"] = trace.max_slot_share
            phrase = (continuation.get("phrase_events") or {}).get("score")
            if (args.motif_weight > 1 or groove_fit is not None) and phrase is not None:
                continuation["pattern_and_audio_score"] = continuation["selection_score"]
                continuation["selection_score"] = .8 * continuation["pattern_and_audio_score"] + .2 * phrase
            metric = continuation["selection_score"]
            say(f"composition {metric:.3f}, symbolic {continuation['score']:.3f}, "
                f"hold {continuation['hold_score']:.3f}, "
                f"{continuation['prompts_above_hold']}/{len(continuation['runs'])} prompts above hold")
            lap("composition")
        better = metric > state["best"]
        tag = f"{epoch:03d}" if part is None else f"{epoch:03d}b{part:03d}"
        if patch_book:
            metrics = dict(checkpoint=tag, minutes=round(elapsed/60, 2), eta=brain.learner.config.eta,
                           train_error=prediction_error(load(), train_eval), test_error=prediction_error(load(), test_eval),
                           composition=metric, symbolic=continuation["score"], hold=continuation["hold_score"],
                           phrase_events=continuation.get("phrase_events"), development=continuation.get("development"),
                           prompt_protocol=continuation["prompt_protocol"],
                           temperature=temp, copy_baseline=copy_baseline, gains=path_gains())
            lap("teacher_forced")
            metrics["evaluation_seconds"] = spent        # filled on to the gate's free runs
            cuts = {}
            metrics.update(coordination_readings(load, lags=args.lag_cuts == "every" or final, spent=cuts))
            lap("coordination")
            spent["coordination_parts"] = cuts
            # long free play against the material: the density of what the brain adds when it hears itself
            metrics["density"] = density_rollout(load(), long_held, prime=8, length=64, passages=args.density_passages,
                                                 temperature=0.11)
            lap("density")
            if args.density_greedy == "every" or final:
                metrics["density_greedy"] = density_rollout(load(), long_held, prime=8, length=64,
                                                            passages=args.density_passages)
                lap("density_greedy")
            test_error = metrics["test_error"]["balanced"]
            if test_error is not None and test_error < state["best_test_error"]:
                best_prediction = True
                state["best_test_error"] = test_error
            train_error = metrics["train_error"]["balanced"]
            if train_error is not None:
                state["best_train_error"] = min(state["best_train_error"], train_error)
            if trace is not None:
                metrics["recall_max_slot_share"] = trace.max_slot_share
            say(f"teacher-forced error train {metrics['train_error']['balanced']:.3f}{turns(metrics['train_error'])}, "
                f"test {metrics['test_error']['balanced']:.3f}{turns(metrics['test_error'])}, eta {metrics['eta']:.5g}")
        say(f"epoch {tag} {elapsed/60:.1f} min post-lesson sound agreement {state['agreed']/max(state['seen'],1):.3f} held-out score {reading['score']:.3f} mods {reading['mods'] or 0:.2f} replay all {rp['mean_all']} (hold {rp['hold_all']}) mods {rp['mods_all']} trained {rp['mean_trained']} "
            + " ".join(f"{n}:keep {reading[n]['keep'] or 0:.2f}/chg {reading[n]['change'] or 0:.2f}/bigram {reading[n]['bigram'] or 0:.2f}" for n in ("beat", "bass", "lead", "air")))
        if better:
            state["best"] = metric
        patience_reached = not better and args.patience and since_best + 1 >= args.patience
        if better or best_prediction or part is not None or final or patience_reached or epoch % args.save_every == 0:
            path = out / f"{args.name}_e{tag}.npz"
            if not patch_book:
                temp, temp_scan = calibration()
            bars_s, mp = free_run(load(), evals, bars_per_corner=args.gate_bars, temperature=temp,
                                  groove_fit=groove_fit, dwell_bars=args.voice_dwell)
            g_s = gate(bars_s, mp, ref)
            lap("gate_sampled")
            if metrics is not None:
                # how the brain's own playing compares with the corpus (change rates, hold lengths): on the curve,
                # so Stats shows it beside the next-bar error
                metrics["gate"] = {"pass": g_s["pass"], "checks": g_s["checks"]}
            bars_g, mp = free_run(load(), evals, bars_per_corner=args.gate_bars, greedy=True)
            g_g = gate(bars_g, mp, ref)
            lap("gate_greedy")
            card = card_for(args, settings, epoch if part is None else round(epoch - 1 + part / max(n_batches, 1), 2), elapsed, reading, g_s, g_g, path, better)
            card["free_run"] = bars_s.tolist()
            if unsettled_counts():
                card["unsettled"] = unsettled_counts()     # windows and readings skipped, weights restored
            card["temperature"], card["temperature_scan"] = temp, temp_scan
            import cadence
            card["cadence_version"] = cadence.__version__
            if patch_sha := os.environ.get("CADANCE_CADENCE_PATCH_SHA"):
                card["cadence_patch_sha256"] = patch_sha
            card["patch_book"] = patch_book
            card["training"] = {k: v for k, v in vars(args).items() if k not in ("out", "feedback")}
            card["training"].update(unique_windows=len(wins), extra_focus_windows=extra,
                                    windows_per_group=grouped.counts if grouped else None)
            if patch_book:
                card["errors"] = metrics
                card["composition"] = continuation
                if groove_fit is not None:
                    card["groove"] = groove_fit
                if args.voice_dwell:
                    card["voice_dwell"] = args.voice_dwell
                card["selection_metric"] = "composition.selection_score"
                if args.recall_probe:
                    cleared = composition.compare(load, focus, temp, args.composition_bars, clear_recall=True,
                                                  groove_fit=groove_fit, dwell_bars=args.voice_dwell,
                                                  prompt_protocol=prompt_protocol)
                    def mute():
                        counterfactual = load()
                        counterfactual.retune(recall_amplitude=0.0)
                        return counterfactual
                    muted = composition.compare(mute, focus, temp, args.composition_bars, groove_fit=groove_fit,
                                               dwell_bars=args.voice_dwell, prompt_protocol=prompt_protocol)
                    card["recall_probe"] = dict(intact=continuation["score"], cleared=cleared["score"],
                                                difference=continuation["score"] - cleared["score"],
                                                muted=muted["score"], drive_difference=continuation["score"] - muted["score"],
                                                phrase_events={"intact": continuation.get("phrase_events"),
                                                               "cleared": cleared.get("phrase_events"),
                                                               "muted": muted.get("phrase_events")},
                                                development={"intact": continuation.get("development"),
                                                             "cleared": cleared.get("development"),
                                                             "muted": muted.get("development")})
                if better and args.audio_eval:
                    mix = path.with_suffix(".wav")
                    os.replace(snapshot.parent / "generated.wav", mix)
                    card["audition"] = mix.name
            metadata = snapshot.with_suffix(".json")
            metadata.write_text(json.dumps(card, allow_nan=False))
            os.replace(snapshot, path)
            state["last_saved"] = str(path)
            os.replace(metadata, path.with_suffix(".json"))
            say(f"saved {path.name} T={temp} gate {'PASS' if g_s['pass'] else 'fail'} (greedy {'pass' if g_g['pass'] else 'fail'}) "
                + "; ".join(f"{c['check']}={c['value']}" for c in g_s["checks"]))
        return better, metrics

    epoch_size = sum(grouped.counts.values()) if grouped else len(wins) + extra
    n_batches = (epoch_size + args.batch - 1) // args.batch
    if resume_fraction and not skip_batches:
        skip_batches = int(round(resume_fraction * n_batches))
    if args.evals_per_epoch < 0:
        ap.error("--evals-per-epoch must be nonnegative")
    if by_marks:
        say(f"full evaluations at minutes {', '.join(f'{m:g}' for m in marks)} and at the end; {n_batches} batches a pass"
            + (f"; light readings within {args.read_share:.0%} of the time" if args.read_share else ""))
    elif args.evals_per_epoch:
        args.save_batches = max(1, -(-n_batches // args.evals_per_epoch))
        say(f"full evaluations every {args.save_batches} of {n_batches} batches and at the end of each pass")
    focus_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 17])) if args.arrangement_focus else None
    detail_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 21])) if args.detail_focus else None
    transpose_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 23])) if args.transpose else None
    interest = None
    if args.interest_weight != 1.0:
        interest = interest_weights(args.streams, train, args.interest_weight)
        if interest is None or args.interest_weight <= 0:
            ap.error("--interest-weight needs a positive weight and a score with an interest reading (v4)")
        pool_windows = [w for g in grouped.groups.values() for w in g] if grouped else wins
        heavy = sum(window_weight(interest, w[0]) != 1.0 for w in pool_windows)
        say(f"interest weight {args.interest_weight:g} on {heavy} of {len(pool_windows)} lesson windows "
            f"(passages at or above the median interest)")
        settings = {**settings, "interest_weight": args.interest_weight}
    if args.transpose and not (common.SYNTH and common.SYNTH.get("kind") == "steps"):
        ap.error("--transpose moves step lines (--synth-vocab steps)")
    batches_done = 0
    # a brain with the direct route takes a label for every slot; the others may be left one open
    open_labels = getattr(getattr(brain, "recall", None), "tokens", None) is None
    ON = list(range(common.SLOTS - L, common.SLOTS)) if common.ONOFF else []    # the on/off slots count with their sounds
    if args.event_slots == "arrangement" and not common.STEPS:
        ap.error("arrangement weighting needs --beat-rows or --steps")
    if args.event_slots == "section" and not common.SECTIONS:
        ap.error("section weighting needs --section")
    kick_present = np.asarray(common.ROWS["kick"]).any(axis=1) if common.ROWS else None
    has_sampler = patch_book and patch_book.get("samples")
    drum_start = L + common.MODS
    kick_slots = list(range(drum_start, drum_start + max(1, common.STEPS // len(common.DRUM_ROWS))))
    EV = {"arrangement": [], "section": [], "kick": [0] + kick_slots + ON[:1],
          "beat": [0] + list(range(drum_start, drum_start + common.STEPS)) + ON[:1],
          "tokens": list(range(L)) + ON, "sounds+kick": list(range(L)) + kick_slots + ON}[args.event_slots]
    if start_epoch > 1:
        # Fresh lesson orders for the epochs that follow: the generators are shared, so they change in place.
        for generator, stream in ((rng, 0), (focus_rng, 17), (detail_rng, 21)):
            if generator is not None:
                generator.bit_generator.state = np.random.default_rng(
                    np.random.SeedSequence([args.seed, stream, start_epoch])).bit_generator.state
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_windows = wins + [focused_windows[i] for i in rng.integers(len(focused_windows), size=extra)] if extra else wins
        synth_lessons = None
        if grouped:
            sources = grouped.draw(with_groups=True)
            epoch_windows = [window for _, window in sources]
            if synth_lesson_groups:
                synth_lessons = np.array([source in synth_lesson_groups for source, _ in sources])
        order = rng.permutation(len(epoch_windows))
        agreed = slots_seen = 0
        if skip_batches:
            say(f"resuming inside pass {epoch}: its first {skip_batches} batches were taught before the stop")
        for b0 in range(0, len(order), args.batch):
            if skip_batches:
                skip_batches -= 1
                continue
            idx = order[b0:b0 + args.batch]
            bars = np.stack([epoch_windows[i][0] for i in idx])        # (batch, window, 4)
            row_interest = (np.array([window_weight(interest, epoch_windows[i][0]) for i in idx])
                            if interest is not None else None)
            if transpose_rng is not None:
                bars = transpose_lines(bars, transpose_rng.integers(-args.transpose, args.transpose + 1, size=len(bars)))
            mood = np.stack([epoch_windows[i][1] for i in idx])
            # score v4: the drum steps the reader left open are no lesson (common.open_steps)
            opened = (np.stack([epoch_windows[i][2] for i in idx])
                      if all(len(epoch_windows[i]) > 2 for i in idx) else None)
            brain.reset()
            p_own = args.own_path * min(1.0, epoch / max(args.own_ramp, 1))
            if args.own_after and (time.time() - t0) / 60 < args.own_after:
                p_own = 0.0                                   # the first minutes hear the score alone
            heard = bars[:, 0]
            own = np.zeros(len(bars), dtype=bool)
            reprises = common.motif_return_ages(bars) if args.motif_weight > 1 or groove_fit is not None else None
            changes = common.arrangement_transitions(bars, kick_present) if focus_rng is not None else None
            if groove_fit is not None:
                controllers = [groove.Groove(groove_fit, [args.seed, epoch, b0, i]) for i in range(len(bars))]
            elif args.voice_dwell:
                controllers = [groove.VoiceDwell(args.voice_dwell) for _ in bars]
            else:
                controllers = None
            clock = slice(common.MOOD - common.CLOCK, common.MOOD)      # the bar clock's bits within a mood row
            phases = mood[..., clock][..., :3] @ np.array([1, 2, 4]) if controllers else None
            developments = None
            if args.development_weight > 1:
                starts = mood[:, 0, clock][:, :2] @ np.array([1, 2])
                developments = np.stack([np.logical_or.reduce(list(composition.development_points(
                    stream, int(start), synth=synth_lessons is None or synth_lessons[idx[i]]).values()))
                    for i, (stream, start) in enumerate(zip(bars, starts))])
            history = []
            try:
                for t in range(args.window - 1):
                    x = observe(heard, mood[:, t + 1])
                    w = None
                    if args.change_weight != 1.0:
                        changed = (bars[:, t + 1] != bars[:, t]).any(axis=1)
                        w = np.where(changed, args.change_weight, 1.0)
                    if args.event_weight != 1.0:
                        if args.event_slots == "section":
                            # the first bar of a new section, and the last bar before one
                            s = common.section_slot()
                            here, nxt = bars[:, t, s], bars[:, t + 1, s]
                            after = bars[:, t + 2, s] if t + 2 < bars.shape[1] else nxt
                            event = (here != nxt) | (nxt != after)
                        elif args.event_slots == "arrangement":
                            current = common.sounds(bars[:, t]) > 0
                            following = common.sounds(bars[:, t + 1]) > 0
                            kick = common.audible_kick(bars[:, t], kick_present)
                            next_kick = common.audible_kick(bars[:, t + 1], kick_present)
                            event = (current != following).any(axis=1) | (kick != next_kick)
                        else:
                            event = (bars[:, t + 1][:, EV] != bars[:, t][:, EV]).any(axis=1)
                        w = (np.ones(len(bars)) if w is None else w) * np.where(event, args.event_weight, 1.0)
                    if has_sampler and args.sampler_weight != 1:
                        sample = common.sample_voices(bars[:,t+1]).any(axis=1)
                        w = (1 if w is None else w) * np.where(sample, args.sampler_weight, 1.)
                    if developments is not None:
                        w = np.maximum(1 if w is None else w, np.where(developments[:, t + 1], args.development_weight, 1.))
                    if row_interest is not None:
                        w = row_interest if w is None else w * row_interest
                    target = bars[:, t + 1]
                    if args.reference_edits and own.any():
                        target = common.reference_edits(bars[:, t], target, heard, own)
                    if opened is not None:
                        target = common.open_steps(target, heard, opened[:, t + 1], leave=open_labels)
                    if reprises is not None:
                        history.append(heard)
                        ages = reprises[:, t + 1]
                        if synth_lessons is not None:
                            ages = np.where(synth_lessons[idx, None], ages, 0)
                        returning = ages.any(axis=1)
                        if returning.any():
                            target = common.heard_reprises(target, ages, history)
                            if args.motif_weight > 1:
                                w = np.maximum(1.0 if w is None else w, np.where(returning, args.motif_weight, 1.0))
                    if args.mask_unheard_targets:
                        target = common.audible_targets(target)
                    if synth_lessons is not None:
                        target[~synth_lessons[idx], L + common.MODS + common.STEPS:common.LIVE] = -1
                    if changes is not None:
                        active_changes = changes[:, t + 1] & (focus_rng.random((len(bars), 1)) < args.arrangement_focus)
                        if active_changes.any():
                            target = common.focus_arrangement(target, active_changes)
                    if detail_rng is not None:
                        focused = ~own & (detail_rng.random(len(bars)) < args.detail_focus)
                        focused &= (common.sounds(bars[:, t]) == common.sounds(bars[:, t + 1])).all(axis=1)
                        if developments is not None:
                            focused &= ~developments[:, t + 1]
                        if changes is not None:
                            focused &= ~active_changes.any(axis=1)
                        if focused.any():
                            target = common.focus_detail(target, focused, detail_rng)
                    if args.teach_returns:
                        # returns only: a changed slot (its label differs from the bar heard) whose label is the value
                        # of none of the bars 2, 4, 8, 16 back in the window is left open; holds and returns are taught
                        target = np.array(target, dtype=int, copy=True)
                        labelled = target >= 0
                        changed = labelled & (target != np.asarray(heard))
                        returning = np.zeros_like(changed)
                        for back in (2, 4, 8, 16):
                            if t + 1 - back >= 0:
                                returning |= target == bars[:, t + 1 - back]
                        drop = changed & ~returning & ~own[:, None]      # own-play rows: every wrong slot taught
                        target[drop] = -1
                        lesson_rows["returns_opened"] += float(drop.sum())
                        lesson_rows["returns_labelled"] += float(labelled.sum())
                    common.hear(brain, x)
                    note_credit(brain.lesson(x, target, weights=w))
                    if args.gain_cap:
                        cap_gains()
                    if common.HEAR_FIRST and not (p_own or controllers):
                        # The bar was heard and taught. Nothing here uses the answer, so no second moment is made.
                        heard = bars[:, t + 1]
                        continue
                    if common.HEAR_FIRST:
                        kept = brain.recall.snapshot()
                    answer = np.array(brain.act(x, greedy=True)).reshape(len(bars), common.SLOTS)
                    if common.HEAR_FIRST:
                        brain.recall.restore(kept)      # the answer leaves no second copy of the bar in the memory
                    agreed += int((common.sounds(answer) == common.sounds(bars[:, t + 1])).sum()); slots_seen += len(bars) * L
                    state["agreed"], state["seen"] = agreed, slots_seen
                    if args.rollout_bars == 1:
                        own = rng.random(len(bars)) < p_own
                    else:
                        bits = args.rollout_bars.bit_length() - 1
                        boundary = ~mood[:, t + 1, clock][:, :bits].any(axis=1)
                        own[boundary] = rng.random(int(boundary.sum())) < p_own
                    if args.own_sample > 0 and own.any():         # ... as the player plays it: sampled, not the greedy commonest value
                        act = common.motor_activity(brain)
                        for i in np.nonzero(own)[0]:
                            answer[i] = common.sample_slots(act[i], heard[i], args.own_sample, rng)
                    if controllers:
                        for i, controller in enumerate(controllers):
                            if own[i]:
                                answer[i] = controller.play(heard[i], answer[i], int(phases[i, t + 1]))
                            else:
                                controller.hear(heard[i], bars[i, t + 1])
                    heard = np.where(own[:, None], answer, bars[:, t + 1])
            except RuntimeError as error:
                if not unsettled_error(error):
                    raise
                # the rest of this batch's windows is skipped; the next batch starts afresh
                note_unsettled("lessons", f"a lesson of epoch {epoch} batch {b0 // args.batch + 1}", error)
            print(f"{datetime.now().strftime('%H:%M:%S')} epoch {epoch} batch {b0 // args.batch + 1}/{n_batches} {(time.time() - t0) / 60:.1f} min", flush=True)
            k = b0 // args.batch + 1
            batches_done += 1
            by_marks = take_schedule(time.time() - t0) or by_marks    # the schedule file, before the caps' last reading
            if args.max_batches and batches_done >= args.max_batches:
                evaluate_and_save(epoch, k if k < n_batches else None, time.time() - t0, final=True)
                say("DONE (batch cap)")
                return
            if k < n_batches and time.time() - t0 >= args.hours * 3600:
                evaluate_and_save(epoch, k, time.time() - t0)
                say("DONE (time cap)")
                return
            first_sample = args.save_first_batch and epoch == 1 and k == 1
            scheduled = args.save_batches and k % args.save_batches == 0
            if by_marks:
                scheduled = k < n_batches and mark_due(time.time() - t0)
            read_now = (reading_due() if args.read_share else args.read_batches and k % args.read_batches == 0)
            if (first_sample or scheduled) and k < n_batches:
                evaluate_and_save(epoch, k, time.time() - t0)        # a checkpoint inside a long epoch
                if deaf():
                    say("DONE (does not listen)")
                    return
            elif read_now and patch_book and (k < n_batches or by_marks):
                light_reading(epoch, k, time.time() - t0)
            elif args.latest_every and args.keep_latest and k % args.latest_every == 0:
                # the newest state between readings, so a lost box costs at most this many batches
                with tempfile.TemporaryDirectory(dir=out) as scratch:
                    snapshot = Path(scratch) / "latest.npz"
                    brain.save(snapshot)
                    keep_latest(snapshot, epoch, k, time.time() - t0)
        for n, group in liked.items():            # the listener's liked windows, taught with extra weight
            for g0 in range(0, len(group), args.batch):
                chunk = group[g0:g0 + args.batch]
                bars = np.stack([c[0] for c in chunk]); mood = np.stack([c[1] for c in chunk])
                brain.reset()
                for t in range(n - 1):
                    x = observe(bars[:, t], mood[:, t + 1])
                    target = bars[:, t + 1]
                    if args.mask_unheard_targets:
                        target = common.audible_targets(target)
                    common.hear(brain, x)
                    note_credit(brain.lesson(x, target, weights=np.full(len(chunk), args.feedback_weight)))
                    if args.gain_cap:
                        cap_gains()
                    if not common.HEAR_FIRST:
                        brain.act(x, greedy=True)
        elapsed = time.time() - t0
        capped = elapsed >= args.hours * 3600
        if by_marks and not (capped or epoch == args.epochs or mark_due(elapsed)):
            continue                              # the pass ends without an evaluation: they come at the marks
        better = evaluate_and_save(epoch, None, elapsed, final=capped or epoch == args.epochs)
        if better:
            since_best = 0
        else:
            since_best += 1
        count_progress(*(state[key] if state[key] < float("inf") else None
                         for key in ("best_test_error", "best_train_error")))
        if deaf():
            break
        if capped or (args.patience and since_best >= args.patience):
            say(f"stop at epoch {epoch}: {'time cap' if capped else 'patience'}")
            break
        if args.error_patience and progress["since"] >= args.error_patience:
            say(f"stop at epoch {epoch}: no new low in {progress['since']} evaluations "
                f"(training {progress['train']:.3f}, test {progress['test']:.3f})")
            break
    say("DONE")


if __name__ == "__main__":
    main()
