"""Check a brain layout before it is trained and tracked.

Three stages, each written into quality/checks/<layout>.json:

  build    the brain composes at full size on the real layout: every motor slot has one owner, the direct
           route is in place, a lesson and an act run, and a saved copy answers as the original does
  bench    the same wiring at a small scale is taught two things: to hold a bar it heard, and to follow a
           rule on top of holding (five slots count up every bar). Holding is the gate. The rule is a
           reading: learned, partly or not learned
  listen   on the real music the held-out next-bar error falls below the level of always writing the
           commonest bar: the brain uses what it hears

Every bar is heard as a moment of its own, then taught or answered. The recall memory then holds the bar
the answer is asked for, in one slot per bar.

With --reserve the brain has a stage-1 shape of PLAN.md (common.RESERVE): "inputs" = idle ear and style inputs,
"kits" = those and room for more kits, "sid" = those and a block of SID register slots per synth, labelled off
or, with --sid-lessons POOL, taught the SID fit of every bar's sound from the first lesson. Every shape has its
own records, quality/checks/<layout>.<shape>.json. The bench then gains a test in two phases. A small brain with idle parts
learns to hold and to follow the rule. Then the idle inputs carry a signal that changes every bar, and a
second rule depends on it: the idle slots follow it, or with idle inputs alone the bench's sound-control slots
do. The reading says whether the first skills are kept and whether the second rule is learned about as well as
by a brain taught both from birth. The same first rule learned without idle parts is recorded beside it. The
reading is recorded and shown; holding stays the gate.

A layout may be trained once build and bench passed on the present code. A training run fills in the
listen stage from its own curve. COMPOSER_SCORE=v2 reads the corrected score (corpus/streams_underground_v2.json
and pool/pool_underground_v2.json).

    python check.py LAYOUT [LAYOUT ...] [--stages build,bench] [--batches 12] [--reserve inputs|kits|sid] [--sid-lessons POOL]
    python check.py --require LAYOUT [--reserve inputs|kits|sid]     exit 0 when the layout may be trained
    python check.py --list [--reserve inputs|kits|sid]
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import hierarchy
import layouts

ROOT = Path(__file__).resolve().parent.parent
CHECKS = ROOT / "quality" / "checks"
SOURCES = ("hierarchy.py", "common.py", "layouts.py")      # the builder a check holds for
PROTOCOL = 4      # how the checks measure (2: every bar is heard first, one memory slot per bar; 3: the
#                   reserved shape and its test in two phases; 4: the second phase's signal changes every bar,
#                   so repeating is always wrong there as it is for the first rule)
RESERVED = False  # --reserve: the stage-1 shape, "inputs", "kits" or "sid"; its records are kept apart from the plain ones
SID_LESSONS = None      # --sid-lessons: the pool of SID fits the full-size stages teach the SID blocks from
HEADS = None            # --recall-heads: the recall heads (the composer's one per layer, 4); records ...-hN.json
RECALL_WINDOW = None    # --recall-window: the recall memory's key spans this many moments (the composer's 2); records ...-wN.json
EAR = False             # --ear: the ear inputs carry the measured levels of the bar just heard
LAGS = None             # --lags 1,2,4,8: heads born at these lags, with --lag-gains (common.born_lags)
PLAN = False            # --plan: every bar carries the plan slots (common.LOOKAHEAD); records ...-plan.json
SECTION = False         # --section: every bar carries the section slot (common.SECTION); records ...-sec.json
PARTS = False           # --parts: every bar is heard as two moments (common.HEAR_PARTS); records ...-lagp.json
LAG_GAINS = None
SCORE = {"v2": "_v2", "v4": "_v4", "v5a": "_v5a", "v5b": "_v5b", "v5c": "_v5c", "v5d": "_v5d", "v5e": "_v5e", "v5f": "_v5f"}.get(os.environ.get("COMPOSER_SCORE"), "")      # which score the real-music stages read
DEVICE = None     # --device cuda: the full-size stages (build, listen) run on the GPU, under GPU_CAP gigabytes
GPU_CAP = 4.0
GAIN_BOUND = 300.0      # the listen stage teaches as the trainer does (aws/train_composer.sh: --gain-bound 300)


def backend():
    """The backend settings of the full-size stages."""
    if not DEVICE or DEVICE == "cpu":
        return {"backend": "cpu"}
    import torch
    share = GPU_CAP * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, share))      # never more than the cap: no other process starves
    return {"backend": "torch", "device": DEVICE}
# The bench layout: the composer's slot groups at a small scale, four values per slot.
VALUES, DRUM, LINE, PATCH = 4, 8, 4, 3
BENCH_SLOTS = hierarchy.LAYERS + hierarchy.MIX + DRUM + 3 * LINE + 3 * PATCH
WINDOW, STREAMS = 32, 8
# The bench's idle parts: three slots per synth after the live ones (value 0 = off) and four inputs after
# everything else. In the second phase the inputs carry a signal of SIGNALS values (0 = none).
IDLE_SLOTS, IDLE_INPUTS, SIGNALS = 3, 4, 4


def idle_blocks(reserve):
    """The synths with a block of idle slots on the bench, as (bass, lead, air) counts of IDLE_SLOTS: all three
    for "sid", the lead alone for "sidlead"."""
    return {"sid": (1, 1, 1), "sidlead": (0, 1, 0)}.get(reserve, (0, 0, 0))
# The second phase reads as passed when the first skills are kept within these shares of error and the second
# rule reaches the level of the brain taught both rules from birth, within this share. Two runs of one layout
# on the same seed differ by about a tenth in their rule reading (0.40 and 0.48 for split, 0.43 and 0.56 for
# trio, on boxes with other work), so a tenth is the bench's noise there.
KEEP_HOLD, KEEP_RULE, LATER_MARGIN = 0.02, 0.10, 0.10
# How far the bench error has to fall from the commonest-answer level toward the best a brain can do.
# The rule counts as learned at half the commonest-answer level and as partly learned below nine tenths of it.
HOLD_SHARE, RULE_SHARE, RULE_PARTLY = 0.75, 0.5, 0.9
# Generation 2 (no direct route): the bench's hold bar of 3/4 of the way from the commonest bar to repeating was set
# for brains whose memory wrote the heard bar into the motor slots. A brain without the route has to learn holding
# through its pathways: its bar is a quarter of the way.
HOLD_SHARE_OPEN = 0.25
TEACH_FIRST = False    # set per layout: a brain without the route is taught the next bar before the act records it


def teaches_first(name):
    """A layout without the direct route (layouts.NO_ROUTE): its lessons come before the act, as Cadence orders them."""
    return layouts.entry(name)["preset"] in layouts.NO_ROUTE


def code_sha():
    """The code a check holds for: a change to the builder or to the way the checks measure asks for a new
    check."""
    digest = hashlib.sha256(f"protocol {PROTOCOL}".encode())
    for name in SOURCES:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def path_of(name):
    return CHECKS / f"{name}{'.' + RESERVED if RESERVED else ''}{'-ear' if EAR else ''}{'-lag' if LAGS else ''}{'p' if PARTS else ''}{f'-w{RECALL_WINDOW}' if RECALL_WINDOW else ''}{f'-h{HEADS}' if HEADS else ''}{'-plan' if PLAN else ''}{'-sec' if SECTION else ''}.json"


def window_settings():
    """The recall memory's key window, when --recall-window sets one (a smaller key: half the address maps)."""
    return {**({"recall_window": int(RECALL_WINDOW)} if RECALL_WINDOW else {}), **({"recall_heads": int(HEADS)} if HEADS else {})}


def lag_settings():
    """The compose settings of heads born at several lags, or none."""
    if not LAGS:
        return {}
    lags = [None if v in ("none", "null") else int(v) for v in LAGS.split(",")]
    gains = [float(v) for v in (LAG_GAINS or "").split(",") if v]
    return {"recall_lags_born": lags, "recall_lag_gains_born": gains} if gains else {"recall_lags_born": lags}


def record(name):
    path = path_of(name)
    return json.loads(path.read_text()) if path.is_file() else {}


def write(name, stage, result):
    CHECKS.mkdir(parents=True, exist_ok=True)
    entry = layouts.entry(name)
    kept = record(name)
    if kept.get("code") != code_sha():          # stages checked on other code do not carry over
        kept = {}
    kept.update(layout=name, title=entry["title"], pathways=entry["pathways"], wiring=entry["wiring"],
                code=code_sha(), reserve=RESERVED)
    kept[stage] = {**result, "at": int(time.time())}
    kept["ready"] = judge(kept)["ready"]
    path = path_of(name)
    part = path.with_suffix(".part")
    part.write_text(json.dumps(kept, indent=1))
    part.replace(path)
    return kept


def judge(kept):
    """What a record says by the present rules. The bench passes when the brain holds; the rule is read as
    learned, partly or not learned; a layout may be trained once it builds and holds."""
    bench = kept.get("bench") or {}
    hold, rule = bench.get("hold"), bench.get("rule")
    holds = bool(hold) and hold["error"] <= hold["bar"]
    reading = None
    if rule:
        reading = ("learned" if rule["error"] <= rule["bar"]
                   else "partly" if rule["error"] <= RULE_PARTLY * rule["commonest"] else "not learned")
    later = bench.get("later")
    grows = bool(later_verdict(later)["passed"]) if later else None      # a reading, as the rule is
    return {"holds": holds, "rule": reading, "later": grows,
            "ready": bool((kept.get("build") or {}).get("passed")) and holds}


def later_verdict(later):
    """The test in two phases by the present margins: the first skills kept, and the second rule learned about
    as well as by the brain taught both rules from birth."""
    before, after, both = later["before"], later["after"], later["from_birth"]
    kept = after["hold"] <= before["hold"] + KEEP_HOLD and after["rule"] <= before["rule"] + KEEP_RULE
    learned = after["second"] <= both["second"] + LATER_MARGIN
    return {"passed": bool(kept and learned), "kept": bool(kept), "learned": bool(learned)}


def ready(name):
    kept = record(name)
    return judge(kept)["ready"] and kept.get("code") == code_sha()


# ---------------------------------------------------------------- bench

def bench_parts(architecture, reserve=False):
    return hierarchy.owners(architecture, DRUM, LINE, PATCH, tuple(IDLE_SLOTS * n for n in idle_blocks(reserve)))


def bench_brain(name, seed, reserve=False):
    """The layout's wiring on the bench layout: region widths a quarter of the real ones. With ``reserve`` the
    brain has the bench's idle parts: four more inputs, and with "sid" three more slots per synth."""
    import common
    from cadence import Brain

    preset, over = layouts.layout(name)
    slots = BENCH_SLOTS + IDLE_SLOTS * sum(idle_blocks(reserve))
    actions = slots * VALUES
    inputs = actions + 2 + (IDLE_INPUTS if reserve else 0)
    settings = dict(inputs=inputs, actions=actions, slots=(VALUES,) * slots, arousal=True,
                    recall_heads=hierarchy.LAYERS, recall_tokens=range(actions), recall_gain=1.0,
                    learned_code=64 * -(-(inputs + 1) // 64), attention=True, learning_surprise=True)
    observation_code = settings["learned_code"]
    settings.update(common.PRESETS[preset])
    settings.update(over)
    settings.update(window_settings())
    if settings.get("learned_code") == "observation":
        settings["learned_code"] = observation_code
    settings.update(modules=tuple(max(12, width // 4) for width in settings["modules"]),
                    recall_capacity=64, recall_position=16)
    born = settings.pop("recall_gain_born", None)
    architecture = hierarchy.PRESETS.get(preset)
    if architecture:
        brain = hierarchy.compose(settings, bench_parts(architecture, reserve), seed, f"bench-{architecture}")
    else:
        brain = Brain.compose(seed=seed, **settings)
    if born is not None:
        common.born_gains(brain, born)      # the value path is born switched on, as in the full-size brain
    if LAGS and LAG_GAINS:
        common.HEAR_PARTS = bool(PARTS)     # the ages the heads are born at follow the moments per bar
        common.born_lags(brain, **{"lags": lag_settings()["recall_lags_born"], "gains": lag_settings()["recall_lag_gains_born"]})
    return brain, architecture


def rule_slots():
    """The slots that follow the rule, the same in every layout: the kit, the first drum step and the first
    note of every synth."""
    lines = hierarchy.LAYERS + hierarchy.MIX + DRUM
    return [0, hierarchy.LAYERS + hierarchy.MIX, lines, lines + LINE, lines + 2 * LINE]


def later_slots(reserve):
    """The slots of the second rule: the idle slots, or with idle inputs alone the bench's sound controls."""
    if sum(idle_blocks(reserve)):
        return list(range(BENCH_SLOTS, BENCH_SLOTS + IDLE_SLOTS * sum(idle_blocks(reserve))))
    return list(range(BENCH_SLOTS - 3 * PATCH, BENCH_SLOTS))


def bench_tracks(count, seed, counting=(), reserve=False, later=False):
    """Bars that hold, with a random edit now and then; the slots in ``counting`` count up every bar.

    With ``reserve`` a signal of zeros comes with the bars, and with "sid" the bars carry the idle slots, all
    "off". With ``later`` the signal takes one of SIGNALS values and a new one every bar, and the slots of the
    second rule follow it: an idle slot is off without a signal and otherwise takes a value set by the signal
    and the slot; with idle inputs alone the sound-control slots take a value set by the signal and the slot.
    Repeating the bar is then always wrong on those slots. Every other slot is the same bar either way."""
    rng = np.random.default_rng(seed)
    free = np.setdiff1d(np.arange(BENCH_SLOTS), counting)
    out = np.zeros((count, WINDOW, BENCH_SLOTS), dtype=int)
    out[:, 0] = rng.integers(0, VALUES, (count, BENCH_SLOTS))
    for t in range(1, WINDOW):
        out[:, t] = out[:, t - 1]
        out[:, t, list(counting)] = (out[:, t - 1, list(counting)] + 1) % VALUES
        for row in np.flatnonzero(rng.random(count) > 0.85):
            edit = rng.choice(free, size=rng.integers(1, 4), replace=False)
            out[row, t, edit] = rng.integers(0, VALUES, len(edit))
    if not reserve:
        return out, None
    signal = np.zeros((count, WINDOW), dtype=int)
    if later:
        draw = np.random.default_rng(seed + 1000)
        signal[:, 0] = draw.integers(0, SIGNALS, count)
        for t in range(1, WINDOW):
            signal[:, t] = (signal[:, t - 1] + draw.integers(1, SIGNALS, count)) % SIGNALS
    if sum(idle_blocks(reserve)):
        which = np.arange(IDLE_SLOTS * sum(idle_blocks(reserve)))
        idle = np.where(signal[..., None] > 0, 1 + (signal[..., None] - 1 + which) % (VALUES - 1), 0)
        return np.concatenate([out, idle], axis=2), signal
    if later:
        out[:, :, later_slots(reserve)] = (signal[..., None] + np.arange(3 * PATCH)) % VALUES
    return out, signal


def bench_observe(bars, signal=None, reserve=False):
    """A bar as the bench brain hears it, with the signal for the bar it is to write. An idle slot that is off
    is not heard, and without a signal the idle inputs stay at zero."""
    total = bars.shape[1]
    x = np.zeros((len(bars), total * VALUES + 2 + (IDLE_INPUTS if reserve else 0)))
    x[np.arange(len(bars))[:, None], np.arange(total) * VALUES + bars] = 1.0
    if total > BENCH_SLOTS:
        x[:, BENCH_SLOTS * VALUES:total * VALUES:VALUES] = 0.0
    x[:, total * VALUES:total * VALUES + 2] = 0.5
    if reserve and signal is not None:
        on = np.flatnonzero(signal > 0)
        x[on, total * VALUES + 2 + signal[on] - 1] = 1.0
    return x


def bench_hear(brain, x, total):
    """The moment a bench bar of ``total`` slots is heard. With PARTS it is two moments, as common.hear makes
    them: first the drum row and the synth lines alone, then the whole bar."""
    if PARTS:
        fast = np.array(x, copy=True)
        first = (hierarchy.LAYERS + hierarchy.MIX) * VALUES
        fast[:, :first] = 0.0
        fast[:, first + (DRUM + 3 * LINE) * VALUES:total * VALUES] = 0.0
        brain.act(fast, greedy=True)
    brain.act(x, greedy=True)


def bench_wrong(brain, data, signal=None, reserve=False):
    """The share of wrong next-bar answers per slot."""
    brain.reset()
    total = data.shape[2]
    wrong = np.zeros(total)
    for t in range(WINDOW - 1):
        x = bench_observe(data[:, t], None if signal is None else signal[:, t + 1], reserve)
        if TEACH_FIRST:                                         # one act answers and writes the moment
            answer = np.asarray(brain.act(x, greedy=True)).reshape(len(data), total)
        else:
            bench_hear(brain, x, total)                         # the bar is heard and written to the memory
            kept = brain.recall.snapshot()
            answer = np.asarray(brain.act(x, greedy=True)).reshape(len(data), total)
            brain.recall.restore(kept)                          # the answer leaves no second copy of the bar
        wrong += (answer != data[:, t + 1]).sum(0)
    return wrong / (len(data) * (WINDOW - 1))


def bench_error(brain, data, slots=None, signal=None, reserve=False):
    """The share of wrong answers over the live slots, or over ``slots``."""
    share = bench_wrong(brain, data, signal, reserve)
    return float(share[:BENCH_SLOTS].mean() if slots is None else share[list(slots)].mean())


def bench_teach(brain, train, signal, lessons, rng, reserve=False, reading=None):
    """Lessons on windows of the training bars, every bar heard and then taught."""
    done = 0
    while done < lessons:
        chosen = rng.choice(len(train), STREAMS, replace=False)
        rows = train[chosen]
        brain.reset()
        for t in range(WINDOW - 1):
            x = bench_observe(rows[:, t], None if signal is None else signal[chosen, t + 1], reserve)
            if TEACH_FIRST:                                     # taught, then heard
                brain.lesson(x, rows[:, t + 1])
                bench_hear(brain, x, rows.shape[2])
            else:
                bench_hear(brain, x, rows.shape[2])             # the bar is heard, then taught
                brain.lesson(x, rows[:, t + 1])
            done += 1
        if reading and (done % (10 * (WINDOW - 1)) == 0 or done >= lessons):
            reading(done)
    return done


def bench_task(name, seed, counting, lessons, reserve=False):
    brain, _ = bench_brain(name, seed, reserve)
    slots = rule_slots() if counting else []
    (train, signal), (test, test_signal) = bench_tracks(64, 11, slots, reserve), bench_tracks(16, 12, slots, reserve)
    watched = slots or None
    live = test[:, :, :BENCH_SLOTS]
    commonest = np.array([np.bincount(train[:, :, s].ravel(), minlength=VALUES).argmax() for s in range(BENCH_SLOTS)])
    pick = (lambda share: share[slots].mean()) if slots else (lambda share: share.mean())
    blind = float(pick((live[:, 1:] != commonest).mean((0, 1))))
    repeat = float(pick((live[:, 1:] != live[:, :-1]).mean((0, 1))))
    curve = [[0, round(bench_error(brain, test, watched, test_signal, reserve), 3)]]
    done = bench_teach(brain, train, signal, lessons, np.random.default_rng(seed), reserve,
                       lambda at: curve.append([at, round(bench_error(brain, test, watched, test_signal, reserve), 3)]))
    return {"error": curve[-1][1], "commonest": round(blind, 3), "repeat": round(repeat, 3), "lessons": done,
            "curve": curve, "neurons": int(brain.connectome.n)}


def bench_later(name, seed, lessons, reserve="sid"):
    """The test in two phases. A brain with idle parts learns to hold and the rule. Then the signal arrives and
    the second rule's slots follow it. It is read before and after the second phase, next to a brain of the
    same shape that was taught both rules from birth for as many lessons in all."""
    counting, idle = rule_slots(), later_slots(reserve)
    others = [slot for slot in range(BENCH_SLOTS) if slot not in counting and slot not in idle]
    first, first_test = bench_tracks(64, 11, counting, reserve), bench_tracks(16, 12, counting, reserve)
    second, second_test = bench_tracks(64, 11, counting, reserve, True), bench_tracks(16, 12, counting, reserve, True)

    def read(brain, data):
        share = bench_wrong(brain, data[0], data[1], reserve)
        return {"hold": round(float(share[others].mean()), 3), "rule": round(float(share[counting].mean()), 3),
                "second": round(float(share[idle].mean()), 3)}

    grown, _ = bench_brain(name, seed, reserve)
    rng = np.random.default_rng(seed)
    bench_teach(grown, first[0], first[1], lessons, rng, reserve)
    before = read(grown, first_test)                    # the first skills, before the signal exists
    untaught = read(grown, second_test)["second"]       # the second rule before any lesson of it
    bench_teach(grown, second[0], second[1], lessons, rng, reserve)
    after = read(grown, second_test)
    born, _ = bench_brain(name, seed, reserve)
    bench_teach(born, second[0], second[1], 2 * lessons, np.random.default_rng(seed), reserve)
    both = read(born, second_test)
    bars = second_test[0]
    commonest = np.array([np.bincount(second[0][:, :, s].ravel(), minlength=VALUES).argmax() for s in idle])
    levels = {"commonest": round(float((bars[:, 1:, idle] != commonest).mean()), 3),
              "repeat": round(float((bars[:, 1:, idle] != bars[:, :-1, idle]).mean()), 3)}
    counting_levels = {"commonest": round(float(np.mean([(first_test[0][:, 1:, s] != np.bincount(first[0][:, :, s].ravel(), minlength=VALUES).argmax()).mean() for s in counting])), 3),
                       "repeat": round(float((first_test[0][:, 1:, counting] != first_test[0][:, :-1, counting]).mean()), 3)}
    result = {"before": before, "after": after, "from_birth": both, "second_untaught": untaught,
              "second_levels": levels, "rule_levels": counting_levels, "lessons": [lessons, lessons],
              "neurons": int(grown.connectome.n)}
    return {**later_verdict(result), **result}


def bench(name, seed=3, lessons=930, later=True):
    """Hold: the error over every live slot against the commonest bar and repeating. Rule: the error on the
    counting slots, where repeating is always wrong and the best is no error at all. With the reserved shape
    the rule is read from the first phase of the test in two phases, and the same rule learned by a brain
    without idle parts is recorded as ``plain``. Without ``later`` the second phase is left out."""
    start = time.monotonic()
    global TEACH_FIRST
    TEACH_FIRST = teaches_first(name)
    hold = bench_task(name, seed, False, lessons, RESERVED)
    share = HOLD_SHARE_OPEN if TEACH_FIRST else HOLD_SHARE
    hold["bar"] = round(hold["commonest"] - share * (hold["commonest"] - hold["repeat"]), 3)
    if not RESERVED or not later:
        rule = bench_task(name, seed, True, lessons, RESERVED)
        rule["bar"] = round(rule["commonest"] * (1 - RULE_SHARE), 3)
        result = {"passed": hold["error"] <= hold["bar"], "hold": hold, "rule": rule}
    else:
        phases = bench_later(name, seed, lessons, RESERVED)
        rule = {"error": phases["before"]["rule"], **phases["rule_levels"], "lessons": lessons}
        rule["bar"] = round(rule["commonest"] * (1 - RULE_SHARE), 3)
        result = {"passed": hold["error"] <= hold["bar"], "hold": hold, "rule": rule, "later": phases}
    if RESERVED:
        plain = bench_task(name, seed, True, lessons)
        result["plain"] = {key: plain[key] for key in ("error", "commonest", "repeat", "lessons", "neurons")}
    return {**result, "seed": seed, "minutes": round((time.monotonic() - start) / 60, 1)}


# ---------------------------------------------------------------- the real layout

def music(eval_tracks=4, teach_first=False):
    """The real layout and corpus, configured as the trainer configures them for the comparison."""
    import common
    import gate
    import timbre
    from train import eval_set

    pool = json.loads((ROOT / f"pool/pool_underground{SCORE}.json").read_text())
    book = timbre.teachers(pool)
    common.ONOFF = False
    common.SYNTH_FILL = True              # a silent synth part is taught the line it last played
    common.HEAR_FIRST = not teach_first   # every bar is heard before it is taught or answered (with the route)
    common.HEAR_PARTS = bool(PARTS)       # as two moments with --parts
    sid_book = timbre.sid_book(json.loads(Path(SID_LESSONS).read_text())) if SID_LESSONS else None
    common.RESERVE, common.SID_LIVE = RESERVED or False, bool(sid_book)      # the stage-1 shape
    common.EAR_LIVE = bool(EAR)
    common.LOOKAHEAD = bool(PLAN)         # the plan slots after the live slots
    common.SECTION = bool(SECTION)        # the section slot after them
    tracks = common.load_streams(ROOT / f"corpus/streams_underground{SCORE}.json", steps=True, rows=None, sid_book=sid_book,
                                 synth={"kind": "steps"}, clock=True, patch_book=book)
    evals = eval_set(tracks, eval_tracks, 33, 0, interior=True)
    train_eval = [t for t in evals if not t["held_out"]]
    test_eval = [t for t in evals if t["held_out"]]
    taught = [t for t in tracks if not t["held_out"]]
    average = gate.commonest_bar(taught)
    levels = {name: {"train": round(gate.prediction_error(None, train_eval, constant=constant)["balanced"], 4),
                     "test": round(gate.prediction_error(None, test_eval, constant=constant)["balanced"], 4)}
              for name, constant in (("commonest", average), ("repeat", None))}
    return taught, train_eval, test_eval, levels


def build(name, seed=1):
    import anatomy
    import common
    from cadence import Brain

    start = time.monotonic()
    taught, _, _, _ = music(teach_first=teaches_first(name))
    preset, over = layouts.layout(name)
    over = {**over, **lag_settings(), **window_settings()}
    brain, settings = common.compose(preset, seed=seed, **backend(), **over)
    problems = []
    if teaches_first(name):
        # generation 2: no route from memory to the motor slots; the heads read at the coordinator on unit keys
        if settings.get("recall_tokens") is not None or settings.get("recall_gain"):
            problems.append("a direct route is wired")
        if settings.get("pathway_recall"):
            problems.append("a recall head reads into a pathway")
        if not getattr(getattr(brain, "recall", None), "key_norm", False):
            problems.append("the recall keys are not unit-normalized")
    else:
        if not (settings.get("recall_tokens") is not None and settings.get("recall_gain")):
            problems.append("no direct route")
        if settings.get("learned_code") and settings["learned_code"] < common.INPUTS:
            problems.append("the learned code is narrower than the observation")
        gains = getattr(getattr(brain, "recall", None), "gains", None)
        if gains is None:
            problems.append("the recall memory has no gains")
    card = {"settings": settings, "seed": seed, "steps": common.STEPS, "patch_book": common.PATCH_BOOK,
            "reserve": RESERVED, "lookahead": bool(PLAN), "section": bool(SECTION)}
    drawn = anatomy.describe(card)
    if not drawn:
        problems.append("the layout cannot be drawn")
    elif sum(output["slots"] for output in drawn["outputs"]) != common.SLOTS:
        problems.append("a motor slot has no owner")
    wins = common.windows(taught, 8, 64)
    bars = np.stack([wins[i][0] for i in (0, 1)]); mood = np.stack([wins[i][1] for i in (0, 1)])
    if RESERVED:
        import timbre
        idle = common.chip_end() - common.LIVE
        if idle != timbre.SID_WIDTH * len(common.chip_layers()):
            problems.append("the SID blocks are not in place")
        if common.SIZES[0] != (common.KIT_CLASSES if RESERVED in ("kits", "sid", "sidlead") else common.T):
            problems.append("the kit slot has the wrong room")
        x = common.observe(bars[:, 0], mood[:, 1])
        # with the ear the first EAR_INPUTS reserved inputs carry what it heard; the style inputs stay idle
        if x[:, -(common.STYLE_INPUTS if EAR else common.RESERVED_INPUTS):].any():
            problems.append("an idle input is heard")
        if EAR and not x[:, -common.RESERVED_INPUTS:-common.STYLE_INPUTS].any():
            problems.append("the ear hears nothing")
        # the SID blocks' input columns (none in a shape without them: LIVE may then be the last slot)
        chip_columns = (slice(common.OFFSETS[common.LIVE], common.OFFSETS[common.chip_end()] if common.chip_end() < common.SLOTS else common.ACTIONS)
                        if idle else slice(0, 0))
        if idle and not common.SID_LIVE and (bars[..., common.LIVE:common.chip_end()].any() or x[:, chip_columns].any()):
            problems.append("an idle SID block is not off or is heard")
        if idle and common.SID_LIVE:
            # with SID lessons a sounding synth's switch is on (the synths with a block: all three, or the lead)
            sounding = bars[..., [1 + i for i in common.chip_layers()]] > 0
            switches = bars[..., common.LIVE:common.chip_end():timbre.SID_WIDTH]
            if not (switches[sounding] == 1).all():
                problems.append("a sounding synth has no SID fit")
    if PLAN:
        plan = bars[..., common.chip_end():common.chip_end() + common.PLAN]
        if common.PLAN != len(common.PLAN_SIZES) or (plan >= np.asarray(common.PLAN_SIZES)).any() or not plan.any():
            problems.append("the plan slots are not in place")
        if not common.observe(bars[:, 0], mood[:, 1])[:, common.OFFSETS[common.chip_end()]:common.ACTIONS].any():
            problems.append("the plan is not heard")
    if SECTION:
        section = bars[..., common.section_slot()]
        if common.SECTIONS != 1 or (section >= len(common.SECTION_CLASSES)).any() or (section < 0).any():
            problems.append("the section slot is not in place")
    brain.reset()
    for t in range(3):
        x = common.observe(bars[:, t], mood[:, t + 1])
        if common.HEAR_FIRST:
            common.hear(brain, x)
            brain.lesson(x, bars[:, t + 1])
        else:                       # taught before the act records the moment
            brain.lesson(x, bars[:, t + 1])
            brain.act(x, greedy=True)
    answer = np.asarray(common.answer(brain, x))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "copy.npz"
        brain.save(path)
        copy = Brain.load(path, **backend())
        x = common.observe(bars[:, 3], mood[:, 4])
        if not np.array_equal(np.asarray(common.answer(brain, x)), np.asarray(common.answer(copy, x))):
            problems.append("a saved copy answers differently")
    if answer.reshape(2, -1).shape[1] != common.SLOTS:
        problems.append("the answer does not fill the layout")
    return {"passed": not problems, "problems": problems, "neurons": int(brain.connectome.n),
            "synapses": int(settings.get("plastic_synapses") or (drawn or {}).get("synapses") or 0),
            "inputs": int(common.INPUTS), "slots": int(common.SLOTS), "live_slots": int(common.LIVE),
            "idle_inputs": int(common.RESERVED_INPUTS), "kit_classes": int(common.SIZES[0]),
            "sid_lessons": bool(common.SID_LIVE), "ear": bool(common.EAR_LIVE), "hear_parts": bool(common.HEAR_PARTS),
            "code": settings.get("learned_code"),
            "modules": list(settings.get("modules") or ()), "score": SCORE.strip("_") or "v1",
            "architecture": settings.get("architecture") or "chain", "device": DEVICE or "cpu",
            "minutes": round((time.monotonic() - start) / 60, 1)}


def listen(name, batches=4, seed=1):
    """A light check on the real music: lesson batches of two 24-bar passages, read after 2, 4, 8 ... batches
    against the commonest bar and repeating, on one training and one held-out passage."""
    import common
    import gate
    from cadence import Brain

    taught, train_eval, test_eval, levels = music(eval_tracks=1, teach_first=teaches_first(name))
    preset, over = layouts.layout(name)
    brain, _ = common.compose(preset, seed=seed, **backend(), **{**over, **lag_settings(), **window_settings()})
    if not teaches_first(name):
        type(brain.recall).GAIN_BOUND = GAIN_BOUND
    wins = common.windows(taught, 24, 64)
    start = time.monotonic()

    def reading(done):
        with tempfile.TemporaryDirectory() as folder:      # an evaluation must not teach: it reads a copy
            path = Path(folder) / "copy.npz"
            brain.save(path)
            row = {"batches": done, "minutes": round((time.monotonic() - start) / 60, 1)}
            for part, passages in (("train", train_eval), ("test", test_eval)):
                read = gate.prediction_error(Brain.load(path, **backend()), passages)
                row[part] = round(read["balanced"], 4)
                if read.get("sid") is not None:
                    row[part + "_sid"] = round(read["sid"], 4)      # the SID blocks, read apart
                gc.collect()        # the copy's memory goes back before the next one is loaded
        return row

    curve = [reading(0)]                # at birth: the direct route holds what was heard before any lesson
    print(json.dumps({"layout": name, "levels": levels, **curve[-1]}), flush=True)
    rng = np.random.default_rng(seed)
    for done in range(1, batches + 1):
        chosen = rng.choice(len(wins), 2, replace=False)
        bars = np.stack([wins[i][0] for i in chosen]); mood = np.stack([wins[i][1] for i in chosen])
        brain.reset()
        for t in range(bars.shape[1] - 1):
            x = common.observe(bars[:, t], mood[:, t + 1])
            if common.HEAR_FIRST:
                common.hear(brain, x)
                brain.lesson(x, bars[:, t + 1])
            else:                   # taught before the act records the moment
                brain.lesson(x, bars[:, t + 1])
                brain.act(x, greedy=True)
        if (done > 1 and done & (done - 1) == 0) or done == batches:
            curve.append(reading(done))
            print(json.dumps({"layout": name, "levels": levels, **curve[-1]}), flush=True)
            write(name, "listen", {"passed": None, "running": True, "levels": levels, "curve": curve, "where": "local"})
    best = min(row["test"] for row in curve)
    return {"passed": best < levels["commonest"]["test"], "levels": levels, "curve": curve, "where": "local",
            "lowest_test": best, "minutes": curve[-1]["minutes"], "device": DEVICE or "cpu",
            "gain_bound": GAIN_BOUND, "score": SCORE.strip("_") or "v1"}


STAGES = {"build": build, "bench": bench, "listen": listen}


def main():
    global RESERVED, DEVICE, GPU_CAP
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("layouts", nargs="*")
    parser.add_argument("--stages", default="build,bench")
    parser.add_argument("--batches", type=int, default=4, help="lesson batches of the listen stage")
    parser.add_argument("--require", metavar="LAYOUT")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--reserve", nargs="?", const="sid", default=None, choices=("inputs", "kits", "sid", "sidlead"),
                        help="check a stage-1 shape: 'inputs' = idle inputs, 'kits' = those and room for more kits, "
                             "'sid' = those and the SID blocks")
    parser.add_argument("--sid-lessons", metavar="POOL", help="with --reserve sid: the full-size stages teach the SID fits of this pool")
    parser.add_argument("--ear", action="store_true", help="with --reserve: the ear inputs carry the measured levels (records <layout>.<shape>-ear.json)")
    parser.add_argument("--lags", help="heads born at these lags, e.g. 1,2,4,8 (records ...-lag.json); with --lag-gains")
    parser.add_argument("--lag-gains", help="the born gain of every head's recall port, e.g. 0.1,0.05,0.075,0.025")
    parser.add_argument("--parts", action="store_true",
                        help="with --lags: every bar is heard as two moments, the drum rows and synth lines alone and "
                             "then the whole bar (common.HEAR_PARTS; records ...-lagp.json)")
    parser.add_argument("--section", action="store_true",
                        help="every bar carries the section slot (normal, breakdown, build; common.SECTION; records ...-sec.json)")
    parser.add_argument("--plan", action="store_true",
                        help="every bar carries the plan slots, the score's summary of the 8 bars after it "
                             "(common.LOOKAHEAD, plan_rows; records ...-plan.json)")
    parser.add_argument("--recall-heads", type=int, help="the recall heads (records ...-hN.json); with --lags one lag per head")
    parser.add_argument("--recall-window", type=int, help="the recall memory's key window in moments (records ...-wN.json)")
    parser.add_argument("--seed", type=int, default=3, help="the seed of the bench")
    parser.add_argument("--device", default=None, help="cuda: build and listen on the GPU (under --gpu-cap gigabytes)")
    parser.add_argument("--gpu-cap", type=float, default=GPU_CAP)
    parser.add_argument("--skip-later", action="store_true",
                        help="bench without the second phase: hold and the rule with idle parts, and the rule without")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    global SID_LESSONS, EAR, LAGS, LAG_GAINS, PARTS, RECALL_WINDOW, HEADS, PLAN, SECTION
    RESERVED = args.reserve or False
    PLAN = bool(args.plan)
    SECTION = bool(args.section)
    SID_LESSONS = args.sid_lessons
    EAR = bool(args.ear)
    RECALL_WINDOW = args.recall_window
    HEADS = args.recall_heads
    LAGS, LAG_GAINS, PARTS = args.lags, args.lag_gains, bool(args.parts)
    if PARTS and not LAGS:
        sys.exit("--parts belongs to heads born at several lags (--lags)")
    DEVICE, GPU_CAP = args.device, args.gpu_cap
    if args.require:
        if args.require not in layouts.LAYOUTS:
            sys.exit(f"unknown layout {args.require}")
        if not ready(args.require):
            sys.exit(f"{args.require} has not passed build and bench on the present builder: "
                     f"python brain/check.py {args.require}{f' --reserve {RESERVED}' if RESERVED else ''}")
        return
    if args.list:
        for name in layouts.LAYOUTS:
            kept = record(name)
            fresh = kept.get("code") == code_sha()
            marks = " ".join(f"{stage}:{'-' if stage not in kept or not fresh else 'ok' if kept[stage].get('passed') else 'wait' if kept[stage].get('passed') is None else 'FAIL'}"
                             for stage in STAGES)
            verdict = judge(kept) if fresh else {"rule": None, "ready": False, "later": None}
            later = "" if verdict["later"] is None else f" later:{'ok' if verdict['later'] else 'FAIL'}"
            print(f"{name:16s} {marks} rule:{verdict['rule'] or '-'}{later} {'READY' if verdict['ready'] else ''}")
        return
    for name in args.layouts or list(layouts.LAYOUTS):
        layouts.layout(name)
        for stage in args.stages.split(","):
            try:
                result = (STAGES[stage](name, batches=args.batches) if stage == "listen"
                          else bench(name, seed=args.seed, later=not args.skip_later) if stage == "bench"
                          else STAGES[stage](name))
            except Exception as exc:          # a layout that cannot be built has failed its check
                result = {"passed": False, "problems": [f"{type(exc).__name__}: {str(exc)[:300]}"]}
            write(name, stage, result)
            shown = {key: value for key, value in result.items() if key not in ("curve", "levels")}
            for part in ("hold", "rule", "plain"):
                if part in shown:
                    shown[part] = {key: value for key, value in shown[part].items() if key != "curve"}
            print(json.dumps({"layout": name, "stage": stage, **shown}), flush=True)


if __name__ == "__main__":
    main()
