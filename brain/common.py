"""Shared layout of the minimal-techno creature: bar state, observation and action coding, presets, statistics.

A bar's state has 12 slots: the sound (token) of each of the 4 layers, then FILTER and SPACE levels of each
layer (0..M-1, the middle level = the sound's own setting). The observation is the one-hot state of the
present bar plus the mood wanted for the next bar; the action is the next bar's state, one settle for all.
Every slot value is a token of the recall memory's value path, so a remembered sound or modulation can
come back.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import hierarchy
import timbre


def local_setting(name, default=""):
    """A setting of this computer alone: the environment, else site/local.json (kept out of the repository)."""
    if os.environ.get(name) is not None:
        return os.environ[name]
    try:
        return json.loads((Path(__file__).resolve().parents[1] / "site" / "local.json").read_text()).get(name, default)
    except (OSError, ValueError):
        return default


# A score may mark its tracks with a style and a group. Where a function prefers one (the passages free runs start
# from, the pace the play temperatures are calibrated to, the lessons of the synth lines), it prefers these two.
REFERENCE_STYLE = local_setting("CADANCE_REFERENCE_STYLE", "reference")
REFERENCE_GROUP = local_setting("CADANCE_REFERENCE_GROUP", "reference")

PATCH_BOOK = None
AUDITED_RECALL_PATCH = "6fd423e9440a96bff42511c5540145d3ddd5efd244f2bd08cf5184ba3b53b700"
AUDITED_RECALL_RELEASE = "0.103.1"

L, MOOD = 4, 2                 # MOOD = energy, brightness (+ one-hot style when the corpus has several styles)
STYLES: list = []
CLOCK = 0                      # 6 when the brain hears the bar clock: the bar count of the bar it writes, as binary bits
CLOCK_OFFSETS = {}             # track id -> bar offset of its lesson clock (the structure lane's grid; play counts from 0)
LAYERS = ("beat", "bass", "lead", "air")
MODS = 8                       # FILTER, SPACE per layer: [f_beat, s_beat, f_bass, s_bass, ...]
T, M = 33, 7                   # tokens per layer (0 = silence), modulation levels (set by configure)
STEPS = 0                      # 48 when the brain writes the beat step by step (kick, snare/clap, hats x 16, each 0/1/2)
ROWS = None                    # or: one slot per drum row, each a whole 16-step row from that voice's row vocabulary
SYNTH = None                   # when the brain writes synth lines: {layer: [16-step note rows]}, 0 rest / 1 tie / 2+pc
SYNTH_LAYERS = ("bass", "lead", "air")
SYN = 0
DRUM_ROWS = ("kick", "snare", "hats")
SIZES = [T] * L + [M] * MODS
OFFSETS = np.cumsum([0] + SIZES[:-1])
SLOTS = L + MODS
INPUTS = sum(SIZES) + MOOD
ACTIONS = sum(SIZES)
CONTENT = None                 # optional: what each layer's sound plays, heard next to its id (set_content)
# The stage-1 shape (PLAN.md). Cadence cannot add inputs to a trained brain, and a brain with the direct
# route takes no new output slots, so a brain of this shape is born with parts of the later stages in
# place. RESERVE names how much:
#   "inputs"  EAR_INPUTS and STYLE_INPUTS inputs after everything else, held at zero
#   "kits"    those, and a kit slot of KIT_CLASSES classes in place of the pool's 65: kits fitted later
#             (breakbeat kits with other snares) arrive as new labels of the same slot. A class without a
#             kit is never labelled and never played
#   "sidlead" as "sid" with one block, the lead's (a brain near the plain width: learned code 2,240 against 3,200)
#   "sid"     those, and after the last live slot one block of SID register slots per synth
#             (timbre.SID_SIZES). The blocks are labelled "off" in every lesson, or with SID lessons
#             (load_streams sid_book, SID_LIVE) they carry the SID fit of the bar's sound from the first lesson
# The reserved parts carry no rule and no feature. LIVE is the number of slots every brain has. Every
# reading of errors, hits and free play counts those alone; the SID blocks have a reading of their own.
# The idle inputs fit the spare width of the learned code and cost nothing. The kit room adds 64 values:
# the learned code grows from 1,792 to 1,856. The SID blocks add 1,395 one-hot values more, and the direct
# route needs every one of them: the learned code is then 3,264. A brain's host memory follows the square of
# that width (measured: 4.7 GB at 1,792, 5.8 at 2,112, 6.3 at 2,240, so about 4.9 GB at 1,856 and about
# 11 GB at 3,264; brain/memory_probe.py, quality/stage1/memory.json).
RESERVE = False                # False, "inputs", "kits", "sid" or "sidlead": set before configure / load_streams; a card records it
SID_LIVE = False               # the brain is taught, hears and plays its SID blocks
SID_RENDER = True              # a brain with live SID blocks is played through the chip while a block's switch is on
EAR_LIVE = False               # the ear inputs carry the measured levels of the bar just heard (ear_inputs)
EAR_INPUTS, STYLE_INPUTS = 16, 4
# The plan (look-ahead lessons, snapshot s1f): with LOOKAHEAD every bar carries PLAN slots after the live slots and
# the SID blocks, the score's summary of the PLAN_BARS bars after it (plan_rows). A brain hears the plan of the bar
# heard and writes the plan of the bar it writes. Readings count the live slots and give the plan a row of its own;
# the player renders the live slots and hears the brain's own plan back through observe.
LOOKAHEAD = False
PLAN_BARS = 8
PLAN_SIZES = (3, 3, 3, 3, 3, 3)     # beat, bass, lead, air: none / some / all; arrangement changes 0 / 1 / 2+; first kickless bar
PLAN = 0                           # the plan slots of the configured shape
# The section (score v5f, structure lane): with SECTION every bar carries one slot after the plan slots, the score's
# section of the bar: 0 normal, 1 breakdown, 2 build (reserved: the score's builds are taught as normal for now).
# Heard with the bar heard, written with the bar written, as the plan.
SECTION = False
SECTION_CLASSES = ("normal", "breakdown", "build")
SECTION_TAUGHT = {"normal": 0, "breakdown": 1, "build": 0}
SECTIONS = 0                       # the section slots of the configured shape (0 or 1)
EAR_NAMES = ("bass", "bass sounds", "bass over the bar", "lead", "lead sounds", "lead over the bar",
             "air", "air sounds", "air over the bar", "drums", "drums sound", "drums over the bar")
KIT_CLASSES = 129              # silence and room for 128 kits; the fitted pool has 64
RESERVED_INPUTS = 0
LIVE = SLOTS


def ear_inputs(levels, bar_db, drum_median, drum_bar_median):
    """What the ear hears of one bar: EAR_INPUTS values, the first twelve named in EAR_NAMES, the rest zero.

    ``levels`` are the bar's levels in dB of [drums, bass, lead, air] over each voice's sounding steps (None
    for a silent voice), ``bar_db`` the same four over the whole bar, the two medians the drums' typical level
    in this recording by either measure. The score carries them per bar (``levels``, ``bar_db``); a player
    takes them from its engine's level meters.

    Per synth voice, three values: its level against the drums, ``(dB - drums dB + 20) / 20`` clipped to
    -1..1 (0 = 20 dB under the kit, 1 = as loud as the kit), 1 when the voice sounds in the bar, and the same
    level over the whole bar. The drums of the bar are the reference; in a bar without drums their median is.
    For the drums: their level against their median, ``(dB - median) / 10`` clipped to -1..1, 1 when they
    sound, and the same over the whole bar."""
    out = np.zeros(EAR_INPUTS)
    scale = lambda value: float(np.clip(value, -1.0, 1.0))
    drums = levels[0]
    reference = drums if drums is not None else drum_median
    bar_reference = bar_db[0] if drums is not None else drum_bar_median
    for voice in range(3):
        level = levels[voice + 1]
        if level is not None:
            out[3 * voice] = scale((level - reference + 20) / 20)
            out[3 * voice + 1] = 1.0
            out[3 * voice + 2] = scale((bar_db[voice + 1] - bar_reference + 20) / 20)
    if drums is not None:
        out[9] = scale((drums - drum_median) / 10)
        out[10] = 1.0
        out[11] = scale((bar_db[0] - drum_bar_median) / 10)
    return out


def ear_track(levels, bar_db):
    """The ear's values for every bar of a recording, (bars, EAR_INPUTS)."""
    heard = [(row[0], whole[0]) for row, whole in zip(levels, bar_db) if row[0] is not None]
    median = float(np.median([a for a, _ in heard])) if heard else -20.0
    bar_median = float(np.median([b for _, b in heard])) if heard else -20.0
    return np.array([ear_inputs(row, whole, median, bar_median) for row, whole in zip(levels, bar_db)])


def chips():
    """Whether the layout has the SID blocks."""
    return RESERVE in (True, "sid", "sidlead")


def chip_layers():
    """The synths (0 bass, 1 lead, 2 air) that carry a SID block, in block order."""
    return (1,) if RESERVE == "sidlead" else (0, 1, 2) if chips() else ()


def chip_end():
    """The slot after the SID blocks: where the plan slots begin (SLOTS without a plan or a section)."""
    return SLOTS - PLAN - SECTIONS


def section_slot():
    """The section's slot (the last one), or None without SECTION."""
    return SLOTS - 1 if SECTIONS else None


def kit_room():
    """Whether the kit slot has room for kits fitted later."""
    return RESERVE in (True, "kits", "sid", "sidlead")
DRUMS = (("kick",), ("snare", "clap"), ("ch",), ("oh", "cym"), ("ltom", "htom", "rim", "cow"))
DEVICES = ("606", "808", "909")
TIMBRE = ("cutoff", "reso", "decay", "envmod", "send_delay", "send_reverb")


def _content_of(layer, token):
    """A sound's heard content: beat = per-voice-group velocities on 16 steps + drum machine model;
    synth layers = gates, pitch and accents on 16 steps + acid/synth + main timbre settings (all 0..1)."""
    if layer == 0:
        out = np.zeros(len(DRUMS) * 16 + len(DEVICES))
        if token:
            hits = token.get("pattern", {}).get("hits", {})
            for g, voices in enumerate(DRUMS):
                for v in voices:
                    out[g * 16:(g + 1) * 16] = np.maximum(out[g * 16:(g + 1) * 16], np.clip(np.asarray(hits.get(v, [0] * 16), float)[:16], 0, 1))
            if token.get("device") in DEVICES:
                out[len(DRUMS) * 16 + DEVICES.index(token["device"])] = 1.0
        return out
    out = np.zeros(16 * 3 + 1 + len(TIMBRE))
    if token:
        p = token.get("pattern", {})
        notes = p.get("notes", [None] * 16)[:16]
        vel = p.get("vel") or [1.0 if n is not None else 0.0 for n in notes]
        acc = p.get("accent") or [0] * 16
        for i, n in enumerate(notes):
            if n is not None:
                out[i] = min(1.0, float(vel[i]) if i < len(vel) and vel[i] is not None else 1.0)
                out[16 + i] = float(np.clip((n - 24) / 72.0, 0, 1))
                out[32 + i] = float(acc[i]) if i < len(acc) and acc[i] is not None else 0.0
        out[48] = 1.0 if token.get("voice") == "acid" else 0.0
        patch = token.get("patch", {})
        for j, k in enumerate(TIMBRE):
            out[49 + j] = float(patch.get(k, patch.get("f_env", 0.0) if k == "envmod" else 0.0))
    return out


def set_content(pool: dict | None):
    """Hear each sound's content from this pool (None: ids only)."""
    global CONTENT, INPUTS
    if pool is None:
        CONTENT = None
    else:
        CONTENT = [np.stack([_content_of(l, layer["tokens"][t] if t < len(layer["tokens"]) else None) for t in range(T)])
                   for l, layer in enumerate(pool["layers"])]
    INPUTS = sum(SIZES) + MOOD + content_width() + RESERVED_INPUTS


def content_width() -> int:
    return 0 if CONTENT is None else sum(c.shape[1] for c in CONTENT)


def configure(tokens_per_layer: int, mod_levels: int = 7, steps: bool = False, styles=None, rows=None, clock=False, synth=None, patch_book=None):
    """steps: the brain writes 48 drum steps; rows: {voice: [16-step rows]} -> one slot per drum row instead."""
    global T, M, SIZES, OFFSETS, INPUTS, ACTIONS, SLOTS, STEPS, STYLES, MOOD, ROWS, CLOCK, SYNTH, SYN
    global PATCH_BOOK, LIVE, RESERVED_INPUTS, PLAN, SECTIONS
    PATCH_BOOK = patch_book
    if patch_book and (not synth or ONOFF):
        raise ValueError("DSP controls require synth lines and embedded synth gates")
    if RESERVE and not patch_book:
        raise ValueError("The reserved shape belongs to the composer with DSP controls")
    STYLES = list(styles or [])
    CLOCK = 6 if clock else 0
    MOOD = 2 + len(STYLES) + CLOCK
    T, M = int(tokens_per_layer), int(mod_levels)
    ROWS = {k: [list(r) for r in v] for k, v in rows.items()} if rows else None
    STEPS = 3 if ROWS else (48 if steps else 0)
    if synth and synth.get("kind") in ("steps", "splitsteps"):
        kind = synth["kind"]
        SYNTH = {"kind": kind}
        syn_sizes = ([14] if kind == "steps" else [3, 12]) * (16 * len(SYNTH_LAYERS))
        SYN = len(syn_sizes)
    elif synth and synth.get("kind") == "factored":
        SYNTH = {"kind": "factored", "rhythm": {k: [list(r) for r in v] for k, v in synth["rhythm"].items()},
                 "contour": {k: [list(c) for c in v] for k, v in synth["contour"].items()}}
        SYN = 3 * len(SYNTH_LAYERS)
        syn_sizes = [n for v in SYNTH_LAYERS for n in (len(SYNTH["rhythm"][v]), len(SYNTH["contour"][v]), 12)]
    else:
        SYNTH = {k: [list(r) for r in v] for k, v in synth.items()} if synth else None
        SYN = 3 if SYNTH else 0
        syn_sizes = [len(SYNTH[v]) for v in SYNTH_LAYERS] if SYNTH else []
    _MEMO.clear()
    if kit_room() and T > KIT_CLASSES:
        raise ValueError("The score has more kits than the reserved shape holds")
    sound_sizes = [KIT_CLASSES if kit_room() else T, 2, 2, 2] if patch_book else [T] * L
    patch_sizes = list(timbre.sizes(patch_book)) * 3 if patch_book else []
    SIZES = sound_sizes + [M] * MODS + ([len(ROWS[v]) for v in DRUM_ROWS] if ROWS else [3] * STEPS) + syn_sizes + patch_sizes + ([2] * L if ONOFF else [])
    LIVE = len(SIZES)
    if chips():
        SIZES = SIZES + list(timbre.SID_SIZES) * len(chip_layers())      # idle: one SID block per synth that has one
    PLAN = len(PLAN_SIZES) if LOOKAHEAD else 0
    SIZES = SIZES + list(PLAN_SIZES) * bool(PLAN)                        # the plan slots
    SECTIONS = 1 if SECTION else 0
    SIZES = SIZES + [len(SECTION_CLASSES)] * SECTIONS                    # the section, last
    RESERVED_INPUTS = EAR_INPUTS + STYLE_INPUTS if RESERVE else 0        # idle: held at zero
    SLOTS = len(SIZES)
    OFFSETS = np.cumsum([0] + SIZES[:-1])
    ACTIONS = sum(SIZES)
    INPUTS = ACTIONS + MOOD + content_width() + RESERVED_INPUTS


def layout_key(card):
    """Cache identity for encoded tracks and compatible feedback windows."""
    import hashlib
    fields = ("streams", "tokens_per_layer", "mod_levels", "styles", "clock", "onoff",
              "steps", "beat_rows", "synth_rows", "synth_fill", "patch_book")
    if card.get("streams_sha"):
        fields += ("streams_sha",)
    if card.get("reserve"):
        fields += ("reserve",)
    if card.get("sid_live"):
        fields += ("sid_live",)
    if card.get("ear"):
        fields += ("ear",)
    if card.get("lookahead"):
        fields += ("lookahead",)
    if card.get("section"):
        fields += ("section",)
    if card.get("echo_lags"):
        fields += ("echo_lags",)
    layout = json.dumps([card.get(k) for k in fields], sort_keys=True)
    return hashlib.sha256(layout.encode()).hexdigest()


# The echo brain (generation 2): besides the bar heard (lag 1) the brain hears the bars ECHO_LAGS back as further
# one-hot slot blocks of the same layout, after the observation's own columns. Empty: the plain observation.
ECHO_LAGS = ()


def brain_inputs():
    """The observation width a brain takes: the bar heard with its mood rows, and with ECHO_LAGS one more slot block
    (ACTIONS columns) per lag beyond the first."""
    return INPUTS + (len(ECHO_LAGS) - 1) * ACTIONS if ECHO_LAGS else INPUTS


def lagged_observation(history, mood):
    """The echo brain's observation of a bar: ``history`` the bars heard so far, oldest first, each (rows, SLOTS), the
    last the bar heard now (lag 1); ``mood`` the mood rows of the bar to write. The bar heard now with its mood rows
    (observe), then for every lag k in ECHO_LAGS beyond the first the bar heard k - 1 bars before it as a one-hot slot
    block with the same sounding rule, or an all-zero block where the history does not reach back that far. Without
    ECHO_LAGS, observe(history[-1], mood). The player builds its observations with this, from the bars it played."""
    x = observe(history[-1], mood)
    if not ECHO_LAGS:
        return x
    blocks = [x]
    for k in ECHO_LAGS[1:]:
        if len(history) >= k:
            blocks.append(observe(history[-k], mood)[:, :ACTIONS])
        else:
            blocks.append(np.zeros((len(x), ACTIONS)))
    return np.hstack(blocks)


class EchoBrain:
    """A Cadence brain of the echo shape behind the trainer's and the readers' plain calls: ``act`` and ``lesson`` take the
    plain observation of the bar heard (observe) and add the lag blocks from the bars this wrapper saw acted on since
    the last reset (the same blocks as lagged_observation). An act records its bar; a lesson does not, so a lesson
    before the act reads the same lags as the act. Everything else is the brain's own."""

    def __init__(self, brain, lags=None):
        object.__setattr__(self, "_brain", brain)
        object.__setattr__(self, "_lags", tuple(lags or ECHO_LAGS))
        object.__setattr__(self, "_heard", [])

    def _extend(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        heard = self._heard
        if heard and len(heard[-1]) != len(x):        # other streams: the lags start over
            heard.clear()
        blocks = [x]
        for k in self._lags[1:]:
            blocks.append(heard[-(k - 1)] if len(heard) >= k - 1 else np.zeros((len(x), ACTIONS)))
        return np.hstack(blocks)

    def act(self, x, *args, **kwargs):
        out = self._brain.act(self._extend(x), *args, **kwargs)
        self._heard.append(np.atleast_2d(np.asarray(x, dtype=float))[:, :ACTIONS].copy())
        del self._heard[:-max(self._lags)]
        return out

    def lesson(self, x, *args, **kwargs):
        return self._brain.lesson(self._extend(x), *args, **kwargs)

    def reset(self, *args, **kwargs):
        self._heard.clear()
        return self._brain.reset(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_brain", "_lags", "_heard"):    # copies and pickles: no delegation
            raise AttributeError(name)
        return getattr(self._brain, name)

    def __setattr__(self, name, value):
        setattr(self._brain, name, value)


def echo(brain):
    """The brain wrapped for the echo shape when ECHO_LAGS is set, else the brain itself."""
    return EchoBrain(brain) if ECHO_LAGS and not isinstance(brain, EchoBrain) else brain


def common_settings():
    code = max(192, 64 * -(-(INPUTS + 1) // 64))     # the value path needs a learned code >= the observation
    return dict(
        inputs=brain_inputs(), actions=ACTIONS, slots=tuple(SIZES), arousal=True,
        recall_heads=L, recall_tokens=range(ACTIONS), recall_gain=1.0, recall_action=True,
        recall_capacity=512, learned_code=code, attention=True, learning_surprise=True,
    )


PRESETS = {
    "base": dict(modules=(128,)),
    "deep": dict(modules=(256, 128)),
    "composer": dict(modules=(256, 128, 96, 64), apical=0.25, teaching="deep",
                     recall_window=2, recall_when=True, recall_position=64, recall_capacity=128,
                     recall_action=False, learning_surprise_share=True),
    "apical": dict(modules=(256, 128, 64), apical=0.25),
    "wide": dict(modules=(512, 256)),
    "nocode": dict(modules=(256, 128), learned_code=None, attention=False),
    "tiny": dict(modules=(32,), recall_capacity=128),
}

# Open labels use the ordinary learned-code recall ports and the address lesson.
# The direct token-to-motor shortcut requires a complete categorical target.
AUDIBLE_RECALL = dict(recall_tokens=None, recall_gain=0.0, recall_action=False)
PRESETS["composer-learned-recall"] = {
    **PRESETS["composer"], **AUDIBLE_RECALL, "learned_code": 512, "recall_temperature": 2.0,
}
PRESETS[hierarchy.PRESET] = {
    **PRESETS["composer-learned-recall"], "modules": hierarchy.WIDTHS, "coordinator_depth": 1,
    "memory_amplitude": 0.0,
}
# One pathway per instrument. The coordinator writes the arrangement: which instruments sound.
PRESETS[hierarchy.INSTRUMENT_PRESET] = {
    **PRESETS[hierarchy.PRESET], "modules": hierarchy.INSTRUMENT_WIDTHS,
}
# The pathway brains with the direct route. A remembered bar reads straight into the motor slots, so a brain can
# hold what it heard; the pathways learn the changes. The route needs a learned code at least as wide as the
# observation and a label for every slot, so these presets teach every slot (a silent part keeps what it had).
PRESETS[hierarchy.ROUTE_PRESET] = {
    **PRESETS["composer"], "modules": hierarchy.WIDTHS, "coordinator_depth": 1, "memory_amplitude": 0.0,
}
PRESETS[hierarchy.INSTRUMENT_ROUTE_PRESET] = {
    **PRESETS[hierarchy.ROUTE_PRESET], "modules": hierarchy.INSTRUMENT_WIDTHS,
}
PRESETS[hierarchy.TRIO_PRESET] = {**PRESETS[hierarchy.ROUTE_PRESET], "modules": hierarchy.TRIO_WIDTHS}
for _narrow, _wide in hierarchy.WIDE_PRESETS.items():       # the same wiring, every region hierarchy.WIDE times as wide
    PRESETS[_wide] = {**PRESETS[_narrow], "modules": hierarchy.wide(PRESETS[_narrow]["modules"])}
PRESETS[hierarchy.PARTS_PRESET] = {**PRESETS[hierarchy.ROUTE_PRESET], "modules": hierarchy.PARTS_WIDTHS}
# Generation 2 (s2, Cadence 0.104.0): no direct route from memory to the motor slots, in lessons and in scoring; the
# four attention heads read at the coordinator (pathway_recall off), on unit-normalized keys (recall_key_norm) at
# Cadence's own recall temperature (0.25, where the delayed-cue chamber passed with unit keys; the 2.0 above was set
# for raw dot products); every pathway's detail hears the shared learned code (pathway_reads).
_S2 = {"pathway_reads": True, "recall_key_norm": True, "recall_temperature": 0.25,
       # the learned code as wide as the observation (1,792 for score v5f), as generation 1 trained it: the code is
       # where the detail of a bar is held, and the details read it (composer-learned-recall's 512 is not taken)
       "learned_code": "observation",
       # the coordinator saturated within the first pass without it (local study, 26 Nov: slope 0.01-0.02 against
       # 0.31-0.34 with a budget of 4); it bounds the learned basal projections into each hidden region
       "learning_drive_budget": 4.0, "learning_credit_window": 15}
PRESETS[hierarchy.DUO_PRESET] = {**PRESETS[hierarchy.PRESET], **_S2, "modules": hierarchy.WIDTHS}
PRESETS[hierarchy.QUAD_PRESET] = {**PRESETS[hierarchy.INSTRUMENT_PRESET], **_S2}
OPEN_LABEL_PRESETS = ("composer-learned-recall", hierarchy.PRESET, hierarchy.INSTRUMENT_PRESET,
                      hierarchy.DUO_PRESET, hierarchy.QUAD_PRESET)


def shape_from(card):
    """Take the shape switches a checkpoint card records. Every reader of a card calls this before it
    configures the layout or loads the score."""
    global RESERVE, SID_LIVE, EAR_LIVE, HEAR_PARTS, LOOKAHEAD, SECTION, ECHO_LAGS
    mode = (card or {}).get("reserve")
    ECHO_LAGS = tuple((card or {}).get("echo_lags") or ())
    RESERVE = "sid" if mode is True else (mode or False)
    SID_LIVE = bool((card or {}).get("sid_live"))
    EAR_LIVE = bool((card or {}).get("ear"))
    HEAR_PARTS = bool((card or {}).get("hear_parts"))
    LOOKAHEAD = bool((card or {}).get("lookahead"))
    SECTION = bool((card or {}).get("section"))


def pathway_slots(architecture):
    """The motor slots of each pathway in the configured layout (hierarchy.owners)."""
    if not (PATCH_BOOK and STEPS and SYN):
        raise ValueError("The hierarchy needs the DSP composer with drum and synth patterns")
    return hierarchy.owners(architecture, STEPS, SYN // len(SYNTH_LAYERS), timbre.width(PATCH_BOOK),
                            tuple(timbre.SID_WIDTH if i in chip_layers() else 0 for i in range(len(SYNTH_LAYERS))),
                            PLAN + SECTIONS)


LAG_BIAS = 1e4         # the head's preference for its lag at birth. The address's logits at the composer's size
#                        are in the thousands (a key of 1,800 values over a temperature of 0.25): 30 changed nothing.


def born_gains(brain, value):
    """The value path's gains at birth: ``value`` on the first head's recall port, zero on every other port.

    Cadence composes the gains at zero and lessons raise them. At the composer's size they rise too slowly
    to matter (0.002 after 95 lessons, where a brain of 8 slots learns 0.5), so the path is born switched on.
    The first head takes the best match, which is the bar just heard. The gains stay learned: every lesson
    still moves them."""
    gains = np.zeros(len(np.asarray(brain.recall.gains)))
    gains[0] = value      # ports come as (recall, next) per head
    brain.recall.gains = gains


def lag_age(lag):
    """The age, when a bar is answered, of the moment a head born at ``lag`` reads: the bar lag - 1 bars
    before the bar heard. With HEAR_PARTS a bar is two moments, and the heads at the lags beyond 1 read the
    moment that holds the drum rows and synth lines alone."""
    if HEAR_PARTS and lag > 1:
        return 2 * (lag - 1) + 1
    return lag - 1


def born_lags(brain, lags, gains, bias=None):
    """Heads born listening at fixed lags: head h prefers the bar that repeats at lag lags[h] (the bar
    lags[h] - 1 bars before the bar heard), and its recall port starts at gains[h]; None leaves a head as
    the library bears it.

    Loops of two and four bars repeat the bar L bars back, not the last one. A head's position term has a
    bias per age (cadence: recall.position_maps, zero at birth); ``bias`` on one age makes the head read that
    bar. The direct route adds the heads' bars at their gains, so every slot takes the value most of the gain
    agrees on. These are initial values of learned parameters, as the born gain is: lessons move the gains,
    the position terms and the addresses.

    With HEAR_PARTS the vote is per row group. The direct route has one gain per read port, for every slot
    alike, so a group cannot have its own gain. A bar is heard as two moments instead (fast_part): the head at
    lag 1 reads the whole bar, the heads at the lags beyond read the moments that hold the drum rows and
    synth lines alone. The arrangement and the sound controls then start as the bar heard, at the first
    head's gain, and the drum rows and synth lines take the vote of all the lags."""
    recall = brain.recall
    bias = LAG_BIAS if bias is None else bias
    maps = []
    for head, (on_x, on_c, base) in enumerate(recall.position_maps):
        base = np.array(base, dtype=float, copy=True)
        if head < len(lags) and lags[head] is not None:
            base[int(recall._bucket(np.array([float(lag_age(lags[head]))]))[0])] = bias
        maps.append((on_x, on_c, base))
    recall.position_maps = maps
    values = np.zeros(len(np.asarray(recall.gains)))
    if len(values):                             # a brain without the direct route has no gains to set
        values[0:2 * len(gains):2] = gains      # ports come as (recall, next) per head
        recall.gains = values


def compose(preset: str, seed: int = 0, **over):
    from cadence import Brain

    global ECHO_LAGS
    if over.get("echo_lags"):            # the echo shape: set before the observation's width is read
        ECHO_LAGS = tuple(int(k) for k in over["echo_lags"])
    kw = common_settings()
    observation_code = kw["learned_code"]
    kw.update(PRESETS[preset])
    kw.update(over)
    if kw.get("learned_code") == "observation":
        kw["learned_code"] = observation_code
    if kw.get("echo_lags"):
        kw["echo_lags"] = list(ECHO_LAGS)
        kw["echo_base"] = INPUTS                     # the lag blocks follow the observation's own columns
    if kw.get("reflex_lag_groups") is not None:
        groups = kw.pop("reflex_lag_groups")
        if groups != "drums_lines":
            raise ValueError("reflex_lag_groups: drums_lines")
        kw["reflex_lag_slots"] = list(range(L + MODS, L + MODS + STEPS + SYN))    # drum rows and synth lines
    born = kw.pop("recall_gain_born", None)
    lags = kw.pop("recall_lags_born", None)                 # with them, the born gain of every head: recall_lag_gains_born
    lag_gains = kw.pop("recall_lag_gains_born", None)
    if preset in hierarchy.PRESETS:
        architecture = hierarchy.PRESETS[preset]
        label = "cadance-hierarchy-v1" if architecture == hierarchy.ARCHITECTURE else f"cadance-{architecture}"
        brain = hierarchy.compose(kw, pathway_slots(architecture), seed, label)
        settings = {k: (list(v) if isinstance(v, (tuple, range)) else v) for k, v in kw.items()}
        settings["architecture"] = architecture
        settings["plastic_synapses"] = int(np.count_nonzero(brain.learner.plastic_synapses))
    else:
        if kw.get("learned_code") is None:
            kw.pop("learned_code", None)
            kw["attention"] = False
        settings = {k: (list(v) if isinstance(v, (tuple, range)) else v) for k, v in kw.items()}
        brain = Brain.compose(seed=seed, **kw)
    if born is not None:
        born_gains(brain, born)
        settings["recall_gain_born"] = born
    if lags is not None:
        if lag_gains is None or len(lag_gains) != len(lags):
            raise ValueError("recall_lags_born needs a born gain for every head (recall_lag_gains_born)")
        born_lags(brain, list(lags), [float(g) for g in lag_gains])
        settings["recall_lags_born"], settings["recall_lag_gains_born"] = list(lags), list(lag_gains)
    return brain, settings


def mute_reflex(brain, lag=None):
    """The reflex arcs (heard slot to the same motor slot; learned code to the motor) set to zero on this brain: the
    reflex_cut reading. False for a brain without one."""
    graph = brain.connectome
    populations = graph.populations
    if "motor" not in populations:
        return False
    # the sensory reflex (heard slot to motor slot) and the code reflex (learned code to the pathways' motor slots)
    sources = np.concatenate([np.asarray(populations[name], dtype=np.int64) for name in ("sensory", "code") if name in populations])
    if lag is not None:          # one echo lag's block only: lag 1 the bar heard now, the others after the observation
        heard = np.asarray(populations["sensory"], dtype=np.int64)
        first = heard.min() if lag == ECHO_LAGS[0] else heard.min() + INPUTS + (ECHO_LAGS.index(lag) - 1) * ACTIONS
        sources = np.arange(first, first + ACTIONS)
    edges = np.isin(np.asarray(graph.pre), sources) & np.isin(np.asarray(graph.post), populations["motor"])
    if not edges.any():
        return False
    efficacy = np.array(brain.brain.efficacy, dtype=float, copy=True)
    efficacy[edges] = 0
    brain.brain.efficacy = efficacy
    return True


def mute_coordination(brain):
    """The coordinator's outgoing links to the pathways (every synapse from the association into a pathway's detail
    or plan) set to zero on this brain, its own motor slots left wired: a private evaluation copy for the
    coordinator_cut reading. False for a brain without pathways."""
    graph = brain.connectome
    populations = graph.populations
    branches = [i for name, index in populations.items() if name.endswith(("_detail", "_plan")) for i in index]
    if "association" not in populations or not branches:
        return False
    edges = np.isin(np.asarray(graph.pre), populations["association"]) & np.isin(np.asarray(graph.post), branches)
    if not edges.any():
        return False
    efficacy = np.array(brain.brain.efficacy, dtype=float, copy=True)
    efficacy[edges] = 0
    brain.brain.efficacy = efficacy
    return True


def mute_upper(brain, phrase_only=False):
    """The upper coordinator's outgoing links (two coordinator layers: every synapse from phrase_context) set to zero
    on this brain, its inputs left wired: the upper_cut reading. With ``phrase_only`` just its link down to the phrase
    coordinator, its own arrangement slots kept: the upper_to_phrase_cut reading. False for a brain without an upper
    coordinator."""
    graph = brain.connectome
    populations = graph.populations
    if "phrase_context" not in populations or "association" not in populations:
        return False
    edges = np.isin(np.asarray(graph.pre), populations["phrase_context"])
    if phrase_only:
        edges &= np.isin(np.asarray(graph.post), populations["association"])
    if not edges.any():
        return False
    efficacy = np.array(brain.brain.efficacy, dtype=float, copy=True)
    efficacy[edges] = 0
    brain.brain.efficacy = efficacy
    return True


def observe(state: np.ndarray, mood: np.ndarray) -> np.ndarray:
    """state (rows, 12) slot values, mood (rows, 2) -> one-hot per slot + wanted mood."""
    state = np.asarray(state, dtype=int).reshape(-1, SLOTS)
    rows = len(state)
    x = np.zeros((rows, INPUTS))
    row_ids = np.arange(rows)[:, None]
    columns = OFFSETS + state
    x[row_ids, columns] = 1.0
    if PATCH_BOOK and PATCH_BOOK.get("hearing") == "sounding":
        # A resting layer keeps its last patch for the teacher. Its inaudible
        # controls and line must not dominate the learned code or recall query.
        width = timbre.width(PATCH_BOOK)
        line = L + MODS + STEPS
        for i in range(3):
            active = state[:, i + 1:i + 2] > 0
            notes = slice(line + i * (SYN // 3), line + (i + 1) * (SYN // 3))
            patch = slice(line + SYN + i * width, line + SYN + (i + 1) * width)
            x[row_ids, columns[:, notes]] = active
            x[row_ids, columns[:, patch]] = active
            if PATCH_BOOK.get("samples"):
                sampler = active & (state[:, patch.start:patch.start+1] == timbre.VOICES.index("sampler"))
                x[row_ids, columns[:, patch.start+len(timbre.SIZES):patch.stop]] = sampler
            if i in chip_layers():
                # A SID block is heard while its layer sounds and its switch is on. Without SID lessons the
                # block is never heard, whatever its slots hold.
                k = chip_layers().index(i)
                chip = slice(LIVE + k * timbre.SID_WIDTH, LIVE + (k + 1) * timbre.SID_WIDTH)
                playing = active & (state[:, chip.start:chip.start + 1] > 0) if SID_LIVE else np.zeros_like(active)
                x[row_ids, columns[:, chip]] = playing
    if SYNTH and SYNTH.get("kind") == "splitsteps":
        start = L + MODS + STEPS
        kinds = state[:, start:start + SYN:2]
        pitch_columns = columns[:, start + 1:start + SYN:2]
        x[row_ids, pitch_columns] *= kinds == 2
    mood = np.atleast_2d(np.asarray(mood, dtype=float))
    x[:, ACTIONS:ACTIONS + MOOD] = mood[:, :MOOD]
    if EAR_LIVE and RESERVED_INPUTS and mood.shape[1] >= MOOD + EAR_INPUTS:
        # A mood row of a score with an ear carries, after the mood, what the ear heard of this bar.
        first = ACTIONS + MOOD + content_width()
        x[:, first:first + EAR_INPUTS] = mood[:, MOOD:MOOD + EAR_INPUTS]
    if CONTENT is not None:
        x[:, ACTIONS + MOOD:ACTIONS + MOOD + content_width()] = np.concatenate([CONTENT[l][state[:, l]] * (state[:, SLOTS - L + l, None] if ONOFF else 1) for l in range(L)], axis=1)
    return x


def sounds(states) -> np.ndarray:
    """What sounds, per layer (0 = silent): with ONOFF the sound slot times the layer's on/off slot."""
    st = np.asarray(states, dtype=int)
    return st[..., :L] * st[..., SLOTS - L:SLOTS] if ONOFF else st[..., :L]


def reference_edits(before, after, played, own):
    """Carry the played pattern through reference holds, with the reference arranging entrances and exits."""
    target = after.copy()
    start = L + MODS
    sounding = (sounds(before) > 0) & (sounds(after) > 0) & (sounds(played) > 0)
    drum_width = STEPS // len(DRUM_ROWS)
    for i, voice in enumerate(DRUM_ROWS):
        row = slice(start + i * drum_width, start + (i + 1) * drum_width)
        if ROWS:
            has_hit = np.asarray(ROWS[voice]).any(axis=1)
            same_audibility = (has_hit[played[:, row]] == has_hit[after[:, row]]).ravel()
        else:
            edited = np.where(before[:, row] != after[:, row], after[:, row], played[:, row])
            keep = own & sounding[:, 0] & (played[:, row].any(axis=1) == before[:, row].any(axis=1))
            keep &= edited.any(axis=1) == after[:, row].any(axis=1)
            target[keep, row] = edited[keep]
            continue
        keep = own & sounding[:, 0] & (before[:, row] == after[:, row]).all(axis=1) & same_audibility
        target[keep, row] = played[keep, row]
    width = SYN // len(SYNTH_LAYERS)
    for i in range(len(SYNTH_LAYERS)):
        slot = start + STEPS + i * width
        line = slice(slot, slot + width)
        kind = SYNTH.get("kind")
        if kind in ("steps", "splitsteps"):
            old, new, heard = before[:, line], after[:, line], played[:, line]
            if kind == "splitsteps":
                old, new, heard = (decode_split_notes(notes) for notes in (old, new, heard))
            edited = np.where(old != new, new, heard)
            keep = own & sounding[:, i + 1] & (heard.any(axis=1) == old.any(axis=1))
            keep &= (heard >= 2).any(axis=1) == (old >= 2).any(axis=1)
            keep &= (edited.any(axis=1) == new.any(axis=1)) & ((edited >= 2).any(axis=1) == (new >= 2).any(axis=1))
            target[keep, line] = encode_split_notes(edited[keep]) if kind == "splitsteps" else edited[keep]
            continue
        candidates = own & sounding[:, i + 1] & (before[:, line] == after[:, line]).all(axis=1)
        for row in np.flatnonzero(candidates):
            # Empty notes cannot stand in for a reference line that keeps playing.
            heard_notes, target_notes = decode_synth(played[row])[i], decode_synth(after[row])[i]
            if (heard_notes.any(), (heard_notes >= 2).any()) == (target_notes.any(), (target_notes >= 2).any()):
                target[row, line] = played[row, line]
    return target


def motif_return_ages(bars):
    """An eight- or sixteen-bar reprise after a different or silent preceding bar."""
    start = L + MODS + STEPS
    lines = bars[..., start:start + SYN].reshape(*bars.shape[:2], len(SYNTH_LAYERS), -1)
    kind = SYNTH.get("kind")
    if kind == "splitsteps":
        lines = decode_split_notes(lines)
    onsets = (lines >= 2).any(axis=-1) if kind in ("steps", "splitsteps") else lines[..., 0] > 0
    playing = (sounds(bars)[..., 1:] > 0) & onsets
    ages = np.zeros(playing.shape, dtype=np.int8)
    for age in (16, 8):
        if bars.shape[1] <= age:
            continue
        same = (lines[:, age:] == lines[:, :-age]).all(axis=3)
        departed = (~playing[:, age - 1:-1] |
                    (lines[:, age - 1:-1] != lines[:, :-age]).any(axis=3))
        reprise = playing[:, age:] & playing[:, :-age] & same & departed
        ages[:, age:][reprise] = age
    return ages


def heard_reprises(target, ages, history):
    """Teach the motif the stream actually heard at the reprise's anchor."""
    result = target.copy()
    start = L + MODS + STEPS
    width = SYN // len(SYNTH_LAYERS)
    for row, layer in zip(*np.nonzero(ages)):
        anchor = history[len(history) - int(ages[row, layer])][row]
        slot = start + layer * width
        line = slice(slot, slot + width)
        kind = SYNTH.get("kind")
        if kind == "splitsteps":
            has_onsets = (anchor[line][::2] == 2).any()
        else:
            has_onsets = (anchor[line] >= 2).any() if kind == "steps" else anchor[slot] > 0
        if sounds(anchor)[layer + 1] > 0 and has_onsets:
            result[row, line] = anchor[line]
    return result


def audible_kick(bars, kick_rows=None):
    """Read kick presence from either a row token or the first drum's steps."""
    start = L + MODS
    hits = (kick_rows[bars[..., start]] if kick_rows is not None else
            bars[..., start:start + STEPS // len(DRUM_ROWS)].any(axis=-1))
    return (sounds(bars)[..., 0] > 0) & hits


def arrangement_transitions(bars, kick_rows=None):
    """Find entrances and exits with two consistent bars on each side."""
    gates = sounds(bars) > 0
    kick = audible_kick(bars, kick_rows)[..., None]
    roles = np.concatenate((gates, kick), axis=-1)
    changes = np.zeros_like(roles)
    changes[:, 2:-1] = ((roles[:, 1:-2] != roles[:, 2:-1]) &
                        (roles[:, :-3] == roles[:, 1:-2]) &
                        (roles[:, 2:-1] == roles[:, 3:]))
    return changes


def focus_arrangement(target, changes):
    """Leave each selected row's unrelated instruments and patch controls open."""
    focus = changes.any(axis=1)
    labels = target.copy()
    keep = np.zeros(labels.shape, dtype=bool)
    keep[:, :L] = changes[:, :L]
    keep[:, 0] |= changes[:, L]
    start = L + MODS
    keep[:, start:start + STEPS // len(DRUM_ROWS)] = changes[:, L, None]
    labels[focus[:, None] & ~keep] = -1
    return labels


def audible_targets(state):
    """Leave motor slots open when their instrument contributes no audio."""
    labels = np.asarray(state, dtype=int).copy().reshape(-1, SLOTS)
    sounding = sounds(labels) > 0
    base = L + MODS
    labels[:, L:base].reshape(-1, L, 2)[~sounding] = -1
    labels[:, base:base+STEPS][~sounding[:, 0]] = -1
    line = base + STEPS
    patch = line + SYN
    labels[:, line:patch].reshape(len(labels), 3, SYN // 3)[~sounding[:, 1:]] = -1
    if SYNTH and SYNTH.get("kind") == "splitsteps":
        kinds = labels[:, line:patch:2]
        pitches = labels[:, line+1:patch:2]
        pitches[kinds != 2] = -1
    controls = labels[:, patch:LIVE].reshape(len(labels), 3, timbre.width(PATCH_BOOK))
    controls[~sounding[:, 1:]] = -1
    if PATCH_BOOK.get("samples"):
        synth = controls[:, :, 0] != timbre.VOICES.index("sampler")
        controls[:, :, -3:][synth] = -1
    return labels


def focus_detail(target, selected, rng):
    """Teach accents and pitched onsets alongside an equal sample of contrasting steps."""
    start = L + MODS
    stop = start + STEPS + SYN
    steps = target[:, start:stop].reshape(len(target), len(DRUM_ROWS) + len(SYNTH_LAYERS), 16)
    positive = steps >= 2  # an accent for drums, a pitched attack for synths
    other = (steps >= 0) & ~positive
    draw = np.where(other, rng.random(steps.shape), np.inf)
    rank = np.argsort(np.argsort(draw, axis=-1), axis=-1)
    keep = positive | (other & (rank < positive.sum(axis=-1, keepdims=True)))
    focused = selected & positive.any(axis=(1, 2))
    labels = target.copy()
    labels[focused] = -1
    labels[focused, start:stop] = np.where(keep, steps, -1).reshape(len(target), -1)[focused]
    return labels


def sample_voices(state):
    """Mask of the three pitched voices currently playing a one-shot."""
    start = L + MODS + STEPS + SYN
    modes = np.asarray(state)[..., start:LIVE:timbre.width(PATCH_BOOK)]
    return (modes == timbre.VOICES.index("sampler")) & (sounds(state)[...,1:] > 0)


def sound_identities(states):
    """Comparable timbres for change/hold statistics. The DSP sound slots are gates."""
    states = np.asarray(states, dtype=int)
    result = sounds(states).copy()
    if PATCH_BOOK:
        width = timbre.width(PATCH_BOOK)
        start = L + MODS + STEPS + SYN
        for layer in range(1, L):
            controls = states[..., start:start + width]
            identity = controls[..., 0].copy()
            for knob in ("wave", "cutoff", "reso", "decay", "drive"):
                identity = 7 * identity + controls[..., 3 + timbre.NAMES.index(knob)]
            if PATCH_BOOK.get("samples"):
                extras = controls[..., len(timbre.SIZES):]
                sample = (extras[..., 0]*7 + extras[..., 1])*2 + extras[..., 2]
                identity = np.where(controls[...,0]==timbre.VOICES.index("sampler"),
                                    identity*168+sample, identity)
            result[..., layer] *= identity + 1
            start += width
    return result


def external(states) -> np.ndarray:
    """States in the plain form every reader of sounds knows: the four sound slots hold what sounds (0 = silent),
    the on/off slots are dropped; modulation, drum rows and synth lines stay where they are."""
    st = np.asarray(states, dtype=int)
    if not ONOFF:
        return st
    return np.concatenate([sounds(st), st[..., L:SLOTS - L]], axis=-1)


def set_sound(state, layer: int, token: int):
    """Make `layer` play `token` (0 = silent) in a state (list or array, changed in place)."""
    if ONOFF:
        if token > 0:
            state[layer] = int(token)
        state[SLOTS - L + layer] = int(token > 0)
    else:
        state[layer] = int(token)
    return state


def beat_pattern(state, kit_token) -> dict | None:
    """The brain's own beat for this bar as an engine pattern on its kit's voices (snare row -> the kit's snare or
    clap, hats row -> its closed hat, open hat or cymbal: whichever the kit's own pattern uses most)."""
    if not STEPS:
        return None
    d = decode_beat(state)
    hits = (kit_token or {}).get("pattern", {}).get("hits", {}) if kit_token else {}
    use = lambda names: max(names, key=lambda v: sum(1 for x in hits.get(v, []) if x and x > 0))
    def vel(row):
        """Hit 0.7, accent 1.0; in a very busy row (> 8 hits: what the reader made of a texture) plain hits are ghost notes."""
        busy = sum(1 for v in row if v) > 8
        return [0.0 if v == 0 else ((0.25 if busy else 0.7) if v == 1 else (0.8 if busy else 1.0)) for v in row]
    return {"hits": {"kick": vel(d[0]), use(("snare", "clap")): vel(d[1]), use(("ch", "oh", "cym")): vel(d[2])}}


def mods_out(state) -> list:
    """The engine's modulation values [[filter, space] x 4] in -1..1 from a state's slots 4..11."""
    mid = (M - 1) / 2
    s = [int(v) for v in np.asarray(state).reshape(-1)[L:L + MODS]]
    return [[(s[2 * l] - mid) / mid, (s[2 * l + 1] - mid) / mid] for l in range(L)]


def mood_rows(mood2, style=None, bars=None):
    """Energy/brightness rows (n, 2) + the one-hot style each row is played in (when the corpus has styles) + the
    bar clock (when on): the bar count of each row's bar as 6 binary bits, so 8-, 16-, 32-bar phrase positions are
    heard (bars = the bar indices; default 0..n-1)."""
    m = np.atleast_2d(np.asarray(mood2, dtype=float))[:, :2]
    cols = [m]
    if STYLES:
        onehot = np.zeros((len(m), len(STYLES)))
        if style in STYLES:
            onehot[:, STYLES.index(style)] = 1.0
        cols.append(onehot)
    if CLOCK:
        idx = np.arange(len(m)) if bars is None else np.asarray(bars, dtype=int).reshape(-1)
        cols.append(((idx[:, None] >> np.arange(CLOCK)) & 1).astype(float))
    return np.hstack(cols) if len(cols) > 1 else m


def row_vocab(data, k=32):
    """Per drum voice the k most common 16-step rows of the TRAINING tracks (the silent row always included)."""
    from collections import Counter
    vocab = {}
    for v, name in enumerate(DRUM_ROWS):
        cnt = Counter()
        for tr in data["tracks"]:
            if tr.get("held_out"):
                continue
            for bar in tr["drums"]:
                cnt[tuple(int(x) for x in bar[v])] += 1
        rows = [tuple([0] * 16)] + [r for r, _ in cnt.most_common(k + 1) if any(r)]
        vocab[name] = [list(r) for r in rows[:k]]
    return vocab


SYNTH_VOCAB = {"kind": "rhythm", "k": 128}   # 11-23: "rows" (k most common exact rows) taught held notes: 0.2 lead
#                                             onsets per bar against 3.4 in the tracks; set per run (train.py --synth-vocab)


def synth_vocab(data, k=None, kind=None):
    """Per synth layer a vocabulary of 16-step note rows (0 rest, 1 tie, 2+ onset at a pitch class) from the TRAINING
    tracks, the silent row always included. kind "rows": the k most common exact rows (most of them held notes);
    kind "rhythm" (default): the rhythms (rest / tie / onset per step) by frequency, a rhythm kept when it differs
    from every kept one in at least two steps, each with its most common pitches, so the vocabulary holds the lines'
    real rhythms, the busy ones too."""
    kind = kind or SYNTH_VOCAB["kind"]; k = k or SYNTH_VOCAB["k"]
    if kind in ("steps", "splitsteps"):
        return {"kind": kind}
    if kind == "factored":
        return factored_vocab(data, k_rhythm=k or 256, k_contour=SYNTH_VOCAB.get("contours", 128))
    if kind == "rhythm":
        return _rhythm_vocab(data, k)
    k = k or 64
    return _rows_vocab(data, k)


def _rhythm_vocab(data, k):
    from collections import Counter, defaultdict
    vocab = {}
    for v, name in enumerate(SYNTH_LAYERS):
        rc, full = Counter(), defaultdict(Counter)
        for tr in data["tracks"]:
            if tr.get("held_out") or "synth" not in tr:
                continue
            for bar in tr["synth"]:
                r = tuple(int(x) for x in bar[v]); h = tuple(2 if x >= 2 else x for x in r)
                rc[h] += 1; full[h][r] += 1
        kept = [tuple([0] * 16)]
        arr = [np.zeros(16, dtype=int)]
        for h, _ in rc.most_common():
            if len(kept) >= k:
                break
            if not any(h):
                continue
            a = np.asarray(h)
            if min(int((a != b).sum()) for b in arr) >= 2:
                kept.append(h); arr.append(a)
        vocab[name] = [list(full[h].most_common(1)[0][0]) if any(h) else [0] * 16 for h in kept]
    return vocab


ONOFF = False          # 11-24: each layer gets an ON/OFF slot (2 values, at the end of the state) and its sound slot keeps
#                        the last sound while the layer is off. Without it "off" is one of 97 values of the sound slot: at
#                        the low reading temperature it beat every single sound, a layer that went off stayed off three
#                        times longer than in the tracks and the brains played drums alone in 21-60% of the bars (tracks
#                        1%). Set from the card ("onoff") before configure(); sounds()/external() read what sounds.
SYNTH_FILL = False     # 11-23: while a synth layer's sound is off (token 0) its line is not heard; with SYNTH_FILL the line
#                        is taught as the last line that sounded (forward, then backward at a track's start) instead of a
#                        rest row, so "rest" is no majority class (round-5 brains wrote rest rows only: bass 1%, lead 0%)
CONTOUR_LEN = 8                 # a contour holds the pitch steps of up to 8 onsets (cycled over longer lines)
_MEMO: dict = {}

HEAR_FIRST = False     # 11-25: a bar is heard as a moment of its own before the brain is taught or answers on it.
#                        The recall memory holds the moments before the present one, so a brain that answers as it
#                        first hears a bar reads the bar before it, and the direct route is one bar late on every
#                        change. Heard first, the bar is in the memory the answer reads. The memory keeps one
#                        slot per bar: the moment the bar was heard.


HEAR_PARTS = False     # 11-26: with HEAR_FIRST a bar is heard as two moments: first its drum rows and synth lines
#                        alone (fast_part), then the whole bar. The memory then holds two slots per bar, and a head
#                        that reads the first of them brings the drum rows and synth lines of that bar and nothing
#                        of its arrangement or sound controls (born_lags). Set from the card ("hear_parts").


def fast_part(x):
    """The first of a bar's two moments (HEAR_PARTS): the observation with the columns of the sounds, the
    mods and the sound controls silent. The drum rows, the synth lines and what is no slot (mood, clock, ear)
    stay as heard."""
    x = np.array(x, dtype=float, copy=True)
    first = L + MODS
    x[..., :OFFSETS[first]] = 0.0
    if first + STEPS + SYN < SLOTS:
        x[..., OFFSETS[first + STEPS + SYN]:ACTIONS] = 0.0
    return x


def hear(brain, x):
    """The moment a bar is heard, before the lesson or the answer on it (HEAR_FIRST). It writes the bar to
    the recall memory: one moment, or with HEAR_PARTS two."""
    if HEAR_FIRST:
        if HEAR_PARTS:
            brain.act(fast_part(x), greedy=True)
        brain.act(x, greedy=True)


def answer(brain, x):
    """One bar from the brain, as a greedy act. With HEAR_FIRST the bar is heard first, and the answer leaves
    no second copy of it in the recall memory."""
    if not HEAR_FIRST:
        return brain.act(x, greedy=True)
    hear(brain, x)
    kept = brain.recall.snapshot()
    out = brain.act(x, greedy=True)
    brain.recall.restore(kept)
    return out



def factored_vocab(data, k_rhythm=256, k_contour=128):
    """The factored synth code (11-23: a whole-row vocabulary fixes the pitches with the rhythm; lead lines taught
    with 13% pitch movement between onsets against 49% in the labels). Per layer three slots: a RHYTHM (16 steps of
    rest / tie / onset, the lines' rhythms by frequency, kept when >= 2 steps from every kept one), a CONTOUR (the
    pitch classes of the first CONTOUR_LEN onsets relative to the first, the most common ones), a ROOT (the first
    onset's pitch class). A bar's line = the rhythm's onsets sung on root + contour (cycled)."""
    from collections import Counter
    rhythm = {k: [[2 if x >= 2 else x for x in r] for r in rows] for k, rows in _rhythm_vocab(data, k_rhythm).items()}
    contour = {}
    for v, name in enumerate(SYNTH_LAYERS):
        cnt = Counter()
        for tr in data["tracks"]:
            if tr.get("held_out") or "synth" not in tr:
                continue
            for bar in tr["synth"]:
                on = [int(x) - 2 for x in bar[v] if x >= 2]
                if on:
                    cnt[tuple((p - on[0]) % 12 for p in on[:CONTOUR_LEN])] += 1
        kept = [(0,)] + [c for c, _ in cnt.most_common(k_contour + 1) if c != (0,)]
        contour[name] = [list(c) for c in kept[:k_contour]]
    return {"kind": "factored", "rhythm": rhythm, "contour": contour}


def _contour_dist(a, b):
    n = min(len(a), len(b))
    d = sum(min((x - y) % 12, (y - x) % 12) for x, y in zip(a[:n], b[:n]))
    return d + 2 * abs(len(a) - len(b))


def encode_factored(row, v):
    """(rhythm, contour, root) indices of a 16-step note row of synth layer v under the factored code."""
    name = SYNTH_LAYERS[v]
    h = tuple(2 if x >= 2 else int(x) for x in row)
    key = ("h", v, h)
    if key not in _MEMO:
        _MEMO[key] = _nearest_note_row(list(h), SYNTH["rhythm"][name])
    on = [int(x) - 2 for x in row if x >= 2]
    if not on:
        return _MEMO[key], 0, 0
    c = tuple((p - on[0]) % 12 for p in on[:CONTOUR_LEN])
    ck = ("c", v, c)
    if ck not in _MEMO:
        _MEMO[ck] = min(range(len(SYNTH["contour"][name])), key=lambda i: _contour_dist(c, SYNTH["contour"][name][i]))
    return _MEMO[key], _MEMO[ck], on[0] % 12


def decode_factored(rhythm_i, contour_i, root, v):
    name = SYNTH_LAYERS[v]
    h = SYNTH["rhythm"][name][int(rhythm_i)]; c = SYNTH["contour"][name][int(contour_i)] or [0]
    out, j = [], 0
    for x in h:
        if x >= 2:
            out.append(2 + (int(root) + int(c[j % len(c)])) % 12); j += 1
        else:
            out.append(int(x))
    return out


def _rows_vocab(data, k=64):
    """Per synth layer the k most common 16-step note rows of the TRAINING tracks (the silent row always included)."""
    from collections import Counter
    vocab = {}
    for v, name in enumerate(SYNTH_LAYERS):
        cnt = Counter()
        for tr in data["tracks"]:
            if tr.get("held_out") or "synth" not in tr:
                continue
            for bar in tr["synth"]:
                cnt[tuple(int(x) for x in bar[v])] += 1
        rows = [tuple([0] * 16)] + [r for r, _ in cnt.most_common(k + 1) if any(r)]
        vocab[name] = [list(r) for r in rows[:k]]
    return vocab


def _nearest_note_row(row, vocab_rows):
    a = np.asarray(vocab_rows); r = np.asarray(row)
    onset_a, onset_r = a >= 2, r >= 2
    d = (onset_a != onset_r).sum(axis=1) * 3 + ((a == 1) != (r == 1)).sum(axis=1) + (onset_a & onset_r & (a != r)).sum(axis=1)
    return int(np.argmin(d))


def encode_split_notes(notes):
    return np.stack((np.minimum(notes, 2), np.maximum(notes - 2, 0)), axis=-1).reshape(
        *notes.shape[:-1], notes.shape[-1] * 2)


def decode_split_notes(steps):
    kind, pitch = steps[..., ::2], steps[..., 1::2]
    return np.where(kind == 2, pitch + 2, kind)


def decode_synth(state) -> np.ndarray | None:
    """The 3 x 16 note rows (0 rest, 1 tie, 2+pc) a state holds, or None without synth writing."""
    if not SYNTH:
        return None
    st = np.asarray(state).reshape(-1)
    tail = st[L + MODS + STEPS:L + MODS + STEPS + SYN]
    if SYNTH.get("kind") == "steps":
        return tail.reshape(len(SYNTH_LAYERS), 16)
    if SYNTH.get("kind") == "splitsteps":
        return decode_split_notes(tail).reshape(len(SYNTH_LAYERS), 16)
    if SYNTH.get("kind") == "factored":
        return np.array([decode_factored(tail[3 * v], tail[3 * v + 1], tail[3 * v + 2], v) for v in range(len(SYNTH_LAYERS))])
    return np.array([SYNTH[v][int(tail[i])] for i, v in enumerate(SYNTH_LAYERS)])


def synth_pattern(row, token, octave=None, held=None, velocities=None) -> dict | None:
    """An engine pattern for one synth layer from a note row, voiced near the register of the layer's sound.
    held: the pitch class (0..11) still sounding at the end of the previous bar. A row that opens with ties holds
    that note over the bar line: it sounds again from step 0 (the engine starts notes per bar). Without it, a bar of
    ties and no new note played nothing (11-23: a lit instrument without sound)."""
    if token is None:
        return None
    if octave is None:
        notes = [n for n in (token.get("pattern", {}).get("notes") or []) if n is not None]
        octave = int(np.median(notes) // 12) if notes else 4
    out_notes, vel, length, slide = [None] * 16, [0.0] * 16, [1] * 16, [0] * 16
    last = None
    for i, v in enumerate(row):
        if v >= 2:
            out_notes[i] = 12 * octave + (int(v) - 2); vel[i] = 0.85; last = i
        elif v == 1 and last is not None:
            length[last] += 1; slide[last] = 1
        elif v == 1 and last is None and i == 0 and token.get("voice") != "sampler":
            # a line that opens with ties holds a note over the bar line: the one still sounding (held), or, when
            # nothing sounds yet (a new sound, a new composition), the sound's own pitch (its fitted pattern):
            # held pads and drones (the air layer's coarse rows) always sound
            pc = held
            if pc is None:
                own = [n for n in (token.get("pattern", {}).get("notes") or []) if n is not None]
                pc = int(np.median(own)) % 12 if own else 0
            out_notes[0] = 12 * octave + int(pc); vel[0] = 0.85; last = 0
        else:
            last = None
    accents = [0] * 16
    if velocities is not None:
        for i, note in enumerate(out_notes):
            if note is not None:
                vel[i] = float(velocities[i % 4])
                accents[i] = int(vel[i] >= 6/7)
    return {"notes": out_notes, "vel": vel, "len": length, "slide": slide, "accent": accents}


def held_at_end(row, held=None):
    """The pitch class sounding at the end of a bar of this note row (None after a rest), given the one held into it."""
    cur = held
    for v in row:
        cur = (int(v) - 2) if v >= 2 else (cur if v == 1 else None)
    return cur


def score_bar(state, pool, held, previous):
    """Turn one internal state into exactly the bar heard by the player and the QA renderer."""
    snd = sounds(state).tolist()
    kits = pool["layers"][0]["tokens"]
    kit = kits[snd[0]] if snd[0] < len(kits) else None
    patterns = [beat_pattern(state, kit) if kit else None, None, None, None]
    patches = [None] * L
    rows = decode_synth(state)
    if rows is not None:
        for layer, row in enumerate(rows, 1):
            register = velocities = None
            if PATCH_BOOK:
                width = timbre.width(PATCH_BOOK)
                start = L + MODS + STEPS + SYN + (layer - 1) * width
                controls = state[start:start + width]
                tok, register = timbre.instrument(controls, PATCH_BOOK)
                if width > len(timbre.PATCH_SIZES):
                    velocities = (np.asarray(controls[len(timbre.PATCH_SIZES):len(timbre.SIZES)]) + 1) / 7
                if layer - 1 in chip_layers() and SID_LIVE and SID_RENDER:
                    block = LIVE + chip_layers().index(layer - 1) * timbre.SID_WIDTH
                    chip = timbre.sid_decode(state[block:block + timbre.SID_WIDTH])
                    if chip:                 # the same notes, register, gain and velocities on the SID voice
                        tok = {**chip, "gain": tok["gain"]}
                patches[layer] = tok if snd[layer] else None
                tok = patches[layer]
            else:
                tokens = pool["layers"][layer]["tokens"]
                tok = tokens[snd[layer]] if snd[layer] < len(tokens) else None
            prior = held[layer - 1] if tok and previous[layer] == snd[layer] else None
            patterns[layer] = synth_pattern(row, tok, octave=register, held=prior, velocities=velocities)
            held[layer - 1] = held_at_end(row, prior) if tok else None
    previous[:] = snd
    return dict(tokens=snd, mods=mods_out(state), patterns=patterns, **({"patches": patches} if PATCH_BOOK else {}))


def _nearest_row(row, vocab_rows):
    a = np.asarray(vocab_rows); r = np.asarray(row)
    d = (np.minimum(a, 1) != np.minimum(r, 1)).sum(axis=1) * 2 + (a != r).sum(axis=1)   # hit/no-hit first, accents second
    return int(np.argmin(d))


def decode_beat(state) -> np.ndarray:
    """The 3 x 16 drum steps a state holds (step slots or row slots)."""
    st = np.asarray(state).reshape(-1)
    tail = st[L + MODS:L + MODS + STEPS]
    if ROWS:
        return np.array([ROWS[v][int(tail[i])] for i, v in enumerate(DRUM_ROWS)])
    return tail.reshape(3, 16)


def simple_beat_bars(bars) -> np.ndarray:
    """Per bar: a beat sound with a simple beat: a kick on at least two quarters, no busy snare row (> 8 hits) and no
    busy hat row (> 12). Free runs (QA) and the player start from such passages, so a kickless stretch or a
    machine-gun snare in what follows is the brain's own doing. Without drum rows: the bars with a beat sound."""
    b = external(bars)
    if not STEPS:
        return b[:, 0] > 0
    out = np.zeros(len(b), dtype=bool)
    for i, x in enumerate(b):
        if x[0] > 0:
            d = decode_beat(x)
            out[i] = (d[0][[0, 4, 8, 12]] > 0).sum() >= 2 and (d[1] > 0).sum() <= 8 and (d[2] > 0).sum() <= 12
    return out


def flow_passages(tracks, k, n, seed, style=REFERENCE_STYLE):
    """k passages of n bars in full flow from distinct training tracks, `style` first (the target style): a beat in
    every bar and a SIMPLE beat (simple_beat_bars) in all but one. Free runs (QA, the calibration) and the player
    start from such passages, so what follows is the brain's own doing."""
    rng = np.random.default_rng(seed)
    train = [t for t in tracks if not t["held_out"] and len(t["bars"]) > n + 2]
    first = [t for t in train if t.get("style") == style]
    out = []
    for pool in ([first, train] if len(first) >= k else [train]):
        for i in rng.permutation(len(pool)):
            tr = pool[i]
            if any(tr is o[0] for o in out):
                continue
            b = np.asarray(tr["bars"]); beat = sounds(b)[:, 0] > 0
            ok = simple_beat_bars(b)
            starts = [s for s in range(0, len(b) - n - 1) if beat[s:s + n].all() and ok[s:s + n].sum() >= n - 1]
            if starts:
                out.append((tr, starts[rng.integers(len(starts))]))
            if len(out) == k:
                return out
    return out


def slot_rates(tracks, style=REFERENCE_STYLE) -> np.ndarray:
    """Per slot, how often its value changes from one bar to the next in the training tracks (of `style` when the
    corpus has that style): the pace of every voice in the music, the target of the reading's calibration."""
    tr = [t for t in tracks if not t["held_out"] and len(t["bars"]) > 16]
    if any(t.get("style") == style for t in tr):
        tr = [t for t in tr if t.get("style") == style]
    ch, n = np.zeros(SLOTS), 0
    for t in tr:
        b = np.asarray(t["bars"])
        ch += (b[1:] != b[:-1]).sum(axis=0); n += len(b) - 1
    return ch / max(n, 1)


def motor_activity(brain) -> np.ndarray:
    """The motor neurons' activations of the settle just made, one row per stream."""
    a = brain._activity().activation
    if hasattr(a, "detach"):                  # torch backend (GPU training): bring the activations to the host
        a = a.detach().cpu().numpy()
    elif hasattr(a, "get") and not isinstance(a, dict):
        a = a.get()
    return np.asarray(a)[:, np.asarray(brain.connectome.populations["motor/actions"])]


CHANGE_TEMPERATURE = None  # when set: WHETHER a slot changes is decided at this temperature (1.0 = the brain's own change
#                            probability, unsharpened); WHAT it changes to keeps the calibrated (low) temperature, so a new
#                            sound is the brain's best guess, which it then holds (11-23: brains know what (~85%), not when)
PICK_TEMPERATURE = None   # when set: the NEW value of a changing slot is drawn at this temperature (the change decision keeps
#                           the calibrated one); at a low single temperature every pick is the commonest value ("off")
LINE_SCALE = None         # when set: the drum rows and synth lines (rhythm, contour, root) are read at their temperature x
#                           LINE_SCALE (11-24: at the low temperature the commonest line wins, and the commonest lines are the
#                           simplest: brains played 1-4 notes per bar and 1-5 different lines per 16 bars; the tracks 5-6 and 6-9)
SYNTH_SCALE = None        # when set: the synth lines alone (rhythm, contour, root, not the drum rows) are read at x SYNTH_SCALE
#                           on top of LINE_SCALE (26 Nov: line_scale 1.6 opened the lines but heated the kick row into a wall)
KICK_SCALE = None         # when set (play only): the kick row alone is read at x KICK_SCALE on top of LINE_SCALE (27 Nov:
#                           an echo brain's drum rows need a higher scale to move at the material's rate, and the kick
#                           row goes busy with it; the kick keeps its own scale)
ENTRY_SCALE = None        # when set: whether a SILENT layer comes back is read at temperature x ENTRY_SCALE (the sound it
#                           comes back with keeps the calibrated temperature). 11-24 listening check: at one low temperature
#                           "stay off" (one value) beats every single sound (96 values), so a layer that went off stayed off
#                           ~16 bars against ~5 in the tracks, and the brains played drums alone in 21-60% of the bars (tracks 1%)


def sample_slots(act, state, temperature, rng, *, separate_lines=False) -> np.ndarray:
    """One bar from one stream's motor activations `act`, with COHERENT changes: each slot's probabilities come from
    its motor neurons at ITS temperature (`temperature`: one number, or one per slot from calibrate_slots), and one
    shared draw per group (the four sounds, the modulation levels, the rows) decides which slots change: a slot
    changes when its change probability (1 - p of what it plays now) exceeds the draw, so voices move together
    when the brain signals a transition and hold together otherwise. Only the new value is drawn per slot.
    With separate_lines, each synth voice gets its own draw for its rhythm, contour, and root."""
    temps = np.broadcast_to(np.asarray(temperature, dtype=float), (SLOTS,))
    if LINE_SCALE is not None:
        temps = temps.copy()
        temps[L + MODS:L + MODS + STEPS + SYN] *= float(LINE_SCALE)
    if SYNTH_SCALE is not None:
        temps = temps.copy()
        temps[L + MODS + STEPS:L + MODS + STEPS + SYN] *= float(SYNTH_SCALE)
    if KICK_SCALE is not None and STEPS:
        temps = temps.copy()
        kick = 1 if ROWS else STEPS // len(DRUM_ROWS)                 # the kick is the first drum row
        temps[L + MODS:L + MODS + kick] *= float(KICK_SCALE)
    offsets = OFFSETS
    new = [int(v) for v in state]
    on = list(range(SLOTS - L, SLOTS)) if ONOFF else []
    line_end = L + MODS + STEPS + SYN
    line_start = L + MODS
    groups = [list(range(L)) + on, range(L, line_start)]
    if (STEPS and not ROWS) or (SYNTH and SYNTH.get("kind") in ("steps", "splitsteps")):
        drums_end = line_start + STEPS
        if STEPS:
            width = STEPS // len(DRUM_ROWS)
            groups.extend(range(i, i + width) for i in range(line_start, drums_end, width))
        if SYN:
            width = SYN // len(SYNTH_LAYERS)
            groups.extend(range(i, i + width) for i in range(drums_end, line_end, width))
    elif separate_lines and SYN:
        synth_start = line_start + STEPS
        width = SYN // len(SYNTH_LAYERS)
        groups.append(range(line_start, synth_start))
        groups.extend(range(i, i + width) for i in range(synth_start, line_end, width))
    else:
        groups.append(range(line_start, line_end))
    if PATCH_BOOK:
        width = timbre.width(PATCH_BOOK)
        groups.extend(range(i, i + width) for i in range(line_end, LIVE, width))
        if chips() and SID_LIVE:
            groups.extend(range(i, i + timbre.SID_WIDTH) for i in range(LIVE, chip_end(), timbre.SID_WIDTH))
    groups.extend([slot] for slot in range(chip_end(), SLOTS))         # every plan slot a choice of its own
    # the classes a slot may play: a kit slot with room only has the kits the score has
    widths = list(SIZES)
    if kit_room() and PATCH_BOOK:
        widths[0] = min(T, SIZES[0])
        if new[0] >= widths[0]:
            new[0] = 0
    for g in groups:
        u = rng.random()
        for s in g:
            z = act[offsets[s]:offsets[s] + widths[s]] / max(float(temps[s]), 1e-4)
            p = np.exp(z - z.max()); p /= p.sum()
            cur = int(new[s])
            if ENTRY_SCALE is not None and s < L and cur == 0:
                ze = act[offsets[s]:offsets[s] + widths[s]] / max(float(temps[s]) * float(ENTRY_SCALE), 1e-4)
                pe = np.exp(ze - ze.max()); pe /= pe.sum()
                stay = pe[0]
            elif CHANGE_TEMPERATURE is not None:
                zc = act[offsets[s]:offsets[s] + widths[s]] / max(float(CHANGE_TEMPERATURE), 1e-4)
                pc = np.exp(zc - zc.max()); pc /= pc.sum()
                stay = pc[cur]
            else:
                stay = p[cur]
            if u < 1.0 - stay:
                if PICK_TEMPERATURE is not None:
                    zp = act[offsets[s]:offsets[s] + widths[s]] / max(float(PICK_TEMPERATURE), 1e-4)
                    q = np.exp(zp - zp.max())
                else:
                    q = p.copy()
                q[cur] = 0.0
                if q.sum() > 0:
                    new[s] = int(rng.choice(len(q), p=q / q.sum()))
    result = np.array(new)
    if chips() and not SID_LIVE:
        result[LIVE:chip_end()] = 0       # an idle block plays "off"
    if SYNTH and SYNTH.get("kind") == "splitsteps":
        notes = result[line_start + STEPS:line_end]
        notes[1::2][notes[::2] != 2] = 0
    return result


def pace_phrase(current, proposed, position):
    """Keep the kit and large line edits at phrase boundaries, leaving small variations audible."""
    if not STEPS or not SYN:
        return proposed
    next_bar = proposed.copy()
    boundary = position % 8 == 0
    beat_start = L + MODS
    drums = slice(beat_start, beat_start + STEPS // len(DRUM_ROWS))
    current_sounds, proposed_sounds = sounds(current), sounds(proposed)
    before, after = current_sounds > 0, proposed_sounds > 0
    kick_before, kick_after = decode_beat(current)[0] > 0, decode_beat(proposed)[0] > 0
    if not boundary and before[0] and kick_before.any():
        if current_sounds[0] != proposed_sounds[0]:
            set_sound(next_bar, 0, int(current_sounds[0]))
        if not after[0] or (kick_before != kick_after).sum() > (kick_before | kick_after).sum() / 2:
            next_bar[drums] = current[drums]

    notes_before = decode_synth(current) * before[1:, None]
    notes_after = decode_synth(proposed) * after[1:, None]
    changed = (notes_before != notes_after).sum(axis=1)
    occupied = ((notes_before != 0) | (notes_after != 0)).sum(axis=1)
    large = np.flatnonzero(2 * changed > occupied)
    if boundary and len(large):
        # Rotate which voice can make the bigger edit when several request one.
        allowed = large[(position // 8) % len(large)]
        large = large[large != allowed]
    width = SYN // len(SYNTH_LAYERS)
    for layer in large:
        start = beat_start + STEPS + layer * width
        next_bar[start:start + width] = current[start:start + width]
        set_sound(next_bar, layer + 1, int(current_sounds[layer + 1]))
    return next_bar


def calibrate_slots(brain, tracks, base: float, style=REFERENCE_STYLE, runs=3, bars=128, rounds=10, seed=23):
    """One temperature PER SLOT, so that every voice changes as often on the brain's own path as it does in the
    tracks. One temperature for all (fitted to the average of the four sounds) let the drum kit change three times
    too often while lead and air never changed: an endless loop instead of an evolving stream (11-23). Per round:
    free runs from passages in full flow, the change rate of every slot, and each slot's temperature moved toward
    its rate in the tracks (warmer = changes more often). Returns {"temperatures", "rates", "targets"}."""
    target = slot_rates(tracks, style)
    temps = np.full(SLOTS, float(base))
    starts = flow_passages(tracks, runs, 8, seed, style)
    floor = 0.5 / (runs * bars)
    rate = np.zeros(SLOTS)
    for r in range(rounds + 1):
        changes = np.zeros(SLOTS); n = 0
        for j, (tr, s) in enumerate(starts):
            rng = np.random.default_rng(seed + 31 * j + r)
            brain.reset()
            b, m = tr["bars"], tr["mood"]
            for t in range(s, s + 7):
                answer(brain, observe(b[t:t + 1], m[t + 1:t + 2]))
            cur = np.asarray(b[s + 7]).copy(); pos = s + 8
            for _ in range(bars):
                answer(brain, observe(cur[None], mood_rows([[0.5, 0.5]], tr.get("style"), bars=[pos])))
                new = sample_slots(motor_activity(brain)[0], cur, temps, rng)
                changes += new != cur; n += 1; cur = new; pos += 1
        rate = changes / max(n, 1)
        if r == rounds:
            break
        step = np.sqrt(np.maximum(target, floor) / np.maximum(rate, floor))
        temps = np.clip(temps * np.clip(step, 0.6, 1.7), 0.01, 3.0)
    brain.reset()
    return {"temperatures": [round(float(t), 4) for t in temps], "rates": [round(float(x), 4) for x in rate],
            "targets": [round(float(x), 4) for x in target], "base": float(base), "style": style}


SLOT_CALIBRATION = False   # 11-23: the fit does not converge on early checkpoints (a slot is frozen or flips every few
#                            bars, little in between), so every reader (player, QA) uses the card's one temperature


def slot_temperatures(brain, tracks, card, cache=None, style=REFERENCE_STYLE):
    """The per-slot temperatures of a checkpoint: from its card, from `cache` (a .slots.json beside it), or fitted
    now (and written to `cache`)."""
    if not SLOT_CALIBRATION:
        return np.full(SLOTS, float(card.get("temperature") or 0.07))
    if card.get("slot_temperatures"):
        return np.asarray(card["slot_temperatures"], dtype=float)
    if cache is not None and Path(cache).exists():
        try:
            saved = json.loads(Path(cache).read_text())
            if len(saved.get("temperatures", [])) == SLOTS:
                return np.asarray(saved["temperatures"], dtype=float)
        except ValueError:
            pass
    fit = calibrate_slots(brain, tracks, card.get("temperature") or 0.07, style)
    if cache is not None:
        Path(cache).write_text(json.dumps(fit))
    return np.asarray(fit["temperatures"], dtype=float)


def load_streams(path: str | Path, steps: bool = False, rows=None, clock=False, synth=None, patch_book=None,
                 synth_lesson_groups=None, sid_book=None):
    """rows: a row vocabulary {voice: rows}, or True to build it from the training tracks. sid_book
    (timbre.sid_book of the SID pool): the SID blocks of the "sid" shape carry the SID fit of every bar's
    sound, the chip's switch on; without it they are "off" in every bar."""
    data = json.loads(Path(path).read_text())
    styles = sorted({tr["style"] for tr in data["tracks"] if tr.get("style")})
    if rows is True:
        rows = row_vocab(data)
    if synth is True:
        vocabulary = data if synth_lesson_groups is None else {**data, "tracks": [
            t for t in data["tracks"] if t.get("group") in synth_lesson_groups
        ]}
        synth = synth_vocab(vocabulary)
    configure(data["tokens_per_layer"], data.get("mod_levels", 7), steps=steps, styles=styles if len(styles) > 1 else None, rows=rows, clock=clock, synth=synth, patch_book=patch_book)
    # Repeated drum bars dominate the corpus. Quantize each distinct row once.
    from functools import lru_cache
    @lru_cache(maxsize=65536)
    def drum_id(voice, row):
        return _nearest_row(row, ROWS[voice])
    mid = (M - 1) // 2
    tracks = []
    for tr in data["tracks"]:
        bars = np.asarray(tr["bars"], dtype=int)
        mods = np.asarray(tr["mods"], dtype=int) + mid if tr.get("mods") else np.full((len(bars), MODS), mid)
        offset = int(CLOCK_OFFSETS.get(tr["id"], 0))          # the score's phrase grid against the bar count
        mood = mood_rows(tr["mood"], tr.get("style"), bars=np.arange(len(bars)) + offset if offset else None)
        assert bars.shape[1] == L and mods.shape == (len(bars), MODS) and len(mood) == len(bars)
        if EAR_LIVE:
            if not RESERVE or "levels" not in tr or "bar_db" not in tr:
                raise ValueError("The ear needs a reserved shape and a score with measured levels")
            # A mood row travels with the bar before it (observe(bars[t], mood[t + 1])), so what the ear
            # heard of bar t goes into row t + 1.
            heard = ear_track(tr["levels"], tr["bar_db"])
            mood = np.hstack([mood, np.vstack([np.zeros((1, EAR_INPUTS)), heard[:-1]])])
        cols = [bars, np.clip(mods, 0, M - 1)]
        if ROWS:
            dr = np.asarray(tr["drums"], dtype=int).reshape(len(bars), 3, 16)
            cols.append(np.array([[drum_id(v, tuple(dr[b, i])) for i, v in enumerate(DRUM_ROWS)] for b in range(len(bars))]))
        elif STEPS:
            drums = np.asarray(tr["drums"], dtype=int).reshape(len(bars), STEPS)
            cols.append(np.clip(drums, 0, 2))
        if SYNTH and SYNTH.get("kind") == "steps":
            cols.append(np.asarray(tr["synth"], dtype=int).reshape(len(bars), SYN))
        elif SYNTH and SYNTH.get("kind") == "splitsteps":
            notes = np.asarray(tr["synth"], dtype=int).reshape(len(bars), -1)
            cols.append(encode_split_notes(notes))
        elif SYNTH and SYNTH.get("kind") == "factored":
            sy = np.asarray(tr["synth"], dtype=int).reshape(len(bars), 3, 16)
            cols.append(np.array([[x for v in range(len(SYNTH_LAYERS)) for x in encode_factored(sy[b, v], v)] for b in range(len(bars))]))
        elif SYNTH:
            sy = np.asarray(tr["synth"], dtype=int).reshape(len(bars), 3, 16)
            cols.append(np.array([[_nearest_note_row(sy[b, i], SYNTH[v]) for i, v in enumerate(SYNTH_LAYERS)] for b in range(len(bars))]))
        if patch_book:
            octaves = np.asarray(tr["synth_oct"], dtype=int) if "synth_oct" in tr else None
            cols.append(timbre.track_controls(bars, octaves, patch_book))
            cols[0] = bars.copy()
            cols[0][:, 1:] = bars[:, 1:] > 0
        allb = np.concatenate(cols, axis=1)
        if SYNTH and SYNTH_FILL:
            per = SYN // len(SYNTH_LAYERS)
            base = L + MODS + STEPS
            for v in range(len(SYNTH_LAYERS)):
                sl = slice(base + per * v, base + per * (v + 1))
                on = allb[:, 1 + v] > 0                                   # the layer's sound plays
                if not on.any():
                    continue
                last = (np.zeros(per, dtype=int) if patch_book else allb[np.argmax(on), sl].copy())                     # before the first sounding bar: that bar's line
                for b in range(len(allb)):
                    if on[b]:
                        last = allb[b, sl].copy()
                    else:
                        allb[b, sl] = last
        if ONOFF:                                             # sound slots keep the last sound; on/off slots at the end
            on = (allb[:, :L] > 0).astype(int)
            for l in range(L):
                col = allb[:, l]
                if on[:, l].any():
                    last = col[np.argmax(on[:, l])]               # before the first sounding bar: that bar's sound
                    for b in range(len(col)):
                        if col[b] > 0:
                            last = col[b]
                        else:
                            col[b] = last
            allb = np.concatenate([allb, on], axis=1)
        if chips():                                           # the SID blocks: the fit of the bar's sound, or "off"
            blocks = (timbre.sid_controls(bars, sid_book, chip_layers()) if sid_book
                      else np.zeros((len(allb), chip_end() - LIVE), dtype=int))
            allb = np.concatenate([allb, blocks], axis=1)
        if PLAN:                                              # the plan: the score's summary of the bars ahead
            allb = np.concatenate([allb, plan_rows(allb)], axis=1)
        if SECTIONS:                                          # the section of every bar (score v5f), else normal
            allb = np.concatenate([allb, section_rows(tr.get("section"), len(allb))[:, None]], axis=1)
        tracks.append(dict(id=tr["id"], title=tr.get("title", tr["id"]), held_out=bool(tr.get("held_out")), style=tr.get("style"),
                           bars=allb, mood=mood, group=tr.get("group"),
                           window_share=data.get("window_shares", {}).get(tr.get("group")), **sure_marks(tr, len(allb))))
    return tracks


SURE = ("drums_sure", "synth_sure")      # per bar: which drum rows (kick, snare, hats) and synths the reader was sure of
OPEN = "drums_open"                      # per bar and drum step (score v4): the reader could not decide this step
LINE_OPEN = "synth_open"                 # per bar and line step (bass, lead, air x 16, as "synth"): the same for the lines
CONTROLS_OPEN = "controls_open"          # per bar and synth layer (score v5f): the layer's sound controls were not read
MARKS = SURE + (OPEN, LINE_OPEN, CONTROLS_OPEN)


def sure_marks(track, bars):
    """The score's marks of a track, one row per bar, where the score carries them: the sure marks per drum row
    and synth, and from score v4 the open drum steps, in the order of the drum-step slots."""
    marks = {}
    for key in SURE:
        value = np.asarray(track.get(key) if track.get(key) is not None else [], dtype=bool)
        if value.shape == (bars, 3):
            marks[key] = value
    if track.get(OPEN) is not None and STEPS and not ROWS:
        value = np.asarray(track[OPEN], dtype=bool).reshape(bars, -1)
        if value.shape == (bars, STEPS):
            marks[OPEN] = value
    if track.get(LINE_OPEN) is not None and STEPS and not ROWS and SYNTH and SYNTH.get("kind") == "steps":
        value = np.asarray(track[LINE_OPEN], dtype=bool).reshape(bars, -1)
        if value.shape == (bars, SYN):
            marks[LINE_OPEN] = value
    if track.get(CONTROLS_OPEN) is not None:
        value = np.asarray(track[CONTROLS_OPEN], dtype=bool).reshape(bars, -1)
        if value.shape[1] == 1:                           # one flag per bar: every synth layer
            value = np.repeat(value, len(SYNTH_LAYERS), axis=1)
        if value.shape == (bars, len(SYNTH_LAYERS)):
            marks[CONTROLS_OPEN] = value
    return marks


def control_slots(layer):
    """The slots an open sound reading (CONTROLS_OPEN) of a synth layer covers: its filter and space levels (the mods)
    and its block of sound controls but the voice and the register (the line's octave), i.e. what the matched sound
    and the level rule write."""
    start = L + MODS + STEPS + SYN
    width = (LIVE - start) // len(SYNTH_LAYERS)
    mods = [L + 2 * (layer + 1), L + 2 * (layer + 1) + 1]
    block = [start + layer * width + k for k in range(2, width)]
    return np.asarray(mods + block, dtype=int)


def passage(track, start, bars):
    """A track cut to ``bars`` bars from ``start``: its bars, its mood rows and its marks."""
    cut = slice(start, start + bars)
    return dict(track, bars=track["bars"][cut], mood=track["mood"][cut],
                **{key: track[key][cut] for key in MARKS if key in track})


def open_steps(target, heard, opened, leave):
    """The lesson's labels with the open drum steps taken out of the lesson (score v4: a step the reader
    could not decide is no lesson). ``opened`` marks the open steps of the taught bar, one row per lesson row:
    STEPS drum steps, or STEPS drum steps and then SYN line steps (open_marks), which follow them in the state,
    and then, where the score marks open sound readings (CONTROLS_OPEN), one flag per synth layer: its
    control_slots are taken out the same way.

    A brain taught with open labels gets no label there (``leave``: -1, as audible_targets leaves a silent
    part). A brain with the direct route must be given a label for every slot (Cadence refuses an open one
    beside recall_tokens), so there the label is the step as heard: the route already answers it, the lesson
    moves nothing toward the undecided reading, and the heard bar still shows the row as read."""
    opened = np.asarray(opened, dtype=bool)
    if not opened.any():
        return target
    target = np.array(target, dtype=int, copy=True)
    layers = len(SYNTH_LAYERS) if opened.shape[-1] == STEPS + SYN + len(SYNTH_LAYERS) else 0
    first = L + MODS
    width = opened.shape[-1] - layers
    steps = target[:, first:first + width]
    steps[opened[:, :width]] = -1 if leave else np.asarray(heard, dtype=int)[:, first:first + width][opened[:, :width]]
    for v in range(layers):
        rows = np.flatnonzero(opened[:, width + v])
        if len(rows):
            slots = control_slots(v)
            target[np.ix_(rows, slots)] = -1 if leave else np.asarray(heard, dtype=int)[np.ix_(rows, slots)]
    return target


def open_marks(track, lines=None, controls=None):
    """A track's open steps per bar for its lesson windows: the drum steps (OPEN), and when the score marks open
    line steps (LINE_OPEN) the drum steps and then the line steps, and when it marks open sound readings
    (CONTROLS_OPEN) the drum steps, the line steps and one flag per synth layer; None when it marks none.
    ``lines`` / ``controls`` True give the track that width also where it marks nothing there (all decided), so
    the windows of a set of tracks stack."""
    drums, opened, unread = track.get(OPEN), track.get(LINE_OPEN), track.get(CONTROLS_OPEN)
    lines = bool(lines) or opened is not None
    controls = bool(controls) or unread is not None
    if drums is None and opened is None and unread is None and not lines and not controls:
        return None
    n = len(track["bars"])
    width = STEPS + (SYN if lines or controls else 0)
    marks = np.zeros((n, width + (len(SYNTH_LAYERS) if controls else 0)), dtype=bool)
    if drums is not None:
        marks[:, :STEPS] = drums
    if opened is not None:
        marks[:, STEPS:STEPS + SYN] = opened
    if unread is not None:
        marks[:, width:] = unread
    return marks


def section_rows(section, bars):
    """The section slot's labels of a track: per bar its class (SECTION_TAUGHT: a build is taught as normal), or
    normal everywhere when the score carries no section."""
    if section is None:
        return np.zeros(bars, dtype=int)
    if len(section) != bars:
        raise ValueError("the score's section must have one entry per bar")
    return np.array([SECTION_TAUGHT[s] if isinstance(s, str) else int(s) for s in section], dtype=int)


def plan_rows(bars):
    """The plan slots of every bar of a track, (bars, PLAN): the summary of the PLAN_BARS bars AFTER each bar.
    The plan of bar b describes bars b + 1 .. b + PLAN_BARS, so a brain writing bar t + 1 writes the plan for bars
    t + 2 .. t + 9 and hears, with bar t, the plan for t + 1 .. t + 8; the same in lessons (the score's) and in
    play (its own). Per voice (beat, bass, lead, air): it sounds in none (0), some (1) or all (2) of those bars;
    then how many of them change which voices sound (0, 1, 2 or more); then the first of them without an audible
    kick (0 none, 1 within four bars, 2 in bars five to eight). Near a track's end the summary reads the bars that
    remain; the last bar keeps the plan of the bar before."""
    bars = np.asarray(bars)
    n = len(bars)
    sounding = sounds(bars) > 0
    kick = audible_kick(bars)
    out = np.zeros((n, len(PLAN_SIZES)), dtype=int)
    for b in range(n):
        ahead = slice(b + 1, min(n, b + 1 + PLAN_BARS))
        if ahead.start >= ahead.stop:
            out[b] = out[b - 1] if b else 0
            continue
        s = sounding[ahead]
        played = s.sum(axis=0)
        out[b, :L] = np.where(played == 0, 0, np.where(played == len(s), 2, 1))
        out[b, L] = min(int((s != sounding[b:ahead.stop - 1]).any(axis=1).sum()), 2)
        quiet = np.flatnonzero(~kick[ahead])
        out[b, L + 1] = 0 if not len(quiet) else 1 if quiet[0] < 4 else 2
    return out


def windows(tracks, length: int, stride: int):
    """Lesson windows (bars, mood); a track with open steps (score v4: drum steps; line steps where the score marks
    them) gives (bars, mood, open steps)."""
    out = []
    lines = any(LINE_OPEN in tr for tr in tracks)            # one width for every window of the set
    controls = any(CONTROLS_OPEN in tr for tr in tracks)
    for tr in tracks:
        n = len(tr["bars"])
        marks = open_marks(tr, lines, controls)
        for s in range(0, max(n - length, 0) + 1, stride):
            if s + length <= n:
                window = (tr["bars"][s:s + length], tr["mood"][s:s + length])
                out.append(window + (marks[s:s + length],) if marks is not None else window)
    return out


def stream_stats(bars: np.ndarray, offset: int = 0) -> dict:
    """Statistics of one state stream (bars, 12) used for the corpus reference and for free runs."""
    identities = sound_identities(bars)
    bars = external(bars)
    n = len(bars)
    change = bars[1:] != bars[:-1]
    change[:, :L] = identities[1:] != identities[:-1]
    stats = {"bars": n}
    any_change = change[:, :L].any(axis=1) if n > 1 else np.zeros(0, bool)
    run, longest = 0, 0
    for c in any_change:
        run = 0 if c else run + 1
        longest = max(longest, run)
    stats["longest_still"] = int(longest)
    idx = np.flatnonzero(any_change) + 1 + offset                 # bar indices where some sound changes
    stats["aligned8"] = float((idx % 8 == 0).mean()) if len(idx) else 0.0
    stats["changes"] = int(len(idx))
    stats["change_any"] = float(any_change.mean()) if n > 1 else 0.0
    stats["mod_change"] = float(change[:, L:L + MODS].mean()) if n > 1 else 0.0
    if STEPS and n:
        d = np.stack([decode_beat(b) for b in bars]) * (bars[:, 0, None, None] > 0)
        beat_on = d.sum(axis=(1, 2)) > 0
        stats["drums"] = {
            "four_floor": float(((d[:, 0, [0, 4, 8, 12]] > 0).all(axis=1))[beat_on].mean()) if beat_on.any() else 0.0,
            "snare_2_4": float(((d[:, 1, [4, 12]] > 0).all(axis=1))[beat_on].mean()) if beat_on.any() else 0.0,
            "kicks_per_bar": float((d[:, 0] > 0).sum(axis=1)[beat_on].mean()) if beat_on.any() else 0.0,
            "hat_density": float((d[:, 2] > 0).mean(axis=1)[beat_on].mean()) if beat_on.any() else 0.0,
            "beat_change": float((d[1:] != d[:-1]).any(axis=(1, 2)).mean()) if n > 1 else 0.0,
        }
    for layer, name in enumerate(LAYERS):
        col = identities[:, layer]
        last_seen, returns = {}, 0
        for i, tok in enumerate(col):
            if tok and tok in last_seen and i - last_seen[tok] > 8:
                returns += 1
            if tok:
                last_seen[tok] = i
        runs, cur_len = [], 1
        for i in range(1, n):
            if col[i] == col[i - 1]:
                cur_len += 1
            else:
                runs.append(cur_len); cur_len = 1
        runs.append(cur_len)
        stats[name] = {
            "run_median": float(np.median(runs)) if runs else 0.0,
            "run_mean": float(np.mean(runs)) if runs else 0.0,
            "change": float(change[:, layer].mean()) if n > 1 else 0.0,
            "silence": float((col == 0).mean()),
            "distinct": int(len(set(col[col > 0].tolist()))),
            "returns_per_100": float(100.0 * returns / max(n, 1)),
            "mod_change": float(change[:, L + 2 * layer:L + 2 * layer + 2].mean()) if n > 1 else 0.0,
        }
    return stats


def corpus_reference(tracks) -> dict:
    """Pooled statistics over the training tracks (the gate's reference)."""
    train = [t for t in tracks if not t["held_out"]] or tracks
    per = [stream_stats(t["bars"]) for t in train]
    w = np.array([p["bars"] for p in per], dtype=float)
    row_weights = np.ones(len(train))
    shares = {t.get("group"): t.get("window_share") for t in train if t.get("window_share")}
    if shares:
        for group, share in shares.items():
            mask = np.array([t.get("group") == group for t in train])
            row_weights[mask] = share * w.sum() / w[mask].sum()
    w = w * row_weights
    w = w / w.sum()
    ref = {"change_any": float(sum(wi * p["change_any"] for wi, p in zip(w, per))),
           **({"drums": {k: float(sum(wi * p["drums"][k] for wi, p in zip(w, per))) for k in per[0]["drums"]}} if STEPS else {}),
           "mod_change": float(sum(wi * p["mod_change"] for wi, p in zip(w, per))),
           "aligned8": float(sum(r*p["aligned8"]*p["changes"] for r,p in zip(row_weights,per)) / max(sum(r*p["changes"] for r,p in zip(row_weights,per)), 1)),
           "longest_still": int(max(p["longest_still"] for p in per))}
    for name in LAYERS:
        ref[name] = {k: float(sum(wi * p[name][k] for wi, p in zip(w, per))) for k in ("change", "silence", "returns_per_100", "mod_change", "run_mean")}
    sums = np.zeros((L, T, MOOD))
    counts = np.zeros((L, T))
    for weight, t in zip(row_weights, train):
        snd = sounds(t["bars"])
        for layer in range(L):
            np.add.at(sums[layer], snd[:, layer], weight*t["mood"])
            np.add.at(counts[layer], snd[:, layer], weight)
    ref["token_mood"] = (sums / np.where(counts>0, counts, 1)[..., None]).tolist()
    ref["token_count"] = counts.tolist()
    return ref


def read_mood(bars: np.ndarray, token_mood) -> np.ndarray:
    tm = np.asarray(token_mood)
    bars = sounds(bars)
    return np.mean([tm[layer][bars[:, layer]] for layer in range(L)], axis=0)
