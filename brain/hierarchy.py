"""One apical Cadence brain with an isolated motor pathway for each musical part.

The layouts share one builder and differ in how many pathways they have. ``rhythm-melody-v1`` gives
the drums and the synth parts a pathway each. ``trio-v1`` has the beat, the bass and one melody
pathway. ``instruments-v1`` gives every instrument its own pathway. ``parts-v1`` splits every synth
into its notes and its sound. In the last three the coordinator writes the arrangement itself:
which instruments sound in the bar.
"""
import re
from dataclasses import replace

import numpy as np


PRESET = "composer-hierarchical"
ARCHITECTURE = "rhythm-melody-v1"
WIDTHS = (112, 128, 96, 128, 64)
BRANCHES = (("rhythm_detail", "rhythm_plan"), ("melody_detail", "melody_plan"))

# Generation 2 (s2): the rhythm and melody pathways under a coordinator that writes the arrangement itself.
DUO_ARCHITECTURE = "duo-v1"
DUO_PRESET = "composer-duo"
QUAD_PRESET = "composer-quad"

INSTRUMENT_PRESET = "composer-instruments"
INSTRUMENT_ARCHITECTURE = "instruments-v1"
INSTRUMENTS = ("beat", "bass", "lead", "air")
INSTRUMENT_WIDTHS = (96, 64) * len(INSTRUMENTS) + (64,)

# Three pathways: the beat, the bass, and one melody pathway for the lead and the air.
TRIO_ARCHITECTURE = "trio-v1"
TRIO_WIDTHS = (96, 64, 96, 64, 160, 96, 64)
# Seven pathways: the beat, and for every synth its notes and its sound apart.
PARTS_ARCHITECTURE = "parts-v1"
PARTS_WIDTHS = (96, 64) + (64, 48) * 6 + (64,)

# The layouts with the direct route from memory to the motor slots (recall_tokens with recall_gain).
ROUTE_PRESET = "composer-hierarchical-route"
INSTRUMENT_ROUTE_PRESET = "composer-instruments-route"
TRIO_PRESET = "composer-trio-route"
PARTS_PRESET = "composer-parts-route"
ROUTE_PRESETS = (ROUTE_PRESET, INSTRUMENT_ROUTE_PRESET, TRIO_PRESET, PARTS_PRESET)

# The same wirings with every region WIDE times as wide (stage 1 of PLAN.md: are the regions too narrow to
# hold the detail of a pattern?). The plan asks for four times, or the largest factor that lets two runs share
# one training GPU of 23 GB. Four times fits: a trainer that frees its evaluation copies (train.py, release)
# holds about 2.5 GB of the GPU at the present width and about 3.7 GB at four times the width after its first
# full evaluation (quality/stage1/memory.json). A lesson batch of the widest layout takes about 1.7 times as long.
WIDE = 4
WIDE_PRESETS = {ROUTE_PRESET: "composer-hierarchical-route-wide", INSTRUMENT_ROUTE_PRESET: "composer-instruments-route-wide",
                TRIO_PRESET: "composer-trio-route-wide"}


def wide(widths, factor=None):
    return tuple(int(round(width * (WIDE if factor is None else factor))) for width in widths)


PRESETS = {PRESET: ARCHITECTURE, INSTRUMENT_PRESET: INSTRUMENT_ARCHITECTURE,
           DUO_PRESET: DUO_ARCHITECTURE, QUAD_PRESET: INSTRUMENT_ARCHITECTURE,
           ROUTE_PRESET: ARCHITECTURE, INSTRUMENT_ROUTE_PRESET: INSTRUMENT_ARCHITECTURE,
           TRIO_PRESET: TRIO_ARCHITECTURE, PARTS_PRESET: PARTS_ARCHITECTURE,
           WIDE_PRESETS[ROUTE_PRESET]: ARCHITECTURE, WIDE_PRESETS[INSTRUMENT_ROUTE_PRESET]: INSTRUMENT_ARCHITECTURE,
           WIDE_PRESETS[TRIO_PRESET]: TRIO_ARCHITECTURE}
# The slots the coordinator writes itself carry this name in ``owners``.
COORDINATOR = "coordinator"
REFLEX_SCALE = 0.1        # the reflex synapses' birth magnitude, a tenth of a projection's
LAYERS, MIX = 4, 8


def owners(architecture, drum, line, patch, sid=0, plan=0):
    """The motor slots each pathway writes, in pathway order.

    The layout holds the four sound slots, two mix controls per layer, ``drum`` drum slots, ``line``
    line slots per synth layer and ``patch`` sound controls per synth layer. A brain of the reserved shape
    has ``sid`` more slots per synth layer after them, the synth's SID registers: they belong to the pathway
    that writes the synth's sound. ``sid`` may also give the count per synth (bass, lead, air): the lead-only
    shape has a block on the lead alone. ``plan`` slots come last (common.plan_rows): the coordinator writes them
    where the layout gives it motor slots, else the pathway that takes the remaining slots."""
    sids = tuple(sid) if isinstance(sid, (tuple, list)) else (sid,) * 3
    base = LAYERS + MIX
    lines = base + drum
    patches = lines + 3 * line
    chips = patches + 3 * patch
    total = chips + sum(sids)
    plans = list(range(total, total + plan))
    if architecture == ARCHITECTURE:
        rhythm = [0, LAYERS, LAYERS + 1, *range(base, lines)]
        taken = set(rhythm)
        return {"rhythm": rhythm, "melody": [slot for slot in range(total + plan) if slot not in taken]}

    def notes(i):      # the line of synth i (bass, lead, air)
        return list(range(lines + i * line, lines + (i + 1) * line))

    def sound(i):      # its two mix controls, its sound controls and its SID registers
        mix = LAYERS + 2 * (i + 1)
        return [mix, mix + 1, *range(patches + i * patch, patches + (i + 1) * patch),
                *range(chips + sum(sids[:i]), chips + sum(sids[:i + 1]))]

    # In these layouts the coordinator writes the arrangement itself: the four sound slots, and the plan.
    if architecture == DUO_ARCHITECTURE:
        rhythm = [LAYERS, LAYERS + 1, *range(base, lines)]
        taken = set(rhythm) | set(range(LAYERS)) | set(plans)
        return {COORDINATOR: list(range(LAYERS)) + plans, "rhythm": rhythm,
                "melody": [slot for slot in range(total) if slot not in taken]}
    parts = {COORDINATOR: list(range(LAYERS)) + plans, "beat": [LAYERS, LAYERS + 1, *range(base, lines)]}
    if architecture == INSTRUMENT_ARCHITECTURE:
        for i, name in enumerate(INSTRUMENTS[1:]):
            parts[name] = sound(i)[:2] + notes(i) + sound(i)[2:]
    elif architecture == TRIO_ARCHITECTURE:
        parts["bass"] = sound(0) + notes(0)
        parts["melody"] = sound(1) + notes(1) + sound(2) + notes(2)
    elif architecture == PARTS_ARCHITECTURE:
        for i, name in enumerate(INSTRUMENTS[1:]):
            parts[name] = notes(i)
            parts[f"{name}_sound"] = sound(i)
    else:
        raise ValueError(f"unknown pathway architecture {architecture!r}")
    return parts


def _head(name):
    """The recall head a read port belongs to: recall, next, recall_2, next_2_when ..."""
    fields = [field for field in name.split("_") if field not in ("recall", "next", "when", "action")]
    return int(fields[0]) if fields else 0


def genome(settings, parts, seed, label="cadance-hierarchy-v1"):
    """The genome of the pathway brain and what ``Brain.compose`` takes beside its connectome.

    ``parts`` maps each pathway to its motor slots; the entry named ``coordinator`` holds the slots
    the coordinator writes itself. Returns ``(genome, inputs, actions, slots, options)``."""
    from cadence import Genome, Projection
    from cadence.regions import Region

    options = dict(settings)
    widths = tuple(options.pop("modules"))
    depth = options.pop("coordinator_depth")
    reads = options.pop("pathway_reads", False)
    recall = options.pop("pathway_recall", False)
    # With two coordinator layers: the recall heads read into the upper one (the section context), which hears the
    # phrase coordinator and the memory; the phrase coordinator keeps the code and the trace and drives the pathways.
    # The upper coordinator then lies before the phrase coordinator and writes the arrangement beside it (see below).
    upper_recall = options.pop("context_recall", False)
    # With context_recall: the upper coordinator's return from the pathways' motor slots, a reciprocal projection born
    # at this share of a projection's scale (0: none). Under the per-slot surprise its own arrangement slots are left
    # open once they are answered as shown, and it had no other return in one hop (see below).
    upper_return = options.pop("upper_return", 0.0)
    # The reflex arc (generation 2, a hedge for holding): learned synapses, born small, from every pathway slot's
    # heard values (its one-hot columns of the observation) to the same slot's motor values. Not the memory: the bar
    # heard now, one block per slot, taught like any synapse.
    reflex = options.pop("reflex", False)
    # with reflex_coordinator the coordinator's own slots (the kit and which instruments sound) get the reflex too
    reflex_coordinator = options.pop("reflex_coordinator", False)
    # The echo brain: the observation holds the bars ECHO lags back as further slot blocks after its own columns
    # (common.lagged_observation); every lag gets the same per-slot reflex blocks to the motor.
    echo_lags = tuple(options.pop("echo_lags", ()) or ())
    echo_base = options.pop("echo_base", None)
    # The code reflex (admissible on Cadence 0.104.0, which keeps the sensory neurons for the learned code): learned
    # synapses, born small, from the shared learned code to every pathway's own motor slots, one-way.
    code_reflex = options.pop("code_reflex", False)
    pathways = [name for name in parts if name != COORDINATOR]
    if len(pathways) < 2:
        raise ValueError("The hierarchy needs at least two pathways")
    if len(widths) != 2 * len(pathways) + 1 or any(type(w) is not int or w < 2 for w in widths):
        raise ValueError("The hierarchy needs a detail and a plan width for each pathway, then the coordinator: "
                         + ", ".join(f"{name} detail/plan" for name in pathways) + ", coordinator")
    if type(depth) is not int or depth not in (1, 2):
        raise ValueError("The hierarchy needs one or two coordinator layers")
    if type(reads) is not bool or type(recall) is not bool or type(upper_recall) is not bool:
        raise ValueError("pathway_reads, pathway_recall and context_recall are switches")
    if type(reflex) is not bool or type(code_reflex) is not bool:
        raise ValueError("reflex and code_reflex are switches")
    if upper_recall and (depth != 2 or recall):
        raise ValueError("context_recall needs two coordinator layers and no pathway recall")
    if isinstance(upper_return, bool) or not isinstance(upper_return, (int, float)) or not 0 <= upper_return <= 1:
        raise ValueError("upper_return is a birth scale from 0 to 1")
    if upper_return and not upper_recall:
        raise ValueError("upper_return needs context_recall")
    if not options.get("learned_code") or not options.get("attention") or not options.get("apical"):
        raise ValueError("The hierarchy uses apical feedback and attention on a learned code")
    # The direct route (recall_tokens with recall_gain) reads a remembered bar straight into the motor slots.
    # It is what lets a brain hold what it heard: without it every run stayed at the level of writing the
    # commonest bar. The pathways learn the changes on top of it.
    if not options.get("recall", True) or options.get("recall_action"):
        raise ValueError("Recall enters through its read ports and the direct route, not through kept commands")
    if options.get("efference_amplitude", 0) or options.get("message") or options.get("longterm"):
        raise ValueError("Use the coordinator's recall and local motor feedback in this comparison")
    if options.get("memory_choice") or options.get("memory_amplitude", 0):
        raise ValueError("A direct episodic read into the motor layer would bypass the coordinator")

    inputs, actions, slots = options.pop("inputs"), options.pop("actions"), options.pop("slots")
    # Genome.composed builds the ports but doesn't apply Brain.compose's attention defaults.
    options.setdefault("recall", True)
    options.setdefault("recall_next", True)
    options.setdefault("memory_amplitude", 0.0)
    base = Genome.composed(inputs, actions, modules=(widths[-1],), slots=slots, seed=seed, **options)
    owner = np.full(len(slots), -1)
    for index, name in enumerate(parts):
        if (owner[parts[name]] >= 0).any():
            raise ValueError("A motor slot has two pathways")
        owner[parts[name]] = index
    if (owner < 0).any() or any(not len(parts[name]) for name in parts):
        raise ValueError("Every pathway needs motor slots, and every motor slot one pathway")
    neurons = np.repeat(owner, slots)
    motor = base.region("motor")
    motor = replace(motor, circuit=motor.circuit.with_populations(
        **{name: np.flatnonzero(neurons == index) for index, name in enumerate(parts)}))

    reflex_slots = ([i for name in pathways for i in parts[name]]
                    + (list(parts.get(COORDINATOR, ())) if reflex_coordinator else [])) if reflex else []
    if reflex:
        from cadence import Connectome
        edges = np.concatenate([[0], np.cumsum(slots)])
        blocks = {f"s{i}": list(range(int(edges[i]), int(edges[i + 1]))) for i in reflex_slots}
        motor = replace(motor, circuit=motor.circuit.with_populations(**blocks))
        heard = base.region("sensory")
        if heard.size < edges[-1]:
            raise ValueError("the reflex needs the heard bar in the first sensory columns, one per slot value")
        for j, k in enumerate(echo_lags[1:], start=1):
            first = int(echo_base) + (j - 1) * int(edges[-1])
            blocks.update({f"l{k}s{i}": list(range(first + int(edges[i]), first + int(edges[i + 1]))) for i in reflex_slots})
            if first + int(edges[-1]) > heard.size:
                raise ValueError("the echo lags need their slot blocks in the sensory columns after the observation")
        heard = Region("sensory", heard.size, circuit=Connectome(heard.size, [], [], [], [],
                       populations={k: v for k, v in blocks.items() if not k.startswith("s") or k[1:].isdigit()}))
    regions = []
    for region in base.regions:
        if reflex and region.name == "sensory":
            regions.append(heard)
            continue
        if region.name == "association":
            for index, name in enumerate(pathways):
                regions.extend((Region(f"{name}_detail", widths[2 * index]),
                                Region(f"{name}_plan", widths[2 * index + 1])))
            if depth == 2 and upper_recall:
                # The upper coordinator that holds the recall heads lies before the phrase coordinator. Cadence lands
                # the half of a reciprocal pair that runs from the later region to the earlier one apically, so here
                # the memory's way down (upper to phrase coordinator) is a basal drive and the way back is the apical
                # return. Laid out after it, the cue reached the phrase coordinator only as apical modulation.
                regions.append(Region("phrase_context", widths[-1]))
        regions.append(motor if region.name == "motor" else region)
        if region.name == "association" and depth == 2 and not upper_recall:
            regions.append(Region("phrase_context", widths[-1]))

    # All mixed inputs (sensory code, trace, and recall) terminate at the coordinator.
    # The lower pathways only exchange information via that coordinator.
    projections = []
    for edge in base.projections:
        if (edge.pre, edge.post) == ("association", "motor"):
            continue
        if recall and edge.post == "association" and edge.pre.startswith(("recall", "next")) \
                and _head(edge.pre) < len(pathways):
            # One recall head per pathway: the head reads into that pathway's plan.
            edge = replace(edge, post=f"{pathways[_head(edge.pre)]}_plan")
        elif upper_recall and edge.post == "association" and edge.pre.startswith(("recall", "next")):
            edge = replace(edge, post="phrase_context")
        projections.append(edge)
    for name in pathways:
        detail, plan = f"{name}_detail", f"{name}_plan"
        projections.extend((Projection(detail, plan), Projection(plan, "association"),
                            Projection(detail, f"motor/{name}"),
                            # A one-way basal drive lights local features. Apical feedback
                            # is normalized by basal activity and cannot start a quiet branch.
                            Projection("association", detail, reciprocal=False)))
        if reads:
            # The pathway also hears the bar itself, so holding a pattern is a local loop.
            projections.append(Projection("code", detail, reciprocal=False))
    if COORDINATOR in parts:
        # The coordinator's own output: which instruments sound.
        projections.append(Projection("association", f"motor/{COORDINATOR}"))
    if depth == 2:
        # The return projection is apical. It can shape both plans through association. (With context_recall the
        # regions lie the other way round: the upper coordinator drives, the phrase coordinator returns.)
        projections.append(Projection("association", "phrase_context"))
        if upper_recall and COORDINATOR in parts:
            # The upper coordinator needs its own return from a taught motor: it writes the arrangement beside the
            # phrase coordinator, and the motor's half of this pair is apical, one hop from the taught slots. With
            # the phrase coordinator as its only way back, eight memory ports set its units' potential and the
            # correction arrived two hops away: on seed 2 of the delayed-cue chamber 81 to 90% of its units sat
            # below rest from lesson 15 on (slope 0.06 to 0.09) and it took 0.02% of the motor's teaching contrast.
            projections.append(Projection("phrase_context", f"motor/{COORDINATOR}"))
    if upper_return:
        # A taught pathway slot returns to the upper coordinator apically in one hop, as its arrangement slots do.
        # Without it, under the per-slot surprise, the delayed-cue chamber's upper coordinator fell to 0.03 to 0.4%
        # of the motor's teaching contrast as soon as the arrangement was right (deep_control_probe, 27 Nov). The
        # forward half lets the upper coordinator write pathway slots itself: born small for that reason.
        projections.extend(Projection("phrase_context", f"motor/{name}", scale=float(upper_return)) for name in pathways)
    for i in reflex_slots:          # appended last: the brain without the reflex develops from the same draws
        projections.append(Projection(f"sensory/s{i}", f"motor/s{i}", reciprocal=False, scale=REFLEX_SCALE))
    for k in echo_lags[1:]:         # then the echo lags' blocks, lag by lag
        projections.extend(Projection(f"sensory/l{k}s{i}", f"motor/s{i}", reciprocal=False, scale=REFLEX_SCALE)
                           for i in reflex_slots)
    if code_reflex:                 # appended after everything else, for the same reason
        if not any(region.name == "code" for region in regions):
            raise ValueError("the code reflex needs a learned code")
        projections.extend(Projection("code", f"motor/{name}", reciprocal=False, scale=REFLEX_SCALE) for name in pathways)

    return Genome(tuple(regions), tuple(projections), f"{label}-{depth}"), inputs, actions, slots, options


def compose(settings, parts, seed, label="cadance-hierarchy-v1"):
    from cadence import Brain, develop

    settings = dict(settings)
    # The reflex's birth: "diagonal" sets, in every slot's block, the synapse from heard value v to motor value v at
    # reflex_gain and the rest of the block at zero (an innate copy that stays plastic); default: random and small.
    birth = settings.pop("reflex_birth", None)
    gain = float(settings.pop("reflex_gain", 1.0))
    # the kit slot (slot 0, 65 classes, each kit in about one recording) may take its own diagonal gain
    kit_gain = settings.pop("reflex_kit_gain", None)
    # the coordinator's own slots (the kit and which instruments sound) may take their own lag-1 diagonal gain
    coordinator_gain = settings.pop("reflex_coordinator_gain", None)
    # the echo lags' diagonal gains (one per lag in echo_lags, the first the bar heard now) and the slots whose
    # lags beyond the first are born on (the rest of those blocks start at zero, plastic)
    lag_gains = settings.pop("reflex_lag_gains", None)
    lag_slots = settings.pop("reflex_lag_slots", None)
    echo_lags, echo_base = tuple(settings.get("echo_lags", ()) or ()), settings.get("echo_base")
    if birth not in (None, "random", "diagonal") or (birth == "diagonal" and not settings.get("reflex")):
        raise ValueError("reflex_birth is random or diagonal, and needs reflex")
    built, inputs, actions, slots, options = genome(settings, parts, seed, label)
    connectome = develop(built, seed=seed)
    if birth == "diagonal":
        from cadence import Connectome
        heard, motor = (np.asarray(connectome.populations[name], dtype=np.int64) for name in ("sensory", "motor"))
        pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
        arc = np.isin(pre, heard) & np.isin(post, motor)
        width = int(np.sum(settings["slots"]))                      # the motor values: one slot block
        local, target = pre - heard.min(), post - motor.min()
        lag = np.where(local < width, 0, -1)                         # which lag block a heard column belongs to
        value = np.where(local < width, local, -1)
        if echo_lags:
            later = local >= int(echo_base)
            lag = np.where(later, 1 + (local - int(echo_base)) // width, lag)
            value = np.where(later, (local - int(echo_base)) % width, value)
        diagonal = arc & (value == target)                          # heard value v of slot s (at any lag) to motor v
        gains = list(lag_gains) if lag_gains is not None else [gain] + [0.0] * max(len(echo_lags) - 1, 0)
        sign = np.array(connectome.sign, dtype=float, copy=True)
        sign[arc] = 0.0
        sign[diagonal] = np.asarray(gains, dtype=float)[lag[diagonal]]
        edges = np.cumsum([0, *settings["slots"]])
        slot = np.searchsorted(edges, target, side="right") - 1
        if lag_slots is not None:                                    # lags beyond the first only on these slots
            sign[diagonal & (lag > 0) & ~np.isin(slot, list(lag_slots))] = 0.0
        if coordinator_gain is not None and COORDINATOR in parts:
            sign[diagonal & (lag == 0) & np.isin(slot, list(parts[COORDINATOR]))] = float(coordinator_gain)
        if kit_gain is not None:
            sign[diagonal & (lag == 0) & (slot == 0)] = float(kit_gain)
        connectome = Connectome(connectome.n, connectome.pre, connectome.post, connectome.count, sign,
                                connectome.populations, connectome.label)
    # The reflex's per-slot populations (sensory/sN, motor/sN) only place its synapses. Kept, they cut the block layout
    # into more than 256 blocks, the layout falls back to one dense block and the drive budget builds it at every
    # lesson; dropped after development, the synapses are the same bit for bit.
    wiring = re.compile(r"(sensory|motor)/(l\d+)?s\d+")
    if any(wiring.fullmatch(name) for name in connectome.populations):
        from cadence import Connectome
        kept = {name: members for name, members in connectome.populations.items() if not wiring.fullmatch(name)}
        connectome = Connectome(connectome.n, connectome.pre, connectome.post, connectome.count, connectome.sign,
                                kept, connectome.label)
    return Brain.compose(inputs, actions, connectome=connectome, slots=slots, seed=seed, **options)
