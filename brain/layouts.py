"""The slate of brain layouts under comparison (generation 2, Cadence 0.104.0).

Every layout is an apical brain of pathways under one coordinator. The coordinator writes the arrangement and reads
the four attention heads of the recall memory; there is no route from memory to the motor slots. Every pathway has a
plan and a detail region, and the details hear the shared learned code. A lesson teaches the next bar before the act
records the moment, and only the motor slots the score has read. The layouts differ in how many pathways they have and
how wide they are. A layout is trained only after it passed its checks (``check.py``); Stats tracks the checked ones.

    python layouts.py                 list the slate
    python layouts.py NAME            the preset and the settings of one layout, as JSON
"""
import json
import sys

import hierarchy

# The value path's gain on the first head's recall port at birth (common.born_gains). At the composer's size
# lessons alone raise it from zero too slowly. The value is set on the bench: at 1.0 the path is so strong that
# the pathways do not learn a change on top of holding, at 0.25 they do and the brain still holds. At full
# size 0.25 reads 38% at birth, where the commonest bar gives 55% and repeating 23%.
BORN_GAIN = 0.25

LAYOUTS = {
    # Generation 2 (s2): no direct route; the coordinator writes the arrangement and reads the four recall heads.
    # The detail regions carry the heard bar from the learned code to the motor (holding is what a brain without
    # the route gets wrong): two and a half times as wide; the plans one and a half times; the coordinator as it was.
    "split": dict(
        preset=hierarchy.DUO_PRESET, pathways=2, title="Rhythm and melody",
        over={"modules": [280, 192, 240, 192, 64]},
        wiring="A drum pathway and a synth pathway, each hearing the learned code, under one coordinator that writes "
               "the arrangement and reads the four recall heads."),
    "inst_reads": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments",
        over={"modules": [240, 96] * 4 + [64]},
        wiring="One pathway per instrument, each hearing the learned code, under one coordinator that writes the "
               "arrangement and reads the four recall heads."),
    # P4: the instruments layout with the sparse reflex, one learned block per pathway slot from its heard values to
    # its motor values (16,331 synapses); needs the library patch quality/library/sensory-reflex-0.104.0.patch, which
    # admits one-way sensory projections beside a learned code (CADENCE_PATCH on the box).
    "inst_reads_sref": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with a reflex",
        over={"modules": [240, 96] * 4 + [64], "reflex": True},
        wiring="The instruments layout, and learned synapses from every pathway slot as heard to the same motor slot."),
    # P5: the sparse reflex born on the diagonal: heard value v of a slot drives motor value v at reflex_gain from
    # birth (generation 1 copied through the memory route from birth); the rest of each block starts at zero. The
    # coordinator's slots (the kit, which instruments sound) have it too: the kit is the slot staying bars get wrong most;
    # its block takes gain 4 (65 classes: at gain 2 it was wrong on 31% of staying bars at birth, at 4 on none).
    "inst_reads_sdia": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with an innate reflex",
        over={"modules": [240, 96] * 4 + [64], "reflex": True, "reflex_coordinator": True, "reflex_birth": "diagonal",
              "reflex_gain": 2.0, "reflex_kit_gain": 4.0},
        wiring="The instruments layout, and a learned copy from every pathway slot as heard to the same motor slot, born on."),
    # The echo brain: the bars 2, 4, 8 and 16 back heard as further slot blocks (common.lagged_observation), each with
    # its own diagonal reflex to the motor; the learned code (1,792) reads every block. Birth gains by measurement:
    # reflex_lag_gains, one per lag; reflex_lag_groups "drums_lines" bears the lags beyond the first on drum rows and
    # synth lines only (the other blocks stay in the graph at zero, plastic).
    "inst_reads_echo": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with an echo",
        over={"modules": [240, 96] * 4 + [64], "reflex": True, "reflex_coordinator": True, "reflex_birth": "diagonal",
              "reflex_gain": 2.0, "reflex_kit_gain": 4.0, "echo_lags": [1, 2, 4, 8, 16],
              "reflex_lag_gains": [2.0, 1.0, 1.0, 1.0, 1.0]},
        wiring="The instruments layout hearing the bars 1, 2, 4, 8 and 16 back, each with an innate copy to the motor."),
    # The echo brain in the drums-and-melody split.
    "split_echo": dict(
        preset=hierarchy.DUO_PRESET, pathways=2, title="Rhythm and melody with an echo",
        over={"modules": [280, 192, 240, 192, 64], "reflex": True, "reflex_coordinator": True, "reflex_birth": "diagonal",
              "reflex_gain": 2.0, "reflex_kit_gain": 4.0, "echo_lags": [1, 2, 4, 8, 16],
              "reflex_lag_gains": [2.0, 1.0, 1.0, 1.0, 1.0]},
        wiring="The rhythm and melody layout hearing the bars 1, 2, 4, 8 and 16 back, each with an innate copy to the motor."),
    # Two coordinator layers (hierarchy context_recall): the upper coordinator reads the four recall heads, drives the
    # phrase coordinator and writes the arrangement beside it; the phrase coordinator drives the pathways. The upper
    # coordinator has a return from the pathways' motor slots, born at a tenth of a projection's scale (upper_return):
    # the per-slot surprise leaves its arrangement slots open once they are right. The innate reflex and the echo as
    # above. Gate: the delayed-cue chamber (deep_control_probe, 27 Nov).
    "inst_reads_sdia2": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with an innate reflex, two coordinator layers",
        over={"modules": [240, 96] * 4 + [64], "reflex": True, "reflex_coordinator": True, "reflex_birth": "diagonal",
              "reflex_gain": 2.0, "reflex_kit_gain": 4.0, "coordinator_depth": 2, "context_recall": True,
              "upper_return": 0.1},
        wiring="The instruments layout with the innate reflex, under a phrase coordinator and an upper coordinator "
               "that reads the four recall heads."),
    "inst_reads_echo2": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with an echo, two coordinator layers",
        over={"modules": [240, 96] * 4 + [64], "reflex": True, "reflex_coordinator": True, "reflex_birth": "diagonal",
              "reflex_gain": 2.0, "reflex_kit_gain": 4.0, "echo_lags": [1, 2, 4, 8, 16],
              "reflex_lag_gains": [2.0, 1.0, 1.0, 1.0, 1.0], "reflex_lag_groups": "drums_lines",
              "coordinator_depth": 2, "context_recall": True, "upper_return": 0.1},
        wiring="The echo layout under a phrase coordinator and an upper coordinator that reads the four recall heads."),
    # The drums-and-melody split with the same innate reflex over all its slots.
    "split_sdia": dict(
        preset=hierarchy.DUO_PRESET, pathways=2, title="Rhythm and melody with an innate reflex",
        over={"modules": [280, 192, 240, 192, 64], "reflex": True, "reflex_coordinator": True,
              "reflex_birth": "diagonal", "reflex_gain": 2.0, "reflex_kit_gain": 4.0},
        wiring="The rhythm and melody layout, and a learned copy from every slot as heard to the same motor slot, born on."),
    # P3: the instruments layout with a code reflex, a short learned arc for holding (a hedge: the generation-2
    # brains get the staying slots wrong where the old route copied them). The coordinator cut and the reflex cut
    # read who does what.
    "inst_reads_cref": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with a code reflex",
        over={"modules": [240, 96] * 4 + [64], "code_reflex": True},
        wiring="The instruments layout, and learned synapses from the shared learned code to every pathway's motor slots."),
}

# The reflex arc (hierarchy ``reflex``: learned synapses from every pathway slot as heard to the same motor slot)
# composes as a genome, but Cadence 0.104.0 refuses it beside a learned code ("a learned code's sensory neurons
# project to the code region alone", generic.py); it joins the slate with a library that admits it.
REFLEX_LAYOUTS = {
    "inst_reads_reflex": dict(
        preset=hierarchy.QUAD_PRESET, pathways=4, title="Instruments with a reflex",
        over={"modules": [240, 96] * 4 + [64], "reflex": True},
        wiring="The instruments layout, and learned synapses from every pathway slot as heard to the same motor slot."),
}

# Generation 1 (s1, the direct route): out of the slate, still built by name for their old snapshots and probes.
S1_LAYOUTS = {
    "chain": dict(
        preset="composer", pathways=0, title="Shared chain",
        wiring="One chain of four regions writes every slot."),
    # The learner reaches one region under the motor and not four (learn_probe, 26 Nov): the shallow chain.
    "shallow": dict(
        preset="composer", pathways=0, title="One region",
        over={"modules": [512]},
        wiring="One region of 512 units under the motor writes every slot."),
    "split_route": dict(
        preset=hierarchy.ROUTE_PRESET, pathways=2, title="Rhythm and melody, direct route (s1)",
        wiring="A drum pathway and a synth pathway under one coordinator."),
    "trio": dict(
        preset=hierarchy.TRIO_PRESET, pathways=3, title="Beat, bass and melody",
        wiring="Three pathways. The coordinator writes which instruments sound."),
    "inst": dict(
        preset=hierarchy.INSTRUMENT_ROUTE_PRESET, pathways=4, title="Instruments",
        wiring="One pathway per instrument. The coordinator writes which instruments sound."),
    "inst_reads_route": dict(
        preset=hierarchy.INSTRUMENT_ROUTE_PRESET, pathways=4, title="Instruments, direct reads, direct route (s1)",
        over={"pathway_reads": True},
        wiring="One pathway per instrument, and every pathway hears the bar itself."),
    "inst_local": dict(
        preset=hierarchy.INSTRUMENT_ROUTE_PRESET, pathways=4, title="Instruments, direct reads and own recall",
        over={"pathway_reads": True, "pathway_recall": True},
        wiring="One pathway per instrument. Every pathway hears the bar itself and reads its own recall head."),
    "inst_deep": dict(
        preset=hierarchy.INSTRUMENT_ROUTE_PRESET, pathways=4, title="Instruments, two coordinator layers",
        over={"coordinator_depth": 2},
        wiring="One pathway per instrument under a coordinator of two layers."),
    "parts": dict(
        preset=hierarchy.PARTS_PRESET, pathways=7, title="Notes and sound apart",
        wiring="The beat, and for every synth one pathway for its notes and one for its sound."),
    # Stage 1 of PLAN.md: the three wirings carried on, each also with every region hierarchy.WIDE times as wide.
    "split_wide": dict(
        preset=hierarchy.WIDE_PRESETS[hierarchy.ROUTE_PRESET], pathways=2, title="Rhythm and melody, wide",
        wiring=f"A drum pathway and a synth pathway under one coordinator, every region {hierarchy.WIDE} times as wide."),
    "inst_reads_wide": dict(
        preset=hierarchy.WIDE_PRESETS[hierarchy.INSTRUMENT_ROUTE_PRESET], pathways=4, title="Instruments, direct reads, wide",
        over={"pathway_reads": True},
        wiring=f"One pathway per instrument, every pathway hears the bar itself, every region {hierarchy.WIDE} times as wide."),
    "trio_wide": dict(
        preset=hierarchy.WIDE_PRESETS[hierarchy.TRIO_PRESET], pathways=3, title="Beat, bass and melody, wide",
        wiring=f"Three pathways, the coordinator writes which instruments sound, every region {hierarchy.WIDE} times as wide."),
}

# The slate of stage 1 (generation 1): born with the reserved shape (common.RESERVE) and trained on the corrected score.
STAGE1 = ("split_route", "inst_reads_route", "trio", "split_wide", "inst_reads_wide", "trio_wide")


NO_ROUTE = (hierarchy.DUO_PRESET, hierarchy.QUAD_PRESET, hierarchy.PRESET, hierarchy.INSTRUMENT_PRESET)


def entry(name):
    """A layout's entry: the slate's, or a generation-1 layout's by name."""
    if name in LAYOUTS:
        return LAYOUTS[name]
    if name in S1_LAYOUTS:
        return S1_LAYOUTS[name]
    raise ValueError(f"unknown layout {name!r}: one of {', '.join(LAYOUTS)}")


def layout(name):
    """The preset and the compose settings a layout adds to it."""
    item = entry(name)
    born = {} if item["preset"] in NO_ROUTE else {"recall_gain_born": BORN_GAIN}     # no route, no born gain
    return item["preset"], {**born, **(item.get("over") or {})}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        preset, over = layout(sys.argv[1])
        print(json.dumps({"preset": preset, "over": over, **{k: entry(sys.argv[1])[k] for k in ("title", "pathways", "wiring")}}))
    else:
        for name, entry in LAYOUTS.items():
            print(f"{name:11s} {entry['pathways']} pathways  {entry['title']}: {entry['wiring']}")
