"""Replay the same held-out prompts with specific changes to a checkpoint's recall."""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np

import common
import composition
import groove
from recall_trace import RecallTrace


def hold_lines(activity, state, temperature, draws):
    """Double the odds of the playing patterns; leave the arrangement and patches alone."""
    activity = activity.copy()
    start = common.L + common.MODS
    active = common.sounds(state) > 0
    synth_layers = len(common.SYNTH_LAYERS)
    layer_widths = [common.STEPS] + [common.SYN // synth_layers] * synth_layers
    slots = np.flatnonzero(np.repeat(active, layer_widths)) + start
    line_temperature = temperature * (common.LINE_SCALE or 1.0)
    activity[common.OFFSETS[slots] + state[slots]] += line_temperature * np.log(2)
    return common.sample_slots(activity, state, temperature, draws)


def dwell_spec(value):
    try:
        bars = [int(part) for part in value.split(",")]
        if len(bars) == 1:
            bars = bars[0]
        groove.VoiceDwell(bars)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a minimum of two bars, or four comma-separated minima") from error
    return bars


def main():
    from cadence import Brain, __version__

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("streams", type=Path)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device")
    parser.add_argument("--prompt-protocol", type=int, choices=(1, 2, 3))
    parser.add_argument("--test-release", help="Require this Cadence version for a cross-release playback comparison")
    parser.add_argument("--bars", type=int, help="Continuation length to use for every mode")
    parser.add_argument("--sampling-seed", type=int, default=23)
    parser.add_argument("--dwell-bars", type=dwell_spec, default=4, help="Minimum bars between larger edits for beat, bass, lead and air")
    parser.add_argument("--only-case", choices=("groove", "synth entry", "kick return", "motif handoff"))
    default_modes = ("intact", "clear", "prompt_only", "no_motor_shortcut", "half_strength", "quarter_strength",
             "sharper_attention", "mute", "softer_attention", "hold_lines")
    parser.add_argument("--mode", action="append", choices=(*default_modes, "separate_lines", "phrase_pacing", "learned_groove", "voice_dwell"))
    parser.add_argument("--audio-pool", type=Path, help="Render every prompt with this checkpoint's instrument pool")
    parser.add_argument("--audio-out", type=Path)
    parser.add_argument("--trace", type=Path, help="Write the intact heads' focus, including individual reads")
    args = parser.parse_args()
    card = json.loads(args.checkpoint.with_suffix(".json").read_text())
    expected_version = args.test_release or card["cadence_version"]
    if __version__ != expected_version:
        parser.error(f"Expected Cadence {expected_version}, found {__version__}")
    if args.bars is not None and args.bars < 17:
        parser.error("--bars must be at least 17")
    if args.sampling_seed < 0:
        parser.error("--sampling-seed must be nonnegative")
    if hashlib.sha256(args.streams.read_bytes()).hexdigest() != card["streams_sha"]:
        parser.error("The stream file differs from the checkpoint's training corpus")
    if bool(args.audio_pool) != bool(args.audio_out):
        parser.error("--audio-pool and --audio-out are used together")
    if args.audio_pool:
        pool_bytes = args.audio_pool.read_bytes()
        if hashlib.sha256(pool_bytes).hexdigest()[:12] != card.get("pool_sha"):
            parser.error("The pool differs from the checkpoint's instrument pool")
        pool = json.loads(pool_bytes)
    common.SYNTH_FILL = card["synth_fill"]
    common.shape_from(card)
    common.ENTRY_SCALE = card["entry_scale"]
    common.LINE_SCALE = card["line_scale"]
    tracks = common.load_streams(args.streams, steps=card.get("steps", False), rows=card["beat_rows"], synth=card["synth_rows"],
                                clock=card["clock"], patch_book=card["patch_book"])
    protocol = args.prompt_protocol or (card.get("composition") or {}).get("prompt_protocol", 1)
    bars = args.bars if args.bars is not None else card["training"]["composition_bars"]
    samplers = {"hold_lines": hold_lines, "separate_lines": partial(common.sample_slots, separate_lines=True)}
    for mode in args.mode or default_modes:
        def load():
            brain = Brain.load(args.checkpoint, backend=args.backend, device=args.device)
            if mode == "no_motor_shortcut":
                brain.recall.gains.fill(0)
            elif mode in ("half_strength", "quarter_strength"):
                brain.retune(recall_amplitude=brain.recall.amplitude / (2 if mode == "half_strength" else 4))
            elif mode == "sharper_attention":
                brain.retune(recall_temperature=brain.recall.temperature / 2)
            elif mode == "softer_attention":
                brain.retune(recall_temperature=brain.recall.temperature * 8)
            elif mode == "mute":
                brain.retune(recall_amplitude=0.0)
            return brain

        trace = RecallTrace(keep_reads=bool(args.trace)) if mode in ("intact", "softer_attention") else None
        audition = None
        if args.audio_out:
            directory = args.audio_out / mode
            directory.mkdir(parents=True, exist_ok=True)
            audition = composition.rendered_comparison(card, pool, directory)
        fitted = None
        if mode == "learned_groove":
            groups = card["training"].get("synth_lesson_group") or [common.REFERENCE_GROUP]
            fitted = groove.fit(tracks, groups)
        result = composition.compare(load, tracks, card["temperature"], bars,
                                     clear_recall=mode == "clear", prompt_only_recall=mode == "prompt_only",
                                     recall_trace=trace, sampler=samplers.get(mode, common.sample_slots),
                                     prompt_protocol=protocol, audition=audition, sampling_seed=args.sampling_seed,
                                     pace_edits=mode == "phrase_pacing", groove_fit=fitted,
                                     dwell_bars=args.dwell_bars if mode == "voice_dwell" else None,
                                     only_case=args.only_case)
        if trace is not None and args.trace:
            output = args.trace if mode == "intact" else args.trace.with_suffix(".softer.json")
            output.write_text(json.dumps({"streams_sha": card["streams_sha"], "checkpoint": card["name"],
                                          "mode": mode, "prompt_protocol": protocol,
                                          "max_slot_share": trace.max_slot_share,
                                          "heads": trace.summary(), "reads": trace.reads},
                                         allow_nan=False))
        print(json.dumps({"checkpoint": card["name"], "mode": mode, "cadence_version": __version__,
                          "prompt_protocol": protocol, "bars": bars, "sampling_seed": args.sampling_seed,
                          "dwell_bars": args.dwell_bars if mode == "voice_dwell" else None,
                          "groove_fit": None if fitted is None else {k: fitted[k] for k in ("version", "examples", "switches", "edits")},
                          "score": result["score"], "hold": result["hold_score"],
                          "selection_score": result["selection_score"],
                          "phrase_events": result.get("phrase_events"), "development": result.get("development"),
                          "max_slot_share": trace.max_slot_share if trace is not None else None,
                          "heads": trace.summary() if trace is not None else None,
                          "prompts": [{"track": run["track"], "prompt_bar": run["prompt_bar"],
                                       "case": run["case"], "score": run["score"], "hold": run["hold_score"],
                                       "reference": run["target"], "played": run["played"],
                                       "continuity": run["continuity"], "audio": run.get("audio")}
                                      for run in result["runs"]]}), flush=True)


if __name__ == "__main__":
    main()
