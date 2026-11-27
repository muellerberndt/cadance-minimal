"""Matched continuations of unseen tracks. Every value is a fraction on [0, 1]."""
import numpy as np

import common
import groove
import timbre


def audible_patterns(bars):
    gates = common.sounds(bars) > 0
    drums = np.stack([common.decode_beat(bar) for bar in bars]) * gates[:, :1, None]
    notes = np.stack([common.decode_synth(bar) for bar in bars]) * gates[:, 1:, None]
    return gates, drums, notes


def detail_profile(drums, notes):
    detail = {}
    for i, voice in enumerate(common.DRUM_ROWS):
        row = drums[:, i]
        active = row.any(axis=1)
        detail[voice] = {"accents": int((row == 2).sum()), "hits": int((row > 0).sum()),
                         "patterns": len(np.unique(row[active], axis=0)), "bars": int(active.sum())}
    for i, voice in enumerate(common.SYNTH_LAYERS):
        line = notes[:, i]
        active = (line >= 2).any(axis=1)
        detail[voice] = {"ties": int((line == 1).sum()), "onsets": int((line >= 2).sum()),
                         "pitch_classes": len(np.unique(line[line >= 2])),
                         "patterns": len(np.unique(line[active], axis=0)), "bars": int(active.sum())}
    return detail


def profile(bars, offset=0):
    gates, drums, notes = audible_patterns(bars)
    result = {
        "quarter_kicks": float((drums[:, 0, ::4] > 0).mean()),
        "busy_snare": float(((drums[:, 1] > 0).sum(axis=1) > 8).mean()),
        "beat_variation": float((drums[1:] != drums[:-1]).any(axis=(1, 2)).mean()),
    }
    bass, lead = notes[:, 0] >= 2, notes[:, 1] >= 2
    result["lead_on_bass_rest"] = float((lead & ~bass).sum() / max(1, lead.sum()))
    entering = (gates[1:] != gates[:-1]).any(axis=1)
    edits = (notes[1:] != notes[:-1]).any(axis=(1, 2)) | (drums[1:] != drums[:-1]).any(axis=(1, 2))
    downbeat = (np.arange(1, len(bars)) + offset) % 8 == 0
    result["entries_on_phrase"] = float((entering & downbeat).sum() / max(1, entering.sum()))
    result["motif_edits_on_phrase"] = float((edits & downbeat).sum() / max(1, edits.sum()))
    for i, name in enumerate(common.SYNTH_LAYERS):
        line = notes[:, i]
        active = (line != 0).any(axis=1)
        result[f"{name}_sounding"] = float(active.mean())
        result[f"{name}_onsets"] = float((line >= 2).mean())
        # A return counts only after a departure. A frozen loop scores zero.
        for lag in (1, 8, 16):
            pairs = active[lag:] & active[:-lag]
            same = (line[lag:] == line[:-lag]).all(axis=1)
            if lag > 1:
                departed = np.zeros(len(same), dtype=bool)
                for j in range(1, lag):
                    departed |= (line[j:j + len(same)] != line[:-lag]).any(axis=1)
                same &= departed
            result[f"{name}_repeat_{lag}"] = float((pairs & same).sum() / max(1, pairs.sum()))
        start = common.L + common.MODS + common.STEPS + common.SYN + i * timbre.width(common.PATCH_BOOK)
        # Only shared, audible controls contribute to timbre motion and only while sounding.
        if timbre.width(common.PATCH_BOOK) > len(timbre.PATCH_SIZES):
            velocity = bars[:, start + len(timbre.PATCH_SIZES):start + len(timbre.SIZES)]
            # Spread of played velocities only; ignore silent layers and unused phases.
            velocity = np.tile((velocity + 1) / 7, (1, 4))
            on = line >= 2
            result[f"{name}_velocity_spread"] = float(velocity[on].std()) if on.any() else 0.
        knobs = [timbre.NAMES.index(k) + 3 for k in ("wave", "cutoff", "reso", "decay", "drive")]
        patch = bars[:, start + np.array(knobs)]
        pairs = active[1:] & active[:-1]
        if common.PATCH_BOOK.get("samples"):
            playing = (line >= 2).any(axis=1) & (bars[:,start] == timbre.VOICES.index("sampler"))
            result[f"{name}_sample_share"] = float(playing.mean())
        result[f"{name}_timbre_change"] = float(((patch[1:] != patch[:-1]).any(axis=1) & pairs).sum() / max(1, pairs.sum()))
    return result


def phrase_events(bars, offset, prompt_bars):
    """Count musical events by their position in the eight-bar phrase."""
    sounds = common.sounds(bars) > 0
    kicks = np.array([common.decode_beat(bar)[0].any() for bar in bars]) & sounds[:, 0]
    lines = np.stack([common.decode_synth(bar) for bar in bars]) * sounds[:, 1:, None]
    active = lines.any(axis=2)
    positions = (offset + np.arange(prompt_bars, len(bars))) % 8

    def phased(events):
        return np.bincount(positions, weights=events[prompt_bars:], minlength=8).astype(int)

    def entrances(heard):
        events = np.zeros(len(bars), dtype=bool)
        events[2:] = heard[2:] & ~heard[1:-1] & ~heard[:-2]
        return phased(events)

    events = {"kick_return": entrances(kicks)}
    for layer, name in enumerate(common.SYNTH_LAYERS):
        line = lines[:, layer]
        heard = active[:, layer]
        returned = np.zeros(len(bars), dtype=bool)
        for lag in (8, 16):
            if lag < len(bars):
                same = (line[lag:] == line[:-lag]).all(axis=1)
                departed = (line[lag - 1:-1] != line[:-lag]).any(axis=1)
                returned[lag:] |= heard[lag:] & same & departed
        events[f"{name}_entrance"] = entrances(heard)
        events[f"{name}_motif_return"] = phased(returned)
    return events


def bar_continuity(bars):
    """Compare decoded, active lines between bars, excluding entrances and exits."""
    drums = np.stack([common.decode_beat(bar) for bar in bars]).reshape(len(bars), -1)
    notes = np.stack([common.decode_synth(bar) for bar in bars])
    notes = np.pad(notes, ((0, 0), (0, 0), (0, drums.shape[1] - notes.shape[2])))
    lines = np.concatenate([drums[:, None, :], notes], axis=1)
    lines *= (common.sounds(bars) > 0)[..., None]
    before, after = lines[:-1], lines[1:]
    carries = before.any(axis=2) & after.any(axis=2)
    changed = (before != after).sum(axis=2)
    occupied = ((before != 0) | (after != 0)).sum(axis=2)
    edit = changed / np.maximum(occupied, 1)
    rewrite = carries & (edit > .5)
    layers = {}
    for i, name in enumerate(common.LAYERS):
        present = carries[:, i]
        count = int(present.sum())
        layers[name] = {"adjacent_sounding": count,
                        "distinct_sounding_patterns": len(np.unique(lines[lines[:, i].any(axis=1), i], axis=0)),
                        "note_edit_fraction": float(edit[present, i].mean()) if count else None,
                        "rewrite_fraction": float(rewrite[present, i].mean()) if count else None}
    if common.PATCH_BOOK:
        start = common.L + common.MODS + common.STEPS + common.SYN
        controls = bars[:, start:common.LIVE].reshape(len(bars), len(common.SYNTH_LAYERS), timbre.width(common.PATCH_BOOK))
        mode = controls[:, :, 0]
        wave = controls[:, :, 3 + timbre.NAMES.index("wave")]
        if common.PATCH_BOOK.get("samples"):
            wave = np.where(mode == timbre.VOICES.index("sampler"), controls[:, :, len(timbre.SIZES)], wave)
        switched = (mode[1:] != mode[:-1]) | (wave[1:] != wave[:-1])
        for i, name in enumerate(common.SYNTH_LAYERS):
            present = carries[:, i + 1]
            layers[name]["instrument_switch"] = float(switched[present, i].mean()) if present.any() else None
    return {"multiple_lines_rewritten": float((rewrite.sum(axis=1) >= 2).mean()), "layers": layers}


def event_agreement(target, generated, repeated):
    """Phrase-position overlap for categories the reference actually demonstrates."""
    readings = {}
    for name, expected in target.items():
        actual, hold = generated[name], repeated[name]
        readings[name] = {"reference": expected.tolist(), "played": actual.tolist(), "hold": hold.tolist()}
    available = [name for name, events in target.items() if events.any()]

    def score(predicted):
        return float(np.mean([2 * np.minimum(target[name], predicted[name]).sum() /
                              (target[name].sum() + predicted[name].sum()) for name in available])) if available else None

    return {"score": score(generated), "hold": score(repeated), "events": readings}


def development_points(bars, offset, synth=True):
    """Read four-bar rises, releases, and the arrival of a new recurring line after both."""
    gates = common.sounds(bars) > 0
    drums = np.stack([common.decode_beat(bar) for bar in bars]) * gates[:, :1, None]
    notes = (np.stack([common.decode_synth(bar) for bar in bars]) * gates[:, 1:, None] if synth else
             np.zeros((len(bars), len(common.SYNTH_LAYERS), 16), dtype=int))
    density = .5 * drums.sum(axis=(1, 2)) + (notes >= 2).sum(axis=(1, 2))
    events = {name: np.zeros(len(bars), dtype=bool) for name in ("buildup", "release", "new_motif_after_release")}

    def resemblance(a, b):
        attacks = (a[:, None, :] >= 2) & (a[:, None, :] == b[None, :, :])
        total = (a >= 2).sum(axis=1)[:, None] + (b >= 2).sum(axis=1)[None, :]
        return 2 * attacks.sum(axis=2) / np.maximum(1, total)

    for boundary in range(4 + (-offset) % 4, len(bars) - 3, 4):
        change = density[boundary:boundary + 4].mean() - density[boundary - 4:boundary].mean()
        events["buildup"][boundary] = change >= 2
        events["release"][boundary] = change <= -2
    if not synth:
        return events
    for release in np.flatnonzero(events["release"]):
        if not events["buildup"][max(0, release - 8):release].any():
            continue
        outgoing = notes[max(0, release - 8):release]
        for handoff in (release + 4, release + 8):
            incoming = notes[handoff:handoff + 4]
            if len(incoming) < 4:
                break
            for i in range(len(common.SYNTH_LAYERS)):
                similarity = resemblance(incoming[:, i], incoming[:, i])
                np.fill_diagonal(similarity, 0)
                recurring = similarity.max(axis=1) >= .6
                novel = resemblance(incoming[:, i], outgoing[:, i]).max(axis=1) < .6
                if (recurring & novel).any():
                    events["new_motif_after_release"][handoff] = True
                    break
            if events["new_motif_after_release"][handoff]:
                break
    return events


def development_events(bars, offset, prompt_bars):
    points = development_points(bars, offset)
    return {name: np.bincount((np.flatnonzero(flags[prompt_bars:]) + prompt_bars + offset) % 16, minlength=16)
            for name, flags in points.items()}


def prompts(test, bars, rng, protocol=2):
    """Fixed musical situations chosen from labels before any candidate is scored."""
    options = []
    for i, tr in enumerate(test):
        seq = tr["bars"]
        length = min(bars, len(seq) - 8)
        gates = common.sounds(seq) > 0
        kicks = np.array([common.decode_beat(b)[0].any() for b in seq]) & gates[:, 0]
        melodies = np.stack([common.decode_synth(b) for b in seq]) * gates[:, 1:, None]
        drums = np.stack([common.decode_beat(b) for b in seq]) * gates[:, :1, None]
        identities = common.sound_identities(seq)
        changed = ((melodies[1:] != melodies[:-1]).any(axis=(1, 2)) |
                   (drums[1:] != drums[:-1]).any(axis=(1, 2)) |
                   (identities[1:] != identities[:-1]).any(axis=1))
        # Presence means a line actually plays, including ties. A gate alone can be silent.
        sounding = melodies.any(axis=2)
        handoffs = development_points(seq, tr.get("offset", 0))["new_motif_after_release"] if protocol == 3 else None
        for start in range(len(seq) - 8 - length + 1):
            pos = start + 8
            future = slice(pos, pos + min(16, length))
            cases = set()
            if kicks[pos-4:pos].sum() >= 3 and kicks[pos:pos+4].sum() >= 3:
                cases.add("groove")
            if ((~sounding[pos-4:pos].any(axis=0)) & sounding[future].any(axis=0)).any():
                cases.add("synth entry")
            if not kicks[pos-4:pos].any() and kicks[future].any():
                cases.add("kick return")
            # The slice keeps up to 20 earlier bars needed to verify the development leading to the handoff.
            if handoffs is not None and handoffs[start + 20:pos + length - 3].any():
                cases.add("motif handoff")
            options.append((i, start, length, cases, bool(changed[pos-1:pos+length-1].any())))
    if protocol == 2:
        return coverage_prompts(options, rng)
    if protocol == 3:
        return coverage_prompts(options, rng, handoffs=True)
    if protocol != 1:
        raise ValueError(f"unknown composition prompt protocol {protocol}")
    selected = []
    used = set()
    for requested in ("groove", "synth entry", "kick return"):
        remaining = [opt for opt in options if opt[0] not in used]
        if not remaining:
            # With two held-out records, use another nonoverlapping passage for the third prompt.
            remaining = [opt for opt in options if all(
                opt[0] != i or opt[1]+8+opt[2] <= start or start+8+length <= opt[1]
                for i, start, length, _ in selected)]
            if not remaining:
                break
        changing = [opt for opt in remaining if opt[4]]
        matches = [opt for opt in changing if requested in opt[3]]
        pool = matches or changing or remaining
        i, start, length, _, varies = pool[int(rng.integers(len(pool)))]
        case = requested if matches else ("other changing passage" if varies else "static passage")
        selected.append((i, start, length, case))
        used.add(i)
    return selected


def coverage_prompts(options, rng, handoffs=False):
    """Reserve scarce musical situations before common ones fill the available passages."""
    requested = ("groove", "synth entry", "kick return")
    by_case = {case: [o for o in options if case in o[3] and o[4]] for case in requested}
    if handoffs:
        matching = [o for o in options if "motif handoff" in o[3] and o[4]]
        if matching:
            requested += ("motif handoff",)
            by_case["motif handoff"] = matching
    selected = {}
    for case in sorted(requested, key=lambda name: len(by_case[name]) or float("inf")):
        used = {o[0] for o in selected.values()}
        picked_spans = {o[:3] for o in selected.values()}

        def disjoint(option):
            i, start, length = option[:3]
            return all(i != picked[0] or start + 8 + length <= picked[1] or
                       picked[1] + 8 + picked[2] <= start for picked in selected.values())

        matches = [o for o in by_case[case] if o[:3] not in picked_spans]
        available = [o for o in options if disjoint(o)]
        pool = ([o for o in matches if o[0] not in used] or [o for o in matches if disjoint(o)] or
                matches or [o for o in available if o[4]] or available)
        if pool:
            i, start, length, cases, varies = pool[int(rng.integers(len(pool)))]
            label = case if case in cases and varies else ("other changing passage" if varies else "static passage")
            selected[case] = (i, start, length, label)
    return [selected[case] for case in requested if case in selected]


def compare(load, tracks, temperature, bars, *, clear_recall=False, prompt_only_recall=False,
            audition=None, recall_trace=None, prompt_protocol=2, sampler=common.sample_slots, sampling_seed=23,
            pace_edits=False, groove_fit=None, dwell_bars=None, only_case=None):
    """Score actual player draws and a hold-last-bar baseline against the same real music."""
    if bars < 17:
        raise ValueError("composition evaluation needs at least 17 continuation bars")
    if prompt_only_recall and (clear_recall or recall_trace is not None):
        raise ValueError("prompt-only recall uses a fixed bank; choose either prompt-only, clearing, or tracing")
    rng = np.random.default_rng(131)
    test = [t for t in tracks if t["held_out"] and len(t["bars"]) >= 25]
    if not test:
        raise ValueError("composition evaluation needs a held-out track of at least 25 bars")
    # Prefer full-length excerpts when the split has them.
    test = [t for t in test if len(t["bars"]) >= bars + 8] or test
    runs = []
    selected = prompts(test, bars, rng, prompt_protocol)
    if only_case:
        selected = [prompt for prompt in selected if prompt[3] == only_case]
        if not selected:
            raise ValueError(f"No {only_case} prompt was available")
    for idx, start, length, case in selected:
        tr = test[idx]
        position = tr.get("offset", 0) + start
        seq = tr["bars"][start:start + 8 + length]
        moods = tr["mood"][start:start + len(seq)]
        brain = load()
        brain.reset()
        for t in range(7):
            common.answer(brain, common.observe(seq[t], moods[t + 1]))
        if prompt_only_recall:
            seed = brain.recall.snapshot()
            bank = {key: seed[key] for key in ("keys", "values", "nexts", "held", "depth") if key in seed}
        state = seq[7]
        if recall_trace is not None:
            recall_trace.prompt(tr["id"], position, seq[:7])
        generated = []
        draws = np.random.default_rng(sampling_seed + int(idx))
        controller = None
        if groove_fit is not None:
            controller = groove.Groove(groove_fit, [sampling_seed, int(idx), 1])
        elif dwell_bars is not None:
            controller = groove.VoiceDwell(dwell_bars)
        if controller is not None:
            controller.prime(seq[:8])
        for local_bar, mood in enumerate(moods[8:], start=8):
            if clear_recall:
                brain.recall.reset(1)
            elif prompt_only_recall:
                # Keep the live query context. Discard the new slot and its write to
                # the last recorded bar's successor before every generated bar.
                brain.recall.restore(brain.recall.snapshot() | bank)
            common.answer(brain, common.observe(state, mood))
            if recall_trace is not None:
                recall_trace.read(brain.recall, state)
            candidate = sampler(common.motor_activity(brain)[0], state, temperature, draws)
            if controller is not None:
                state = controller.play(state, candidate, position + local_bar)
            else:
                state = common.pace_phrase(state, candidate, position + local_bar) if pace_edits else candidate
            generated.append(state)
        generated = np.asarray(generated)
        repeated = np.repeat(seq[7:8], length, axis=0)
        target = profile(seq[8:], position + 8)
        hold = profile(repeated, position + 8)
        played = profile(generated, position + 8)
        events = event_agreement(
            phrase_events(seq, position, 8),
            phrase_events(np.concatenate([seq[:8], generated]), position, 8),
            phrase_events(np.concatenate([seq[:8], repeated]), position, 8),
        )
        development = event_agreement(
            development_events(seq, position, 8),
            development_events(np.concatenate([seq[:8], generated]), position, 8),
            development_events(np.concatenate([seq[:8], repeated]), position, 8),
        )
        del brain
        def agreement(read):
            return 1 - float(np.mean([abs(target[k] - read[k]) for k in target]))
        runs.append(dict(track=tr["id"], case=case, prompt_bar=position, bars=length, target=target, played=played,
                         hold_score=agreement(hold), score=agreement(played), phrase_events=events, development=development,
                         detail={"reference": detail_profile(*audible_patterns(seq[8:])[1:]),
                                 "played": detail_profile(*audible_patterns(generated)[1:])},
                         continuity={"reference": bar_continuity(seq[7:]),
                                     "played": bar_continuity(np.concatenate([seq[7:8], generated]))}))
        if audition is not None:
            runs[-1]["audio"] = audition(generated, seq[8:], position + 8, len(runs) - 1)
    result = dict(prompt_protocol=prompt_protocol, score=float(np.mean([r["score"] for r in runs])),
                  hold_score=float(np.mean([r["hold_score"] for r in runs])), runs=runs)
    result["over_hold"] = result["score"] - result["hold_score"]
    result["prompts_above_hold"] = sum(r["score"] > r["hold_score"] for r in runs)
    event_runs = [r["phrase_events"] for r in runs if r["phrase_events"]["score"] is not None]
    if event_runs:
        result["phrase_events"] = {k: float(np.mean([r[k] for r in event_runs])) for k in ("score", "hold")}
    development_runs = [r["development"] for r in runs if r["development"]["score"] is not None]
    if development_runs:
        result["development"] = {k: float(np.mean([r[k] for r in development_runs])) for k in ("score", "hold")}
    if audition is not None:
        from listen import merge
        result["audio"] = merge([r["audio"] for r in runs])
        # A frozen or inaudible stem must influence checkpoint selection even if
        # many sparse symbolic features happen to agree with their references.
        result["selection_score"] = (result["score"] * result["audio"]["score"]) ** .5
    else:
        result["selection_score"] = result["score"]
    return result


def rendered_comparison(card, pool, directory):
    from listen import stems, measure
    def audition(generated, reference, position, index):
        summaries = {}
        suffix = f"_{index + 1:02d}" if index else ""
        for name, stream in (("generated", generated), ("reference", reference)):
            wav = directory / f"{name}{suffix}.wav"
            db = stems(card, pool, stream, wav)
            summaries[name], _ = measure(stream, db, position)
        a, b = summaries["generated"], summaries["reference"]
        errors = [abs(a["heard"][key] - b["heard"][key]) for key in common.LAYERS]
        errors.extend(abs(a[k] - b[k]) for k in ("drums_only", "dead"))
        summaries["score"] = 1 - float(np.mean(errors))
        return summaries
    return audition
