"""Held-out reading, replay from memory and the free-run gate for a minimal-techno creature.

held_out(): teacher-forced next-bar answers on the held-out tracks, sounds scored separately on bars that
keep and bars that change (copy-last-bar is perfect on keeps, so the change bars carry the signal), plus
the modulation levels.
replay(): each track played from memory (seeded, then on its own output with the track's mood path).
free_run() + gate(): the brain plays on its own while the wanted mood tours the pad; statistics against
the corpus reference.
"""
from __future__ import annotations

import numpy as np

import common
import timbre
from common import L, LAYERS, observe, read_mood, stream_stats

TOUR = [(0.2, 0.25), (0.8, 0.3), (0.8, 0.8), (0.3, 0.75)]   # (energy, brightness) corners, N bars each


def _act(brain, state, mood, greedy=True):
    assert greedy
    return np.asarray(common.answer(brain, observe(state, mood))).reshape(-1, common.SLOTS)


def bigram_change_baseline(tracks):
    """Per layer: on bars that change, guess the most frequent other successor of the current token."""
    T = common.T
    trans = np.zeros((L, T, T))
    for tr in tracks:
        if tr["held_out"]:
            continue
        b = common.sounds(tr["bars"])
        for layer in range(L):
            np.add.at(trans[layer], (b[:-1, layer], b[1:, layer]), 1)
    for layer in range(L):
        np.fill_diagonal(trans[layer], -1)
    return trans.argmax(axis=2)


def held_out(brain, tracks):
    guess = bigram_change_baseline(tracks)
    hits = {name: dict(keep=[0, 0], change=[0, 0], detect=[0, 0], false=[0, 0], bigram=[0, 0]) for name in LAYERS}
    mods = [0, 0]
    for tr in tracks:
        if not tr["held_out"]:
            continue
        bars, mood = tr["bars"], tr["mood"]
        snd = common.sounds(bars)                                # what sounds (with on/off slots: sound x on)
        brain.reset()
        for t in range(len(bars) - 1):
            pred = _act(brain, bars[t:t + 1], mood[t + 1:t + 2], greedy=True)[0]
            mods[0] += int((pred[L:L + common.MODS] == bars[t + 1, L:L + common.MODS]).sum()); mods[1] += common.MODS
            psnd = common.sounds(pred)
            for layer, name in enumerate(LAYERS):
                cur, nxt, p = snd[t, layer], snd[t + 1, layer], psnd[layer]
                h = hits[name]
                if nxt == cur:
                    h["keep"][0] += int(p == nxt); h["keep"][1] += 1
                    h["false"][0] += int(p != cur); h["false"][1] += 1
                else:
                    h["change"][0] += int(p == nxt); h["change"][1] += 1
                    h["detect"][0] += int(p != cur); h["detect"][1] += 1
                    h["bigram"][0] += int(guess[layer, cur] == nxt); h["bigram"][1] += 1
    out = {}
    for name, h in hits.items():
        out[name] = {k: (v[0] / v[1] if v[1] else None) for k, v in h.items()}
        out[name]["change_bars"] = h["change"][1]
    keep = [out[n]["keep"] for n in LAYERS if out[n]["keep"] is not None]
    change = [out[n]["change"] for n in LAYERS if out[n]["change"] is not None]
    out["score"] = 0.5 * float(np.mean(keep or [0])) + 0.5 * float(np.mean(change or [0]))
    out["copy_score"] = 0.5
    out["mods"] = mods[0] / mods[1] if mods[1] else None
    return out


def commonest_bar(tracks):
    """The bar that holds every slot's commonest value over these tracks: what a brain writes that has learned the
    average of its lessons and nothing of what it hears."""
    bars = np.concatenate([t["bars"] for t in tracks])
    return np.array([np.bincount(bars[:, slot]).argmax() for slot in range(bars.shape[1])])


READ_STREAMS = 8      # how many passages a reading answers side by side


def next_bars(brain, tracks, streams=None):
    """The brain's next bar after every bar of every track: per track the answers to its bars 0 .. n - 2.

    The tracks are read side by side, ``streams`` at a time, each as a stream of its own: one act answers a
    bar of every track. A shorter track repeats its last bar until the longest has ended, and those answers
    are dropped. Read one track at a time, an evaluation of eight passages took eight times the acts and a
    training run spent half its time reading."""
    out = [None] * len(tracks)
    size = streams or READ_STREAMS
    for first in range(0, len(tracks), size):
        group = tracks[first:first + size]
        longest = max(len(tr["bars"]) for tr in group)
        at = lambda tr, i, key: tr[key][min(i, len(tr[key]) - 1)]
        answers = [[] for _ in group]
        brain.reset()
        for t in range(longest - 1):
            answer = _act(brain, np.stack([at(tr, t, "bars") for tr in group]),
                          np.stack([at(tr, t + 1, "mood") for tr in group]), greedy=True)
            for row, tr in enumerate(group):
                if t < len(tr["bars"]) - 1:
                    answers[row].append(answer[row])
        out[first:first + size] = answers
    return out


def rollouts(brain, tracks, prime=8, length=16, streams=None, most=0):
    """Free-running continuations from held-out openings: per track and opening, the brain hears ``prime`` bars of
    the score (teacher-forced), then plays ``length`` bars hearing its own answers, greedy. Openings start at
    bar 0, prime + length, ... while a whole opening and continuation fit; with ``most`` the first opening of each
    track comes first, so a cap keeps one continuation per track before it takes a second. Returns
    ``(pieces, answers)``: each piece is the score's passage from the last primed bar to the end of the
    continuation (with its marks), and ``answers`` the brain's bars aligned with it, ready for
    ``prediction_error(..., answers=answers)``."""
    span = prime + length
    jobs = []
    for tr in tracks:
        for start in range(0, len(tr["bars"]) - span + 1, span):
            jobs.append((tr, start))
    if most:
        jobs = sorted(jobs, key=lambda job: job[1])[:most]       # stable: the tracks keep their order per opening
    pieces, answers = [], []
    size = streams or READ_STREAMS
    for first in range(0, len(jobs), size):
        group = jobs[first:first + size]
        brain.reset()
        heard = np.stack([tr["bars"][start] for tr, start in group])
        played = [[] for _ in group]
        for t in range(span - 1):
            mood = np.stack([tr["mood"][start + t + 1] for tr, start in group])
            answer = _act(brain, heard, mood, greedy=True)
            if t + 1 < prime:                                # the opening: the score's next bar is heard
                heard = np.stack([tr["bars"][start + t + 1] for tr, start in group])
            else:                                            # the continuation: the brain hears itself
                for row in range(len(group)):
                    played[row].append(answer[row])
                heard = answer
        for (tr, start), bars in zip(group, played):
            pieces.append(common.passage(tr, start + prime - 1, length + 1))
            answers.append(bars)
    return pieces, answers


def bar_density(bars):
    """Per bar of a passage the audible events: kick steps, drum hits, bass and lead notes (onsets), sounding voices."""
    bars = np.asarray(bars, dtype=int).reshape(-1, common.SLOTS)
    sound = common.sounds(bars) > 0
    drums = bars[:, L + common.MODS:L + common.MODS + common.STEPS] > 0
    drums = drums & sound[:, :1]                      # heard only while the beat sounds
    width = common.STEPS // len(common.DRUM_ROWS)
    notes = np.stack([common.decode_synth(b) for b in bars]) >= 2      # (bars, 3, 16): onsets
    notes = notes & sound[:, 1:L, None]
    return {"kick": drums[:, :width].sum(axis=1), "drums": drums.sum(axis=1), "bass": notes[:, 0].sum(axis=1),
            "lead": notes[:, 1].sum(axis=1), "voices": sound.sum(axis=1)}


DENSITY_SEGMENTS = ((1, 8), (9, 16), (17, 32), (33, 64))


def change_rates(bars, history):
    """Per bar of a passage the slots that changed against the bar before, by group (arrangement, drum rows, synth
    lines, sound controls), split into RETURNS (the new value is the one the slot held 2, 4, 8 or 16 bars back) and
    NEW; ``history`` the bars before the passage (oldest first), so the first bars have their lags."""
    bars = np.asarray(bars, dtype=int).reshape(-1, common.SLOTS)
    every = np.concatenate([np.asarray(history, dtype=int).reshape(-1, common.SLOTS), bars])
    first = len(every) - len(bars)
    base = L + common.MODS
    groups = {"arrangement": slice(0, L), "drum_rows": slice(base, base + common.STEPS),
              "synth_line": slice(base + common.STEPS, base + common.STEPS + common.SYN),
              "synth_controls": slice(base + common.STEPS + common.SYN, common.LIVE)}
    out = {f"{g}_{kind}": [] for g in groups for kind in ("returns", "new")}
    for i in range(first, len(every)):
        now, before = every[i], every[i - 1]
        changed = now != before
        back = [every[i - k] for k in (2, 4, 8, 16) if i - k >= 0]
        returning = changed & (np.any([b == now for b in back], axis=0) if back else np.zeros_like(changed))
        for g, cut in groups.items():
            out[f"{g}_returns"].append(int(returning[cut].sum()))
            out[f"{g}_new"].append(int((changed & ~returning)[cut].sum()))
    return {k: np.asarray(v) for k, v in out.items()}


def density_rollout(brain, tracks, prime=8, length=64, passages=4, streams=None, temperature=0.0, seed=0):
    """Long free play from held-out openings against the material's own continuation: per segment of the played bars
    (bars 1-8, 9-16, 17-32, 33-64 of the continuation, as far as ``length`` reaches) the mean kick steps, drum hits,
    bass notes, lead notes and sounding voices per bar, the brain's beside the material's, and the density ratio of
    the last segment (drum hits + bass notes + lead notes, brain over material). Greedy play, the brain hearing its own
    bars after the opening."""
    span = prime + length
    rng = np.random.default_rng(seed)
    chosen = [tr for tr in tracks if len(tr["bars"]) >= span][:passages]
    if not chosen:
        return None
    pieces, played, openings = [], [], []
    size = streams or READ_STREAMS
    for first in range(0, len(chosen), size):
        group = chosen[first:first + size]
        brain.reset()
        heard = np.stack([tr["bars"][0] for tr in group])
        out = [[] for _ in group]
        for t in range(span - 1):
            mood = np.stack([tr["mood"][t + 1] for tr in group])
            answer = _act(brain, heard, mood, greedy=True)
            if t + 1 < prime:
                heard = np.stack([tr["bars"][t + 1] for tr in group])
            else:
                if temperature > 0:          # sampled as the player samples (common.sample_slots)
                    act = common.motor_activity(brain)
                    answer = np.stack([common.sample_slots(act[row], heard[row], temperature, rng)
                                       for row in range(len(group))])
                for row in range(len(group)):
                    out[row].append(answer[row])
                heard = answer
        for tr, bars in zip(group, out):
            pieces.append(np.asarray(tr["bars"][prime:span]))
            played.append(np.asarray(bars))
            openings.append(np.asarray(tr["bars"][:prime]))
    segments = {}
    ratio = None
    for lo, hi in DENSITY_SEGMENTS:
        if lo > length:
            break
        hi = min(hi, length)
        def mean(sets):
            stats = [bar_density(b[lo - 1:hi]) for b in sets]
            return {k: round(float(np.mean(np.concatenate([st[k] for st in stats]))), 3) for k in stats[0]}
        brain_side, material = mean(played), mean(pieces)
        def rates(sets):
            per = [change_rates(b, opening) for b, opening in zip(sets, openings)]
            return {k: round(float(np.mean(np.concatenate([st[k][lo - 1:hi] for st in per]))), 3) for k in per[0]}
        brain_side["changes"], material["changes"] = rates(played), rates(pieces)
        segments[f"{lo}-{hi}"] = {"brain": brain_side, "material": material}
        events = lambda d: d["drums"] + d["bass"] + d["lead"]
        ratio = events(brain_side) / events(material) if events(material) else None
    return {"segments": segments, "density_ratio_last": None if ratio is None else round(ratio, 3),
            "passages": len(chosen), "bars": length, "temperature": temperature}


def prediction_error(brain, tracks, constant=None, answers=None):
    """Exact next-bar slot error. Passing no brain measures repeating the previous bar, or with ``constant``
    writing that one bar whatever was heard; ``answers`` (per track, one bar per bar after its first) scores
    bars played elsewhere, e.g. a free-running continuation (rollouts)."""
    groups = ("arrangement", "drum_rows", "synth_line", "synth_controls")
    wrong = np.zeros(len(groups), dtype=np.int64)
    seen = np.zeros_like(wrong)
    base = L + common.MODS
    width = (common.LIVE-base-common.STEPS-common.SYN)//3       # live slots only: a reserved block is not read
    samples = common.PATCH_BOOK.get("samples")
    kind = (common.SYNTH or {}).get("kind")
    detail = {key: [0, 0, 0] for key in ("drum_hits", "drum_accents", "synth_attacks", "synth_notes", "synth_ties")} if (
        common.STEPS and not common.ROWS and kind in ("steps", "splitsteps")) else {}
    right_sample = misses = false_sample = positive = negative = 0
    # the same slots again, counted for the instrument that plays them: its sound slot, and its pattern,
    # line and sound controls while it sounds in the reference
    part_wrong = np.zeros(L, dtype=np.int64)
    part_seen = np.zeros_like(part_wrong)
    chip = common.chips() and common.SID_LIVE          # the SID blocks are taught: they get a reading of their own
    chip_wrong = chip_seen = 0
    # the acid voice's own row (score v5a): line steps of a synth whose taught sound is the acid voice, and its notes
    acid_voice = timbre.VOICES.index("acid") if common.PATCH_BOOK and "acid" in timbre.VOICES else None
    acid = {"wrong": 0, "seen": 0, "ref": 0, "played": 0, "hit": 0}
    open_lines = [[0, 0] for _ in range(3)]       # per synth: open line steps and steps read while it sounds
    open_sounds = [[0, 0] for _ in range(3)]      # per synth: sounding bars whose sound the score did not read
    kit = [0, 0, 0, 0]                            # slot 0, the beat's kit: wrong on staying bars, staying bars, wrong on changes, changes
    # the changed live slots split: RETURNS (the new value is the one the slot held 2, 4, 8 or 16 bars before the bar
    # written) and NEW (no such lag has it): wrong and seen for each
    echo = {"returns": [0, 0], "new": [0, 0]}
    plan_wrong = plan_seen = 0           # the plan slots (common.plan_rows), read apart from the live slots
    section_wrong = section_seen = 0     # the section slot (common.section_rows)
    # the loudness slots of the synth patches, read apart while the synth sounds: the gain step and the level
    # control, as the share of wrong answers and as the mean distance in steps
    loud_slots = {"gain": 2, "level": 3 + timbre.NAMES.index("level")} if common.PATCH_BOOK else {}
    loud = {key: [0, 0, 0] for key in loud_slots}
    # the levels: the drum level (hit or accent) on the steps both bars hit, the synths' level control while the
    # synth sounds, and the synths' velocity on the attacks both bars play (one velocity slot per sixteenth
    # position within a beat)
    level_names = ("drum_levels", "synth_level", "synth_accents")
    levels = {key: [0, 0] for key in level_names} if detail and common.PATCH_BOOK else {}
    velocity = len(timbre.PATCH_SIZES)
    # the same four groups again, split by whether the taught slot differs from the bar heard: "change" is the
    # share of the changing slots answered wrong (repeating the bar gets every one of them wrong) and "stay" the
    # share of the unchanged slots answered wrong (repeating gets none wrong)
    turned = np.zeros((2, len(groups)), dtype=np.int64)
    turned_seen = np.zeros_like(turned)
    # the drum rows and synth lines again on the bars the score's reader was sure of: a drum row or a line
    # counts where the reader was sure of it in the bar heard and in the bar taught. The rest of the error on
    # these rows includes the reader's flicker, which nothing can predict.
    sure_wrong = np.zeros(2, dtype=np.int64)
    sure_seen = np.zeros_like(sure_wrong)
    marked = bool(detail) and all(all(len(tr.get(key, ())) == len(tr["bars"]) for key in common.SURE) for tr in tracks)
    # score v4: a drum step the reader left open in the taught bar is no lesson and counts in no error, for the
    # brain and for repeating alike
    every = np.ones(common.STEPS, dtype=bool)
    if answers is None:
        answers = next_bars(brain, tracks) if brain is not None else None
    for index, tr in enumerate(tracks):
        bars, mood = tr["bars"], tr["mood"]
        for t, expected in enumerate(bars[1:]):
            actual = (constant if constant is not None else bars[t]) if answers is None else answers[index][t]
            mistakes = actual != expected
            moved = np.asarray(bars[t]) != expected
            changed = moved[:common.LIVE]
            if changed.any():
                back = [bars[t + 1 - k][:common.LIVE] for k in (2, 4, 8, 16) if t + 1 - k >= 0]
                returning = changed & (np.any([b == expected[:common.LIVE] for b in back], axis=0) if back
                                       else np.zeros_like(changed))
                fresh = changed & ~returning
                echo["returns"][0] += int(mistakes[:common.LIVE][returning].sum()); echo["returns"][1] += int(returning.sum())
                echo["new"][0] += int(mistakes[:common.LIVE][fresh].sum()); echo["new"][1] += int(fresh.sum())
            if moved[0]:
                kit[2] += int(mistakes[0]); kit[3] += 1
            else:
                kit[0] += int(mistakes[0]); kit[1] += 1
            drum = slice(base, base+common.STEPS if expected[0] else base)
            counted = ~tr[common.OPEN][t + 1] if common.OPEN in tr else every      # the drum steps that count
            drum_errors = mistakes[drum][counted[:drum.stop - drum.start]]
            drum_moved = moved[drum][counted[:drum.stop - drum.start]]
            lead_start = base+common.STEPS
            ctrl_start = lead_start+common.SYN
            active = expected[1:L] > 0
            lines = mistakes[lead_start:ctrl_start].reshape(3, -1)
            moved_lines = moved[lead_start:ctrl_start].reshape(3, -1)
            # the line steps that count: an open line step (score LINE_OPEN) counts for nobody
            line_counted = (~np.asarray(tr[common.LINE_OPEN][t + 1], dtype=bool).reshape(3, -1)
                            if common.LINE_OPEN in tr else np.ones((3, common.SYN // 3), dtype=bool))
            if common.LINE_OPEN in tr:
                for v in np.flatnonzero(expected[1:L] > 0):
                    open_lines[v][0] += int((~line_counted[v]).sum()); open_lines[v][1] += line_counted.shape[1]
            if detail or kind == "splitsteps":
                notes, played_notes = common.decode_synth(expected), common.decode_synth(actual)
            if kind == "splitsteps":
                lines = notes != played_notes
                moved_lines = common.decode_synth(bars[t]) != notes
            if detail:
                drums = common.decode_beat(expected) * (expected[0] > 0) * counted.reshape(3, -1)
                played_drums = common.decode_beat(actual) * (actual[0] > 0) * counted.reshape(3, -1)
                notes = notes * active[:, None] * line_counted
                played_notes = played_notes * (actual[1:L, None] > 0) * line_counted
                for key, ref, played, match in (
                        ("drum_hits", drums > 0, played_drums > 0, True),
                        ("drum_accents", drums == 2, played_drums == 2, True),
                        ("synth_attacks", notes >= 2, played_notes >= 2, True),
                        ("synth_notes", notes >= 2, played_notes >= 2, notes == played_notes),
                        ("synth_ties", notes == 1, played_notes == 1, True)):
                    counts = detail[key]
                    counts[0] += int((ref & played & match).sum())
                    counts[1] += int(ref.sum())
                    counts[2] += int(played.sum())
            controls = mistakes[ctrl_start:common.LIVE].reshape(3, width)
            moved_controls = moved[ctrl_start:common.LIVE].reshape(3, width)
            # the sound controls that count: a layer whose sound the score did not read (score CONTROLS_OPEN) counts
            # with its voice and register only, for the brain and for repeating alike
            unread = (np.asarray(tr[common.CONTROLS_OPEN][t + 1], dtype=bool) if common.CONTROLS_OPEN in tr
                      else np.zeros(3, dtype=bool))
            ctrl_counted = np.ones((3, width), dtype=bool)
            ctrl_counted[unread, 2:] = False
            read = active & ~unread
            if common.CONTROLS_OPEN in tr:
                for v in np.flatnonzero(active):
                    open_sounds[v][0] += int(unread[v]); open_sounds[v][1] += 1
            if levels:
                shared = (drums > 0) & (played_drums > 0)
                levels["drum_levels"][0] += int((drums != played_drums)[shared].sum())
                levels["drum_levels"][1] += int(shared.sum())
                level = ctrl_start + np.flatnonzero(read) * width + loud_slots["level"]
                levels["synth_level"][0] += int((np.asarray(actual)[level] != expected[level]).sum())
                levels["synth_level"][1] += len(level)
                want = np.asarray(expected[ctrl_start:common.LIVE]).reshape(3, width)[:, velocity:velocity + 4]
                gave = np.asarray(actual[ctrl_start:common.LIVE]).reshape(3, width)[:, velocity:velocity + 4]
                attacks = (notes >= 2) & (played_notes >= 2) & ~unread[:, None]
                beat = np.arange(attacks.shape[1]) % 4
                levels["synth_accents"][0] += int(((want != gave)[:, beat] & attacks).sum())
                levels["synth_accents"][1] += int(attacks.sum())
            if samples:
                shown = common.sample_voices(expected)
                played = common.sample_voices(actual)
                positive += int(shown.sum())
                negative += int((~shown).sum())
                misses += int((shown & ~played).sum())
                false_sample += int((played & ~shown).sum())
                indices = slice(ctrl_start + width - 3, common.LIVE, width)
                right_sample += int((shown & played & ~mistakes[indices]).sum())
                masked = np.repeat(active[:, None], width, axis=1)
                masked[:, -3:] &= expected[ctrl_start:common.LIVE:width, None] == 2
                masked &= ctrl_counted
                control_errors = controls[masked]
                moved_controls = moved_controls[masked]
            else:
                masked = np.repeat(active[:, None], width, axis=1) & ctrl_counted
                control_errors = controls[masked]
                moved_controls = moved_controls[masked]
            for key, slot in loud_slots.items():
                where = ctrl_start + np.flatnonzero(read) * width + slot
                loud[key][0] += int((actual[where] != expected[where]).sum())
                loud[key][1] += int(np.abs(np.asarray(actual)[where].astype(int) - np.asarray(expected)[where].astype(int)).sum())
                loud[key][2] += len(where)
            if common.PLAN:
                plan_wrong += int(mistakes[common.chip_end():common.chip_end() + common.PLAN].sum()); plan_seen += common.PLAN
            if common.SECTIONS:
                section_wrong += int(mistakes[common.section_slot()]); section_seen += 1
            if chip:
                layers = list(common.chip_layers())       # the synths with a block: all three, or the lead alone
                blocks = mistakes[common.LIVE:common.chip_end()].reshape(len(layers), -1)[np.asarray(active)[layers]]
                chip_wrong += int(blocks.sum()); chip_seen += blocks.size
            if acid_voice is not None:
                for v in np.flatnonzero(active):
                    if expected[ctrl_start + v * width] == acid_voice:
                        counted_v = line_counted[v]
                        acid["wrong"] += int(lines[v][counted_v].sum()); acid["seen"] += int(counted_v.sum())
                        if detail:
                            want, got = notes[v] * counted_v, played_notes[v] * counted_v
                            acid["ref"] += int((want >= 2).sum()); acid["played"] += int((got >= 2).sum())
                            acid["hit"] += int(((want >= 2) & (got >= 2) & (want == got)).sum())
            slices = (mistakes[:L], drum_errors, lines[active][line_counted[active]], control_errors)
            wrong += [s.sum() for s in slices]
            seen += [s.size for s in slices]
            if marked:
                rows = tr["drums_sure"][t] & tr["drums_sure"][t + 1]
                if expected[0]:
                    steps = mistakes[base:base + common.STEPS].reshape(len(rows), -1)[rows][counted.reshape(len(rows), -1)[rows]]
                    sure_wrong[0] += int(steps.sum()); sure_seen[0] += steps.size
                both = active & tr["synth_sure"][t] & tr["synth_sure"][t + 1]
                heard = lines[both][line_counted[both]]
                sure_wrong[1] += int(heard.sum()); sure_seen[1] += heard.size
            for i, (bad, change) in enumerate(zip(slices, (moved[:L], drum_moved, moved_lines[active][line_counted[active]], moved_controls))):
                turned[0, i] += int(bad[change].sum()); turned_seen[0, i] += int(change.sum())
                turned[1, i] += int(bad[~change].sum()); turned_seen[1, i] += int((~change).sum())
            own = [[mistakes[:1], drum_errors]] + [[mistakes[1+i:2+i]] for i in range(3)]
            for i in np.flatnonzero(active):
                own[1+i] += [lines[i][line_counted[i]], controls[i][masked[i]]]
            part_wrong += [sum(int(s.sum()) for s in part) for part in own]
            part_seen += [sum(s.size for s in part) for part in own]
    errors = {key: float(bad/total) if total else None for key,bad,total in zip(groups,wrong,seen)}
    values = [x for x in errors.values() if x is not None]
    result = {"balanced": float(np.mean(values)) if values else None, **errors,
              "instruments": {name: float(bad/total) if total else None
                              for name, bad, total in zip(common.LAYERS, part_wrong, part_seen)}}
    for name, bad, total in zip(("change", "stay"), turned, turned_seen):
        parts = {key: float(b / n) if n else None for key, b, n in zip(groups, bad, total)}
        known = [x for x in parts.values() if x is not None]
        result[name] = {"balanced": float(np.mean(known)) if known else None, **parts}
    result["changing"] = {key: float(c / (c + k)) if c + k else None for key, c, k in zip(groups, *turned_seen)}
    if marked:
        result["sure"] = {**{key: float(bad / total) if total else None
                             for key, bad, total in zip(("drum_rows", "synth_line"), sure_wrong, sure_seen)},
                          "share": {key: float(part / total) if total else None
                                    for key, part, total in zip(("drum_rows", "synth_line"), sure_seen, seen[1:3])}}
    if levels:
        parts = {key: bad / total if total else None for key, (bad, total) in levels.items()}
        known = [x for x in parts.values() if x is not None]
        result["levels"] = {"balanced": float(np.mean(known)) if known else None, **parts}
    if chip:
        result["sid"] = float(chip_wrong/chip_seen) if chip_seen else None
    if acid["seen"]:
        result["acid"] = {"line_error": acid["wrong"] / acid["seen"], "line_steps": acid["seen"],
                          "note_f1": (2 * acid["hit"] / (acid["ref"] + acid["played"])) if acid["ref"] + acid["played"] else None}
    if common.PLAN:
        result["plan"] = float(plan_wrong/plan_seen) if plan_seen else None
    if common.SECTIONS:
        result["section"] = float(section_wrong/section_seen) if section_seen else None
    if any(seen_ for _, seen_ in open_lines):
        # the share of line steps the score leaves open, per synth: a rise in note F1 is read against it
        result["open_lines"] = {name: (bad / n if n else None) for name, (bad, n) in zip(("bass", "lead", "air"), open_lines)}
    result["changed_split"] = {key: {"error": w / n if n else None, "slots": n} for key, (w, n) in echo.items()}
    result["kit"] = {"stay": kit[0] / kit[1] if kit[1] else None, "change": kit[2] / kit[3] if kit[3] else None,
                     "changes": kit[3]}
    if any(seen_ for _, seen_ in open_sounds):
        result["open_sounds"] = {name: (bad / n if n else None) for name, (bad, n) in zip(("bass", "lead", "air"), open_sounds)}
    if loud:
        result["loudness"] = {key: {"error": wrong/seen if seen else None, "steps": far/seen if seen else None}
                              for key, (wrong, far, seen) in loud.items()}
    if detail:
        result["detail"] = {key: {"f1": 2*hits/(ref+played) if ref+played else None, "reference": ref, "played": played}
                            for key, (hits, ref, played) in detail.items()}
    if samples:
        result["sampler"] = {"targets":positive, "miss_rate":misses/positive if positive else None,
                             "identity_accuracy":right_sample/positive if positive else None,
                             "false_trigger_rate":false_sample/negative if negative else None}
    return result


def replay(brain, tracks, bars=128, seed_bars=4):
    """Play each track from memory, all tracks side by side (one stream each): seeded with its first bars,
    then the brain answers greedily on its own output while the track's own mood path is given. Per track
    and layer the share of bars whose sound equals the track's, 'mods' the share of modulation levels,
    'hold' the same share for holding the seed bar forever."""
    use = [tr for tr in tracks if len(tr["bars"]) > seed_bars]
    n = [min(bars, len(tr["bars"]) - seed_bars) for tr in use]
    N = max(n)
    def at(tr, i, key):                 # a track's row i, its last row past its end
        a = tr[key]
        return a[min(i, len(a) - 1)]
    brain.reset()
    for t in range(seed_bars - 1):
        common.answer(brain, observe(np.stack([at(tr, t, "bars") for tr in use]), np.stack([at(tr, t + 1, "mood") for tr in use])))
    cur = np.stack([at(tr, seed_bars - 1, "bars") for tr in use])
    hit = np.zeros((len(use), L)); mhit = np.zeros(len(use))
    for k in range(N):
        i = seed_bars - 1 + k
        cur = np.asarray(common.answer(brain, observe(cur, np.stack([at(tr, i + 1, "mood") for tr in use])))).reshape(len(use), common.SLOTS)
        for r, tr in enumerate(use):
            if k < n[r]:
                hit[r] += common.sounds(cur[r]) == common.sounds(tr["bars"][i + 1])
                mhit[r] += float((cur[r, L:L + common.MODS] == tr["bars"][i + 1, L:L + common.MODS]).mean())
    out = {}
    for r, tr in enumerate(use):
        b = tr["bars"]
        acc = hit[r] / n[r]
        sb = common.sounds(b)
        hold = (sb[seed_bars:seed_bars + n[r]] == sb[seed_bars - 1]).mean(axis=0)
        out[tr["id"]] = {"title": tr["title"], "held_out": tr["held_out"], "bars": int(n[r]),
                         **{LAYERS[l]: round(float(acc[l]), 3) for l in range(L)},
                         "mean": round(float(acc.mean()), 3), "hold": round(float(hold.mean()), 3),
                         "mods": round(float(mhit[r] / n[r]), 3)}
    rows = [v for v in out.values() if isinstance(v, dict)]
    trained = [v["mean"] for v in rows if not v["held_out"]]
    out["mean_trained"] = round(float(np.mean(trained)), 3) if trained else None
    out["mean_all"] = round(float(np.mean([v["mean"] for v in rows])), 3)
    out["hold_all"] = round(float(np.mean([v["hold"] for v in rows])), 3)
    out["mods_all"] = round(float(np.mean([v["mods"] for v in rows])), 3)
    return out


def free_run(brain, tracks, bars_per_corner=64, greedy=False, temperature=None, seed_bars=8, groove_fit=None, dwell_bars=None):
    held = [t for t in tracks if t["held_out"]] or tracks       # seed with the held-out track that has the most beat
    held = [t for t in held if len(t["bars"]) >= seed_bars]
    seed = max(held, key=lambda t: float((common.sounds(t["bars"])[:, 0] > 0).mean()))
    tour = np.repeat(np.array(TOUR, dtype=float), bars_per_corner, axis=0)
    start = seed_bars + (seed.get("offset", 0) if groove_fit is not None or dwell_bars is not None else 0)
    mood_path = common.mood_rows(tour, seed.get("style"), bars=np.arange(start, start + len(tour)))
    brain.reset()
    for t in range(seed_bars - 1):
        _act(brain, seed["bars"][t:t + 1], seed["mood"][t + 1:t + 2], greedy=True)
    cur = seed["bars"][seed_bars - 1:seed_bars]
    out = []
    rng = np.random.default_rng(23)
    controller = None
    if groove_fit is not None:
        from groove import Groove

        controller = Groove(groove_fit, [23, 1])
    elif dwell_bars is not None:
        from groove import VoiceDwell

        controller = VoiceDwell(dwell_bars)
    if controller is not None:
        controller.prime(seed["bars"][:seed_bars])
    for k in range(len(mood_path)):
        x = observe(cur, mood_path[k:k + 1])
        action = common.answer(brain, x)
        proposed = (np.asarray(action).reshape(-1) if greedy else
                    common.sample_slots(common.motor_activity(brain)[0], cur[0], temperature or 0.07, rng))
        cur = (controller.play(cur[0], proposed, start + k) if controller else proposed)[None]
        out.append(cur[0])
    return np.asarray(out, dtype=int), mood_path


def gate(bars, mood_path, ref, offset=8) -> dict:
    st = stream_stats(bars, offset=offset)
    read = read_mood(bars, ref["token_mood"])
    corr = []
    for k in range(2):
        a, b = mood_path[:, k], read[:, k]
        corr.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0)
    checks = []
    for name in LAYERS:
        rc, c = ref[name]["change"], st[name]["change"]
        lo, hi = 0.4 * rc, max(2.5 * rc, rc + 0.05)
        checks.append({"check": f"{name} change rate", "value": round(c, 3), "range": [round(lo, 3), round(hi, 3)], "pass": lo <= c <= hi})
    rm = ref["mod_change"]
    lo, hi = 0.4 * rm, max(2.5 * rm, rm + 0.05)
    checks.append({"check": "modulation change rate", "value": round(st["mod_change"], 3), "range": [round(lo, 3), round(hi, 3)], "pass": lo <= st["mod_change"] <= hi})
    ra = ref.get("aligned8", 0.0)
    checks.append({"check": "share of sound changes on 8-bar boundaries", "value": round(st["aligned8"], 3), "range": [round(0.6 * ra, 3), 1.0], "pass": st["aligned8"] >= 0.6 * ra})
    checks.append({"check": "longest stretch without any sound change (bars)", "value": st["longest_still"], "range": [0, 96], "pass": st["longest_still"] <= 96})
    for name in LAYERS:                      # sounds hold about as long as in the corpus
        rr, r = ref[name]["run_mean"], st[name]["run_mean"]
        checks.append({"check": f"{name} mean hold (bars)", "value": round(r, 2), "range": [round(0.5 * rr, 2), round(2.0 * rr, 2)], "pass": 0.5 * rr <= r <= 2.0 * rr})
    bs, rs = st["beat"]["silence"], ref["beat"]["silence"]   # the kick keeps going, and breaks still happen
    lo, hi = max(0.0, rs - 0.25), min(1.0, rs + 0.25)
    checks.append({"check": "beat silence share (breaks)", "value": round(bs, 3), "range": [round(lo, 3), round(hi, 3)], "pass": lo <= bs <= hi})
    if "drums" in st and "drums" in ref:           # the beat it writes, against the corpus's beats
        for k, lo_f, hi_f, lab in (("four_floor", 0.6, 1.4, "share of beat bars with four kicks"), ("snare_2_4", 0.5, 1.6, "share with snare/clap on 2 and 4"),
                                   ("beat_change", 0.4, 2.5, "beat change rate"), ("hat_density", 0.5, 1.6, "hat density")):
            rv, v = ref["drums"][k], st["drums"][k]
            lo, hi = lo_f * rv, max(hi_f * rv, rv + 0.05)
            checks.append({"check": lab, "value": round(v, 3), "range": [round(lo, 3), round(hi, 3)], "pass": lo <= v <= hi})
    checks.append({"check": "energy follows the mood pad (corr)", "value": round(corr[0], 3), "range": [0.2, 1], "pass": corr[0] >= 0.2})
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "mood_corr": corr, "stats": st}


TEMPS = (0.02, 0.05, 0.07, 0.085, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3)


def calibrate(load, tracks, ref, bars_per_corner=32, temps=TEMPS, groove_fit=None, dwell_bars=None):
    """Choose the sampling temperature whose free run changes sounds as often as the corpus does (per-layer
    change rate, averaged over layers). `load()` returns a fresh copy of the brain. Returns (best T, scan)."""
    target = float(np.mean([ref[n]["change"] for n in LAYERS]))
    scan = []
    if common.PATCH_BOOK:
        from composition import profile
        real = [profile(tr["bars"]) for tr in tracks if not tr["held_out"] and len(tr["bars"]) >= 17]
        if not real:
            raise ValueError("DSP temperature calibration needs at least 17 consecutive training bars")
        examples = [tr for tr in tracks if not tr["held_out"] and len(tr["bars"]) >= 17]
        weights = [tr["window_share"] / sum(t.get("group")==tr["group"] for t in examples)
                   if tr.get("window_share") else 1.0 for tr in examples]
        reference = {k: float(np.average([p[k] for p in real], weights=weights)) for k in real[0]}
    for t in temps:
        bars, _ = free_run(load(), tracks, bars_per_corner=bars_per_corner, temperature=t,
                           groove_fit=groove_fit, dwell_bars=dwell_bars)
        stats = stream_stats(bars)
        rate = float(np.mean([stats[n]["change"] for n in LAYERS]))
        error = abs(rate - target)
        if common.PATCH_BOOK:
            actual = profile(bars, offset=8)
            error += float(np.mean([abs(actual[k] - v) for k, v in reference.items()]))
        scan.append({"temperature": t, "change": round(rate, 3), "error": error})
    best = min(scan, key=lambda r: r["error"])
    return best["temperature"], {"target": round(target, 3), "scan": scan}
