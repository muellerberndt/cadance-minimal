"""Learn when to keep, replace, or drop each of the composer's voices."""
from collections import Counter

import numpy as np

import common
import timbre


AGE_BINS = (1, 2, 4, 8, 16, 32)
PHASES = 8
VERSION = 2


def patterns(state):
    audible = common.sounds(state) > 0
    kick = common.decode_beat(state)[0] > 0
    notes = common.decode_synth(state)
    return np.vstack((kick, notes)) * audible[:, None]


def large_edits(before, after):
    occupied = ((before != 0) | (after != 0)).sum(axis=1)
    return 2 * (before != after).sum(axis=1) > occupied


def patch_identity(layer):
    start = common.L + common.MODS + common.STEPS + common.SYN
    width = timbre.width(common.PATCH_BOOK)
    slots = [0, 1, 2, 3 + timbre.NAMES.index("wave")]
    if common.PATCH_BOOK.get("samples"):
        slots.extend(range(len(timbre.SIZES), width))
    return start + layer * width + np.asarray(slots)


class VoiceDwell:
    """Give the brain's larger edits a few bars to settle into the groove."""
    def __init__(self, bars, lets_go=None):
        bars = np.asarray(bars)
        if bars.shape not in ((), (common.L,)) or bars.dtype.kind not in "iu" or (bars < 2).any():
            raise ValueError("Voice dwell needs at least two bars for each voice")
        self.bars = np.broadcast_to(bars, (common.L,))
        self.age = np.ones(common.L, dtype=int)
        # lets_go: per voice, whether the dwell lets a sounding voice leave (it still holds sound changes and
        # large edits); default: it holds exits too
        self.lets_go = np.broadcast_to(np.asarray(lets_go if lets_go is not None else False, dtype=bool), (common.L,))

    def _changes(self, current, proposed):
        old, new = common.sounds(current), common.sounds(proposed)
        edits = large_edits(patterns(current), patterns(proposed))
        changed = edits | (old != new)
        for layer in range(len(common.SYNTH_LAYERS)):
            slots = patch_identity(layer)
            if old[layer + 1] and new[layer + 1] and np.any(current[slots] != proposed[slots]):
                changed[layer + 1] = True
        return old, edits, changed

    def prime(self, bars):
        for current, played in zip(bars[:-1], bars[1:]):
            self.hear(current, played)

    def hear(self, current, played):
        _, _, changed = self._changes(np.asarray(current), np.asarray(played))
        self.age = np.where(changed, 1, self.age + 1)

    def play(self, current, proposed, position):
        current = np.asarray(current)
        old, edits, changed = self._changes(current, proposed)
        played = proposed.copy()
        keep = changed & (old > 0) & (self.age < self.bars)
        if self.lets_go.any():
            keep &= ~(self.lets_go & (common.sounds(proposed) == 0))       # a voice that leaves may leave
        beat = common.L + common.MODS
        if keep[0]:
            common.set_sound(played, 0, int(old[0]))
            if edits[0]:
                kick = slice(beat, beat + common.STEPS // len(common.DRUM_ROWS))
                played[kick] = current[kick]
        line_width = common.SYN // len(common.SYNTH_LAYERS)
        for layer in np.flatnonzero(keep[1:]):
            voice = layer + 1
            common.set_sound(played, voice, int(old[voice]))
            start = beat + common.STEPS + layer * line_width
            line = slice(start, start + line_width)
            if edits[voice]:
                played[line] = current[line]
            slots = patch_identity(layer)
            played[slots] = current[slots]
        self.hear(current, played)
        return played


def advance(presence_age, pattern_age, before, after, changed):
    left, right = before.any(axis=1), after.any(axis=1)
    return (np.where(left != right, 1, presence_age + 1),
            np.where(left & right & ~changed, pattern_age + 1, 1))


def fit(tracks, synth_groups):
    """Estimate entrance, exit and rewrite rates by elapsed bars and phrase position."""
    shape = (common.L, 2, len(AGE_BINS) + 1, PHASES)
    presence_trials = np.zeros(shape, dtype=int)
    switches = np.zeros_like(presence_trials)
    pattern_trials = np.zeros_like(presence_trials)
    edits = np.zeros_like(presence_trials)
    layers = np.arange(common.L)
    beat = common.L + common.MODS
    kick = slice(beat, beat + common.STEPS // len(common.DRUM_ROWS))
    kick_entries = [Counter() for _ in range(PHASES)]
    heard_kicks = Counter()
    breaks, clean_breaks = [], []
    for track in tracks:
        if track["held_out"] or len(track["bars"]) < 2:
            continue
        taught = layers if track.get("group") in synth_groups else layers[:1]
        bars = track["bars"]
        before = patterns(bars[0])
        offset = track.get("offset", 0)
        last_kick = offset if before[0].any() else None
        presence_age = pattern_age = np.ones(common.L, dtype=int)
        for position, bar in enumerate(bars[1:], start=offset + 1):
            after = patterns(bar)
            left, right = before.any(axis=1), after.any(axis=1)
            changed = large_edits(before, after)
            phase = position % PHASES
            if right[0]:
                anchor = (int(common.sounds(bar)[0]), *map(int, bar[kick]))
                heard_kicks[anchor] += 1
                if not left[0]:
                    kick_entries[phase][anchor] += 1
                if last_kick is not None and position > last_kick + 1:
                    duration = position - last_kick - 1
                    breaks.append(duration)
                    if track.get("group") in synth_groups:
                        clean_breaks.append(duration)
                last_kick = position
            entry = (taught, left[taught].astype(int), np.searchsorted(AGE_BINS, presence_age[taught]), phase)
            presence_trials[entry] += 1
            switches[entry] += left[taught] != right[taught]
            sounding = taught[(left & right)[taught]]
            rewrite = (sounding, 1, np.searchsorted(AGE_BINS, pattern_age[sounding]), phase)
            pattern_trials[rewrite] += 1
            edits[rewrite] += changed[sounding]
            presence_age, pattern_age = advance(presence_age, pattern_age, before, after, changed)
            before = after
    if np.any(presence_trials.sum(axis=(1, 2, 3)) == 0):
        raise ValueError("Groove training needs examples for every instrument")

    def smooth(events, trials):
        # Back off from sparse phase/age combinations to the same voice and audibility.
        voice = (events.sum(axis=(2, 3), keepdims=True) + 1) / (trials.sum(axis=(2, 3), keepdims=True) + 2)
        age = (events.sum(axis=3, keepdims=True) + 8 * voice) / (trials.sum(axis=3, keepdims=True) + 8)
        return ((events + 8 * age) / (trials + 8)).tolist()

    break_limit = int(np.ceil(np.percentile(clean_breaks or breaks, 95) / PHASES) * PHASES) if breaks else None
    return {"version": VERSION, "switch": smooth(switches, presence_trials),
            "rewrite": smooth(edits, pattern_trials),
            "kick_entries": [list((entries or heard_kicks).most_common(1)[0][0]) if heard_kicks else None
                             for entries in kick_entries],
            "kick_break_limit": break_limit,
            "examples": presence_trials.sum(axis=(1, 2, 3)).tolist(),
            "switches": switches.sum(axis=(1, 2, 3)).tolist(),
            "edits": edits.sum(axis=(1, 2, 3)).tolist()}


class Groove:
    def __init__(self, fitted, seed):
        shape = (common.L, 2, len(AGE_BINS) + 1, PHASES)
        if fitted["version"] != VERSION:
            raise ValueError("Unsupported groove training data")
        self.switch = np.asarray(fitted["switch"], dtype=float)
        self.rewrite = np.asarray(fitted["rewrite"], dtype=float)
        if self.switch.shape != shape or self.rewrite.shape != shape:
            raise ValueError("Groove training data has the wrong shape")
        self.rng = np.random.default_rng(seed)
        self.kick_entries = fitted.get("kick_entries") or [None] * PHASES
        self.kick_break_limit = fitted.get("kick_break_limit")
        self.kick_return_phase = int(np.argmax(self.switch[0, 0, -1]))
        self.presence_age = np.ones(common.L, dtype=int)
        self.pattern_age = np.ones(common.L, dtype=int)
        self.proposal_rate = np.full(common.L, .5)
        self.last_sounding = [None] * common.L

    def prime(self, bars):
        if len(bars):
            self.hear(bars[0], bars[0])
            self.presence_age.fill(1)
            self.pattern_age.fill(1)
        for current, played in zip(bars[:-1], bars[1:]):
            self.hear(current, played)

    def hear(self, current, played):
        before, after = patterns(current), patterns(played)
        self.presence_age, self.pattern_age = advance(
            self.presence_age, self.pattern_age, before, after, large_edits(before, after))
        played = np.asarray(played).copy()
        for layer in np.flatnonzero(after.any(axis=1)):
            self.last_sounding[layer] = played

    def play(self, current, proposed, position):
        played = self._propose(current, proposed, position)
        self.hear(current, played)
        return played

    def _propose(self, current, proposed, position):
        current = np.asarray(current)
        before, after = patterns(current), patterns(proposed)
        audible, wanted = before.any(axis=1), after.any(axis=1)
        phase = position % PHASES
        layers = np.arange(common.L)
        switches = self.switch[layers, audible.astype(int), np.searchsorted(AGE_BINS, self.presence_age), phase]
        playing = audible ^ (self.rng.random(common.L) < switches)
        if (self.kick_break_limit and not audible[0] and self.presence_age[0] >= self.kick_break_limit
                and phase == self.kick_return_phase):
            playing[0] = True
        changes = large_edits(before, after) & audible & wanted
        self.proposal_rate += (changes - self.proposal_rate) / 16
        rate = self.rewrite[layers, 1, np.searchsorted(AGE_BINS, self.pattern_age), phase]
        replace = self.rng.random(common.L) * np.maximum(self.proposal_rate, .125) < rate

        played = proposed.copy()
        beat = common.L + common.MODS
        kick = slice(beat, beat + common.STEPS // len(common.DRUM_ROWS))
        if not playing[0]:
            played[kick] = 0
        elif not wanted[0]:
            source = current if audible[0] else self.last_sounding[0]
            if source is not None:
                common.set_sound(played, 0, int(common.sounds(source)[0]))
                played[kick] = source[kick]
            elif self.kick_entries[phase] is not None:
                kit, *pattern = self.kick_entries[phase]
                common.set_sound(played, 0, kit)
                played[kick] = pattern
        elif audible[0] and not replace[0]:
            common.set_sound(played, 0, int(common.sounds(current)[0]))
            if changes[0]:
                played[kick] = current[kick]

        start = beat + common.STEPS
        line_width = common.SYN // len(common.SYNTH_LAYERS)
        patch_width = timbre.width(common.PATCH_BOOK)
        for layer in range(len(common.SYNTH_LAYERS)):
            voice = layer + 1
            if not playing[voice]:
                common.set_sound(played, voice, 0)
                continue
            line_start = start + layer * line_width
            line = slice(line_start, line_start + line_width)
            patch = start + common.SYN + layer * patch_width
            if not wanted[voice]:
                source = current if audible[voice] else self.last_sounding[voice]
                if source is not None:
                    played[line] = source[line]
                    common.set_sound(played, voice, int(common.sounds(source)[voice]))
                    if not audible[voice]:
                        played[patch:patch + patch_width] = source[patch:patch + patch_width]
            if audible[voice] and (not wanted[voice] or not replace[voice]):
                if changes[voice]:
                    played[line] = current[line]
                slots = patch_identity(layer)
                played[slots] = current[slots]
        return played


class PhraseVariation(Groove):
    """Keep four-bar motifs, with small variations and a late answering figure."""

    def __init__(self, fitted, seed):
        super().__init__(fitted, seed)
        self.anchor = None

    def _propose(self, current, proposed, position):
        played = super()._propose(current, proposed, position)
        phase = position % 4
        if phase == 0 or self.anchor is None:
            self.anchor = played.copy()
            return played
        beat = common.L + common.MODS
        if played[0] and played[0] == self.anchor[0]:
            self._answer_drums(played, beat, phase)
        if self.anchor[2] and (played[2] or phase == 1):
            self._answer_lead(played, beat + common.STEPS + 16, phase)
        return played

    def _answer_drums(self, played, beat, phase):
        # Keep the kick as written. A few changes in the upper voices make a fill
        # legible against the accent pattern set out in the first bar.
        for voice in (1, 2):
            slots = slice(beat + 16 * voice, beat + 16 * (voice + 1))
            motif, answer = self.anchor[slots], played[slots].copy()
            if not motif.any() or not answer.any():
                continue
            row = played[slots]
            row[:] = motif
            start = 4 if phase == 1 else 8
            end = 8 if phase == 1 else 16
            changes = np.flatnonzero(motif[start:end] != answer[start:end]) + start
            # Give syncopated attacks and anticipations first choice of the edit budget.
            steps = sorted(changes, key=lambda step: (step % 4 == 0, -int(answer[step])))
            for step in steps[:phase]:
                row[step] = answer[step]

    def _answer_lead(self, played, start, phase):
        line = slice(start, start + 16)
        motif = self.anchor[line]
        answer = played[line].copy()
        if not np.any(motif >= 2) or (phase > 1 and not np.any(answer >= 2)):
            return
        if not played[2]:
            common.set_sound(played, 2, int(self.anchor[2]))
            start_patch = common.L + common.MODS + common.STEPS + common.SYN
            width = timbre.width(common.PATCH_BOOK)
            patch = slice(start_patch + width, start_patch + 2 * width)
            played[patch] = self.anchor[patch]
        row = played[line]
        row[:] = motif
        if phase > 1:
            palette = motif
            if phase == 3 and self.anchor[1]:
                palette = np.concatenate((motif, self.anchor[start - 16:start]))
            pitches = np.unique(palette[palette >= 2])
            late = 8 if phase == 3 else 12
            accents = (answer[late:] >= 2) & (answer[late:] != motif[late:])
            for step in np.flatnonzero(accents)[:2 if phase == 3 else 1] + late:
                distance = np.abs(pitches - answer[step])
                row[step] = pitches[np.argmin(np.minimum(distance, 12 - distance))]
        slots = patch_identity(1)
        played[slots] = self.anchor[slots]


def player_controller(card, seed):
    if card.get("voice_dwell"):
        return VoiceDwell(card["voice_dwell"], card.get("dwell_lets_go"))
    if card.get("groove"):
        if card.get("phrase_variation") and (common.ROWS or common.STEPS != 48
                                             or (common.SYNTH or {}).get("kind") != "steps"):
            raise ValueError("Phrase variations need step-level drum and synth voices")
        cls = PhraseVariation if card.get("phrase_variation") else Groove
        return cls(card["groove"], seed)
    return None
