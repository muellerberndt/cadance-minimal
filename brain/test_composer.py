"""Regression checks for the composer checkpoint and both sides of its audio interface."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import common
import composition
import groove
import gate
import quality
import timbre
from recall_trace import RecallTrace

ROOT = Path(__file__).resolve().parents[1]
# A stubbed prediction_error's answer for the trainer's rollout readings (prediction_error(None, pieces, answers=...)):
# the free-running continuations are scored on the brain's own bars, so they take no value from a test's queue.
ROLLOUT = {"balanced": .3, "arrangement": .3, "stay": {"balanced": .1}, "change": {"balanced": .5}}
# The whole-run tests stub the long free-play density reading (density_rollout: 64 bars of sampled and greedy play at
# every full evaluation, 32 at every light reading) as None, the trainer's own value when no held-out passage fits.


class ComposerTests(unittest.TestCase):
    def setUp(self):
        self.pool = json.loads((ROOT / "pool/pool_r6.json").read_text())
        self.book = timbre.teachers(self.pool)
        common.ONOFF = False
        common.set_content(None)
        rows = {voice: [[0]*16, [1, 0, 0, 0]*4] for voice in common.DRUM_ROWS}
        lines = {layer: [[0]*16, [2, 1, 0, 0]*4, [4, 0, 0, 0]*4] for layer in common.SYNTH_LAYERS}
        common.configure(97, rows=rows, synth=lines, clock=True, patch_book=self.book)
        self.state = np.zeros(common.SLOTS, dtype=int)
        self.state[1] = self.state[15] = 1
        self.start = common.L + common.MODS + common.STEPS + common.SYN
        self.state[self.start:] = np.concatenate([layer[1] for layer in self.book["layers"]])

    def test_controls_are_causal_through_silence(self):
        token = np.array([[0, 0, 0, 0], [0, 6, 0, 0], [0, 0, 0, 0]])
        octave = np.full((3, 3), -1)
        octave[1, 0] = 2
        controls = timbre.track_controls(token, octave, self.book)
        np.testing.assert_array_equal(controls[0, :len(timbre.SIZES)], self.book["layers"][0][0])
        np.testing.assert_array_equal(controls[1], controls[2])
        self.assertEqual(controls[1, 1], 3)
        self.assertEqual(common.SIZES[1:4], [2, 2, 2])

    def test_render_responds_to_motor_without_pool_synths(self):
        # Remove every synth preset. The generated patch and pattern are sufficient.
        for layer in self.pool["layers"][1:]:
            layer["tokens"] = [None]
        self.state[self.start:self.start + 3] = (0, 3, 5)
        for k, v in dict(level=4, sustain=4, cutoff=6, wave=0, fm_amt=0, attack=0, ftype=0).items():
            self.state[self.start + 3 + timbre.NAMES.index(k)] = v
        def score():
            return common.score_bar(self.state, self.pool, [None]*3, [None]*4)
        first = score()
        self.state[self.start + 3 + timbre.NAMES.index("wave")] = 6
        second = score()
        self.assertEqual(first["patterns"], second["patterns"])
        self.assertGreater(common.sound_identities(self.state)[1], 1)
        js = """
          import {renderScore, stats} from './engine/render.mjs';
          import fs from 'node:fs';
          const [pool, a, b] = JSON.parse(fs.readFileSync(0, 'utf8'));
          const play = bar => renderScore({bars:[bar]}, pool, {sr:22050, layer:1, tail:0, dry:true});
          const A = play(a), B = play(b);
          if (!(stats(A.L,A.R).rms > .001 && stats(B.L,B.R).rms > .001)) process.exit(1);
          let delta=0;
          for(let i=0;i<A.L.length;i++) delta+=Math.abs(A.L[i]-B.L[i]);
          if (!(delta/A.L.length>.001)) process.exit(2);
        """
        subprocess.run(["node", "--input-type=module", "-e", js], cwd=ROOT,
                       input=json.dumps([self.pool, first, second]), text=True, check=True)
        self.state[1] = 0
        self.assertIsNone(score()["patches"][1])

    def test_recall_input_ignores_silent_synth_controls(self):
        self.state[1] = 0
        other = self.state.copy()
        other[self.start + 3] = (other[self.start + 3] + 1) % 7
        mood = common.mood_rows([[0.2, 0.6]])
        np.testing.assert_array_equal(common.observe(self.state, mood), common.observe(other, mood))
        self.state[1] = other[1] = 1
        self.assertFalse(np.array_equal(common.observe(self.state, mood), common.observe(other, mood)))

    def test_only_sounding_instruments_are_taught(self):
        lead = self.state.copy()
        lead[:4] = [1, 0, 1, 0]
        original = np.stack([self.state, lead])
        labels = common.audible_targets(original)
        width = timbre.width(self.book)
        expected = np.concatenate([[True] * 4, [False] * 2, [True] * 2, [False] * 4,
                                   [False] * 3, [True, False, False], [True] * width, [False] * (2 * width)])
        np.testing.assert_array_equal(labels[0] >= 0, expected)
        self.assertTrue((labels[1, 12:15] >= 0).all())
        self.assertTrue((labels[1, self.start:self.start + width] == -1).all())
        self.assertTrue((labels[1, self.start + width:self.start + 2 * width] >= 0).all())
        np.testing.assert_array_equal(labels[labels >= 0], original[labels >= 0])
        self.assertTrue((original >= 0).all())

    def test_sampler_parameters_are_taught_only_in_sampler_mode(self):
        book = dict(self.book, version=timbre.SAMPLE_VERSION, samples=["hit"])
        common.configure(97, rows=common.ROWS, synth=common.SYNTH, patch_book=book)
        states = np.zeros((2, common.SLOTS), dtype=int)
        states[:, 1] = 1
        states[1, self.start] = timbre.VOICES.index("sampler")
        extras = slice(self.start + len(timbre.SIZES), self.start + timbre.width(book))
        states[:, extras] = [0, 5, 1]
        np.testing.assert_array_equal(common.audible_targets(states)[:, extras], [[-1, -1, -1], [0, 5, 1]])

    def test_recall_trace_points_to_the_bar_actually_read(self):
        silent = np.zeros_like(self.state)
        kick = silent.copy()
        kick[0] = kick[12] = 1
        trace = RecallTrace(keep_reads=True)
        trace.prompt("groove", 100, [silent, kick, silent])
        trace.read(SimpleNamespace(strength=False, attended_heads=(np.array([[.1, .8, .1]]),)), self.state)
        head = trace.summary()[0]
        self.assertAlmostEqual(head["mean_lag"], 2.0)
        self.assertAlmostEqual(head["kick_entrance"], .8)
        self.assertAlmostEqual(head["kick_entrance_uniform"], 1 / 3)
        self.assertEqual(trace.reads[0]["retrieved"][0]["bar"], 101)
        self.assertAlmostEqual(trace.max_slot_share, .8)

    def test_continuity_excludes_silence_and_measures_rewritten_lines(self):
        sounding = self.state.copy()
        sounding[0] = sounding[12] = 1
        replacement = sounding.copy()
        replacement[15] = 2
        replacement[self.start + 3 + timbre.NAMES.index("wave")] = 6
        silent = replacement.copy()
        silent[1] = 0
        result = composition.bar_continuity(np.stack([sounding, sounding, replacement, silent]))
        self.assertEqual(result["layers"]["beat"]["note_edit_fraction"], 0)
        self.assertEqual(result["layers"]["bass"]["adjacent_sounding"], 2)
        self.assertEqual(result["layers"]["bass"]["rewrite_fraction"], .5)
        self.assertEqual(result["layers"]["bass"]["instrument_switch"], .5)
        self.assertIsNone(result["layers"]["lead"]["note_edit_fraction"])

    def test_pattern_hold_leaves_arrangement_patches_and_silent_layers_untouched(self):
        from probe_recall import hold_lines

        start = common.L + common.MODS
        activity = np.zeros(common.ACTIONS)
        # In this fixture only the bass is active.
        bass = start + common.STEPS
        with patch.object(common, "LINE_SCALE", None), patch.object(common, "sample_slots", return_value=self.state) as sample:
            hold_lines(activity, self.state, .08, None)
        received = sample.call_args.args[0]
        expected = np.zeros_like(activity)
        expected[common.OFFSETS[bass] + self.state[bass]] = .08 * np.log(2)
        np.testing.assert_array_equal(received, expected)
        self.assertFalse(activity.any())

    def test_phrase_pacing_keeps_the_pulse_and_allows_a_layer_to_change_at_a_boundary(self):
        current = self.state.copy()
        beat = common.L + common.MODS
        line = beat + common.STEPS
        current[:common.L] = 1
        current[beat] = 1
        current[line:line + common.SYN] = 1
        proposed = current.copy()
        proposed[0] = 2
        proposed[beat] = 0
        proposed[beat + 2] = 1
        proposed[line:line + common.SYN] = 2
        proposed[self.start + 1] = 2

        offbeat = common.pace_phrase(current, proposed, 7)
        self.assertEqual(offbeat[0], current[0])
        self.assertEqual(offbeat[beat], current[beat])
        self.assertEqual(offbeat[beat + 2], proposed[beat + 2])
        self.assertEqual(offbeat[self.start + 1], proposed[self.start + 1])
        np.testing.assert_array_equal(offbeat[line:line + common.SYN], [1, 1, 1])

        boundary = common.pace_phrase(current, proposed, 8)
        self.assertEqual(boundary[0], proposed[0])
        self.assertEqual(boundary[beat], proposed[beat])
        np.testing.assert_array_equal(boundary[line:line + common.SYN], [1, 2, 1])

    def test_voice_dwell_holds_large_edits_and_allows_timbre_motion(self):
        current = self.state.copy()
        beat = common.L + common.MODS
        current[0] = current[beat] = 1
        proposed = current.copy()
        proposed[0] = 2
        proposed[beat] = 0
        proposed[beat + 2] = 1
        proposed[15] = 2
        controls = self.start + 3
        cutoff = controls + timbre.NAMES.index("cutoff")
        proposed[cutoff] = (current[cutoff] + 1) % common.SIZES[cutoff]
        player = groove.VoiceDwell(4)
        for bar in range(1, 4):
            current = player.play(current, proposed, bar)
            self.assertEqual(current[0], 1)
            self.assertEqual(current[beat], 1)
            self.assertEqual(current[beat + 2], 1)
            self.assertEqual(current[15], 1)
            self.assertEqual(current[cutoff], proposed[cutoff])
        np.testing.assert_array_equal(player.play(current, proposed, 4), proposed)

    def test_groove_training_uses_clean_synth_labels_and_excludes_held_out_tracks(self):
        changed = self.state.copy()
        changed[15] = 2
        clean = {"group": "album", "held_out": False, "bars": [self.state, self.state, changed]}
        mix = dict(clean, group="mix", bars=[changed, self.state])
        holdout = dict(mix, held_out=True)
        result = groove.fit([clean, mix, holdout], {"album"})
        self.assertEqual(result["examples"], [3, 2, 2, 2])
        self.assertEqual(result["edits"], [0, 1, 0, 0])
        self.assertEqual(result, groove.fit([clean, mix], {"album"}))

    def test_groove_keeps_sound_identity_and_still_allows_fills_and_patch_motion(self):
        current = self.state.copy()
        beat = common.L + common.MODS
        current[0] = current[beat] = 1
        proposed = current.copy()
        proposed[0] = 2
        proposed[beat] = 0
        proposed[beat + 2] = 1
        proposed[15] = 2
        wave = self.start + 3 + timbre.NAMES.index("wave")
        cutoff = self.start + 3 + timbre.NAMES.index("cutoff")
        proposed[wave] = (current[wave] + 1) % common.SIZES[wave]
        proposed[cutoff] = (current[cutoff] + 1) % common.SIZES[cutoff]
        fitted = groove.fit([{"group": "album", "held_out": False, "bars": [current, proposed]}], {"album"})
        controller = groove.Groove(fitted, 0)
        controller.switch.fill(0)
        controller.rewrite.fill(0)
        result = controller.play(current.tolist(), proposed, 4)
        self.assertEqual(result[0], current[0])
        self.assertEqual(result[beat], current[beat])
        self.assertEqual(result[beat + 2], proposed[beat + 2])
        self.assertEqual(result[15], current[15])
        self.assertEqual(result[wave], current[wave])
        self.assertEqual(result[cutoff], proposed[cutoff])
        np.testing.assert_array_equal(controller.presence_age, [2] * common.L)
        np.testing.assert_array_equal(controller.pattern_age, [2, 2, 1, 1])
        controller.switch[0, 1].fill(1)
        broken = controller.play(current, proposed, 5)
        self.assertEqual(broken[beat], 0)
        controller.switch[0, 0].fill(1)
        returned = controller.play(broken, proposed, 6)
        self.assertEqual(returned[0], current[0])
        self.assertEqual(returned[beat], current[beat])

    def test_groove_learns_first_kick_return_without_rewriting_the_hat(self):
        beat = common.L + common.MODS
        rest = self.state.copy()
        rest[0] = rest[beat+2] = 1
        kick = rest.copy()
        kick[0], kick[beat] = 2, 1
        fitted = groove.fit([{"group": "album", "held_out": False, "bars": [rest, kick, rest]}], {"album"})
        controller = groove.Groove(fitted, 0)
        controller.prime([rest])
        controller.switch.fill(0)
        controller.switch[0, 0].fill(1)
        answer = controller.play(rest, rest, 1)
        self.assertEqual(answer[0], 2)
        self.assertEqual(answer[beat], 1)
        self.assertEqual(answer[beat+2], rest[beat+2])

        fitted["kick_break_limit"] = 8
        controller = groove.Groove(fitted, 0)
        controller.switch.fill(0)
        controller.prime([rest] * 8)
        answer = controller.play(rest, rest, controller.kick_return_phase)
        self.assertEqual(answer[beat], 1)
        self.assertEqual(answer[beat+2], rest[beat+2])

    def test_phrase_variation_preserves_the_motif_and_uses_the_brains_answer(self):
        common.configure(97, steps=True, synth={"kind": "steps"}, clock=True, patch_book=self.book)
        beat = common.L + common.MODS
        hats, lead = beat + 32, beat + common.STEPS + 16
        anchor = np.zeros(common.SLOTS, dtype=int)
        anchor[:3] = 1
        anchor[[beat, beat + 16 + 4, hats + 2, hats + 14]] = 2
        anchor[lead - 16] = 9
        anchor[[lead, lead + 6]] = [2, 5]
        proposed = anchor.copy()
        proposed[hats + 2] = 0
        proposed[hats + 15] = 2
        proposed[lead] = 11
        proposed[lead + 13] = 9
        training = {"group": "album", "held_out": False, "bars": [anchor, proposed]}
        controller = groove.PhraseVariation(groove.fit([training], {"album"}), 0)
        controller.switch.fill(0)
        controller.rewrite.fill(1)
        first = controller.play(anchor, anchor, 0)
        middle = controller.play(first, proposed, 1)
        end = controller.play(middle, proposed, 3)
        np.testing.assert_array_equal(middle[lead:lead + 16], anchor[lead:lead + 16])
        np.testing.assert_array_equal(end[lead:lead + 8], anchor[lead:lead + 8])
        np.testing.assert_array_equal(end[beat:beat + 16], proposed[beat:beat + 16])
        self.assertEqual(end[hats + 2], 2)
        self.assertEqual(end[hats + 15], 2)
        self.assertEqual(end[lead + 13], 9)
        np.testing.assert_array_equal(controller.last_sounding[2], end)

    def test_reference_edits_keep_own_line_only_while_the_reference_holds(self):
        previous = np.tile(self.state, (5, 1))
        target = previous.copy()
        played = previous.copy()
        played[:4, 15] = 2
        played[4, 15] = 0
        target[1, 15] = 2                  # a real edit takes priority
        played[1, 15] = 1
        target[2, 1] = 0                   # the reference's exit takes priority
        played[:, self.start] = 1
        taught = common.reference_edits(previous, target, played, np.array([1, 1, 1, 0, 1], dtype=bool))
        np.testing.assert_array_equal(taught[:, 15], [2, 2, 1, 1, 1])
        np.testing.assert_array_equal(taught[:, :common.L], target[:, :common.L])
        np.testing.assert_array_equal(taught[:, self.start:], target[:, self.start:])
        np.testing.assert_array_equal(target[:, 15], [1, 2, 1, 1, 1])

    def test_reprise_uses_the_played_motif_after_a_departure(self):
        bars = np.tile(self.state, (1, 17, 1))
        bars[0, 1:8, 1] = 0
        ages = common.motif_return_ages(bars)
        self.assertEqual(ages[0, 8, 0], 8)
        self.assertFalse(ages[0, 16].any())
        history = list(bars[0, :8, None].copy())
        history[0][0, 15] = 2
        result = common.heard_reprises(bars[:, 8], ages[:, 8], history)
        self.assertEqual(result[0, 15], 2)
        self.assertEqual(bars[0, 8, 15], 1)
        np.testing.assert_array_equal(result[:, self.start:], bars[:, 8, self.start:])
        history[0][0, 1] = 0
        np.testing.assert_array_equal(common.heard_reprises(bars[:, 8], ages[:, 8], history), bars[:, 8])

    def test_arrangement_focus_teaches_stable_kick_and_bass_changes(self):
        bars = np.tile(self.state, (1, 7, 1))
        bars[0, :, 0] = 1
        kick_slot = common.L + common.MODS
        bars[0, 2:4, kick_slot] = 1
        bars[0, :2, 1] = 0
        bars[0, 4:6, 1] = 0
        changes = common.arrangement_transitions(bars, np.array([False, True]))
        self.assertEqual(list(np.flatnonzero(changes[0, :, 1])), [2, 4])
        self.assertEqual(list(np.flatnonzero(changes[0, :, common.L])), [2, 4])
        target = bars[:, 2]
        focused = common.focus_arrangement(target, changes[:, 2])
        np.testing.assert_array_equal(np.flatnonzero(focused[0] >= 0), [0, 1, kick_slot])
        self.assertEqual(focused[0, kick_slot], 1)
        self.assertEqual(focused[0, 1], 1)
        self.assertTrue((target >= 0).all())
        np.testing.assert_array_equal(common.focus_arrangement(bars[:, 3], changes[:, 3]), bars[:, 3])

    def test_prompt_only_keeps_recorded_bars_and_live_query_context(self):
        track = {"id": "held-out", "held_out": True, "bars": np.tile(self.state, (25, 1)),
                 "mood": common.mood_rows([[.2, .6]] * 25)}
        state = {"keys": (), "values": (), "nexts": (), "held": (), "past": 0}
        visited = []
        recall = Mock(snapshot=lambda: state.copy(), restore=lambda bank: state.update(bank))

        def act(*args, **kwargs):
            visited.append((state["keys"], state["past"]))
            for key in ("keys", "values", "nexts", "held"):
                state[key] += (state["past"],)
            state["past"] += 1

        brain = Mock(act=act, recall=recall)
        with patch.object(common, "motor_activity", return_value=[None]):
            composition.compare(lambda: brain, [track], .08, 17, prompt_only_recall=True,
                                sampler=lambda a, previous, t, rng: previous)
        self.assertEqual(visited[7:], [(tuple(range(7)), i) for i in range(7, 24)])

    def test_composition_renders_every_prompt(self):
        def track(i):
            return {"id": str(i), "held_out": True, "bars": np.tile(self.state, (25, 1)),
                    "mood": common.mood_rows([[.2, .6]] * 25)}

        brain = Mock()
        audition = Mock(side_effect=lambda g, r, p, i: {"score": i / 2})
        with patch.object(common, "motor_activity", return_value=[None]):
            result = composition.compare(lambda: brain, [track(i) for i in range(3)], .08, 17,
                                         audition=audition, sampler=lambda a, previous, t, rng: previous)
        self.assertEqual(audition.call_count, 3)
        self.assertEqual(result["audio"]["score"], .5)
        self.assertEqual([run["audio"]["score"] for run in result["runs"]], [0, .5, 1])

    def test_learned_recall_preset_keeps_four_heads_and_uses_code_only_keys(self):
        brain, settings = common.compose("composer-learned-recall", backend="cpu")
        self.assertEqual(settings["modules"], [256, 128, 96, 64])
        self.assertEqual(brain.recall.heads, 4)
        self.assertEqual(brain.recall.key_width, 512)
        self.assertIsNone(brain.recall.tokens)
        self.assertAlmostEqual(brain.recall.temperature, 2.0)

    def test_phrase_event_score_rewards_returns_on_phase_and_skips_absent_events(self):
        silent = np.zeros_like(self.state)
        bar = self.state.copy()
        bar[0] = bar[12] = 1
        states = np.tile(silent, (24, 1))
        states[[0, 8]] = bar
        delayed = states.copy()
        delayed[8] = silent
        delayed[9] = bar
        hold = np.concatenate([states[:8], np.repeat(states[7:8], 16, axis=0)])
        target = composition.phrase_events(states, offset=3, prompt_bars=8)
        self.assertEqual(target["bass_motif_return"][3], 1)
        self.assertEqual(target["kick_return"][3], 1)
        self.assertEqual(target["lead_entrance"].sum(), 0)
        frozen = composition.phrase_events(hold, offset=3, prompt_bars=8)
        assert composition.event_agreement(target, target, frozen)["score"] == 1.0
        assert composition.event_agreement(target, target, frozen)["hold"] == 0.0
        off_phase = composition.phrase_events(delayed, offset=3, prompt_bars=8)
        assert composition.event_agreement(target, off_phase, frozen)["score"] == 0.0

    def test_step_composer_keeps_own_details_and_teaches_reference_edits(self):
        common.configure(97, steps=True, synth={"kind": "steps"}, clock=True, patch_book=self.book)
        before = np.zeros((1, common.SLOTS), dtype=int)
        before[0, :2] = 1
        beat = common.L + common.MODS
        line = beat + common.STEPS
        before[0, [beat, beat + 34, line + 3, line + 4]] = [2, 1, 7, 1]
        after = before.copy()
        after[0, [beat + 38, line + 11]] = [2, 9]
        heard = before.copy()
        heard[0, [beat + 34, line + 3]] = [2, 5]
        edited = common.reference_edits(before, after, heard, np.array([True]))[0]
        self.assertTrue(common.audible_kick(edited))
        self.assertEqual(common.decode_beat(edited)[2].tolist(), [0, 0, 2, 0, 0, 0, 2] + [0]*9)
        self.assertEqual(common.decode_synth(edited)[0].tolist(), [0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 9] + [0]*4)
        labels = common.audible_targets(edited)[0]
        np.testing.assert_array_equal(labels[line:line+16], edited[line:line+16])
        self.assertTrue((labels[line+16:line+48] == -1).all())
        sequence = np.repeat(before[None, :, :], 9, axis=1)
        sequence[0, 1:8, 1] = 0
        self.assertEqual(common.motif_return_ages(sequence)[0, 8, 0], 8)

    def test_development_records_a_release_into_a_recurring_new_motif(self):
        common.configure(97, steps=True, synth={"kind": "steps"}, clock=True, patch_book=self.book)
        states = np.zeros((16, common.SLOTS), dtype=int)
        states[:, :2] = 1
        beat = common.L + common.MODS
        line = beat + common.STEPS
        states[:, beat:beat+16:4] = 2
        states[4:8, beat+32:beat+48:2] = 1
        states[:12, line+2:line+16:4] = 6
        states[12:, line+2:line+16:4] = 9
        states[12:, line+14] = [9, 10, 11, 12]
        expected = composition.development_events(states, 0, 4)
        self.assertEqual({key: np.flatnonzero(values).tolist() for key, values in expected.items()},
                         {"buildup": [4], "release": [8], "new_motif_after_release": [12]})
        hold = composition.development_events(np.repeat(states[:1], 16, axis=0), 0, 4)
        self.assertEqual(composition.event_agreement(expected, expected, hold)["hold"], 0)
        self.assertFalse(composition.development_points(states, 0, synth=False)["new_motif_after_release"].any())

    def test_prompt_picker_reserves_the_held_out_handoff(self):
        options = [(0, 0, 64, {"groove", "motif handoff"}, True), (1, 0, 64, {"groove"}, True),
                   (2, 0, 64, {"synth entry"}, True), (3, 0, 64, {"kick return"}, True)]
        picks = composition.coverage_prompts(options, np.random.default_rng(22), handoffs=True)
        self.assertEqual(picks[-1], (0, 0, 64, "motif handoff"))

    def test_step_detail_metrics_count_pitch_errors_and_missed_accents(self):
        common.configure(97, steps=True, synth={"kind": "steps"}, clock=True, patch_book=self.book)
        bars = np.zeros((2, common.SLOTS), dtype=int)
        bars[:, :2] = 1
        beat = common.L + common.MODS
        line = beat + common.STEPS
        bars[:, beat] = [1, 2]
        bars[:, line+3] = [3, 7]
        bars[:, line+4] = 1
        result = gate.prediction_error(None, [{"bars": bars, "mood": None}])
        detail = result["detail"]
        # One wrong drum step and one wrong bass note, each counted for the instrument that plays it.
        parts = result["instruments"]
        self.assertAlmostEqual(parts["beat"], 1 / (1 + common.STEPS))
        self.assertAlmostEqual(parts["bass"], 1 / (17 + timbre.width(self.book)))
        self.assertEqual((parts["lead"], parts["air"]), (0, 0))
        self.assertEqual(detail["drum_hits"]["f1"], 1)
        self.assertEqual(detail["drum_accents"]["f1"], 0)
        self.assertEqual(detail["synth_notes"]["f1"], 0)
        self.assertEqual(detail["synth_ties"]["f1"], 1)

    def test_split_steps_teach_pitch_only_on_attacks(self):
        common.configure(97, steps=True, synth={"kind": "splitsteps"}, clock=True, patch_book=self.book)
        state = np.zeros(common.SLOTS, dtype=int)
        state[1] = 1
        notes = np.array([0, 8, 1, 13] + [0]*12)
        start = common.L + common.MODS + common.STEPS
        state[start:start+32] = common.encode_split_notes(notes)
        np.testing.assert_array_equal(common.decode_synth(state)[0], notes)
        label = common.audible_targets(state)[0]
        self.assertEqual(label[start:start+8].tolist(), [0, -1, 2, 6, 1, -1, 2, 11])
        empty = state.copy()
        empty[start+1] = 7
        mood = common.mood_rows([[.5,.5]])
        np.testing.assert_array_equal(common.observe(empty, mood), common.observe(state, mood))
        edited = state.copy()
        edited[start+6:start+8] = [2, 9]
        played = empty.copy()
        played[start+3] = 2
        replacement = common.reference_edits(state[None], edited[None], played[None], [True])[0]
        self.assertEqual(common.decode_synth(replacement)[0][:4].tolist(), [0, 4, 1, 11])

    def test_focused_lessons_keep_accents_and_pitches_with_matching_contrasts(self):
        common.configure(97, steps=True, synth={"kind": "steps"}, clock=True, patch_book=self.book)
        bars = np.zeros((3, common.SLOTS), dtype=int)
        bars[:, :2] = 1
        beat = common.L + common.MODS
        line = beat + common.STEPS
        bars[:2, [beat+3, beat+6, line+4]] = [2, 2, 9]
        target = common.audible_targets(bars)
        actual = common.focus_detail(target, [True, False, True], np.random.default_rng(4))
        self.assertTrue((actual[0, :beat] == -1).all())
        self.assertTrue((actual[0, line+16:] == -1).all())
        self.assertEqual(actual[0, [beat+3, beat+6, line+4]].tolist(), [2, 2, 9])
        self.assertEqual(np.count_nonzero(actual[0] == 0), 3)
        np.testing.assert_array_equal(actual[1:], target[1:])

    def test_prompt_picker_reserves_rare_kick_return(self):
        options = [
            (0, 0, 80, {"groove"}, True),
            (1, 0, 80, {"synth entry"}, True),
            (1, 20, 80, {"kick return", "synth entry"}, True),
            (0, 100, 80, {"groove"}, True),
        ]
        picks = composition.coverage_prompts(options, np.random.default_rng(131))
        self.assertEqual([p[3] for p in picks], ["groove", "synth entry", "kick return"])
        self.assertEqual(picks[2][:2], (1, 20))

    def test_audio_render_cli_runs_through_symlink_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="Cadance render ") as directory:
            work = Path(directory)
            render = work / "render.mjs"
            render.symlink_to(ROOT / "engine/render.mjs")
            score = work / "score.json"
            score.write_text(json.dumps({"pool": self.pool, "bars": [[1, 0, 0, 0]]}))
            wav = work / "stem.wav"
            subprocess.run(["node", str(render), str(score), str(wav), "--layer", "0", "--tail", "0"],
                           check=True, capture_output=True)
            self.assertGreater(wav.stat().st_size, 44)

    def test_replay_scores_the_final_requested_bar(self):
        bars = np.tile(self.state, (6, 1))
        bars[-1, 0] = 1
        track = {"id": "phrase", "title": "phrase", "held_out": True, "bars": bars,
                 "mood": common.mood_rows([[.5, .5]] * len(bars))}
        brain = Mock()
        brain.act.side_effect = [bars[[i]] for i in (1, 2, 3, 4, 5)]
        result = gate.replay(brain, [track], bars=2)
        self.assertEqual(result["phrase"]["beat"], 1)
        self.assertEqual(result["phrase"]["hold"], .875)
        self.assertEqual(brain.act.call_count, 5)

    def test_sampler_score_counts_a_false_trigger_during_a_reference_rest(self):
        book = dict(self.book, version=timbre.SAMPLE_VERSION, samples=["hit"])
        common.configure(97, rows=common.ROWS, synth=common.SYNTH, patch_book=book)
        silent = np.zeros(common.SLOTS, dtype=int)
        played = silent.copy()
        played[1] = 1
        played[self.start] = timbre.VOICES.index("sampler")
        track = {"bars": np.stack([silent, silent]), "mood": common.mood_rows([[.5, .5]] * 2)}
        with patch.object(gate, "_act", return_value=played[None]):
            result = gate.prediction_error(Mock(), [track])
        self.assertAlmostEqual(result["sampler"]["false_trigger_rate"], 1 / 3)

    def test_training_keeps_weights_if_quality_falls_or_evaluation_fails(self):
        import train

        scores = iter([.8, .7, .6])
        errors = iter((.9, .9, .8, .8, .7, .7, .75, .75))

        def compare(*args, **kwargs):
            score = next(scores)
            return dict(selection_score=score, score=score, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            if constant is not None:
                return {"balanced": .95}          # always the commonest bar
            return {"balanced": next(errors) if brain is not None else .2}

        pool = ROOT / "pool/pool_underground.json"
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "train.py", "--name", "audit", "--out", directory,
                "--streams", str(ROOT / "corpus/streams_underground.json"),
                "--pool", pool.name, "--synth-controls", str(pool),
                "--preset", "composer-learned-recall", "--set", "modules=[24,16,12,8]",
                "--set", "learned_code=24", "--beat-rows", "--synth-rows",
                "--synth-lesson-group", common.REFERENCE_GROUP,
                "--synth-vocab", "factored:8:4", "--clock", "--window", "8", "--stride", "1024",
                "--batch", "32", "--epochs", "3", "--eval-tracks", "1", "--eval-bars", "25",
                "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                "--save-every", "5",
            ]
            with (patch.object(sys, "argv", argv),
                  patch.object(train.composition, "compare", side_effect=compare),
                  patch.object(train, "prediction_error", side_effect=prediction_error),
                  patch.object(train, "density_rollout", return_value=None)):
                train.main()
            best_prediction = Path(directory) / "audit_e002.npz"
            self.assertTrue(best_prediction.is_file())
            self.assertFalse(json.loads(best_prediction.with_suffix(".json").read_text())["best"])
            last = Path(directory) / "audit_e003.npz"
            self.assertTrue(last.is_file())
            self.assertFalse(json.loads(last.with_suffix(".json").read_text())["best"])

            retry_dir = Path(directory) / "retry"
            argv[argv.index("--out") + 1] = str(retry_dir)
            with (patch.object(sys, "argv", argv),
                  patch.object(train.composition, "compare", side_effect=OSError("renderer failed")),
                  self.assertRaisesRegex(OSError, "renderer failed")):
                train.main()
            recovered = retry_dir / "evaluation-recovery.npz"
            self.assertTrue(recovered.is_file())
            self.assertEqual(common.layout_key(json.loads(recovered.with_suffix(".json").read_text())),
                             common.layout_key(json.loads(last.with_suffix(".json").read_text())))

    def test_a_stuck_run_stops_and_a_resumed_run_continues_its_curve(self):
        import train

        def compare(*args, **kwargs):
            return dict(selection_score=.5, score=.5, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        readings = []

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            if constant is not None:
                return {"balanced": .95}          # always the commonest bar
            return {"balanced": readings.pop(0) if brain is not None else .2}

        pool = ROOT / "pool/pool_underground.json"
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "train.py", "--name", "steady", "--out", directory,
                "--streams", str(ROOT / "corpus/streams_underground.json"),
                "--pool", pool.name, "--synth-controls", str(pool),
                "--preset", "composer-learned-recall", "--set", "modules=[24,16,12,8]",
                "--set", "learned_code=24", "--beat-rows", "--synth-rows",
                "--synth-lesson-group", common.REFERENCE_GROUP,
                "--synth-vocab", "factored:8:4", "--clock", "--window", "8", "--stride", "1024",
                "--batch", "32", "--epochs", "9", "--eval-tracks", "1", "--eval-bars", "25",
                "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                "--save-every", "1", "--error-patience", "2",
            ]
            curve = Path(directory) / "metrics.jsonl"

            def run(arguments, errors):
                readings[:] = [value for pair in errors for value in pair]      # training, then test
                with (patch.object(sys, "argv", arguments),
                      patch.object(train.composition, "compare", side_effect=compare),
                      patch.object(train, "prediction_error", side_effect=prediction_error),
                      patch.object(train, "density_rollout", return_value=None)):
                    train.main()
                self.assertEqual(readings, [])
                return [json.loads(line) for line in curve.read_text().splitlines()]

            # A lower training error alone is progress (epoch 2). Two evaluations without a new low of
            # either error end the run.
            rows = run(argv, [(.9, .9), (.8, .8), (.7, .8), (.7, .8), (.7, .8)])
            self.assertEqual([row["checkpoint"] for row in rows], ["000", "001", "002", "003", "004"])
            last = Path(directory) / "steady_e004.npz"
            self.assertTrue(last.is_file())

            # Resumed, the run goes on with its next epoch on the same curve, and stops again when stuck.
            rows = run(argv + ["--init", str(last), "--resume"], [(.7, .7), (.7, .7), (.7, .7)])
            self.assertEqual([row["checkpoint"] for row in rows],
                             ["000", "001", "002", "003", "004", "005", "006", "007"])
            self.assertEqual([row.get("tag") for row in rows].count("untrained"), 1)
            minutes = [row["minutes"] for row in rows]
            self.assertEqual(minutes, sorted(minutes))
            card = json.loads((Path(directory) / "steady_e007.json").read_text())
            self.assertEqual(card["epoch"], 7)
            self.assertGreaterEqual(card["minutes"], json.loads(last.with_suffix(".json").read_text())["minutes"])

    def test_a_run_of_the_slate_hears_every_bar_first_and_stops_when_it_does_not_listen(self):
        import train

        def compare(*args, **kwargs):
            return dict(selection_score=.5, score=.5, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        readings = []

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            if constant is not None:
                return {"balanced": .6}           # always the commonest bar
            return {"balanced": readings.pop(0) if brain is not None else .2}

        pool = ROOT / "pool/pool_underground.json"

        def run(name, directory, errors, *extra):
            argv = [
                "train.py", "--name", name, "--out", directory,
                "--streams", str(ROOT / "corpus/streams_underground.json"),
                "--pool", pool.name, "--synth-controls", str(pool),
                "--layout", "chain", "--set", "modules=[24,16,12,8]", "--hear-first", "--listen-patience", "1",
                "--beat-rows", "--synth-rows", "--synth-fill", "--synth-vocab", "factored:8:4", "--clock",
                "--window", "8", "--stride", "4096", "--batch", "32", "--eval-tracks", "1", "--eval-bars", "25",
                "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                "--save-every", "1", *extra,
            ]
            readings[:] = [value for pair in errors for value in pair]      # training, then test
            with (patch.object(sys, "argv", argv),
                  patch.object(train.composition, "compare", side_effect=compare),
                  patch.object(train, "prediction_error", side_effect=prediction_error),
                  patch.object(train, "density_rollout", return_value=None),
                  patch.object(train.common, "hear", wraps=train.common.hear) as heard,
                  patch.object(train.common, "answer", wraps=train.common.answer) as answered):
                train.main()
            self.assertEqual(readings, [])
            self.assertTrue(train.common.HEAR_FIRST)
            self.assertGreater(heard.call_count, 0)       # every lesson is on a bar heard first
            self.assertGreater(answered.call_count, 0)    # and every reading answers a bar heard first
            return [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]

        try:
            with tempfile.TemporaryDirectory() as directory:
                # An evaluation at or above the commonest-bar level, with a patience of one: the run ends itself.
                rows = run("deaf", directory, [(.9, .9), (.7, .7)], "--epochs", "9")
                self.assertEqual([row["checkpoint"] for row in rows], ["000", "001"])
                self.assertEqual(rows[0]["layout"], "chain")
                self.assertTrue(rows[0]["hear_first"])
                self.assertEqual(rows[0]["blind_baseline"]["test"]["balanced"], .6)
                self.assertFalse(any(row.get("listens") for row in rows))
                card = json.loads((Path(directory) / "deaf_e001.json").read_text())
                self.assertTrue(card["hear_first"])
                self.assertEqual(card["layout"], "chain")
                self.assertEqual(card["settings"]["recall_gain"], 1.0)
            with tempfile.TemporaryDirectory() as directory:
                # Below that level, the run listens and goes on to its last epoch.
                rows = run("listens", directory, [(.9, .9), (.5, .5), (.5, .5)], "--epochs", "2")
                self.assertEqual([row["checkpoint"] for row in rows], ["000", "001", "002"])
                self.assertEqual([bool(row.get("listens")) for row in rows], [False, True, False])
            with tempfile.TemporaryDirectory() as directory:
                # The stage-1 recipe: idle inputs for the later stages, evenly spaced evaluations, the rate floor.
                rows = run("reserved", directory, [(.9, .9), (.5, .5)], "--epochs", "1", "--reserve", "inputs",
                           "--evals-per-epoch", "3", "--plateau", "3", "--plateau-floor", "0.1")
                self.assertEqual([row["checkpoint"] for row in rows], ["000", "001"])
                reserved = json.loads((Path(directory) / "reserved_e001.json").read_text())
                self.assertEqual(reserved["reserve"], "inputs")
                self.assertEqual(reserved["settings"]["inputs"],
                                 card["settings"]["inputs"] + common.EAR_INPUTS + common.STYLE_INPUTS)
                self.assertEqual(reserved["settings"]["slots"], card["settings"]["slots"])
                self.assertGreaterEqual(reserved["settings"]["learned_code"], reserved["settings"]["inputs"])
                self.assertFalse(card.get("reserve"))
        finally:
            common.HEAR_FIRST = False
            common.RESERVE = False

    def test_a_run_evaluates_at_its_marks_and_reads_within_its_share(self):
        import train

        def compare(*args, **kwargs):
            return dict(selection_score=.5, score=.5, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            return {"balanced": .6 if constant is not None else .2 if brain is None else .5}

        pool = ROOT / "pool/pool_underground.json"
        try:
            with tempfile.TemporaryDirectory() as directory:
                argv = [
                    "train.py", "--name", "marks", "--out", directory,
                    "--streams", str(ROOT / "corpus/streams_underground.json"),
                    "--pool", pool.name, "--synth-controls", str(pool),
                    "--layout", "chain", "--set", "modules=[24,16,12,8]", "--hear-first", "--reserve", "inputs",
                    "--beat-rows", "--synth-rows", "--synth-fill", "--synth-vocab", "factored:8:4", "--clock",
                    "--window", "8", "--stride", "4096", "--batch", "8", "--eval-tracks", "1", "--eval-bars", "25",
                    "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                    "--save-every", "1", "--epochs", "2", "--evals-per-epoch", "3",
                    "--eval-minutes", "600", "--read-share", "0.999", "--keep-latest",
                ]
                with (patch.object(sys, "argv", argv),
                      patch.object(train.composition, "compare", side_effect=compare),
                      patch.object(train, "prediction_error", side_effect=prediction_error),
                      patch.object(train, "density_rollout", return_value=None)):
                    train.main()
                rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
                latest = json.loads((Path(directory) / "marks_latest.json").read_text())
                self.assertTrue((Path(directory) / "marks_latest.npz").is_file())
            full = [row for row in rows[1:] if not row.get("light")]
            light = [row for row in rows if row.get("light")]
            self.assertEqual([row["checkpoint"] for row in full], ["002"])      # no mark came: the one at the end
            self.assertGreaterEqual(len(light), 3)                              # a share near 1 reads after every batch
            self.assertIn("001b", light[0]["checkpoint"])
            self.assertTrue(any(row["checkpoint"].startswith("002b") for row in light))
            self.assertTrue(all("reading_seconds" in row for row in light))
            self.assertEqual(latest["latest"], light[-1]["checkpoint"])        # the newest state, with a checkpoint's card
            self.assertEqual(latest["layout"], "chain")
            self.assertEqual(latest["reserve"], "inputs")
        finally:
            common.HEAR_FIRST = False

    def test_a_run_reads_its_error_between_the_evaluations(self):
        import train

        def compare(*args, **kwargs):
            return dict(selection_score=.5, score=.5, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        message = "brain did not settle within 1024 steps and a rescue of as many: residual=0.02, tolerance=0.003; no action issued"
        counted = {"readings": 0}

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            if constant is not None:
                return {"balanced": .6}
            if brain is None:
                return {"balanced": .2}
            counted["readings"] += 1
            if counted["readings"] == 5:          # the second light reading does not settle
                raise RuntimeError(message)
            return {"balanced": .5}

        pool = ROOT / "pool/pool_underground.json"
        try:
            with tempfile.TemporaryDirectory() as directory:
                argv = [
                    "train.py", "--name", "light", "--out", directory,
                    "--streams", str(ROOT / "corpus/streams_underground.json"),
                    "--pool", pool.name, "--synth-controls", str(pool),
                    "--layout", "chain", "--set", "modules=[24,16,12,8]", "--hear-first", "--reserve", "inputs",
                    "--beat-rows", "--synth-rows", "--synth-fill", "--synth-vocab", "factored:8:4", "--clock",
                    "--window", "8", "--stride", "4096", "--batch", "2", "--eval-tracks", "1", "--eval-bars", "25",
                    "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                    "--save-every", "1", "--epochs", "1", "--read-batches", "1", "--evals-per-epoch", "1",
                ]
                with (patch.object(sys, "argv", argv),
                      patch.object(train.composition, "compare", side_effect=compare),
                      patch.object(train, "prediction_error", side_effect=prediction_error),
                      patch.object(train, "density_rollout", return_value=None)):
                    train.main()
                rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            light = [row for row in rows if row.get("light")]
            self.assertGreaterEqual(len(light), 3)                       # a reading after every batch but the last
            self.assertEqual(rows[0]["checkpoint"], "000")
            self.assertEqual(rows[-1]["checkpoint"], "001")              # the pass ends with its full evaluation
            self.assertTrue(all("gpu" in row for row in rows))           # every row says what the trainer holds of the GPU
            self.assertEqual(light[0]["test_error"]["balanced"], .5)
            self.assertTrue(light[1]["unsettled"])                       # skipped and counted, the run goes on
            self.assertNotIn("test_error", light[1])
            self.assertEqual(rows[-1]["unsettled_counts"], {"lessons": 0, "readings": 1, "restores": 0})
        finally:
            common.HEAR_FIRST = False
            common.RESERVE = False

    def test_a_layout_run_hears_its_own_playing_from_the_first_pass(self):
        import train

        def compare(*args, **kwargs):
            return dict(selection_score=.5, score=.5, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            return {"balanced": .6 if constant is not None else .2 if brain is None else .5}

        pool = ROOT / "pool/pool_underground.json"
        try:
            with tempfile.TemporaryDirectory() as directory:
                argv = [
                    "train.py", "--name", "own", "--out", directory,
                    "--streams", str(ROOT / "corpus/streams_underground.json"),
                    "--pool", pool.name, "--synth-controls", str(pool),
                    "--layout", "chain", "--set", "modules=[24,16,12,8]", "--hear-first", "--reserve", "inputs",
                    "--beat-rows", "--synth-rows", "--synth-fill", "--synth-vocab", "factored:8:4", "--clock",
                    "--window", "17", "--stride", "4096", "--batch", "32", "--eval-tracks", "1", "--eval-bars", "25",
                    "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                    "--save-every", "1", "--epochs", "1",
                    "--own-path", "0.35", "--own-ramp", "1", "--rollout-bars", "8", "--own-sample", "0.11",
                ]
                with (patch.object(sys, "argv", argv),
                      patch.object(train.composition, "compare", side_effect=compare),
                      patch.object(train, "prediction_error", side_effect=prediction_error),
                      patch.object(train, "density_rollout", return_value=None),
                      patch.object(train.common, "sample_slots", wraps=train.common.sample_slots) as sampled):
                    train.main()
                rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
                card = json.loads((Path(directory) / "own_e001.json").read_text())
            self.assertEqual([row["checkpoint"] for row in rows], ["000", "001"])
            self.assertEqual(card["training"]["own_path"], 0.35)
            self.assertEqual(card["training"]["rollout_bars"], 8)
            self.assertGreater(sampled.call_count, 0)        # in lessons the brain heard bars it sampled itself
        finally:
            common.HEAR_FIRST = False
            common.RESERVE = False

    def test_a_run_survives_a_brain_that_does_not_settle(self):
        import train

        message = ("brain did not settle within 1024 steps and a rescue of as many: residual=0.0233385, "
                   "tolerance=0.003; no action issued")

        def compare(*args, **kwargs):
            return dict(selection_score=.5, score=.5, hold_score=.9, prompts_above_hold=0,
                        runs=[{}], prompt_protocol=2)

        counted = {"readings": 0, "heard": 0}

        def prediction_error(brain, tracks, constant=None, answers=None):
            if answers is not None:
                return dict(ROLLOUT)
            if constant is not None:
                return {"balanced": .6}
            if brain is None:
                return {"balanced": .2}
            counted["readings"] += 1
            if counted["readings"] == 5:          # the first reading of the second evaluation
                raise RuntimeError(message)
            return {"balanced": .5}

        hear_first = common.hear

        def hear(brain, x):
            # lessons hear in the trainer's own loop; the readings' free-running continuations hear through
            # common.answer and are not counted (the caller is the first frame outside unittest.mock)
            caller = sys._getframe(1)
            while caller.f_code.co_filename.endswith("mock.py"):
                caller = caller.f_back
            if caller.f_code.co_name == "main":
                counted["heard"] += 1
                if counted["heard"] == 8:         # the first lesson of the second pass (a pass has seven)
                    raise RuntimeError(message)
            return hear_first(brain, x)

        pool = ROOT / "pool/pool_underground.json"
        try:
            with tempfile.TemporaryDirectory() as directory:
                argv = [
                    "train.py", "--name", "shaky", "--out", directory,
                    "--streams", str(ROOT / "corpus/streams_underground.json"),
                    "--pool", pool.name, "--synth-controls", str(pool),
                    "--layout", "chain", "--set", "modules=[24,16,12,8]", "--hear-first",
                    "--beat-rows", "--synth-rows", "--synth-fill", "--synth-vocab", "factored:8:4", "--clock",
                    "--window", "8", "--stride", "4096", "--batch", "32", "--eval-tracks", "1", "--eval-bars", "25",
                    "--gate-bars", "5", "--calibration-bars", "5", "--temps", ".08", "--replay-bars", "5",
                    "--save-every", "1", "--epochs", "3", "--unsettled-patience", "2",
                ]
                with (patch.object(sys, "argv", argv),
                      patch.object(train.composition, "compare", side_effect=compare),
                      patch.object(train, "prediction_error", side_effect=prediction_error),
                      patch.object(train, "density_rollout", return_value=None),
                      patch.object(train.common, "hear", side_effect=hear)):
                    train.main()
                rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
                log = (Path(directory) / "shaky.log").read_text()
                saved = sorted(path.name for path in Path(directory).glob("shaky_e*.npz"))
                card = json.loads((Path(directory) / "shaky_e003.json").read_text())
            # the run went through its three passes: the unsettled lesson and the unsettled evaluation were skipped
            self.assertEqual([row["checkpoint"] for row in rows], ["000", "001", "002", "003"])
            self.assertEqual(counted["readings"], 7)
            self.assertTrue(rows[2]["unsettled"])
            self.assertNotIn("test_error", rows[2])
            self.assertEqual(saved, ["shaky_e001.npz", "shaky_e003.npz"])      # no checkpoint from the failed evaluation
            # two in a row: the weights of the last checkpoint came back at half the rate, and the curve and card say so
            counts = {"lessons": 1, "readings": 1, "restores": 1}
            self.assertEqual(rows[2]["unsettled_counts"], counts)
            self.assertEqual(rows[3]["unsettled_counts"], counts)
            self.assertEqual(card["unsettled"], counts)
            self.assertAlmostEqual(rows[3]["eta"], rows[1]["eta"] / 2)
            self.assertIn("did not settle in a lesson of epoch 2 batch 1; skipped", log)
            self.assertIn("did not settle in the evaluation 002; skipped", log)
            self.assertIn("weights of shaky_e001.npz restored, learning rate halved", log)
        finally:
            common.HEAR_FIRST = False
            common.RESERVE = False

    def test_generated_accents_reach_the_pattern(self):
        self.state[15] = 1
        common.SYNTH["bass"][1] = [2, 2, 2, 2] * 4
        self.state[self.start + len(timbre.PATCH_SIZES):self.start + len(timbre.SIZES)] = [6, 1, 3, 0]
        bar = common.score_bar(self.state, self.pool, [None]*3, [None]*4)
        np.testing.assert_allclose(bar["patterns"][1]["vel"][:4], [1, 2/7, 4/7, 1/7])
        self.assertEqual(bar["patterns"][1]["accent"][:4], [1, 0, 0, 0])

    def test_motif_return_requires_intervening_change(self):
        seq = np.tile(self.state, (32, 1))
        self.assertEqual(composition.profile(seq)["bass_repeat_8"], 0)
        seq[4::8, 15] = 2
        self.assertGreater(composition.profile(seq)["bass_repeat_8"], 0)
        keys = quality.music_keys(seq)
        self.assertNotEqual(keys[3], keys[4])

    def test_validation_prompt_covers_groove_entry_and_return(self):
        seq = np.tile(self.state, (25, 1))
        groove = seq.copy()
        groove[:, 0] = groove[:, 12] = 1
        groove[12, 15] = 2
        entry = seq.copy()
        entry[:8, 1] = 0
        back = seq.copy()
        back[8:, 0] = back[8:, 12] = 1
        tracks = [dict(bars=b) for b in (seq, groove, entry, back)]
        choices = composition.prompts(tracks, 17, np.random.default_rng(1))
        self.assertEqual([c[3] for c in choices], ["groove", "synth entry", "kick return"])
        self.assertEqual(len({c[0] for c in choices}), 3)

    def test_sampler_patch_and_worklet_engine_share_the_same_controls(self):
        import base64
        samples = (np.sin(np.arange(8000)*.09)*np.exp(-np.arange(8000)/1000)*16000).astype("<i2")
        pcm = base64.b64encode(samples.tobytes()).decode("ascii")
        self.pool["sample_bank"] = {name:dict(pcm16=pcm, rate=8000, root=48) for name in ("a","b")}
        book = timbre.teachers(self.pool)
        self.assertEqual(book["version"], "dsp-v3")
        self.assertEqual(timbre.sizes(book)[-3:], (2,7,2))
        token = {"voice":"sampler", "sample":"b", "patch":{"start":0,"reverse":1}, "pattern":{"notes":[48]}}
        row = timbre.encode(token, 0, book)
        patch, register = timbre.instrument(row, book)
        self.assertEqual((patch["voice"], patch["sample"], patch["patch"]["reverse"], register),
                         ("sampler","b",1,4))
        self.assertTrue(all(note is None for note in common.synth_pattern([1]*16, patch, held=0)["notes"]))
        js = """
          import fs from 'node:fs';
          import {renderScore, stats} from './engine/render.mjs';
          const [pool, patch] = JSON.parse(fs.readFileSync(0,'utf8'));
          const play = (root, reverse, present=true) => {
            const p = {...patch, patch:{...patch.patch, reverse}, sample: present ? patch.sample : 'missing'};
            return renderScore({bars:[{tokens:[0,1,0,0], patches:[null,p], patterns:[null,
              {notes:[root,...Array(15).fill(null)],vel:[1,...Array(15).fill(0)],len:[5,...Array(15).fill(1)]}]}]},
              pool,{sr:22050,layer:1,tail:0,dry:true});
          };
          const forward=play(48,0), reverse=play(48,1), pitched=play(60,0), missing=play(48,0,false);
          const delta=(a,b)=>a.L.reduce((sum,x,i)=>sum+Math.abs(x-b.L[i]),0);
          if (stats(forward.L,forward.R).rms<.003 || stats(reverse.L,reverse.R).nonfinite ||
              delta(forward,reverse)<5 || delta(forward,pitched)<5 || stats(missing.L,missing.R).peak!==0) process.exit(1);
        """
        subprocess.run(["node","--input-type=module","-e",js],cwd=ROOT,
                       input=json.dumps([self.pool,patch]),text=True,check=True)

    def test_plateau_requires_both_validation_readings_to_stall(self):
        from train import Plateau
        schedule = Plateau(.2, 2)
        self.assertEqual(schedule.next_rate(.2, .7, .4), .2)
        self.assertEqual(schedule.next_rate(.2, .7, .4), .2)
        self.assertEqual(schedule.next_rate(.2, .7, .42), .2)
        self.assertEqual(schedule.next_rate(.2, .7, .42), .2)
        self.assertAlmostEqual(schedule.next_rate(.2, .7, .42), .14)

    def test_plateau_goes_down_to_its_floor(self):
        from train import Plateau
        self.assertAlmostEqual(Plateau(.2, 1).floor, .05)             # a quarter of the starting rate, as before
        schedule = Plateau(.2, 1, .1)                                  # stage 1: down to a tenth
        rate = .2
        schedule.next_rate(rate, .7, .4)                               # the first reading sets the best
        for _ in range(12):
            rate = schedule.next_rate(rate, .7, .4)
        self.assertAlmostEqual(rate, .02)

    def test_balanced_windows_rotate_without_crossing_recordings(self):
        from train import SourceWindows
        tracks = []
        for group, length, share, offset in (("album", 8, .5, 0), ("ej", 12, .25, 100), ("mix", 20, .25, 200)):
            bars = np.arange(offset, offset+length)
            tracks.append(dict(group=group, window_share=share, bars=bars, mood=bars))
        schedule = SourceWindows(tracks, 2, 2, np.random.default_rng(4))
        self.assertEqual(schedule.counts, {"album":4, "ej":2, "mix":2})
        first = []
        for _ in range(3):
            draws = schedule.draw()
            self.assertTrue(all(b[1] == b[0]+1 for b,m in draws))
            first.extend(b[0] for b,m in draws if 100<=b[0]<200)
        self.assertEqual(len(set(first)), 6)

    def test_two_heldout_records_use_disjoint_prompts(self):
        seq = np.tile(self.state, (110, 1))
        seq[::15, 15] = 2
        prompts = composition.prompts([dict(bars=seq)]*2, 17, np.random.default_rng(8))
        self.assertEqual(len(prompts), 3)
        for k, (i, a, n, _) in enumerate(prompts):
            for j, b, m, _ in prompts[k+1:]:
                self.assertTrue(i != j or a+8+n <= b or b+8+m <= a)

    def test_legacy_layout_after_composer(self):
        common.configure(97)
        state = np.zeros(common.SLOTS, dtype=int)
        state[1] = 2
        self.assertEqual(len(state), 12)
        bar = common.score_bar(state, self.pool, [None]*3, [None]*4)
        self.assertNotIn("patches", bar)
        self.assertEqual(bar["tokens"], [0, 2, 0, 0])


if __name__ == "__main__":
    unittest.main()
