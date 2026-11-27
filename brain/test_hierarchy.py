"""The two pathways share a brain, but cannot communicate below the coordinator."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import anatomy
import common
import hierarchy
import timbre

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "pool/pool_underground.json"
SID_POOL = ROOT / "pool/pool_underground_sid.json"
NO_SID_POOL = "needs the fitted SID pool (pool/pool_underground_sid.json), which is not shipped"
HAS_SID_POOL = SID_POOL.is_file() and not os.environ.get("CADANCE_SYNTHETIC_POOL")


def sound_pool():
    """The fitted sound pool when the checkout has it. A fresh clone has none (the pool is not shipped), and these
    tests read only its shape: four layers (beat, bass, lead, air) of 65 sounds each, every control in range. Then a
    synthetic pool of that shape stands in. CADANCE_SYNTHETIC_POOL=1 runs the tests as a clone does: the synthetic pool,
    and the tests of the fitted SID pool skipped."""
    if POOL.is_file() and not os.environ.get("CADANCE_SYNTHETIC_POOL"):
        return json.loads(POOL.read_text())
    layers = [{"name": "beat", "tokens": [None] * 65}]
    for i, name in enumerate(("bass", "lead", "air")):
        tokens = [None] + [{"voice": "synth",
                            "patch": {key: ((j + 3 * k) % 7) / 6 for k, key in enumerate(timbre.NAMES)},
                            "pattern": {"notes": [36 + 12 * i + j % 12, None] * 8, "vel": [0.85] * 16}}
                           for j in range(1, 65)]
        layers.append({"name": name, "tokens": tokens})
    return {"version": 1, "layers": layers}


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        pool = sound_pool()
        common.ONOFF = False
        common.RESERVE = common.SID_LIVE = False
        common.set_content(None)
        common.configure(65, steps=True, clock=True, synth={"kind": "steps"}, patch_book=timbre.teachers(pool))

    def make_brain(self, depth=1):
        return common.compose(hierarchy.PRESET, backend="cpu", learned_code=48,
                              modules=(16, 12, 16, 12, 10), coordinator_depth=depth)[0]

    def assert_topology(self, brain, depth):
        graph = brain.connectome
        pops = graph.populations
        roles = np.full(graph.n, "", dtype=object)
        for name, ids in pops.items():
            if "/" not in name or name in ("motor/rhythm", "motor/melody"):
                roles[list(ids)] = name
        expected = {
            ("sensory", "code"), ("code", "association"), ("prefrontal", "code"),
            ("prefrontal", "association"),
        }
        apical = set()
        if depth == 2:
            expected.update({("association", "phrase_context"), ("phrase_context", "association")})
            apical.add(("phrase_context", "association"))
        for kind, (detail, plan) in zip(("rhythm", "melody"), hierarchy.BRANCHES, strict=True):
            motor = f"motor/{kind}"
            expected.update({(detail, plan), (plan, detail), (plan, "association"), ("association", plan),
                             (detail, motor), (motor, detail), ("association", detail)})
            apical.update({(plan, detail), ("association", plan), (motor, detail)})
        for name in pops:
            if name.startswith(("recall", "next")):
                expected.add((name, "association"))
        found = set(zip(roles[graph.pre], roles[graph.post]))
        self.assertEqual({(left, right) for left, right in found if left != right}, expected)
        flags = brain.brain.apical.synapses
        self.assertEqual(set(zip(roles[graph.pre[flags]], roles[graph.post[flags]])), apical)

        # No categorical output is split across branches or read by the wrong one.
        rhythm = {0, 4, 5, *range(12, 60)}
        for slot, (first, width) in enumerate(zip(common.OFFSETS, common.SIZES, strict=True)):
            motor = np.asarray(pops["motor"])[first:first + width]
            self.assertTrue((roles[motor] == ("motor/rhythm" if slot in rhythm else "motor/melody")).all())
        self.assertEqual(len(brain.learner.slot_sizes), common.SLOTS)

    def test_both_variants_only_join_at_the_coordinator(self):
        for depth in (1, 2):
            with self.subTest(depth=depth):
                self.assert_topology(self.make_brain(depth), depth)

    def test_two_layer_coordinator_learns_and_round_trips_as_one_brain(self):
        from cadence import Brain

        brain = self.make_brain(2)
        before = brain.brain.weights.copy()
        for i in range(3):
            current = np.zeros((1, common.SLOTS), dtype=int)
            current[0, :2] = [1, 1]
            current[0, 12 + i] = 2
            current[0, 60 + i] = 4
            mood = common.mood_rows([[.5, .5]], bars=[i])
            x = common.observe(current, mood)
            brain.lesson(x, common.audible_targets(current))
            brain.act(x, greedy=True)
        self.assertGreater(np.linalg.norm(brain.brain.weights - before), 0)
        changed = np.abs(brain.brain.weights - before) > 0
        for kind in ("rhythm", "melody"):
            motor = np.isin(brain.connectome.post, brain.connectome.populations[f"motor/{kind}"])
            self.assertTrue((changed & motor).any())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duet.npz"
            brain.save(path)
            loaded = Brain.load(path, backend="cpu")
            self.assert_topology(loaded, 2)
            self.assertTrue(loaded.attention)
            self.assertEqual(loaded.recall.heads, 4)
            self.assertEqual(loaded.recall.capacity, 128)
            self.assertEqual(loaded.hippocampus.amplitude, 0)
            np.testing.assert_array_equal(brain.act(x, greedy=True), loaded.act(x, greedy=True))

    def test_invalid_depth_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "one or two coordinator"):
            self.make_brain(3)

    def make_instruments(self, **over):
        return common.compose(hierarchy.INSTRUMENT_PRESET, backend="cpu", learned_code=48,
                              modules=(16, 12) * 4 + (10,), **over)

    @staticmethod
    def roles(brain):
        graph = brain.connectome
        roles = np.full(graph.n, "", dtype=object)
        for name, ids in graph.populations.items():
            if "/" not in name or name.startswith("motor/"):
                roles[list(ids)] = name
        return roles

    def edges(self, brain):
        graph, roles = brain.connectome, self.roles(brain)
        return {(left, right) for left, right in zip(roles[graph.pre], roles[graph.post]) if left != right}

    def test_every_instrument_has_a_pathway_and_the_coordinator_writes_the_arrangement(self):
        brain, settings = self.make_instruments()
        self.assertEqual(settings["architecture"], hierarchy.INSTRUMENT_ARCHITECTURE)
        graph, roles = brain.connectome, self.roles(brain)
        expected = {("sensory", "code"), ("code", "association"), ("prefrontal", "code"),
                    ("prefrontal", "association"), ("association", "motor/coordinator"),
                    ("motor/coordinator", "association")}
        apical = {("motor/coordinator", "association")}
        for name in hierarchy.INSTRUMENTS:
            detail, plan, motor = f"{name}_detail", f"{name}_plan", f"motor/{name}"
            expected.update({(detail, plan), (plan, detail), (plan, "association"), ("association", plan),
                             (detail, motor), (motor, detail), ("association", detail)})
            apical.update({(plan, detail), ("association", plan), (motor, detail)})
        expected.update((name, "association") for name in graph.populations if name.startswith(("recall", "next")))
        self.assertEqual(self.edges(brain), expected)
        flags = brain.brain.apical.synapses
        self.assertEqual(set(zip(roles[graph.pre[flags]], roles[graph.post[flags]])), apical)

        # The coordinator owns the four sound slots. Every other slot belongs to one instrument, whole.
        parts = common.pathway_slots(hierarchy.INSTRUMENT_ARCHITECTURE)
        self.assertEqual(parts[hierarchy.COORDINATOR], [0, 1, 2, 3])
        self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(common.SLOTS)))
        motor = np.asarray(graph.populations["motor"])
        for name, slots in parts.items():
            for slot in slots:
                first = common.OFFSETS[slot]
                self.assertTrue((roles[motor[first:first + common.SIZES[slot]]] == f"motor/{name}").all())

        before = brain.brain.weights.copy()
        for i in range(3):
            current = np.zeros((1, common.SLOTS), dtype=int)
            current[0, :4] = [1, 1, 1, 1]
            current[0, 12 + i] = 2
            current[0, 60 + i] = 4
            x = common.observe(current, common.mood_rows([[.5, .5]], bars=[i]))
            brain.lesson(x, common.audible_targets(current))
            brain.act(x, greedy=True)
        changed = np.abs(brain.brain.weights - before) > 0
        for name in parts:
            self.assertTrue((changed & np.isin(graph.post, graph.populations[f"motor/{name}"])).any(), name)

    def test_a_pathway_can_hear_the_bar_itself_and_own_a_recall_head(self):
        brain, _ = self.make_instruments(pathway_reads=True, pathway_recall=True)
        edges = self.edges(brain)
        ports = [name for name in brain.connectome.populations if name.startswith(("recall", "next"))]
        self.assertEqual(brain.recall.heads, len(hierarchy.INSTRUMENTS))
        for head, name in enumerate(hierarchy.INSTRUMENTS):
            self.assertIn(("code", f"{name}_detail"), edges)
            own = [port for port in ports if hierarchy._head(port) == head]
            self.assertTrue(own)
            for port in own:
                self.assertIn((port, f"{name}_plan"), edges)
                self.assertNotIn((port, "association"), edges)

    def test_the_upper_coordinator_with_the_recall_heads_drives_the_phrase_coordinator_and_writes_the_arrangement(self):
        brain, _ = self.make_instruments(coordinator_depth=2, context_recall=True)
        graph, roles, edges = brain.connectome, self.roles(brain), self.edges(brain)
        flags = brain.brain.apical.synapses
        apical = set(zip(roles[graph.pre[flags]], roles[graph.post[flags]]))
        # the memory reads into the upper coordinator alone
        ports = [name for name in graph.populations if name.startswith(("recall", "next"))]
        self.assertTrue(ports)
        for port in ports:
            self.assertIn((port, "phrase_context"), edges)
            self.assertNotIn((port, "association"), edges)
        # it lies before the phrase coordinator: its drive down is basal, the way back is the apical return
        self.assertIn(("phrase_context", "association"), edges)
        self.assertNotIn(("phrase_context", "association"), apical)
        self.assertIn(("association", "phrase_context"), apical)
        # it writes the arrangement beside the phrase coordinator, so the taught slots return to it in one hop
        self.assertIn(("phrase_context", "motor/coordinator"), edges)
        self.assertIn(("motor/coordinator", "phrase_context"), apical)
        for name in hierarchy.INSTRUMENTS:
            self.assertNotIn(("phrase_context", f"motor/{name}"), edges)
        plan = brain.learner._credit_plan()
        upper = np.asarray(graph.populations["phrase_context"])
        self.assertTrue((plan.motor_distance[upper] == 1).all())
        # with upper_return it also has a return from every pathway's motor slots, apical, one hop; not without
        back, _ = self.make_instruments(coordinator_depth=2, context_recall=True, upper_return=0.1)
        roles, flags = self.roles(back), back.brain.apical.synapses
        returns = set(zip(roles[back.connectome.pre[flags]], roles[back.connectome.post[flags]]))
        for name in hierarchy.INSTRUMENTS:
            self.assertIn(("phrase_context", f"motor/{name}"), self.edges(back))
            self.assertIn((f"motor/{name}", "phrase_context"), returns)
            self.assertNotIn(("association", f"motor/{name}"), self.edges(back))
        with self.assertRaisesRegex(ValueError, "upper_return needs context_recall"):
            self.make_instruments(coordinator_depth=2, upper_return=0.1)
        # two layers without the heads up there are wired as they were
        plain, _ = self.make_instruments(coordinator_depth=2)
        roles, flags = self.roles(plain), plain.brain.apical.synapses
        self.assertIn(("phrase_context", "association"),
                      set(zip(roles[plain.connectome.pre[flags]], roles[plain.connectome.post[flags]])))
        self.assertNotIn(("phrase_context", "motor/coordinator"), self.edges(plain))

    def test_every_layout_gives_every_slot_one_pathway(self):
        counts = {hierarchy.ARCHITECTURE: 2, hierarchy.TRIO_ARCHITECTURE: 3,
                  hierarchy.INSTRUMENT_ARCHITECTURE: 4, hierarchy.PARTS_ARCHITECTURE: 7}
        for architecture, pathways in counts.items():
            with self.subTest(architecture=architecture):
                parts = common.pathway_slots(architecture)
                self.assertEqual(len([name for name in parts if name != hierarchy.COORDINATOR]), pathways)
                self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(common.SLOTS)))
        trio = common.pathway_slots(hierarchy.TRIO_ARCHITECTURE)
        four = common.pathway_slots(hierarchy.INSTRUMENT_ARCHITECTURE)
        self.assertEqual(sorted(trio["melody"]), sorted(four["lead"] + four["air"]))
        self.assertEqual(sorted(trio["bass"]), sorted(four["bass"]))
        split = common.pathway_slots(hierarchy.PARTS_ARCHITECTURE)
        for name in hierarchy.INSTRUMENTS[1:]:
            self.assertEqual(sorted(split[name] + split[f"{name}_sound"]), sorted(four[name]))
            self.assertEqual(len(split[name]), common.SYN // 3)          # the notes alone

    def test_the_slate_has_no_route_from_memory_to_the_motor_and_wires_its_pathways(self):
        import layouts

        for name, entry in layouts.LAYOUTS.items():
            with self.subTest(layout=name):
                preset, over = layouts.layout(name)
                settings = {**common.common_settings(), **common.PRESETS[preset], **over}
                # generation 2: no route from memory to the motor slots (the recall reaches hidden regions alone), four
                # attention heads, a learned code as wide as the observation, open labels
                self.assertIn(preset, layouts.NO_ROUTE)
                self.assertIsNone(settings["recall_tokens"])
                self.assertEqual(settings["recall_gain"], 0.0)
                self.assertFalse(settings["recall_action"])
                self.assertEqual(settings["recall_heads"], common.L)
                self.assertEqual(settings["learned_code"], "observation")
                self.assertIn(preset, common.OPEN_LABEL_PRESETS)
                parts = common.pathway_slots(hierarchy.PRESETS[preset])
                pathways = [part for part in parts if part != hierarchy.COORDINATOR]
                self.assertEqual(len(pathways), entry["pathways"])
                self.assertEqual(len(settings["modules"]), 2 * len(pathways) + 1)
        # Generation 1, still built by name for its old snapshots and probes: the route, a remembered bar read into the
        # motor slots through a code as wide as the observation.
        for name, entry in layouts.S1_LAYOUTS.items():
            with self.subTest(layout=name):
                preset, over = layouts.layout(name)
                settings = {**common.common_settings(), **common.PRESETS[preset], **over}
                self.assertEqual(list(settings["recall_tokens"]), list(range(common.ACTIONS)))
                self.assertEqual(settings["recall_gain"], 1.0)
                self.assertGreaterEqual(settings["learned_code"], common.INPUTS)
                self.assertNotIn(preset, common.OPEN_LABEL_PRESETS)
                if preset not in hierarchy.PRESETS:
                    self.assertEqual(entry["pathways"], 0)
                    continue
                parts = common.pathway_slots(hierarchy.PRESETS[preset])
                pathways = [part for part in parts if part != hierarchy.COORDINATOR]
                self.assertEqual(len(pathways), entry["pathways"])
                self.assertEqual(len(settings["modules"]), 2 * len(pathways) + 1)
        # The innate copies of the slate's reflex and echo layouts, on a small copy of each (the arcs do not depend on
        # the widths): heard value v of a slot drives motor value v, at reflex_gain for the bar heard (the kit slot at
        # reflex_kit_gain) and at reflex_lag_gains for the bars further back (with reflex_lag_groups on the drum rows
        # and synth lines alone); every other synapse of the arcs is born at zero.
        edges = np.cumsum([0, *common.SIZES])
        width = int(edges[-1])
        lines = list(range(common.L + common.MODS, common.L + common.MODS + common.STEPS + common.SYN))
        for name in layouts.LAYOUTS:
            preset, over = layouts.layout(name)
            if over.get("reflex_birth") != "diagonal":
                continue
            with self.subTest(innate=name):
                small = {**over, "modules": [max(8, w // 16) for w in over["modules"]], "learned_code": 48}
                try:
                    brain, _ = common.compose(preset, backend="cpu", **small)
                    graph = brain.connectome
                    heard, motor = (np.asarray(graph.populations[key]) for key in ("sensory", "motor"))
                    pre, post = np.asarray(graph.pre), np.asarray(graph.post)
                    arc = np.isin(pre, heard) & np.isin(post, motor)
                    sign = np.asarray(graph.sign, dtype=float)[arc]
                    column, target = pre[arc] - heard.min(), post[arc] - motor.min()
                    later = column >= common.INPUTS                     # the lag blocks follow the observation's columns
                    lag = np.where(later, 1 + (column - common.INPUTS) // width, 0)
                    value = np.where(later, (column - common.INPUTS) % width, column)
                    diagonal = value == target
                    self.assertFalse(sign[~diagonal].any())
                    lags = over.get("echo_lags") or [1]
                    self.assertEqual(sorted(set(lag[diagonal].tolist())), list(range(len(lags))))
                    for k in range(len(lags)):                         # every slot's every value, at every lag
                        self.assertEqual(int((diagonal & (lag == k)).sum()), width)
                    slot = np.searchsorted(edges, target, side="right") - 1
                    expected = np.asarray(over.get("reflex_lag_gains") or [over["reflex_gain"]], dtype=float)[lag]
                    expected[(lag == 0) & (slot == 0)] = over["reflex_kit_gain"]
                    if over.get("reflex_lag_groups"):
                        expected[(lag > 0) & ~np.isin(slot, lines)] = 0.0
                    np.testing.assert_array_equal(sign[diagonal], expected[diagonal])
                    self.assertEqual(np.asarray(brain.recall.gains).size, 0)        # no recall port into the motor
                finally:
                    common.ECHO_LAGS = ()
        # The wiring of the two route layouts, on a small copy without the route (the topology is the same).
        small = dict(learned_code=48, **common.AUDIBLE_RECALL)
        for preset, widths in ((hierarchy.TRIO_PRESET, (16, 12) * 3 + (10,)), (hierarchy.PARTS_PRESET, (16, 12) * 7 + (10,))):
            with self.subTest(preset=preset):
                brain, settings = common.compose(preset, backend="cpu", modules=widths, **small)
                edges, parts = self.edges(brain), common.pathway_slots(settings["architecture"])
                self.assertIn(("association", "motor/coordinator"), edges)
                for name in parts:
                    if name == hierarchy.COORDINATOR:
                        continue
                    for edge in ((f"{name}_detail", f"{name}_plan"), (f"{name}_plan", "association"),
                                 (f"{name}_detail", f"motor/{name}"), ("association", f"{name}_detail")):
                        self.assertIn(edge, edges)
                    # a pathway reaches no other pathway's outputs
                    for other in parts:
                        if other not in (name, hierarchy.COORDINATOR):
                            self.assertNotIn((f"{name}_detail", f"motor/{other}"), edges)

    def test_the_value_path_is_born_switched_on(self):
        import layouts
        from cadence import Brain

        for name in layouts.S1_LAYOUTS:                     # the route layouts (generation 1)
            self.assertEqual(layouts.layout(name)[1]["recall_gain_born"], layouts.BORN_GAIN)
        for name in layouts.LAYOUTS:                        # the slate: no route, no born gain
            self.assertNotIn("recall_gain_born", layouts.layout(name)[1])
        # Cadence composes the gains at zero. The slate sets the first head's recall port and leaves the rest.
        brain = Brain.compose(12, 12, slots=(4,) * 3, modules=(8,), seed=0, recall=True, recall_heads=2,
                              recall_tokens=range(12), recall_gain=1.0)
        self.assertTrue((np.asarray(brain.recall.gains) == 0).all())
        common.born_gains(brain, layouts.BORN_GAIN)
        np.testing.assert_array_equal(np.asarray(brain.recall.gains), [layouts.BORN_GAIN, 0, 0, 0])
        # a lesson still moves the gains: they stay learned
        x = np.zeros((2, 12)); x[0, [0, 5, 9]] = 1; x[1, [1, 4, 10]] = 1
        for _ in range(4):
            brain.act(x, greedy=True)
            brain.lesson(x, np.array([[1, 2, 0], [3, 0, 1]]))
        self.assertFalse(np.array_equal(np.asarray(brain.recall.gains), [layouts.BORN_GAIN, 0, 0, 0]))
        # the birth value is no part of the wiring: the drawing of a brain does not change with it
        _, settings = common.compose(hierarchy.TRIO_PRESET, backend="cpu", modules=(16, 12) * 3 + (10,),
                                     learned_code=48, **common.AUDIBLE_RECALL)
        card = {"settings": {k: v for k, v in settings.items() if k != "plastic_synapses"}, "seed": 0, "steps": True,
                "onoff": False, "patch_book": common.PATCH_BOOK}
        born = {**card, "settings": {**card["settings"], "recall_gain_born": 1.0}}
        self.assertEqual(anatomy.describe(born), anatomy.describe(card))

    def test_the_drawn_layout_is_the_brain(self):
        cases = [(hierarchy.PRESET, dict(modules=(16, 12, 16, 12, 10), coordinator_depth=2)),
                 (hierarchy.INSTRUMENT_PRESET, dict(modules=(16, 12) * 4 + (10,))),
                 (hierarchy.INSTRUMENT_PRESET, dict(modules=(16, 12) * 4 + (10,), pathway_reads=True,
                                                    pathway_recall=True)),
                 ("composer-learned-recall", dict(modules=(24, 16, 12, 8)))]
        for preset, over in cases:
            with self.subTest(preset=preset, over=sorted(over)):
                brain, settings = common.compose(preset, backend="cpu", learned_code=48, **over)
                graph = brain.connectome
                settings = {k: v for k, v in settings.items() if k != "plastic_synapses"}
                card = {"settings": settings, "seed": 0, "steps": True, "onoff": False,
                        "patch_book": common.PATCH_BOOK}
                drawn = anatomy.describe(card)
                self.assertEqual(drawn["neurons"], graph.n)
                self.assertEqual(drawn["synapses"], int(np.count_nonzero(brain.learner.plastic_synapses)))
                self.assertEqual({region["name"]: region["size"] for region in drawn["regions"]},
                                 {name: len(ids) for name, ids in graph.populations.items() if "/" not in name})
                self.assertEqual(sum(output["slots"] for output in drawn["outputs"]), common.SLOTS)
                self.assertEqual(drawn["recall"]["heads"], brain.recall.heads)
                if preset in hierarchy.PRESETS:
                    roles = self.roles(brain)
                    flags = brain.brain.apical.synapses
                    feedback = {(link["to"], link["from"]) for link in drawn["links"] if link["kind"] == "feedback"}
                    self.assertEqual(feedback, set(zip(roles[graph.pre[flags]], roles[graph.post[flags]])))
                    lands = {name: drawn["recall"]["lands"][hierarchy._head(name)]
                             for name in graph.populations if name.startswith(("recall", "next"))}
                    self.assertEqual({(port, where) for port, where in lands.items()},
                                     {edge for edge in self.edges(brain) if edge[0] in lands})


class ReservedShapeTests(unittest.TestCase):
    """The stage-1 shape: idle inputs and idle SID slots for the later stages, and the live parts as they were."""

    def setUp(self):
        self.pool = sound_pool()
        common.ONOFF = False
        common.set_content(None)
        self.configure(False)

    def tearDown(self):
        self.configure(False)

    def configure(self, reserve, live=False):
        common.RESERVE, common.SID_LIVE = ("sid" if reserve is True else reserve), live
        common.configure(65, steps=True, clock=True, synth={"kind": "steps"}, patch_book=timbre.teachers(self.pool))

    def bars(self, count, seed=0):
        rng = np.random.default_rng(seed)
        bars = (rng.random((count, common.SLOTS)) * np.asarray(common.SIZES)).astype(int)
        bars[:, common.LIVE:] = 0
        bars[:, 1:4] = 1
        return bars

    def test_the_idle_parts_are_added_and_the_live_parts_stay(self):
        import gate

        plain = dict(slots=common.SLOTS, actions=common.ACTIONS, inputs=common.INPUTS)
        bars = self.bars(6)
        mood = np.full((6, common.MOOD), 0.5)
        heard = common.observe(bars, mood)
        repeat = gate.prediction_error(None, [{"bars": bars, "mood": mood}])
        self.configure(True)
        chips = 3 * sum(timbre.SID_SIZES)
        room = common.KIT_CLASSES - common.T                    # the kit slot's room for kits fitted later
        self.assertEqual((common.LIVE, common.SLOTS), (plain["slots"], plain["slots"] + 3 * timbre.SID_WIDTH))
        self.assertEqual(common.ACTIONS, plain["actions"] + room + chips)
        self.assertEqual(common.INPUTS, plain["inputs"] + room + chips + common.EAR_INPUTS + common.STYLE_INPUTS)
        self.assertGreaterEqual(common.common_settings()["learned_code"], common.INPUTS)
        wide = np.concatenate([bars, np.zeros((6, common.SLOTS - common.LIVE), dtype=int)], axis=1)
        x = common.observe(wide, mood)
        # the live slots and the mood are heard as before (the kit slot is wider), an idle block and the idle
        # inputs are silent
        live = plain["actions"] + room
        np.testing.assert_array_equal(x[:, :common.T], heard[:, :common.T])
        self.assertFalse(x[:, common.T:common.KIT_CLASSES].any())
        np.testing.assert_array_equal(x[:, common.KIT_CLASSES:live], heard[:, common.T:plain["actions"]])
        np.testing.assert_array_equal(x[:, common.ACTIONS:common.ACTIONS + common.MOOD], heard[:, plain["actions"]:])
        self.assertFalse(x[:, live:common.ACTIONS].any())
        self.assertFalse(x[:, common.ACTIONS + common.MOOD:].any())
        # whatever an idle block holds, it is not heard and not read
        wide[:, common.LIVE:] = 1
        np.testing.assert_array_equal(common.observe(wide, mood), x)
        again = gate.prediction_error(None, [{"bars": wide, "mood": mood}])
        self.assertEqual(again, repeat)
        # a sampled bar leaves an idle block off
        rng = np.random.default_rng(1)
        sampled = common.sample_slots(rng.normal(size=common.ACTIONS), wide[0], 0.5, rng)
        self.assertEqual(len(sampled), common.SLOTS)
        self.assertFalse(sampled[common.LIVE:].any())
        # and never a kit the score does not have, however strongly an empty class is driven
        loud = np.zeros(common.ACTIONS); loud[common.T:common.KIT_CLASSES] = 50.0
        for seed in range(5):
            self.assertLess(common.sample_slots(loud, wide[0], 1.0, np.random.default_rng(seed))[0], common.T)

    def test_idle_inputs_alone_leave_the_brain_its_size(self):
        plain = dict(slots=common.SLOTS, actions=common.ACTIONS, inputs=common.INPUTS, code=common.common_settings()["learned_code"])
        bars = self.bars(4)
        mood = np.full((4, common.MOOD), 0.5)
        heard = common.observe(bars, mood)
        self.configure("inputs")
        self.assertEqual((common.SLOTS, common.LIVE, common.ACTIONS), (plain["slots"], plain["slots"], plain["actions"]))
        self.assertEqual(common.INPUTS, plain["inputs"] + common.EAR_INPUTS + common.STYLE_INPUTS)
        self.assertEqual(common.common_settings()["learned_code"], plain["code"])      # they fit the code's spare width
        x = common.observe(bars, mood)
        np.testing.assert_array_equal(x[:, :plain["inputs"]], heard)
        self.assertFalse(x[:, plain["inputs"]:].any())
        self.assertEqual(common.pathway_slots(hierarchy.TRIO_ARCHITECTURE),
                         hierarchy.owners(hierarchy.TRIO_ARCHITECTURE, common.STEPS, common.SYN // 3, timbre.width(common.PATCH_BOOK)))
        self.assertNotEqual(common.layout_key({"reserve": "inputs"}), common.layout_key({"reserve": "sid"}))

    def test_room_for_kits_widens_the_kit_slot_alone(self):
        plain = dict(slots=common.SLOTS, actions=common.ACTIONS, sizes=list(common.SIZES))
        self.configure("kits")
        self.assertEqual((common.SLOTS, common.LIVE), (plain["slots"], plain["slots"]))
        self.assertEqual(common.SIZES[0], common.KIT_CLASSES)
        self.assertEqual(list(common.SIZES[1:]), plain["sizes"][1:])
        self.assertEqual(common.ACTIONS, plain["actions"] + common.KIT_CLASSES - common.T)
        self.assertEqual(common.RESERVED_INPUTS, common.EAR_INPUTS + common.STYLE_INPUTS)
        self.assertEqual(common.common_settings()["learned_code"], 1856)
        self.assertNotEqual(common.layout_key({"reserve": "kits"}), common.layout_key({"reserve": "inputs"}))

    @unittest.skipUnless(HAS_SID_POOL, NO_SID_POOL)
    def test_sid_lessons_label_every_sounding_bar_with_its_sounds_sid_fit(self):
        sounds = json.loads(SID_POOL.read_text())
        book = timbre.sid_book(sounds)
        self.assertEqual([len(table) for table in book["layers"]], [len(layer["tokens"]) for layer in self.pool["layers"][1:]])
        #                 kit bass lead air
        tokens = np.array([[1, 0, 3, 0], [1, 5, 3, 0], [1, 0, 0, 0], [1, 7, 4, 0]])
        blocks = timbre.sid_controls(tokens, book).reshape(4, 3, timbre.SID_WIDTH)
        self.assertFalse(blocks[0, 0].any())                                         # off before the bass first sounds
        self.assertEqual(blocks[1, 0].tolist(), timbre.sid_encode(sounds["layers"][1]["tokens"][5]))
        self.assertEqual(blocks[2, 0].tolist(), blocks[1, 0].tolist())               # carried through a silent bar
        self.assertEqual(blocks[3, 0].tolist(), timbre.sid_encode(sounds["layers"][1]["tokens"][7]))
        self.assertEqual(blocks[3, 1].tolist(), timbre.sid_encode(sounds["layers"][2]["tokens"][4]))
        self.assertFalse(blocks[:, 2].any())                                         # the air never sounds here
        # a score loaded with the fits has them in its bars, heard while the layer sounds, and a reading of their own
        import gate
        self.configure("sid", live=True)
        bars = self.bars(3)
        bars[:, common.LIVE:] = np.tile(np.concatenate([timbre.sid_encode(sounds["layers"][i]["tokens"][2]) for i in (1, 2, 3)]), (3, 1))
        mood = np.full((3, common.MOOD), 0.5)
        x = common.observe(bars, mood)
        first = common.OFFSETS[common.LIVE]
        self.assertTrue((x[:, first:common.ACTIONS].sum(axis=1) == 3 * timbre.SID_WIDTH).all())
        knob = common.L + common.MODS + common.STEPS + common.SYN + 3
        self.assertTrue(x[0, common.OFFSETS[knob]:common.OFFSETS[knob + len(timbre.NAMES)]].any())   # the ordinary knobs too
        reading = gate.prediction_error(None, [{"bars": bars, "mood": mood}])
        self.assertEqual(reading["sid"], 0.0)                                        # repeating a held fit is right
        other = bars.copy(); other[1:, common.LIVE + 5] += 1
        self.assertGreater(gate.prediction_error(None, [{"bars": other, "mood": mood}])["sid"], 0.0)
        token, _ = None, None
        common.SID_RENDER = True
        played = common.score_bar(bars[0], self.pool, [None] * 3, [0] * 4)
        self.assertEqual([patch["voice"] for patch in played["patches"][1:]], ["sid"] * 3)

    def test_the_ear_hears_each_voice_against_the_drums(self):
        heard = common.ear_inputs([-18.0, -18.0, -38.0, None], [-22.0, -20.0, -52.0, -60.0], -16.0, -21.0)
        self.assertEqual(len(heard), common.EAR_INPUTS)
        named = dict(zip(common.EAR_NAMES, heard))
        self.assertAlmostEqual(named["bass"], 1.0)                 # as loud as the kit
        self.assertAlmostEqual(named["lead"], 0.0)                 # twenty dB under it
        self.assertEqual((named["bass sounds"], named["lead sounds"], named["air sounds"]), (1.0, 1.0, 0.0))
        self.assertEqual((named["air"], named["air over the bar"]), (0.0, 0.0))        # a silent voice is not heard
        self.assertAlmostEqual(named["lead over the bar"], (-52.0 + 22.0 + 20) / 20)
        self.assertAlmostEqual(named["drums"], -0.2)               # two dB under their median
        self.assertEqual(named["drums sound"], 1.0)
        self.assertFalse(heard[len(common.EAR_NAMES):].any())
        without = dict(zip(common.EAR_NAMES, common.ear_inputs([None, -30.0, None, None], [-89.0, -31.0, -80.0, -80.0], -16.0, -21.0)))
        self.assertAlmostEqual(without["bass"], (-30.0 + 16.0 + 20) / 20)              # no drums: against their median
        self.assertEqual((without["drums"], without["drums sound"]), (0.0, 0.0))
        # in a score with an ear, what was heard of bar t travels with the mood row of bar t + 1 into the ear inputs
        self.configure("inputs")
        common.EAR_LIVE = True
        try:
            levels = [[-18.0, -18.0, -38.0, None], [-18.0, None, -28.0, -30.0], [-17.0, -20.0, None, None]]
            whole = [[-22.0, -20.0, -52.0, -60.0], [-22.0, -70.0, -40.0, -35.0], [-21.0, -24.0, -70.0, -70.0]]
            ear = common.ear_track(levels, whole)
            self.assertEqual(ear.shape, (3, common.EAR_INPUTS))
            mood = np.hstack([np.full((3, common.MOOD), 0.5), np.vstack([np.zeros((1, common.EAR_INPUTS)), ear[:-1]])])
            bars = self.bars(3)
            x = common.observe(bars[:2], mood[1:3])
            first = common.ACTIONS + common.MOOD
            np.testing.assert_allclose(x[:, first:first + common.EAR_INPUTS], ear[:2])
            np.testing.assert_allclose(x[:, common.ACTIONS:first], 0.5)
            self.assertFalse(x[:, first + common.EAR_INPUTS:].any())              # the style inputs stay idle
            plain = common.observe(bars[:2], np.full((2, common.MOOD), 0.5))      # a mood row without an ear: silent ears
            self.assertFalse(plain[:, first:].any())
        finally:
            common.EAR_LIVE = False

    def test_heads_are_born_listening_at_their_lags(self):
        import check
        from cadence import Brain

        def attended(brain):
            rng = np.random.default_rng(0)
            bars = rng.integers(0, check.VALUES, (1, 10, check.BENCH_SLOTS))
            brain.reset()
            for t in range(9):
                x = check.bench_observe(bars[:, t])
                brain.act(x, greedy=True)
            brain.act(x, greedy=True)                           # the answer on bar 8: the heads read
            return [int(np.asarray(a)[0].argmax()) for a in brain.recall.attended_heads]

        # the route layout (generation 1): each head's recall port into the motor starts at its gain
        brain, _ = check.bench_brain("split_route", 3)
        common.born_lags(brain, [1, 2, 4, None], [0.1, 0.05, 0.075, 0.0], bias=1e4)
        np.testing.assert_allclose(np.asarray(brain.recall.gains), [0.1, 0, 0.05, 0, 0.075, 0, 0, 0])
        self.assertEqual(attended(brain)[:3], [8, 7, 5])        # the bars that repeat at lags 1, 2 and 4
        # the slate (generation 2) has no route: no recall port gains, and the heads at the coordinator still read
        # the bars at their lags
        brain, _ = check.bench_brain("split", 3)
        common.born_lags(brain, [1, 2, 4, None], [0.1, 0.05, 0.075, 0.0], bias=1e4)
        self.assertEqual(np.asarray(brain.recall.gains).size, 0)
        self.assertEqual(attended(brain)[:3], [8, 7, 5])
        # the settings travel through compose, are recorded, and a saved copy keeps them
        self.configure(False)
        made, settings = common.compose(hierarchy.ROUTE_PRESET, backend="cpu", learned_code=48, modules=(16, 12, 16, 12, 10),
                                        recall_tokens=None, recall_gain=0.0,
                                        recall_lags_born=[1, 2, 4, 8], recall_lag_gains_born=[0.0, 0.0, 0.0, 0.0])
        self.assertEqual(settings["recall_lags_born"], [1, 2, 4, 8])
        lagged = [np.flatnonzero(np.asarray(m[2])).tolist() for m in made.recall.position_maps]
        self.assertEqual(lagged, [[0], [1], [3], [7]])

    def test_a_reading_answers_its_passages_side_by_side(self):
        import gate

        self.configure("inputs")
        brain, _ = common.compose(hierarchy.ROUTE_PRESET, backend="cpu", learned_code=common.common_settings()["learned_code"],
                                  modules=(16, 12, 16, 12, 10), recall_gain_born=0.25)
        tracks = [{"bars": self.bars(n, seed), "mood": np.full((n, common.MOOD), 0.5)} for seed, n in enumerate((6, 4, 5))]
        common.HEAR_FIRST = True
        try:
            alone = gate.next_bars(brain, tracks, streams=1)
            together = gate.next_bars(brain, tracks, streams=3)
            self.assertEqual([len(a) for a in together], [5, 3, 4])            # one answer per bar that has a next
            for one, side in zip(alone, together):
                np.testing.assert_array_equal(np.asarray(one), np.asarray(side))
            reading = gate.prediction_error(brain, tracks)
            self.assertEqual(reading["change"]["balanced"] is not None, True)
            self.assertAlmostEqual(gate.prediction_error(None, tracks)["change"]["balanced"], 1.0)
            self.assertEqual(gate.prediction_error(None, tracks)["stay"]["balanced"], 0.0)
            self.assertEqual(set(reading["levels"]), {"balanced", "drum_levels", "synth_level", "synth_accents"})
        finally:
            common.HEAR_FIRST = False

    def test_an_open_drum_step_neither_teaches_nor_counts(self):
        import gate

        self.configure("inputs")
        first = common.L + common.MODS
        step = first + 20                                      # a snare step
        heard = self.bars(1)[0]
        heard[0], heard[step] = 1, 0                           # the kit sounds, the step is silent in the bar heard
        taught = heard.copy()
        taught[step] = 1                                       # the reader's undecided hit in the next bar
        opened = np.zeros((2, common.STEPS), dtype=bool)
        opened[1, 20] = True
        mood = np.full((2, common.MOOD), 0.5)
        raw = {"drums_open": opened.reshape(2, 3, 16).astype(int).tolist(), "drums_sure": [[1, 1, 1]] * 2, "synth_sure": [[0, 0, 0]] * 2}
        marks = common.sure_marks(raw, 2)
        np.testing.assert_array_equal(marks[common.OPEN], opened)
        track = {"bars": np.stack([heard, taught]), "mood": mood, **marks}
        plain = {"bars": np.stack([heard, taught]), "mood": mood}
        # the counts: repeating is wrong on the step where it counts and is not asked where it is open
        self.assertGreater(gate.prediction_error(None, [plain])["drum_rows"], 0.0)
        self.assertEqual(gate.prediction_error(None, [track])["drum_rows"], 0.0)
        self.assertEqual(gate.prediction_error(None, [track])["sure"]["drum_rows"], 0.0)
        cut = common.passage(dict(track, bars=np.stack([heard, heard, taught]), mood=np.full((3, common.MOOD), 0.5),
                                  **{common.OPEN: np.concatenate([opened[:1], opened])}), 1, 2)
        np.testing.assert_array_equal(cut[common.OPEN], opened)                 # a passage keeps its marks
        self.assertEqual(len(common.windows([track], 2, 1)[0]), 3)              # and a lesson window carries them
        self.assertEqual(len(common.windows([plain], 2, 1)[0]), 2)
        # the labels: with the direct route the step is taught as heard, with open labels it is left open
        self.assertEqual(common.open_steps(taught[None], heard[None], opened[1:], leave=False)[0, step], heard[step])
        self.assertEqual(common.open_steps(taught[None], heard[None], opened[1:], leave=True)[0, step], -1)
        np.testing.assert_array_equal(np.delete(common.open_steps(taught[None], heard[None], opened[1:], leave=False)[0], step),
                                      np.delete(taught, step))
        # the lesson: taught as written the brain moves toward the hit; with the step open it does not
        common.HEAR_FIRST = True
        try:
            reached = {}
            for name, label in (("taught", taught[None]), ("open", common.open_steps(taught[None], heard[None], opened[1:], leave=False))):
                brain, _ = common.compose(hierarchy.ROUTE_PRESET, backend="cpu", seed=5, learned_code=common.common_settings()["learned_code"],
                                          modules=(16, 12, 16, 12, 10), recall_gain_born=0.25)
                x = common.observe(heard[None], mood[:1])
                for _ in range(12):
                    brain.reset()
                    common.hear(brain, x)
                    brain.lesson(x, label)
                brain.reset()
                common.answer(brain, x)
                reached[name] = float(common.motor_activity(brain)[0, common.OFFSETS[step] + 1])
            self.assertGreater(reached["taught"], reached["open"])
        finally:
            common.HEAR_FIRST = False

    def test_an_open_line_step_neither_teaches_nor_counts(self):
        import gate

        self.configure("inputs")
        first = common.L + common.MODS
        step = first + common.STEPS + 16 + 5                   # the lead's sixth step
        heard = self.bars(1)[0]
        heard[0], heard[2] = 1, 1                              # the kit and the lead sound
        heard[step] = 0                                        # a rest in the bar heard
        taught = heard.copy()
        taught[step] = 7                                       # the reader's undecided note in the next bar
        lines = np.zeros((2, common.SYN), dtype=bool)
        lines[1, 16 + 5] = True
        mood = np.full((2, common.MOOD), 0.5)
        raw = {"synth_open": lines.reshape(2, 3, 16).astype(int).tolist(), "drums_sure": [[1, 1, 1]] * 2,
               "synth_sure": [[1, 1, 1]] * 2}
        marks = common.sure_marks(raw, 2)
        np.testing.assert_array_equal(marks[common.LINE_OPEN], lines)
        track = {"bars": np.stack([heard, taught]), "mood": mood, **marks}
        plain = {"bars": np.stack([heard, taught]), "mood": mood}
        self.assertGreater(gate.prediction_error(None, [plain])["synth_line"], 0.0)
        reading = gate.prediction_error(None, [track])
        self.assertEqual(reading["synth_line"], 0.0)                              # counted for nobody
        self.assertAlmostEqual(reading["open_lines"]["lead"], 1 / 16)
        window = common.windows([track], 2, 1)[0]
        self.assertEqual(window[2].shape, (2, common.STEPS + common.SYN))         # drum steps, then line steps
        self.assertTrue(window[2][1, common.STEPS + 16 + 5])
        label = common.open_steps(taught[None], heard[None], window[2][1:], leave=False)
        self.assertEqual(label[0, step], heard[step])                             # taught as the step heard
        np.testing.assert_array_equal(np.delete(label[0], step), np.delete(taught, step))
        self.assertEqual(common.open_steps(taught[None], heard[None], window[2][1:], leave=True)[0, step], -1)

    def test_an_unread_sound_neither_teaches_nor_counts(self):
        import gate

        self.configure("inputs")
        start = common.L + common.MODS + common.STEPS + common.SYN
        width = (common.LIVE - start) // 3
        heard = self.bars(1)[0]
        heard[0], heard[1], heard[2] = 1, 1, 1                 # the kit, the bass and the lead sound
        taught = heard.copy()
        lead = 1
        cutoff = start + lead * width + 3 + timbre.NAMES.index("cutoff")
        voice = start + lead * width
        taught[cutoff] = (heard[cutoff] + 1) % 7               # the matched sound's cutoff moves in the next bar
        taught[common.L + 2 * (lead + 1)] += 1                 # and the rule's filter level of the lead
        unread = np.array([[False, False, False], [False, True, False]])
        mood = np.full((2, common.MOOD), 0.5)
        marks = common.sure_marks({"controls_open": unread.astype(int).tolist()}, 2)
        np.testing.assert_array_equal(marks[common.CONTROLS_OPEN], unread)
        np.testing.assert_array_equal(common.sure_marks({"controls_open": [0, 1]}, 2)[common.CONTROLS_OPEN],
                                      np.array([[0, 0, 0], [1, 1, 1]], dtype=bool))     # one flag per bar: every layer
        self.assertIn(cutoff, common.control_slots(lead))
        self.assertIn(common.L + 2 * (lead + 1), common.control_slots(lead))
        self.assertNotIn(voice, common.control_slots(lead))                            # voice and register stay
        self.assertNotIn(voice + 1, common.control_slots(lead))
        track = {"bars": np.stack([heard, taught]), "mood": mood, **marks}
        plain = {"bars": np.stack([heard, taught]), "mood": mood}
        self.assertGreater(gate.prediction_error(None, [plain])["synth_controls"], 0.0)
        reading = gate.prediction_error(None, [track])
        self.assertEqual(reading["synth_controls"], 0.0)                               # counted for nobody
        self.assertAlmostEqual(reading["open_sounds"]["lead"], 1.0)
        self.assertAlmostEqual(reading["open_sounds"]["bass"], 0.0)
        switched = taught.copy()
        switched[voice] = (heard[voice] + 1) % 3                                       # a change of voice still counts
        self.assertGreater(gate.prediction_error(None, [dict(track, bars=np.stack([heard, switched]))])["synth_controls"], 0.0)
        window = common.windows([track], 2, 1)[0]
        self.assertEqual(window[2].shape, (2, common.STEPS + common.SYN + 3))       # drum steps, line steps, layers
        label = common.open_steps(taught[None], heard[None], window[2][1:], leave=False)
        np.testing.assert_array_equal(label[0], heard)                                 # taught as heard
        left = common.open_steps(taught[None], heard[None], window[2][1:], leave=True)[0]
        self.assertTrue((left[common.control_slots(lead)] == -1).all())
        np.testing.assert_array_equal(np.delete(left, common.control_slots(lead)), np.delete(taught, common.control_slots(lead)))

    def test_a_bar_heard_as_two_moments_gives_the_older_lags_its_rows_alone(self):
        import check

        self.configure(False)
        bars = self.bars(2)
        x = common.observe(bars, np.full((2, common.MOOD), 0.5))
        fast = common.fast_part(x)
        rows = slice(common.OFFSETS[common.L + common.MODS], common.OFFSETS[common.L + common.MODS + common.STEPS + common.SYN])
        np.testing.assert_array_equal(fast[:, rows], x[:, rows])                    # drum rows and synth lines as heard
        np.testing.assert_array_equal(fast[:, common.ACTIONS:], x[:, common.ACTIONS:])  # mood and clock as heard
        self.assertEqual(fast[:, :rows.start].sum() + fast[:, rows.stop:common.ACTIONS].sum(), 0)
        self.assertGreater(x[:, :rows.start].sum(), 0)
        check.PARTS = common.HEAR_PARTS = True
        try:
            self.assertEqual([common.lag_age(lag) for lag in (1, 2, 4, 8)], [0, 3, 7, 15])
            brain, _ = check.bench_brain("split", 3)
            common.born_lags(brain, [1, 2, 4, None], [0.25, 0.25, 0.25, 0.0])
            rng = np.random.default_rng(0)
            bars = rng.integers(0, check.VALUES, (1, 10, check.BENCH_SLOTS))
            brain.reset()
            for t in range(9):
                x = check.bench_observe(bars[:, t])
                check.bench_hear(brain, x, check.BENCH_SLOTS)
            brain.act(x, greedy=True)                           # the answer on bar 8: the heads read
            attended = [int(np.asarray(a)[0].argmax()) for a in brain.recall.attended_heads]
            # bar t is moment 2t (its rows alone) and moment 2t + 1 (the whole bar)
            self.assertEqual(attended[:3], [17, 14, 10])        # bar 8 whole; the rows of bars 7 and 5
        finally:
            check.PARTS = common.HEAR_PARTS = False

    def test_a_synths_sid_block_belongs_to_the_pathway_of_its_sound(self):
        self.configure(True)
        chip = lambda i: set(range(common.LIVE + i * timbre.SID_WIDTH, common.LIVE + (i + 1) * timbre.SID_WIDTH))
        for architecture, homes in ((hierarchy.ARCHITECTURE, ("melody",) * 3),
                                    (hierarchy.INSTRUMENT_ARCHITECTURE, ("bass", "lead", "air")),
                                    (hierarchy.TRIO_ARCHITECTURE, ("bass", "melody", "melody")),
                                    (hierarchy.PARTS_ARCHITECTURE, ("bass_sound", "lead_sound", "air_sound"))):
            with self.subTest(architecture=architecture):
                parts = common.pathway_slots(architecture)
                self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(common.SLOTS)))
                for i, home in enumerate(homes):
                    self.assertLessEqual(chip(i), set(parts[home]))

    @unittest.skipUnless(HAS_SID_POOL, NO_SID_POOL)
    def test_the_lead_only_shape_has_one_block_on_the_lead(self):
        import gate
        sounds = json.loads(SID_POOL.read_text())
        book = timbre.sid_book(sounds)
        self.configure("sid")
        full = common.SLOTS
        self.configure("sidlead", live=True)
        self.assertEqual(common.chip_layers(), (1,))
        self.assertEqual(common.SLOTS, full - 2 * timbre.SID_WIDTH)
        lead = set(range(common.LIVE, common.SLOTS))
        for architecture, home in ((hierarchy.ARCHITECTURE, "melody"), (hierarchy.INSTRUMENT_ARCHITECTURE, "lead"),
                                   (hierarchy.TRIO_ARCHITECTURE, "melody"), (hierarchy.PARTS_ARCHITECTURE, "lead_sound")):
            with self.subTest(architecture=architecture):
                parts = common.pathway_slots(architecture)
                self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(common.SLOTS)))
                self.assertLessEqual(lead, set(parts[home]))
        #                 kit bass lead air
        tokens = np.array([[1, 5, 0, 0], [1, 5, 3, 0], [1, 7, 4, 2]])
        blocks = timbre.sid_controls(tokens, book, common.chip_layers())
        self.assertEqual(blocks.shape, (3, timbre.SID_WIDTH))
        self.assertFalse(blocks[0].any())                                             # off before the lead sounds
        self.assertEqual(blocks[2].tolist(), timbre.sid_encode(sounds["layers"][2]["tokens"][4]))
        bars = self.bars(3)
        bars[:, common.LIVE:] = blocks[2]
        mood = np.full((3, common.MOOD), 0.5)
        x = common.observe(bars, mood)
        self.assertEqual(int(x[:, common.OFFSETS[common.LIVE]:common.ACTIONS].sum(axis=1).max()), timbre.SID_WIDTH)
        self.assertEqual(gate.prediction_error(None, [{"bars": bars, "mood": mood}])["sid"], 0.0)
        common.SID_RENDER = True
        played = common.score_bar(bars[0], self.pool, [None] * 3, [0] * 4)
        self.assertEqual([(patch or {}).get("voice") for patch in played["patches"][1:]][1], "sid")
        self.assertNotEqual([(patch or {}).get("voice") for patch in played["patches"][1:]][0], "sid")
        card = {"settings": {"slots": list(common.SIZES)}, "patch_book": common.PATCH_BOOK, "steps": True,
                "reserve": "sidlead"}
        kinds, _, _, _, sid, _ = anatomy.slot_kinds(card)
        self.assertEqual((kinds.count("reserved"), sid), (timbre.SID_WIDTH, (0, timbre.SID_WIDTH, 0)))

    def test_the_plan_summarises_the_eight_bars_after_each_bar(self):
        import gate
        self.configure("inputs")
        plain = (common.SLOTS, common.ACTIONS)
        common.LOOKAHEAD = True
        try:
            self.configure("inputs")
            self.assertEqual(common.PLAN, len(common.PLAN_SIZES))
            self.assertEqual((common.SLOTS, common.ACTIONS), (plain[0] + common.PLAN, plain[1] + sum(common.PLAN_SIZES)))
            self.assertEqual(common.chip_end(), common.LIVE)
            for architecture, home in ((hierarchy.ARCHITECTURE, "melody"), (hierarchy.INSTRUMENT_ARCHITECTURE, hierarchy.COORDINATOR),
                                       (hierarchy.TRIO_ARCHITECTURE, hierarchy.COORDINATOR)):
                with self.subTest(architecture=architecture):
                    parts = common.pathway_slots(architecture)
                    self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(common.SLOTS)))
                    self.assertLessEqual(set(range(common.chip_end(), common.SLOTS)), set(parts[home]))
            # twelve bars: the bass enters at bar 3, the kick leaves at bar 6 and comes back at bar 8, the air never plays
            bars = self.bars(12)[:, :common.chip_end()]
            bars[:, 0] = 1
            bars[:, 1:4] = 0
            bars[3:, 1] = 1
            bars[:, 2] = 1
            kick = slice(common.L + common.MODS, common.L + common.MODS + common.STEPS // 3)
            bars[:, kick] = 0
            bars[:, kick.start] = 1
            bars[6:8, kick] = 0
            plan = common.plan_rows(bars)
            self.assertEqual(plan.shape, (12, common.PLAN))
            beat, bass, lead, air, changes, quiet = plan.T
            self.assertEqual(bass[0], 1)              # bars 1..8: the bass plays in some of them
            self.assertEqual(bass[2], 2)              # bars 3..10: in all of them
            self.assertEqual((lead[0], air[0]), (2, 0))
            self.assertEqual(changes[0], 1)           # one change of which voices sound (bar 3)
            self.assertEqual(changes[3], 0)
            self.assertEqual(quiet[0], 2)             # the first kickless bar is bar 6: the sixth bar ahead
            self.assertEqual(quiet[3], 1)             # from bar 3 it is the third bar ahead
            self.assertEqual(quiet[8], 0)             # no kickless bar after bar 8
            self.assertEqual(plan[11].tolist(), plan[10].tolist())   # the last bar keeps the plan before it
            full = np.concatenate([bars, plan], axis=1)
            mood = np.full((12, common.MOOD), 0.5)
            x = common.observe(full, mood)
            self.assertTrue((x[:, common.OFFSETS[common.chip_end()]:common.ACTIONS].sum(axis=1) == common.PLAN).all())
            reading = gate.prediction_error(None, [{"bars": full, "mood": mood}])
            self.assertIn("plan", reading)
            card = {"settings": {"slots": list(common.SIZES)}, "patch_book": common.PATCH_BOOK, "steps": True,
                    "reserve": "inputs", "lookahead": True}
            kinds, *_, plan_slots = anatomy.slot_kinds(card)
            self.assertEqual((kinds.count("plan"), plan_slots), (common.PLAN, common.PLAN))
        finally:
            common.LOOKAHEAD = False
            self.configure("inputs")

    def test_the_section_slot_follows_the_plan_and_is_heard(self):
        import gate
        self.configure("inputs")
        plain = (common.SLOTS, common.ACTIONS)
        common.LOOKAHEAD = common.SECTION = True
        try:
            self.configure("inputs")
            self.assertEqual(common.SECTIONS, 1)
            self.assertEqual(common.SLOTS, plain[0] + common.PLAN + 1)
            self.assertEqual(common.ACTIONS, plain[1] + sum(common.PLAN_SIZES) + len(common.SECTION_CLASSES))
            self.assertEqual((common.chip_end(), common.section_slot()), (common.LIVE, common.SLOTS - 1))
            parts = common.pathway_slots(hierarchy.ARCHITECTURE)
            self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(common.SLOTS)))
            self.assertIn(common.section_slot(), parts["melody"])
            self.assertEqual(common.section_rows(["normal", "breakdown", "build", "breakdown"], 4).tolist(), [0, 1, 0, 1])
            self.assertEqual(common.section_rows(None, 3).tolist(), [0, 0, 0])
            with self.assertRaises(ValueError):
                common.section_rows(["normal"], 2)
            bars = self.bars(12)[:, :common.chip_end()]
            plan = common.plan_rows(bars)
            section = common.section_rows(["normal"] * 6 + ["breakdown"] * 4 + ["normal"] * 2, 12)[:, None]
            full = np.concatenate([bars, plan, section], axis=1)
            mood = np.full((12, common.MOOD), 0.5)
            x = common.observe(full, mood)
            heard = x[:, common.OFFSETS[common.section_slot()]:common.ACTIONS]
            self.assertEqual(heard.argmax(axis=1).tolist(), section[:, 0].tolist())
            self.assertTrue((heard.sum(axis=1) == 1).all())
            reading = gate.prediction_error(None, [{"bars": full, "mood": mood}])
            self.assertIn("section", reading)
            self.assertIn("plan", reading)
            card = {"settings": {"slots": list(common.SIZES)}, "patch_book": common.PATCH_BOOK, "steps": True,
                    "reserve": "inputs", "lookahead": True, "section": True}
            kinds, *_, plan_slots = anatomy.slot_kinds(card)
            self.assertEqual(plan_slots, common.PLAN + 1)
        finally:
            common.LOOKAHEAD = common.SECTION = False
            self.configure("inputs")

    def test_a_brain_of_the_reserved_shape_learns_and_is_drawn(self):
        self.configure(True)
        brain, settings = common.compose(hierarchy.INSTRUMENT_PRESET, backend="cpu", learned_code=48,
                                         modules=(16, 12) * 4 + (10,))
        bars = self.bars(3)
        mood = np.full((3, common.MOOD), 0.5)
        x = common.observe(bars[:2], mood[:2])
        brain.act(x, greedy=True)
        brain.lesson(x, bars[1:])
        self.assertEqual(np.asarray(brain.act(x, greedy=True)).reshape(2, -1).shape[1], common.SLOTS)
        card = {"settings": {k: v for k, v in settings.items() if k != "plastic_synapses"}, "seed": 0, "steps": True,
                "onoff": False, "patch_book": common.PATCH_BOOK, "reserve": True}
        drawn = anatomy.describe(card)
        self.assertEqual(drawn["neurons"], brain.connectome.n)
        self.assertEqual(sum(output["slots"] for output in drawn["outputs"]), common.SLOTS)
        reserved = {output["name"]: output["parts"].get("reserved", 0) for output in drawn["outputs"]}
        self.assertEqual(reserved, {"coordinator": 0, "beat": 0, "bass": timbre.SID_WIDTH, "lead": timbre.SID_WIDTH,
                                    "air": timbre.SID_WIDTH})
        # the same card without the mark describes another brain: the mark is part of the layout
        self.assertNotEqual(common.layout_key({**card, "tokens_per_layer": 65}),
                            common.layout_key({**card, "tokens_per_layer": 65, "reserve": False}))

    @unittest.skipUnless(HAS_SID_POOL, NO_SID_POOL)
    def test_a_sid_block_holds_a_fitted_sid_sound(self):
        sounds = json.loads(SID_POOL.read_text())
        self.assertEqual(len(timbre.SID_NAMES), 36)
        self.assertIsNone(timbre.sid_decode(timbre.sid_off()))
        for layer in sounds["layers"][1:]:
            for token in [t for t in layer["tokens"] if t][:8]:
                block = timbre.sid_encode(token)
                self.assertEqual(block[0], 1)
                self.assertTrue(all(0 < value < size for value, size in zip(block[1:], timbre.SID_SIZES[1:])))
                back = timbre.sid_decode(block)["patch"]
                for name in timbre.SID_NAMES:
                    levels = timbre.SID_LEVELS[name.rstrip("123")]
                    self.assertLessEqual(abs(back[name] - token["patch"][name]) * (levels - 1), 0.5 + 1e-9)
        # with SID lessons a block whose switch is on is heard while its layer sounds
        self.configure(True, live=True)
        bars = self.bars(1)
        bars[0, common.LIVE:common.LIVE + timbre.SID_WIDTH] = timbre.sid_encode(sounds["layers"][1]["tokens"][1])
        x = common.observe(bars, np.full((1, common.MOOD), 0.5))
        first = common.OFFSETS[common.LIVE]
        self.assertEqual(x[0, first:first + sum(timbre.SID_SIZES)].sum(), timbre.SID_WIDTH)

    def test_the_wide_layouts_keep_their_wiring(self):
        import layouts

        self.assertEqual(set(layouts.STAGE1) - set(layouts.LAYOUTS) - set(layouts.S1_LAYOUTS), set())
        # the narrow wirings of the wide layouts are the route layouts (the generation-2 slate took the names split and
        # inst_reads for its own wirings)
        for name, narrow in (("split", "split_route"), ("inst_reads", "inst_reads_route"), ("trio", "trio")):
            preset, over = layouts.layout(narrow)
            wide, wide_over = layouts.layout(name + "_wide")
            self.assertEqual(hierarchy.PRESETS[wide], hierarchy.PRESETS[preset])
            self.assertEqual(wide_over, over)
            self.assertEqual(tuple(common.PRESETS[wide]["modules"]),
                             tuple(hierarchy.WIDE * width for width in common.PRESETS[preset]["modules"]))
            narrow = {k: v for k, v in common.PRESETS[preset].items() if k != "modules"}
            self.assertEqual({k: v for k, v in common.PRESETS[wide].items() if k != "modules"}, narrow)

    def test_the_bench_has_idle_parts_and_a_second_rule_for_them(self):
        import check

        plain, none = check.bench_tracks(8, 11, check.rule_slots())
        idle, quiet = check.bench_tracks(8, 11, check.rule_slots(), "sid")
        later, signal = check.bench_tracks(8, 11, check.rule_slots(), "sid", True)
        self.assertIsNone(none)
        np.testing.assert_array_equal(idle[:, :, :check.BENCH_SLOTS], plain)
        np.testing.assert_array_equal(later[:, :, :check.BENCH_SLOTS], plain)
        self.assertFalse(idle[:, :, check.BENCH_SLOTS:].any() or quiet.any())
        self.assertTrue((signal[:, 1:] != signal[:, :-1]).all())                 # a new signal every bar
        chosen = later[:, :, check.later_slots("sid")]
        self.assertTrue(((chosen > 0).all(axis=2) == (signal > 0)).all())       # off without a signal
        for value in range(1, check.SIGNALS):                                   # one answer per signal value
            self.assertEqual(len({tuple(row) for row in chosen[signal == value]}), 1)
        x = check.bench_observe(idle[:, 0], None, "sid")
        self.assertFalse(x[:, check.BENCH_SLOTS * check.VALUES:-check.IDLE_INPUTS - 2].any() or x[:, -check.IDLE_INPUTS:].any())
        x = check.bench_observe(later[:, 0], signal[:, 1], "sid")
        np.testing.assert_array_equal(x[:, -check.IDLE_INPUTS:].sum(axis=1), signal[:, 1] > 0)
        # idle inputs alone: the same bars, and in the second phase the sound controls follow the signal
        alone, quiet = check.bench_tracks(8, 11, check.rule_slots(), "inputs")
        moved, signal2 = check.bench_tracks(8, 11, check.rule_slots(), "inputs", True)
        np.testing.assert_array_equal(alone, plain)
        np.testing.assert_array_equal(signal2, signal)
        controls = check.later_slots("inputs")
        rest = [slot for slot in range(check.BENCH_SLOTS) if slot not in controls]
        np.testing.assert_array_equal(moved[:, :, rest], plain[:, :, rest])
        for value in range(check.SIGNALS):
            self.assertEqual(len({tuple(row) for row in moved[:, :, controls][signal2 == value]}), 1)
        self.assertEqual(check.bench_observe(moved[:, 0], signal2[:, 1], "inputs").shape[1],
                         check.BENCH_SLOTS * check.VALUES + 2 + check.IDLE_INPUTS)
        brain, _ = check.bench_brain("trio", 3, "sid")
        parts = check.bench_parts(hierarchy.TRIO_ARCHITECTURE, "sid")
        self.assertEqual(sorted(slot for slots in parts.values() for slot in slots), list(range(later.shape[2])))
        self.assertEqual(np.asarray(brain.act(x, greedy=True)).reshape(8, -1).shape[1], later.shape[2])
