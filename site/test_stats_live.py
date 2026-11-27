"""An existing stats connection sees file updates from the checkpoint watcher."""
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import serve


class StatsStreamTests(unittest.TestCase):
    def test_stats_keep_the_branch_and_coordinator_layout(self):
        widths = [112, 128, 96, 128, 64]
        point = serve.training_stats.point({}, "duet", {"settings": {
            "architecture": "rhythm-melody-v1", "modules": widths, "coordinator_depth": 2}})
        self.assertEqual(point["architecture"], "rhythm-melody-v1")
        self.assertEqual(point["modules"], widths)
        self.assertEqual(point["coordinator_depth"], 2)

    def test_every_candidate_carries_its_brain_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            installed = {"name": "duet_e001", "candidate": "duet", "patch_book": {"version": "dsp-v2"},
                         "streams_sha": "demo", "settings": {"modules": [8]}}
            (checkpoints / "duet_e001.json").write_text(json.dumps(installed))
            # A run that was never installed keeps its card beside its curve.
            run = root / "quality/composer/solo"
            run.mkdir(parents=True)
            row = {"checkpoint": "001", "minutes": 1, "train_error": {"balanced": .7},
                   "test_error": {"balanced": .8, "instruments": {"bass": .6}}}
            (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
            (run / "solo_e001.json").write_text(json.dumps({"name": "solo_e001", "settings": {"modules": [4]}}))
            # The trainer writes the layout with its first evaluation, before any card exists.
            fresh = root / "quality/composer/fresh"
            fresh.mkdir()
            first = dict(row, checkpoint="000", tag="untrained", anatomy={"architecture": "instruments-v1"})
            (fresh / "metrics.jsonl").write_text(json.dumps(first) + "\n")
            seen = []

            def describe(card):
                seen.append(card.get("name"))
                return {"widths": card["settings"]["modules"]}

            curves = serve.training_stats.curves(root, checkpoints, root / "pinned", describe=describe)
            layouts = {group["candidate"]: group.get("layout") for group in curves}
            self.assertEqual(layouts, {"duet": {"widths": [8]}, "solo": {"widths": [4]},
                                       "fresh": {"architecture": "instruments-v1"}})
            self.assertEqual(sorted(seen), ["duet_e001", "solo_e001"])
            solo = next(group for group in curves if group["candidate"] == "solo")
            self.assertEqual(solo["points"][0]["test_parts"]["instruments"], {"bass": .6})

    def test_a_run_s_newest_checkpoint_becomes_loadable_for_a_listen(self):
        import os
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "quality/composer/solo"
            run.mkdir(parents=True)
            for tag in ("e001", "e002"):
                (run / f"solo_{tag}.npz").write_bytes(tag.encode())
                (run / f"solo_{tag}.json").write_text(json.dumps({"name": f"solo_{tag}", "candidate": "solo"}))
            os.utime(run / "solo_e001.npz", (1, 1))
            calls = []

            def aws(arguments, **options):
                calls.append(arguments[1:3])
                if arguments[1:3] == ["s3", "cp"]:             # the fetch writes the file it was asked for
                    Path(arguments[4]).write_bytes(arguments[3].rsplit("/", 1)[1].encode())
                    return Mock(stdout="", stderr="", returncode=0)
                return Mock(stdout=bucket[0] if "/checkpoints/" in arguments[3] else "", returncode=0)

            bucket = [""]
            with (patch.multiple(serve, ROOT=root, CHECKPOINTS=root / "checkpoints", PINNED=root / "pinned",
                                 AUDITION=root / "audition"),
                  patch("subprocess.run", side_effect=aws)):
                # The newest checkpoint beside the local curve is copied where the player loads from.
                self.assertEqual(serve.audition_checkpoint("solo"), "solo_e002")
                self.assertEqual((root / "audition/solo_e002.npz").read_bytes(), b"e002")
                self.assertNotIn(["s3", "cp"], calls)
                # A newer one in the run's folder of the bucket is fetched.
                bucket[0] = "2099-01-01 00:00:00   34128416 solo_e003.npz\n2099-01-01 00:00:01     169490 solo_e003.json\n"
                self.assertEqual(serve.audition_checkpoint("solo"), "solo_e003")
                self.assertEqual((root / "audition/solo_e003.npz").read_bytes(), b"solo_e003.npz")
                self.assertEqual(serve._dir("audition"), root / "audition")
                with self.assertRaisesRegex(ValueError, "no checkpoint"):
                    serve.audition_checkpoint("absent")
                with self.assertRaisesRegex(ValueError, "unknown candidate"):
                    serve.audition_checkpoint("../solo")

    def test_the_menu_only_grows_by_itself_when_asked_to(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = root / "brains.json"
            entries.write_text("[]")
            installed = {"r8_new_e001": {"name": "r8_new_e001", "candidate": "r8_new", "dir": "checkpoints",
                                         "epoch": 1, "composition_score": .7, "quality": .7,
                                         "created": "2026-11-25T04:00:00Z"}}
            flag = root / ".auto-list"
            with patch.multiple(serve, BRAINS=entries, PLAYER_SELECTION=root / "selection", AUTO_LIST=flag, _last_auto=[0]):
                self.assertEqual(serve._brains(installed), [])
                flag.touch()
                serve._last_auto[0] = 0
                self.assertEqual([entry["name"] for entry in serve._brains(installed)], ["r8_new_e001"])

    def test_stats_tell_which_runs_have_a_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ("busy", "idle", "lost"):
                folder = root / "quality/composer" / run
                folder.mkdir(parents=True)
                row = {"checkpoint": "001", "minutes": 1, "train_error": {"balanced": .7}, "test_error": {"balanced": .8}}
                (folder / "metrics.jsonl").write_text(json.dumps(row) + "\n")
            box = {"instance": "i-1", "type": "g6.2xlarge", "since": "2026-11-25T03:52:15+00:00"}
            now = 1_000_000
            (root / "quality/composer/training-status-a.json").write_text(json.dumps(
                {"at": now - 30, "tag": "a", "done": False, "running": {"busy": box}}))
            # A mirror that stopped reporting with a box alive cannot say whether that run still trains.
            (root / "quality/composer/training-status-b.json").write_text(json.dumps(
                {"at": now - 3000, "tag": "b", "done": False, "running": {"lost": box}}))
            states, checked = serve.training_stats.training(root, now=now)
            self.assertEqual(states, {"busy": box, "lost": None})
            self.assertEqual(checked, now - 30)
            with (patch.multiple(serve, ROOT=root, CHECKPOINTS=root / "checkpoints", PINNED=root / "pinned"),
                  patch("time.time", return_value=now)):
                curves = serve.Handler._curves(None)
            self.assertEqual({brain["candidate"]: brain["training"] for brain in curves["brains"]},
                             {"busy": box, "idle": False, "lost": None})
            self.assertEqual(curves["mirror"], {"at": now - 30})

    def test_stats_show_the_slate_with_the_checks_made_on_the_present_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks = root / "quality/checks"
            checks.mkdir(parents=True)
            code = serve.check.code_sha()
            bench = {"passed": True, "hold": {"error": .06, "commonest": .75, "repeat": .02, "bar": .2, "lessons": 930, "curve": [[0, .75]]},
                     "rule": {"error": .1, "commonest": .75, "repeat": 1., "bar": .37, "lessons": 930}}
            (checks / "inst.json").write_text(json.dumps(
                {"layout": "inst", "code": code, "ready": True, "bench": bench,
                 "build": {"passed": True, "problems": [], "neurons": 34176, "synapses": 1}}))
            # a check made before the builder changed does not count
            (checks / "chain.json").write_text(json.dumps({"layout": "chain", "code": "older", "ready": True, "bench": bench}))
            run = root / "quality/composer/slate_inst_s1_a"
            run.mkdir(parents=True)
            rows = [{"checkpoint": "000", "minutes": 0, "train_error": {"balanced": .76}, "test_error": {"balanced": .75},
                     "layout": "inst", "blind_baseline": {"train": {"balanced": .53}, "test": {"balanced": .54}}},
                    {"checkpoint": "001", "minutes": 9, "train_error": {"balanced": .4}, "test_error": {"balanced": .5}}]
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch.multiple(serve, ROOT=root, CHECKPOINTS=root / "checkpoints", PINNED=root / "pinned"):
                curves = serve.Handler._curves(None)
            slate = {item["layout"]: item for item in curves["slate"]}
            self.assertEqual(list(slate), list(serve.layouts.LAYOUTS))
            self.assertTrue(slate["inst"]["ready"])
            self.assertEqual(slate["inst"]["stages"]["build"]["neurons"], 34176)
            self.assertEqual(slate["inst"]["stages"]["bench"]["hold"]["error"], .06)
            self.assertNotIn("curve", slate["inst"]["stages"]["bench"]["hold"])
            self.assertFalse(slate["chain"]["ready"])
            self.assertEqual(slate["chain"]["stages"], {})
            self.assertEqual(slate["parts"]["pathways"], 7)
            # the run names its layout and carries the commonest-bar level its curve is read against
            brain = curves["brains"][0]
            self.assertEqual(brain["slate"], "inst")
            self.assertEqual(brain["blind"]["test"]["balanced"], .54)

    def test_curves_are_read_again_only_after_a_file_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "quality/composer/steady"
            run.mkdir(parents=True)
            metrics = run / "metrics.jsonl"
            row = {"checkpoint": "001", "minutes": 1, "train_error": {"balanced": .7}, "test_error": {"balanced": .8}}
            metrics.write_text(json.dumps(row) + "\n")
            with patch.multiple(serve, ROOT=root, CHECKPOINTS=root / "checkpoints", PINNED=root / "pinned"):
                with patch.object(serve.training_stats, "curves", wraps=serve.training_stats.curves) as read:
                    first = serve.Handler._curves(None)["brains"]
                    self.assertEqual(serve.Handler._curves(None)["brains"], first)
                    self.assertEqual(read.call_count, 1)
                    with metrics.open("a") as file:
                        file.write(json.dumps(dict(row, checkpoint="002", minutes=2)) + "\n")
                    self.assertEqual(len(serve.Handler._curves(None)["brains"][0]["points"]), 2)
                    self.assertEqual(read.call_count, 2)

    def test_featured_playback_variant_keeps_source_training_stats(self):
        name = "r8_apical_detail_e007b016_returns"
        installed = {name: {"name": name, "dir": "pinned", "featured": True,
                            "candidate": "r8_apical_detail", "quality": .6}}
        entries = []
        self.assertTrue(serve._auto_add(entries, installed))
        self.assertEqual(entries[0]["name"], name)
        self.assertFalse(entries[0]["auto"])
        point = serve.training_stats.point({"test_error": {"balanced": .6}, "composition": .8}, name,
                                           {"source_checkpoint": "r8_apical_detail_e007b016",
                                            "playback_review": {"detail": .605}})
        self.assertEqual(point["test"], .6)
        self.assertIsNone(point["composition"])
        self.assertEqual(point["playback_review"]["detail"], .605)

    def test_saved_selection_recovers_from_automatic_supersession(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "r8_apical_chosen_e001"
            saved = root / "selection"
            saved.write_text(name)
            entries = root / "brains.json"
            entries.write_text(json.dumps([{"name": name, "auto": True, "deleted": True,
                                           "reason": "Training checkpoint · superseded"}]))
            with patch.multiple(serve, PLAYER_SELECTION=saved, BRAINS=entries, _last_auto=[0], AUTO_CAP=0):
                restored = serve._brains({name: {"name": name, "dir": "pinned"}})
            self.assertFalse(restored[0]["deleted"])
            self.assertEqual(restored[0]["dir"], "pinned")

    def test_playability_requires_the_checkpoint_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            name = "r8_apical_fixed_e001"
            card = {"name": name, "candidate": "r8_apical_fixed", "patch_book": {"version": "dsp-v2"},
                    "streams_sha": "demo", "cadence_patch_sha256": "correct"}
            (checkpoints / f"{name}.json").write_text(json.dumps(card))
            (checkpoints / f"{name}.npz").touch()
            (checkpoints / "index.json").write_text(json.dumps([card]))
            for runtime, playable in (("wrong", False), ("correct", True)):
                with patch.multiple(serve, CHECKPOINTS=checkpoints, PINNED=root / "pinned",
                                    SUPPORTED_CHECKPOINT_PATCH=runtime):
                    self.assertEqual(name in serve._installed(), playable)
                    points = serve.training_stats.curves(root, checkpoints, root / "pinned", True, runtime)[0]["points"]
                    self.assertEqual(points[0]["playable"], playable)

    def test_regular_composer_checkpoints_remain_listed_together(self):
        entries = []
        installed = {f"r8_apical_fixed_e{epoch:03d}": {"name": f"r8_apical_fixed_e{epoch:03d}",
                     "candidate": "r8_apical_fixed", "dir": "checkpoints", "epoch": epoch, "composition_score": .65,
                     "quality": .65, "created": f"2026-11-24T12:{epoch:02d}:00Z"} for epoch in (1, 2, 3)}
        self.assertTrue(serve._auto_add(entries, installed))
        self.assertEqual([e["name"] for e in entries if not e["deleted"]], list(installed))
        self.assertTrue(all(e["dir"] == "checkpoints" for e in entries))

    def test_stats_exposes_publication_time_and_uses_the_player_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            name = "r8_apical_fixture_e001"
            created = "2026-11-24T16:00:00Z"
            card = {"name": name, "candidate": "r8_apical_fixture", "created": created,
                    "patch_book": {"version": "dsp-v2"}, "streams_sha": "demo",
                    "errors": {"train_error": {"balanced": .66},
                               "test_error": {"balanced": .69, "detail": {"drum_accents": {"f1": .3}}}},
                    "composition": {"development": {"score": .4, "hold": .1}}}
            (checkpoints / f"{name}.json").write_text(json.dumps(card))
            (checkpoints / f"{name}.npz").touch()
            with patch.multiple(serve, ROOT=root, CHECKPOINTS=checkpoints, PINNED=root / "pinned"):
                pending = serve.Handler._curves(None)["brains"][0]["points"][0]
                self.assertEqual(pending["created"], created)
                self.assertFalse(pending["playable"])
                (checkpoints / "index.json").write_text(json.dumps([card]))
                ready = serve.Handler._curves(None)["brains"][0]["points"][0]
                self.assertTrue(ready["playable"])
                self.assertEqual(ready["test"], .69)
                self.assertEqual(ready["test_parts"]["detail"]["drum_accents"]["f1"], .3)
                self.assertEqual(ready["development"], {"score": .4, "hold": .1})

    def test_connection_delivers_a_new_evaluation_without_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "quality/composer" / "live_test"
            run.mkdir(parents=True)
            metrics = run / "metrics.jsonl"
            first = {"checkpoint": "001", "minutes": 1, "train_error": {"balanced": .75},
                     "test_error": {"balanced": .78}}
            metrics.write_text(json.dumps(first) + "\n")
            with patch.multiple(serve, ROOT=root, CHECKPOINTS=root / "checkpoints", PINNED=root / "pinned"):
                with ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler) as server:
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        url = f"http://127.0.0.1:{server.server_port}/api/curves/events"
                        with urllib.request.urlopen(url, timeout=3) as response:
                            self.assertEqual(response.headers.get_content_type(), "text/event-stream")

                            def checkpoints():
                                event = response.readline()
                                self.assertTrue(event.startswith(b"data: "))
                                self.assertEqual(response.readline(), b"\n")
                                return json.loads(event[6:])["brains"][0]["points"]

                            self.assertEqual([p["name"] for p in checkpoints()], ["live_test_e001"])
                            second = dict(first, checkpoint="002", minutes=2, test_error={"balanced": .62})
                            temp = metrics.with_suffix(".part")
                            temp.write_text(metrics.read_text() + json.dumps(second) + "\n")
                            temp.replace(metrics)
                            observed = checkpoints()
                            self.assertEqual([p["name"] for p in observed], ["live_test_e001", "live_test_e002"])
                            self.assertEqual(observed[-1]["test"], .62)
                    finally:
                        server.shutdown()
                        thread.join()
