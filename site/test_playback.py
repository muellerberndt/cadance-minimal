"""Regression tests for checkpoint switching and concurrent player requests."""
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import serve


class ObservedLayout:
    def __init__(self):
        self.lock = threading.RLock()
        self.requested = threading.Event()

    def acquire(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            self.requested.set()
        return self.lock.acquire(*args, **kwargs)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *_):
        self.release()


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    return SimpleNamespace(path=path, headers={"Content-Length": str(len(data))},
                           rfile=io.BytesIO(data), _send=Mock())


class PlaybackTests(unittest.TestCase):
    def test_stats_can_load_a_checkpoint_before_the_menu_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            name = "groove_e003"
            card = {"name": name, "candidate": "groove", "pool": "pool_v0.json"}
            weights = directory / f"{name}.npz"
            weights.touch()
            (directory / f"{name}.json").write_text(json.dumps(card))
            (directory / "index.json").write_text(json.dumps([card]))
            player = serve.Creature()
            with (patch.multiple(serve, CHECKPOINTS=directory, PINNED=directory / "pinned"),
                  patch.object(serve, "cards", return_value=[]),
                  patch.object(serve.cadence.Brain, "load") as load):
                loaded, _ = player._read(name)
            self.assertEqual(loaded["name"], name)
            load.assert_called_once_with(weights, backend="cpu")

    def test_groove_playback_feeds_the_played_bar_back_to_the_brain(self):
        creature = serve.Creature()
        creature.brain = object()
        creature.card = {"name": "groove"}
        creature.pool = {}
        creature.ensure_layout = Mock()
        creature.with_view = False
        creature.bar_pos = 24
        creature.state = np.zeros(serve.common.SLOTS, dtype=int).tolist()
        proposed = np.ones(serve.common.SLOTS, dtype=int)
        played = np.zeros_like(proposed)
        played[1] = 1
        creature.groove = Mock()
        creature.groove.play.return_value = played
        with (patch.object(serve, "observe", return_value=None) as observe,
              patch.object(serve, "sample_bar", return_value=proposed),
              patch.object(serve.common, "score_bar", return_value={})):
            result = creature.next([.5, .5], 0)
            _, candidate, position = creature.groove.play.call_args.args
            np.testing.assert_array_equal(candidate, proposed)
            self.assertEqual(position, 24)
            self.assertEqual(result["bar"], 1)
            creature.next([.5, .5], 0)
            np.testing.assert_array_equal(observe.call_args.args[0][0], played)

    def test_bar_request_does_not_lock_the_creature_while_waiting_for_layout(self):
        for path in ("/api/next", "/api/deck/next"):
            with self.subTest(path=path):
                layout = ObservedLayout()
                creature = serve.Creature()
                creature.brain = object()
                creature.next = Mock(return_value={"bar": 1})
                handler = post(path)
                request = threading.Thread(target=serve.Handler.do_POST, args=(handler,), daemon=True)
                with patch.multiple(serve, CREATURE=creature, LAYOUT=layout, _DECKS=[creature, None, None],
                                    MIX={"incoming": None, "bar": 0}):
                    with layout:
                        request.start()
                        reached = layout.requested.wait(1)
                        unlocked = creature.lock.acquire(blocking=False)
                        if unlocked:
                            creature.lock.release()
                    request.join(2)
                self.assertTrue(reached)
                self.assertTrue(unlocked, "bar request took the creature lock before the layout lock")
                self.assertFalse(request.is_alive())
                self.assertEqual(handler._send.call_args.args[1]["bar"], 1)

    def test_failed_checkpoint_prime_preserves_current_composition(self):
        creature = serve.Creature()
        creature.brain = Mock()
        creature.card = {"name": "old", "tokens_per_layer": 8, "mod_levels": 7, "pool": "pool_v0.json"}
        creature.state = [1] + [0] * 11
        creature._configure(creature.card)
        original = (creature.brain, creature.card, creature.state, creature.pool)
        new_card = dict(creature.card, name="new", tokens_per_layer=12, temperature=.08)
        bad_brain = Mock()
        bad_brain.act.side_effect = ValueError("invalid seed")
        seed = ([creature.state] * 2, [[.5, .5]] * 3, "bad", 0)
        with patch.object(serve, "_LAYOUT_OWNER", [creature]), patch.object(creature, "_read", return_value=(new_card, bad_brain)):
            with patch.object(serve.Creature, "_seed", return_value=seed):
                with self.assertRaisesRegex(ValueError, "invalid seed"):
                    creature.load("new")
            self.assertEqual((creature.brain, creature.card, creature.state, creature.pool), original)
            creature.ensure_layout()
            self.assertEqual(serve.common.T, 8)

    def test_feedback_uses_the_rated_checkpoint_layout(self):
        player = serve.Creature()
        player.card = {"name": "player", "tokens_per_layer": 8, "mod_levels": 7}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feedback.jsonl"
            with patch.object(serve, "FEEDBACK", path), patch.object(serve.common, "T", 24), patch.object(serve.common, "M", 5):
                player.feedback(1, None)
            feedback = json.loads(path.read_text())
        self.assertEqual((feedback["tokens_per_layer"], feedback["mod_levels"]), (8, 7))

    def test_switching_to_demo_cancels_the_old_handover(self):
        player = serve.Creature()
        player.brain = object()
        handler = post("/api/load", {"name": "demo", "mix": True})
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection"
            with patch.multiple(serve, CREATURE=player, MIX={"incoming": object(), "bar": 3},
                                PLAYER_SELECTION=selection):
                serve.Handler.do_POST(handler)
                self.assertIsNone(serve.MIX["incoming"])
            self.assertEqual(selection.read_text(), "demo")
        self.assertIsNone(player.brain)
