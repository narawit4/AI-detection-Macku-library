import importlib.util
import importlib
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from sound_service import ToggleSoundPlayer, _PygameBackend


class ToggleSoundPlayerTests(unittest.TestCase):
    def test_sound_service_module_is_available(self):
        """Fails if hotkey audio support is absent from the application."""
        self.assertIsNotNone(importlib.util.find_spec("sound_service"))

    def test_toggle_sound_player_is_available(self):
        module = importlib.import_module("sound_service")
        self.assertIsNotNone(getattr(module, "ToggleSoundPlayer", None))

    def test_play_queues_ogg_cues_without_waiting_for_audio(self):
        entered = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        played = []

        class Backend:
            def set_volume(self, volume):
                played.append(("volume", volume))

            def play(self, enabled):
                entered.set()
                release.wait(1)
                played.append(("cue", enabled))
                if len([item for item in played if item[0] == "cue"]) == 2:
                    completed.set()

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            received_paths = []

            def backend_factory(on_path, off_path):
                received_paths.extend((on_path, off_path))
                return Backend()

            player = ToggleSoundPlayer(
                Path(directory), backend_factory=backend_factory
            )
            player.play(True)
            self.assertTrue(entered.wait(1))
            player.play(False)
            release.set()
            self.assertTrue(completed.wait(1))
            player.close()

        self.assertEqual(
            played,
            [("volume", 0.7), ("cue", True),
             ("volume", 0.7), ("cue", False)],
        )
        self.assertEqual(
            [path.name for path in received_paths], ["ON.ogg", "OFF.ogg"]
        )

    def test_muting_suppresses_hotkey_cues_but_force_allows_preview(self):
        completed = threading.Event()
        played = []

        class Backend:
            def set_volume(self, volume):
                played.append(("volume", volume))

            def play(self, enabled):
                played.append(("cue", enabled))
                completed.set()

            def close(self):
                pass

        player = ToggleSoundPlayer(
            Path("sound"), backend_factory=lambda _on, _off: Backend(),
            enabled=False, volume=25,
        )
        player.play(True)
        player.play(False, force=True)
        self.assertTrue(completed.wait(1))
        player.close()
        self.assertEqual(played, [("volume", 0.25), ("cue", False)])

    def test_playback_failure_closes_backend_before_recovery(self):
        failed_closed = threading.Event()
        recovered_played = threading.Event()
        backends = []

        class FailedBackend:
            closed_calls = 0

            def set_volume(self, volume):
                pass

            def play(self, enabled):
                raise RuntimeError("playback failed")

            def close(self):
                self.closed_calls += 1
                failed_closed.set()

        class RecoveredBackend:
            closed_calls = 0

            def set_volume(self, volume):
                pass

            def play(self, enabled):
                recovered_played.set()

            def close(self):
                self.closed_calls += 1

        def factory(_on_path, _off_path):
            backend = FailedBackend() if not backends else RecoveredBackend()
            backends.append(backend)
            return backend

        player = ToggleSoundPlayer(Path("sound"), backend_factory=factory)
        with self.assertLogs(level="ERROR") as logs:
            player.play(True)
            self.assertTrue(failed_closed.wait(1))
        self.assertEqual(backends[0].closed_calls, 1)
        self.assertIn("Could not play hotkey sound", logs.output[0])

        player.play(False)
        self.assertTrue(recovered_played.wait(1))
        player.close()
        self.assertEqual(backends[1].closed_calls, 1)

    def test_configuration_failure_closes_backend_before_discarding_it(self):
        closed = threading.Event()
        calls = {"play": 0}

        class Backend:
            closed_calls = 0

            def set_volume(self, volume):
                raise RuntimeError("configuration failed")

            def play(self, enabled):
                calls["play"] += 1

            def close(self):
                self.closed_calls += 1
                closed.set()

        backend = Backend()
        player = ToggleSoundPlayer(
            Path("sound"), backend_factory=lambda _on, _off: backend,
        )
        with self.assertLogs(level="ERROR") as logs:
            player.play(True)
            self.assertTrue(closed.wait(1))
        self.assertEqual(backend.closed_calls, 1)
        self.assertEqual(calls["play"], 0)
        self.assertIn("Could not play hotkey sound", logs.output[0])
        player.close()
        self.assertEqual(backend.closed_calls, 1)

    def test_partial_pygame_initialization_quits_mixer_even_if_stop_fails(self):
        calls = []

        class Channel:
            def stop(self):
                calls.append("stop")
                raise RuntimeError("stop failed")

        class Mixer:
            def init(self):
                calls.append("init")

            def Channel(self, index):
                calls.append(("channel", index))
                return Channel()

            def Sound(self, path):
                calls.append(("sound", path))
                raise RuntimeError("sound failed")

            def quit(self):
                calls.append("quit")

        pygame = SimpleNamespace(mixer=Mixer())
        with mock.patch.dict(sys.modules, {"pygame": pygame}):
            with self.assertLogs(level="ERROR") as logs:
                with self.assertRaisesRegex(RuntimeError, "sound failed"):
                    _PygameBackend(Path("ON.ogg"), Path("OFF.ogg"))

        self.assertEqual(calls, ["init", ("channel", 0), ("sound", "ON.ogg"),
                                 "stop", "quit"])
        self.assertIn("Could not clean up partial hotkey sound backend", logs.output[0])

    def test_pygame_mixer_init_failure_still_attempts_cleanup(self):
        calls = []

        class Mixer:
            def init(self):
                calls.append("init")
                raise RuntimeError("init failed")

            def quit(self):
                calls.append("quit")

        pygame = SimpleNamespace(mixer=Mixer())
        with mock.patch.dict(sys.modules, {"pygame": pygame}):
            with self.assertRaisesRegex(RuntimeError, "init failed"):
                _PygameBackend(Path("ON.ogg"), Path("OFF.ogg"))

        self.assertEqual(calls, ["init", "quit"])

    def test_pygame_close_logs_secondary_cleanup_failure(self):
        calls = []

        class Channel:
            def stop(self):
                calls.append("stop")
                raise RuntimeError("channel stop failed")

        class Mixer:
            def init(self):
                calls.append("init")

            def Channel(self, index):
                calls.append(("channel", index))
                return Channel()

            def Sound(self, path):
                calls.append(("sound", path))
                return SimpleNamespace(set_volume=lambda _volume: None)

            def quit(self):
                calls.append("quit")
                raise RuntimeError("mixer quit failed")

        pygame = SimpleNamespace(mixer=Mixer())
        with mock.patch.dict(sys.modules, {"pygame": pygame}):
            backend = _PygameBackend(Path("ON.ogg"), Path("OFF.ogg"))
            with self.assertLogs(level="ERROR") as logs:
                with self.assertRaisesRegex(RuntimeError, "channel stop failed"):
                    backend.close()

        self.assertEqual(
            calls,
            ["init", ("channel", 0), ("sound", "ON.ogg"),
             ("sound", "OFF.ogg"), "stop", "quit"],
        )
        self.assertIn("Could not quit hotkey sound mixer", logs.output[0])

    def test_close_failure_is_contained_and_next_cue_recovers(self):
        close_logged = threading.Event()
        recovered_played = threading.Event()
        backends = []
        log_messages = []

        class FailedBackend:
            def set_volume(self, volume):
                pass

            def play(self, enabled):
                raise RuntimeError("playback failed")

            def close(self):
                raise RuntimeError("close failed")

        class RecoveredBackend:
            closed_calls = 0

            def set_volume(self, volume):
                pass

            def play(self, enabled):
                recovered_played.set()

            def close(self):
                self.closed_calls += 1

        def factory(_on_path, _off_path):
            backend = FailedBackend() if not backends else RecoveredBackend()
            backends.append(backend)
            return backend

        player = ToggleSoundPlayer(Path("sound"), backend_factory=factory)

        def record_log(message, *args, **kwargs):
            log_messages.append(message)
            if message == "Could not close hotkey sound player":
                close_logged.set()

        with mock.patch("sound_service.logging.exception", side_effect=record_log):
            player.play(True)
            self.assertTrue(close_logged.wait(1))
        player.play(False)
        self.assertTrue(recovered_played.wait(1))
        player.close()
        self.assertEqual(backends[1].closed_calls, 1)
        self.assertEqual(
            log_messages,
            ["Could not play hotkey sound", "Could not close hotkey sound player"],
        )

    def test_concurrent_and_second_close_close_backend_once(self):
        played = threading.Event()

        class Backend:
            closed_calls = 0

            def set_volume(self, volume):
                pass

            def play(self, enabled):
                played.set()

            def close(self):
                self.closed_calls += 1

        backend = Backend()
        player = ToggleSoundPlayer(
            Path("sound"), backend_factory=lambda _on, _off: backend,
        )
        player.play(True)
        self.assertTrue(played.wait(1))

        start = threading.Barrier(3)
        closers = [
            threading.Thread(target=lambda: (start.wait(), player.close()))
            for _ in range(2)
        ]
        for closer in closers:
            closer.start()
        start.wait()
        for closer in closers:
            closer.join(1)
            self.assertFalse(closer.is_alive())

        player.close()
        self.assertEqual(backend.closed_calls, 1)


if __name__ == "__main__":
    unittest.main()
