import importlib.util
import importlib
from pathlib import Path
import tempfile
import threading
import unittest

from sound_service import ToggleSoundPlayer


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


if __name__ == "__main__":
    unittest.main()
