import unittest

from jitter_app.device.hotkeys import HotkeyEdgeDetector, HotkeyWatcher


class HotkeyTests(unittest.TestCase):
    def test_edge_detector_fires_once_per_press(self):
        detector = HotkeyEdgeDetector()
        self.assertTrue(detector.update(True))
        self.assertFalse(detector.update(True))
        self.assertFalse(detector.update(False))
        self.assertTrue(detector.update(True))

    def test_watcher_poll_once_invokes_callback_only_on_down_edge(self):
        states = iter((0x8000, 0x8000, 0, 0x8000))
        calls = []
        watcher = HotkeyWatcher(0xBD, lambda: calls.append("toggle"), key_state=lambda _vk: next(states))
        for _index in range(4):
            watcher.poll_once()
        self.assertEqual(calls, ["toggle", "toggle"])

    def test_changing_vk_resets_the_held_edge(self):
        current = {0xBD: 0x8000, 0x77: 0x8000}
        calls = []
        watcher = HotkeyWatcher(0xBD, lambda: calls.append("toggle"), key_state=current.get)
        watcher.poll_once()
        watcher.set_vk(0x77)
        watcher.poll_once()
        self.assertEqual(calls, ["toggle", "toggle"])
