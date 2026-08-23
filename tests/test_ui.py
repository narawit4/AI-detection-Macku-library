import tkinter as tk
import unittest

from ui import JitterApp


class StubStore:
    def __init__(self):
        self.saved = []

    def load(self):
        from settings import AppConfig, LoadOutcome
        return LoadOutcome(AppConfig())

    def save(self, config):
        self.saved.append(config)


class StubService:
    def __init__(self, event_sink):
        self.event_sink = event_sink
        self.connected = False
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def connect(self):
        self.started += 1

    def reconnect(self):
        return None

    def start_motion(self, _settings_provider, duration_s=None):
        self.started += 1
        return self.connected

    def stop_motion(self, reason="manual"):
        self.stopped += 1

    def close(self):
        self.closed += 1


class StubHotkey:
    def __init__(self, vk, callback):
        self.vk = vk
        self.callback = callback
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def set_vk(self, vk):
        self.vk = vk

    def stop(self):
        self.stopped += 1


def widget_texts(widget):
    values = []
    try:
        text = widget.cget("text")
    except tk.TclError:
        text = ""
    if text:
        values.append(str(text))
    for child in widget.winfo_children():
        values.extend(widget_texts(child))
    return values


class JitterLayoutTests(unittest.TestCase):
    def setUp(self):
        self.service = None
        self.store = StubStore()

        def service_factory(event_sink):
            self.service = StubService(event_sink)
            return self.service

        self.app = JitterApp(
            config_store=self.store,
            service_factory=service_factory,
            hotkey_factory=StubHotkey,
            auto_start=False,
        )
        self.app.withdraw()

    def tearDown(self):
        try:
            self.app.destroy()
        except tk.TclError:
            pass

    def test_window_is_fixed_size_focused_dashboard(self):
        self.app.update_idletasks()
        self.assertEqual(tuple(map(int, self.app.resizable())), (0, 0))
        width, height = map(int, self.app.geometry().split("+")[0].split("x"))
        self.assertGreaterEqual(width, 700)
        self.assertGreaterEqual(height, 650)

    def test_required_actions_are_present_and_stop_is_outside_advanced(self):
        texts = widget_texts(self.app)
        for expected in ("Reconnect", "Enable Jitter", "Test 3s", "STOP", "Advanced Settings"):
            self.assertIn(expected, texts)
        stop = self.app.stop_button
        ancestor = stop.master
        while ancestor is not self.app:
            self.assertIsNot(ancestor, self.app.advanced_frame)
            ancestor = ancestor.master

    def test_advanced_toggle_does_not_change_outer_geometry(self):
        self.app.update_idletasks()
        before = self.app.geometry().split("+")[0]
        self.app.toggle_advanced()
        self.app.update_idletasks()
        after = self.app.geometry().split("+")[0]
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
