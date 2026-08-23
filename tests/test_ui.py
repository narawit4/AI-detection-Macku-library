import tkinter as tk
import unittest

from ui import JitterApp
from makcu_service import ServiceEvent


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
        self.reconnects = 0

    def connect(self):
        self.started += 1

    def reconnect(self):
        self.reconnects += 1
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
            self.app.close_app()
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


class JitterRuntimeTests(JitterLayoutTests):
    def test_start_runtime_starts_hotkey_and_connection_once(self):
        self.app.start_runtime()
        self.app.start_runtime()
        self.assertEqual(self.app.hotkey_watcher.started, 1)
        self.assertEqual(self.service.started, 1)

    def test_enabled_trigger_starts_and_release_stops_motion(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertGreaterEqual(self.service.started, 1)
        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.assertGreaterEqual(self.service.stopped, 1)

    def test_modifier_gate_requires_both_buttons(self):
        self.service.connected = True
        self.app.modifier_var.set("Right")
        self.app.on_bindings_changed()
        self.app.set_enabled(True)
        started = self.service.started
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertEqual(self.service.started, started)
        self.app.handle_service_event(ServiceEvent("button", ("Right", True)))
        self.assertGreater(self.service.started, started)

    def test_stop_disables_and_clears_trigger_state(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.trigger_gate.update_button("Left", True)
        self.app.emergency_stop("Stopped by user")
        self.assertFalse(self.app.enabled)
        self.assertFalse(self.app.trigger_gate.active)
        self.assertEqual(self.app.runtime_state_var.get(), "Disabled")

    def test_disconnect_performs_emergency_stop(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.connection_state_var.get(), "Disconnected")

    def test_test_run_bypasses_trigger_but_requires_connection(self):
        self.service.connected = False
        self.app.start_test_run()
        stopped_count = self.service.stopped
        self.service.connected = True
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "Testing")
        self.assertGreaterEqual(self.service.started, 1)
        self.assertEqual(self.service.stopped, stopped_count)

    def test_captured_hotkey_updates_watcher_and_persisted_name(self):
        self.app.apply_captured_hotkey(0x77, "F8")
        self.assertEqual(self.app.hotkey_watcher.vk, 0x77)
        self.assertEqual(self.app.hotkey_name_var.get(), "F8")

    def test_save_config_writes_current_independent_bindings(self):
        self.app.trigger_var.set("Mouse4")
        self.app.modifier_var.set("None")
        self.app.save_config()
        self.assertEqual(self.store.saved[-1].trigger, "Mouse4")
        self.assertEqual(self.store.saved[-1].modifier, "None")

    def test_close_stops_hotkey_motion_and_service(self):
        self.app.start_runtime()
        self.app.close_app()
        self.assertEqual(self.app.hotkey_watcher.stopped, 1)
        self.assertGreaterEqual(self.service.stopped, 1)
        self.assertEqual(self.service.closed, 1)

    def test_service_events_are_marshaled_to_tk_thread(self):
        self.app.queue_service_event(ServiceEvent("connected", "Fake Makcu"))
        self.assertEqual(self.app.connection_state_var.get(), "Disconnected")
        self.app.update()
        self.assertEqual(self.app.connection_state_var.get(), "Connected")

    def test_global_hotkey_toggles_on_tk_thread(self):
        self.service.connected = True
        self.app._hotkey_pressed()
        self.assertFalse(self.app.enabled)
        self.app.update()
        self.assertTrue(self.app.enabled)

    def test_reconnect_delegates_to_service(self):
        self.app.start_runtime()
        self.app.reconnect()
        self.assertEqual(self.service.reconnects, 1)

    def test_timed_test_run_restores_armed_state(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "Testing")
        self.app.handle_service_event(ServiceEvent("motion_stopped", "duration_complete"))
        self.assertEqual(self.app.runtime_state_var.get(), "Armed")
        self.assertTrue(self.app.enabled)

    def test_invalid_motion_edit_keeps_last_snapshot(self):
        previous = self.app.get_motion_settings()
        self.app.motion_angle_deg_var.set("not-a-number")
        self.app.update()
        self.assertEqual(self.app.get_motion_settings(), previous)
        self.assertIn("motion_angle_deg", self.app._invalid_motion_keys)
        self.app.motion_angle_deg_var.set("180")
        self.app.update()
        self.assertEqual(self.app.get_motion_settings().angle_deg, 180.0)

    def test_future_schema_save_protection_is_honored(self):
        self.app._save_allowed = False
        self.app.save_config()
        self.assertEqual(self.store.saved, [])

    def test_hotkey_capture_skips_makcu_mouse_virtual_keys(self):
        def key_state(vk):
            return 0x8000 if vk in (0x01, 0x77) else 0

        self.app._get_async_key_state = key_state
        self.app.capture_hotkey()
        self.assertFalse(self.app._capturing_hotkey)
        self.assertEqual(self.app.hotkey_watcher.vk, 0x77)


if __name__ == "__main__":
    unittest.main()
