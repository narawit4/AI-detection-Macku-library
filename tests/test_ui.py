import tkinter as tk
from tkinter import ttk
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

    def test_window_is_fixed_size_compact_xp_dashboard(self):
        self.app.update_idletasks()
        self.assertEqual(tuple(map(int, self.app.resizable())), (0, 0))
        self.assertEqual(self.app.geometry().split("+")[0], "640x560")
        self.assertEqual(self.app.cget("background"), "#ECE9D8")

    def _is_descendant(self, widget, ancestor):
        current = widget
        while current is not self.app:
            if current is ancestor:
                return True
            current = current.master
        return False

    def test_compact_dashboard_keeps_only_primary_motion_controls(self):
        self.assertFalse(self._is_descendant(self.app.motion_strength_pps_entry,
                                             self.app.advanced_frame))
        self.assertFalse(self._is_descendant(self.app.jitter_rate_hz_entry,
                                             self.app.advanced_frame))
        secondary_widgets = {
            "Hotkey": self.app.hotkey_button,
            "Angle": self.app.motion_angle_deg_entry,
            "Horizontal": self.app.horizontal_jitter_pps_entry,
            "Vertical": self.app.vertical_jitter_pps_entry,
            "Randomness": self.app.jitter_randomness_percent_entry,
            "Axis Phase": self.app.jitter_axis_phase_deg_entry,
            "Smoothness": self.app.smoothness_percent_entry,
            "Ramp time": self.app.ramp_up_ms_entry,
            "Update rate": self.app.update_rate_hz_entry,
            "Maximum step": self.app.max_step_px_entry,
            "Acceleration": self.app.acceleration_pps2_entry,
            "Deceleration": self.app.deceleration_pps2_entry,
            "Waveform": self.app.waveform_combo,
            "Motion Curve": self.app.motion_curve_combo,
        }
        for setting, widget in secondary_widgets.items():
            with self.subTest(setting=setting):
                self.assertTrue(self._is_descendant(widget,
                                                    self.app.advanced_frame))

    def test_runtime_group_keeps_stop_always_visible(self):
        self.assertTrue(self._is_descendant(self.app.stop_button,
                                            self.app.runtime_frame))
        self.assertFalse(self._is_descendant(self.app.stop_button,
                                             self.app.advanced_frame))

    def test_stop_remains_inside_application_viewport_when_advanced_is_scrolled(self):
        self.app.deiconify()
        self.app.toggle_advanced()
        self.app.update()
        self.app.canvas.yview_moveto(1.0)
        self.app.update()

        app_left = self.app.winfo_rootx()
        app_top = self.app.winfo_rooty()
        app_right = app_left + self.app.winfo_width()
        app_bottom = app_top + self.app.winfo_height()
        stop_left = self.app.stop_button.winfo_rootx()
        stop_top = self.app.stop_button.winfo_rooty()
        stop_right = stop_left + self.app.stop_button.winfo_width()
        stop_bottom = stop_top + self.app.stop_button.winfo_height()

        self.assertGreaterEqual(stop_left, app_left)
        self.assertGreaterEqual(stop_top, app_top)
        self.assertLessEqual(stop_right, app_right)
        self.assertLessEqual(stop_bottom, app_bottom)

    def test_runtime_group_uses_the_approved_title(self):
        self.assertEqual(self.app.runtime_frame.cget("text"), "Runtime")

    def test_setup_group_combines_bindings_preset_and_test_run(self):
        group_titles = [
            child.cget("text") for child in self.app.content.winfo_children()
            if isinstance(child, ttk.LabelFrame)
        ]
        self.assertIn("Setup", group_titles)
        setup = next(
            child for child in self.app.content.winfo_children()
            if isinstance(child, ttk.LabelFrame) and child.cget("text") == "Setup"
        )
        for expected in ("Trigger", "Modifier", "Preset", "Test 3s"):
            self.assertIn(expected, widget_texts(setup))
        self.assertNotIn("Actions", group_titles)

    def test_luna_blue_styles_are_registered(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("XP.Title.TFrame", "background"), "#0054E3")
        self.assertEqual(style.lookup("XP.Group.TLabelframe", "background"), "#ECE9D8")
        self.assertEqual(style.lookup("XP.Danger.TButton", "foreground"), "#A00000")

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

    def test_test_run_ignores_queued_normal_stop_before_duration_completion(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertTrue(self.app._normal_motion_started)
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "Testing")
        self.app.handle_service_event(ServiceEvent("motion_stopped", "test_run"))
        self.assertEqual(self.app.runtime_state_var.get(), "Testing")
        self.app.handle_service_event(ServiceEvent("motion_stopped", "duration_complete"))
        self.assertEqual(self.app.runtime_state_var.get(), "Armed")

    def test_toggle_is_blocked_while_test_run_is_active(self):
        self.service.connected = True
        self.app.start_test_run()
        self.assertFalse(self.app.enabled)
        self.app.toggle_enabled()
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "Testing")

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

    def test_hotkey_capture_accepts_each_mouse_button_on_a_new_down_edge(self):
        down = set()

        def key_state(vk):
            return 0x8000 if vk in down else 0

        self.app._get_async_key_state = key_state
        mouse_buttons = {
            0x01: "Mouse Left",
            0x02: "Mouse Right",
            0x04: "Mouse Middle",
            0x05: "Mouse4",
            0x06: "Mouse5",
        }
        for vk, expected_name in mouse_buttons.items():
            with self.subTest(vk=vk):
                down.clear()
                self.app.capture_hotkey()
                self.assertTrue(self.app._capturing_hotkey)
                down.add(vk)
                self.app._poll_hotkey_capture()
                self.assertFalse(self.app._capturing_hotkey)
                self.assertEqual(self.app.hotkey_watcher.vk, vk)
                self.assertEqual(self.app.hotkey_name_var.get(), expected_name)

    def test_stale_callbacks_do_not_raise_when_tk_scheduling_is_tearing_down(self):
        def raising_after(*_args):
            raise tk.TclError("application has been destroyed")

        self.app.after = raising_after
        self.app.queue_service_event(ServiceEvent("connected"))
        self.app._hotkey_pressed()

    def test_motion_snapshot_access_is_lock_protected(self):
        class CountingLock:
            def __init__(self):
                self.enters = 0

            def __enter__(self):
                self.enters += 1
                return self

            def __exit__(self, *_args):
                return False

        lock = CountingLock()
        self.app._motion_lock = lock
        self.app.get_motion_settings()
        self.app.motion_strength_pps_var.set("81")
        self.app.update()
        self.assertGreaterEqual(lock.enters, 2)

    def test_preset_clears_stale_invalid_entry_style(self):
        self.app.motion_angle_deg_var.set("not-a-number")
        self.app.update()
        self.assertEqual(self.app.motion_angle_deg_entry.cget("style"), "Invalid.TEntry")
        self.app.preset_var.set("Balanced")
        self.app.apply_preset()
        self.assertEqual(self.app.motion_angle_deg_entry.cget("style"), "App.TEntry")


if __name__ == "__main__":
    unittest.main()
