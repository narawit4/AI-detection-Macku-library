import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest

from ui import JitterApp
from makcu_service import ServiceEvent
from xp_widgets import XPGlossySlider


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
        self.assertEqual(self.app.cget("background"), "#F4F1E6")

    def test_internal_brand_banner_is_not_rendered(self):
        brand_banners = [
            text for text in widget_texts(self.app)
            if text.startswith("Jitter") and "Makcu Control" in text
        ]
        self.assertEqual(brand_banners, [])

    def test_native_window_title_carries_the_brand_name(self):
        expected = "Jitter " + chr(0x2014) + " Makcu Control"
        self.assertEqual(self.app.title(), expected)

    def test_every_numeric_control_uses_xp_glossy_slider(self):
        numeric_keys = (
            "motion_angle_deg",
            "motion_strength_pps",
            "horizontal_jitter_pps",
            "vertical_jitter_pps",
            "jitter_rate_hz",
            "jitter_randomness_percent",
            "jitter_axis_phase_deg",
            "smoothness_percent",
            "ramp_up_ms",
            "update_rate_hz",
            "max_step_px",
            "acceleration_pps2",
            "deceleration_pps2",
        )
        for key in numeric_keys:
            with self.subTest(key=key):
                self.assertIsInstance(
                    getattr(self.app, f"{key}_scale"),
                    XPGlossySlider,
                )

    def test_glossy_slider_user_change_updates_exact_entry_and_snapshot(self):
        slider = self.app.motion_strength_pps_scale
        slider._set_from_user(123)
        self.app.update()
        self.assertEqual(self.app.motion_strength_pps_var.get(), "123")
        self.assertEqual(self.app.get_motion_settings().strength_pps, 123.0)

    def test_exact_entry_and_preset_changes_update_glossy_slider_silently(self):
        slider = self.app.motion_strength_pps_scale
        self.app.motion_strength_pps_var.set("77")
        self.app.update()
        self.assertEqual(slider.get(), 77.0)
        self.app.preset_var.set("Balanced")
        self.app.apply_preset()
        self.app.update()
        self.assertEqual(
            slider.get(),
            self.app.get_motion_settings().strength_pps,
        )

    def _is_descendant(self, widget, ancestor):
        current = widget
        while current is not self.app:
            if current is ancestor:
                return True
            current = current.master
        return False

    def test_command_center_uses_fixed_status_body_footer_runtime_order(self):
        regions = (
            self.app.status_strip,
            self.app.command_center,
            self.app.footer_frame,
            self.app.runtime_frame,
        )
        self.assertTrue(all(widget.master is self.app.shell for widget in regions))
        self.assertEqual(
            [int(widget.grid_info()["row"]) for widget in regions],
            [0, 1, 2, 3],
        )
        self.assertEqual(self.app.shell.grid_rowconfigure(1)["weight"], 1)

    def test_setup_and_tools_stay_in_fixed_left_column(self):
        fixed_widgets = (
            self.app.trigger_combo,
            self.app.modifier_combo,
            self.app.preset_combo,
            self.app.hotkey_button,
            self.app.reconnect_button,
            self.app.test_button,
            self.app.advanced_toggle,
        )
        for widget in fixed_widgets:
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.left_column))
                self.assertFalse(self._is_descendant(widget, self.app.right_content))

    def test_quick_and_advanced_controls_live_in_right_workspace(self):
        right_widgets = (
            self.app.motion_strength_pps_entry,
            self.app.jitter_rate_hz_entry,
            self.app.motion_angle_deg_entry,
            self.app.waveform_combo,
            self.app.motion_curve_combo,
        )
        for widget in right_widgets:
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.right_content))
        self.assertFalse(self._is_descendant(self.app.hotkey_button,
                                             self.app.advanced_frame))

    def test_advanced_uses_approved_two_column_grid(self):
        expected_positions = {
            "motion_angle_deg": (0, 0),
            "horizontal_jitter_pps": (0, 1),
            "vertical_jitter_pps": (1, 0),
            "jitter_randomness_percent": (1, 1),
            "jitter_axis_phase_deg": (2, 0),
            "smoothness_percent": (2, 1),
            "ramp_up_ms": (3, 0),
            "update_rate_hz": (3, 1),
            "max_step_px": (4, 0),
            "acceleration_pps2": (4, 1),
            "deceleration_pps2": (5, 0),
        }
        for key, expected in expected_positions.items():
            with self.subTest(key=key):
                block = getattr(self.app, f"{key}_entry").master.master
                info = block.grid_info()
                self.assertEqual((int(info["row"]), int(info["column"])), expected)
                self.assertIs(block.master, self.app.advanced_grid)

    def test_advanced_choices_span_the_full_grid_width(self):
        for combo in (self.app.waveform_combo, self.app.motion_curve_combo):
            with self.subTest(combo=str(combo)):
                choice_row = combo.master
                info = choice_row.grid_info()
                self.assertEqual(int(info["columnspan"]), 2)
                self.assertIs(choice_row.master, self.app.advanced_grid)

    def test_runtime_actions_have_equal_fixed_weight(self):
        self.assertIs(self.app.enable_button.master, self.app.runtime_frame)
        self.assertIs(self.app.stop_button.master, self.app.runtime_frame)
        self.assertEqual(int(self.app.enable_button.grid_info()["column"]), 0)
        self.assertEqual(int(self.app.stop_button.grid_info()["column"]), 2)
        self.assertIn("ew", self.app.enable_button.grid_info()["sticky"])
        self.assertIn("ew", self.app.stop_button.grid_info()["sticky"])
        self.assertEqual(self.app.runtime_frame.grid_columnconfigure(0)["weight"],
                         self.app.runtime_frame.grid_columnconfigure(2)["weight"])

    def test_footer_and_runtime_are_outside_scrollable_workspace(self):
        for widget in (self.app.footer_frame, self.app.runtime_frame,
                       self.app.enable_button, self.app.stop_button):
            with self.subTest(widget=str(widget)):
                self.assertFalse(self._is_descendant(widget, self.app.right_host))

    def test_status_strip_combines_device_and_connection_state(self):
        self.assertTrue(self._is_descendant(self.app.device_label,
                                            self.app.status_strip))
        self.assertTrue(self._is_descendant(self.app.connection_label,
                                            self.app.status_strip))
        self.assertFalse(self._is_descendant(self.app.reconnect_button,
                                             self.app.status_strip))

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

    def test_luna_blue_styles_are_registered(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("XP.Title.TFrame", "background"), "#2F69B3")
        self.assertEqual(style.lookup("XP.Group.TLabelframe", "background"), "#F4F1E6")

    def test_xp_remastered_buttons_use_high_contrast_palette(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("XP.Primary.TButton", "background"),
                         "#356FAF")
        self.assertEqual(style.lookup("XP.Primary.TButton", "foreground"),
                         "#FFFFFF")
        self.assertEqual(style.lookup("XP.Danger.TButton", "background"),
                         "#C74652")
        self.assertEqual(style.lookup("XP.Danger.TButton", "foreground"),
                         "#FFFFFF")
        self.assertEqual(style.lookup("XP.Secondary.TButton", "background"),
                         "#F7F3E7")
        self.assertEqual(self.app.enable_button.cget("style"),
                         "XP.Primary.TButton")
        self.assertEqual(self.app.stop_button.cget("style"),
                         "XP.Danger.TButton")

    def test_xp_remastered_buttons_show_hover_press_and_focus_states(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("XP.Primary.TButton", "background",
                                      ("active",)), "#5B92CC")
        self.assertEqual(style.lookup("XP.Primary.TButton", "background",
                                      ("pressed", "active")), "#244F7D")
        self.assertEqual(style.lookup("XP.Danger.TButton", "background",
                                      ("active",)), "#DF6670")
        self.assertEqual(style.lookup("XP.Danger.TButton", "background",
                                      ("pressed", "active")), "#942F38")
        self.assertEqual(style.lookup("XP.Secondary.TButton", "background",
                                      ("active",)), "#E5EEF8")
        self.assertEqual(style.lookup("XP.Secondary.TButton", "relief",
                                      ("pressed",)), "sunken")
        self.assertEqual(style.lookup("XP.Primary.TButton", "bordercolor",
                                      ("focus",)), "#E4A43A")

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

    def test_advanced_starts_collapsed_and_expands_below_quick(self):
        self.assertFalse(self.app._advanced_visible)
        self.assertFalse(self.app.advanced_state_var.get())
        self.assertFalse(self.app.advanced_frame.winfo_manager())
        self.assertEqual(self.app.advanced_toggle.cget("text"),
                         "Advanced Settings ▼")

        self.app.toggle_advanced()
        self.app.update_idletasks()

        self.assertTrue(self.app._advanced_visible)
        self.assertEqual(self.app.advanced_frame.winfo_manager(), "pack")
        children = self.app.right_content.pack_slaves()
        self.assertLess(children.index(self.app.quick_frame),
                        children.index(self.app.advanced_frame))
        self.assertEqual(self.app.advanced_toggle.cget("text"),
                         "Advanced Settings ▲")

    def test_collapsing_advanced_returns_right_workspace_to_top(self):
        self.app.deiconify()
        self.app.toggle_advanced()
        self.app.update()
        self.app.right_canvas.yview_moveto(1.0)
        self.app.toggle_advanced()
        self.app.update()
        self.assertEqual(self.app.right_canvas.yview()[0], 0.0)

    def test_mousewheel_scrolls_only_over_right_workspace(self):
        self.app.deiconify()
        self.app.toggle_advanced()
        self.app.update()
        self.app.right_canvas.yview_moveto(0.0)

        right_event = SimpleNamespace(
            delta=-120,
            x_root=self.app.right_canvas.winfo_rootx() + 10,
            y_root=self.app.right_canvas.winfo_rooty() + 10,
        )
        self.assertEqual(self.app._on_right_mousewheel(right_event), "break")
        self.app.update_idletasks()
        after_right = self.app.right_canvas.yview()[0]
        self.assertGreater(after_right, 0.0)

        left_event = SimpleNamespace(
            delta=-120,
            x_root=self.app.left_column.winfo_rootx() + 10,
            y_root=self.app.left_column.winfo_rooty() + 10,
        )
        self.assertIsNone(self.app._on_right_mousewheel(left_event))
        self.assertEqual(self.app.right_canvas.yview()[0], after_right)


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
        self.assertEqual(
            self.app.footer_var.get(),
            "Invalid value for motion angle deg",
        )
        self.app.preset_var.set("Balanced")
        self.app.apply_preset()
        self.assertEqual(self.app.motion_angle_deg_entry.cget("style"), "App.TEntry")
        self.assertEqual(self.app.footer_var.get(), "Ready")

    def test_valid_motion_edit_clears_stale_invalid_footer(self):
        self.app.motion_strength_pps_var.set("not-a-number")
        self.app.update()
        self.assertEqual(
            self.app.footer_var.get(),
            "Invalid value for motion strength pps",
        )

        self.app.motion_strength_pps_var.set("75")
        self.app.update()

        self.assertEqual(self.app.motion_strength_pps_entry.cget("style"), "App.TEntry")
        self.assertEqual(self.app.footer_var.get(), "Ready")


if __name__ == "__main__":
    unittest.main()
