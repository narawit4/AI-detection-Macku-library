import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import threading
import unittest

from ui import JitterApp
from makcu_service import ServiceEvent
from liquid_widgets import LiquidIconButton, LiquidSlider


class StubStore:
    def __init__(self, config=None):
        self.saved = []
        self.config = config

    def load(self):
        from settings import AppConfig, LoadOutcome
        return LoadOutcome(self.config or AppConfig())

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


def contrast_ratio(first, second):
    def luminance(color):
        channels = [int(color[index:index + 2], 16) / 255.0
                    for index in (1, 3, 5)]
        linear = [channel / 12.92 if channel <= 0.04045
                  else ((channel + 0.055) / 1.055) ** 2.4
                  for channel in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


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

    def test_window_is_fixed_size_liquid_control_deck(self):
        self.app.update_idletasks()
        self.assertEqual(self.app.geometry().split("+")[0], "780x640")
        self.assertFalse(self.app.resizable()[0])
        self.assertFalse(self.app.resizable()[1])

    def test_internal_brand_banner_is_not_rendered(self):
        brand_banners = [
            text for text in widget_texts(self.app)
            if text.startswith("Jitter") and "Makcu Control" in text
        ]
        self.assertEqual(brand_banners, [])

    def test_native_window_title_carries_the_brand_name(self):
        expected = "Jitter " + chr(0x2014) + " Makcu Control"
        self.assertEqual(self.app.title(), expected)

    def test_theme_toggle_applies_dark_palette_and_persists_choice(self):
        style = ttk.Style(self.app)
        self.assertEqual(self.app.theme_button.icon, "☾")
        self.assertEqual(self.app.theme_tooltip_text,
                         "Switch to Dark Mode")
        self.app.toggle_theme()
        self.app.update_idletasks()

        self.assertEqual(self.app.theme_var.get(), "dark")
        self.assertEqual(self.app.cget("background"), "#0D1420")
        self.assertEqual(style.lookup("Liquid.Body.TLabel", "foreground"),
                         "#EEF8FF")
        self.assertEqual(self.app.motion_strength_pps_scale.cget("background"),
                         "#0D1420")
        self.assertEqual(self.app.theme_button.icon, "☀")
        self.assertEqual(self.app.theme_tooltip_text,
                         "Switch to Light Mode")
        self.assertEqual(self.app.theme_button.accessible_name,
                         "Switch to Light Mode")
        self.assertEqual(self.app.nav.cget("background"), "#0D1420")
        self.assertEqual(self.app.nav.itemcget("glass", "fill"), "#172232")
        self.assertEqual(self.app.nav.itemcget("lens", "fill"), "#63E6FF")
        self.assertEqual(self.app.theme_button.itemcget("surface", "fill"),
                         "#202F43")

        self.app._cancel_after("_save_after_id")
        self.app.save_config()
        self.assertEqual(self.store.saved[-1].theme, "dark")

    def test_every_numeric_control_uses_liquid_slider(self):
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
                    LiquidSlider,
                )

    def test_liquid_slider_user_change_updates_exact_entry_and_snapshot(self):
        slider = self.app.motion_strength_pps_scale
        slider._set_from_user(123)
        self.app.update()
        self.assertEqual(self.app.motion_strength_pps_var.get(), "123")
        self.assertEqual(self.app.get_motion_settings().strength_pps, 123.0)

    def test_exact_entry_and_preset_changes_update_liquid_slider_silently(self):
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

    def test_shell_region_order_is_identity_nav_page_runtime_footer(self):
        regions = (
            self.app.identity_frame,
            self.app.navigation_frame,
            self.app.page_host,
            self.app.runtime_frame,
            self.app.footer_frame,
        )
        self.assertEqual(
            [int(widget.grid_info()["row"]) for widget in regions],
            [0, 1, 2, 3, 4],
        )

    def test_navigation_owns_control_motion_and_advanced_pages(self):
        self.assertEqual(self.app.nav.labels, ("Control", "Motion", "Advanced"))
        self.assertEqual(
            self.app.pages,
            (self.app.control_page, self.app.motion_page, self.app.advanced_page),
        )
        for widget in (
            self.app.trigger_combo,
            self.app.modifier_combo,
            self.app.preset_combo,
            self.app.hotkey_button,
            self.app.device_label,
        ):
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.control_page))
        for widget in (
            self.app.motion_strength_pps_entry,
            self.app.jitter_rate_hz_entry,
        ):
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.motion_page))
        for key in (
            "motion_angle_deg",
            "horizontal_jitter_pps",
            "vertical_jitter_pps",
            "jitter_randomness_percent",
            "jitter_axis_phase_deg",
            "smoothness_percent",
            "ramp_up_ms",
            "update_rate_hz",
            "max_step_px",
            "acceleration_pps2",
            "deceleration_pps2",
        ):
            with self.subTest(key=key):
                self.assertTrue(self._is_descendant(
                    getattr(self.app, f"{key}_entry"), self.app.advanced_page
                ))
        for widget in (self.app.waveform_combo, self.app.motion_curve_combo):
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.advanced_page))

    def test_mini_actions_are_liquid_icon_buttons(self):
        for button in (self.app.reconnect_button, self.app.test_button,
                       self.app.theme_button):
            self.assertIsInstance(button, LiquidIconButton)
            self.assertIs(button.master, self.app.navigation_actions)
        self.assertEqual(
            [button.icon for button in (
                self.app.reconnect_button,
                self.app.test_button,
                self.app.theme_button,
            )],
            ["↻", "▶", "☾"],
        )

    def test_mini_actions_remain_visible_on_every_page(self):
        self.app.deiconify()
        self.app.update()
        for index in range(3):
            with self.subTest(index=index):
                self.app.select_page(index)
                self.app.update_idletasks()
                self.assertTrue(all(button.winfo_ismapped() for button in (
                    self.app.reconnect_button,
                    self.app.test_button,
                    self.app.theme_button,
                )))

    def test_stop_is_visible_on_every_navigation_page(self):
        self.app.deiconify()
        for index in range(3):
            with self.subTest(index=index):
                self.app.select_page(index)
                self.app.update()
                self.assertEqual(self.app.stop_button.winfo_ismapped(), 1)

    def test_close_cancels_navigation_animation(self):
        self.app.deiconify()
        self.app.update()
        self.app.nav.select(2)
        self.assertIsNotNone(self.app.nav._animation_after_id)
        animation_states_at_service_close = []
        original_close = self.service.close

        def observe_service_close():
            animation_states_at_service_close.append(
                self.app.nav._animation_after_id
            )
            original_close()

        self.service.close = observe_service_close
        self.app.close_app()
        self.assertEqual(animation_states_at_service_close, [None])
        self.assertIsNone(self.app.nav._animation_after_id)

    def test_advanced_canvas_belongs_only_to_advanced_page(self):
        self.assertTrue(self._is_descendant(self.app.advanced_canvas,
                                            self.app.advanced_page))
        self.assertFalse(self._is_descendant(self.app.stop_button,
                                             self.app.advanced_page))

    def test_invalid_advanced_edit_does_not_change_page(self):
        self.app.select_page(2)
        self.app.horizontal_jitter_pps_var.set("not-a-number")
        self.app._motion_changed("horizontal_jitter_pps")
        self.assertEqual(self.app.nav.selected_index, 2)
        self.assertTrue(self.app.footer_var.get().startswith("Invalid value for "))

    def test_mini_actions_keep_icon_button_size_and_tooltips(self):
        for button in (self.app.reconnect_button, self.app.test_button,
                       self.app.theme_button):
            with self.subTest(button=str(button)):
                self.assertEqual(int(button.cget("width")), 34)
                self.assertEqual(button.pack_info()["side"], "left")
        self.assertEqual(self.app.reconnect_tooltip_text, "Reconnect Makcu")
        self.assertEqual(self.app.test_tooltip_text, "Test Run 3s")
        self.assertEqual(self.app.reconnect_button.accessible_name,
                         "Reconnect Makcu")
        self.assertEqual(self.app.test_button.accessible_name, "Test Run 3s")
        self.assertEqual(self.app.theme_button.accessible_name,
                         "Switch to Dark Mode")
        self.assertTrue(self.app.reconnect_button.bind("<Enter>"))
        self.assertTrue(self.app.test_button.bind("<Enter>"))

        event = SimpleNamespace(widget=self.app.reconnect_button)
        self.app._show_action_tooltip(event, self.app.reconnect_tooltip_text)
        tooltip = self.app._action_tooltip
        self.assertEqual(tooltip.winfo_children()[0].cget("text"),
                         "Reconnect Makcu")
        self.app._hide_action_tooltip()
        self.assertIsNone(self.app._action_tooltip)

    def test_mini_action_tooltips_are_available_from_keyboard_focus(self):
        self.app.deiconify()
        self.app.update()
        cases = (
            (self.app.reconnect_button, "Reconnect Makcu", "_action_tooltip"),
            (self.app.test_button, "Test Run 3s", "_action_tooltip"),
            (self.app.theme_button, "Switch to Dark Mode", "_theme_tooltip"),
        )
        for button, expected_text, tooltip_attribute in cases:
            with self.subTest(button=str(button)):
                self.assertTrue(button.bind("<FocusIn>"))
                self.assertTrue(button.bind("<FocusOut>"))
                button.event_generate("<FocusIn>")
                tooltip = getattr(self.app, tooltip_attribute)
                self.assertIsNotNone(tooltip)
                self.assertEqual(
                    tooltip.winfo_children()[0].cget("text"),
                    expected_text,
                )
                button.event_generate("<FocusOut>")
                self.assertIsNone(getattr(self.app, tooltip_attribute))

    def test_select_page_shows_one_page_without_resetting_values(self):
        self.app.motion_strength_pps_var.set("123")
        self.app.select_page(2)
        self.assertEqual(self.app.nav.selected_index, 2)
        self.assertEqual(self.app.page_host.grid_slaves(), [self.app.pages[2]])
        self.app.select_page(1)
        self.assertEqual(self.app.motion_strength_pps_var.get(), "123")

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
        waveform_row = self.app.waveform_combo.master
        curve_row = self.app.motion_curve_combo.master
        self.assertIsNot(waveform_row, curve_row)
        for combo, expected_row in (
            (self.app.waveform_combo, 6),
            (self.app.motion_curve_combo, 7),
        ):
            with self.subTest(combo=str(combo)):
                choice_row = combo.master
                info = choice_row.grid_info()
                self.assertEqual(int(info["row"]), expected_row)
                self.assertEqual(int(info["columnspan"]), 2)
                self.assertIs(choice_row.master, self.app.advanced_grid)

    def test_runtime_actions_have_equal_fixed_weight(self):
        self.assertIs(self.app.enable_button.master, self.app.runtime_frame)
        self.assertIs(self.app.stop_button.master, self.app.runtime_frame)
        self.assertEqual(int(self.app.enable_button.grid_info()["column"]), 0)
        self.assertEqual(int(self.app.stop_button.grid_info()["column"]), 2)
        self.assertIn("ew", self.app.enable_button.grid_info()["sticky"])
        self.assertIn("ew", self.app.stop_button.grid_info()["sticky"])
        for column in (0, 2):
            with self.subTest(column=column):
                config = self.app.runtime_frame.grid_columnconfigure(column)
                self.assertEqual(config["weight"], 1)
                self.assertEqual(config["uniform"], "runtime_actions")
        self.assertEqual(
            self.app.runtime_frame.grid_columnconfigure(1)["weight"],
            2,
        )

    def test_footer_and_runtime_are_outside_scrollable_workspace(self):
        for widget in (self.app.footer_frame, self.app.runtime_frame,
                       self.app.enable_button, self.app.stop_button):
            with self.subTest(widget=str(widget)):
                self.assertFalse(self._is_descendant(widget, self.app.advanced_host))

    def test_identity_shows_connection_and_control_shows_device_summary(self):
        self.assertTrue(self._is_descendant(self.app.device_label,
                                            self.app.control_page))
        self.assertTrue(self._is_descendant(self.app.connection_label,
                                            self.app.identity_frame))
        self.assertFalse(self._is_descendant(self.app.reconnect_button,
                                             self.app.identity_frame))

    def test_theme_toggle_lives_in_navigation_not_identity(self):
        self.assertIs(self.app.theme_button.master, self.app.navigation_actions)
        self.assertFalse(self._is_descendant(self.app.theme_button,
                                             self.app.identity_frame))
        self.assertFalse(self._is_descendant(self.app.theme_button,
                                             self.app.advanced_host))
        self.assertEqual(self.app.theme_button.pack_info()["side"], "left")

    def test_theme_icon_tooltip_appears_on_hover_and_is_removed(self):
        self.app._show_theme_tooltip()
        tooltip = self.app._theme_tooltip
        self.assertIsNotNone(tooltip)
        self.assertEqual(tooltip.winfo_children()[0].cget("text"),
                         "Switch to Dark Mode")

        self.app._hide_theme_tooltip()
        self.assertIsNone(self.app._theme_tooltip)
        self.assertEqual(tooltip.winfo_exists(), 0)

    def test_runtime_group_keeps_stop_always_visible(self):
        self.assertTrue(self._is_descendant(self.app.stop_button,
                                            self.app.runtime_frame))
        self.assertFalse(self._is_descendant(self.app.stop_button,
                                             self.app.advanced_frame))

    def test_stop_remains_inside_application_viewport_when_advanced_is_scrolled(self):
        self.app.deiconify()
        self.app.select_page(2)
        self.app.update()
        self.app.advanced_canvas.yview_moveto(1.0)
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

    def test_liquid_styles_are_registered(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("Liquid.App.TFrame", "background"), "#F2F7FA")
        self.assertEqual(style.lookup("Liquid.Surface.TFrame", "background"),
                         "#E5F0F5")
        self.assertEqual(
            self.app.tk.splitlist(
                style.lookup("Liquid.Title.TLabel", "font")
            ),
            ("Segoe UI", "18", "bold"),
        )

    def test_advanced_scrollbar_uses_liquid_colors_in_both_themes(self):
        """Fails if Advanced scrolling falls back to the platform theme."""
        style = ttk.Style(self.app)
        self.assertEqual(
            self.app.advanced_scrollbar.cget("style"),
            "Liquid.Vertical.TScrollbar",
        )
        for expected_trough, expected_thumb, expected_arrow in (
            ("#E5F0F5", "#FFFFFF", "#263640"),
            ("#172232", "#202F43", "#EEF8FF"),
        ):
            with self.subTest(theme=self.app.theme_var.get()):
                self.assertEqual(
                    style.lookup("Liquid.Vertical.TScrollbar", "troughcolor"),
                    expected_trough,
                )
                self.assertEqual(
                    style.lookup("Liquid.Vertical.TScrollbar", "background"),
                    expected_thumb,
                )
                self.assertEqual(
                    style.lookup("Liquid.Vertical.TScrollbar", "arrowcolor"),
                    expected_arrow,
                )
            self.app.toggle_theme()

    def test_combobox_popups_use_liquid_colors_in_both_themes(self):
        """Fails if classic Tk popup Listboxes ignore the active theme."""
        combos = (
            self.app.trigger_combo,
            self.app.modifier_combo,
            self.app.preset_combo,
            self.app.waveform_combo,
            self.app.motion_curve_combo,
        )
        for expected in (
            ("#FFFFFF", "#263640", "#55DDF6", "#07252C"),
            ("#202F43", "#EEF8FF", "#63E6FF", "#07252C"),
        ):
            for combo in combos:
                with self.subTest(theme=self.app.theme_var.get(), combo=str(combo)):
                    popdown = self.app.tk.call(
                        "ttk::combobox::PopdownWindow", str(combo)
                    )
                    listbox = f"{popdown}.f.l"
                    actual = tuple(
                        self.app.tk.call(listbox, "cget", option)
                        for option in (
                            "-background",
                            "-foreground",
                            "-selectbackground",
                            "-selectforeground",
                        )
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        self.app.tk.call(
                            f"{popdown}.f.sb", "cget", "-style"
                        ),
                        "Liquid.Vertical.TScrollbar",
                    )
            self.app.toggle_theme()

    def test_light_disabled_secondary_button_remains_readable(self):
        """Fails if disabled secondary text drops below readable contrast."""
        style = ttk.Style(self.app)
        normal_background = style.lookup(
            "Liquid.Secondary.TButton", "background"
        )
        disabled_background = style.lookup(
            "Liquid.Secondary.TButton", "background", ("disabled",)
        )
        disabled_foreground = style.lookup(
            "Liquid.Secondary.TButton", "foreground", ("disabled",)
        )
        self.assertNotEqual(disabled_background, normal_background)
        self.assertGreaterEqual(
            contrast_ratio(disabled_background, disabled_foreground),
            4.5,
        )

    def test_liquid_buttons_use_high_contrast_palette(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "background"),
                         "#55DDF6")
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "foreground"),
                         "#07252C")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "background"),
                         "#C74652")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "foreground"),
                         "#FFFFFF")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "background"),
                         "#FFFFFF")
        self.assertEqual(self.app.enable_button.cget("style"),
                         "Liquid.Primary.TButton")
        self.assertEqual(self.app.stop_button.cget("style"),
                         "Liquid.Danger.TButton")

    def test_liquid_buttons_show_hover_press_and_focus_states(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "background",
                                      ("active",)), "#79E8FA")
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "background",
                                      ("pressed", "active")), "#33BDD8")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "background",
                                      ("active",)), "#DF6670")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "background",
                                      ("pressed", "active")), "#9F3140")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "background",
                                      ("active",)), "#D6F5FA")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "relief",
                                      ("pressed",)), "sunken")
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "bordercolor",
                                      ("focus",)), "#8B5CF6")

        self.app.toggle_theme()
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "background",
                                      ("disabled",)), "#34465C")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "foreground",
                                      ("disabled",)), "#91A5B8")

    def test_required_actions_are_present_and_stop_is_outside_advanced(self):
        texts = widget_texts(self.app)
        for expected in ("Enable Jitter", "STOP", "Advanced Settings"):
            self.assertIn(expected, texts)
        self.assertEqual(self.app.reconnect_button.icon, "↻")
        self.assertEqual(self.app.test_button.icon, "▶")
        stop = self.app.stop_button
        ancestor = stop.master
        while ancestor is not self.app:
            self.assertIsNot(ancestor, self.app.advanced_frame)
            ancestor = ancestor.master

    def test_page_selection_does_not_change_outer_geometry(self):
        self.app.update_idletasks()
        before = self.app.geometry().split("+")[0]
        self.app.select_page(2)
        self.app.update_idletasks()
        after = self.app.geometry().split("+")[0]
        self.assertEqual(after, before)

    def test_advanced_controls_are_mounted_in_the_persistent_page(self):
        self.assertEqual(self.app.advanced_frame.winfo_manager(), "pack")
        self.assertTrue(self._is_descendant(self.app.advanced_frame,
                                            self.app.advanced_page))
        self.assertIs(self.app.advanced_canvas.master, self.app.advanced_host)

    def test_mousewheel_scrolls_only_over_advanced_page(self):
        self.app.deiconify()
        self.app.select_page(2)
        self.app.update()
        self.app.advanced_canvas.yview_moveto(0.0)

        advanced_event = SimpleNamespace(
            delta=-120,
            x_root=self.app.advanced_canvas.winfo_rootx() + 10,
            y_root=self.app.advanced_canvas.winfo_rooty() + 10,
        )
        self.assertEqual(self.app._on_advanced_mousewheel(advanced_event), "break")
        self.app.update_idletasks()
        after_advanced = self.app.advanced_canvas.yview()[0]
        self.assertGreater(after_advanced, 0.0)

        self.app.select_page(0)
        self.app.update_idletasks()
        control_event = SimpleNamespace(
            delta=-120,
            x_root=self.app.control_page.winfo_rootx() + 10,
            y_root=self.app.control_page.winfo_rooty() + 10,
        )
        self.assertIsNone(self.app._on_advanced_mousewheel(control_event))
        self.assertEqual(self.app.advanced_canvas.yview()[0], after_advanced)


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

    def test_close_cancels_queue_polling_callback(self):
        queue_poll_id = self.app._ui_pump_after_id
        self.assertIsNotNone(queue_poll_id)
        cancelled = []
        original_after_cancel = self.app.after_cancel

        def recording_after_cancel(callback_id):
            cancelled.append(callback_id)
            return original_after_cancel(callback_id)

        self.app.after_cancel = recording_after_cancel
        self.app.close_app()
        self.assertIn(queue_poll_id, cancelled)
        self.assertIsNone(self.app._ui_pump_after_id)

    def test_queue_drain_yields_after_a_bounded_batch(self):
        self.app._cancel_after("_ui_pump_after_id")
        handled = []
        scheduled = []
        self.app.handle_service_event = handled.append

        def recording_after(delay, callback):
            scheduled.append((delay, callback))
            return f"scheduled-{len(scheduled)}"

        self.app.after = recording_after
        for index in range(200):
            self.app._ui_queue.put(("service", ServiceEvent("item", index)))

        self.app._drain_ui_queue()

        self.assertGreater(len(handled), 0)
        self.assertLess(len(handled), 200)
        self.assertEqual(scheduled[-1][0], 0)

    def test_queue_drain_recovers_from_handler_failure_and_keeps_polling(self):
        self.app._cancel_after("_ui_pump_after_id")
        hotkeys = []
        scheduled = []

        def failing_handler(_event):
            raise RuntimeError("bad service event")

        def recording_after(delay, callback):
            scheduled.append((delay, callback))
            return f"scheduled-{len(scheduled)}"

        self.app.handle_service_event = failing_handler
        self.app.toggle_enabled = lambda: hotkeys.append("handled")
        self.app.after = recording_after
        self.app._ui_queue.put(("service", ServiceEvent("bad")))
        self.app._ui_queue.put(("hotkey", None))

        with self.assertLogs(level="ERROR") as captured_logs:
            try:
                self.app._drain_ui_queue()
            except RuntimeError as exc:
                self.fail(f"queue handler failure escaped the UI pump: {exc}")
            if not hotkeys:
                scheduled[-1][1]()

        self.assertEqual(hotkeys, ["handled"])
        self.assertTrue(scheduled)
        self.assertIn("UI queue handler failed", captured_logs.output[0])

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

    def test_test_run_button_is_disabled_only_while_test_is_active(self):
        self.assertTrue(self.app.test_button._enabled)
        self.app.start_test_run()
        self.assertTrue(self.app.test_button._enabled)

        self.service.connected = True
        self.app.start_test_run()
        self.assertFalse(self.app.test_button._enabled)
        self.app.handle_service_event(
            ServiceEvent("motion_stopped", "duration_complete")
        )
        self.assertTrue(self.app.test_button._enabled)

        self.app.start_test_run()
        self.assertFalse(self.app.test_button._enabled)
        self.app.emergency_stop("Stopped by user")
        self.assertTrue(self.app.test_button._enabled)

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

    def test_worker_callbacks_never_call_tk_scheduling_directly(self):
        release = threading.Event()
        after_called = threading.Event()

        def blocking_after(*_args):
            after_called.set()
            release.wait(1.0)

        self.app.after = blocking_after
        workers = (
            threading.Thread(
                target=self.app.queue_service_event,
                args=(ServiceEvent("connected"),),
            ),
            threading.Thread(target=self.app._hotkey_pressed),
        )
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(0.1)
            self.assertFalse(after_called.is_set())
            self.assertTrue(all(not worker.is_alive() for worker in workers))
        finally:
            release.set()
            for worker in workers:
                worker.join(1.0)

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
        self.assertEqual(self.app.motion_angle_deg_entry.cget("style"),
                         "Liquid.Invalid.TEntry")
        self.assertEqual(
            self.app.footer_var.get(),
            "Invalid value for motion angle deg",
        )
        self.app.preset_var.set("Balanced")
        self.app.apply_preset()
        self.assertEqual(self.app.motion_angle_deg_entry.cget("style"),
                         "Liquid.Entry.TEntry")
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

        self.assertEqual(self.app.motion_strength_pps_entry.cget("style"),
                         "Liquid.Entry.TEntry")
        self.assertEqual(self.app.footer_var.get(), "Ready")


if __name__ == "__main__":
    unittest.main()
