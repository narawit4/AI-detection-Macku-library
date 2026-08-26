import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import threading
import unittest

from ai_service import AiEvent
from ai_targeting import AimSettings
from ui import JitterApp
from makcu_service import ServiceEvent
from liquid_widgets import LiquidIconButton, LiquidSlider
from motion import MotionSettings
from settings import AppConfig
from sound_service import ToggleSoundPlayer


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
        self.reconnect_hook = None
        self.motion_calls = []
        self.ai_motion_calls = []
        self.stop_reasons = []
        self.stop_hook = None
        self._motion_active = False
        self.motion_active_hook = None
        self.motion_generation = 0
        self.active_motion_generation = None
        self.start_motion_hook = None
        self.start_ai_motion_hook = None

    @property
    def motion_active(self):
        # Mirror the production service's lock-linearized property read while
        # still allowing a terminal callback to be queued immediately after
        # the observed value.  This makes ownership handoff races completely
        # deterministic without sleeping.
        active = self._motion_active
        if self.motion_active_hook is not None:
            self.motion_active_hook()
        return active

    @motion_active.setter
    def motion_active(self, active):
        self._motion_active = bool(active)

    def connect(self):
        self.started += 1

    def reconnect(self):
        self.reconnects += 1
        if self.reconnect_hook is not None:
            self.reconnect_hook()
        # Match MakcuService.reconnect(): it signals cancellation internally
        # without crossing the public stop_motion() return barrier on Tk's
        # thread, then invalidates the retiring connection generation.
        self.connected = False
        self.motion_active = False
        self.active_motion_generation = None
        return None

    def start_motion(self, settings_provider, duration_s=None):
        return self.start_motion_source(settings_provider, duration_s) is not None

    def start_motion_source(self, settings_provider, duration_s=None):
        self.started += 1
        self.motion_calls.append((settings_provider, duration_s))
        source = None
        if self.connected:
            if self.motion_active:
                source = self.active_motion_generation
            else:
                self.motion_generation += 1
                source = self.motion_generation
                self.active_motion_generation = source
                self.motion_active = True
        if self.start_motion_hook is not None:
            self.start_motion_hook()
        return source

    def start_ai_motion(
        self,
        snapshot_provider,
        settings_provider,
        duration_s=None,
    ):
        return self.start_ai_motion_source(
            snapshot_provider,
            settings_provider,
            duration_s,
        ) is not None

    def start_ai_motion_source(
        self,
        snapshot_provider,
        settings_provider,
        duration_s=None,
    ):
        self.started += 1
        self.ai_motion_calls.append(
            (snapshot_provider, settings_provider, duration_s)
        )
        source = None
        if self.connected:
            if self.motion_active:
                source = self.active_motion_generation
            else:
                self.motion_generation += 1
                source = self.motion_generation
                self.active_motion_generation = source
                self.motion_active = True
        if self.start_ai_motion_hook is not None:
            self.start_ai_motion_hook()
        return source

    def stop_motion(self, reason="manual"):
        self.stopped += 1
        self.stop_reasons.append(reason)
        if self.stop_hook is not None:
            self.stop_hook(reason)

    def emit(self, event):
        if event.kind in {"motion_error", "motion_stopped"}:
            source = event.motion_generation
            if source is None:
                source = self.active_motion_generation
                event = ServiceEvent(event.kind, event.payload, source)
            if source == self.active_motion_generation:
                self.motion_active = False
                self.active_motion_generation = None
        self.event_sink(event)

    def close(self):
        self.closed += 1


class StubAiService:
    def __init__(self):
        self.event_sink = None
        self.start_calls = []
        self.stop_calls = []
        self.closed = 0
        self.snapshot = object()
        self.generation = 0
        self.active_generation = None
        self.start_result = True
        self.start_exception = None
        self.stop_hook = None

    def with_sink(self, event_sink):
        self.event_sink = event_sink
        return self

    def start(self, settings_provider):
        self.start_calls.append(settings_provider)
        if self.start_exception is not None:
            raise self.start_exception
        if not self.start_result:
            return self.start_result
        self.generation += 1
        self.active_generation = self.generation
        return self.active_generation

    def stop(self, reason="manual"):
        self.stop_calls.append(reason)
        if self.stop_hook is not None:
            self.stop_hook(self.active_generation)
        self.active_generation = None

    def emit(self, event, *, generation=None):
        source_generation = (
            self.active_generation if generation is None else generation
        )
        if (
            self.active_generation is None
            or source_generation != self.active_generation
        ):
            return False
        self.event_sink(event)
        return True

    def close(self):
        self.closed += 1

    def latest_snapshot(self):
        return self.snapshot


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


class StubSounds:
    def __init__(self):
        self.played = []
        self.forced = []
        self.configured = []
        self.closed = 0

    def play(self, enabled, *, force=False):
        self.played.append(bool(enabled))
        self.forced.append(bool(force))

    def configure(self, *, enabled, volume):
        self.configured.append((bool(enabled), int(volume)))

    def close(self):
        self.closed += 1


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


def descendant_widgets(widget):
    values = []
    for child in widget.winfo_children():
        values.append(child)
        values.extend(descendant_widgets(child))
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
        self.app = None
        self.make_app()

    def make_app(self, *, config=None):
        self.service = None
        self.store = StubStore(config or AppConfig())
        self.ai = StubAiService()

        def service_factory(event_sink):
            self.service = StubService(event_sink)
            return self.service

        self.app = JitterApp(
            config_store=self.store,
            service_factory=service_factory,
            ai_service_factory=lambda sink: self.ai.with_sink(sink),
            hotkey_factory=StubHotkey,
            sound_player=StubSounds(),
            auto_start=False,
        )
        self.app.withdraw()
        return self.app

    def tearDown(self):
        try:
            if self.app is not None:
                self.app.close_app()
        except tk.TclError:
            pass

    def test_ai_controls_reflect_config_and_mode(self):
        self.app.close_app()
        app = self.make_app(config=AppConfig(
            mode="ai_aim",
            ai=AimSettings(0.5, 0.6, 0.7, 30),
        ))

        self.assertEqual(app.mode_var.get(), "ai_aim")
        self.assertEqual(app.ai_vars["confidence"].get(), "0.5")
        self.assertEqual(app.ai_vars["aim_strength"].get(), "0.6")
        self.assertEqual(app.ai_vars["smoothing"].get(), "0.7")
        self.assertEqual(app.ai_vars["max_step"].get(), "30")
        self.assertEqual(app.ai_status_var.get(), "Stopped")
        self.assertEqual(app.ai_fps_var.get(), "0 FPS")
        self.assertFalse(app.enabled)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.enable_button.cget("text"), "Enable AI Aim")
        self.assertEqual(app.motion_hero_card.winfo_manager(), "")
        self.assertEqual(app.ai_settings_card.winfo_manager(), "grid")

    def test_ai_service_is_injected_after_widgets_without_autostart(self):
        self.assertIs(self.ai.event_sink.__self__, self.app)
        self.assertEqual(self.ai.event_sink.__func__, self.app.queue_ai_event.__func__)
        self.assertEqual(self.ai.start_calls, [])

    def test_mode_selector_shows_only_the_selected_motion_cards(self):
        self.assertEqual(self.app.mode_combo.cget("values"), ("Jitter", "AI Aim"))
        self.assertEqual(self.app.motion_hero_card.winfo_manager(), "grid")
        self.assertEqual(self.app.motion_summary_card.winfo_manager(), "grid")
        self.assertEqual(self.app.ai_settings_card.winfo_manager(), "")
        self.assertEqual(self.app.ai_status_card.winfo_manager(), "")

        self.app.mode_var.set("ai_aim")
        self.app.on_mode_changed()

        self.assertEqual(self.app.motion_hero_card.winfo_manager(), "")
        self.assertEqual(self.app.motion_summary_card.winfo_manager(), "")
        self.assertEqual(self.app.ai_settings_card.winfo_manager(), "grid")
        self.assertEqual(self.app.ai_status_card.winfo_manager(), "grid")
        self.app.update_idletasks()
        self.assertEqual(self.app.geometry().split("+")[0], "840x620")
        self.assertEqual(self.app.stop_button.winfo_manager(), "grid")

    def test_ai_numeric_controls_use_approved_ranges_and_exact_entries(self):
        expected = {
            "confidence": (0.05, 0.95, 0.01),
            "aim_strength": (0.05, 2.0, 0.01),
            "smoothing": (0.0, 0.95, 0.01),
            "max_step": (1.0, 127.0, 1.0),
        }
        for key, limits in expected.items():
            with self.subTest(key=key):
                slider = getattr(self.app, f"ai_{key}_scale")
                entry = getattr(self.app, f"ai_{key}_entry")
                self.assertIsInstance(slider, LiquidSlider)
                self.assertEqual(
                    (slider.from_, slider.to, slider.resolution),
                    limits,
                )
                self.assertEqual(entry.cget("textvariable"), str(self.app.ai_vars[key]))

    def test_get_ai_settings_returns_configured_immutable_snapshot(self):
        self.assertEqual(self.app.get_ai_settings(), self.app.config.ai)

    def test_ai_slider_change_updates_exact_entry_and_snapshot(self):
        self.app.ai_confidence_scale._set_from_user(0.55)
        self.app.update()

        self.assertEqual(self.app.ai_vars["confidence"].get(), "0.55")
        self.assertEqual(self.app.get_ai_settings().confidence, 0.55)

    def test_window_is_fixed_size_liquid_split_console(self):
        self.app.update_idletasks()
        self.assertEqual(self.app.geometry().split("+")[0], "840x620")
        self.assertEqual(self.app.resizable(), (False, False))

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
        self.assertEqual(self.app.pulse_size_px_scale.cget("background"),
                         "#172232")
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

    def test_navigation_palette_api_tracks_the_active_theme(self):
        palette_builder = getattr(self.app, "_navigation_palette", None)
        self.assertIsNotNone(palette_builder)
        self.assertEqual(
            palette_builder()["background"],
            "#F2F7FA",
        )
        self.app.toggle_theme()
        self.assertEqual(
            palette_builder()["background"],
            "#0D1420",
        )

    def test_every_numeric_control_uses_liquid_slider(self):
        numeric_keys = ("pulse_size_px", "pulse_rate_hz")
        for key in numeric_keys:
            with self.subTest(key=key):
                self.assertIsInstance(
                    getattr(self.app, f"{key}_scale"),
                    LiquidSlider,
                )

    def test_liquid_slider_user_change_updates_exact_entry_and_snapshot(self):
        slider = self.app.pulse_size_px_scale
        slider._set_from_user(4)
        self.app.update()
        self.assertEqual(self.app.pulse_size_px_var.get(), "4")
        self.assertEqual(self.app.get_motion_settings().pulse_size_px, 4.0)

    def test_exact_entry_and_preset_changes_update_liquid_slider_silently(self):
        slider = self.app.pulse_size_px_scale
        self.app.pulse_size_px_var.set("7")
        self.app.update()
        self.assertEqual(slider.get(), 7.0)
        self.app.preset_var.set("Balanced")
        self.app.apply_preset()
        self.app.update()
        self.assertEqual(
            slider.get(),
            self.app.get_motion_settings().pulse_size_px,
        )

    def _is_descendant(self, widget, ancestor):
        current = widget
        while current is not self.app:
            if current is ancestor:
                return True
            current = current.master
        return False

    def _descendants(self, widget):
        for child in widget.winfo_children():
            yield child
            yield from self._descendants(child)

    def _assert_device_summary_contained(self):
        self.app.deiconify()
        self.app.update()
        label_right = (
            self.app.device_label.winfo_rootx()
            + self.app.device_label.winfo_width()
        )
        card_right = (
            self.app.control_device_card.winfo_rootx()
            + self.app.control_device_card.winfo_width()
        )
        self.assertLessEqual(len(self.app.device_status_var.get()), 40)
        self.assertGreater(
            self.app.control_bindings_card.winfo_width(),
            self.app.control_device_card.winfo_width(),
        )
        self.assertLessEqual(label_right, card_right)

    def test_shell_uses_persistent_rail_and_console_columns(self):
        self.assertIs(self.app.navigation_rail.master, self.app.shell)
        self.assertIs(self.app.console_workspace.master, self.app.shell)
        self.assertEqual(int(self.app.navigation_rail.grid_info()["column"]), 0)
        self.assertEqual(int(self.app.console_workspace.grid_info()["column"]), 1)
        self.assertEqual(int(self.app.navigation_rail.cget("width")), 176)
        self.assertEqual(self.app.nav.orientation, "vertical")
        self.app.deiconify()
        self.app.update()
        self.assertEqual(self.app.navigation_rail.winfo_width(), 176)

    def test_rail_owns_identity_connection_navigation_and_mini_actions(self):
        for widget in (
            self.app.rail_identity,
            self.app.connection_indicator,
            self.app.nav,
            self.app.navigation_actions,
        ):
            with self.subTest(widget=str(widget)):
                self.assertTrue(
                    self._is_descendant(widget, self.app.navigation_rail)
                )
        for button in (
            self.app.reconnect_button,
            self.app.test_button,
            self.app.theme_button,
        ):
            with self.subTest(button=str(button)):
                self.assertIs(button.master, self.app.navigation_actions)
        self.assertEqual(
            self.app.navigation_actions.pack_info()["side"],
            "bottom",
        )

    def test_workspace_keeps_page_footer_runtime_order(self):
        widgets = (
            self.app.page_host,
            self.app.footer_frame,
            self.app.runtime_frame,
        )
        self.assertEqual(
            [int(widget.grid_info()["row"]) for widget in widgets],
            [0, 1, 2],
        )
        for widget in widgets:
            with self.subTest(widget=str(widget)):
                self.assertIs(widget.master, self.app.console_workspace)

    def test_split_console_shell_preserves_semantic_layers_in_both_themes(self):
        """Fails if the shell drops its graded and rounded Canvas surfaces."""
        shell = getattr(self.app, "shell", None)
        self.assertIsInstance(shell, tk.Canvas)
        self.app.deiconify()
        self.app.update()

        required_tags = (
            "rail-surface",
            "workspace-band",
            "rounded-surface",
            "floating-panel",
            "floating-panel-rail",
            "floating-panel-page",
            "floating-panel-runtime",
        )
        themed_layers = {}
        for theme in ("light", "dark"):
            with self.subTest(theme=theme):
                self.assertEqual(self.app.theme_var.get(), theme)
                for tag in required_tags:
                    self.assertTrue(shell.find_withtag(tag), tag)
                self.assertFalse(shell.find_withtag("panel-highlight"))
                for item in shell.find_withtag("floating-panel"):
                    self.assertTrue(shell.itemcget(item, "outline"))
                    self.assertEqual(float(shell.itemcget(item, "width")), 1.0)
                rail_fill = shell.itemcget(
                    shell.find_withtag("rail-surface")[0], "fill"
                )
                workspace_fills = tuple(
                    shell.itemcget(item, "fill")
                    for item in shell.find_withtag("workspace-band")
                )
                self.assertNotIn(rail_fill, workspace_fills)
                themed_layers[theme] = (rail_fill, workspace_fills)
            self.app.toggle_theme()
            self.app.update()
        self.assertNotEqual(themed_layers["light"], themed_layers["dark"])

    def test_connection_indicator_has_glow_and_semantic_state_tags(self):
        """Fails if connection state returns to a text-only indicator."""
        indicator = getattr(self.app, "connection_indicator", None)
        self.assertIsInstance(indicator, tk.Canvas)
        self.assertLessEqual(indicator.winfo_reqwidth(), 24)
        self.assertTrue(indicator.find_withtag("status-glow"))
        self.assertTrue(indicator.find_withtag("status-marker"))
        self.assertTrue(indicator.find_withtag("status-disconnected"))

        disconnected_fill = indicator.itemcget("status-marker", "fill")
        self.app.handle_service_event(ServiceEvent("connecting"))
        self.assertTrue(indicator.find_withtag("status-connecting"))
        connecting_fill = indicator.itemcget("status-marker", "fill")
        self.app.handle_service_event(ServiceEvent("connected", "Fake Makcu"))
        self.assertTrue(indicator.find_withtag("status-connected"))
        connected_fill = indicator.itemcget("status-marker", "fill")
        self.assertEqual(len({disconnected_fill, connecting_fill, connected_fill}), 3)

    def test_navigation_contains_control_motion_and_settings(self):
        self.assertEqual(self.app.nav.labels, ("Control", "Motion", "Settings"))
        self.assertEqual(
            self.app.pages,
            (self.app.control_page, self.app.motion_page, self.app.settings_page),
        )
        self.assertFalse(hasattr(self.app, "advanced_page"))
        self.assertFalse(hasattr(self.app, "advanced_canvas"))
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
            self.app.pulse_size_px_entry,
            self.app.pulse_rate_hz_entry,
            self.app.ramp_mode_combo,
        ):
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.motion_page))

    def test_settings_page_exposes_persisted_sound_controls(self):
        self.assertTrue(self.app.sound_enabled_var.get())
        self.assertEqual(self.app.sound_volume_var.get(), "70")
        self.assertEqual(self.app.sound_volume_scale.from_, 0.0)
        self.assertEqual(self.app.sound_volume_scale.to, 100.0)
        for widget in (
            self.app.sound_enabled_check,
            self.app.sound_volume_entry,
            self.app.sound_volume_scale,
            self.app.test_on_button,
            self.app.test_off_button,
        ):
            with self.subTest(widget=str(widget)):
                self.assertTrue(self._is_descendant(widget, self.app.settings_page))

    def test_sound_preview_uses_two_compact_labeled_rows(self):
        actions = self.app.test_on_button.master
        self.assertIs(self.app.test_off_button.master, actions)
        labels = {
            child.cget("text"): child
            for child in actions.winfo_children()
            if isinstance(child, ttk.Label)
        }
        self.assertEqual(set(labels), {"ARMED CUE", "DISABLED CUE"})
        separators = [
            child for child in actions.winfo_children()
            if isinstance(child, ttk.Separator)
        ]
        self.assertEqual(len(separators), 1)

        self.assertEqual(self.app.test_on_button.cget("text"), "\u25b6")
        self.assertEqual(self.app.test_off_button.cget("text"), "\u25b6")
        self.assertEqual(
            self.app.test_on_button.cget("style"),
            "Liquid.CompactPrimary.TButton",
        )
        self.assertEqual(
            self.app.test_off_button.cget("style"),
            "Liquid.CompactSecondary.TButton",
        )
        self.assertEqual(int(self.app.test_on_button.cget("width")), 2)
        self.assertEqual(int(self.app.test_off_button.cget("width")), 2)

        armed_grid = labels["ARMED CUE"].grid_info()
        disabled_grid = labels["DISABLED CUE"].grid_info()
        on_grid = self.app.test_on_button.grid_info()
        off_grid = self.app.test_off_button.grid_info()
        self.assertEqual(
            (
                int(armed_grid["row"]),
                int(armed_grid["column"]),
                int(on_grid["row"]),
                int(on_grid["column"]),
            ),
            (0, 0, 0, 1),
        )
        self.assertEqual(
            (
                int(disabled_grid["row"]),
                int(disabled_grid["column"]),
                int(off_grid["row"]),
                int(off_grid["column"]),
            ),
            (2, 0, 2, 1),
        )
        self.assertEqual(int(separators[0].grid_info()["row"]), 1)
        self.assertEqual(int(separators[0].grid_info()["columnspan"]), 2)
        self.assertEqual(int(actions.grid_columnconfigure(0)["weight"]), 1)

    def test_sound_preview_buttons_invoke_their_matching_cues(self):
        self.app.test_on_button.invoke()
        self.app.test_off_button.invoke()

        self.assertEqual(self.app.sound_player.played[-2:], [True, False])
        self.assertEqual(self.app.sound_player.forced[-2:], [True, True])

    def test_settings_page_uses_header_and_three_to_two_dashboard(self):
        self.assertEqual(self.app.settings_title_label.cget("text"), "SETTINGS")
        self.assertTrue(
            self._is_descendant(
                self.app.settings_title_label, self.app.settings_page
            )
        )
        self.assertEqual(
            tuple(
                int(self.app.settings_content.grid_columnconfigure(column)[
                    "weight"
                ])
                for column in (0, 1)
            ),
            (3, 2),
        )
        self.assertEqual(
            int(self.app.sound_feedback_card.grid_info()["column"]), 0
        )
        self.assertEqual(
            int(self.app.sound_preview_card.grid_info()["column"]), 1
        )
        self.assertTrue(
            self._is_descendant(
                self.app.sound_preview_card, self.app.settings_content
            )
        )

    def test_settings_has_no_duplicate_theme_controls(self):
        self.app.deiconify()
        self.app.select_page(2)
        self.app.update()
        self.assertTrue(self.app.settings_page.winfo_ismapped())
        self.assertTrue(self.app.stop_button.winfo_ismapped())
        visible_text = set(widget_texts(self.app.settings_page))
        self.assertNotIn("APPEARANCE", visible_text)
        self.assertNotIn("Dark", visible_text)
        self.assertNotIn("Light", visible_text)
        self.assertTrue(
            self._is_descendant(
                self.app.theme_button, self.app.navigation_actions
            )
        )
        self.assertLessEqual(
            self.app.stop_button.winfo_rooty()
            + self.app.stop_button.winfo_height()
            - self.app.winfo_rooty(),
            self.app.winfo_height(),
        )

    def test_settings_small_text_meets_contrast_in_both_themes(self):
        style = ttk.Style(self.app)
        for theme in ("light", "dark"):
            cases = (
                (
                    "eyebrow",
                    "Liquid.SettingsEyebrow.TLabel",
                    "Liquid.App.TFrame",
                ),
                (
                    "card-copy",
                    "Liquid.CardBody.TLabel",
                    "Liquid.SettingsCard.TFrame",
                ),
                (
                    "volume-unit",
                    "Liquid.VolumeUnit.TLabel",
                    "Liquid.SettingsCard.TFrame",
                ),
                (
                    "page-muted",
                    "Liquid.Muted.TLabel",
                    "Liquid.App.TFrame",
                ),
                (
                    "field-caption",
                    "Liquid.DropdownLabel.TLabel",
                    "Liquid.SettingsCard.TFrame",
                ),
            )
            for name, text_style, background_style in cases:
                foreground = style.lookup(text_style, "foreground")
                background = style.lookup(background_style, "background")
                with self.subTest(theme=theme, label=name):
                    self.assertGreaterEqual(
                        contrast_ratio(foreground, background), 4.5
                    )
            self.app.toggle_theme()

    def test_settings_volume_slider_blends_with_card_in_both_themes(self):
        style = ttk.Style(self.app)
        for theme in ("light", "dark"):
            with self.subTest(theme=theme):
                self.assertEqual(
                    self.app.sound_volume_scale.cget("background"),
                    style.lookup("Liquid.SettingsCard.TFrame", "background"),
                )
            self.app.toggle_theme()

    def test_motion_page_exposes_only_paired_pulse_controls(self):
        self.assertEqual(
            set(self.app.motion_vars),
            {"pulse_size_px", "pulse_rate_hz", "ramp_mode"},
        )
        self.assertEqual(
            self.app.ramp_mode_combo.cget("values"),
            ("Instant", "Smooth"),
        )
        self.assertEqual(
            self.app.preset_values,
            ("Custom", "Soft", "Balanced", "Strong"),
        )
        self.assertEqual(self.app.pulse_rate_hz_scale.from_, 20.0)
        self.assertEqual(self.app.pulse_rate_hz_scale.to, 120.0)
        self.assertEqual(self.app.pulse_rate_hz_var.get(), "60")

    def test_motion_numeric_entries_use_compact_width(self):
        for entry in (
            self.app.pulse_size_px_entry,
            self.app.pulse_rate_hz_entry,
        ):
            with self.subTest(entry=str(entry)):
                self.assertEqual(int(entry.cget("width")), 5)

    def test_dropdowns_use_modern_field_panels(self):
        fields = (
            (self.app.trigger_combo, "Trigger"),
            (self.app.modifier_combo, "Modifier"),
            (self.app.preset_combo, "Preset"),
            (self.app.ramp_mode_combo, "Ramp Mode"),
        )
        for combo, label_text in fields:
            with self.subTest(field=label_text):
                self.assertEqual(combo.winfo_manager(), "grid")
                info = combo.grid_info()
                self.assertEqual(info["sticky"], "ew")
                self.assertEqual(
                    combo.master.cget("style"), "Liquid.DropdownField.TFrame"
                )
                self.assertEqual(combo.cget("style"), "Liquid.Modern.TCombobox")
                labels = [
                    child for child in combo.master.winfo_children()
                    if isinstance(child, ttk.Label)
                ]
                self.assertEqual(len(labels), 1)
                self.assertEqual(labels[0].cget("text"), label_text.upper())
                self.assertEqual(
                    labels[0].cget("style"), "Liquid.DropdownLabel.TLabel"
                )
                self.assertEqual(int(labels[0].grid_info()["row"]), 0)
                self.assertEqual(int(info["row"]), 1)

    def test_modern_dropdown_style_has_spacious_accent_states(self):
        style = ttk.Style(self.app)
        expectations = (
            ("#FFFFFF", "#D6F5FA", "#E5F0F5", "#55DDF6"),
            ("#202F43", "#2A3B52", "#172232", "#63E6FF"),
        )
        for normal, hover, focus, accent in expectations:
            padding = style.lookup("Liquid.Modern.TCombobox", "padding")
            padding_values = (
                padding
                if isinstance(padding, (tuple, list))
                else self.app.tk.splitlist(str(padding))
            )
            self.assertEqual(
                tuple(int(value) for value in padding_values),
                (10, 7),
            )
            self.assertEqual(
                style.lookup(
                    "Liquid.Modern.TCombobox",
                    "fieldbackground",
                    ("readonly",),
                ),
                normal,
            )
            self.assertEqual(
                style.lookup(
                    "Liquid.Modern.TCombobox",
                    "fieldbackground",
                    ("readonly", "active"),
                ),
                hover,
            )
            self.assertEqual(
                style.lookup(
                    "Liquid.Modern.TCombobox",
                    "fieldbackground",
                    ("readonly", "focus"),
                ),
                focus,
            )
            self.assertEqual(
                style.lookup(
                    "Liquid.Modern.TCombobox", "arrowcolor", ("readonly",)
                ),
                accent,
            )
            self.assertIn(
                "Rounded", style.layout("Liquid.Modern.TCombobox")[0][0]
            )
            self.app.toggle_theme()

    def test_dropdown_rounded_element_renders_a_focus_state(self):
        class CapturingStyle:
            def __init__(self):
                self.created = None

            def element_names(self):
                return ()

            def element_create(self, *args, **kwargs):
                self.created = (args, kwargs)

        style = CapturingStyle()
        original_factory = self.app._rounded_style_image
        self.app._rounded_style_image = (
            lambda fill, border: f"image:{fill}:{border}"
        )
        try:
            self.app._install_rounded_element(
                style,
                "DropdownFocusProbe",
                ("normal", "hover", "pressed", "disabled", "border"),
                focus=("focus-fill", "focus-border"),
            )
        finally:
            self.app._rounded_style_image = original_factory

        args, _kwargs = style.created
        self.assertIn(
            ("focus", "image:focus-fill:focus-border"),
            args,
        )

    def test_binding_dropdown_cards_form_equal_two_column_row(self):
        trigger_card = self.app.trigger_combo.master
        modifier_card = self.app.modifier_combo.master
        self.assertIs(trigger_card.master, self.app.control_bindings_card)
        self.assertIs(modifier_card.master, self.app.control_bindings_card)
        self.assertEqual(int(trigger_card.grid_info()["row"]), 2)
        self.assertEqual(int(modifier_card.grid_info()["row"]), 2)
        self.assertEqual(
            (int(trigger_card.grid_info()["column"]),
             int(modifier_card.grid_info()["column"])),
            (0, 1),
        )
        self.assertEqual(
            tuple(
                int(self.app.control_bindings_card.grid_columnconfigure(column)[
                    "weight"
                ])
                for column in (0, 1)
            ),
            (1, 1),
        )

    def test_navigation_uses_compact_equal_button_layout(self):
        self.assertEqual(int(self.app.nav.cget("height")), 168)
        first = self.app.nav._item_bounds(0)
        second = self.app.nav._item_bounds(1)
        self.assertEqual(first[3] - first[1], second[3] - second[1])

    def test_split_console_control_uses_exact_three_to_two_columns(self):
        self.assertEqual(
            int(self.app.control_bindings_card.grid_info()["column"]), 0
        )
        self.assertEqual(
            int(self.app.control_device_card.grid_info()["column"]), 1
        )
        self.assertEqual(
            tuple(
                int(self.app.control_page.grid_columnconfigure(column)["weight"])
                for column in (0, 1)
            ),
            (3, 2),
        )
        for widget in (
            self.app.trigger_combo,
            self.app.modifier_combo,
            self.app.hotkey_button,
        ):
            self.assertTrue(
                self._is_descendant(widget, self.app.control_bindings_card)
            )
        self.assertTrue(
            self._is_descendant(self.app.preset_combo,
                                self.app.control_device_card)
        )
        self.assertTrue(
            self._is_descendant(self.app.device_label,
                                self.app.control_device_card)
        )

    def test_control_page_uses_dashboard_header_and_surface_cards(self):
        self.assertEqual(self.app.control_title_label.cget("text"), "CONTROL")
        self.assertEqual(
            int(self.app.control_title_label.master.grid_info()["row"]), 0
        )
        for card in (
            self.app.control_bindings_card,
            self.app.control_device_card,
        ):
            with self.subTest(card=str(card)):
                self.assertEqual(
                    card.cget("style"), "Liquid.SettingsCard.TFrame"
                )
                self.assertEqual(int(card.grid_info()["row"]), 1)

    def test_split_console_device_summary_stays_inside_device_card(self):
        """Fails if real Makcu diagnostics overflow the Device card."""
        self.app.handle_service_event(ServiceEvent(
            "connected",
            "{'port': 'COM3', 'description': 'USB-Enhanced-SERIAL CH343 "
            "(COM3)', 'vid': '0x1a86', 'pid': '0x55d3'}",
        ))
        self.assertEqual(self.app.device_status_var.get(), "Makcu on COM3")
        self._assert_device_summary_contained()

    def test_split_console_device_summary_parses_pipe_inside_metadata(self):
        payload = (
            "{'description': 'USB | Serial', 'port': 'COM6'} | 1.0"
        )
        self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(self.app.device_status_var.get(), "Makcu on COM6")
        self._assert_device_summary_contained()

    def test_split_console_device_summary_normalizes_valid_port(self):
        payload = "{'port': ' com6 '} | 1.0"
        self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(self.app.device_status_var.get(), "Makcu on COM6")
        self._assert_device_summary_contained()

    def test_split_console_device_summary_contains_malformed_metadata(self):
        payload = "{'port': 'COM7'"
        self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(
            self.app.device_status_var.get(), "Makcu device connected"
        )
        self._assert_device_summary_contained()

    def test_split_console_device_summary_contains_very_long_diagnostic(self):
        payload = "unstructured diagnostic " + "x" * 100
        self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(
            self.app.device_status_var.get(), "Makcu device connected"
        )
        self._assert_device_summary_contained()

    def test_split_console_device_summary_contains_missing_port(self):
        payload = "{'description': 'USB | Serial'} | 1.0"
        self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(
            self.app.device_status_var.get(), "Makcu device connected"
        )
        self._assert_device_summary_contained()

    def test_split_console_device_summary_contains_oversized_port(self):
        payload = "{'port': 'COM" + "9" * 100 + "'} | 1.0"
        self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(
            self.app.device_status_var.get(), "Makcu device connected"
        )
        self._assert_device_summary_contained()

    def test_split_console_device_summary_logs_complete_diagnostic(self):
        payload = (
            "{'description': 'USB | Serial', 'port': 'COM6'} | "
            "firmware diagnostic 1.0"
        )
        with self.assertLogs(level="INFO") as captured:
            self.app.handle_service_event(ServiceEvent("connected", payload))
        self.assertEqual(self.app.device_status_var.get(), "Makcu on COM6")
        self.assertTrue(any(payload in line for line in captured.output))

    def test_split_console_motion_uses_exact_three_to_two_columns(self):
        self.assertEqual(int(self.app.motion_hero_card.grid_info()["column"]), 0)
        self.assertEqual(
            int(self.app.motion_summary_card.grid_info()["column"]), 1
        )
        self.assertEqual(
            tuple(
                int(self.app.motion_page.grid_columnconfigure(column)["weight"])
                for column in (0, 1)
            ),
            (3, 2),
        )
        for key in ("pulse_size_px", "pulse_rate_hz"):
            for suffix in ("scale", "entry"):
                with self.subTest(key=key, suffix=suffix):
                    self.assertTrue(
                        self._is_descendant(
                            getattr(self.app, f"{key}_{suffix}"),
                            self.app.motion_hero_card,
                        )
                    )
        self.assertTrue(
            self._is_descendant(
                self.app.motion_summary_label, self.app.motion_summary_card
            )
        )
        self.app.deiconify()
        self.app.select_page(1)
        self.app.update()
        self.assertAlmostEqual(
            self.app.motion_hero_card.winfo_width()
            / self.app.motion_summary_card.winfo_width(),
            1.5,
            delta=0.03,
        )

    def test_motion_page_uses_dashboard_header_and_live_readouts(self):
        self.assertEqual(self.app.motion_title_label.cget("text"), "MOTION")
        self.assertEqual(
            int(self.app.motion_title_label.master.grid_info()["row"]), 0
        )
        for card in (
            self.app.motion_hero_card,
            self.app.motion_summary_card,
        ):
            with self.subTest(card=str(card)):
                self.assertEqual(
                    card.cget("style"), "Liquid.SettingsCard.TFrame"
                )
                self.assertEqual(int(card.grid_info()["row"]), 1)
        for readout, variable in (
            (
                self.app.motion_size_readout,
                self.app.motion_snapshot_size_var,
            ),
            (
                self.app.motion_rate_readout,
                self.app.motion_snapshot_rate_var,
            ),
            (
                self.app.motion_ramp_readout,
                self.app.motion_snapshot_ramp_var,
            ),
        ):
            with self.subTest(readout=str(readout)):
                self.assertTrue(
                    self._is_descendant(readout, self.app.motion_summary_card)
                )
                self.assertEqual(
                    readout.cget("textvariable"), str(variable)
                )

    def test_motion_snapshot_text_is_not_clipped_at_fixed_window_size(self):
        self.app.deiconify()
        self.app.select_page(1)
        self.app.update()
        labels = {}
        for widget in descendant_widgets(self.app.motion_summary_frame):
            if isinstance(widget, ttk.Label):
                labels[str(widget.cget("text"))] = widget
        for text in ("PULSE SIZE", "PULSE RATE", "RAMP MODE"):
            with self.subTest(text=text):
                self.assertGreaterEqual(
                    labels[text].winfo_width(), labels[text].winfo_reqwidth()
                )
        self.assertLessEqual(
            int(self.app.motion_summary_label.cget("wraplength")),
            self.app.motion_summary_frame.winfo_width(),
        )
        self.assertLessEqual(
            self.app.motion_summary_frame.winfo_reqheight(),
            self.app.motion_summary_frame.winfo_height(),
        )
        snapshot_copy = next(
            widget
            for widget in self.app.motion_summary_card.winfo_children()
            if isinstance(widget, ttk.Label)
            and str(widget.cget("text")).startswith("The immutable profile")
        )
        self.assertLessEqual(
            int(snapshot_copy.cget("wraplength")),
            self.app.motion_summary_frame.winfo_width(),
        )

    def test_motion_page_has_snapshot_backed_live_summary(self):
        """Fails if Motion lacks a visible summary of the active snapshot."""
        summary_var = getattr(self.app, "motion_summary_var", None)
        self.assertIsInstance(summary_var, tk.StringVar)
        self.assertTrue(
            self._is_descendant(self.app.motion_summary_label, self.app.motion_page)
        )
        self.assertEqual(
            self.app.motion_summary_label.cget("textvariable"),
            str(summary_var),
        )
        self.assertEqual(
            summary_var.get(),
            "2 px paired pulse at 60 Hz | Smooth",
        )

    def test_motion_summary_refreshes_after_edit_and_preset(self):
        """Fails if the summary drifts from the immutable motion snapshot."""
        summary_var = getattr(self.app, "motion_summary_var", None)
        self.assertIsInstance(summary_var, tk.StringVar)
        self.app.pulse_size_px_var.set("4")
        self.app.update()
        self.assertEqual(self.app.get_motion_settings().pulse_size_px, 4.0)
        self.assertEqual(
            summary_var.get(),
            "4 px paired pulse at 60 Hz | Smooth",
        )

        self.app.preset_var.set("Strong")
        self.app.apply_preset()
        self.app.update()
        self.assertEqual(
            self.app.get_motion_settings(),
            MotionSettings(4.0, 100.0, "Instant"),
        )
        self.assertEqual(
            summary_var.get(),
            "4 px paired pulse at 100 Hz | Instant",
        )

    def test_motion_summary_describes_paired_pulse_snapshot(self):
        self.app._replace_motion_snapshot(
            MotionSettings(2.0, 30.0, "Smooth")
        )
        self.assertEqual(
            self.app.motion_summary_var.get(),
            "2 px paired pulse at 30 Hz | Smooth",
        )

    def test_pulse_edits_update_immutable_snapshot(self):
        self.app.pulse_size_px_var.set("4")
        self.app.pulse_rate_hz_var.set("45")
        self.app.ramp_mode_var.set("Instant")
        self.app._motion_changed("pulse_size_px")
        self.assertEqual(
            self.app.get_motion_settings(),
            MotionSettings(4.0, 45.0, "Instant"),
        )

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

    def test_split_console_keeps_actions_footer_runtime_and_stop_on_every_page(self):
        self.app.deiconify()
        self.app.update()
        for index in range(2):
            with self.subTest(index=index):
                self.app.select_page(index)
                self.app.update_idletasks()
                self.assertTrue(all(widget.winfo_ismapped() for widget in (
                    self.app.navigation_actions,
                    self.app.reconnect_button,
                    self.app.test_button,
                    self.app.theme_button,
                    self.app.footer_frame,
                    self.app.runtime_frame,
                    self.app.stop_button,
                )))

    def test_stop_is_visible_on_every_navigation_page(self):
        self.app.deiconify()
        for index in range(2):
            with self.subTest(index=index):
                self.app.select_page(index)
                self.app.update()
                self.assertEqual(self.app.stop_button.winfo_ismapped(), 1)

    def test_split_console_close_cancels_all_widget_callbacks_before_service_close(self):
        self.app.deiconify()
        self.app.update()
        self.app.nav.select(1)
        self.assertIsNotNone(self.app.nav._animation_after_id)
        sliders = [
            widget for widget in self._descendants(self.app)
            if isinstance(widget, LiquidSlider)
        ]
        self.assertTrue(sliders)
        for slider in sliders:
            slider._schedule_hide_bubble()
            self.assertIsNotNone(slider._bubble_after_id)
        self.app._show_action_tooltip(
            SimpleNamespace(widget=self.app.reconnect_button),
            self.app.reconnect_tooltip_text,
        )
        self.app._show_theme_tooltip()
        callback_states_at_service_close = []
        original_close = self.service.close

        def observe_service_close():
            callback_states_at_service_close.append((
                self.app._closing,
                self.app.nav._animation_after_id,
                tuple(slider._bubble_after_id for slider in sliders),
                self.app._action_tooltip,
                self.app._theme_tooltip,
            ))
            original_close()

        self.service.close = observe_service_close
        self.app.close_app()
        self.assertEqual(
            callback_states_at_service_close,
            [(True, None, tuple(None for _slider in sliders), None, None)],
        )
        self.assertIsNone(self.app.nav._animation_after_id)

    def test_invalid_motion_edit_does_not_change_page(self):
        self.app.select_page(1)
        self.app.pulse_size_px_var.set("not-a-number")
        self.app._motion_changed("pulse_size_px")
        self.assertEqual(self.app.nav.selected_index, 1)
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

    def test_split_console_page_and_theme_changes_preserve_state_and_geometry(self):
        self.app.pulse_size_px_var.set("4")
        self.app.pulse_rate_hz_var.set("45")
        self.app.ramp_mode_var.set("Instant")
        self.app.trigger_var.set("Mouse4")
        self.app.modifier_var.set("Right")
        self.app.preset_var.set("Custom")
        self.app.deiconify()
        self.app.update()
        expected_summary = self.app.motion_summary_var.get()
        expected_motion_values = {
            key: variable.get() for key, variable in self.app.motion_vars.items()
        }
        expected_control_values = (
            self.app.trigger_var.get(),
            self.app.modifier_var.get(),
            self.app.preset_var.get(),
            self.app.hotkey_name_var.get(),
        )
        expected_geometry = self.app.geometry()

        for index in (1, 0):
            with self.subTest(index=index):
                self.app.select_page(index)
                self.app.toggle_theme()
                self.app.update_idletasks()
                self.assertEqual(self.app.nav.selected_index, index)
                self.assertEqual(
                    self.app.page_host.grid_slaves(), [self.app.pages[index]]
                )
                self.assertEqual(self.app.motion_summary_var.get(), expected_summary)
                self.assertEqual(
                    {
                        key: variable.get()
                        for key, variable in self.app.motion_vars.items()
                    },
                    expected_motion_values,
                )
                self.assertEqual(
                    (
                        self.app.trigger_var.get(),
                        self.app.modifier_var.get(),
                        self.app.preset_var.get(),
                        self.app.hotkey_name_var.get(),
                    ),
                    expected_control_values,
                )
                self.assertEqual(self.app.geometry(), expected_geometry)

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

    def test_footer_and_runtime_are_owned_by_the_persistent_workspace(self):
        for widget in (self.app.footer_frame, self.app.runtime_frame,
                       self.app.enable_button, self.app.stop_button):
            with self.subTest(widget=str(widget)):
                self.assertTrue(
                    self._is_descendant(widget, self.app.console_workspace)
                )

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

    def test_liquid_styles_are_registered(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("Liquid.App.TFrame", "background"), "#F2F7FA")
        self.assertEqual(style.lookup("Liquid.Surface.TFrame", "background"),
                         "#E5F0F5")
        self.assertEqual(
            self.app.tk.splitlist(
                style.lookup("Liquid.Title.TLabel", "font")
            ),
            ("Consolas", "18", "bold"),
        )

    def test_application_named_fonts_use_consolas(self):
        """Fails if an unstyled Tk control falls back to another family."""
        for name in (
            "TkDefaultFont", "TkTextFont", "TkFixedFont",
            "TkMenuFont", "TkHeadingFont", "TkCaptionFont",
            "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont",
        ):
            self.assertEqual(
                self.app.tk.call("font", "actual", name, "-family"),
                "Consolas",
            )

    def test_combobox_popups_use_liquid_colors_in_both_themes(self):
        """Fails if classic Tk popup Listboxes ignore the active theme."""
        combos = (
            self.app.trigger_combo,
            self.app.modifier_combo,
            self.app.preset_combo,
            self.app.ramp_mode_combo,
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
                    self.assertEqual(
                        self.app.tk.call(listbox, "cget", "-relief"), "flat"
                    )
                    self.assertEqual(
                        int(self.app.tk.call(listbox, "cget", "-borderwidth")),
                        0,
                    )
                    self.assertEqual(
                        int(self.app.tk.call(
                            listbox, "cget", "-highlightthickness"
                        )),
                        1,
                    )
                    self.assertEqual(
                        self.app.tk.call(listbox, "cget", "-activestyle"),
                        "none",
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

    def test_text_contrast_across_themes_and_interaction_states(self):
        """Fails if status or STOP text becomes unreadable in any theme state."""
        style = ttk.Style(self.app)
        for theme in ("light", "dark"):
            cases = (
                (
                    "stop-normal",
                    style.lookup("Liquid.Danger.TButton", "foreground"),
                    style.lookup("Liquid.Danger.TButton", "background"),
                ),
                (
                    "stop-hover",
                    style.lookup(
                        "Liquid.Danger.TButton", "foreground", ("active",)
                    ),
                    style.lookup(
                        "Liquid.Danger.TButton", "background", ("active",)
                    ),
                ),
                (
                    "stop-pressed",
                    style.lookup(
                        "Liquid.Danger.TButton",
                        "foreground",
                        ("pressed", "active"),
                    ),
                    style.lookup(
                        "Liquid.Danger.TButton",
                        "background",
                        ("pressed", "active"),
                    ),
                ),
                *(
                    (
                        f"status-{state.lower()}",
                        style.lookup(
                            f"Liquid.Status{state}.TLabel", "foreground"
                        ),
                        style.lookup(
                            f"Liquid.Status{state}.TLabel", "background"
                        ),
                    )
                    for state in ("Disconnected", "Connecting", "Connected")
                ),
            )
            for state, foreground, background in cases:
                with self.subTest(theme=theme, state=state):
                    self.assertGreaterEqual(
                        contrast_ratio(foreground, background),
                        4.5,
                    )
            self.app.toggle_theme()

    def test_mini_icon_contrast_across_themes_and_interaction_states(self):
        """Fails if a mini-action symbol lacks non-text contrast."""
        for theme in ("light", "dark"):
            palette = self.app._icon_palette()
            cases = (
                ("normal", "icon", "surface"),
                ("hover", "icon", "surface_hover"),
                ("pressed", "icon", "surface_pressed"),
                ("disabled", "icon_disabled", "surface_disabled"),
            )
            for state, icon_role, surface_role in cases:
                with self.subTest(theme=theme, state=state):
                    self.assertGreaterEqual(
                        contrast_ratio(
                            palette[icon_role], palette[surface_role]
                        ),
                        3.0,
                    )
            self.app.toggle_theme()

    def test_liquid_buttons_use_high_contrast_palette(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "background"),
                         "#55DDF6")
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "foreground"),
                         "#07252C")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "background"),
                         "#B83246")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "foreground"),
                         "#FFFFFF")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "background"),
                         "#FFFFFF")
        self.assertEqual(self.app.enable_button.cget("style"),
                         "Liquid.Primary.TButton")
        self.assertEqual(self.app.stop_button.cget("style"),
                         "Liquid.Danger.TButton")

    def test_liquid_buttons_show_hover_and_press_states_with_rounded_borders(self):
        style = ttk.Style(self.app)
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "background",
                                      ("active",)), "#79E8FA")
        self.assertEqual(style.lookup("Liquid.Primary.TButton", "background",
                                      ("pressed", "active")), "#33BDD8")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "background",
                                      ("active",)), "#C74652")
        self.assertEqual(style.lookup("Liquid.Danger.TButton", "background",
                                      ("pressed", "active")), "#9F3140")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "background",
                                      ("active",)), "#D6F5FA")
        for name in ("Primary", "Secondary", "Danger"):
            widget_style = f"Liquid.{name}.TButton"
            self.assertEqual(str(style.lookup(widget_style, "borderwidth")), "0")
            self.assertEqual(style.lookup(widget_style, "relief", ("pressed",)), "flat")
            self.assertEqual(str(style.lookup(widget_style, "focusthickness")), "0")
            self.assertIn("Rounded", style.layout(widget_style)[0][0])

        self.app.toggle_theme()
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "background",
                                      ("disabled",)), "#34465C")
        self.assertEqual(style.lookup("Liquid.Secondary.TButton", "foreground",
                                      ("disabled",)), "#91A5B8")

    def test_form_controls_use_rounded_borders_without_square_frames(self):
        """Fails if a native square border reappears around a form control."""
        style = ttk.Style(self.app)
        for widget_style in (
            "Liquid.Entry.TEntry",
            "Liquid.Invalid.TEntry",
            "Liquid.Modern.TCombobox",
        ):
            self.assertEqual(str(style.lookup(widget_style, "borderwidth")), "0")
            self.assertIn("Rounded", style.layout(widget_style)[0][0])
        for widget_style in (
            "Liquid.Card.TLabelframe",
            "Liquid.Vertical.TScrollbar",
        ):
            self.assertEqual(str(style.lookup(widget_style, "borderwidth")), "0")

    def test_required_actions_are_present(self):
        texts = widget_texts(self.app)
        for expected in ("Enable Jitter", "STOP"):
            self.assertIn(expected, texts)
        self.assertEqual(self.app.reconnect_button.icon, "↻")
        self.assertEqual(self.app.test_button.icon, "▶")
    def test_page_selection_does_not_change_outer_geometry(self):
        self.app.update_idletasks()
        before = self.app.geometry().split("+")[0]
        self.app.select_page(1)
        self.app.update_idletasks()
        after = self.app.geometry().split("+")[0]
        self.assertEqual(after, before)


class JitterRuntimeTests(JitterLayoutTests):
    def make_connected_ai_app(self):
        self.app.close_app()
        app = self.make_app(config=AppConfig(mode="ai_aim"))
        self.service.connected = True
        return app

    def prepare_retiring_test_source(self, mode, cancellation):
        self.app.close_app()
        app = self.make_app(config=AppConfig(mode=mode))
        self.service.connected = True
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        if mode == "ai_aim":
            app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        source = self.service.active_motion_generation
        self.assertIs(type(source), int)

        if cancellation == "release":
            app.handle_service_event(ServiceEvent("button", ("Left", False)))
        elif cancellation == "rebind":
            app.trigger_var.set("Mouse4")
            app.on_bindings_changed()
        elif cancellation == "stop":
            app.emergency_stop("Stopped by user")
        else:
            self.fail(f"unknown cancellation: {cancellation}")

        self.assertTrue(self.service.motion_active)
        self.assertEqual(app._retiring_motion_generation, source)
        return app, source

    def motion_calls_for_mode(self, mode):
        return (
            self.service.ai_motion_calls
            if mode == "ai_aim" else self.service.motion_calls
        )

    def ready_pending_ai_test(self, app, mode):
        if mode == "ai_aim" and not app._ai_ready:
            app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

    def assert_test_started_from_fresh_source(
        self,
        app,
        mode,
        retiring_source,
        calls_before_test,
    ):
        calls = self.motion_calls_for_mode(mode)
        fresh_source = self.service.active_motion_generation
        self.assertIs(type(fresh_source), int)
        self.assertNotEqual(fresh_source, retiring_source)
        self.assertEqual(len(calls), calls_before_test + 1)
        self.assertEqual(calls[-1][-1], 3.0)
        self.assertEqual(
            app._motion_mode,
            "test_ai" if mode == "ai_aim" else "test",
        )
        self.assertEqual(app._expected_motion_generation, fresh_source)
        self.assertEqual(app.runtime_state_var.get(), "TESTING")
        self.assertFalse(app.test_button._enabled)
        return fresh_source

    def test_release_rebind_and_stop_defer_both_tests_for_stopped_or_error(self):
        for mode in ("jitter", "ai_aim"):
            for cancellation in ("release", "rebind", "stop"):
                for terminal_kind in ("motion_stopped", "motion_error"):
                    with self.subTest(
                        mode=mode,
                        cancellation=cancellation,
                        terminal=terminal_kind,
                    ):
                        app, retiring_source = self.prepare_retiring_test_source(
                            mode,
                            cancellation,
                        )
                        calls = self.motion_calls_for_mode(mode)
                        calls_before_test = len(calls)

                        app.start_test_run()
                        self.ready_pending_ai_test(app, mode)

                        self.assertEqual(len(calls), calls_before_test)
                        self.assertEqual(app.runtime_state_var.get(), "TESTING")
                        self.assertFalse(app.test_button._enabled)
                        self.service.emit(ServiceEvent(
                            terminal_kind,
                            (
                                "RuntimeError: retiring worker failed"
                                if terminal_kind == "motion_error"
                                else f"{cancellation}_complete"
                            ),
                            retiring_source,
                        ))
                        app._drain_ui_queue()

                        fresh_source = self.assert_test_started_from_fresh_source(
                            app,
                            mode,
                            retiring_source,
                            calls_before_test,
                        )
                        app._cancel_after("_ui_pump_after_id")
                        self.service.emit(ServiceEvent(
                            "motion_stopped",
                            "duration_complete",
                            fresh_source,
                        ))
                        app._drain_ui_queue()

                        restored = cancellation != "stop"
                        self.assertEqual(app.enabled, restored)
                        self.assertIsNone(app._motion_mode)
                        self.assertTrue(app.test_button._enabled)
                        self.assertEqual(
                            app.runtime_state_var.get(),
                            "ARMED" if restored else "DISABLED",
                        )

    def test_retiring_terminal_queued_during_test_request_starts_once(self):
        for mode in ("jitter", "ai_aim"):
            for terminal_kind in ("motion_stopped", "motion_error"):
                with self.subTest(mode=mode, terminal=terminal_kind):
                    app, retiring_source = self.prepare_retiring_test_source(
                        mode,
                        "release",
                    )
                    calls = self.motion_calls_for_mode(mode)
                    calls_before_test = len(calls)
                    callback_count = 0

                    def emit_during_ownership_read():
                        nonlocal callback_count
                        callback_count += 1
                        self.service.motion_active_hook = None
                        self.service.emit(ServiceEvent(
                            terminal_kind,
                            (
                                "RuntimeError: retiring worker failed"
                                if terminal_kind == "motion_error"
                                else "trigger_released"
                            ),
                            retiring_source,
                        ))

                    self.service.motion_active_hook = emit_during_ownership_read
                    app.start_test_run()
                    self.ready_pending_ai_test(app, mode)
                    app._drain_ui_queue()

                    self.assertEqual(callback_count, 1)
                    self.assert_test_started_from_fresh_source(
                        app,
                        mode,
                        retiring_source,
                        calls_before_test,
                    )

    def test_wrong_and_duplicate_retiring_sources_cannot_launch_twice(self):
        for mode in ("jitter", "ai_aim"):
            with self.subTest(mode=mode):
                app, retiring_source = self.prepare_retiring_test_source(
                    mode,
                    "release",
                )
                calls = self.motion_calls_for_mode(mode)
                calls_before_test = len(calls)
                app.start_test_run()
                self.ready_pending_ai_test(app, mode)

                self.service.emit(ServiceEvent(
                    "motion_error",
                    "RuntimeError: wrong source",
                    retiring_source + 1000,
                ))
                app._drain_ui_queue()
                self.assertEqual(len(calls), calls_before_test)
                self.assertEqual(
                    app._motion_mode,
                    "test_ai_loading" if mode == "ai_aim" else "test_pending",
                )

                app._cancel_after("_ui_pump_after_id")
                self.service.emit(ServiceEvent(
                    "motion_stopped",
                    "trigger_released",
                    retiring_source,
                ))
                app._drain_ui_queue()
                fresh_source = self.assert_test_started_from_fresh_source(
                    app,
                    mode,
                    retiring_source,
                    calls_before_test,
                )

                app._cancel_after("_ui_pump_after_id")
                self.service.emit(ServiceEvent(
                    "motion_error",
                    "RuntimeError: duplicate retiring source",
                    retiring_source,
                ))
                app._drain_ui_queue()
                self.assertEqual(len(calls), calls_before_test + 1)
                self.assertEqual(
                    self.service.active_motion_generation,
                    fresh_source,
                )
                self.assertFalse(app.test_button._enabled)

    def test_pending_test_handoff_is_cleared_by_every_cancellation_path(self):
        for mode in ("jitter", "ai_aim"):
            for cancellation in (
                "stop",
                "disable",
                "mode_switch",
                "disconnect",
                "new_cancellation",
                "shutdown",
            ):
                with self.subTest(mode=mode, cancellation=cancellation):
                    app, retiring_source = self.prepare_retiring_test_source(
                        mode,
                        "release",
                    )
                    calls = self.motion_calls_for_mode(mode)
                    calls_before_test = len(calls)
                    app.start_test_run()
                    self.ready_pending_ai_test(app, mode)
                    self.assertEqual(len(calls), calls_before_test)

                    if cancellation == "stop":
                        app.emergency_stop("Stopped by user")
                    elif cancellation == "disable":
                        app.set_enabled(False)
                    elif cancellation == "mode_switch":
                        app.mode_var.set(
                            "jitter" if mode == "ai_aim" else "ai_aim"
                        )
                        app.on_mode_changed()
                    elif cancellation == "disconnect":
                        self.service.connected = False
                        app.handle_service_event(ServiceEvent(
                            "disconnected",
                            "Device lost",
                        ))
                    elif cancellation == "new_cancellation":
                        app.trigger_var.set("Mouse5")
                        app.on_bindings_changed()
                    else:
                        app.close_app()

                    self.assertIsNone(
                        getattr(app, "_deferred_motion_action", None)
                    )
                    if cancellation != "shutdown":
                        self.service.emit(ServiceEvent(
                            "motion_stopped",
                            "trigger_released",
                            retiring_source,
                        ))
                        app._drain_ui_queue()
                    self.assertEqual(len(calls), calls_before_test)
                    self.assertIsNone(app._motion_mode)
                    self.assertTrue(app.test_button._enabled)

    def test_ai_enable_arms_detection_but_not_movement(self):
        app = self.make_connected_ai_app()

        app.set_enabled(True)

        self.assertEqual(self.ai.start_calls, [app.get_ai_settings])
        self.assertEqual(self.service.motion_calls, [])
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertEqual(app.runtime_state_var.get(), "ARMED")

    def test_ai_ready_and_active_gate_start_ai_motion(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.assertEqual(len(self.service.ai_motion_calls), 1)
        self.assertEqual(self.service.motion_calls, [])
        snapshot_provider, settings_provider, duration_s = (
            self.service.ai_motion_calls[0]
        )
        self.assertIs(snapshot_provider.__self__, self.ai)
        self.assertEqual(snapshot_provider(), self.ai.snapshot)
        self.assertEqual(settings_provider(), app.get_ai_settings())
        self.assertIsNone(duration_s)
        self.assertEqual(app.runtime_state_var.get(), "MOVING")

    def test_ai_gate_held_before_ready_starts_motion_on_ready(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertEqual(self.service.ai_motion_calls, [])

        app.handle_ai_event(AiEvent("ready", "CPUExecutionProvider"))

        self.assertEqual(len(self.service.ai_motion_calls), 1)

    def test_ai_gate_release_stops_only_motion_and_leaves_capture_armed(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        ai_stop_count = len(self.ai.stop_calls)

        app.handle_service_event(ServiceEvent("button", ("Left", False)))

        self.assertIn("trigger_released", self.service.stop_reasons)
        self.assertEqual(len(self.ai.stop_calls), ai_stop_count)
        self.assertTrue(app.enabled)
        self.assertEqual(app.runtime_state_var.get(), "ARMED")

    def test_ai_modifier_gate_requires_both_buttons(self):
        app = self.make_connected_ai_app()
        app.modifier_var.set("Right")
        app.on_bindings_changed()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertEqual(self.service.ai_motion_calls, [])
        app.handle_service_event(ServiceEvent("button", ("Right", True)))

        self.assertEqual(len(self.service.ai_motion_calls), 1)

    def test_trigger_change_stops_active_motion_in_both_modes(self):
        for mode in ("jitter", "ai_aim"):
            with self.subTest(mode=mode):
                self.app.close_app()
                app = self.make_app(config=AppConfig(mode=mode))
                self.service.connected = True
                app.set_enabled(True)
                if mode == "ai_aim":
                    app.handle_ai_event(
                        AiEvent("ready", "DmlExecutionProvider")
                    )
                app.handle_service_event(
                    ServiceEvent("button", ("Left", True))
                )
                self.assertEqual(app.runtime_state_var.get(), "MOVING")
                ai_stop_count = len(self.ai.stop_calls)

                app.trigger_var.set("Mouse4")
                app.on_bindings_changed()

                self.assertIn("bindings_changed", self.service.stop_reasons)
                self.assertFalse(app._normal_motion_started)
                self.assertEqual(app.runtime_state_var.get(), "ARMED")
                self.assertTrue(app.enabled)
                self.assertEqual(len(self.ai.stop_calls), ai_stop_count)

    def test_modifier_change_stops_active_motion_in_both_modes(self):
        for mode in ("jitter", "ai_aim"):
            with self.subTest(mode=mode):
                self.app.close_app()
                app = self.make_app(config=AppConfig(
                    mode=mode,
                    modifier="Right",
                ))
                self.service.connected = True
                app.set_enabled(True)
                if mode == "ai_aim":
                    app.handle_ai_event(
                        AiEvent("ready", "CPUExecutionProvider")
                    )
                app.handle_service_event(
                    ServiceEvent("button", ("Left", True))
                )
                app.handle_service_event(
                    ServiceEvent("button", ("Right", True))
                )
                self.assertEqual(app.runtime_state_var.get(), "MOVING")
                ai_stop_count = len(self.ai.stop_calls)

                app.modifier_var.set("Mouse4")
                app.on_bindings_changed()

                self.assertIn("bindings_changed", self.service.stop_reasons)
                self.assertFalse(app._normal_motion_started)
                self.assertEqual(app.runtime_state_var.get(), "ARMED")
                self.assertTrue(app.enabled)
                self.assertEqual(len(self.ai.stop_calls), ai_stop_count)

    def test_disconnected_ai_enable_keeps_ai_mode_label_and_stays_disabled(self):
        app = self.make_connected_ai_app()
        self.service.connected = False

        app.set_enabled(True)

        self.assertFalse(app.enabled)
        self.assertEqual(app.enable_button.cget("text"), "Enable AI Aim")
        self.assertEqual(self.ai.start_calls, [])

    def test_jitter_mode_never_starts_ai_runtime_or_ai_motion(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.assertEqual(len(self.service.motion_calls), 1)
        self.assertEqual(self.ai.start_calls, [])
        self.assertEqual(self.service.ai_motion_calls, [])

    def test_mode_change_immediately_cancels_ai_and_makcu(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))

        app.mode_var.set("jitter")
        app.on_mode_changed()

        self.assertFalse(app.enabled)
        self.assertEqual(app.mode_var.get(), "jitter")
        self.assertEqual(self.ai.stop_calls[-1], "Mode changed")
        self.assertEqual(self.service.stop_reasons[-1], "Mode changed")
        self.assertEqual(app.ai_settings_card.winfo_manager(), "")

    def test_ai_events_are_marshaled_to_the_tk_thread(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        self.assertTrue(
            self.ai.emit(AiEvent("ready", "DmlExecutionProvider"))
        )
        self.assertEqual(app.ai_status_var.get(), "Stopped")

        app.update()

        self.assertEqual(app.ai_status_var.get(), "Ready (DirectML)")
        self.assertEqual(app.ai_provider_var.get(), "DirectML")

    def test_queued_ready_and_press_from_old_arm_do_not_cross_stop_and_rearm(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        old_generation = self.ai.active_generation
        self.assertTrue(self.ai.emit(
            AiEvent("ready", "DmlExecutionProvider"),
            generation=old_generation,
        ))
        self.service.emit(ServiceEvent("button", ("Left", True)))

        app.emergency_stop("Stopped by user")
        app.set_enabled(True)
        app._drain_ui_queue()

        self.assertNotEqual(self.ai.active_generation, old_generation)
        self.assertFalse(app._ai_ready)
        self.assertFalse(app.trigger_gate.active)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertEqual(app.runtime_state_var.get(), "ARMED")

    def test_inflight_ready_during_stop_cannot_activate_new_ai_test(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        old_generation = self.ai.active_generation

        def emit_ready_during_stop(generation):
            self.assertEqual(generation, old_generation)
            self.assertTrue(self.ai.emit(
                AiEvent("ready", "DmlExecutionProvider"),
                generation=generation,
            ))

        self.ai.stop_hook = emit_ready_during_stop
        app.emergency_stop("Stopped by user")
        self.ai.stop_hook = None

        app.start_test_run()
        app._drain_ui_queue()

        self.assertFalse(app._ai_ready)
        self.assertEqual(app._motion_mode, "test_ai_loading")
        self.assertIsNotNone(app._ai_test_pending_generation)
        self.assertEqual(self.service.ai_motion_calls, [])

    def test_queued_error_from_old_ai_arm_is_discarded_after_mode_switch(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        self.assertTrue(self.ai.emit(
            AiEvent("error", "OldGenerationError: stale")
        ))

        app.mode_var.set("jitter")
        app.on_mode_changed()
        app._drain_ui_queue()

        self.assertEqual(app.mode_var.get(), "jitter")
        self.assertEqual(app.ai_status_var.get(), "Stopped")
        self.assertNotIn("ai_error", self.service.stop_reasons)
        self.assertEqual(app.footer_var.get(), "Jitter mode selected")

    def test_queued_motion_terminal_from_old_arm_cannot_end_new_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.service.emit(ServiceEvent("motion_stopped", "old_motion"))

        self.app.emergency_stop("Stopped by user")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.app._drain_ui_queue()

        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_rebind_terminal_cannot_clear_motion_started_by_fresh_binding_edge(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.service.stop_hook = lambda _reason: self.service.emit(
            ServiceEvent("motion_stopped", "bindings_changed")
        )
        self.app.trigger_var.set("Mouse4")
        self.app.on_bindings_changed()
        self.service.stop_hook = None
        self.app.handle_service_event(
            ServiceEvent("button", ("Mouse4", True))
        )

        self.app._drain_ui_queue()

        self.assertEqual(len(self.service.motion_calls), 2)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_terminal_emitted_after_rebind_stop_cannot_clear_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.app.trigger_var.set("Mouse4")
        self.app.on_bindings_changed()
        self.service.emit(
            ServiceEvent("motion_stopped", "bindings_changed")
        )
        self.app.handle_service_event(
            ServiceEvent("button", ("Mouse4", True))
        )
        self.app._drain_ui_queue()

        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_terminal_emitted_after_release_cannot_clear_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.service.emit(
            ServiceEvent("motion_stopped", "trigger_released")
        )
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.app._drain_ui_queue()

        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_jitter_release_then_press_waits_for_old_stopped_before_fresh_start(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.assertEqual(len(self.service.motion_calls), 1)
        self.assertEqual(self.service.active_motion_generation, old_source)
        self.assertFalse(self.app._normal_motion_started)

        self.service.emit(ServiceEvent(
            "motion_stopped",
            "trigger_released",
            old_source,
        ))
        self.app._drain_ui_queue()

        fresh_source = self.service.active_motion_generation
        self.assertIs(type(fresh_source), int)
        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(len(self.service.motion_calls), 2)
        self.assertEqual(self.app._expected_motion_generation, fresh_source)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_ai_rebind_then_press_waits_for_old_stopped_before_fresh_start(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        app.trigger_var.set("Mouse4")
        app.on_bindings_changed()
        app.handle_service_event(ServiceEvent("button", ("Mouse4", True)))

        self.assertEqual(len(self.service.ai_motion_calls), 1)
        self.assertEqual(self.service.active_motion_generation, old_source)
        self.assertFalse(app._normal_motion_started)

        self.service.emit(ServiceEvent(
            "motion_stopped",
            "bindings_changed",
            old_source,
        ))
        app._drain_ui_queue()

        fresh_source = self.service.active_motion_generation
        self.assertIs(type(fresh_source), int)
        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(len(self.service.ai_motion_calls), 2)
        self.assertEqual(app._expected_motion_generation, fresh_source)
        self.assertTrue(app._normal_motion_started)
        self.assertEqual(app.runtime_state_var.get(), "MOVING")

    def test_jitter_stop_rearm_then_press_waits_for_old_error_before_fresh_start(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        self.app.emergency_stop("Stopped by user")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.assertEqual(len(self.service.motion_calls), 1)
        self.assertEqual(self.service.active_motion_generation, old_source)
        self.assertFalse(self.app._normal_motion_started)

        self.service.emit(ServiceEvent(
            "motion_error",
            "RuntimeError: canceled Jitter worker",
            old_source,
        ))
        self.app._drain_ui_queue()

        fresh_source = self.service.active_motion_generation
        self.assertIs(type(fresh_source), int)
        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(len(self.service.motion_calls), 2)
        self.assertEqual(self.app._expected_motion_generation, fresh_source)
        self.assertTrue(self.app.enabled)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_ai_stop_rearm_then_press_waits_for_old_error_before_fresh_start(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "CPUExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        app.emergency_stop("Stopped by user")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "CPUExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.assertEqual(len(self.service.ai_motion_calls), 1)
        self.assertEqual(self.service.active_motion_generation, old_source)
        self.assertFalse(app._normal_motion_started)

        self.service.emit(ServiceEvent(
            "motion_error",
            "RuntimeError: canceled AI worker",
            old_source,
        ))
        app._drain_ui_queue()

        fresh_source = self.service.active_motion_generation
        self.assertIs(type(fresh_source), int)
        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(len(self.service.ai_motion_calls), 2)
        self.assertEqual(app._expected_motion_generation, fresh_source)
        self.assertEqual(app.ai_status_var.get(), "Ready (CPU)")
        self.assertTrue(app.enabled)
        self.assertTrue(app._normal_motion_started)
        self.assertEqual(app.runtime_state_var.get(), "MOVING")

    def test_pending_restart_is_canceled_when_gate_releases_before_terminal(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "trigger_released",
            old_source,
        ))
        self.app._drain_ui_queue()

        self.assertEqual(len(self.service.motion_calls), 1)
        self.assertIsNone(self.service.active_motion_generation)
        self.assertTrue(self.app.enabled)
        self.assertFalse(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_pending_restart_is_canceled_by_stop_before_terminal(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.app.emergency_stop("Stopped by user")
        self.service.emit(ServiceEvent(
            "motion_error",
            "RuntimeError: canceled before restart",
            old_source,
        ))
        self.app._drain_ui_queue()

        self.assertEqual(len(self.service.motion_calls), 1)
        self.assertIsNone(self.service.active_motion_generation)
        self.assertFalse(self.app.enabled)
        self.assertFalse(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_old_terminal_after_stop_and_rearm_cannot_clear_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.app.emergency_stop("Stopped by user")
        self.app.set_enabled(True)
        self.service.emit(
            ServiceEvent("motion_stopped", "Stopped by user")
        )
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.app._drain_ui_queue()

        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_old_terminal_cannot_complete_a_directly_started_test(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.service.emit(ServiceEvent("motion_stopped", "old_motion"))

        self.app.start_test_run()
        self.app._drain_ui_queue()

        self.assertEqual(self.app._motion_mode, "test")
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        self.assertFalse(self.app.test_button._enabled)

    def test_old_terminal_during_jitter_test_start_is_ignored(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.service.start_motion_hook = lambda: self.service.emit(
            ServiceEvent("motion_stopped", "trigger_released")
        )

        self.app.start_test_run()
        self.service.start_motion_hook = None
        self.app._drain_ui_queue()

        self.assertEqual(self.app._motion_mode, "test")
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        self.assertFalse(self.app.test_button._enabled)

    def test_old_terminal_during_ai_test_start_is_ignored(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.start_test_run()
        self.service.start_ai_motion_hook = lambda: self.service.emit(
            ServiceEvent("motion_stopped", "bindings_changed")
        )

        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.service.start_ai_motion_hook = None
        app._drain_ui_queue()

        self.assertEqual(app._motion_mode, "test_ai")
        self.assertEqual(app.runtime_state_var.get(), "TESTING")
        self.assertFalse(app.test_button._enabled)

    def test_reentrant_terminal_during_stop_handoff_starts_test_once(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertTrue(self.service.motion_active)
        self.service.stop_hook = lambda _reason: self.service.emit(
            ServiceEvent("motion_stopped", "test_run")
        )

        self.app.start_test_run()
        self.service.stop_hook = None
        self.app._drain_ui_queue()

        self.assertEqual(self.app._motion_mode, "test")
        self.assertEqual(len(self.service.motion_calls), 2)
        self.assertEqual(self.service.motion_calls[-1][1], 3.0)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")

    def test_late_jitter_test_completion_after_restoration_cannot_clear_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.start_test_run()
        test_source = self.service.active_motion_generation

        self.service.emit(
            ServiceEvent("motion_stopped", "duration_complete", test_source)
        )
        self.app._drain_ui_queue()
        self.app._cancel_after("_ui_pump_after_id")
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.service.emit(
            ServiceEvent("motion_stopped", "duration_complete", test_source)
        )
        self.app._drain_ui_queue()

        self.assertTrue(self.app.enabled)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_late_ai_test_completion_after_restoration_cannot_clear_fresh_motion(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.start_test_run()
        test_source = self.service.active_motion_generation
        self.service.emit(
            ServiceEvent("motion_stopped", "duration_complete", test_source)
        )
        app._drain_ui_queue()
        app._cancel_after("_ui_pump_after_id")
        app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.service.emit(
            ServiceEvent("motion_stopped", "duration_complete", test_source)
        )
        app._drain_ui_queue()

        self.assertTrue(app._normal_motion_started)
        self.assertEqual(app.runtime_state_var.get(), "MOVING")

    def test_old_jitter_duration_after_new_test_start_cannot_complete_it(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.start_test_run()
        old_source = self.service.active_motion_generation
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            old_source,
        ))
        self.app._drain_ui_queue()
        self.app._cancel_after("_ui_pump_after_id")

        self.app.start_test_run()
        fresh_source = self.service.active_motion_generation
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            old_source,
        ))
        self.app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(self.app._motion_mode, "test")
        self.assertFalse(self.app.test_button._enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")

    def test_old_jitter_duration_during_new_test_start_cannot_complete_it(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.start_test_run()
        old_source = self.service.active_motion_generation
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            old_source,
        ))
        self.app._drain_ui_queue()
        self.app._cancel_after("_ui_pump_after_id")
        self.service.start_motion_hook = lambda: self.service.emit(
            ServiceEvent(
                "motion_stopped",
                "duration_complete",
                old_source,
            )
        )

        self.app.start_test_run()
        self.service.start_motion_hook = None
        fresh_source = self.service.active_motion_generation
        self.app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(self.app._motion_mode, "test")
        self.assertFalse(self.app.test_button._enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")

    def test_old_ai_duration_after_new_test_start_cannot_complete_it(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.start_test_run()
        old_source = self.service.active_motion_generation
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            old_source,
        ))
        app._drain_ui_queue()
        app._cancel_after("_ui_pump_after_id")

        app.start_test_run()
        fresh_source = self.service.active_motion_generation
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            old_source,
        ))
        app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(app._motion_mode, "test_ai")
        self.assertFalse(app.test_button._enabled)
        self.assertEqual(app.runtime_state_var.get(), "TESTING")

    def test_old_ai_duration_during_new_test_start_cannot_complete_it(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.start_test_run()
        old_source = self.service.active_motion_generation
        self.service.emit(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            old_source,
        ))
        app._drain_ui_queue()
        app._cancel_after("_ui_pump_after_id")
        self.service.start_ai_motion_hook = lambda: self.service.emit(
            ServiceEvent(
                "motion_stopped",
                "duration_complete",
                old_source,
            )
        )

        app.start_test_run()
        self.service.start_ai_motion_hook = None
        fresh_source = self.service.active_motion_generation
        app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertEqual(app._motion_mode, "test_ai")
        self.assertFalse(app.test_button._enabled)
        self.assertEqual(app.runtime_state_var.get(), "TESTING")

    def test_stale_motion_error_after_rebind_cannot_disable_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        self.app.trigger_var.set("Mouse4")
        self.app.on_bindings_changed()
        self.service.emit(ServiceEvent(
            "motion_error",
            "RuntimeError: stale rebind failure",
            old_source,
        ))
        self.app.handle_service_event(ServiceEvent("button", ("Mouse4", True)))
        fresh_source = self.service.active_motion_generation
        self.app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertTrue(self.app.enabled)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_stale_motion_error_after_stop_cannot_disable_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation

        self.app.emergency_stop("Stopped by user")
        self.app.set_enabled(True)
        self.service.emit(ServiceEvent(
            "motion_error",
            "RuntimeError: stale stopped failure",
            old_source,
        ))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        fresh_source = self.service.active_motion_generation
        self.app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertTrue(self.app.enabled)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_stale_motion_error_during_next_start_cannot_disable_fresh_motion(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation
        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        # The old worker has released its motion slot, but its sink callback is
        # still delayed until the next start is in progress.
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.service.start_motion_hook = lambda: self.service.emit(
            ServiceEvent(
                "motion_error",
                "RuntimeError: stale during-start failure",
                old_source,
            )
        )

        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.service.start_motion_hook = None
        fresh_source = self.service.active_motion_generation
        self.app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertTrue(self.app.enabled)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_stale_ai_motion_error_during_next_start_cannot_disable_fresh_motion(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        old_source = self.service.active_motion_generation
        app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.service.start_ai_motion_hook = lambda: self.service.emit(
            ServiceEvent(
                "motion_error",
                "RuntimeError: stale AI during-start failure",
                old_source,
            )
        )

        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.service.start_ai_motion_hook = None
        fresh_source = self.service.active_motion_generation
        app._drain_ui_queue()

        self.assertNotEqual(fresh_source, old_source)
        self.assertTrue(app.enabled)
        self.assertTrue(app._normal_motion_started)
        self.assertEqual(app.runtime_state_var.get(), "MOVING")

    def test_current_normal_motion_terminal_returns_moving_to_armed(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        source = self.service.active_motion_generation

        self.app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "worker_complete",
            source,
        ))

        self.assertTrue(self.app.enabled)
        self.assertFalse(self.app._normal_motion_started)
        self.assertIsNone(self.app._expected_motion_generation)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_ai_status_fps_provider_and_stopped_events_are_concise(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("loading"))
        self.assertEqual(app.ai_status_var.get(), "Loading")
        self.assertEqual(app.ai_fps_var.get(), "0 FPS")

        app.handle_ai_event(AiEvent("ready", "CPUExecutionProvider"))
        self.assertEqual(app.ai_status_var.get(), "Ready (CPU)")
        self.assertEqual(app.ai_provider_var.get(), "CPU")
        app.handle_ai_event(AiEvent("fps", 59.75))
        self.assertEqual(app.ai_fps_var.get(), "59.8 FPS")

        app.handle_ai_event(AiEvent("stopped", "manual"))
        self.assertEqual(app.ai_status_var.get(), "Stopped")
        self.assertEqual(app.ai_fps_var.get(), "0 FPS")
        self.assertEqual(app.ai_provider_var.get(), "No provider")

    def test_unsolicited_ready_is_ignored_while_ai_runtime_is_inactive(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")

        app.queue_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app._drain_ui_queue()

        self.assertFalse(app._ai_ready)
        self.assertEqual(app.ai_status_var.get(), "Stopped")
        self.assertEqual(app.ai_provider_var.get(), "No provider")

    def test_ai_error_logs_detail_stops_motion_and_disables_without_hiding_mode(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))

        with self.assertLogs(level="ERROR") as captured:
            app.handle_ai_event(AiEvent("error", "SecretDetectorError: details"))

        self.assertFalse(app.enabled)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.ai_status_var.get(), "Error")
        self.assertEqual(app.mode_var.get(), "ai_aim")
        self.assertIn("ai_error", self.service.stop_reasons)
        self.assertIn("ai_error", self.ai.stop_calls)
        self.assertNotIn("SecretDetectorError", app.footer_var.get())
        self.assertIn("SecretDetectorError", captured.output[0])

    def test_ai_enable_start_none_logs_and_returns_to_disabled(self):
        app = self.make_connected_ai_app()
        self.ai.start_result = None

        with self.assertLogs(level="ERROR") as captured:
            app.set_enabled(True)

        self.assertFalse(app.enabled)
        self.assertFalse(app._ai_runtime_active)
        self.assertFalse(app._ai_ready)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.ai_status_var.get(), "Error")
        self.assertTrue(app.test_button._enabled)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertIn("AI runtime did not start", captured.output[0])
        self.assertNotIn("None", app.footer_var.get())

    def test_ai_enable_start_exception_logs_and_returns_to_disabled(self):
        app = self.make_connected_ai_app()
        self.ai.start_exception = RuntimeError("private enable failure")

        with self.assertLogs(level="ERROR") as captured:
            app.set_enabled(True)

        self.assertFalse(app.enabled)
        self.assertFalse(app._ai_runtime_active)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.ai_status_var.get(), "Error")
        self.assertIn("private enable failure", "\n".join(captured.output))
        self.assertNotIn("private enable failure", app.footer_var.get())

    def test_ai_test_start_false_unwinds_loading_state(self):
        app = self.make_connected_ai_app()
        self.ai.start_result = False

        with self.assertLogs(level="ERROR") as captured:
            app.start_test_run()

        self.assertFalse(app.enabled)
        self.assertFalse(app._ai_runtime_active)
        self.assertFalse(app._ai_ready)
        self.assertIsNone(app._motion_mode)
        self.assertIsNone(app._ai_test_pending_generation)
        self.assertFalse(app._test_start_pending)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.ai_status_var.get(), "Error")
        self.assertTrue(app.test_button._enabled)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertIn("AI runtime did not start", captured.output[0])

    def test_ai_test_start_exception_unwinds_loading_state(self):
        app = self.make_connected_ai_app()
        self.ai.start_exception = RuntimeError("private test failure")

        with self.assertLogs(level="ERROR") as captured:
            app.start_test_run()

        self.assertFalse(app.enabled)
        self.assertFalse(app._ai_runtime_active)
        self.assertIsNone(app._motion_mode)
        self.assertIsNone(app._ai_test_pending_generation)
        self.assertFalse(app._test_start_pending)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.ai_status_var.get(), "Error")
        self.assertTrue(app.test_button._enabled)
        self.assertIn("private test failure", "\n".join(captured.output))
        self.assertNotIn("private test failure", app.footer_var.get())

    def test_ai_test_ready_before_start_begins_timed_motion_immediately(self):
        app = self.make_connected_ai_app()
        app._cancel_after("_ui_pump_after_id")
        app.set_enabled(True)
        self.assertTrue(
            self.ai.emit(AiEvent("ready", "DmlExecutionProvider"))
        )
        app._drain_ui_queue()

        app.start_test_run()

        self.assertEqual(app._motion_mode, "test_ai")
        self.assertEqual(app.runtime_state_var.get(), "TESTING")
        self.assertEqual(len(self.service.ai_motion_calls), 1)
        self.assertEqual(self.service.ai_motion_calls[0][2], 3.0)
        self.assertEqual(self.ai.start_calls, [app.get_ai_settings])

    def test_ai_test_waits_for_asynchronous_ready_without_blocking_tk(self):
        app = self.make_connected_ai_app()

        app.start_test_run()

        self.assertEqual(app._motion_mode, "test_ai_loading")
        self.assertIsNotNone(app._ai_test_pending_generation)
        self.assertEqual(self.ai.start_calls, [app.get_ai_settings])
        self.assertEqual(self.service.ai_motion_calls, [])
        app.handle_ai_event(AiEvent("loading"))
        self.assertEqual(self.service.ai_motion_calls, [])

        app.handle_ai_event(AiEvent("ready", "CPUExecutionProvider"))

        self.assertEqual(app._motion_mode, "test_ai")
        self.assertEqual(len(self.service.ai_motion_calls), 1)
        self.assertEqual(self.service.ai_motion_calls[0][2], 3.0)

    def test_ai_test_waits_for_active_motion_stop_before_timed_start(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertEqual(len(self.service.ai_motion_calls), 1)
        normal_source = self.service.active_motion_generation

        app.start_test_run()

        self.assertEqual(app._motion_mode, "test_ai_loading")
        self.assertEqual(len(self.service.ai_motion_calls), 1)
        self.assertEqual(self.service.stop_reasons[-1], "test_run")

        app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "test_run",
            normal_source,
        ))

        self.assertEqual(app._motion_mode, "test_ai")
        self.assertEqual(len(self.service.ai_motion_calls), 2)
        self.assertEqual(self.service.ai_motion_calls[-1][2], 3.0)

    def test_ai_test_load_error_aborts_and_clears_pending_generation(self):
        app = self.make_connected_ai_app()
        app.start_test_run()

        with self.assertLogs(level="ERROR"):
            app.handle_ai_event(AiEvent("error", "ModelContractError: invalid"))

        self.assertIsNone(app._ai_test_pending_generation)
        self.assertIsNone(app._motion_mode)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertTrue(app.test_button._enabled)
        self.assertEqual(app.ai_status_var.get(), "Error")

    def test_ai_test_stop_clears_pending_and_late_ready_cannot_move(self):
        app = self.make_connected_ai_app()
        app.start_test_run()

        app.emergency_stop("Stopped by user")
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertIsNone(app._ai_test_pending_generation)
        self.assertIsNone(app._motion_mode)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertFalse(app.enabled)

    def test_ai_test_duration_completion_restores_disabled_and_stops_ai(self):
        app = self.make_connected_ai_app()
        app.start_test_run()
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        test_source = self.service.active_motion_generation

        app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            test_source,
        ))

        self.assertFalse(app.enabled)
        self.assertIsNone(app._motion_mode)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertTrue(app.test_button._enabled)
        self.assertEqual(self.ai.stop_calls[-1], "test_complete")

    def test_ai_test_duration_completion_restores_prior_armed_state(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        stop_count = len(self.ai.stop_calls)

        app.start_test_run()
        test_source = self.service.active_motion_generation
        app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            test_source,
        ))

        self.assertTrue(app.enabled)
        self.assertEqual(app.runtime_state_var.get(), "ARMED")
        self.assertEqual(len(self.ai.stop_calls), stop_count)
        self.assertEqual(app.ai_status_var.get(), "Ready (DirectML)")

    def test_ai_test_disconnect_cancels_pending_and_disallows_late_ready(self):
        app = self.make_connected_ai_app()
        app.start_test_run()

        app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertIsNone(app._ai_test_pending_generation)
        self.assertIsNone(app._motion_mode)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertFalse(app.enabled)

    def test_ai_test_mode_change_cancels_pending_and_disallows_late_ready(self):
        app = self.make_connected_ai_app()
        app.start_test_run()

        app.mode_var.set("jitter")
        app.on_mode_changed()
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertIsNone(app._ai_test_pending_generation)
        self.assertIsNone(app._motion_mode)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertEqual(app.mode_var.get(), "jitter")

    def test_save_config_persists_selected_mode_and_ai_snapshot(self):
        app = self.make_connected_ai_app()
        values = {
            "confidence": "0.55",
            "aim_strength": "1.25",
            "smoothing": "0.45",
            "max_step": "42",
        }
        for key, value in values.items():
            app.ai_vars[key].set(value)
        app.update()

        app._cancel_after("_save_after_id")
        app.save_config()

        saved = self.store.saved[-1]
        self.assertEqual(saved.mode, "ai_aim")
        self.assertEqual(saved.ai, AimSettings(0.55, 1.25, 0.45, 42))

    def test_invalid_ai_entry_keeps_snapshot_marks_entry_and_does_not_save(self):
        previous = self.app.get_ai_settings()
        self.app._cancel_after("_save_after_id")

        self.app.ai_vars["confidence"].set("0.01")
        self.app.update_idletasks()

        self.assertEqual(self.app.get_ai_settings(), previous)
        self.assertIn("confidence", self.app._invalid_ai_keys)
        self.assertEqual(
            self.app.ai_confidence_entry.cget("style"),
            "Liquid.Invalid.TEntry",
        )
        self.assertIsNone(self.app._save_after_id)
        self.assertEqual(self.store.saved, [])

    def test_fractional_max_step_is_invalid_and_valid_edit_recovers(self):
        previous = self.app.get_ai_settings()
        self.app.ai_vars["max_step"].set("12.5")
        self.app.update_idletasks()
        self.assertEqual(self.app.get_ai_settings(), previous)
        self.assertEqual(
            self.app.ai_max_step_entry.cget("style"),
            "Liquid.Invalid.TEntry",
        )

        self.app.ai_vars["max_step"].set("12")
        self.app.update_idletasks()

        self.assertEqual(self.app.get_ai_settings().max_step, 12)
        self.assertEqual(
            self.app.ai_max_step_entry.cget("style"),
            "Liquid.Entry.TEntry",
        )
        self.assertEqual(self.app.footer_var.get(), "Ready")

    def test_ai_snapshot_access_and_replacement_are_lock_protected(self):
        class CountingLock:
            def __init__(self):
                self.enters = 0

            def __enter__(self):
                self.enters += 1
                return self

            def __exit__(self, *_args):
                return False

        lock = CountingLock()
        self.app._ai_lock = lock
        self.app.get_ai_settings()
        self.app.ai_vars["confidence"].set("0.5")
        self.app.update_idletasks()

        self.assertGreaterEqual(lock.enters, 2)

    def test_toggle_is_blocked_while_ai_test_is_loading(self):
        app = self.make_connected_ai_app()
        app.start_test_run()

        app.toggle_enabled()

        self.assertFalse(app.enabled)
        self.assertEqual(app._motion_mode, "test_ai_loading")
        self.assertIn("Test Run is active", app.footer_var.get())

    def test_runtime_keeps_dashboard_and_stop_mounted_while_connecting(self):
        self.app.start_runtime()
        self.app.handle_service_event(ServiceEvent("connecting"))
        self.app.update_idletasks()

        self.assertEqual(self.app.shell.winfo_manager(), "pack")
        self.assertEqual(self.app.stop_button.winfo_manager(), "grid")
        self.assertEqual(self.app.connection_state_var.get(), "Connecting")
        self.assertEqual(self.app.device_status_var.get(), "Connecting to Makcu...")

    def test_startup_connects_once_without_scheduling_ui_reconnect(self):
        self.app._cancel_after("_ui_pump_after_id")
        scheduled = []
        self.app.after = lambda delay, callback: scheduled.append(
            (delay, callback)
        ) or f"scheduled-{len(scheduled)}"

        self.app.start_runtime()

        self.assertEqual(self.service.started, 1)
        self.assertEqual(self.service.reconnects, 0)
        self.assertEqual(scheduled, [])

    def test_disconnect_does_not_schedule_ui_reconnect(self):
        self.app.start_runtime()
        self.app._cancel_after("_ui_pump_after_id")
        scheduled = []
        self.app.after = lambda delay, callback: scheduled.append(
            (delay, callback)
        ) or f"scheduled-{len(scheduled)}"

        self.app.handle_service_event(ServiceEvent("disconnected", "Not found"))

        for _delay, callback in tuple(scheduled):
            callback()

        self.assertEqual(scheduled, [])
        self.assertEqual(self.service.reconnects, 0)

    def test_disconnect_keeps_selected_page_and_runtime_layout_mounted(self):
        self.app.start_runtime()
        self.app.handle_service_event(ServiceEvent("connected", "Makcu on COM3"))
        self.app.select_page(2)
        self.app.update_idletasks()
        expected_geometry = self.app.geometry()

        self.app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
        self.app.update_idletasks()

        self.assertEqual(self.app.shell.winfo_manager(), "pack")
        self.assertEqual(self.app.page_host.grid_slaves(), [self.app.settings_page])
        self.assertEqual(self.app.stop_button.winfo_manager(), "grid")
        self.assertEqual(self.app.geometry(), expected_geometry)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_disconnect_keeps_focused_configuration_input_accessible(self):
        self.app.deiconify()
        self.app.start_runtime()
        self.app.handle_service_event(ServiceEvent("connected", "Makcu on COM3"))
        self.app.select_page(2)
        self.app.update()
        self.app.sound_volume_entry.focus_force()
        self.app.update()

        self.app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
        self.app.update_idletasks()

        self.assertIs(self.app.focus_get(), self.app.sound_volume_entry)
        self.assertEqual(self.app.sound_volume_entry.winfo_viewable(), 1)

    def test_app_creates_default_nonblocking_sound_player(self):
        app = JitterApp(
            config_store=StubStore(),
            service_factory=StubService,
            hotkey_factory=StubHotkey,
            auto_start=False,
        )
        try:
            self.assertIsInstance(app.sound_player, ToggleSoundPlayer)
        finally:
            app.close_app()

    def test_fresh_balanced_preset_still_starts_disabled(self):
        self.assertEqual(self.app.preset_var.get(), "Balanced")
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_runtime_state_uses_exact_uppercase_vocabulary(self):
        """Fails if a runtime transition emits stale or mixed-case wording."""
        observed = [self.app.runtime_state_var.get()]
        self.service.connected = True
        self.app.set_enabled(True)
        observed.append(self.app.runtime_state_var.get())
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        observed.append(self.app.runtime_state_var.get())
        self.app.emergency_stop("Stopped by user")
        observed.append(self.app.runtime_state_var.get())
        self.app.start_test_run()
        observed.append(self.app.runtime_state_var.get())

        self.assertEqual(
            observed,
            ["DISABLED", "ARMED", "MOVING", "DISABLED", "TESTING"],
        )

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
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(self.ai.stop_calls[-1], "Stopped by user")

    def test_disconnect_performs_emergency_stop(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.connection_state_var.get(), "Disconnected")
        self.assertEqual(self.ai.stop_calls[-1], "Device disconnected")

    def test_test_run_bypasses_trigger_but_requires_connection(self):
        self.service.connected = False
        self.app.start_test_run()
        stopped_count = self.service.stopped
        self.service.connected = True
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        self.assertGreaterEqual(self.service.started, 1)
        self.assertEqual(self.service.stopped, stopped_count)

    def test_test_run_ignores_queued_normal_stop_before_duration_completion(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertTrue(self.app._normal_motion_started)
        normal_source = self.service.active_motion_generation
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "test_run",
            normal_source,
        ))
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        test_source = self.service.active_motion_generation
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            test_source,
        ))
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_toggle_is_blocked_while_test_run_is_active(self):
        self.service.connected = True
        self.app.start_test_run()
        self.assertFalse(self.app.enabled)
        self.app.toggle_enabled()
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")

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
        self.assertGreaterEqual(len(self.ai.stop_calls), 1)
        self.assertEqual(self.ai.closed, 1)

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
            self.app._ui_queue.put(
                ("service", None, ServiceEvent("item", index))
            )

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
        self.app._ui_queue.put(("service", None, ServiceEvent("bad")))
        self.app._ui_queue.put(
            ("hotkey", self.app._hotkey_event_epoch, None)
        )

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

    def test_global_hotkey_disable_stops_ai_immediately(self):
        app = self.make_connected_ai_app()
        app.set_enabled(True)

        app._hotkey_pressed()
        app.update()

        self.assertFalse(app.enabled)
        self.assertEqual(self.ai.stop_calls[-1], "Disabled by user")

    def test_paused_inflight_hotkey_before_stop_cannot_reenable_after_stop(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app._cancel_after("_ui_pump_after_id")
        captured = threading.Event()
        resume = threading.Event()
        worker_ident = {"value": None}

        class PauseAfterFirstWorkerCapture:
            def __init__(self):
                self._lock = threading.Lock()
                self._paused = False

            def __enter__(self):
                self._lock.acquire()
                return self

            def __exit__(self, *_args):
                self._lock.release()
                if (
                    threading.get_ident() == worker_ident["value"]
                    and not self._paused
                ):
                    self._paused = True
                    captured.set()
                    resume.wait(2.0)
                return False

        self.app._hotkey_epoch_lock = PauseAfterFirstWorkerCapture()

        def press_hotkey():
            worker_ident["value"] = threading.get_ident()
            self.app._hotkey_pressed()

        worker = threading.Thread(target=press_hotkey)
        worker.start()
        try:
            self.assertTrue(captured.wait(2.0))
            self.app.emergency_stop("Stopped by user")
        finally:
            resume.set()
            worker.join(2.0)

        self.assertFalse(worker.is_alive())
        self.app._drain_ui_queue()
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_global_hotkey_plays_on_and_off_cues_after_state_changes(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()
        self.app._drain_ui_queue()
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()
        self.app._drain_ui_queue()
        self.assertEqual(self.app.sound_player.played, [True, False])

    def test_ui_enable_button_does_not_play_hotkey_cue(self):
        self.service.connected = True
        self.app.toggle_enabled()
        self.app.toggle_enabled()
        self.assertEqual(self.app.sound_player.played, [])

    def test_sound_controls_apply_immediately_and_save(self):
        self.app.sound_enabled_var.set(False)
        self.app.sound_volume_var.set("25")
        self.app.apply_sound_settings()
        self.assertEqual(self.app.sound_player.configured[-1], (False, 25))
        self.app._cancel_after("_save_after_id")
        self.app.save_config()
        self.assertFalse(self.store.saved[-1].sound_enabled)
        self.assertEqual(self.store.saved[-1].sound_volume, 25)

    def test_sound_preview_buttons_bypass_mute_at_selected_volume(self):
        self.app.sound_enabled_var.set(False)
        self.app.sound_volume_var.set("40")
        self.app.apply_sound_settings()
        self.app.preview_sound(True)
        self.app.preview_sound(False)
        self.assertEqual(self.app.sound_player.played[-2:], [True, False])
        self.assertEqual(self.app.sound_player.forced[-2:], [True, True])

    def test_invalid_sound_volume_is_repaired_before_save(self):
        self.app.sound_volume_var.set("loud")
        self.app.save_config()
        self.assertEqual(self.app.sound_volume_var.get(), "70")
        self.assertEqual(self.store.saved[-1].sound_volume, 70)

    def test_explicit_reconnect_action_delegates_to_service_exactly_once(self):
        self.app.start_runtime()
        self.app.reconnect_button.command()
        self.assertEqual(self.service.reconnects, 1)

    def test_explicit_reconnect_action_starts_runtime_with_one_reconnect(self):
        self.app.reconnect_button.command()

        self.assertEqual(self.app.hotkey_watcher.started, 1)
        self.assertEqual(self.service.started, 0)
        self.assertEqual(self.service.reconnects, 1)
        self.assertNotIn("Reconnect requested", self.service.stop_reasons)

    def test_reconnect_does_not_call_the_synchronous_service_stop_barrier(self):
        stop_barrier_called = threading.Event()
        self.service.stop_hook = lambda _reason: stop_barrier_called.set()

        self.app.reconnect()

        self.assertFalse(stop_barrier_called.is_set())
        self.assertEqual(self.service.reconnects, 1)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_reconnect_stops_normal_jitter_and_ai_motion_before_delegating(self):
        for mode in ("jitter", "ai_aim"):
            with self.subTest(mode=mode):
                self.app.close_app()
                app = self.make_app(config=AppConfig(mode=mode))
                app.start_runtime()
                self.service.connected = True
                app.set_enabled(True)
                if mode == "ai_aim":
                    app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
                    app.queue_ai_event(AiEvent("ready", "CPUExecutionProvider"))
                app.handle_service_event(ServiceEvent("button", ("Left", True)))
                source = self.service.active_motion_generation
                calls = self.motion_calls_for_mode(mode)
                calls_before = len(calls)
                app.queue_service_event(ServiceEvent("button", ("Left", True)))

                observed = []
                self.service.reconnect_hook = lambda: observed.append((
                    app.enabled,
                    app._normal_motion_started,
                    app._ai_runtime_active,
                    app._expected_motion_generation,
                    app._motion_mode,
                    app.test_button._enabled,
                ))
                app.reconnect()
                self.service.reconnect_hook = None

                self.assertEqual(
                    observed,
                    [(False, False, False, None, None, True)],
                )
                self.assertEqual(app.runtime_state_var.get(), "DISABLED")
                self.assertNotIn("Reconnect requested", self.service.stop_reasons)
                if mode == "ai_aim":
                    self.assertIn("Reconnect requested", self.ai.stop_calls)

                app.queue_service_event(ServiceEvent(
                    "motion_error",
                    "PrivateOldMotionFailure: token=secret",
                    source,
                ))
                app._cancel_after("_ui_pump_after_id")
                app._drain_ui_queue()
                self.assertFalse(app.enabled)
                self.assertFalse(app._ai_ready)
                self.assertEqual(len(calls), calls_before)
                self.assertEqual(app.runtime_state_var.get(), "DISABLED")

    def test_reconnect_clears_deferred_normal_restart_without_a_terminal(self):
        for mode in ("jitter", "ai_aim"):
            with self.subTest(mode=mode):
                self.app.close_app()
                app = self.make_app(config=AppConfig(mode=mode))
                app.start_runtime()
                self.service.connected = True
                app.set_enabled(True)
                if mode == "ai_aim":
                    app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
                app.handle_service_event(ServiceEvent("button", ("Left", True)))
                retiring_source = self.service.active_motion_generation
                app.handle_service_event(ServiceEvent("button", ("Left", False)))
                app.handle_service_event(ServiceEvent("button", ("Left", True)))
                calls = self.motion_calls_for_mode(mode)
                calls_before = len(calls)
                self.assertEqual(app._deferred_motion_action.kind, "normal")

                app.reconnect()

                self.assertIsNone(app._deferred_motion_action)
                self.assertIsNone(app._motion_mode)
                self.assertTrue(app.test_button._enabled)
                app.queue_service_event(ServiceEvent(
                    "motion_stopped",
                    "trigger_released",
                    retiring_source,
                ))
                app._cancel_after("_ui_pump_after_id")
                app._drain_ui_queue()
                self.assertEqual(len(calls), calls_before)
                self.assertFalse(app.enabled)
                self.assertEqual(app.runtime_state_var.get(), "DISABLED")

    def test_reconnect_cancels_active_jitter_and_ai_tests_without_a_terminal(self):
        for mode in ("jitter", "ai_aim"):
            with self.subTest(mode=mode):
                self.app.close_app()
                app = self.make_app(config=AppConfig(mode=mode))
                app.start_runtime()
                self.service.connected = True
                app.set_enabled(True)
                if mode == "ai_aim":
                    app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
                app.start_test_run()
                test_source = self.service.active_motion_generation
                self.assertEqual(
                    app._motion_mode,
                    "test_ai" if mode == "ai_aim" else "test",
                )
                self.assertFalse(app.test_button._enabled)

                app.reconnect()

                self.assertFalse(app.enabled)
                self.assertIsNone(app._motion_mode)
                self.assertFalse(app._test_start_pending)
                self.assertTrue(app.test_button._enabled)
                self.assertEqual(app.runtime_state_var.get(), "DISABLED")
                app.queue_service_event(ServiceEvent(
                    "motion_stopped",
                    "duration_complete",
                    test_source,
                ))
                app._cancel_after("_ui_pump_after_id")
                app._drain_ui_queue()
                self.assertFalse(app.enabled)
                self.assertIsNone(app._motion_mode)
                self.assertTrue(app.test_button._enabled)

    def test_reconnect_cancels_loading_ai_test_and_discards_late_ready(self):
        app = self.make_connected_ai_app()
        app.start_runtime()
        app.start_test_run()
        self.assertEqual(app._motion_mode, "test_ai_loading")
        self.assertFalse(app.test_button._enabled)
        app.queue_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        app.reconnect()
        app._cancel_after("_ui_pump_after_id")
        app._drain_ui_queue()

        self.assertFalse(app.enabled)
        self.assertFalse(app._ai_runtime_active)
        self.assertFalse(app._ai_ready)
        self.assertIsNone(app._ai_test_pending_generation)
        self.assertIsNone(app._motion_mode)
        self.assertTrue(app.test_button._enabled)
        self.assertEqual(self.service.ai_motion_calls, [])
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")

    def test_reconnect_disables_armed_state_and_discards_queued_hotkey(self):
        self.app.start_runtime()
        self.service.connected = True
        self.app.set_enabled(True)
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()

        self.app.reconnect()
        self.app._drain_ui_queue()

        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(self.app.enable_button.cget("text"), "Enable Jitter")
        self.assertEqual(self.app.footer_var.get(), "Connecting to Makcu...")

    def test_makcu_motion_error_logs_detail_but_footer_is_fixed_and_actionable(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        source = self.service.active_motion_generation
        private_detail = (
            "SerialException: COM3 access denied; api_key=TOP-SECRET; "
            "trace=C:\\private\\driver.py:417"
        )

        with self.assertLogs(level="ERROR") as captured:
            self.app.handle_service_event(ServiceEvent(
                "motion_error",
                private_detail,
                source,
            ))

        self.assertEqual(
            self.app.footer_var.get(),
            "Makcu movement failed; reconnect and try again",
        )
        self.assertNotIn("TOP-SECRET", self.app.footer_var.get())
        self.assertIn(private_detail, "\n".join(captured.output))
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_timed_test_run_restores_armed_state(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped",
            "duration_complete",
            self.service.active_motion_generation,
        ))
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")
        self.assertTrue(self.app.enabled)

    def test_test_run_button_is_disabled_only_while_test_is_active(self):
        self.assertTrue(self.app.test_button._enabled)
        self.app.start_test_run()
        self.assertTrue(self.app.test_button._enabled)

        self.service.connected = True
        self.app.start_test_run()
        self.assertFalse(self.app.test_button._enabled)
        self.app.handle_service_event(
            ServiceEvent(
                "motion_stopped",
                "duration_complete",
                self.service.active_motion_generation,
            )
        )
        self.assertTrue(self.app.test_button._enabled)

        self.app.start_test_run()
        self.assertFalse(self.app.test_button._enabled)
        self.app.emergency_stop("Stopped by user")
        self.assertTrue(self.app.test_button._enabled)

    def test_invalid_motion_edit_keeps_last_snapshot(self):
        previous = self.app.get_motion_settings()
        previous_readouts = (
            self.app.motion_snapshot_size_var.get(),
            self.app.motion_snapshot_rate_var.get(),
            self.app.motion_snapshot_ramp_var.get(),
        )
        self.app.pulse_size_px_var.set("not-a-number")
        self.app.update()
        self.assertEqual(self.app.get_motion_settings(), previous)
        self.assertEqual(
            (
                self.app.motion_snapshot_size_var.get(),
                self.app.motion_snapshot_rate_var.get(),
                self.app.motion_snapshot_ramp_var.get(),
            ),
            previous_readouts,
        )
        self.assertIn("pulse_size_px", self.app._invalid_motion_keys)
        self.app.pulse_size_px_var.set("4")
        self.app.update()
        self.assertEqual(self.app.get_motion_settings().pulse_size_px, 4.0)

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
            threading.Thread(
                target=self.app.queue_ai_event,
                args=(AiEvent("loading"),),
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
        self.app.pulse_size_px_var.set("4")
        self.app.update()
        self.assertGreaterEqual(lock.enters, 2)

    def test_preset_clears_stale_invalid_entry_style(self):
        self.app.pulse_size_px_var.set("not-a-number")
        self.app.update()
        self.assertEqual(self.app.pulse_size_px_entry.cget("style"),
                         "Liquid.Invalid.TEntry")
        self.assertEqual(
            self.app.footer_var.get(),
            "Invalid value for pulse size px",
        )
        self.app.preset_var.set("Balanced")
        self.app.apply_preset()
        self.assertEqual(self.app.pulse_size_px_entry.cget("style"),
                         "Liquid.Entry.TEntry")
        self.assertEqual(self.app.footer_var.get(), "Ready")

    def test_valid_motion_edit_clears_stale_invalid_footer(self):
        self.app.pulse_size_px_var.set("not-a-number")
        self.app.update()
        self.assertEqual(
            self.app.footer_var.get(),
            "Invalid value for pulse size px",
        )

        self.app.pulse_size_px_var.set("4")
        self.app.update()

        self.assertEqual(self.app.pulse_size_px_entry.cget("style"),
                         "Liquid.Entry.TEntry")
        self.assertEqual(self.app.footer_var.get(), "Ready")


if __name__ == "__main__":
    unittest.main()
