import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import threading
import unittest

from ai_service import AiEvent
from ai_targeting import AimSettings
from combined_motion import MotionSources
from ui import JitterApp
from makcu_service import ServiceEvent
from liquid_widgets import LiquidIconButton, LiquidSlider
from motion import MotionSettings
from overlay import OverlaySetupError
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
        self.composite_motion_calls = []
        self.stop_reasons = []
        self.cancel_reasons = []
        self.stop_hook = None
        self._motion_active = False
        self.motion_active_hook = None
        self.motion_generation = 0
        self.active_motion_generation = None
        self.start_motion_hook = None
        self.start_ai_motion_hook = None
        self.start_composite_motion_hook = None

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

    def start_composite_motion_source(
        self,
        sources,
        motion_provider,
        target_provider,
        aim_provider,
        duration_s=None,
    ):
        self.started += 1
        call = SimpleNamespace(
            sources=sources,
            motion_provider=motion_provider,
            target_provider=target_provider,
            aim_provider=aim_provider,
            duration_s=duration_s,
        )
        self.composite_motion_calls.append(call)
        source = None
        if self.connected:
            if self.motion_active:
                source = self.active_motion_generation
            else:
                self.motion_generation += 1
                source = self.motion_generation
                self.active_motion_generation = source
                self.motion_active = True
        if self.start_composite_motion_hook is not None:
            self.start_composite_motion_hook()
        return source

    def stop_motion(self, reason="manual"):
        self.stopped += 1
        self.stop_reasons.append(reason)
        if self.stop_hook is not None:
            self.stop_hook(reason)

    def cancel_motion(self, reason="manual"):
        self.stopped += 1
        self.stop_reasons.append(reason)
        self.cancel_reasons.append(reason)
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
        self.detection_snapshot = object()
        self.generation = 0
        self.active_generation = None
        self.start_result = True
        self.start_exception = None
        self.stop_hook = None

    def with_sink(self, event_sink):
        self.event_sink = event_sink
        return self

    def start(self, settings_provider, zoom_gate_provider=None):
        self.start_calls.append((settings_provider, zoom_gate_provider))
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

    def latest_detection_snapshot(self):
        return self.detection_snapshot


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


class StubOverlay:
    def __init__(self):
        self.shown = self.hidden = self.closed = 0
        self.cleared = 0
        self.rendered = []
        self.render_options = []
        self.show_error = None
        self.hide_error = None
        self.render_error = None

    def show(self):
        if self.show_error is not None:
            raise self.show_error
        self.shown += 1

    def render(
        self,
        snapshot,
        *,
        now,
        color=None,
        show_heads=None,
    ):
        if self.render_error is not None:
            raise self.render_error
        self.rendered.append((snapshot, now))
        self.render_options.append((color, show_heads))

    def clear(self):
        self.cleared += 1

    def hide(self):
        self.hidden += 1
        if self.hide_error is not None:
            raise self.hide_error

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

    def make_app(self, *, config=None, clock=None):
        self.service = None
        self.store = StubStore(config or AppConfig())
        self.ai = StubAiService()
        self.overlay = StubOverlay()
        self.sounds = StubSounds()
        self.cancelled_callbacks = []

        def service_factory(event_sink):
            self.service = StubService(event_sink)
            return self.service

        self.app = JitterApp(
            config_store=self.store,
            service_factory=service_factory,
            ai_service_factory=lambda sink: self.ai.with_sink(sink),
            hotkey_factory=StubHotkey,
            overlay_factory=lambda _root: self.overlay,
            sound_player=self.sounds,
            clock=clock or (lambda: 123.5),
            auto_start=False,
        )
        original_after_cancel = self.app.after_cancel

        def recording_after_cancel(callback_id):
            self.cancelled_callbacks.append(callback_id)
            return original_after_cancel(callback_id)

        self.app.after_cancel = recording_after_cancel
        self.app.withdraw()
        return self.app

    def tearDown(self):
        try:
            if self.app is not None:
                self.app.close_app()
        except tk.TclError:
            pass

    def test_launches_with_all_runtime_switches_off_and_no_mode_selector(self):
        self.assertFalse(self.app.jitter_selected)
        self.assertFalse(self.app.ai_selected)
        self.assertFalse(self.app.master_armed)
        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.app.master_button.cget("text"), "Enable Selected")
        self.assertEqual(self.app.jitter_source_button.cget("text"), "Jitter OFF")
        self.assertEqual(self.app.ai_source_button.cget("text"), "AI Aim OFF")
        self.assertFalse(hasattr(self.app, "mode_combo"))

    def test_ai_controls_reflect_config_without_restoring_runtime_selection(self):
        self.app.close_app()
        app = self.make_app(config=AppConfig(
            ai=AimSettings(0.5, 0.6, 0.7, 30),
        ))

        self.assertEqual(app.ai_vars["confidence"].get(), "0.5")
        self.assertEqual(app.ai_vars["aim_strength"].get(), "0.6")
        self.assertEqual(app.ai_vars["smoothing"].get(), "0.7")
        self.assertEqual(app.ai_vars["max_step"].get(), "30")
        self.assertEqual(app.ai_status_var.get(), "Stopped")
        self.assertEqual(app.ai_fps_var.get(), "0 FPS")
        self.assertFalse(app.jitter_selected)
        self.assertFalse(app.ai_selected)
        self.assertFalse(app.master_armed)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.master_button.cget("text"), "Enable Selected")
        self.assertEqual(app.motion_hero_card.winfo_manager(), "grid")
        self.assertEqual(app.ai_settings_card.winfo_manager(), "grid")

    def test_overlay_preferences_reflect_config_while_runtime_starts_off(self):
        self.app.close_app()
        app = self.make_app(config=AppConfig(
            overlay_color="#00cc88",
            overlay_head_visible=False,
        ))

        self.assertEqual(app.overlay_color, "#00cc88")
        self.assertIs(app.overlay_head_visible, False)
        self.assertFalse(app.overlay_visible)

    def test_overlay_color_button_applies_choice_and_schedules_save(self):
        self.app._color_chooser = lambda **_kwargs: (
            (0.0, 204.0, 136.0),
            "#00CC88",
        )
        self.app._cancel_after("_save_after_id")
        button = self.app.overlay_color_button
        button.invoke()

        self.assertEqual(
            self.app.overlay_color, "#00cc88"
        )
        self.assertIsNotNone(self.app._save_after_id)
        self.assertIn("#00CC88", button.cget("text"))

    def test_overlay_color_cancel_keeps_current_choice(self):
        self.app._color_chooser = lambda **_kwargs: (None, None)
        before = self.app.overlay_color
        button = self.app.overlay_color_button
        self.app._cancel_after("_save_after_id")
        button.invoke()

        self.assertEqual(self.app.overlay_color, before)
        self.assertIsNone(self.app._save_after_id)

    def test_overlay_color_chooser_error_keeps_current_choice(self):
        def fail_chooser(**_kwargs):
            raise tk.TclError("chooser failed")

        self.app._color_chooser = fail_chooser
        before = self.app.overlay_color
        self.app._cancel_after("_save_after_id")

        with self.assertLogs(level="ERROR"):
            self.app.overlay_color_button.invoke()

        self.assertEqual(self.app.overlay_color, before)
        self.assertIsNone(self.app._save_after_id)
        self.assertEqual(
            self.app.footer_var.get(), "Could not open the color chooser"
        )

    def test_head_boxes_button_toggles_visibility_and_schedules_save(self):
        self.app._cancel_after("_save_after_id")
        button = self.app.overlay_head_button
        button.invoke()

        self.assertIs(
            self.app.overlay_head_visible, False
        )
        self.assertIsNotNone(self.app._save_after_id)
        self.assertEqual(button.cget("text"), "Head Boxes OFF")

    def test_ai_service_is_injected_after_widgets_without_autostart(self):
        self.assertIs(self.ai.event_sink.__self__, self.app)
        self.assertEqual(self.ai.event_sink.__func__, self.app.queue_ai_event.__func__)
        self.assertEqual(self.ai.start_calls, [])

    def test_motion_scroll_keeps_both_source_settings_available(self):
        self.assertIsInstance(self.app.motion_scroll_canvas, tk.Canvas)
        for card in (
            self.app.motion_hero_card,
            self.app.motion_summary_card,
            self.app.ai_settings_card,
            self.app.ai_status_card,
        ):
            self.assertIs(card.master, self.app.motion_scroll_content)
            self.assertEqual(card.winfo_manager(), "grid")
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

    def test_save_config_persists_only_settings_not_runtime_state(self):
        self.app._replace_motion_snapshot(MotionSettings(4.0, 45.0, "Instant"))
        self.app._replace_ai_snapshot(AimSettings(0.5, 0.6, 0.7, 30))
        self.app.trigger_var.set("Right")
        self.app.modifier_var.set("Mouse4")
        self.app.hotkey_watcher.set_vk(65)
        self.app.hotkey_name_var.set("A")
        self.app.preset_var.set("Strong")
        self.app.theme_var.set("dark")
        self.app.sound_enabled_var.set(False)
        self.app.sound_volume_var.set("35")
        self.app.jitter_selected = True
        self.app.ai_selected = True
        self.app.master_armed = True
        self.app.overlay_visible = True
        self.app.overlay_color = "#00cc88"
        self.app.overlay_head_visible = False

        self.app.save_config()

        self.assertEqual(self.store.saved[-1], AppConfig(
            motion=MotionSettings(4.0, 45.0, "Instant"),
            ai=AimSettings(0.5, 0.6, 0.7, 30),
            trigger="Right",
            modifier="Mouse4",
            hotkey_vk=65,
            hotkey_name="A",
            selected_preset="Strong",
            theme="dark",
            sound_enabled=False,
            sound_volume=35,
            overlay_color="#00cc88",
            overlay_head_visible=False,
        ))
        for name in (
            "mode", "jitter_selected", "ai_selected", "master_armed",
            "overlay_visible", "target", "detections", "fps", "provider",
        ):
            self.assertFalse(hasattr(self.store.saved[-1], name))

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
                self.assertEqual(int(card.grid_info()["row"]), 0)
        for card in (
            self.app.ai_settings_card,
            self.app.ai_status_card,
        ):
            with self.subTest(card=str(card)):
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
        for expected in (
            "Jitter OFF", "AI Aim OFF", "Enable Selected", "Overlay OFF", "STOP"
        ):
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
    def drain_ui_queue(self):
        self.app._cancel_after("_ui_pump_after_id")
        self.app._drain_ui_queue()

    def prepare_armed_sources(self, sources, *, gate_active=False):
        self.service.connected = True
        self.app.jitter_selected = sources.jitter
        self.app.ai_selected = sources.ai
        self.app._render_runtime_controls()
        self.app.set_master(True)
        if sources.ai:
            self.app.handle_ai_event(
                AiEvent("ready", "DmlExecutionProvider")
            )
        if gate_active:
            self.app.handle_service_event(
                ServiceEvent("button", ("Left", True))
            )

    def test_adaptive_zoom_gate_requires_connected_normal_ai_movement_gate(self):
        self.service.connected = True
        self.app.toggle_ai_source()
        self.app.set_master(True)
        self.assertFalse(self.app.get_adaptive_zoom_gate())
        _settings_provider, zoom_provider = self.ai.start_calls[-1]
        self.assertIs(zoom_provider.__self__, self.app)
        self.assertIs(
            zoom_provider.__func__, self.app.get_adaptive_zoom_gate.__func__
        )
        self.assertFalse(zoom_provider())

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertTrue(self.app.get_adaptive_zoom_gate())

        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_overlay_only_jitter_only_and_test_run_never_enable_zoom_gate(self):
        self.service.connected = True
        self.app.toggle_overlay()
        self.assertFalse(self.app.get_adaptive_zoom_gate())
        self.app.toggle_overlay()
        self.app.toggle_jitter_source()
        self.app.set_master(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertFalse(self.app.get_adaptive_zoom_gate())
        self.app.emergency_stop()
        self.app.ai_selected = True
        self.app.start_test_run()
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_configured_modifier_requires_both_buttons_for_zoom_gate(self):
        self.app.modifier_var.set("Right")
        self.app.on_bindings_changed()
        self.prepare_armed_sources(MotionSources(False, True))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertFalse(self.app.get_adaptive_zoom_gate())
        self.app.handle_service_event(ServiceEvent("button", ("Right", True)))
        self.assertTrue(self.app.get_adaptive_zoom_gate())
        self.app.handle_service_event(ServiceEvent("button", ("Right", False)))
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_source_removal_and_hotkey_disable_clear_zoom_gate(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.assertTrue(self.app.get_adaptive_zoom_gate())
        self.app.toggle_ai_source()
        self.assertFalse(self.app.get_adaptive_zoom_gate())

        self.app.close_app()
        self.make_app()
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()
        self.drain_ui_queue()
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_disconnect_clears_zoom_gate(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app.handle_service_event(ServiceEvent("disconnected", "lost"))
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_ai_error_clears_zoom_gate(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(AiEvent("error", "failed"))
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_stop_clears_zoom_gate(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app.emergency_stop("Stopped")
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_close_clears_zoom_gate(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app.close_app()
        self.assertFalse(self.app.get_adaptive_zoom_gate())

    def test_zoom_metric_starts_one_x_and_tracks_valid_events(self):
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
        self.assertIn("ZOOM", widget_texts(self.app))
        self.app.handle_ai_event(AiEvent("zoom", 2.0))
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app.handle_ai_event(AiEvent("zoom", 1.5))
        self.assertEqual(self.app.ai_zoom_var.get(), "1.5×")
        self.app.handle_ai_event(AiEvent("zoom", 2.0))
        self.assertEqual(self.app.ai_zoom_var.get(), "2.0×")
        self.app.handle_ai_event(AiEvent("zoom", "invalid"))
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")

    def test_stop_disconnect_and_ai_stop_reset_zoom_metric(self):
        self.app.ai_zoom_var.set("2.0×")
        self.app.emergency_stop()
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
        self.app.ai_zoom_var.set("1.5×")
        self.app.handle_service_event(ServiceEvent("disconnected", "lost"))
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
        self.app._ai_runtime_active = True
        self.app.ai_zoom_var.set("2.0×")
        self.app.handle_ai_event(AiEvent("stopped", "manual"))
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")

    def test_trigger_release_resets_zoom_metric_without_waiting_for_ai_frame(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app.handle_ai_event(AiEvent("zoom", 1.5))
        self.assertEqual(self.app.ai_zoom_var.get(), "1.5×")
        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.assertFalse(self.app.get_adaptive_zoom_gate())
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")

    def test_late_same_epoch_zoom_after_trigger_release_keeps_metric_reset(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app._cancel_after("_ui_pump_after_id")
        self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
        self.app.queue_ai_event(AiEvent("zoom", 2.0))
        self.drain_ui_queue()

        self.assertFalse(self.app.get_adaptive_zoom_gate())
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")

    def test_stale_queued_zoom_event_cannot_change_metric_after_stop(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app._cancel_after("_ui_pump_after_id")
        self.app.queue_ai_event(AiEvent("zoom", 2.0))
        self.app.emergency_stop("Stopped")
        self.drain_ui_queue()
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")

    def test_ai_error_falls_back_to_jitter_after_exact_retiring_generation(self):
        self.prepare_armed_sources(
            MotionSources(True, True), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.toggle_overlay()
        self.overlay.hide_error = RuntimeError("clear failed")

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(
                AiEvent("error", "RuntimeError: AI service failed")
            )

        self.assertTrue(self.app.jitter_selected)
        self.assertFalse(self.app.ai_selected)
        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.closed, 1)
        self.assertTrue(self.app.master_armed)
        self.assertEqual(self.service.cancel_reasons[-1], "ai_error")
        self.assertEqual(len(self.service.composite_motion_calls), 1)

        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "ai_error", retiring + 1
        ))
        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "ai_error", retiring
        ))
        self.assertEqual(
            self.service.composite_motion_calls[-1].sources,
            MotionSources(True, False),
        )

    def test_ai_error_disarms_ai_only_source(self):
        self.prepare_armed_sources(MotionSources(False, True))

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(
                AiEvent("error", "RuntimeError: AI service failed")
            )

        self.assertFalse(self.app.ai_selected)
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_overlay_only_ai_error_does_not_interrupt_jitter_motion(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        generation = self.service.active_motion_generation
        self.app.toggle_overlay()
        self.overlay.hide_error = RuntimeError("clear failed")

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(
                AiEvent("error", "RuntimeError: overlay detector failed")
            )

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.closed, 1)
        self.assertTrue(self.app.jitter_selected)
        self.assertFalse(self.app.ai_selected)
        self.assertTrue(self.app.master_armed)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app._expected_motion_generation, generation)
        self.assertEqual(self.service.active_motion_generation, generation)
        self.assertNotIn("ai_error", self.service.cancel_reasons)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_overlay_only_ai_error_does_not_abort_jitter_test(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app.set_master(True)
        self.app.start_test_run()
        generation = self.service.active_motion_generation
        self.app.toggle_overlay()

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(
                AiEvent("error", "RuntimeError: overlay detector failed")
            )

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.app._motion_mode, "test_jitter")
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app._expected_motion_generation, generation)
        self.assertEqual(self.service.active_motion_generation, generation)
        self.assertNotIn("ai_error", self.service.cancel_reasons)
        self.assertFalse(self.app.test_button._enabled)

        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "duration_complete", generation
        ))
        self.assertTrue(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_disconnect_disarms_motion_but_keeps_overlay_demand(self):
        self.app.toggle_overlay()
        self.prepare_armed_sources(MotionSources(False, True))

        self.app.handle_service_event(
            ServiceEvent("disconnected", "Device lost")
        )

        self.assertFalse(self.app.master_armed)
        self.assertTrue(self.app.overlay_visible)
        self.assertTrue(self.app._ai_runtime_active)
        self.assertTrue(self.app.ai_selected)

    def test_disconnect_recovery_start_failure_hides_existing_overlay(self):
        self.app.toggle_overlay()
        self.app.handle_ai_event(AiEvent("stopped", "worker ended"))
        self.ai.start_result = False
        self.overlay.hide_error = RuntimeError("clear failed")

        with self.assertLogs(level="ERROR"):
            self.app.handle_service_event(
                ServiceEvent("disconnected", "Device lost")
            )

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.hidden, 1)
        self.assertEqual(self.overlay.closed, 1)
        self.assertFalse(self.app._ai_runtime_active)

    def test_overlay_toggle_hide_failure_closes_native_overlay(self):
        self.app.toggle_overlay()
        self.overlay.hide_error = RuntimeError("clear failed")

        with self.assertLogs(level="ERROR"):
            self.app.toggle_overlay()

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.hidden, 1)
        self.assertEqual(self.overlay.closed, 1)
        self.assertEqual(self.ai.stop_calls[-1], "Overlay disabled")

    def test_stop_hide_failure_closes_native_overlay(self):
        self.app.toggle_overlay()
        self.overlay.hide_error = RuntimeError("clear failed")

        with self.assertLogs(level="ERROR"):
            self.app.emergency_stop("Stopped by user")

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.hidden, 1)
        self.assertEqual(self.overlay.closed, 1)

    def test_stop_hides_overlay_but_preserves_source_choices(self):
        self.prepare_armed_sources(MotionSources(True, True))
        self.app.toggle_overlay()

        self.app.emergency_stop("Stopped by user")

        self.assertFalse(self.app.master_armed)
        self.assertFalse(self.app.overlay_visible)
        self.assertTrue(self.app.jitter_selected)
        self.assertTrue(self.app.ai_selected)
        self.assertEqual(self.overlay.hidden, 1)

    def test_disconnect_during_test_never_restores_master_on_late_completion(self):
        self.prepare_armed_sources(MotionSources(True, False))
        self.app.start_test_run()
        generation = self.service.active_motion_generation

        self.app.handle_service_event(
            ServiceEvent("disconnected", "Device lost")
        )
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "duration_complete", generation
        ))

        self.assertFalse(self.app.master_armed)
        self.assertIsNone(self.app._motion_mode)

    def test_close_cancels_overlay_poll_and_closes_overlay(self):
        self.app.toggle_overlay()
        callback = self.app._overlay_after_id

        self.app.close_app()

        self.assertIn(callback, self.cancelled_callbacks)
        self.assertEqual(self.overlay.closed, 1)

    def test_hotkey_toggles_master_and_plays_only_successful_state_changes(self):
        self.service.connected = True
        self.app._cancel_after("_ui_pump_after_id")

        self.app._hotkey_pressed()
        self.drain_ui_queue()
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.sounds.played, [])

        self.app.toggle_jitter_source()
        self.app._hotkey_pressed()
        self.drain_ui_queue()
        self.assertTrue(self.app.master_armed)
        self.app._hotkey_pressed()
        self.drain_ui_queue()
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.sounds.played, [True, False])

    def test_two_legitimate_queued_hotkeys_preserve_toggle_parity(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app._cancel_after("_ui_pump_after_id")

        self.app._hotkey_pressed()
        self.app._hotkey_pressed()
        self.drain_ui_queue()

        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(self.sounds.played, [True, False])

    def test_stale_queued_hotkey_cannot_rearm_after_stop(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()

        self.app.emergency_stop("Stopped by user")
        self.drain_ui_queue()

        self.assertFalse(self.app.master_armed)

    def test_queued_hotkey_cannot_run_after_close(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()

        self.app.close_app()
        self.app._drain_ui_queue()

        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.sounds.played, [])

    def test_test_run_uses_selected_source_matrix(self):
        for sources in (
            MotionSources(True, False),
            MotionSources(False, True),
            MotionSources(True, True),
        ):
            with self.subTest(sources=sources):
                self.app.close_app()
                app = self.make_app()
                self.service.connected = True
                app.jitter_selected = sources.jitter
                app.ai_selected = sources.ai

                app.start_test_run()

                if sources.ai:
                    self.assertEqual(self.service.composite_motion_calls, [])
                    app.handle_ai_event(
                        AiEvent("ready", "DmlExecutionProvider")
                    )
                self.assertEqual(
                    self.service.composite_motion_calls[-1].sources,
                    sources,
                )
                self.assertEqual(
                    self.service.composite_motion_calls[-1].duration_s,
                    3.0,
                )

    def test_test_run_rejects_no_sources(self):
        self.service.connected = True

        self.app.start_test_run()

        self.assertIsNone(self.app._motion_mode)
        self.assertIn("Select Jitter or AI Aim", self.app.footer_var.get())
        self.assertEqual(self.service.composite_motion_calls, [])

    def test_ai_test_ready_first_still_waits_for_exact_retiring_worker(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.toggle_ai_source()

        self.app.start_test_run()
        self.assertEqual(self.app._motion_mode, "test_combined_loading")
        self.assertTrue(self.app._test_waiting_for_motion_stop)
        self.assertFalse(self.app.master_armed)

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "test_run", retiring + 100
        ))
        self.assertEqual(len(self.service.composite_motion_calls), 1)

        self.service.emit(ServiceEvent(
            "motion_stopped", "test_run", retiring
        ))
        self.drain_ui_queue()

        self.assertEqual(len(self.service.composite_motion_calls), 2)
        self.assertEqual(
            self.service.composite_motion_calls[-1].sources,
            MotionSources(True, True),
        )
        self.assertEqual(
            self.service.composite_motion_calls[-1].duration_s,
            3.0,
        )

    def test_ai_test_retiring_worker_first_still_waits_for_ready(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.toggle_ai_source()

        self.app.start_test_run()
        self.service.emit(ServiceEvent(
            "motion_stopped", "sources_changed", retiring
        ))
        self.drain_ui_queue()

        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.assertFalse(self.app._test_waiting_for_motion_stop)
        self.assertEqual(self.app._motion_mode, "test_combined_loading")

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertEqual(len(self.service.composite_motion_calls), 2)
        self.assertEqual(
            self.service.composite_motion_calls[-1].sources,
            MotionSources(True, True),
        )
        self.assertEqual(
            self.service.composite_motion_calls[-1].duration_s,
            3.0,
        )
        self.assertFalse(self.app.master_armed)

    def test_duration_complete_restores_prior_master_and_source_choices(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app.set_master(True)
        selected = self.app._selected_sources()

        self.app.start_test_run()
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        generation = self.service.active_motion_generation
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "duration_complete", generation
        ))

        self.assertTrue(self.app.master_armed)
        self.assertEqual(self.app._selected_sources(), selected)
        self.assertIsNone(self.app._motion_mode)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_ai_test_suspends_master_while_loading_and_running(self):
        self.service.connected = True
        self.app.toggle_ai_source()
        self.app.set_master(True)

        self.app.start_test_run()

        self.assertEqual(self.app._motion_mode, "test_ai_loading")
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertEqual(self.app._motion_mode, "test_ai")
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "TESTING")
        generation = self.service.active_motion_generation
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "duration_complete", generation
        ))
        self.assertTrue(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_test_run_keeps_master_off_when_it_started_disarmed(self):
        self.service.connected = True
        self.app.toggle_jitter_source()

        self.app.start_test_run()
        generation = self.service.active_motion_generation
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "duration_complete", generation
        ))

        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_overlay_only_starts_ai_without_makcu_and_hiding_stops_final_demand(self):
        self.service.connected = False

        self.app.toggle_overlay()

        self.assertTrue(self.app.overlay_visible)
        self.assertEqual(self.overlay.shown, 1)
        self.assertEqual(len(self.ai.start_calls), 1)
        self.assertEqual(self.service.composite_motion_calls, [])

        self.app.toggle_overlay()

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.hidden, 1)
        self.assertEqual(self.ai.stop_calls[-1], "Overlay disabled")

    def test_hiding_overlay_does_not_reload_or_stop_armed_ai(self):
        self.service.connected = True
        self.app.toggle_ai_source()
        self.app.set_master(True)
        self.app.toggle_overlay()
        starts = len(self.ai.start_calls)

        self.app.toggle_overlay()

        self.assertEqual(len(self.ai.start_calls), starts)
        self.assertEqual(self.ai.stop_calls, [])

    def test_overlay_setup_failure_turns_off_only_overlay(self):
        self.app.toggle_ai_source()
        self.overlay.show_error = OverlaySetupError("affinity failed")

        with self.assertLogs(level="ERROR"):
            self.app.toggle_overlay()

        self.assertFalse(self.app.overlay_visible)
        self.assertTrue(self.app.ai_selected)
        self.assertEqual(self.ai.start_calls, [])

    def test_overlay_ai_start_failure_hides_overlay_and_starts_no_motion(self):
        self.ai.start_result = False

        with self.assertLogs(level="ERROR"):
            self.app.toggle_overlay()

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.shown, 1)
        self.assertEqual(self.overlay.hidden, 1)
        self.assertEqual(self.service.composite_motion_calls, [])

    def test_overlay_poll_renders_detection_snapshot_with_injected_clock(self):
        self.app.toggle_overlay()

        self.assertEqual(
            self.overlay.rendered[-1],
            (self.ai.detection_snapshot, 123.5),
        )
        self.assertEqual(
            self.overlay.render_options[-1],
            ("#ff2b2b", True),
        )
        self.assertIsNotNone(self.app._overlay_after_id)

    def test_head_boxes_off_reaches_overlay_render(self):
        self.app.overlay_head_button.invoke()

        self.app.toggle_overlay()

        self.assertEqual(
            self.overlay.render_options[-1],
            ("#ff2b2b", False),
        )

    def test_overlay_render_error_turns_off_overlay_and_final_ai_demand(self):
        self.overlay.render_error = RuntimeError("canvas failed")

        with self.assertLogs(level="ERROR"):
            self.app.toggle_overlay()

        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.overlay.closed, 1)
        self.assertEqual(self.ai.stop_calls[-1], "Overlay failed")

    def test_master_rejects_empty_selection_and_disconnected_device(self):
        self.app.toggle_master()
        self.assertFalse(self.app.master_armed)
        self.assertIn("Select Jitter or AI Aim", self.app.footer_var.get())
        self.app.toggle_jitter_source()
        self.app.toggle_master()
        self.assertFalse(self.app.master_armed)
        self.assertEqual(
            self.app.footer_var.get(), "Makcu device is not connected"
        )

    def test_both_selected_start_one_combined_worker_when_gate_activates(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app.toggle_ai_source()
        self.app.toggle_master()
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))

        self.assertEqual(
            self.service.composite_motion_calls[-1].sources,
            MotionSources(True, True),
        )
        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.assertTrue(self.app.master_armed)

    def test_source_change_cancels_exact_generation_before_restart(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app.set_master(True)
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        retiring = self.service.active_motion_generation

        self.app.toggle_ai_source()
        self.assertEqual(self.service.cancel_reasons[-1], "sources_changed")
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "sources_changed", retiring + 99
        ))
        self.assertEqual(len(self.service.composite_motion_calls), 1)

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "sources_changed", retiring
        ))
        self.assertEqual(
            self.service.composite_motion_calls[-1].sources,
            MotionSources(True, True),
        )
        self.assertNotEqual(
            self.service.active_motion_generation,
            retiring,
        )

    def test_removing_final_source_disarms_master_and_preserves_selection(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app.set_master(True)

        self.app.toggle_jitter_source()

        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app._selected_sources(), MotionSources(False, False))
        self.assertEqual(self.app.master_button.cget("text"), "Enable Selected")

    def test_ai_start_failure_while_jitter_moves_keeps_current_worker(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        generation = self.service.active_motion_generation
        self.ai.start_result = False

        with self.assertLogs(level="ERROR"):
            self.app.toggle_ai_source()

        self.assertFalse(self.app.ai_selected)
        self.assertTrue(self.app.master_armed)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")
        self.assertEqual(self.service.active_motion_generation, generation)
        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.assertNotIn("sources_changed", self.service.cancel_reasons)

    def test_test_run_waits_for_worker_already_retiring_after_gate_release(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", False))
        )

        self.app.start_test_run()

        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.assertTrue(self.app._test_waiting_for_motion_stop)
        self.service.emit(ServiceEvent(
            "motion_stopped", "trigger_released", retiring
        ))
        self.drain_ui_queue()
        self.assertEqual(len(self.service.composite_motion_calls), 2)
        self.assertEqual(
            self.service.composite_motion_calls[-1].duration_s, 3.0
        )

    def test_reentrant_stop_terminal_during_test_handoff_starts_once(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.service.stop_hook = lambda _reason: self.service.emit(
            ServiceEvent("motion_stopped", "test_run", retiring)
        )

        self.app.start_test_run()
        fresh = self.service.active_motion_generation

        self.assertNotEqual(fresh, retiring)
        self.assertEqual(len(self.service.composite_motion_calls), 2)
        self.assertEqual(self.app._motion_mode, "test_jitter")
        self.assertEqual(self.app._expected_motion_generation, fresh)
        self.drain_ui_queue()
        self.assertEqual(len(self.service.composite_motion_calls), 2)
        self.assertEqual(self.app._expected_motion_generation, fresh)

    def test_terminal_queued_during_retiring_worker_probe_starts_test_once(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", False))
        )

        def emit_terminal_during_probe():
            self.service.motion_active_hook = None
            self.service.emit(ServiceEvent(
                "motion_stopped", "trigger_released", retiring
            ))

        self.service.motion_active_hook = emit_terminal_during_probe
        self.app.start_test_run()

        self.assertTrue(self.app._test_waiting_for_motion_stop)
        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.drain_ui_queue()
        self.assertEqual(len(self.service.composite_motion_calls), 2)
        self.assertEqual(self.app._motion_mode, "test_jitter")
        self.assertEqual(
            self.service.composite_motion_calls[-1].duration_s, 3.0
        )

    def test_stale_terminal_from_composite_start_hook_cannot_clear_fresh_worker(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", False))
        )
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", True))
        )
        hook_calls = []

        def emit_stale_terminal():
            hook_calls.append("called")
            self.service.emit(ServiceEvent(
                "motion_stopped", "late duplicate", retiring
            ))

        self.service.start_composite_motion_hook = emit_stale_terminal
        self.service.emit(ServiceEvent(
            "motion_stopped", "trigger_released", retiring
        ))
        self.drain_ui_queue()
        fresh = self.service.active_motion_generation

        self.assertEqual(hook_calls, ["called"])
        self.assertNotEqual(fresh, retiring)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(self.app._expected_motion_generation, fresh)
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")

    def test_binding_change_cancels_pending_test_handoff(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", False))
        )
        self.app.start_test_run()
        self.assertIsNotNone(self.app._deferred_motion_action)
        self.assertFalse(self.app.master_armed)

        self.app.trigger_var.set("Mouse4")
        self.app.on_bindings_changed()

        self.assertIsNone(self.app._motion_mode)
        self.assertIsNone(self.app._deferred_motion_action)
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")
        self.assertTrue(self.app.test_button._enabled)
        self.service.emit(ServiceEvent(
            "motion_stopped", "trigger_released", retiring
        ))
        self.drain_ui_queue()
        self.assertEqual(len(self.service.composite_motion_calls), 1)

    def test_ai_only_start_failure_disarms_and_deselects_ai(self):
        self.service.connected = True
        self.app.ai_selected = True
        self.ai.start_result = False

        with self.assertLogs(level="ERROR"):
            self.app.set_master(True)

        self.assertFalse(self.app.ai_selected)
        self.assertFalse(self.app.master_armed)
        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")

    def test_combined_start_failure_falls_back_to_armed_jitter(self):
        self.service.connected = True
        self.app.jitter_selected = True
        self.app.ai_selected = True
        self.ai.start_exception = RuntimeError("load failed")

        with self.assertLogs(level="ERROR"):
            self.app.set_master(True)

        self.assertTrue(self.app.jitter_selected)
        self.assertFalse(self.app.ai_selected)
        self.assertTrue(self.app.master_armed)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_gate_release_cancels_composite_but_keeps_master_and_ai_demand(self):
        self.prepare_armed_sources(
            MotionSources(True, True), gate_active=True
        )

        self.app.handle_service_event(
            ServiceEvent("button", ("Left", False))
        )

        self.assertEqual(self.service.cancel_reasons[-1], "trigger_released")
        self.assertTrue(self.app.master_armed)
        self.assertTrue(self.app._ai_runtime_active)
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")

    def test_modifier_gate_requires_both_buttons_for_composite_worker(self):
        self.app.modifier_var.set("Right")
        self.app.on_bindings_changed()
        self.prepare_armed_sources(MotionSources(True, True))

        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertEqual(self.service.composite_motion_calls, [])
        self.app.handle_service_event(ServiceEvent("button", ("Right", True)))

        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.assertEqual(
            self.service.composite_motion_calls[-1].sources,
            MotionSources(True, True),
        )

    def test_binding_change_waits_for_exact_retiring_source_before_restart(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        retiring = self.service.active_motion_generation

        self.app.trigger_var.set("Mouse4")
        self.app.on_bindings_changed()
        self.app.handle_service_event(
            ServiceEvent("button", ("Mouse4", True))
        )
        self.assertEqual(len(self.service.composite_motion_calls), 1)
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "bindings_changed", retiring + 9
        ))
        self.assertEqual(len(self.service.composite_motion_calls), 1)

        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "bindings_changed", retiring
        ))
        self.assertEqual(len(self.service.composite_motion_calls), 2)

    def test_old_terminal_cannot_clear_fresh_composite_generation(self):
        self.prepare_armed_sources(
            MotionSources(True, False), gate_active=True
        )
        old_generation = self.service.active_motion_generation
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", False))
        )
        self.service.motion_active = False
        self.service.active_motion_generation = None
        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "trigger_released", old_generation
        ))
        self.app.handle_service_event(
            ServiceEvent("button", ("Left", True))
        )
        fresh_generation = self.service.active_motion_generation

        self.app.handle_service_event(ServiceEvent(
            "motion_stopped", "trigger_released", old_generation
        ))

        self.assertNotEqual(fresh_generation, old_generation)
        self.assertTrue(self.app._normal_motion_started)
        self.assertEqual(
            self.app._expected_motion_generation, fresh_generation
        )

    def test_test_run_disables_sources_but_leaves_overlay_and_stop_available(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        selected = self.app._selected_sources()

        self.app.start_test_run()

        self.assertEqual(str(self.app.jitter_source_button.cget("state")), "disabled")
        self.assertEqual(str(self.app.ai_source_button.cget("state")), "disabled")
        self.assertFalse(self.app.test_button._enabled)
        self.assertEqual(str(self.app.overlay_button.cget("state")), "normal")
        self.assertEqual(str(self.app.stop_button.cget("state")), "normal")
        self.app.toggle_jitter_source()
        self.assertEqual(self.app._selected_sources(), selected)
        self.app.toggle_overlay()
        self.assertTrue(self.app.overlay_visible)

    def test_stop_cancels_loading_ai_test_and_late_ready_cannot_move(self):
        self.service.connected = True
        self.app.toggle_ai_source()
        self.app.start_test_run()
        self.assertEqual(self.app._motion_mode, "test_ai_loading")

        self.app.emergency_stop("Stopped by user")
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertIsNone(self.app._motion_mode)
        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.service.composite_motion_calls, [])

    def test_ai_error_during_loading_test_aborts_without_master_restore(self):
        self.service.connected = True
        self.app.toggle_ai_source()
        self.app.set_master(True)
        self.app.start_test_run()

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(AiEvent("error", "capture failed"))

        self.assertIsNone(self.app._motion_mode)
        self.assertFalse(self.app.master_armed)
        self.assertFalse(self.app.ai_selected)
        self.assertTrue(self.app.test_button._enabled)

    def test_current_makcu_motion_error_stops_runtime_but_preserves_sources(self):
        self.prepare_armed_sources(
            MotionSources(True, True), gate_active=True
        )
        self.app.toggle_overlay()
        selected = self.app._selected_sources()
        generation = self.service.active_motion_generation

        with self.assertLogs(level="ERROR"):
            self.app.handle_service_event(ServiceEvent(
                "motion_error", "controller failed", generation
            ))

        self.assertFalse(self.app.master_armed)
        self.assertFalse(self.app.overlay_visible)
        self.assertEqual(self.app._selected_sources(), selected)
        self.assertEqual(
            self.app.footer_var.get(),
            "Makcu movement failed; reconnect and try again",
        )

    def test_worker_events_enter_tk_only_through_the_queue(self):
        self.service.connected = True
        self.app.toggle_jitter_source()
        self.app.set_master(True)
        self.app._cancel_after("_ui_pump_after_id")

        thread = threading.Thread(
            target=lambda: self.service.emit(
                ServiceEvent("button", ("Left", True))
            )
        )
        thread.start()
        thread.join()
        self.assertEqual(self.service.composite_motion_calls, [])

        self.drain_ui_queue()
        self.assertEqual(len(self.service.composite_motion_calls), 1)

    def test_unsolicited_ai_ready_is_ignored_without_runtime_demand(self):
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.assertFalse(self.app._ai_ready)
        self.assertEqual(self.app.ai_status_var.get(), "Stopped")

    def test_start_runtime_is_idempotent(self):
        self.app.start_runtime()
        self.app.start_runtime()

        self.assertEqual(self.app.hotkey_watcher.started, 1)
        self.assertEqual(self.service.started, 1)

    def test_reconnect_disarms_and_discards_queued_hotkey_but_keeps_sources(self):
        self.prepare_armed_sources(MotionSources(True, False))
        self.app._cancel_after("_ui_pump_after_id")
        self.app._hotkey_pressed()
        selected = self.app._selected_sources()

        self.app.reconnect()
        self.drain_ui_queue()

        self.assertFalse(self.app.master_armed)
        self.assertEqual(self.app._selected_sources(), selected)
        self.assertEqual(self.service.reconnects, 1)

    def test_close_stops_every_owned_service(self):
        self.app.toggle_overlay()

        self.app.close_app()

        self.assertEqual(self.app.hotkey_watcher.stopped, 1)
        self.assertEqual(self.service.closed, 1)
        self.assertEqual(self.ai.closed, 1)
        self.assertEqual(self.overlay.closed, 1)
        self.assertEqual(self.sounds.closed, 1)

    def test_runtime_status_and_ai_metrics_use_concise_vocabulary(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.assertEqual(self.app.runtime_state_var.get(), "ARMED")
        self.app.handle_ai_event(AiEvent("fps", 37.25))
        self.assertEqual(self.app.ai_fps_var.get(), "37.2 FPS")
        self.assertEqual(self.app.ai_provider_var.get(), "DirectML")
        self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
        self.assertEqual(self.app.runtime_state_var.get(), "MOVING")
        self.app.emergency_stop("Stopped")
        self.assertEqual(self.app.runtime_state_var.get(), "DISABLED")
