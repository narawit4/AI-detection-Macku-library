import tkinter as tk
from tkinter import ttk
from dataclasses import replace
from types import SimpleNamespace
import threading
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from jitter_app.ai.model_selection import (
    ModelChoice,
    ModelValidationEvent,
    bundled_model_choice,
)
from jitter_app.ai.capture import CENTER_320, FULL_DISPLAY
from jitter_app.ai.service import AiEvent
from jitter_app.ai.targeting import (
    AimSettings,
    DEFAULT_RESPONSE_CURVE,
    response_curve_value,
)
from jitter_app.motion.combined import MotionSources
from jitter_app.device.display_timing import RuntimeCadence
from jitter_app.presentation.ui import JitterApp
from jitter_app.device.makcu import ServiceEvent
from jitter_app.presentation.widgets import (
    CollapsibleSection,
    LiquidSlider,
)
from jitter_app.motion.engine import MotionSettings
from jitter_app.presentation.overlay import OverlaySetupError, OverlayStyle
from jitter_app.config.store import AppConfig
from jitter_app.presentation.sound import ToggleSoundPlayer


class StubStore:
    def __init__(self, config=None):
        self.saved = []
        self.config = config

    def load(self):
        from jitter_app.config.store import AppConfig, LoadOutcome
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
        self.stop_exception = None
        self.stop_hook = None
        self.reset_targeting_calls = 0
        self.targeting_revision = 0

    def with_sink(self, event_sink):
        self.event_sink = event_sink
        return self

    def start(
        self,
        settings_provider,
        zoom_gate_provider=None,
        *,
        model_path=None,
        capture_mode=CENTER_320,
    ):
        self.start_calls.append(
            (settings_provider, zoom_gate_provider, model_path, capture_mode)
        )
        if self.start_exception is not None:
            raise self.start_exception
        if not self.start_result:
            return self.start_result
        self.generation += 1
        self.active_generation = self.generation
        return self.active_generation

    def stop(self, reason="manual"):
        self.stop_calls.append(reason)
        if self.stop_exception is not None:
            raise self.stop_exception
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

    def reset_targeting(self):
        self.reset_targeting_calls += 1
        self.targeting_revision += 1
        self.snapshot = None
        self.detection_snapshot = None
        return self.targeting_revision


class StrictDuplicateStartAiService(StubAiService):
    """Mirror production ownership: a duplicate start keeps its generation."""

    def start(
        self,
        settings_provider,
        zoom_gate_provider=None,
        *,
        model_path=None,
        capture_mode=CENTER_320,
    ):
        self.start_calls.append(
            (settings_provider, zoom_gate_provider, model_path, capture_mode)
        )
        if self.start_exception is not None:
            raise self.start_exception
        if not self.start_result:
            return self.start_result
        if self.active_generation is not None:
            return self.active_generation
        self.generation += 1
        self.active_generation = self.generation
        return self.active_generation


class StubModelValidator:
    def __init__(self):
        self.event_sink = None
        self.start_calls = []
        self.cancelled = 0
        self.closed = 0
        self.start_result = True

    def with_sink(self, event_sink):
        self.event_sink = event_sink
        return self

    def start(self, choice, token):
        self.start_calls.append((choice, token))
        return self.start_result

    def cancel(self):
        self.cancelled += 1

    def close(self):
        self.closed += 1

    def emit(self, event):
        self.event_sink(event)


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
        self.styles = []
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
        runtime=None,
        style=None,
    ):
        if self.render_error is not None:
            raise self.render_error
        self.rendered.append((snapshot, now))
        self.render_options.append((color, show_heads, runtime))
        self.styles.append(style)

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

    def make_app(
        self,
        *,
        config=None,
        clock=None,
        runtime_cadence=None,
        use_default_factories=False,
        falsey_factories=False,
        strict_duplicate_ai=False,
    ):
        self.service = None
        self.store = StubStore(config or AppConfig())
        self.ai = (
            StrictDuplicateStartAiService()
            if strict_duplicate_ai else StubAiService()
        )
        self.model_validator = StubModelValidator()
        self.model_dialog_result = ""
        self.overlay = StubOverlay()
        self.sounds = StubSounds()
        self.cancelled_callbacks = []

        def service_factory(event_sink):
            self.service = StubService(event_sink)
            return self.service

        ai_service_factory = lambda sink: self.ai.with_sink(sink)
        if falsey_factories:
            class FalseyFactory:
                def __init__(self, factory):
                    self.factory = factory

                def __bool__(self):
                    return False

                def __call__(self, event_sink):
                    return self.factory(event_sink)

            service_factory = FalseyFactory(service_factory)
            ai_service_factory = FalseyFactory(ai_service_factory)

        self.app = JitterApp(
            config_store=self.store,
            service_factory=None if use_default_factories else service_factory,
            ai_service_factory=(
                None
                if use_default_factories
                else ai_service_factory
            ),
            model_validator_factory=lambda sink: self.model_validator.with_sink(sink),
            model_file_chooser=lambda **_kwargs: self.model_dialog_result,
            hotkey_factory=StubHotkey,
            overlay_factory=lambda _root: self.overlay,
            sound_player=self.sounds,
            clock=clock or (lambda: 123.5),
            auto_start=False,
            runtime_cadence=(
                runtime_cadence
                if runtime_cadence is not None
                else RuntimeCadence(None, 120, 1000)
            ),
        )
        original_after_cancel = self.app.after_cancel

        def recording_after_cancel(callback_id):
            self.cancelled_callbacks.append(callback_id)
            return original_after_cancel(callback_id)

        self.app.after_cancel = recording_after_cancel
        self.app.withdraw()
        return self.app

    def drain_model_ui_queue(self):
        self.app._cancel_after("_ui_pump_after_id")
        self.app._drain_ui_queue()

    def begin_custom_model_switch(self, filename):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / filename
        path.write_bytes(b"model")
        self.model_dialog_result = str(path)
        self.app.browse_ai_model()
        return self.model_validator.start_calls[-1]

    @staticmethod
    def validated_model_choice(choice, input_size=320):
        return replace(choice, input_size=input_size)

    def test_app_default_factories_receive_runtime_cadence(self):
        self.app.close_app()
        cadence = RuntimeCadence(144, 144, 1000)
        makcu_kwargs = []
        ai_kwargs = []

        def make_service(event_sink, **kwargs):
            makcu_kwargs.append(kwargs)
            self.service = StubService(event_sink)
            return self.service

        def make_ai_service(event_sink, **kwargs):
            ai_kwargs.append(kwargs)
            return self.ai.with_sink(event_sink)

        with (
            mock.patch("jitter_app.presentation.ui.MakcuService", side_effect=make_service),
            mock.patch("jitter_app.presentation.ui.AiService", side_effect=make_ai_service),
        ):
            app = self.make_app(
                runtime_cadence=cadence,
                use_default_factories=True,
            )

        self.assertEqual(app.runtime_cadence, cadence)
        self.assertEqual(makcu_kwargs, [{"ai_poll_hz": 1000}])
        self.assertEqual(ai_kwargs, [{"capture_fps": 144}])
        self.assertEqual(
            app.ai_cadence_var.get(),
            "DISPLAY 144 HZ · SERVO 1000 HZ",
        )

    def test_fallback_cadence_status_is_explicit(self):
        self.app.close_app()
        app = self.make_app(
            runtime_cadence=RuntimeCadence(None, 120, 1000),
        )

        self.assertEqual(
            app.ai_cadence_var.get(),
            "DISPLAY AUTO · SERVO 1000 HZ",
        )

    def test_injected_service_factories_keep_one_argument_contract(self):
        self.app.close_app()
        app = self.make_app(
            runtime_cadence=RuntimeCadence(165, 165, 330),
            falsey_factories=True,
        )

        self.assertIs(app.service, self.service)
        self.assertIs(app.ai_service, self.ai)

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
        self.assertEqual(app.target_area_var.get(), "Head")
        self.assertEqual(app.ai_status_var.get(), "Stopped")
        self.assertEqual(app.ai_fps_var.get(), "0 FPS")
        self.assertFalse(app.jitter_selected)
        self.assertFalse(app.ai_selected)
        self.assertFalse(app.master_armed)
        self.assertEqual(app.runtime_state_var.get(), "DISABLED")
        self.assertEqual(app.master_button.cget("text"), "Enable Selected")
        self.assertEqual(app.motion_hero_card.winfo_manager(), "grid")
        self.assertEqual(app.ai_settings_card.winfo_manager(), "grid")

    def test_target_area_starts_head_changes_live_without_scheduling_save(self):
        self.app.close_app()
        app = self.make_app(config=AppConfig(
            ai=AimSettings(target_area="upper_body"),
        ))
        self.assertEqual(app.target_area_var.get(), "Head")
        self.assertEqual(tuple(app.target_area_combo.cget("values")), (
            "Head", "Upper Body", "Chest",
        ))

        app._cancel_after("_save_after_id")
        app.target_area_var.set("Chest")
        app.ai_zoom_var.set("2.0×")
        app._target_area_changed()

        self.assertEqual(app.get_ai_settings().target_area, "chest")
        self.assertEqual(self.ai.reset_targeting_calls, 1)
        self.assertEqual(app.ai_zoom_var.get(), "1.0×")
        self.assertIsNone(app._save_after_id)

    def test_capture_mode_starts_centered_and_shares_target_row(self):
        self.assertEqual(self.app._capture_mode, CENTER_320)
        self.assertEqual(self.app.capture_mode_var.get(), "Center 320")
        self.assertEqual(
            tuple(self.app.capture_mode_combo.cget("values")),
            ("Center 320", "Full Display"),
        )
        self.assertEqual(
            self.app.capture_mode_combo.master.master,
            self.app.target_area_combo.master.master,
        )

    def test_capture_mode_is_runtime_only_and_new_app_restores_center(self):
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        self.app.save_config()
        self.assertFalse(hasattr(self.store.saved[-1], "capture_mode"))
        config = self.app.config
        self.app.close_app()
        app = self.make_app(config=config)
        self.assertEqual(app._capture_mode, CENTER_320)

    def test_idle_capture_mode_change_does_not_start_ai(self):
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
        self.assertEqual(self.ai.start_calls, [])
        self.assertEqual(self.ai.stop_calls, [])

    def test_active_capture_mode_change_restarts_ai_and_keeps_overlay(self):
        self.app.toggle_overlay()
        self.ai.emit(AiEvent("ready", "DmlExecutionProvider"))
        self.drain_ui_queue()
        starts = len(self.ai.start_calls)

        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()

        self.assertTrue(self.app.overlay_visible)
        self.assertEqual(self.ai.stop_calls[-1], "Capture mode changed")
        self.assertEqual(len(self.ai.start_calls), starts + 1)
        self.assertEqual(self.ai.start_calls[-1][3], FULL_DISPLAY)
        self.assertEqual(str(self.app.capture_mode_combo.cget("state")), "disabled")
        self.ai.emit(AiEvent("ready", "DmlExecutionProvider"))
        self.drain_ui_queue()
        self.assertEqual(str(self.app.capture_mode_combo.cget("state")), "readonly")

    def test_combined_motion_continues_during_capture_mode_restart(self):
        self.prepare_armed_sources(
            MotionSources(True, True), gate_active=True
        )
        self.app.toggle_overlay()
        motion_generation = self.app._expected_motion_generation
        cancellations = len(self.service.cancel_reasons)

        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()

        self.assertTrue(self.app.master_armed)
        self.assertEqual(
            self.app._selected_sources(), MotionSources(True, True)
        )
        self.assertTrue(self.app.trigger_gate.active)
        self.assertTrue(self.app.overlay_visible)
        self.assertEqual(
            self.app._expected_motion_generation, motion_generation
        )
        self.assertEqual(len(self.service.cancel_reasons), cancellations)
        self.assertTrue(self.app._normal_motion_started)

    def test_invalid_capture_mode_label_restores_current_selection(self):
        self.app.capture_mode_var.set("Unsupported")
        self.app._capture_mode_changed()

        self.assertEqual(self.app._capture_mode, CENTER_320)
        self.assertEqual(self.app.capture_mode_var.get(), "Center 320")
        self.assertEqual(self.ai.start_calls, [])
        self.assertEqual(self.ai.stop_calls, [])

    def test_model_validation_guards_capture_mode_lifecycle(self):
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        self.begin_custom_model_switch("validating.onnx")
        starts = list(self.ai.start_calls)
        stops = list(self.ai.stop_calls)

        self.assertEqual(
            str(self.app.capture_mode_combo.cget("state")), "disabled"
        )
        self.app.capture_mode_var.set("Center 320")
        self.app._capture_mode_changed()

        self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
        self.assertEqual(self.app.capture_mode_var.get(), "Full Display")
        self.assertEqual(self.ai.start_calls, starts)
        self.assertEqual(self.ai.stop_calls, stops)

    def test_every_test_motion_mode_guards_capture_mode_lifecycle(self):
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        test_modes = (
            "test_jitter_pending",
            "test_jitter",
            "test_ai_loading",
            "test_ai",
            "test_combined_loading",
            "test_combined",
        )
        for motion_mode in test_modes:
            with self.subTest(motion_mode=motion_mode):
                self.app._motion_mode = motion_mode
                self.app._render_runtime_controls()
                starts = list(self.ai.start_calls)
                stops = list(self.ai.stop_calls)

                self.assertEqual(
                    str(self.app.capture_mode_combo.cget("state")),
                    "disabled",
                )
                self.app.capture_mode_var.set("Center 320")
                self.app._capture_mode_changed()

                self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
                self.assertEqual(
                    self.app.capture_mode_var.get(), "Full Display"
                )
                self.assertEqual(self.ai.start_calls, starts)
                self.assertEqual(self.ai.stop_calls, stops)

    def test_loading_and_existing_restart_guard_capture_mode_lifecycle(self):
        self.app.toggle_overlay()
        starts = len(self.ai.start_calls)
        self.assertEqual(
            str(self.app.capture_mode_combo.cget("state")), "disabled"
        )

        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        self.assertEqual(self.app._capture_mode, CENTER_320)
        self.assertEqual(self.app.capture_mode_var.get(), "Center 320")
        self.assertEqual(len(self.ai.start_calls), starts)
        self.assertEqual(self.ai.stop_calls, [])

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.assertEqual(
            str(self.app.capture_mode_combo.cget("state")), "readonly"
        )
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        starts = len(self.ai.start_calls)
        stops = len(self.ai.stop_calls)

        self.app.capture_mode_var.set("Center 320")
        self.app._capture_mode_changed()
        self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
        self.assertEqual(self.app.capture_mode_var.get(), "Full Display")
        self.assertEqual(len(self.ai.start_calls), starts)
        self.assertEqual(len(self.ai.stop_calls), stops)

    def test_test_candidate_and_rollback_generations_keep_capture_mode(self):
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        self.service.connected = True
        self.app.ai_selected = True
        self.app.start_test_run()
        self.assertEqual(self.ai.start_calls[-1][3], FULL_DISPLAY)

        self.app.close_app()
        app = self.make_app()
        app.capture_mode_var.set("Full Display")
        app._capture_mode_changed()
        app.toggle_overlay()
        app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        choice, token = self.begin_custom_model_switch("candidate.onnx")
        validated = self.validated_model_choice(choice)
        self.model_validator.emit(
            ModelValidationEvent("ready", token, validated)
        )
        self.drain_model_ui_queue()
        self.assertEqual(self.ai.start_calls[-1][2], validated.path)
        self.assertEqual(self.ai.start_calls[-1][3], FULL_DISPLAY)

        app.handle_ai_event(AiEvent("error", "candidate failed"))
        self.assertEqual(
            self.ai.start_calls[-1][2], app._model_switch.previous.path
        )
        self.assertEqual(self.ai.start_calls[-1][3], FULL_DISPLAY)

    def test_stop_keeps_runtime_capture_mode_and_new_app_starts_centered(self):
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()
        self.app.emergency_stop("Stopped by user")

        self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
        self.assertEqual(self.app.capture_mode_var.get(), "Full Display")
        self.app.close_app()
        app = self.make_app(config=self.app.config)
        self.assertEqual(app._capture_mode, CENTER_320)

    def test_false_and_exception_capture_restarts_use_ai_failure_policy(self):
        for failure in (False, RuntimeError("restart failed")):
            with self.subTest(failure=type(failure).__name__):
                self.app.close_app()
                app = self.make_app()
                self.prepare_armed_sources(
                    MotionSources(True, True), gate_active=True
                )
                app.toggle_overlay()
                generation = app._expected_motion_generation
                cancellations = len(self.service.cancel_reasons)
                if isinstance(failure, Exception):
                    self.ai.start_exception = failure
                else:
                    self.ai.start_result = failure

                app.capture_mode_var.set("Full Display")
                with self.assertLogs(level="ERROR"):
                    app._capture_mode_changed()

                self.assertFalse(app.overlay_visible)
                self.assertFalse(app.ai_selected)
                self.assertTrue(app.jitter_selected)
                self.assertTrue(app.master_armed)
                self.assertTrue(app.trigger_gate.active)
                self.assertEqual(
                    app._expected_motion_generation, generation
                )
                self.assertEqual(
                    len(self.service.cancel_reasons), cancellations
                )
                self.assertEqual(app._capture_mode, FULL_DISPLAY)
                self.assertFalse(app._capture_mode_switching)

    def test_async_capture_restart_error_clears_switch_before_failure_policy(self):
        self.prepare_armed_sources(
            MotionSources(True, True), gate_active=True
        )
        self.app.toggle_overlay()
        self.app.capture_mode_var.set("Full Display")
        self.app._capture_mode_changed()

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(AiEvent("error", "capture failed"))

        self.assertFalse(self.app._capture_mode_switching)
        self.assertEqual(self.app._capture_mode, FULL_DISPLAY)
        self.assertFalse(self.app.overlay_visible)
        self.assertFalse(self.app.ai_selected)
        self.assertTrue(self.app.jitter_selected)
        self.assertTrue(self.app.master_armed)

    def test_stop_disconnect_and_shutdown_clear_capture_restart_state(self):
        for action in ("stop", "disconnect", "shutdown"):
            with self.subTest(action=action):
                self.app.close_app()
                app = self.make_app()
                app.toggle_overlay()
                app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
                app.capture_mode_var.set("Full Display")
                app._capture_mode_changed()
                self.assertTrue(app._capture_mode_switching)

                if action == "stop":
                    app.emergency_stop("Stopped by user")
                elif action == "disconnect":
                    app._handle_disconnect("Device disconnected")
                else:
                    app.close_app()

                self.assertFalse(app._capture_mode_switching)
                if action == "stop":
                    self.assertEqual(
                        str(app.capture_mode_combo.cget("state")),
                        "readonly",
                    )
                elif action == "disconnect":
                    app.handle_ai_event(
                        AiEvent("ready", "DmlExecutionProvider")
                    )
                    self.assertEqual(
                        str(app.capture_mode_combo.cget("state")),
                        "readonly",
                    )

    def test_stale_ai_events_cannot_mutate_capture_replacement(self):
        for stale_event in (
            AiEvent("ready", "CPUExecutionProvider"),
            AiEvent("error", "old capture failed"),
        ):
            with self.subTest(kind=stale_event.kind):
                self.app.close_app()
                app = self.make_app()
                app.toggle_overlay()
                app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
                app.queue_ai_event(stale_event)

                app.capture_mode_var.set("Full Display")
                app._capture_mode_changed()
                self.drain_ui_queue()

                self.assertTrue(app._capture_mode_switching)
                self.assertTrue(app._ai_runtime_active)
                self.assertTrue(app.overlay_visible)
                self.assertEqual(app._capture_mode, FULL_DISPLAY)
                self.assertEqual(app.ai_status_var.get(), "Loading")

    def test_each_capture_direction_makes_exactly_one_stop_and_start(self):
        self.app.toggle_overlay()
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        for label, mode in (
            ("Full Display", FULL_DISPLAY),
            ("Center 320", CENTER_320),
        ):
            with self.subTest(label=label):
                starts = len(self.ai.start_calls)
                stops = len(self.ai.stop_calls)
                self.app.capture_mode_var.set(label)
                self.app._capture_mode_changed()

                self.assertEqual(len(self.ai.start_calls), starts + 1)
                self.assertEqual(len(self.ai.stop_calls), stops + 1)
                self.assertEqual(self.ai.start_calls[-1][3], mode)
                self.app.handle_ai_event(
                    AiEvent("ready", "DmlExecutionProvider")
                )
                self.assertEqual(
                    self.app.footer_var.get(),
                    f"AI capture ready: {label}",
                )

    def test_motion_worker_survives_capture_restart_with_cleared_ai_target(self):
        for sources in (
            MotionSources(False, True),
            MotionSources(True, True),
        ):
            with self.subTest(sources=sources):
                self.app.close_app()
                app = self.make_app()
                self.prepare_armed_sources(sources, gate_active=True)
                call = self.service.composite_motion_calls[-1]
                generation = app._expected_motion_generation
                cancellations = len(self.service.cancel_reasons)

                app.capture_mode_var.set("Full Display")
                app._capture_mode_changed()

                self.assertIsNone(call.target_provider())
                self.assertEqual(
                    app._expected_motion_generation, generation
                )
                self.assertEqual(
                    len(self.service.cancel_reasons), cancellations
                )
                self.assertTrue(app._normal_motion_started)

    def test_save_config_keeps_target_area_runtime_only(self):
        self.app.target_area_var.set("Chest")
        self.app._target_area_changed()
        self.app._cancel_after("_save_after_id")

        self.app.save_config()

        saved = self.store.saved[-1]
        self.assertEqual(saved.ai.target_area, "head")
        self.assertEqual(self.app.get_ai_settings().target_area, "chest")

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
        self.app.open_overlay_customizer()
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
        self.app.open_overlay_customizer()
        self.app._color_chooser = lambda **_kwargs: (None, None)
        before = self.app.overlay_color
        button = self.app.overlay_color_button
        self.app._cancel_after("_save_after_id")
        button.invoke()

        self.assertEqual(self.app.overlay_color, before)
        self.assertIsNone(self.app._save_after_id)

    def test_overlay_color_chooser_error_keeps_current_choice(self):
        self.app.open_overlay_customizer()

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
        self.app.open_overlay_customizer()
        self.app._cancel_after("_save_after_id")
        button = self.app.overlay_head_button
        button.invoke()

        self.assertIs(
            self.app.overlay_head_visible, False
        )
        self.assertIsNotNone(self.app._save_after_id)
        self.assertEqual(button.cget("text"), "Head Boxes OFF")

    def test_player_boxes_button_is_runtime_only_and_reaches_overlay_style(self):
        self.app.open_overlay_customizer()
        self.assertTrue(
            hasattr(self.app, "overlay_player_button"),
            "Overlay customization must expose a Player Boxes control",
        )
        self.app._cancel_after("_save_after_id")

        self.app.overlay_player_button.invoke()
        style = self.app._overlay_style_snapshot()

        self.assertFalse(style.show_players)
        self.assertEqual(
            self.app.overlay_player_button.cget("text"),
            "Player Boxes OFF",
        )
        self.assertIsNone(self.app._save_after_id)

    def test_detection_custom_controls_publish_width_and_label_mode_runtime_only(self):
        self.app.open_overlay_customizer()
        self.assertTrue(hasattr(self.app, "overlay_box_width_entry"))
        self.assertTrue(hasattr(self.app, "overlay_box_width_scale"))
        self.assertTrue(hasattr(self.app, "overlay_label_mode_combo"))
        self.app._cancel_after("_save_after_id")
        self.app.overlay_box_width_var.set("20")
        self.app.overlay_label_mode_var.set("Class + Confidence")

        self.app._overlay_entry_changed("box_width")
        self.app._overlay_style_changed()
        style = self.app._overlay_style_snapshot()

        self.assertEqual(self.app.overlay_box_width_var.get(), "8")
        self.assertEqual(style.box_width, 8)
        self.assertEqual(style.label_mode, "class_confidence")
        self.assertIsNone(self.app._save_after_id)

    def test_hud_placement_controls_publish_corner_offsets_and_font_size(self):
        self.app.open_overlay_customizer()
        for name in (
            "overlay_hud_corner_combo",
            "overlay_hud_offset_x_entry",
            "overlay_hud_offset_x_scale",
            "overlay_hud_offset_y_entry",
            "overlay_hud_offset_y_scale",
            "overlay_hud_font_size_entry",
            "overlay_hud_font_size_scale",
        ):
            self.assertTrue(hasattr(self.app, name), name)
        self.app._cancel_after("_save_after_id")
        self.app.overlay_hud_corner_var.set("Bottom Right")
        self.app.overlay_hud_offset_x_var.set("40")
        self.app.overlay_hud_offset_y_var.set("50")
        self.app.overlay_hud_font_size_var.set("18")

        for key in ("hud_offset_x", "hud_offset_y", "hud_font_size"):
            self.app._overlay_entry_changed(key)
        self.app._overlay_style_changed()
        style = self.app._overlay_style_snapshot()

        self.assertEqual(style.hud_corner, "bottom_right")
        self.assertEqual(style.hud_offset_x, 40)
        self.assertEqual(style.hud_offset_y, 50)
        self.assertEqual(style.hud_font_size, 18)
        self.assertIsNone(self.app._save_after_id)

    def test_hud_visibility_color_and_metric_buttons_are_runtime_only(self):
        self.app.open_overlay_customizer()
        for name in (
            "overlay_hud_button",
            "overlay_hud_color_button",
            "overlay_hud_fps_button",
            "overlay_hud_provider_button",
            "overlay_hud_zoom_button",
            "overlay_hud_lock_button",
        ):
            self.assertTrue(hasattr(self.app, name), name)
        self.app._color_chooser = lambda **_kwargs: (
            (0.0, 204.0, 136.0),
            "#00CC88",
        )
        self.app._cancel_after("_save_after_id")

        self.app.overlay_hud_color_button.invoke()
        self.app.overlay_hud_button.invoke()
        self.app.overlay_hud_fps_button.invoke()
        self.app.overlay_hud_zoom_button.invoke()
        style = self.app._overlay_style_snapshot()

        self.assertFalse(style.hud_visible)
        self.assertEqual(style.hud_color, "#00cc88")
        self.assertFalse(style.hud_show_fps)
        self.assertTrue(style.hud_show_provider)
        self.assertFalse(style.hud_show_zoom)
        self.assertTrue(style.hud_show_lock)
        self.assertEqual(style.box_color, "#ff2b2b")
        self.assertIsNone(self.app._save_after_id)

    def test_reset_overlay_restores_complete_default_and_saves_only_legacy_preferences(self):
        self.app.open_overlay_customizer()
        self.assertTrue(hasattr(self.app, "overlay_reset_button"))
        self.app.overlay_color = "#00cc88"
        self.app.overlay_head_visible = False
        self.app.overlay_player_visible = False
        self.app.overlay_box_width_var.set("8")
        self.app.overlay_label_mode_var.set("Class + Confidence")
        self.app.overlay_hud_visible = False
        self.app.overlay_hud_corner_var.set("Bottom Right")
        self.app.overlay_hud_offset_x_var.set("40")
        self.app.overlay_hud_offset_y_var.set("50")
        self.app.overlay_hud_font_size_var.set("18")
        self.app.overlay_hud_color = "#123456"
        self.app.overlay_hud_show_fps = False
        self.app.overlay_hud_show_provider = False
        self.app.overlay_hud_show_zoom = False
        self.app.overlay_hud_show_lock = False
        self.app._cancel_after("_save_after_id")

        self.app.overlay_reset_button.invoke()

        self.assertEqual(self.app._overlay_style_snapshot(), OverlayStyle())
        self.assertIsNotNone(self.app._save_after_id)

    def test_overlay_style_snapshot_uses_safe_defaults_while_exact_input_is_incomplete(self):
        self.app.overlay_box_width_var.set("")
        self.app.overlay_hud_offset_x_var.set("-")
        self.app.overlay_hud_offset_y_var.set("not a number")
        self.app.overlay_hud_font_size_var.set("")

        try:
            style = self.app._overlay_style_snapshot()
        except ValueError as exc:
            self.fail(f"Incomplete exact input must not stop the overlay: {exc}")

        self.assertEqual(style.box_width, 2)
        self.assertEqual(style.hud_offset_x, 8)
        self.assertEqual(style.hud_offset_y, 8)
        self.assertEqual(style.hud_font_size, 10)

    def test_ai_service_is_injected_after_widgets_without_autostart(self):
        self.assertIs(self.ai.event_sink.__self__, self.app)
        self.assertEqual(self.ai.event_sink.__func__, self.app.queue_ai_event.__func__)
        self.assertEqual(self.ai.start_calls, [])

    def test_model_row_starts_with_bundled_default_and_keeps_fixed_shell(self):
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        self.assertEqual(
            str(self.app.use_default_model_button.cget("state")), "disabled"
        )
        self.app.update_idletasks()
        self.assertEqual(self.app.geometry().split("+")[0], "840x620")
        self.assertEqual(self.app.stop_button.winfo_manager(), "grid")

    def test_browse_cancel_does_not_change_model_or_schedule_save(self):
        self.app._cancel_after("_save_after_id")

        self.app.model_browse_button.invoke()

        self.assertEqual(self.model_validator.start_calls, [])
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        self.assertIsNone(self.app._save_after_id)

    def test_model_label_shows_only_validated_input_size(self):
        self.assertEqual(
            self.app._model_label(bundled_model_choice()),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        pending = ModelChoice(Path("custom.onnx"), "custom.onnx", False)
        validated = ModelChoice(
            Path("custom.onnx"), "custom.onnx", False, 640
        )
        self.assertEqual(
            self.app._model_label(pending), "Custom \u00b7 custom.onnx"
        )
        self.assertEqual(
            self.app._model_label(validated),
            "Custom \u00b7 custom.onnx \u00b7 640\u00d7640",
        )

    def test_ready_event_replaces_pending_candidate_with_validated_choice(self):
        pending, token = self.begin_custom_model_switch("custom.onnx")
        validated = replace(pending, input_size=160)

        self.model_validator.emit(
            ModelValidationEvent("ready", token, validated)
        )
        self.drain_model_ui_queue()

        self.assertEqual(self.app._model_choice, validated)
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Custom \u00b7 custom.onnx \u00b7 160\u00d7160",
        )

    def test_ready_event_with_same_token_but_different_path_is_ignored(self):
        pending, token = self.begin_custom_model_switch("expected.onnx")
        wrong = ModelChoice(Path("other.onnx"), "other.onnx", False, 640)

        self.model_validator.emit(ModelValidationEvent("ready", token, wrong))
        self.drain_model_ui_queue()

        self.assertEqual(self.app._model_switch.candidate, pending)
        self.assertEqual(
            self.app.ai_model_var.get(), "Loading \u00b7 expected.onnx"
        )

    def test_ready_event_without_validated_input_size_is_ignored(self):
        self.app.toggle_overlay()
        previous = self.app._model_choice
        pending, token = self.begin_custom_model_switch("pending.onnx")
        starts = len(self.ai.start_calls)

        self.model_validator.emit(ModelValidationEvent("ready", token, pending))
        self.drain_model_ui_queue()

        self.assertEqual(self.app._model_switch.candidate, pending)
        self.assertEqual(self.app._model_switch.phase, "validating")
        self.assertEqual(self.app._model_choice, previous)
        self.assertEqual(
            self.app.ai_model_var.get(), "Loading \u00b7 pending.onnx"
        )
        self.assertEqual(len(self.ai.start_calls), starts)

    def test_invalid_input_size_footer_is_actionable_without_path_leak(self):
        pending, token = self.begin_custom_model_switch("private-name.onnx")

        self.model_validator.emit(ModelValidationEvent(
            "error",
            token,
            pending,
            "ModelContractError",
            "AI model input must use a 160, 320, or 640 square input",
        ))
        self.drain_model_ui_queue()

        footer = self.app.footer_var.get()
        self.assertIn("160, 320, or 640", footer)
        self.assertNotIn(str(pending.path.parent), footer)

    def test_idle_candidate_commits_after_matching_validation_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.onnx"
            path.write_bytes(b"model")
            self.model_dialog_result = str(path)
            self.app.model_browse_button.invoke()
            choice, token = self.model_validator.start_calls[-1]
            self.assertEqual(self.app.ai_model_var.get(), "Loading \u00b7 custom.onnx")
            self.assertEqual(
                str(self.app.model_browse_button.cget("state")), "disabled"
            )
            validated = self.validated_model_choice(choice, 160)
            self.model_validator.emit(
                ModelValidationEvent("ready", token, validated)
            )
            self.drain_model_ui_queue()

        self.assertEqual(
            self.app.ai_model_var.get(),
            "Custom \u00b7 custom.onnx \u00b7 160\u00d7160",
        )
        self.assertEqual(self.ai.start_calls, [])
        self.assertIsNone(self.app._save_after_id)

    def test_use_default_runs_the_same_validation_flow_without_saving(self):
        custom, custom_token = self.begin_custom_model_switch("custom.onnx")
        self.model_validator.emit(
            ModelValidationEvent(
                "ready",
                custom_token,
                self.validated_model_choice(custom, 640),
            )
        )
        self.drain_model_ui_queue()
        self.app._cancel_after("_save_after_id")

        self.app.use_default_model_button.invoke()
        default, default_token = self.model_validator.start_calls[-1]
        self.assertTrue(default.is_default)
        self.assertEqual(
            self.app.ai_model_var.get(), "Loading \u00b7 all_games_320.onnx"
        )
        self.model_validator.emit(
            ModelValidationEvent("ready", default_token, default)
        )
        self.drain_model_ui_queue()

        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        self.assertIsNone(self.app._save_after_id)

    def test_invalid_model_selection_keeps_previous_choice_without_path_leak(self):
        for filename, expected_footer in (
            ("wrong.txt", "Select an ONNX model file"),
            ("missing.onnx", "Selected model file was not found"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / filename
                if filename == "wrong.txt":
                    path.write_bytes(b"not a model")
                self.model_dialog_result = str(path)
                with self.assertLogs(level="ERROR") as logs:
                    self.app.browse_ai_model()

                self.assertEqual(
                    self.app.ai_model_var.get(),
                    "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
                )
                self.assertEqual(self.ai.start_calls, [])
                self.assertEqual(self.app.footer_var.get(), expected_footer)
                self.assertNotIn(str(path), self.app.footer_var.get())
                self.assertIn(str(path), "\n".join(logs.output))

    def test_model_switch_is_unavailable_while_test_run_is_active(self):
        self.app._motion_mode = "test_jitter"
        self.app._render_model_controls()

        self.app.browse_ai_model()

        self.assertEqual(self.model_validator.start_calls, [])
        self.assertEqual(
            self.app.footer_var.get(), "Test Run is active; use STOP to cancel"
        )

    def test_single_page_shell_orders_fixed_chrome_around_one_scroll_region(self):
        widgets = (
            self.app.topbar_frame,
            self.app.dashboard_frame,
            self.app.footer_frame,
            self.app.runtime_frame,
        )
        self.assertEqual(
            [int(widget.grid_info()["row"]) for widget in widgets],
            [0, 1, 2, 3],
        )
        self.assertTrue(
            all(widget.master is self.app.console_workspace for widget in widgets)
        )
        self.assertIsInstance(self.app.dashboard_scroll_canvas, tk.Canvas)
        self.assertFalse(hasattr(self.app, "nav"))
        self.assertFalse(hasattr(self.app, "pages"))

    def test_dashboard_has_five_ordered_independent_sections(self):
        self.assertEqual(
            tuple(section.title for section in self.app.sections),
            ("CONTROL", "JITTER", "AI AIM", "OVERLAY", "SETTINGS"),
        )
        self.assertTrue(
            all(isinstance(section, CollapsibleSection) for section in self.app.sections)
        )
        self.assertTrue(self.app.control_section.expanded)
        self.assertTrue(all(not section.expanded for section in self.app.sections[1:]))
        self.app.ai_section.set_expanded(True)
        self.app.overlay_section.set_expanded(True)
        self.assertTrue(self.app.control_section.expanded)
        self.assertTrue(self.app.ai_section.expanded)
        self.assertTrue(self.app.overlay_section.expanded)

    def test_overlay_section_opens_custom_controls_in_separate_window(self):
        self.assertTrue(
            hasattr(self.app, "overlay_customize_button"),
            "Overlay section must expose the separate customizer action",
        )
        self.assertEqual(
            self.app.overlay_customize_button.cget("text"),
            "Customize Overlay",
        )
        self.assertTrue(
            self._is_descendant(
                self.app.overlay_customize_button,
                self.app.overlay_section.body,
            )
        )

        self.app.deiconify()
        self.app.overlay_customize_button.invoke()
        self.app.update()

        customizer = self.app.overlay_custom_window
        self.assertIsInstance(customizer, tk.Toplevel)
        self.assertEqual(customizer.title(), "Customize Overlay")
        self.assertTrue(customizer.winfo_viewable())
        self.assertTrue(
            self._is_descendant(self.app.overlay_color_button, customizer)
        )
        self.assertFalse(
            self._is_descendant(
                self.app.overlay_color_button,
                self.app.dashboard_content,
            )
        )

    def test_opening_overlay_customizer_twice_reuses_one_window(self):
        self.app.deiconify()
        self.app.open_overlay_customizer()
        first = self.app.overlay_custom_window

        self.app.open_overlay_customizer()
        self.app.update()

        self.assertIs(self.app.overlay_custom_window, first)
        self.assertEqual(
            [
                child
                for child in self.app.winfo_children()
                if isinstance(child, tk.Toplevel)
                and child.title() == "Customize Overlay"
            ],
            [first],
        )

    def test_overlay_customizer_keeps_all_controls_inside_fixed_window(self):
        self.app.deiconify()
        self.app.open_overlay_customizer()
        self.app.update()
        customizer = self.app.overlay_custom_window
        controls = (
            self.app.overlay_color_button,
            self.app.overlay_head_button,
            self.app.overlay_player_button,
            self.app.overlay_box_width_scale,
            self.app.overlay_label_mode_combo,
            self.app.overlay_hud_corner_combo,
            self.app.overlay_hud_offset_x_scale,
            self.app.overlay_hud_offset_y_scale,
            self.app.overlay_hud_font_size_scale,
            self.app.overlay_hud_button,
            self.app.overlay_hud_color_button,
            self.app.overlay_hud_fps_button,
            self.app.overlay_hud_provider_button,
            self.app.overlay_hud_zoom_button,
            self.app.overlay_hud_lock_button,
        )
        left = customizer.winfo_rootx()
        top = customizer.winfo_rooty()
        right = left + customizer.winfo_width()
        bottom = top + customizer.winfo_height()

        for control in controls:
            with self.subTest(control=control):
                self.assertGreaterEqual(control.winfo_rootx(), left)
                self.assertGreaterEqual(control.winfo_rooty(), top)
                self.assertLessEqual(
                    control.winfo_rootx() + control.winfo_width(),
                    right,
                )
                self.assertLessEqual(
                    control.winfo_rooty() + control.winfo_height(),
                    bottom,
                )

    def test_overlay_customizer_keeps_action_labels_unclipped(self):
        self.app.deiconify()
        self.app.open_overlay_customizer()
        self.app.update()

        for button in (
            self.app.overlay_color_button,
            self.app.overlay_head_button,
            self.app.overlay_player_button,
            self.app.overlay_hud_button,
            self.app.overlay_hud_color_button,
        ):
            with self.subTest(button=button):
                self.assertGreaterEqual(
                    button.winfo_width(),
                    button.winfo_reqwidth(),
                )

    def test_closing_overlay_customizer_keeps_detection_overlay_enabled(self):
        self.app.toggle_overlay()
        self.app.deiconify()
        self.app.open_overlay_customizer()
        customizer = self.app.overlay_custom_window

        self.app.close_overlay_customizer()
        self.app.update()

        self.assertTrue(self.app.overlay_visible)
        self.assertFalse(
            customizer.winfo_exists() and customizer.winfo_viewable()
        )

    def test_reopening_overlay_customizer_normalizes_incomplete_numeric_edits(self):
        self.app.deiconify()
        self.app.open_overlay_customizer()
        self.app.overlay_box_width_var.set("")
        self.app.overlay_hud_offset_x_var.set("-")
        self.app.overlay_hud_offset_y_var.set("9999")
        self.app.overlay_hud_font_size_var.set("0")
        self.app.close_overlay_customizer()

        try:
            self.app.open_overlay_customizer()
        except (tk.TclError, TypeError, ValueError) as exc:
            self.fail(f"reopening the customizer raised {type(exc).__name__}: {exc}")
        self.app.update()

        self.assertEqual(
            (
                self.app.overlay_box_width_var.get(),
                self.app.overlay_hud_offset_x_var.get(),
                self.app.overlay_hud_offset_y_var.get(),
                self.app.overlay_hud_font_size_var.get(),
            ),
            ("2", "8", "500", "8"),
        )
        self.assertTrue(self.app.overlay_custom_window.winfo_viewable())

    def test_open_overlay_customizer_cleans_up_failed_window_build(self):
        self.app.deiconify()
        with mock.patch.object(
            self.app,
            "_dropdown_field",
            side_effect=tk.TclError("partial build failed"),
        ):
            with self.assertRaisesRegex(tk.TclError, "partial build failed"):
                self.app.open_overlay_customizer()

        try:
            self.assertIsNone(self.app.overlay_custom_window)
            self.assertFalse(
                any(
                    isinstance(child, tk.Toplevel)
                    and child.title() == "Customize Overlay"
                    for child in self.app.winfo_children()
                )
            )
        finally:
            leaked = self.app.overlay_custom_window
            self.app.overlay_custom_window = None
            if leaked is not None and leaked.winfo_exists():
                leaked.destroy()

        try:
            self.app.toggle_theme()
        except tk.TclError as exc:
            self.fail(f"theme toggle used a stale partial-build slider: {exc}")

    def test_theme_toggle_after_closing_overlay_customizer_ignores_stale_sliders(self):
        self.app.deiconify()
        self.app.open_overlay_customizer()
        self.app.close_overlay_customizer()
        before = self.app.theme_var.get()

        try:
            self.app.toggle_theme()
        except tk.TclError as exc:
            self.fail(f"theme toggle used a stale customizer slider: {exc}")

        self.assertNotEqual(self.app.theme_var.get(), before)

    def test_overlay_customizer_native_background_tracks_theme(self):
        self.app.deiconify()
        self.app.open_overlay_customizer()
        customizer = self.app.overlay_custom_window
        before = str(customizer.cget("background"))

        self.app.toggle_theme()
        self.app.update()

        self.assertEqual(
            str(customizer.cget("background")),
            self.app._palette["window"],
        )
        self.assertEqual(
            str(self.app.overlay_box_width_scale.cget("background")),
            self.app._palette["surface"],
        )
        self.assertNotEqual(str(customizer.cget("background")), before)

    def test_dashboard_uses_one_vertical_scrollbar_and_no_motion_scroll(self):
        vertical = [
            widget
            for widget in descendant_widgets(self.app.console_workspace)
            if isinstance(widget, ttk.Scrollbar)
            and str(widget.cget("orient")) == "vertical"
        ]
        self.assertEqual(vertical, [self.app.dashboard_scrollbar])
        self.assertFalse(hasattr(self.app, "motion_scrollbar"))

    def test_single_page_shell_keeps_semantic_layers_in_both_themes(self):
        shell = self.app.shell
        self.app.deiconify()
        self.app.update()
        required_tags = (
            "workspace-band", "rounded-surface", "floating-panel",
            "floating-panel-topbar", "floating-panel-dashboard",
            "floating-panel-runtime",
        )
        themed_layers = {}
        for theme in ("light", "dark"):
            with self.subTest(theme=theme):
                self.assertEqual(self.app.theme_var.get(), theme)
                for tag in required_tags:
                    self.assertTrue(shell.find_withtag(tag), tag)
                fills = tuple(
                    shell.itemcget(item, "fill")
                    for item in shell.find_withtag("workspace-band")
                )
                themed_layers[theme] = fills
            self.app.toggle_theme()
            self.app.update()
        self.assertNotEqual(themed_layers["light"], themed_layers["dark"])

    def test_control_section_keeps_three_to_two_card_layout(self):
        self.assertEqual(
            tuple(
                int(self.app.control_section.body.grid_columnconfigure(column)["weight"])
                for column in (0, 1)
            ),
            (3, 2),
        )
        self.assertEqual(int(self.app.control_bindings_card.grid_info()["column"]), 0)
        self.assertEqual(int(self.app.control_device_card.grid_info()["column"]), 1)

    def test_settings_section_keeps_three_to_two_card_layout(self):
        self.assertIs(self.app.settings_content, self.app.settings_section.body)
        self.assertEqual(
            tuple(
                int(self.app.settings_content.grid_columnconfigure(column)["weight"])
                for column in (0, 1)
            ),
            (3, 2),
        )
        self.assertIs(self.app.theme_button.master, self.app.test_on_button.master)

    def test_jitter_section_keeps_compact_snapshot_summary(self):
        self.assertIs(self.app.jitter_section.summary, self.app.motion_summary_var)
        self.assertEqual(
            self.app.motion_summary_var.get(),
            "2 px paired pulse at 60 Hz | Smooth",
        )

    def test_collapsed_sections_keep_compact_headers_at_fixed_window_size(self):
        self.app.jitter_section.set_expanded(True)
        self.app.deiconify()
        self.app.update()
        self.assertLessEqual(
            self.app.jitter_section.header_button.winfo_reqheight(),
            self.app.jitter_section.winfo_height(),
        )
        self.assertNotIn("LIVE SNAPSHOT", widget_texts(self.app.jitter_section.body))

    def test_collapsed_section_summaries_follow_validated_live_state(self):
        self.assertIn("No sources", self.app.control_section_summary_var.get())
        self.assertEqual(
            self.app.motion_summary_var.get(),
            "2 px paired pulse at 60 Hz | Smooth",
        )
        self.assertIn("Head", self.app.ai_section_summary_var.get())
        self.assertIn("Center 320", self.app.ai_section_summary_var.get())
        self.assertIn("Overlay Off", self.app.overlay_section_summary_var.get())
        self.assertIn("Sound On", self.app.settings_section_summary_var.get())

        self.app.toggle_jitter_source()
        self.app.pulse_size_px_var.set("4")
        self.app._motion_changed("pulse_size_px")
        self.app.overlay_player_visible = False
        self.app._render_runtime_controls()
        self.app.sound_volume_var.set("45")
        self.app.apply_sound_settings()

        self.assertIn("Jitter", self.app.control_section_summary_var.get())
        self.assertIn("4 px", self.app.jitter_section.summary.get())
        self.assertIn("Head", self.app.overlay_section_summary_var.get())
        self.assertNotIn("Player", self.app.overlay_section_summary_var.get())
        self.assertIn("45%", self.app.settings_section_summary_var.get())

    def test_invalid_ai_text_does_not_replace_valid_ai_summary(self):
        before = self.app.ai_section_summary_var.get()
        self.app.ai_vars["aim_strength"].set("invalid")
        self.app._ai_changed("aim_strength")
        self.assertEqual(self.app.ai_section_summary_var.get(), before)

    def test_ai_model_summary_get_failure_does_not_skip_targeting_reset(self):
        """Fails if a summary getter aborts target-change runtime work."""
        resets = self.ai.reset_targeting_calls
        with mock.patch.object(
            self.app.ai_model_var,
            "get",
            side_effect=tk.TclError("summary getter failed"),
        ):
            self.app.target_area_var.set("Upper Body")
            self.app._target_area_changed()

        self.assertEqual(self.ai.reset_targeting_calls, resets + 1)
        self.assertEqual(self.app.get_ai_settings().target_area, "upper_body")

    def test_long_ai_model_summary_keeps_target_and_strength_visible(self):
        self.app.ai_model_var.set("Custom · " + ("long-model-name-" * 12) + ".onnx")
        self.app._refresh_section_summaries()

        summary = self.app.ai_section_summary_var.get()
        self.assertLessEqual(len(summary), 72)
        self.assertIn("Head", summary)
        self.assertIn("Strength", summary)
        self.assertIn("Center 320", summary)

    def test_main_dashboard_excludes_ai_runtime_readouts(self):
        visible = set(widget_texts(self.app.dashboard_content))
        self.assertNotIn("AI RUNTIME", visible)
        self.assertNotIn(self.app.ai_fps_var.get(), visible)
        self.assertNotIn(self.app.ai_provider_var.get(), visible)
        self.assertNotIn(self.app.ai_cadence_var.get(), visible)

    def test_theme_action_exists_once_inside_settings(self):
        self.assertTrue(
            self._is_descendant(self.app.theme_button, self.app.settings_section.body)
        )
        self.assertFalse(self._is_descendant(self.app.theme_button, self.app.topbar_frame))
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Dark Mode")
        self.app.theme_button.invoke()
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Light Mode")
        self.assertIn("Dark", self.app.settings_section_summary_var.get())

    def test_session_actions_and_sound_previews_use_compact_labels(self):
        self.assertEqual(self.app.reconnect_button.cget("text"), "Reconnect")
        self.assertEqual(self.app.test_button.cget("text"), "Test 3s")
        self.assertEqual(self.app.test_on_button.cget("text"), "Play Armed Cue")
        self.assertEqual(self.app.test_off_button.cget("text"), "Play Disabled Cue")

    def test_numeric_controls_still_pair_slider_and_exact_entry(self):
        self.app.open_overlay_customizer()
        pairs = (
            (self.app.pulse_size_px_scale, self.app.pulse_size_px_entry),
            (self.app.pulse_rate_hz_scale, self.app.pulse_rate_hz_entry),
            (self.app.ai_confidence_scale, self.app.ai_confidence_entry),
            (self.app.ai_aim_strength_scale, self.app.ai_aim_strength_entry),
            (self.app.ai_smoothing_scale, self.app.ai_smoothing_entry),
            (self.app.ai_max_step_scale, self.app.ai_max_step_entry),
            (self.app.overlay_box_width_scale, self.app.overlay_box_width_entry),
            (self.app.overlay_hud_offset_x_scale, self.app.overlay_hud_offset_x_entry),
            (self.app.overlay_hud_offset_y_scale, self.app.overlay_hud_offset_y_entry),
            (self.app.overlay_hud_font_size_scale, self.app.overlay_hud_font_size_entry),
            (self.app.sound_volume_scale, self.app.sound_volume_entry),
        )
        for slider, entry in pairs:
            self.assertIsInstance(slider, LiquidSlider)
            self.assertIsInstance(entry, ttk.Entry)
            self.assertEqual(int(entry.cget("width")), 5)

    def test_fixed_runtime_dock_keeps_stop_visible_at_bottom_of_dashboard(self):
        self.app.deiconify()
        for section in self.app.sections:
            section.set_expanded(True)
        self.app.update()
        self.app.dashboard_scroll_canvas.yview_moveto(1.0)
        self.app.update()
        self.assertTrue(self.app.footer_frame.winfo_ismapped())
        self.assertTrue(self.app.runtime_frame.winfo_ismapped())
        self.assertTrue(self.app.stop_button.winfo_ismapped())
        self.assertIs(self.app.stop_button.master, self.app.runtime_frame)
        self.assertLessEqual(
            self.app.stop_button.winfo_rooty() + self.app.stop_button.winfo_height(),
            self.app.winfo_rooty() + self.app.winfo_height(),
        )

    def test_mouse_wheel_on_dashboard_control_scrolls_shared_canvas(self):
        self.app.deiconify()
        for section in self.app.sections:
            section.set_expanded(True)
        self.app.update()
        self.app.dashboard_scroll_canvas.yview_moveto(0.0)
        before = self.app.dashboard_scroll_canvas.yview()

        self.app.pulse_size_px_entry.event_generate("<MouseWheel>", delta=-120)
        self.app.update()

        self.assertGreater(self.app.dashboard_scroll_canvas.yview()[0], before[0])

    def test_mouse_wheel_on_fixed_runtime_control_does_not_scroll_dashboard(self):
        self.app.deiconify()
        for section in self.app.sections:
            section.set_expanded(True)
        self.app.update()
        self.app.dashboard_scroll_canvas.yview_moveto(0.0)
        before = self.app.dashboard_scroll_canvas.yview()

        self.app.stop_button.event_generate("<MouseWheel>", delta=-120)
        self.app.update()

        self.assertEqual(self.app.dashboard_scroll_canvas.yview(), before)

    def test_each_existing_control_belongs_to_its_approved_section(self):
        self.app.open_overlay_customizer()
        ownership = {
            self.app.control_section.body: (
                self.app.jitter_source_button, self.app.ai_source_button,
                self.app.trigger_combo, self.app.modifier_combo,
                self.app.hotkey_button, self.app.preset_combo,
                self.app.device_label, self.app.reconnect_button,
                self.app.test_button,
            ),
            self.app.jitter_section.body: (
                self.app.pulse_size_px_scale, self.app.pulse_size_px_entry,
                self.app.pulse_rate_hz_scale, self.app.pulse_rate_hz_entry,
                self.app.ramp_mode_combo,
            ),
            self.app.ai_section.body: (
                self.app.ai_confidence_scale, self.app.ai_confidence_entry,
                self.app.ai_aim_strength_scale, self.app.ai_aim_strength_entry,
                self.app.ai_smoothing_scale, self.app.ai_smoothing_entry,
                self.app.ai_max_step_scale, self.app.ai_max_step_entry,
                self.app.target_area_combo, self.app.capture_mode_combo,
                self.app.model_browse_button,
                self.app.use_default_model_button, self.app.ai_curve_canvas,
                self.app.ai_curve_reset_button,
            ),
            self.app.overlay_section.body: (
                self.app.overlay_button, self.app.overlay_customize_button,
            ),
            self.app.settings_section.body: (
                self.app.sound_enabled_check, self.app.sound_volume_scale,
                self.app.sound_volume_entry, self.app.test_on_button,
                self.app.test_off_button, self.app.theme_button,
            ),
        }
        for section_body, widgets in ownership.items():
            for widget in widgets:
                with self.subTest(section=section_body, widget=widget):
                    self.assertTrue(self._is_descendant(widget, section_body))

        customizer = self.app.overlay_custom_window
        custom_controls = (
            self.app.overlay_reset_button,
            self.app.overlay_color_button,
            self.app.overlay_head_button,
            self.app.overlay_player_button,
            self.app.overlay_box_width_scale,
            self.app.overlay_box_width_entry,
            self.app.overlay_label_mode_combo,
            self.app.overlay_hud_button,
            self.app.overlay_hud_color_button,
            self.app.overlay_hud_corner_combo,
            self.app.overlay_hud_offset_x_scale,
            self.app.overlay_hud_offset_y_scale,
            self.app.overlay_hud_font_size_scale,
        )
        for widget in custom_controls:
            with self.subTest(widget=widget):
                self.assertTrue(self._is_descendant(widget, customizer))
                self.assertFalse(
                    self._is_descendant(widget, self.app.dashboard_content)
                )

    def test_curve_exact_edit_updates_live_snapshot_and_schedules_save(self):
        self.app._cancel_after("_save_after_id")
        self.app.ai_curve_vars[2].set("42")
        self.app._curve_entry_changed(2)

        self.assertEqual(self.app.get_ai_settings().response_curve[2], 0.42)
        self.assertIsNotNone(self.app._save_after_id)

    def test_curve_edit_preserves_last_valid_scalars_when_scalar_text_is_invalid(self):
        self.app.close_app()
        initial = AimSettings(0.5, 0.6, 0.7, 30)
        app = self.make_app(config=AppConfig(ai=initial))
        app.ai_vars["confidence"].set("invalid")
        self.assertIs(app.get_ai_settings(), initial)

        app.ai_curve_vars[2].set("42")
        app._curve_entry_changed(2)

        self.assertEqual(
            app.get_ai_settings(),
            AimSettings(0.5, 0.6, 0.7, 30, (0.0, 0.12, 0.42, 0.68, 1.0)),
        )

    def test_scalar_edit_preserves_fractional_curve_from_last_valid_snapshot(self):
        self.app.close_app()
        curve = (0.0, 0.123, 0.357, 0.689, 0.997)
        app = self.make_app(config=AppConfig(
            ai=AimSettings(response_curve=curve),
        ))

        app.ai_vars["aim_strength"].set("0.5")

        self.assertEqual(app.get_ai_settings().aim_strength, 0.5)
        self.assertEqual(app.get_ai_settings().response_curve, curve)

    def test_reset_curve_restores_default(self):
        self.app.ai_curve_vars[1].set("20")
        self.app._curve_entry_changed(1)

        self.app.ai_curve_reset_button.invoke()

        self.assertEqual(
            self.app.get_ai_settings().response_curve,
            DEFAULT_RESPONSE_CURVE,
        )

    def test_curve_uses_four_exact_entries_and_a_fixed_zero_node(self):
        self.assertNotIn("response_curve", self.app.ai_vars)
        self.assertEqual(
            {index: variable.get() for index, variable in self.app.ai_curve_vars.items()},
            {1: "12", 2: "35", 3: "68", 4: "100"},
        )
        self.assertEqual(set(self.app.ai_curve_entries), {1, 2, 3, 4})
        self.assertTrue(self.app.ai_curve_canvas.find_withtag("ai-curve-node-0"))
        self.assertEqual(
            self.app.ai_curve_canvas.tag_bind("ai-curve-node-0", "<ButtonPress-1>"),
            "",
        )
        self.assertTrue(
            self.app.ai_curve_canvas.tag_bind(
                "ai-curve-node-1", "<ButtonPress-1>"
            )
        )
        sampled_coords = self.app.ai_curve_canvas.coords("ai-curve-sample")
        self.assertGreater(len(sampled_coords), 10)

    def test_response_curve_restores_from_config(self):
        self.app.close_app()
        curve = (0.0, 0.2, 0.4, 0.8, 0.9)
        app = self.make_app(config=AppConfig(
            ai=AimSettings(response_curve=curve),
        ))

        self.assertEqual(app.get_ai_settings().response_curve, curve)
        self.assertEqual(
            {index: variable.get() for index, variable in app.ai_curve_vars.items()},
            {1: "20", 2: "40", 3: "80", 4: "90"},
        )

    def test_curve_accepts_exact_zero_and_hundred_percent_boundaries(self):
        self.app.ai_curve_vars[1].set("0")
        self.app._curve_entry_changed(1)
        self.app.ai_curve_vars[4].set("100")
        self.app._curve_entry_changed(4)

        self.assertEqual(
            self.app.get_ai_settings().response_curve,
            (0.0, 0.0, 0.35, 0.68, 1.0),
        )
        self.assertEqual(
            self.app.ai_curve_entries[1].cget("style"),
            "Liquid.Entry.TEntry",
        )
        self.assertEqual(
            self.app.ai_curve_entries[4].cget("style"),
            "Liquid.Entry.TEntry",
        )

    def test_invalid_curve_edits_style_only_affected_entry_without_mutation(self):
        original = self.app.get_ai_settings()
        for raw in ("not-a-number", "-1", "101", "5"):
            with self.subTest(raw=raw):
                self.app._cancel_after("_save_after_id")
                self.app.ai_curve_vars[2].set(raw)
                self.app._curve_entry_changed(2)

                self.assertIs(self.app.get_ai_settings(), original)
                self.assertIsNone(self.app._save_after_id)
                self.assertEqual(
                    self.app.ai_curve_entries[2].cget("style"),
                    "Liquid.Invalid.TEntry",
                )
                for index in (1, 3, 4):
                    self.assertEqual(
                        self.app.ai_curve_entries[index].cget("style"),
                        "Liquid.Entry.TEntry",
                    )
                self.assertIn("response curve", self.app.footer_var.get().lower())

    def test_later_curve_interaction_keeps_actual_invalid_entry_styled(self):
        original = self.app.get_ai_settings()
        self.app.ai_curve_vars[2].set("invalid")
        self.app._curve_entry_changed(2)
        self.assertEqual(
            self.app.ai_curve_entries[2].cget("style"),
            "Liquid.Invalid.TEntry",
        )

        self.app.ai_curve_vars[3].set("70")
        self.app._curve_entry_changed(3)

        self.assertIs(self.app.get_ai_settings(), original)
        self.assertEqual(
            self.app.ai_curve_entries[2].cget("style"),
            "Liquid.Invalid.TEntry",
        )
        self.assertEqual(
            self.app.ai_curve_entries[3].cget("style"),
            "Liquid.Entry.TEntry",
        )

    def test_curve_order_error_owner_survives_unrelated_edit_until_corrected(self):
        original = self.app.get_ai_settings()
        self.app._cancel_after("_save_after_id")
        self.app.ai_curve_vars[1].set("50")
        self.app._curve_entry_changed(1)

        self.assertIs(self.app.get_ai_settings(), original)
        self.assertIsNone(self.app._save_after_id)
        self.assertEqual(
            self.app.ai_curve_entries[1].cget("style"),
            "Liquid.Invalid.TEntry",
        )
        self.assertEqual(
            self.app.ai_curve_entries[2].cget("style"),
            "Liquid.Entry.TEntry",
        )

        self.app.ai_curve_vars[3].set("70")
        self.app._curve_entry_changed(3)

        self.assertIs(self.app.get_ai_settings(), original)
        self.assertIsNone(self.app._save_after_id)
        self.assertEqual(
            self.app.ai_curve_entries[1].cget("style"),
            "Liquid.Invalid.TEntry",
        )
        for index in (2, 3, 4):
            self.assertEqual(
                self.app.ai_curve_entries[index].cget("style"),
                "Liquid.Entry.TEntry",
            )

        self.app.ai_curve_vars[1].set("30")
        self.app._curve_entry_changed(1)

        self.assertEqual(
            self.app.get_ai_settings().response_curve,
            (0.0, 0.3, 0.35, 0.7, 1.0),
        )
        for entry in self.app.ai_curve_entries.values():
            self.assertEqual(entry.cget("style"), "Liquid.Entry.TEntry")

    def test_curve_drag_clamps_adjustable_node_between_neighbors(self):
        self.app._curve_drag_started(2)
        self.app._curve_dragged(SimpleNamespace(y=-1000))
        self.assertEqual(self.app.get_ai_settings().response_curve[2], 0.68)

        self.app._curve_dragged(SimpleNamespace(y=10000))
        self.app._curve_drag_ended()
        self.assertEqual(self.app.get_ai_settings().response_curve[2], 0.12)
        self.assertEqual(self.app.get_ai_settings().response_curve[0], 0.0)

    def test_curve_real_canvas_drag_survives_redraw_until_release(self):
        self.app.deiconify()
        self.app.ai_section.set_expanded(True)
        self.app.dashboard_scroll_canvas.yview_moveto(1.0)
        self.app.update()
        canvas = self.app.ai_curve_canvas
        node_bounds = canvas.bbox("ai-curve-node-2")
        x = (node_bounds[0] + node_bounds[2]) // 2
        y = (node_bounds[1] + node_bounds[3]) // 2

        canvas.event_generate("<Motion>", x=x, y=y)
        canvas.event_generate("<ButtonPress-1>", x=x, y=y)
        self.app.update()
        self.assertEqual(self.app._curve_drag_index, 2)

        canvas.event_generate("<B1-Motion>", x=x, y=y - 20)
        self.app.update()
        first_motion = self.app.get_ai_settings().response_curve[2]
        self.assertNotEqual(first_motion, DEFAULT_RESPONSE_CURVE[2])

        canvas.event_generate("<B1-Motion>", x=x, y=y + 20)
        self.app.update()
        second_motion = self.app.get_ai_settings().response_curve[2]
        self.assertNotEqual(second_motion, first_motion)

        canvas.event_generate("<ButtonRelease-1>", x=x, y=y + 20)
        self.app.update()
        self.assertIsNone(self.app._curve_drag_index)

    def test_curve_redraw_tracks_theme_palette(self):
        self.app.update_idletasks()
        canvas = self.app.ai_curve_canvas
        self.assertEqual(canvas.cget("background"), "#FFFFFF")
        self.assertEqual(canvas.itemcget("ai-curve-sample", "fill"), "#55DDF6")

        self.app.toggle_theme()
        self.app.update_idletasks()

        self.assertEqual(canvas.cget("background"), "#202F43")
        self.assertEqual(canvas.itemcget("ai-curve-sample", "fill"), "#63E6FF")

    def test_curve_sample_is_unsmoothed_response_curve_polyline(self):
        self.app.ai_section.set_expanded(True)
        self.app.update_idletasks()
        canvas = self.app.ai_curve_canvas
        actual = canvas.coords("ai-curve-sample")
        left, top, right, bottom = self.app._curve_plot_bounds()
        plot_width = right - left
        plot_height = bottom - top
        curve = self.app.get_ai_settings().response_curve
        expected = []
        for sample in range(65):
            normalized_x = sample / 64.0
            expected.extend((
                left + normalized_x * plot_width,
                bottom - response_curve_value(curve, normalized_x) * plot_height,
            ))

        self.assertEqual(len(actual), len(expected))
        for actual_coord, expected_coord in zip(actual, expected):
            self.assertAlmostEqual(actual_coord, expected_coord)
        self.assertEqual(canvas.itemcget("ai-curve-sample", "smooth"), "0")

    def test_curve_redraw_is_safe_after_canvas_destroy(self):
        self.app.ai_curve_canvas.destroy()

        self.app._redraw_ai_curve()
        self.app._curve_drag_started(2)
        self.app._curve_dragged(SimpleNamespace(y=10))
        self.app._curve_drag_ended()

        self.assertEqual(self.app.get_ai_settings().response_curve, DEFAULT_RESPONSE_CURVE)

    def test_save_config_persists_curve_without_canvas_or_runtime_state(self):
        curve = (0.0, 0.2, 0.42, 0.8, 1.0)
        for index, value in enumerate(curve[1:], start=1):
            self.app.ai_curve_vars[index].set(str(round(value * 100)))
        self.app._curve_entry_changed(2)
        self.app._cancel_after("_save_after_id")

        self.app.save_config()

        saved = self.store.saved[-1]
        self.assertEqual(saved.ai.response_curve, curve)
        self.assertFalse(hasattr(saved, "ai_curve_canvas"))
        self.assertFalse(hasattr(saved, "ai_curve_vars"))

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
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Dark Mode")
        self.app.toggle_theme()
        self.app.update_idletasks()

        self.assertEqual(self.app.theme_var.get(), "dark")
        self.assertEqual(self.app.cget("background"), "#0D1420")
        self.assertEqual(style.lookup("Liquid.Body.TLabel", "foreground"),
                         "#EEF8FF")
        self.assertEqual(self.app.pulse_size_px_scale.cget("background"),
                         "#172232")
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Light Mode")

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
        self.app._model_choice = ModelChoice(
            Path("C:/private/models/custom.onnx"),
            "custom.onnx",
            False,
            640,
        )

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
            "model_path", "model_name", "model_input_size", "input_size",
        ):
            self.assertFalse(hasattr(self.store.saved[-1], name))

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

    def test_settings_section_exposes_persisted_sound_controls(self):
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
                self.assertTrue(self._is_descendant(widget, self.app.settings_section.body))

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

        self.assertEqual(self.app.test_on_button.cget("text"), "Play Armed Cue")
        self.assertEqual(self.app.test_off_button.cget("text"), "Play Disabled Cue")
        self.assertEqual(
            self.app.test_on_button.cget("style"),
            "Liquid.CompactPrimary.TButton",
        )
        self.assertEqual(
            self.app.test_off_button.cget("style"),
            "Liquid.CompactSecondary.TButton",
        )
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

    def test_jitter_section_exposes_only_paired_pulse_controls(self):
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
        self.assertEqual(int(trigger_card.grid_info()["row"]), 1)
        self.assertEqual(int(modifier_card.grid_info()["row"]), 1)
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

    def test_jitter_section_has_snapshot_backed_live_summary(self):
        """Fails if Motion lacks a visible summary of the active snapshot."""
        summary_var = getattr(self.app, "motion_summary_var", None)
        self.assertIsInstance(summary_var, tk.StringVar)
        self.assertIs(self.app.jitter_section.summary, summary_var)
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

    def test_split_console_close_cancels_all_widget_callbacks_before_service_close(self):
        self.app.deiconify()
        self.app.update()
        sliders = [
            widget for widget in self._descendants(self.app)
            if isinstance(widget, LiquidSlider)
        ]
        self.assertTrue(sliders)
        for slider in sliders:
            slider._schedule_hide_bubble()
            self.assertIsNotNone(slider._bubble_after_id)
        callback_states_at_service_close = []
        original_close = self.service.close

        def observe_service_close():
            callback_states_at_service_close.append((
                self.app._closing,
                tuple(slider._bubble_after_id for slider in sliders),
            ))
            original_close()

        self.service.close = observe_service_close
        self.app.close_app()
        self.assertEqual(
            callback_states_at_service_close,
            [(True, tuple(None for _slider in sliders))],
        )

    def test_invalid_motion_edit_does_not_change_page(self):
        self.app.jitter_section.set_expanded(True)
        self.app.pulse_size_px_var.set("not-a-number")
        self.app._motion_changed("pulse_size_px")
        self.assertTrue(self.app.jitter_section.expanded)
        self.assertTrue(self.app.footer_var.get().startswith("Invalid value for "))

    def test_session_and_theme_actions_use_labeled_ttk_buttons(self):
        labels = {
            self.app.reconnect_button: "Reconnect",
            self.app.test_button: "Test 3s",
            self.app.theme_button: "Switch to Dark Mode",
        }
        for button, label in labels.items():
            with self.subTest(button=str(button)):
                self.assertEqual(button.cget("text"), label)
                self.assertEqual(button.winfo_manager(), "grid")

    def test_theme_action_activates_from_keyboard_focus(self):
        self.app.settings_section.set_expanded(True)
        self.app.deiconify()
        self.app.update()
        self.app.theme_button.focus_set()
        self.app.theme_button.invoke()
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Light Mode")

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

        for section in (self.app.jitter_section, self.app.control_section):
            with self.subTest(section=section.title):
                section.set_expanded(True)
                self.app.toggle_theme()
                self.app.update_idletasks()
                self.assertTrue(section.expanded)
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

    def test_theme_action_updates_its_label(self):
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Dark Mode")
        self.app.theme_button.invoke()
        self.assertEqual(self.app.theme_button.cget("text"), "Switch to Light Mode")

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
        self.app.open_overlay_customizer()
        combos = (
            self.app.trigger_combo,
            self.app.modifier_combo,
            self.app.preset_combo,
            self.app.ramp_mode_combo,
            self.app.target_area_combo,
            self.app.capture_mode_combo,
            self.app.overlay_label_mode_combo,
            self.app.overlay_hud_corner_combo,
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

        self.app.close_overlay_customizer()
        self.app._apply_combobox_popup_palette()

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

    def test_labeled_action_text_contrast_across_themes_and_interaction_states(self):
        """Fails if compact action labels become unreadable."""
        for theme in ("light", "dark"):
            cases = (
                ("Liquid.Primary.TButton", None),
                ("Liquid.Primary.TButton", "active"),
                ("Liquid.Secondary.TButton", None),
                ("Liquid.Secondary.TButton", "active"),
            )
            style = ttk.Style(self.app)
            for widget_style, state in cases:
                with self.subTest(theme=theme, state=state or "normal"):
                    states = () if state is None else (state,)
                    self.assertGreaterEqual(
                        contrast_ratio(
                            style.lookup(widget_style, "foreground", states),
                            style.lookup(widget_style, "background", states),
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
        self.assertEqual(self.app.reconnect_button.cget("text"), "Reconnect")
        self.assertEqual(self.app.test_button.cget("text"), "Test 3s")
    def drain_ui_queue(self):
        self.app._cancel_after("_ui_pump_after_id")
        self.app._drain_ui_queue()

    def test_active_switch_stops_motion_and_ai_then_restarts_candidate_on_validation(self):
        self.prepare_armed_sources(MotionSources(True, True), gate_active=True)
        retiring = self.service.active_motion_generation
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.onnx"
            path.write_bytes(b"model")
            self.model_dialog_result = str(path)
            self.app.model_browse_button.invoke()
            choice, token = self.model_validator.start_calls[-1]

            self.assertIn("model_switch", self.service.cancel_reasons)
            self.assertEqual(self.ai.stop_calls[-1], "Model switch")
            self.assertEqual(self.ai.reset_targeting_calls, 1)
            self.assertFalse(self.app._ai_ready)
            self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
            self.assertTrue(self.app.master_armed)
            self.assertTrue(self.app.jitter_selected)
            self.assertTrue(self.app.ai_selected)
            self.assertFalse(self.app.overlay_visible)

            validated = self.validated_model_choice(choice, 640)
            self.model_validator.emit(
                ModelValidationEvent("ready", token, validated)
            )
            self.drain_ui_queue()

        self.assertEqual(self.ai.start_calls[-1][2], choice.path)
        self.assertEqual(self.app.ai_model_var.get(), "Loading · custom.onnx")
        self.assertEqual(self.app._model_switch.phase, "starting_candidate")
        self.assertFalse(self.app._normal_motion_started)

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Custom · custom.onnx · 640×640",
        )
        self.assertIsNone(self.app._model_switch)
        self.assertTrue(self.app._ai_ready)
        self.assertFalse(self.app._normal_motion_started)
        self.service.emit(ServiceEvent(
            "motion_stopped", "model_switch", retiring
        ))
        self.drain_ui_queue()
        self.assertTrue(self.app._normal_motion_started)

    def test_model_controls_remain_available_for_each_active_demand(self):
        for sources, overlay in (
            (MotionSources(False, True), False),
            (MotionSources(True, True), False),
            (MotionSources(False, False), True),
        ):
            with self.subTest(sources=sources, overlay=overlay):
                self.make_app()
                if overlay:
                    self.app.toggle_overlay()
                else:
                    self.prepare_armed_sources(sources)
                self.assertEqual(
                    str(self.app.model_browse_button.cget("state")), "normal"
                )

    def test_validation_failure_restarts_previous_model_when_demand_still_exists(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )
        choice, token = self.begin_custom_model_switch("bad.onnx")
        self.model_validator.emit(
            ModelValidationEvent("error", token, choice, "ModelContractError")
        )
        self.drain_ui_queue()
        self.assertEqual(self.ai.start_calls[-1][2], self.app._model_choice.path)
        self.assertEqual(self.app._model_switch.phase, "starting_rollback")
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.assertIsNone(self.app._model_switch)
        self.assertEqual(
            self.app.footer_var.get(),
            "Model rejected: AI model validation failed; "
            "restored all_games_320.onnx",
        )

    def test_candidate_runtime_error_rolls_back_once(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )
        choice, token = self.begin_custom_model_switch("loads-then-fails.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        self.app.handle_ai_event(AiEvent("error", "RuntimeError: AI service failed"))
        self.assertEqual(self.app._model_switch.phase, "starting_rollback")
        self.assertEqual(self.ai.start_calls[-1][2], self.app._model_switch.previous.path)

    def test_reconcile_preserves_started_candidate_until_its_ready_event(self):
        self.prepare_armed_sources(MotionSources(False, True))
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("candidate.onnx")
        validated = self.validated_model_choice(choice)
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, validated
        ))
        self.drain_ui_queue()

        self.app.toggle_overlay()

        self.assertIsNotNone(self.app._model_switch)
        self.assertEqual(self.app._model_switch.phase, "starting_candidate")
        self.assertEqual(self.app._model_choice, previous)
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, validated)

    def test_removing_demand_cancels_started_candidate_and_ignores_late_ready(self):
        self.prepare_armed_sources(MotionSources(False, True))
        choice, token = self.begin_custom_model_switch("candidate.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        previous = self.app._model_switch.previous
        self.app.queue_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.app.set_master(False)
        self.drain_ui_queue()

        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertFalse(self.app._ai_runtime_active)
        self.assertFalse(self.app._ai_ready)
        self.assertEqual(self.app.ai_status_var.get(), "Stopped")
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default · all_games_320.onnx · 320×320",
        )
        self.assertEqual(str(self.app.model_browse_button.cget("state")), "normal")
        self.assertFalse(self.app._normal_motion_started)

    def test_stop_cancels_started_rollback_and_ignores_late_ready(self):
        self.prepare_armed_sources(MotionSources(False, True))
        choice, token = self.begin_custom_model_switch("rollback.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        self.app.handle_ai_event(AiEvent("error", "candidate failed"))
        previous = self.app._model_switch.previous
        self.app.queue_ai_event(AiEvent("ready", "DmlExecutionProvider"))

        self.app.emergency_stop("Stopped by user")
        self.drain_ui_queue()

        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertFalse(self.app._ai_runtime_active)
        self.assertFalse(self.app._ai_ready)
        self.assertEqual(self.app.ai_status_var.get(), "Stopped")
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default · all_games_320.onnx · 320×320",
        )
        self.assertEqual(str(self.app.model_browse_button.cget("state")), "normal")
        self.assertFalse(self.app._normal_motion_started)

    def test_stop_exception_cleans_pending_switch_without_restarting_ai(self):
        for phase in ("candidate", "rollback"):
            with self.subTest(phase=phase):
                self.make_app()
                self.prepare_armed_sources(MotionSources(False, True))
                choice, token = self.begin_custom_model_switch(f"{phase}.onnx")
                self.model_validator.emit(
                    ModelValidationEvent(
                        "ready", token, self.validated_model_choice(choice)
                    )
                )
                self.drain_ui_queue()
                if phase == "rollback":
                    self.app.handle_ai_event(AiEvent("error", "candidate failed"))
                previous = self.app._model_switch.previous
                starts_before_stop = len(self.ai.start_calls)
                self.ai.stop_exception = RuntimeError("stop failed")
                failure = None
                try:
                    if phase == "candidate":
                        self.app.set_master(False)
                    else:
                        self.app.emergency_stop("Stopped by user")
                except RuntimeError as error:
                    failure = error

                self.assertIsNone(failure)
                self.assertIsNone(self.app._model_switch)
                self.assertEqual(self.app._model_choice, previous)
                self.assertFalse(self.app._ai_runtime_active)
                self.assertFalse(self.app._ai_ready)
                self.assertEqual(self.app.ai_status_var.get(), "Stopped")
                self.assertEqual(
                    self.app.ai_model_var.get(),
                    "Default · all_games_320.onnx · 320×320",
                )
                self.assertEqual(
                    str(self.app.model_browse_button.cget("state")), "normal"
                )
                self.assertEqual(len(self.ai.start_calls), starts_before_stop)

    def test_rollback_error_enters_existing_fail_closed_path_without_retry(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )
        choice, token = self.begin_custom_model_switch("bad-runtime.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        self.app.handle_ai_event(AiEvent("error", "candidate failed"))
        starts_before_failure = len(self.ai.start_calls)
        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(AiEvent("error", "rollback failed"))
        self.assertEqual(len(self.ai.start_calls), starts_before_failure)
        self.assertIsNone(self.app._model_switch)
        self.assertFalse(self.app.ai_selected)
        self.assertFalse(self.app.master_armed)

    def test_validation_thread_start_failure_restarts_previous_model_once(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )
        self.model_validator.start_result = False
        self.begin_custom_model_switch("thread-fails.onnx")
        self.assertEqual(self.app._model_switch.phase, "starting_rollback")
        self.assertEqual(self.ai.start_calls[-1][2], self.app._model_switch.previous.path)

    def test_test_run_invalidates_pending_model_validation(self):
        candidate, token = self.begin_custom_model_switch("custom.onnx")
        self.service.connected = True
        self.app.jitter_selected = True

        self.app.start_test_run()
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(candidate)
        ))
        self.drain_model_ui_queue()

        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )

    def test_transient_ai_demand_invalidates_pending_model_validation(self):
        candidate, token = self.begin_custom_model_switch("custom.onnx")

        self.app.toggle_overlay()
        self.app.toggle_overlay()
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(candidate)
        ))
        self.drain_model_ui_queue()

        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )

    def test_stop_invalidates_switch_before_late_validation_ready(self):
        self.app.toggle_overlay()
        choice, token = self.begin_custom_model_switch("late.onnx")
        starts = len(self.ai.start_calls)

        self.app.emergency_stop("Stopped by user")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()

        self.assertIsNone(self.app._model_switch)
        self.assertEqual(len(self.ai.start_calls), starts)
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        self.assertFalse(self.app.overlay_visible)

    def test_stale_ready_from_cancelled_switch_cannot_commit_after_new_switch(self):
        first, first_token = self.begin_custom_model_switch("first.onnx")

        self.app.emergency_stop()
        second, second_token = self.begin_custom_model_switch("second.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", first_token, self.validated_model_choice(first)
        ))
        self.drain_ui_queue()

        self.assertEqual(self.app._model_switch.token, second_token)
        self.assertEqual(
            self.app.ai_model_var.get(), "Loading \u00b7 second.onnx"
        )

    def test_removing_ai_source_cancels_candidate_and_restarts_previous_for_overlay(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.app.toggle_overlay()
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("candidate.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        starts_before_removal = len(self.ai.start_calls)

        self.app.toggle_ai_source()

        self.assertTrue(self.app.overlay_visible)
        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertEqual(
            self.ai.start_calls[-1][2], previous.path,
        )
        self.assertEqual(len(self.ai.start_calls), starts_before_removal + 1)

    def test_model_controls_are_disabled_for_every_test_mode(self):
        modes = (
            "test_jitter_pending",
            "test_jitter",
            "test_ai_loading",
            "test_ai",
            "test_combined_loading",
            "test_combined",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                self.app._motion_mode = mode
                self.app._render_runtime_controls()
                self.assertEqual(
                    str(self.app.model_browse_button.cget("state")), "disabled"
                )
                self.assertEqual(
                    str(self.app.use_default_model_button.cget("state")), "disabled"
                )
                self.assertEqual(
                    str(self.app.stop_button.cget("state")), "normal"
                )

    def test_browse_handler_refuses_direct_call_during_test(self):
        self.app._motion_mode = "test_ai_loading"

        self.app.browse_ai_model()

        self.assertEqual(self.model_validator.start_calls, [])
        self.assertEqual(
            self.app.footer_var.get(), "Test Run is active; use STOP to cancel"
        )

    def test_busy_switch_rejects_repeated_browse_and_default_commands(self):
        choice, token = self.begin_custom_model_switch("first.onnx")
        calls = list(self.model_validator.start_calls)

        self.app.browse_ai_model()
        self.app.use_default_ai_model()

        self.assertEqual(self.model_validator.start_calls, calls)
        self.assertEqual(self.app._model_switch.token, token)
        self.assertEqual(self.app._model_switch.candidate, choice)

    def test_disconnect_cancels_candidate_and_restarts_previous_for_overlay(self):
        self.prepare_armed_sources(MotionSources(False, True))
        self.app.toggle_overlay()
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("disconnect.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()

        self.app._handle_disconnect("Device disconnected")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()

        self.assertTrue(self.app.overlay_visible)
        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertEqual(self.ai.start_calls[-1][2], previous.path)

    def test_close_invalidates_switch_token_before_validator_is_closed(self):
        choice, token = self.begin_custom_model_switch("close.onnx")
        starts = len(self.ai.start_calls)

        self.app.close_app()
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))

        self.assertGreater(self.app._model_switch_token, token)
        self.assertEqual(self.model_validator.closed, 1)
        self.assertEqual(len(self.ai.start_calls), starts)

    def test_model_controls_stay_disabled_through_candidate_and_rollback(self):
        self.prepare_armed_sources(MotionSources(False, True))
        choice, token = self.begin_custom_model_switch("controls.onnx")
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "disabled"
        )
        self.assertEqual(
            str(self.app.stop_button.cget("state")), "normal"
        )

        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        self.assertEqual(self.app._model_switch.phase, "starting_candidate")
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "disabled"
        )

        self.app.handle_ai_event(AiEvent("error", "candidate failed"))
        self.assertEqual(self.app._model_switch.phase, "starting_rollback")
        self.assertEqual(
            str(self.app.use_default_model_button.cget("state")), "disabled"
        )

        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        self.assertIsNone(self.app._model_switch)
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )

    def test_adding_ai_source_cancels_idle_validation_before_starting_committed_model(self):
        self.prepare_armed_sources(MotionSources(True, False))
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("addition.onnx")
        starts_before_addition = len(self.ai.start_calls)

        self.app.toggle_ai_source()
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()

        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertEqual(len(self.ai.start_calls), starts_before_addition + 1)
        self.assertEqual(self.ai.start_calls[-1][2], previous.path)

    def test_unarmed_ai_removal_restarts_committed_model_for_overlay_demand(self):
        self.app.toggle_overlay()
        self.app.ai_selected = True
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("unarmed.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        starts_before_removal = len(self.ai.start_calls)

        self.app.toggle_ai_source()

        self.assertFalse(self.app.master_armed)
        self.assertTrue(self.app.overlay_visible)
        self.assertEqual(self.app._model_choice, previous)
        self.assertIsNone(self.app._model_switch)
        self.assertEqual(len(self.ai.start_calls), starts_before_removal + 1)
        self.assertEqual(self.ai.start_calls[-1][2], previous.path)
        self.assertEqual(self.service.composite_motion_calls, [])

    def test_rollback_terminal_error_invalidates_switch_and_restores_label(self):
        self.prepare_armed_sources(MotionSources(False, True))
        choice, token = self.begin_custom_model_switch("rollback-terminal.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        self.app.handle_ai_event(AiEvent("error", "candidate failed"))

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(AiEvent("error", "rollback failed"))

        self.assertIsNone(self.app._model_switch)
        self.assertGreater(self.app._model_switch_token, token)
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        self.assertEqual(self.app.ai_status_var.get(), "Error")
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )

    def test_synchronous_rollback_start_failure_invalidates_switch_and_restores_label(self):
        self.prepare_armed_sources(MotionSources(False, True))
        choice, token = self.begin_custom_model_switch("rollback-start.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        self.ai.start_result = False

        with self.assertLogs(level="ERROR"):
            self.app.handle_ai_event(AiEvent("error", "candidate failed"))

        self.assertIsNone(self.app._model_switch)
        self.assertGreater(self.app._model_switch_token, token)
        self.assertEqual(
            self.app.ai_model_var.get(),
            "Default \u00b7 all_games_320.onnx \u00b7 320\u00d7320",
        )
        self.assertEqual(self.app.ai_status_var.get(), "Error")
        self.assertEqual(
            str(self.app.model_browse_button.cget("state")), "normal"
        )

    def test_candidate_error_contains_stop_failure_and_starts_one_rollback(self):
        self.prepare_armed_sources(MotionSources(False, True))
        choice, token = self.begin_custom_model_switch("stop-error.onnx")
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()
        previous = self.app._model_switch.previous
        self.ai.stop_exception = RuntimeError("stop failed")
        starts_before_error = len(self.ai.start_calls)

        failure = None
        try:
            self.app.handle_ai_event(AiEvent("error", "candidate failed"))
        except RuntimeError as error:
            failure = error

        self.assertIsNone(failure)
        self.assertEqual(self.app._model_switch.phase, "starting_rollback")
        self.assertEqual(len(self.ai.start_calls), starts_before_error + 1)
        self.assertEqual(self.ai.start_calls[-1][2], previous.path)
        self.assertEqual(self.app._model_switch.token, token)

    def test_jitter_only_motion_survives_idle_model_changes_with_and_without_overlay(self):
        for overlay in (False, True):
            with self.subTest(overlay=overlay):
                self.make_app()
                self.prepare_armed_sources(MotionSources(True, False), gate_active=True)
                if overlay:
                    self.app.toggle_overlay()
                motion_generation = self.service.active_motion_generation
                motion_calls = len(self.service.composite_motion_calls)
                cancellation_reasons = list(self.service.cancel_reasons)

                custom, custom_token = self.begin_custom_model_switch("jitter.onnx")
                self.model_validator.emit(
                    ModelValidationEvent(
                        "ready",
                        custom_token,
                        self.validated_model_choice(custom),
                    )
                )
                self.drain_ui_queue()
                if overlay:
                    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

                self.app.use_default_ai_model()
                default, default_token = self.model_validator.start_calls[-1]
                self.model_validator.emit(
                    ModelValidationEvent("ready", default_token, default)
                )
                self.drain_ui_queue()
                if overlay:
                    self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))

                self.assertTrue(self.app._normal_motion_started)
                self.assertEqual(self.service.active_motion_generation, motion_generation)
                self.assertEqual(len(self.service.composite_motion_calls), motion_calls)
                self.assertEqual(self.service.cancel_reasons, cancellation_reasons)

    def test_overlay_demand_cancels_idle_validation_before_strict_duplicate_start(self):
        self.make_app(strict_duplicate_ai=True)
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("overlay-race.onnx")

        self.app.toggle_overlay()
        starts_after_overlay = len(self.ai.start_calls)
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()

        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertEqual(len(self.ai.start_calls), starts_after_overlay)
        self.assertEqual(self.ai.start_calls[-1][2], previous.path)

    def test_master_demand_cancels_idle_validation_before_strict_duplicate_start(self):
        self.make_app(strict_duplicate_ai=True)
        self.service.connected = True
        self.app.ai_selected = True
        previous = self.app._model_choice
        choice, token = self.begin_custom_model_switch("master-race.onnx")

        self.app.set_master(True)
        starts_after_master = len(self.ai.start_calls)
        self.model_validator.emit(ModelValidationEvent(
            "ready", token, self.validated_model_choice(choice)
        ))
        self.drain_ui_queue()

        self.assertIsNone(self.app._model_switch)
        self.assertEqual(self.app._model_choice, previous)
        self.assertEqual(len(self.ai.start_calls), starts_after_master)
        self.assertEqual(self.ai.start_calls[-1][2], previous.path)

    def test_idle_switch_terminal_paths_reset_ai_runtime_status(self):
        for outcome in ("success", "rejected", "cancelled"):
            with self.subTest(outcome=outcome):
                self.make_app()
                choice, token = self.begin_custom_model_switch(f"{outcome}.onnx")
                if outcome == "success":
                    self.model_validator.emit(
                        ModelValidationEvent(
                            "ready", token, self.validated_model_choice(choice)
                        )
                    )
                    self.drain_ui_queue()
                elif outcome == "rejected":
                    self.model_validator.emit(
                        ModelValidationEvent("error", token, choice, "invalid")
                    )
                    self.drain_ui_queue()
                else:
                    self.app.emergency_stop("Stopped by user")

                self.assertEqual(self.app.ai_status_var.get(), "Stopped")
                self.assertEqual(self.app.ai_fps_var.get(), "0 FPS")
                self.assertEqual(self.app.ai_provider_var.get(), "No provider")
                self.assertEqual(
                    self.app.ai_zoom_var.get(),
                    "1.0\N{MULTIPLICATION SIGN}",
                )

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
        (
            _settings_provider,
            zoom_provider,
            model_path,
            capture_mode,
        ) = self.ai.start_calls[-1]
        self.assertEqual(model_path, self.app._model_choice.path)
        self.assertEqual(capture_mode, CENTER_320)
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

    def test_zoom_runtime_value_starts_one_x_and_tracks_valid_events(self):
        self.assertEqual(self.app.ai_zoom_var.get(), "1.0×")
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

    def test_target_area_change_rejects_prechange_queued_zoom_event(self):
        self.prepare_armed_sources(MotionSources(False, True), gate_active=True)
        self.app._cancel_after("_ui_pump_after_id")
        old_revision = self.app._ai_targeting_revision
        self.app.queue_ai_event(AiEvent("zoom", 2.0, old_revision))

        self.app.target_area_var.set("Chest")
        self.app._target_area_changed()
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
        self.assertEqual(str(self.app.test_button.cget("state")), "disabled")

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

    def test_ai_test_run_provider_reads_live_curve_with_zoom_disabled(self):
        self.service.connected = True
        self.app.ai_selected = True

        self.app.start_test_run()
        self.app.handle_ai_event(AiEvent("ready", "DmlExecutionProvider"))
        call = self.service.composite_motion_calls[-1]

        self.assertFalse(self.app.get_adaptive_zoom_gate())
        self.app.ai_curve_vars[2].set("42")
        self.app._curve_entry_changed(2)
        self.assertEqual(call.aim_provider().response_curve[2], 0.42)
        self.assertFalse(self.app.get_adaptive_zoom_gate())

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
        self.app.ai_fps_var.set("73.5 FPS")
        self.app.ai_provider_var.set("DirectML")
        self.app.toggle_overlay()

        self.assertEqual(
            self.overlay.rendered[-1],
            (self.ai.detection_snapshot, 123.5),
        )
        self.assertEqual(
            self.overlay.render_options[-1],
            (
                "#ff2b2b",
                True,
                ("73.5 FPS", "DirectML", "1.0×"),
            ),
        )
        self.assertIsNotNone(self.app._overlay_after_id)

    def test_overlay_poll_publishes_complete_default_runtime_style(self):
        self.app.toggle_overlay()

        self.assertEqual(
            self.overlay.styles[-1],
            OverlayStyle(
                box_color="#ff2b2b",
                show_heads=True,
                hud_color="#ff2b2b",
            ),
        )

    def test_head_boxes_off_reaches_overlay_render(self):
        self.app.open_overlay_customizer()
        self.app.overlay_head_button.invoke()

        self.app.toggle_overlay()

        self.assertEqual(
            self.overlay.render_options[-1],
            (
                "#ff2b2b",
                False,
                ("0 FPS", "No provider", "1.0×"),
            ),
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
        self.assertEqual(str(self.app.test_button.cget("state")), "normal")
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
        self.assertEqual(str(self.app.test_button.cget("state")), "disabled")
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
        self.assertEqual(str(self.app.test_button.cget("state")), "normal")

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

    def test_expansion_and_theme_changes_preserve_values_and_outer_geometry(self):
        self.app.deiconify()
        self.app.update()
        expected_geometry = self.app.geometry()
        expected_motion = self.app.get_motion_settings()
        expected_ai = self.app.get_ai_settings()
        expected_bindings = (
            self.app.trigger_var.get(),
            self.app.modifier_var.get(),
            self.app.hotkey_name_var.get(),
        )
        for section in self.app.sections:
            section.set_expanded(True)
        self.app.toggle_theme()
        self.app.dashboard_scroll_canvas.yview_moveto(1.0)
        self.app.update_idletasks()
        self.assertEqual(self.app.geometry(), expected_geometry)
        self.assertEqual(self.app.get_motion_settings(), expected_motion)
        self.assertEqual(self.app.get_ai_settings(), expected_ai)
        self.assertEqual(
            (
                self.app.trigger_var.get(),
                self.app.modifier_var.get(),
                self.app.hotkey_name_var.get(),
            ),
            expected_bindings,
        )

    def test_section_expansion_is_not_serialized(self):
        self.app.ai_section.set_expanded(True)
        self.app.settings_section.set_expanded(True)
        self.app._cancel_after("_save_after_id")
        self.app.save_config()
        saved = self.store.saved[-1]
        self.assertFalse(hasattr(saved, "section_state"))
        self.assertFalse(hasattr(saved, "expanded_sections"))

    def test_all_sections_expanded_create_scroll_without_moving_fixed_controls(self):
        self.app.deiconify()
        for section in self.app.sections:
            section.set_expanded(True)
        self.app.update()
        scrollregion = tuple(
            float(value)
            for value in self.app.tk.splitlist(
                self.app.dashboard_scroll_canvas.cget("scrollregion")
            )
        )
        self.assertGreater(scrollregion[3] - scrollregion[1],
                           self.app.dashboard_scroll_canvas.winfo_height())
        self.assertTrue(self.app.footer_frame.winfo_ismapped())
        self.assertTrue(self.app.runtime_frame.winfo_ismapped())
        self.assertTrue(self.app.stop_button.winfo_ismapped())
        fixed_positions = (
            self.app.footer_frame.winfo_rooty(),
            self.app.runtime_frame.winfo_rooty(),
            self.app.stop_button.winfo_rooty(),
        )
        self.app.dashboard_scroll_canvas.yview_moveto(1.0)
        self.app.update()
        self.assertEqual(
            (
                self.app.footer_frame.winfo_rooty(),
                self.app.runtime_frame.winfo_rooty(),
                self.app.stop_button.winfo_rooty(),
            ),
            fixed_positions,
        )
        self.assertTrue(self.app.footer_frame.winfo_ismapped())
        self.assertTrue(self.app.runtime_frame.winfo_ismapped())
        self.assertTrue(self.app.stop_button.winfo_ismapped())
