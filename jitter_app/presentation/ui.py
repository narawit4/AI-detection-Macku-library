"""Tkinter dashboard and safe runtime wiring for the standalone Jitter app."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import io
import logging
import math
import queue
import re
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import colorchooser, filedialog, ttk
import tokenize
from typing import Any, Callable, Mapping

from jitter_app.ai.capture import (
    CENTER_320,
    FULL_DISPLAY,
    validated_capture_mode,
)
from jitter_app.ai.service import AiEvent, AiService
from jitter_app.ai.model_selection import (
    ModelChoice,
    ModelSelectionError,
    ModelValidationEvent,
    ModelValidator,
    bundled_model_choice,
    external_model_choice,
)
from jitter_app.ai.targeting import (
    AIM_LIMITS,
    AimSettings,
    DEFAULT_RESPONSE_CURVE,
    aim_settings_from_mapping,
    aim_settings_to_mapping,
    response_curve_value,
    validated_response_curve,
)
from jitter_app.motion.combined import MotionSources
from jitter_app.device.display_timing import RuntimeCadence, detect_runtime_cadence
from jitter_app.device.hotkeys import HotkeyWatcher
from jitter_app.device.makcu import MakcuService, ServiceEvent
from jitter_app.motion.engine import (
    MOTION_LIMITS,
    MOTION_PRESETS,
    RAMP_MODES,
    MotionSettings,
    TriggerGate,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
)
from jitter_app.config.store import AppConfig, ConfigStore, normalize_overlay_color
from .widgets import CollapsibleSection, LiquidSlider
from .sound import ToggleSoundPlayer
from .overlay import (
    MAX_FRAME_AGE_S,
    DetectionOverlay,
    OverlaySetupError,
    OverlayStyle,
)
from jitter_app.resources import sound_directory


_UI_QUEUE_MAX_BATCH = 50
_UI_QUEUE_TIME_SLICE_S = 0.005
_UI_QUEUE_IDLE_DELAY_MS = 15
_RUNTIME_STATE_LABELS = {
    "disabled": "DISABLED",
    "armed": "ARMED",
    "testing": "TESTING",
    "moving": "MOVING",
}
_DEVICE_SUMMARY_FALLBACK = "Makcu device connected"
_DEVICE_SUMMARY_MAX_CHARS = 40
_DEVICE_PORT_PATTERN = re.compile(r"COM[0-9]{1,5}", re.IGNORECASE)
_TEST_MOTION_MODES = {
    "test_jitter_pending",
    "test_jitter",
    "test_ai_loading",
    "test_ai",
    "test_combined_loading",
    "test_combined",
}
_LIFECYCLE_SERVICE_EVENTS = frozenset({
    "button",
})
_AI_CONTROL_SPECS = {
    "confidence": ("Confidence", 0.05, 0.95, 0.01),
    "aim_strength": ("Aim Strength", 0.05, 2.0, 0.01),
    "smoothing": ("Smoothing", 0.0, 0.95, 0.01),
    "max_step": ("Max Step", 1.0, 127.0, 1.0),
}


def _overlay_poll_interval_ms(capture_fps: int) -> int:
    if type(capture_fps) is not int or capture_fps <= 0:
        capture_fps = 120
    return max(1, int(1000 / min(capture_fps, 240)))


_TARGET_AREA_LABELS = {
    "head": "Head",
    "upper_body": "Upper Body",
    "chest": "Chest",
}
_TARGET_AREA_VALUES = {
    label: value for value, label in _TARGET_AREA_LABELS.items()
}
_CAPTURE_MODE_LABELS = {
    CENTER_320: "Center 320",
    FULL_DISPLAY: "Full Display",
}
_CAPTURE_MODE_VALUES = {
    label: value for value, label in _CAPTURE_MODE_LABELS.items()
}


@dataclass(frozen=True)
class _DeferredMotionAction:
    kind: str
    retiring_source: Any
    lifecycle_epoch: int
    sources: MotionSources
    test_generation: int | None = None


@dataclass(frozen=True)
class _ModelSwitch:
    token: int
    candidate: ModelChoice
    previous: ModelChoice
    phase: str
    failure: str | None = None


@dataclass(frozen=True)
class _DeferredAiStart:
    context: str
    model_choice: ModelChoice
    capture_mode: str
    lifecycle_epoch: int
    kind: str
    test_generation: int | None = None


@dataclass(frozen=True)
class _ActiveAiLifecycle:
    request: _DeferredAiStart
    generation: Any
    event_epoch: int
    model_token: int | None = None


DARK_PALETTE = {
    "window": "#0D1420", "surface": "#172232", "raised": "#202F43",
    "border": "#34465C", "text": "#EEF8FF", "muted": "#91A5B8",
    "accent": "#63E6FF", "accent_hover": "#8CECFF",
    "accent_pressed": "#3CC7E1", "green": "#42D392",
    "amber": "#F6C85F", "red": "#FF6B78", "danger": "#C23147",
    "danger_hover": "#CF3B4E", "danger_pressed": "#A52F42",
    "disabled_surface": "#34465C", "disabled_text": "#91A5B8",
    "icon_disabled": "#91A5B8", "green_glow": "#194A3B",
    "amber_glow": "#4A3D21", "red_glow": "#4B2730",
    "focus": "#FFE08A",
}

LIGHT_PALETTE = {
    "window": "#F2F7FA", "surface": "#E5F0F5", "raised": "#FFFFFF",
    "border": "#B9CBD5", "text": "#263640", "muted": "#617581",
    "accent": "#55DDF6", "accent_hover": "#79E8FA",
    "accent_pressed": "#33BDD8", "green": "#146C4D",
    "amber": "#945F00", "red": "#B83246", "danger": "#B83246",
    "danger_hover": "#C74652", "danger_pressed": "#9F3140",
    "disabled_surface": "#D5E0E5", "disabled_text": "#4A5E69",
    "icon_disabled": "#4A5E69", "green_glow": "#B9E6D4",
    "amber_glow": "#F1D89A", "red_glow": "#F1B8C0",
    "focus": "#8B5CF6",
}

_BACKGROUND_BANDS = {
    "dark": ("#0A111C", "#0C1623", "#0D1A2A", "#102035", "#12253C"),
    "light": ("#F8FBFD", "#F4F9FB", "#EFF6F9", "#EAF4F8", "#E5F1F6"),
}

FONT_FAMILY = "Consolas"
BODY_FONT = (FONT_FAMILY, 10)
SMALL_FONT = (FONT_FAMILY, 9)
TITLE_FONT = (FONT_FAMILY, 18, "bold")
SECTION_FONT = (FONT_FAMILY, 9, "bold")


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def _motion_summary_text(settings: MotionSettings) -> str:
    return (
        f"{_display_value(settings.pulse_size_px)} px paired pulse at "
        f"{_display_value(settings.pulse_rate_hz)} Hz | {settings.ramp_mode}"
    )


def _compact_section_summary(*parts: object, limit: int = 72) -> str:
    text = " | ".join(
        " ".join(str(part).split())
        for part in parts
        if str(part).strip()
    )
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _first_serialized_diagnostic(text: str) -> Any:
    """Safely decode the first diagnostic before a top-level pipe."""
    tokens = []
    depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.OP:
                if token.string == "|" and depth == 0:
                    break
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}" and depth:
                    depth -= 1
            if token.type not in {tokenize.ENDMARKER, tokenize.NEWLINE}:
                tokens.append(token)
        return ast.literal_eval(tokenize.untokenize(tokens).strip())
    except (SyntaxError, ValueError, tokenize.TokenError):
        return None


def _device_summary_text(payload: Any) -> str:
    if payload is None:
        return _DEVICE_SUMMARY_FALLBACK
    details: Any = payload
    text = str(payload).strip()
    if not text:
        return _DEVICE_SUMMARY_FALLBACK
    if isinstance(payload, str):
        details = _first_serialized_diagnostic(text)
    if isinstance(details, Mapping):
        raw_port = details.get("port")
        if isinstance(raw_port, str):
            port = raw_port.strip().upper()
            if _DEVICE_PORT_PATTERN.fullmatch(port):
                summary = f"Makcu on {port}"
                if len(summary) <= _DEVICE_SUMMARY_MAX_CHARS:
                    return summary
        return _DEVICE_SUMMARY_FALLBACK
    if text.startswith(("{", "[", "(")):
        return _DEVICE_SUMMARY_FALLBACK
    if len(text) <= _DEVICE_SUMMARY_MAX_CHARS:
        return text
    return _DEVICE_SUMMARY_FALLBACK


class JitterApp(tk.Tk):
    """Fixed-size Liquid Split Console for Jitter.

    The factories make the shell hardware-free in tests and give the runtime
    layer a narrow seam for the real Makcu and global-hotkey services.
    """

    def __init__(
        self,
        *,
        config_store: ConfigStore | None = None,
        service_factory: Callable[[Callable[[Any], None]], Any] | None = None,
        ai_service_factory: Callable[[Callable[[AiEvent], None]], Any] | None = None,
        model_validator_factory: (
            Callable[[Callable[[ModelValidationEvent], None]], Any] | None
        ) = None,
        model_file_chooser: Callable[..., str] | None = None,
        hotkey_factory: Callable[[int, Callable[[], None]], Any] | None = None,
        overlay_factory: Callable[[tk.Misc], Any] | None = None,
        sound_player: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
        auto_start: bool = True,
        runtime_cadence: RuntimeCadence | None = None,
    ) -> None:
        super().__init__()
        for font_name in (
            "TkDefaultFont", "TkTextFont", "TkFixedFont",
            "TkMenuFont", "TkHeadingFont", "TkCaptionFont",
            "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont",
        ):
            tkfont.nametofont(font_name, root=self).configure(
                family=FONT_FAMILY,
            )
        self.option_add("*Font", BODY_FONT)
        self.title("Jitter " + chr(0x2014) + " Makcu Control")
        self.geometry("840x620")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.runtime_cadence = runtime_cadence or detect_runtime_cadence()
        self._overlay_poll_delay_ms = _overlay_poll_interval_ms(
            self.runtime_cadence.capture_fps
        )
        self._last_overlay_render_key = None

        self.config_store = config_store or ConfigStore()
        self.load_outcome = self.config_store.load()
        self.config: AppConfig = self.load_outcome.config
        self._theme = self.config.theme
        self.configure(background=self._palette["window"])
        self._save_allowed = bool(self.load_outcome.save_allowed)
        self._closed = False
        self._runtime_started = False
        self._closing = False
        self._save_after_id: str | None = None
        self._capture_after_id: str | None = None
        self._ui_pump_after_id: str | None = None
        self._overlay_after_id: str | None = None
        self._ui_queue: queue.SimpleQueue[
            tuple[str, int | None, Any]
        ] = queue.SimpleQueue()
        self._ai_event_epoch = 0
        self._ai_targeting_revision = 0
        self._motion_event_epoch = 0
        self._hotkey_event_epoch = 0
        self._hotkey_epoch_lock = threading.Lock()
        self._capturing_hotkey = False
        self._capture_seen_down = False
        self._capture_prev_down: dict[int, bool] = {}
        self._updating_motion_controls = False
        self._invalid_motion_keys: set[str] = set()
        self._updating_ai_controls = False
        self._invalid_ai_keys: set[str] = set()
        self._updating_ai_curve_controls = False
        self._curve_order_error_owners: dict[int, int] = {}
        self._curve_drag_index: int | None = None
        self.jitter_selected = False
        self.ai_selected = False
        self.master_armed = False
        self.overlay_visible = False
        self.overlay_color = self.config.overlay_color
        self.overlay_head_visible = self.config.overlay_head_visible
        self.overlay_player_visible = True
        self.overlay_hud_visible = True
        self.overlay_hud_color = self.overlay_color
        self.overlay_hud_show_fps = True
        self.overlay_hud_show_provider = True
        self.overlay_hud_show_zoom = True
        self.overlay_hud_show_lock = True
        self._color_chooser = colorchooser.askcolor
        self._clock = clock
        self._motion_mode: str | None = None
        self._test_restore_master = False
        self._test_sources: MotionSources | None = None
        self._test_start_pending = False
        self._normal_motion_started = False
        self._expected_motion_generation: Any | None = None
        # STOP is a movement barrier, not a worker-termination join.  Keep the
        # exact canceled source until its queued terminal releases the slot.
        self._retiring_motion_generation: Any | None = None
        self._deferred_motion_action: _DeferredMotionAction | None = None
        self._ai_ready = False
        self._ai_provider: str | None = None
        self._ai_runtime_active = False
        self._capture_mode = CENTER_320
        self._capture_mode_switching = False
        self._capture_restart_pending = False
        self._model_start_pending: tuple[str, int] | None = None
        self._deferred_ai_start: _DeferredAiStart | None = None
        self._active_ai_lifecycle: _ActiveAiLifecycle | None = None
        self._model_choice = bundled_model_choice()
        self._model_switch_token = 0
        self._model_switch: _ModelSwitch | None = None
        self._test_generation = 0
        self._test_pending_generation: int | None = None
        self._test_waiting_for_motion_stop = False
        self._rounded_style_images: dict[str, tuple[tk.PhotoImage, ...]] = {}
        self._motion_lock = threading.RLock()
        self._motion_snapshot: MotionSettings = self.config.motion
        self._ai_lock = threading.RLock()
        self._ai_snapshot: AimSettings = (
            self.config.ai
            if self.config.ai.target_area == "head"
            else replace(self.config.ai, target_area="head")
        )
        self._adaptive_zoom_gate = False
        self._trigger_lock_counter = 0
        self._trigger_lock_epoch: int | None = None
        self._trigger_lock_owner: str | None = None
        self._physical_buttons_down: set[str] = set()
        self._hotkey_vk = int(self.config.hotkey_vk)

        self._configure_styles()
        self._create_variables()
        self._build_page()
        self.model_validator_factory = (
            model_validator_factory
            if model_validator_factory is not None
            else ModelValidator
        )
        self._model_file_chooser = (
            model_file_chooser
            if model_file_chooser is not None
            else filedialog.askopenfilename
        )
        self.model_validator = self.model_validator_factory(
            self.queue_model_validation_event
        )
        self._render_model_controls()
        self.overlay = (overlay_factory or DetectionOverlay)(self)

        cadence = self.runtime_cadence
        self.service_factory = (
            service_factory
            if service_factory is not None
            else lambda sink: MakcuService(sink, ai_poll_hz=cadence.servo_hz)
        )
        self.ai_service_factory = (
            ai_service_factory
            if ai_service_factory is not None
            else lambda sink: AiService(sink, capture_fps=cadence.capture_fps)
        )
        self.hotkey_factory = hotkey_factory or HotkeyWatcher
        self.service = self.service_factory(self.queue_service_event)
        self.ai_service = self.ai_service_factory(self.queue_ai_event)
        self.hotkey_watcher = self.hotkey_factory(
            self.config.hotkey_vk, self._hotkey_pressed
        )
        self.sound_player = (
            sound_player
            if sound_player is not None
            else ToggleSoundPlayer(
                sound_directory(),
                enabled=self.config.sound_enabled,
                volume=self.config.sound_volume,
            )
        )
        self.sound_player.configure(
            enabled=self.config.sound_enabled,
            volume=self.config.sound_volume,
        )
        # Keep the short alias for integrations that used the shell seam.
        self.hotkey = self.hotkey_watcher
        self.trigger_gate = TriggerGate(self.config.trigger, self.config.modifier)
        # Task 7 intentionally does not start background services.  Task 8
        # owns start_runtime() and the lifecycle transitions.
        self.auto_start = bool(auto_start)
        self._install_runtime_bindings()
        self._ui_pump_after_id = self.after(0, self._drain_ui_queue)
        if self.auto_start:
            self.start_runtime()

    # ---- setup ---------------------------------------------------------

    def _rounded_style_image(
        self,
        fill: str,
        border: str,
        *,
        size: int = 24,
        radius: int = 8,
    ) -> tk.PhotoImage:
        image = tk.PhotoImage(master=self, width=size, height=size)
        inner_radius = max(1, radius - 1)

        def inside(px: int, py: int, inset: int, corner_radius: int) -> bool:
            low = inset + corner_radius
            high = size - inset - corner_radius - 1
            nearest_x = min(max(px, low), high)
            nearest_y = min(max(py, low), high)
            return (
                (px - nearest_x) * (px - nearest_x)
                + (py - nearest_y) * (py - nearest_y)
                <= corner_radius * corner_radius
            )

        for y in range(size):
            outer = [x for x in range(size) if inside(x, y, 0, radius)]
            if outer:
                image.put(border, to=(outer[0], y, outer[-1] + 1, y + 1))
            inner = [x for x in range(size) if inside(x, y, 1, inner_radius)]
            if inner:
                image.put(fill, to=(inner[0], y, inner[-1] + 1, y + 1))
        return image

    def _install_rounded_element(
        self,
        style: ttk.Style,
        role: str,
        colors: tuple[str, str, str, str, str],
        *,
        focus: tuple[str, str] | None = None,
    ) -> str:
        element = f"Liquid.Rounded.{self._theme}.{role}"
        if element in style.element_names():
            return element
        normal, active, pressed, disabled, border = colors
        images = [
            self._rounded_style_image(normal, border),
            self._rounded_style_image(active, border),
            self._rounded_style_image(pressed, border),
            self._rounded_style_image(disabled, border),
        ]
        state_images: list[tuple[str, tk.PhotoImage]] = [
            ("disabled", images[3]),
            ("pressed", images[2]),
        ]
        if focus is not None:
            focus_image = self._rounded_style_image(*focus)
            images.append(focus_image)
            state_images.append(("focus", focus_image))
        state_images.append(("active", images[1]))
        self._rounded_style_images[element] = tuple(images)
        style.element_create(
            element,
            "image",
            images[0],
            *state_images,
            border=8,
            sticky="nsew",
        )
        return element

    def _configure_styles(self) -> None:
        p = self._palette
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        secondary_hover = "#2A3B52" if self._theme == "dark" else "#D6F5FA"
        secondary_pressed = p["surface"] if self._theme == "dark" else "#B7EFF8"
        danger_hover = p["danger_hover"]
        danger_pressed = p["danger_pressed"]
        disabled_background = p["disabled_surface"]
        disabled_text = p["disabled_text"]
        settings_accent = p["accent"] if self._theme == "dark" else "#0B7186"
        settings_muted = p["muted"] if self._theme == "dark" else "#526771"
        primary_element = self._install_rounded_element(
            style,
            "Primary",
            (p["accent"], p["accent_hover"], p["accent_pressed"],
             disabled_background, p["accent_pressed"]),
        )
        secondary_element = self._install_rounded_element(
            style,
            "Secondary",
            (p["raised"], secondary_hover, secondary_pressed,
             disabled_background, p["border"]),
        )
        danger_element = self._install_rounded_element(
            style,
            "Danger",
            (p["danger"], danger_hover, danger_pressed,
             disabled_background, p["red"]),
        )
        entry_element = self._install_rounded_element(
            style,
            "Entry",
            (p["raised"], p["raised"], p["raised"],
             disabled_background, p["border"]),
        )
        invalid_element = self._install_rounded_element(
            style,
            "Invalid",
            (p["raised"], p["raised"], p["raised"],
             disabled_background, p["red"]),
        )
        dropdown_element = self._install_rounded_element(
            style,
            "Dropdown",
            (p["raised"], secondary_hover, p["surface"],
             disabled_background, p["border"]),
            focus=(p["surface"], p["accent"]),
        )
        section_element = self._install_rounded_element(
            style,
            "Section",
            (
                p["surface"], p["raised"], p["surface"],
                disabled_background, p["border"],
            ),
            focus=(p["surface"], p["accent"]),
        )
        button_layout = lambda element: [
            (element, {
                "sticky": "nsew",
                "children": [("Button.padding", {
                    "sticky": "nsew",
                    "children": [("Button.label", {"sticky": "nsew"})],
                })],
            }),
        ]

        style.configure("Liquid.App.TFrame", background=p["window"])
        style.configure("Liquid.Surface.TFrame", background=p["surface"],
                        bordercolor=p["border"], relief="flat", borderwidth=0)
        style.configure(
            "Liquid.Section.TButton",
            background=p["surface"],
            foreground=p["text"],
            bordercolor=p["border"],
            font=(FONT_FAMILY, 10, "bold"),
            padding=(10, 8),
            anchor="w",
        )
        style.map(
            "Liquid.Section.TButton",
            background=[("pressed", p["surface"]), ("active", p["raised"])],
            foreground=[("disabled", disabled_text)],
        )
        style.configure(
            "Liquid.SectionBody.TFrame",
            background=p["surface"],
            bordercolor=p["border"],
            relief="flat",
            borderwidth=0,
        )
        style.configure("Liquid.Title.TLabel", background=p["surface"],
                        foreground=p["text"], font=TITLE_FONT)
        style.configure("Liquid.Subtitle.TLabel", background=p["surface"],
                        foreground=p["muted"], font=SMALL_FONT)
        style.configure(
            "Liquid.SettingsEyebrow.TLabel",
            background=p["window"],
            foreground=settings_accent,
            font=(FONT_FAMILY, 9, "bold"),
        )
        style.configure(
            "Liquid.SettingsTitle.TLabel",
            background=p["window"],
            foreground=p["text"],
            font=(FONT_FAMILY, 22, "bold"),
        )
        style.configure(
            "Liquid.SettingsCard.TFrame",
            background=p["surface"],
            bordercolor=p["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Liquid.CardTitle.TLabel",
            background=p["surface"],
            foreground=p["text"],
            font=(FONT_FAMILY, 12, "bold"),
        )
        style.configure(
            "Liquid.CardBody.TLabel",
            background=p["surface"],
            foreground=settings_muted,
            font=SMALL_FONT,
        )
        style.configure(
            "Liquid.CardText.TLabel",
            background=p["surface"],
            foreground=p["text"],
            font=BODY_FONT,
        )
        style.configure(
            "Liquid.Metric.TFrame",
            background=p["raised"],
            bordercolor=p["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Liquid.MetricLabel.TLabel",
            background=p["raised"],
            foreground=settings_muted,
            font=(FONT_FAMILY, 8, "bold"),
        )
        style.configure(
            "Liquid.MetricValue.TLabel",
            background=p["raised"],
            foreground=p["text"],
            font=(FONT_FAMILY, 22, "bold"),
        )
        style.configure(
            "Liquid.MetricUnit.TLabel",
            background=p["raised"],
            foreground=settings_accent,
            font=(FONT_FAMILY, 9, "bold"),
        )
        style.configure(
            "Liquid.Volume.TLabel",
            background=p["surface"],
            foreground=p["text"],
            font=(FONT_FAMILY, 30, "bold"),
        )
        style.configure(
            "Liquid.VolumeUnit.TLabel",
            background=p["surface"],
            foreground=settings_accent,
            font=(FONT_FAMILY, 12, "bold"),
        )
        style.configure("Liquid.Body.TLabel", background=p["window"],
                        foreground=p["text"], font=BODY_FONT)
        style.configure("Liquid.Muted.TLabel", background=p["window"],
                        foreground=settings_muted, font=SMALL_FONT)
        style.configure(
            "Liquid.TCheckbutton",
            background=p["window"],
            foreground=p["text"],
            font=BODY_FONT,
            focuscolor=p["focus"],
        )
        style.map(
            "Liquid.TCheckbutton",
            background=[("active", p["window"])],
            foreground=[("disabled", p["muted"])],
        )
        style.configure(
            "Liquid.Surface.TCheckbutton",
            background=p["surface"],
            foreground=p["text"],
            font=BODY_FONT,
            focuscolor=p["focus"],
        )
        style.map(
            "Liquid.Surface.TCheckbutton",
            background=[("active", p["surface"])],
            foreground=[("disabled", p["muted"])],
        )
        style.configure("Liquid.Card.TLabelframe", background=p["window"],
                        foreground=p["text"], relief="flat", borderwidth=0)
        style.configure("Liquid.Card.TLabelframe.Label", background=p["window"],
                        foreground=p["muted"], font=SECTION_FONT)
        style.configure(
            "Liquid.Field.TLabelframe",
            background=p["surface"],
            bordercolor=p["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Liquid.Field.TLabelframe.Label",
            background=p["surface"],
            foreground=settings_muted,
            font=(FONT_FAMILY, 8, "bold"),
        )
        style.configure(
            "Liquid.DropdownField.TFrame",
            background=p["surface"],
            relief="flat",
            borderwidth=0,
        )
        style.configure(
            "Liquid.DropdownLabel.TLabel",
            background=p["surface"],
            foreground=settings_muted,
            font=(FONT_FAMILY, 8, "bold"),
        )
        style.configure("Liquid.Primary.TButton", background=p["accent"],
                        foreground="#07252C", bordercolor=p["accent_pressed"],
                        focuscolor=p["accent"], focusthickness=0,
                        relief="flat", borderwidth=0,
                        font=(FONT_FAMILY, 10, "bold"), padding=(14, 8))
        style.map("Liquid.Primary.TButton",
                  background=[("disabled", disabled_background),
                              ("pressed", p["accent_pressed"]),
                              ("active", p["accent_hover"])],
                  foreground=[("disabled", disabled_text)],
                  relief=[("pressed", "flat"), ("!pressed", "flat")])
        style.configure("Liquid.Secondary.TButton", background=p["raised"],
                        foreground=p["text"], bordercolor=p["border"],
                        focuscolor=p["raised"], focusthickness=0,
                        relief="flat", borderwidth=0,
                        font=BODY_FONT, padding=(12, 7))
        style.map("Liquid.Secondary.TButton",
                  background=[("disabled", disabled_background),
                              ("pressed", secondary_pressed),
                              ("active", secondary_hover)],
                  foreground=[("disabled", disabled_text)],
                  relief=[("pressed", "flat"), ("!pressed", "flat")])
        style.configure(
            "Liquid.CompactPrimary.TButton",
            background=p["accent"],
            foreground="#07252C",
            bordercolor=p["accent_pressed"],
            focuscolor=p["accent"],
            focusthickness=0,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 9, "bold"),
            padding=(7, 4),
        )
        style.map(
            "Liquid.CompactPrimary.TButton",
            background=[
                ("disabled", disabled_background),
                ("pressed", p["accent_pressed"]),
                ("active", p["accent_hover"]),
            ],
            foreground=[("disabled", disabled_text)],
            relief=[("pressed", "flat"), ("!pressed", "flat")],
        )
        style.configure(
            "Liquid.CompactSecondary.TButton",
            background=p["raised"],
            foreground=p["text"],
            bordercolor=p["border"],
            focuscolor=p["raised"],
            focusthickness=0,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 9, "bold"),
            padding=(7, 4),
        )
        style.map(
            "Liquid.CompactSecondary.TButton",
            background=[
                ("disabled", disabled_background),
                ("pressed", secondary_pressed),
                ("active", secondary_hover),
            ],
            foreground=[("disabled", disabled_text)],
            relief=[("pressed", "flat"), ("!pressed", "flat")],
        )
        style.configure("Liquid.Danger.TButton", background=p["danger"],
                        foreground="#FFFFFF", bordercolor=p["red"],
                        focuscolor=p["danger"], focusthickness=0,
                        relief="flat", borderwidth=0,
                        font=(FONT_FAMILY, 10, "bold"), padding=(14, 8))
        style.map("Liquid.Danger.TButton",
                  background=[("disabled", disabled_background),
                              ("pressed", danger_pressed),
                              ("active", danger_hover)],
                  foreground=[("disabled", disabled_text)],
                  relief=[("pressed", "flat"), ("!pressed", "flat")])
        for state, color in (
            ("Disconnected", p["red"]),
            ("Connecting", p["amber"]),
            ("Connected", p["green"]),
        ):
            style.configure(f"Liquid.Status{state}.TLabel",
                            background=p["surface"], foreground=color,
                            font=(FONT_FAMILY, 10, "bold"))
        style.configure("Liquid.Entry.TEntry", fieldbackground=p["raised"],
                        foreground=p["text"], insertcolor=p["text"],
                        padding=(6, 4), relief="flat", borderwidth=0)
        style.configure("Liquid.Invalid.TEntry", fieldbackground=p["raised"],
                        foreground=p["text"], insertcolor=p["text"],
                        padding=(6, 4), relief="flat", borderwidth=0)
        style.configure(
            "Liquid.Modern.TCombobox",
            fieldbackground=p["raised"],
            background=p["raised"],
            foreground=p["text"],
            arrowcolor=p["accent"],
            padding=(10, 7),
            relief="flat",
            borderwidth=0,
            font=BODY_FONT,
        )
        style.map(
            "Liquid.Modern.TCombobox",
            fieldbackground=[
                ("disabled", disabled_background),
                ("focus", p["surface"]),
                ("active", secondary_hover),
                ("readonly", p["raised"]),
            ],
            background=[
                ("disabled", disabled_background),
                ("focus", p["surface"]),
                ("active", secondary_hover),
                ("readonly", p["raised"]),
            ],
            foreground=[
                ("disabled", disabled_text),
                ("readonly", p["text"]),
            ],
            arrowcolor=[
                ("disabled", disabled_text),
                ("readonly", p["accent"]),
            ],
        )
        style.configure(
            "Liquid.Vertical.TScrollbar",
            background=p["raised"],
            troughcolor=p["surface"],
            arrowcolor=p["text"],
            borderwidth=0,
            relief="flat",
            darkcolor=p["raised"],
            lightcolor=p["raised"],
        )
        style.layout("Liquid.Section.TButton", button_layout(section_element))
        style.layout("Liquid.Primary.TButton", button_layout(primary_element))
        style.layout("Liquid.Secondary.TButton", button_layout(secondary_element))
        style.layout(
            "Liquid.CompactPrimary.TButton", button_layout(primary_element)
        )
        style.layout(
            "Liquid.CompactSecondary.TButton", button_layout(secondary_element)
        )
        style.layout("Liquid.Danger.TButton", button_layout(danger_element))
        entry_layout = lambda element: [
            (element, {
                "sticky": "nsew",
                "children": [("Entry.padding", {
                    "sticky": "nsew",
                    "children": [("Entry.textarea", {"sticky": "nsew"})],
                })],
            }),
        ]
        style.layout("Liquid.Entry.TEntry", entry_layout(entry_element))
        style.layout("Liquid.Invalid.TEntry", entry_layout(invalid_element))
        style.layout("Liquid.Modern.TCombobox", [
            (dropdown_element, {
                "sticky": "nsew",
                "children": [
                    ("Combobox.downarrow", {"side": "right", "sticky": "ns"}),
                    ("Combobox.padding", {
                        "sticky": "nsew",
                        "children": [("Combobox.textarea", {"sticky": "nsew"})],
                    }),
                ],
            }),
        ])
        style.map(
            "Liquid.Vertical.TScrollbar",
            background=[
                ("disabled", disabled_background),
                ("pressed", p["accent_pressed"]),
                ("active", p["accent_hover"]),
            ],
            arrowcolor=[("disabled", disabled_text)],
        )

    @property
    def _palette(self) -> Mapping[str, str]:
        return DARK_PALETTE if self._theme == "dark" else LIGHT_PALETTE

    def _create_variables(self) -> None:
        self.connection_status_var = tk.StringVar(self, "Disconnected")
        self.runtime_status_var = tk.StringVar(
            self, _RUNTIME_STATE_LABELS["disabled"]
        )
        self.connection_state_var = self.connection_status_var
        self.runtime_state_var = self.runtime_status_var
        self.device_status_var = tk.StringVar(self, "Makcu device not connected")
        self.trigger_var = tk.StringVar(self, self.config.trigger)
        self.modifier_var = tk.StringVar(self, self.config.modifier)
        self.hotkey_name_var = tk.StringVar(self, self.config.hotkey_name)
        self.preset_var = tk.StringVar(self, self._selected_preset())
        self.footer_var = tk.StringVar(self, "Ready")
        self.theme_var = tk.StringVar(self, self._theme)
        self.sound_enabled_var = tk.BooleanVar(
            self, self.config.sound_enabled
        )
        self.sound_volume_var = tk.StringVar(
            self, str(self.config.sound_volume)
        )
        self.ai_status_var = tk.StringVar(self, "Stopped")
        self.ai_fps_var = tk.StringVar(self, "0 FPS")
        self.ai_provider_var = tk.StringVar(self, "No provider")
        self.ai_model_var = tk.StringVar(
            self, self._model_label(self._model_choice)
        )
        cadence = self.runtime_cadence
        self.ai_cadence_var = tk.StringVar(
            self,
            (
                f"DISPLAY {cadence.display_hz} HZ · SERVO {cadence.servo_hz} HZ"
                if cadence.display_hz is not None
                else f"DISPLAY AUTO · SERVO {cadence.servo_hz} HZ"
            ),
        )
        self.ai_zoom_var = tk.StringVar(self, "1.0×")
        ai_mapping = aim_settings_to_mapping(self._ai_snapshot)
        self.ai_vars = {
            key: tk.StringVar(self, ai_mapping[key])
            for key in _AI_CONTROL_SPECS
        }
        self.target_area_var = tk.StringVar(
            self,
            _TARGET_AREA_LABELS["head"],
        )
        self.capture_mode_var = tk.StringVar(
            self,
            _CAPTURE_MODE_LABELS[CENTER_320],
        )
        response_curve = validated_response_curve(self.config.ai.response_curve)
        self.ai_curve_vars = {
            index: tk.StringVar(
                self, str(int(round(response_curve[index] * 100.0)))
            )
            for index in range(1, 5)
        }
        self.overlay_box_width_var = tk.StringVar(self, "2")
        self.overlay_label_mode_var = tk.StringVar(self, "Off")
        self.overlay_hud_corner_var = tk.StringVar(self, "Top Left")
        self.overlay_hud_offset_x_var = tk.StringVar(self, "8")
        self.overlay_hud_offset_y_var = tk.StringVar(self, "8")
        self.overlay_hud_font_size_var = tk.StringVar(self, "10")
        self._overlay_box_width_bounds = (1, 8)
        self._overlay_hud_offset_x_bounds = (0, 500)
        self._overlay_hud_offset_y_bounds = (0, 500)
        self._overlay_hud_font_size_bounds = (8, 24)
        self.motion_summary_var = tk.StringVar(
            self, _motion_summary_text(self._motion_snapshot)
        )
        self.control_section_summary_var = tk.StringVar(self, "No sources")
        self.ai_section_summary_var = tk.StringVar(self, "Default model")
        self.overlay_section_summary_var = tk.StringVar(self, "Overlay Off")
        self.settings_section_summary_var = tk.StringVar(self, "Sound On")
        self.motion_snapshot_size_var = tk.StringVar(
            self, _display_value(self._motion_snapshot.pulse_size_px)
        )
        self.motion_snapshot_rate_var = tk.StringVar(
            self, _display_value(self._motion_snapshot.pulse_rate_hz)
        )
        self.motion_snapshot_ramp_var = tk.StringVar(
            self, self._motion_snapshot.ramp_mode
        )

        mapping = motion_settings_to_mapping(self.config.motion)
        self.motion_vars: dict[str, tk.Variable] = {}
        for key, value in mapping.items():
            variable = tk.StringVar(self, _display_value(value))
            self.motion_vars[key] = variable
            setattr(self, f"{key}_var", variable)

    def _selected_preset(self) -> str:
        choices = self.preset_values
        return self.config.selected_preset if self.config.selected_preset in choices else "Custom"

    @property
    def preset_values(self) -> tuple[str, ...]:
        # Custom is display-only and represents a non-preset combination in
        # the current controls.
        return ("Custom", *MOTION_PRESETS.keys())

    def _build_page(self) -> None:
        self.shell = tk.Canvas(
            self,
            background=self._palette["window"],
            highlightthickness=0,
            borderwidth=0,
            takefocus=False,
        )
        self.shell.pack(fill="both", expand=True)
        self.shell.rowconfigure(0, weight=1)
        self.shell.columnconfigure(0, weight=1)
        self.shell.bind("<Configure>", self._redraw_shell_art, add="+")
        self.console_workspace = ttk.Frame(
            self.shell, style="Liquid.App.TFrame", padding=(12, 10)
        )
        self.console_workspace.grid(row=0, column=0, sticky="nsew")
        self.console_workspace.columnconfigure(0, weight=1)
        self.console_workspace.rowconfigure(1, weight=1)

        self._build_topbar()
        self._build_dashboard()
        self._build_footer()
        self._build_main_control_card()
        self._apply_combobox_popup_palette()
        for panel in (self.topbar_frame, self.dashboard_frame, self.runtime_frame):
            panel.bind("<Configure>", self._redraw_shell_art, add="+")
        self._redraw_shell_art()

    def _build_topbar(self) -> None:
        self.topbar_frame = ttk.Frame(
            self.console_workspace,
            style="Liquid.Surface.TFrame",
            padding=(12, 8),
        )
        self.topbar_frame.grid(row=0, column=0, sticky="ew")
        self.topbar_frame.columnconfigure(0, weight=1)
        self.identity_frame = ttk.Frame(
            self.topbar_frame, style="Liquid.Surface.TFrame"
        )
        self.identity_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.identity_frame, text="JITTER", style="Liquid.Title.TLabel"
        ).pack(anchor="w")
        connection_row = ttk.Frame(
            self.topbar_frame, style="Liquid.Surface.TFrame"
        )
        connection_row.grid(row=0, column=1, sticky="e")
        self.connection_indicator = tk.Canvas(
            connection_row,
            width=18,
            height=18,
            background=self._palette["surface"],
            highlightthickness=0,
            borderwidth=0,
            takefocus=False,
        )
        self.connection_indicator.pack(side="left", padx=(0, 5), pady=(1, 0))
        self.connection_label = ttk.Label(
            connection_row,
            textvariable=self.connection_status_var,
            style="Liquid.StatusDisconnected.TLabel",
        )
        self.connection_label.pack(side="left")
        self._redraw_connection_indicator()

    def _build_dashboard(self) -> None:
        self.dashboard_frame = ttk.Frame(
            self.console_workspace, style="Liquid.App.TFrame"
        )
        self.dashboard_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 6))
        self.dashboard_frame.columnconfigure(0, weight=1)
        self.dashboard_frame.rowconfigure(0, weight=1)
        self.dashboard_scroll_canvas = tk.Canvas(
            self.dashboard_frame, background=self._palette["window"],
            highlightthickness=0, borderwidth=0, takefocus=False,
        )
        self.dashboard_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self.dashboard_scrollbar = ttk.Scrollbar(
            self.dashboard_frame, orient="vertical",
            style="Liquid.Vertical.TScrollbar",
            command=self.dashboard_scroll_canvas.yview,
        )
        self.dashboard_scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        self.dashboard_scroll_canvas.configure(
            yscrollcommand=self.dashboard_scrollbar.set
        )
        self.dashboard_content = ttk.Frame(
            self.dashboard_scroll_canvas, style="Liquid.App.TFrame"
        )
        self.dashboard_content.columnconfigure(0, weight=1)
        self._dashboard_scroll_window = self.dashboard_scroll_canvas.create_window(
            (0, 0), window=self.dashboard_content, anchor="nw"
        )
        self.dashboard_content.bind(
            "<Configure>", self._refresh_dashboard_scrollregion, add="+"
        )
        self.dashboard_scroll_canvas.bind(
            "<Configure>", self._resize_dashboard_content, add="+"
        )
        self.bind("<MouseWheel>", self._scroll_dashboard, add="+")
        definitions = (
            ("control_section", 1, "Control", self.control_section_summary_var, True),
            ("jitter_section", 2, "Jitter", self.motion_summary_var, False),
            ("ai_section", 3, "AI Aim", self.ai_section_summary_var, False),
            ("overlay_section", 4, "Overlay", self.overlay_section_summary_var, False),
            ("settings_section", 5, "Settings", self.settings_section_summary_var, False),
        )
        sections = []
        for row, (attribute, number, title, summary, expanded) in enumerate(definitions):
            section = CollapsibleSection(
                self.dashboard_content, number=number, title=title,
                summary=summary, expanded=expanded,
            )
            section.grid(row=row, column=0, sticky="ew", pady=(0, 7))
            setattr(self, attribute, section)
            sections.append(section)
        self.sections = tuple(sections)
        self._build_control_section(self.control_section.body)
        self._build_jitter_section(self.jitter_section.body)
        self._build_ai_section(self.ai_section.body)
        self._build_overlay_section(self.overlay_section.body)
        self._build_settings_section(self.settings_section.body)
        self._refresh_section_summaries()

    def toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_var.set(self._theme)
        self._configure_styles()
        self.configure(background=self._palette["window"])
        if self._overlay_customizer_exists():
            self.overlay_custom_window.configure(
                background=self._palette["window"]
            )
        self.shell.configure(background=self._palette["window"])
        self.dashboard_scroll_canvas.configure(background=self._palette["window"])
        self._redraw_shell_art()
        self._redraw_connection_indicator()
        self._redraw_ai_curve()
        self.theme_button.configure(
            text=(
                "Switch to Light Mode"
                if self._theme == "dark"
                else "Switch to Dark Mode"
            )
        )
        self._apply_combobox_popup_palette()
        slider_palette = self._slider_palette()
        for widget in self.winfo_children():
            self._apply_slider_palette(widget, slider_palette)
        surface_slider_names = (
            "sound_volume_scale",
            "pulse_size_px_scale",
            "pulse_rate_hz_scale",
            "ai_confidence_scale",
            "ai_aim_strength_scale",
            "ai_smoothing_scale",
            "ai_max_step_scale",
        )
        if self._overlay_customizer_exists():
            surface_slider_names += (
                "overlay_box_width_scale",
                "overlay_hud_offset_x_scale",
                "overlay_hud_offset_y_scale",
                "overlay_hud_font_size_scale",
            )
        for name in surface_slider_names:
            surface_slider = getattr(self, name, None)
            if surface_slider is not None:
                surface_slider.set_palette(
                    self._slider_palette(on_surface=True)
                )
        self._refresh_section_summaries()
        self._schedule_save()

    def _apply_slider_palette(self, widget: tk.Misc,
                              palette: Mapping[str, str]) -> None:
        if isinstance(widget, LiquidSlider):
            widget.set_palette(palette)
        for child in widget.winfo_children():
            self._apply_slider_palette(child, palette)

    def _cancel_slider_callbacks(self, widget: tk.Misc) -> None:
        if isinstance(widget, LiquidSlider):
            widget.cancel_pending_callbacks()
        for child in widget.winfo_children():
            self._cancel_slider_callbacks(child)

    def _slider_palette(self, *, on_surface: bool = False) -> dict[str, str]:
        p = self._palette
        return {
            "background": p["surface"] if on_surface else p["window"],
            "rail": p["border"],
            "fill": p["accent"], "thumb": p["raised"],
            "thumb_border": p["accent_pressed"],
            "halo": p["surface"], "text": p["text"],
            "bubble": p["raised"], "bubble_text": p["text"],
            "focus": p["focus"], "disabled": p["border"],
            "disabled_text": p["muted"],
        }

    def _apply_combobox_popup_palette(self) -> None:
        p = self._palette
        for name in (
            "trigger_combo",
            "modifier_combo",
            "preset_combo",
            "ramp_mode_combo",
            "target_area_combo",
            "capture_mode_combo",
            "overlay_label_mode_combo",
            "overlay_hud_corner_combo",
        ):
            combo = getattr(self, name, None)
            if combo is None:
                continue
            try:
                if not combo.winfo_exists():
                    continue
                popdown = self.tk.call(
                    "ttk::combobox::PopdownWindow", str(combo)
                )
                self.tk.call(
                    f"{popdown}.f.l",
                    "configure",
                    "-background", p["raised"],
                    "-foreground", p["text"],
                    "-selectbackground", p["accent"],
                    "-selectforeground", "#07252C",
                    "-font", (FONT_FAMILY, 10),
                    "-relief", "flat",
                    "-borderwidth", 0,
                    "-highlightthickness", 1,
                    "-highlightbackground", p["border"],
                    "-highlightcolor", p["accent"],
                    "-activestyle", "none",
                )
                self.tk.call(
                    f"{popdown}.f.sb",
                    "configure",
                    "-style", "Liquid.Vertical.TScrollbar",
                )
            except tk.TclError:
                logging.debug(
                    "Could not apply combobox popup palette to %s", combo,
                    exc_info=True,
                )

    def _card(self, title: str, parent: tk.Misc) -> ttk.LabelFrame:
        card = ttk.LabelFrame(
            parent, text=title, style="Liquid.Card.TLabelframe",
            padding=(12, 8, 12, 10)
        )
        card.pack(fill="x", pady=(0, 9))
        return card

    def _build_main_control_card(self) -> None:
        self.runtime_frame = ttk.Frame(
            self.console_workspace,
            style="Liquid.Surface.TFrame",
            padding=(8, 6),
        )
        self.runtime_frame.grid(
            row=3, column=0, sticky="ew", pady=(4, 0)
        )
        self.runtime_frame.columnconfigure(0, weight=1, uniform="runtime_actions")
        self.runtime_frame.columnconfigure(1, weight=2)
        self.runtime_frame.columnconfigure(2, weight=1, uniform="runtime_actions")
        self.master_button = ttk.Button(
            self.runtime_frame,
            text="Enable Selected",
            style="Liquid.Primary.TButton",
            command=lambda: self.toggle_master(),
        )
        self.master_button.grid(row=0, column=0, sticky="ew")
        self.enable_button = self.master_button
        state = ttk.Frame(self.runtime_frame, style="Liquid.Surface.TFrame")
        state.grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Label(
            state, text="RUNTIME", style="Liquid.Subtitle.TLabel"
        ).pack(anchor="center")
        ttk.Label(state, textvariable=self.runtime_status_var,
                  style="Liquid.Subtitle.TLabel",
                  font=(FONT_FAMILY, 10, "bold")).pack(anchor="center")
        self.stop_button = ttk.Button(self.runtime_frame, text="STOP",
                                      style="Liquid.Danger.TButton",
                                      command=self.emergency_stop)
        self.stop_button.grid(row=0, column=2, sticky="ew")

    def _build_control_section(self, parent: ttk.Frame) -> None:
        self.control_frame = parent
        parent.columnconfigure(0, weight=3, uniform="control")
        parent.columnconfigure(1, weight=2, uniform="control")
        self.control_bindings_card = ttk.Frame(
            parent,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.control_bindings_card.grid(
            row=0, column=0, sticky="nsew", padx=(0, 6)
        )
        self.control_bindings_card.columnconfigure(0, weight=1, uniform="binding")
        self.control_bindings_card.columnconfigure(1, weight=1, uniform="binding")
        self.control_device_card = ttk.Frame(
            parent,
            style="Liquid.SettingsCard.TFrame",
            padding=(16, 16, 16, 18),
        )
        self.control_device_card.grid(
            row=0, column=1, sticky="nsew", padx=(6, 0)
        )
        self.control_device_card.columnconfigure(0, weight=1)

        ttk.Label(
            self.control_bindings_card,
            text="INPUT BINDINGS",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            self.control_device_card,
            text="DEVICE SETUP",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        device_row = ttk.Frame(
            self.control_device_card,
            style="Liquid.Metric.TFrame",
            padding=(12, 10),
        )
        device_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            device_row, text="MAKCU STATUS", style="Liquid.MetricLabel.TLabel"
        ).pack(anchor="w")
        self.device_label = ttk.Label(
            device_row,
            textvariable=self.device_status_var,
            style="Liquid.MetricValue.TLabel",
            font=(FONT_FAMILY, 11, "bold"),
            anchor="w",
            justify="left",
            wraplength=190,
        )
        self.device_label.pack(anchor="w", fill="x", pady=(5, 0))

        def combo_card(parent, row, column, label, variable, values, width,
                       *, columnspan=1, padx=(0, 0), pady=(0, 10)):
            field, combo = self._dropdown_field(
                parent,
                label=label,
                variable=variable,
                values=values,
                width=width,
            )
            field.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky="ew",
                padx=padx,
                pady=pady,
            )
            return combo

        self.trigger_combo = combo_card(
            self.control_bindings_card, 1, 0, "Trigger", self.trigger_var,
            ("Left", "Right", "Middle", "Mouse4", "Mouse5"), 10,
            padx=(0, 5),
        )
        self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.modifier_combo = combo_card(
            self.control_bindings_card, 1, 1, "Modifier", self.modifier_var,
            ("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"), 10,
            padx=(5, 0),
        )
        self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.source_field = ttk.Frame(
            self.control_device_card,
            style="Liquid.DropdownField.TFrame",
        )
        self.source_field.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.source_field.columnconfigure(0, weight=1, uniform="sources")
        self.source_field.columnconfigure(1, weight=1, uniform="sources")
        ttk.Label(
            self.source_field,
            text="MOTION SOURCES",
            style="Liquid.DropdownLabel.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self.jitter_source_button = ttk.Button(
            self.source_field,
            text="Jitter OFF",
            style="Liquid.Secondary.TButton",
            command=lambda: self.toggle_jitter_source(),
        )
        self.jitter_source_button.grid(
            row=1, column=0, sticky="ew", padx=(0, 3)
        )
        self.ai_source_button = ttk.Button(
            self.source_field,
            text="AI Aim OFF",
            style="Liquid.Secondary.TButton",
            command=lambda: self.toggle_ai_source(),
        )
        self.ai_source_button.grid(
            row=1, column=1, sticky="ew", padx=(3, 0)
        )
        self.preset_combo = combo_card(
            self.control_device_card, 3, 0, "Preset", self.preset_var,
            self.preset_values, 14,
            pady=(10, 0),
        )
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        self.hotkey_button = ttk.Button(
            self.control_bindings_card,
            text=f"Hotkey: {self.hotkey_name_var.get()}",
            style="Liquid.Secondary.TButton",
            command=self.capture_hotkey,
        )
        self.hotkey_button.grid(
            row=2, column=0, columnspan=2, sticky="sew", pady=(12, 0)
        )

        self._build_control_actions(self.control_device_card)

    def _build_control_actions(self, parent: ttk.Frame) -> None:
        self.control_action_row = ttk.Frame(
            parent, style="Liquid.Surface.TFrame"
        )
        self.control_action_row.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.control_action_row.columnconfigure(0, weight=1)
        self.control_action_row.columnconfigure(1, weight=1)
        self.reconnect_button = ttk.Button(
            self.control_action_row,
            text="Reconnect",
            style="Liquid.Secondary.TButton",
            command=self.reconnect,
        )
        self.test_button = ttk.Button(
            self.control_action_row,
            text="Test 3s",
            style="Liquid.Primary.TButton",
            command=self.test_run,
        )
        self.reconnect_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.test_button.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    def _numeric_control(self, parent: tk.Misc, row: int, column: int,
                         label: str, key: str, low: float, high: float,
                         resolution: float = 1.0) -> None:
        block = ttk.Frame(parent, style="Liquid.Surface.TFrame")
        block.grid(row=row, column=column, sticky="ew", padx=5, pady=4)
        parent.columnconfigure(column, weight=1)
        top = ttk.Frame(block, style="Liquid.Surface.TFrame")
        top.pack(fill="x")
        ttk.Label(
            top, text=label.upper(), style="Liquid.CardBody.TLabel"
        ).pack(side="left")
        entry = ttk.Entry(top, textvariable=self.motion_vars[key], width=5,
                          style="Liquid.Entry.TEntry", justify="right")
        entry.pack(side="right")
        slider = LiquidSlider(
            block,
            from_=low,
            to=high,
            resolution=resolution,
            command=lambda value, name=key: self._scale_changed(name, value),
            palette=self._slider_palette(on_surface=True),
        )
        slider.set(float(self.motion_vars[key].get()))
        slider.pack(fill="x", pady=(2, 0))
        setattr(self, f"{key}_entry", entry)
        setattr(self, f"{key}_scale", slider)

    def _ai_numeric_control(
        self,
        parent: tk.Misc,
        row: int,
        column: int,
        label: str,
        key: str,
        low: float,
        high: float,
        resolution: float,
    ) -> None:
        block = ttk.Frame(parent, style="Liquid.Surface.TFrame")
        block.grid(row=row, column=column, sticky="ew", padx=5, pady=4)
        parent.columnconfigure(column, weight=1)
        top = ttk.Frame(block, style="Liquid.Surface.TFrame")
        top.pack(fill="x")
        ttk.Label(
            top, text=label.upper(), style="Liquid.CardBody.TLabel"
        ).pack(side="left")
        entry = ttk.Entry(
            top,
            textvariable=self.ai_vars[key],
            width=5,
            style="Liquid.Entry.TEntry",
            justify="right",
        )
        entry.pack(side="right")
        slider = LiquidSlider(
            block,
            from_=low,
            to=high,
            resolution=resolution,
            command=lambda value, name=key: self._ai_scale_changed(name, value),
            palette=self._slider_palette(on_surface=True),
        )
        slider.set(float(self.ai_vars[key].get()))
        slider.pack(fill="x", pady=(2, 0))
        setattr(self, f"ai_{key}_entry", entry)
        setattr(self, f"ai_{key}_scale", slider)

    def _overlay_numeric_control(
        self,
        parent: tk.Misc,
        *,
        label: str,
        key: str,
        low: int,
        high: int,
    ) -> ttk.Frame:
        block = ttk.Frame(parent, style="Liquid.Surface.TFrame")
        block.columnconfigure(0, weight=1)
        top = ttk.Frame(block, style="Liquid.Surface.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            top,
            text=label.upper(),
            style="Liquid.CardBody.TLabel",
        ).pack(side="left")
        variable = getattr(self, f"overlay_{key}_var")
        entry = ttk.Entry(
            top,
            textvariable=variable,
            width=5,
            justify="right",
            style="Liquid.Entry.TEntry",
        )
        entry.pack(side="right")
        slider = LiquidSlider(
            block,
            from_=low,
            to=high,
            resolution=1,
            command=lambda value, name=key: self._overlay_scale_changed(
                name, value
            ),
            palette=self._slider_palette(on_surface=True),
        )
        slider.set(float(variable.get()))
        slider.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        entry.bind(
            "<FocusOut>",
            lambda _event, name=key: self._overlay_entry_changed(name),
        )
        entry.bind(
            "<Return>",
            lambda _event, name=key: self._overlay_entry_changed(name),
        )
        setattr(self, f"overlay_{key}_entry", entry)
        setattr(self, f"overlay_{key}_scale", slider)
        setattr(self, f"_overlay_{key}_bounds", (low, high))
        return block

    def _dropdown_field(
        self,
        parent: tk.Misc,
        *,
        label: str,
        variable: tk.Variable,
        values: tuple[str, ...],
        width: int | None = None,
    ) -> tuple[ttk.Frame, ttk.Combobox]:
        field = ttk.Frame(parent, style="Liquid.DropdownField.TFrame")
        field.columnconfigure(0, weight=1)
        ttk.Label(
            field,
            text=label.upper(),
            style="Liquid.DropdownLabel.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        options: dict[str, Any] = {
            "textvariable": variable,
            "values": values,
            "state": "readonly",
            "style": "Liquid.Modern.TCombobox",
        }
        if width is not None:
            options["width"] = width
        combo = ttk.Combobox(field, **options)
        combo.grid(row=1, column=0, sticky="ew")
        return field, combo

    def _build_jitter_section(self, parent: ttk.Frame) -> None:
        self.quick_frame = parent
        parent.columnconfigure(0, weight=1)
        self.motion_hero_card = ttk.Frame(parent, style="Liquid.SettingsCard.TFrame",
                                           padding=(18, 16, 18, 18))
        self.motion_hero_card.grid(row=0, column=0, sticky="ew")
        self.motion_hero_card.columnconfigure(0, weight=1)
        ttk.Label(self.motion_hero_card, text="MOTION SHAPE",
                  style="Liquid.CardTitle.TLabel",
                  font=(FONT_FAMILY, 12, "bold")).grid(row=0, column=0, sticky="w")
        self.quick_grid = ttk.Frame(self.motion_hero_card, style="Liquid.Surface.TFrame")
        self.quick_grid.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.quick_grid.columnconfigure(0, weight=1, uniform="quick")
        self.quick_grid.columnconfigure(1, weight=1, uniform="quick")
        for index, control in enumerate((("Pulse Size", "pulse_size_px", 1, 8, 1),
                                         ("Pulse Rate", "pulse_rate_hz", 20, 120, 1))):
            self._numeric_control(self.quick_grid, index // 2, index % 2, *control)
        ramp_row, self.ramp_mode_combo = self._dropdown_field(
            self.quick_grid, label="Ramp Mode",
            variable=self.motion_vars["ramp_mode"], values=RAMP_MODES)
        ramp_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 4))
    def _build_ai_section(self, parent: ttk.Frame) -> None:
        self.ai_settings_card = ttk.Frame(
            parent, style="Liquid.SettingsCard.TFrame", padding=(18, 16, 18, 18)
        )
        self.ai_settings_card.grid(row=0, column=0, sticky="ew")
        self.ai_settings_card.columnconfigure(0, weight=1)
        ttk.Label(
            self.ai_settings_card, text="AI AIM SETTINGS",
            style="Liquid.CardTitle.TLabel", font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ai_controls_grid = ttk.Frame(
            self.ai_settings_card, style="Liquid.Surface.TFrame"
        )
        self.ai_controls_grid.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.ai_controls_grid.columnconfigure(0, weight=1, uniform="ai_controls")
        self.ai_controls_grid.columnconfigure(1, weight=1, uniform="ai_controls")
        for index, (key, spec) in enumerate(_AI_CONTROL_SPECS.items()):
            self._ai_numeric_control(
                self.ai_controls_grid, index // 2, index % 2, spec[0], key,
                spec[1], spec[2], spec[3],
            )
        self.ai_target_row = ttk.Frame(
            self.ai_settings_card, style="Liquid.Surface.TFrame"
        )
        self.ai_target_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self.ai_target_row.columnconfigure(0, weight=1, uniform="ai_target")
        self.ai_target_row.columnconfigure(1, weight=1, uniform="ai_target")
        target_area_field, self.target_area_combo = self._dropdown_field(
            self.ai_target_row, label="Target Area", variable=self.target_area_var,
            values=tuple(_TARGET_AREA_VALUES),
        )
        target_area_field.grid(row=0, column=0, sticky="ew", padx=5)
        self.target_area_combo.bind("<<ComboboxSelected>>", self._target_area_changed)
        capture_mode_field, self.capture_mode_combo = self._dropdown_field(
            self.ai_target_row,
            label="Capture Mode",
            variable=self.capture_mode_var,
            values=tuple(_CAPTURE_MODE_VALUES),
        )
        capture_mode_field.grid(row=0, column=1, sticky="ew", padx=5)
        self.capture_mode_combo.bind(
            "<<ComboboxSelected>>", self._capture_mode_changed
        )
        self.ai_model_frame = ttk.Frame(
            self.ai_settings_card, style="Liquid.Surface.TFrame", padding=(5, 10)
        )
        self.ai_model_frame.grid(row=3, column=0, sticky="ew")
        self.ai_model_frame.columnconfigure(0, weight=1)
        self.ai_model_frame.columnconfigure(1, weight=1)
        ttk.Label(
            self.ai_model_frame, text="MODEL", style="Liquid.CardBody.TLabel"
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            self.ai_model_frame, textvariable=self.ai_model_var,
            style="Liquid.CardText.TLabel", wraplength=300,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 6))
        self.model_browse_button = ttk.Button(
            self.ai_model_frame, text="Browse...", style="Liquid.Secondary.TButton",
            command=self.browse_ai_model,
        )
        self.model_browse_button.grid(row=2, column=0, sticky="ew", padx=(0, 3))
        self.use_default_model_button = ttk.Button(
            self.ai_model_frame, text="Use Default", style="Liquid.Secondary.TButton",
            command=self.use_default_ai_model,
        )
        self.use_default_model_button.grid(row=2, column=1, sticky="ew", padx=(3, 0))
        self._build_ai_curve_card(parent)

    def _build_overlay_section(self, parent: ttk.Frame) -> None:
        self.overlay_custom_window: tk.Toplevel | None = None
        self.overlay_control_card = ttk.Frame(
            parent,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 14, 18, 14),
        )
        self.overlay_control_card.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )
        self.overlay_control_card.columnconfigure(0, weight=1)
        ttk.Label(
            self.overlay_control_card,
            text="OVERLAY VISIBILITY",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.overlay_control_card,
            text="Detection boxes and HUD are independent from motion sources.",
            style="Liquid.CardBody.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        actions = ttk.Frame(
            self.overlay_control_card,
            style="Liquid.SettingsCard.TFrame",
        )
        actions.grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))
        self.overlay_button = ttk.Button(
            actions,
            text="Overlay OFF",
            style="Liquid.Secondary.TButton",
            command=self.toggle_overlay,
        )
        self.overlay_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.overlay_customize_button = ttk.Button(
            actions,
            text="Customize Overlay",
            style="Liquid.Secondary.TButton",
            command=self.open_overlay_customizer,
        )
        self.overlay_customize_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )

    def _overlay_customizer_exists(self) -> bool:
        window = self.overlay_custom_window
        if window is None:
            return False
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            return False

    def _normalize_overlay_numeric_values(self) -> None:
        defaults = {
            "box_width": 2,
            "hud_offset_x": 8,
            "hud_offset_y": 8,
            "hud_font_size": 10,
        }
        for key, default in defaults.items():
            variable = getattr(self, f"overlay_{key}_var")
            low, high = getattr(self, f"_overlay_{key}_bounds")
            try:
                value = int(variable.get())
            except (TypeError, ValueError):
                value = default
            variable.set(str(max(low, min(high, value))))

    def open_overlay_customizer(self) -> None:
        if self._overlay_customizer_exists():
            window = self.overlay_custom_window
            window.deiconify()
            window.lift()
            window.focus_set()
            return

        self._normalize_overlay_numeric_values()
        window = tk.Toplevel(self)
        self.overlay_custom_window = window
        try:
            window.withdraw()
            window.title("Customize Overlay")
            window.geometry("760x520")
            window.resizable(False, False)
            window.transient(self)
            window.configure(background=self._palette["window"])
            window.protocol("WM_DELETE_WINDOW", self.close_overlay_customizer)
            self._build_overlay_customizer_contents(window)
            self._apply_combobox_popup_palette()
            self._render_runtime_controls()
            window.deiconify()
            window.lift()
            window.focus_set()
        except Exception:
            self.overlay_custom_window = None
            try:
                self._cancel_slider_callbacks(window)
                window.destroy()
            except Exception:
                pass
            raise

    def close_overlay_customizer(self) -> None:
        window = self.overlay_custom_window
        self.overlay_custom_window = None
        if window is None:
            return
        try:
            self._cancel_slider_callbacks(window)
            window.destroy()
        except tk.TclError:
            pass

    def _build_overlay_customizer_contents(self, parent: tk.Toplevel) -> None:
        self.overlay_custom_card = ttk.Frame(
            parent,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.overlay_custom_card.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12,
        )
        self.overlay_custom_card.columnconfigure(0, weight=1, uniform="overlay")
        self.overlay_custom_card.columnconfigure(1, weight=1, uniform="overlay")
        ttk.Label(
            self.overlay_custom_card,
            text="OVERLAY CUSTOM",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        overlay_header_actions = ttk.Frame(
            self.overlay_custom_card,
            style="Liquid.SettingsCard.TFrame",
        )
        overlay_header_actions.grid(row=0, column=1, sticky="e")
        overlay_header_actions.columnconfigure(0, weight=1)
        self.overlay_reset_button = ttk.Button(
            overlay_header_actions,
            text="Reset Overlay",
            style="Liquid.Secondary.TButton",
            command=self.reset_overlay_customization,
        )
        self.overlay_reset_button.grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(
            self.overlay_custom_card,
            text=(
                "Box color and Head Boxes are saved; other controls reset "
                "on launch."
            ),
            style="Liquid.CardBody.TLabel",
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(5, 12),
        )
        detection = ttk.Frame(
            self.overlay_custom_card,
            style="Liquid.Surface.TFrame",
        )
        detection.grid(row=2, column=0, sticky="new", padx=(0, 6))
        for column in range(2):
            detection.columnconfigure(column, weight=1, uniform="boxes")
        self.overlay_color_button = ttk.Button(
            detection,
            text=f"Box Color {self.overlay_color.upper()}",
            style="Liquid.Secondary.TButton",
            command=self.choose_overlay_color,
        )
        self.overlay_color_button.grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        self.overlay_head_button = ttk.Button(
            detection,
            text=(
                "Head Boxes ON"
                if self.overlay_head_visible else "Head Boxes OFF"
            ),
            style=(
                "Liquid.Primary.TButton"
                if self.overlay_head_visible else "Liquid.Secondary.TButton"
            ),
            command=self.toggle_overlay_heads,
        )
        self.overlay_head_button.grid(
            row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0)
        )
        self.overlay_player_button = ttk.Button(
            detection,
            text="Player Boxes ON",
            style="Liquid.Primary.TButton",
            command=self.toggle_overlay_players,
        )
        self.overlay_player_button.grid(
            row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0)
        )
        width_control = self._overlay_numeric_control(
            detection,
            label="Box Width",
            key="box_width",
            low=1,
            high=8,
        )
        width_control.grid(
            row=2, column=0, sticky="ew", padx=(0, 6), pady=(10, 0)
        )
        label_field, self.overlay_label_mode_combo = self._dropdown_field(
            detection,
            label="Box Label",
            variable=self.overlay_label_mode_var,
            values=("Off", "Class", "Class + Confidence"),
        )
        label_field.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=(10, 0),
        )
        self.overlay_label_mode_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._overlay_style_changed(),
        )

        hud = ttk.Frame(
            self.overlay_custom_card,
            style="Liquid.Surface.TFrame",
        )
        hud.grid(row=2, column=1, sticky="new", padx=(6, 0))
        hud.columnconfigure(0, weight=1)
        hud.columnconfigure(1, weight=1)
        corner_field, self.overlay_hud_corner_combo = self._dropdown_field(
            hud,
            label="HUD Corner",
            variable=self.overlay_hud_corner_var,
            values=("Top Left", "Top Right", "Bottom Left", "Bottom Right"),
        )
        corner_field.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.overlay_hud_corner_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._overlay_style_changed(),
        )
        for column, (label, key, low, high) in enumerate((
            ("HUD X Offset", "hud_offset_x", 0, 500),
            ("HUD Y Offset", "hud_offset_y", 0, 500),
        )):
            control = self._overlay_numeric_control(
                hud,
                label=label,
                key=key,
                low=low,
                high=high,
            )
            control.grid(
                row=1,
                column=column,
                sticky="ew",
                padx=(0, 5) if column == 0 else (5, 0),
                pady=(10, 0),
            )
        font_control = self._overlay_numeric_control(
            hud,
            label="HUD Font Size",
            key="hud_font_size",
            low=8,
            high=24,
        )
        font_control.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )
        hud_actions = ttk.Frame(hud, style="Liquid.Surface.TFrame")
        hud_actions.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )
        hud_actions.columnconfigure(0, weight=1)
        hud_actions.columnconfigure(1, weight=1)
        self.overlay_hud_button = ttk.Button(
            hud_actions,
            text="HUD ON",
            style="Liquid.Primary.TButton",
            command=self.toggle_overlay_hud,
        )
        self.overlay_hud_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        self.overlay_hud_color_button = ttk.Button(
            hud_actions,
            text=f"HUD Color {self.overlay_hud_color.upper()}",
            style="Liquid.Secondary.TButton",
            command=self.choose_overlay_hud_color,
        )
        self.overlay_hud_color_button.grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        metrics = ttk.Frame(hud, style="Liquid.Surface.TFrame")
        metrics.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        for index, (label, key) in enumerate((
            ("FPS", "fps"),
            ("Provider", "provider"),
            ("Zoom", "zoom"),
            ("Lock", "lock"),
        )):
            button = ttk.Button(
                metrics,
                text=f"{label} ON",
                style="Liquid.Primary.TButton",
                command=lambda name=key: self.toggle_overlay_hud_metric(name),
            )
            button.grid(
                row=index // 2,
                column=index % 2,
                sticky="ew",
                padx=(0, 4) if index % 2 == 0 else (4, 0),
                pady=(0, 4) if index < 2 else (4, 0),
            )
            setattr(self, f"overlay_hud_{key}_button", button)

    def _build_ai_curve_card(self, parent: ttk.Frame) -> None:
        self.ai_curve_card = ttk.Frame(
            parent,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.ai_curve_card.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(8, 0),
        )
        self.ai_curve_card.columnconfigure(0, weight=1)

        heading = ttk.Frame(
            self.ai_curve_card, style="Liquid.SettingsCard.TFrame"
        )
        heading.grid(row=0, column=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading,
            text="AI RESPONSE CURVE",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ai_curve_reset_button = ttk.Button(
            heading,
            text="Reset Curve",
            style="Liquid.Secondary.TButton",
            command=self._reset_ai_curve,
        )
        self.ai_curve_reset_button.grid(row=0, column=1, sticky="e")
        self.ai_curve_canvas = tk.Canvas(
            self.ai_curve_card,
            height=132,
            background=self._palette["raised"],
            highlightthickness=0,
            borderwidth=0,
            takefocus=False,
        )
        self.ai_curve_canvas.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.ai_curve_canvas.bind(
            "<Configure>", self._redraw_ai_curve, add="+"
        )
        self.ai_curve_canvas.bind("<B1-Motion>", self._curve_dragged, add="+")
        self.ai_curve_canvas.bind(
            "<ButtonRelease-1>",
            lambda _event: self._curve_drag_ended(),
            add="+",
        )

        exact = ttk.Frame(
            self.ai_curve_card, style="Liquid.SettingsCard.TFrame"
        )
        exact.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for column in range(5):
            exact.columnconfigure(column, weight=1, uniform="curve_exact")
        fixed = ttk.Frame(exact, style="Liquid.SettingsCard.TFrame")
        fixed.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Label(
            fixed, text="0% DISTANCE", style="Liquid.CardBody.TLabel"
        ).pack(anchor="center")
        ttk.Label(
            fixed,
            text="0%",
            style="Liquid.CardText.TLabel",
            font=(FONT_FAMILY, 11, "bold"),
        ).pack(anchor="center", pady=(4, 0))

        self.ai_curve_entries: dict[int, ttk.Entry] = {}
        for index, distance in enumerate((25, 50, 75, 100), start=1):
            block = ttk.Frame(exact, style="Liquid.SettingsCard.TFrame")
            block.grid(row=0, column=index, sticky="ew", padx=5)
            ttk.Label(
                block,
                text=f"{distance}% DISTANCE",
                style="Liquid.CardBody.TLabel",
            ).pack(anchor="center")
            entry = ttk.Entry(
                block,
                textvariable=self.ai_curve_vars[index],
                width=5,
                style="Liquid.Entry.TEntry",
                justify="right",
            )
            entry.pack(anchor="center", pady=(4, 0))
            entry.bind(
                "<FocusOut>",
                lambda _event, node=index: self._curve_entry_changed(node),
            )
            entry.bind(
                "<Return>",
                lambda _event, node=index: self._curve_entry_changed(node),
            )
            self.ai_curve_entries[index] = entry
        self._redraw_ai_curve()

    def _ai_curve_canvas_alive(self) -> bool:
        canvas = getattr(self, "ai_curve_canvas", None)
        if canvas is None:
            return False
        try:
            return bool(canvas.winfo_exists())
        except tk.TclError:
            if self._closing or self._closed:
                return False
            raise

    def _curve_plot_bounds(self) -> tuple[float, float, float, float]:
        canvas = self.ai_curve_canvas
        width = max(280, canvas.winfo_width(), canvas.winfo_reqwidth())
        height = max(140, canvas.winfo_height(), canvas.winfo_reqheight())
        return 42.0, 16.0, float(width - 18), float(height - 28)

    def _redraw_ai_curve(self, _event: tk.Event | None = None) -> None:
        if self._closing or not self._ai_curve_canvas_alive():
            return
        canvas = self.ai_curve_canvas
        p = self._palette
        try:
            canvas.configure(background=p["raised"])
            canvas.delete("all")
            left, top, right, bottom = self._curve_plot_bounds()
            plot_width = right - left
            plot_height = bottom - top

            for percent in (0, 25, 50, 75, 100):
                y = bottom - (percent / 100.0) * plot_height
                canvas.create_line(
                    left,
                    y,
                    right,
                    y,
                    fill=p["border"],
                    width=1,
                    tags=("ai-curve-grid",),
                )
                canvas.create_text(
                    left - 7,
                    y,
                    text=str(percent),
                    fill=p["muted"],
                    font=SMALL_FONT,
                    anchor="e",
                    tags=("ai-curve-label",),
                )
                x = left + (percent / 100.0) * plot_width
                canvas.create_line(
                    x,
                    top,
                    x,
                    bottom,
                    fill=p["border"],
                    width=1,
                    tags=("ai-curve-grid",),
                )
                canvas.create_text(
                    x,
                    bottom + 13,
                    text=str(percent),
                    fill=p["muted"],
                    font=SMALL_FONT,
                    anchor="n",
                    tags=("ai-curve-label",),
                )

            curve = self.get_ai_settings().response_curve
            sample_points: list[float] = []
            for sample in range(65):
                normalized_x = sample / 64.0
                normalized_y = response_curve_value(curve, normalized_x)
                sample_points.extend((
                    left + normalized_x * plot_width,
                    bottom - normalized_y * plot_height,
                ))
            canvas.create_line(
                *sample_points,
                fill=p["accent"],
                width=3,
                tags=("ai-curve-sample",),
            )

            for index, value in enumerate(curve):
                x = left + (index / 4.0) * plot_width
                y = bottom - value * plot_height
                node_tag = f"ai-curve-node-{index}"
                if index:
                    canvas.create_oval(
                        x - 10,
                        y - 10,
                        x + 10,
                        y + 10,
                        fill=p["raised"],
                        outline="",
                        tags=("ai-curve-hit", node_tag),
                    )
                canvas.create_oval(
                    x - 5,
                    y - 5,
                    x + 5,
                    y + 5,
                    fill=p["accent" if index else "muted"],
                    outline=p["text"],
                    width=1,
                    tags=("ai-curve-node", node_tag),
                )
                if index:
                    canvas.tag_bind(
                        node_tag,
                        "<ButtonPress-1>",
                        lambda _event, node=index: self._curve_drag_started(node),
                    )
        except tk.TclError:
            if self._closing or not self._ai_curve_canvas_alive():
                return
            raise

    def _curve_entry_validation(
        self,
        changed_index: int,
    ) -> tuple[
        tuple[float, float, float, float, float] | None,
        set[int],
    ]:
        percentages: list[int | None] = [0]
        invalid: set[int] = set()
        for index in range(1, 5):
            raw = self.ai_curve_vars[index].get().strip()
            try:
                value = int(raw)
                if not 0 <= value <= 100:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                value = None
                invalid.add(index)
            percentages.append(value)
        for index in range(1, 5):
            previous = percentages[index - 1]
            current = percentages[index]
            if previous is None or current is None or current >= previous:
                continue
            owner = self._curve_order_error_owners.get(index)
            if owner is None:
                owner = (
                    changed_index
                    if changed_index in {index - 1, index} and changed_index > 0
                    else index
                )
            self._curve_order_error_owners[index] = owner
            invalid.add(owner)
        self._curve_order_error_owners = {
            index: owner
            for index, owner in self._curve_order_error_owners.items()
            if (
                percentages[index - 1] is not None
                and percentages[index] is not None
                and percentages[index] < percentages[index - 1]
            )
        }
        if invalid:
            return None, invalid
        curve = validated_response_curve(
            tuple(value / 100.0 for value in percentages)
        )
        return curve, set()

    def _current_ai_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            name: variable.get() for name, variable in self.ai_vars.items()
        }
        mapping["response_curve"] = list(
            self.get_ai_settings().response_curve
        )
        mapping["target_area"] = _TARGET_AREA_VALUES.get(
            self.target_area_var.get(), "head"
        )
        return mapping

    def _target_area_changed(self, _event: tk.Event | None = None) -> None:
        if self._closing:
            return
        target_area = _TARGET_AREA_VALUES.get(
            self.target_area_var.get(), "head"
        )
        self._replace_ai_snapshot(
            replace(self.get_ai_settings(), target_area=target_area)
        )
        self._invalidate_trigger_lock_epoch()
        self.ai_zoom_var.set("1.0×")

    def _curve_entry_changed(self, index: int) -> None:
        if self._updating_ai_curve_controls or self._closing:
            return
        curve, invalid = self._curve_entry_validation(index)
        if invalid:
            for node, entry in self.ai_curve_entries.items():
                entry.configure(
                    style=(
                        "Liquid.Invalid.TEntry"
                        if node in invalid else "Liquid.Entry.TEntry"
                    )
                )
            self.footer_var.set(
                "Response curve must use ordered whole percentages from 0 to 100"
            )
            return
        assert curve is not None
        for entry in self.ai_curve_entries.values():
            entry.configure(style="Liquid.Entry.TEntry")
        if self.footer_var.get().startswith("Response curve must "):
            self.footer_var.set("Ready")
        self._replace_ai_snapshot(
            replace(self.get_ai_settings(), response_curve=curve)
        )
        self._redraw_ai_curve()
        self._schedule_save()

    def _reset_ai_curve(self) -> None:
        if self._closing:
            return
        self._curve_order_error_owners.clear()
        self._updating_ai_curve_controls = True
        try:
            for index, value in enumerate(DEFAULT_RESPONSE_CURVE[1:], start=1):
                self.ai_curve_vars[index].set(str(int(round(value * 100.0))))
        finally:
            self._updating_ai_curve_controls = False
        for entry in self.ai_curve_entries.values():
            entry.configure(style="Liquid.Entry.TEntry")
        self._replace_ai_snapshot(
            replace(
                self.get_ai_settings(),
                response_curve=DEFAULT_RESPONSE_CURVE,
            )
        )
        self.footer_var.set("Response curve reset")
        self._redraw_ai_curve()
        self._schedule_save()

    def _curve_drag_started(self, index: int) -> None:
        if (
            self._closing
            or index not in self.ai_curve_vars
            or not self._ai_curve_canvas_alive()
        ):
            return
        self._curve_drag_index = index

    def _curve_dragged(self, event: Any) -> None:
        index = self._curve_drag_index
        if (
            index is None
            or self._closing
            or not self._ai_curve_canvas_alive()
        ):
            return
        _left, top, _right, bottom = self._curve_plot_bounds()
        y = max(top, min(bottom, float(event.y)))
        percentage = int(round((bottom - y) * 100.0 / (bottom - top)))
        curve = self.get_ai_settings().response_curve
        lower = int(math.ceil(curve[index - 1] * 100.0))
        upper = (
            int(math.floor(curve[index + 1] * 100.0))
            if index < 4 else 100
        )
        percentage = max(lower, min(upper, percentage))
        self._updating_ai_curve_controls = True
        try:
            self.ai_curve_vars[index].set(str(percentage))
        finally:
            self._updating_ai_curve_controls = False
        self._curve_entry_changed(index)

    def _curve_drag_ended(self) -> None:
        self._curve_drag_index = None

    def _cancel_ai_curve_callbacks(self) -> None:
        self._curve_drag_index = None
        if not self._ai_curve_canvas_alive():
            return
        try:
            self.ai_curve_canvas.unbind("<Configure>")
            self.ai_curve_canvas.unbind("<B1-Motion>")
            self.ai_curve_canvas.unbind("<ButtonRelease-1>")
            for index in range(1, 5):
                node_tag = f"ai-curve-node-{index}"
                self.ai_curve_canvas.tag_unbind(node_tag, "<ButtonPress-1>")
        except tk.TclError:
            if self._closing or not self._ai_curve_canvas_alive():
                return
            raise

    def _refresh_dashboard_scrollregion(
        self, _event: tk.Event | None = None
    ) -> None:
        self.dashboard_scroll_canvas.configure(
            scrollregion=self.dashboard_scroll_canvas.bbox("all")
        )

    def _resize_dashboard_content(self, event: tk.Event) -> None:
        self.dashboard_scroll_canvas.itemconfigure(
            self._dashboard_scroll_window,
            width=max(1, int(event.width)),
        )

    def _scroll_dashboard(self, event: tk.Event) -> str | None:
        widget = event.widget
        while widget is not None:
            if widget is self.dashboard_frame:
                break
            widget = getattr(widget, "master", None)
        else:
            return None
        delta = int(getattr(event, "delta", 0))
        if delta:
            self.dashboard_scroll_canvas.yview_scroll(
                -1 if delta > 0 else 1, "units"
            )
        return "break"

    def _build_settings_section(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        self.settings_content = parent
        self.settings_content.columnconfigure(0, weight=3, uniform="settings")
        self.settings_content.columnconfigure(1, weight=2, uniform="settings")
        self.settings_content.rowconfigure(0, weight=1)

        self.sound_feedback_card = ttk.Frame(
            self.settings_content,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.sound_feedback_card.grid(
            row=0, column=0, sticky="nsew", padx=(0, 6)
        )
        self.sound_feedback_card.columnconfigure(0, weight=1)
        self.sound_feedback_card.rowconfigure(5, weight=1)
        ttk.Label(
            self.sound_feedback_card,
            text="SOUND FEEDBACK",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        sound_toggle_row = ttk.Frame(
            self.sound_feedback_card, style="Liquid.Surface.TFrame"
        )
        sound_toggle_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        sound_toggle_row.columnconfigure(0, weight=1)
        ttk.Label(
            sound_toggle_row,
            text="HOTKEY CUES",
            style="Liquid.CardBody.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.sound_enabled_check = ttk.Checkbutton(
            sound_toggle_row,
            text="Enabled",
            variable=self.sound_enabled_var,
            command=self.apply_sound_settings,
            style="Liquid.Surface.TCheckbutton",
        )
        self.sound_enabled_check.grid(row=0, column=1, sticky="e")
        ttk.Separator(
            self.sound_feedback_card, orient="horizontal"
        ).grid(row=2, column=0, sticky="ew", pady=12)

        volume_control = ttk.Frame(
            self.sound_feedback_card, style="Liquid.Surface.TFrame"
        )
        volume_control.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        volume_control.columnconfigure(0, weight=1)
        self.sound_volume_entry = ttk.Entry(
            volume_control,
            textvariable=self.sound_volume_var,
            width=5,
            justify="right",
            style="Liquid.Entry.TEntry",
        )
        self.sound_volume_entry.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.sound_volume_scale = LiquidSlider(
            volume_control,
            from_=0,
            to=100,
            resolution=1,
            command=self._sound_volume_changed,
            palette=self._slider_palette(on_surface=True),
        )
        self.sound_volume_scale.set(float(self.config.sound_volume))
        self.sound_volume_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            volume_control, text="0", style="Liquid.CardBody.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(1, 0))
        ttk.Label(
            volume_control, text="100", style="Liquid.CardBody.TLabel"
        ).grid(row=1, column=0, sticky="e", pady=(1, 0))

        self.sound_preview_card = ttk.Frame(
            self.settings_content,
            style="Liquid.SettingsCard.TFrame",
            padding=(16, 14, 16, 16),
        )
        self.sound_preview_card.grid(
            row=0, column=1, sticky="nsew", padx=(6, 0)
        )
        self.sound_preview_card.columnconfigure(0, weight=1)
        self.sound_preview_card.rowconfigure(2, weight=1)
        ttk.Label(
            self.sound_preview_card,
            text="PREVIEW",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(
            self.sound_preview_card, style="Liquid.Surface.TFrame"
        )
        actions.grid(row=1, column=0, sticky="new", pady=(8, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Label(
            actions,
            text="ARMED CUE",
            style="Liquid.CardBody.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.test_on_button = ttk.Button(
            actions,
            text="Play Armed Cue",
            style="Liquid.CompactPrimary.TButton",
            command=lambda: self.preview_sound(True),
        )
        self.test_on_button.grid(row=0, column=1, sticky="e")
        ttk.Separator(actions, orient="horizontal").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=10
        )
        ttk.Label(
            actions,
            text="DISABLED CUE",
            style="Liquid.CardBody.TLabel",
        ).grid(row=2, column=0, sticky="w")
        self.test_off_button = ttk.Button(
            actions,
            text="Play Disabled Cue",
            style="Liquid.CompactSecondary.TButton",
            command=lambda: self.preview_sound(False),
        )
        self.test_off_button.grid(row=2, column=1, sticky="e")
        self.theme_button = ttk.Button(
            actions,
            text=(
                "Switch to Light Mode" if self._theme == "dark"
                else "Switch to Dark Mode"
            ),
            style="Liquid.Secondary.TButton",
            command=self.toggle_theme,
        )
        self.theme_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_footer(self) -> None:
        self.footer_frame = ttk.Frame(
            self.console_workspace, style="Liquid.App.TFrame"
        )
        self.footer_frame.grid(
            row=2, column=0, sticky="ew", pady=(0, 2)
        )
        self.footer_label = ttk.Label(
            self.footer_frame,
            textvariable=self.footer_var,
            style="Liquid.Muted.TLabel",
            anchor="w",
        )
        self.footer_label.pack(side="left", fill="x", expand=True)

    # ---- shell interactions -------------------------------------------

    def _redraw_shell_art(self, _event: tk.Event | None = None) -> None:
        if not hasattr(self, "shell"):
            return
        self.shell.delete("shell-art")
        width = max(self.shell.winfo_width(), self.shell.winfo_reqwidth())
        height = max(self.shell.winfo_height(), self.shell.winfo_reqheight())
        shell_left = self.shell.winfo_rootx()
        shell_top = self.shell.winfo_rooty()
        workspace = getattr(self, "console_workspace", None)
        workspace_left = 0
        if workspace is not None:
            workspace_left = max(
                0, workspace.winfo_rootx() - shell_left
            )
        bands = _BACKGROUND_BANDS[self._theme]
        for index, color in enumerate(bands):
            top = round(height * index / len(bands))
            bottom = round(height * (index + 1) / len(bands))
            self.shell.create_rectangle(
                workspace_left,
                top,
                width,
                bottom,
                fill=color,
                outline="",
                tags=(
                    "shell-art",
                    "background-band",
                    "workspace-band",
                    f"background-band-{index}",
                    f"workspace-band-{index}",
                ),
            )

        p = self._palette
        for name, attribute in (
            ("topbar", "topbar_frame"),
            ("dashboard", "dashboard_frame"),
            ("runtime", "runtime_frame"),
        ):
            panel = getattr(self, attribute, None)
            if panel is None:
                continue
            left = panel.winfo_rootx() - shell_left - 4
            top = panel.winfo_rooty() - shell_top - 4
            right = left + max(panel.winfo_width(), panel.winfo_reqwidth()) + 8
            bottom = top + max(panel.winfo_height(), panel.winfo_reqheight()) + 8
            if right <= left or bottom <= top:
                continue
            radius = min(18, max(4, (bottom - top) // 3))
            points = (
                left + radius, top, right - radius, top,
                right, top, right, top + radius,
                right, bottom - radius, right, bottom,
                right - radius, bottom, left + radius, bottom,
                left, bottom, left, bottom - radius,
                left, top + radius, left, top,
            )
            panel_tags = (
                "shell-art",
                "rounded-surface",
                "floating-panel",
                f"floating-panel-{name}",
            )
            self.shell.create_polygon(
                points,
                smooth=True,
                splinesteps=24,
                fill=p["surface"],
                outline=p["border"],
                width=1,
                tags=panel_tags,
            )
        self.shell.tag_lower("shell-art")

    def _redraw_connection_indicator(self) -> None:
        indicator = getattr(self, "connection_indicator", None)
        if indicator is None:
            return
        state = self.connection_status_var.get().lower()
        role = {
            "connected": "green",
            "connecting": "amber",
        }.get(state, "red")
        p = self._palette
        indicator.configure(background=p["surface"])
        indicator.delete("all")
        state_tag = f"status-{state if state in {'connected', 'connecting'} else 'disconnected'}"
        indicator.create_oval(
            1,
            1,
            17,
            17,
            fill=p[f"{role}_glow"],
            outline="",
            tags=("status-glow", state_tag),
        )
        indicator.create_oval(
            5,
            5,
            13,
            13,
            fill=p[role],
            outline=p["raised"],
            width=1,
            tags=("status-marker", state_tag),
        )
        indicator.create_oval(
            7,
            6,
            9,
            8,
            fill=p["raised"],
            outline="",
            tags=("status-highlight", state_tag),
        )

    def _set_connection_state(self, state: str) -> None:
        self.connection_status_var.set(state)
        self.connection_label.configure(style=f"Liquid.Status{state}.TLabel")
        self._redraw_connection_indicator()
        self._refresh_section_summaries()

    def _set_section_summary(
        self, variable: tk.StringVar, *parts: object
    ) -> None:
        try:
            variable.set(_compact_section_summary(*parts))
        except (tk.TclError, RuntimeError, TypeError, ValueError):
            logging.debug("Could not refresh dashboard summary", exc_info=True)

    def _refresh_section_summaries(self) -> None:
        if self._closing:
            return
        try:
            source_names = [
                name
                for selected, name in (
                    (self.jitter_selected, "Jitter"),
                    (self.ai_selected, "AI Aim"),
                )
                if selected
            ]
            sources = " + ".join(source_names) or "No sources"
            self._set_section_summary(
                self.control_section_summary_var,
                sources,
                self.trigger_var.get(),
                self.preset_var.get(),
                self.connection_status_var.get(),
            )

            aim = self.get_ai_settings()
            target = _TARGET_AREA_LABELS.get(aim.target_area, "Head")
            capture_mode = _CAPTURE_MODE_LABELS[self._capture_mode]
            strength = f"Strength {_display_value(aim.aim_strength)}"
            ai_details = _compact_section_summary(target, capture_mode, strength)
            model_limit = max(3, 72 - len(ai_details) - 3)
            model = _compact_section_summary(
                self.ai_model_var.get(), limit=model_limit
            )
            self._set_section_summary(
                self.ai_section_summary_var,
                model,
                target,
                capture_mode,
                strength,
            )

            box_names = [
                name
                for visible, name in (
                    (self.overlay_head_visible, "Head"),
                    (self.overlay_player_visible, "Player"),
                )
                if visible
            ]
            boxes = " + ".join(box_names) or "No boxes"
            hud = (
                f"HUD {self.overlay_hud_corner_var.get()}"
                if self.overlay_hud_visible else "HUD Off"
            )
            self._set_section_summary(
                self.overlay_section_summary_var,
                f"Overlay {'On' if self.overlay_visible else 'Off'}",
                boxes,
                hud,
            )

            try:
                volume = max(0, min(100, int(self.sound_volume_var.get())))
            except (TypeError, ValueError):
                volume = self.config.sound_volume
            self._set_section_summary(
                self.settings_section_summary_var,
                f"Sound {'On' if self.sound_enabled_var.get() else 'Off'}",
                f"{volume}%",
                self.theme_var.get().title(),
            )
        except (tk.TclError, RuntimeError, TypeError, ValueError):
            logging.debug("Could not refresh dashboard summaries", exc_info=True)

    def _set_test_button_enabled(self, enabled: bool) -> None:
        self.test_button.configure(state="normal" if enabled else "disabled")

    # ---- runtime wiring -----------------------------------------------

    def _install_runtime_bindings(self) -> None:
        """Install variable/widget callbacks after the shell exists."""
        for key, variable in self.motion_vars.items():
            variable.trace_add("write", lambda *_args, name=key: self._motion_changed(name))
            entry = getattr(self, f"{key}_entry", None)
            if entry is not None:
                entry.bind("<FocusOut>", lambda _event, name=key: self._motion_changed(name))
                entry.bind("<Return>", lambda _event, name=key: self._motion_changed(name))
        for key, variable in self.ai_vars.items():
            variable.trace_add(
                "write", lambda *_args, name=key: self._ai_changed(name)
            )
            entry = getattr(self, f"ai_{key}_entry", None)
            if entry is not None:
                entry.bind(
                    "<FocusOut>",
                    lambda _event, name=key: self._ai_changed(name),
                )
                entry.bind(
                    "<Return>",
                    lambda _event, name=key: self._ai_changed(name),
                )
        self.sound_volume_entry.bind(
            "<FocusOut>", lambda _event: self.apply_sound_settings()
        )
        self.sound_volume_entry.bind(
            "<Return>", lambda _event: self.apply_sound_settings()
        )

    def _sound_volume_changed(self, value: str) -> None:
        self.sound_volume_var.set(str(int(round(float(value)))))
        self.apply_sound_settings()

    def apply_sound_settings(self) -> None:
        try:
            volume = int(self.sound_volume_var.get())
        except (TypeError, ValueError):
            volume = self.config.sound_volume
        volume = max(0, min(100, volume))
        self.sound_volume_var.set(str(volume))
        self.sound_volume_scale.set(volume)
        self.sound_player.configure(
            enabled=self.sound_enabled_var.get(),
            volume=volume,
        )
        self._schedule_save()
        self._refresh_section_summaries()

    def preview_sound(self, enabled: bool) -> None:
        self.apply_sound_settings()
        self.sound_player.play(bool(enabled), force=True)

    def _scale_changed(self, key: str, value: str) -> None:
        if self._updating_motion_controls:
            return
        self._updating_motion_controls = True
        try:
            self.motion_vars[key].set(value)
        finally:
            self._updating_motion_controls = False
        self._motion_changed(key)

    def _ai_scale_changed(self, key: str, value: str) -> None:
        if self._updating_ai_controls:
            return
        self._updating_ai_controls = True
        try:
            self.ai_vars[key].set(value)
        finally:
            self._updating_ai_controls = False
        self._ai_changed(key)

    def _overlay_scale_changed(self, key: str, value: str) -> None:
        variable = getattr(self, f"overlay_{key}_var")
        variable.set(str(int(round(float(value)))))
        self._overlay_style_changed()

    def _overlay_entry_changed(self, key: str) -> None:
        variable = getattr(self, f"overlay_{key}_var")
        low, high = getattr(self, f"_overlay_{key}_bounds")
        defaults = {
            "box_width": 2,
            "hud_offset_x": 8,
            "hud_offset_y": 8,
            "hud_font_size": 10,
        }
        try:
            value = int(variable.get())
        except (TypeError, ValueError):
            value = defaults[key]
        value = max(low, min(high, value))
        variable.set(str(value))
        getattr(self, f"overlay_{key}_scale").set(value)
        self._overlay_style_changed()

    def _overlay_style_changed(self) -> None:
        self.footer_var.set("Overlay customization updated")
        self._refresh_section_summaries()

    def _overlay_int_value(self, key: str, default: int) -> int:
        try:
            value = int(getattr(self, f"overlay_{key}_var").get())
        except (TypeError, ValueError):
            value = default
        low, high = getattr(self, f"_overlay_{key}_bounds")
        return max(low, min(high, value))

    def _bindings_event(self, _event: tk.Event | None = None) -> None:
        self.on_bindings_changed()

    def _set_runtime_state(self, state: str) -> None:
        self.runtime_status_var.set(_RUNTIME_STATE_LABELS[state])

    def start_runtime(self) -> None:
        if self._runtime_started or self._closing:
            return
        self._runtime_started = True
        self.hotkey_watcher.start()
        self.service.connect()
        if self.load_outcome.warning:
            self.footer_var.set(self.load_outcome.warning)

    def reconnect(self) -> None:
        if self._closing:
            return
        if not self._runtime_started:
            self._runtime_started = True
            self.hotkey_watcher.start()
        # MakcuService.reconnect() signals its motion generation immediately
        # and crosses the blocking move barrier on its daemon lifecycle
        # worker.  Keep Tk responsible only for local/AI invalidation here.
        self.emergency_stop(
            "Reconnect requested",
            stop_device_motion=False,
        )
        try:
            self.service.reconnect()
            self.footer_var.set("Connecting to Makcu...")
        except Exception as exc:
            self.footer_var.set(f"Reconnect failed: {type(exc).__name__}")

    @property
    def enabled(self) -> bool:
        """Read-only compatibility view of the authoritative Master state."""
        return self.master_armed

    def _selected_sources(self) -> MotionSources:
        return MotionSources(self.jitter_selected, self.ai_selected)

    @staticmethod
    def _model_label(choice: ModelChoice) -> str:
        prefix = "Default" if choice.is_default else "Custom"
        label = f"{prefix} · {choice.display_name}"
        if choice.input_size is not None:
            label += f" · {choice.input_size}×{choice.input_size}"
        return label

    def _model_changes_unavailable(self) -> bool:
        return (
            self._model_switch is not None
            or self._model_start_pending is not None
            or self._capture_mode_switching
            or self._capture_restart_pending
            or self._deferred_ai_start is not None
            or self._motion_mode in _TEST_MOTION_MODES
            or self._closing
        )

    def _render_model_controls(self) -> None:
        busy = self._model_switch is not None
        testing = self._motion_mode in _TEST_MOTION_MODES
        disabled = (
            busy
            or self._model_start_pending is not None
            or self._capture_mode_switching
            or self._capture_restart_pending
            or self._deferred_ai_start is not None
            or testing
            or self._closing
        )
        self.model_browse_button.configure(
            state="disabled" if disabled else "normal"
        )
        self.use_default_model_button.configure(
            state=(
                "disabled"
                if disabled or self._model_choice.is_default
                else "normal"
            )
        )
        self._render_capture_mode_control()
        self._refresh_section_summaries()

    def _render_capture_mode_control(self) -> None:
        disabled = (
            self._closing
            or self._motion_mode in _TEST_MOTION_MODES
            or self._model_switch is not None
            or self._capture_mode_switching
            or self._capture_restart_pending
            or self._deferred_ai_start is not None
            or (self._ai_runtime_active and not self._ai_ready)
        )
        self.capture_mode_combo.configure(
            state="disabled" if disabled else "readonly"
        )

    def _capture_mode_changed(self, _event: tk.Event | None = None) -> None:
        label = self.capture_mode_var.get()
        mode = _CAPTURE_MODE_VALUES.get(label)
        try:
            mode = validated_capture_mode(mode)
        except ValueError:
            self.capture_mode_var.set(_CAPTURE_MODE_LABELS[self._capture_mode])
            return
        guarded = (
            self._closing
            or self._motion_mode in _TEST_MOTION_MODES
            or self._model_switch is not None
            or self._model_start_pending is not None
            or self._capture_mode_switching
            or self._capture_restart_pending
            or self._deferred_ai_start is not None
            or (self._ai_runtime_active and not self._ai_ready)
        )
        if guarded:
            self.capture_mode_var.set(_CAPTURE_MODE_LABELS[self._capture_mode])
            self._render_capture_mode_control()
            return
        if mode == self._capture_mode:
            return
        self._capture_mode = mode
        if not self._ai_runtime_active:
            self._render_capture_mode_control()
            self._refresh_section_summaries()
            return

        self._capture_mode_switching = True
        self._capture_restart_pending = False
        self.footer_var.set(f"Switching AI capture to {label}...")
        self._render_model_controls()
        self._refresh_section_summaries()
        if not self._invalidate_trigger_lock_epoch():
            try:
                self._stop_ai_runtime("Capture mode change failed")
            except Exception:
                logging.exception(
                    "AI runtime cleanup failed after targeting reset error"
                )
            self._fail_capture_mode_switch()
            return
        try:
            self._stop_ai_runtime("Capture mode changed")
        except Exception:
            logging.exception("AI runtime stop failed during capture switch")
            self._fail_capture_mode_switch()
            return
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        self.ai_status_var.set("Loading")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        self.ai_zoom_var.set("1.0\N{MULTIPLICATION SIGN}")
        self._capture_restart_pending = True
        self._continue_capture_mode_switch()

    def _continue_capture_mode_switch(self) -> None:
        if not self._capture_restart_pending:
            return
        if self._closing or not self._ai_runtime_required():
            self._capture_restart_pending = False
            return
        try:
            worker_active = bool(self.ai_service.worker_active)
        except Exception:
            logging.exception("AI worker retirement probe failed")
            self._fail_capture_mode_switch()
            return
        if worker_active:
            return
        self._capture_restart_pending = False
        if self._start_ai_runtime(
            "Capture mode changed",
            capture_mode=self._capture_mode,
            lifecycle_kind="capture",
        ):
            return
        self._fail_capture_mode_switch()

    def _continue_deferred_ai_start(self) -> None:
        capture_transition = (
            self._capture_mode_switching or self._capture_restart_pending
        )
        model_transition = (
            self._model_start_pending is not None
            or (
                self._model_switch is not None
                and self._model_switch.phase
                in {"starting_candidate", "starting_rollback"}
            )
        )
        if (
            (capture_transition and model_transition)
            or (
                self._deferred_ai_start is not None
                and (capture_transition or model_transition)
            )
        ):
            logging.error("Conflicting AI lifecycle starts were rejected")
            self._capture_restart_pending = False
            self._capture_mode_switching = False
            self._model_start_pending = None
            self._deferred_ai_start = None
            self._handle_ai_runtime_error("Conflicting AI lifecycle transitions")
            return
        self._continue_capture_mode_switch()
        self._continue_model_start()
        self._continue_general_ai_start()

    def _general_ai_start_request(self, context: str) -> _DeferredAiStart:
        is_test = (
            self._motion_mode in {"test_ai_loading", "test_combined_loading"}
            and self._test_pending_generation is not None
        )
        return _DeferredAiStart(
            context=context,
            model_choice=self._model_choice,
            capture_mode=self._capture_mode,
            lifecycle_epoch=self._ai_event_epoch,
            kind="test" if is_test else "normal",
            test_generation=(
                self._test_pending_generation if is_test else None
            ),
        )

    def _general_ai_start_is_current(self, request: _DeferredAiStart) -> bool:
        if (
            self._closing
            or request.lifecycle_epoch != self._ai_event_epoch
            or not self._ai_runtime_required()
            or request.model_choice != self._model_choice
            or request.capture_mode != self._capture_mode
            or self._capture_mode_switching
            or self._capture_restart_pending
            or self._model_switch is not None
            or self._model_start_pending is not None
        ):
            return False
        if request.kind == "test":
            return (
                request.test_generation is not None
                and request.test_generation == self._test_pending_generation
                and self._motion_mode
                in {"test_ai_loading", "test_combined_loading"}
                and self._test_start_pending
            )
        return request.kind == "normal" and self._motion_mode is None

    def _request_general_ai_start(self, context: str) -> bool:
        request = self._general_ai_start_request(context)
        try:
            worker_active = bool(self.ai_service.worker_active)
        except Exception:
            logging.exception("AI worker retirement probe failed")
            return False
        if worker_active:
            self._deferred_ai_start = request
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self._sync_adaptive_zoom_gate()
            self.ai_status_var.set("Loading")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            self.ai_zoom_var.set("1.0\N{MULTIPLICATION SIGN}")
            self._render_model_controls()
            return True
        return self._start_ai_runtime(
            request.context,
            model_choice=request.model_choice,
            capture_mode=request.capture_mode,
            lifecycle_kind=request.kind,
            test_generation=request.test_generation,
        )

    def _continue_general_ai_start(self) -> None:
        request = self._deferred_ai_start
        if request is None:
            return
        if not self._general_ai_start_is_current(request):
            self._deferred_ai_start = None
            self._render_model_controls()
            return
        try:
            worker_active = bool(self.ai_service.worker_active)
        except Exception:
            logging.exception("AI worker retirement probe failed")
            self._deferred_ai_start = None
            self._handle_deferred_ai_start_failure(request)
            return
        if worker_active:
            return
        self._deferred_ai_start = None
        started = self._start_ai_runtime(
            request.context,
            model_choice=request.model_choice,
            capture_mode=request.capture_mode,
            lifecycle_kind=request.kind,
            test_generation=request.test_generation,
        )
        if not started:
            self._handle_deferred_ai_start_failure(request)

    def _handle_deferred_ai_start_failure(
        self, request: _DeferredAiStart
    ) -> None:
        if (
            request.kind == "test"
            and request.test_generation == self._test_pending_generation
        ):
            self._abort_test_run("AI Test Run could not start")
            return
        self._hide_overlay_after_ai_failure()
        if self.master_armed and self.ai_selected:
            self._handle_ai_start_failure()
        else:
            self._render_runtime_controls()
            self.footer_var.set("Overlay could not start AI detection")

    def _continue_model_start(self) -> None:
        pending = self._model_start_pending
        if pending is None:
            return
        kind, token = pending
        switch = self._model_switch
        expected_phase = {
            "candidate": "starting_candidate",
            "rollback": "starting_rollback",
        }.get(kind)
        if (
            expected_phase is None
            or switch is None
            or switch.token != token
            or switch.phase != expected_phase
        ):
            self._model_start_pending = None
            return
        if self._closing or not self._ai_runtime_required():
            self._model_start_pending = None
            return
        try:
            worker_active = bool(self.ai_service.worker_active)
        except Exception:
            logging.exception("AI worker retirement probe failed")
            self._model_start_pending = None
            if kind == "candidate":
                self._start_model_rollback(
                    switch, "candidate retirement probe failed"
                )
            else:
                self._handle_ai_runtime_error(
                    "AI model rollback retirement probe failed"
                )
            return
        if worker_active:
            return
        self._model_start_pending = None
        choice = switch.candidate if kind == "candidate" else switch.previous
        context = "Model switch" if kind == "candidate" else "Model rollback"
        if self._start_ai_runtime(
            context,
            model_choice=choice,
            capture_mode=self._capture_mode,
            lifecycle_kind=f"model_{kind}",
            lifecycle_token=token,
        ):
            return
        if kind == "candidate":
            self._start_model_rollback(switch, "candidate startup failed")
        else:
            self._handle_ai_runtime_error("AI model rollback failed")

    def _fail_capture_mode_switch(self) -> None:
        self._capture_restart_pending = False
        self._deferred_ai_start = None
        self._capture_mode_switching = False
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        self.ai_status_var.set("Error")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        self.ai_zoom_var.set("1.0\N{MULTIPLICATION SIGN}")
        self._hide_overlay_after_ai_failure()
        self._handle_ai_start_failure()
        self._render_runtime_controls()

    def browse_ai_model(self) -> None:
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        if self._model_changes_unavailable():
            self.footer_var.set("Stop AI before changing the model")
            return
        try:
            selected = self._model_file_chooser(
                title="Select AI Aim ONNX Model",
                filetypes=(("ONNX models", "*.onnx"), ("All files", "*.*")),
                parent=self,
            )
        except Exception:
            logging.exception("AI model chooser failed")
            self.footer_var.set("Could not open the model chooser")
            return
        if not selected:
            return
        try:
            candidate = external_model_choice(selected)
        except ModelSelectionError as error:
            logging.error("AI model selection rejected for %s", selected)
            self.footer_var.set(str(error))
            return
        self._begin_model_switch(candidate)

    def use_default_ai_model(self) -> None:
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        if self._model_changes_unavailable():
            self.footer_var.set("Stop AI before changing the model")
            return
        self._begin_model_switch(bundled_model_choice())

    def _begin_model_switch(self, candidate: ModelChoice) -> None:
        if (
            self._model_start_pending is not None
            or self._capture_mode_switching
            or self._capture_restart_pending
            or self._deferred_ai_start is not None
        ):
            self.footer_var.set("Wait for the current AI transition to finish")
            self._render_model_controls()
            return
        if candidate.path == self._model_choice.path:
            self.footer_var.set(f"Using model: {candidate.display_name}")
            return
        self._model_switch_token += 1
        switch = _ModelSwitch(
            self._model_switch_token,
            candidate,
            self._model_choice,
            "validating",
        )
        self._model_switch = switch
        self.ai_model_var.set(f"Loading · {candidate.display_name}")
        self._render_model_controls()
        if self.ai_selected and (
            self._normal_motion_started
            or self._expected_motion_generation is not None
        ):
            self._stop_motion_runtime("model_switch")
            self._normal_motion_started = False
            self._set_runtime_state(
                "armed" if self.master_armed else "disabled"
            )
        self._invalidate_trigger_lock_epoch()
        self.ai_zoom_var.set("1.0\N{MULTIPLICATION SIGN}")
        if self._ai_runtime_active:
            self._stop_ai_runtime("Model switch")
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        self.ai_status_var.set("Loading")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        if not self.model_validator.start(candidate, switch.token):
            self._start_model_rollback(
                switch, "validation worker could not start"
            )

    def _finish_model_switch(self, choice: ModelChoice, footer: str) -> None:
        self._model_start_pending = None
        self._model_choice = choice
        self._model_switch = None
        self._normalize_idle_ai_runtime_status()
        self.ai_model_var.set(self._model_label(choice))
        self._render_model_controls()
        self.footer_var.set(footer)

    def _normalize_idle_ai_runtime_status(self) -> None:
        if self._ai_runtime_required():
            return
        self.ai_status_var.set("Stopped")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        self.ai_zoom_var.set("1.0\N{MULTIPLICATION SIGN}")

    def _cancel_model_switch(self, reason: str) -> None:
        switch = self._model_switch
        if switch is None:
            return
        self._model_start_pending = None
        self._model_switch_token += 1
        try:
            self.model_validator.cancel()
        except Exception:
            logging.exception("AI model validation cancellation failed")
        try:
            if switch.phase in {"starting_candidate", "starting_rollback"}:
                try:
                    self._stop_ai_runtime("Model switch cancelled")
                except Exception:
                    logging.exception("AI model switch cleanup failed")
                finally:
                    self._ai_ready = False
                    self._ai_provider = None
                    self._ai_runtime_active = False
                    self._sync_adaptive_zoom_gate()
        finally:
            self._model_switch = None
            self._model_choice = switch.previous
            self._normalize_idle_ai_runtime_status()
            self.ai_model_var.set(self._model_label(self._model_choice))
            self._render_model_controls()
        logging.info("AI model switch cancelled: %s", reason)

    def queue_model_validation_event(self, event: ModelValidationEvent) -> None:
        if self._closing or self._closed:
            return
        self._ui_queue.put(("model_validation", None, event))

    def handle_model_validation_event(self, event: ModelValidationEvent) -> None:
        switch = self._model_switch
        if (
            switch is None
            or event.token != switch.token
            or event.choice.path != switch.candidate.path
        ):
            return
        if event.kind == "error":
            self._start_model_rollback(
                switch, event.safe_message or "AI model validation failed"
            )
            return
        if event.kind != "ready" or event.choice.input_size is None:
            return
        validated_switch = replace(switch, candidate=event.choice)
        self._model_switch = validated_switch
        if not self._ai_runtime_required():
            self._finish_model_switch(
                event.choice,
                f"Using model: {event.choice.display_name}",
            )
            return
        self._start_validated_model_generation(validated_switch)

    def _start_validated_model_generation(self, switch: _ModelSwitch) -> None:
        if self._model_switch != switch or switch.phase != "validating":
            return
        starting = replace(switch, phase="starting_candidate")
        self._model_switch = starting
        self._render_model_controls()
        self._model_start_pending = ("candidate", starting.token)
        self._continue_model_start()

    def _start_model_rollback(
        self,
        switch: _ModelSwitch,
        failure: str,
    ) -> None:
        if self._model_switch != switch:
            return
        logging.error(
            "AI model %s rejected: %s",
            switch.candidate.path,
            failure,
        )
        try:
            if self._ai_runtime_active:
                self._stop_ai_runtime("Model rollback")
        except Exception:
            logging.exception("AI model rollback cleanup failed")
        finally:
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self._sync_adaptive_zoom_gate()
        if not self._ai_runtime_required():
            self._finish_model_switch(
                switch.previous,
                f"Model rejected: {failure}; "
                f"restored {switch.previous.display_name}",
            )
            return
        rollback = replace(
            switch,
            phase="starting_rollback",
            failure=failure,
        )
        self._model_switch = rollback
        self.ai_model_var.set(f"Loading · {rollback.previous.display_name}")
        self._render_model_controls()
        self._model_start_pending = ("rollback", rollback.token)
        self._continue_model_start()

    def _render_runtime_controls(self) -> None:
        testing = self._motion_mode in _TEST_MOTION_MODES
        self.jitter_source_button.configure(
            text=f"Jitter {'ON' if self.jitter_selected else 'OFF'}",
            style=(
                "Liquid.Primary.TButton"
                if self.jitter_selected else "Liquid.Secondary.TButton"
            ),
            state="disabled" if testing else "normal",
        )
        self.ai_source_button.configure(
            text=f"AI Aim {'ON' if self.ai_selected else 'OFF'}",
            style=(
                "Liquid.Primary.TButton"
                if self.ai_selected else "Liquid.Secondary.TButton"
            ),
            state="disabled" if testing else "normal",
        )
        self.master_button.configure(
            text="Disable Selected" if self.master_armed else "Enable Selected"
        )
        self.overlay_button.configure(
            text=f"Overlay {'ON' if self.overlay_visible else 'OFF'}",
            style=(
                "Liquid.Primary.TButton"
                if self.overlay_visible else "Liquid.Secondary.TButton"
            ),
        )
        if self._overlay_customizer_exists():
            self.overlay_color_button.configure(
                text=f"Box Color {self.overlay_color.upper()}"
            )
            self.overlay_head_button.configure(
                text=(
                    "Head Boxes ON"
                    if self.overlay_head_visible else "Head Boxes OFF"
                ),
                style=(
                    "Liquid.Primary.TButton"
                    if self.overlay_head_visible else "Liquid.Secondary.TButton"
                ),
            )
            self.overlay_player_button.configure(
                text=(
                    "Player Boxes ON"
                    if self.overlay_player_visible else "Player Boxes OFF"
                ),
                style=(
                    "Liquid.Primary.TButton"
                    if self.overlay_player_visible else "Liquid.Secondary.TButton"
                ),
            )
            self.overlay_hud_button.configure(
                text=f"HUD {'ON' if self.overlay_hud_visible else 'OFF'}",
                style=(
                    "Liquid.Primary.TButton"
                    if self.overlay_hud_visible else "Liquid.Secondary.TButton"
                ),
            )
            self.overlay_hud_color_button.configure(
                text=f"HUD Color {self.overlay_hud_color.upper()}"
            )
            for label, key in (
                ("FPS", "fps"),
                ("Provider", "provider"),
                ("Zoom", "zoom"),
                ("Lock", "lock"),
            ):
                enabled = getattr(self, f"overlay_hud_show_{key}")
                getattr(self, f"overlay_hud_{key}_button").configure(
                    text=f"{label} {'ON' if enabled else 'OFF'}",
                    style=(
                        "Liquid.Primary.TButton"
                        if enabled else "Liquid.Secondary.TButton"
                    ),
                )
        self._render_model_controls()
        self._render_capture_mode_control()
        self._refresh_section_summaries()

    def choose_overlay_color(self) -> None:
        try:
            choice = self._color_chooser(
                initialcolor=self.overlay_color,
                parent=self,
                title="Overlay Box Color",
            )
        except tk.TclError:
            logging.exception("Overlay color chooser failed")
            self.footer_var.set("Could not open the color chooser")
            return
        selected = (
            choice[1]
            if isinstance(choice, tuple) and len(choice) > 1 else None
        )
        if selected is None:
            return
        self.overlay_color = normalize_overlay_color(selected)
        self._render_runtime_controls()
        self._schedule_save()
        self.footer_var.set(f"Overlay color set to {self.overlay_color.upper()}")

    def toggle_overlay_heads(self) -> None:
        self.overlay_head_visible = not self.overlay_head_visible
        self._render_runtime_controls()
        self._schedule_save()
        state = "shown" if self.overlay_head_visible else "hidden"
        self.footer_var.set(f"Overlay head boxes {state}")

    def toggle_overlay_players(self) -> None:
        self.overlay_player_visible = not self.overlay_player_visible
        self._render_runtime_controls()
        state = "shown" if self.overlay_player_visible else "hidden"
        self.footer_var.set(f"Overlay player boxes {state}")

    def toggle_overlay_hud(self) -> None:
        self.overlay_hud_visible = not self.overlay_hud_visible
        self._render_runtime_controls()
        self._overlay_style_changed()

    def toggle_overlay_hud_metric(self, key: str) -> None:
        attribute = f"overlay_hud_show_{key}"
        setattr(self, attribute, not getattr(self, attribute))
        self._render_runtime_controls()
        self._overlay_style_changed()

    def choose_overlay_hud_color(self) -> None:
        try:
            choice = self._color_chooser(
                initialcolor=self.overlay_hud_color,
                parent=self,
                title="Overlay HUD Color",
            )
        except tk.TclError:
            logging.exception("Overlay HUD color chooser failed")
            self.footer_var.set("Could not open the color chooser")
            return
        selected = (
            choice[1]
            if isinstance(choice, tuple) and len(choice) > 1 else None
        )
        if selected is None:
            return
        self.overlay_hud_color = normalize_overlay_color(selected)
        self._render_runtime_controls()
        self._overlay_style_changed()

    def reset_overlay_customization(self) -> None:
        defaults = OverlayStyle()
        self.overlay_color = defaults.box_color
        self.overlay_head_visible = defaults.show_heads
        self.overlay_player_visible = defaults.show_players
        self.overlay_box_width_var.set(str(defaults.box_width))
        self.overlay_label_mode_var.set("Off")
        self.overlay_hud_visible = defaults.hud_visible
        self.overlay_hud_corner_var.set("Top Left")
        self.overlay_hud_offset_x_var.set(str(defaults.hud_offset_x))
        self.overlay_hud_offset_y_var.set(str(defaults.hud_offset_y))
        self.overlay_hud_font_size_var.set(str(defaults.hud_font_size))
        self.overlay_hud_color = defaults.hud_color
        self.overlay_hud_show_fps = defaults.hud_show_fps
        self.overlay_hud_show_provider = defaults.hud_show_provider
        self.overlay_hud_show_zoom = defaults.hud_show_zoom
        self.overlay_hud_show_lock = defaults.hud_show_lock
        for key in (
            "box_width",
            "hud_offset_x",
            "hud_offset_y",
            "hud_font_size",
        ):
            value = int(getattr(self, f"overlay_{key}_var").get())
            getattr(self, f"overlay_{key}_scale").set(value)
        self._render_runtime_controls()
        self._schedule_save()
        self.footer_var.set("Overlay customization reset")

    def toggle_master(self) -> None:
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        self.set_master(not self.master_armed)

    def toggle_enabled(self) -> None:
        self.toggle_master()

    def set_enabled(self, enabled: bool) -> None:
        self.set_master(enabled)

    def set_master(self, armed: bool) -> None:
        armed = bool(armed)
        if not armed:
            self._retire_owned_trigger_lock_epoch()
            self.master_armed = False
            self._normal_motion_started = False
            self.trigger_gate.clear()
            self._sync_adaptive_zoom_gate()
            if not self._ai_runtime_required():
                self._cancel_model_switch("Master disabled")
            self._stop_motion_runtime("Disabled by user")
            self._reconcile_ai_runtime("Master disabled")
            self._set_runtime_state("disabled")
            self._render_runtime_controls()
            self.footer_var.set("Selected sources disabled")
            return
        if self._closing:
            return
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        sources = self._selected_sources()
        if not sources.any:
            self.footer_var.set("Select Jitter or AI Aim first")
            return
        if not self.service.connected:
            self.footer_var.set("Makcu device is not connected")
            return
        self._motion_event_epoch += 1
        self.master_armed = True
        self._motion_mode = None
        self._normal_motion_started = False
        self._expected_motion_generation = None
        self._deferred_motion_action = None
        self.trigger_gate.clear()
        self._sync_adaptive_zoom_gate()
        if not self._reconcile_ai_runtime("Master enabled"):
            self._handle_ai_start_failure()
            self._render_runtime_controls()
            return
        self._sync_adaptive_zoom_gate()
        self._set_runtime_state("armed")
        self._render_runtime_controls()
        self.footer_var.set("Selected sources armed")

    def toggle_jitter_source(self) -> None:
        self._toggle_motion_source("jitter")

    def toggle_ai_source(self) -> None:
        self._toggle_motion_source("ai")

    def _toggle_motion_source(self, source_name: str) -> None:
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        old_sources = self._selected_sources()
        self._end_trigger_lock_epoch()
        if source_name == "jitter":
            self.jitter_selected = not self.jitter_selected
        elif source_name == "ai":
            self.ai_selected = not self.ai_selected
        else:
            raise ValueError(f"unknown motion source: {source_name}")
        sources = self._selected_sources()
        self._sync_adaptive_zoom_gate()
        self._cancel_model_switch("Motion sources changed")
        if not self.master_armed:
            if self._ai_runtime_required():
                self._reconcile_ai_runtime("Motion sources updated")
            self._render_runtime_controls()
            self.footer_var.set("Motion sources updated")
            return
        if not sources.any:
            self.set_master(False)
            self.footer_var.set("No motion sources selected; Master disabled")
            return

        adding_ai = not old_sources.ai and sources.ai
        if adding_ai and not self._reconcile_ai_runtime("AI source selected"):
            self._handle_ai_start_failure()
            self._render_runtime_controls()
            return

        was_moving = self._normal_motion_started
        if was_moving:
            self._stop_motion_runtime("sources_changed")
            self._normal_motion_started = False
            self._set_runtime_state("armed")
            if self.trigger_gate.active:
                self._defer_motion_action("normal", sources=sources)

        if not adding_ai:
            self._reconcile_ai_runtime("Motion sources updated")
        self._sync_adaptive_zoom_gate()
        if self.trigger_gate.active and not was_moving:
            self._normal_motion_started = self._start_gated_motion()
            if self._normal_motion_started:
                self._set_runtime_state("moving")
        self._render_runtime_controls()
        self.footer_var.set("Motion sources updated")

    def _ai_runtime_required(self) -> bool:
        return (
            self.overlay_visible
            or (self.master_armed and self.ai_selected)
            or self._motion_mode in {
                "test_ai_loading",
                "test_ai",
                "test_combined_loading",
                "test_combined",
            }
        )

    def _reconcile_ai_runtime(self, context: str) -> bool:
        required = self._ai_runtime_required()
        switch = self._model_switch
        if required and switch is not None and switch.phase == "validating":
            self._cancel_model_switch(f"{context}: AI demand started")
        if required and not self._ai_runtime_active:
            if self._capture_restart_pending or self._model_start_pending:
                self._continue_deferred_ai_start()
                return bool(
                    self._capture_restart_pending
                    or self._model_start_pending
                    or self._ai_runtime_active
                )
            if self._deferred_ai_start is not None:
                self._deferred_ai_start = self._general_ai_start_request(context)
                self._continue_deferred_ai_start()
                return bool(
                    self._deferred_ai_start is not None
                    or self._ai_runtime_active
                )
            started = self._request_general_ai_start(context)
            if not started:
                self._hide_overlay_after_ai_failure()
            return started
        if not required:
            self._cancel_model_switch(context)
            try:
                if self._ai_runtime_active:
                    self._sync_adaptive_zoom_gate()
                    self._stop_ai_runtime(context)
            except Exception:
                logging.exception(
                    "AI runtime stop failed during %s", context
                )
            finally:
                self._capture_restart_pending = False
                self._model_start_pending = None
                self._deferred_ai_start = None
                self._capture_mode_switching = False
                self._ai_ready = False
                self._ai_provider = None
                self._ai_runtime_active = False
                self._sync_adaptive_zoom_gate()
                self.ai_status_var.set("Stopped")
                self.ai_fps_var.set("0 FPS")
                self.ai_provider_var.set("No provider")
        return True

    def _hide_overlay_after_ai_failure(self) -> None:
        if not self.overlay_visible:
            return
        self.overlay_visible = False
        self._cancel_after("_overlay_after_id")
        self._hide_overlay_fail_closed(
            "Detection overlay hide failed after AI error"
        )

    def _hide_overlay_fail_closed(self, failure_message: str) -> None:
        self._last_overlay_render_key = None
        try:
            self.overlay.hide()
        except Exception:
            logging.exception(failure_message)
            try:
                self.overlay.close()
            except Exception:
                logging.exception(
                    "Detection overlay destruction failed after hide error"
                )

    def toggle_overlay(self) -> None:
        if self.overlay_visible:
            self.overlay_visible = False
            self._cancel_after("_overlay_after_id")
            self._hide_overlay_fail_closed("Detection overlay hide failed")
            if not self._ai_runtime_required():
                self._cancel_model_switch("Overlay disabled")
            self._reconcile_ai_runtime("Overlay disabled")
            self._render_runtime_controls()
            self.footer_var.set("Overlay disabled")
            return

        self._last_overlay_render_key = None
        try:
            self.overlay.show()
        except OverlaySetupError:
            logging.exception("Detection overlay setup failed")
            self.overlay_visible = False
            self._render_runtime_controls()
            self.footer_var.set("Overlay unavailable; check app.log")
            return
        except Exception:
            logging.exception("Detection overlay setup failed")
            self.overlay_visible = False
            self._render_runtime_controls()
            self.footer_var.set("Overlay unavailable; check app.log")
            return

        self._last_overlay_render_key = None
        self.overlay_visible = True
        self._render_runtime_controls()
        if not self._reconcile_ai_runtime("Overlay enabled"):
            self._render_runtime_controls()
            self.footer_var.set("Overlay could not start AI detection")
            return
        self.footer_var.set("Overlay enabled")
        self._poll_overlay()

    def _poll_overlay(self) -> None:
        self._overlay_after_id = None
        if self._closing or not self.overlay_visible:
            return
        try:
            snapshot = self.ai_service.latest_detection_snapshot()
            now = self._clock()
            runtime = (
                self.ai_fps_var.get(),
                self.ai_provider_var.get(),
                self.ai_zoom_var.get(),
            )
            style = self._overlay_style_snapshot()
            if snapshot is None:
                snapshot_key = ("none",)
            elif hasattr(snapshot, "sequence") and hasattr(
                snapshot,
                "captured_at",
            ):
                is_fresh = (
                    max(0.0, now - snapshot.captured_at)
                    <= MAX_FRAME_AGE_S
                )
                snapshot_key = (
                    snapshot.sequence,
                    snapshot.captured_at,
                    is_fresh,
                )
            else:
                snapshot_key = ("opaque", id(snapshot))
            render_key = (snapshot_key, runtime, style)
            if render_key != self._last_overlay_render_key:
                self.overlay.render(
                    snapshot,
                    now=now,
                    color=self.overlay_color,
                    show_heads=self.overlay_head_visible,
                    runtime=runtime,
                    style=style,
                )
                self._last_overlay_render_key = render_key
        except Exception:
            logging.exception("Detection overlay rendering failed")
            self._disable_overlay_after_error()
            return
        try:
            self._overlay_after_id = self.after(
                self._overlay_poll_delay_ms,
                self._poll_overlay,
            )
        except (tk.TclError, RuntimeError):
            self._overlay_after_id = None

    def _overlay_style_snapshot(self) -> OverlayStyle:
        label_modes = {
            "Off": "off",
            "Class": "class",
            "Class + Confidence": "class_confidence",
        }
        hud_corners = {
            "Top Left": "top_left",
            "Top Right": "top_right",
            "Bottom Left": "bottom_left",
            "Bottom Right": "bottom_right",
        }
        return OverlayStyle(
            box_color=self.overlay_color,
            show_heads=self.overlay_head_visible,
            show_players=self.overlay_player_visible,
            box_width=self._overlay_int_value("box_width", 2),
            label_mode=label_modes.get(
                self.overlay_label_mode_var.get(), "off"
            ),
            hud_corner=hud_corners.get(
                self.overlay_hud_corner_var.get(), "top_left"
            ),
            hud_offset_x=self._overlay_int_value("hud_offset_x", 8),
            hud_offset_y=self._overlay_int_value("hud_offset_y", 8),
            hud_font_size=self._overlay_int_value("hud_font_size", 10),
            hud_color=self.overlay_hud_color,
            hud_visible=self.overlay_hud_visible,
            hud_show_fps=self.overlay_hud_show_fps,
            hud_show_provider=self.overlay_hud_show_provider,
            hud_show_zoom=self.overlay_hud_show_zoom,
            hud_show_lock=self.overlay_hud_show_lock,
        )

    def _disable_overlay_after_error(self) -> None:
        self.overlay_visible = False
        self._last_overlay_render_key = None
        self._cancel_after("_overlay_after_id")
        try:
            self.overlay.close()
        except Exception:
            logging.exception("Detection overlay cleanup failed")
        self._reconcile_ai_runtime("Overlay failed")
        self._render_runtime_controls()
        self.footer_var.set("Overlay stopped; check app.log")

    def _start_ai_runtime(
        self,
        context: str,
        *,
        model_choice: ModelChoice | None = None,
        capture_mode: str | None = None,
        lifecycle_kind: str = "normal",
        lifecycle_token: int | None = None,
        test_generation: int | None = None,
    ) -> bool:
        choice = model_choice or self._model_choice
        generation_capture_mode = (
            self._capture_mode if capture_mode is None else capture_mode
        )
        request = _DeferredAiStart(
            context=context,
            model_choice=choice,
            capture_mode=generation_capture_mode,
            lifecycle_epoch=self._ai_event_epoch,
            kind=lifecycle_kind,
            test_generation=test_generation,
        )
        if not self._ai_runtime_active:
            self._ai_event_epoch += 1
        event_epoch = self._ai_event_epoch
        self._ai_runtime_active = True
        self._active_ai_lifecycle = None
        self._sync_adaptive_zoom_gate()
        self._render_capture_mode_control()
        try:
            generation = self.ai_service.start(
                self.get_ai_settings,
                self.get_adaptive_zoom_gate,
                self.get_trigger_lock_epoch,
                model_path=choice.path,
                capture_mode=generation_capture_mode,
            )
        except Exception:
            logging.exception("AI runtime could not start during %s", context)
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self._active_ai_lifecycle = None
            self._sync_adaptive_zoom_gate()
            self._render_capture_mode_control()
            self.ai_status_var.set("Error")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            return False
        if not generation:
            logging.error("AI runtime did not start during %s", context)
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self._active_ai_lifecycle = None
            self._sync_adaptive_zoom_gate()
            self._render_capture_mode_control()
            self.ai_status_var.set("Error")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            return False
        self._active_ai_lifecycle = _ActiveAiLifecycle(
            request=request,
            generation=generation,
            event_epoch=event_epoch,
            model_token=lifecycle_token,
        )
        return True

    def _handle_ai_start_failure(self) -> None:
        """Fail closed to Jitter when normal armed AI demand cannot start."""
        self._retire_owned_trigger_lock_epoch()
        self._capture_restart_pending = False
        self._deferred_ai_start = None
        self._capture_mode_switching = False
        self.ai_selected = False
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        if self.jitter_selected and self.master_armed:
            self._set_runtime_state(
                "moving" if self._normal_motion_started else "armed"
            )
            self.footer_var.set("AI Aim stopped; Jitter remains available")
        else:
            self.master_armed = False
            self.trigger_gate.clear()
            self._set_runtime_state("disabled")
            self.footer_var.set("AI Aim could not start; Master disabled")
        self._advance_hotkey_epoch()

    def _stop_ai_runtime(self, reason: str) -> None:
        self._sync_adaptive_zoom_gate()
        try:
            self.ai_service.stop(reason)
        finally:
            self._ai_event_epoch += 1
            self._active_ai_lifecycle = None

    def _stop_motion_runtime(
        self,
        reason: str,
        *,
        stop_device_motion: bool = True,
    ) -> None:
        self._deferred_motion_action = None
        retiring_source = self._expected_motion_generation
        self._expected_motion_generation = None
        if retiring_source is not None:
            self._retiring_motion_generation = retiring_source
        try:
            if stop_device_motion:
                self.service.cancel_motion(reason)
        finally:
            self._motion_event_epoch += 1

    def _defer_motion_action(
        self,
        kind: str,
        *,
        sources: MotionSources | None = None,
        test_generation: int | None = None,
    ) -> bool:
        retiring_source = self._retiring_motion_generation
        if retiring_source is None:
            return False
        # A terminal can be queued immediately after this linearized service
        # read.  Install the action before Tk drains that queue; the exact
        # retiring source is then the only event allowed to consume it.
        if not bool(getattr(self.service, "motion_active", True)):
            return False
        self._deferred_motion_action = _DeferredMotionAction(
            kind=kind,
            retiring_source=retiring_source,
            lifecycle_epoch=self._motion_event_epoch,
            sources=sources or self._selected_sources(),
            test_generation=test_generation,
        )
        return True

    def _advance_hotkey_epoch(self) -> None:
        with self._hotkey_epoch_lock:
            self._hotkey_event_epoch += 1

    def test_run(self) -> None:
        self.start_test_run()

    def start_test_run(self) -> None:
        if self._closing:
            return
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is already active")
            return
        sources = self._selected_sources()
        if not sources.any:
            self.footer_var.set("Select Jitter or AI Aim first")
            return
        if not self.service.connected:
            self.footer_var.set("Makcu device is not connected")
            return

        self._cancel_model_switch("Test Run started")
        self._deferred_motion_action = None
        self._test_restore_master = self.master_armed
        self.master_armed = False
        self._end_trigger_lock_epoch()
        self.trigger_gate.clear()
        self._test_sources = sources
        self._test_generation += 1
        generation = self._test_generation
        self._test_pending_generation = generation
        self._test_start_pending = True
        if sources.ai:
            self._motion_mode = (
                "test_combined_loading" if sources.jitter
                else "test_ai_loading"
            )
        else:
            self._motion_mode = "test_jitter_pending"

        self._sync_adaptive_zoom_gate()

        if self._normal_motion_started:
            self._stop_motion_runtime("test_run")
            self._normal_motion_started = False
        self._test_waiting_for_motion_stop = self._defer_motion_action(
            "test",
            sources=sources,
            test_generation=generation,
        )
        self._set_runtime_state("testing")
        self._set_test_button_enabled(False)
        self._render_runtime_controls()

        if sources.ai and not self._reconcile_ai_runtime("Test Run"):
            self._abort_test_run("AI Test Run could not start")
            return
        if self._test_waiting_for_motion_stop:
            return
        if sources.ai and not self._ai_ready:
            self.footer_var.set("Loading AI for Test Run")
            return
        if not self._begin_test_motion(generation):
            self._abort_test_run("Test Run could not start")

    def _begin_test_motion(self, generation: int) -> bool:
        sources = self._test_sources
        if (
            sources is None
            or self._test_pending_generation != generation
            or not self._test_start_pending
            or self._test_waiting_for_motion_stop
            or not bool(self.service.connected)
            or (sources.ai and not self._ai_ready)
        ):
            return False
        if sources.ai:
            self._motion_mode = (
                "test_combined" if sources.jitter else "test_ai"
            )
        else:
            self._motion_mode = "test_jitter"
        self._test_start_pending = False
        self._test_pending_generation = None
        self._motion_event_epoch += 1
        if sources.ai:
            if not self._reset_targeting_for_trigger_lock(
                "AI Test 3s start"
            ):
                return False
            self._publish_trigger_lock_epoch(
                self._next_trigger_lock_epoch(), "test"
            )
        started = self._request_motion_start(sources, duration_s=3.0)
        if started:
            self.footer_var.set("Test Run active")
            return True
        return False

    def _abort_test_run(self, message: str) -> None:
        self._restore_after_test(restore_master=False)
        self.footer_var.set(message)

    def _restore_after_test(self, *, restore_master: bool = True) -> None:
        test_epoch = self._clear_trigger_lock_epoch("test")
        if test_epoch is not None:
            self._reset_targeting_for_trigger_lock("AI Test 3s cleanup")
        restore = (
            restore_master
            and self._test_restore_master
            and bool(self.service.connected)
        )
        self._motion_event_epoch += 1
        self._expected_motion_generation = None
        self._deferred_motion_action = None
        self._retiring_motion_generation = None
        self._motion_mode = None
        self._test_sources = None
        self._test_start_pending = False
        self._test_pending_generation = None
        self._test_waiting_for_motion_stop = False
        self._test_restore_master = False
        self._set_test_button_enabled(True)
        self.trigger_gate.clear()
        self.master_armed = bool(restore)
        if restore:
            self._set_runtime_state("armed")
        else:
            self._set_runtime_state("disabled")
        self._sync_adaptive_zoom_gate()
        self._reconcile_ai_runtime("Test Run complete")
        self._render_runtime_controls()

    def emergency_stop(
        self,
        reason: str = "Stopped",
        *,
        stop_device_motion: bool = True,
    ) -> None:
        stop_reason = str(reason or "Stopped")
        self._retire_owned_trigger_lock_epoch()
        self._capture_restart_pending = False
        self._capture_mode_switching = False
        self.master_armed = False
        self._sync_adaptive_zoom_gate()
        was_overlay_visible = self.overlay_visible
        self.overlay_visible = False
        self._cancel_after("_overlay_after_id")
        if was_overlay_visible:
            self._hide_overlay_fail_closed(
                "Detection overlay hide failed during STOP"
            )
        self._normal_motion_started = False
        self._deferred_motion_action = None
        self._motion_mode = None
        self._test_sources = None
        self._test_start_pending = False
        self._test_generation += 1
        self._test_pending_generation = None
        self._test_waiting_for_motion_stop = False
        self._test_restore_master = False
        self.trigger_gate.clear()
        self._sync_adaptive_zoom_gate()
        self._cancel_model_switch(stop_reason)
        try:
            self._stop_motion_runtime(
                stop_reason,
                stop_device_motion=stop_device_motion,
            )
        except Exception:
            pass
        try:
            self._reconcile_ai_runtime(stop_reason)
        except Exception:
            logging.exception("AI runtime stop failed during STOP")
        self._advance_hotkey_epoch()
        self._set_runtime_state("disabled")
        self._render_runtime_controls()
        self._set_test_button_enabled(True)
        self.footer_var.set(stop_reason)

    def _handle_disconnect(self, reason: str) -> None:
        self._retire_owned_trigger_lock_epoch()
        self._capture_mode_switching = False
        self.master_armed = False
        self._sync_adaptive_zoom_gate()
        self._normal_motion_started = False
        self._deferred_motion_action = None
        self._motion_mode = None
        self._test_sources = None
        self._test_start_pending = False
        self._test_generation += 1
        self._test_pending_generation = None
        self._test_waiting_for_motion_stop = False
        self._test_restore_master = False
        self.trigger_gate.clear()
        self._sync_adaptive_zoom_gate()
        self._cancel_model_switch(reason)
        try:
            self._stop_motion_runtime(reason)
        except Exception:
            logging.exception("Makcu motion stop failed during disconnect")
        try:
            self._reconcile_ai_runtime(reason)
        except Exception:
            logging.exception("AI runtime reconciliation failed on disconnect")
        self._advance_hotkey_epoch()
        self._set_runtime_state("disabled")
        self._render_runtime_controls()
        self._set_test_button_enabled(True)
        self.footer_var.set(reason)

    def _consume_deferred_motion_action(self, source: Any) -> None:
        action = self._deferred_motion_action
        self._deferred_motion_action = None
        self._retiring_motion_generation = None
        if action is None or action.retiring_source != source:
            return
        if (
            action.lifecycle_epoch != self._motion_event_epoch
            or self._closing
            or not bool(self.service.connected)
            or self._selected_sources() != action.sources
        ):
            return

        if action.kind == "normal":
            if (
                not self.master_armed
                or self._motion_mode is not None
                or not self.trigger_gate.active
            ):
                return
            self._normal_motion_started = self._start_gated_motion()
            if self._normal_motion_started:
                self._set_runtime_state("moving")
            return

        if action.kind != "test":
            return
        if (
            self._motion_mode not in _TEST_MOTION_MODES
            or not self._test_start_pending
            or not self._test_waiting_for_motion_stop
            or action.test_generation is None
            or self._test_pending_generation != action.test_generation
        ):
            return
        self._test_waiting_for_motion_stop = False
        sources = self._test_sources
        if sources is not None and sources.ai and not self._ai_ready:
            return
        if not self._begin_test_motion(action.test_generation):
            self._abort_test_run("Test Run could not start")

    def handle_service_event(self, event: ServiceEvent) -> None:
        if self._closing:
            return
        kind = event.kind
        if kind in {"motion_error", "motion_stopped"}:
            source = event.motion_generation
            current_source = (
                source is not None
                and source == self._expected_motion_generation
            )
            retiring_source = (
                source is not None
                and not current_source
                and source == self._retiring_motion_generation
            )
            if not current_source and not retiring_source:
                return
            if retiring_source:
                self._consume_deferred_motion_action(source)
                return
        if kind == "connecting":
            self._set_connection_state("Connecting")
            self.device_status_var.set("Connecting to Makcu...")
        elif kind in {"connected", "reconnected"}:
            self._set_connection_state("Connected")
            if event.payload:
                logging.info("Makcu connected: %s", event.payload)
            self.device_status_var.set(_device_summary_text(event.payload))
            self.footer_var.set("Makcu connected")
        elif kind == "disconnected":
            self._set_connection_state("Disconnected")
            self.device_status_var.set(str(event.payload or "Makcu device not connected"))
            self._handle_disconnect("Device disconnected")
        elif kind == "button":
            try:
                button, pressed = event.payload
            except (TypeError, ValueError):
                return
            button = str(button)
            pressed = bool(pressed)
            was_physically_down = button in self._physical_buttons_down
            if pressed:
                self._physical_buttons_down.add(button)
            else:
                self._physical_buttons_down.discard(button)
            self.trigger_gate.update_button(button, pressed)
            is_trigger_button = button == self.trigger_gate.trigger
            if self._motion_mode not in _TEST_MOTION_MODES:
                if is_trigger_button and pressed and not was_physically_down:
                    self._begin_trigger_lock_epoch()
                elif (
                    is_trigger_button
                    and not pressed
                    and was_physically_down
                ):
                    self._end_trigger_lock_epoch()
            self._sync_adaptive_zoom_gate()
            if not self.trigger_gate.active:
                action = self._deferred_motion_action
                if action is not None and action.kind == "normal":
                    self._deferred_motion_action = None
            if self.master_armed and self._motion_mode is None:
                if self.trigger_gate.active and not self._normal_motion_started:
                    self._normal_motion_started = self._start_gated_motion()
                    if self._normal_motion_started:
                        self._set_runtime_state("moving")
                elif not self.trigger_gate.active and self._normal_motion_started:
                    self._stop_motion_runtime("trigger_released")
                    self._normal_motion_started = False
                    self._set_runtime_state("armed")
        elif kind == "motion_error":
            self._expected_motion_generation = None
            logging.error("Makcu motion error: %s", event.payload)
            self.emergency_stop(
                "Makcu movement failed; reconnect and try again"
            )
        elif kind == "motion_stopped":
            reason = str(event.payload or "")
            if (
                self._motion_mode in {
                    "test_jitter", "test_ai", "test_combined"
                }
                and reason == "duration_complete"
            ):
                self._restore_after_test()
                self.footer_var.set("Test Run complete")
            elif self._motion_mode is None and self._normal_motion_started:
                self._expected_motion_generation = None
                self._normal_motion_started = False
                if self.master_armed:
                    self._set_runtime_state("armed")

    def _request_motion_start(
        self,
        sources: MotionSources,
        duration_s: float | None = None,
    ) -> bool:
        """Reserve motion and remember the Makcu generation returned by start."""
        self._expected_motion_generation = None
        source = self.service.start_composite_motion_source(
            sources,
            self.get_motion_settings,
            self.ai_service.latest_snapshot,
            self.get_ai_settings,
            duration_s=duration_s,
            targeting_epoch_provider=(
                self.ai_service.current_targeting_revision
            ),
        )
        if source is None or source is False:
            return False
        self._expected_motion_generation = source
        return True

    def _start_gated_motion(self) -> bool:
        sources = self._selected_sources()
        if not sources.any or (sources.ai and not self._ai_ready):
            return False
        if self._defer_motion_action("normal", sources=sources):
            return False
        self._deferred_motion_action = None
        return self._request_motion_start(sources)

    def handle_ai_event(self, event: AiEvent) -> None:
        if self._closing:
            return
        kind = event.kind
        active_lifecycle = self._active_ai_lifecycle
        if (
            kind in {"loading", "ready", "fps", "zoom", "error"}
            and not self._ai_runtime_active
        ):
            return
        if (
            kind in {"loading", "ready", "fps", "zoom", "error", "stopped"}
            and (
                (
                    active_lifecycle is None
                    and event.generation is not None
                )
                or (
                    active_lifecycle is not None
                    and (
                        event.generation != active_lifecycle.generation
                        or active_lifecycle.event_epoch != self._ai_event_epoch
                    )
                )
            )
        ):
            return
        if kind == "loading":
            self._ai_runtime_active = True
            self._sync_adaptive_zoom_gate()
            self._ai_ready = False
            self._ai_provider = None
            self.ai_status_var.set("Loading")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            self._render_capture_mode_control()
            self.ai_zoom_var.set("1.0×")
        elif kind == "ready":
            switch = self._model_switch
            lifecycle_kind = (
                active_lifecycle.request.kind
                if active_lifecycle is not None else None
            )
            lifecycle_token = (
                active_lifecycle.model_token
                if active_lifecycle is not None else None
            )
            if (
                switch is not None
                and switch.phase == "starting_candidate"
                and lifecycle_kind == "model_candidate"
                and lifecycle_token == switch.token
            ):
                self._finish_model_switch(
                    switch.candidate,
                    f"Using model: {switch.candidate.display_name}",
                )
            elif (
                switch is not None
                and switch.phase == "starting_rollback"
                and lifecycle_kind == "model_rollback"
                and lifecycle_token == switch.token
            ):
                failure = switch.failure or "AI model validation failed"
                self._finish_model_switch(
                    switch.previous,
                    f"Model rejected: {failure}; "
                    f"restored {switch.previous.display_name}",
                )
            raw_provider = str(event.payload or "Unknown")
            provider = {
                "DmlExecutionProvider": "DirectML",
                "CPUExecutionProvider": "CPU",
            }.get(raw_provider, raw_provider)
            self._ai_ready = True
            self._ai_provider = raw_provider
            self._ai_runtime_active = True
            self._sync_adaptive_zoom_gate()
            self.ai_status_var.set(f"Ready ({provider})")
            self.ai_provider_var.set(provider)
            capture_mode_switching = bool(
                self._capture_mode_switching
                and lifecycle_kind == "capture"
                and active_lifecycle is not None
                and active_lifecycle.request.capture_mode == self._capture_mode
            )
            if capture_mode_switching:
                self._capture_restart_pending = False
                self._capture_mode_switching = False
            self._render_model_controls()
            if capture_mode_switching:
                self.footer_var.set(
                    f"AI capture ready: "
                    f"{_CAPTURE_MODE_LABELS[self._capture_mode]}"
                )
            if (
                self._motion_mode in {
                    "test_ai_loading", "test_combined_loading"
                }
                and self._test_pending_generation is not None
                and not self._test_waiting_for_motion_stop
            ):
                generation = self._test_pending_generation
                if not self._begin_test_motion(generation):
                    self._abort_test_run("AI Test Run could not start")
            elif (self.master_armed and self.ai_selected
                    and self._motion_mode is None
                    and self.trigger_gate.active
                    and not self._normal_motion_started):
                self._normal_motion_started = self._start_gated_motion()
                if self._normal_motion_started:
                    self._set_runtime_state("moving")
        elif kind == "fps":
            try:
                fps = float(event.payload)
                if not math.isfinite(fps):
                    raise ValueError
                fps = max(0.0, fps)
            except (TypeError, ValueError, OverflowError):
                fps = 0.0
            rendered = f"{fps:.1f}".rstrip("0").rstrip(".")
            self.ai_fps_var.set(f"{rendered} FPS")
        elif kind == "zoom":
            if (
                event.targeting_revision is not None
                and event.targeting_revision != self._ai_targeting_revision
            ):
                return
            if not self.get_adaptive_zoom_gate():
                self.ai_zoom_var.set("1.0×")
                return
            try:
                factor = float(event.payload)
                if not math.isfinite(factor) or factor not in {1.0, 1.5, 2.0}:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                factor = 1.0
            self.ai_zoom_var.set(f"{factor:.1f}×")
        elif kind == "error":
            self._capture_restart_pending = False
            if self._capture_mode_switching:
                self._capture_mode_switching = False
            switch = self._model_switch
            if switch is not None and switch.phase == "starting_candidate":
                failure = str(event.payload or "AI service failed")
                self._start_model_rollback(switch, failure)
                return
            if switch is not None and switch.phase == "starting_rollback":
                self._handle_ai_runtime_error(event.payload)
                return
            self._handle_ai_runtime_error(event.payload)
        elif kind == "stopped":
            self._capture_restart_pending = False
            self._capture_mode_switching = False
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self._active_ai_lifecycle = None
            self._sync_adaptive_zoom_gate()
            self.ai_status_var.set("Stopped")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            self._render_capture_mode_control()

    def _handle_ai_runtime_error(self, payload: object) -> None:
        logging.error("AI runtime error: %s", payload)
        self._retire_owned_trigger_lock_epoch()
        self._deferred_ai_start = None
        self._active_ai_lifecycle = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        test_sources = self._test_sources
        ai_motion_demand = (
            (self.master_armed and self.ai_selected)
            or (
                self._motion_mode in _TEST_MOTION_MODES
                and test_sources is not None
                and test_sources.ai
            )
        )
        if not ai_motion_demand:
            was_overlay_visible = self.overlay_visible
            self.overlay_visible = False
            self._cancel_after("_overlay_after_id")
            if was_overlay_visible:
                self._hide_overlay_fail_closed(
                    "Overlay hide failed after AI error"
                )
            self._cancel_model_switch("AI runtime error")
            try:
                self._stop_ai_runtime("ai_error")
            except Exception:
                logging.exception("AI runtime stop failed after error")
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self._sync_adaptive_zoom_gate()
            self.ai_status_var.set("Error")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            self._render_runtime_controls()
            self.footer_var.set("Overlay stopped; AI detection failed")
            return
        was_master_armed = self.master_armed
        was_test_run = self._motion_mode in _TEST_MOTION_MODES
        gate_active = self.trigger_gate.active
        jitter_fallback = was_master_armed and self.jitter_selected
        had_motion = (
            self._normal_motion_started
            or self._expected_motion_generation is not None
        )

        self.ai_selected = False
        was_overlay_visible = self.overlay_visible
        self.overlay_visible = False
        self._cancel_after("_overlay_after_id")
        if was_overlay_visible:
            self._hide_overlay_fail_closed(
                "Overlay hide failed after AI error"
            )
        self._cancel_model_switch("AI runtime error")
        try:
            self._stop_ai_runtime("ai_error")
        except Exception:
            logging.exception("AI runtime stop failed after error")
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self._sync_adaptive_zoom_gate()
        self.ai_status_var.set("Error")
        self.ai_fps_var.set("0 FPS")
        self.ai_provider_var.set("No provider")
        self._normal_motion_started = False
        self._motion_mode = None
        self._test_sources = None
        self._test_start_pending = False
        self._deferred_motion_action = None
        self._test_generation += 1
        self._test_pending_generation = None
        self._test_waiting_for_motion_stop = False
        self._test_restore_master = False
        if had_motion or was_test_run:
            try:
                self._stop_motion_runtime("ai_error")
            except Exception:
                logging.exception("Motion stop failed after AI error")

        self.master_armed = bool(jitter_fallback and not was_test_run)
        self._sync_adaptive_zoom_gate()
        if self.master_armed:
            self._set_runtime_state("armed")
            if gate_active:
                self._normal_motion_started = self._start_gated_motion()
                if self._normal_motion_started:
                    self._set_runtime_state("moving")
        else:
            self.trigger_gate.clear()
            self._set_runtime_state("disabled")
        self._advance_hotkey_epoch()
        self._render_runtime_controls()
        self._set_test_button_enabled(True)
        self.footer_var.set("AI Aim stopped; Jitter remains available")

    def queue_service_event(self, event: ServiceEvent) -> None:
        # This method is intentionally the only service-to-Tk handoff.
        if self._closing or self._closed:
            return
        epoch = (
            self._motion_event_epoch
            if event.kind in _LIFECYCLE_SERVICE_EVENTS else None
        )
        self._ui_queue.put(("service", epoch, event))

    def queue_ai_event(self, event: AiEvent) -> None:
        if self._closing or self._closed:
            return
        self._ui_queue.put(("ai", self._ai_event_epoch, event))

    def _hotkey_pressed(self) -> None:
        with self._hotkey_epoch_lock:
            epoch = self._hotkey_event_epoch
        if self._capturing_hotkey or self._closing:
            return
        if self._closed:
            return
        self._ui_queue.put(("hotkey", epoch, None))

    def _drain_ui_queue(self) -> None:
        self._ui_pump_after_id = None
        if self._closing or self._closed:
            return
        next_delay_ms = _UI_QUEUE_IDLE_DELAY_MS
        deadline = time.monotonic() + _UI_QUEUE_TIME_SLICE_S
        processed = 0
        try:
            while processed < _UI_QUEUE_MAX_BATCH:
                try:
                    kind, epoch, payload = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    current_epoch = {
                        "ai": self._ai_event_epoch,
                        "service": self._motion_event_epoch,
                        "hotkey": self._hotkey_event_epoch,
                    }.get(kind)
                    if epoch is not None and epoch != current_epoch:
                        processed += 1
                        continue
                    if kind == "service":
                        self.handle_service_event(payload)
                    elif kind == "ai":
                        self.handle_ai_event(payload)
                    elif kind == "model_validation":
                        self.handle_model_validation_event(payload)
                    elif kind == "hotkey":
                        was_enabled = self.enabled
                        self.toggle_enabled()
                        if (self.enabled != was_enabled
                                and self.sound_player is not None):
                            self.sound_player.play(self.enabled)
                except Exception:
                    logging.exception("UI queue handler failed for %s event", kind)
                processed += 1
                if time.monotonic() >= deadline:
                    next_delay_ms = 0
                    break
            if processed >= _UI_QUEUE_MAX_BATCH or not self._ui_queue.empty():
                next_delay_ms = 0
        finally:
            self._continue_deferred_ai_start()
            if not self._closing and not self._closed:
                try:
                    self._ui_pump_after_id = self.after(
                        next_delay_ms,
                        self._drain_ui_queue,
                    )
                except (tk.TclError, RuntimeError):
                    self._ui_pump_after_id = None

    def _get_async_key_state(self, vk: int) -> int:
        try:
            import ctypes
            return int(ctypes.windll.user32.GetAsyncKeyState(vk))
        except (AttributeError, OSError):
            return 0

    def capture_hotkey(self) -> None:
        if self._capturing_hotkey or self._closing:
            return
        self._capturing_hotkey = True
        self._capture_seen_down = False
        self._capture_prev_down = self._capture_key_state()
        self.hotkey_button.configure(text="Press a key or mouse button...", state="disabled")
        self.footer_var.set("Press a key or mouse button (Esc cancels)")
        self._poll_hotkey_capture()

    def _poll_hotkey_capture(self) -> None:
        if not self._capturing_hotkey or self._closing:
            return
        previous = self._capture_prev_down
        current: dict[int, bool] = {}
        for vk in range(1, 256):
            is_down = bool(self._get_async_key_state(vk) & 0x8000)
            current[vk] = is_down
            if not is_down or previous.get(vk, False):
                continue
            if vk == 0x1B:
                self._capture_prev_down = current
                self._cancel_hotkey_capture()
                return
            self._capture_prev_down = current
            self.apply_captured_hotkey(vk, self._format_hotkey_name(vk))
            return
        self._capture_prev_down = current
        self._cancel_after("_capture_after_id")
        try:
            self._capture_after_id = self.after(40, self._poll_hotkey_capture)
        except (tk.TclError, RuntimeError):
            self._capture_after_id = None
            self._capturing_hotkey = False

    def _capture_key_state(self) -> dict[int, bool]:
        return {
            vk: bool(self._get_async_key_state(vk) & 0x8000)
            for vk in range(1, 256)
        }

    @staticmethod
    def _format_hotkey_name(vk: int) -> str:
        if 0x70 <= vk <= 0x7B:
            return f"F{vk - 0x6F}"
        if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
            return chr(vk)
        names = {0x01: "Mouse Left", 0x02: "Mouse Right", 0x04: "Mouse Middle",
                 0x05: "Mouse4", 0x06: "Mouse5",
                 0x20: "Space", 0x09: "Tab", 0x0D: "Enter", 0x10: "Shift",
                 0x11: "Ctrl", 0x12: "Alt", 0x2D: "Insert", 0x2E: "Delete"}
        return names.get(vk, f"VK {vk}")

    def _cancel_hotkey_capture(self) -> None:
        self._capturing_hotkey = False
        self.hotkey_button.configure(text=f"Hotkey: {self.hotkey_name_var.get()}", state="normal")
        self.footer_var.set("Hotkey capture cancelled")
        self._cancel_after("_capture_after_id")

    def apply_captured_hotkey(self, vk: int, name: str) -> None:
        self._capturing_hotkey = False
        self._cancel_after("_capture_after_id")
        self.hotkey_watcher.set_vk(int(vk))
        self._hotkey_vk = int(vk)
        self.hotkey_name_var.set(str(name))
        self.hotkey_button.configure(text=f"Hotkey: {name}", state="normal")
        self.footer_var.set(f"Hotkey set to {name}")
        self._schedule_save()

    def on_bindings_changed(self) -> None:
        trigger = self.trigger_var.get()
        modifier = self.modifier_var.get()
        self._end_trigger_lock_epoch()
        action = self._deferred_motion_action
        self._deferred_motion_action = None
        if action is not None and action.kind == "test":
            self._restore_after_test(restore_master=False)
            self.footer_var.set("Bindings changed; Test Run canceled")
        self.trigger_gate.configure(trigger, modifier)
        self._sync_adaptive_zoom_gate()
        if self._normal_motion_started:
            self._stop_motion_runtime("bindings_changed")
            self._normal_motion_started = False
            if self.enabled:
                self._set_runtime_state("armed")
        self._schedule_save()
        self._refresh_section_summaries()

    def get_motion_settings(self) -> MotionSettings:
        with self._motion_lock:
            return self._motion_snapshot

    def get_ai_settings(self) -> AimSettings:
        with self._ai_lock:
            return self._ai_snapshot

    def get_adaptive_zoom_gate(self) -> bool:
        with self._ai_lock:
            return self._adaptive_zoom_gate

    def get_trigger_lock_epoch(self) -> int | None:
        with self._ai_lock:
            return self._trigger_lock_epoch

    def _publish_trigger_lock_epoch(
        self,
        epoch: int | None,
        owner: str | None,
    ) -> None:
        if (epoch is None) != (owner is None) or owner not in {
            None, "normal", "test",
        }:
            raise ValueError("Trigger-lock epoch and owner must agree")
        with self._ai_lock:
            self._trigger_lock_epoch = epoch
            self._trigger_lock_owner = owner

    def _clear_trigger_lock_epoch(
        self,
        expected_owner: str | None = None,
    ) -> int | None:
        with self._ai_lock:
            if (
                expected_owner is not None
                and self._trigger_lock_owner != expected_owner
            ):
                return None
            epoch = self._trigger_lock_epoch
            self._trigger_lock_epoch = None
            self._trigger_lock_owner = None
            return epoch

    def _next_trigger_lock_epoch(self) -> int:
        with self._ai_lock:
            self._trigger_lock_counter += 1
            return self._trigger_lock_counter

    def _raw_trigger_press_eligible(self) -> bool:
        return bool(
            not self._closing
            and self.service.connected
            and self.master_armed
            and self.ai_selected
            and self._motion_mode is None
        )

    def _reset_targeting_for_trigger_lock(self, context: str) -> bool:
        try:
            revision = self.ai_service.reset_targeting()
        except Exception:
            logging.exception("AI targeting reset failed during %s", context)
            self._clear_trigger_lock_epoch()
            self.footer_var.set("AI target lock stopped; check app.log")
            return False
        self._ai_targeting_revision = revision
        return True

    def _begin_trigger_lock_epoch(self) -> None:
        with self._ai_lock:
            if self._trigger_lock_owner is not None:
                return
        if not self._raw_trigger_press_eligible():
            return
        if not self._reset_targeting_for_trigger_lock("Trigger press"):
            return
        self._publish_trigger_lock_epoch(
            self._next_trigger_lock_epoch(), "normal"
        )

    def _end_trigger_lock_epoch(self) -> None:
        epoch = self._clear_trigger_lock_epoch("normal")
        if epoch is None:
            return
        self._reset_targeting_for_trigger_lock("Trigger release")

    def _retire_owned_trigger_lock_epoch(self) -> None:
        epoch = self._clear_trigger_lock_epoch()
        if epoch is None:
            return
        self._reset_targeting_for_trigger_lock("Trigger lock retirement")

    def _invalidate_trigger_lock_epoch(self) -> bool:
        epoch = self.get_trigger_lock_epoch()
        try:
            revision = self.ai_service.invalidate_trigger_lock(epoch)
        except Exception:
            logging.exception("AI Trigger lock invalidation failed")
            self._clear_trigger_lock_epoch()
            self.footer_var.set("AI target lock stopped; check app.log")
            return False
        self._ai_targeting_revision = revision
        return True

    def _sync_adaptive_zoom_gate(self) -> None:
        active = bool(
            not self._closing
            and self.service.connected
            and self._ai_runtime_active
            and self.master_armed
            and self.ai_selected
            and self._motion_mode is None
            and self.trigger_gate.active
        )
        with self._ai_lock:
            self._adaptive_zoom_gate = active
        if not active:
            self.ai_zoom_var.set("1.0×")

    def _replace_ai_snapshot(self, settings: AimSettings) -> None:
        with self._ai_lock:
            self._ai_snapshot = settings
        self._refresh_section_summaries()

    def _replace_motion_snapshot(self, settings: MotionSettings) -> None:
        with self._motion_lock:
            self._motion_snapshot = settings
        self.motion_snapshot_size_var.set(_display_value(settings.pulse_size_px))
        self.motion_snapshot_rate_var.set(_display_value(settings.pulse_rate_hz))
        self.motion_snapshot_ramp_var.set(settings.ramp_mode)
        self.motion_summary_var.set(_motion_summary_text(settings))

    def _motion_changed(self, key: str) -> None:
        if self._updating_motion_controls or self._closing:
            return
        mapping = {name: variable.get() for name, variable in self.motion_vars.items()}
        invalid = self._invalid_motion_values(mapping)
        if invalid:
            self._invalid_motion_keys = invalid
            for name in invalid:
                entry = getattr(self, f"{name}_entry", None)
                if entry is not None:
                    entry.configure(style="Liquid.Invalid.TEntry")
            self.footer_var.set(f"Invalid value for {key.replace('_', ' ')}")
            return
        self._invalid_motion_keys.clear()
        if self.footer_var.get().startswith("Invalid value for "):
            self.footer_var.set("Ready")
        for name in self.motion_vars:
            entry = getattr(self, f"{name}_entry", None)
            if entry is not None:
                entry.configure(style="Liquid.Entry.TEntry")
            scale = getattr(self, f"{name}_scale", None)
            if scale is not None:
                try:
                    self._updating_motion_controls = True
                    scale.set(float(self.motion_vars[name].get()))
                except (TypeError, ValueError, tk.TclError):
                    pass
                finally:
                    self._updating_motion_controls = False
        self._replace_motion_snapshot(motion_settings_from_mapping(mapping))
        if self.preset_var.get() != "Custom":
            self._updating_motion_controls = True
            try:
                self.preset_var.set("Custom")
            finally:
                self._updating_motion_controls = False
        self._schedule_save()
        self._refresh_section_summaries()

    def _ai_changed(self, key: str) -> None:
        if self._updating_ai_controls or self._closing:
            return
        mapping = self._current_ai_mapping()
        invalid = self._invalid_ai_values(mapping)
        if invalid:
            self._invalid_ai_keys = invalid
            for name in self.ai_vars:
                entry = getattr(self, f"ai_{name}_entry", None)
                if entry is not None:
                    entry.configure(
                        style=(
                            "Liquid.Invalid.TEntry"
                            if name in invalid else "Liquid.Entry.TEntry"
                        )
                    )
            self.footer_var.set(f"Invalid value for {key.replace('_', ' ')}")
            return
        self._invalid_ai_keys.clear()
        if self.footer_var.get().startswith("Invalid value for "):
            self.footer_var.set("Ready")
        for name in self.ai_vars:
            entry = getattr(self, f"ai_{name}_entry", None)
            if entry is not None:
                entry.configure(style="Liquid.Entry.TEntry")
            scale = getattr(self, f"ai_{name}_scale", None)
            if scale is not None:
                try:
                    self._updating_ai_controls = True
                    scale.set(float(self.ai_vars[name].get()))
                except (TypeError, ValueError, tk.TclError):
                    pass
                finally:
                    self._updating_ai_controls = False
        self._replace_ai_snapshot(aim_settings_from_mapping(mapping))
        if key == "confidence":
            self._invalidate_trigger_lock_epoch()
        self._schedule_save()

    @staticmethod
    def _invalid_motion_values(mapping: Mapping[str, Any]) -> set[str]:
        invalid: set[str] = set()
        for key, (low, high) in MOTION_LIMITS.items():
            raw = mapping.get(key)
            try:
                value = float(raw)
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError
            except (TypeError, ValueError):
                invalid.add(key)
        return invalid

    @staticmethod
    def _invalid_ai_values(mapping: Mapping[str, Any]) -> set[str]:
        invalid: set[str] = set()
        for key, (_label, control_low, control_high, _step) in (
            _AI_CONTROL_SPECS.items()
        ):
            domain_low, domain_high = AIM_LIMITS[key]
            low = max(control_low, domain_low)
            high = min(control_high, domain_high)
            raw = mapping.get(key)
            try:
                value = float(raw)
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError
                if key == "max_step" and not value.is_integer():
                    raise ValueError
            except (TypeError, ValueError):
                invalid.add(key)
        return invalid

    def apply_preset(self, _event: tk.Event | None = None) -> None:
        name = self.preset_var.get()
        if name == "Custom":
            return
        preset = MOTION_PRESETS.get(name)
        if preset is None:
            return
        mapping = motion_settings_to_mapping(MotionSettings())
        mapping.update(preset)
        settings = motion_settings_from_mapping(mapping)
        self._updating_motion_controls = True
        try:
            for key, value in motion_settings_to_mapping(settings).items():
                variable = self.motion_vars[key]
                variable.set(str(value))
                scale = getattr(self, f"{key}_scale", None)
                if scale is not None:
                    scale.set(float(value))
        finally:
            self._updating_motion_controls = False
        self._invalid_motion_keys.clear()
        if self.footer_var.get().startswith("Invalid value for "):
            self.footer_var.set("Ready")
        for key in self.motion_vars:
            entry = getattr(self, f"{key}_entry", None)
            if entry is not None:
                entry.configure(style="Liquid.Entry.TEntry")
        self._replace_motion_snapshot(settings)
        self._schedule_save()
        self._refresh_section_summaries()

    def _schedule_save(self) -> None:
        if self._closing or not self._save_allowed:
            return
        self._cancel_after("_save_after_id")
        self._save_after_id = self.after(250, self.save_config)

    def _cancel_after(self, attribute: str) -> None:
        callback_id = getattr(self, attribute, None)
        if callback_id is None:
            return
        try:
            self.after_cancel(callback_id)
        except tk.TclError:
            pass
        setattr(self, attribute, None)

    def save_config(self) -> None:
        self._save_after_id = None
        if not self._save_allowed or self._closed:
            return
        try:
            sound_volume = int(self.sound_volume_var.get())
        except (TypeError, ValueError):
            sound_volume = self.config.sound_volume
        sound_volume = max(0, min(100, sound_volume))
        self.sound_volume_var.set(str(sound_volume))
        config = AppConfig(
            motion=self.get_motion_settings(),
            ai=replace(self.get_ai_settings(), target_area="head"),
            trigger=self.trigger_var.get(),
            modifier=self.modifier_var.get(),
            hotkey_vk=self._current_hotkey_vk(),
            hotkey_name=self.hotkey_name_var.get(),
            selected_preset=self.preset_var.get() or "Custom",
            theme=self.theme_var.get(),
            sound_enabled=self.sound_enabled_var.get(),
            sound_volume=sound_volume,
            overlay_color=self.overlay_color,
            overlay_head_visible=self.overlay_head_visible,
        )
        try:
            self.config_store.save(config)
            self.config = config
        except Exception as exc:
            self.footer_var.set(f"Could not save config: {type(exc).__name__}")

    def close_app(self) -> None:
        if self._closed or self._closing:
            return
        self._closing = True
        self._capture_restart_pending = False
        self._model_start_pending = None
        self._deferred_ai_start = None
        self._active_ai_lifecycle = None
        self._capture_mode_switching = False
        self._cancel_ai_curve_callbacks()
        for widget in self.winfo_children():
            self._cancel_slider_callbacks(widget)
        self._cancel_after("_save_after_id")
        self._cancel_after("_capture_after_id")
        self._cancel_after("_ui_pump_after_id")
        self._cancel_after("_overlay_after_id")
        self._capturing_hotkey = False
        self.emergency_stop("Stopped on close")
        try:
            self.overlay.close()
        except Exception:
            logging.exception("Detection overlay close failed")
        try:
            self.hotkey_watcher.stop()
        except Exception:
            pass
        try:
            self.service.close()
        except Exception:
            pass
        try:
            self.ai_service.close()
        except Exception:
            pass
        try:
            self.model_validator.close()
        except Exception:
            pass
        if self.sound_player is not None:
            try:
                self.sound_player.close()
            except Exception:
                pass
        self._save_allowed = bool(self.load_outcome.save_allowed)
        self.save_config()
        self._closed = True
        self.destroy()

    def _current_hotkey_vk(self) -> int:
        value = self._hotkey_vk
        value = getattr(self.hotkey_watcher, "vk", value)
        if value is None:
            value = getattr(self.hotkey_watcher, "_vk", self.config.hotkey_vk)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(self.config.hotkey_vk)


__all__ = ["JitterApp"]
