"""Tkinter dashboard and safe runtime wiring for the standalone Jitter app."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import io
import logging
import math
from pathlib import Path
import queue
import re
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
import tokenize
from typing import Any, Callable, Mapping

from ai_service import AiEvent, AiService
from ai_targeting import (
    AIM_LIMITS,
    AimSettings,
    aim_settings_from_mapping,
    aim_settings_to_mapping,
)
from hotkeys import HotkeyWatcher
from makcu_service import MakcuService, ServiceEvent
from motion import (
    MOTION_LIMITS,
    MOTION_PRESETS,
    RAMP_MODES,
    MotionSettings,
    TriggerGate,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
)
from settings import AppConfig, ConfigStore
from liquid_widgets import LiquidIconButton, LiquidNavigation, LiquidSlider
from sound_service import ToggleSoundPlayer


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
_MODE_LABELS = {"jitter": "Jitter", "ai_aim": "AI Aim"}
_TEST_MOTION_MODES = {
    "test", "test_pending", "test_ai_loading", "test_ai",
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


@dataclass(frozen=True)
class _DeferredMotionAction:
    kind: str
    retiring_source: Any
    lifecycle_epoch: int
    mode: str
    ai_test_generation: int | None = None

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
        hotkey_factory: Callable[[int, Callable[[], None]], Any] | None = None,
        sound_player: Any | None = None,
        auto_start: bool = True,
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
        self._ui_queue: queue.SimpleQueue[
            tuple[str, int | None, Any]
        ] = queue.SimpleQueue()
        self._ai_event_epoch = 0
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
        self.enabled = False
        self._motion_mode: str | None = None
        self._test_restore_enabled = False
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
        self._ai_test_generation = 0
        self._ai_test_pending_generation: int | None = None
        self._ai_test_waiting_for_motion_stop = False
        self._rounded_style_images: dict[str, tuple[tk.PhotoImage, ...]] = {}
        self._motion_lock = threading.RLock()
        self._motion_snapshot: MotionSettings = self.config.motion
        self._ai_lock = threading.RLock()
        self._ai_snapshot: AimSettings = self.config.ai
        self._hotkey_vk = int(self.config.hotkey_vk)

        self._configure_styles()
        self._create_variables()
        self._build_page()

        self.service_factory = service_factory or (lambda sink: MakcuService(sink))
        self.ai_service_factory = ai_service_factory or (lambda sink: AiService(sink))
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
                Path(__file__).resolve().parent / "sound",
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

        style.configure("Liquid.App.TFrame", background=p["window"])
        style.configure("Liquid.Surface.TFrame", background=p["surface"],
                        bordercolor=p["border"], relief="flat", borderwidth=0)
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
        button_layout = lambda element: [
            (element, {
                "sticky": "nsew",
                "children": [("Button.padding", {
                    "sticky": "nsew",
                    "children": [("Button.label", {"sticky": "nsew"})],
                })],
            }),
        ]
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
        self.mode_var = tk.StringVar(self, self.config.mode)
        self.mode_display_var = tk.StringVar(
            self, _MODE_LABELS.get(self.config.mode, _MODE_LABELS["jitter"])
        )
        self.ai_status_var = tk.StringVar(self, "Stopped")
        self.ai_fps_var = tk.StringVar(self, "0 FPS")
        self.ai_provider_var = tk.StringVar(self, "No provider")
        self.ai_vars = {
            key: tk.StringVar(self, value)
            for key, value in aim_settings_to_mapping(self.config.ai).items()
        }
        self.motion_summary_var = tk.StringVar(
            self, _motion_summary_text(self._motion_snapshot)
        )
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
        self.shell.columnconfigure(0, weight=0, minsize=176)
        self.shell.columnconfigure(1, weight=1)
        self.shell.rowconfigure(0, weight=1)
        self.shell.bind("<Configure>", self._redraw_shell_art, add="+")

        self.navigation_rail = ttk.Frame(
            self.shell,
            width=176,
            style="Liquid.Surface.TFrame",
            padding=(12, 14),
        )
        self.navigation_rail.grid(
            row=0, column=0, sticky="ns", padx=(12, 8), pady=12
        )
        self.navigation_rail.grid_propagate(False)

        self.rail_identity = ttk.Frame(
            self.navigation_rail, style="Liquid.Surface.TFrame"
        )
        self.rail_identity.pack(side="top", fill="x")
        # Preserve the established identity seam for existing integrations.
        self.identity_frame = self.rail_identity
        self._build_identity()

        self.navigation_frame = ttk.Frame(
            self.navigation_rail, style="Liquid.Surface.TFrame"
        )
        self.navigation_frame.pack(side="top", fill="x", pady=(18, 0))
        self.nav = LiquidNavigation(
            self.navigation_frame,
            labels=("Control", "Motion", "Settings"),
            command=self.select_page,
            palette=self._navigation_palette(),
            orientation="vertical",
            width=152,
            height=168,
        )
        self.nav.pack(fill="x")
        self._build_navigation_actions()

        self.console_workspace = ttk.Frame(
            self.shell, style="Liquid.App.TFrame"
        )
        self.console_workspace.grid(
            row=0, column=1, sticky="nsew", padx=(8, 14), pady=(12, 10)
        )
        self.console_workspace.columnconfigure(0, weight=1)
        self.console_workspace.rowconfigure(0, weight=1)

        self.page_host = ttk.Frame(
            self.console_workspace, style="Liquid.App.TFrame"
        )
        self.page_host.grid(row=0, column=0, sticky="nsew")
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)
        self.control_page = ttk.Frame(self.page_host, style="Liquid.App.TFrame")
        self.motion_page = ttk.Frame(self.page_host, style="Liquid.App.TFrame")
        self.settings_page = ttk.Frame(self.page_host, style="Liquid.App.TFrame")
        self.pages = (
            self.control_page, self.motion_page, self.settings_page,
        )
        for page in self.pages:
            page.grid(row=0, column=0, sticky="nsew")

        self._build_trigger_card()
        self._build_quick_card()
        self._build_settings_page()
        self._apply_combobox_popup_palette()
        self.select_page(0)
        self._build_main_control_card()
        self._build_footer()
        for panel in (
            self.navigation_rail,
            self.page_host,
            self.runtime_frame,
        ):
            panel.bind("<Configure>", self._redraw_shell_art, add="+")
        self._redraw_shell_art()

    def _build_identity(self) -> None:
        identity_copy = ttk.Frame(
            self.identity_frame, style="Liquid.Surface.TFrame"
        )
        identity_copy.pack(side="top", fill="x")
        ttk.Label(
            identity_copy, text="Jitter", style="Liquid.Title.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            identity_copy,
            text="MAKCU MOTION",
            style="Liquid.Subtitle.TLabel",
        ).pack(anchor="w", pady=(1, 0))
        connection_row = ttk.Frame(
            self.identity_frame, style="Liquid.Surface.TFrame"
        )
        connection_row.pack(side="top", fill="x", pady=(14, 0))
        self.connection_label = ttk.Label(
            connection_row,
            textvariable=self.connection_status_var,
            style="Liquid.StatusDisconnected.TLabel",
        )
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
        self.connection_label.pack(side="left")
        self._redraw_connection_indicator()

    def toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_var.set(self._theme)
        self._configure_styles()
        self.configure(background=self._palette["window"])
        self.shell.configure(background=self._palette["window"])
        self._redraw_shell_art()
        self._redraw_connection_indicator()
        self.nav.set_palette(self._navigation_palette())
        self.theme_button.icon = "☀" if self._theme == "dark" else "☾"
        self.theme_tooltip_text = (
            "Switch to Light Mode" if self._theme == "dark"
            else "Switch to Dark Mode"
        )
        self.theme_button.accessible_name = self.theme_tooltip_text
        self._hide_theme_tooltip()
        self._apply_combobox_popup_palette()
        icon_palette = self._icon_palette()
        for button in (
            self.reconnect_button,
            self.test_button,
            self.theme_button,
        ):
            button.set_palette(icon_palette)
        slider_palette = self._slider_palette()
        for widget in self.winfo_children():
            self._apply_slider_palette(widget, slider_palette)
        for name in (
            "sound_volume_scale",
            "pulse_size_px_scale",
            "pulse_rate_hz_scale",
            "ai_confidence_scale",
            "ai_aim_strength_scale",
            "ai_smoothing_scale",
            "ai_max_step_scale",
        ):
            surface_slider = getattr(self, name, None)
            if surface_slider is not None:
                surface_slider.set_palette(
                    self._slider_palette(on_surface=True)
                )
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

    def _navigation_palette(self) -> dict[str, str]:
        p = self._palette
        return {
            "background": p["window"], "surface": p["surface"],
            "surface_highlight": p["raised"], "border": p["border"],
            "lens": p["accent"],
            "lens_highlight": "#B8F6FF" if self._theme == "dark" else "#C7F8FF",
            "text": p["text"], "selected_text": "#07252C",
            "focus": p["focus"],
        }

    def _icon_palette(self) -> dict[str, str]:
        p = self._palette
        return {
            "background": p["window"], "surface": p["raised"],
            "surface_hover": (
                "#2A3B52" if self._theme == "dark" else "#D6F5FA"
            ),
            "surface_pressed": p["surface"],
            "surface_disabled": p["disabled_surface"], "border": p["border"],
            "icon": p["text"], "icon_disabled": p["icon_disabled"],
            "highlight": p["surface"], "focus": p["focus"],
        }

    def _apply_combobox_popup_palette(self) -> None:
        p = self._palette
        for combo in (
            self.trigger_combo,
            self.modifier_combo,
            self.mode_combo,
            self.preset_combo,
            self.ramp_mode_combo,
        ):
            try:
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
            padding=(10, 8),
        )
        self.runtime_frame.grid(
            row=2, column=0, sticky="ew", pady=(8, 0)
        )
        self.runtime_frame.columnconfigure(0, weight=1, uniform="runtime_actions")
        self.runtime_frame.columnconfigure(1, weight=2)
        self.runtime_frame.columnconfigure(2, weight=1, uniform="runtime_actions")
        self.enable_button = ttk.Button(
            self.runtime_frame,
            text=f"Enable {_MODE_LABELS.get(self.mode_var.get(), 'Jitter')}",
            style="Liquid.Primary.TButton",
            command=self.toggle_enabled,
        )
        self.enable_button.grid(row=0, column=0, sticky="ew")
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

    def _build_dashboard_header(
        self,
        parent: ttk.Frame,
        eyebrow: str,
        title: str,
        subtitle: str,
    ) -> tuple[ttk.Frame, ttk.Label]:
        header = ttk.Frame(parent, style="Liquid.App.TFrame")
        header.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14)
        )
        ttk.Label(
            header,
            text=eyebrow,
            style="Liquid.SettingsEyebrow.TLabel",
        ).pack(anchor="w")
        title_label = ttk.Label(
            header,
            text=title,
            style="Liquid.SettingsTitle.TLabel",
            font=(FONT_FAMILY, 22, "bold"),
        )
        title_label.pack(anchor="w", pady=(2, 0))
        ttk.Label(
            header,
            text=subtitle,
            style="Liquid.Muted.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        return header, title_label

    def _build_trigger_card(self) -> None:
        self.control_page.columnconfigure(0, weight=3, uniform="control")
        self.control_page.columnconfigure(1, weight=2, uniform="control")
        self.control_page.rowconfigure(1, weight=1)
        self.control_header_frame, self.control_title_label = (
            self._build_dashboard_header(
                self.control_page,
                "INPUT AND DEVICE SETUP",
                "CONTROL",
                "Choose how Jitter arms and which motion preset is active.",
            )
        )
        self.control_bindings_card = ttk.Frame(
            self.control_page,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.control_bindings_card.grid(
            row=1, column=0, sticky="nsew", padx=(0, 6)
        )
        self.control_bindings_card.columnconfigure(0, weight=1, uniform="binding")
        self.control_bindings_card.columnconfigure(1, weight=1, uniform="binding")
        self.control_bindings_card.rowconfigure(3, weight=1)
        self.control_device_card = ttk.Frame(
            self.control_page,
            style="Liquid.SettingsCard.TFrame",
            padding=(16, 16, 16, 18),
        )
        self.control_device_card.grid(
            row=1, column=1, sticky="nsew", padx=(6, 0)
        )
        self.control_device_card.columnconfigure(0, weight=1)
        self.control_device_card.rowconfigure(3, weight=1)
        # Preserve the established public seam for integrations.
        self.control_frame = self.control_bindings_card

        ttk.Label(
            self.control_bindings_card,
            text="INPUT BINDINGS",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            self.control_bindings_card,
            text="Hold the trigger and optional modifier to begin movement.",
            style="Liquid.CardBody.TLabel",
            wraplength=330,
            justify="left",
        ).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 14)
        )

        ttk.Label(
            self.control_device_card,
            text="DEVICE SETUP",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.control_device_card,
            text="Monitor Makcu and choose a motion preset.",
            style="Liquid.CardBody.TLabel",
            wraplength=205,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 14))
        device_row = ttk.Frame(
            self.control_device_card,
            style="Liquid.Metric.TFrame",
            padding=(12, 10),
        )
        device_row.grid(row=2, column=0, sticky="ew")
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
            self.control_bindings_card, 2, 0, "Trigger", self.trigger_var,
            ("Left", "Right", "Middle", "Mouse4", "Mouse5"), 10,
            padx=(0, 5),
        )
        self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.modifier_combo = combo_card(
            self.control_bindings_card, 2, 1, "Modifier", self.modifier_var,
            ("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"), 10,
            padx=(5, 0),
        )
        self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.mode_combo = combo_card(
            self.control_device_card, 3, 0, "Mode", self.mode_display_var,
            tuple(_MODE_LABELS.values()), 14,
            pady=(14, 0),
        )
        self.mode_combo.bind("<<ComboboxSelected>>", self._mode_selected)
        self.preset_combo = combo_card(
            self.control_device_card, 4, 0, "Preset", self.preset_var,
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
            row=4, column=0, columnspan=2, sticky="sew", pady=(16, 0)
        )

    def _build_navigation_actions(self) -> None:
        self.navigation_actions = ttk.Frame(
            self.navigation_rail, style="Liquid.Surface.TFrame"
        )
        self.navigation_actions.pack(side="bottom", anchor="center")
        self.reconnect_tooltip_text = "Reconnect Makcu"
        self.test_tooltip_text = "Test Run 3s"
        self._action_tooltip: tk.Toplevel | None = None
        self.reconnect_button = LiquidIconButton(
            self.navigation_actions,
            icon="↻",
            accessible_name=self.reconnect_tooltip_text,
            command=self.reconnect,
            palette=self._icon_palette(),
        )
        self.test_button = LiquidIconButton(
            self.navigation_actions,
            icon="▶",
            accessible_name=self.test_tooltip_text,
            command=self.test_run,
            palette=self._icon_palette(),
        )
        self.theme_tooltip_text = (
            "Switch to Light Mode" if self._theme == "dark"
            else "Switch to Dark Mode"
        )
        self._theme_tooltip: tk.Toplevel | None = None
        self.theme_button = LiquidIconButton(
            self.navigation_actions,
            icon="☀" if self._theme == "dark" else "☾",
            accessible_name=self.theme_tooltip_text,
            command=self.toggle_theme,
            palette=self._icon_palette(),
        )
        self.reconnect_button.pack(side="left", padx=(0, 5))
        self.test_button.pack(side="left", padx=(0, 5))
        self.theme_button.pack(side="left")
        self.reconnect_button.bind(
            "<Enter>",
            lambda event: self._show_action_tooltip(
                event, self.reconnect_tooltip_text
            ),
            add="+",
        )
        self.test_button.bind(
            "<Enter>",
            lambda event: self._show_action_tooltip(
                event, self.test_tooltip_text
            ),
            add="+",
        )
        self.reconnect_button.bind(
            "<Leave>", self._hide_action_tooltip, add="+"
        )
        self.test_button.bind("<Leave>", self._hide_action_tooltip, add="+")
        self.reconnect_button.bind(
            "<FocusIn>",
            lambda event: self._show_action_tooltip(
                event, self.reconnect_tooltip_text
            ),
            add="+",
        )
        self.test_button.bind(
            "<FocusIn>",
            lambda event: self._show_action_tooltip(
                event, self.test_tooltip_text
            ),
            add="+",
        )
        self.reconnect_button.bind(
            "<FocusOut>", self._hide_action_tooltip, add="+"
        )
        self.test_button.bind("<FocusOut>", self._hide_action_tooltip, add="+")
        self.theme_button.bind(
            "<Enter>", self._show_theme_tooltip, add="+"
        )
        self.theme_button.bind(
            "<Leave>", self._hide_theme_tooltip, add="+"
        )
        self.theme_button.bind(
            "<FocusIn>", self._show_theme_tooltip, add="+"
        )
        self.theme_button.bind(
            "<FocusOut>", self._hide_theme_tooltip, add="+"
        )

    def _show_action_tooltip(self, event: tk.Event, text: str) -> None:
        self._hide_action_tooltip()
        widget = getattr(event, "widget", None)
        if widget is None or self._closing:
            return
        try:
            if not widget.winfo_exists():
                return
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(
                f"+{widget.winfo_rootx()}+{widget.winfo_rooty() - 26}"
            )
            p = self._palette
            tk.Label(
                tooltip,
                text=text,
                background=p["raised"],
                foreground=p["text"],
                borderwidth=1,
                relief="solid",
                font=SMALL_FONT,
                padx=5,
                pady=2,
            ).pack()
        except (tk.TclError, RuntimeError):
            return
        self._action_tooltip = tooltip

    def _hide_action_tooltip(self, _event: tk.Event | None = None) -> None:
        tooltip = getattr(self, "_action_tooltip", None)
        if tooltip is not None:
            try:
                tooltip.destroy()
            except tk.TclError:
                pass
            self._action_tooltip = None

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

    def _build_quick_card(self) -> None:
        self.motion_page.columnconfigure(0, weight=3, uniform="motion")
        self.motion_page.columnconfigure(1, weight=2, uniform="motion")
        self.motion_page.rowconfigure(1, weight=1)
        self.motion_header_frame, self.motion_title_label = (
            self._build_dashboard_header(
                self.motion_page,
                "PAIRED PULSE ENGINE",
                "MOTION",
                "Tune movement strength, cadence, and acceleration feel.",
            )
        )
        self.motion_hero_card = ttk.Frame(
            self.motion_page,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.motion_hero_card.grid(
            row=1, column=0, sticky="nsew", padx=(0, 6)
        )
        self.motion_hero_card.columnconfigure(0, weight=1)
        self.motion_hero_card.rowconfigure(2, weight=1)
        self.motion_summary_card = ttk.Frame(
            self.motion_page,
            style="Liquid.SettingsCard.TFrame",
            padding=(16, 16, 16, 18),
        )
        self.motion_summary_card.grid(
            row=1, column=1, sticky="nsew", padx=(6, 0)
        )
        self.motion_summary_card.columnconfigure(0, weight=1)
        self.motion_summary_card.rowconfigure(2, weight=1)
        # Preserve the established public seams for integrations.
        self.quick_frame = self.motion_hero_card
        ttk.Label(
            self.motion_hero_card,
            text="MOTION SHAPE",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.motion_hero_card,
            text="Adjust the two-dimensional paired pulse sent to Makcu.",
            style="Liquid.CardBody.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 12))
        self.quick_grid = ttk.Frame(
            self.motion_hero_card, style="Liquid.Surface.TFrame"
        )
        self.quick_grid.grid(row=2, column=0, sticky="new")
        self.quick_grid.columnconfigure(0, weight=1, uniform="quick")
        self.quick_grid.columnconfigure(1, weight=1, uniform="quick")
        controls = (
            ("Pulse Size", "pulse_size_px", 1, 8, 1),
            ("Pulse Rate", "pulse_rate_hz", 20, 120, 1),
        )
        for index, control in enumerate(controls):
            self._numeric_control(self.quick_grid, index // 2, index % 2, *control)
        ramp_row, self.ramp_mode_combo = self._dropdown_field(
            self.quick_grid,
            label="Ramp Mode",
            variable=self.motion_vars["ramp_mode"],
            values=RAMP_MODES,
        )
        ramp_row.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 4)
        )

        ttk.Label(
            self.motion_summary_card,
            text="LIVE SNAPSHOT",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.motion_summary_card,
            text="The immutable profile currently shared with the mover.",
            style="Liquid.CardBody.TLabel",
            wraplength=200,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 12))
        self.motion_summary_frame = ttk.Frame(
            self.motion_summary_card,
            style="Liquid.Surface.TFrame",
            padding=0,
        )
        self.motion_summary_frame.grid(row=2, column=0, sticky="nsew")
        self.motion_summary_frame.columnconfigure(0, weight=1, uniform="metric")
        self.motion_summary_frame.columnconfigure(1, weight=1, uniform="metric")
        size_metric = ttk.Frame(
            self.motion_summary_frame,
            style="Liquid.Metric.TFrame",
            padding=(10, 6),
        )
        size_metric.grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        ttk.Label(
            size_metric, text="PULSE SIZE", style="Liquid.MetricLabel.TLabel"
        ).pack(anchor="w")
        size_value = ttk.Frame(size_metric, style="Liquid.Metric.TFrame")
        size_value.pack(anchor="w", pady=(3, 0))
        self.motion_size_readout = ttk.Label(
            size_value,
            textvariable=self.motion_snapshot_size_var,
            style="Liquid.MetricValue.TLabel",
            font=(FONT_FAMILY, 22, "bold"),
        )
        self.motion_size_readout.pack(side="left")
        ttk.Label(
            size_value, text=" px", style="Liquid.MetricUnit.TLabel"
        ).pack(side="left", pady=(8, 0))

        rate_metric = ttk.Frame(
            self.motion_summary_frame,
            style="Liquid.Metric.TFrame",
            padding=(10, 6),
        )
        rate_metric.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Label(
            rate_metric, text="PULSE RATE", style="Liquid.MetricLabel.TLabel"
        ).pack(anchor="w")
        rate_value = ttk.Frame(rate_metric, style="Liquid.Metric.TFrame")
        rate_value.pack(anchor="w", pady=(3, 0))
        self.motion_rate_readout = ttk.Label(
            rate_value,
            textvariable=self.motion_snapshot_rate_var,
            style="Liquid.MetricValue.TLabel",
            font=(FONT_FAMILY, 22, "bold"),
        )
        self.motion_rate_readout.pack(side="left")
        ttk.Label(
            rate_value, text=" Hz", style="Liquid.MetricUnit.TLabel"
        ).pack(side="left", pady=(8, 0))

        ramp_metric = ttk.Frame(
            self.motion_summary_frame,
            style="Liquid.Metric.TFrame",
            padding=(10, 6),
        )
        ramp_metric.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Label(
            ramp_metric, text="RAMP MODE", style="Liquid.MetricLabel.TLabel"
        ).pack(anchor="w")
        self.motion_ramp_readout = ttk.Label(
            ramp_metric,
            textvariable=self.motion_snapshot_ramp_var,
            style="Liquid.MetricValue.TLabel",
            font=(FONT_FAMILY, 13, "bold"),
        )
        self.motion_ramp_readout.pack(anchor="w", pady=(4, 0))

        ttk.Label(
            self.motion_summary_frame,
            text="ACTIVE PROFILE",
            style="Liquid.CardBody.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.motion_summary_label = ttk.Label(
            self.motion_summary_frame,
            textvariable=self.motion_summary_var,
            style="Liquid.CardText.TLabel",
            wraplength=200,
            justify="left",
        )
        self.motion_summary_label.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(3, 0)
        )

        self.ai_settings_card = ttk.Frame(
            self.motion_page,
            style="Liquid.SettingsCard.TFrame",
            padding=(18, 16, 18, 18),
        )
        self.ai_settings_card.grid(
            row=1, column=0, sticky="nsew", padx=(0, 6)
        )
        self.ai_settings_card.columnconfigure(0, weight=1)
        ttk.Label(
            self.ai_settings_card,
            text="AI AIM SETTINGS",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.ai_settings_card,
            text="Tune target acceptance and smooth closed-loop movement.",
            style="Liquid.CardBody.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 12))
        self.ai_controls_grid = ttk.Frame(
            self.ai_settings_card, style="Liquid.Surface.TFrame"
        )
        self.ai_controls_grid.grid(row=2, column=0, sticky="new")
        self.ai_controls_grid.columnconfigure(0, weight=1, uniform="ai_controls")
        self.ai_controls_grid.columnconfigure(1, weight=1, uniform="ai_controls")
        for index, (key, spec) in enumerate(_AI_CONTROL_SPECS.items()):
            self._ai_numeric_control(
                self.ai_controls_grid,
                index // 2,
                index % 2,
                spec[0],
                key,
                spec[1],
                spec[2],
                spec[3],
            )

        self.ai_status_card = ttk.Frame(
            self.motion_page,
            style="Liquid.SettingsCard.TFrame",
            padding=(16, 16, 16, 18),
        )
        self.ai_status_card.grid(
            row=1, column=1, sticky="nsew", padx=(6, 0)
        )
        self.ai_status_card.columnconfigure(0, weight=1)
        ttk.Label(
            self.ai_status_card,
            text="AI RUNTIME",
            style="Liquid.CardTitle.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.ai_status_card,
            text="Capture, provider, and inference health stay visible here.",
            style="Liquid.CardBody.TLabel",
            wraplength=200,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 12))
        for row, (label, variable) in enumerate((
            ("STATUS", self.ai_status_var),
            ("INFERENCE", self.ai_fps_var),
            ("PROVIDER", self.ai_provider_var),
        ), start=2):
            metric = ttk.Frame(
                self.ai_status_card,
                style="Liquid.Metric.TFrame",
                padding=(10, 8),
            )
            metric.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            ttk.Label(
                metric, text=label, style="Liquid.MetricLabel.TLabel"
            ).pack(anchor="w")
            ttk.Label(
                metric,
                textvariable=variable,
                style="Liquid.MetricValue.TLabel",
                font=(FONT_FAMILY, 12, "bold"),
                wraplength=190,
                justify="left",
            ).pack(anchor="w", pady=(4, 0))
        self._show_mode_panels()

    def _build_settings_page(self) -> None:
        self.settings_page.columnconfigure(0, weight=1)
        self.settings_page.rowconfigure(1, weight=1)

        self.settings_header_frame = ttk.Frame(
            self.settings_page, style="Liquid.App.TFrame"
        )
        self.settings_header_frame.grid(
            row=0, column=0, sticky="ew", pady=(0, 16)
        )
        ttk.Label(
            self.settings_header_frame,
            text="PERSONALIZE YOUR CONTROL DECK",
            style="Liquid.SettingsEyebrow.TLabel",
        ).pack(anchor="w")
        self.settings_title_label = ttk.Label(
            self.settings_header_frame,
            text="SETTINGS",
            style="Liquid.SettingsTitle.TLabel",
            font=(FONT_FAMILY, 22, "bold"),
        )
        self.settings_title_label.pack(anchor="w", pady=(2, 0))
        ttk.Label(
            self.settings_header_frame,
            text="Tune the audible feedback used by the global hotkey.",
            style="Liquid.Muted.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        self.settings_content = ttk.Frame(
            self.settings_page, style="Liquid.App.TFrame"
        )
        self.settings_content.grid(row=1, column=0, sticky="nsew")
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
        ttk.Label(
            self.sound_feedback_card,
            text=(
                "Hear a clean cue whenever the global hotkey arms or "
                "disables Jitter."
            ),
            style="Liquid.CardBody.TLabel",
            wraplength=300,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 16))

        sound_toggle_row = ttk.Frame(
            self.sound_feedback_card, style="Liquid.Surface.TFrame"
        )
        sound_toggle_row.grid(row=2, column=0, sticky="ew")
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
        ).grid(row=3, column=0, sticky="ew", pady=16)

        volume_readout = ttk.Frame(
            self.sound_feedback_card, style="Liquid.Surface.TFrame"
        )
        volume_readout.grid(row=4, column=0, sticky="ew")
        volume_readout.columnconfigure(0, weight=1)
        ttk.Label(
            volume_readout,
            text="OUTPUT LEVEL",
            style="Liquid.CardBody.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        volume_value = ttk.Frame(
            volume_readout, style="Liquid.Surface.TFrame"
        )
        volume_value.grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            volume_value,
            textvariable=self.sound_volume_var,
            style="Liquid.Volume.TLabel",
            font=(FONT_FAMILY, 30, "bold"),
        ).pack(side="left")
        ttk.Label(
            volume_value,
            text="%",
            style="Liquid.VolumeUnit.TLabel",
            font=(FONT_FAMILY, 12, "bold"),
        ).pack(side="left", padx=(3, 0), pady=(13, 0))
        self.sound_volume_entry = ttk.Entry(
            volume_readout,
            textvariable=self.sound_volume_var,
            width=5,
            justify="right",
            style="Liquid.Entry.TEntry",
        )
        self.sound_volume_entry.grid(row=1, column=1, sticky="e", padx=(12, 0))

        volume_control = ttk.Frame(
            self.sound_feedback_card, style="Liquid.Surface.TFrame"
        )
        volume_control.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        volume_control.columnconfigure(0, weight=1)
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
        ttk.Label(
            self.sound_preview_card,
            text="Check both cues at the selected volume.",
            style="Liquid.CardBody.TLabel",
            wraplength=185,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(4, 12))

        actions = ttk.Frame(
            self.sound_preview_card, style="Liquid.Surface.TFrame"
        )
        actions.grid(row=2, column=0, sticky="new")
        actions.columnconfigure(0, weight=1)
        ttk.Label(
            actions,
            text="ARMED CUE",
            style="Liquid.CardBody.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.test_on_button = ttk.Button(
            actions,
            text="\u25b6",
            width=2,
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
            text="\u25b6",
            width=2,
            style="Liquid.CompactSecondary.TButton",
            command=lambda: self.preview_sound(False),
        )
        self.test_off_button.grid(row=2, column=1, sticky="e")

    def _build_footer(self) -> None:
        self.footer_frame = ttk.Frame(
            self.console_workspace, style="Liquid.App.TFrame"
        )
        self.footer_frame.grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )
        self.footer_label = ttk.Label(
            self.footer_frame,
            textvariable=self.footer_var,
            style="Liquid.Muted.TLabel",
            anchor="w",
        )
        self.footer_label.pack(side="left", fill="x", expand=True)

    def _show_theme_tooltip(self, _event: tk.Event | None = None) -> None:
        self._hide_theme_tooltip()
        if self._closing:
            return
        widget = getattr(_event, "widget", self.theme_button)
        try:
            if not widget.winfo_exists():
                return
            tooltip = tk.Toplevel(self)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(
                f"+{widget.winfo_rootx()}+{widget.winfo_rooty() - 26}"
            )
            p = self._palette
            tk.Label(
                tooltip,
                text=self.theme_tooltip_text,
                background=p["raised"],
                foreground=p["text"],
                borderwidth=1,
                relief="solid",
                font=SMALL_FONT,
                padx=5,
                pady=2,
            ).pack()
        except (tk.TclError, RuntimeError):
            return
        self._theme_tooltip = tooltip

    def _hide_theme_tooltip(self, _event: tk.Event | None = None) -> None:
        tooltip = getattr(self, "_theme_tooltip", None)
        if tooltip is not None:
            try:
                tooltip.destroy()
            except tk.TclError:
                pass
            self._theme_tooltip = None

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
        workspace_left = 176
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
            ("rail", "navigation_rail"),
            ("page", "page_host"),
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
            if name == "rail":
                panel_tags += ("rail-surface",)
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

    def select_page(self, index: int) -> None:
        selected = min(len(self.pages) - 1, max(0, int(index)))
        for page in self.pages:
            page.grid_remove()
        self.pages[selected].grid()
        if self.nav.selected_index != selected:
            self.nav.select(selected, notify=False)

    def _mode_selected(self, _event: tk.Event | None = None) -> None:
        selected_label = self.mode_display_var.get()
        selected_mode = next(
            (mode for mode, label in _MODE_LABELS.items()
             if label == selected_label),
            "jitter",
        )
        self.mode_var.set(selected_mode)
        self.on_mode_changed()

    def on_mode_changed(self) -> None:
        selected = self.mode_var.get()
        if selected not in _MODE_LABELS:
            selected = "jitter"
        self.emergency_stop("Mode changed")
        self.mode_var.set(selected)
        self.mode_display_var.set(_MODE_LABELS[selected])
        self._show_mode_panels()
        self.enable_button.configure(text=f"Enable {_MODE_LABELS[selected]}")
        self.footer_var.set(f"{_MODE_LABELS[selected]} mode selected")
        self._schedule_save()

    def _show_mode_panels(self) -> None:
        if self.mode_var.get() == "ai_aim":
            self.motion_hero_card.grid_remove()
            self.motion_summary_card.grid_remove()
            self.ai_settings_card.grid()
            self.ai_status_card.grid()
        else:
            self.ai_settings_card.grid_remove()
            self.ai_status_card.grid_remove()
            self.motion_hero_card.grid()
            self.motion_summary_card.grid()

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

    def toggle_enabled(self) -> None:
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        self.set_enabled(not self.enabled)

    def _start_ai_runtime(self, context: str) -> bool:
        if not self._ai_runtime_active:
            self._ai_event_epoch += 1
        self._ai_runtime_active = True
        try:
            generation = self.ai_service.start(self.get_ai_settings)
        except Exception as exc:
            logging.exception("AI runtime could not start during %s", context)
            self.handle_ai_event(
                AiEvent("error", f"{type(exc).__name__}: AI service failed")
            )
            return False
        if not generation:
            logging.error("AI runtime did not start during %s", context)
            self.handle_ai_event(
                AiEvent("error", "AI service returned no generation")
            )
            return False
        return True

    def _stop_ai_runtime(self, reason: str) -> None:
        try:
            self.ai_service.stop(reason)
        finally:
            self._ai_event_epoch += 1

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
        ai_test_generation: int | None = None,
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
            mode=self.mode_var.get(),
            ai_test_generation=ai_test_generation,
        )
        return True

    def _advance_hotkey_epoch(self) -> None:
        with self._hotkey_epoch_lock:
            self._hotkey_event_epoch += 1

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if not enabled:
            self.emergency_stop("Disabled by user")
            return
        if self._closing:
            return
        if enabled and self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        if not self.service.connected:
            self.enabled = False
            self._set_runtime_state("disabled")
            self.enable_button.configure(
                text=f"Enable {_MODE_LABELS.get(self.mode_var.get(), 'Jitter')}"
            )
            self.footer_var.set("Makcu device is not connected")
            return
        self._motion_event_epoch += 1
        self._advance_hotkey_epoch()
        self.enabled = True
        self._motion_mode = None
        self._normal_motion_started = False
        self._expected_motion_generation = None
        self._deferred_motion_action = None
        self.trigger_gate.clear()
        self._set_runtime_state("armed")
        mode_label = _MODE_LABELS.get(self.mode_var.get(), "Jitter")
        self.enable_button.configure(text=f"Disable {mode_label}")
        if self.mode_var.get() == "ai_aim":
            if not self._start_ai_runtime("AI Aim enable"):
                return
            self.footer_var.set("AI Aim armed")
        else:
            self.footer_var.set("Jitter armed")

    def test_run(self) -> None:
        self.start_test_run()

    def start_test_run(self) -> None:
        if self._closing:
            return
        if not self.service.connected:
            self.footer_var.set("Makcu device is not connected")
            return
        if self._motion_mode in _TEST_MOTION_MODES:
            self.footer_var.set("Test Run is already active")
            return
        if self.mode_var.get() == "ai_aim":
            self._start_ai_test_run()
            return
        self._deferred_motion_action = None
        self._test_restore_enabled = self.enabled
        normal_motion_active = self._normal_motion_started
        if normal_motion_active:
            self._stop_motion_runtime("test_run")
            self._normal_motion_started = False
        self._set_runtime_state("testing")
        self.test_button.set_enabled(False)
        self._motion_mode = "test_pending"
        self._test_start_pending = True
        if self._defer_motion_action("jitter_test"):
            # The normal worker emits a queued stop event after stop_motion().
            # Start the timed worker only after that event, so the service's
            # idempotent start contract cannot accidentally reuse the normal
            # worker.
            return
        started = self._begin_test_motion()
        if not started:
            self.footer_var.set("Test Run could not start")

    def _start_ai_test_run(self) -> None:
        self._deferred_motion_action = None
        self._test_restore_enabled = self.enabled
        self._ai_test_generation += 1
        generation = self._ai_test_generation
        self._ai_test_pending_generation = generation
        if self._normal_motion_started:
            self._stop_motion_runtime("test_run")
            self._normal_motion_started = False
        self._motion_mode = "test_ai_loading"
        self._test_start_pending = True
        self._set_runtime_state("testing")
        self.test_button.set_enabled(False)
        self._ai_test_waiting_for_motion_stop = self._defer_motion_action(
            "ai_test",
            ai_test_generation=generation,
        )

        if self._ai_ready and not self._ai_test_waiting_for_motion_stop:
            if not self._begin_ai_test_motion(generation):
                self.footer_var.set("AI Test Run could not start")
            return
        if not self._ai_runtime_active:
            self._start_ai_runtime("AI Test Run")

    def _begin_ai_test_motion(self, generation: int) -> bool:
        if (self._motion_mode != "test_ai_loading"
                or self._ai_test_pending_generation != generation
                or self._ai_test_waiting_for_motion_stop
                or not self._ai_ready):
            return False
        self._motion_mode = "test_ai"
        self._test_start_pending = False
        self._ai_test_pending_generation = None
        self._motion_event_epoch += 1
        started = self._request_motion_start(ai=True, duration_s=3.0)
        if not started:
            self._restore_after_test()
        return started

    def _begin_test_motion(self) -> bool:
        self._motion_mode = "test"
        self._test_start_pending = False
        self._motion_event_epoch += 1
        started = self._request_motion_start(ai=False, duration_s=3.0)
        if not started:
            self._motion_mode = None
            self.test_button.set_enabled(True)
            self._restore_after_test()
        return started

    def _restore_after_test(self) -> None:
        was_ai_test = self._motion_mode in {"test_ai_loading", "test_ai"}
        restore = self._test_restore_enabled and bool(self.service.connected)
        self._motion_event_epoch += 1
        self._expected_motion_generation = None
        self._deferred_motion_action = None
        self._motion_mode = None
        self._test_start_pending = False
        self._ai_test_pending_generation = None
        self._ai_test_waiting_for_motion_stop = False
        self.test_button.set_enabled(True)
        mode_label = _MODE_LABELS.get(self.mode_var.get(), "Jitter")
        if restore:
            self.enabled = True
            self._set_runtime_state("armed")
            self.enable_button.configure(text=f"Disable {mode_label}")
        else:
            self.enabled = False
            self.trigger_gate.clear()
            self._set_runtime_state("disabled")
            self.enable_button.configure(text=f"Enable {mode_label}")
            if was_ai_test:
                try:
                    self._stop_ai_runtime("test_complete")
                except Exception:
                    pass
                self._ai_ready = False
                self._ai_provider = None
                self._ai_runtime_active = False

    def emergency_stop(
        self,
        reason: str = "Stopped",
        *,
        stop_device_motion: bool = True,
    ) -> None:
        stop_reason = str(reason or "Stopped")
        self.enabled = False
        self._normal_motion_started = False
        self._deferred_motion_action = None
        self._motion_mode = None
        self._test_start_pending = False
        self._ai_test_generation += 1
        self._ai_test_pending_generation = None
        self._ai_test_waiting_for_motion_stop = False
        self._ai_ready = False
        self._ai_provider = None
        self._ai_runtime_active = False
        self.trigger_gate.clear()
        try:
            self._stop_ai_runtime(stop_reason)
        except Exception:
            pass
        try:
            self._stop_motion_runtime(
                stop_reason,
                stop_device_motion=stop_device_motion,
            )
        except Exception:
            pass
        self._advance_hotkey_epoch()
        self._set_runtime_state("disabled")
        self.enable_button.configure(
            text=f"Enable {_MODE_LABELS.get(self.mode_var.get(), 'Jitter')}"
        )
        self.test_button.set_enabled(True)
        self.footer_var.set(stop_reason)

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
            or self.mode_var.get() != action.mode
        ):
            return

        if action.kind == "normal":
            if (
                not self.enabled
                or self._motion_mode is not None
                or not self.trigger_gate.active
            ):
                return
            self._normal_motion_started = self._start_gated_motion()
            if self._normal_motion_started:
                self._set_runtime_state("moving")
            return

        if action.kind == "jitter_test":
            if (
                action.mode != "jitter"
                or self._motion_mode != "test_pending"
                or not self._test_start_pending
            ):
                return
            if not self._begin_test_motion():
                self.footer_var.set("Test Run could not start")
            return

        if action.kind != "ai_test":
            return
        if (
            action.mode != "ai_aim"
            or self._motion_mode != "test_ai_loading"
            or not self._test_start_pending
            or not self._ai_test_waiting_for_motion_stop
            or action.ai_test_generation is None
            or self._ai_test_pending_generation != action.ai_test_generation
        ):
            return
        self._ai_test_waiting_for_motion_stop = False
        if self._ai_ready and not self._begin_ai_test_motion(
            action.ai_test_generation
        ):
            self.footer_var.set("AI Test Run could not start")

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
            self.emergency_stop("Device disconnected")
        elif kind == "button":
            try:
                button, pressed = event.payload
            except (TypeError, ValueError):
                return
            self.trigger_gate.update_button(str(button), bool(pressed))
            if not self.trigger_gate.active:
                action = self._deferred_motion_action
                if action is not None and action.kind == "normal":
                    self._deferred_motion_action = None
            if self.enabled and self._motion_mode is None:
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
                self._motion_mode == "test"
                and reason == "duration_complete"
            ):
                self._restore_after_test()
                self.footer_var.set("Test Run complete")
            elif (
                self._motion_mode == "test_ai"
                and reason == "duration_complete"
            ):
                self._restore_after_test()
                self.footer_var.set("Test Run complete")
            elif self._motion_mode is None and self._normal_motion_started:
                self._expected_motion_generation = None
                self._normal_motion_started = False
                if self.enabled:
                    self._set_runtime_state("armed")

    def _request_motion_start(
        self,
        *,
        ai: bool,
        duration_s: float | None = None,
    ) -> bool:
        """Reserve motion and remember the Makcu generation returned by start."""
        self._expected_motion_generation = None
        if ai:
            source_start = getattr(
                self.service,
                "start_ai_motion_source",
                None,
            )
            if callable(source_start):
                source = source_start(
                    self.ai_service.latest_snapshot,
                    self.get_ai_settings,
                    duration_s=duration_s,
                )
            else:
                started = self.service.start_ai_motion(
                    self.ai_service.latest_snapshot,
                    self.get_ai_settings,
                    duration_s=duration_s,
                )
                source = (
                    getattr(self.service, "motion_generation", started)
                    if started else None
                )
        else:
            source_start = getattr(
                self.service,
                "start_motion_source",
                None,
            )
            if callable(source_start):
                source = source_start(
                    self.get_motion_settings,
                    duration_s=duration_s,
                )
            else:
                started = self.service.start_motion(
                    self.get_motion_settings,
                    duration_s=duration_s,
                )
                source = (
                    getattr(self.service, "motion_generation", started)
                    if started else None
                )
        if source is None or source is False:
            return False
        self._expected_motion_generation = source
        return True

    def _start_gated_motion(self) -> bool:
        ai = self.mode_var.get() == "ai_aim"
        if ai and not self._ai_ready:
            return False
        if self._defer_motion_action("normal"):
            return False
        self._deferred_motion_action = None
        return self._request_motion_start(ai=ai)

    def handle_ai_event(self, event: AiEvent) -> None:
        if self._closing:
            return
        kind = event.kind
        if (
            kind in {"loading", "ready", "fps", "error"}
            and not self._ai_runtime_active
        ):
            return
        if kind == "loading":
            self._ai_runtime_active = True
            self._ai_ready = False
            self._ai_provider = None
            self.ai_status_var.set("Loading")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
        elif kind == "ready":
            raw_provider = str(event.payload or "Unknown")
            provider = {
                "DmlExecutionProvider": "DirectML",
                "CPUExecutionProvider": "CPU",
            }.get(raw_provider, raw_provider)
            self._ai_ready = True
            self._ai_provider = raw_provider
            self._ai_runtime_active = True
            self.ai_status_var.set(f"Ready ({provider})")
            self.ai_provider_var.set(provider)
            if (self._motion_mode == "test_ai_loading"
                    and self._ai_test_pending_generation is not None
                    and not self._ai_test_waiting_for_motion_stop):
                generation = self._ai_test_pending_generation
                if not self._begin_ai_test_motion(generation):
                    self.footer_var.set("AI Test Run could not start")
            elif (self.enabled and self.mode_var.get() == "ai_aim"
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
        elif kind == "error":
            logging.error("AI runtime error: %s", event.payload)
            try:
                self._stop_ai_runtime("ai_error")
            except Exception:
                logging.exception("AI runtime stop failed after error")
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self.ai_status_var.set("Error")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")
            self.enabled = False
            self._normal_motion_started = False
            self._motion_mode = None
            self._test_start_pending = False
            self._deferred_motion_action = None
            self._ai_test_generation += 1
            self._ai_test_pending_generation = None
            self._ai_test_waiting_for_motion_stop = False
            self.trigger_gate.clear()
            try:
                self._stop_motion_runtime("ai_error")
            except Exception:
                pass
            self._advance_hotkey_epoch()
            self._set_runtime_state("disabled")
            self.enable_button.configure(text="Enable AI Aim")
            self.test_button.set_enabled(True)
            self.footer_var.set(
                "AI Aim stopped; switch to Jitter or try again"
            )
        elif kind == "stopped":
            self._ai_ready = False
            self._ai_provider = None
            self._ai_runtime_active = False
            self.ai_status_var.set("Stopped")
            self.ai_fps_var.set("0 FPS")
            self.ai_provider_var.set("No provider")

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
        action = self._deferred_motion_action
        self._deferred_motion_action = None
        if action is not None and action.kind in {"jitter_test", "ai_test"}:
            self._restore_after_test()
            self.footer_var.set("Bindings changed; Test Run canceled")
        self.trigger_gate.configure(trigger, modifier)
        if self._normal_motion_started:
            self._stop_motion_runtime("bindings_changed")
            self._normal_motion_started = False
            if self.enabled:
                self._set_runtime_state("armed")
        self._schedule_save()

    def get_motion_settings(self) -> MotionSettings:
        with self._motion_lock:
            return self._motion_snapshot

    def get_ai_settings(self) -> AimSettings:
        with self._ai_lock:
            return self._ai_snapshot

    def _replace_ai_snapshot(self, settings: AimSettings) -> None:
        with self._ai_lock:
            self._ai_snapshot = settings

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

    def _ai_changed(self, key: str) -> None:
        if self._updating_ai_controls or self._closing:
            return
        mapping = {name: variable.get() for name, variable in self.ai_vars.items()}
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
            ai=self.get_ai_settings(),
            mode=self.mode_var.get(),
            trigger=self.trigger_var.get(),
            modifier=self.modifier_var.get(),
            hotkey_vk=self._current_hotkey_vk(),
            hotkey_name=self.hotkey_name_var.get(),
            selected_preset=self.preset_var.get() or "Custom",
            theme=self.theme_var.get(),
            sound_enabled=self.sound_enabled_var.get(),
            sound_volume=sound_volume,
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
        self.nav.cancel_animation()
        for widget in self.winfo_children():
            self._cancel_slider_callbacks(widget)
        self._hide_action_tooltip()
        self._hide_theme_tooltip()
        self._cancel_after("_save_after_id")
        self._cancel_after("_capture_after_id")
        self._cancel_after("_ui_pump_after_id")
        self._capturing_hotkey = False
        self.emergency_stop("Stopped on close")
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
