"""Tkinter dashboard and safe runtime wiring for the standalone Jitter app."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
import math
import queue
import threading
import time
from typing import Mapping
from typing import Any, Callable

from hotkeys import HotkeyWatcher
from makcu_service import MakcuService, ServiceEvent
from motion import (
    JITTER_WAVEFORMS,
    MOTION_CURVES,
    MOTION_LIMITS,
    MOTION_PRESETS,
    MotionSettings,
    TriggerGate,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
)
from settings import AppConfig, ConfigStore
from liquid_widgets import LiquidIconButton, LiquidNavigation, LiquidSlider


_UI_QUEUE_MAX_BATCH = 50
_UI_QUEUE_TIME_SLICE_S = 0.005
_UI_QUEUE_IDLE_DELAY_MS = 15
_RUNTIME_STATE_LABELS = {
    "disabled": "DISABLED",
    "armed": "ARMED",
    "testing": "TESTING",
    "moving": "MOVING",
}

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

FONT_FAMILY = "Segoe UI"
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
        f"Strength {_display_value(settings.strength_pps)} px/s | "
        f"Angle {_display_value(settings.angle_deg)} deg | "
        f"Jitter {_display_value(settings.horizontal_jitter_pps)} x "
        f"{_display_value(settings.vertical_jitter_pps)} px/s at "
        f"{_display_value(settings.jitter_rate_hz)} Hz | "
        f"{settings.jitter_waveform} | "
        f"Smooth {_display_value(settings.smoothness)}%"
    )


class JitterApp(tk.Tk):
    """Fixed-size Liquid Control Deck for Jitter.

    The factories make the shell hardware-free in tests and give the runtime
    layer a narrow seam for the real Makcu and global-hotkey services.
    """

    def __init__(
        self,
        *,
        config_store: ConfigStore | None = None,
        service_factory: Callable[[Callable[[Any], None]], Any] | None = None,
        hotkey_factory: Callable[[int, Callable[[], None]], Any] | None = None,
        auto_start: bool = True,
    ) -> None:
        super().__init__()
        self.title("Jitter " + chr(0x2014) + " Makcu Control")
        self.geometry("780x640")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        self.config_store = config_store or ConfigStore()
        self.load_outcome = self.config_store.load()
        self.config: AppConfig = self.load_outcome.config
        self._theme = self.config.theme
        self.configure(background=self._palette["window"])
        self._save_allowed = bool(self.load_outcome.save_allowed)
        self._advanced_visible = False
        self._closed = False
        self._runtime_started = False
        self._closing = False
        self._save_after_id: str | None = None
        self._capture_after_id: str | None = None
        self._ui_pump_after_id: str | None = None
        self._ui_queue: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        self._capturing_hotkey = False
        self._capture_seen_down = False
        self._capture_prev_down: dict[int, bool] = {}
        self._updating_motion_controls = False
        self._invalid_motion_keys: set[str] = set()
        self.enabled = False
        self._motion_mode: str | None = None
        self._test_restore_enabled = False
        self._test_start_pending = False
        self._normal_motion_started = False
        self._motion_lock = threading.RLock()
        self._motion_snapshot: MotionSettings = self.config.motion
        self._hotkey_vk = int(self.config.hotkey_vk)

        self._configure_styles()
        self._create_variables()
        self._build_page()
        self.bind("<MouseWheel>", self._on_advanced_mousewheel, add="+")

        self.service_factory = service_factory or (lambda sink: MakcuService(sink))
        self.hotkey_factory = hotkey_factory or HotkeyWatcher
        self.service = self.service_factory(self.queue_service_event)
        self.hotkey_watcher = self.hotkey_factory(
            self.config.hotkey_vk, self._hotkey_pressed
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

        style.configure("Liquid.App.TFrame", background=p["window"])
        style.configure("Liquid.Surface.TFrame", background=p["surface"],
                        bordercolor=p["border"], relief="flat", borderwidth=0)
        style.configure("Liquid.Title.TLabel", background=p["surface"],
                        foreground=p["text"], font=TITLE_FONT)
        style.configure("Liquid.Subtitle.TLabel", background=p["surface"],
                        foreground=p["muted"], font=SMALL_FONT)
        style.configure("Liquid.Body.TLabel", background=p["window"],
                        foreground=p["text"], font=BODY_FONT)
        style.configure("Liquid.Muted.TLabel", background=p["window"],
                        foreground=p["muted"], font=SMALL_FONT)
        style.configure("Liquid.Card.TLabelframe", background=p["window"],
                        foreground=p["text"], bordercolor=p["border"],
                        relief="solid", borderwidth=1)
        style.configure("Liquid.Card.TLabelframe.Label", background=p["window"],
                        foreground=p["muted"], font=SECTION_FONT)
        style.configure("Liquid.Primary.TButton", background=p["accent"],
                        foreground="#07252C", bordercolor=p["accent_pressed"],
                        focuscolor=p["focus"], focusthickness=1,
                        relief="raised", borderwidth=1,
                        font=(FONT_FAMILY, 10, "bold"), padding=(14, 8))
        style.map("Liquid.Primary.TButton",
                  background=[("disabled", disabled_background),
                              ("pressed", p["accent_pressed"]),
                              ("active", p["accent_hover"])],
                  foreground=[("disabled", disabled_text)],
                  bordercolor=[("focus", p["focus"]),
                               ("!focus", p["accent_pressed"])],
                  relief=[("pressed", "sunken"), ("!pressed", "raised")])
        style.configure("Liquid.Secondary.TButton", background=p["raised"],
                        foreground=p["text"], bordercolor=p["border"],
                        focuscolor=p["focus"], focusthickness=1,
                        relief="raised", borderwidth=1,
                        font=BODY_FONT, padding=(12, 7))
        style.map("Liquid.Secondary.TButton",
                  background=[("disabled", disabled_background),
                              ("pressed", secondary_pressed),
                              ("active", secondary_hover)],
                  foreground=[("disabled", disabled_text)],
                  bordercolor=[("focus", p["focus"]),
                               ("!focus", p["border"])],
                  relief=[("pressed", "sunken"), ("!pressed", "raised")])
        style.configure("Liquid.Danger.TButton", background=p["danger"],
                        foreground="#FFFFFF", bordercolor=p["red"],
                        focuscolor=p["focus"], focusthickness=1,
                        relief="raised", borderwidth=1,
                        font=(FONT_FAMILY, 10, "bold"), padding=(14, 8))
        style.map("Liquid.Danger.TButton",
                  background=[("disabled", disabled_background),
                              ("pressed", danger_pressed),
                              ("active", danger_hover)],
                  foreground=[("disabled", disabled_text)],
                  bordercolor=[("focus", p["focus"]),
                               ("!focus", p["red"])],
                  relief=[("pressed", "sunken"), ("!pressed", "raised")])
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
                        padding=(6, 4), bordercolor=p["border"])
        style.configure("Liquid.Invalid.TEntry", fieldbackground=p["raised"],
                        foreground=p["text"], insertcolor=p["text"],
                        padding=(6, 4), bordercolor=p["red"])
        style.configure("Liquid.Readonly.TCombobox", fieldbackground=p["raised"],
                        background=p["raised"], foreground=p["text"],
                        arrowcolor=p["text"], padding=(5, 4),
                        bordercolor=p["border"])
        style.map("Liquid.Readonly.TCombobox",
                  fieldbackground=[("readonly", p["raised"])],
                  foreground=[("readonly", p["text"])])
        style.configure(
            "Liquid.Vertical.TScrollbar",
            background=p["raised"],
            troughcolor=p["surface"],
            bordercolor=p["border"],
            arrowcolor=p["text"],
            darkcolor=p["border"],
            lightcolor=p["raised"],
        )
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
        self.advanced_state_var = tk.BooleanVar(self, False)
        self.theme_var = tk.StringVar(self, self._theme)
        self.motion_summary_var = tk.StringVar(
            self, _motion_summary_text(self._motion_snapshot)
        )

        mapping = motion_settings_to_mapping(self.config.motion)
        self.motion_vars: dict[str, tk.Variable] = {}
        for key, value in mapping.items():
            variable: tk.Variable
            if key == "jitter_enabled":
                variable = tk.BooleanVar(self, bool(value))
            else:
                variable = tk.StringVar(self, _display_value(value))
            self.motion_vars[key] = variable
            setattr(self, f"{key}_var", variable)

    def _selected_preset(self) -> str:
        choices = self.preset_values
        return self.config.selected_preset if self.config.selected_preset in choices else "Custom"

    @property
    def preset_values(self) -> tuple[str, ...]:
        # Custom is display-only: its numeric values are the current controls,
        # including the intentionally non-preset default combination.
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
        self.shell.columnconfigure(0, weight=1)
        self.shell.rowconfigure(2, weight=1)
        self.shell.bind("<Configure>", self._redraw_shell_art, add="+")

        self.identity_frame = ttk.Frame(
            self.shell, style="Liquid.Surface.TFrame", padding=(14, 8)
        )
        self.identity_frame.grid(
            row=0, column=0, sticky="ew", padx=16, pady=(14, 8)
        )
        self._build_identity()

        self.navigation_frame = ttk.Frame(
            self.shell, style="Liquid.App.TFrame"
        )
        self.navigation_frame.grid(
            row=1, column=0, sticky="ew", padx=16, pady=(0, 8)
        )
        self.nav = LiquidNavigation(
            self.navigation_frame,
            labels=("Control", "Motion", "Advanced"),
            command=self.select_page,
            palette=self._navigation_palette(),
        )
        self.nav.pack(side="left")
        self._build_navigation_actions()

        self.page_host = ttk.Frame(self.shell, style="Liquid.App.TFrame")
        self.page_host.grid(row=2, column=0, sticky="nsew", padx=16)
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)
        self.control_page = ttk.Frame(self.page_host, style="Liquid.App.TFrame")
        self.motion_page = ttk.Frame(self.page_host, style="Liquid.App.TFrame")
        self.advanced_page = ttk.Frame(self.page_host, style="Liquid.App.TFrame")
        self.pages = (self.control_page, self.motion_page, self.advanced_page)
        for page in self.pages:
            page.grid(row=0, column=0, sticky="nsew")

        self._build_trigger_card()
        self._build_advanced_workspace()
        self._build_quick_card()
        self._build_advanced_card()
        self._apply_combobox_popup_palette()
        self.select_page(0)
        self._build_main_control_card()
        self._build_footer()
        for panel in (
            self.identity_frame,
            self.page_host,
            self.runtime_frame,
        ):
            panel.bind("<Configure>", self._redraw_shell_art, add="+")
        self._redraw_shell_art()

    def _build_identity(self) -> None:
        identity_copy = ttk.Frame(
            self.identity_frame, style="Liquid.Surface.TFrame"
        )
        identity_copy.pack(side="left", fill="x", expand=True)
        ttk.Label(
            identity_copy, text="Jitter", style="Liquid.Title.TLabel"
        ).pack(side="left")
        ttk.Label(
            identity_copy,
            text="  Smooth Makcu motion control",
            style="Liquid.Subtitle.TLabel",
        ).pack(side="left", pady=(7, 0))
        ttk.Label(
            self.identity_frame,
            text="MAKCU",
            style="Liquid.Subtitle.TLabel",
        ).pack(side="left", padx=(10, 5))
        self.connection_label = ttk.Label(
            self.identity_frame,
            textvariable=self.connection_status_var,
            style="Liquid.StatusDisconnected.TLabel",
        )
        self.connection_label.pack(side="right")
        self.connection_indicator = tk.Canvas(
            self.identity_frame,
            width=18,
            height=18,
            background=self._palette["surface"],
            highlightthickness=0,
            borderwidth=0,
            takefocus=False,
        )
        self.connection_indicator.pack(side="right", padx=(8, 3), pady=(1, 0))
        self._redraw_connection_indicator()

    def toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_var.set(self._theme)
        self._configure_styles()
        self.configure(background=self._palette["window"])
        self.shell.configure(background=self._palette["window"])
        self.advanced_canvas.configure(background=self._palette["window"])
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

    def _slider_palette(self) -> dict[str, str]:
        p = self._palette
        return {
            "background": p["window"], "rail": p["border"],
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
            self.preset_combo,
            self.waveform_combo,
            self.motion_curve_combo,
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

    def _build_advanced_workspace(self) -> None:
        self.advanced_host = ttk.Frame(
            self.advanced_page, style="Liquid.App.TFrame"
        )
        self.advanced_host.pack(fill="both", expand=True)
        self.advanced_host.rowconfigure(0, weight=1)
        self.advanced_host.columnconfigure(0, weight=1)
        self.advanced_canvas = tk.Canvas(
            self.advanced_host,
            background=self._palette["window"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.advanced_scrollbar = ttk.Scrollbar(
            self.advanced_host,
            orient="vertical",
            command=self.advanced_canvas.yview,
            style="Liquid.Vertical.TScrollbar",
        )
        self.advanced_canvas.configure(yscrollcommand=self.advanced_scrollbar.set)
        self.advanced_canvas.grid(row=0, column=0, sticky="nsew")
        self.advanced_scrollbar.grid(row=0, column=1, sticky="ns")
        self.advanced_content = ttk.Frame(
            self.advanced_canvas,
            style="Liquid.App.TFrame",
            padding=(0, 0, 4, 8),
        )
        self.advanced_content_window = self.advanced_canvas.create_window(
            (0, 0),
            window=self.advanced_content,
            anchor="nw",
        )
        self.canvas = self.advanced_canvas
        self.content = self.advanced_content
        self.content_window = self.advanced_content_window
        # Keep the established widget seams until the scrolling follow-up
        # renames the remaining right-workspace compatibility aliases.
        self.right_host = self.advanced_host
        self.right_canvas = self.advanced_canvas
        self.right_scrollbar = self.advanced_scrollbar
        self.right_content = self.advanced_content
        self.right_content_window = self.advanced_content_window
        self.advanced_content.bind("<Configure>", self._refresh_scrollregion)
        self.advanced_canvas.bind("<Configure>", self._resize_content_window)

    def _card(self, title: str, parent: tk.Misc) -> ttk.LabelFrame:
        card = ttk.LabelFrame(
            parent, text=title, style="Liquid.Card.TLabelframe",
            padding=(12, 8, 12, 10)
        )
        card.pack(fill="x", pady=(0, 9))
        return card

    def _build_main_control_card(self) -> None:
        self.runtime_frame = ttk.Frame(
            self.shell, style="Liquid.Surface.TFrame", padding=(10, 8)
        )
        self.runtime_frame.grid(
            row=3, column=0, sticky="ew", padx=16, pady=(8, 0)
        )
        self.runtime_frame.columnconfigure(0, weight=1, uniform="runtime_actions")
        self.runtime_frame.columnconfigure(1, weight=2)
        self.runtime_frame.columnconfigure(2, weight=1, uniform="runtime_actions")
        self.enable_button = ttk.Button(self.runtime_frame, text="Enable Jitter",
                                        style="Liquid.Primary.TButton",
                                        command=self.toggle_enabled)
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

    def _build_trigger_card(self) -> None:
        self.control_frame = self._card("Control", self.control_page)
        self.control_frame.columnconfigure(0, weight=1)

        device_row = ttk.Frame(
            self.control_frame, style="Liquid.App.TFrame"
        )
        device_row.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        ttk.Label(
            device_row, text="Device", style="Liquid.Muted.TLabel"
        ).pack(side="left")
        self.device_label = ttk.Label(
            device_row,
            textvariable=self.device_status_var,
            style="Liquid.Body.TLabel",
        )
        self.device_label.pack(side="right")

        def combo_row(row, label, variable, values, width):
            ttk.Label(self.control_frame, text=label,
                      style="Liquid.Body.TLabel").grid(row=row, column=0, sticky="w")
            combo = ttk.Combobox(
                self.control_frame,
                textvariable=variable,
                values=values,
                state="readonly",
                style="Liquid.Readonly.TCombobox",
                width=width,
            )
            combo.grid(row=row + 1, column=0, sticky="ew", pady=(2, 6))
            return combo

        self.trigger_combo = combo_row(
            1, "Trigger", self.trigger_var,
            ("Left", "Right", "Middle", "Mouse4", "Mouse5"), 14,
        )
        self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.modifier_combo = combo_row(
            3, "Modifier", self.modifier_var,
            ("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"), 14,
        )
        self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.preset_combo = combo_row(
            5, "Preset", self.preset_var, self.preset_values, 14,
        )
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        self.hotkey_button = ttk.Button(
            self.control_frame,
            text=f"Hotkey: {self.hotkey_name_var.get()}",
            style="Liquid.Secondary.TButton",
            command=self.capture_hotkey,
        )
        self.hotkey_button.grid(row=7, column=0, sticky="ew", pady=(3, 0))

    def _build_navigation_actions(self) -> None:
        self.navigation_actions = ttk.Frame(
            self.navigation_frame, style="Liquid.App.TFrame"
        )
        self.navigation_actions.pack(side="right")
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
        block = ttk.Frame(parent, style="Liquid.App.TFrame")
        block.grid(row=row, column=column, sticky="ew", padx=5, pady=4)
        parent.columnconfigure(column, weight=1)
        top = ttk.Frame(block, style="Liquid.App.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=label, style="Liquid.Body.TLabel").pack(side="left")
        entry = ttk.Entry(top, textvariable=self.motion_vars[key], width=8,
                          style="Liquid.Entry.TEntry", justify="right")
        entry.pack(side="right")
        slider = LiquidSlider(
            block,
            from_=low,
            to=high,
            resolution=resolution,
            command=lambda value, name=key: self._scale_changed(name, value),
            palette=self._slider_palette(),
        )
        slider.set(float(self.motion_vars[key].get()))
        slider.pack(fill="x", pady=(2, 0))
        setattr(self, f"{key}_entry", entry)
        setattr(self, f"{key}_scale", slider)

    def _build_quick_card(self) -> None:
        self.quick_frame = self._card("Motion", self.motion_page)
        self.quick_grid = ttk.Frame(
            self.quick_frame, style="Liquid.App.TFrame"
        )
        self.quick_grid.pack(fill="x")
        self.quick_grid.columnconfigure(0, weight=1, uniform="quick")
        self.quick_grid.columnconfigure(1, weight=1, uniform="quick")
        controls = (
            ("Strength", "motion_strength_pps", 0, 500, 1),
            ("Jitter Rate", "jitter_rate_hz", 0.1, 60, 0.1),
        )
        for index, control in enumerate(controls):
            self._numeric_control(self.quick_grid, index // 2, index % 2, *control)
        self.motion_summary_frame = ttk.Frame(
            self.quick_frame,
            style="Liquid.Surface.TFrame",
            padding=(10, 6),
        )
        self.motion_summary_frame.pack(fill="x", pady=(8, 0))
        ttk.Label(
            self.motion_summary_frame,
            text="LIVE SNAPSHOT",
            style="Liquid.Subtitle.TLabel",
            font=SECTION_FONT,
        ).pack(anchor="w")
        self.motion_summary_label = ttk.Label(
            self.motion_summary_frame,
            textvariable=self.motion_summary_var,
            style="Liquid.Subtitle.TLabel",
            wraplength=680,
        )
        self.motion_summary_label.pack(anchor="w", fill="x", pady=(2, 0))

    def _build_advanced_card(self) -> None:
        self.advanced_frame = ttk.LabelFrame(
            self.advanced_content, text="Advanced Settings",
            style="Liquid.Card.TLabelframe",
            padding=(11, 8, 11, 10))
        self.advanced_frame.pack(fill="x", pady=(0, 9))
        self.advanced_grid = ttk.Frame(
            self.advanced_frame, style="Liquid.App.TFrame"
        )
        self.advanced_grid.pack(fill="x")
        self.advanced_grid.columnconfigure(0, weight=1, uniform="advanced")
        self.advanced_grid.columnconfigure(1, weight=1, uniform="advanced")
        controls = (
            (0, 0, "Angle", "motion_angle_deg", 0, 360, 1),
            (0, 1, "Horizontal", "horizontal_jitter_pps", 0, 500, 1),
            (1, 0, "Vertical", "vertical_jitter_pps", 0, 500, 1),
            (1, 1, "Randomness", "jitter_randomness_percent", 0, 100, 1),
            (2, 0, "Axis Phase", "jitter_axis_phase_deg", 0, 360, 1),
            (2, 1, "Smoothness", "smoothness_percent", 1, 100, 1),
            (3, 0, "Ramp (ms)", "ramp_up_ms", 0, 2000, 1),
            (3, 1, "Update Rate", "update_rate_hz", 20, 500, 1),
            (4, 0, "Max Step", "max_step_px", 1, 50, 1),
            (4, 1, "Acceleration", "acceleration_pps2", 1, 10000, 1),
            (5, 0, "Deceleration", "deceleration_pps2", 1, 10000, 1),
        )
        for row, column, label, key, low, high, resolution in controls:
            self._numeric_control(
                self.advanced_grid, row, column, label, key, low, high, resolution
            )

        waveform_row = ttk.Frame(
            self.advanced_grid, style="Liquid.App.TFrame"
        )
        waveform_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 3))
        ttk.Label(waveform_row, text="Waveform",
                  style="Liquid.Body.TLabel").pack(side="left")
        self.waveform_combo = ttk.Combobox(
            waveform_row, textvariable=self.motion_vars["jitter_waveform"],
            values=JITTER_WAVEFORMS, state="readonly",
            style="Liquid.Readonly.TCombobox")
        self.waveform_combo.pack(side="right", fill="x", expand=True, padx=(8, 0))

        curve_row = ttk.Frame(
            self.advanced_grid, style="Liquid.App.TFrame"
        )
        curve_row.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Label(curve_row, text="Motion Curve",
                  style="Liquid.Body.TLabel").pack(side="left")
        self.motion_curve_combo = ttk.Combobox(
            curve_row, textvariable=self.motion_vars["motion_curve"],
            values=MOTION_CURVES, state="readonly",
            style="Liquid.Readonly.TCombobox")
        self.motion_curve_combo.pack(side="right", fill="x", expand=True, padx=(8, 0))

    def _build_footer(self) -> None:
        self.footer_frame = ttk.Frame(self.shell, style="Liquid.App.TFrame")
        self.footer_frame.grid(
            row=4, column=0, sticky="ew", padx=16, pady=(6, 10)
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
        bands = _BACKGROUND_BANDS[self._theme]
        for index, color in enumerate(bands):
            top = round(height * index / len(bands))
            bottom = round(height * (index + 1) / len(bands))
            self.shell.create_rectangle(
                0,
                top,
                width,
                bottom,
                fill=color,
                outline=color,
                tags=("shell-art", "background-band", f"background-band-{index}"),
            )

        p = self._palette
        for name, attribute in (
            ("identity", "identity_frame"),
            ("page", "page_host"),
            ("runtime", "runtime_frame"),
        ):
            panel = getattr(self, attribute, None)
            if panel is None:
                continue
            left = panel.winfo_x() - 4
            top = panel.winfo_y() - 4
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
            self.shell.create_polygon(
                points,
                smooth=True,
                splinesteps=24,
                fill=p["surface"],
                outline=p["border"],
                width=1,
                tags=(
                    "shell-art",
                    "rounded-surface",
                    "floating-panel",
                    f"floating-panel-{name}",
                ),
            )
            self.shell.create_line(
                left + radius,
                top + 2,
                right - radius,
                top + 2,
                fill=p["raised"],
                width=1,
                tags=("shell-art", "panel-highlight", f"panel-highlight-{name}"),
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

    def _refresh_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.advanced_canvas.configure(
            scrollregion=self.advanced_canvas.bbox("all")
        )

    def _resize_content_window(self, event: tk.Event) -> None:
        self.advanced_canvas.itemconfigure(
            self.advanced_content_window, width=event.width
        )

    def toggle_advanced(self) -> None:
        self._advanced_visible = True
        self.advanced_state_var.set(True)
        self.select_page(2)

    @staticmethod
    def _is_descendant_of(widget: tk.Misc | None, ancestor: tk.Misc) -> bool:
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_advanced_mousewheel(self, event) -> str | None:
        if self.nav.selected_index != 2:
            return None
        target = self.winfo_containing(event.x_root, event.y_root)
        if target is None and self.advanced_host.winfo_ismapped():
            host_left = self.advanced_host.winfo_rootx()
            host_top = self.advanced_host.winfo_rooty()
            host_right = host_left + self.advanced_host.winfo_width()
            host_bottom = host_top + self.advanced_host.winfo_height()
            if host_left <= event.x_root < host_right and host_top <= event.y_root < host_bottom:
                target = self.advanced_host
        if not self._is_descendant_of(target, self.advanced_host):
            return None
        bounds = self.advanced_canvas.bbox("all")
        if bounds is None or bounds[3] <= self.advanced_canvas.winfo_height():
            return None
        direction = -1 if event.delta > 0 else 1
        self.advanced_canvas.yview_scroll(direction, "units")
        return "break"

    # ---- runtime wiring -----------------------------------------------

    def _install_runtime_bindings(self) -> None:
        """Install variable/widget callbacks after the shell exists."""
        for key, variable in self.motion_vars.items():
            variable.trace_add("write", lambda *_args, name=key: self._motion_changed(name))
            entry = getattr(self, f"{key}_entry", None)
            if entry is not None:
                entry.bind("<FocusOut>", lambda _event, name=key: self._motion_changed(name))
                entry.bind("<Return>", lambda _event, name=key: self._motion_changed(name))

    def _scale_changed(self, key: str, value: str) -> None:
        if self._updating_motion_controls:
            return
        self._updating_motion_controls = True
        try:
            self.motion_vars[key].set(value)
        finally:
            self._updating_motion_controls = False
        self._motion_changed(key)

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
            self.start_runtime()
            return
        try:
            self.service.reconnect()
            self.footer_var.set("Connecting to Makcu...")
        except Exception as exc:
            self.footer_var.set(f"Reconnect failed: {type(exc).__name__}")

    def toggle_enabled(self) -> None:
        if self._motion_mode in {"test", "test_pending"}:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        self.set_enabled(not self.enabled)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if not enabled:
            self.emergency_stop("Disabled by user")
            return
        if self._closing:
            return
        if enabled and self._motion_mode in {"test", "test_pending"}:
            self.footer_var.set("Test Run is active; use STOP to cancel")
            return
        if not self.service.connected:
            self.enabled = False
            self._set_runtime_state("disabled")
            self.enable_button.configure(text="Enable Jitter")
            self.footer_var.set("Makcu device is not connected")
            return
        self.enabled = True
        self._motion_mode = None
        self._normal_motion_started = False
        self.trigger_gate.clear()
        self._set_runtime_state("armed")
        self.enable_button.configure(text="Disable Jitter")
        self.footer_var.set("Jitter armed")

    def test_run(self) -> None:
        self.start_test_run()

    def start_test_run(self) -> None:
        if self._closing:
            return
        if not self.service.connected:
            self.footer_var.set("Makcu device is not connected")
            return
        if self._motion_mode in {"test", "test_pending"}:
            self.footer_var.set("Test Run is already active")
            return
        self._test_restore_enabled = self.enabled
        normal_motion_active = self._normal_motion_started
        if normal_motion_active:
            self.service.stop_motion("test_run")
            self._normal_motion_started = False
        self._set_runtime_state("testing")
        self.test_button.set_enabled(False)
        if normal_motion_active:
            # The normal worker emits a queued stop event after stop_motion().
            # Start the timed worker only after that event, so the service's
            # idempotent start contract cannot accidentally reuse the normal
            # worker.
            self._motion_mode = "test_pending"
            self._test_start_pending = True
            return
        started = self._begin_test_motion()
        if not started:
            self.footer_var.set("Test Run could not start")

    def _begin_test_motion(self) -> bool:
        self._motion_mode = "test"
        self._test_start_pending = False
        started = bool(self.service.start_motion(self.get_motion_settings, duration_s=3.0))
        if not started:
            self._motion_mode = None
            self.test_button.set_enabled(True)
            self._restore_after_test()
        return started

    def _restore_after_test(self) -> None:
        restore = self._test_restore_enabled and bool(self.service.connected)
        self._motion_mode = None
        self._test_start_pending = False
        self.test_button.set_enabled(True)
        if restore:
            self.enabled = True
            self._set_runtime_state("armed")
            self.enable_button.configure(text="Disable Jitter")
        else:
            self.enabled = False
            self.trigger_gate.clear()
            self._set_runtime_state("disabled")
            self.enable_button.configure(text="Enable Jitter")

    def emergency_stop(self, reason: str = "Stopped") -> None:
        self.enabled = False
        self._normal_motion_started = False
        self._motion_mode = None
        self._test_start_pending = False
        self.trigger_gate.clear()
        try:
            self.service.stop_motion(str(reason or "Stopped"))
        except Exception:
            pass
        self._set_runtime_state("disabled")
        self.enable_button.configure(text="Enable Jitter")
        self.test_button.set_enabled(True)
        self.footer_var.set(str(reason or "Stopped"))

    def handle_service_event(self, event: ServiceEvent) -> None:
        if self._closing:
            return
        kind = event.kind
        if kind == "connecting":
            self._set_connection_state("Connecting")
            self.device_status_var.set("Connecting to Makcu...")
        elif kind in {"connected", "reconnected"}:
            self._set_connection_state("Connected")
            self.device_status_var.set(str(event.payload or "Makcu device connected"))
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
            if self.enabled and self._motion_mode is None:
                if self.trigger_gate.active and not self._normal_motion_started:
                    self._normal_motion_started = bool(
                        self.service.start_motion(self.get_motion_settings)
                    )
                    if self._normal_motion_started:
                        self._set_runtime_state("moving")
                elif not self.trigger_gate.active and self._normal_motion_started:
                    self.service.stop_motion("trigger_released")
                    self._normal_motion_started = False
                    self._set_runtime_state("armed")
        elif kind == "motion_error":
            self.emergency_stop(f"Motion error: {event.payload}")
        elif kind == "motion_stopped":
            if self._motion_mode == "test_pending" and self._test_start_pending:
                if not self._begin_test_motion():
                    self.footer_var.set("Test Run could not start")
            elif self._motion_mode == "test":
                self._restore_after_test()
                self.footer_var.set("Test Run complete")
            elif self.enabled:
                self._normal_motion_started = False
                self._set_runtime_state("armed")

    def queue_service_event(self, event: ServiceEvent) -> None:
        # This method is intentionally the only service-to-Tk handoff.
        if self._closing or self._closed:
            return
        self._ui_queue.put(("service", event))

    def _hotkey_pressed(self) -> None:
        if self._capturing_hotkey or self._closing:
            return
        if self._closed:
            return
        self._ui_queue.put(("hotkey", None))

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
                    kind, payload = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == "service":
                        self.handle_service_event(payload)
                    elif kind == "hotkey":
                        self.toggle_enabled()
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
        self.trigger_gate.configure(trigger, modifier)
        self._schedule_save()

    def get_motion_settings(self) -> MotionSettings:
        with self._motion_lock:
            return self._motion_snapshot

    def _replace_motion_snapshot(self, settings: MotionSettings) -> None:
        with self._motion_lock:
            self._motion_snapshot = settings
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
                variable.set(bool(value) if key == "jitter_enabled" else str(value))
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
        config = AppConfig(
            motion=self.get_motion_settings(),
            trigger=self.trigger_var.get(),
            modifier=self.modifier_var.get(),
            hotkey_vk=self._current_hotkey_vk(),
            hotkey_name=self.hotkey_name_var.get(),
            selected_preset=self.preset_var.get() or "Custom",
            theme=self.theme_var.get(),
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
