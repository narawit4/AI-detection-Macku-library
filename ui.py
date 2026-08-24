"""Tkinter dashboard and safe runtime wiring for the standalone Jitter app."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import math
import queue
import threading
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
from xp_widgets import LiquidXPNav, XPGlossySlider


# Shared Windows XP Remastered palette and typography.
XP_WINDOW = "#F4F1E6"
XP_PANEL = "#FFFFFF"
XP_BLUE = "#2F69B3"
XP_BLUE_LIGHT = "#77A7DD"
XP_BLUE_DARK = "#214E86"
XP_PRIMARY = "#356FAF"
XP_PRIMARY_HOVER = "#5B92CC"
XP_PRIMARY_PRESSED = "#244F7D"
XP_SECONDARY = "#F7F3E7"
XP_SECONDARY_HOVER = "#E5EEF8"
XP_SECONDARY_PRESSED = "#CEDDED"
XP_BORDER = "#A6B0BA"
XP_TEXT = "#20252A"
XP_MUTED = "#66717A"
XP_GREEN = "#287A45"
XP_AMBER = "#A26700"
XP_RED = "#9C2230"
XP_DANGER_BG = "#C74652"
XP_DANGER_HOVER = "#DF6670"
XP_DANGER_PRESSED = "#942F38"
XP_FOCUS = "#E4A43A"

DARK_PALETTE = {
    "window": "#171B22", "panel": "#222833", "blue": "#285A91",
    "blue_dark": "#8CBCEB", "primary": "#4C8CCC", "primary_hover": "#67A5DF",
    "primary_pressed": "#326A9E", "secondary": "#303846",
    "secondary_hover": "#3D4858", "secondary_pressed": "#252C36",
    "border": "#566273", "text": "#E7ECF3", "muted": "#AAB4C2",
    "green": "#63C985", "amber": "#F1B84B", "red": "#FF7380",
    "danger": "#B83E4A", "danger_hover": "#D25763", "danger_pressed": "#8F3039",
    "focus": "#F2B84B",
}

LIGHT_PALETTE = {
    "window": XP_WINDOW, "panel": XP_PANEL, "blue": XP_BLUE,
    "blue_dark": XP_BLUE_DARK, "primary": XP_PRIMARY,
    "primary_hover": XP_PRIMARY_HOVER, "primary_pressed": XP_PRIMARY_PRESSED,
    "secondary": XP_SECONDARY, "secondary_hover": XP_SECONDARY_HOVER,
    "secondary_pressed": XP_SECONDARY_PRESSED, "border": XP_BORDER,
    "text": XP_TEXT, "muted": XP_MUTED, "green": XP_GREEN,
    "amber": XP_AMBER, "red": XP_RED, "danger": XP_DANGER_BG,
    "danger_hover": XP_DANGER_HOVER, "danger_pressed": XP_DANGER_PRESSED,
    "focus": XP_FOCUS,
}

FONT_FAMILY = "Tahoma"
BODY_FONT = (FONT_FAMILY, 9)
SMALL_FONT = (FONT_FAMILY, 8)
TITLE_FONT = (FONT_FAMILY, 10, "bold")
SECTION_FONT = (FONT_FAMILY, 8, "bold")


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


class JitterApp(tk.Tk):
    """Fixed-size, one-page Focused Dashboard for Jitter.

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
        self.geometry("640x560")
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
        style.configure("XP.App.TFrame", background=p["window"])
        style.configure("XP.Status.TFrame", background=p["panel"],
                        bordercolor=p["border"], relief="solid", borderwidth=1)
        style.configure("XP.Title.TFrame", background=p["blue"])
        style.configure("XP.Title.TLabel", background=p["blue"], foreground="#FFFFFF",
                        font=TITLE_FONT)
        style.configure("XP.Group.TLabelframe", background=p["window"],
                        foreground=p["blue_dark"], bordercolor=p["border"],
                        relief="groove", borderwidth=1)
        style.configure("XP.Group.TLabelframe.Label", background=p["window"],
                        foreground=p["blue_dark"], font=SECTION_FONT)
        style.configure("XP.Group.TLabel", background=p["window"],
                        foreground=p["text"], font=BODY_FONT)
        style.configure("XP.Muted.TLabel", background=p["window"],
                        foreground=p["muted"], font=SMALL_FONT)
        style.configure("XP.Primary.TButton", background=p["primary"],
                        foreground="#FFFFFF", bordercolor=p["blue_dark"],
                        focuscolor=p["focus"], focusthickness=1,
                        relief="raised", borderwidth=1,
                        font=(FONT_FAMILY, 9, "bold"), padding=(12, 6))
        style.map("XP.Primary.TButton",
                  background=[("disabled", "#C4CBD3"),
                              ("pressed", p["primary_pressed"]),
                              ("active", p["primary_hover"])],
                  foreground=[("disabled", p["muted"])],
                  bordercolor=[("focus", p["focus"]),
                               ("!focus", p["blue_dark"])],
                  relief=[("pressed", "sunken"), ("!pressed", "raised")])
        style.configure("XP.Secondary.TButton", background=p["secondary"],
                        foreground=p["text"], bordercolor=p["border"],
                        focuscolor=p["focus"], focusthickness=1,
                        relief="raised", borderwidth=1,
                        font=BODY_FONT, padding=(10, 5))
        style.map("XP.Secondary.TButton",
                  background=[("disabled", "#D4D4CF"),
                              ("pressed", p["secondary_pressed"]),
                              ("active", p["secondary_hover"])],
                  foreground=[("disabled", p["muted"])],
                  bordercolor=[("focus", p["focus"]),
                               ("!focus", p["border"])],
                  relief=[("pressed", "sunken"), ("!pressed", "raised")])
        style.configure("XP.Danger.TButton", background=p["danger"],
                        foreground="#FFFFFF", bordercolor=p["red"],
                        focuscolor=p["focus"], focusthickness=1,
                        relief="raised", borderwidth=1,
                        font=(FONT_FAMILY, 9, "bold"), padding=(12, 6))
        style.map("XP.Danger.TButton",
                  background=[("disabled", "#C4B7B8"),
                              ("pressed", p["danger_pressed"]),
                              ("active", p["danger_hover"])],
                  foreground=[("disabled", "#F4EEEE")],
                  bordercolor=[("focus", p["focus"]),
                               ("!focus", p["red"])],
                  relief=[("pressed", "sunken"), ("!pressed", "raised")])
        style.configure("XP.StatusDisconnected.TLabel", background=p["window"], foreground=p["red"], font=BODY_FONT)
        style.configure("XP.StatusConnecting.TLabel", background=p["window"], foreground=p["amber"], font=BODY_FONT)
        style.configure("XP.StatusConnected.TLabel", background=p["window"], foreground=p["green"], font=BODY_FONT)
        style.configure("App.TEntry", fieldbackground=p["panel"], foreground=p["text"], insertcolor=p["text"], padding=(5, 3), bordercolor=p["border"])
        style.configure("Invalid.TEntry", fieldbackground=p["panel"], foreground=p["text"], insertcolor=p["text"], padding=(5, 3), bordercolor=p["red"])
        style.configure("XP.TCombobox", fieldbackground=p["panel"], background=p["panel"], foreground=p["text"], arrowcolor=p["text"], padding=(4, 3), bordercolor=p["border"])
        style.map("XP.TCombobox", fieldbackground=[("readonly", p["panel"])], foreground=[("readonly", p["text"])])

    @property
    def _palette(self) -> Mapping[str, str]:
        return DARK_PALETTE if self._theme == "dark" else LIGHT_PALETTE

    def _create_variables(self) -> None:
        self.connection_status_var = tk.StringVar(self, "Disconnected")
        self.runtime_status_var = tk.StringVar(self, "Disabled")
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
        self.shell = ttk.Frame(self, style="XP.App.TFrame", padding=(8, 8, 8, 8))
        self.shell.pack(fill="both", expand=True)
        self.shell.columnconfigure(0, weight=1)
        self.shell.rowconfigure(2, weight=1)

        self.status_strip = ttk.Frame(self.shell, style="XP.Status.TFrame",
                                      padding=(7, 5))
        self.status_strip.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self._build_status_strip()

        self.navigation_frame = ttk.Frame(self.shell, style="XP.App.TFrame")
        self.navigation_frame.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        self.nav = LiquidXPNav(
            self.navigation_frame,
            labels=("Setup", "Motion", "Advanced"),
            command=self.select_page,
            palette=self._nav_palette(),
        )
        self.nav.pack(side="left")
        self._build_navigation_actions()

        self.page_host = ttk.Frame(self.shell, style="XP.App.TFrame")
        self.page_host.grid(row=2, column=0, sticky="nsew")
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)
        self.setup_page = ttk.Frame(self.page_host, style="XP.App.TFrame")
        self.motion_page = ttk.Frame(self.page_host, style="XP.App.TFrame")
        self.advanced_page = ttk.Frame(self.page_host, style="XP.App.TFrame")
        self.pages = (self.setup_page, self.motion_page, self.advanced_page)
        for page in self.pages:
            page.grid(row=0, column=0, sticky="nsew")

        self._build_trigger_card()
        self._build_advanced_workspace()
        self._build_quick_card()
        self._build_advanced_card()
        self.select_page(0)
        self._build_footer()
        self._build_main_control_card()

    def _build_status_strip(self) -> None:
        self.device_label = ttk.Label(
            self.status_strip,
            textvariable=self.device_status_var,
            style="XP.Muted.TLabel",
        )
        self.device_label.pack(side="left", fill="x", expand=True)
        ttk.Label(self.status_strip, text="Connection:",
                  style="XP.Muted.TLabel").pack(side="left", padx=(8, 4))
        self.connection_label = ttk.Label(
            self.status_strip,
            textvariable=self.connection_status_var,
            style="XP.StatusDisconnected.TLabel",
        )
        self.connection_label.pack(side="right")

    def toggle_theme(self) -> None:
        self._theme = "light" if self._theme == "dark" else "dark"
        self.theme_var.set(self._theme)
        self._configure_styles()
        self.configure(background=self._palette["window"])
        self.advanced_canvas.configure(background=self._palette["window"])
        self.nav.set_palette(self._nav_palette())
        self.theme_button.configure(
            text="☀" if self._theme == "dark" else "☾"
        )
        self.theme_tooltip_text = (
            "Switch to Light Mode" if self._theme == "dark"
            else "Switch to Dark Mode"
        )
        self._hide_theme_tooltip()
        slider_palette = self._slider_palette()
        for widget in self.winfo_children():
            self._apply_slider_palette(widget, slider_palette)
        self._schedule_save()

    def _apply_slider_palette(self, widget: tk.Misc,
                              palette: Mapping[str, str]) -> None:
        if isinstance(widget, XPGlossySlider):
            widget.set_palette(palette)
        for child in widget.winfo_children():
            self._apply_slider_palette(child, palette)

    def _slider_palette(self) -> dict[str, str]:
        p = self._palette
        return {
            "background": p["window"], "rail": p["panel"],
            "rail_outline": p["border"], "hover": p["secondary_hover"],
            "shadow": "#0C0F14" if self._theme == "dark" else "#75828E",
            "thumb": p["secondary"], "thumb_pressed": p["primary"],
            "thumb_outline": p["blue_dark"], "highlight": p["text"],
            "bubble": p["panel"], "text": p["text"], "focus": p["focus"],
        }

    def _nav_palette(self) -> dict[str, str]:
        p = self._palette
        return {
            "background": p["window"], "capsule": p["secondary"],
            "capsule_outline": p["border"], "pill": p["primary"],
            "pill_highlight": "#FFFFFF", "text": p["text"],
            "active_text": "#FFFFFF", "focus": p["focus"],
        }

    def _build_advanced_workspace(self) -> None:
        self.advanced_host = ttk.Frame(self.advanced_page, style="XP.App.TFrame")
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
        )
        self.advanced_canvas.configure(yscrollcommand=self.advanced_scrollbar.set)
        self.advanced_canvas.grid(row=0, column=0, sticky="nsew")
        self.advanced_scrollbar.grid(row=0, column=1, sticky="ns")
        self.advanced_content = ttk.Frame(
            self.advanced_canvas,
            style="XP.App.TFrame",
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
        card = ttk.LabelFrame(parent, text=title, style="XP.Group.TLabelframe",
                              padding=(8, 6, 8, 8))
        card.pack(fill="x", pady=(0, 9))
        return card

    def _build_main_control_card(self) -> None:
        self.runtime_frame = ttk.Frame(self.shell, style="XP.App.TFrame")
        self.runtime_frame.grid(row=4, column=0, sticky="ew")
        self.runtime_frame.columnconfigure(0, weight=1, uniform="runtime_actions")
        self.runtime_frame.columnconfigure(1, weight=2)
        self.runtime_frame.columnconfigure(2, weight=1, uniform="runtime_actions")
        self.enable_button = ttk.Button(self.runtime_frame, text="Enable Jitter",
                                        style="XP.Primary.TButton",
                                        command=self.toggle_enabled)
        self.enable_button.grid(row=0, column=0, sticky="ew")
        state = ttk.Frame(self.runtime_frame, style="XP.App.TFrame")
        state.grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Label(state, text="RUNTIME", style="XP.Muted.TLabel").pack(anchor="center")
        ttk.Label(state, textvariable=self.runtime_status_var,
                  style="XP.Group.TLabel", font=(FONT_FAMILY, 8, "bold")).pack(anchor="center")
        self.stop_button = ttk.Button(self.runtime_frame, text="STOP", style="XP.Danger.TButton",
                                      command=self.emergency_stop)
        self.stop_button.grid(row=0, column=2, sticky="ew")

    def _build_trigger_card(self) -> None:
        self.setup_frame = self._card("Control Setup", self.setup_page)
        self.setup_frame.columnconfigure(0, weight=1)

        def combo_row(row, label, variable, values, width):
            ttk.Label(self.setup_frame, text=label,
                      style="XP.Group.TLabel").grid(row=row, column=0, sticky="w")
            combo = ttk.Combobox(
                self.setup_frame,
                textvariable=variable,
                values=values,
                state="readonly",
                style="XP.TCombobox",
                width=width,
            )
            combo.grid(row=row + 1, column=0, sticky="ew", pady=(2, 6))
            return combo

        self.trigger_combo = combo_row(
            0, "Trigger", self.trigger_var,
            ("Left", "Right", "Middle", "Mouse4", "Mouse5"), 14,
        )
        self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.modifier_combo = combo_row(
            2, "Modifier", self.modifier_var,
            ("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"), 14,
        )
        self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.preset_combo = combo_row(
            4, "Preset", self.preset_var, self.preset_values, 14,
        )
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        self.hotkey_button = ttk.Button(
            self.setup_frame,
            text=f"Hotkey: {self.hotkey_name_var.get()}",
            style="XP.Secondary.TButton",
            command=self.capture_hotkey,
        )
        self.hotkey_button.grid(row=6, column=0, sticky="ew", pady=(2, 0))

    def _build_navigation_actions(self) -> None:
        self.navigation_actions = ttk.Frame(
            self.navigation_frame, style="XP.App.TFrame"
        )
        self.navigation_actions.pack(side="right")
        self.reconnect_tooltip_text = "Reconnect Makcu"
        self.test_tooltip_text = "Test Run 3s"
        self._action_tooltip: tk.Toplevel | None = None
        self.reconnect_button = ttk.Button(
            self.navigation_actions,
            text="↻",
            style="XP.Secondary.TButton",
            command=self.reconnect,
            width=3,
        )
        self.test_button = ttk.Button(
            self.navigation_actions,
            text="▶",
            style="XP.Secondary.TButton",
            command=self.test_run,
            width=3,
        )
        self.theme_tooltip_text = (
            "Switch to Light Mode" if self._theme == "dark"
            else "Switch to Dark Mode"
        )
        self._theme_tooltip: tk.Toplevel | None = None
        self.theme_button = ttk.Button(
            self.navigation_actions,
            text="☀" if self._theme == "dark" else "☾",
            style="XP.Secondary.TButton",
            command=self.toggle_theme,
            width=3,
        )
        self.reconnect_button.pack(side="left", padx=(0, 5))
        self.test_button.pack(side="left", padx=(0, 5))
        self.theme_button.pack(side="left")
        self.reconnect_button.bind(
            "<Enter>",
            lambda event: self._show_action_tooltip(
                event, self.reconnect_tooltip_text
            ),
        )
        self.test_button.bind(
            "<Enter>",
            lambda event: self._show_action_tooltip(
                event, self.test_tooltip_text
            ),
        )
        self.reconnect_button.bind("<Leave>", self._hide_action_tooltip)
        self.test_button.bind("<Leave>", self._hide_action_tooltip)
        self.theme_button.bind("<Enter>", self._show_theme_tooltip)
        self.theme_button.bind("<Leave>", self._hide_theme_tooltip)

    def _show_action_tooltip(self, event: tk.Event, text: str) -> None:
        self._hide_action_tooltip()
        widget = event.widget
        tooltip = tk.Toplevel(self)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(
            f"+{widget.winfo_rootx()}+{widget.winfo_rooty() - 26}"
        )
        p = self._palette
        tk.Label(
            tooltip,
            text=text,
            background=p["panel"],
            foreground=p["text"],
            borderwidth=1,
            relief="solid",
            font=SMALL_FONT,
            padx=5,
            pady=2,
        ).pack()
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
        block = ttk.Frame(parent, style="XP.App.TFrame")
        block.grid(row=row, column=column, sticky="ew", padx=5, pady=4)
        parent.columnconfigure(column, weight=1)
        top = ttk.Frame(block, style="XP.App.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=label, style="XP.Group.TLabel").pack(side="left")
        entry = ttk.Entry(top, textvariable=self.motion_vars[key], width=8,
                          style="App.TEntry", justify="right")
        entry.pack(side="right")
        slider = XPGlossySlider(
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
        self.quick_frame = self._card("Quick Jitter", self.motion_page)
        self.quick_grid = ttk.Frame(self.quick_frame, style="XP.App.TFrame")
        self.quick_grid.pack(fill="x")
        self.quick_grid.columnconfigure(0, weight=1, uniform="quick")
        self.quick_grid.columnconfigure(1, weight=1, uniform="quick")
        controls = (
            ("Strength", "motion_strength_pps", 0, 500, 1),
            ("Jitter Rate", "jitter_rate_hz", 0.1, 60, 0.1),
        )
        for index, control in enumerate(controls):
            self._numeric_control(self.quick_grid, index // 2, index % 2, *control)

    def _build_advanced_card(self) -> None:
        self.advanced_frame = ttk.LabelFrame(
            self.advanced_content, text="Advanced Settings", style="XP.Group.TLabelframe",
            padding=(11, 8, 11, 10))
        self.advanced_frame.pack(fill="x", pady=(0, 9))
        self.advanced_grid = ttk.Frame(self.advanced_frame, style="XP.App.TFrame")
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

        waveform_row = ttk.Frame(self.advanced_grid, style="XP.App.TFrame")
        waveform_row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=(8, 3))
        ttk.Label(waveform_row, text="Waveform",
                  style="XP.Group.TLabel").pack(side="left")
        self.waveform_combo = ttk.Combobox(
            waveform_row, textvariable=self.motion_vars["jitter_waveform"],
            values=JITTER_WAVEFORMS, state="readonly", style="XP.TCombobox")
        self.waveform_combo.pack(side="right", fill="x", expand=True, padx=(8, 0))

        curve_row = ttk.Frame(self.advanced_grid, style="XP.App.TFrame")
        curve_row.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=3)
        ttk.Label(curve_row, text="Motion Curve",
                  style="XP.Group.TLabel").pack(side="left")
        self.motion_curve_combo = ttk.Combobox(
            curve_row, textvariable=self.motion_vars["motion_curve"],
            values=MOTION_CURVES, state="readonly", style="XP.TCombobox")
        self.motion_curve_combo.pack(side="right", fill="x", expand=True, padx=(8, 0))

    def _build_footer(self) -> None:
        self.footer_frame = ttk.Frame(self.shell, style="XP.App.TFrame")
        self.footer_frame.grid(row=3, column=0, sticky="ew", pady=(6, 3))
        self.footer_label = ttk.Label(
            self.footer_frame,
            textvariable=self.footer_var,
            style="XP.Muted.TLabel",
            anchor="w",
        )
        self.footer_label.pack(side="left", fill="x", expand=True)

    def _show_theme_tooltip(self, _event: tk.Event | None = None) -> None:
        self._hide_theme_tooltip()
        tooltip = tk.Toplevel(self)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(
            f"+{self.theme_button.winfo_rootx()}+"
            f"{self.theme_button.winfo_rooty() - 26}"
        )
        p = self._palette
        tk.Label(
            tooltip,
            text=self.theme_tooltip_text,
            background=p["panel"],
            foreground=p["text"],
            borderwidth=1,
            relief="solid",
            font=SMALL_FONT,
            padx=5,
            pady=2,
        ).pack()
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
            self.runtime_status_var.set("Disabled")
            self.enable_button.configure(text="Enable Jitter")
            self.footer_var.set("Makcu device is not connected")
            return
        self.enabled = True
        self._motion_mode = None
        self._normal_motion_started = False
        self.trigger_gate.clear()
        self.runtime_status_var.set("Armed")
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
        self.runtime_status_var.set("Testing")
        self.test_button.configure(state="disabled")
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
            self.test_button.configure(state="normal")
            self._restore_after_test()
        return started

    def _restore_after_test(self) -> None:
        restore = self._test_restore_enabled and bool(self.service.connected)
        self._motion_mode = None
        self._test_start_pending = False
        self.test_button.configure(state="normal")
        if restore:
            self.enabled = True
            self.runtime_status_var.set("Armed")
            self.enable_button.configure(text="Disable Jitter")
        else:
            self.enabled = False
            self.trigger_gate.clear()
            self.runtime_status_var.set("Disabled")
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
        self.runtime_status_var.set("Disabled")
        self.enable_button.configure(text="Enable Jitter")
        self.test_button.configure(state="normal")
        self.footer_var.set(str(reason or "Stopped"))

    def handle_service_event(self, event: ServiceEvent) -> None:
        if self._closing:
            return
        kind = event.kind
        if kind == "connecting":
            self.connection_status_var.set("Connecting")
            self.connection_label.configure(style="XP.StatusConnecting.TLabel")
            self.device_status_var.set("Connecting to Makcu...")
        elif kind in {"connected", "reconnected"}:
            self.connection_status_var.set("Connected")
            self.connection_label.configure(style="XP.StatusConnected.TLabel")
            self.device_status_var.set(str(event.payload or "Makcu device connected"))
            self.footer_var.set("Makcu connected")
        elif kind == "disconnected":
            self.connection_status_var.set("Disconnected")
            self.connection_label.configure(style="XP.StatusDisconnected.TLabel")
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
                        self.runtime_status_var.set("Triggered")
                elif not self.trigger_gate.active and self._normal_motion_started:
                    self.service.stop_motion("trigger_released")
                    self._normal_motion_started = False
                    self.runtime_status_var.set("Armed")
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
                self.runtime_status_var.set("Armed")

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
        while True:
            try:
                kind, payload = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "service":
                self.handle_service_event(payload)
            elif kind == "hotkey":
                self.toggle_enabled()
        self._ui_pump_after_id = self.after(15, self._drain_ui_queue)

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
                    entry.configure(style="Invalid.TEntry")
            self.footer_var.set(f"Invalid value for {key.replace('_', ' ')}")
            return
        self._invalid_motion_keys.clear()
        if self.footer_var.get().startswith("Invalid value for "):
            self.footer_var.set("Ready")
        for name in self.motion_vars:
            entry = getattr(self, f"{name}_entry", None)
            if entry is not None:
                entry.configure(style="App.TEntry")
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
                entry.configure(style="App.TEntry")
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
