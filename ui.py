"""Tkinter shell for the standalone Jitter dashboard.

This module deliberately contains the page structure and widget bindings only.
Connection, motion, hotkey, and persistence behavior is added by the runtime
integration layer after the shell has a stable public surface.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from hotkeys import HotkeyWatcher
from makcu_service import MakcuService
from motion import (
    JITTER_WAVEFORMS,
    MOTION_CURVES,
    MOTION_LIMITS,
    MOTION_PRESETS,
    motion_settings_to_mapping,
)
from settings import AppConfig, ConfigStore


# Shared Cyber Minimal palette.  Keep these values in one place so controls
# introduced by later runtime work use the same visual language.
GRAPHITE = "#151a1f"
PANEL = "#20272e"
PANEL_ALT = "#29323b"
BORDER = "#35424d"
CYAN = "#18d4e7"
GREEN = "#4bd17d"
AMBER = "#e4b44f"
RED = "#f05d68"
TEXT = "#edf4f7"
MUTED = "#9aa9b4"

FONT_FAMILY = "Segoe UI"
BODY_FONT = (FONT_FAMILY, 9)
SMALL_FONT = (FONT_FAMILY, 8)
TITLE_FONT = (FONT_FAMILY, 18, "bold")
SECTION_FONT = (FONT_FAMILY, 10, "bold")


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
        self.title("Jitter")
        self.geometry("720x680")
        self.resizable(False, False)
        self.configure(background=GRAPHITE)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        self.config_store = config_store or ConfigStore()
        self.load_outcome = self.config_store.load()
        self.config: AppConfig = self.load_outcome.config
        self._advanced_visible = False
        self._closed = False

        self._configure_styles()
        self._create_variables()
        self._build_page()

        self.service_factory = service_factory or (lambda sink: MakcuService(sink))
        self.hotkey_factory = hotkey_factory or HotkeyWatcher
        self.service = self.service_factory(self.queue_service_event)
        self.hotkey = self.hotkey_factory(
            self.config.hotkey_vk, self._hotkey_pressed
        )
        # Task 7 intentionally does not start background services.  Task 8
        # owns start_runtime() and the lifecycle transitions.
        self.auto_start = bool(auto_start)

    # ---- setup ---------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=GRAPHITE)
        style.configure("Card.TLabelframe", background=PANEL, foreground=TEXT,
                        bordercolor=BORDER, relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=PANEL,
                        foreground=TEXT, font=SECTION_FONT)
        style.configure("App.TLabel", background=GRAPHITE, foreground=TEXT,
                        font=BODY_FONT)
        style.configure("Card.TLabel", background=PANEL, foreground=TEXT,
                        font=BODY_FONT)
        style.configure("Muted.TLabel", background=GRAPHITE, foreground=MUTED,
                        font=SMALL_FONT)
        style.configure("CardMuted.TLabel", background=PANEL, foreground=MUTED,
                        font=SMALL_FONT)
        style.configure("StatusDisconnected.TLabel", background=GRAPHITE,
                        foreground=RED, font=BODY_FONT)
        style.configure("StatusConnecting.TLabel", background=GRAPHITE,
                        foreground=AMBER, font=BODY_FONT)
        style.configure("StatusConnected.TLabel", background=GRAPHITE,
                        foreground=GREEN, font=BODY_FONT)
        style.configure("Primary.TButton", background=CYAN, foreground=GRAPHITE,
                        font=(FONT_FAMILY, 9, "bold"), padding=(12, 7), borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#63e7f1"),
                                                   ("disabled", "#43757b")])
        style.configure("Secondary.TButton", background=PANEL_ALT,
                        foreground=TEXT, font=BODY_FONT, padding=(9, 5), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#3b4b58")],
                  foreground=[("disabled", "#8a969e")])
        style.configure("Danger.TButton", background=RED, foreground="#ffffff",
                        font=(FONT_FAMILY, 10, "bold"), padding=(16, 7), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#ff7b84")])
        style.configure("App.TEntry", fieldbackground=PANEL_ALT,
                        foreground=TEXT, insertcolor=TEXT, padding=(5, 3),
                        bordercolor=BORDER)
        style.configure("App.TCombobox", fieldbackground=PANEL_ALT,
                        background=PANEL_ALT, foreground=TEXT, arrowcolor=CYAN,
                        padding=(4, 3))
        style.map("App.TCombobox", fieldbackground=[("readonly", PANEL_ALT)],
                  foreground=[("readonly", TEXT)])

    def _create_variables(self) -> None:
        self.connection_status_var = tk.StringVar(self, "Disconnected")
        self.runtime_status_var = tk.StringVar(self, "Disabled")
        self.device_status_var = tk.StringVar(self, "Makcu device not connected")
        self.trigger_var = tk.StringVar(self, self.config.trigger)
        self.modifier_var = tk.StringVar(self, self.config.modifier)
        self.hotkey_name_var = tk.StringVar(self, self.config.hotkey_name)
        self.preset_var = tk.StringVar(self, self._selected_preset())
        self.footer_var = tk.StringVar(self, "Ready")
        self.advanced_state_var = tk.BooleanVar(self, False)

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
        shell = ttk.Frame(self, style="App.TFrame")
        shell.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(shell, background=GRAPHITE, highlightthickness=0,
                                borderwidth=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.content = ttk.Frame(self.canvas, style="App.TFrame", padding=(14, 12, 14, 18))
        self.content_window = self.canvas.create_window((0, 0), window=self.content,
                                                         anchor="nw")
        self.content.bind("<Configure>", self._refresh_scrollregion)
        self.canvas.bind("<Configure>", self._resize_content_window)

        self._build_header()
        self._build_device_card()
        self._build_main_control_card()
        self._build_trigger_card()
        self._build_action_card()
        self._build_quick_card()
        self._build_advanced_card()
        self._build_footer()

    def _build_header(self) -> None:
        header = ttk.Frame(self.content, style="App.TFrame")
        header.pack(fill="x", pady=(0, 9))
        ttk.Label(header, text="JITTER", style="App.TLabel", font=TITLE_FONT).pack(side="left")
        ttk.Label(header, text="  MAKCU CONTROL", style="Muted.TLabel",
                  font=(FONT_FAMILY, 8, "bold")).pack(side="left", pady=(5, 0))
        self.connection_label = ttk.Label(
            header, textvariable=self.connection_status_var,
            style="StatusDisconnected.TLabel")
        self.connection_label.pack(side="right", pady=(4, 0))

    def _card(self, title: str) -> ttk.LabelFrame:
        card = ttk.LabelFrame(self.content, text=title, style="Card.TLabelframe",
                              padding=(11, 8, 11, 10))
        card.pack(fill="x", pady=(0, 9))
        return card

    def _build_device_card(self) -> None:
        card = self._card("Device")
        self.device_label = ttk.Label(card, textvariable=self.device_status_var,
                                      style="CardMuted.TLabel")
        self.device_label.pack(side="left", fill="x", expand=True)
        self.reconnect_button = ttk.Button(card, text="Reconnect",
                                            style="Secondary.TButton",
                                            command=self.reconnect)
        self.reconnect_button.pack(side="right")

    def _build_main_control_card(self) -> None:
        card = self._card("Main Control")
        self.enable_button = ttk.Button(card, text="Enable Jitter",
                                        style="Primary.TButton",
                                        command=self.toggle_enabled)
        self.enable_button.pack(side="left")
        state = ttk.Frame(card, style="Card.TLabelframe")
        state.pack(side="right", padx=(12, 0))
        ttk.Label(state, text="RUNTIME", style="CardMuted.TLabel").pack(anchor="e")
        ttk.Label(state, textvariable=self.runtime_status_var,
                  style="Card.TLabel", font=(FONT_FAMILY, 10, "bold")).pack(anchor="e")

    def _build_trigger_card(self) -> None:
        card = self._card("Trigger")
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)
        ttk.Label(card, text="Trigger", style="Card.TLabel").grid(row=0, column=0,
                                                                     sticky="w", padx=(0, 6))
        self.trigger_combo = ttk.Combobox(card, textvariable=self.trigger_var,
                                          values=("Left", "Right", "Middle", "Mouse4", "Mouse5"),
                                          state="readonly", style="App.TCombobox", width=11)
        self.trigger_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(card, text="Modifier", style="Card.TLabel").grid(row=0, column=2,
                                                                       sticky="w", padx=(0, 6))
        self.modifier_combo = ttk.Combobox(card, textvariable=self.modifier_var,
                                           values=("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"),
                                           state="readonly", style="App.TCombobox", width=11)
        self.modifier_combo.grid(row=0, column=3, sticky="ew", padx=(0, 12))
        self.hotkey_button = ttk.Button(card, text="Hotkey: -",
                                        style="Secondary.TButton",
                                        command=self.capture_hotkey)
        self.hotkey_button.grid(row=0, column=4, sticky="e")

    def _build_action_card(self) -> None:
        card = self._card("Actions")
        ttk.Label(card, text="Preset", style="Card.TLabel").pack(side="left", padx=(0, 7))
        self.preset_combo = ttk.Combobox(card, textvariable=self.preset_var,
                                         values=self.preset_values, state="readonly",
                                         style="App.TCombobox", width=17)
        self.preset_combo.pack(side="left", padx=(0, 10))
        self.test_button = ttk.Button(card, text="Test 3s", style="Secondary.TButton",
                                      command=self.test_run)
        self.test_button.pack(side="left")
        # STOP lives in this always-visible card, never in advanced settings.
        self.stop_button = ttk.Button(card, text="STOP", style="Danger.TButton",
                                      command=self.emergency_stop)
        self.stop_button.pack(side="right")

    def _numeric_control(self, parent: tk.Misc, row: int, column: int,
                         label: str, key: str, low: float, high: float,
                         resolution: float = 1.0) -> None:
        block = ttk.Frame(parent, style="Card.TLabelframe")
        block.grid(row=row, column=column, sticky="ew", padx=5, pady=4)
        parent.columnconfigure(column, weight=1)
        top = ttk.Frame(block, style="Card.TLabelframe")
        top.pack(fill="x")
        ttk.Label(top, text=label, style="Card.TLabel").pack(side="left")
        entry = ttk.Entry(top, textvariable=self.motion_vars[key], width=8,
                          style="App.TEntry", justify="right")
        entry.pack(side="right")
        slider = tk.Scale(
            block, from_=low, to=high, resolution=resolution, orient="horizontal",
            showvalue=False, variable=self.motion_vars[key], highlightthickness=0,
            bd=0, relief="flat", troughcolor=PANEL_ALT, activebackground=CYAN,
            background=PANEL, foreground=TEXT, sliderrelief="flat",
        )
        slider.pack(fill="x", pady=(2, 0))
        setattr(self, f"{key}_entry", entry)
        setattr(self, f"{key}_scale", slider)

    def _build_quick_card(self) -> None:
        card = self._card("Quick Jitter")
        grid = ttk.Frame(card, style="Card.TLabelframe")
        grid.pack(fill="x")
        controls = (
            ("Angle", "motion_angle_deg", 0, 360, 1),
            ("Strength", "motion_strength_pps", 0, 500, 1),
            ("Horizontal", "horizontal_jitter_pps", 0, 500, 1),
            ("Vertical", "vertical_jitter_pps", 0, 500, 1),
            ("Jitter Rate", "jitter_rate_hz", 0.1, 60, 0.1),
        )
        for index, control in enumerate(controls):
            self._numeric_control(grid, index // 2, index % 2, *control)

    def _build_advanced_card(self) -> None:
        self.advanced_frame = ttk.LabelFrame(
            self.content, text="Advanced Settings", style="Card.TLabelframe",
            padding=(11, 8, 11, 10))
        grid = ttk.Frame(self.advanced_frame, style="Card.TLabelframe")
        grid.pack(fill="x")
        self._numeric_control(grid, 0, 0, "Randomness", "jitter_randomness_percent", 0, 100)
        self._numeric_control(grid, 0, 1, "Axis Phase", "jitter_axis_phase_deg", 0, 360)
        self._numeric_control(grid, 1, 0, "Smoothness", "smoothness_percent", 1, 100)
        self._numeric_control(grid, 1, 1, "Ramp (ms)", "ramp_up_ms", 0, 2000)
        self._numeric_control(grid, 2, 0, "Update Rate", "update_rate_hz", 20, 500)
        self._numeric_control(grid, 2, 1, "Max Step", "max_step_px", 1, 50)
        self._numeric_control(grid, 3, 0, "Acceleration", "acceleration_pps2", 1, 10000)
        self._numeric_control(grid, 3, 1, "Deceleration", "deceleration_pps2", 1, 10000)

        ttk.Label(grid, text="Waveform", style="Card.TLabel").grid(row=4, column=0,
                                                                       sticky="w", padx=5, pady=(8, 3))
        self.waveform_combo = ttk.Combobox(
            grid, textvariable=self.motion_vars["jitter_waveform"],
            values=JITTER_WAVEFORMS, state="readonly", style="App.TCombobox")
        self.waveform_combo.grid(row=4, column=1, sticky="ew", padx=5, pady=(8, 3))
        ttk.Label(grid, text="Motion Curve", style="Card.TLabel").grid(row=5, column=0,
                                                                          sticky="w", padx=5, pady=3)
        self.motion_curve_combo = ttk.Combobox(
            grid, textvariable=self.motion_vars["motion_curve"],
            values=MOTION_CURVES, state="readonly", style="App.TCombobox")
        self.motion_curve_combo.grid(row=5, column=1, sticky="ew", padx=5, pady=3)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.advanced_toggle = ttk.Button(self.content, text="Advanced Settings",
                                           style="Secondary.TButton",
                                           command=self.toggle_advanced)
        self.advanced_toggle.pack(fill="x", pady=(0, 9))

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.content, style="App.TFrame")
        footer.pack(fill="x", pady=(0, 3))
        self.footer_label = ttk.Label(footer, textvariable=self.footer_var,
                                      style="Muted.TLabel", anchor="w")
        self.footer_label.pack(fill="x")

    # ---- shell interactions -------------------------------------------

    def _refresh_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content_window(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.content_window, width=event.width)

    def toggle_advanced(self) -> None:
        if self._advanced_visible:
            self.advanced_frame.pack_forget()
            self._advanced_visible = False
            self.advanced_state_var.set(False)
        else:
            self.advanced_frame.pack(fill="x", pady=(0, 9), before=self.advanced_toggle)
            self._advanced_visible = True
            self.advanced_state_var.set(True)
        self.update_idletasks()
        self._refresh_scrollregion()

    # The following callbacks are intentional shell seams.  Task 8 replaces
    # their no-op behavior with the runtime state machine.
    def reconnect(self) -> None:
        self.footer_var.set("Reconnect is available when runtime services start.")

    def toggle_enabled(self) -> None:
        self.footer_var.set("Enable Jitter is available when runtime services start.")

    def test_run(self) -> None:
        self.footer_var.set("Test Run is available when runtime services start.")

    def emergency_stop(self) -> None:
        self.footer_var.set("Stopped")

    def capture_hotkey(self) -> None:
        self.footer_var.set("Hotkey capture is available when runtime services start.")

    def queue_service_event(self, _event: Any) -> None:
        # Service callbacks must not touch Tk from worker threads.  Runtime
        # wiring adds the after(0, ...) handoff in Task 8.
        return None

    def _hotkey_pressed(self) -> None:
        return None

    def close_app(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.destroy()


__all__ = ["JitterApp"]
