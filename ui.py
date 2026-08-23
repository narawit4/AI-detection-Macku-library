"""Tkinter dashboard and safe runtime wiring for the standalone Jitter app."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import math
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


# Shared Windows XP Luna palette and typography.
XP_WINDOW = "#ECE9D8"
XP_PANEL = "#FFFFFF"
XP_BLUE = "#0054E3"
XP_BLUE_LIGHT = "#3C8CF0"
XP_BORDER = "#7F9DB9"
XP_TEXT = "#111111"
XP_MUTED = "#555555"
XP_GREEN = "#287025"
XP_AMBER = "#A76500"
XP_RED = "#A00000"

FONT_FAMILY = "Tahoma"
BODY_FONT = (FONT_FAMILY, 8)
SMALL_FONT = (FONT_FAMILY, 8)
TITLE_FONT = (FONT_FAMILY, 9, "bold")
SECTION_FONT = (FONT_FAMILY, 8)


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
        self.geometry("640x560")
        self.resizable(False, False)
        self.configure(background=XP_WINDOW)
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        self.config_store = config_store or ConfigStore()
        self.load_outcome = self.config_store.load()
        self.config: AppConfig = self.load_outcome.config
        self._save_allowed = bool(self.load_outcome.save_allowed)
        self._advanced_visible = False
        self._closed = False
        self._runtime_started = False
        self._closing = False
        self._save_after_id: str | None = None
        self._capture_after_id: str | None = None
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
        if self.auto_start:
            self.start_runtime()

    # ---- setup ---------------------------------------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("XP.App.TFrame", background=XP_WINDOW)
        style.configure("XP.Title.TFrame", background=XP_BLUE)
        style.configure("XP.Title.TLabel", background=XP_BLUE, foreground="#FFFFFF",
                        font=TITLE_FONT)
        style.configure("XP.Group.TLabelframe", background=XP_WINDOW,
                        foreground="#164E9E", bordercolor="#ACA899",
                        relief="groove", borderwidth=1)
        style.configure("XP.Group.TLabelframe.Label", background=XP_WINDOW,
                        foreground="#164E9E", font=SECTION_FONT)
        style.configure("XP.Group.TLabel", background=XP_WINDOW,
                        foreground=XP_TEXT, font=BODY_FONT)
        style.configure("XP.Muted.TLabel", background=XP_WINDOW,
                        foreground=XP_MUTED, font=SMALL_FONT)
        style.configure("XP.Primary.TButton", foreground=XP_TEXT,
                        font=(FONT_FAMILY, 8, "bold"), padding=(10, 4))
        style.configure("XP.Secondary.TButton", foreground=XP_TEXT,
                        font=BODY_FONT, padding=(8, 3))
        style.configure("XP.Danger.TButton", foreground=XP_RED,
                        font=(FONT_FAMILY, 8, "bold"), padding=(10, 4))
        style.configure("XP.StatusDisconnected.TLabel", background=XP_WINDOW,
                        foreground=XP_RED, font=BODY_FONT)
        style.configure("XP.StatusConnecting.TLabel", background=XP_WINDOW,
                        foreground=XP_AMBER, font=BODY_FONT)
        style.configure("XP.StatusConnected.TLabel", background=XP_WINDOW,
                        foreground=XP_GREEN, font=BODY_FONT)
        style.configure("App.TEntry", fieldbackground=XP_PANEL, foreground=XP_TEXT,
                        insertcolor=XP_TEXT, padding=(5, 3), bordercolor=XP_BORDER)
        style.configure("Invalid.TEntry", fieldbackground=XP_PANEL,
                        foreground=XP_TEXT, insertcolor=XP_TEXT, padding=(5, 3),
                        bordercolor=XP_RED)
        style.configure("XP.TCombobox", fieldbackground=XP_PANEL, background=XP_PANEL,
                        foreground=XP_TEXT, arrowcolor=XP_TEXT, padding=(4, 3),
                        bordercolor=XP_BORDER)
        style.map("XP.TCombobox", fieldbackground=[("readonly", XP_PANEL)],
                  foreground=[("readonly", XP_TEXT)])

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
        shell = ttk.Frame(self, style="XP.App.TFrame")
        shell.pack(fill="both", expand=True)

        self.fixed_content = ttk.Frame(
            shell, style="XP.App.TFrame", padding=(8, 8, 8, 0)
        )
        self.fixed_content.pack(fill="x")
        self._build_header()
        self._build_device_card()
        self._build_main_control_card()

        scroll_host = ttk.Frame(shell, style="XP.App.TFrame")
        scroll_host.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(scroll_host, background=XP_WINDOW, highlightthickness=0,
                                borderwidth=0)
        scrollbar = ttk.Scrollbar(scroll_host, orient="vertical",
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.content = ttk.Frame(self.canvas, style="XP.App.TFrame", padding=(8, 8, 8, 12))
        self.content_window = self.canvas.create_window((0, 0), window=self.content,
                                                         anchor="nw")
        self.content.bind("<Configure>", self._refresh_scrollregion)
        self.canvas.bind("<Configure>", self._resize_content_window)

        self._build_trigger_card()
        self._build_action_card()
        self._build_quick_card()
        self._build_advanced_card()
        self._build_footer()

    def _build_header(self) -> None:
        header = ttk.Frame(self.fixed_content, style="XP.Title.TFrame")
        header.pack(fill="x", pady=(0, 9))
        ttk.Label(header, text="Jitter \N{EM DASH} Makcu Control", style="XP.Title.TLabel").pack(
            side="left", padx=8, pady=5)
        summary = ttk.Frame(self.fixed_content, style="XP.App.TFrame")
        summary.pack(fill="x", pady=(0, 7))
        ttk.Label(summary, text="Connection:", style="XP.Muted.TLabel").pack(side="left")
        self.connection_label = ttk.Label(
            summary, textvariable=self.connection_status_var,
            style="XP.StatusDisconnected.TLabel")
        self.connection_label.pack(side="left", padx=(4, 0))

    def _card(self, title: str, parent: tk.Misc | None = None) -> ttk.LabelFrame:
        if parent is None:
            parent = self.content
        card = ttk.LabelFrame(parent, text=title, style="XP.Group.TLabelframe",
                              padding=(8, 6, 8, 8))
        card.pack(fill="x", pady=(0, 9))
        return card

    def _build_device_card(self) -> None:
        card = self._card("Device", self.fixed_content)
        self.device_label = ttk.Label(card, textvariable=self.device_status_var,
                                      style="XP.Muted.TLabel")
        self.device_label.pack(side="left", fill="x", expand=True)
        self.reconnect_button = ttk.Button(card, text="Reconnect",
                                            style="XP.Secondary.TButton",
                                            command=self.reconnect)
        self.reconnect_button.pack(side="right")

    def _build_main_control_card(self) -> None:
        self.runtime_frame = self._card("Runtime", self.fixed_content)
        card = self.runtime_frame
        self.enable_button = ttk.Button(card, text="Enable Jitter",
                                        style="XP.Primary.TButton",
                                        command=self.toggle_enabled)
        self.enable_button.pack(side="left")
        state = ttk.Frame(card, style="XP.App.TFrame")
        state.pack(side="left", expand=True, padx=12)
        ttk.Label(state, text="RUNTIME", style="XP.Muted.TLabel").pack(anchor="center")
        ttk.Label(state, textvariable=self.runtime_status_var,
                  style="XP.Group.TLabel", font=(FONT_FAMILY, 8, "bold")).pack(anchor="center")
        self.stop_button = ttk.Button(card, text="STOP", style="XP.Danger.TButton",
                                      command=self.emergency_stop)
        self.stop_button.pack(side="right")

    def _build_trigger_card(self) -> None:
        self.setup_frame = self._card("Setup")
        card = self.setup_frame
        card.columnconfigure(1, weight=1)
        card.columnconfigure(3, weight=1)
        ttk.Label(card, text="Trigger", style="XP.Group.TLabel").grid(row=0, column=0,
                                                                     sticky="w", padx=(0, 6))
        self.trigger_combo = ttk.Combobox(card, textvariable=self.trigger_var,
                                          values=("Left", "Right", "Middle", "Mouse4", "Mouse5"),
                                          state="readonly", style="XP.TCombobox", width=11)
        self.trigger_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        ttk.Label(card, text="Modifier", style="XP.Group.TLabel").grid(row=0, column=2,
                                                                       sticky="w", padx=(0, 6))
        self.modifier_combo = ttk.Combobox(card, textvariable=self.modifier_var,
                                           values=("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"),
                                           state="readonly", style="XP.TCombobox", width=11)
        self.modifier_combo.grid(row=0, column=3, sticky="ew", padx=(0, 12))
        self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)

    def _build_action_card(self) -> None:
        card = ttk.Frame(self.setup_frame, style="XP.App.TFrame")
        card.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        ttk.Label(card, text="Preset", style="XP.Group.TLabel").pack(side="left", padx=(0, 7))
        self.preset_combo = ttk.Combobox(card, textvariable=self.preset_var,
                                         values=self.preset_values, state="readonly",
                                         style="XP.TCombobox", width=17)
        self.preset_combo.pack(side="left", padx=(0, 10))
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        self.test_button = ttk.Button(card, text="Test 3s", style="XP.Secondary.TButton",
                                       command=self.test_run)
        self.test_button.pack(side="left")

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
        slider = tk.Scale(
            block, from_=low, to=high, resolution=resolution, orient="horizontal",
            showvalue=False, highlightthickness=0,
            bd=0, relief="flat", troughcolor=XP_BORDER, activebackground=XP_BLUE_LIGHT,
            background=XP_WINDOW, foreground=XP_TEXT, sliderrelief="flat",
            command=lambda value, name=key: self._scale_changed(name, value),
        )
        slider.set(float(self.motion_vars[key].get()))
        slider.pack(fill="x", pady=(2, 0))
        setattr(self, f"{key}_entry", entry)
        setattr(self, f"{key}_scale", slider)

    def _build_quick_card(self) -> None:
        card = self._card("Quick Jitter")
        grid = ttk.Frame(card, style="XP.App.TFrame")
        grid.pack(fill="x")
        controls = (
            ("Strength", "motion_strength_pps", 0, 500, 1),
            ("Jitter Rate", "jitter_rate_hz", 0.1, 60, 0.1),
        )
        for index, control in enumerate(controls):
            self._numeric_control(grid, index // 2, index % 2, *control)

    def _build_advanced_card(self) -> None:
        self.advanced_frame = ttk.LabelFrame(
            self.content, text="Advanced Settings", style="XP.Group.TLabelframe",
            padding=(11, 8, 11, 10))
        self.hotkey_button = ttk.Button(
            self.advanced_frame, text=f"Hotkey: {self.hotkey_name_var.get()}",
            style="XP.Secondary.TButton", command=self.capture_hotkey)
        self.hotkey_button.pack(anchor="w", pady=(0, 5))
        grid = ttk.Frame(self.advanced_frame, style="XP.App.TFrame")
        grid.pack(fill="x")
        self._numeric_control(grid, 1, 0, "Angle", "motion_angle_deg", 0, 360, 1)
        self._numeric_control(grid, 1, 1, "Horizontal", "horizontal_jitter_pps", 0, 500, 1)
        self._numeric_control(grid, 2, 0, "Vertical", "vertical_jitter_pps", 0, 500, 1)
        self._numeric_control(grid, 3, 0, "Randomness", "jitter_randomness_percent", 0, 100)
        self._numeric_control(grid, 3, 1, "Axis Phase", "jitter_axis_phase_deg", 0, 360)
        self._numeric_control(grid, 4, 0, "Smoothness", "smoothness_percent", 1, 100)
        self._numeric_control(grid, 4, 1, "Ramp (ms)", "ramp_up_ms", 0, 2000)
        self._numeric_control(grid, 5, 0, "Update Rate", "update_rate_hz", 20, 500)
        self._numeric_control(grid, 5, 1, "Max Step", "max_step_px", 1, 50)
        self._numeric_control(grid, 6, 0, "Acceleration", "acceleration_pps2", 1, 10000)
        self._numeric_control(grid, 6, 1, "Deceleration", "deceleration_pps2", 1, 10000)

        ttk.Label(grid, text="Waveform", style="XP.Group.TLabel").grid(row=7, column=0,
                                                                       sticky="w", padx=5, pady=(8, 3))
        self.waveform_combo = ttk.Combobox(
            grid, textvariable=self.motion_vars["jitter_waveform"],
            values=JITTER_WAVEFORMS, state="readonly", style="XP.TCombobox")
        self.waveform_combo.grid(row=7, column=1, sticky="ew", padx=5, pady=(8, 3))
        ttk.Label(grid, text="Motion Curve", style="XP.Group.TLabel").grid(row=8, column=0,
                                                                          sticky="w", padx=5, pady=3)
        self.motion_curve_combo = ttk.Combobox(
            grid, textvariable=self.motion_vars["motion_curve"],
            values=MOTION_CURVES, state="readonly", style="XP.TCombobox")
        self.motion_curve_combo.grid(row=8, column=1, sticky="ew", padx=5, pady=3)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.advanced_toggle = ttk.Button(self.content, text="Advanced Settings",
                                           style="XP.Secondary.TButton",
                                           command=self.toggle_advanced)
        self.advanced_toggle.pack(fill="x", pady=(0, 9))

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.content, style="XP.App.TFrame")
        footer.pack(fill="x", pady=(0, 3))
        self.footer_label = ttk.Label(footer, textvariable=self.footer_var,
                                      style="XP.Muted.TLabel", anchor="w")
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
        try:
            self.after(0, self.handle_service_event, event)
        except (tk.TclError, RuntimeError):
            # A worker can race close_app() between the state check and Tk's
            # command registration.  Teardown must remain quiet and safe.
            return

    def _hotkey_pressed(self) -> None:
        if self._capturing_hotkey or self._closing:
            return
        if self._closed:
            return
        try:
            self.after(0, self.toggle_enabled)
        except (tk.TclError, RuntimeError):
            return

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
        finally:
            self._updating_motion_controls = False
        self._invalid_motion_keys.clear()
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
        self._cancel_after("_save_after_id")
        self._cancel_after("_capture_after_id")
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
