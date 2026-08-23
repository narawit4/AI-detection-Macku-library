"""Tkinter dashboard and safe runtime wiring for the standalone Jitter app."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import math
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
        self._save_allowed = bool(self.load_outcome.save_allowed)
        self._advanced_visible = False
        self._closed = False
        self._runtime_started = False
        self._closing = False
        self._save_after_id: str | None = None
        self._capture_after_id: str | None = None
        self._capturing_hotkey = False
        self._capture_seen_down = False
        self._updating_motion_controls = False
        self._invalid_motion_keys: set[str] = set()
        self.enabled = False
        self._motion_mode: str | None = None
        self._test_restore_enabled = False
        self._normal_motion_started = False
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
        style.configure("Invalid.TEntry", fieldbackground="#4a252b",
                        foreground="#ffb8bd", insertcolor="#ffb8bd",
                        padding=(5, 3), bordercolor=RED)
        style.configure("App.TCombobox", fieldbackground=PANEL_ALT,
                        background=PANEL_ALT, foreground=TEXT, arrowcolor=CYAN,
                        padding=(4, 3))
        style.map("App.TCombobox", fieldbackground=[("readonly", PANEL_ALT)],
                  foreground=[("readonly", TEXT)])

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
        self.trigger_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        ttk.Label(card, text="Modifier", style="Card.TLabel").grid(row=0, column=2,
                                                                       sticky="w", padx=(0, 6))
        self.modifier_combo = ttk.Combobox(card, textvariable=self.modifier_var,
                                           values=("None", "Left", "Right", "Middle", "Mouse4", "Mouse5"),
                                           state="readonly", style="App.TCombobox", width=11)
        self.modifier_combo.grid(row=0, column=3, sticky="ew", padx=(0, 12))
        self.modifier_combo.bind("<<ComboboxSelected>>", self._bindings_event)
        self.hotkey_button = ttk.Button(card, text=f"Hotkey: {self.hotkey_name_var.get()}",
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
        self.preset_combo.bind("<<ComboboxSelected>>", self.apply_preset)
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
            showvalue=False, highlightthickness=0,
            bd=0, relief="flat", troughcolor=PANEL_ALT, activebackground=CYAN,
            background=PANEL, foreground=TEXT, sliderrelief="flat",
            command=lambda value, name=key: self._scale_changed(name, value),
        )
        slider.set(float(self.motion_vars[key].get()))
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
        self.set_enabled(not self.enabled)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if not enabled:
            self.emergency_stop("Disabled by user")
            return
        if self._closing:
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
        self._test_restore_enabled = self.enabled
        if self._normal_motion_started:
            self.service.stop_motion("test_run")
            self._normal_motion_started = False
        self._motion_mode = "test"
        self.runtime_status_var.set("Testing")
        self.test_button.configure(state="disabled")
        started = self.service.start_motion(self.get_motion_settings, duration_s=3.0)
        if not started:
            self._motion_mode = None
            self.test_button.configure(state="normal")
            self._restore_after_test()
            self.footer_var.set("Test Run could not start")

    def _restore_after_test(self) -> None:
        restore = self._test_restore_enabled and bool(self.service.connected)
        self._motion_mode = None
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
            self.connection_label.configure(style="StatusConnecting.TLabel")
            self.device_status_var.set("Connecting to Makcu...")
        elif kind in {"connected", "reconnected"}:
            self.connection_status_var.set("Connected")
            self.connection_label.configure(style="StatusConnected.TLabel")
            self.device_status_var.set(str(event.payload or "Makcu device connected"))
            self.footer_var.set("Makcu connected")
        elif kind == "disconnected":
            self.connection_status_var.set("Disconnected")
            self.connection_label.configure(style="StatusDisconnected.TLabel")
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
            if self._motion_mode == "test":
                self._restore_after_test()
                self.footer_var.set("Test Run complete")
            elif self.enabled:
                self._normal_motion_started = False
                self.runtime_status_var.set("Armed")

    def queue_service_event(self, event: ServiceEvent) -> None:
        # This method is intentionally the only service-to-Tk handoff.
        self.after(0, self.handle_service_event, event)

    def _hotkey_pressed(self) -> None:
        if self._capturing_hotkey or self._closing:
            return
        self.after(0, self.toggle_enabled)

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
        self.hotkey_button.configure(text="Press a key...", state="disabled")
        self.footer_var.set("Press a keyboard key (Esc cancels)")
        self._poll_hotkey_capture()

    def _poll_hotkey_capture(self) -> None:
        if not self._capturing_hotkey or self._closing:
            return
        mouse_vks = {0x01, 0x02, 0x04, 0x05, 0x06}
        for vk in range(1, 256):
            if vk in mouse_vks:
                continue
            if not (self._get_async_key_state(vk) & 0x8000):
                continue
            if vk == 0x1B:
                self._cancel_hotkey_capture()
                return
            self.apply_captured_hotkey(vk, self._format_hotkey_name(vk))
            return
        self._capture_after_id = self.after(40, self._poll_hotkey_capture)

    @staticmethod
    def _format_hotkey_name(vk: int) -> str:
        if 0x70 <= vk <= 0x7B:
            return f"F{vk - 0x6F}"
        if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
            return chr(vk)
        names = {0x20: "Space", 0x09: "Tab", 0x0D: "Enter", 0x10: "Shift",
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
        return self._motion_snapshot

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
        self._motion_snapshot = motion_settings_from_mapping(mapping)
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
        self._motion_snapshot = settings
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
            motion=self._motion_snapshot,
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
