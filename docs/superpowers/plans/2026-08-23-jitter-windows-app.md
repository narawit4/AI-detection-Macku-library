# Jitter Windows App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new standalone one-page Windows Jitter controller that connects to Makcu, provides safe smooth two-axis motion, and contains none of EverFall's AI or auxiliary feature stack.

**Architecture:** Keep the deterministic motion domain in `motion.py`, persistence in `settings.py`, Windows hotkey polling in `hotkeys.py`, and all Makcu I/O/workers in `makcu_service.py`. `ui.py` owns only Tk state, event marshalling, input gating, and presentation; `main.py` owns process setup and the single-instance mutex.

**Tech Stack:** Python 3.10+, Tkinter/ttk, `makcu==2.3.1`, `unittest`, Windows `ctypes`, and Nuitka for explicit on-demand packaging.

**Spec:** `docs/superpowers/specs/2026-08-23-jitter-windows-app-design.md`

## Global Constraints

- The project root is exactly `C:\Users\User\Desktop\Jitter` and is independent from EverFall Jitter.
- The application is Windows-only and uses Python 3.10 or newer.
- Runtime dependencies are limited to `makcu==2.3.1` and Python's standard library.
- The UI is one fixed-size approximately 720 by 680 pixel English Tkinter page using the approved Focused Dashboard layout.
- Jitter starts Disabled on every process launch; held-button and Moving state are never persisted.
- Makcu movement is two-dimensional; the scroll wheel is not a Z motion axis.
- Tk widgets and Tk variables are accessed only on the Tk main thread.
- AI, ONNX, training, profiles, overlay, tray, Pillow, Pystray, and Torch are out of scope.
- `config.json`, `config.json.bak`, and `app.log` are independent local user data and remain ignored by Git.
- Normal source development runs syntax/tests/import checks only. Never run Nuitka unless the user explicitly requests a packaged build.
- Every production change follows RED, GREEN, then refactor; every task ends with a focused commit.

## File map

- `main.py`: logging setup, base-directory selection, Windows mutex, process entry point.
- `ui.py`: Focused Dashboard widgets, Tk state, service-event marshalling, trigger gate, hotkey capture, save debounce, and shutdown.
- `motion.py`: limits, defaults, six presets, immutable `MotionSettings`, coercion, `TriggerGate`, and `SmoothMotionEngine`.
- `makcu_service.py`: `ServiceEvent`, Makcu connection generations, callbacks, reconnect, movement worker, Test Run timeout, and cleanup.
- `hotkeys.py`: injectable edge detector and daemon global-hotkey watcher.
- `settings.py`: schema-one `AppConfig`, load outcome, paths, validation, atomic writes, and backup.
- `tests/test_motion.py`: motion settings, presets, trigger gate, and deterministic engine tests.
- `tests/test_settings.py`: config schema, corruption, atomic replacement, backup, and future-schema protection.
- `tests/test_hotkeys.py`: hotkey edge behavior and virtual-key updates.
- `tests/test_makcu_service.py`: fake-controller connection, callback, reconnect, motion, Test Run, and shutdown tests.
- `tests/test_ui.py`: Tk smoke/layout, trigger gating, event marshalling, STOP, config, and shutdown tests.
- `tests/test_entrypoints.py`: mutex, dependency, launcher, packaging-help, and forbidden-import guards.
- `requirements.txt`: pinned runtime dependency.
- `run_gui.bat`: source launcher.
- `gen.bat`: explicit verification plus Nuitka packaging entry point.
- `README.md`: setup, usage, settings, verification, hardware check, and on-demand build instructions.

---

### Task 1: Motion settings, validation, and presets

**Files:**
- Create: `motion.py`
- Create: `tests/test_motion.py`

**Interfaces:**
- Consumes: only Python standard-library types.
- Produces: `MotionSettings`, `MOTION_DEFAULTS`, `MOTION_LIMITS`, `MOTION_PRESETS`, `MOTION_CURVES`, `JITTER_WAVEFORMS`, `motion_settings_from_mapping(raw)`, and `motion_settings_to_mapping(settings)`.

- [ ] **Step 1: Write the failing settings and preset tests**

Create `tests/test_motion.py` with these initial tests:

```python
import unittest
from dataclasses import replace

from motion import (
    JITTER_WAVEFORMS,
    MOTION_CURVES,
    MOTION_DEFAULTS,
    MOTION_PRESETS,
    MotionSettings,
    motion_settings_from_mapping,
    motion_settings_to_mapping,
)


class MotionSettingsTests(unittest.TestCase):
    def test_defaults_match_the_approved_strong_jitter_starting_point(self):
        settings = motion_settings_from_mapping({})
        self.assertEqual(settings.angle_deg, 90.0)
        self.assertEqual(settings.strength_pps, 80.0)
        self.assertEqual(settings.horizontal_jitter_pps, 55.0)
        self.assertEqual(settings.vertical_jitter_pps, 40.0)
        self.assertEqual(settings.update_rate_hz, 240.0)

    def test_numeric_values_are_clamped_and_invalid_choices_use_defaults(self):
        settings = motion_settings_from_mapping({
            "motion_angle_deg": -10,
            "motion_strength_pps": 900,
            "jitter_rate_hz": "bad",
            "jitter_waveform": "Saw",
            "motion_curve": "Instant",
        })
        self.assertEqual(settings.angle_deg, 0.0)
        self.assertEqual(settings.strength_pps, 500.0)
        self.assertEqual(settings.jitter_rate_hz, float(MOTION_DEFAULTS["jitter_rate_hz"]))
        self.assertIn(settings.jitter_waveform, JITTER_WAVEFORMS)
        self.assertIn(settings.motion_curve, MOTION_CURVES)

    def test_all_six_approved_presets_round_trip(self):
        self.assertEqual(
            tuple(MOTION_PRESETS),
            ("Ultra Stable", "Soft", "Balanced", "Fast Response", "Strong Shake", "Extreme"),
        )
        for name, raw in MOTION_PRESETS.items():
            settings = motion_settings_from_mapping(raw)
            restored = motion_settings_from_mapping(motion_settings_to_mapping(settings))
            self.assertEqual(restored, settings, name)

    def test_motion_settings_are_immutable(self):
        settings = MotionSettings()
        with self.assertRaises(AttributeError):
            settings.strength_pps = 20


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_motion.py -v
```

Expected: import failure for missing `motion`.

- [ ] **Step 3: Implement immutable settings, exact limits, and six presets**

Create `motion.py` with a frozen dataclass whose public field names are:

```python
@dataclass(frozen=True)
class MotionSettings:
    angle_deg: float = 90.0
    strength_pps: float = 80.0
    jitter_enabled: bool = True
    horizontal_jitter_pps: float = 55.0
    vertical_jitter_pps: float = 40.0
    smoothness: float = 25.0
    update_rate_hz: float = 240.0
    ramp_up_ms: float = 80.0
    jitter_rate_hz: float = 14.0
    jitter_randomness: float = 25.0
    jitter_axis_phase_deg: float = 90.0
    jitter_waveform: str = "Random blend"
    max_step_px: int = 8
    acceleration_pps2: float = 2500.0
    deceleration_pps2: float = 3500.0
    motion_curve: str = "S-curve"
```

Define the exact mapping keys used in JSON/UI:

```python
MOTION_DEFAULTS = {
    "motion_angle_deg": "90",
    "motion_strength_pps": "80",
    "jitter_enabled": True,
    "horizontal_jitter_pps": "55",
    "vertical_jitter_pps": "40",
    "smoothness_percent": "25",
    "update_rate_hz": "240",
    "ramp_up_ms": "80",
    "jitter_rate_hz": "14",
    "jitter_randomness_percent": "25",
    "jitter_axis_phase_deg": "90",
    "jitter_waveform": "Random blend",
    "max_step_px": "8",
    "acceleration_pps2": "2500",
    "deceleration_pps2": "3500",
    "motion_curve": "S-curve",
}

MOTION_LIMITS = {
    "motion_angle_deg": (0.0, 360.0),
    "motion_strength_pps": (0.0, 500.0),
    "horizontal_jitter_pps": (0.0, 500.0),
    "vertical_jitter_pps": (0.0, 500.0),
    "smoothness_percent": (1.0, 100.0),
    "update_rate_hz": (20.0, 500.0),
    "ramp_up_ms": (0.0, 2000.0),
    "jitter_rate_hz": (0.1, 60.0),
    "jitter_randomness_percent": (0.0, 100.0),
    "jitter_axis_phase_deg": (0.0, 360.0),
    "max_step_px": (1.0, 50.0),
    "acceleration_pps2": (1.0, 10000.0),
    "deceleration_pps2": (1.0, 10000.0),
}

MOTION_CURVES = ("Linear", "Ease-in", "S-curve")
JITTER_WAVEFORMS = ("Sine", "Triangle", "Square", "Random blend")
```

Define `MOTION_PRESETS` with the exact six mappings from the approved spec. Use these values:

```python
MOTION_PRESETS = {
    "Ultra Stable": {"motion_strength_pps": "15", "horizontal_jitter_pps": "0", "vertical_jitter_pps": "0", "smoothness_percent": "95", "ramp_up_ms": "400", "max_step_px": "1", "acceleration_pps2": "60", "deceleration_pps2": "160", "motion_curve": "S-curve", "jitter_enabled": False},
    "Soft": {"motion_strength_pps": "25", "horizontal_jitter_pps": "1", "vertical_jitter_pps": "0", "smoothness_percent": "90", "ramp_up_ms": "300", "max_step_px": "1", "acceleration_pps2": "80", "deceleration_pps2": "180", "motion_curve": "S-curve", "jitter_enabled": True},
    "Balanced": {"motion_strength_pps": "40", "horizontal_jitter_pps": "2", "vertical_jitter_pps": "0", "smoothness_percent": "80", "ramp_up_ms": "250", "max_step_px": "2", "acceleration_pps2": "120", "deceleration_pps2": "240", "motion_curve": "S-curve", "jitter_enabled": True},
    "Fast Response": {"motion_strength_pps": "60", "horizontal_jitter_pps": "2", "vertical_jitter_pps": "1", "smoothness_percent": "55", "ramp_up_ms": "100", "max_step_px": "2", "acceleration_pps2": "260", "deceleration_pps2": "400", "motion_curve": "Ease-in", "jitter_enabled": True},
    "Strong Shake": {"motion_strength_pps": "80", "horizontal_jitter_pps": "90", "vertical_jitter_pps": "70", "smoothness_percent": "18", "update_rate_hz": "240", "ramp_up_ms": "40", "jitter_rate_hz": "16", "jitter_randomness_percent": "20", "jitter_axis_phase_deg": "90", "jitter_waveform": "Random blend", "max_step_px": "10", "acceleration_pps2": "4000", "deceleration_pps2": "5000", "motion_curve": "Linear", "jitter_enabled": True},
    "Extreme": {"motion_strength_pps": "120", "horizontal_jitter_pps": "180", "vertical_jitter_pps": "150", "smoothness_percent": "5", "update_rate_hz": "360", "ramp_up_ms": "0", "jitter_rate_hz": "24", "jitter_randomness_percent": "40", "jitter_axis_phase_deg": "135", "jitter_waveform": "Square", "max_step_px": "18", "acceleration_pps2": "8000", "deceleration_pps2": "9000", "motion_curve": "Linear", "jitter_enabled": True},
}
```

Implement `motion_settings_from_mapping()` by merging raw values over `MOTION_DEFAULTS`, clamping every numeric key through `MOTION_LIMITS`, validating curve/waveform membership, and constructing `MotionSettings`. Implement `motion_settings_to_mapping()` with the same JSON keys and compact numeric strings so round trips compare equal.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_motion.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit the settings domain**

```powershell
git add motion.py tests/test_motion.py
git commit -m "feat: define Jitter motion settings"
```

---

### Task 2: Deterministic motion engine and trigger gate

**Files:**
- Modify: `motion.py`
- Modify: `tests/test_motion.py`

**Interfaces:**
- Consumes: `MotionSettings` from Task 1.
- Produces: `SmoothMotionEngine.step(settings, dt, elapsed, rng=random) -> tuple[int, int]` and `TriggerGate(trigger="Left", modifier="None")` with `update_button()`, `configure()`, `clear()`, and `active`.

- [ ] **Step 1: Add failing engine and gate tests**

Append these tests to `tests/test_motion.py`:

```python
import random

from motion import SmoothMotionEngine, TriggerGate


class SmoothMotionEngineTests(unittest.TestCase):
    def test_angle_uses_screen_coordinates(self):
        engine = SmoothMotionEngine()
        settings = replace(MotionSettings(), angle_deg=90, strength_pps=100,
                           jitter_enabled=False, smoothness=0, ramp_up_ms=0,
                           acceleration_pps2=10000, max_step_px=50)
        x, y = engine.step(settings, 0.1, 1.0, random.Random(1))
        self.assertEqual(x, 0)
        self.assertGreater(y, 0)

    def test_fractional_motion_accumulates(self):
        engine = SmoothMotionEngine()
        settings = replace(MotionSettings(), angle_deg=0, strength_pps=3,
                           jitter_enabled=False, smoothness=0, ramp_up_ms=0,
                           acceleration_pps2=10000, max_step_px=50)
        reports = [engine.step(settings, 0.1, 1.0, random.Random(1))[0] for _ in range(10)]
        self.assertEqual(sum(reports), 3)

    def test_balanced_jitter_has_near_zero_net_drift(self):
        engine = SmoothMotionEngine()
        settings = replace(MotionSettings(), strength_pps=0, jitter_enabled=True,
                           horizontal_jitter_pps=20, vertical_jitter_pps=0,
                           jitter_rate_hz=1, jitter_randomness=0, jitter_waveform="Sine",
                           smoothness=0, ramp_up_ms=0, acceleration_pps2=10000,
                           max_step_px=50)
        reports = [engine.step(settings, 0.01, 1.0, random.Random(2))[0] for _ in range(100)]
        self.assertLessEqual(abs(sum(reports)), 1)

    def test_max_step_discards_excess_without_backlog(self):
        engine = SmoothMotionEngine()
        strong = replace(MotionSettings(), angle_deg=0, strength_pps=500,
                         jitter_enabled=False, smoothness=0, ramp_up_ms=0,
                         acceleration_pps2=10000, max_step_px=2)
        stopped = replace(strong, strength_pps=0)
        self.assertEqual(engine.step(strong, 0.1, 1.0, random.Random(3))[0], 2)
        self.assertEqual(engine.step(stopped, 0.1, 1.0, random.Random(3))[0], 0)


class TriggerGateTests(unittest.TestCase):
    def test_modifier_is_required_when_configured(self):
        gate = TriggerGate(trigger="Left", modifier="Right")
        gate.update_button("Left", True)
        self.assertFalse(gate.active)
        gate.update_button("Right", True)
        self.assertTrue(gate.active)
        gate.update_button("Right", False)
        self.assertFalse(gate.active)

    def test_reconfigure_and_clear_drop_held_state(self):
        gate = TriggerGate(trigger="Left", modifier="None")
        gate.update_button("Left", True)
        self.assertTrue(gate.active)
        gate.configure("Mouse4", "None")
        self.assertFalse(gate.active)
        gate.update_button("Mouse4", True)
        gate.clear()
        self.assertFalse(gate.active)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the Task 1 focused command. Expected: imports fail for `SmoothMotionEngine` and `TriggerGate`.

- [ ] **Step 3: Implement the engine and gate**

Implement `SmoothMotionEngine` with these persistent float fields: `velocity_x`, `velocity_y`, `residual_x`, `residual_y`, `filtered_x`, `filtered_y`, and `jitter_phase`. In `step()`:

```python
dt = max(0.0, min(float(dt), 0.1))
self.jitter_phase = (self.jitter_phase + math.tau * settings.jitter_rate_hz * dt) % math.tau
angle = math.radians(settings.angle_deg)
target_x = math.cos(angle) * settings.strength_pps + jitter_x
target_y = math.sin(angle) * settings.strength_pps + jitter_y
progress = 1.0 if settings.ramp_up_ms <= 0 else min(1.0, elapsed / (settings.ramp_up_ms / 1000.0))
tau = (max(0.0, min(settings.smoothness, 100.0)) / 100.0) ** 2 * 0.250
alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
```

Generate Sine/Triangle/Square/Random blend waves, mix configured randomness, apply axis phase, apply Linear/Ease-in/S-curve ramping, smooth the target, limit vector acceleration/deceleration, and accumulate fractional reports. Clamp each integer axis to `max_step_px`; set residual to `total - raw` so the clamped-away excess is discarded.

Implement the gate exactly around string button names:

```python
@dataclass
class TriggerGate:
    trigger: str = "Left"
    modifier: str = "None"
    trigger_held: bool = False
    modifier_held: bool = False

    @property
    def active(self) -> bool:
        return self.trigger_held and (self.modifier == "None" or self.modifier_held)

    def update_button(self, name: str, pressed: bool) -> None:
        if name == self.trigger:
            self.trigger_held = pressed
        if self.modifier != "None" and name == self.modifier:
            self.modifier_held = pressed

    def configure(self, trigger: str, modifier: str) -> None:
        self.trigger = trigger
        self.modifier = modifier
        self.clear()

    def clear(self) -> None:
        self.trigger_held = False
        self.modifier_held = False
```

- [ ] **Step 4: Run all motion tests and verify GREEN**

Run the focused motion command. Expected: 10 tests pass.

- [ ] **Step 5: Commit the engine**

```powershell
git add motion.py tests/test_motion.py
git commit -m "feat: add smooth Jitter engine"
```

---

### Task 3: Independent schema-one configuration

**Files:**
- Create: `settings.py`
- Create: `tests/test_settings.py`

**Interfaces:**
- Consumes: `MotionSettings`, `motion_settings_from_mapping()`, and `motion_settings_to_mapping()`.
- Produces: `SCHEMA_VERSION = 1`, `AppConfig`, `LoadOutcome`, `ConfigStore.load() -> LoadOutcome`, `ConfigStore.save(config)`, and `runtime_base_dir()`.

- [ ] **Step 1: Write failing config tests**

Create `tests/test_settings.py`:

```python
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from motion import MotionSettings
from settings import AppConfig, ConfigStore, SCHEMA_VERSION


class ConfigStoreTests(unittest.TestCase):
    def test_missing_config_returns_safe_disabled_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome = ConfigStore(Path(directory) / "config.json").load()
        self.assertEqual(outcome.config.motion, MotionSettings())
        self.assertEqual(outcome.config.trigger, "Left")
        self.assertEqual(outcome.config.modifier, "None")
        self.assertTrue(outcome.save_allowed)

    def test_valid_config_round_trips_without_runtime_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            config = AppConfig(
                motion=replace(MotionSettings(), strength_pps=123.0),
                trigger="Mouse4", modifier="Right", hotkey_vk=0x77,
                hotkey_name="F8", selected_preset="Custom",
            )
            store.save(config)
            document = json.loads(path.read_text(encoding="utf-8"))
            outcome = store.load()
        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        self.assertNotIn("enabled", document)
        self.assertNotIn("moving", document)
        self.assertEqual(outcome.config, config)

    def test_second_save_keeps_previous_document_as_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            store = ConfigStore(path)
            first = AppConfig(selected_preset="Soft")
            second = AppConfig(selected_preset="Extreme")
            store.save(first)
            store.save(second)
            backup = json.loads((Path(str(path) + ".bak")).read_text(encoding="utf-8"))
        self.assertEqual(backup["selected_preset"], "Soft")

    def test_future_schema_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = {"schema_version": SCHEMA_VERSION + 1, "future": True}
            path.write_text(json.dumps(original), encoding="utf-8")
            store = ConfigStore(path)
            outcome = store.load()
            self.assertFalse(outcome.save_allowed)
            with self.assertRaises(PermissionError):
                store.save(outcome.config)
            restored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(restored, original)

    def test_corrupt_json_uses_defaults_and_reports_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("not json", encoding="utf-8")
            outcome = ConfigStore(path).load()
        self.assertTrue(outcome.save_allowed)
        self.assertIsNotNone(outcome.warning)
        self.assertEqual(outcome.config, AppConfig())
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_settings.py -v
```

Expected: import failure for missing `settings`.

- [ ] **Step 3: Implement config models and atomic store**

Use these public models:

```python
SCHEMA_VERSION = 1
VALID_BUTTONS = ("Left", "Right", "Middle", "Mouse4", "Mouse5")

@dataclass(frozen=True)
class AppConfig:
    motion: MotionSettings = field(default_factory=MotionSettings)
    trigger: str = "Left"
    modifier: str = "None"
    hotkey_vk: int = 0xBD
    hotkey_name: str = "-"
    selected_preset: str = "Strong Shake"

@dataclass(frozen=True)
class LoadOutcome:
    config: AppConfig
    save_allowed: bool = True
    warning: str | None = None
```

`ConfigStore.load()` validates the root object, handles corrupt JSON, rejects future schemas without changing the file, validates Trigger/Modifier against `VALID_BUTTONS`, clamps hotkey VK to 1-255, and delegates motion coercion to Task 1. Store the future-schema state on the instance so `save()` raises `PermissionError` after such a load.

`ConfigStore.save()` serializes a document with `schema_version`, `motion`, `trigger`, `modifier`, `hotkey_vk`, `hotkey_name`, and `selected_preset`; it must not serialize runtime enabled or held state. Write `config.json.tmp`, flush, call `os.fsync`, copy the current file to `.bak`, and call `os.replace`. Remove a leftover temporary file after an `OSError` and re-raise.

Implement `runtime_base_dir()` using `NUITKA_ONEFILE_DIRECTORY`, then compiled `sys.executable`, then `__file__`, so packaged config remains beside `Jitter.exe`.

- [ ] **Step 4: Run config and motion tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p "test_*settings.py" -v
python -m unittest discover -s tests -p test_motion.py -v
```

Expected: both commands pass.

- [ ] **Step 5: Commit independent persistence**

```powershell
git add settings.py tests/test_settings.py
git commit -m "feat: add independent Jitter config"
```

---

### Task 4: Edge-triggered Windows global hotkey

**Files:**
- Create: `hotkeys.py`
- Create: `tests/test_hotkeys.py`

**Interfaces:**
- Consumes: a Windows virtual-key integer and an injected `key_state(vk) -> int` callable.
- Produces: `HotkeyEdgeDetector.update(is_down) -> bool` and `HotkeyWatcher(vk, callback, key_state=None, poll_interval=0.04)` with `start()`, `set_vk()`, `poll_once()`, and `stop()`.

- [ ] **Step 1: Write failing hotkey tests**

Create `tests/test_hotkeys.py`:

```python
import unittest

from hotkeys import HotkeyEdgeDetector, HotkeyWatcher


class HotkeyTests(unittest.TestCase):
    def test_edge_detector_fires_once_per_press(self):
        detector = HotkeyEdgeDetector()
        self.assertTrue(detector.update(True))
        self.assertFalse(detector.update(True))
        self.assertFalse(detector.update(False))
        self.assertTrue(detector.update(True))

    def test_watcher_poll_once_invokes_callback_only_on_down_edge(self):
        states = iter((0x8000, 0x8000, 0, 0x8000))
        calls = []
        watcher = HotkeyWatcher(0xBD, lambda: calls.append("toggle"), key_state=lambda _vk: next(states))
        for _index in range(4):
            watcher.poll_once()
        self.assertEqual(calls, ["toggle", "toggle"])

    def test_changing_vk_resets_the_held_edge(self):
        current = {0xBD: 0x8000, 0x77: 0x8000}
        calls = []
        watcher = HotkeyWatcher(0xBD, lambda: calls.append("toggle"), key_state=current.get)
        watcher.poll_once()
        watcher.set_vk(0x77)
        watcher.poll_once()
        self.assertEqual(calls, ["toggle", "toggle"])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_hotkeys.py -v
```

Expected: import failure for missing `hotkeys`.

- [ ] **Step 3: Implement detector and watcher**

Use an internal lock around the active VK and detector. The default `key_state` is `ctypes.windll.user32.GetAsyncKeyState`. `poll_once()` treats bit `0x8000` as down and calls the callback only when `HotkeyEdgeDetector.update()` returns true. `start()` creates one daemon thread; `_run()` repeatedly calls `poll_once()` and waits with a stop event; `stop()` sets that event. Catch and ignore only callback exceptions after logging them through `logging.exception`.

Use this edge detector exactly:

```python
class HotkeyEdgeDetector:
    def __init__(self):
        self._was_down = False

    def update(self, is_down: bool) -> bool:
        fired = bool(is_down) and not self._was_down
        self._was_down = bool(is_down)
        return fired

    def reset(self) -> None:
        self._was_down = False
```

- [ ] **Step 4: Run hotkey tests and verify GREEN**

Run the focused hotkey command. Expected: 3 tests pass.

- [ ] **Step 5: Commit hotkey polling**

```powershell
git add hotkeys.py tests/test_hotkeys.py
git commit -m "feat: add global Jitter hotkey"
```

---

### Task 5: Makcu connection service and generation safety

**Files:**
- Create: `makcu_service.py`
- Create: `tests/test_makcu_service.py`

**Interfaces:**
- Consumes: `makcu.create_controller`, `MakcuConnectionError`, and `MouseButton`.
- Produces: `BUTTON_NAMES`, `ServiceEvent(kind, payload=None)`, and `MakcuService(event_sink, controller_factory=create_controller)` with `connect()`, `reconnect()`, `connected`, `controller`, and `close()`.

- [ ] **Step 1: Write failing connection-service tests**

Create `tests/test_makcu_service.py` with a fake controller:

```python
import threading
import unittest

from makcu_service import MakcuService, ServiceEvent


class FakeController:
    def __init__(self):
        self.connection_callback = None
        self.button_callback = None
        self.monitoring = False
        self.disconnected = False
        self.moves = []

    def on_connection_change(self, callback):
        self.connection_callback = callback

    def enable_button_monitoring(self, enabled):
        self.monitoring = enabled

    def set_button_callback(self, callback):
        self.button_callback = callback

    def get_device_info(self):
        return "Fake Makcu"

    def get_firmware_version(self):
        return "1.0"

    def move(self, x, y):
        self.moves.append((x, y))

    def disconnect(self):
        self.disconnected = True


class MakcuConnectionTests(unittest.TestCase):
    def test_successful_worker_configures_controller_and_emits_connected(self):
        controller = FakeController()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(controller.monitoring)
        self.assertIsNotNone(controller.connection_callback)
        self.assertIsNotNone(controller.button_callback)
        self.assertTrue(service.connected)
        self.assertEqual(events[-1].kind, "connected")
        self.assertIn("Fake Makcu", events[-1].payload)

    def test_connection_failure_emits_disconnected_without_controller(self):
        events = []
        def failing_factory(**_kwargs):
            raise RuntimeError("not found")
        service = MakcuService(events.append, controller_factory=failing_factory)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertFalse(service.connected)
        self.assertEqual(events[-1], ServiceEvent("disconnected", "RuntimeError: not found"))

    def test_old_generation_controller_is_disconnected_and_ignored(self):
        old = FakeController()
        service = MakcuService(lambda _event: None, controller_factory=lambda **_kwargs: old)
        generation = service._begin_connection()
        service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(old.disconnected)
        self.assertIsNone(service.controller)

    def test_disconnect_signal_during_setup_prevents_controller_install(self):
        class DropsDuringSetup(FakeController):
            def on_connection_change(self, callback):
                super().on_connection_change(callback)
                callback(False)
        controller = DropsDuringSetup()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        self.assertTrue(controller.disconnected)
        self.assertFalse(service.connected)
        self.assertEqual(events[-1].kind, "disconnected")

    def test_button_callback_emits_normalized_name_and_pressed_state(self):
        controller = FakeController()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller)
        generation = service._begin_connection()
        service._connect_worker(generation)
        service._button_event("Left", True)
        self.assertEqual(events[-1], ServiceEvent("button", ("Left", True)))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_makcu_service.py -v
```

Expected: import failure for missing `makcu_service`.

- [ ] **Step 3: Implement connection service**

Define `ServiceEvent` as a frozen dataclass. Map real `MouseButton` values to the five names `Left`, `Right`, `Middle`, `Mouse4`, and `Mouse5`; `_button_event()` must also accept an already normalized string for tests.

`_begin_connection()` increments a lock-protected generation, clears the active controller, emits `ServiceEvent("connecting")`, and returns the generation. `connect()` starts `_connect_worker(generation)` on a daemon thread.

`_connect_worker()` calls:

```python
controller = self._controller_factory(debug=False, auto_reconnect=True)
controller.on_connection_change(lambda connected, g=generation: self._connection_changed(g, connected))
controller.enable_button_monitoring(True)
controller.set_button_callback(self._button_event)
```

Read device details without failing the connection if diagnostics are unavailable. Track the latest connection signal together with its generation; if `False` arrives during setup, disconnect instead of installing that controller. Install the controller only if its generation is current and no setup-time disconnect was seen; otherwise disconnect that exact controller. On failure, emit `ServiceEvent("disconnected", "TypeName: message")` only for the current generation.

`_connection_changed()` updates connection state only for the active generation and emits `reconnected` or `disconnected`. `reconnect()` invalidates the old generation, disconnects the exact prior controller on a daemon worker, and starts a fresh connection. `close()` invalidates generations and disconnects the exact controller without touching Tk. Task 6 adds motion cancellation to these transitions once the movement worker exists.

- [ ] **Step 4: Run connection tests and verify GREEN**

Run the focused Makcu command. Expected: 5 tests pass.

- [ ] **Step 5: Commit connection service**

```powershell
git add makcu_service.py tests/test_makcu_service.py
git commit -m "feat: add Makcu connection service"
```

---

### Task 6: Makcu movement worker and interruptible Test Run

**Files:**
- Modify: `makcu_service.py`
- Modify: `tests/test_makcu_service.py`

**Interfaces:**
- Consumes: `MotionSettings`, `SmoothMotionEngine`, and a thread-safe `settings_provider() -> MotionSettings`.
- Produces: `MakcuService.start_motion(settings_provider, duration_s=None) -> bool`, `stop_motion()`, `motion_active`, and terminal `ServiceEvent("motion_stopped", reason)`.

- [ ] **Step 1: Add failing movement-worker tests**

Append:

```python
from motion import MotionSettings


class ConstantEngine:
    def step(self, _settings, _dt, _elapsed):
        return 2, -1


class MakcuMovementTests(unittest.TestCase):
    def connected_service(self):
        controller = FakeController()
        events = []
        service = MakcuService(events.append, controller_factory=lambda **_kwargs: controller,
                               engine_factory=ConstantEngine)
        generation = service._begin_connection()
        service._connect_worker(generation)
        return service, controller, events

    def test_start_motion_sends_reports_and_stop_is_interruptible(self):
        service, controller, _events = self.connected_service()
        self.assertTrue(service.start_motion(lambda: MotionSettings()))
        deadline = threading.Event()
        for _index in range(100):
            if controller.moves:
                break
            deadline.wait(0.005)
        service.stop_motion()
        service.join_motion(1.0)
        count_after_stop = len(controller.moves)
        deadline.wait(0.03)
        self.assertGreater(count_after_stop, 0)
        self.assertEqual(len(controller.moves), count_after_stop)

    def test_timed_motion_finishes_and_emits_test_complete(self):
        service, controller, events = self.connected_service()
        self.assertTrue(service.start_motion(lambda: MotionSettings(update_rate_hz=500), duration_s=0.02))
        service.join_motion(1.0)
        self.assertGreater(len(controller.moves), 0)
        self.assertEqual(events[-1], ServiceEvent("motion_stopped", "duration_complete"))

    def test_disconnect_stops_motion_before_emitting_disconnected(self):
        service, _controller, events = self.connected_service()
        service.start_motion(lambda: MotionSettings())
        generation = service.connection_generation
        service._connection_changed(generation, False)
        service.join_motion(1.0)
        self.assertFalse(service.motion_active)
        self.assertEqual(events[-1].kind, "disconnected")
```

- [ ] **Step 2: Run the Makcu test and verify RED**

Run the focused Makcu command. Expected: constructor rejects `engine_factory` or movement methods are missing.

- [ ] **Step 3: Implement generation-safe motion**

Add a separate motion generation, stop event, active flag, and thread. `start_motion()` returns false when disconnected, is idempotent when already active, clears the stop event, captures the generation, and starts one daemon worker.

The worker uses `time.perf_counter()`, creates one engine from `engine_factory`, reads the latest immutable settings each iteration, sends non-zero reports only, and waits interruptibly:

```python
interval = 1.0 / max(settings.update_rate_hz, 20.0)
self._motion_stop.wait(max(0.0, interval - (time.perf_counter() - tick_started)))
```

Stop when duration expires, service closes, controller disconnects, the stop event is set, or generation changes. On `move()` exception emit `ServiceEvent("motion_error", "TypeName: message")`, then stop. A normal timed exit emits `ServiceEvent("motion_stopped", "duration_complete")`; manual stops use `manual`, and disconnect uses `disconnected`. Ensure a late old worker cannot clear the active flag for a newer generation. Update Task 5's `_connection_changed()`, `reconnect()`, and `close()` to call `stop_motion()` before emitting disconnect or replacing/closing the controller.

Expose `join_motion(timeout)` only as a bounded lifecycle/testing helper; production UI must never call it on the Tk thread.

- [ ] **Step 4: Run all Makcu and motion tests and verify GREEN**

Run:

```powershell
python -m unittest discover -s tests -p test_makcu_service.py -v
python -m unittest discover -s tests -p test_motion.py -v
```

Expected: both commands pass.

- [ ] **Step 5: Commit movement worker**

```powershell
git add makcu_service.py tests/test_makcu_service.py
git commit -m "feat: add interruptible Makcu movement"
```

---

### Task 7: Focused Dashboard UI shell

**Files:**
- Create: `ui.py`
- Create: `tests/test_ui.py`

**Interfaces:**
- Consumes: `AppConfig`, `ConfigStore`, `HotkeyWatcher`, motion constants, `TriggerGate`, `MakcuService`, and `ServiceEvent`.
- Produces: `JitterApp(tk.Tk)` with injectable `config_store`, `service_factory`, `hotkey_factory`, and `auto_start` arguments.

- [ ] **Step 1: Write failing Tk layout smoke tests**

Create `tests/test_ui.py`:

```python
import tkinter as tk
import unittest

from ui import JitterApp


class StubStore:
    def __init__(self):
        self.saved = []

    def load(self):
        from settings import AppConfig, LoadOutcome
        return LoadOutcome(AppConfig())

    def save(self, config):
        self.saved.append(config)


class StubService:
    def __init__(self, event_sink):
        self.event_sink = event_sink
        self.connected = False
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def connect(self):
        self.started += 1

    def reconnect(self):
        return None

    def start_motion(self, _settings_provider, duration_s=None):
        self.started += 1
        return self.connected

    def stop_motion(self, reason="manual"):
        self.stopped += 1

    def close(self):
        self.closed += 1


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


class JitterLayoutTests(unittest.TestCase):
    def setUp(self):
        self.service = None
        self.store = StubStore()
        def service_factory(event_sink):
            self.service = StubService(event_sink)
            return self.service
        self.app = JitterApp(
            config_store=self.store,
            service_factory=service_factory,
            hotkey_factory=StubHotkey,
            auto_start=False,
        )
        self.app.withdraw()

    def tearDown(self):
        try:
            self.app.destroy()
        except tk.TclError:
            pass

    def test_window_is_fixed_size_focused_dashboard(self):
        self.app.update_idletasks()
        self.assertEqual(tuple(map(int, self.app.resizable())), (0, 0))
        width, height = map(int, self.app.geometry().split("+")[0].split("x"))
        self.assertGreaterEqual(width, 700)
        self.assertGreaterEqual(height, 650)

    def test_required_actions_are_present_and_stop_is_outside_advanced(self):
        texts = widget_texts(self.app)
        for expected in ("Reconnect", "Enable Jitter", "Test 3s", "STOP", "Advanced Settings"):
            self.assertIn(expected, texts)
        stop = self.app.stop_button
        ancestor = stop.master
        while ancestor is not self.app:
            self.assertIsNot(ancestor, self.app.advanced_frame)
            ancestor = ancestor.master

    def test_advanced_toggle_does_not_change_outer_geometry(self):
        self.app.update_idletasks()
        before = self.app.geometry().split("+")[0]
        self.app.toggle_advanced()
        self.app.update_idletasks()
        after = self.app.geometry().split("+")[0]
        self.assertEqual(after, before)
```

- [ ] **Step 2: Run the UI test and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -v
```

Expected: import failure for missing `ui`.

- [ ] **Step 3: Build the approved page without runtime behavior**

Create `JitterApp(tk.Tk)`, set title `Jitter`, fixed geometry `720x680`, `resizable(False, False)`, and `WM_DELETE_WINDOW` to `close_app`. Use module-level graphite/cyan/green/amber/red palette and shared fonts/styles.

Build one scrollable content canvas with a header and these stable public widget attributes used by tests and Task 8: `connection_label`, `device_label`, `enable_button`, `stop_button`, `test_button`, `trigger_combo`, `modifier_combo`, `hotkey_button`, `preset_combo`, `advanced_frame`, and `footer_label`.

Define Tk variables for every Task 1 mapping key, plus connection status, runtime status, trigger, modifier, hotkey name, and preset. Quick controls are Angle, Strength, Horizontal, Vertical, and Rate. Each uses a Scale and Entry bound to the same StringVar. Advanced controls use comboboxes for waveform/curve and Scale+Entry pairs for numeric values.

Keep `advanced_frame` initially unpacked. `toggle_advanced()` packs/unpacks it inside the scrollable content and refreshes the canvas scrollregion without changing geometry. Place the STOP button in the Action card above the Quick Jitter card.

- [ ] **Step 4: Run UI smoke tests and verify GREEN**

Run the focused UI command. Expected: 3 tests pass without leaving a window open.

- [ ] **Step 5: Commit the UI shell**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: add focused Jitter dashboard"
```

---

### Task 8: UI runtime wiring, safety, config, and shutdown

**Files:**
- Modify: `ui.py`
- Modify: `tests/test_ui.py`

**Interfaces:**
- Consumes: all Task 3-7 public interfaces.
- Produces: operational `start_runtime()`, `set_enabled()`, `emergency_stop()`, `handle_service_event()`, `start_test_run()`, `apply_preset()`, `capture_hotkey()`, `apply_captured_hotkey()`, `save_config()`, and `close_app()` methods.

- [ ] **Step 1: Add failing runtime-state tests**

Append to `tests/test_ui.py`:

```python
from makcu_service import ServiceEvent


class JitterRuntimeTests(JitterLayoutTests):
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
        self.assertEqual(self.app.runtime_state_var.get(), "Disabled")

    def test_disconnect_performs_emergency_stop(self):
        self.service.connected = True
        self.app.set_enabled(True)
        self.app.handle_service_event(ServiceEvent("disconnected", "Device lost"))
        self.assertFalse(self.app.enabled)
        self.assertEqual(self.app.connection_state_var.get(), "Disconnected")

    def test_test_run_bypasses_trigger_but_requires_connection(self):
        self.service.connected = False
        self.app.start_test_run()
        stopped_count = self.service.stopped
        self.service.connected = True
        self.app.start_test_run()
        self.assertEqual(self.app.runtime_state_var.get(), "Testing")
        self.assertGreaterEqual(self.service.started, 1)
        self.assertEqual(self.service.stopped, stopped_count)

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
```

- [ ] **Step 2: Run UI tests and verify RED**

Run the focused UI command. Expected: runtime methods or state properties are missing.

- [ ] **Step 3: Wire services and enforce state transitions**

In `__init__`, load `LoadOutcome`, create a lock-protected immutable motion snapshot, initialize `TriggerGate`, construct service with `self.queue_service_event`, construct hotkey watcher with `lambda: self.after(0, self.toggle_enabled)`, and keep `_runtime_started`, `_closing`, `_save_allowed`, and `_save_after_id` flags.

`queue_service_event(event)` must do only `self.after(0, self.handle_service_event, event)`. `handle_service_event()` maps connection events to header colors/text, maps button events through `TriggerGate`, starts motion only when enabled and gate.active, stops motion on release, and calls emergency stop on disconnect/motion error.

`set_enabled(True)` requires a connection before entering Armed. `set_enabled(False)` delegates to emergency stop. `emergency_stop(reason)` disables, clears gate, stops service motion immediately, updates button/runtime state, and places the reason in the footer.

`start_test_run()` requires `service.connected`, saves whether the app was enabled, stops an active normal run, enters Testing, and calls `start_motion(self.get_motion_settings, duration_s=3.0)`. On `motion_stopped/duration_complete`, restore Armed only if it was enabled before Test Run; otherwise restore Disabled.

Validate each UI edit through `motion_settings_from_mapping()`. Invalid Entry text keeps the last immutable snapshot and marks that Entry/footer red; valid changes replace the snapshot and schedule a 250 ms save. Applying a preset merges its mapping over defaults, refreshes all variables, and selects the preset name.

Hotkey capture pauses normal watcher toggling, polls virtual keys 1-255 with `after(40, ...)`, uses Escape to cancel, records the first down edge, and passes it to `apply_captured_hotkey(vk, name)`. That method updates the watcher VK and displayed name and schedules config save. Do not offer mouse buttons in capture because Makcu Trigger/Modifier owns them.

`save_config()` creates `AppConfig` from the snapshot and UI bindings and skips writes when future-schema `save_allowed` is false. `close_app()` is idempotent: cancel pending `after` callbacks, emergency-stop, stop hotkey watcher, request service close, save valid config, and destroy Tk without joining workers.

- [ ] **Step 4: Run UI and complete unit suite**

Run:

```powershell
python -m unittest discover -s tests -p test_ui.py -v
python -m unittest discover -s tests -v
```

Expected: UI tests and the entire current suite pass.

- [ ] **Step 5: Commit runtime integration**

```powershell
git add ui.py tests/test_ui.py
git commit -m "feat: wire safe Jitter runtime"
```

---

### Task 9: Entry point, launchers, packaging, and documentation

**Files:**
- Create: `main.py`
- Create: `requirements.txt`
- Create: `run_gui.bat`
- Create: `gen.bat`
- Create: `README.md`
- Create: `tests/test_entrypoints.py`
- Modify: `.gitignore`
- Modify: `AGENTS.md` only if implemented commands or paths differ from the approved instructions.

**Interfaces:**
- Consumes: `JitterApp` and `runtime_base_dir()`.
- Produces: `ensure_single_instance(kernel32=None)`, `configure_logging(base_dir)`, `main()`, source and packaging entry points, and user documentation.

- [ ] **Step 1: Write failing entry-point and script tests**

Create `tests/test_entrypoints.py`:

```python
import subprocess
import unittest
from pathlib import Path

import main


ROOT = Path(__file__).parents[1]


class FakeKernel32:
    def __init__(self, last_error):
        self.last_error = last_error
        self.closed = []

    def CreateMutexW(self, _security, _owner, _name):
        return 123

    def GetLastError(self):
        return self.last_error

    def CloseHandle(self, handle):
        self.closed.append(handle)


class EntryPointTests(unittest.TestCase):
    def test_second_instance_returns_no_handle_and_closes_duplicate(self):
        kernel32 = FakeKernel32(last_error=183)
        self.assertIsNone(main.ensure_single_instance(kernel32))
        self.assertEqual(kernel32.closed, [123])

    def test_first_instance_keeps_mutex_handle(self):
        kernel32 = FakeKernel32(last_error=0)
        self.assertEqual(main.ensure_single_instance(kernel32), 123)
        self.assertEqual(kernel32.closed, [])

    def test_runtime_requirements_exclude_removed_feature_stacks(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertEqual(requirements.strip(), "makcu==2.3.1")
        for forbidden in ("torch", "onnx", "pillow", "pystray", "ultralytics"):
            self.assertNotIn(forbidden, requirements)

    def test_gen_help_is_safe_and_documents_explicit_build(self):
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(ROOT / "gen.bat"), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("build-output\\jitter.exe", completed.stdout.lower())

    def test_source_tree_does_not_import_removed_stacks(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.glob("*.py")
        ).lower()
        for forbidden in ("ai_tracker", "ai_training", "onnxruntime", "torch", "pystray"):
            self.assertNotIn(forbidden, source)
```

- [ ] **Step 2: Run entry-point tests and verify RED**

Run:

```powershell
python -m unittest discover -s tests -p test_entrypoints.py -v
```

Expected: import failure for missing `main` or missing launcher files.

- [ ] **Step 3: Implement Windows process entry point**

In `main.py`, define mutex name `Local\Jitter_Makcu_Controller_Mutex` and duplicate error `183`. `ensure_single_instance()` uses injected `kernel32` or `ctypes.windll.kernel32`, calls `CreateMutexW`, and closes/returns `None` on duplicate. Keep the first-instance handle alive in a module global until process exit.

`configure_logging(base_dir)` writes timestamped INFO diagnostics to `app.log` with `logging.FileHandler(encoding="utf-8")`. `main()` resolves the base directory, configures logging, acquires the mutex, shows a native message box and returns on duplicate, then constructs `ConfigStore(base_dir / "config.json")`, creates `JitterApp`, and enters `mainloop()`.

- [ ] **Step 4: Add exact dependency and launch scripts**

Create `requirements.txt`:

```text
makcu==2.3.1
```

Create `run_gui.bat`:

```bat
@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 pause
```

Create `gen.bat` with `--help` handled before dependency installation. The normal path runs these commands in order and exits on the first failure:

```bat
python -m pip install -r requirements.txt Nuitka ordered-set zstandard
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
python -m nuitka --onefile --mingw64 --assume-yes-for-downloads --progress-bar=none --windows-console-mode=disable --enable-plugin=tk-inter --output-filename=Jitter.exe --output-dir=build-output main.py
```

Redirect Nuitka output to `build-output\build.log`; print the log on failure. Help output must state `Builds build-output\Jitter.exe on explicit request.` Do not run the normal path during implementation verification.

- [ ] **Step 5: Write README and finalize ignore rules**

Document Windows/Python/Makcu requirements, installation, `python main.py`, `run_gui.bat`, the one-page control flow, Trigger/Modifier rules, global hotkey, STOP, Test 3s, config/log locations, verification commands, hardware checklist, and explicit `gen.bat` packaging. State clearly that normal feature work does not build an executable.

Ensure `.gitignore` includes `.venv/`, `__pycache__/`, `*.py[cod]`, `config.json`, `config.json.bak`, `app.log`, `build-output/`, `dist/`, `*.build/`, `*.dist/`, and `.superpowers/`.

- [ ] **Step 6: Run final non-packaging verification**

Run:

```powershell
python -m py_compile main.py ui.py motion.py makcu_service.py hotkeys.py settings.py
python -m unittest discover -s tests -v
python -c "import makcu"
cmd.exe /d /c gen.bat --help
git diff --check
```

Expected: syntax exits 0, all tests pass, Makcu imports, help exits 0 without starting Nuitka, and Git reports no whitespace errors.

- [ ] **Step 7: Perform a source UI smoke run**

Run:

```powershell
python main.py
```

Verify the window is 720x680, English, uses Focused Dashboard A, keeps STOP visible with Advanced open, remains responsive while connecting without hardware, and exits completely on close. Inspect only this project's `app.log`. With hardware attached, additionally verify Trigger/Modifier, Test 3s, hotkey, STOP, cable disconnect, and reconnect.

- [ ] **Step 8: Commit the complete runnable application**

```powershell
git add .gitignore AGENTS.md README.md main.py requirements.txt run_gui.bat gen.bat tests/test_entrypoints.py
git commit -m "feat: deliver standalone Jitter app"
```

Do not run `gen.bat` without `--help` and do not claim the executable build succeeds unless a separate explicitly requested build is run and its exit code is verified.
